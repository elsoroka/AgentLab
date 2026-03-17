"""
GenericAgent implementation for AgentLab

This module defines a `GenericAgent` class and its associated arguments for use in the AgentLab framework. \
The `GenericAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `GenericAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from warnings import warn
import traceback
import bgym
from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
from agentlab.agents import dynamic_prompting as dp
from .planner_agent_prompt import PlannerSystemPrompt, ExecutorSystemPrompt, PlannerPromptFlags
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, HumanMessage

from agentlab.llm.tracking import cost_tracker_decorator

from agentlab.llm.llm_utils import retry

from queue import Queue
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PlanningAgentArgs(AgentArgs):
    planner_model_args: BaseModelArgs = None
    executor_model_args: BaseModelArgs = None
    flags: PlannerPromptFlags = None
    max_retry: int = 1

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"PlanningAgent-{self.planner_model_args.model_name}-{self.executor_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True
        
        """Override the action set for the planner, keeping the original low level action set for the executor."""
        self.flags.action.planner_action_set = deepcopy(DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["plannerhighlevel"])

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        # for backward compatibility with old traces
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        # verify if we can remove this
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.planner_model_args.temperature = 0 # does not work with gpt-5
        self.executor_model_args.temperature = 0

    def prepare(self):
        self.executor_model_args.prepare_server()
        return self.planner_model_args.prepare_server()

    def close(self):
        self.executor_model_args.close_server()
        return self.planner_model_args.close_server()

    def make_agent(self):
        return PlanningAgent(
            planner_model_args=self.planner_model_args,
            executor_model_args=self.executor_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
        )


class PlanningAgent(Agent):

    def __init__(
        self,
        planner_model_args: BaseModelArgs,
        executor_model_args: BaseModelArgs,
        flags: PlannerPromptFlags,
        max_retry: int = 1,
    ):
        self.plan = None
        self.plan_step = 0

        self.planner_llm = planner_model_args.make_model()
        self.executor_llm = executor_model_args.make_model()
        self.planner_model_args = planner_model_args
        self.executor_model_args = executor_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.planner_action_set = self.flags.action.planner_action_set.make_action_set()
        self.executor_action_set = self.flags.action.action_set.make_action_set()

        self._obs_preprocessor = dp.make_obs_preprocessor(self.flags.obs)

        self._check_flag_constancy()
        self.reset(seed=None)

        # Executor management
        self.action_queue = Queue()
        self.observation_queue = Queue()

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def extract_action(self, answer):
        answer = answer.strip()
        if '`' in answer:
            answer = answer.strip('`')
        return answer


    @cost_tracker_decorator
    def get_executor_action(self, obs, specific_task_prompt:str):
        # Example implementation
        self.action_prompt = dp.ActionPrompt(self.action_set, action_flags=self.flags.action)
        self.instructions = dp.GoalInstructions(specific_task_prompt)
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.hints.prompt}\
""")

        answer = self.executor_llm(prompt)
        action = self.extract_action(answer)
        info = {
            #"think": chain_of_thought,
            "messages": [prompt, answer],
            "action": action,
            "stats": {"prompt_length": len(prompt), "answer_length": len(answer)},
            #"some_other_info": "webagents are great",
        }
        return action, info

    @cost_tracker_decorator
    def get_action(self, obs):
        self.obs_history.append(obs)
        self.observation_queue.put(obs)

        ans_dict = dict()
        stats = dict()
        model_args = dict()
        chat_messages = []
        planner_stats = dict()

        if self.plan is None:
            model_args = self.planner_model_args
            try:
                system_prompt = SystemMessage(dp.PlannerSystemPromptElement().prompt)
                main_prompt = PlannerSystemPrompt(
                    self.planner_action_set,
                    obs_history=self.obs_history,
                    actions=self.actions,
                    memories=self.memories,
                    thoughts=self.thoughts,
                    previous_plan=self.plan,
                    step=self.plan_step,
                    flags=self.flags,
                )

                max_prompt_tokens, max_trunc_itr = self._get_maxes()

                human_prompt = dp.fit_tokens(
                    shrinkable=main_prompt,
                    max_prompt_tokens=max_prompt_tokens,
                    model_name=self.executor_model_args.model_name,
                    max_iterations=max_trunc_itr,
                    additional_prompts=[],
                )
                chat_messages = Discussion([system_prompt, human_prompt])
                ans_dict = retry(
                    self.planner_llm,
                    chat_messages,
                    n_retry=self.max_retry,
                    parser=main_prompt._parse_answer,
                )
                print("ans_dict: ", ans_dict)
                
                stats = self.planner_llm.get_stats()
                self.plan = ans_dict["plan"].split("<plan>")[1].split("</plan>")[0]
                self.plan_step = 0
                print("plan: ", self.plan)
                planner_stats = self.planner_llm.get_stats()
                planner_stats["n_retry"] = self.max_retry + 1
                planner_stats["busted_retry"] = 1
                # launch the plan in a thread
                self.executor_thread_pool = ThreadPoolExecutor(max_workers=1)
                def tmp(plan:str):
                    # pass in the queues as globals

                    exec(plan, globals={
                        "action_queue": self.action_queue,
                        "observation_queue": self.observation_queue,
                        "noop": self.noop_action,
                        "search_for_page": self.search_for_page,
                        "open_page": self.open_page,
                        "close_page":self.close_page,
                        "go_back":self.go_back,
                        "go_forward":self.go_forward,
                        "navigate_to_page":self.navigate_to_page,
                        "extract_information_from_page":self.extract_information_from_page,
                        "fill_text_field":self.fill_text_field,
                        "press_button":self.press_button,
                        "select_option":self.select_option,
                        "generic_action":self.generic_action,
                        "add_to_cart":self.add_to_cart,
                        "checkout":self.checkout,
                    })

                    self.action_queue.put("DONE")

                self.executor_thread_pool.submit(tmp, self.plan)
                
            except Exception as e:
                print("Exception: ", e)
                # print the traceback
                print(traceback.format_exc())
                ans_dict = dict(
                    action=None,
                    n_retry=self.max_retry + 1,
                    busted_retry=1,
                )
                

        # Now the plan is running, so we get an action from the threaded executor 
        print("Entering a blocking queue.get()")
        ans_dict = self.action_queue.get()
        self.plan_step += 1
        model_args = self.executor_model_args
        stats = self.executor_llm.get_stats()

        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"planner_stats": planner_stats, "chat_model_args": asdict(model_args), #"eco_logits": eco_impacts.dict()
            },
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = None
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.executor_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.executor_model_args.max_total_tokens,
            self.executor_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr


    # Here is where we define the executor actions.
    