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
import time
from warnings import warn
import functools
import logging
import bgym

logger = logging.getLogger(__name__)
# logger config to allow debug messages
logging.basicConfig(level=logging.DEBUG)
# allow messages from threads
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
from agentlab.agents import dynamic_prompting as dp
from .planner_agent_prompt import PlannerSystemPrompt, ExecutorSystemPrompt, PlannerPromptFlags
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, HumanMessage
from .executor_prompts import (
    search_on_page_prompt,
    navigate_to_page_prompt,
    extract_information_from_page_prompt,
    fill_text_field_prompt,
    press_button_prompt,
    select_option_prompt,
    add_to_cart_prompt,
    checkout_prompt,
)
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
        #self.actions.append(None) # TODO remove

        # action things
        self.navigate_to_page = functools.partial(self.generic_action, task_prompt=navigate_to_page_prompt)
        self.extract_information_from_page = functools.partial(self.generic_action, task_prompt=extract_information_from_page_prompt)
        self.fill_text_field = functools.partial(self.generic_action, task_prompt=fill_text_field_prompt)
        self.press_button = functools.partial(self.generic_action, task_prompt=press_button_prompt)
        self.select_option = functools.partial(self.generic_action, task_prompt=select_option_prompt)
        self.checkout = functools.partial(self.generic_action, task_prompt=checkout_prompt)

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
        self.observation_queue.put(obs)
        self.obs_history.append(obs)

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
                    model_name=self.planner_model_args.model_name,
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
                stats = self.planner_llm.get_stats()
                if "<plan>" in ans_dict["plan"]:
                    self.plan = ans_dict["plan"].split("<plan>")[1].split("</plan>")[0]
                else:
                    self.plan = ans_dict["plan"]

                self.plan_step = 0
                logger.info("plan: %s", self.plan)
                planner_stats = self.planner_llm.get_stats()
                planner_stats["n_retry"] = self.max_retry + 1
                planner_stats["busted_retry"] = 1
                model_args = self.planner_model_args

                self.plan_step = ans_dict.get("step", self.plan_step)
                #self.actions.append(ans_dict.get("action", None))
                #self.memories.append(ans_dict.get("memory", None))
                #self.thoughts.append(ans_dict.get("think", None))

                agent_info = AgentInfo(
                    think=ans_dict.get("think", None),
                    chat_messages=chat_messages,
                    stats=stats,
                    extra_info={"planner_stats": planner_stats, "chat_model_args": asdict(model_args), #"eco_logits": eco_impacts.dict()
                    },
                )

                self.last_agent_info = agent_info

                # launch the plan in a thread
                self.executor_thread_pool = ThreadPoolExecutor(max_workers=1)
                
                def tmp(plan:str):
                    try:
                        exec(plan, {
                            "action_queue": self.action_queue,
                            "observation_queue": self.observation_queue,
                            "noop": self.noop,
                            "search_on_page": self.search_on_page,
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
                    except Exception as e:
                        logger.exception("Exception in executor: %s", e)

                    finished_action = {
                        "action": "noop()",
                        "n_retry": 0,
                        "busted_retry": 0,
                    }
                    self.action_queue.put((finished_action, None))

                self.executor_thread_pool.submit(tmp, self.plan)
                
            except Exception as e:
                logger.exception("Exception in planner: %s", e)
                ans_dict = dict(
                    action=None,
                    n_retry=self.max_retry + 1,
                    busted_retry=1,
                )
                

        # Now the plan is running, so we get an action from the threaded executor
        logger.debug("mainloop: entering a blocking action_queue.get()")
        result = self.action_queue.get()
    
        ans_dict, agent_info = result

        self.actions.append(ans_dict.get("action", None))
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        self.plan_step += 1

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


    # The executor runs in a thread
    # note: ignore potentialconcurrency issues for now, we will fix them later.
    # Here is where we define the executor actions.
    def noop(self):
        # make ans_dict
        ans_dict = {
            "action": "noop()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.last_agent_info))
        return None

    def go_back(self):
        ans_dict = {
            "action": "go_back()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.last_agent_info))
        return None

    def go_forward(self):
        ans_dict = {
            "action": "go_forward()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.last_agent_info))
        return None

    def open_page(self, url:str):
        ans_dict = {
            "action": "new_tab()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.last_agent_info))
        ans_dict = {
            "action": f"goto('{url}')",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.last_agent_info))
        return None

    def close_page(self):
        ans_dict = {
            "action": "tab_close()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.last_agent_info))
        return None
    
    def search_on_page(self, url:str, search_text:str):
        self.open_page(url)
        return self.generic_action(task_prompt=search_on_page_prompt(search_text))


    def add_to_cart(self, url:str, item_description:str):
        self.open_page(url)
        return self.generic_action(task_prompt=add_to_cart_prompt(item_description))


    def generic_action(self, *args, **kwargs):
        n_steps = 0
        while n_steps < 10:
            logger.debug(f"generic_action step {n_steps}: Entering blocking observation queue.get()")
            # wait for previous actions to be consumed in the main thread
            while not self.action_queue.empty() or len(self.obs_history) == 0 or len(self.actions) >= len(self.obs_history):
                time.sleep(1.0)
            while True:
                obs = self.observation_queue.get()
                logger.debug("retrieved observation from queue, queue is empty: %s", self.observation_queue.empty())
                #self.obs_history.append(obs)
                if self.observation_queue.empty():
                    break
            n_steps += 1
            
            task_prompt_text = kwargs.get("task_prompt", "")
            kwargs_copy = deepcopy(kwargs)
            kwargs_copy.pop("task_prompt")
            logger.debug("task_prompt: %s", task_prompt_text)
            # this forces the task prompt to be the last message in the chat history
            #self.obs_history.append({"chat_messages": [{"role": "user", "text": task_prompt_text}]})

            system_prompt = SystemMessage(dp.SystemPrompt().prompt)
            logger.debug(f"actions: {len(self.actions)}")
            logger.debug(f"observation history: {len(self.obs_history)}")
            for a in self.actions:
                logger.debug(f"Final action: {str(a)[0:20]}")
            for o in self.obs_history:
                logger.debug(f"observation: {str(o)[0:20]}")

            main_prompt = ExecutorSystemPrompt(
                    self.executor_action_set,
                    goal=task_prompt_text,
                    obs_history=self.obs_history,
                    actions=self.actions,
                    memories=self.memories,
                    thoughts=self.thoughts,
                    previous_plan=self.plan,
                    step=self.plan_step,
                    flags=self.flags,
                    )
            logger.debug("main_prompt: %s", main_prompt.prompt)

            max_prompt_tokens, max_trunc_itr = self._get_maxes()

            human_prompt = dp.fit_tokens(
                shrinkable=main_prompt,
                max_prompt_tokens=max_prompt_tokens,
                model_name=self.executor_model_args.model_name,
                max_iterations=max_trunc_itr,
                additional_prompts=[f"\n<your task>\n{task_prompt_text}\n</your task>"],
            )

            try:
                chat_messages = Discussion([system_prompt, human_prompt])
                ans_dict = retry(
                    self.executor_llm,
                    chat_messages,
                    n_retry=self.max_retry,
                    parser=main_prompt._parse_answer,
                )
                ans_dict["busted_retry"] = 0
                # inferring the number of retries, TODO: make this less hacky
                ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
            except ParseError as e:
                ans_dict = dict(
                    action=None,
                    n_retry=self.max_retry + 1,
                    busted_retry=1,
                )

            stats = self.executor_llm.get_stats()
            stats["n_retry"] = ans_dict["n_retry"]
            stats["busted_retry"] = ans_dict["busted_retry"]

            agent_info = AgentInfo(
                think=ans_dict.get("think", None),
                chat_messages=chat_messages,
                stats=stats,
                extra_info={"executor_model_args": asdict(self.executor_model_args),# "eco_logits": eco_impacts.dict()
                },
            )
            self.last_agent_info = agent_info
            self.action_queue.put((ans_dict, agent_info))
        
        return "Failure"

