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
from typing import Optional, Union
from re import S
import time, json
from warnings import warn
import functools
import logging
import bgym

logger = logging.getLogger(__name__)

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
import threading

@dataclass
class PlanningAgentArgs(AgentArgs):
    planner_model_args: BaseModelArgs = None
    executor_model_args: BaseModelArgs = None
    flags: PlannerPromptFlags = None
    max_retry: int = 1
    max_steps: int = 50 # 50 matches the WebMall paper
    plan_from_file = None

    def __post_init__(self):
        self.keyed_plans = dict()

        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"PlanningAgent-{self.planner_model_args.model_name}-{self.executor_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass
            
        if self.plan_from_file:
            self.load_plan_from_file(self.plan_from_file)

    def load_plan_from_file(self, plan_from_file: str):
        with open(plan_from_file, 'r') as file:
            data = [json.loads(line) for line in file.readlines()]
        if data[0].keys() != data[1].keys():
            # data [0] is config
            data = data[1:]
        
        self.keyed_plans = dict()
        for plan in data:
            self.keyed_plans[plan['id']] = plan['clean_response'] if 'clean_response' in plan else None
        

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True
        
        """Override the action set for the planner, keeping the original low level action set for the executor."""
        self.flags.action.planner_action_set = deepcopy(DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["plannerhighlevel"])

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["executorwebarena"])

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
            max_steps=self.max_steps,
            keyed_plans=self.keyed_plans if len(self.keyed_plans) > 0 else None
        )


class PlanningAgent(Agent):

    def __init__(
        self,
        planner_model_args: BaseModelArgs,
        executor_model_args: BaseModelArgs,
        flags: PlannerPromptFlags,
        max_retry: int = 2,
        max_steps: int = 50,
        keyed_plans: dict = None,
    ):
        self.plan = None
        self.plan_step = 0
        self.keyed_plans = keyed_plans

        if not self.keyed_plans:
            self.planner_llm = planner_model_args.make_model()
        
        self.executor_llm = executor_model_args.make_model()
        self.planner_model_args = planner_model_args
        self.executor_model_args = executor_model_args
        self.max_retry = max_retry
        self.max_steps = max_steps
        self.get_action_count = 0
        self.keyed_plans = keyed_plans

        self.flags = flags
        if not self.keyed_plans:
            self.planner_action_set = self.flags.action.planner_action_set.make_action_set()
        self.executor_action_set = self.flags.action.action_set.make_action_set()

        self._obs_preprocessor = dp.make_obs_preprocessor(self.flags.obs)

        # Stop event: set when the step limit is reached so the executor thread can exit cleanly.
        self._stop_event = threading.Event()

        self._check_flag_constancy()
        self.reset(seed=None)

        # Executor management
        self.action_queue = Queue()
        self.observation_queue = Queue()

        # history of actions etc. when we reset them for each executor task
        self.all_actions = []
        self.all_memories = []
        self.all_thoughts = []
        self.all_obs_history = []

        self.dummy_agent_info = AgentInfo(
            think=None,
            chat_messages=None,
            stats=dict(
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_cost=0,
            ),
            extra_info={},
        )

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def extract_action(self, answer):
        answer = answer.strip()
        if '`' in answer:
            answer = answer.strip('`')
        return answer

    def make_and_start_plan(self, obs:dict):
        """Call the PlannerAgent to make the high-level plan in code to call the executor agent.
        The plan is run in a thread using exec().
        """
        self.obs_history.append(obs)
        # assumption: self.obs_history has at least one observation.
        assert(len(self.obs_history) > 0)

        model_args = self.planner_model_args
        try:
            # system prompt for the PLANNER agent which prompts it to make a plan, no action.
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

            if not self.keyed_plans:
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
                planner_stats = self.planner_llm.get_stats()
                self.plan = ans_dict["plan"]
            
            else:
                self.plan = self.keyed_plans[obs["task_id"]]
                # TODO fix we would need to import a tokenizer to count the tokens in the plan
                planner_stats = dict(
                    total_tokens=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_cost=0,
                )
            
            if "<plan>" in self.plan:
                self.plan = self.plan.split("<plan>")[1].split("</plan>")[0]
            if '```' in self.plan:
                self.plan = self.plan.strip('`')
            if self.plan.startswith('python'):
                self.plan = self.plan.split("python")[1]
            self.plan = self.plan.strip()
            self.plan_step = 0
            logger.info("plan: %s", self.plan)

            planner_stats["n_retry"] = self.max_retry + 1
            planner_stats["busted_retry"] = 1
            model_args = self.planner_model_args

            self.plan_step = ans_dict.get("step", self.plan_step)

            agent_info = AgentInfo(
                think=ans_dict.get("think", None),
                chat_messages=chat_messages,
                stats=planner_stats,
                extra_info={"planner_stats": planner_stats, "chat_model_args": asdict(model_args), #"eco_logits": eco_impacts.dict()
                },
            )

            self.obs_history.pop()

            # launch the plan in a thread
            self.executor_thread_pool = ThreadPoolExecutor(max_workers=1)
            self.executor_thread_pool.submit(self.execute_plan, self.plan)
            #self.waiting_for_action.clear()
            
        except Exception as e:
            logger.exception("Exception in planner: %s", e)
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
    
    def execute_plan(self, plan:str):
        # assumption: plan is valid Python code
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
                "get_open_tabs":self.get_open_tabs,
            })
        except Exception as e:
            # print the traceback
            logger.exception("Exception in executor: %s", e, exc_info=True)

        finished_action = {
            "action": "finished_plan()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((finished_action, self.dummy_agent_info))

    
    def safe_parse_int(self, value:Optional[str])->Union[int, float]:
        if value is None:
            return float("NaN")
        
        value = ''.join([v for v in value if v.isdigit()])
        try:
            return int(value)
        except ValueError:
            return float("NaN")
    
    def safe_parse_float(self, value:Optional[str])->float:
        if value is None or value == "":
            return None
        value = ''.join([v for v in value if v.isdigit() or v == '.' or v == ','])
        # is this one of them that swaps the use of a comma and a decimal point?
        if len(value.split(',')[-1]) == 2:
            if '.' in value:
                value = value.replace('.', '')
            value = value.replace(',', '.')
        
        try:
            return float(value)
        except ValueError:
            return None
    
    @cost_tracker_decorator
    def get_action(self, obs):
        """ The original get_action function
        To adapt this to our planner/executor model,
        the planner/executor consume observations from the observation queue and produce actions on the action queue.
        this function puts the new observation in the observation queue and consumes an action from the action queue.
        """
        if len(self.actions) > 0 or len(self.all_actions) > 0:
            try:
                self.action_queue.task_done() # corresponds to the previous action
            except Exception:
                pass

        self.observation_queue.put(obs)

        if self.plan is None:
            self.get_action_count = 0
            self.make_and_start_plan(obs)

        self.get_action_count += 1

        # Now the plan is running, so we get an action from the threaded executor
        logger.debug("mainloop: entering a blocking action_queue.get()")

        ans_dict, agent_info  = self.action_queue.get()

        if ans_dict.get("action", None) == "finished_plan()":
            self.plan = None
            self.plan_step = 0
            ans_dict['action'] = None
            logger.info("finished_plan() received, stopping executor thread.")

        # If we have reached the step limit, signal the executor thread to stop.
        # Call task_done for the action we just got (normally done at the start of the next
        # get_action call), then drain any buffered actions so the executor's action_queue.join()
        # unblocks and the thread can exit cleanly.
        if self.get_action_count >= self.max_steps:
            logger.info("Step limit (%d) reached; signalling executor to stop.", self.max_steps)
            self._stop_event.set()
            #try:
                #self.action_queue.task_done()
            #except Exception:
                #pass
            #while not self.action_queue.empty():
                #try:
                    #self.action_queue.get_nowait()
                    #self.action_queue.task_done()
                #except Exception:
                    #break

        self.executor_thread_pool.shutdown(wait=False)

        self.actions.append(ans_dict.get("action", None))
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        if hasattr(self, 'actions'):
            self.all_actions = self.actions
        if hasattr(self, 'memories'):
            self.all_memories = self.memories
        if hasattr(self, 'thoughts'):
            self.all_thoughts = self.thoughts
        if hasattr(self, 'obs_history'):
            self.all_obs_history = self.obs_history

        self.seed = seed
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []
        logger.info(f"{'='*20} OBS HISTORY AND ACTION HISTORY HAS BEEN RESET {'='*20}")

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

    def clean_and_parse_executor_action(self, raw_action:str)->Optional[str]:
        if "report_result" in raw_action:
            # strip off ' and " and extract the report_result("result")" string
            if '=' in raw_action:
                raw_action = raw_action.split("=")[1]
                
            else:
                raw_action = raw_action.split("(")[1].split(")")[0]
            raw_action = raw_action.split(")")[0].strip("""'" """)
            return raw_action
        elif 'done' in raw_action:
            return True
        elif 'report_infeasible' in raw_action:
            return False
        else:
            return None

    def get_open_tabs(self) -> list[str]:
        """Return URLs of all currently open tabs, based on the latest observation."""
        if self.obs_history:
            return list(self.obs_history[-1].get("open_pages_urls", []))
        if self.all_obs_history:
            return list(self.all_obs_history[-1].get("open_pages_urls", []))
        return []

    def noop(self):
        self.action_queue.join()
        self.observation_queue.join()
        # make ans_dict
        ans_dict = {
            "action": "noop()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        # TODO the self.last_agent_info is a stupid hack to stop the AgentLab framework from crashing.
        self.action_queue.put((ans_dict, self.dummy_agent_info))
        return None

    def go_back(self):
        self.action_queue.join()
        
        ans_dict = {
            "action": "go_back()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.dummy_agent_info))
        return None

    def go_forward(self):
        self.action_queue.join()
        ans_dict = {
            "action": "go_forward()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.dummy_agent_info))
        return None

    def open_page(self, url:str):
        self.action_queue.join()

        ans_dict = {
                "action": "new_tab()",
                "n_retry": 0,
                "busted_retry": 0,
            }
        self.action_queue.put((ans_dict, self.dummy_agent_info))
        ans_dict = {
            "action": f"goto('{url}')",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.dummy_agent_info))
        return True

    def close_page(self):
        ans_dict = {
            "action": f"tab_close()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((ans_dict, self.dummy_agent_info))
        return True
    

    def navigate_to_page(self, description:str):
        """Navigate to a page that fits the given description. Return True if successful, False otherwise.

        Examples:
        navigate_to_page("The home page of this website.")
        """
        self.action_queue.join()
        self.reset()
        final_result = None
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=navigate_to_page_prompt(description))

        #self.action_queue.join()
        
        if type(final_result) != bool:
            final_result = False
        logger.info(f"navigate_to_page({description}) returned {final_result}")
        return final_result

    

    def extract_information_from_page(self, description:str, _type:str="str"):
        """Extract text from the current page that fits the given description and matches the given type.
        Guaranteed to return a value of the given type or None if the information cannot be found.

        Examples:
        extract_information_from_page("The lowest price of the product.", float)
        """
        self.action_queue.join()
        final_result = None
        self.reset()
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=extract_information_from_page_prompt(description, _type))
        
        raw_result = final_result
        logger.info(f"extract_information_from_page({description}) returned raw result {raw_result}")
        if final_result is not None and final_result is not False:
            if _type == "int":
                final_result = self.safe_parse_int(final_result)
            elif _type == "float":
                final_result = self.safe_parse_float(final_result)
            elif _type == "str":
                final_result = str(final_result)
        
        else:
            final_result = None

        #self.action_queue.join()
        logger.info(f"extract_information_from_page({description}) returned {final_result} from raw result {raw_result}")
        return final_result

    def search_on_page(self, url:str=None, search_text:str=None, selection_criteria='', search_page_url:str=None)->Optional[list[str]]:
        """Open the search_page_url and search for the search_text. Return a list of page URLs that matche the selection criteria as a string, or None if not found.

        Examples:
        search_on_page("https://www.google.com", "Python")
        """
        # stupid hack here, we should really fix the planner to not do this
        if not url:
            url =  search_page_url
        self.action_queue.join()
        final_result = None
        self.reset()
        self.open_page(url)
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=search_on_page_prompt(search_text, selection_criteria))
        
        #self.action_queue.join()
        
        if type(final_result) != str:
            final_result = ''
        logger.info(f"search_on_page({url}, {search_text}, {selection_criteria}) returned {final_result}")
        return final_result



    def add_to_cart(self, url:str, item_description:str):
        """Add the product to the cart. Return True if successful, False otherwise.

        Examples:
        add_to_cart("product_url", "The product description") # returns True because this is a product page
        """
        self.action_queue.join()
        final_result = None
        self.reset()
        self.open_page(url)

        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=add_to_cart_prompt(item_description))
        
        #self.action_queue.join()
        if type(final_result) != bool:
            final_result = False
        logger.info(f"add_to_cart({item_description}) returned {final_result}")
        return final_result

    def checkout(self, payment_and_shipping_information:str):
        """Checkout from the current page. Return True if successful, False otherwise.

        Examples:
        checkout("A string containing payment information and shipping address") # while on a web shopping site with at least one item in the cart, returns True
        """
        self.action_queue.join()
        final_result = None
        self.reset()
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=checkout_prompt(payment_and_shipping_information))
        
        #self.action_queue.join()
        if type(final_result) != bool:
            final_result = False
        logger.info(f"checkout({payment_and_shipping_information}) returned {final_result}")
        return final_result


    def fill_text_field(self, field_description:str, text:str)->bool:
        """Fill the text field with the given text. Return True if successful, False otherwise.

        Examples:
        fill_text_field("The email field", "example@example.com")
        """
        self.action_queue.join()
        final_result = None
        self.reset()
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=fill_text_field_prompt(field_description, text))
        
        #self.action_queue.join()
        if type(final_result) != bool:
            final_result = False
        logger.info(f"fill_text_field({field_description}, {text}) returned {final_result}")
        return final_result
    

    def press_button(self, button_description:str)->bool:
        """Press the button with the given description. Return True if successful, False otherwise.

        Examples:
        press_button("The submit button")
        """
        self.action_queue.join()
        final_result = None
        self.reset()
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=press_button_prompt(button_description))
        
        #self.action_queue.join()
        if type(final_result) != bool:
            final_result = False
        logger.info(f"press_button({button_description}) returned {final_result}")
        return final_result
 

    def select_option(self, option_description:str)->bool:
        """Select the option with the given description. Return True if successful, False otherwise.

        Examples:
        select_option("Ground shipping")
        """
        self.action_queue.join()
        final_result = None
        self.reset()
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(task_prompt=select_option_prompt(option_description))
        
        #self.action_queue.join()
        if type(final_result) != bool:
            final_result = False
        logger.info(f"select_option({option_description}) returned {final_result}")
        return final_result
    


    def generic_action(self, *args, **kwargs):
        self.action_queue.join()
        final_result = None
        self.reset()
        while final_result is None and not self._stop_event.is_set():
            ans_dict, agent_info, final_result = self.generic_action_step(*args, **kwargs)
        
        #self.action_queue.join()
        return final_result
    

    def generic_action_step(self, *args, **kwargs):
        logger.debug("Entering blocking observation queue.get()")

        # Get at least one observation from the queue
        # We have to get at least one because otherwise we aren't waiting for the result of the previous action.
        # There can be more than one if the previous action was hardcoded, such as opening a tab or going to a URL.
        # wait for previous actions to be consumed in the main thread
        self.action_queue.join()

        if self._stop_event.is_set():
            logger.info("generic_action_step: stop event set, exiting.")
            return None, None, None

        while not self.observation_queue.empty():
            obs = self.observation_queue.get()
            self.obs_history.append(obs)
            self.observation_queue.task_done()

            logger.debug("retrieved observation from queue, queue is empty: %s", self.observation_queue.empty())

            if self.observation_queue.empty():
                break
    
        self.all_obs_history += self.obs_history[:-(len(self.actions)+1)]
        self.obs_history = self.obs_history[-(len(self.actions)+1):]

        # make sure there is always an observation, better to have an old one than none
        if len(self.obs_history) == 0:
            self.obs_history.append(self.all_obs_history[-1])
    
        # final_result is ONLY returned when the executor task is complete, meaning
        # done, report_result, or report_infeasible is present in the action (s).
        final_result = None

        task_prompt = kwargs.get("task_prompt", "")
        kwargs_copy = deepcopy(kwargs)
        kwargs_copy.pop("task_prompt")
        logger.debug("task_prompt: %s", task_prompt.prompt)

        # Another stupid hack
        # without this, we cannot remove the high-level goal ("find x product", "purchase x product", etc. from the prompt)
        # this replaces the high-level goal with the executor's subgoal (e.g. "search for x", "find y information on this page")
        last_obs = deepcopy(self.obs_history[-1])
        high_level_goal = last_obs.get("goal", "")
        self.obs_history[-1]['goal'] = task_prompt.prompt

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        logger.debug(f"actions: {len(self.actions)}")
        logger.debug(f"observation history: {len(self.obs_history)}")
        #for a in self.actions:
            #logger.debug(f"Final action: {str(a)[0:20]}")
        #for o in self.obs_history:
            #logger.debug(f"observation: {str(o)[0:20]}")
        logger.debug(f"task_prompt: {task_prompt.prompt}")
        main_prompt = ExecutorSystemPrompt(
                self.executor_action_set,
                goal=task_prompt,
                obs_history=self.obs_history,
                actions=self.actions,
                memories=self.memories,
                thoughts=self.thoughts,
                previous_plan=self.plan,
                step=self.plan_step,
                flags=self.flags,
                )
        #logger.debug("main_prompt: %s", main_prompt.prompt)

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.executor_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=[],
        )

        ans_dict = None
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

        # have we got a final result?
        # There can be more than one action here so we have to catch the rest of them and enqueue them
        # for example if we have the actions "report_result(url="url")\ntab_close()"" we want to set ans_dict['action'] to "tab_close()" and final_result to "url"
        clean_action = ''
        final_result = None
        if ans_dict['action'] is None:
            ans_dict['action'] = ''

        for line in ans_dict["action"].split("\n"):
            if "report_result" in line or "done" in line or "report_infeasible" in line:
                if final_result is None:
                    final_result = self.clean_and_parse_executor_action(line)
                # we have a common issue of done() and report_result() being used at the same time
                # so we need to save the result of report_result and return it instead of True if done() is also returned

            else:
                clean_action += line + "\n"

        ans_dict["action"] = clean_action

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

        if clean_action != '':
            self.action_queue.put((ans_dict, agent_info))
        self.obs_history[-1] = last_obs

        return ans_dict, agent_info, final_result

