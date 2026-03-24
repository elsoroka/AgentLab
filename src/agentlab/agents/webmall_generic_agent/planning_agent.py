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
from re import S
import time
from warnings import warn
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
import threading

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
        self.max_executor_steps = 10  # max LLM steps per sub-task before giving up

        self._obs_preprocessor = dp.make_obs_preprocessor(self.flags.obs)

        self._check_flag_constancy()
        self.reset(seed=None)

        # Queues for executor <-> main loop communication.
        # Design: each action put in action_queue is matched by one observation put in
        # observation_queue (by get_action after BrowserGym steps the env).
        # _execute_one_action() puts one action and then blocks waiting for the resulting obs.
        # This serializes the action→obs cycle without any join()/Event() gymnastics.
        self.action_queue = Queue()
        self.observation_queue = Queue()
        self.last_obs = None  # most recent obs, for sub-tasks that call reset() before generic_action_step

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def extract_action(self, answer):
        answer = answer.strip()
        if '`' in answer:
            answer = answer.strip('`')
        return answer


    def make_and_start_plan(self, obs:dict):
        self.obs_history.append(obs)
        # assumption: self.obs_history has at least one observation.
        assert(len(self.obs_history) > 0)

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

            agent_info = AgentInfo(
                think=ans_dict.get("think", None),
                chat_messages=chat_messages,
                stats=stats,
                extra_info={"planner_stats": planner_stats, "chat_model_args": asdict(model_args)},
            )

            self.last_agent_info = agent_info
            self.obs_history.pop()

            # launch the plan in a thread
            self.executor_thread_pool = ThreadPoolExecutor(max_workers=1)
            self.executor_thread_pool.submit(self.execute_plan, self.plan)

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
            })
        except Exception as e:
            # print the traceback
            logger.exception("Exception in executor: %s", e, exc_info=True)

        # Signal the main loop that this plan is done. Reset plan so get_action will re-plan.
        # Put the sentinel directly (no obs wait) — main loop returns noop, then re-plans.
        self.plan = None
        finished_action = {
            "action": "noop()",
            "n_retry": 0,
            "busted_retry": 0,
        }
        self.action_queue.put((finished_action, self.last_agent_info))

    # -------------------------------------------------------------------------
    # Core synchronization primitive: put ONE action, wait for ONE observation.
    # Every browser action goes through here, ensuring the executor always sees
    # the fresh post-action observation before deciding what to do next.
    # -------------------------------------------------------------------------

    # Prefixes that identify a line as a meta-action (a signal to the planning loop,
    # not a real BrowserGym browser command). We match on the START of the stripped
    # line to avoid false-positives when these words appear inside string arguments
    # (e.g., fill('11', 'Infeasible: report_infeasible()')).
    _META_PREFIXES = ('report_result(', 'report_infeasible(', 'done()')

    def _is_meta_action_line(self, line: str) -> bool:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in self._META_PREFIXES):
            return True
        # Handle assignment form: result = report_result(...)
        if '=' in stripped and 'report_result(' in stripped:
            lhs, _, rhs = stripped.partition('=')
            if rhs.strip().startswith('report_result('):
                return True
        return False

    def _to_browser_action(self, action_str: str) -> str:
        """Strip meta-action lines (done/report_result/report_infeasible) from an action string.

        Meta-actions are signals from the executor LLM to the planning loop — they are NOT
        valid BrowserGym actions. Sending them to the browser would prematurely end the episode
        (done()) or cause errors. We keep them in ans_dict["action"] for loop termination
        detection while sending only real browser actions to the environment.

        Detection uses startswith on the stripped line so that meta-keyword strings appearing
        inside fill() arguments do NOT cause the fill line to be incorrectly stripped.
        """
        if action_str is None:
            return 'noop()'
        lines = [
            l for l in action_str.split('\n')
            if l.strip() and not self._is_meta_action_line(l)
        ]
        return '\n'.join(lines) if lines else 'noop()'

    def _execute_one_action(self, action_str: str, agent_info=None) -> dict:
        """Put one browser action in the queue and block until the resulting observation arrives.

        Returns the observation dict that resulted from executing the action.
        """
        ans_dict = {
            "action": action_str,
            "n_retry": 0,
            "busted_retry": 0,
        }
        info = agent_info if agent_info is not None else self.last_agent_info
        self.action_queue.put((ans_dict, info))
        obs = self.observation_queue.get()
        self.observation_queue.task_done()
        self.last_obs = obs
        return obs

    #@cost_tracker_decorator
    def get_action(self, obs):
        """Called by BrowserGym each step. Delivers the new obs to the executor thread
        and retrieves the next action to execute."""

        if self.plan is None:
            # First call (or after plan completion): start a new plan.
            # Don't put obs in the queue yet — the executor hasn't issued its first
            # action yet, so there is nothing waiting for an obs.
            self.make_and_start_plan(obs)
        else:
            # Deliver the obs that resulted from the previous action so the executor
            # (blocked in _execute_one_action) can continue.
            self.observation_queue.put(obs)

        logger.debug("mainloop: blocking on action_queue.get()")
        result = self.action_queue.get()
        ans_dict, agent_info = result

        self.actions.append(ans_dict.get("action", None))
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
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


    # -------------------------------------------------------------------------
    # Executor action primitives — each issues browser actions and waits for obs.
    # These run in the executor thread, called from within exec(plan, ...).
    # -------------------------------------------------------------------------

    def clean_and_parse_executor_action(self, raw_action: str):
        """Parse the executor LLM's termination signal and return a Python value.

        Returns:
          str  — the extracted result string (report_result)
          True — task completed successfully (done)
          None — task is infeasible on this page/store (report_infeasible)
          False — action was not a termination signal (should not reach callers)
        """
        if "report_result" in raw_action:
            # Extract the argument from report_result(url="...") or report_result("text")
            if '=' in raw_action:
                raw_action = raw_action.split("=")[1]
            else:
                raw_action = raw_action.split("(")[1].split(")")[0]
            raw_action = raw_action.split(")")[0].strip("""'" """)
            return raw_action
        elif 'done' in raw_action:
            return True
        elif 'report_infeasible' in raw_action:
            # Return None so callers can use `if result:` to detect infeasibility.
            # Previously returned a truthy "Infeasible: ..." string which caused
            # infeasible stores to be appended to offer_urls in the plan.
            return None
        else:
            return False

    def noop(self):
        obs = self._execute_one_action("noop()")
        self.obs_history.append(obs)

    def go_back(self):
        obs = self._execute_one_action("go_back()")
        self.obs_history.append(obs)

    def go_forward(self):
        obs = self._execute_one_action("go_forward()")
        self.obs_history.append(obs)

    def open_page(self, url:str):
        """Open url in a new tab. Appends the post-navigation obs to obs_history."""
        self._execute_one_action("new_tab()")   # discard intermediate obs
        obs = self._execute_one_action(f"goto('{url}')")
        self.obs_history.append(obs)

    def close_page(self):
        obs = self._execute_one_action("tab_close()")
        self.obs_history.append(obs)


    def navigate_to_page(self, description:str):
        """Navigate to a page that fits the given description. Return True if successful, False otherwise.

        Examples:
        navigate_to_page("The home page of this website.")
        """
        self.reset()
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=navigate_to_page_prompt(description))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("navigate_to_page FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("navigate_to_page CONTINUING: %s", ans_dict["action"])
        logger.warning("navigate_to_page exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def extract_information_from_page(self, description:str):
        """Extract text from the current page that fits the given description. Return the text as a string.

        Examples:
        extract_information_from_page("The lowest price of the product.")
        """
        self.reset()
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=extract_information_from_page_prompt(description))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("extract_information_from_page FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("extract_information_from_page CONTINUING: %s", ans_dict["action"])
        logger.warning("extract_information_from_page exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def search_on_page(self, url:str, search_text:str):
        """Navigate to url, search for search_text, and return the product page URL if an exact match is found, else None.

        IMPORTANT: Do NOT call open_page(url) before this — search_on_page handles navigation internally.
        Do NOT call extract_information_from_page after this — the returned URL IS the product page.

        Examples:
        url = search_on_page("http://localhost:8081", "Asus ROG Ryujin II 360mm")
        if url:
            offer_urls.append(url)
        """
        self.reset()
        self.open_page(url)
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=search_on_page_prompt(search_text))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("search_on_page FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("search_on_page CONTINUING: %s", ans_dict["action"])
        logger.warning("search_on_page exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def add_to_cart(self, url:str, item_description:str):
        """Add the product to the cart. Return True if successful, False otherwise.

        Examples:
        add_to_cart("product_url", "The product description") # returns True because this is a product page
        """
        self.reset()
        self.open_page(url)

        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=add_to_cart_prompt(item_description))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("add_to_cart FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("add_to_cart CONTINUING: %s", ans_dict["action"])
        logger.warning("add_to_cart exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def checkout(self, payment_and_shipping_information:str):
        """Checkout from the current page. Return True if successful, False otherwise.

        Examples:
        checkout("A string containing payment information and shipping address") # while on a web shopping site with at least one item in the cart, returns True
        """
        self.reset()
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=checkout_prompt(payment_and_shipping_information))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("checkout FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("checkout CONTINUING: %s", ans_dict["action"])
        logger.warning("checkout exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def fill_text_field(self, field_description:str, text:str)->bool:
        """Fill the text field with the given text. Return True if successful, False otherwise.

        Examples:
        fill_text_field("The email field", "example@example.com")
        """
        self.reset()
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=fill_text_field_prompt(field_description, text))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("fill_text_field FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("fill_text_field CONTINUING: %s", ans_dict["action"])
        logger.warning("fill_text_field exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def press_button(self, button_description:str)->bool:
        """Press the button with the given description. Return True if successful, False otherwise.

        Examples:
        press_button("The submit button")
        """
        self.reset()
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=press_button_prompt(button_description))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("press_button FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("press_button CONTINUING: %s", ans_dict["action"])
        logger.warning("press_button exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def select_option(self, option_description:str)->bool:
        """Select the option with the given description. Return True if successful, False otherwise.

        Examples:
        select_option("Ground shipping")
        """
        self.reset()
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(task_prompt=select_option_prompt(option_description))
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("select_option FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("select_option CONTINUING: %s", ans_dict["action"])
        logger.warning("select_option exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def generic_action(self, *args, **kwargs):
        self.reset()
        for _step in range(self.max_executor_steps):
            ans_dict, agent_info = self.generic_action_step(*args, **kwargs)
            ans_dict["action"] = str(ans_dict["action"])
            if "report_result" in ans_dict["action"] or "done" in ans_dict["action"] or "report_infeasible" in ans_dict["action"]:
                logger.debug("generic_action FINISHING: %s", ans_dict["action"])
                return self.clean_and_parse_executor_action(ans_dict["action"])
            logger.debug("generic_action CONTINUING: %s", ans_dict["action"])
        logger.warning("generic_action exceeded max_executor_steps (%d), giving up", self.max_executor_steps)
        return None


    def generic_action_step(self, *args, **kwargs):
        """Ask the executor LLM what to do next, execute the action, and wait for the obs.

        Uses obs_history[-1] (the current page state) for the LLM prompt.
        If obs_history is empty (e.g. after reset() without a preceding open_page()),
        falls back to self.last_obs.
        """
        # Ensure we have at least one observation in history for the LLM prompt.
        if not self.obs_history:
            if self.last_obs is not None:
                self.obs_history.append(self.last_obs)
            else:
                raise RuntimeError("generic_action_step called with no observation available")

        task_prompt = kwargs.get("task_prompt", "")
        kwargs_copy = deepcopy(kwargs)
        kwargs_copy.pop("task_prompt", None)
        logger.debug("task_prompt: %s", task_prompt)
        logger.debug("actions: %d  obs_history: %d", len(self.actions), len(self.obs_history))

        # Temporarily override the goal in the last obs for the LLM prompt, then restore it.
        last_obs_backup = deepcopy(self.obs_history[-1])
        self.obs_history[-1]['goal'] = task_prompt.prompt

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

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
            extra_info={"executor_model_args": asdict(self.executor_model_args)},
        )
        self.last_agent_info = agent_info

        # Restore the obs before appending the new one.
        self.obs_history[-1] = last_obs_backup

        # Execute the action and get the resulting observation.
        # Strip meta-actions (done/report_result/report_infeasible) from the browser action —
        # these are loop-termination signals for the executor, not real browser commands.
        browser_action = self._to_browser_action(ans_dict["action"])
        new_obs = self._execute_one_action(browser_action, agent_info)
        self.obs_history.append(new_obs)

        return ans_dict, agent_info
