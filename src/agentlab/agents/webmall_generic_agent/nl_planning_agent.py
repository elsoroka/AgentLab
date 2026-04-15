"""
NlPlanningAgent implementation for AgentLab

This module defines a `NlPlanningAgent` class and its associated arguments for use in the AgentLab framework. \
The `NlPlanningAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `NlPlanningAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

import logging
from copy import deepcopy

from Browsergym.browsergym.experiments.src.browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS

logger = logging.getLogger(__name__)
from dataclasses import asdict, dataclass
from warnings import warn

import bgym
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, HumanMessage, ParseError, SystemMessage, parse_html_tags_raise, retry as llm_retry

from agentlab.llm.tracking import cost_tracker_decorator


from .generic_agent_prompt import GenericPromptFlags, MainPrompt
from .nl_planning_agent_prompt import NlPlanningStepPrompt

@dataclass
class NlPlanningAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"NlPlanningAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["nlplannerhighlevel"])
        # use nlplannerwebarena

        # for backward compatibility with old traces
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        # verify if we can remove this
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return NlPlanningAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class NlPlanningAgent(Agent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
    ):

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        self._check_flag_constancy()
        self.reset(seed=None)

        self.full_obs_history = []
        self.full_action_history = []
        self.full_memories = []
        self.full_thoughts = []

    def _generate_nl_plan(self, obs) -> list[str]:
        """Call the LLM once to produce a high-level natural language plan for the task."""
        goal_object = obs.get("goal_object", [{"type": "text", "text": str(obs.get("goal", ""))}])
        system_prompt = SystemMessage(dp.NlPlanningSystemPrompt().prompt)
        human_prompt = HumanMessage(dp.NlPlanGoalPrompt(goal_object).prompt)
        chat_messages = Discussion([system_prompt, human_prompt])

        def parse_plan(text):
            try:
                result = parse_html_tags_raise(text, keys=["plan"])
                steps = [line.strip() for line in result["plan"].splitlines() if line.strip()]
                return {"plan": steps}
            except ParseError:
                logger.error(f"_generate_nl_plan: failed to parse plan: {text}")
                return {"plan": [text]}

        ans_dict = llm_retry(
            self.chat_llm,
            chat_messages,
            n_retry=self.max_retry,
            parser=parse_plan,
        )
        logger.info(f"_generate_nl_plan: plan is {ans_dict.get('plan', 'No plan generated')}")
        return ans_dict.get("plan", "No plan generated")

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):

        self.obs_history.append(obs)

        # On the first step, generate a high-level NL plan before acting.
        if self.high_level_plan is None:
            self.high_level_plan = self._generate_nl_plan(obs)
            self.plan = self.high_level_plan[0]
            self.plan_step = 0

        current_step_idx = max(0, min(self.plan_step, len(self.high_level_plan) - 1))
        main_prompt = NlPlanningStepPrompt(
            action_set=self.action_set,
            current_step=self.high_level_plan[current_step_idx],
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = llm_retry(
                self.chat_llm,
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

        # did we get the next_step action?
        if ans_dict["action"] is not None:
            for line in ans_dict["action"].split("\n"):
                line = line.strip()
                if line.startswith("go_to_next_step("):
                    notes = line.split("(")[1].split(")")[0]
                    self.notes_from_previous_step.append(notes)

                    self.plan_step += 1

                    # clear and save histories
                    self.full_action_history += self.actions
                    self.full_obs_history += self.obs_history
                    self.obs_history = self.obs_history[-1:]
                    self.actions = []
                    self.full_memories += self.memories
                    self.full_thoughts += self.thoughts
                    self.memories = self.notes_from_previous_step
                    self.thoughts = []
                    
                    logger.info(f"go_to_next_step: notes from previous step: {notes}")
                    logger.info(f"next step is {self.plan_step}/{len(self.high_level_plan)}: {self.high_level_plan[self.plan_step]}")

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)

        step_val = ans_dict.get("step", self.plan_step)
        try:
            new_step = int(step_val)
        except (ValueError, TypeError):
            new_step = self.plan_step
        # If the step advanced, seed self.plan from the next high-level step
        if new_step != self.plan_step and self.high_level_plan is not None:
            new_idx = max(0, min(new_step, len(self.high_level_plan) - 1))
            self.plan = self.high_level_plan[new_idx]
        self.plan_step = new_step
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.high_level_plan = None
        self.notes_from_previous_step = []
        self.plan = "No plan yet"
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
            if not self.chat_model_args.vision_support:
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
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr
