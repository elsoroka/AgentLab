"""
Prompt builder for NlPlanningAgent.

The executor sees the current high-level plan step as its goal (via
ExecutorGoalInstructions) rather than the full task description, mirroring
the structure of ExecutorSystemPrompt used by the code-planning agent.
"""

from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.webmall_generic_agent.generic_agent_prompt import (
    GenericPromptFlags,
    MainPrompt,
    Plan,
    Memory,
    Criticise,
)
from agentlab.llm.llm_utils import HumanMessage


class NlPlanningStepPrompt(MainPrompt):
    """MainPrompt variant that shows the current plan step as the goal.

    Replaces GoalInstructions (full task description) with
    ExecutorGoalInstructions(current_step), so the executor is focused
    on exactly one high-level step at a time.
    """

    def __init__(
        self,
        action_set: AbstractActionSet,
        current_step: str,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: GenericPromptFlags,
    ) -> None:
        super().__init__(action_set, obs_history, actions, memories, thoughts, previous_plan, step, flags)
        # Override instructions: show the current plan step instead of the full goal
        self.instructions = dp.ExecutorGoalInstructions(
            current_step, extra_instructions=flags.extra_instructions
        )
