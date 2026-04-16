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


class NotesFromPreviousStep(dp.PromptElement):
    """Shows accumulated notes passed via go_to_next_step() from prior plan steps."""

    def __init__(self, notes: list[str]) -> None:
        super().__init__(visible=bool(notes))
        if notes:
            formatted = "\n".join(f"- {n}" for n in notes)
            self._prompt = f"\n# Notes from previous steps\n{formatted}\n"
        else:
            self._prompt = ""

    def _parse_answer(self, text_answer):
        return {}


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
        notes_from_previous_step: list[str] = None,
    ) -> None:
        super().__init__(action_set, obs_history, actions, memories, thoughts, previous_plan, step, flags)
        # Override instructions: show the current plan step instead of the full goal
        self.instructions = dp.ExecutorGoalInstructions(
            current_step, extra_instructions=flags.extra_instructions
        )
        self.notes = NotesFromPreviousStep(notes_from_previous_step or [])

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.notes.prompt}\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.hints.prompt}\
{self.be_cautious.prompt}\
{self.think.prompt}\
{self.plan.prompt}\
{self.memory.prompt}\
{self.criticise.prompt}\
"""
        )

        if self.flags.use_abstract_example:
            prompt.add_text(
                f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.think.abstract_ex}\
{self.plan.abstract_ex}\
{self.memory.abstract_ex}\
{self.criticise.abstract_ex}\
{self.action_prompt.abstract_ex}\
"""
            )

        if self.flags.use_concrete_example:
            prompt.add_text(
                f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.think.concrete_ex}\
{self.plan.concrete_ex}\
{self.memory.concrete_ex}\
{self.criticise.concrete_ex}\
{self.action_prompt.concrete_ex}\
"""
            )
        return self.obs.add_screenshot(prompt)

