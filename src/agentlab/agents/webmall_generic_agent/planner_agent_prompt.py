"""
Prompt builder for GenericAgent

It is based on the dynamic_prompting module from the agentlab package.
"""

import logging
from dataclasses import dataclass

from browsergym.core import action
from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage, parse_html_tags_raise


@dataclass
class PlannerPromptFlags(dp.Flags):
    """
    A class to represent various flags used to control features in an application.

    Attributes:
        use_plan (bool): Ask the LLM to provide a plan.
        use_criticise (bool): Ask the LLM to first draft and criticise the action before producing it.
        use_thinking (bool): Enable a chain of thoughts.
        use_concrete_example (bool): Use a concrete example of the answer in the prompt for a generic task.
        use_abstract_example (bool): Use an abstract example of the answer in the prompt.
        use_hints (bool): Add some human-engineered hints to the prompt.
        enable_chat (bool): Enable chat mode, where the agent can interact with the user.
        max_prompt_tokens (int): Maximum number of tokens allowed in the prompt.
        be_cautious (bool): Instruct the agent to be cautious about its actions.
        extra_instructions (Optional[str]): Extra instructions to provide to the agent.
        add_missparsed_messages (bool): When retrying, add the missparsed messages to the prompt.
        flag_group (Optional[str]): Group of flags used.
    """

    obs: dp.ObsFlags
    action: dp.ActionFlags
    use_plan: bool = True  #
    use_criticise: bool = False  #
    use_thinking: bool = False
    use_memory: bool = False  #
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    max_trunc_itr: int = 20
    flag_group: str = None


class ExecutorSystemPrompt(dp.Shrinkable):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: PlannerPromptFlags,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = dp.GoalInstructions(
                obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(
            obs_history[-1],
            self.flags.obs,
        )

        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

        def time_for_caution():
            # no need for caution if we're in single action mode
            return flags.be_cautious and (
                flags.action.action_set.multiaction or flags.action.action_set == "python"
            )

        self.be_cautious = dp.BeCautious(visible=time_for_caution)
        self.think = dp.Think(visible=lambda: flags.use_thinking)
        self.hints = dp.Hints(visible=lambda: flags.use_hints)
        #self.plan = Plan(previous_plan, step, lambda: flags.use_plan)  # TODO add previous plan
        #self.criticise = Criticise(visible=lambda: flags.use_criticise)
#        self.memory = Memory(visible=lambda: flags.use_memory)

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
"""
#{self.be_cautious.prompt}\
#{self.think.prompt}\
#{self.plan.prompt}\
#{self.memory.prompt}\
#{self.criticise.prompt}\
#"""
        )

        __prompt = """You are an expert web navigator. Your task is to choose the most appropriate Python function to call for the given webpage content and task.
"""

        if self.flags.use_abstract_example:
            prompt.add_text(__prompt)
        
        return self.obs.add_screenshot(prompt)


    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        try:
            ans_dict = super()._parse_answer(text_answer)
        except Exception as e:
            return text_answer


class PlannerSystemPrompt(dp.Shrinkable):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict] = [],
        actions: list[str] = [],
        memories: list[str] = [],
        thoughts: list[str] = [],
        previous_plan: str = "",
        step: int = 0,
        flags: PlannerPromptFlags = None,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = dp.PlannerGoalInstructions(
                obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(
            obs_history[-1],
            self.flags.obs,
        )

        self.action_prompt = dp.PlannerActionPromptElement(action_set, action_flags=flags.action)

        #def time_for_caution():
            ## no need for caution if we're in single action mode
            #return flags.be_cautious and (
                #flags.action.action_set.multiaction or flags.action.action_set == "python"
            #)

        #self.be_cautious = dp.BeCautious(visible=time_for_caution)
        #self.think = dp.Think(visible=lambda: flags.use_thinking)
        #self.hints = dp.Hints(visible=lambda: flags.use_hints)
        #self.plan = Plan(previous_plan, step, lambda: flags.use_plan)  # TODO add previous plan
        #self.criticise = Criticise(visible=lambda: flags.use_criticise)
        #self.memory = Memory(visible=lambda: flags.use_memory)
    

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}
"""
        )

        return self.obs.add_screenshot(prompt)


    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        if text_answer.startswith('`') or text_answer.endswith('`'):
            text_answer = text_answer.strip('`')
        
        try:
            compile(text_answer, '<string>', 'exec')
        except Exception as e:
            print(e)
        
        ans_dict = {}
        ans_dict['plan'] = text_answer
        return ans_dict

    
# note **Managing tabs**
#new_tab() Open a new tab.
#tab_close() Close the current tab.. maps to close_page
#tab_focus(index) Bring a tab to front (activate tab).

#**Navigating the browser**
#go_back() Navigate to the previous page in history. in high level plan
#go_forward() Navigate to the next page in history. in high level plan
#goto(url) Navigate to a url


class Memory(dp.PromptElement): # Do not use
    _prompt = ""  # provided in the abstract and concrete examples

    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions. Some tasks require to
remember hints from previous steps in order to solve it.
</memory>
"""

    _concrete_ex = """
<memory>
I clicked on bid "32" to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
</memory>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["memory"], merge_multiple=True)


class Plan(dp.PromptElement):
    def __init__(self, previous_plan, plan_step, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self.previous_plan = previous_plan
        self._prompt = f"""
# Plan:

You just executed step {plan_step} of the previously proposed plan:\n{previous_plan}\n
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary.
"""

    _abstract_ex = """
<plan>
Provide a multi step plan that will guide you to accomplish the goal. There
should always be steps to verify if the previous action had an effect. The plan
can be revisited at each steps. Specifically, if there was something unexpected.
The plan should be cautious and favor exploring befor submitting.
</plan>

<step>Integer specifying the step of current action
</step>
"""

    _concrete_ex = """
<plan>
1. fill form (failed)
    * type first name
    * type last name
2. Try to activate the form
    * click on tab 2
3. fill form again
    * type first name
    * type last name
4. verify and submit
    * verify form is filled
    * submit if filled, if not, replan
</plan>

<step>2</step>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["plan", "step"])


class Criticise(dp.PromptElement):
    _prompt = ""

    _abstract_ex = """
<action_draft>
Write a first version of what you think is the right action.
</action_draft>

<criticise>
Criticise action_draft. What could be wrong with it? Enumerate reasons why it
could fail. Did your past actions had the expected effect? Make sure you're not
repeating the same mistakes.
</criticise>
"""

    _concrete_ex = """
<action_draft>
click("32")
</action_draft>

<criticise>
click("32") might not work because the element is not visible yet. I need to
explore the page to find a way to activate the form.
</criticise>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["action_draft", "criticise"])
