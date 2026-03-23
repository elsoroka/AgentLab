from AgentLab.src.agentlab.agents.dynamic_prompting import PromptElement


def search_on_page_prompt(search_text: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""Find the search bar on the current page and search for the text: {search_text}.
    Select the best-matching result and return its URL as a string using the syntax
    <action>
    report_result(url="https://www.example.com")
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p

def navigate_to_page_prompt(description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, navigate to the page described by the following description: {description}.
    When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p

def extract_information_from_page_prompt(description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, extract the information described by the following description: {description}.
    Return the information as a string.
    When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p

def fill_text_field_prompt(field_description: str, text: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, find the text field described by the following description: {field_description} with the following text: {text}.
When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p

def press_button_prompt(button_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, press the button described by the following description: {button_description}.
When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p

def select_option_prompt(bid: str, options: str | list[str]) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, select the option described by the following description: {bid} with the following options: {options}.
    When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p

def add_to_cart_prompt(item_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""If the current page describes a product for sale, add it to the cart. If there are multiple variants of the product, use the following description to choose one: {item_description}.
    When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p

def checkout_prompt(payment_and_shipping_information: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, check out the items in the cart. Provide the following payment and shipping information: {payment_and_shipping_information}.
When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
"""
    return p