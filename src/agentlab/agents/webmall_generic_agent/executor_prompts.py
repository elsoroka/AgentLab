from AgentLab.src.agentlab.agents.dynamic_prompting import PromptElement


def search_on_page_prompt(search_text: str, selection_criteria="best match") -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""Search for the product {search_text} and gather the URLS for the pages that best match the selection criteria: {selection_criteria}. If more than one URL matches, return the URLS as a string separated by ###.
    Example:
    <action>
    report_result(url="https://www.example1.com###https://www.example2.com")
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If search fails two times in a row, return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p

def navigate_to_page_prompt(description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""Starting from the current page, navigate to the page described by the following description: {description}.
    When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If navigation fails two times in a row, return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p

def extract_information_from_page_prompt(description: str, _type:str="str") -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, extract the information described by the following description: {description} and return it as a {_type} using the function report_result() If the information is not on this page, return report_infeasible().
    If you return a number, make sure to remove all non-numeric characters and punctuation first.
    When finished, return
    <action>
    report_result(result)
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If extraction fails two times in a row, return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p

def fill_text_field_prompt(field_description: str, text: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, find the text field described by the following description: {field_description} with the following text: {text}. If the text field is not on this page, report that the action is not possible.
When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If filling the text field fails two times in a row, return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p

def press_button_prompt(button_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, press the button described by the following description: {button_description}. If the button is not on this page, report that the action is not possible.
When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If pressing the button fails two times in a row, return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p

def select_option_prompt(bid: str, options: str | list[str]) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, select the option described by the following description: {bid} with the following options: {options}. If the option is not on this page, report that the action is not possible.
    When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If selecting the option fails two times in a row, return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p

def add_to_cart_prompt(item_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""If the current page describes a product for sale, add it to the cart. If there are multiple variants of the product, use the following description to choose one: {item_description}. If the product is not on this page, report that the action is not possible.
    When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If adding to the cart fails two times in a row return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p

def checkout_prompt(payment_and_shipping_information: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, check out the items in the cart. Provide the following payment and shipping information: {payment_and_shipping_information}.
    Finally, finish the checkout process.
When finished, return
    <action>
    done()
    </action>
    If the action is not possible, return
    <action>
    report_infeasible()
    </action>
    If checkout fails two times in a row, return report_infeasible(). Don't keep trying the same thing over and over.
"""
    return p