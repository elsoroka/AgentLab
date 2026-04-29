from AgentLab.src.agentlab.agents.dynamic_prompting import PromptElement


def search_on_page_prompt(search_text: str, selection_criteria="best match", website_url: str = 'https://localhost:8081') -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""Search for the product {search_text} on the website {website_url} and gather the URLS for the pages that best match the selection criteria: {selection_criteria}. If more than one URL matches the selection criteria, return the URLS as a string separated by ###.
    Example: url="https://www.example1.com###https://www.example2.com"
    Hints:
    * Pay close attention to the requirements in the task description.
"""
    return p

def navigate_to_page_prompt(description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""Starting from the current page, navigate to the page described by: {description}.
"""
    return p

def extract_information_from_page_prompt(description: str, url: str, _type:str="str") -> str:
    p = PromptElement(visible=True)
    page_ref = f"the page at {url}"
    p._prompt = f"""On {page_ref}, extract the information described by the following description: {description} and return it as a {_type} using the function report_result() If the information is not on this page, return report_infeasible().
    If you return a number, make sure it can be parsed by Puthon `float()`. Example: $1,200.00 -> 1200.00
"""
    return p

def fill_text_field_prompt(field_description: str, text: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, find the text field described by: {field_description} with the text: {text}. If the text field is not on this page, report that the action is not possible.
"""
    return p

def press_button_prompt(button_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, press the button described by: {button_description}. If the button is not on this page, report that the action is not possible.
"""
    return p

def select_option_prompt(bid: str, options: str | list[str]) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, select the option described by: {bid} with the options: {options}. If the option is not on this page, report that the action is not possible.
"""
    return p

def add_to_cart_prompt(item_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""If the current page describes a product for sale, add it to the cart. If there are multiple variants of the product, use the following description to choose one: {item_description}. If the product is not on this page, report that the action is not possible.
"""
    return p

def checkout_prompt(payment_and_shipping_information: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, check out the items in the cart. Provide the following payment and shipping information: {payment_and_shipping_information}.
    Finally, finish the checkout process.

"""
    return p