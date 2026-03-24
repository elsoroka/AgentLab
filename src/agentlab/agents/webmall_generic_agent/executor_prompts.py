from AgentLab.src.agentlab.agents.dynamic_prompting import PromptElement

_ACTION_FORMAT_NOTE = """\
Always respond using <action>...</action> XML tags. Do NOT use Python code blocks (```).
"""


def search_on_page_prompt(search_text: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""Goal: find the product "{search_text}" in THIS store (current tab only — ignore all other open tabs).

Step-by-step workflow:
1. Check your action history first:
   - If you already CLICKED a product link in a previous step AND the current page is a product detail page (URL contains /product/ or page shows a product title, price, and "Add to cart" button), you are ALREADY on the target page. Immediately report it:
<action>
report_result(url="<current page URL>")
</action>
   - Do NOT click anything again. Do NOT search again. Just report the URL.

2. If you have NOT yet searched, use the store's search box to search for "{search_text}".

3. Look at what happens after the search:
   a) SEARCH RESULTS LIST: scan for a product whose name is an exact or very close match to "{search_text}". If found, click it to open its product page. After clicking, on the NEXT step immediately report the URL (do not click again).
   b) DIRECT REDIRECT to a single product page: the store auto-redirected. Check if the title is an exact match to "{search_text}". If YES, report the URL. If NOT, call report_infeasible() immediately.

4. If no exact match found:
<action>
report_infeasible()
</action>

Hard rules:
- FIRST CHECK: If the current page is already a product detail page, STOP and report_result immediately.
- Only consider the CURRENT tab. Do NOT copy URLs from other browser tabs.
- You have at most 2 search attempts. After 2 failed attempts call report_infeasible().
- NEVER repeat a search already in your action history — check the history and call report_infeasible() instead.
- Do NOT try alternate search terms. Either find the exact product or declare infeasible.
- After clicking a product link, the VERY NEXT action must be report_result — never click again.
{_ACTION_FORMAT_NOTE}"""
    return p


def navigate_to_page_prompt(description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""Navigate to the page described as: {description}.
When finished:
<action>
done()
</action>
If not possible:
<action>
report_infeasible()
</action>
{_ACTION_FORMAT_NOTE}"""
    return p


def extract_information_from_page_prompt(description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, extract the information described by: {description}.
Return it as a plain string:
<action>
report_result("the extracted text here")
</action>
If the information cannot be found:
<action>
report_infeasible()
</action>
Do NOT navigate away or click anything — just read and report.
{_ACTION_FORMAT_NOTE}"""
    return p


def fill_text_field_prompt(field_description: str, text: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, find the text field described as "{field_description}" and fill it with exactly this text:
{text}

IMPORTANT: Only fill the field. Do NOT click any buttons, do NOT submit any forms, do NOT press Enter.
Once the field is filled:
<action>
done()
</action>
If the field cannot be found:
<action>
report_infeasible()
</action>
{_ACTION_FORMAT_NOTE}"""
    return p


def press_button_prompt(button_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, click the button described as "{button_description}".

IMPORTANT: Only click that button. Do NOT modify or fill any text fields — leave all field contents exactly as they are.
Once the button is clicked:
<action>
done()
</action>
If the button cannot be found:
<action>
report_infeasible()
</action>
{_ACTION_FORMAT_NOTE}"""
    return p


def select_option_prompt(bid: str, options: str | list[str]) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""On the current page, select the option described as: {bid} with options: {options}.
When finished:
<action>
done()
</action>
If not possible:
<action>
report_infeasible()
</action>
{_ACTION_FORMAT_NOTE}"""
    return p


def add_to_cart_prompt(item_description: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""If the current page describes a product for sale, add it to the cart. If there are multiple variants, use this description to choose: {item_description}.
When finished:
<action>
done()
</action>
If not possible:
<action>
report_infeasible()
</action>
{_ACTION_FORMAT_NOTE}"""
    return p


def checkout_prompt(payment_and_shipping_information: str) -> str:
    p = PromptElement(visible=True)
    p._prompt = f"""From the current page, check out the items in the cart using this payment and shipping information: {payment_and_shipping_information}.
When finished:
<action>
done()
</action>
If not possible:
<action>
report_infeasible()
</action>
{_ACTION_FORMAT_NOTE}"""
    return p
