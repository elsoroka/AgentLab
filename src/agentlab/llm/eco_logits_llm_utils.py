import logging
from typing import TYPE_CHECKING
from agentlab.llm.llm_utils import Discussion, ParseError
if TYPE_CHECKING:
    from agentlab.llm.eco_logits_chat_api import ChatModelEcoLogits


def retry(
    chat: "ChatModelEcoLogits",
    messages: "Discussion",
    n_retry: int,
    parser: callable,
    log: bool = True,
):
    """Retry querying the chat models with the response from the parser until it
    returns a valid value.

    If the answer is not valid, it will retry and append to the chat the  retry
    message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be slow
    and expensive.

    Args:
        chat (ChatModel): a ChatModel object taking a list of messages and
            returning a list of answers, all in OpenAI format.
        messages (list): the list of messages so far. This list will be modified with
            the new messages and the retry messages.
        n_retry (int): the maximum number of sequential retries.
        parser (callable): a function taking a message and retruning a parsed value,
            or raising a ParseError
        log (bool): whether to log the retry messages.

    Returns:
        dict: the parsed value, with a string at key "action".

    Raises:
        ParseError: if the parser could not parse the response after n_retry retries.
    """
    tries = 0
    while tries < n_retry:
        # TODO: add cummulation of eco metrics if tries fail. 
        answer, impacts = chat(messages)
        # TODO: could we change this to not use inplace modifications ?
        messages.append(answer)
        try:
            return parser(answer["content"]), impacts
        except ParseError as parsing_error:
            tries += 1
            if log:
                msg = f"Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer['content']}\n[User]:\n{str(parsing_error)}"
                logging.info(msg)
            messages.append(dict(role="user", content=str(parsing_error)))
            
    raise ParseError(f"Could not parse a valid value after {n_retry} retries.")