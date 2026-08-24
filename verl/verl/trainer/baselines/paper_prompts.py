"""Exact student and self-Teacher prompts used by the SDPO/SRPO papers.

The Science prompt is copied from ``lasgroup/SDPO`` commit
``7c457fc1b1f636ae794eb0362ba37d4743b06fbc``. The SRPO Tool Use
system message and Teacher template are copied from the arXiv source of
Appendix B.3/B.6 in arXiv:2604.02288v1. Dynamic Tool Use API documentation
and the final user question are record-specific and are therefore preserved
byte-for-byte from the official SDPO train/test records.
"""

from __future__ import annotations

from typing import Any


SDPO_OFFICIAL_PROMPT_PROFILE = "sdpo_official_v2"
SRPO_PAPER_PROMPT_PROFILE = "srpo_v1"

SCIENCE_SYSTEM_PROMPT = """
Given a question and four options, please select the right answer. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. Do not restate the answer text. For example, if the answer is "A", just output:
<answer>
A
</answer>
"""

SCIENCE_USER_SUFFIX = "\nPlease reason step by step."

SRPO_TOOLUSE_SYSTEM_PROMPT = (
    "You are a tool-use assistant. Solve each request by reasoning about the task and "
    "calling the provided tools when needed.\n"
    "Use only the tools provided in the user message.\n"
    "Follow the required response format exactly.\n"
    "For every tool call, output exactly these lines in this order:\n"
    "Thought: <brief reason>\n"
    "Action: <the exact tool name from the tool list>\n"
    "Action Input: <one valid JSON object on the next line>\n"
    "After Action:, write only the exact tool name; never write 'Use ...'. "
    "Write 'Action Input:' exactly, never 'Input:' or 'Action_Input:'. "
    "Do not use Markdown code fences and do not write an Output: block; the tool runner supplies the output.\n"
    "Example:\n"
    "Thought: Search for the requested API.\n"
    "Action: searchAPIs\n"
    "Action Input: {\"query\": \"OpenWeatherMap\", \"limit\": 1}"
)

TOOLUSE_USER_PREFIX = "Your task is to answer the user's question using available tools."
TOOLUSE_REQUIRED_FRAGMENTS = (
    "You have access to the following tools:",
    "Use the following format:",
    "Thought: you should always think about what to do",
    "Action: the action to take, should be one of the tool names.",
    "Action Input: the input to the action, must be in JSON format.",
    "Begin!",
    "Question:",
)

SDPO_REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
SDPO_SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
SDPO_FEEDBACK_TEMPLATE = (
    "\nThe following is feedback from your unsuccessful earlier attempt:\n\n"
    "{feedback_raw}\n\n"
)


def _data_source_family(data_source: Any) -> str:
    normalized = str(data_source or "").strip().lower()
    if normalized in {
        "sciknoweval",
        "science",
        "biology",
        "chemistry",
        "material",
        "materials",
        "physics",
    }:
        return "science"
    if normalized in {"tooluse", "tool_use", "toolalpaca"}:
        return "tooluse"
    raise ValueError(f"Unsupported SDPO/SRPO paper data source: {data_source!r}")


def _copy_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("SDPO/SRPO paper prompts require a non-empty message list")
    copied: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("SDPO/SRPO paper prompt messages must be dictionaries")
        copied.append(dict(message))
    return copied


def apply_paper_prompt_profile(
    messages: Any,
    *,
    data_source: Any,
    profile: str,
) -> list[dict[str, Any]]:
    """Validate and, where the SRPO paper requires it, complete the prompt.

    The upstream SDPO JSONL already contains the complete Science system/user
    messages and the complete dynamic Tool Use user message. This function
    validates the fixed paper scaffold without reconstructing or normalizing
    record-specific content. SRPO additionally specifies a Tool
    Use system prompt, so that profile inserts it when the upstream record has
    no system message.
    """

    if profile not in {SDPO_OFFICIAL_PROMPT_PROFILE, SRPO_PAPER_PROMPT_PROFILE}:
        raise ValueError(f"Unsupported SDPO/SRPO paper prompt profile: {profile!r}")
    copied = _copy_messages(messages)
    family = _data_source_family(data_source)
    if copied[-1].get("role") != "user":
        raise ValueError("SDPO/SRPO paper prompt must end with a user message")
    user_content = copied[-1].get("content")
    if not isinstance(user_content, str) or not user_content:
        raise ValueError("SDPO/SRPO paper prompt requires non-empty user content")

    if family == "science":
        system_messages = [message for message in copied if message.get("role") == "system"]
        if len(system_messages) != 1 or system_messages[0].get("content") != SCIENCE_SYSTEM_PROMPT:
            raise ValueError("Science prompt does not match the complete official SDPO system prompt")
        if not user_content.endswith(SCIENCE_USER_SUFFIX):
            raise ValueError("Science prompt does not match the official SDPO user suffix")
        return copied

    for fragment in TOOLUSE_REQUIRED_FRAGMENTS:
        if fragment not in user_content:
            raise ValueError(f"Tool Use paper prompt is missing required fragment: {fragment!r}")
    if not user_content.startswith(TOOLUSE_USER_PREFIX):
        raise ValueError("Tool Use prompt does not match the paper user-template prefix")

    system_indices = [index for index, message in enumerate(copied) if message.get("role") == "system"]
    if profile == SDPO_OFFICIAL_PROMPT_PROFILE:
        if system_indices:
            raise ValueError("Official SDPO Tool Use records do not contain a system message")
        return copied

    if not system_indices:
        return [{"role": "system", "content": SRPO_TOOLUSE_SYSTEM_PROMPT}, *copied]
    if len(system_indices) != 1 or copied[system_indices[0]].get("content") != SRPO_TOOLUSE_SYSTEM_PROMPT:
        raise ValueError("SRPO Tool Use system message does not match Appendix B.3")
    if system_indices[0] != 0:
        raise ValueError("SRPO Tool Use system message must be the first message")
    return copied


def build_self_teacher_messages(raw_prompt: Any, sibling_response: str) -> list[dict[str, str]]:
    """Build the exact operational SDPO/SRPO Correct-solution reprompt."""

    copied = _copy_messages(raw_prompt)
    if copied[-1].get("role") != "user":
        raise ValueError("Self-Teacher prompt must end with a user message")
    prompt = copied[-1].get("content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Self-Teacher prompt requires non-empty user content")
    sibling_response = str(sibling_response or "")
    if not sibling_response.strip():
        raise ValueError("Self-Teacher prompt requires a non-empty successful sibling")
    solution = SDPO_SOLUTION_TEMPLATE.format(
        successful_previous_attempt=sibling_response
    )
    reprompt = SDPO_REPROMPT_TEMPLATE.format(
        prompt=prompt,
        solution=solution,
        feedback="",
    )
    return [*copied[:-1], {"role": "user", "content": reprompt}]


__all__ = [
    "SCIENCE_SYSTEM_PROMPT",
    "SCIENCE_USER_SUFFIX",
    "SDPO_FEEDBACK_TEMPLATE",
    "SDPO_OFFICIAL_PROMPT_PROFILE",
    "SDPO_REPROMPT_TEMPLATE",
    "SDPO_SOLUTION_TEMPLATE",
    "SRPO_PAPER_PROMPT_PROFILE",
    "SRPO_TOOLUSE_SYSTEM_PROMPT",
    "apply_paper_prompt_profile",
    "build_self_teacher_messages",
]
