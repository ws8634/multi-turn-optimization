import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    SUMMARY = "summary"


VALID_ROLES = {r.value for r in Role}


@dataclass
class Message:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {self.role}. Must be one of {VALID_ROLES}")

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    def content_bytes(self) -> int:
        return len(self.content.encode("utf-8"))


class ValidationError(Exception):
    pass


@dataclass
class CompressionResult:
    messages: List[Message]
    stats: dict
    stderr_output: List[str]


def validate_messages(messages: List[Message]) -> None:
    if not messages:
        raise ValidationError("Messages list is empty")

    system_indices: List[int] = []
    prev_role: Optional[str] = None

    for i, msg in enumerate(messages):
        if msg.role == Role.SYSTEM.value:
            system_indices.append(i)
        elif msg.role in (Role.USER.value, Role.ASSISTANT.value):
            if prev_role == msg.role:
                raise ValidationError(
                    f"Consecutive {msg.role} messages at position {i-1} and {i}"
                )
            prev_role = msg.role
        elif msg.role == Role.SUMMARY.value:
            prev_role = msg.role
        else:
            raise ValidationError(f"Invalid role '{msg.role}' at position {i}")

    if len(system_indices) == 0:
        raise ValidationError("No system message found")
    if len(system_indices) > 1:
        raise ValidationError(f"Multiple system messages found: {len(system_indices)}")
    if system_indices[0] != 0:
        raise ValidationError(f"System message must be first, found at position {system_indices[0]}")


def truncate_preview(content: str, max_bytes: int = 80) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content

    truncated = encoded[:max_bytes]

    while len(truncated) > 0:
        try:
            preview = truncated.decode("utf-8")
            return preview + "..."
        except UnicodeDecodeError:
            truncated = truncated[:-1]

    return "..."


def get_non_system_messages(messages: List[Message]) -> List[Message]:
    if messages and messages[0].role == Role.SYSTEM.value:
        return messages[1:]
    return messages


def calculate_total_bytes(messages: List[Message]) -> int:
    return sum(m.content_bytes() for m in messages)


def fold_pair(user_msg: Message, assistant_msg: Message) -> Message:
    content = f"[ROUND u={user_msg.content}| a={assistant_msg.content}]"
    return Message(role=Role.SUMMARY.value, content=content)


def merge_adjacent_summaries(messages: List[Message]) -> List[Message]:
    if not messages:
        return messages

    result: List[Message] = []
    pending_summaries: List[str] = []

    for msg in messages:
        if msg.role == Role.SUMMARY.value:
            pending_summaries.append(msg.content)
        else:
            if pending_summaries:
                merged_content = " ".join(pending_summaries)
                result.append(Message(role=Role.SUMMARY.value, content=merged_content))
                pending_summaries = []
            result.append(msg)

    if pending_summaries:
        merged_content = " ".join(pending_summaries)
        result.append(Message(role=Role.SUMMARY.value, content=merged_content))

    return result


def find_next_foldable_pair(messages: List[Message]) -> Optional[Tuple[int, int]]:
    for i in range(len(messages) - 1):
        if messages[i].role == Role.USER.value and messages[i + 1].role == Role.ASSISTANT.value:
            return (i, i + 1)
    return None


def compress_conversation(messages: List[Message], budget: int) -> CompressionResult:
    validate_messages(messages)

    system_msg: Optional[Message] = None
    working_messages: List[Message] = []

    if messages[0].role == Role.SYSTEM.value:
        system_msg = messages[0]
        working_messages = messages[1:]
    else:
        working_messages = messages

    stderr_output: List[str] = []
    stats = {
        "initial_bytes": calculate_total_bytes(working_messages),
        "budget": budget,
        "fold_count": 0,
        "discarded_users": [],
        "triggered": False,
    }

    if budget == 0:
        result_messages = [system_msg] if system_msg else []
        stats["triggered"] = True
        stderr_output.append("Compression triggered: budget=0, all non-system messages removed")
        return CompressionResult(
            messages=result_messages,
            stats=stats,
            stderr_output=stderr_output,
        )

    current_bytes = calculate_total_bytes(working_messages)

    if current_bytes <= budget:
        result_messages = [system_msg] + working_messages if system_msg else working_messages
        stderr_output.append(
            f"Compression not triggered: current {current_bytes} bytes <= budget {budget} bytes"
        )
        return CompressionResult(
            messages=result_messages,
            stats=stats,
            stderr_output=stderr_output,
        )

    stats["triggered"] = True
    stderr_output.append(
        f"Compression triggered: current {current_bytes} bytes > budget {budget} bytes"
    )

    while True:
        current_bytes = calculate_total_bytes(working_messages)
        if current_bytes <= budget:
            break

        fold_pair_idx = find_next_foldable_pair(working_messages)

        if fold_pair_idx is None:
            user_indices = [
                i for i, m in enumerate(working_messages)
                if m.role == Role.USER.value
            ]
            if user_indices:
                discard_idx = user_indices[0]
                discarded_msg = working_messages.pop(discard_idx)
                preview = truncate_preview(discarded_msg.content)
                stats["discarded_users"].append(preview)
                stderr_output.append(f"Warning: discarded unpaired user message: {preview}")
                continue
            else:
                break

        i, j = fold_pair_idx
        user_msg = working_messages[i]
        assistant_msg = working_messages[j]

        folded = fold_pair(user_msg, assistant_msg)
        working_messages = working_messages[:i] + [folded] + working_messages[j + 1:]

        working_messages = merge_adjacent_summaries(working_messages)
        stats["fold_count"] += 1

    result_messages = (
        [system_msg] + working_messages if system_msg else working_messages
    )
    stats["final_bytes"] = calculate_total_bytes(
        get_non_system_messages(result_messages)
    )

    return CompressionResult(
        messages=result_messages,
        stats=stats,
        stderr_output=stderr_output,
    )


def dry_run_stats(messages: List[Message], budget: int) -> dict:
    validate_messages(messages)

    working_messages = get_non_system_messages(messages)
    current_bytes = calculate_total_bytes(working_messages)

    stats = {
        "initial_bytes": current_bytes,
        "budget": budget,
        "fold_count": 0,
        "would_discard_users": [],
        "triggered": current_bytes > budget,
    }

    if budget == 0:
        return stats

    temp_messages = working_messages.copy()

    while True:
        current_bytes = calculate_total_bytes(temp_messages)
        if current_bytes <= budget:
            break

        fold_pair_idx = find_next_foldable_pair(temp_messages)

        if fold_pair_idx is None:
            user_indices = [
                i for i, m in enumerate(temp_messages)
                if m.role == Role.USER.value
            ]
            if user_indices:
                discard_idx = user_indices[0]
                discarded_msg = temp_messages.pop(discard_idx)
                preview = truncate_preview(discarded_msg.content)
                stats["would_discard_users"].append(preview)
                continue
            else:
                break

        i, j = fold_pair_idx
        user_msg = temp_messages[i]
        assistant_msg = temp_messages[j]

        folded = fold_pair(user_msg, assistant_msg)
        temp_messages = temp_messages[:i] + [folded] + temp_messages[j + 1:]

        temp_messages = merge_adjacent_summaries(temp_messages)
        stats["fold_count"] += 1

    return stats
