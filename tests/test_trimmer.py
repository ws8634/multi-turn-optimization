import pytest

from context_trimmer.core import (
    Message,
    CompressionResult,
    ValidationError,
    compress_conversation,
    dry_run_stats,
    merge_adjacent_summaries,
    fold_pair,
)
from context_trimmer.api import Session


class TestValidation:
    def test_no_system_message_error(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        with pytest.raises(ValidationError, match="No system message found"):
            compress_conversation(messages, budget=100)

    def test_multiple_system_messages_error(self):
        messages = [
            Message(role="system", content="You are a helper"),
            Message(role="system", content="Another system"),
            Message(role="user", content="Hello"),
        ]
        with pytest.raises(ValidationError, match="Multiple system messages found"):
            compress_conversation(messages, budget=100)

    def test_system_not_first_error(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="system", content="You are a helper"),
        ]
        with pytest.raises(ValidationError, match="System message must be first"):
            compress_conversation(messages, budget=100)

    def test_consecutive_user_error(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="U1"),
            Message(role="user", content="U2"),
        ]
        with pytest.raises(ValidationError, match="Consecutive user messages"):
            compress_conversation(messages, budget=100)

    def test_consecutive_assistant_error(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="U1"),
            Message(role="assistant", content="A1"),
            Message(role="assistant", content="A2"),
        ]
        with pytest.raises(ValidationError, match="Consecutive assistant messages"):
            compress_conversation(messages, budget=100)


class TestNoCompression:
    def test_under_budget_no_compression(self):
        messages = [
            Message(role="system", content="You are a helper"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = compress_conversation(messages, budget=200)

        assert not result.stats["triggered"]
        assert len(result.messages) == 3
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "assistant"
        assert any("not triggered" in line for line in result.stderr_output)

    def test_equal_budget_no_compression(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        current_bytes = len("Hello".encode("utf-8")) + len("Hi".encode("utf-8"))
        result = compress_conversation(messages, budget=current_bytes)

        assert not result.stats["triggered"]
        assert len(result.messages) == 3


class TestCompression:
    def test_single_pair_fold_triggered(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="U1"),
            Message(role="assistant", content="A1"),
        ]
        original_bytes = len("U1".encode()) + len("A1".encode())
        budget = original_bytes - 1

        result = compress_conversation(messages, budget=budget)

        assert result.stats["triggered"]
        assert result.stats["fold_count"] == 1
        assert len(result.messages) == 2
        assert result.messages[1].role == "summary"
        assert result.messages[1].content == "[ROUND u=U1| a=A1]"

    def test_multiple_pairs_fold_in_order(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="U1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="U2"),
            Message(role="assistant", content="A2"),
        ]
        original_bytes = (
            len("U1".encode()) + len("A1".encode()) +
            len("U2".encode()) + len("A2".encode())
        )
        budget = original_bytes - 1

        result = compress_conversation(messages, budget=budget)

        assert result.stats["triggered"]
        assert result.stats["fold_count"] >= 1

    def test_adjacent_summary_merge(self):
        msg1 = Message(role="summary", content="[S1]")
        msg2 = Message(role="summary", content="[S2]")
        msg3 = Message(role="user", content="U3")

        merged = merge_adjacent_summaries([msg1, msg2, msg3])

        assert len(merged) == 2
        assert merged[0].role == "summary"
        assert merged[0].content == "[S1] [S2]"
        assert merged[1].role == "user"

    def test_fold_pair_template(self):
        user = Message(role="user", content="What's your name?")
        assistant = Message(role="assistant", content="I'm a bot.")

        folded = fold_pair(user, assistant)

        assert folded.role == "summary"
        assert folded.content == "[ROUND u=What's your name?| a=I'm a bot.]"

    def test_summary_content_counts_toward_budget(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="summary", content="[OLD ROUND u=X| a=Y]"),
        ]
        summary_bytes = len("[OLD ROUND u=X| a=Y]".encode())
        budget = summary_bytes - 1

        result = compress_conversation(messages, budget=budget)

        assert result.stats["triggered"]


class TestDiscardUnpairedUser:
    def test_discard_unpaired_user_when_needed(self):
        long_user = "x" * 50
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content=long_user),
        ]
        original_bytes = len(long_user.encode())
        budget = original_bytes - 1

        result = compress_conversation(messages, budget=budget)

        assert result.stats["triggered"]
        assert len(result.stats["discarded_users"]) > 0
        assert any("discarded" in line.lower() for line in result.stderr_output)

    def test_warning_stderr_on_discard(self):
        long_content = "x" * 100
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content=long_content),
        ]
        budget = 50
        result = compress_conversation(messages, budget=budget)

        assert result.stats["triggered"]
        discarded = any("discarded" in line.lower() for line in result.stderr_output)
        if result.stats["discarded_users"]:
            assert discarded


class TestBudgetZero:
    def test_budget_zero_removes_all_non_system(self):
        messages = [
            Message(role="system", content="You are a helper"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="summary", content="[Old]"),
        ]
        result = compress_conversation(messages, budget=0)

        assert result.stats["triggered"]
        assert len(result.messages) == 1
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are a helper"
        assert any("budget=0" in line for line in result.stderr_output)

    def test_budget_zero_empty_system(self):
        messages = [
            Message(role="system", content=""),
            Message(role="user", content="Hello"),
        ]
        result = compress_conversation(messages, budget=0)

        assert len(result.messages) == 1
        assert result.messages[0].role == "system"


class TestSingleUserEdgeCase:
    def test_only_user_no_assistant(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="Just me"),
        ]
        budget = len("Just me".encode())
        result = compress_conversation(messages, budget=budget)

        assert not result.stats["triggered"]
        assert len(result.messages) == 2
        assert result.messages[1].role == "user"
        assert result.messages[1].content == "Just me"

    def test_only_user_no_assistant_under_budget(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="Just me"),
        ]
        budget = len("Just me".encode()) * 2
        result = compress_conversation(messages, budget=budget)

        assert not result.stats["triggered"]
        assert result.messages[1].role == "user"


class TestDryRun:
    def test_dry_run_no_modification(self):
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="U1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="U2"),
            Message(role="assistant", content="A2"),
        ]
        original_bytes = (
            len("U1".encode()) + len("A1".encode()) +
            len("U2".encode()) + len("A2".encode())
        )
        budget = original_bytes - 1
        stats = dry_run_stats(messages, budget)

        assert stats["triggered"]
        assert stats["fold_count"] >= 1


class TestSessionAPI:
    def test_session_from_dicts(self):
        dicts = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hello"},
        ]
        session = Session.from_dicts(dicts)

        assert len(session.get_messages()) == 2
        assert session.get_messages()[0].role == "system"

    def test_session_add_turn(self):
        session = Session()
        session.add_message("system", "Sys")
        session.add_turn("U1", "A1")

        messages = session.get_messages()
        assert len(messages) == 3
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    def test_session_compress(self):
        session = Session.from_dicts([
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
        ])
        original_bytes = len("U1".encode()) + len("A1".encode())
        budget = original_bytes - 1
        result = session.compress(budget, output_stderr=False)

        assert result.stats["fold_count"] == 1
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[1].role == "summary"

    def test_session_current_bytes(self):
        session = Session.from_dicts([
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        expected = len("Hello".encode()) + len("Hi".encode())
        assert session.current_bytes() == expected

    def test_session_to_dicts(self):
        dicts = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "U1"},
        ]
        session = Session.from_dicts(dicts)
        output = session.to_dicts()

        assert output == dicts


class TestContentTruncation:
    def test_long_content_preview(self):
        from context_trimmer.core import truncate_preview

        content = "x" * 100
        preview = truncate_preview(content, max_chars=80)

        assert len(preview) == 83
        assert preview.endswith("...")

    def test_short_content_no_truncation(self):
        from context_trimmer.core import truncate_preview

        content = "Short text"
        preview = truncate_preview(content, max_chars=80)

        assert preview == content
