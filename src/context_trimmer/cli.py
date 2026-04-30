import argparse
import json
import sys
from typing import Optional, TextIO

from context_trimmer.core import ValidationError
from context_trimmer.api import Session


def read_json_input(source: Optional[str] = None) -> dict:
    if source:
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return json.load(sys.stdin)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-turn dialogue context trimmer"
    )
    parser.add_argument(
        "budget",
        type=int,
        help="Budget B (>= 0) in UTF-8 bytes for non-system messages",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to JSON input file. If not provided, read from stdin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fold count without modifying output",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.budget < 0:
        print(f"Error: budget must be >= 0, got {args.budget}", file=sys.stderr)
        return 1

    try:
        data = read_json_input(args.input)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1

    if "messages" not in data:
        print("Error: Missing 'messages' field in input", file=sys.stderr)
        return 1

    messages = data["messages"]
    if not isinstance(messages, list):
        print("Error: 'messages' must be an array", file=sys.stderr)
        return 1

    try:
        session = Session.from_dicts(messages)
        session.validate()
    except ValidationError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        stats = session.dry_run(args.budget)
        print(f"Initial bytes: {stats['initial_bytes']}")
        print(f"Budget: {stats['budget']}")
        print(f"Would trigger compression: {'yes' if stats['triggered'] else 'no'}")
        print(f"Would fold {stats['fold_count']} pair(s)")
        if stats.get("would_discard_users"):
            for preview in stats["would_discard_users"]:
                print(f"Would discard user message: {preview}")
        return 0

    result = session.compress(args.budget)

    output_data = {"messages": session.to_dicts()}
    json.dump(output_data, sys.stdout, ensure_ascii=False, indent=2)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
