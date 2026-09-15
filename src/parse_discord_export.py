"""
Parse DiscordChatExporter JSON export(s) into a raw text dump of one
person's messages, one per line, suitable for training a char-level
language model.

Usage:
    python parse_discord_export.py --input export1.json export2.json \
        --author "YourDiscordUsername" \
        --output corpus.txt

If you're not sure of the exact author name/id to filter on, run once
with --list-authors to see everyone who appears in the export(s).
"""

import argparse
import json
import re
import sys
from pathlib import Path


def load_messages(path: Path):
    """Load a DiscordChatExporter JSON export and return its message list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def list_authors(paths):
    """Print every distinct author name/id found across the given export files."""
    seen = {}
    for path in paths:
        for msg in load_messages(path):
            author = msg.get("author", {})
            key = (author.get("id"), author.get("name"))
            seen[key] = seen.get(key, 0) + 1
    print("Authors found across export(s):")
    for (author_id, name), count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  name={name!r}  id={author_id}  messages={count}")


def clean_text(text: str) -> str:
    """
    Strip Discord-specific noise from a message's raw content:
    - custom emoji codes like <:name:123456789>
    - user/role/channel mentions like <@123456789>, <@&123>, <#123>
    - URLs
    - collapse excess whitespace
    """
    if not text:
        return ""

    # Custom emoji: <:name:id> or <a:name:id> (animated)
    text = re.sub(r"<a?:\w+:\d+>", "", text)

    # Mentions: <@id>, <@!id>, <@&id> (role), <#id> (channel)
    text = re.sub(r"<@!?&?\d+>", "", text)
    text = re.sub(r"<#\d+>", "", text)

    # URLs
    text = re.sub(r"https?://\S+", "", text)

    # Collapse whitespace/newlines within a single message down to single spaces,
    # since we want one line per message in the output.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_author_messages(paths, author_name=None, author_id=None,
                             min_length=1, exclude_bots=True):
    """
    Walk all given export files and yield cleaned message strings from the
    matching author, in file order (DiscordChatExporter exports are already
    chronological within a file).
    """
    for path in paths:
        messages = load_messages(path)
        for msg in messages:
            # Skip non-default message types (system messages: pins, calls,
            # member joins, etc. show up in the export as other "type" values).
            if msg.get("type") != "Default":
                continue

            author = msg.get("author", {})

            if exclude_bots and author.get("isBot"):
                continue

            if author_id is not None and author.get("id") != author_id:
                continue
            if author_name is not None and author.get("name") != author_name:
                continue

            content = clean_text(msg.get("content", ""))

            # Skip empty messages (e.g. a message that was ONLY an attachment,
            # sticker, or embed with no text content).
            if len(content) < min_length:
                continue

            yield content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more DiscordChatExporter JSON export files.",
    )
    parser.add_argument(
        "--output", default="corpus.txt",
        help="Path to write the flat text corpus to (default: corpus.txt).",
    )
    parser.add_argument(
        "--author", default=None,
        help="Exact Discord username to filter on (as it appears in the export's "
             "author.name field). Use --list-authors first if unsure.",
    )
    parser.add_argument(
        "--author-id", default=None,
        help="Discord user ID to filter on instead of username. More reliable "
             "than --author if the account has changed its display name.",
    )
    parser.add_argument(
        "--min-length", type=int, default=1,
        help="Minimum character length for a message to be kept (default: 1, "
             "i.e. keep anything non-empty). Raise this to filter out low-signal "
             "one-word messages like 'lol' or 'ok' if you want a denser corpus.",
    )
    parser.add_argument(
        "--list-authors", action="store_true",
        help="Instead of writing output, list all authors found in the input "
             "files and exit. Use this first to find the exact name/id to filter on.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.input]
    for p in paths:
        if not p.exists():
            sys.exit(f"Input file not found: {p}")

    if args.list_authors:
        list_authors(paths)
        return

    if not args.author and not args.author_id:
        sys.exit(
            "You must specify --author or --author-id (or run with "
            "--list-authors first to find the right value)."
        )

    messages = list(
        extract_author_messages(
            paths,
            author_name=args.author,
            author_id=args.author_id,
            min_length=args.min_length,
        )
    )

    if not messages:
        sys.exit(
            "No matching messages found. Double check the --author / --author-id "
            "value with --list-authors."
        )

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(messages))

    total_chars = sum(len(m) for m in messages)
    print(f"Wrote {len(messages)} messages ({total_chars:,} characters) to {output_path}")
    if total_chars < 200_000:
        print(
            "Note: this is a fairly small corpus. A char-level model can still "
            "train on it, but consider including multiple DM/channel exports "
            "with --input to get more text if generation quality looks weak."
        )


if __name__ == "__main__":
    main()