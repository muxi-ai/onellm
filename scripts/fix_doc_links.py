#!/usr/bin/env python3
"""
Automatically fix documentation links by removing .md extensions.
"""

import re
from pathlib import Path


def fix_markdown_links(docs_dir: Path) -> int:
    """Remove .md extensions from markdown links."""
    fixed_count = 0

    # Pattern for markdown links with .md extension
    md_link_pattern = re.compile(r'(\[[^\]]+\]\([^)]+)\.md(\))')

    for md_file in docs_dir.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace .md extensions in links
        new_content, replacements = md_link_pattern.subn(r'\1\2', content)

        if replacements > 0:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            rel_path = md_file.relative_to(docs_dir.parent)
            print(f"✓ Fixed {replacements} link(s) in {rel_path}")
            fixed_count += replacements

    return fixed_count


def main():
    """Fix all documentation links."""
    docs_dir = Path(__file__).parent.parent / "docs"

    if not docs_dir.exists():
        print(f"Error: docs directory not found at {docs_dir}")
        return 1

    print(f"Fixing documentation links in {docs_dir}...\n")

    fixed_count = fix_markdown_links(docs_dir)

    print(f"\n✅ Fixed {fixed_count} links total")
    return 0


if __name__ == "__main__":
    exit(main())
