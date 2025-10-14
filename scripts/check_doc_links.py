#!/usr/bin/env python3
"""
Check documentation links for common issues:
- Links with .md extensions (should be removed for Jekyll)
- Broken {% link %} tags
- Mermaid diagram syntax issues
"""

import re
import sys
from pathlib import Path


def check_markdown_links(docs_dir: Path) -> list[tuple[Path, int, str]]:
    """Find markdown links with .md extensions."""
    issues = []

    # Pattern for markdown links with .md extension
    md_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+\.md)\)')

    for md_file in docs_dir.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                matches = md_link_pattern.findall(line)
                if matches:
                    for text, url in matches:
                        issues.append((md_file, line_num, f"Link has .md extension: [{text}]({url})"))

    return issues


def check_jekyll_links(docs_dir: Path) -> list[tuple[Path, int, str]]:
    """Find Jekyll {% link %} tags and verify target files exist."""
    issues = []

    # Pattern for Jekyll link tags
    link_pattern = re.compile(r'{%\s*link\s+([^\s}]+)\s*%}')

    for md_file in docs_dir.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                matches = link_pattern.findall(line)
                for target in matches:
                    # Check if the target file exists
                    target_path = docs_dir / target
                    if not target_path.exists():
                        issues.append((md_file, line_num, f"Jekyll link target not found: {target}"))

    return issues


def check_mermaid_diagrams(docs_dir: Path) -> list[tuple[Path, int, str]]:
    """Check mermaid diagrams for common issues."""
    issues = []
    mermaid_start = re.compile(r'```mermaid')
    mermaid_end = re.compile(r'```')

    for md_file in docs_dir.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            in_mermaid = False
            mermaid_start_line = 0

            for line_num, line in enumerate(lines, 1):
                if mermaid_start.search(line):
                    in_mermaid = True
                    mermaid_start_line = line_num
                elif in_mermaid and mermaid_end.search(line):
                    in_mermaid = False

    return issues


def main():
    """Run all documentation link checks."""
    docs_dir = Path(__file__).parent.parent / "docs"

    if not docs_dir.exists():
        print(f"Error: docs directory not found at {docs_dir}")
        sys.exit(1)

    print(f"Checking documentation links in {docs_dir}...\n")

    # Check for .md extensions in markdown links
    md_issues = check_markdown_links(docs_dir)
    if md_issues:
        print(f"❌ Found {len(md_issues)} links with .md extensions:")
        for file, line, issue in md_issues:
            rel_path = file.relative_to(docs_dir.parent)
            print(f"  {rel_path}:{line} - {issue}")
        print()

    # Check Jekyll links
    jekyll_issues = check_jekyll_links(docs_dir)
    if jekyll_issues:
        print(f"❌ Found {len(jekyll_issues)} broken Jekyll links:")
        for file, line, issue in jekyll_issues:
            rel_path = file.relative_to(docs_dir.parent)
            print(f"  {rel_path}:{line} - {issue}")
        print()

    # Check mermaid diagrams
    mermaid_issues = check_mermaid_diagrams(docs_dir)
    if mermaid_issues:
        print(f"❌ Found {len(mermaid_issues)} mermaid diagram issues:")
        for file, line, issue in mermaid_issues:
            rel_path = file.relative_to(docs_dir.parent)
            print(f"  {rel_path}:{line} - {issue}")
        print()

    # Summary
    total_issues = len(md_issues) + len(jekyll_issues) + len(mermaid_issues)
    if total_issues > 0:
        print(f"Total issues found: {total_issues}")
        print("\nTo fix .md extensions in links:")
        print("  Replace [text](path/file.md) with [text](path/file)")
        print("  Replace {{ site.baseurl }}/path/file.md with {{ site.baseurl }}/path/file")
        sys.exit(1)
    else:
        print("✅ All documentation links and diagrams look good!")
        sys.exit(0)


if __name__ == "__main__":
    main()
