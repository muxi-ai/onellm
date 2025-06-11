#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Text cleaning utilities for OneLLM.

This module provides utilities for cleaning Unicode artifacts from AI model responses.
Many AI models inject invisible Unicode characters that can be detected by plagiarism
and AI detection tools. This module helps normalize text to safe ASCII equivalents.
"""

import re


def clean_unicode_artifacts(text: str) -> str:
    """
    Clean invisible Unicode artifacts from AI model responses.

    This function removes problematic Unicode characters that AI models often inject,
    such as zero-width spaces and other invisible characters that can be detected
    by plagiarism and AI detection tools, while preserving all legitimate text
    content including multilingual characters.

    Args:
        text: The input text containing potential Unicode artifacts

    Returns:
        The cleaned text with invisible characters removed and typographic
        characters normalized, while preserving all legitimate content

    Examples:
        >>> clean_unicode_artifacts('"Hello" — World\u200B')
        '"Hello" — World'

        >>> clean_unicode_artifacts('Text with\u00a0spaces')
        'Text with spaces'

        >>> clean_unicode_artifacts('こんにちは\u200B世界')  # Japanese with zero-width space
        'こんにちは 世界'
    """
    if not text:
        return text

    # Normalize common typographic characters to ASCII equivalents
    replacements = {
        # spaces
        '\u202f': " ",  # Narrow no-break space → normal space
        '\u00a0': " ",  # Non-breaking space → normal space
        '\u2003': " ",  # Em space → normal space
        '\u2009': " ",  # Thin space → normal space
        # invisible characters
        '\u200B': '',   # Zero-width space
        '\u200C': '',   # Zero-width non-joiner
        '\u200D': '',   # Zero-width joiner
        '\uFEFF': '',   # Byte order mark
        '\u200A': '',   # Word joiner
        '\u200E': '',   # Left-to-right mark
        '\u200F': '',   # Right-to-left mark
        '\u2028': '',   # Line separator
        '\u2029': '',   # Paragraph separator
        '\u202A': '',   # Left-to-right embedding
        '\u202B': '',   # Right-to-left embedding
        '\u202C': '',   # Pop directional formatting
        '\u202D': '',   # Left-to-right override
        '\u202E': '',   # Right-to-left override
        '\u2060': '',   # Word joiner
        '\u2061': '',   # Function application
        '\u2062': '',   # Invisible times
        '\u2063': '',   # Invisible separator
        '\u2064': '',   # Invisible plus
        '\u2065': '',   # Invisible separator
        '\u2066': '',   # Left-to-right isolate
        '\u2067': '',   # Right-to-left isolate
        '\u2068': '',   # First strong isolate
        '\u2069': '',   # Pop directional isolate
        '\u206A': '',   # Inhibit symmetric swapping
        '\u206B': '',   # Activate symmetric swapping
        '\u206C': '',   # Inhibit Arabic form shaping
        '\u206D': '',   # Activate Arabic form shaping
        '\u206E': '',   # National digit shapes
        '\u206F': '',   # Nominal digit shapes
        '\u034F': '',   # Combining grapheme joiner
        '\uFE00': '',   # Variation selector-1
        '\uFE01': '',   # Variation selector-2
        '\uFE02': '',   # Variation selector-3
        '\uFE03': '',   # Variation selector-4
        '\uFE04': '',   # Variation selector-5
        '\uFE05': '',   # Variation selector-6
        '\uFE06': '',   # Variation selector-7
        '\uFE07': '',   # Variation selector-8
        '\uFE08': '',   # Variation selector-9
        '\uFE09': '',   # Variation selector-10
        '\uFE0A': '',   # Variation selector-11
        '\uFE0B': '',   # Variation selector-12
        '\uFE0C': '',   # Variation selector-13
        '\uFE0D': '',   # Variation selector-14
        '\uFE0E': '',   # Variation selector-15
        '\uFE0F': '',   # Variation selector-16
        # quotes
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u00AB': '"',  # French opening quote → normal quote
        '\u00BB': '"',  # French closing quote → normal quote
        # dashes
        '\u2011': '-',  # Non-breaking hyphen → regular hyphen
        '\u2013': '–',  # En dash
        '\u2014': '—',  # Em dash

    }

    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # Remove trailing whitespace on every line
    text = re.sub(r'[ \t]+(\r?\n)', r'\1', text)

    return text

