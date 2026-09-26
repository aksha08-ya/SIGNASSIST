"""Conservative context-aware sentence builder for SIGNASSIST Phase 10.

This module provides rule-based phrase template matching, lexical normalization,
question-word detection, capitalization, and punctuation formatting.

Note on Scope:
This is a conservative phrase reconstruction and context formatting helper.
It deliberately avoids claiming full ISL-to-English neural or dependency-based grammar translation.
"""

import re
from typing import Dict, List, Optional


class ContextAwareSentenceBuilder:
    """Conservative, rule-based sentence builder and context formatter."""

    def __init__(
        self,
        enable_context_rules: bool = True,
        auto_punctuation: bool = True,
    ) -> None:
        self.enable_context_rules = enable_context_rules
        self.auto_punctuation = auto_punctuation

        # Common phrase templates (upper-case gloss list tuple -> formatted template)
        self.phrase_templates: Dict[tuple, str] = {
            ("ME", "NAME"): "My name is",
            ("MY", "NAME"): "My name is",
            ("THANK", "YOU"): "Thank you",
            ("GOOD", "MORNING"): "Good morning",
            ("GOOD", "NIGHT"): "Good night",
            ("NICE", "MEET", "YOU"): "Nice to meet you",
            ("HOW", "ARE", "YOU"): "How are you",
            ("HELP", "ME"): "Please help me",
            ("YOU", "HELP", "ME"): "Can you help me",
        }

        # Question words triggering '?' ending
        self.question_words = {"WHERE", "WHAT", "WHO", "WHY", "HOW", "WHEN", "WHICH"}

        # Lexical normalization mapping
        self.lexical_map = {
            "ME": "I",
            "MY": "my",
            "YOU": "you",
            "YOUR": "your",
            "HE": "he",
            "SHE": "she",
            "WE": "we",
            "THEY": "they",
        }

    def build_sentence(self, glosses: List[str]) -> str:
        """Assembles a list of sign glosses into a formatted English sentence.

        Args:
            glosses: List of uppercase gloss strings.

        Returns:
            Formatted sentence string with capitalization and punctuation.
        """
        if not glosses:
            return ""

        clean_glosses = [g.strip().upper() for g in glosses if g.strip()]
        if not clean_glosses:
            return ""

        if not self.enable_context_rules:
            # Simple space joining without phrase rules
            raw_text = " ".join(clean_glosses)
            return self.format_text(raw_text, is_question=False)

        # 1. Check for exact prefix phrase templates
        matched_prefix = ""
        remaining_glosses: List[str] = []

        # Check multi-token template matching
        for template_key, template_val in self.phrase_templates.items():
            k_len = len(template_key)
            if len(clean_glosses) >= k_len and tuple(clean_glosses[:k_len]) == template_key:
                matched_prefix = template_val
                remaining_glosses = clean_glosses[k_len:]
                break

        if matched_prefix:
            if remaining_glosses:
                rem_words = [
                    self.lexical_map.get(g, g.capitalize()) for g in remaining_glosses
                ]
                sentence_body = f"{matched_prefix} {' '.join(rem_words)}"
            else:
                sentence_body = matched_prefix
        else:
            # 2. General lexical normalization fallback
            words: List[str] = []
            for idx, g in enumerate(clean_glosses):
                if idx == 0 and g == "ME":
                    words.append("I")
                elif g in self.lexical_map:
                    words.append(self.lexical_map[g])
                else:
                    words.append(g.lower())

            sentence_body = " ".join(words)

        # 3. Question detection
        has_question_word = any(
            g in self.question_words for g in clean_glosses
        )

        return self.format_text(sentence_body, is_question=has_question_word)

    def format_text(self, text: str, is_question: bool = False) -> str:
        """Formats text with proper capitalization and ending punctuation.

        Args:
            text: Input raw text string.
            is_question: Flag indicating if sentence is a question.

        Returns:
            Formatted sentence string.
        """
        text = text.strip()
        if not text:
            return ""

        # Normalize spaces
        text = re.sub(r"\s+", " ", text)

        # Capitalize first character
        if len(text) > 1:
            text = text[0].upper() + text[1:]
        else:
            text = text.upper()

        # Fix isolated ' i ' to ' I '
        text = re.sub(r"\b(i)\b", "I", text)

        if not self.auto_punctuation:
            return text

        # Strip trailing punctuation before adding standard ending
        text = text.rstrip(".?!,;")

        if is_question:
            text += "?"
        else:
            text += "."

        return text
