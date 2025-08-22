#!/usr/bin/env python
"""Test safety filter behavior for prompt templates."""

import unittest

from safety.filter import SafetyFilter, filter_conversation


class TestPromptTemplate(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.safety_filter = SafetyFilter()
        self.safe_pashto = "سلام وروره! دا یو ښه ورځ دی."
        self.safe_english = "Hello, how can I help you today?"
        self.unsafe_content = "This contains violence and harm"
        self.pii_content = "My phone number is 555-1234"

    def test_safe_pashto_content(self):
        """Test safe Pashto content passes filter."""
        is_safe, violations = self.safety_filter.check_text(self.safe_pashto)

        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)
        self.assertEqual(self.safety_filter.get_safety_score(self.safe_pashto), 1.0)

    def test_safe_english_content(self):
        """Test safe English content passes filter."""
        is_safe, violations = self.safety_filter.check_text(self.safe_english)

        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)
        self.assertEqual(self.safety_filter.get_safety_score(self.safe_english), 1.0)

    def test_unsafe_content_detection(self):
        """Test unsafe content is detected."""
        is_safe, violations = self.safety_filter.check_text(self.unsafe_content)

        self.assertFalse(is_safe)
        self.assertGreater(len(violations), 0)
        self.assertLess(self.safety_filter.get_safety_score(self.unsafe_content), 1.0)

    def test_pii_detection(self):
        """Test PII detection."""
        is_safe, violations = self.safety_filter.check_text(self.pii_content)

        self.assertFalse(is_safe)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("Sensitive pattern" in v for v in violations))

    def test_filter_text_replacement(self):
        """Test text filtering with replacement."""
        filtered = self.safety_filter.filter_text(self.unsafe_content)

        self.assertNotEqual(filtered, self.unsafe_content)
        self.assertIn("[FILTERED]", filtered)

    def test_custom_banned_words(self):
        """Test custom banned words."""
        custom_filter = SafetyFilter(custom_banned_words=["testing"])

        is_safe, violations = custom_filter.check_text("This is a testing message")

        self.assertFalse(is_safe)
        self.assertTrue(any("testing" in v for v in violations))

    def test_empty_text_handling(self):
        """Test empty text handling."""
        is_safe, violations = self.safety_filter.check_text("")

        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)
        self.assertEqual(self.safety_filter.get_safety_score(""), 1.0)

    def test_very_long_text(self):
        """Test very long text detection."""
        long_text = "A" * 6000  # Exceeds 5000 char limit

        is_safe, violations = self.safety_filter.check_text(long_text)

        self.assertFalse(is_safe)
        self.assertTrue(any("too long" in v for v in violations))

    def test_conversation_filtering(self):
        """Test conversation filtering functionality."""
        messages = [
            {"role": "user", "content": self.safe_pashto},
            {"role": "assistant", "content": self.safe_english},
            {"role": "user", "content": self.unsafe_content},
        ]

        filtered_messages = filter_conversation(messages, self.safety_filter)

        self.assertEqual(len(filtered_messages), 3)

        # First two should be unchanged
        self.assertEqual(filtered_messages[0]["content"], self.safe_pashto)
        self.assertEqual(filtered_messages[1]["content"], self.safe_english)

        # Third should be filtered
        self.assertIn("[FILTERED]", filtered_messages[2]["content"])
        self.assertIn("safety_violations", filtered_messages[2])

    def test_safety_score_range(self):
        """Test safety score is in valid range."""
        test_texts = [
            "",
            self.safe_pashto,
            self.unsafe_content,
            self.pii_content,
            "Multiple violations: violence harm illegal",
        ]

        for text in test_texts:
            score = self.safety_filter.get_safety_score(text)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_case_insensitive_filtering(self):
        """Test case-insensitive filtering."""
        upper_case_unsafe = "THIS CONTAINS VIOLENCE"

        is_safe, violations = self.safety_filter.check_text(upper_case_unsafe)

        self.assertFalse(is_safe)
        self.assertGreater(len(violations), 0)


if __name__ == "__main__":
    unittest.main()
