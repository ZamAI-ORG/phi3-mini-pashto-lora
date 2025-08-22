#!/usr/bin/env python
"""Simple heuristic safety filter for Pashto content."""

import re
from typing import List, Tuple, Dict

# Basic banned words/phrases (extend as needed)
BANNED_WORDS_PASHTO = [
    # Add Pashto banned words here
    "ناوړه",  # inappropriate
    "ګناه",   # harmful
]

BANNED_WORDS_ENGLISH = [
    "violence", "hate", "harm", "illegal", "dangerous",
    "weapon", "drug", "suicide", "murder", "terrorist"
]

SENSITIVE_PATTERNS = [
    r"\b(?:phone|email|address|password)\b",  # PII patterns
    r"\d{4}-\d{4}-\d{4}-\d{4}",              # Credit card pattern
    r"\b\d{3}-\d{2}-\d{4}\b",                # SSN pattern
]

class SafetyFilter:
    def __init__(self, custom_banned_words: List[str] = None):
        self.banned_words = (BANNED_WORDS_PASHTO + BANNED_WORDS_ENGLISH + 
                           (custom_banned_words or []))
        self.patterns = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_PATTERNS]
    
    def check_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text is safe.
        
        Returns:
            (is_safe, list_of_violations)
        """
        if not text:
            return True, []
        
        violations = []
        text_lower = text.lower()
        
        # Check banned words
        for word in self.banned_words:
            if word.lower() in text_lower:
                violations.append(f"Banned word: {word}")
        
        # Check sensitive patterns
        for pattern in self.patterns:
            if pattern.search(text):
                violations.append(f"Sensitive pattern: {pattern.pattern}")
        
        # Additional heuristics
        if len(text) > 5000:
            violations.append("Text too long")
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def filter_text(self, text: str, replacement: str = "[FILTERED]") -> str:
        """Replace banned content with replacement text."""
        filtered_text = text
        
        for word in self.banned_words:
            filtered_text = re.sub(
                re.escape(word), 
                replacement, 
                filtered_text, 
                flags=re.IGNORECASE
            )
        
        for pattern in self.patterns:
            filtered_text = pattern.sub(replacement, filtered_text)
        
        return filtered_text
    
    def get_safety_score(self, text: str) -> float:
        """Get safety score between 0 (unsafe) and 1 (safe)."""
        is_safe, violations = self.check_text(text)
        if is_safe:
            return 1.0
        
        # Simple scoring based on number of violations
        score = max(0.0, 1.0 - (len(violations) * 0.2))
        return score

def filter_conversation(messages: List[Dict[str, str]], 
                       safety_filter: SafetyFilter = None) -> List[Dict[str, str]]:
    """Filter a conversation (list of messages)."""
    if safety_filter is None:
        safety_filter = SafetyFilter()
    
    filtered_messages = []
    for message in messages:
        content = message.get("content", "")
        is_safe, violations = safety_filter.check_text(content)
        
        if is_safe:
            filtered_messages.append(message)
        else:
            # Log violation or replace with filtered version
            filtered_content = safety_filter.filter_text(content)
            filtered_message = message.copy()
            filtered_message["content"] = filtered_content
            filtered_message["safety_violations"] = violations
            filtered_messages.append(filtered_message)
    
    return filtered_messages

# Example usage
if __name__ == "__main__":
    filter_obj = SafetyFilter()
    
    test_texts = [
        "سلام وروره! دا یو ښه ورځ دی.",  # Safe Pashto
        "This is a safe message.",       # Safe English
        "Call me at 555-1234",          # PII
        "Violence is bad",               # Banned word
    ]
    
    for text in test_texts:
        is_safe, violations = filter_obj.check_text(text)
        score = filter_obj.get_safety_score(text)
        print(f"Text: {text}")
        print(f"Safe: {is_safe}, Score: {score:.2f}")
        if violations:
            print(f"Violations: {violations}")
        print("-" * 40)