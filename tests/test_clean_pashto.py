#!/usr/bin/env python
"""Test BLEU/chrF metrics for perfect self-match scores."""

import unittest
from evaluation.metrics import compute_bleu, compute_chrf, compute_all_metrics

class TestCleanPashto(unittest.TestCase):
    
    def setUp(self):
        """Set up test cases."""
        self.pashto_text = "سلام نړۍ! دا یو ښه ورځ دی."
        self.english_text = "Hello world! This is a good day."
        self.multi_sentence = "پښتو ژبه ښکلې ده. دا زموږ مورنۍ ژبه ده."
    
    def test_bleu_perfect_match(self):
        """Test BLEU score for perfect self-match."""
        predictions = [self.pashto_text]
        references = [self.pashto_text]
        
        result = compute_bleu(predictions, references)
        
        self.assertAlmostEqual(result['bleu'], 100.0, places=1)
        self.assertEqual(result['bleu_bp'], 1.0)
        self.assertEqual(result['bleu_ratio'], 1.0)
    
    def test_chrf_perfect_match(self):
        """Test chrF score for perfect self-match."""
        predictions = [self.pashto_text]
        references = [self.pashto_text]
        
        result = compute_chrf(predictions, references)
        
        self.assertAlmostEqual(result['chrf'], 100.0, places=1)
        self.assertEqual(result['chrf_bp'], 1.0)
        self.assertEqual(result['chrf_ratio'], 1.0)
    
    def test_all_metrics_perfect_match(self):
        """Test all metrics for perfect self-match."""
        predictions = [self.pashto_text, self.english_text]
        references = [self.pashto_text, self.english_text]
        
        result = compute_all_metrics(predictions, references)
        
        self.assertAlmostEqual(result['bleu'], 100.0, places=1)
        self.assertAlmostEqual(result['chrf'], 100.0, places=1)
        self.assertEqual(result['num_predictions'], 2)
        self.assertEqual(result['num_references'], 2)
    
    def test_multi_sentence_perfect_match(self):
        """Test metrics with multi-sentence text."""
        predictions = [self.multi_sentence]
        references = [self.multi_sentence]
        
        bleu_result = compute_bleu(predictions, references)
        chrf_result = compute_chrf(predictions, references)
        
        self.assertAlmostEqual(bleu_result['bleu'], 100.0, places=1)
        self.assertAlmostEqual(chrf_result['chrf'], 100.0, places=1)
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        predictions = []
        references = []
        
        bleu_result = compute_bleu(predictions, references)
        chrf_result = compute_chrf(predictions, references)
        
        self.assertEqual(bleu_result['bleu'], 0.0)
        self.assertEqual(chrf_result['chrf'], 0.0)
    
    def test_partial_match(self):
        """Test metrics with partial matches."""
        predictions = ["سلام نړۍ! دا یو ډېر ښه ورځ دی."]  # Slightly different
        references = [self.pashto_text]
        
        bleu_result = compute_bleu(predictions, references)
        chrf_result = compute_chrf(predictions, references)
        
        # Should be high but not perfect
        self.assertGreater(bleu_result['bleu'], 70.0)
        self.assertLess(bleu_result['bleu'], 100.0)
        self.assertGreater(chrf_result['chrf'], 80.0)
        self.assertLess(chrf_result['chrf'], 100.0)
    
    def test_multiple_references_format(self):
        """Test handling of multiple references per prediction."""
        predictions = [self.pashto_text]
        references = [[self.pashto_text, "سلام ورور! دا ښه ورځ دی."]]
        
        bleu_result = compute_bleu(predictions, references)
        chrf_result = compute_chrf(predictions, references)
        
        # Should get perfect match with first reference
        self.assertAlmostEqual(bleu_result['bleu'], 100.0, places=1)
        self.assertAlmostEqual(chrf_result['chrf'], 100.0, places=1)

if __name__ == '__main__':
    unittest.main()