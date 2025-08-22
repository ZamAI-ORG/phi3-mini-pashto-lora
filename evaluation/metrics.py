#!/usr/bin/env python
"""Metrics utilities for Pashto model evaluation."""

import sacrebleu
from typing import List, Dict, Any

def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU score using sacrebleu."""
    if not predictions or not references:
        return {"bleu": 0.0}
    
    # Handle single reference per prediction
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    
    bleu = sacrebleu.corpus_bleu(predictions, references)
    return {
        "bleu": bleu.score,
        "bleu_bp": bleu.bp,
        "bleu_ratio": bleu.ratio,
        "bleu_hyp_len": bleu.sys_len,
        "bleu_ref_len": bleu.ref_len
    }

def compute_chrf(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute chrF score using sacrebleu."""
    if not predictions or not references:
        return {"chrf": 0.0}
    
    # Handle single reference per prediction
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    
    chrf = sacrebleu.corpus_chrf(predictions, references)
    return {
        "chrf": chrf.score,
        "chrf_bp": chrf.bp,
        "chrf_ratio": chrf.ratio,
        "chrf_hyp_len": chrf.sys_len,
        "chrf_ref_len": chrf.ref_len
    }

def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Compute both BLEU and chrF scores."""
    bleu_scores = compute_bleu(predictions, references)
    chrf_scores = compute_chrf(predictions, references)
    
    return {
        **bleu_scores,
        **chrf_scores,
        "num_predictions": len(predictions),
        "num_references": len(references)
    }