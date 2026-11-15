"""Depiction Style Centring (DSC): training-free removal of depiction bias from frozen image embeddings."""
from .method import (DEPICTIONS, TEMPLATES, DSC, dsc, centre_by_group, gate_open, load_siglip, name_depictions,
                     word_embeddings)
from .metrics import evaluate, depiction_bias, pair_precision, per_query

__version__ = "1.0.0"
__all__ = ["DEPICTIONS", "TEMPLATES", "DSC", "dsc", "centre_by_group", "gate_open", "load_siglip", "name_depictions",
           "word_embeddings", "evaluate", "depiction_bias", "pair_precision", "per_query"]
