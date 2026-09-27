"""
AirWrite Character Recognition Package
Provides letter segmentation, stroke normalization, dataset collection,
and machine-learning-based character classification.
"""

from .feature_extractor import StrokeFeatureExtractor
from .dataset_manager import DatasetManager
from .letter_segmenter import LetterSegmenter
from .recognizer import CharacterRecognizer

__all__ = [
    "StrokeFeatureExtractor",
    "DatasetManager",
    "LetterSegmenter",
    "CharacterRecognizer",
]
