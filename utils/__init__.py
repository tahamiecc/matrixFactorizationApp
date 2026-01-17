"""
Utility functions module
"""

from .data_loader import generate_sample_data, load_sample_images, generate_text_corpus
from .visualization import plot_ratings_matrix, plot_recommendations, plot_image_grid

__all__ = [
    'generate_sample_data',
    'load_sample_images',
    'generate_text_corpus',
    'plot_ratings_matrix',
    'plot_recommendations',
    'plot_image_grid'
]

