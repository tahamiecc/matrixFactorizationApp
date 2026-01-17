"""
Matrix Factorization Algorithms Module
Klasik ve Modern algoritmaları içerir
"""

# Klasik algoritmalar
from .svd import SVDRecommender, SVDNoiseReducer
from .pca import PCAAnalyzer
from .nmf import NMFImageProcessor, NMFTopicModeler
from .als import ALSRecommender

# Modern Deep Learning algoritmalar - PyTorch tabanlı (Python 3.14 uyumlu ✅)
PYTORCH_AVAILABLE = False
try:
    import torch
    from .ncf import NCFRecommender
    from .autoencoder import DenoisingAutoencoder, VAERecommender
    from .fm import FactorizationMachine, DeepFM
    from .transformer import TransformerRecommender
    from .gnn import GNNRecommender
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PyTorch yüklü değil. Modern algoritmalar kullanılamayacak: {e}")
    print("   Yüklemek için: pip install torch torch-geometric")

# Modern algoritmalar PyTorch ile kullanılabilir
MODERN_ALGORITHMS_AVAILABLE = PYTORCH_AVAILABLE

__all__ = [
    # Klasik
    'SVDRecommender',
    'SVDNoiseReducer',
    'PCAAnalyzer',
    'NMFImageProcessor',
    'NMFTopicModeler',
    'ALSRecommender',
]

if PYTORCH_AVAILABLE:
    __all__.extend([
        # PyTorch tabanlı modern algoritmalar
        'NCFRecommender',
        'DenoisingAutoencoder',
        'VAERecommender',
        'FactorizationMachine',
        'DeepFM',
        'TransformerRecommender',
        'GNNRecommender',
    ])

