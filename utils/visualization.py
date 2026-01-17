"""
Görselleştirme fonksiyonları
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import scipy.sparse


def plot_ratings_matrix(rating_matrix, figsize=(12, 8)):
    """
    Rating matrisini görselleştirir
    
    Args:
        rating_matrix: Rating matrisi (dense veya sparse)
        figsize: Grafik boyutu
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Rating dağılımı - sparse matrix desteği
    if scipy.sparse.issparse(rating_matrix):
        # Sparse matrix için sadece non-zero değerleri al
        ratings_flat = rating_matrix.data
    else:
        # Dense matrix için NaN olmayan değerleri al
        ratings_flat = rating_matrix[~np.isnan(rating_matrix)]
    
    ax1.hist(ratings_flat, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Frekans')
    ax1.set_title('Rating Dağılımı')
    ax1.grid(True, alpha=0.3)
    
    # Rating matrisi heatmap (ilk 50x50) - sparse matrix desteği
    matrix_vis = rating_matrix[:50, :50]
    
    # Sparse matrix ise dense'e çevir (50x50 küçük olduğu için sorun değil)
    if scipy.sparse.issparse(matrix_vis):
        # numpy.matrix veya scipy.sparse matrix olabilir
        if hasattr(matrix_vis, 'A'):
            matrix_vis = np.array(matrix_vis.A)
        elif hasattr(matrix_vis, 'toarray'):
            matrix_vis = matrix_vis.toarray()
        else:
            matrix_vis = np.array(matrix_vis)
    
    # Mask oluştur (NaN değerler için)
    mask = np.isnan(matrix_vis) if not scipy.sparse.issparse(matrix_vis) else None
    
    sns.heatmap(matrix_vis, cmap='YlOrRd', cbar=True, 
                ax=ax2, vmin=1, vmax=5, 
                mask=mask)
    ax2.set_xlabel('Ürünler')
    ax2.set_ylabel('Kullanıcılar')
    ax2.set_title('Rating Matrisi (İlk 50x50)')
    
    plt.tight_layout()
    return fig


def plot_recommendations(user_idx, recommendations, item_names=None, figsize=(10, 6)):
    """
    Önerileri görselleştirir
    
    Args:
        user_idx: Kullanıcı indeksi
        recommendations: (item_indices, ratings) tuple
        item_names: Ürün isimleri (opsiyonel)
        figsize: Grafik boyutu
        
    Returns:
        matplotlib figure
    """
    item_indices, ratings = recommendations
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if item_names is not None:
        labels = [item_names[i] for i in item_indices]
    else:
        labels = [f'Ürün {i+1}' for i in item_indices]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, ratings, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Tahmin Edilen Rating')
    ax.set_title(f'Kullanıcı {user_idx+1} için Öneriler')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_image_grid(images, image_shape, n_cols=5, figsize=(15, 10), titles=None):
    """
    Görüntü grid'i çizer
    
    Args:
        images: Görüntü matrisi (n_images x n_pixels)
        image_shape: Görüntü boyutu (height, width)
        n_cols: Grid'deki sütun sayısı
        figsize: Grafik boyutu
        titles: Görüntü başlıkları (opsiyonel)
        
    Returns:
        matplotlib figure
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for i in range(n_rows * n_cols):
        if i < n_images:
            img = images[i].reshape(image_shape)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            if titles is not None and i < len(titles):
                axes[i].set_title(titles[i], fontsize=8)
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_topic_words(topics, n_words=10, figsize=(15, 10)):
    """
    Topic'lerin kelimelerini görselleştirir
    
    Args:
        topics: Topic dictionary (topic_name: [(word, score), ...])
        n_words: Gösterilecek kelime sayısı
        figsize: Grafik boyutu
        
    Returns:
        matplotlib figure
    """
    n_topics = len(topics)
    fig, axes = plt.subplots(1, n_topics, figsize=figsize)
    
    if n_topics == 1:
        axes = [axes]
    
    for idx, (topic_name, words_scores) in enumerate(topics.items()):
        words, scores = zip(*words_scores[:n_words])
        y_pos = np.arange(len(words))
        
        axes[idx].barh(y_pos, scores, alpha=0.7)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(words)
        axes[idx].set_xlabel('Önem Skoru')
        axes[idx].set_title(topic_name)
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_noise_reduction_comparison(original, noisy, denoised, figsize=(15, 5)):
    """
    Gürültü temizleme sonuçlarını karşılaştırır
    
    Args:
        original: Orijinal veri
        noisy: Gürültülü veri
        denoised: Temizlenmiş veri
        figsize: Grafik boyutu
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Orijinal
    im1 = axes[0].imshow(original, cmap='viridis', aspect='auto')
    axes[0].set_title('Orijinal Veri')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Gürültülü
    im2 = axes[1].imshow(noisy, cmap='viridis', aspect='auto')
    axes[1].set_title('Gürültülü Veri')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Temizlenmiş
    im3 = axes[2].imshow(denoised, cmap='viridis', aspect='auto')
    axes[2].set_title('Temizlenmiş Veri')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig

