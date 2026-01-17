"""
Gürültü Temizleme Örneği
SVD kullanarak gürültü temizleme
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.svd import SVDNoiseReducer
from utils.data_loader import generate_sample_data, generate_noisy_data
from utils.visualization import plot_noise_reduction_comparison


def example_noise_reduction():
    """SVD ile gürültü temizleme örneği"""
    print("=" * 60)
    print("SVD ile Gürültü Temizleme")
    print("=" * 60)
    
    # Örnek veri oluştur
    print("\n1. Örnek veri oluşturuluyor...")
    X, _ = generate_sample_data(n_samples=200, n_features=100, n_clusters=5)
    print(f"   Veri boyutu: {X.shape}")
    
    # Gürültü ekle
    print("\n2. Gürültü ekleniyor...")
    noise_level = 0.2
    X_noisy = generate_noisy_data(X, noise_level=noise_level)
    
    # Orijinal ve gürültülü veri arasındaki fark
    mse_original = np.mean((X - X_noisy)**2)
    print(f"   Gürültü MSE: {mse_original:.4f}")
    
    # SVD ile gürültü temizleme
    print("\n3. SVD ile gürültü temizleniyor...")
    noise_reducer = SVDNoiseReducer(n_components=None, threshold=0.95)
    noise_reducer.fit(X_noisy)
    
    X_denoised = noise_reducer.denoise(X_noisy)
    
    # Sonuçları değerlendir
    mse_denoised = np.mean((X - X_denoised)**2)
    noise_reduction_ratio = noise_reducer.get_noise_reduction_ratio()
    
    print(f"   Kullanılan bileşen sayısı: {noise_reducer.n_components}")
    print(f"   Varyans korunma oranı: {noise_reduction_ratio:.4f}")
    print(f"   Temizlenmiş veri MSE: {mse_denoised:.4f}")
    print(f"   İyileştirme: {((mse_original - mse_denoised) / mse_original * 100):.2f}%")
    
    # Farklı bileşen sayıları için analiz
    print("\n4. Farklı bileşen sayıları analiz ediliyor...")
    component_counts, variance_ratios = noise_reducer.get_optimal_components(X_noisy)
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Orijinal veri
    im1 = axes[0, 0].imshow(X[:50, :50], cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Orijinal Veri')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Gürültülü veri
    im2 = axes[0, 1].imshow(X_noisy[:50, :50], cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Gürültülü Veri')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Temizlenmiş veri
    im3 = axes[1, 0].imshow(X_denoised[:50, :50], cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Temizlenmiş Veri')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Varyans korunma grafiği
    axes[1, 1].plot(component_counts, variance_ratios, 'o-')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% Eşiği')
    axes[1, 1].axvline(x=noise_reducer.n_components, color='g', 
                      linestyle='--', label=f'Seçilen ({noise_reducer.n_components})')
    axes[1, 1].set_xlabel('Bileşen Sayısı')
    axes[1, 1].set_ylabel('Kümülatif Varyans Oranı')
    axes[1, 1].set_title('Varyans Korunma Analizi')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_reduction_results.png', dpi=150, bbox_inches='tight')
    print("\n   Grafik 'noise_reduction_results.png' olarak kaydedildi.")
    
    return noise_reducer, X, X_noisy, X_denoised


def example_image_denoising():
    """Görüntü gürültü temizleme örneği"""
    print("\n" + "=" * 60)
    print("Görüntü Gürültü Temizleme")
    print("=" * 60)
    
    # Basit bir görüntü oluştur
    print("\n1. Örnek görüntü oluşturuluyor...")
    image_size = (64, 64)
    image = np.zeros(image_size)
    
    # Basit bir pattern oluştur
    center_x, center_y = image_size[1] // 2, image_size[0] // 2
    radius = 20
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = 1.0
    
    # Gürültü ekle
    print("\n2. Gürültü ekleniyor...")
    noise = np.random.normal(0, 0.3, image_size)
    noisy_image = np.clip(image + noise, 0, 1)
    
    # Gürültü temizle
    print("\n3. Gürültü temizleniyor...")
    image_flat = noisy_image.flatten().reshape(1, -1)
    noise_reducer = SVDNoiseReducer(n_components=10, threshold=0.9)
    denoised_flat = noise_reducer.denoise(image_flat)
    denoised_image = denoised_flat.reshape(image_size)
    denoised_image = np.clip(denoised_image, 0, 1)
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title('Gürültülü Görüntü')
    axes[1].axis('off')
    
    axes[2].imshow(denoised_image, cmap='gray')
    axes[2].set_title('Temizlenmiş Görüntü')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_denoising_results.png', dpi=150, bbox_inches='tight')
    print("   Grafik 'image_denoising_results.png' olarak kaydedildi.")
    
    return image, noisy_image, denoised_image


if __name__ == "__main__":
    # Genel gürültü temizleme
    noise_reducer, X, X_noisy, X_denoised = example_noise_reduction()
    
    # Görüntü gürültü temizleme
    image, noisy_image, denoised_image = example_image_denoising()
    
    plt.show()

