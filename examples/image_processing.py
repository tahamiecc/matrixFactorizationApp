"""
Görüntü İşleme Örneği
NMF kullanarak görüntü analizi ve sıkıştırma
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.nmf import NMFImageProcessor
from utils.data_loader import load_sample_images
from utils.visualization import plot_image_grid


def example_image_processing():
    """NMF ile görüntü işleme örneği"""
    print("=" * 60)
    print("NMF ile Görüntü İşleme")
    print("=" * 60)
    
    # Görüntü verisi yükle
    print("\n1. Görüntü verisi yükleniyor...")
    images_flat, image_shape = load_sample_images(n_images=100)
    print(f"   Görüntü sayısı: {len(images_flat)}")
    print(f"   Görüntü boyutu: {image_shape}")
    print(f"   Toplam piksel sayısı: {np.prod(image_shape)}")
    
    # NMF modeli
    print("\n2. NMF modeli eğitiliyor...")
    n_components = 20
    nmf_image = NMFImageProcessor(n_components=n_components)
    nmf_image.fit(images_flat)
    
    # Basis görüntüleri
    print("\n3. Temel görüntüler (basis images) çıkarılıyor...")
    basis_images = nmf_image.get_basis_images(image_shape)
    print(f"   Basis görüntü sayısı: {len(basis_images)}")
    print(f"   Basis görüntü boyutu: {basis_images[0].shape}")
    
    # Yeniden oluşturma
    print("\n4. Görüntüler yeniden oluşturuluyor...")
    reconstructed = nmf_image.reconstruct()
    
    # Sıkıştırma oranı
    compression_ratio = nmf_image.get_compression_ratio(image_shape)
    print(f"\n5. Sıkıştırma analizi:")
    print(f"   Sıkıştırma oranı: {compression_ratio:.2f}x")
    print(f"   Orijinal boyut: {np.prod(image_shape)} piksel")
    print(f"   Sıkıştırılmış boyut: {n_components * (image_shape[0] + image_shape[1])} parametre")
    
    # Görselleştirme
    print("\n6. Görselleştirme oluşturuluyor...")
    
    # Orijinal görüntüler
    fig1 = plot_image_grid(images_flat[:10], image_shape, n_cols=5, 
                           figsize=(15, 6),
                           titles=[f'Orijinal {i+1}' for i in range(10)])
    plt.suptitle('Orijinal Görüntüler (İlk 10)', fontsize=16, y=1.02)
    plt.savefig('original_images.png', dpi=150, bbox_inches='tight')
    print("   'original_images.png' kaydedildi.")
    
    # Basis görüntüleri
    basis_flat = basis_images.reshape(len(basis_images), -1)
    fig2 = plot_image_grid(basis_flat, image_shape, n_cols=5, 
                          figsize=(15, 8),
                          titles=[f'Basis {i+1}' for i in range(len(basis_images))])
    plt.suptitle('Temel Görüntüler (Basis Images)', fontsize=16, y=1.02)
    plt.savefig('basis_images.png', dpi=150, bbox_inches='tight')
    print("   'basis_images.png' kaydedildi.")
    
    # Yeniden oluşturulmuş görüntüler
    fig3 = plot_image_grid(reconstructed[:10], image_shape, n_cols=5, 
                          figsize=(15, 6),
                          titles=[f'Yeniden Oluşturulmuş {i+1}' for i in range(10)])
    plt.suptitle('Yeniden Oluşturulmuş Görüntüler (İlk 10)', fontsize=16, y=1.02)
    plt.savefig('reconstructed_images.png', dpi=150, bbox_inches='tight')
    print("   'reconstructed_images.png' kaydedildi.")
    
    # Karşılaştırma
    fig4, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        # Orijinal
        axes[0, i].imshow(images_flat[i].reshape(image_shape), cmap='gray')
        axes[0, i].set_title(f'Orijinal {i+1}')
        axes[0, i].axis('off')
        
        # Yeniden oluşturulmuş
        axes[1, i].imshow(reconstructed[i].reshape(image_shape), cmap='gray')
        axes[1, i].set_title(f'Yeniden Oluşturulmuş {i+1}')
        axes[1, i].axis('off')
        
        # Fark
        diff = np.abs(images_flat[i].reshape(image_shape) - 
                     reconstructed[i].reshape(image_shape))
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'Fark {i+1}')
        axes[2, i].axis('off')
    
    plt.suptitle('Orijinal vs Yeniden Oluşturulmuş Görüntüler', fontsize=16)
    plt.savefig('image_comparison.png', dpi=150, bbox_inches='tight')
    print("   'image_comparison.png' kaydedildi.")
    
    # Hata analizi
    mse = np.mean((images_flat - reconstructed)**2, axis=1)
    print(f"\n7. Hata analizi:")
    print(f"   Ortalama MSE: {np.mean(mse):.6f}")
    print(f"   Maksimum MSE: {np.max(mse):.6f}")
    print(f"   Minimum MSE: {np.min(mse):.6f}")
    
    return nmf_image, images_flat, reconstructed, basis_images


def example_image_compression():
    """Görüntü sıkıştırma örneği"""
    print("\n" + "=" * 60)
    print("Görüntü Sıkıştırma")
    print("=" * 60)
    
    images_flat, image_shape = load_sample_images(n_images=50)
    
    # Farklı bileşen sayıları için test
    n_components_list = [5, 10, 20, 30, 50]
    compression_ratios = []
    mse_values = []
    
    print("\nFarklı bileşen sayıları test ediliyor...")
    for n_comp in n_components_list:
        nmf = NMFImageProcessor(n_components=n_comp)
        nmf.fit(images_flat)
        reconstructed = nmf.reconstruct()
        
        mse = np.mean((images_flat - reconstructed)**2)
        compression = nmf.get_compression_ratio(image_shape)
        
        compression_ratios.append(compression)
        mse_values.append(mse)
        
        print(f"  {n_comp} bileşen: Sıkıştırma={compression:.2f}x, MSE={mse:.6f}")
    
    # Görselleştirme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(n_components_list, compression_ratios, 'o-')
    ax1.set_xlabel('Bileşen Sayısı')
    ax1.set_ylabel('Sıkıştırma Oranı')
    ax1.set_title('Sıkıştırma Oranı vs Bileşen Sayısı')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(n_components_list, mse_values, 'o-')
    ax2.set_xlabel('Bileşen Sayısı')
    ax2.set_ylabel('MSE')
    ax2.set_title('Yeniden Oluşturma Hatası vs Bileşen Sayısı')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compression_analysis.png', dpi=150, bbox_inches='tight')
    print("\n   'compression_analysis.png' kaydedildi.")
    
    return n_components_list, compression_ratios, mse_values


if __name__ == "__main__":
    # Görüntü işleme
    nmf_image, images, reconstructed, basis = example_image_processing()
    
    # Sıkıştırma analizi
    n_comp_list, comp_ratios, mse_vals = example_image_compression()
    
    plt.show()

