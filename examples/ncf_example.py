"""
Neural Collaborative Filtering (NCF) Örnek Kullanımı
Modern Deep Learning tabanlı öneri sistemi
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algorithms.ncf import NCFRecommender
from utils.data_loader import generate_rating_matrix

def example_ncf_recommender():
    """NCF öneri sistemi örneği"""
    print("=" * 60)
    print("Neural Collaborative Filtering (NCF) - Öneri Sistemi")
    print("=" * 60)
    
    # Veri oluştur
    print("\n1. Veri oluşturuluyor...")
    rating_matrix = generate_rating_matrix(n_users=200, n_items=100, sparsity=0.6)
    print(f"   Rating matrisi boyutu: {rating_matrix.shape}")
    print(f"   Sparsity: {np.isnan(rating_matrix).sum() / rating_matrix.size:.2%}")
    
    # Train-test split
    train_size = int(0.8 * rating_matrix.shape[0])
    train_matrix = rating_matrix[:train_size]
    test_matrix = rating_matrix[train_size:]
    
    # Model oluştur ve eğit
    print("\n2. NCF modeli oluşturuluyor...")
    ncf_model = NCFRecommender(
        n_factors=50,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.2
    )
    
    print("3. Model eğitiliyor (bu biraz zaman alabilir)...")
    history = ncf_model.fit(
        train_matrix,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )
    
    # Eğitim geçmişi
    print("\n4. Eğitim geçmişi:")
    if history:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('NCF Eğitim Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('NCF Eğitim Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ncf_training_history.png', dpi=150, bbox_inches='tight')
        print("   Grafik kaydedildi: ncf_training_history.png")
    
    # Tahminler
    print("\n5. Tahminler yapılıyor...")
    predictions = ncf_model.predict_all(explicit_scale=True)
    print(f"   Tahmin matrisi boyutu: {predictions.shape}")
    print(f"   Tahmin aralığı: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Öneriler
    print("\n6. Öneriler üretiliyor...")
    user_idx = 0
    item_indices, predicted_ratings = ncf_model.recommend(
        user_idx,
        n_recommendations=10,
        rating_matrix=train_matrix
    )
    
    print(f"\n   Kullanıcı {user_idx} için Top 10 Öneri:")
    print("   " + "-" * 50)
    for i, (item_idx, rating) in enumerate(zip(item_indices, predicted_ratings), 1):
        print(f"   {i:2d}. Ürün {item_idx+1:3d} - Tahmin: {rating:.2f}")
    
    # Değerlendirme
    print("\n7. Model değerlendiriliyor...")
    rmse = ncf_model.evaluate(test_matrix, explicit_scale=True)
    print(f"   Test RMSE: {rmse:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ NCF örneği tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        example_ncf_recommender()
    except ImportError as e:
        print(f"❌ Hata: {e}")
        print("💡 TensorFlow yüklü olduğundan emin olun: pip install tensorflow")
    except Exception as e:
        print(f"❌ Hata: {e}")

