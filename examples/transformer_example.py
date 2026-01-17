"""
Transformer-based Sequential Recommendation Örneği
TikTok, YouTube gibi sıralı öneri sistemleri için
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from algorithms.transformer import TransformerRecommender

def example_transformer_recommender():
    """Transformer öneri sistemi örneği"""
    print("=" * 60)
    print("Transformer - Sequential Recommendation")
    print("=" * 60)
    
    # Sequential veri oluştur (her kullanıcı için item sequence)
    print("\n1. Sequential veri oluşturuluyor...")
    n_users = 100
    n_items = 200
    max_seq_length = 50
    
    user_sequences = []
    for u in range(n_users):
        # Her kullanıcı için rastgele item sequence
        seq_length = np.random.randint(10, max_seq_length)
        sequence = np.random.choice(n_items, size=seq_length, replace=False).tolist()
        user_sequences.append(sequence)
        if u < 3:
            print(f"   Kullanıcı {u} sequence: {sequence[:10]}... (uzunluk: {len(sequence)})")
    
    print(f"   Toplam kullanıcı: {n_users}")
    print(f"   Toplam ürün: {n_items}")
    
    # Model oluştur
    print("\n2. Transformer modeli oluşturuluyor...")
    transformer_model = TransformerRecommender(
        n_items=n_items,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_length=max_seq_length
    )
    
    print("3. Model eğitiliyor (bu biraz zaman alabilir)...")
    history = transformer_model.fit(
        user_sequences,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Sıradaki item tahmini
    print("\n4. Sıradaki item tahminleri yapılıyor...")
    
    for user_idx in range(3):
        sequence = user_sequences[user_idx][:-1]  # Son item hariç
        next_item_true = user_sequences[user_idx][-1]  # Gerçek sonraki item
        
        item_indices, probabilities = transformer_model.predict_next(sequence)
        
        print(f"\n   Kullanıcı {user_idx}:")
        print(f"   Geçmiş sequence: {sequence[-5:]}")
        print(f"   Gerçek sonraki item: {next_item_true + 1}")
        print(f"   Top 5 tahmin:")
        for i, (item_idx, prob) in enumerate(zip(item_indices[:5], probabilities[:5]), 1):
            is_correct = "✅" if item_idx == next_item_true else "  "
            print(f"      {is_correct} {i}. Ürün {item_idx + 1} - Olasılık: {prob:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Transformer örneği tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        example_transformer_recommender()
    except ImportError as e:
        print(f"❌ Hata: {e}")
        print("💡 TensorFlow yüklü olduğundan emin olun: pip install tensorflow")
    except Exception as e:
        print(f"❌ Hata: {e}")

