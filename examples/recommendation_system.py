"""
Öneri Sistemi Örneği
SVD ve ALS algoritmalarını kullanarak öneri sistemi oluşturma
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.svd import SVDRecommender
from algorithms.als import ALSRecommender
from utils.data_loader import generate_rating_matrix
from utils.visualization import plot_ratings_matrix, plot_recommendations


def example_svd_recommender():
    """SVD tabanlı öneri sistemi örneği"""
    print("=" * 60)
    print("SVD Tabanlı Öneri Sistemi")
    print("=" * 60)
    
    # Örnek veri oluştur
    print("\n1. Örnek rating matrisi oluşturuluyor...")
    rating_matrix = generate_rating_matrix(n_users=100, n_items=50, sparsity=0.7)
    print(f"   Rating matrisi boyutu: {rating_matrix.shape}")
    print(f"   Dolu rating sayısı: {np.sum(~np.isnan(rating_matrix))}")
    
    # Train-test split
    np.random.seed(42)
    mask = ~np.isnan(rating_matrix)
    test_indices = np.random.choice(np.where(mask)[0], 
                                   size=int(0.2 * np.sum(mask)), 
                                   replace=False)
    test_mask = np.zeros_like(mask, dtype=bool)
    test_mask.flat[test_indices] = True
    
    train_matrix = rating_matrix.copy()
    train_matrix[test_mask] = np.nan
    
    test_matrix = np.full_like(rating_matrix, np.nan)
    test_matrix[test_mask] = rating_matrix[test_mask]
    
    # SVD modeli
    print("\n2. SVD modeli eğitiliyor...")
    svd_model = SVDRecommender(n_components=20)
    svd_model.fit(train_matrix)
    
    # Tekil değerleri göster
    singular_values = svd_model.get_singular_values()
    print(f"   Kullanılan tekil değer sayısı: {len(singular_values)}")
    print(f"   İlk 5 tekil değer: {singular_values[:5]}")
    
    # Tahmin
    print("\n3. Tahminler yapılıyor...")
    rmse = svd_model.evaluate(test_matrix)
    print(f"   Test RMSE: {rmse:.4f}")
    
    # Öneriler
    print("\n4. Öneriler üretiliyor...")
    user_idx = 0
    predictions = svd_model.predict_all()[user_idx]
    rated_items = ~np.isnan(rating_matrix[user_idx])
    predictions[rated_items] = -np.inf
    
    top_items = np.argsort(predictions)[::-1][:10]
    top_ratings = predictions[top_items]
    
    print(f"   Kullanıcı {user_idx+1} için en iyi 5 öneri:")
    for i, (item, rating) in enumerate(zip(top_items[:5], top_ratings[:5]), 1):
        print(f"   {i}. Ürün {item+1}: {rating:.2f}")
    
    return svd_model, rating_matrix


def example_als_recommender():
    """ALS tabanlı öneri sistemi örneği"""
    print("\n" + "=" * 60)
    print("ALS Tabanlı Öneri Sistemi")
    print("=" * 60)
    
    # Örnek veri oluştur
    print("\n1. Örnek rating matrisi oluşturuluyor...")
    rating_matrix = generate_rating_matrix(n_users=100, n_items=50, sparsity=0.7)
    
    # Train-test split
    np.random.seed(42)
    mask = ~np.isnan(rating_matrix)
    test_indices = np.random.choice(np.where(mask)[0], 
                                   size=int(0.2 * np.sum(mask)), 
                                   replace=False)
    test_mask = np.zeros_like(mask, dtype=bool)
    test_mask.flat[test_indices] = True
    
    train_matrix = rating_matrix.copy()
    train_matrix[test_mask] = np.nan
    
    test_matrix = np.full_like(rating_matrix, np.nan)
    test_matrix[test_mask] = rating_matrix[test_mask]
    
    # ALS modeli
    print("\n2. ALS modeli eğitiliyor...")
    als_model = ALSRecommender(n_factors=20, regularization=0.1, iterations=15)
    als_model.fit(train_matrix, implicit=False)
    
    # Tahmin
    print("\n3. Tahminler yapılıyor...")
    rmse = als_model.evaluate(test_matrix)
    print(f"   Test RMSE: {rmse:.4f}")
    
    # Öneriler
    print("\n4. Öneriler üretiliyor...")
    user_idx = 0
    recommendations = als_model.recommend(user_idx, n_recommendations=10, 
                                         exclude_rated=True, 
                                         rating_matrix=rating_matrix)
    
    print(f"   Kullanıcı {user_idx+1} için en iyi 5 öneri:")
    for i, (item, rating) in enumerate(zip(recommendations[0][:5], 
                                          recommendations[1][:5]), 1):
        print(f"   {i}. Ürün {item+1}: {rating:.2f}")
    
    # Benzer item'lar
    print("\n5. Benzer item'lar bulunuyor...")
    item_idx = 0
    similar_items = als_model.get_similar_items(item_idx, n_similar=5)
    print(f"   Ürün {item_idx+1} ile benzer 5 ürün:")
    for i, (item, similarity) in enumerate(zip(similar_items[0][:5], 
                                               similar_items[1][:5]), 1):
        print(f"   {i}. Ürün {item+1}: Benzerlik = {similarity:.4f}")
    
    return als_model, rating_matrix


def compare_algorithms():
    """SVD ve ALS algoritmalarını karşılaştır"""
    print("\n" + "=" * 60)
    print("Algoritma Karşılaştırması")
    print("=" * 60)
    
    rating_matrix = generate_rating_matrix(n_users=100, n_items=50, sparsity=0.7)
    
    # Train-test split
    np.random.seed(42)
    mask = ~np.isnan(rating_matrix)
    test_indices = np.random.choice(np.where(mask)[0], 
                                   size=int(0.2 * np.sum(mask)), 
                                   replace=False)
    test_mask = np.zeros_like(mask, dtype=bool)
    test_mask.flat[test_indices] = True
    
    train_matrix = rating_matrix.copy()
    train_matrix[test_mask] = np.nan
    
    test_matrix = np.full_like(rating_matrix, np.nan)
    test_matrix[test_mask] = rating_matrix[test_mask]
    
    # SVD
    print("\nSVD eğitiliyor...")
    svd_model = SVDRecommender(n_components=20)
    svd_model.fit(train_matrix)
    svd_rmse = svd_model.evaluate(test_matrix)
    
    # ALS
    print("ALS eğitiliyor...")
    als_model = ALSRecommender(n_factors=20, regularization=0.1, iterations=15)
    als_model.fit(train_matrix)
    als_rmse = als_model.evaluate(test_matrix)
    
    print("\n" + "-" * 60)
    print("Sonuçlar:")
    print(f"  SVD RMSE: {svd_rmse:.4f}")
    print(f"  ALS RMSE: {als_rmse:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    # SVD örneği
    svd_model, rating_matrix = example_svd_recommender()
    
    # ALS örneği
    als_model, _ = example_als_recommender()
    
    # Karşılaştırma
    compare_algorithms()

