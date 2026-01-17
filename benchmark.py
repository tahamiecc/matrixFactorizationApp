"""
Performans Karşılaştırma ve Benchmark Modülü
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.svd import SVDRecommender
from algorithms.als import ALSRecommender
from algorithms.pca import PCAAnalyzer
from algorithms.nmf import NMFImageProcessor, NMFTopicModeler
from utils.data_loader import (
    generate_rating_matrix,
    generate_sample_data,
    load_sample_images,
    generate_text_corpus
)


class BenchmarkSuite:
    """Algoritma performans karşılaştırma suite'i"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_recommendation_algorithms(self, n_users=100, n_items=50, 
                                           sparsity=0.7, n_runs=3):
        """
        Öneri sistemi algoritmalarını karşılaştırır
        
        Args:
            n_users: Kullanıcı sayısı
            n_items: Ürün sayısı
            sparsity: Eksik veri oranı
            n_runs: Çalıştırma sayısı (ortalama için)
            
        Returns:
            Sonuçlar DataFrame'i
        """
        print("=" * 60)
        print("Öneri Sistemi Algoritmaları Benchmark")
        print("=" * 60)
        
        results = []
        
        for run in range(n_runs):
            print(f"\nÇalıştırma {run + 1}/{n_runs}...")
            
            # Veri oluştur
            rating_matrix = generate_rating_matrix(
                n_users=n_users, 
                n_items=n_items, 
                sparsity=sparsity,
                random_state=42 + run
            )
            
            # Train-test split
            np.random.seed(42 + run)
            mask = ~np.isnan(rating_matrix)
            test_indices = np.random.choice(
                np.where(mask)[0], 
                size=int(0.2 * np.sum(mask)), 
                replace=False
            )
            test_mask = np.zeros_like(mask, dtype=bool)
            test_mask.flat[test_indices] = True
            
            train_matrix = rating_matrix.copy()
            train_matrix[test_mask] = np.nan
            
            test_matrix = np.full_like(rating_matrix, np.nan)
            test_matrix[test_mask] = rating_matrix[test_mask]
            
            # SVD
            print("  SVD eğitiliyor...")
            start_time = time.time()
            svd_model = SVDRecommender(n_components=20)
            svd_model.fit(train_matrix)
            svd_train_time = time.time() - start_time
            
            start_time = time.time()
            svd_rmse = svd_model.evaluate(test_matrix)
            svd_test_time = time.time() - start_time
            
            results.append({
                'Run': run + 1,
                'Algorithm': 'SVD',
                'RMSE': svd_rmse,
                'Train Time (s)': svd_train_time,
                'Test Time (s)': svd_test_time
            })
            
            # ALS
            print("  ALS eğitiliyor...")
            start_time = time.time()
            als_model = ALSRecommender(n_factors=20, regularization=0.1, iterations=15)
            als_model.fit(train_matrix)
            als_train_time = time.time() - start_time
            
            start_time = time.time()
            als_rmse = als_model.evaluate(test_matrix)
            als_test_time = time.time() - start_time
            
            results.append({
                'Run': run + 1,
                'Algorithm': 'ALS',
                'RMSE': als_rmse,
                'Train Time (s)': als_train_time,
                'Test Time (s)': als_test_time
            })
        
        # Sonuçları özetle
        results_df = pd.DataFrame(results)
        summary = results_df.groupby('Algorithm').agg({
            'RMSE': ['mean', 'std'],
            'Train Time (s)': ['mean', 'std'],
            'Test Time (s)': ['mean', 'std']
        }).round(4)
        
        print("\n" + "=" * 60)
        print("Sonuçlar:")
        print("=" * 60)
        print(summary)
        
        return results_df, summary
    
    def benchmark_pca(self, n_samples=500, n_features=50, n_runs=3):
        """
        PCA performansını test eder
        
        Args:
            n_samples: Örnek sayısı
            n_features: Özellik sayısı
            n_runs: Çalıştırma sayısı
            
        Returns:
            Sonuçlar DataFrame'i
        """
        print("=" * 60)
        print("PCA Benchmark")
        print("=" * 60)
        
        results = []
        
        for run in range(n_runs):
            print(f"\nÇalıştırma {run + 1}/{n_runs}...")
            
            X, _ = generate_sample_data(
                n_samples=n_samples, 
                n_features=n_features,
                random_state=42 + run
            )
            
            # PCA
            start_time = time.time()
            pca = PCAAnalyzer(n_components=None)
            X_transformed = pca.fit_transform(X)
            fit_time = time.time() - start_time
            
            # Yeniden oluşturma
            start_time = time.time()
            X_reconstructed = pca.inverse_transform(X_transformed)
            reconstruct_time = time.time() - start_time
            
            # Hata
            mse = np.mean((X - X_reconstructed)**2)
            variance_explained = np.sum(pca.explained_variance_ratio_[:10])
            
            results.append({
                'Run': run + 1,
                'Fit Time (s)': fit_time,
                'Reconstruct Time (s)': reconstruct_time,
                'MSE': mse,
                'Variance Explained (10 comp)': variance_explained
            })
        
        results_df = pd.DataFrame(results)
        summary = results_df.agg(['mean', 'std']).round(4)
        
        print("\n" + "=" * 60)
        print("Sonuçlar:")
        print("=" * 60)
        print(summary)
        
        return results_df, summary
    
    def benchmark_nmf_image(self, n_images=100, n_components_list=[10, 20, 30]):
        """
        NMF görüntü işleme performansını test eder
        
        Args:
            n_images: Görüntü sayısı
            n_components_list: Test edilecek bileşen sayıları
            
        Returns:
            Sonuçlar DataFrame'i
        """
        print("=" * 60)
        print("NMF Görüntü İşleme Benchmark")
        print("=" * 60)
        
        images_flat, image_shape = load_sample_images(n_images=n_images)
        
        results = []
        
        for n_comp in n_components_list:
            print(f"\n{n_comp} bileşen test ediliyor...")
            
            start_time = time.time()
            nmf = NMFImageProcessor(n_components=n_comp)
            nmf.fit(images_flat)
            fit_time = time.time() - start_time
            
            start_time = time.time()
            reconstructed = nmf.reconstruct()
            reconstruct_time = time.time() - start_time
            
            mse = np.mean((images_flat - reconstructed)**2)
            compression_ratio = nmf.get_compression_ratio(image_shape)
            
            results.append({
                'Components': n_comp,
                'Fit Time (s)': fit_time,
                'Reconstruct Time (s)': reconstruct_time,
                'MSE': mse,
                'Compression Ratio': compression_ratio
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print("Sonuçlar:")
        print("=" * 60)
        print(results_df)
        
        return results_df
    
    def benchmark_nmf_topic(self, n_documents=200, n_topics_list=[3, 5, 7, 10]):
        """
        NMF topic modeling performansını test eder
        
        Args:
            n_documents: Doküman sayısı
            n_topics_list: Test edilecek topic sayıları
            
        Returns:
            Sonuçlar DataFrame'i
        """
        print("=" * 60)
        print("NMF Topic Modeling Benchmark")
        print("=" * 60)
        
        documents = generate_text_corpus(n_documents=n_documents)
        
        results = []
        
        for n_topics in n_topics_list:
            print(f"\n{n_topics} topic test ediliyor...")
            
            start_time = time.time()
            nmf = NMFTopicModeler(n_topics=n_topics, max_iter=200)
            nmf.fit(documents, max_features=500, min_df=2, max_df=0.95)
            fit_time = time.time() - start_time
            
            coherence = nmf.get_topic_coherence()
            topics = nmf.get_topics(n_words=10)
            
            # Ortalama topic kalitesi (en yüksek skorlu kelimelerin ortalaması)
            avg_topic_quality = np.mean([
                np.mean([score for _, score in words[:5]]) 
                for words in topics.values()
            ])
            
            results.append({
                'Topics': n_topics,
                'Fit Time (s)': fit_time,
                'Coherence': coherence,
                'Avg Topic Quality': avg_topic_quality
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print("Sonuçlar:")
        print("=" * 60)
        print(results_df)
        
        return results_df
    
    def plot_benchmark_results(self, results_df, metric='RMSE', title='Benchmark Sonuçları'):
        """
        Benchmark sonuçlarını görselleştirir
        
        Args:
            results_df: Sonuçlar DataFrame'i
            metric: Görselleştirilecek metrik
            title: Grafik başlığı
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'Algorithm' in results_df.columns:
            # Algoritma karşılaştırması
            algorithms = results_df['Algorithm'].unique()
            means = results_df.groupby('Algorithm')[metric].mean()
            stds = results_df.groupby('Algorithm')[metric].std()
            
            ax.bar(means.index, means.values, yerr=stds.values, 
                  alpha=0.7, capsize=5)
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            # Diğer metrikler
            if metric in results_df.columns:
                ax.plot(results_df.index, results_df[metric], 'o-')
                ax.set_xlabel('Run')
                ax.set_ylabel(metric)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def run_full_benchmark():
    """Tüm benchmark'ları çalıştırır"""
    suite = BenchmarkSuite()
    
    # Öneri sistemleri
    rec_results, rec_summary = suite.benchmark_recommendation_algorithms(
        n_users=100, n_items=50, n_runs=3
    )
    
    # PCA
    pca_results, pca_summary = suite.benchmark_pca(n_samples=500, n_features=50, n_runs=3)
    
    # NMF görüntü
    nmf_image_results = suite.benchmark_nmf_image(n_images=100, n_components_list=[10, 20, 30])
    
    # NMF topic
    nmf_topic_results = suite.benchmark_nmf_topic(n_documents=200, n_topics_list=[3, 5, 7, 10])
    
    return {
        'recommendation': (rec_results, rec_summary),
        'pca': (pca_results, pca_summary),
        'nmf_image': nmf_image_results,
        'nmf_topic': nmf_topic_results
    }


if __name__ == "__main__":
    results = run_full_benchmark()
    
    # Görselleştirme
    suite = BenchmarkSuite()
    
    # Öneri sistemleri karşılaştırması
    fig1 = suite.plot_benchmark_results(
        results['recommendation'][0], 
        metric='RMSE',
        title='Öneri Sistemi Algoritmaları - RMSE Karşılaştırması'
    )
    plt.savefig('benchmark_recommendation.png', dpi=150, bbox_inches='tight')
    
    # NMF görüntü bileşen analizi
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    nmf_img = results['nmf_image']
    ax1.plot(nmf_img['Components'], nmf_img['MSE'], 'o-')
    ax1.set_xlabel('Bileşen Sayısı')
    ax1.set_ylabel('MSE')
    ax1.set_title('NMF Görüntü - MSE vs Bileşen Sayısı')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(nmf_img['Components'], nmf_img['Compression Ratio'], 'o-')
    ax2.set_xlabel('Bileşen Sayısı')
    ax2.set_ylabel('Sıkıştırma Oranı')
    ax2.set_title('NMF Görüntü - Sıkıştırma Oranı vs Bileşen Sayısı')
    ax2.grid(True, alpha=0.3)
    plt.savefig('benchmark_nmf_image.png', dpi=150, bbox_inches='tight')
    
    # NMF topic analizi
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    nmf_topic = results['nmf_topic']
    ax1.plot(nmf_topic['Topics'], nmf_topic['Coherence'], 'o-')
    ax1.set_xlabel('Topic Sayısı')
    ax1.set_ylabel('Tutarlılık')
    ax1.set_title('NMF Topic - Tutarlılık vs Topic Sayısı')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(nmf_topic['Topics'], nmf_topic['Avg Topic Quality'], 'o-')
    ax2.set_xlabel('Topic Sayısı')
    ax2.set_ylabel('Ortalama Topic Kalitesi')
    ax2.set_title('NMF Topic - Kalite vs Topic Sayısı')
    ax2.grid(True, alpha=0.3)
    plt.savefig('benchmark_nmf_topic.png', dpi=150, bbox_inches='tight')
    
    print("\nBenchmark tamamlandı! Grafikler kaydedildi.")
    plt.show()

