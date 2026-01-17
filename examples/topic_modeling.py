"""
Topic Modeling Örneği
NMF kullanarak metin madenciliği ve topic modeling
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.nmf import NMFTopicModeler
from utils.data_loader import generate_text_corpus
from utils.visualization import plot_topic_words


def example_topic_modeling():
    """NMF ile topic modeling örneği"""
    print("=" * 60)
    print("NMF ile Topic Modeling")
    print("=" * 60)
    
    # Metin korpusu oluştur
    print("\n1. Metin korpusu oluşturuluyor...")
    documents = generate_text_corpus(n_documents=200)
    print(f"   Doküman sayısı: {len(documents)}")
    print(f"   Örnek doküman: {documents[0][:100]}...")
    
    # NMF modeli
    print("\n2. NMF modeli eğitiliyor...")
    nmf_model = NMFTopicModeler(n_topics=5, max_iter=200)
    nmf_model.fit(documents, max_features=500, min_df=2, max_df=0.95)
    
    # Topic'leri göster
    print("\n3. Topic'ler analiz ediliyor...")
    topics = nmf_model.get_topics(n_words=10)
    
    for topic_name, words_scores in topics.items():
        print(f"\n   {topic_name}:")
        for word, score in words_scores[:5]:
            print(f"      - {word}: {score:.4f}")
    
    # Doküman-topic dağılımı
    print("\n4. Doküman-topic dağılımları hesaplanıyor...")
    doc_topics = nmf_model.get_document_topics()
    print(f"   Doküman-topic matrisi boyutu: {doc_topics.shape}")
    
    # Her doküman için en yüksek skorlu topic
    dominant_topics = np.argmax(doc_topics, axis=1)
    print("\n   İlk 10 dokümanın dominant topic'leri:")
    for i in range(min(10, len(documents))):
        topic_idx = dominant_topics[i]
        topic_score = doc_topics[i, topic_idx]
        print(f"   Doküman {i+1}: Topic {topic_idx+1} (skor: {topic_score:.3f})")
    
    # Yeni doküman tahmini
    print("\n5. Yeni doküman için topic tahmini...")
    new_doc = "computer software algorithm data network system programming"
    top_topic, all_scores = nmf_model.predict_topic(new_doc)
    print(f"   Doküman: '{new_doc}'")
    print(f"   En uygun topic: Topic {top_topic+1} (skor: {all_scores[top_topic]:.3f})")
    print(f"   Tüm topic skorları: {all_scores}")
    
    # Topic tutarlılığı
    coherence = nmf_model.get_topic_coherence()
    print(f"\n6. Topic tutarlılığı: {coherence:.4f}")
    
    # Görselleştirme
    print("\n7. Görselleştirme oluşturuluyor...")
    fig = plot_topic_words(topics, n_words=10, figsize=(18, 6))
    plt.savefig('topic_modeling_results.png', dpi=150, bbox_inches='tight')
    print("   Grafik 'topic_modeling_results.png' olarak kaydedildi.")
    
    # Doküman-topic dağılımı heatmap
    fig2, ax = plt.subplots(figsize=(12, 8))
    import seaborn as sns
    sns.heatmap(doc_topics[:50].T, cmap='YlOrRd', ax=ax, 
                yticklabels=[f'Topic {i+1}' for i in range(nmf_model.n_topics)],
                xticklabels=[f'Doc {i+1}' for i in range(50)])
    ax.set_xlabel('Dokümanlar')
    ax.set_ylabel('Topic\'ler')
    ax.set_title('Doküman-Topic Dağılımı (İlk 50 Doküman)')
    plt.tight_layout()
    plt.savefig('document_topic_distribution.png', dpi=150, bbox_inches='tight')
    print("   Grafik 'document_topic_distribution.png' olarak kaydedildi.")
    
    return nmf_model, topics, doc_topics


def example_topic_keywords():
    """Topic keyword analizi"""
    print("\n" + "=" * 60)
    print("Topic Keyword Analizi")
    print("=" * 60)
    
    documents = generate_text_corpus(n_documents=150)
    nmf_model = NMFTopicModeler(n_topics=5)
    nmf_model.fit(documents)
    
    print("\nHer topic için anahtar kelimeler:")
    for topic_idx in range(nmf_model.n_topics):
        keywords = nmf_model.get_topic_keywords(topic_idx, n_words=15)
        print(f"\nTopic {topic_idx + 1}:")
        for word, score in keywords[:10]:
            print(f"  {word}: {score:.4f}")
    
    return nmf_model


if __name__ == "__main__":
    # Topic modeling
    nmf_model, topics, doc_topics = example_topic_modeling()
    
    # Keyword analizi
    nmf_keywords = example_topic_keywords()
    
    plt.show()

