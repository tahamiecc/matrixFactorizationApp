"""
Veri yükleme ve örnek veri oluşturma fonksiyonları
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, fetch_olivetti_faces
import warnings
warnings.filterwarnings('ignore')


def generate_sample_data(n_samples=1000, n_features=50, n_clusters=5, random_state=42):
    """
    PCA için örnek veri oluşturur
    
    Args:
        n_samples: Örnek sayısı
        n_features: Özellik sayısı
        n_clusters: Küme sayısı
        random_state: Rastgelelik için seed
        
    Returns:
        (X, y) tuple - Veri ve etiketler
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        cluster_std=2.0
    )
    return X, y


def generate_rating_matrix(n_users=100, n_items=50, sparsity=0.7, random_state=42):
    """
    Öneri sistemi için örnek rating matrisi oluşturur
    
    Args:
        n_users: Kullanıcı sayısı
        n_items: Ürün sayısı
        sparsity: Eksik veri oranı (0-1)
        random_state: Rastgelelik için seed
        
    Returns:
        Rating matrisi (NaN değerlerle)
    """
    np.random.seed(random_state)
    
    # Gerçekçi bir rating matrisi oluştur
    # Bazı kullanıcılar daha yüksek, bazıları daha düşük rating verir
    user_biases = np.random.normal(0, 0.8, n_users)  # Daha fazla çeşitlilik
    item_biases = np.random.normal(0, 0.6, n_items)  # Daha fazla çeşitlilik
    
    # Latent faktörler - daha fazla çeşitlilik için
    n_factors = 15  # Daha fazla faktör
    user_factors = np.random.normal(0, 1.2, (n_users, n_factors))  # Daha fazla varyans
    item_factors = np.random.normal(0, 1.2, (n_items, n_factors))
    
    # Rating matrisi - daha çeşitli rating'ler için
    rating_matrix = (
        3.0 +  # Biraz daha düşük global ortalama (daha fazla çeşitlilik için)
        user_biases[:, np.newaxis] +
        item_biases[np.newaxis, :] +
        np.dot(user_factors, item_factors.T) * 0.3 +  # Faktör etkisini artır
        np.random.normal(0, 0.6, (n_users, n_items))  # Daha fazla gürültü
    )
    
    # 1-5 aralığına sınırla
    rating_matrix = np.clip(rating_matrix, 1, 5)
    
    # Daha fazla çeşitlilik için bazı rating'leri ekstrem değerlere çek
    extreme_mask = np.random.random((n_users, n_items)) < 0.1  # %10 ekstrem
    rating_matrix[extreme_mask] = np.random.choice([1.0, 1.5, 4.5, 5.0], 
                                                   size=np.sum(extreme_mask))
    
    # Sparsity uygula (bazı rating'leri NaN yap)
    mask = np.random.random((n_users, n_items)) < sparsity
    rating_matrix[mask] = np.nan
    
    return rating_matrix


def load_sample_images(n_images=100):
    """
    Örnek görüntü verisi yükler veya oluşturur
    
    Args:
        n_images: Yüklenecek görüntü sayısı
        
    Returns:
        Görüntü matrisi ve görüntü boyutları
    """
    try:
        # Olivetti faces dataset'ini dene
        faces = fetch_olivetti_faces()
        images = faces.images[:n_images]
        image_shape = images[0].shape
        # Görüntüleri düzleştir
        images_flat = images.reshape(n_images, -1)
        return images_flat, image_shape
    except:
        # Eğer yüklenemezse, sentetik görüntüler oluştur
        np.random.seed(42)
        image_shape = (64, 64)
        n_pixels = np.prod(image_shape)
        
        # Basit pattern'ler oluştur
        images_flat = []
        for i in range(n_images):
            # Farklı pattern'ler
            pattern = np.random.choice(['circle', 'square', 'noise'])
            if pattern == 'circle':
                img = create_circle_image(image_shape, i)
            elif pattern == 'square':
                img = create_square_image(image_shape, i)
            else:
                img = np.random.random(image_shape)
            images_flat.append(img.flatten())
        
        return np.array(images_flat), image_shape


def create_circle_image(shape, seed):
    """Daire pattern'i oluşturur"""
    np.random.seed(seed)
    h, w = shape
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 3 + np.random.randint(-5, 5)
    
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img = np.zeros(shape)
    img[mask] = 1.0
    return img


def create_square_image(shape, seed):
    """Kare pattern'i oluşturur"""
    np.random.seed(seed)
    h, w = shape
    size = min(w, h) // 3 + np.random.randint(-5, 5)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    
    img = np.zeros(shape)
    img[start_y:start_y+size, start_x:start_x+size] = 1.0
    return img


def generate_text_corpus(n_documents=100):
    """
    Topic modeling için örnek metin korpusu oluşturur
    
    Args:
        n_documents: Doküman sayısı
        
    Returns:
        Doküman listesi
    """
    # Farklı topic'ler için kelime setleri
    topics = {
        'technology': ['computer', 'software', 'algorithm', 'data', 'network', 
                      'system', 'digital', 'code', 'programming', 'internet'],
        'science': ['research', 'experiment', 'hypothesis', 'theory', 'discovery',
                   'analysis', 'method', 'study', 'observation', 'evidence'],
        'sports': ['game', 'player', 'team', 'match', 'championship', 'victory',
                  'competition', 'training', 'coach', 'stadium'],
        'health': ['medicine', 'treatment', 'patient', 'disease', 'health',
                  'doctor', 'hospital', 'therapy', 'diagnosis', 'recovery'],
        'business': ['company', 'market', 'profit', 'investment', 'strategy',
                    'management', 'customer', 'revenue', 'growth', 'finance']
    }
    
    documents = []
    np.random.seed(42)
    
    for i in range(n_documents):
        # Her doküman için 1-2 topic seç
        selected_topics = np.random.choice(list(topics.keys()), 
                                         size=np.random.randint(1, 3), 
                                         replace=False)
        
        # Her topic'ten 5-10 kelime seç
        doc_words = []
        for topic in selected_topics:
            n_words = np.random.randint(5, 10)
            words = np.random.choice(topics[topic], size=n_words, replace=True)
            doc_words.extend(words)
        
        # Dokümanı oluştur
        document = ' '.join(doc_words)
        documents.append(document)
    
    return documents


def generate_noisy_data(original_data, noise_level=0.1):
    """
    Gürültülü veri oluşturur (SVD noise reduction için)
    
    Args:
        original_data: Orijinal veri matrisi
        noise_level: Gürültü seviyesi (0-1)
        
    Returns:
        Gürültülü veri matrisi
    """
    noise = np.random.normal(0, noise_level, original_data.shape)
    noisy_data = original_data + noise
    return noisy_data


def load_rating_data_from_file(file, user_col=None, item_col=None, rating_col=None):
    """
    CSV/Excel dosyasından rating verisi yükler ve rating matrisine çevirir
    
    Args:
        file: Yüklenen dosya (Streamlit file_uploader'dan veya BytesIO)
        user_col: Kullanıcı ID sütunu adı (None ise otomatik tespit)
        item_col: Ürün ID sütunu adı (None ise otomatik tespit)
        rating_col: Rating sütunu adı (None ise otomatik tespit)
        
    Returns:
        (rating_matrix, user_mapping, item_mapping) tuple
        - rating_matrix: n_users x n_items numpy array (NaN değerlerle)
        - user_mapping: Orijinal user ID'lerden indekslere mapping dict
        - item_mapping: Orijinal item ID'lerden indekslere mapping dict
    """
    # Streamlit file_uploader'dan gelen dosya stream'i bir kez okununca tükenir
    # Bu yüzden dosya içeriğini hafızaya al
    import io
    
    # Dosya adını al (BytesIO veya file objesi olabilir)
    file_name = getattr(file, 'name', 'unknown.csv')
    
    # Dosya içeriğini oku
    try:
        # Eğer zaten BytesIO ise, içeriği al
        if isinstance(file, io.BytesIO):
            file_content = file.getvalue()
            file_bytes = io.BytesIO(file_content)
        else:
            # Streamlit file_uploader'dan gelen dosya
            file_content = file.read()
            file_bytes = io.BytesIO(file_content)
    except AttributeError:
        # Eğer read() metodu yoksa, direkt kullan
        file_bytes = file
    
    # Dosya tipine göre yükle
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_bytes)
    elif file_name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_bytes)
    else:
        raise ValueError("Desteklenen formatlar: CSV, Excel (.xlsx, .xls)")
    
    # Dosya boş mu kontrol et
    if df.empty:
        raise ValueError("Dosya boş! Lütfen en az bir satır veri içeren bir dosya yükleyin.")
    
    # Sütun kontrolü
    if len(df.columns) == 0:
        raise ValueError("Dosyada hiç sütun bulunamadı!")
    
    if len(df.columns) < 3:
        raise ValueError(
            f"Dosyada yeterli sütun yok! En az 3 sütun gerekli (user_id, item_id, rating).\n"
            f"Mevcut sütunlar ({len(df.columns)} adet): {list(df.columns)}"
        )
    
    # Sütun isimlerini otomatik tespit et
    if user_col is None:
        # Olası kullanıcı sütunu isimleri
        possible_user_cols = ['user_id', 'user', 'userId', 'UserID', 'userid', 
                             'customer_id', 'customer', 'CustomerID']
        user_col = None
        for col in df.columns:
            if col.lower() in [c.lower() for c in possible_user_cols]:
                user_col = col
                break
        if user_col is None:
            # İlk sütunu dene
            user_col = df.columns[0]
    
    if item_col is None:
        # Olası ürün sütunu isimleri
        possible_item_cols = ['item_id', 'item', 'itemId', 'ItemID', 'itemid',
                             'product_id', 'product', 'ProductID', 'movie_id', 'movie']
        item_col = None
        for col in df.columns:
            if col.lower() in [c.lower() for c in possible_item_cols]:
                item_col = col
                break
        if item_col is None:
            # İkinci sütunu dene
            item_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    if rating_col is None:
        # Olası rating sütunu isimleri
        possible_rating_cols = ['rating', 'Rating', 'RATING', 'score', 'Score',
                               'value', 'Value', 'preference', 'Preference']
        rating_col = None
        for col in df.columns:
            if col.lower() in [c.lower() for c in possible_rating_cols]:
                rating_col = col
                break
        if rating_col is None:
            # Üçüncü sütunu dene
            rating_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]
    
    # Gerekli sütunları kontrol et
    required_cols = [user_col, item_col, rating_col]
    if not all(col in df.columns for col in required_cols):
        available_cols = list(df.columns)
        raise ValueError(
            f"Gerekli sütunlar bulunamadı!\n"
            f"Tespit edilen sütunlar: {required_cols}\n"
            f"Dosyadaki mevcut sütunlar: {available_cols}\n"
            f"Lütfen dosyanızda 'user_id', 'item_id', 'rating' gibi sütunlar olduğundan emin olun."
        )
    
    # Veriyi temizle
    df_clean = df[[user_col, item_col, rating_col]].copy()
    original_count = len(df_clean)
    df_clean = df_clean.dropna()
    after_dropna_count = len(df_clean)
    
    # Rating değerlerini sayısal yap
    df_clean[rating_col] = pd.to_numeric(df_clean[rating_col], errors='coerce')
    after_numeric_count = len(df_clean)
    df_clean = df_clean.dropna()
    final_count = len(df_clean)
    
    # Boş veri kontrolü - daha detaylı hata mesajı
    if df_clean.empty:
        # Diagnostik bilgileri topla
        error_details = []
        error_details.append(f"Tespit edilen sütunlar: user_col='{user_col}', item_col='{item_col}', rating_col='{rating_col}'")
        error_details.append(f"Dosyadaki toplam satır sayısı: {len(df)}")
        error_details.append(f"Seçilen sütunlardaki toplam satır: {original_count}")
        error_details.append(f"dropna() sonrası satır sayısı: {after_dropna_count}")
        error_details.append(f"to_numeric() sonrası satır sayısı: {after_numeric_count}")
        error_details.append(f"Son temizleme sonrası satır sayısı: {final_count}")
        
        # Örnek veri göster
        if len(df) > 0:
            error_details.append(f"\nDosyadaki ilk 5 satır örneği:")
            error_details.append(str(df.head()))
            error_details.append(f"\nSeçilen sütunların veri tipleri:")
            error_details.append(f"  {user_col}: {df[user_col].dtype}")
            error_details.append(f"  {item_col}: {df[item_col].dtype}")
            error_details.append(f"  {rating_col}: {df[rating_col].dtype}")
            
            # Rating sütunundaki benzersiz değerler (ilk 10)
            unique_ratings = df[rating_col].dropna().unique()[:10]
            error_details.append(f"\nRating sütunundaki örnek değerler (ilk 10): {list(unique_ratings)}")
        
        error_msg = "Dosyada geçerli rating verisi bulunamadı!\n\n"
        error_msg += "Olası nedenler:\n"
        error_msg += "1. Rating sütunu sayısal değil (metin, boş, vb.)\n"
        error_msg += "2. Yanlış sütunlar seçilmiş olabilir\n"
        error_msg += "3. Tüm satırlarda eksik veri (NaN) var\n\n"
        error_msg += "Detaylar:\n" + "\n".join(error_details)
        error_msg += "\n\nLütfen dosyanızda en az bir tane geçerli (user_id, item_id, rating) üçlüsü olduğundan emin olun."
        error_msg += "\nRating değerleri sayısal olmalıdır (örn: 1, 2, 3, 4, 5 veya 0.5, 1.0, 2.5 gibi)."
        
        raise ValueError(error_msg)
    
    # Rating aralığını kontrol et ve normalize et (1-5 aralığına)
    if len(df_clean) == 0:
        raise ValueError("Dosyada hiç geçerli rating değeri yok!")
    
    min_rating = df_clean[rating_col].min()
    max_rating = df_clean[rating_col].max()
    
    # min/max kontrolü (eğer tüm değerler aynıysa)
    if pd.isna(min_rating) or pd.isna(max_rating):
        raise ValueError("Dosyada geçerli rating değeri bulunamadı! Tüm rating değerleri NaN veya sayısal değil.")
    
    if max_rating > 5 or min_rating < 1:
        # Rating'leri 1-5 aralığına normalize et
        df_clean[rating_col] = 1 + (df_clean[rating_col] - min_rating) / (max_rating - min_rating) * 4
    
    # User ve Item ID'lerini indekslere çevir
    unique_users = df_clean[user_col].unique()
    unique_items = df_clean[item_col].unique()
    
    user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(unique_users))}
    item_mapping = {item_id: idx for idx, item_id in enumerate(sorted(unique_items))}
    
    # Rating matrisi oluştur - büyük matrisler için sparse kullan
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    # Büyük matrisler için sparse matrix kullan (10M+ hücre)
    matrix_size = n_users * n_items
    use_sparse = matrix_size > 10_000_000  # 10 milyon hücreden büyükse
    
    if use_sparse:
        # Sparse matrix oluştur (sadece mevcut rating'leri sakla)
        from scipy.sparse import csr_matrix
        rows = []
        cols = []
        values = []
        
        for _, row in df_clean.iterrows():
            user_idx = user_mapping[row[user_col]]
            item_idx = item_mapping[row[item_col]]
            rows.append(user_idx)
            cols.append(item_idx)
            values.append(row[rating_col])
        
        rating_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float64
        )
    else:
        # Küçük matrisler için dense matrix
        rating_matrix = np.full((n_users, n_items), np.nan)
        
        # Veriyi matrise doldur
        for _, row in df_clean.iterrows():
            user_idx = user_mapping[row[user_col]]
            item_idx = item_mapping[row[item_col]]
            rating_matrix[user_idx, item_idx] = row[rating_col]
    
    return rating_matrix, user_mapping, item_mapping


def load_rating_matrix_from_file(file):
    """
    Zaten rating matrisi formatında olan CSV/Excel dosyasını yükler
    
    Args:
        file: Yüklenen dosya (Streamlit file_uploader'dan veya BytesIO)
        
    Returns:
        Rating matrisi (numpy array, NaN değerlerle)
    """
    # Streamlit file_uploader'dan gelen dosya stream'i bir kez okununca tükenir
    # Bu yüzden dosya içeriğini hafızaya al
    import io
    
    # Dosya adını al (BytesIO veya file objesi olabilir)
    file_name = getattr(file, 'name', 'unknown.csv')
    
    # Dosya içeriğini oku
    try:
        # Eğer zaten BytesIO ise, içeriği al
        if isinstance(file, io.BytesIO):
            file_content = file.getvalue()
            file_bytes = io.BytesIO(file_content)
        else:
            # Streamlit file_uploader'dan gelen dosya
            file_content = file.read()
            file_bytes = io.BytesIO(file_content)
    except AttributeError:
        # Eğer read() metodu yoksa, direkt kullan
        file_bytes = file
    
    # Dosya tipine göre yükle
    if file_name.endswith('.csv'):
        # CSV dosyası için delimiter tespiti ve farklı encoding'ler dene
        delimiters = [',', ';', '\t', '|']
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        df = None
        last_error = None
        
        # Önce delimiter tespiti yap
        file_bytes.seek(0)
        try:
            first_line_bytes = file_bytes.readline()
            first_line = first_line_bytes.decode('utf-8', errors='ignore')
        except:
            # Eğer decode edilemezse, dosyayı string olarak oku
            file_bytes.seek(0)
            first_line = file_bytes.read(1000).decode('utf-8', errors='ignore').split('\n')[0]
        
        # En uygun delimiter'i bul
        detected_delimiter = ','
        max_cols = 0
        for delim in delimiters:
            cols = first_line.split(delim)
            if len(cols) > max_cols:
                max_cols = len(cols)
                detected_delimiter = delim
        
        # Farklı encoding'lerle dene
        for encoding in encodings:
            try:
                file_bytes.seek(0)  # Dosyayı başa al
                
                # ÖNCE index_col OLMADAN OKU (daha güvenli)
                # Bu şekilde "No columns to parse" hatasından kaçınırız
                df_temp = pd.read_csv(
                    file_bytes, 
                    sep=detected_delimiter, 
                    encoding=encoding, 
                    engine='python',
                    header=0  # İlk satırı header olarak kullan
                )
                
                # Dosya boş mu kontrol et
                if df_temp.empty:
                    raise ValueError("CSV dosyası boş! Lütfen geçerli bir veri dosyası yükleyin.")
                
                # En az 2 sütun olmalı
                if df_temp.shape[1] < 2:
                    raise ValueError(f"CSV dosyasında yeterli sütun yok! {df_temp.shape[1]} sütun bulundu, en az 2 sütun olmalı (1 kullanıcı ID + en az 1 ürün sütunu).")
                
                # İlk sütunu index yap (Matrix Format için)
                df = df_temp.set_index(df_temp.columns[0])
                
                # Başarılı oldu, döngüden çık
                break
                
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_error = e
                continue
            except ValueError as e:
                # ValueError'ları direkt yükselt (boş dosya, yetersiz sütun vb.)
                raise e
            except Exception as e:
                # Diğer hatalar için
                error_str = str(e)
                if "No columns to parse" in error_str:
                    # Delimiter yanlış olabilir, otomatik tespit dene
                    try:
                        file_bytes.seek(0)
                        # Delimiter=None ile otomatik tespit
                        df_temp = pd.read_csv(
                            file_bytes, 
                            sep=None, 
                            encoding=encoding, 
                            engine='python',
                            header=0
                        )
                        if df_temp.shape[1] >= 2:
                            df = df_temp.set_index(df_temp.columns[0])
                            break
                        else:
                            last_error = ValueError(f"CSV dosyasında yeterli sütun yok! {df_temp.shape[1]} sütun bulundu.")
                    except Exception as e2:
                        last_error = e2
                        continue
                else:
                    last_error = e
                    continue
        
        if df is None:
            error_msg = f"CSV dosyası okunamadı: {str(last_error) if last_error else 'Bilinmeyen hata'}"
            error_msg += "\n\n💡 İpuçları:"
            error_msg += "\n- CSV dosyasının ilk sütunu kullanıcı ID'leri olmalı"
            error_msg += "\n- En az 2 sütun olmalı (1 kullanıcı ID + en az 1 ürün sütunu)"
            error_msg += f"\n- Delimiter olarak şunlar deneniyor: {', '.join(delimiters)}"
            error_msg += "\n- Dosya formatını kontrol edin"
            raise ValueError(error_msg)
            
    elif file_name.endswith(('.xlsx', '.xls')):
        try:
            # Excel dosyası için de BytesIO kullan
            file_bytes.seek(0)
            df = pd.read_excel(file_bytes, index_col=0)
            # En az 1 sütun olmalı (index hariç)
            if df.shape[1] == 0:
                # index_col=0 olmadan dene
                file_bytes.seek(0)
                df_temp = pd.read_excel(file_bytes)
                if df_temp.shape[1] > 1:
                    df = df_temp.set_index(df_temp.columns[0])
                else:
                    raise ValueError("Excel dosyasında yeterli sütun yok! En az 2 sütun olmalı (1 kullanıcı ID + en az 1 ürün sütunu).")
        except Exception as e:
            error_msg = f"Excel dosyası okunamadı: {str(e)}"
            error_msg += "\n\n💡 İpuçları:"
            error_msg += "\n- Excel dosyasının ilk sütunu kullanıcı ID'leri olmalı"
            error_msg += "\n- En az 2 sütun olmalı (1 kullanıcı ID + en az 1 ürün sütunu)"
            raise ValueError(error_msg)
    else:
        raise ValueError("Desteklenen formatlar: CSV, Excel (.xlsx, .xls)")
    
    # Boş dosya kontrolü
    if df.empty:
        raise ValueError("Dosya boş! Lütfen geçerli bir veri dosyası yükleyin.")
    
    # DataFrame'i numpy array'e çevir
    rating_matrix = df.values
    
    # Shape kontrolü
    if rating_matrix.shape[0] == 0 or rating_matrix.shape[1] == 0:
        raise ValueError(f"Veri matrisi boş! Shape: {rating_matrix.shape}. En az 1 satır ve 1 sütun olmalı.")
    
    # NaN değerleri koru
    try:
        rating_matrix = rating_matrix.astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Rating değerleri sayısal değil! Lütfen tüm rating değerlerinin sayısal olduğundan emin olun. Hata: {str(e)}")
    
    # En az bir tane geçerli (NaN olmayan) değer olmalı
    valid_values = np.sum(~np.isnan(rating_matrix))
    if valid_values == 0:
        raise ValueError("Dosyada hiç geçerli rating değeri yok! Tüm değerler NaN. Lütfen en az bir tane sayısal rating değeri olduğundan emin olun.")
    
    return rating_matrix

