import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import gc
import os

class SparseBM25:
    """
    Кастомная реализация алгоритма Okapi BM25 поверх разреженных матриц (scipy.sparse).
    Оптимизирована для работы с большими корпусами текстов без переполнения RAM.
    Штрафует за чрезмерную длину документа (защита от SEO-спама в описаниях).
    """
    def __init__(self, k1=1.5, b=0.75, max_features=150000):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=max_features)
        
    def fit_transform(self, corpus):
        X = self.vectorizer.fit_transform(corpus)
        N = X.shape[0]
        
        # Подсчет Document Frequency (DF) и Inverse Document Frequency (IDF)
        df = np.bincount(X.indices, minlength=X.shape[1])
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
        
        doc_lengths = X.sum(axis=1).A1
        avgdl = doc_lengths.mean()
        
        # Расчет Term Frequency (TF) с учетом длины документа
        X_bm25 = X.copy().astype(np.float32)
        row_indices = np.repeat(np.arange(N), np.diff(X.indptr))
        X_bm25.data = (X.data * (self.k1 + 1)) / (X.data + self.k1 * (1 - self.b + self.b * (doc_lengths[row_indices] / avgdl)))
        
        return X_bm25.dot(sp.diags(self.idf))
        
    def transform(self, queries):
        Q = self.vectorizer.transform(queries)
        Q.data = np.ones_like(Q.data) 
        return Q

def generate_reproducible_submission():
    print("1. Загрузка данных")
    data_dir = 'dataset'
    queries_df = pd.read_parquet(os.path.join(data_dir, 'benchmark_queries.parquet'))
    items_df = pd.read_parquet(os.path.join(data_dir, 'benchmark_items.parquet'))
    train_full = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))

    # Стриппинг ID для защиты от смещения форматов (string vs float) при джойнах
    queries_df['query_id'] = queries_df['query_id'].astype(str).str.strip()
    items_df['item_id'] = items_df['item_id'].astype(str).str.strip()
    train_full['item_id'] = train_full['item_id'].astype(str).str.strip()
    
    # Исключаем "мертвые" клики на объявления, которых нет в активном пуле
    valid_items = set(items_df['item_id'])
    train_full = train_full[train_full['item_id'].isin(valid_items)]

    items_df['item_location_id'] = items_df['item_location_id'].fillna(-1).astype(int)
    items_df['item_category_id'] = items_df['item_category_id'].fillna(-1).astype(int)
    
    for df in [train_full, queries_df]:
        df['search_location_id'] = df['search_location_id'].fillna(-1).astype(int)
        df['search_category'] = df.get('search_category', pd.Series(-1, index=df.index)).fillna(-1).astype(int)
        df['search_query_clean'] = df['search_query'].fillna('').astype(str).str.lower().str.strip()

    print("2. Построение исторических двудольных графов")
    # full_geo_history: строгий граф совпадений (Запрос + Локация) -> Успешные объявления
    full_geo_history = train_full.groupby(['search_query_clean', 'search_location_id'])['item_id'].apply(set).to_dict()
    # full_global_history: общий граф (Запрос) -> Успешные объявления (используется для удаленных услуг)
    full_global_history = train_full.groupby('search_query_clean')['item_id'].apply(set).to_dict()
    
    item_ids_array = items_df['item_id'].values
    # Логарифмическая популярность объявления используется как tie-breaker при равном скоре текстов
    full_pop = np.log1p(np.array([train_full['item_id'].value_counts().to_dict().get(id_, 0) for id_ in item_ids_array])) * 0.05
    
    del train_full
    gc.collect()

    print("3. Препроцессинг текстовых признаков")
    def clean_text(df, cols):
        return df[cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower().str.strip()
    
    # Утраиваем заголовок при склейке.
    items_df['item_text_word'] = (
        items_df['item_title_raw'].fillna('').astype(str).str.lower() + " " + 
        items_df['item_title_raw'].fillna('').astype(str).str.lower() + " " + 
        items_df['item_title_raw'].fillna('').astype(str).str.lower() + " " + 
        clean_text(items_df, ['item_infm_params_text', 'item_description_raw'])
    )
    queries_df['query_text'] = clean_text(queries_df, ['search_query', 'search_infm_params_text'])

    print("4. Векторизация (BM25)...")
    bm25 = SparseBM25(max_features=150000)
    i_vec_word_T = bm25.fit_transform(items_df['item_text_word']).T
    q_vec_word_sub = bm25.transform(queries_df['query_text'])
    
    # Освобождаем память перед тяжелым перемножением матриц
    items_df.drop(columns=['item_text_word'], inplace=True)
    queries_df.drop(columns=['query_text'], inplace=True)
    del bm25
    gc.collect()

    print("5. Генерация кандидатов (Инференс)")
    i_locs = items_df['item_location_id'].values[None, :]
    i_cats = items_df['item_category_id'].values[None, :]
    item_id_to_idx = {id_: idx for idx, id_ in enumerate(item_ids_array)}
    i_locs_flat = items_df['item_location_id'].values
    
    sub_locs_all = queries_df['search_location_id'].values[:, None]
    sub_cats_all = queries_df['search_category'].values[:, None]
    
    all_top_indices = []
    batch_size = 1000 
    
    # Батчевая обработка запросов защищает от Out of Memory ошибок
    for start_idx in tqdm(range(0, q_vec_word_sub.shape[0], batch_size), desc="Инференс"):
        end_idx = min(start_idx + batch_size, q_vec_word_sub.shape[0])
        batch_queries = queries_df.iloc[start_idx:end_idx]
        
        # Получаем базовый текстовый скор
        sim_matrix = q_vec_word_sub[start_idx:end_idx].dot(i_vec_word_T).toarray()
        
        # Применяем мета-мультипликаторы (услуги обладают высокой локальностью)
        s_locs = sub_locs_all[start_idx:end_idx]
        sim_matrix[(s_locs == i_locs) & (s_locs != -1)] *= 5.0 
        
        s_cats = sub_cats_all[start_idx:end_idx]
        sim_matrix[(s_cats == i_cats) & (s_cats != -1)] *= 2.0
        
        # Инъекция исторических графовых эвристик (Behavioral Signals)
        for batch_i, (_, row) in enumerate(batch_queries.iterrows()):
            q_text = row['search_query_clean']
            q_loc = row['search_location_id']
            
            if (q_text, q_loc) in full_geo_history and q_loc != -1:
                # Идеальный кандидат: совпал и текст прошлого запроса, и регион
                for item in full_geo_history[(q_text, q_loc)]:
                    if item in item_id_to_idx: 
                        sim_matrix[batch_i, item_id_to_idx[item]] += 100.0
            elif q_text in full_global_history:
                # Soft Fallback логика для обработки оффлайн- и онлайн-услуг
                for item in full_global_history[q_text]:
                    if item in item_id_to_idx:
                        idx = item_id_to_idx[item]
                        if i_locs_flat[idx] == q_loc and q_loc != -1:
                            # Услуга в том же регионе, просто поиск был без привязки к гео
                            sim_matrix[batch_i, idx] += 50.0 
                        else:
                            # Удаленная услуга. Даем мягкий буст, который сработает только при высокой текстовой релевантности.
                            sim_matrix[batch_i, idx] += 15.0 
                            
        # Добавляем tie-breaker (популярность) и находим Топ-50
        sim_matrix += full_pop[None, :]
        current_k = min(50, sim_matrix.shape[1])
        
        # argpartition работает за O(N), что значительно быстрее полной сортировки
        batch_top_indices = np.argpartition(sim_matrix, -current_k, axis=1)[:, -current_k:]
        batch_top_scores = np.take_along_axis(sim_matrix, batch_top_indices, axis=1)
        
        # Сортируем только извлеченные Топ-50 по убыванию скора
        sorted_indices = np.take_along_axis(batch_top_indices, np.argsort(-batch_top_scores, axis=1), axis=1)
        
        all_top_indices.append(sorted_indices)
        
        del sim_matrix
        gc.collect()
    
    print("6. Экспорт файла answer.csv")
    predictions = [' '.join(item_ids_array[row]) for row in np.vstack(all_top_indices)]
    answer_df = pd.DataFrame({'query_id': queries_df['query_id'], 'answer': predictions})
    answer_df.to_csv('answer.csv', index=False)
    print("7. Файл answer.csv успешно сгенерирован.")

if __name__ == '__main__':
    generate_reproducible_submission()
