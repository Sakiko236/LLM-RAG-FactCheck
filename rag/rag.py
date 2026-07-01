import json
from sentence_transformers import SentenceTransformer, util
from itertools import islice
import faiss
import torch
import numpy as np
import argparse
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer
import re

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"解析文件时发生错误: {e}")
        return None


def write_index(embeddings):
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype('float32')

    d = embeddings.shape[1]
    nlist = 1500
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    sample_size = int(10e4)
    indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    sampled_data = embeddings[indices]
    index.train(sampled_data)

    index.add(embeddings)

    return index


def embed(my_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2', device=device)
    sentences = list(my_dict.values())
    embeddings = model.encode(sentences, batch_size=512, show_progress_bar=True)

    return embeddings


def search(index, query_embeddings):
    index.nprobe = 10
    k = 5

    distances, indices = index.search(query_embeddings, k)
    print("最近邻索引 ID:", indices)
    print("相似度分值:", distances)

    return indices


def load_test_claims(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted_data = {
        claim_id: item_content["claim_text"] 
        for claim_id, item_content in data.items()
    }

    return extracted_data


def encode_bm25(input_dict, max_features=100000, k1=1.5, b=0.75):
    
    # 1. 拆分字典
    keys = list(input_dict.keys())
    corpus = list(input_dict.values())
    
    # 2. 词频统计 (获取标准的稀疏矩阵 CountMatrix)
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', dtype=np.float32)
    X = vectorizer.fit_transform(corpus)  # 得到的是一个 CSR (Compressed Sparse Row) 矩阵
    
    N = X.shape[0]  # 文档总数 (120万)
    
    # 3. 计算文档长度
    # X.sum(axis=1) 得到每个文档的单词总数
    doc_lengths = np.array(X.sum(axis=1)).squeeze().astype(np.float32)
    avg_doc_len = doc_lengths.mean()
    
    # 4. 高效计算 BM25 的 TF 项
    # 不拆除稀疏矩阵，直接利用 CSR 的 indptr 数组，在 C 层面快恢复出每个非零元素对应的行索引（文档ID）
    row_indices = np.repeat(np.arange(N, dtype=np.int32), np.diff(X.indptr))
    lengths_for_data = doc_lengths[row_indices]
    
    # 获取原始词频数据并转换成 float32
    data = X.data.astype(np.float32)
    
    # BM25 TF 项公式: (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)))
    denominator = data + k1 * (1.0 - b + b * (lengths_for_data / avg_doc_len))
    numerator = data * (k1 + 1.0)
    tf_bm25_data = numerator / denominator
    
    # 直接复用原矩阵的列索引(indices)和行偏移指针(indptr)，零拷贝构建新矩阵
    X_bm25_tf = csr_matrix((tf_bm25_data, X.indices, X.indptr), shape=X.shape)
    
    # 5. 计算 BM25 的 IDF 项
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
    
    # 标准 BM25 IDF 公式: ln(1 + (N - df + 0.5) / (df + 0.5))
    idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    idf[idf < 0] = 0.0  # 裁剪负权重（防止高频词出现负分）
    
    # 6. TF 与 IDF 矩阵相乘，并保存结果
    idf_diag = diags(idf)
    bm25_matrix = X_bm25_tf.dot(idf_diag)
    
    
    # 保存元数据（Key 的顺序以及词表映射，方便后续检索对应）
    meta_data = {
        'keys': keys,
        'vocabulary': vectorizer.vocabulary_
    }

    print(f"矩阵形状 (Shape): {bm25_matrix.shape}")
    
    return bm25_matrix, meta_data


def search_bm25(queries, bm25_matrix, meta_data, top_k=5):
    vocab = meta_data['vocabulary']
    keys = meta_data['keys']
    
    # 文本预处理必须与之前保持一致（如英文停用词）
    vectorizer = CountVectorizer(vocabulary=vocab, stop_words='english', dtype=np.float32)
    
    # 2. 将查询转换为稀疏矩阵
    Q_matrix = vectorizer.transform(queries)
    
    # 3. 矩阵乘法计算得分
    scores_matrix = Q_matrix.dot(bm25_matrix.T)
    
    indices = []
    distances = []
    
    # 4. 遍历每个查询的结果，提取 Top-K
    for i in range(len(queries)):
        # 获取第 i 个查询的得分行 (依然是 CSR 稀疏格式)
        row = scores_matrix.getrow(i)
        
        # 提取非零得分及其对应的列索引（即文档的 index）
        doc_indices = row.indices
        scores = row.data
        
        # 如果查询词全都不在词表中，或者没有任何文档包含查询词
        if len(scores) == 0:
            results.append({"query": queries[i], "hits": []})
            continue
            
        # 5. 高效寻找 Top-K
        if len(scores) <= top_k:
            # 命中数量少于 top_k，直接全排序
            sorted_idx = np.argsort(-scores)
        else:
            # 使用 argpartition 找到最大的 top_k 个元素的相对位置 (时间复杂度 O(N))
            top_k_pos = np.argpartition(scores, -top_k)[-top_k:]
            # 对这 k 个元素在局部进行降序排序
            sorted_idx = top_k_pos[np.argsort(-scores[top_k_pos])]
            
        # 映射回 120 万文档的全局索引，并提取得分
        top_k_doc_indices = doc_indices[sorted_idx]
        top_k_scores = scores[sorted_idx]
        
        indices.append(top_k_doc_indices.tolist())
        distances.append(top_k_scores.tolist())
        
    print("最近邻索引 ID:", indices)
    print("相似度分值:", distances)

    return indices


def search_number(evidence_dict, queries):
    num_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
    
    ev_nums_list = []
    ev_doc_ids_list = []
    
    # 2. 提取所有 evidence 中的数字并打平 (Flatten)
    for doc_id, text in evidence_dict.items():
        # 剔除常见的千分位逗号，防止 1,000 被识别为 1 和 000
        clean_text = text.replace(",", "")
        nums = [float(x) for x in num_pattern.findall(clean_text)]
        
        if nums:
            ev_nums_list.extend(nums)
            # 记录这些数字对应的 doc_id
            ev_doc_ids_list.extend([doc_id] * len(nums))
            
    # 转为 NumPy 数组，利用 C 级别底层加速向量化比较
    ev_nums_arr = np.array(ev_nums_list, dtype=np.float32)
    ev_doc_ids_arr = np.array(ev_doc_ids_list)
    
    indices = []
    
    # 3. 批量处理 Queries
    for q_text in queries:
        clean_q_text = q_text.replace(",", "")
        q_nums = [float(x) for x in num_pattern.findall(clean_q_text)]
        
        # 如果 Query 中没有数字，直接返回空列表
        if not q_nums:
            indices.append([])
            continue
            
        hit_doc_ids = set()
        
        # 遍历 Query 中的每一个数字，计算区间并使用 Numpy 查找
        for q_val in q_nums:
            # 使用绝对值计算偏差，确保负数（如 -10 的 +-20% 是 [-12, -8]）逻辑正确
            delta = 0.2 * abs(q_val)
            lower_bound = q_val - delta
            upper_bound = q_val + delta
            
            # 核心：向量化寻找落在区间内的数字索引
            mask = (ev_nums_arr >= lower_bound) & (ev_nums_arr <= upper_bound)
            
            # 提取对应的 doc_ids 并加入集合去重
            matched_ids = ev_doc_ids_arr[mask]
            hit_doc_ids.update(matched_ids)
            
        # 将当前 query 命中的去重 doc_ids 存入结果
        indices.append(list(hit_doc_ids))
        
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="输入运行模式，transformer | bm25 | number", default="bm25")
    args = parser.parse_args()

    evidence_dict = load_json_file('data/evidence/evidence.json')

    if evidence_dict:
        print(f"成功加载，字典包含 {len(evidence_dict)} 个一级键。")

    if args.mode == "transformer":
        embeddings = embed(evidence_dict)

        if embeddings is not None:
            print(f"成功嵌入，包含 {len(embeddings)} 个向量。")

        index = write_index(embeddings)

        if index is not None:
            print("索引已创建。")

        test_dict = load_test_claims("data/test-claims.json")

        if test_dict:
            print(f"成功加载，字典包含 {len(test_dict)} 个一级键。")

        embeddings = embed(test_dict)

        indices = search(index, embeddings[4:5])[0]

        for indice in indices:
            key_name = f"evidence-{indice}"
            evidence_text = evidence_dict.get(key_name, "未在本地字典中找到该证据")
            print(f"Key: {key_name}, Value: {evidence_text}")

        true_evidences = [
            "evidence-52981",
            "evidence-264761",
            "evidence-947243",
            "evidence-424102"
        ]

        print("True evidences:")

        for true_evidence in true_evidences:
            evidence_text = evidence_dict.get(true_evidence, "未在本地字典中找到该证据")
            print(f"Key: {true_evidence}, Value: {evidence_text}")

        # 实验表明两个问题：
        # 1.embedding对词语分配的注意力有误，比如错误分配过多权重给"Greenland"，导致搜索结果里缺少对"iceburger"的匹配
        # 2.embedding几乎无法匹配数字

    if args.mode == "bm25":
        bm25_matrix, meta_data = encode_bm25(evidence_dict)

        queries = [
            "Greenland has only lost a tiny fraction of its ice mass"
        ]

        # 执行检索
        indices = search_bm25(queries, bm25_matrix, meta_data, top_k=5)[0]

        # 打印结果
        for indice in indices:
            key_name = f"evidence-{indice}"
            evidence_text = evidence_dict.get(key_name, "未在本地字典中找到该证据")
            print(f"Key: {key_name}, Value: {evidence_text}")

    if args.mode == "number":
        queries = [
            "Volcanoes emit around billion tonnes of CO per year."
        ]

        indices = search_number(evidence_dict, queries)[0]

        if not indices:
            print("无数据。")
        else:
            for indice in indices:
                evidence_text = evidence_dict.get(indice, "未在本地字典中找到该证据")
                print(f"Key: {indice}, Value: {evidence_text}")




