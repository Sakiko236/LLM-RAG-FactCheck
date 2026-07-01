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

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    sentences = list(my_dict.values())
    embeddings = model.encode(sentences, batch_size=512, show_progress_bar=True)

    return embeddings


def search(index, query_embeddings, top_k=5):
    index.nprobe = 10
    k = top_k

    distances, indices = index.search(query_embeddings, k)

    prefixed_indices = [[f"evidence-{item}" for item in sublist] for sublist in indices]

    return prefixed_indices


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

    print(f"BM25矩阵形状 (Shape): {bm25_matrix.shape}")
    
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
            indices.append([])
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

    prefixed_indices = [[f"evidence-{item}" for item in sublist] for sublist in indices]

    return prefixed_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="输入运行模式，transformer | bm25", default="bm25")
    args = parser.parse_args()

    evidence_dict = load_json_file('data/evidence/evidence.json')

    with open('data/test-claims.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    test_dict = load_test_claims("data/test-claims.json")

    if test_dict:
        print(f"测试集成功加载，字典包含 {len(test_dict)} 个一级键。")

    if args.mode == "transformer":
        embeddings = embed(evidence_dict)
        if embeddings is not None:
            print(f"证据集成功嵌入，包含 {len(embeddings)} 个向量。")

        index = write_index(embeddings)
        if index is not None:
            print("索引已创建。")

        embeddings = embed(test_dict)
        prefixed_indices = search(index, embeddings, top_k=16)

        # 实验表明两个问题：
        # 1.embedding对词语分配的注意力有误，比如错误分配过多权重给"Greenland"，导致搜索结果里缺少对"iceburger"的匹配
        # 2.embedding几乎无法匹配数字

    if args.mode == "bm25":
        bm25_matrix, meta_data = encode_bm25(evidence_dict)
        sentences = list(test_dict.values())
        prefixed_indices = search_bm25(sentences, bm25_matrix, meta_data, top_k=16)

    # 安全检查：确保预测数量与真实标签数量一致
    if len(prefixed_indices) != len(ground_truth):
        print(f"警告：预测结果的数量 ({len(prefixed_indices)}) 与真实样本的数量 ({len(ground_truth)}) 不一致！")

    # 1. 初始化统计变量
    total_tp = 0  # 预测正确：在预测中，且在真实中
    total_fp = 0  # 预测错误：在预测中，但不在真实中
    total_fn = 0  # 漏报：不在预测中，但在真实中

    predictions_dict = {}

    # 2. 遍历比对并构建新的预测字典
    # 使用 zip 将真实数据和预测列表按顺序打包
    for (claim_id, claim_data), predicted_evs in zip(ground_truth.items(), prefixed_indices):
    
        # 转换为集合 (Set) 以便进行高效的交集和差集运算
        true_set = set(claim_data.get("evidences", []))
        pred_set = set(predicted_evs)
    
        # 计算当前 claim 的 TP, FP, FN
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
    
        # 累加到全局统计中
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
        # 构建与原始格式相同的预测字典
        predictions_dict[claim_id] = {
            "claim_text": claim_data["claim_text"],
            "claim_label": "UNLABELED", 
            "evidences": list(pred_set) # 替换为预测出的 evidences
        }

    # 3. 计算最终的 Precision, Recall 和 F1 Score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 4. 打印结果
    print("========== 评估结果 ==========")
    print(f"准确率 (Precision): {precision:.4f} ({precision * 100:.2f}%)")
    print(f"召回率 (Recall):    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score:           {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print("==============================")

    # 5. 保存为新的 JSON 文件
    output_filename = 'predicted-claims.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(predictions_dict, f, indent=4, ensure_ascii=False)

    print(f"\n预测结果已成功保存至: {output_filename}")




