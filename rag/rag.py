import json
from sentence_transformers import SentenceTransformer, util
import faiss
import torch
import numpy as np
import argparse
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer


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
    # 如果数据量小于100,000，取实际长度
    sample_size = min(sample_size, embeddings.shape[0])
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
    print("->正在进行BM25编码...")
    keys = list(input_dict.keys())
    corpus = list(input_dict.values())
    
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', dtype=np.float32)
    X = vectorizer.fit_transform(corpus)
    
    N = X.shape[0]
    
    doc_lengths = np.array(X.sum(axis=1)).squeeze().astype(np.float32)
    avg_doc_len = doc_lengths.mean()
    
    row_indices = np.repeat(np.arange(N, dtype=np.int32), np.diff(X.indptr))
    lengths_for_data = doc_lengths[row_indices]
    
    data = X.data.astype(np.float32)

    denominator = data + k1 * (1.0 - b + b * (lengths_for_data / avg_doc_len))
    numerator = data * (k1 + 1.0)
    tf_bm25_data = numerator / denominator

    X_bm25_tf = csr_matrix((tf_bm25_data, X.indices, X.indptr), shape=X.shape)
    
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)

    idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    idf[idf < 0] = 0.0  # 已修复：将全角括号 '）' 修改为半角括号 ')'

    idf_diag = diags(idf)
    bm25_matrix = X_bm25_tf.dot(idf_diag)
    
    meta_data = {
        'keys': keys,
        'vocabulary': vectorizer.vocabulary_
    }
    
    return bm25_matrix, meta_data


def search_bm25(queries, bm25_matrix, meta_data, top_k=5):
    print(f"->正在进行BM25搜索top_{top_k}...")
    vocab = meta_data['vocabulary']
    keys = meta_data['keys']
    
    vectorizer = CountVectorizer(vocabulary=vocab, stop_words='english', dtype=np.float32)

    Q_matrix = vectorizer.transform(queries)

    scores_matrix = Q_matrix.dot(bm25_matrix.T)
    
    indices = []
    distances = []

    for i in range(len(queries)):
        row = scores_matrix.getrow(i)
        doc_indices = row.indices
        scores = row.data
        if len(scores) == 0:
            indices.append([])
            continue

        if len(scores) <= top_k:
            sorted_idx = np.argsort(-scores)
        else:
            top_k_pos = np.argpartition(scores, -top_k)[-top_k:]
            sorted_idx = top_k_pos[np.argsort(-scores[top_k_pos])]

        top_k_doc_indices = doc_indices[sorted_idx]
        top_k_scores = scores[sorted_idx]
        
        indices.append(top_k_doc_indices.tolist())
        distances.append(top_k_scores.tolist())

    prefixed_indices = [[f"evidence-{item}" for item in sublist] for sublist in indices]

    return prefixed_indices


def reciprocal_rank_fusion(list1, list2, k=60, top_k=16):
    """
    通过 RRF 算法融合两个检索结果列表。
    list1, list2 结构均为: [[doc1, doc2, ...], [doc1, doc2, ...], ...]
    """
    print(f"->正在进行 RRF 融合 (输出 top {top_k})...")
    fused_results = []
    for q_idx in range(len(list1)):
        scores = {}
        
        # 处理第一个列表
        for rank, doc_id in enumerate(list1[q_idx]):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
            
        # 处理第二个列表
        for rank, doc_id in enumerate(list2[q_idx]):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
            
        # 按 RRF 得分降序排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # 截取 top_k
        fused_results.append([doc_id for doc_id, score in sorted_docs[:top_k]])
        
    return fused_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 新增 rrf 模式说明
    parser.add_argument("--mode", type=str, help="输入运行模式，transformer | bm25 | rrf", default="bm25")
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

    elif args.mode == "bm25":
        bm25_matrix, meta_data = encode_bm25(evidence_dict)
        sentences = list(test_dict.values())
        prefixed_indices = search_bm25(sentences, bm25_matrix, meta_data, top_k=16)

    elif args.mode == "rrf":
        # 融合时，先各自召回更多的候选文档（例如60个）有助于提升交叉排序的准确率
        retrieve_pool_size = 60
        
        # 1. 运行 Transformer 召回
        print("\n--- 第一步: 运行 Transformer 检索 ---")
        dense_embeddings = embed(evidence_dict)
        index = write_index(dense_embeddings)
        test_embeddings = embed(test_dict)
        transformer_indices = search(index, test_embeddings, top_k=retrieve_pool_size)
        
        # 2. 运行 BM25 召回
        print("\n--- 第二步: 运行 BM25 检索 ---")
        bm25_matrix, meta_data = encode_bm25(evidence_dict)
        sentences = list(test_dict.values())
        bm25_indices = search_bm25(sentences, bm25_matrix, meta_data, top_k=retrieve_pool_size)
        
        # 3. 运行 RRF 融合
        print("\n--- 第三步: 运行倒排融合 (RRF) ---")
        prefixed_indices = reciprocal_rank_fusion(
            transformer_indices, 
            bm25_indices, 
            k=60, 
            top_k=16
        )

    if len(prefixed_indices) != len(ground_truth):
        print(f"警告：预测结果的数量 ({len(prefixed_indices)}) 与真实样本的数量 ({len(ground_truth)}) 不一致！")

    total_tp = 0 
    total_fp = 0
    total_fn = 0

    predictions_dict = {}

    for (claim_id, claim_data), predicted_evs in zip(ground_truth.items(), prefixed_indices):

        true_set = set(claim_data.get("evidences", []))
        pred_set = set(predicted_evs)

        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        predictions_dict[claim_id] = {
            "claim_text": claim_data["claim_text"],
            "claim_label": "UNLABELED", 
            "evidences": list(pred_set)
        }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("========== 评估结果 ==========")
    print(f"准确率 (Precision): {precision:.4f} ({precision * 100:.2f}%)")
    print(f"召回率 (Recall):    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score:           {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print("==============================")

    output_filename = 'predicted-claims.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(predictions_dict, f, indent=4, ensure_ascii=False)

    print(f"\n预测结果已成功保存至: {output_filename}")