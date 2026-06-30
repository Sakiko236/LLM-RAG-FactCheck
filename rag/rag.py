import json
from sentence_transformers import SentenceTransformer, util
from itertools import islice
import faiss
import torch
import numpy as np
import argparse

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"解析文件时发生错误: {e}")
        return None


def write_faiss(embeddings):
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

    faiss.write_index(index, "tmp/embeddings.index")


def embed(my_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2', device=device)
    sentences = list(my_dict.values())
    embeddings = model.encode(sentences, batch_size=512, show_progress_bar=True)

    return embeddings


def search(query_embeddings):
    index = faiss.read_index("tmp/embeddings.index")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="输入运行模式，index:构建index | test:生成召回结果", default="test")
    args = parser.parse_args()

    if args.mode == "index":
        evidence_path = 'data/evidence/evidence.json'
        evidence_dict = load_json_file(evidence_path)

        if evidence_dict:
            print(f"成功加载，字典包含 {len(evidence_dict)} 个一级键。")

        for k, v in islice(evidence_dict.items(), 5):
            print(f"Key: {k}, Value: {v}")

        embeddings = embed(evidence_dict)

        if embeddings is not None:
            print(f"成功嵌入，包含 {len(embeddings)} 个向量。")

        write_faiss(embeddings)

        print("向量已保存到本地。")

    if args.mode == "test":
        test_dict = load_test_claims("data/test-claims.json")

        if test_dict:
            print(f"成功加载，字典包含 {len(test_dict)} 个一级键。")

        for k, v in islice(test_dict.items(), 5):
            print(f"Key: {k}, Value: {v}")

        embeddings = embed(test_dict)

        indices = search(embeddings[4:5])[0]

        evidence_path = 'data/evidence/evidence.json'
        evidence_dict = load_json_file(evidence_path)

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

        for true_evidence in true_evidences:
            evidence_text = evidence_dict.get(true_evidence, "未在本地字典中找到该证据")
            print(f"Key: {true_evidence}, Value: {evidence_text}")

        # 实验表明两个问题：
        # 1.embedding对词语分配的注意力有误，比如错误分配过多权重给"Greenland"，导致搜索结果里缺少对"iceburger"的匹配
        # 2.embedding几乎无法匹配数字







