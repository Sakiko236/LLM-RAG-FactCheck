import json

evidence_path = 'data/evidence/evidence.json'

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"解析文件时发生错误: {e}")
        return None

evidence_dict = load_json_file(evidence_path)

if evidence_dict:
    print(f"成功加载，字典包含 {len(evidence_dict)} 个一级键。")