import os
import sys
import json
import re
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claim_verifier import ClaimVerifier
from rag.rag_pipeline import RAGPipeline

def extract_label(text):
    text_upper = text.upper()
    matches = re.finditer(r'\b(SUPPORTS|REFUTES|NOT_ENOUGH_INFO|DISPUTED)\b', text_upper)
    found_labels = [m.group(0) for m in matches]
    
    if found_labels:
        return found_labels[-1]
    else:
        return "NOT_ENOUGH_INFO" 

def generate_test_predictions_batched(base_model_id, lora_path, test_file_path, output_filepath, batch_size=3):
    print("Initializing ClaimVerifier (Model & LoRA)...")
    verifier = ClaimVerifier(base_model_id=base_model_id, lora_path=lora_path)
    
    print("Initializing RAG Pipeline (Retriever & Re-ranker)...")
    rag = RAGPipeline()
    
    print(f"Loading Unlabelled Test Set from {test_file_path}...")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    output_data = {}
    raw_output_data = {}
    
    items = list(test_data.items())
    total_claims = len(items)
    
    print(f"\nStarting Prediction on {total_claims} claims (Batch Size: {batch_size})...")
    
    for i in tqdm(range(0, total_claims, batch_size), desc="Predicting Batches", colour="green"):
        batch_items = items[i:i + batch_size]
        
        batch_claim_ids = []
        batch_claim_texts = []
        batch_evidence_ids = []
        
        for claim_id, claim_info in batch_items:
            claim_text = claim_info.get('claim_text', '')
            
            top_evidence = rag.process_claim(claim_text)
            evidence_ids = [ev['id'] for ev in top_evidence]
            
            batch_claim_ids.append(claim_id)
            batch_claim_texts.append(claim_text)
            batch_evidence_ids.append(evidence_ids)
            
        model_outputs = verifier.predict(batch_claim_texts, batch_evidence_ids)

        for claim_id, claim_text, ev_ids, model_output in zip(batch_claim_ids, batch_claim_texts, batch_evidence_ids, model_outputs):
            predicted_label = extract_label(model_output)
            
            output_data[claim_id] = {
                "claim_text": claim_text,
                "claim_label": predicted_label,
                "evidences": ev_ids
            }
            
            raw_output_data[claim_id] = {
                "claim_text": claim_text,
                "raw_output": model_output,
                "predicted_label": predicted_label
            }
            
    print(f"\nSaving predictions to {output_filepath}...")
    
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    raw_output_filepath = os.path.join(output_dir, "raw_output.json")
    print(f"Saving raw outputs to {raw_output_filepath}...")
    with open(raw_output_filepath, 'w', encoding='utf-8') as f:
        json.dump(raw_output_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ All Done! Results saved to {output_filepath} and {raw_output_filepath}.")

if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LORA_PATH = "../../model/qwen-cot-lora-final"
    TEST_FILE = "../../data/test-claims-unlabelled.json"
    OUTPUT_FILE = "results/test-output.json" 
    
    generate_test_predictions_batched(
        base_model_id=BASE_MODEL,
        lora_path=None,
        test_file_path=TEST_FILE,
        output_filepath=OUTPUT_FILE,
        batch_size=2
    )