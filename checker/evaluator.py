import os
import sys
import json
import re
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claim_verifier import ClaimVerifier
from rag.rag_pipeline import RAGPipeline

class Evaluator:
    def __init__(self, base_model, lora_path, groundtruth_path, batch_size=2, save_raw_output=False, verbose=False):
        self.base_model = base_model
        self.lora_path = lora_path
        self.groundtruth_path = groundtruth_path
        self.batch_size = batch_size
        self.save_raw_output = save_raw_output
        self.verbose = verbose
        self.groundtruth = self._load_groundtruth()

    def _load_groundtruth(self):
        try:
            with open(self.groundtruth_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("Error loading groundtruth json file:", self.groundtruth_path)
            print("Error details:", e)
            raise SystemExit

    def _extract_label(self, text):
        text_upper = text.upper()
        matches = re.finditer(r'\b(SUPPORTS|REFUTES|NOT_ENOUGH_INFO|DISPUTED)\b', text_upper)
        found_labels = [m.group(0) for m in matches]
        
        if found_labels:
            return found_labels[-1]
        else:
            return "NOT_ENOUGH_INFO"

    def generate_predictions(self):
        print("Initializing ClaimVerifier (Model & LoRA)...")
        verifier = ClaimVerifier(base_model_id=self.base_model, lora_path=self.lora_path)
        
        print("Initializing RAG Pipeline (Retriever & Re-ranker)...")
        rag = RAGPipeline()

        predictions = {}
        raw_outputs = {}
        
        items = list(self.groundtruth.items())
        total_claims = len(items)
        
        print(f"\nStarting Prediction on {total_claims} claims (Batch Size: {self.batch_size})...")
        
        for i in tqdm(range(0, total_claims, self.batch_size), desc="Predicting Batches", colour="green"):
            batch_items = items[i:i + self.batch_size]
            
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
                predicted_label = self._extract_label(model_output)
                
                predictions[claim_id] = {
                    "claim_text": claim_text,
                    "claim_label": predicted_label,
                    "evidences": ev_ids
                }
                
                raw_outputs[claim_id] = {
                    "claim_text": claim_text,
                    "raw_output": model_output,
                    "predicted_label": predicted_label
                }

        if self.save_raw_output:
            raw_output_filepath = "results/raw_output_dev.json"
            os.makedirs(os.path.dirname(raw_output_filepath), exist_ok=True)
            with open(raw_output_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_outputs, f, indent=4, ensure_ascii=False)
            print(f"Saved raw outputs to {raw_output_filepath}")
            
        return predictions

    def evaluate(self, predictions):
        print("\nEvaluating predictions...")
        try:
            f_scores, acc = [], []

            for claim_id, claim in sorted(self.groundtruth.items()):
                if claim_id in predictions and \
                    "claim_label" in predictions[claim_id] and \
                    "evidences" in predictions[claim_id]:

                    #check claim level label
                    instance_correct = 0.0
                    if predictions[claim_id]["claim_label"] == claim["claim_label"]:
                        instance_correct = 1.0
                    
                    #check retrieved evidences
                    evidence_correct = 0
                    evidence_recall = 0.0
                    evidence_precision = 0.0
                    evidence_fscore = 0.0
                    if type(predictions[claim_id]["evidences"]) == list and (len(predictions[claim_id]["evidences"]) > 0):
                        top_six_ev = set(predictions[claim_id]["evidences"])
                        for gr_ev in claim["evidences"]:
                            if gr_ev in top_six_ev:
                                evidence_correct += 1
                        if evidence_correct > 0:
                            evidence_recall = float(evidence_correct) / len(claim["evidences"])
                            evidence_precision = \
                                float(evidence_correct) / len(predictions[claim_id]["evidences"])
                            evidence_fscore = (2*evidence_precision*evidence_recall)/(evidence_precision+evidence_recall)

                    if self.verbose:
                        print("groundtruth =", claim)
                        print("predictions =", predictions[claim_id])
                        print("instance accuracy =", instance_correct)
                        print("evidence recall =", evidence_recall)
                        print("evidence precision =", evidence_precision)
                        print("evidence fscore =", evidence_fscore, "\n\n")

                    #add the metric results
                    acc.append(instance_correct)
                    f_scores.append(evidence_fscore)

            #compute aggregate performance
            mean_f = np.mean(f_scores if len(f_scores) > 0 else [0.0])
            mean_acc = np.mean(acc if len(acc) > 0 else [0.0])
            if mean_f == 0.0 and mean_acc == 0.0:
                hmean = 0.0
            else:
                hmean = (2*mean_f*mean_acc)/(mean_f+mean_acc)

            print("Evidence Retrieval F-score (F)    =", mean_f)
            print("Claim Classification Accuracy (A) =", mean_acc)
            print("Harmonic Mean of F and A          =", hmean)
                    
        except Exception as error:
            print("Error:", error)
            raise SystemExit

    def run(self):
        predictions = self.generate_predictions()
        self.evaluate(predictions)

if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LORA_PATH = "../../model/qwen-cot-lora-final"
    GROUNDTRUTH_PATH = "../../data/dev-claims.json"
    
    BATCH_SIZE = 2
    SAVE_RAW_OUTPUT = True
    VERBOSE = False
    
    evaluator = Evaluator(
        base_model=BASE_MODEL,
        lora_path=None,
        groundtruth_path=GROUNDTRUTH_PATH,
        batch_size=BATCH_SIZE,
        save_raw_output=SAVE_RAW_OUTPUT,
        verbose=VERBOSE
    )
    
    evaluator.run()
