import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_pipeline import RAGPipeline
from fine_tuning.retrieve_evidence import get_evidence_text

class ClaimVerifier:
    def __init__(self, base_model_id="Qwen/Qwen3.5-2B", lora_path=None):
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Configuring 4-bit Quantization...")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )

        print(f"Loading Base Model: {base_model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=compute_dtype,
        )

        if lora_path and os.path.exists(lora_path):
            print(f"Applying LoRA weights from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("LoRA loaded successfully!")
        else:
            print("No valid LoRA path provided. Using Base Model for inference.")
            
        self.rag_pipeline = None

    def fetch_evidence(self, claim_text, evidence_ids=None):
        evidence_texts = []
        
        if evidence_ids and len(evidence_ids) > 0:
            for eid in evidence_ids:
                text = get_evidence_text(eid)
                if text:
                    evidence_texts.append(f"[{eid}] {text}")
        else:
            print("No evidence IDs provided. Triggering RAG Pipeline...")
            if self.rag_pipeline is None:
                self.rag_pipeline = RAGPipeline()
            
            top_evidence = self.rag_pipeline.process_claim(claim_text, top_k_retrieve=20)
            for ev in top_evidence:
                evidence_texts.append(f"[{ev['id']}] {ev['text']}")
                
        return "\n".join(evidence_texts)

    def predict(self, claims_texts, batch_evidence_ids, few_shot=False):
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text_prompts = []
        
        for claim_text, evidence_ids in zip(claims_texts, batch_evidence_ids):
            combined_evidence = self.fetch_evidence(claim_text, evidence_ids)
            
            messages = [
                {"role": "system", "content": """You are a climate science fact-checker. Your task is to explain the logical connection between the Evidence and the Claim to justify the Label. Use the format: "Let's analyze step by step: [Reasoning] Therefore, the conclusion is: [Label]." conclude your response with one of the following four labels: 
                [SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED]."""},
                {"role": "user", "content": f"Evidence:\n{combined_evidence}\n\nClaim:\n{claim_text}"}
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            text_prompts.append(prompt)

        inputs = self.tokenizer(
            text_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        input_length = inputs.input_ids.shape[1]
        clean_responses = []
        
        for output in outputs:
            generated_tokens = output[input_length:]
            full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            clean_response = full_response
            for word in ["Evidence:", "Claim:", "Label:"]:
                if word in clean_response:
                    clean_response = clean_response.split(word)[0]
            
            clean_responses.append(clean_response.strip())

        return clean_responses