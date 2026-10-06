import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
class LocalLLM:
    def __init__(self,model_name):
        print(f"[LLM] Loading local model: {model_name}")
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32,device_map="cpu")
        self.model.eval(); print("[LLM] Model loaded on CPU")
    def answer(self,question,contexts):
        blocks=[]
        for i,x in enumerate(contexts,1):
            loc=x["document_name"]+(f", page {x['page']}" if x.get("page") is not None else "")
            blocks.append(f"[SOURCE {i}] {loc}\n{x['text']}")
        context="\n\n".join(blocks)
        prompt=f'''You are a private-document question answering assistant.\nAnswer the user's question using ONLY the supplied document context.\nRules: do not invent facts; do not use outside knowledge; if unsupported say "I could not find sufficient information in the provided documents."; cite supporting sources as [SOURCE N].\n\nDOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nANSWER:\n'''
        messages=[{"role":"user","content":prompt}]
        try: rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(rendered,return_tensors="pt",truncation=True,max_length=4096)
        with torch.no_grad(): out=self.model.generate(**inputs,max_new_tokens=300,do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
        gen=out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen,skip_special_tokens=True).strip()
