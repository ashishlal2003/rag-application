from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(context_docs, query):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    input_text = f"Context: {' '.join(context_docs)}\n\nQuery: {query}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)