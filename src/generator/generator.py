from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
import toml
import os

config = toml.load("config.toml")
google_api_key = config["api"]["GOOGLE_API_KEY"]
print(f"Google API Key: {google_api_key}")
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_response_with_gemini(doc_texts, query):
    context = " ".join(doc_texts)
    prompt = f"Context: {context}\n\nQuery: {query}"

    try:
        response = model.generate_content(prompt)
        generate_response = response.text
        return generate_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {e}"
    
def generate_response(context_docs, query):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    input_text = f"Context: {' '.join(context_docs)}\n\nQuery: {query}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)