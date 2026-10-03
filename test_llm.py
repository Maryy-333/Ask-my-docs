import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"] # we will add a backup model below

def ask(prompt, retries=4):
    for model in MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                print(f"[{model}] attempt {attempt + 1} failed: {str(e)[:80]}")
                time.sleep(2 ** attempt)  # wait 1s, 2s, 4s, 8s
    return "All attempts failed."

print(ask("Explain RAG in two sentences."))

