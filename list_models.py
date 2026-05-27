import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env ファイルを読み込む
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    print(f".env file found at: {env_path}")
    load_dotenv(env_path)
else:
    print("Warning: .env file NOT found in the current directory.")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY is NOT set in environment variables.")
    print("Please check your .env file format (e.g., GEMINI_API_KEY=your_key_here)")
else:
    # キーの一部を表示して確認（セキュリティのため先頭5文字のみ）
    print(f"API Key found (prefix): {api_key[:5]}...")
    
    try:
        genai.configure(api_key=api_key)
        print("\n--- Available Models ---")
        models = genai.list_models()
        found = False
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                found = True
        if not found:
            print("No models supporting 'generateContent' were found for this API key.")
    except Exception as e:
        print(f"Error occurred: {e}")
