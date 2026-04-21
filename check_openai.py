import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file
load_dotenv()

# Get the key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ ERROR: No OPENAI_API_KEY found in .env file.")
else:
    print(f"🔍 Testing key (starting with: {api_key[:8]}...)")
    
    client = OpenAI(api_key=api_key)
    
    try:
        # A very tiny request to minimize cost and check validity
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=5
        )
        print("✅ SUCCESS: Your OpenAI key is valid! Response received:")
        print(f"   -> \"{response.choices[0].message.content.strip()}\"")
    
    except Exception as e:
        print(f"❌ FAILED: Your key may be invalid or expired.")
        print(f"   Detail: {e}")
