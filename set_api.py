import os
import google.generativeai as genai

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB_ZJuVpd68q8Icjh6E2sd8nAf2XSMJAmA"
genai.api_key = os.environ["GOOGLE_API_KEY"]

print("API key set successfully!")
