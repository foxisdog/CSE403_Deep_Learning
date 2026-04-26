from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")
print(token)