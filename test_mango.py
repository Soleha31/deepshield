import os, certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri = os.getenv("MONGO_URI")

client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=10000)
client.admin.command("ping")
print("MongoDB connected")