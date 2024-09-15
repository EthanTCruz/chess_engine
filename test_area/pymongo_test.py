from pymongo import MongoClient

# Replace with the IP of your node and the NodePort
client = MongoClient('mongodb://192.168.68.50:30017')
try:
    # The ping command is a way to test connectivity to the database
    client.admin.command('ping')
    print("MongoDB connection successful")
except Exception as e:
    print(f"Error: {e}")
