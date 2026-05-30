import os
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app) 

# 1. Veilige Firebase koppeling
firebase_creds_dict = json.loads(os.getenv("FIREBASE_JSON"))
cred = credentials.Certificate(firebase_creds_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# 2. Geoptimaliseerd model laden voor minder geheugengebruik
HF_TOKEN = os.getenv("HF_TOKEN")
model_path = snapshot_download(repo_id="Littendekitten/Orbit-Model", repo_type="dataset", token=HF_TOKEN)

print("Orbit model wordt zuinig geladen...")
# We laden in float16 (half precisie) en gebruiken low_cpu_mem_usage om crashen te voorkomen
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get("user_id", "anoniem")
    message = data.get("message")
    
    # AI antwoord genereren
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Opslaan in Firestore
    db.collection('chats').document(user_id).collection('history').add({
        "message": message,
        "reply": reply,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    
    return jsonify({"reply": reply})

# 3. Server starten (Render poort)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
