import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app) 

# 1. Veilige Firebase koppeling
firebase_creds_dict = json.loads(os.getenv("FIREBASE_JSON"))
cred = credentials.Certificate(firebase_creds_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# 2. Hugging Face API Setup (De externe hersenen)
HF_TOKEN = os.getenv("HF_TOKEN")
# We gebruiken een heel slim, snel model dat via de API beschikbaar is
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get("user_id", "anoniem")
    message = data.get("message")
    
    # 3. AI antwoord ophalen via het internet (kost Render GEEN geheugen!)
    try:
        payload = {"inputs": message, "parameters": {"max_new_tokens": 150}}
        response = requests.post(API_URL, headers=headers, json=payload)
        response_data = response.json()
        
        # Haal de gegenereerde tekst uit het antwoord
        if isinstance(response_data, list) and 'generated_text' in response_data[0]:
            # Haal de vraag van de gebruiker uit het antwoord
            reply = response_data[0]['generated_text'].replace(message, "").strip()
        else:
            reply = "Ik ben even in de war, probeer het nog eens!"
    except Exception as e:
        reply = "Mijn brein is even offline. Check de logs!"

    # 4. Opslaan in Firestore
    db.collection('chats').document(user_id).collection('history').add({
        "message": message,
        "reply": reply,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    
    return jsonify({"reply": reply})

# 5. Server starten
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
