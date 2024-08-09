import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import json
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
import re
import joblib

# SpaCy modelini yükle
nlp = spacy.load("tr_core_news_trf")

# Tokenizer ve model yükleme
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased").to(device)

# Sentiment etiketlerini rakamlarla eşleştiren bir sözlük
label_map = {0: "olumsuz", 1: "nötr", 2: "olumlu"}
reverse_label_map = {v: k for k, v in label_map.items()}

# Modelinizi yükleyin
model = joblib.load('C:/Users/PC/Downloads/Entity-Emotion-AI-main/Entity-Emotion-AI-main/teknofest_model_final_2.joblib')


def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def encode(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def predict_sentiment(text, model):
    preprocessed_text = clean_text(text)
    features = encode(preprocessed_text).reshape(1, -1)
    prediction = model.predict(features)[0]
    return label_map.get(prediction, "Unknown")


def split_text_with_punctuation(text):
    doc = nlp(text)
    sentences = []
    current_sentence = []
    conjunctions = {"ama", "ancak", "fakat", "çünkü", "o yüzden", "dolayısıyla",
                    "halbuki", "ne var ki", "ya da", "üstelik", "buna rağmen", "lakin"}
    punctuations = {'.', '?', '!', ',', ';', ':', '(', ')', '-', '—', '...'}

    for token in doc:
        if token.pos_ == "CCONJ" and token.text.lower() in conjunctions:
            if current_sentence:
                sentences.append(" ".join(current_sentence).strip())
                current_sentence = []
        elif token.text in punctuations:
            if current_sentence:
                if token.text == ',':
                    sentences.append(" ".join(current_sentence).strip() + '.')
                elif token.text == '...':
                    sentences.append(" ".join(current_sentence).strip() + '.')
                else:
                    sentences.append(" ".join(current_sentence).strip() + token.text)
                current_sentence = []
        else:
            current_sentence.append(token.text)

    if current_sentence:
        sentences.append(" ".join(current_sentence).strip())

    sentences = [sent for sent in sentences if sent]
    return sentences


def identify_entities(text):
    doc = nlp(text)
    excluded_labels = {"GPE", "LOC", "FAC", "EVENT", "DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL",
                       "CARDINAL", "LAW"}
    mentions = re.findall(r'@\w+', text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ not in excluded_labels]
    mentions = [(mention, "MENTION") for mention in mentions if mention not in [ent[0] for ent in entities]]
    entities.extend(mentions)

    # Filtreleme işlemi: tek karakterli veya sadece sayı olan entity'leri çıkar
    entities = [entity for entity in entities if len(entity[0]) > 1 and not entity[0].isdigit()]

    return entities


def clean_entity(entity):
    # Burada kesme işareti ile gelen ekleri kaldırıyoruz
    entity = re.sub(r"'.*$", "", entity)
    return entity.strip()


def is_punctuation_only(text):
    return all(char in '.,;!?–—:"' for char in text.strip())


def is_invalid_context(text):
    punctuation_only = is_punctuation_only(text)
    conjunctions = {"ve", "ama", "ancak", "fakat", "çünkü", "o yüzden", "dolayısıyla",
                    "halbuki", "ne var ki", "ya da", "üstelik", "buna rağmen", "lakin"}
    words = set(text.lower().strip().split())
    conjunction_only = all(word in conjunctions for word in words)
    return punctuation_only or conjunction_only


def entity_context_analysis(json_data):
    text = json_data.get("text", "")
    entities = identify_entities(text)
    entity_context_map = {}
    contexts = split_text_with_punctuation(text)

    # Cümle sonu entity'lerini başta değerlendirmek için:
    if len(entities) > 1:
        for entity_text, label in entities:
            cleaned_entity = clean_entity(entity_text)
            if cleaned_entity not in entity_context_map:
                entity_context_map[cleaned_entity] = {'label': label, 'contexts': []}
            for context in contexts:
                if entity_text in context:
                    # Cümle sonunda yer alan entity'leri, baştaki cümleye dahil et
                    if context.endswith(entity_text):
                        # Önceki cümle ile birleştir
                        if len(entity_context_map[cleaned_entity]['contexts']) > 0:
                            entity_context_map[cleaned_entity]['contexts'][-1] += " " + context
                        else:
                            entity_context_map[cleaned_entity]['contexts'].append(context)
                    else:
                        entity_context_map[cleaned_entity]['contexts'].append(context)
    else:
        if entities:
            entity_text, label = entities[0]
            cleaned_entity = clean_entity(entity_text)
            if len(contexts) == 1 and entity_text in contexts[0]:
                entity_context_map[cleaned_entity] = {'label': label, 'contexts': contexts}
            else:
                entity_context_map[cleaned_entity] = {'label': label, 'contexts': contexts}
    return entity_context_map


def process_text():
    input_text = text_input.get("1.0", tk.END).strip()
    try:
        json_data = json.loads(input_text)
    except json.JSONDecodeError as e:
        update_output_text(f"JSON Decode Error: {e}")
        return

    text = json_data.get("text", "")
    contexts = split_text_with_punctuation(text)

    # Cümlelere bölünmüş hali terminale yazdırma
    print("Cümlelere Bölünmüş Hali:")
    for sentence in contexts:
        print(sentence)

    results = {
        "entity_list": [],
        "results": []
    }

    entity_context_map = entity_context_analysis(json_data)
    seen_entities = set()  # Aynı entity ve sentiment kombinasyonunu bir kez eklemek için set

    for entity_text, data in entity_context_map.items():
        contexts = data['contexts']

        for context in contexts:
            predicted_sentiment = predict_sentiment(context, model)

            # Context ve sentiment sonucunu terminale yazdırma
            print(f"Entity: {entity_text}")
            print(f"Context: {context}")
            print(f"Predicted Sentiment: {predicted_sentiment}")
            print("-" * 50)

            # Aynı entity ve sentiment kombinasyonu daha önce görülmemişse, results listesine ekle
            entity_sentiment_key = (entity_text, predicted_sentiment)
            if entity_sentiment_key not in seen_entities:
                seen_entities.add(entity_sentiment_key)
                if entity_text not in results["entity_list"]:
                    results["entity_list"].append(entity_text)
                results["results"].append({
                    "entity": entity_text,
                    "sentiment": predicted_sentiment
                })

    update_output_text(json.dumps(results, ensure_ascii=False, indent=2))


# Tkinter arayüzü
window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("900x700")  # Pencere boyutunu ayarla

# Arka Plan Resmini Ayarla
try:
    background_image = Image.open("green_bg.jpg")
    background_photo = ImageTk.PhotoImage(background_image)
    background_label = tk.Label(window, image=background_photo)
    background_label.place(relwidth=1, relheight=1)
except Exception as e:
    print(f"Background image load error: {e}")

# Input Text
tk.Label(window, text="Enter JSON text:", bg='white').pack(pady=5)
text_input = scrolledtext.ScrolledText(window, width=100, height=15, wrap=tk.WORD, bg='light gray', font=("Arial", 12))
text_input.pack(pady=5)

# Process Button
tk.Button(window, text="Process", command=process_text, bg='blue', fg='white', font=("Arial", 12)).pack(pady=5)

# Output Text (ScrolledText)
tk.Label(window, text="Output:", bg='white').pack(pady=5)
output_text = scrolledtext.ScrolledText(window, width=100, height=15, wrap=tk.WORD, bg='light gray', font=("Arial", 12))
output_text.pack(pady=5)

def update_output_text(text):
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, text)
    output_text.config(state=tk.DISABLED)

window.mainloop()
