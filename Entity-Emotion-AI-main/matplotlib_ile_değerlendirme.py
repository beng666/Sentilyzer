import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModel
import torch

# Modeli yükleme
model_path = 'C:/Users/PC/Downloads/Entity-Emotion-AI-main/Entity-Emotion-AI-main/teknofest_model_final_2.joblib'
model = joblib.load(model_path)
print("Model başarıyla yüklendi.")

# BERT Tokenizer ve Model Yükleme
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

def feature_extraction(text, tokenizer, bert, max_length=512):
    if not isinstance(text, str):
        print("Invalid text input:", text)
        return np.zeros((768,))
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']
    with torch.no_grad():
        outputs = bert(input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# Test verisini yükleme
test_file_path = 'C:/Users/PC/Downloads/Entity-Emotion-AI-main/Entity-Emotion-AI-main/preprocessed_test.csv'
df_test = pd.read_csv(test_file_path)

# Verileri işleme
texts = df_test['text'].tolist()
true_labels = df_test['label'].map({"Negative": 0, "Notr": 1, "Positive": 2}).tolist()  # Etiketlerin sayısal karşılıkları

# Özellik çıkarımı
X_test = [feature_extraction(text, tokenizer, bert) for text in texts]
y_test = np.array(true_labels)

# Model ile tahmin yapma
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Negative", "Notr", "Positive"], output_dict=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Notr", "Positive"], yticklabels=["Negative", "Notr", "Positive"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Precision, Recall ve F1-Score Grafikleri
labels = ["Negative", "Notr", "Positive"]
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1_score = [report[label]['f1-score'] for label in labels]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Labels')
ax.set_title('Precision, Recall and F1-Score by Label')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
