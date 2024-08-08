import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import joblib
import os

# Gerekli NLTK verilerini indir
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\u[A-Za-z0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

def remove_stopwords(text):
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words('turkish'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(file_path, output_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].astype(str)
    data['text'] = data['text'].apply(clean_text)
    data['text'] = data['text'].apply(remove_stopwords)
    data.to_csv(output_path, index=False)
    print(f"Veriler başarıyla {output_path} dosyasına kaydedildi.")

# Tokenizer ve model yükleme
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased").to('cuda')

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
    input_ids = encoded_text['input_ids'].to('cuda')
    attention_mask = encoded_text['attention_mask'].to('cuda')
    with torch.no_grad():
        outputs = bert(input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# Verileri ön işleme
preprocess_data('train.csv', 'preprocessed_data.csv')
preprocess_data('test.csv', 'preprocessed_test.csv')

df_train = pd.read_csv('preprocessed_data.csv')
df_test = pd.read_csv('preprocessed_test.csv')

label_map = {"Negative": 0, "Notr": 1, "Positive": 2}

def labelencoder_data(df, label_map):
    print("Number of NaN values before mapping:", df['label'].isna().sum())
    df['label'] = df['label'].map(label_map)
    print("Number of NaN values after mapping:", df['label'].isna().sum())
    return df

df_train_clean = labelencoder_data(df_train, label_map)
df_test_clean = labelencoder_data(df_test, label_map)

# NaN değerleri temizleme
df_train_clean = df_train_clean.dropna(subset=['label'])
df_test_clean = df_test_clean.dropna(subset=['label'])

df_train_clean.to_csv('preprocessed_data_encoded.csv', index=False)
df_test_clean.to_csv('preprocessed_test_encoded.csv', index=False)

print("Training dataset:")
print(df_train_clean.head())
print("Number of NaN values in training labels:", df_train_clean['label'].isna().sum())
print("Unique labels in training dataset:", df_train_clean['label'].unique())

print("\nTest dataset:")
print(df_test_clean.head())
print("Number of NaN values in test labels:", df_test_clean['label'].isna().sum())
print("Unique labels in test dataset:", df_test_clean['label'].unique())

X_train = []
y_train = []
X_test = []
y_test = []

max_length = 512

for index, row in df_train_clean.iterrows():
    text = row['text']
    label = row['label']
    features = feature_extraction(text, tokenizer, bert)
    X_train.append(features)
    y_train.append(label)

for index, row in df_test_clean.iterrows():
    text = row['text']
    label = row['label']
    features = feature_extraction(text, tokenizer, bert)
    X_test.append(features)
    y_test.append(label)

X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train)
X_test = np.array(X_test, dtype=object)
y_test = np.array(y_test)

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, activation='tanh', solver='adam',
                      alpha=1e-5, learning_rate='constant', verbose=1, early_stopping=True)

model.fit(X_train.tolist(), y_train)

y_pred = model.predict(X_test.tolist())

print(classification_report(y_test, y_pred))

# Modeli kaydetme
model_path = 'C:/Users/PC/PycharmProjects/teknofest/teknofest_model_final.joblib'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print("Model başarıyla kaydedildi:", model_path)

# Modeli yükleme
loaded_model = joblib.load(model_path)
print("Model başarıyla yüklendi.")
