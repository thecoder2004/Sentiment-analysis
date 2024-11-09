from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import nltk
import re
from nltk.tokenize import word_tokenize
import csv
import matplotlib.pyplot as plt

class SentimentDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item["labels"] = self.labels[idx]
        return item

# Sử dụng tokenizer của PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Tải mô hình PhoBERT cho bài toán phân loại (3 lớp)
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)

# Chuyển mô hình sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def toke(text):
    return word_tokenize(text)

def toke_to_string(tok):
    return ' '.join(tok)

def clean_length(words):
    return [word for word in words if len(word) > 2]

def preprocess(text):
    text = re.sub(r'http\S+', '', text)
    text = text.lower()  # Lowercase the text
    tokens = toke(text)
    clean_text = toke_to_string(tokens)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)  # Remove irrelevant characters
    clean_text = re.sub(r'(.)\1+', r'\1', clean_text)  # Thay thế các từ kiểu như quáaaaaaaaa
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

# Chuẩn bị dữ liệu từ file CSV
texts = []
labels = []

with open('preprossessing.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    for row in reader:
        texts.append(row['content'])
        labels.append(int(row['rating']) + 1)  # Giả định rating là -1 (tiêu cực), 0 (trung tính), 1 (tích cực)

# Tokenize dữ liệu
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
labels = torch.tensor(labels)

# Tạo dataset và chia thành các tập train, validation và test
dataset = SentimentDataset(inputs, labels)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Tạo DataLoader cho mỗi tập
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Huấn luyện mô hình
num_epochs = 3
for epoch in range(num_epochs):
    #Train
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # Giải phóng các biến không cần thiết
        del outputs, loss
        torch.cuda.empty_cache()  # Giải phóng bộ nhớ cache của GPU
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    # Validation
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Không tính toán gradient cho validation
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()
            
            # Lấy các dự đoán và nhãn thực tế để tính độ chính xác
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Giải phóng bộ nhớ sau mỗi epoch
    torch.cuda.empty_cache()

# Đánh giá mô hình trên tập test và vẽ confusion matrix
def evaluate(model, dataloader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
    
    accuracy = accuracy_score(true_labels, preds)
    print(f"Accuracy: {accuracy:.4f}")
    return true_labels, preds

# Đánh giá trên tập test và vẽ confusion matrix
true_labels, preds = evaluate(model, test_loader)

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tiêu cực", "Trung tính", "Tích cực"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix PhoBERT Model")
plt.show()

model.save_pretrained('model')
tokenizer.save_pretrained('tokenizer')
