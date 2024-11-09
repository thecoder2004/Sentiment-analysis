from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Bước 1: Tải lại model và tokenizer
model = AutoModelForSequenceClassification.from_pretrained('model')
tokenizer = AutoTokenizer.from_pretrained('tokenizer')

# Bước 2: Chuẩn bị dữ liệu đầu vào
input_text = "sản phẩm này rất tốt"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Bước 3: Dự đoán nhãn cho dữ liệu mới
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = logits.argmax(dim=1).item()

# Bước 4: Kiểm tra nhãn dự đoán
label_mapping = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
print(f"Nhận xét: '{input_text}' được phân loại là: {label_mapping[predicted_label]}")
