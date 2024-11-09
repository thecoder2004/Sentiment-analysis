import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('training.csv', encoding='utf-8-sig');


# Tách các từ trong tu dien và lấy danh sách các từ duy nhất
dicts = set()
abbreviations = {}
with open('Vietnamese-words.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # Tách câu thành các từ và in ra
        words = line.split()
        for word in words:
            dicts.add(word)

            
with open('viet-tat.txt', 'r', encoding='utf-8') as file:
        for line in file:
            # Loại bỏ khoảng trắng thừa ở hai đầu và bỏ qua dòng trống
            line = line.strip()
            if not line:
                continue
            
            # Tách từ viết tắt và ý nghĩa theo dấu hai chấm ':'
            key, value = line.split('-', 1)
            
            # Lưu vào dictionary
            abbreviations[key] = value.strip()

def expand_abbreviations(sentence):
    words = sentence.split()  # Tách câu thành từng từ
    expanded_sentence = []
    
    for word in words:
        # Kiểm tra xem từ có trong dictionary không
        if word in abbreviations:
            expanded_sentence.append(abbreviations[word])  # Thay thế bằng ý nghĩa
        else:
            expanded_sentence.append(word)  # Giữ nguyên từ gốc
    
    return ' '.join(expanded_sentence)  # Ghép các từ lại thành câu



# Vòng lặp để thêm cột cho mỗi từ duy nhất
def wordtovector(X):
    X = X.apply(lambda x: str(x).lower() if x is not None else '')
    X = X.apply(lambda x: expand_abbreviations(x) if x is not None else '')
    X = X.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    one_hot_encoded = pd.DataFrame({
        word: df['content'].apply(lambda x: 1 if word in x.split() else 0) for word in dicts
    })
    return one_hot_encoded



one_hot_encoded = wordtovector(df['content'])

df = pd.concat([df['content'],one_hot_encoded, df['rating']], axis = 1)


df.to_csv('final_output.csv', encoding = 'utf-8-sig')


