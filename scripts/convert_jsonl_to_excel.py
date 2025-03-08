import json
import pandas as pd
import os

# 输入和输出文件路径
input_file = '../data/MCQs_2730.jsonl'
output_file = '../data/MCQs_2730.xlsx'

# 读取JSONL文件
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 跳过空行
            data.append(json.loads(line))

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 处理answers列表，将其转换为单独的列
max_answers = max(len(item['answers']) for item in data)
for i in range(max_answers):
    col_name = f'answer_{chr(65+i)}'  # A, B, C, D...
    df[col_name] = df['answers'].apply(lambda x: x[i] if i < len(x) else None)

# 删除原始answers列表
df = df.drop('answers', axis=1)

# 保存为Excel文件
df.to_excel(output_file, index=False)

print(f"文件已成功转换并保存为: {output_file}")