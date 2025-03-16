import json
import random
import pandas as pd
import os

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

# 按domain分组并随机选择
def select_random_by_domain(data, n=25):
    # 按domain分组
    domain_groups = {}
    for item in data:
        domain = item.get('domain', 'unknown')
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(item)
    
    # 从每个domain随机选择n个题目
    selected = []
    for domain, items in domain_groups.items():
        # 如果该domain的题目数量少于n，则全部选择
        if len(items) <= n:
            selected.extend(items)
            print(f"Domain '{domain}' 只有 {len(items)} 个题目，全部选择")
        else:
            selected.extend(random.sample(items, n))
            print(f"Domain '{domain}' 随机选择了 {n} 个题目")
    
    return selected

# 将数据转换为DataFrame并保存为Excel
def save_to_excel(data, output_file):
    # 准备DataFrame数据
    df_data = []
    for item in data:
        # 将answers列表转换为字符串
        answers_str = "\n".join([f"{chr(65+i)}. {ans}" for i, ans in enumerate(item.get('answers', []))])
        
        df_data.append({
            'question': item.get('question', ''),
            'answers': answers_str,
            'label': item.get('label', ''),
            'language': item.get('language', ''),
            'ability': item.get('ability', ''),
            'domain': item.get('domain', '')
        })
    
    # 创建DataFrame
    df = pd.DataFrame(df_data)
    
    # 保存为Excel
    df.to_excel(output_file, index=False)
    print(f"已将选择的题目保存到 {output_file}")

# 将数据保存为JSONL文件
def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"已将选择的题目保存到 {output_file}")

if __name__ == "__main__":
    input_file = "../data/SAQs_270.jsonl"
    output_excel = "../data/selected_saqs1.xlsx"
    output_jsonl = "../data/selected_saqs1.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    
    # 读取数据
    data = read_jsonl(input_file)
    print(f"共读取 {len(data)} 个题目")
    
    # 随机选择
    selected = select_random_by_domain(data, n=1)
    print(f"共选择 {len(selected)} 个题目")
    
    # 同时保存为Excel和JSONL格式
    save_to_excel(selected, output_excel)
    save_to_jsonl(selected, output_jsonl)