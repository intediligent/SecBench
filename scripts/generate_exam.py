# -*- coding: utf-8 -*-
import json

def generate_exam(jsonl_path, output_md):
    with open(jsonl_path, 'r', encoding='utf-8') as f, \
         open(output_md, 'w', encoding='utf-8') as md:

        md.write("# 网络安全知识测试卷\n\n## 一、单选题（每题5分，共100分）\n\n---\n")
        
        question_count = 0
        
        for line in f:
            if line.strip():
                data = json.loads(line)
                question_count += 1
                
                # 生成题目和答案
                md.write(f"**题目{question_count}**  \n")
                md.write(f"{data['question']}  \n")
                for i, ans in enumerate(data['answers']):
                    md.write(f"{chr(65+i)}. {ans}  \n")
                md.write(f"\n**参考答案**：{data['label']}\n")  # 新增答案行
                md.write("\n---\n\n")

# 使用示例
generate_exam('../data/selected_mcqs.jsonl', '../data/exam.md') 