import json
import requests
import os
import time
import argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import random

class LLMEvaluator:
    def __init__(self, api_key, api_url, model_name, jsonl_path, show_stats=True):
        """
        初始化评测器
        
        参数:
        api_key: API密钥
        api_url: 模型API的URL
        model_name: 被评测的模型名称
        jsonl_path: 包含评测问题的JSONL文件路径
        show_stats: 是否显示数据集统计信息
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.jsonl_path = jsonl_path
        self.questions = self.load_questions(show_stats=show_stats)
        self.results = []
        self.stats = {
            "total": 0,
            "correct": 0,
            "domains": defaultdict(lambda: {"total": 0, "correct": 0}),
            "abilities": defaultdict(lambda: {"total": 0, "correct": 0}),
            "languages": defaultdict(lambda: {"total": 0, "correct": 0})
        }
        
    def load_questions(self, show_stats=True):
        """
        加载JSONL文件中的问题并显示数据集统计信息
        
        参数:
        show_stats: 是否显示数据集统计信息
        
        返回:
        问题列表
        """
        questions = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('...'):
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if show_stats and questions:
            self._show_dataset_stats(questions)
        
        return questions
    
    def _show_dataset_stats(self, questions):
        """
        显示数据集统计信息
        
        参数:
        questions: 问题列表
        """
        total_count = len(questions)
        
        # 统计领域分布
        domain_counts = {}
        for q in questions:
            domain = q.get('domain', '未知')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # 统计语言分布
        language_counts = {}
        for q in questions:
            language = q.get('language', '未知')
            language_counts[language] = language_counts.get(language, 0) + 1
        
        # 统计能力分布
        ability_counts = {}
        for q in questions:
            ability = q.get('ability', '未知')
            ability_counts[ability] = ability_counts.get(ability, 0) + 1
        
        # 打印统计信息
        print(f"\n{'='*50}")
        print(f"数据集统计信息 - {self.jsonl_path}")
        print(f"{'='*50}")
        print(f"总问题数: {total_count}")
        
        # 打印领域分布
        print(f"\n{'-'*20} 领域分布 {'-'*20}")
        domain_df = pd.DataFrame([
            {"领域": domain, "问题数": count, "占比": count/total_count}
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        print(domain_df.to_string(index=False, formatters={"占比": "{:.2%}".format}))
        
        # 打印语言分布
        print(f"\n{'-'*20} 语言分布 {'-'*20}")
        language_df = pd.DataFrame([
            {"语言": language, "问题数": count, "占比": count/total_count}
            for language, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        print(language_df.to_string(index=False, formatters={"占比": "{:.2%}".format}))
        
        # 打印能力分布
        print(f"\n{'-'*20} 能力分布 {'-'*20}")
        ability_df = pd.DataFrame([
            {"能力": ability, "问题数": count, "占比": count/total_count}
            for ability, count in sorted(ability_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        print(ability_df.to_string(index=False, formatters={"占比": "{:.2%}".format}))
        
        # 示例问题
        print(f"\n{'-'*20} 示例问题 {'-'*20}")
        example = random.choice(questions)
        print(f"问题: {example['question']}")
        for i, ans in enumerate(example['answers']):
            print(f"{chr(65+i)}. {ans}")
        print(f"正确答案: {example['label']}")
        print(f"领域: {example.get('domain', '未知')}")
        print(f"能力: {example.get('ability', '未知')}")
        print(f"语言: {example.get('language', '未知')}")
        
        print(f"\n{'='*50}\n")
    
    def get_api_request(self, model_name, formatted_question, stream=False):
        """
        根据模型名称生成适当的API请求体
        
        参数:
        model_name: 模型名称
        formatted_question: 格式化后的问题
        stream: 是否使用流式响应模式
        
        返回:
        API请求数据
        """
        # 默认格式(OpenAI兼容格式)
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": formatted_question}],
            "temperature": 0.01,
            "stream": stream
        }
        
        # Ollama格式
        if "ollama" in self.api_url.lower():
            data = {
                "model": model_name,
                "prompt": formatted_question,
                "stream": stream
            }
        
        # LocalAI格式类似OpenAI
        
        # HuggingFace格式
        elif "huggingface" in self.api_url.lower():
            data = {
                "inputs": formatted_question,
                "parameters": {
                    "temperature": 0,
                    "max_new_tokens": 100
                }
            }
            # HuggingFace API可能不支持流式响应
        
        return data
    
    def call_api(self, question, answers, debug=False, timeout=30, stream=False):
        """
        调用大模型API
        
        参数:
        question: 问题文本
        answers: 选项列表
        debug: 是否打印调试信息
        timeout: API请求超时时间(秒)
        stream: 是否使用流式响应模式
        
        返回:
        模型的回答
        """
        # 格式化问题和选项
        formatted_question = f"你是专业的网络安全专家，请回答以下选择题，只需给出选项字母（A、B、C或D）作为答案：\n\n{question}\n"
        for i, answer in enumerate(answers):
            formatted_question += f"{chr(65+i)}. {answer}\n"
        formatted_question += "\n请直接回答选项字母，例如：A"
        
        if debug:
            print("\n" + "="*50)
            print("调用API - 请求信息:")
            print(f"模型: {self.model_name}")
            print(f"问题内容:\n{formatted_question}")
            print(f"流式模式: {'开启' if stream else '关闭'}")
            print("="*50)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 根据不同API调整请求体
        data = self.get_api_request(self.model_name, formatted_question, stream=stream)
        
        # 重试计数
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                if debug:
                    print(headers, data)
                    print(f"正在发送请求到: {self.api_url} (超时设置: {timeout}秒)")
                
                start_time = time.time()
                
                if stream:
                    # 处理流式响应
                    full_content = ""
                    
                    with requests.post(self.api_url, headers=headers, json=data, timeout=timeout, stream=True) as response:
                        response.raise_for_status()
                        
                        if debug:
                            print("接收流式响应...")
                        
                        # 处理OpenAI兼容格式的流式响应
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: ') and line != 'data: [DONE]':
                                    try:
                                        json_str = line[6:]  # 去掉 'data: ' 前缀
                                        chunk = json.loads(json_str)
                                        
                                        if "choices" in chunk and len(chunk["choices"]) > 0:
                                            delta = chunk["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                content_chunk = delta["content"]
                                                full_content += content_chunk
                                                
                                                if debug:
                                                    print(content_chunk, end="", flush=True)
                                    except Exception as e:
                                        if debug:
                                            print(f"解析流式响应出错: {e}")
                                            print(f"原始行: {line}")
                    
                    if debug:
                        print("\n流式响应接收完成")
                    
                    content = full_content
                    response_time = time.time() - start_time
                    
                    # 过滤<think>标签内容
                    content = self.filter_thinking(content)
                    
                    if debug:
                        print("\n" + "-"*50)
                        print(f"流式响应总结 (耗时: {response_time:.2f}秒):")
                        print(f"模型回答:\n{content}")
                        print("-"*50)
                    
                    return content
                else:
                    # 处理普通响应
                    response = requests.post(self.api_url, headers=headers, json=data, timeout=timeout)
                    response_time = time.time() - start_time
                    
                    if debug:
                        print(f"请求耗时: {response_time:.2f}秒")
                        print(response)

                    response.raise_for_status()
                    
                    # 尝试解析JSON响应
                    try:
                        result = response.json()
                        
                        # 标准OpenAI兼容格式处理
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            
                            # 过滤<think>标签内容
                            content = self.filter_thinking(content)
                            
                            if debug:
                                print("\n" + "-"*50)
                                print(f"API响应 (耗时: {response_time:.2f}秒):")
                                print(f"响应状态码: {response.status_code}")
                                print(f"模型回答:\n{content}")
                                print("-"*50)
                                
                            return content
                        
                        # 其他可能的JSON响应格式
                        if "response" in result:
                            content = result["response"]
                            content = self.filter_thinking(content)
                            return content
                            
                        if "output" in result:
                            content = result["output"]
                            content = self.filter_thinking(content)
                            return content
                            
                        if "generation" in result:
                            content = result["generation"]
                            content = self.filter_thinking(content)
                            return content
                        
                        if debug:
                            print("\n" + "-"*50)
                            print("API响应格式未识别，但返回了JSON:")
                            print(f"响应内容: {result}")
                            print("-"*50)
                            
                    except json.JSONDecodeError:
                        # 非JSON响应处理
                        content = response.text
                        content = self.filter_thinking(content)
                        
                        if debug:
                            print("\n" + "-"*50)
                            print("收到非JSON响应:")
                            print(f"原始响应:\n{content}")
                            print("-"*50)
                            
                        return content
                    
                return None
            except requests.exceptions.Timeout:
                retry_count += 1
                if debug:
                    print(f"\n请求超时 (已尝试 {retry_count}/{max_retries})")
                
                if retry_count < max_retries:
                    wait_time = retry_count * 5  # 递增等待时间
                    if debug:
                        print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"API请求在 {timeout} 秒内未响应，已达到最大重试次数 ({max_retries})")
                    return None
            except Exception as e:
                if debug:
                    print("\n" + "-"*50)
                    print(f"API调用出错: {str(e)}")
                    print(f"详细错误: {repr(e)}")
                    if 'response' in locals():
                        print(f"响应状态码: {response.status_code if response else 'N/A'}")
                        try:
                            print(f"响应内容: {response.text if response else 'N/A'}")
                        except:
                            pass
                    print("-"*50)
                
                print(f"API调用出错: {e}")
                
                # 对于非超时错误，也可以尝试重试
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 3
                    print(f"等待 {wait_time} 秒后重试... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return None
    
    def filter_thinking(self, text):
        """
        过滤掉模型回答中的<think>标签内容
        
        参数:
        text: 模型回答文本
        
        返回:
        过滤后的文本
        """
        if not text:
            return text
        
        # 过滤<think>标签内容
        import re
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        
        # 过滤其他可能的思考格式
        filtered_text = re.sub(r'\[思考过程\].*?\[/思考过程\]', '', filtered_text, flags=re.DOTALL)
        filtered_text = re.sub(r'\[THINKING\].*?\[/THINKING\]', '', filtered_text, flags=re.DOTALL)
        
        return filtered_text.strip()
    
    def extract_answer(self, response_text):
        """
        从模型回答中提取选项
        
        参数:
        response_text: 模型的回答文本
        
        返回:
        提取的选项（A、B、C、D或多选情况下的组合）
        """
        if not response_text:
            return None
        
        # 过滤思考标签内容
        response_text = self.filter_thinking(response_text)
        
        # 转换为大写以统一处理
        response_text = response_text.upper()
        
        # 处理多选情况
        multi_options = ['ABC', 'ABD', 'ACD', 'BCD', 'AB', 'AC', 'AD', 'BC', 'BD', 'CD']
        for combo in multi_options:
            if combo in response_text:
                return combo
        
        # 使用正则表达式匹配独立的选项字母
        import re
        match = re.search(r'\b([A-D])\b', response_text)
        if match:
            return match.group(1)
        
        # 寻找直接的选项标记
        options = ['A', 'B', 'C', 'D']
        patterns = []
        
        # 中文模式
        for option in options:
            patterns.extend([
                f"答案是{option}", f"答案:{option}", f"答案： {option}", 
                f"选择{option}", f"选{option}", f"{option}是正确的",
                f"答案应该是{option}", f"应选{option}", f"正确答案是{option}"
            ])
        
        # 英文模式
        for option in options:
            patterns.extend([
                f"THE ANSWER IS {option}", f"ANSWER IS {option}", f"ANSWER: {option}",
                f"OPTION {option}", f"CHOOSE {option}", f"{option} IS CORRECT",
                f"THE CORRECT ANSWER IS {option}", f"I CHOOSE {option}", f"ANSWER = {option}"
            ])
        
        for pattern in patterns:
            if pattern in response_text:
                for opt in options:
                    if opt in pattern:
                        return opt
        
        # 查找单独出现的选项（如模型只回答了"A"）
        if len(response_text.strip()) <= 5:  # 考虑可能有些空格或标点
            for option in options:
                if option in response_text:
                    return option
        
        # 计算每个选项在回答中出现的位置
        positions = {}
        for option in options:
            pos = response_text.find(option)
            if pos != -1:
                positions[option] = pos
        
        # 如果找到了选项，返回位置最靠前的
        if positions:
            return min(positions, key=positions.get)
        
        return None
    
    def evaluate_single(self, question_data, debug=False, timeout=30, stream=False):
        """
        评估单个问题
        
        参数:
        question_data: 问题数据
        debug: 是否打印调试信息
        timeout: API请求超时时间(秒)
        stream: 是否使用流式响应模式
        """
        question = question_data["question"]
        answers = question_data["answers"]
        correct_label = question_data["label"]
        domain = question_data["domain"]
        ability = question_data["ability"]
        language = question_data["language"]
        
        if debug:
            print(f"\n处理问题: {question}")
            print(f"正确答案: {correct_label}")
        
        model_response = self.call_api(question, answers, debug=debug, timeout=timeout, stream=stream)
        
        if model_response:
            extracted_answer = self.extract_answer(model_response)
            is_correct = extracted_answer == correct_label
            
            if debug:
                print(f"提取的答案: {extracted_answer}")
                print(f"是否正确: {'✓' if is_correct else '✗'}")
            
            result = {
                "question": question,
                "answers": answers,
                "correct_label": correct_label,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "domain": domain,
                "ability": ability,
                "language": language
            }
            
            # 更新统计数据
            self.stats["total"] += 1
            if is_correct:
                self.stats["correct"] += 1
                
            self.stats["domains"][domain]["total"] += 1
            if is_correct:
                self.stats["domains"][domain]["correct"] += 1
                
            self.stats["abilities"][ability]["total"] += 1
            if is_correct:
                self.stats["abilities"][ability]["correct"] += 1
                
            self.stats["languages"][language]["total"] += 1
            if is_correct:
                self.stats["languages"][language]["correct"] += 1
                
            return result
        return None
    
    def run_evaluation(self, limit=None, sample=None, domain=None, language=None, debug=False, timeout=30, live_update=True, update_interval=10, stream=False):
        """
        运行完整评测
        
        参数:
        limit: 评测的问题数量限制（None表示评测所有问题）
        sample: 随机抽样的问题数量
        domain: 只评测特定领域的问题
        language: 只评测特定语言的问题
        debug: 是否打印调试信息
        timeout: API请求超时时间(秒)
        live_update: 是否实时显示评测结果
        update_interval: 实时更新的间隔（评测多少个问题后更新一次）
        stream: 是否使用流式响应模式
        """
        questions_to_evaluate = self.questions
        
        # 应用过滤条件
        if domain:
            questions_to_evaluate = [q for q in questions_to_evaluate if q["domain"] == domain]
        if language:
            questions_to_evaluate = [q for q in questions_to_evaluate if q["language"] == language]
        
        # 应用限制或抽样
        if sample and sample < len(questions_to_evaluate):
            import random
            questions_to_evaluate = random.sample(questions_to_evaluate, sample)
        elif limit and limit < len(questions_to_evaluate):
            questions_to_evaluate = questions_to_evaluate[:limit]
        
        total_questions = len(questions_to_evaluate)
        print(f"开始评测，共{total_questions}个问题")
        
        # 初始化实时进度条
        if live_update and not debug:
            from tqdm import tqdm
            pbar = tqdm(total=total_questions, desc="评测进度")
            
            # 在进度条下方预留空间用于显示实时结果
            print("\n" * 5)  # 预留空间
        
        completed = 0
        start_time = time.time()
        
        for i, question_data in enumerate(questions_to_evaluate):
            if debug:
                result = self.evaluate_single(question_data, debug=debug, timeout=timeout, stream=stream)
            else:
                # 在非调试模式下，不显示单个问题的详细信息
                result = self.evaluate_single(question_data, debug=False, timeout=timeout, stream=stream)
                
            if result:
                self.results.append(result)
                completed += 1
                
                # 更新进度条
                if live_update and not debug:
                    pbar.update(1)
                    
                    # 定期更新实时统计信息
                    if (i + 1) % update_interval == 0 or i == total_questions - 1:
                        current_accuracy = self.stats["correct"] / self.stats["total"] if self.stats["total"] > 0 else 0
                        elapsed_time = time.time() - start_time
                        avg_time_per_q = elapsed_time / (i + 1)
                        remaining_time = avg_time_per_q * (total_questions - i - 1)
                        
                        # 将光标移动到保留的区域，并显示实时结果
                        print(f"\033[{6}A")  # 向上移动6行
                        print(f"\033[K当前进度: {i+1}/{total_questions} ({(i+1)/total_questions:.1%})")
                        print(f"\033[K当前准确率: {current_accuracy:.2%} ({self.stats['correct']}/{self.stats['total']})")
                        print(f"\033[K已用时间: {elapsed_time:.1f}秒 (平均每题: {avg_time_per_q:.1f}秒)")
                        print(f"\033[K预计剩余: {remaining_time:.1f}秒 (约{remaining_time/60:.1f}分钟)")
                        
                        # 显示领域分布
                        domain_counts = {}
                        for r in self.results:
                            domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
                        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        domain_str = ", ".join([f"{d}: {c}" for d, c in top_domains])
                        print(f"\033[K主要领域: {domain_str}")
                        
                        print("\033[6B")  # 光标恢复
        
        if live_update and not debug:
            pbar.close()
        
        elapsed_time = time.time() - start_time
        print(f"\n测评完成! 共评测{completed}/{total_questions}个问题，用时{elapsed_time:.1f}秒，平均每题{elapsed_time/total_questions:.1f}秒")
        
        # 生成最终报告
        self.generate_report()
    
    def generate_report(self):
        """生成评测报告"""
        overall_accuracy = self.stats["correct"] / self.stats["total"] if self.stats["total"] > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"大模型评测报告 - {self.model_name}")
        print(f"{'='*50}")
        print(f"总问题数: {self.stats['total']}")
        print(f"正确回答数: {self.stats['correct']}")
        print(f"总体准确率: {overall_accuracy:.2%}")
        
        # 按领域统计
        print(f"\n{'-'*20} 领域统计 {'-'*20}")
        domain_df = pd.DataFrame([
            {
                "领域": domain,
                "问题数": stats["total"],
                "正确数": stats["correct"],
                "准确率": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            }
            for domain, stats in self.stats["domains"].items()
        ])
        domain_df = domain_df.sort_values("准确率", ascending=False)
        print(domain_df.to_string(index=False, formatters={"准确率": "{:.2%}".format}))
        
        # 按能力统计
        print(f"\n{'-'*20} 能力统计 {'-'*20}")
        ability_df = pd.DataFrame([
            {
                "能力": ability,
                "问题数": stats["total"],
                "正确数": stats["correct"],
                "准确率": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            }
            for ability, stats in self.stats["abilities"].items()
        ])
        ability_df = ability_df.sort_values("准确率", ascending=False)
        print(ability_df.to_string(index=False, formatters={"准确率": "{:.2%}".format}))
        
        # 按语言统计
        print(f"\n{'-'*20} 语言统计 {'-'*20}")
        language_df = pd.DataFrame([
            {
                "语言": language,
                "问题数": stats["total"],
                "正确数": stats["correct"],
                "准确率": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            }
            for language, stats in self.stats["languages"].items()
        ])
        language_df = language_df.sort_values("准确率", ascending=False)
        print(language_df.to_string(index=False, formatters={"准确率": "{:.2%}".format}))
        
        # 保存详细结果到CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"../results/{self.model_name}_evaluation_results.csv", index=False, encoding="utf-8-sig")
        print(f"\n详细评测结果已保存到: ../results/{self.model_name}_evaluation_results.csv")

def main():
    parser = argparse.ArgumentParser(description="大模型评测工具")
    parser.add_argument("--api_key", default="sk-proj-1234567890", help="API密钥")
    parser.add_argument("--api_url", default="http://localhost:11434/v1/chat/completions", help="API URL")
    parser.add_argument("--model", default="deepseek-r1:1.5b", help="模型名称")
    parser.add_argument("--data", default="../data/selected_mcqs.jsonl", help="测评数据集路径")
    parser.add_argument("--limit", type=int, help="评测问题数量限制")
    parser.add_argument("--sample", type=int, help="随机抽样评测的问题数量")
    parser.add_argument("--domain", help="只评测特定领域的问题")
    parser.add_argument("--language", help="只评测特定语言的问题")
    parser.add_argument("--debug", action="store_true", help="启用调试模式，打印详细信息")
    parser.add_argument("--timeout", type=int, default=30, help="API请求超时时间(秒)")
    parser.add_argument("--no-live", action="store_true", help="禁用实时显示评测结果")
    parser.add_argument("--update-interval", type=int, default=10, help="实时更新的间隔数")
    parser.add_argument("--no-stats", action="store_true", help="不显示数据集统计信息")
    parser.add_argument("--stream", action="store_true", help="使用流式响应模式")
    
    args = parser.parse_args()
    
    evaluator = LLMEvaluator(
        api_key=args.api_key,
        api_url=args.api_url,
        model_name=args.model,
        jsonl_path=args.data,
        show_stats=not args.no_stats
    )
    
    evaluator.run_evaluation(
        limit=args.limit,
        sample=args.sample,
        domain=args.domain,
        language=args.language,
        debug=args.debug,
        timeout=args.timeout,
        live_update=not args.no_live,
        update_interval=args.update_interval,
        stream=args.stream
    )

if __name__ == "__main__":
    main() 