import json
import os
from typing import Dict, List, Set
import glob
from collections import Counter
from opencc import OpenCC
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def clean_text(text: str) -> str:
    """清理文本，去除无用字符和格式"""
    # 去除多余的空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符
    text = re.sub(r'[^\u4e00-\u9fff。，、；：？！""''（）《》\s]', '', text)
    # 合并多个标点符号
    text = re.sub(r'[。，、；：？！]{2,}', '。', text)
    return text.strip()

def load_history_files(directory: str, verbose: bool = True) -> List[Dict]:
    """加载并预处理历史文献文件"""
    history_files = glob.glob(os.path.join(directory, "*.json"))
    all_records = []
    
    # 初始化繁简转换器
    cc = OpenCC('t2s')
    
    for file_path in tqdm(history_files, desc="Loading history files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 如果是单个文本，转换为列表形式
                if isinstance(data, str):
                    data = [data]
                
                # 如果verbose模式，打印文件信息
                if verbose:
                    print(f"\nProcessing file: {os.path.basename(file_path)}")
                    print(f"Original data type: {type(data)}")
                    print(f"Original data structure: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                
                all_records.append({
                    "file_path": file_path,
                    "content": data
                })
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_records

def extract_text_content(records: List[Dict]) -> List[Dict]:
    """提取并预处理文本内容"""
    texts = []
    doc_id = 0
    cc = OpenCC('t2s')  # 繁简转换器
    
    def extract_text_from_dict(d: Dict, file_path: str):
        nonlocal doc_id
        text = ""
        
        if isinstance(d, str):
            # 处理字符串类型的内容
            text = d
        elif isinstance(d, dict):
            # 处理字典类型，递归提取文本
            for value in d.values():
                if isinstance(value, (dict, list)):
                    extract_text_from_dict(value, file_path)
                elif isinstance(value, str):
                    text = value
                else:
                    continue
        elif isinstance(d, list):
            # 处理列表类型，递归提取文本
            for item in d:
                extract_text_from_dict(item, file_path)
            return
        else:
            return
        
        # 文本预处理
        text = text.strip()
        if text:  # 确保文本非空
            # 繁体转简体
            text = cc.convert(text)
            # 清理文本
            text = clean_text(text)
            
            if len(text) > 10:  # 确保文本长度足够
                texts.append({
                    "doc_id": doc_id,
                    "text": text,
                    "source_file": file_path,
                    "keywords": []
                })
                doc_id += 1
    
    # 处理所有记录
    for record in records:
        extract_text_from_dict(record["content"], record["file_path"])
    
    # 打印样例检查
    print("\n=== Processed Text Samples ===")
    for i in range(min(5, len(texts))):
        print(f"\nSample {i + 1}:")
        print(f"Source: {texts[i]['source_file']}")
        print(f"Doc ID: {texts[i]['doc_id']}")
        print(f"Text: {texts[i]['text'][:150]}...")
        print(f"Length: {len(texts[i]['text'])} characters")
    
    print(f"\nTotal processed texts: {len(texts)}")
    
    return texts

def create_prompt(text: str) -> str:
    """Create prompt for keyword extraction."""
    history_prompt = """
    请分析以下中国古代历史文本段落，提取最重要的3-5个关键词（包括人物、地点、事件、时间等），
    并说明每个关键词的类型（如：人物、地点、事件等）和重要性。

    文本段落：{text}

    请以JSON格式返回，格式如下：
    {{"keywords": [
        {{"word": "关键词1", "type": "类型", "importance": "重要性说明"}},
        ...
    ]}}
    """
    taoist_prompt = """
    请分析以下中国古代道教文本段落，提取最重要的5-10个关键词（包括人物、术语、理论、概念等），
    并说明每个关键词的类型（如：人物、地点、事件等）和重要性。

    文本段落：{text}

    请以JSON格式返回，格式如下：
    {{"keywords": [
        {{"word": "关键词1", "type": "类型", "importance": "重要性说明"}},
        ...
    ]}}
    """
    return taoist_prompt

def extract_json_from_response(response: str) -> Dict:
    """增强版JSON解析，支持多种异常场景处理"""
    try:
        # 场景1：处理用```json包裹的标准响应
        if matches := re.findall(r'```json(.*?)```', response, re.DOTALL):
            json_str = matches[0].strip()  # 取第一个匹配的JSON块
            
        # 场景2：处理未包裹但直接输出JSON的情况
        else:
            json_str = response.strip()
            # 自动去除可能的首尾非JSON字符（如自然语言说明）
            json_str = re.sub(r'^[^{[]*', '', json_str)  # 去除开头非JSON字符
            json_str = re.sub(r'[^}\]]*$', '', json_str)  # 去除结尾非JSON字符

        # 严格解析（允许尾部存在其他内容）
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(json_str)
        return obj
    
    # 统一异常处理
    except (AttributeError, json.JSONDecodeError, IndexError) as e:
        print(f"JSON解析失败 | 错误类型：{type(e).__name__} | 原始响应：\n{response}")
        return {}
    except Exception as e:
        print(f"未知解析错误：{e} | 原始响应：\n{response}")
        return {}

def batch_analyze_texts(texts: List[Dict], batch_size: int = 32) -> List[Dict]:
    """Analyze texts in batches using vLLM."""
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=4,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=512,
    )

    all_prompts = [create_prompt(text_doc['text']) for text_doc in texts]
    
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_prompts = all_prompts[i:i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        batch_results = []
        
        for j, output in enumerate(outputs):
            try:
                response = output.outputs[0].text
                result = extract_json_from_response(response)
                
            except Exception as e:
                print(f"Error processing document {i + j}: {e}\nResponse content: {response}")
                result = {}  # 保持结构一致性
                
            batch_results.append(result)  # 修正为append

        output_file = f"analyzed_taoist_texts/analyzed_taoist_texts_{i//batch_size}.json"
        os.makedirs("analyzed_taoist_texts", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        results.extend(batch_results)
        
    return results

def perform_statistical_analysis(analyzed_texts: List[Dict]) -> Dict:
    """Perform statistical analysis on the analyzed texts."""
    # 统计关键词频率
    keyword_counter = Counter()
    keyword_types = {}
    keyword_docs = {}  # 记录每个关键词出现在哪些文档中
    
    for doc in analyzed_texts:
        # 如果没有keyword，则跳过
        if 'keywords' not in doc:
            continue
        for kw in doc['keywords']:
            # 如果没有word，则跳过
            if 'word' not in kw:
                continue
            word = kw['word']
            keyword_counter[word] += 1
            keyword_types[word] = kw['type']
            if word not in keyword_docs:
                keyword_docs[word] = set()
            keyword_docs[word].add(doc['doc_id'])
    
    # 出现频率最高的10个关键词及次数
    print(sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:10])
    # 计算关键词共现关系
    cooccurrence = {}
    for doc in analyzed_texts:
        # 如果没有keywords，则跳过
        if 'keywords' not in doc:
            continue
        keywords = [kw['word'] for kw in doc['keywords'] if 'word' in kw]
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i+1:]:
                pair = tuple(sorted([kw1, kw2]))
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
    
    result = {
        "keyword_frequencies": dict(keyword_counter),
        "keyword_types": keyword_types,
        "keyword_documents": {k: list(v) for k, v in keyword_docs.items()},
        #"keyword_cooccurrence": cooccurrence,
        "total_documents": len(analyzed_texts),
        "total_keywords": len(keyword_counter)
    }
    return result


def main():
    # Configuration
    history_dir = "./chinese/dao"
    output_file = "taoist_analysis.json"
    
    # Load and process files
    print("Loading and preprocessing history files...")
    records = load_history_files(history_dir, verbose=True)
    texts = extract_text_content(records)
    
    # 保存预处理后的文本，方便检查，由于文本量较大，可以分为多个文件保存
    batch_size = 1000
    directory = "preprocessed_taoist_texts"
    os.makedirs(directory, exist_ok=True)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with open(os.path.join(directory, f"preprocessed_texts_{i//batch_size}.json"), 'w', encoding='utf-8') as f:
            json.dump(batch_texts, f, ensure_ascii=False, indent=2)
    
    # Batch analyze texts using vLLM
    print("Analyzing texts with vLLM...")
    analyzed_texts = batch_analyze_texts(texts, batch_size=1000)

    
    # Perform statistical analysis
    #print("Performing statistical analysis...")
    #statistics = perform_statistical_analysis(analyzed_texts)
    
    # Save results
    #output = {
    #    "analyzed_texts": analyzed_texts,
    #    "statistics": statistics
    #}
    
    #with open(output_file, 'w', encoding='utf-8') as f:
    #    json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis complete. Results saved to {output_file}")


if __name__ == "__main__":
    # walk through the anlyzed_texts directory and merge all json files into one list
    analyzed_texts = []
    for file in os.listdir("analyzed_texts"):
        with open(os.path.join("analyzed_texts", file), 'r', encoding='utf-8') as f:
            analyzed_texts.extend(json.load(f))
    # 只保留dict类型的analyzed_texts
    analyzed_texts = [text for text in analyzed_texts if isinstance(text, dict)]
    # 给每个analyzed_texts添加doc_id
    for i, text in enumerate(analyzed_texts):
        text['doc_id'] = i
    statistics = perform_statistical_analysis(analyzed_texts)
    with open("statistics.json", 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    