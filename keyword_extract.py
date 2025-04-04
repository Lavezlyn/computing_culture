from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import jieba
import json
from collections import defaultdict
import pkuseg
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Union

def extract_text_from_json(json_obj):
    """递归提取JSON中的所有文本值"""
    texts = []
    if isinstance(json_obj, dict):
        for value in json_obj.values():
            texts.extend(extract_text_from_json(value))
    elif isinstance(json_obj, list):
        for item in json_obj:
            texts.extend(extract_text_from_json(item))
    elif isinstance(json_obj, str):
        texts.append(json_obj)
    return texts

def get_word_embedding(model, tokenizer, word: str) -> np.ndarray:
    """获取单个词的BERT嵌入向量"""
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用[CLS]标记的输出作为词向量
    return outputs.last_hidden_state[0][0].numpy()

def calculate_similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """计算词向量之间的相似度矩阵"""
    return cosine_similarity(embeddings)

def extract_keywords_from_file(file_path: str, use_classical_chinese: bool = True) -> Dict:
    # 使用古文BERT
    model_name = "ethanyt/guwenbert-large"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    kw_model = KeyBERT(model)
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有文本
    all_texts = extract_text_from_json(data)
    combined_text = ' '.join(all_texts)
    
    # 分词处理
    if use_classical_chinese:
        # 使用pkuseg进行古文分词
        seg = pkuseg.pkuseg(postag=True)  # 启用词性标注
        words_with_pos = seg.cut(combined_text)
        # 只保留名词、动词、形容词等实意词
        valid_pos = {'n', 'v', 'a', 'i', 'j', 'l'}  # 名词、动词、形容词、成语、简称、习语
        words = []
        pos_tags = []
        for word, pos in words_with_pos:
            if pos[0] in valid_pos:
                words.append(word)
                pos_tags.append(pos)
        words = ' '.join(words)
    else:
        # 使用jieba进行现代文分词
        words = ' '.join(jieba.cut(combined_text))
        pos_tags = []
    
    # 扩展的古文停用词
    classical_stop_words = [
        '之', '其', '或', '亦', '方', '未', '既', '而', '及', '若', '乃', '則', '子曰', '曰',
        '的', '了', '以', '者', '矣', '于', '於', '何', '也', '之', '云', '焉', '矣', '耳',
        '乎', '哉', '夫', '盖', '诸', '夫', '与', '也', '所', '故', '是', '非', '否', '然',
        '此', '已', '而', '且', '以', '至', '致', '遂', '尚', '皆', '曾', '尝', '且', '又',
        '无', '有', '其', '不', '可', '矣', '焉', '欤', '耳', '夫', '盖',
    ]
    
    # 提取关键词
    keywords = kw_model.extract_keywords(
        words,
        keyphrase_ngram_range=(1, 1),
        stop_words=classical_stop_words,
        top_n=100
    )
    
    # 获取关键词的词向量
    keyword_embeddings = []
    enhanced_keywords = []
    
    for keyword, score in keywords:
        # 获取词向量
        embedding = get_word_embedding(model, tokenizer, keyword)
        keyword_embeddings.append(embedding)
        
        # 获取词性（如果可用）
        pos = None
        if use_classical_chinese and keyword in words.split():
            idx = words.split().index(keyword)
            if idx < len(pos_tags):
                pos = pos_tags[idx]
        
        enhanced_keywords.append({
            "keyword": keyword,
            "score": score,
            "pos": pos,
            "embedding": embedding.tolist()  # 转换为列表以便JSON序列化
        })
    
    # 计算相似度矩阵
    similarity_matrix = calculate_similarity_matrix(keyword_embeddings)
    
    # 为每个关键词添加相似度信息
    for i, keyword_info in enumerate(enhanced_keywords):
        # 获取与其他关键词的相似度
        similarities = {
            enhanced_keywords[j]["keyword"]: float(similarity_matrix[i][j])
            for j in range(len(enhanced_keywords))
            if i != j
        }
        # 只保留相似度最高的前5个
        keyword_info["similarities"] = dict(sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
    
    return {
        "keywords": enhanced_keywords,
        "metadata": {
            "total_keywords": len(enhanced_keywords),
            "model": model_name,
            "use_classical_chinese": use_classical_chinese
        }
    }

if __name__ == "__main__":
    file_path = "./chinese/base/n028.json"
    file_name = file_path.split(".json")[0].split("/")[-1]
    output_path = f"./keyword/{file_name}_keywords.json"
    results = extract_keywords_from_file(file_path, use_classical_chinese=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    