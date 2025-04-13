from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import jieba
import json
from collections import defaultdict
import hanlp
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
    # 使用能识别繁体的BERT
    model_name = "Jihuai/bert-ancient-chinese"
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
        # 使用HanLP进行古文分词
        tokenizer_hanlp = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
        words = tokenizer_hanlp(combined_text)
        pos_tags = tagger(words)
        
        # 只保留名词、动词、形容词等实意词
        valid_pos = {'NN', 'VV', 'JJ', 'VA', 'NR', 'FW'}  # 名词、动词、形容词、专有名词等
        filtered_words = []
        filtered_pos = []
        for word, pos in zip(words, pos_tags):
            if pos in valid_pos:
                filtered_words.append(word)
                filtered_pos.append(pos)
        words = ' '.join(filtered_words)
        pos_tags = filtered_pos
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
        if use_classical_chinese and keyword in filtered_words:
            idx = filtered_words.index(keyword)
            if idx < len(filtered_pos):
                pos = filtered_pos[idx]
        
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
    
    # 在返回结果前移除embedding字段
    for keyword_info in enhanced_keywords:
        if "embedding" in keyword_info:
            del keyword_info["embedding"]
    
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
    