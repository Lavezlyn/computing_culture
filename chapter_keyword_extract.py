from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import jieba
import json
from collections import defaultdict
import pkuseg
import torch
import numpy as np
import os
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Union

# 繁体字停用词表（按词性分类）
TRADITIONAL_STOP_WORDS = {
    # 虚词
    '虚词': [
        '之', '其', '或', '亦', '方', '未', '既', '而', '及', '若', '乃', '則', '曰', '矣', 
        '于', '於', '何', '也', '云', '焉', '耳', '乎', '哉', '夫', '盖', '诸', '与', '也', 
        '所', '故', '是', '非', '否', '然', '此', '已', '而', '且', '以', '至', '致', '遂', 
        '尚', '皆', '曾', '尝', '且', '又', '无', '有', '其', '不', '可', '矣', '焉', '欤', 
        '耳', '夫', '盖'
    ],
    # 代词
    '代词': [
        '吾', '我', '汝', '尔', '彼', '此', '是', '之', '其', '谁', '何', '孰', '某', '某甲', 
        '某乙', '某丙', '某丁', '某戊', '某己', '某庚', '某辛', '某壬', '某癸'
    ],
    # 语气词
    '语气词': [
        '也', '矣', '焉', '耳', '乎', '哉', '夫', '盖', '欤', '耶', '邪', '也夫', '也哉', 
        '也耶', '也邪', '矣夫', '矣哉', '矣耶', '矣邪', '焉耳', '焉哉', '焉耶', '焉邪'
    ],
    # 连词
    '连词': [
        '而', '及', '若', '乃', '則', '与', '且', '以', '至', '致', '遂', '尚', '皆', '曾', 
        '尝', '又', '且夫', '且如', '且使', '且令', '且将', '且欲', '且愿', '且望', '且冀'
    ],
    # 副词
    '副词': [
        '未', '既', '方', '亦', '或', '所', '故', '是', '非', '否', '然', '此', '已', '无', 
        '有', '不', '可', '必', '定', '诚', '实', '真', '确', '确然', '确乎', '确然乎'
    ],
    # 时间词
    '时间词': [
        '昔', '今', '古', '往', '来', '前', '后', '先', '后', '初', '终', '始', '末', '旦', 
        '夕', '朝', '暮', '晨', '昏', '昼', '夜', '春', '夏', '秋', '冬', '年', '月', '日'
    ],
    # 数量词
    '数量词': [
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿', 
        '兆', '半', '双', '对', '群', '众', '诸', '多', '少', '几', '若干', '些许', '些许'
    ],
    # 称谓词
    '称谓词': [
        '子', '君', '臣', '王', '帝', '皇', '后', '妃', '太子', '公子', '公主', '夫人', 
        '大人', '小人', '君子', '小人', '士', '大夫', '卿', '相', '将', '帅', '师', '傅'
    ]
}

def extract_keywords_from_text(text: str, model, tokenizer, kw_model, use_classical_chinese: bool = True) -> List[Dict]:
    """从单个文本中提取关键词"""
    # 分词处理
    if use_classical_chinese:
        # 使用pkuseg进行古文分词
        seg = pkuseg.pkuseg(postag=True)  # 启用词性标注
        words_with_pos = seg.cut(text)
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
        words = ' '.join(jieba.cut(text))
        pos_tags = []
    
    # 合并所有停用词
    all_stop_words = []
    for category in TRADITIONAL_STOP_WORDS.values():
        all_stop_words.extend(category) 
    
    # 提取关键词
    keywords = kw_model.extract_keywords(
        words,
        keyphrase_ngram_range=(1, 1),
        stop_words=all_stop_words,
        top_n=10  # 每个章节取10个关键词
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
        # 只保留相似度最高的前3个
        keyword_info["similarities"] = dict(sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3])
    
    return enhanced_keywords

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

def process_chapter_keywords(file_path: str, use_classical_chinese: bool = True) -> Dict:
    """处理文件中的每个章节并提取关键词"""
    # 使用BERT模型
    model_name = "ethanyt/guwenbert-large"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    kw_model = KeyBERT(model)
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 存储结果
    results = {}
    
    # 处理每个章节
    for book, chapters in data.items():
        results[book] = {}
        for chapter, texts in chapters.items():
            # 将章节中的所有文本合并
            combined_text = ' '.join(texts)
            # 提取关键词
            keywords = extract_keywords_from_text(
                combined_text,
                model,
                tokenizer,
                kw_model,
                use_classical_chinese
            )
            results[book][chapter] = {
                "keywords": keywords,
                "metadata": {
                    "total_keywords": len(keywords),
                    "model": model_name,
                    "use_classical_chinese": use_classical_chinese
                }
            }
    
    return results

if __name__ == "__main__":
    file_dir = "./chinese/base"
    output_dir = "./chapter_keyword"
    os.makedirs(output_dir, exist_ok=True)
    for file_name in tqdm.tqdm(os.listdir(file_dir), desc="Processing files"):
        if file_name.endswith(".json"):
            file_path = os.path.join(file_dir, file_name)
            results = process_chapter_keywords(file_path, use_classical_chinese=True)
            output_path = os.path.join(output_dir, f"{file_name}_chapter_keywords.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"章节关键词已保存到: {output_path}")
