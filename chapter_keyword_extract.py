from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import jieba
import json
from collections import defaultdict
import hanlp
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
        '大人', '小人', '君子', '小人', '士', '大夫', '卿', '相', '将', '帅', '师', '傅', '丞相', '御史', '伊尹'
    ],
    # 人名地名
    '人名地名': [
        '夫子', '顏淵', '子貢', '曾子', '子夏', '孟懿子', '孟武伯', '樊遲', '季康子', '子游', '子貢', '孟孫', '甯武子', '孟武伯', '晏平', '季文子', '子文', '閔子騫', '子華', '巫馬', '陳司', '孟子', '昭公', '葉公', '子路', '孟敬子', '百里', '武王', '孔子', '子罕', '閔子騫', '閔子', '白圭', '季子', '顏路', '皋陶', '司馬牛', '葉公', '冉子', '臧武仲', '齊桓公', '公明賈', '桓公', '晉文公', '伯玉', '康子', '臧文仲', '衛靈公', '伯夷', '陳亢', '顓臾', '齊景公', '公山', '首陽', '柳下惠', '季桓子', '魯孔丘', '箕子', '子張', '吴王', '襄平', '晉國', '黃帝', '司馬', '秦國', '杜摯', '甘龍', '伏羲', '郭偃', '孝公', '魏襄王', '東郭敞', '昊英', '黃鵠', '鄢郢', '江漢', '莊蹻', '鄧林', '楚國', '秦楚', '梁惠王', '梁襄王', '晉文', '齊桓', '齊國', '孟軻', '魯平公', '閔子', '齊王', '子襄', '陳臻', '孟仲子', '子思', '墨翟', '楊墨', '楊氏', '趙簡子', '楊朱', '薛居州', '楚大夫', '子敖', '尹公', '徐子', '沈猶', '秦繆公', '咸丘蒙', '晉平公', '季桓子', '子思', '孟季子', '趙孟', '公都子'
    ]
}

def extract_keywords_from_text(text: str, model, tokenizer, kw_model, use_classical_chinese: bool = True) -> List[Dict]:
    """从单个文本中提取关键词"""
    # 分词处理
    if use_classical_chinese:
        # 使用HanLP进行古文分词
        tokenizer_hanlp = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
        words = tokenizer_hanlp(text)
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
    
    # 计算相似度矩阵（只有在有关键词的情况下）
    if keyword_embeddings:  # 检查是否有关键词
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
    else:
        # 如果没有关键词，返回空列表
        return []
    
    # 在返回结果前移除embedding字段
    for keyword_info in enhanced_keywords:
        if "embedding" in keyword_info:
            del keyword_info["embedding"]
    
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
    model_name = "Jihuai/bert-ancient-chinese"
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
    
    # 获取所有json文件，保持原始顺序
    # all_files = [f for f in os.listdir(file_dir) if f.endswith(".json")]
    # 只处理《论语》《孟子》《商君书》《孙子兵法》
    all_files = ["n007.json", "n009.json", "n016.json", "n019.json"]
    
    # 从第start_index个文件开始处理
    start_index = 0
    files_to_process = all_files[start_index:]
    
    for file_name in tqdm.tqdm(all_files, desc=f"Processing files (starting from {start_index+1})"):
        file_path = os.path.join(file_dir, file_name)
        results = process_chapter_keywords(file_path, use_classical_chinese=True)
        output_path = os.path.join(output_dir, f"{file_name}_chapter_keywords.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"已处理第 {start_index + 1 + files_to_process.index(file_name)} 个文件，章节关键词已保存到: {output_path}")
