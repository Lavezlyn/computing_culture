import json
import os
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import hanlp
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import time

class KeywordFrequencyAnalyzer:
    def __init__(self, chapter_keyword_dir: str, target_docs_dir: str):
        print("初始化分析器...")
        self.chapter_keyword_dir = chapter_keyword_dir
        self.target_docs_dir = target_docs_dir
        
        # 检查GPU是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        print("加载BERT模型...")
        self.model_name = "Jihuai/bert-ancient-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        # 将模型移动到GPU
        self.model = self.model.to(self.device)
        print("加载HanLP分词器...")
        self.hanlp_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        print("初始化完成！")
        
    def load_chapter_keywords(self) -> Dict:
        """Load all chapter keyword files"""
        print("\n加载章节关键词文件...")
        chapter_keywords = {}
        files = [f for f in os.listdir(self.chapter_keyword_dir) if f.endswith("_chapter_keywords.json")]
        for file_name in tqdm(files, desc="加载关键词文件"):
            with open(os.path.join(self.chapter_keyword_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                chapter_keywords[file_name] = data
        print(f"成功加载 {len(chapter_keywords)} 个关键词文件")
        return chapter_keywords
    
    def load_target_documents(self) -> Dict:
        """Load all target documents to analyze"""
        print("\n加载目标文档...")
        target_docs = {}
        files = [f for f in os.listdir(self.target_docs_dir) if f.endswith(".json")]
        for file_name in tqdm(files, desc="加载目标文档"):
            with open(os.path.join(self.target_docs_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                target_docs[file_name] = data
        print(f"成功加载 {len(target_docs)} 个目标文档")
        return target_docs
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get BERT embedding for a word"""
        inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # 将输入数据移动到GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 将结果移回CPU并转换为numpy数组
        return outputs.last_hidden_state[0][0].cpu().numpy()
    
    def calculate_semantic_similarity(self, word1: str, word2: str) -> float:
        """Calculate semantic similarity between two words using BERT embeddings"""
        emb1 = self.get_word_embedding(word1)
        emb2 = self.get_word_embedding(word2)
        return float(cosine_similarity([emb1], [emb2])[0][0])
    
    def batch_calculate_similarities(self, keywords: List[str], words: List[str], batch_size: int = 256) -> Dict[str, List[Tuple[float, str]]]:
        """批量计算语义相似度，返回所有相似度大于阈值的匹配词及其相似度"""
        similarities = {}
        
        # 预先计算所有目标词的嵌入向量
        word_inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True, max_length=512)
        word_inputs = {k: v.to(self.device) for k, v in word_inputs.items()}
        with torch.no_grad():
            word_outputs = self.model(**word_inputs)
        word_embeddings = word_outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 将关键词分批处理
        for i in range(0, len(keywords), batch_size):
            batch_keywords = keywords[i:i + batch_size]
            
            # 批量获取关键词的嵌入向量
            keyword_inputs = self.tokenizer(batch_keywords, return_tensors="pt", padding=True, truncation=True, max_length=512)
            keyword_inputs = {k: v.to(self.device) for k, v in keyword_inputs.items()}
            with torch.no_grad():
                keyword_outputs = self.model(**keyword_inputs)
            keyword_embeddings = keyword_outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(keyword_embeddings, word_embeddings)
            
            # 对每个关键词，找到所有相似度大于阈值的匹配词
            for idx, keyword in enumerate(batch_keywords):
                # 获取所有相似度大于阈值的匹配
                matches = []
                for word_idx, similarity in enumerate(similarity_matrix[idx]):
                    if similarity > 0.7:  # 阈值
                        matches.append((float(similarity), words[word_idx]))
                
                if matches:
                    # 按相似度降序排序
                    matches.sort(key=lambda x: x[0], reverse=True)
                    similarities[keyword] = matches
            
            # 清理当前批次的GPU内存
            del keyword_inputs, keyword_outputs
            torch.cuda.empty_cache()
        
        # 清理所有GPU内存
        del word_inputs, word_outputs
        torch.cuda.empty_cache()
        
        return similarities
    
    def save_intermediate_results(self, results: Dict, chapter_file: str, doc_file: str, output_dir: str):
        """保存中间处理结果"""
        # 创建中间结果目录
        intermediate_dir = os.path.join(output_dir, "intermediate_results")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # 保存当前文档的结果
        result_file = os.path.join(intermediate_dir, f"{chapter_file}_{doc_file}_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results[chapter_file][doc_file], f, ensure_ascii=False, indent=4)
        
        # 保存当前进度信息
        progress_file = os.path.join(intermediate_dir, "progress.json")
        progress_info = {
            "last_processed": {
                "chapter_file": chapter_file,
                "doc_file": doc_file,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_info, f, ensure_ascii=False, indent=4)
        
        print(f"  已保存中间结果到: {result_file}")

    def load_intermediate_results(self, output_dir: str) -> Tuple[Dict, str, str]:
        """加载中间处理结果"""
        intermediate_dir = os.path.join(output_dir, "intermediate_results")
        if not os.path.exists(intermediate_dir):
            return {}, "", ""
        
        # 加载进度信息
        progress_file = os.path.join(intermediate_dir, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_info = json.load(f)
                last_chapter = progress_info["last_processed"]["chapter_file"]
                last_doc = progress_info["last_processed"]["doc_file"]
                print(f"找到上次处理进度: {last_chapter} - {last_doc}")
        else:
            return {}, "", ""
        
        # 加载已处理的结果
        results = {}
        for file_name in os.listdir(intermediate_dir):
            if file_name.endswith("_results.json"):
                parts = file_name.split("_results.json")[0].split("_")
                chapter_file = "_".join(parts[:-1])
                doc_file = parts[-1]
                
                if chapter_file not in results:
                    results[chapter_file] = {}
                
                with open(os.path.join(intermediate_dir, file_name), 'r', encoding='utf-8') as f:
                    results[chapter_file][doc_file] = json.load(f)
        
        return results, last_chapter, last_doc

    def analyze_keyword_frequencies(self) -> Dict:
        """Analyze keyword frequencies across target documents"""
        print("\n开始分析关键词频率...")
        chapter_keywords = self.load_chapter_keywords()
        target_docs = self.load_target_documents()
        
        # 创建输出目录
        output_dir = "./fa_influence_ru_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # 尝试加载中间结果
        results, last_chapter, last_doc = self.load_intermediate_results(output_dir)
        
        # 计算需要处理的总章节数（关键词文件数 × 目标文档数 × 每个文档的章节数）
        total_chapters = 0
        for doc_file, doc_data in target_docs.items():
            doc_chapters = sum(len(chapters) for chapters in doc_data.values())
            total_chapters += doc_chapters * len(chapter_keywords)
        
        # 计算已处理的章节数（包括中间结果）
        processed_chapters = 0
        for chapter_file, doc_results in results.items():
            for doc_file, metrics in doc_results.items():
                # 计算每个文档中已处理的章节数
                processed_chapters += len(set(
                    chapter for match_type in ['direct_matches', 'semantic_matches', 'tfidf_scores']
                    for keyword_data in metrics[match_type].values()
                    for chapter in keyword_data.get('chapters', [])
                ))
        
        # 创建总体进度条
        with tqdm(total=len(chapter_keywords), desc="处理关键词文件") as file_pbar:
            for chapter_file, book_data in chapter_keywords.items():
                # 如果已经处理过这个文件，跳过
                if chapter_file in results and all(doc in results[chapter_file] for doc in target_docs):
                    print(f"\n跳过已处理的文件: {chapter_file}")
                    file_pbar.update(1)
                    continue
                
                print(f"\n处理文件: {chapter_file}")
                if chapter_file not in results:
                    results[chapter_file] = {}
                
                # Extract all keywords with their metadata
                all_keywords = {}
                for book, chapters in book_data.items():
                    for chapter, data in chapters.items():
                        for keyword_info in data["keywords"]:
                            keyword = keyword_info["keyword"]
                            if keyword not in all_keywords:
                                all_keywords[keyword] = {
                                    "score": keyword_info["score"],
                                    "pos": keyword_info["pos"],
                                    "similarities": keyword_info["similarities"],
                                    "chapters": []
                                }
                            all_keywords[keyword]["chapters"].append(f"{book}_{chapter}")
                
                # Analyze each target document
                with tqdm(total=len(target_docs), desc="分析目标文档", leave=False) as doc_pbar:
                    for doc_file, doc_data in target_docs.items():
                        # 如果已经处理过这个文档，跳过
                        if doc_file in results[chapter_file]:
                            print(f"  跳过已处理的文档: {doc_file}")
                            doc_pbar.update(1)
                            continue
                        
                        print(f"  分析文档: {doc_file}")
                        doc_results = {
                            "direct_matches": defaultdict(lambda: {
                                "count": 0,
                                "chapters": [],
                                "score": 0.0,
                                "pos": "",
                                "similarities": {}
                            }),
                            "semantic_matches": defaultdict(lambda: {
                                "matched_word": "",
                                "similarity": 0.0,
                                "chapters": [],
                                "score": 0.0,
                                "pos": "",
                                "original_similarities": {}
                            }),
                            "tfidf_scores": defaultdict(lambda: {
                                "score": 0.0,
                                "chapters": [],
                                "original_score": 0.0,
                                "pos": "",
                                "similarities": {}
                            })
                        }
                        
                        # Process each chapter in the target document
                        with tqdm(total=sum(len(chapters) for chapters in doc_data.values()), 
                                desc="  处理章节", leave=False) as chapter_pbar:
                            for book, chapters in doc_data.items():
                                for chapter, texts in chapters.items():
                                    processed_chapters += 1
                                    
                                    # Combine all texts in the chapter
                                    combined_text = ' '.join(texts)
                                    
                                    # Tokenize the text
                                    words = self.hanlp_tokenizer(combined_text)
                                    
                                    # Calculate direct matches
                                    for word in words:
                                        if word in all_keywords:
                                            doc_results["direct_matches"][word]["count"] += 1
                                            doc_results["direct_matches"][word]["chapters"].append(f"{book}_{chapter}")
                                            doc_results["direct_matches"][word]["score"] = all_keywords[word]["score"]
                                            doc_results["direct_matches"][word]["pos"] = all_keywords[word]["pos"]
                                            doc_results["direct_matches"][word]["similarities"] = all_keywords[word]["similarities"]
                                    
                                    # 批量计算语义相似度
                                    semantic_similarities = self.batch_calculate_similarities(
                                        list(all_keywords.keys()),
                                        words
                                    )
                                    
                                    # 更新语义匹配结果
                                    for keyword, matches in semantic_similarities.items():
                                        if matches:  # 如果有关键词匹配
                                            # 只保存相似度最高的匹配
                                            best_match = matches[0]  # matches已经按相似度降序排序
                                            doc_results["semantic_matches"][keyword]["matched_word"] = best_match[1]  # 保存匹配词
                                            doc_results["semantic_matches"][keyword]["similarity"] = best_match[0]  # 保存相似度
                                            
                                            # 记录所有相似度超过阈值的词的出现章节
                                            for similarity, word in matches:
                                                if similarity > 0.7:  # 阈值
                                                    doc_results["semantic_matches"][keyword]["chapters"].append(f"{book}_{chapter}")
                                            
                                            # 去重章节列表
                                            doc_results["semantic_matches"][keyword]["chapters"] = list(set(doc_results["semantic_matches"][keyword]["chapters"]))
                                            
                                            doc_results["semantic_matches"][keyword]["score"] = all_keywords[keyword]["score"]
                                            doc_results["semantic_matches"][keyword]["pos"] = all_keywords[keyword]["pos"]
                                            doc_results["semantic_matches"][keyword]["original_similarities"] = all_keywords[keyword]["similarities"]
                                    
                                    # Calculate TF-IDF scores
                                    vectorizer = TfidfVectorizer()
                                    tfidf_matrix = vectorizer.fit_transform([combined_text])
                                    feature_names = vectorizer.get_feature_names_out()
                                    
                                    for keyword, keyword_info in all_keywords.items():
                                        if keyword in feature_names:
                                            idx = list(feature_names).index(keyword)
                                            doc_results["tfidf_scores"][keyword]["score"] = float(tfidf_matrix[0, idx])
                                            doc_results["tfidf_scores"][keyword]["chapters"].append(f"{book}_{chapter}")
                                            doc_results["tfidf_scores"][keyword]["original_score"] = keyword_info["score"]
                                            doc_results["tfidf_scores"][keyword]["pos"] = keyword_info["pos"]
                                            doc_results["tfidf_scores"][keyword]["similarities"] = keyword_info["similarities"]
                                    
                                    chapter_pbar.update(1)
                                    chapter_pbar.set_postfix({
                                        "当前章节": f"{book}_{chapter}",
                                        "文档进度": f"{chapter_pbar.n}/{chapter_pbar.total}",
                                        "总进度": f"{processed_chapters}/{total_chapters}"
                                    })
                        
                        results[chapter_file][doc_file] = doc_results
                        
                        # 保存中间结果
                        self.save_intermediate_results(results, chapter_file, doc_file, output_dir)
                        
                        doc_pbar.update(1)
                        doc_pbar.set_postfix({
                            "已处理文档": f"{list(target_docs.keys()).index(doc_file) + 1}/{len(target_docs)}",
                            "当前文件": chapter_file,
                            "文档进度": f"{doc_pbar.n}/{doc_pbar.total}"
                        })
                
                file_pbar.update(1)
                file_pbar.set_postfix({
                    "已处理文件": f"{list(chapter_keywords.keys()).index(chapter_file) + 1}/{len(chapter_keywords)}",
                    "总进度": f"{processed_chapters}/{total_chapters} 章节"
                })
        
        print("\n关键词频率分析完成！")
        return results
    
    def generate_statistics(self, results: Dict) -> pd.DataFrame:
        """Generate summary statistics from the analysis results"""
        print("\n生成统计信息...")
        stats = []
        
        for chapter_file, doc_results in tqdm(results.items(), desc="生成统计信息"):
            for doc_file, metrics in doc_results.items():
                # Calculate summary statistics
                direct_matches = sum(m["count"] for m in metrics["direct_matches"].values())
                semantic_matches = sum(1 for m in metrics["semantic_matches"].values() if m["matched_word"])  # 只统计有匹配的情况
                avg_tfidf = np.mean([m["score"] for m in metrics["tfidf_scores"].values()]) if metrics["tfidf_scores"] else 0
                
                # Calculate average scores
                avg_direct_score = np.mean([m["score"] for m in metrics["direct_matches"].values()]) if metrics["direct_matches"] else 0
                avg_semantic_score = np.mean([m["similarity"] for m in metrics["semantic_matches"].values() if m["matched_word"]]) if any(m["matched_word"] for m in metrics["semantic_matches"].values()) else 0
                
                stats.append({
                    "Chapter_File": chapter_file,
                    "Target_Doc": doc_file,
                    "Direct_Matches": direct_matches,
                    "Semantic_Matches": semantic_matches,
                    "Avg_TFIDF": avg_tfidf,
                    "Avg_Direct_Score": avg_direct_score,
                    "Avg_Semantic_Score": avg_semantic_score
                })
        
        return pd.DataFrame(stats)
    
    def visualize_results(self, stats_df: pd.DataFrame, output_dir: str):
        """Generate visualizations of the analysis results"""
        print("\n生成可视化结果...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create heatmap of direct matches
        print("生成直接匹配热力图...")
        plt.figure(figsize=(12, 8))
        pivot_direct = stats_df.pivot(index="Chapter_File", columns="Target_Doc", values="Direct_Matches")
        sns.heatmap(pivot_direct, annot=True, cmap="YlOrRd")
        plt.title("Direct Keyword Matches Across Documents")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "direct_matches_heatmap.png"))
        plt.close()
        
        # Create heatmap of semantic matches
        print("生成语义匹配热力图...")
        plt.figure(figsize=(12, 8))
        pivot_semantic = stats_df.pivot(index="Chapter_File", columns="Target_Doc", values="Semantic_Matches")
        sns.heatmap(pivot_semantic, annot=True, cmap="YlOrRd")
        plt.title("Semantic Keyword Matches Across Documents")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "semantic_matches_heatmap.png"))
        plt.close()
        
        # Create heatmap of TF-IDF scores
        print("生成TF-IDF分数热力图...")
        plt.figure(figsize=(12, 8))
        pivot_tfidf = stats_df.pivot(index="Chapter_File", columns="Target_Doc", values="Avg_TFIDF")
        sns.heatmap(pivot_tfidf, annot=True, cmap="YlOrRd")
        plt.title("Average TF-IDF Scores Across Documents")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tfidf_scores_heatmap.png"))
        plt.close()
        
        # Create heatmap of average scores
        print("生成平均分数热力图...")
        plt.figure(figsize=(12, 8))
        pivot_scores = stats_df.pivot(index="Chapter_File", columns="Target_Doc", values="Avg_Direct_Score")
        sns.heatmap(pivot_scores, annot=True, cmap="YlOrRd")
        plt.title("Average Keyword Scores Across Documents")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "average_scores_heatmap.png"))
        plt.close()
        
        print("可视化结果生成完成！")

def main():
    start_time = time.time()
    print("="*50)
    print("开始关键词频率分析")
    print("="*50)
    
    # Initialize analyzer
    analyzer = KeywordFrequencyAnalyzer(
        chapter_keyword_dir="./chapter_keyword",
        target_docs_dir="./target"
    )
    
    # Run analysis
    results = analyzer.analyze_keyword_frequencies()
    
    # Generate statistics
    stats_df = analyzer.generate_statistics(results)
    
    # Save results
    output_dir = "./keyword_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n保存分析结果...")
    # Save detailed results
    with open(os.path.join(output_dir, "detailed_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Save statistics
    stats_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
    
    # Generate visualizations
    analyzer.visualize_results(stats_df, output_dir)
    
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "="*50)
    print(f"分析完成！总耗时: {duration:.2f} 秒")
    print(f"结果已保存到: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main() 