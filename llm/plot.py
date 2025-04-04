"""
plot.py - 大规模关键词数据分析与可视化脚本
功能包含：
1. 高效读取和处理190,000+关键词数据
2. 多维度统计分析
3. 自动生成可视化图表(PDF格式)
4. 人文意义特征提取
5. 结果保存为JSON和PDF
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages

def load_data(filepath):
    """高效加载大规模JSON数据"""
    print("正在加载数据...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {sum(len(entry['keywords']) for entry in data)} 条关键词数据")
    return data

def process_data(data):
    """内存优化处理"""
    print("正在处理数据...")
    
    # 使用更高效的数据类型
    dtypes = {
        'word': 'category',
        'type': 'category',
        'importance': 'string'
    }
    
    # 分块处理大数据
    chunks = (pd.DataFrame(entry['keywords']) for entry in data)
    df = pd.concat(chunks, ignore_index=True).astype(dtypes)
    
    # 优化内存使用
    df['word_length'] = df['word'].str.len().astype('uint8')
    df['importance_length'] = df['importance'].str.len().astype('uint16')
    
    return df

def analyze_data(df):
    """执行多维度分析（修复类型转换问题）"""
    print("正在执行分析...")
    
    # 转换所有数值为Python原生类型
    type_dist = df['type'].value_counts().astype(int).to_dict()
    top50_words = {k: int(v) for k, v in df['word'].value_counts().head(50).items()}
    
    analysis = {
        # 基础统计（显式类型转换）
        "total_keywords": int(len(df)),
        "unique_words": int(df['word'].nunique()),
        "type_distribution": type_dist,
        
        # 词频分析
        "top50_words": top50_words,
        "long_tail_analysis": {
            "appear_once": int((df['word'].value_counts() == 1).sum()),
            "appear_twice": int((df['word'].value_counts() == 2).sum())
        },
        
        # 文本特征分析（添加float转换）
        "word_length_stats": {
            "mean": float(df['word_length'].mean()),
            "max": int(df['word_length'].max()),
            "min": int(df['word_length'].min())
        },
        
        # 关联分析（索引转为字符串）
        "type_word_relationship": {
            str(k): str(v) for k, v in 
            df.groupby('type')['word'].agg(lambda x: x.value_counts().index[0]).items()
        }
    }
    
    # 人文特征提取（确保Counter使用原生类型）
    analysis["humanistic_insights"] = {
        "dominant_domain": str(max(type_dist, key=type_dist.get)),
        "most_versatile_word": str(df.groupby('word')['type'].nunique().idxmax()),
        "common_importance_terms": [
            (str(term[0]), int(term[1])) 
            for term in extract_importance_terms(df['importance'])
        ]
    }
    
    return analysis

def extract_importance_terms(text_series, top_n=20):
    """从重要性描述中提取关键术语（返回原生类型）"""
    words = []
    for text in text_series:
        words.extend([str(word.strip("，。！？")) for word in text.split() if len(word) > 1])
    return Counter(words).most_common(top_n)

def visualize_results(df, analysis, output_pdf="analysis_report.pdf"):
    """生成可视化图表并保存为PDF"""
    print("生成可视化图表...")
    
    with PdfPages(output_pdf) as pdf:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        # 类型分布图
        plt.figure(figsize=(10,6))
        df['type'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title("关键词类型分布")
        pdf.savefig()
        plt.close()
        
        # 高频词柱状图（前20）
        plt.figure(figsize=(12,6))
        df['word'].value_counts().head(20).plot.bar()
        plt.title("Top 20高频词分布")
        plt.xticks(rotation=45)
        pdf.savefig()
        plt.close()
        
        # 词云生成
        plt.figure(figsize=(15,10))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(analysis['top50_words'])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("高频词词云")
        pdf.savefig()
        plt.close()
        
        # 文本长度分布
        plt.figure(figsize=(10,6))
        df['word_length'].hist(bins=30)
        plt.title("关键词长度分布")
        plt.xlabel("字符数量")
        plt.ylabel("出现频次")
        pdf.savefig()
        plt.close()

def main(input_file="merged_keywords.json", output_json="statistics.json"):
    # 添加内存优化参数
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('compute.use_numexpr', True)
    
    # 数据加载和处理
    raw_data = load_data(input_file)
    df = process_data(raw_data)
    
    # 执行分析
    analysis_results = analyze_data(df)
    
    # 保存分析结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    # 生成可视化
    visualize_results(df, analysis_results)
    
    print(f"分析完成！结果已保存至 {output_json} 和 analysis_report.pdf")

if __name__ == "__main__":
    main()