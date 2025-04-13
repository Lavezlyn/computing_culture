import json
import networkx as nx
from pyvis.network import Network
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

def load_keywords(file_path: str) -> Dict:
    """加载关键词JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_network(keywords_data: Dict, similarity_threshold: float = 0.5) -> nx.Graph:
    """构建关键词语义关系网络"""
    G = nx.Graph()
    
    # 添加节点
    for keyword_info in keywords_data['keywords']:
        keyword = keyword_info['keyword']
        score = keyword_info['score']
        pos = keyword_info['pos']
        
        # 节点属性
        node_attrs = {
            'score': score,
            'pos': pos,
            'size': score * 20  # 根据重要性得分调整节点大小
        }
        
        # 根据词性设置不同的颜色
        if pos:
            if pos.startswith('n'):  # 名词
                node_attrs['color'] = '#FF9999'
            elif pos.startswith('v'):  # 动词
                node_attrs['color'] = '#99FF99'
            elif pos.startswith('a'):  # 形容词
                node_attrs['color'] = '#9999FF'
            else:
                node_attrs['color'] = '#CCCCCC'
        else:
            node_attrs['color'] = '#CCCCCC'
        
        G.add_node(keyword, **node_attrs)
    
    # 添加边
    for i, keyword_info in enumerate(keywords_data['keywords']):
        keyword1 = keyword_info['keyword']
        similarities = keyword_info['similarities']
        
        for keyword2, similarity in similarities.items():
            if similarity >= similarity_threshold:
                G.add_edge(
                    keyword1,
                    keyword2,
                    weight=similarity,
                    title=f"相似度: {similarity:.3f}"
                )
    
    return G

def create_interactive_network(G: nx.Graph, output_path: str):
    """创建交互式网络可视化"""
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # 添加节点
    for node in G.nodes(data=True):
        net.add_node(
            node[0],
            label=node[0],
            size=node[1]['size'],
            color=node[1]['color'],
            title=f"词性: {node[1]['pos']}<br>重要性: {node[1]['score']:.3f}"
        )
    
    # 添加边
    for edge in G.edges(data=True):
        net.add_edge(
            edge[0],
            edge[1],
            value=edge[2]['weight'],
            title=edge[2]['title']
        )
    
    # 保存为HTML文件
    net.save_graph(output_path)

def analyze_network(G: nx.Graph) -> Dict:
    """分析网络特征"""
    analysis = {
        "基本统计": {
            "节点数": G.number_of_nodes(),
            "边数": G.number_of_edges(),
            "平均度": np.mean([d for n, d in G.degree()]),
            "平均聚类系数": nx.average_clustering(G)
        },
        "中心性分析": {
            "度中心性": dict(nx.degree_centrality(G)),
            "介数中心性": dict(nx.betweenness_centrality(G)),
            "特征向量中心性": dict(nx.eigenvector_centrality(G, max_iter=1000))
        },
        "社区检测": {
            "社区": list(nx.community.louvain_communities(G))
        }
    }
    
    # 按词性统计
    pos_stats = defaultdict(int)
    for node in G.nodes(data=True):
        if node[1]['pos']:
            pos_stats[node[1]['pos']] += 1
    analysis["词性统计"] = dict(pos_stats)
    
    return analysis

def main():
    # 设置输入输出路径
    input_file = "./chapter_keyword/n007.json_chapter_keywords.json"
    output_html = "./keyword/n007_network.html"
    output_analysis = "./keyword/n007_network_analysis.json"
    
    # 加载关键词数据
    keywords_data = load_keywords(input_file)
    
    # 构建网络
    G = build_network(keywords_data, similarity_threshold=0.5)
    
    # 创建交互式可视化
    create_interactive_network(G, output_html)
    
    # 分析网络特征
    analysis = analyze_network(G)
    
    # 保存分析结果
    with open(output_analysis, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=4)
    
    print(f"网络可视化已保存到: {output_html}")
    print(f"网络分析结果已保存到: {output_analysis}")

if __name__ == "__main__":
    main() 