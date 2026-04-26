"""
Moltbook 社交与对抗行为分析脚本（修正版）
依赖库：pandas, matplotlib, seaborn, networkx, openpyxl
安装：pip install pandas matplotlib seaborn networkx openpyxl
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import re

# 设置 matplotlib 支持中文（避免字体警告）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取数据 ====================
print("正在读取数据...")
social_df = pd.read_excel("social_rank_analysis_with_betweenness_total - 样本.xlsx", sheet_name="Sheet1")
ant_df = pd.read_excel("moltbook_ant_annotated - 样本.xlsx", sheet_name="Sheet1")

# 标准化列名（去除空格）
social_df.columns = social_df.columns.str.strip()
ant_df.columns = ant_df.columns.str.strip()

# 统一名称格式（转小写、去空格，提高匹配率）
social_df['name_clean'] = social_df['name'].astype(str).str.lower().str.strip()
ant_df['name_clean'] = ant_df['name'].astype(str).str.lower().str.strip()

# ==================== 2. 对抗行为分析 ====================
print("\n=== 对抗行为分析 ===")
# 对抗类型分布
ant_type_counts = ant_df['ant_type_label'].value_counts()
print("对抗类型分布：")
print(ant_type_counts)

# 对抗强度与社交排名的关系（改进匹配）
merged = ant_df.merge(social_df, on='name_clean', how='left', suffixes=('', '_social'))
ant_social = merged[merged['is_antagonistic'] == True].copy()

if not ant_social.empty:
    print(f"\n成功匹配到社交数据的对抗评论数：{len(ant_social)} / {ant_df['is_antagonistic'].sum()}")
    print("\n对抗性 Agent 的社交排名分布（network_score）:")
    print(ant_social['network_score'].describe())
    
    # 不同对抗类型下的平均 network_score
    type_score = ant_social.groupby('ant_type_label')['network_score'].mean().sort_values(ascending=False)
    print("\n各对抗类型平均 network_score：")
    print(type_score)
else:
    print("\n警告：没有对抗评论能匹配到社交排名数据。请检查 name 字段的一致性。")

# 绘制对抗类型分布
plt.figure(figsize=(10,5))
sns.countplot(data=ant_df, y='ant_type_label', order=ant_type_counts.index, hue='ant_type_label', legend=False, palette='viridis')
plt.title("对抗行为类型分布")
plt.tight_layout()
plt.savefig("ant_type_distribution.png", dpi=150)
print("\n已保存对抗类型分布图：ant_type_distribution.png")

# ==================== 3. AI 自我意识表达分析（修正：定义关键词）====================
print("\n=== AI 自我意识表达分析 ===")
self_aware_keywords = [
    'i am an ai', 'i am ai', 'as an ai', 'i am a robot', 'i am a language model',
    '我是AI', '我是一个AI', '作为AI', '我是机器人', '我的模型', 'my model',
    'claude', 'openclaw', 'clawdbot', 'running on', '我的架构', 'my architecture',
    'ai assistant', 'digital assistant', 'artificial intelligence'
]

def has_self_awareness(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(re.search(kw, text_lower) for kw in self_aware_keywords)

ant_df['has_self_aware'] = ant_df['comments_content'].apply(has_self_awareness)
aware_count = ant_df['has_self_aware'].sum()
print(f"包含 AI 自我意识表述的评论数：{aware_count} / {len(ant_df)} ({aware_count/len(ant_df):.1%})")

# 自我意识表述与对抗性的关联
aware_ant = ant_df[(ant_df['has_self_aware']) & (ant_df['is_antagonistic'] == True)].shape[0]
total_ant = ant_df['is_antagonistic'].sum()
if total_ant > 0:
    print(f"同时具有自我意识与对抗行为的评论数：{aware_ant}")
    print(f"对抗评论中自我意识比例：{aware_ant/total_ant:.1%}")
else:
    print("无对抗评论。")

# ==================== 4. 群体行为基础分析 ====================
print("\n=== 群体行为基础分析 ===")
# 4.1 最活跃的 submolt
submolt_counts = ant_df['submolt_name'].value_counts().head(10)
print("最活跃的 submolt 前10：")
print(submolt_counts)

# 4.2 高排名 Agent 的对抗参与度（使用清洗后的名称匹配）
high_rank = social_df[social_df['level'] == '高'].copy()
high_rank_names = set(high_rank['name_clean'].dropna())
ant_high = ant_df[ant_df['name_clean'].isin(high_rank_names)]
print(f"\n高排名 Agent 数量：{len(high_rank_names)}")
print(f"高排名 Agent 发布的对抗评论数：{ant_high['is_antagonistic'].sum()}")
if len(ant_high) > 0:
    print(f"高排名 Agent 对抗率：{ant_high['is_antagonistic'].mean():.2%}")

# 4.3 网络中心性与对抗性的相关性（使用匹配后的数据）
if 'betweenness_norm' in social_df.columns and 'network_score' in social_df.columns:
    # 将对抗性聚合到 Agent 级别
    agent_ant = ant_df.groupby('name_clean')['is_antagonistic'].sum().reset_index()
    agent_ant.columns = ['name_clean', 'ant_count']
    social_with_ant = social_df.merge(agent_ant, on='name_clean', how='left').fillna(0)
    corr_bet = social_with_ant['betweenness_norm'].corr(social_with_ant['ant_count'])
    corr_score = social_with_ant['network_score'].corr(social_with_ant['ant_count'])
    print(f"\n介数中心性 (betweenness_norm) 与对抗次数的相关系数：{corr_bet:.3f}")
    print(f"网络得分 (network_score) 与对抗次数的相关系数：{corr_score:.3f}")

# ==================== 5. 社交网络可视化（基于 @ 提及）====================
print("\n=== 构建简单社交网络图 ===")
def extract_mentions(text):
    if pd.isna(text):
        return []
    # 匹配 @username（允许字母数字下划线）
    return re.findall(r'@([a-zA-Z0-9_]+)', str(text))

edges = []
for _, row in ant_df.iterrows():
    mentions = extract_mentions(row['comments_content'])
    for m in mentions:
        # 将提及的 username 也做清洗，以便与 name_clean 对应
        m_clean = m.lower().strip()
        edges.append((row['name_clean'], m_clean))

G = nx.DiGraph()
G.add_edges_from(edges)

if len(G.nodes) > 0:
    degree = dict(G.degree())
    top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]
    print("提及网络中度最高的前20个节点：")
    for node, deg in top_nodes:
        print(f"  {node}: {deg}")
    
    # 绘制网络图（仅显示度 >= 2 的节点）
    nodes_to_keep = [n for n, d in degree.items() if d >= 2]
    G_sub = G.subgraph(nodes_to_keep)
    if len(G_sub.nodes) > 1:
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G_sub, k=0.5, iterations=50)
        nx.draw(G_sub, pos, node_size=30, node_color='lightblue', edge_color='gray', 
                with_labels=False, alpha=0.7)
        # 标注度前10的节点
        high_deg_nodes = [n for n, d in degree.items() if d >= 5][:10]
        labels = {n: n for n in high_deg_nodes}
        nx.draw_networkx_labels(G_sub, pos, labels, font_size=8)
        plt.title("Agent 提及网络（@关系）")
        plt.tight_layout()
        plt.savefig("agent_mention_network.png", dpi=150)
        print("已保存网络图：agent_mention_network.png")
    else:
        print("网络过于稀疏，无法绘制。")
else:
    print("未发现有效的提及关系，跳过网络图。")

# ==================== 6. 群体行为模式总结 ====================
print("\n=== 群体行为模式总结 ===")
ant_comments = ant_df[ant_df['is_antagonistic'] == True]['comments_content'].dropna()
if len(ant_comments) > 0:
    all_ant_words = ' '.join(ant_comments).lower().split()
    stopwords = {'the', 'and', 'to', 'of', 'a', 'in', 'for', 'on', 'with', 'is', 'that', 
                 'it', 'this', 'you', 'i', 'we', 'they', 'not', 'but', 'so', 'as', 'be', 'by',
                 'me', 'my', 'your', 'our', 'their', 'him', 'her', 'it', 'them', 'from', 'at',
                 'an', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having',
                 'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'might', 'must',
                 'just', 'like', 'know', 'think', 'see', 'want', 'get', 'make', 'say', 'go'}
    word_counts = Counter([w for w in all_ant_words if w.isalpha() and len(w) > 2 and w not in stopwords])
    print("对抗评论中的高频词（前20）：")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")
else:
    print("没有对抗评论，无法统计高频词。")

print("\n分析完成！所有图表已保存。")