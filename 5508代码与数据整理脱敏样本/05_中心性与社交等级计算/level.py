import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果 SimHei 无效，可尝试 ['Microsoft YaHei'] 或 ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 1. 加载网络数据
# ------------------------------
df_edges = pd.read_excel('network_advance.xlsx', sheet_name='Sheet1')
G = nx.from_pandas_edgelist(df_edges, 'source', 'target', edge_attr='weight', create_using=nx.DiGraph())

# ------------------------------
# 2. 计算网络中心性指标（包含介数中心性）
# ------------------------------
print("计算网络中心性...")

# 度中心性
degree_centrality = nx.degree_centrality(G)

# 加权度中心性（入度+出度，有向图分别计算再求和）
in_degree_weighted = dict(G.in_degree(weight='weight'))
out_degree_weighted = dict(G.out_degree(weight='weight'))
weighted_degree = {node: in_degree_weighted.get(node,0) + out_degree_weighted.get(node,0) for node in G.nodes()}

# PageRank
pagerank = nx.pagerank(G, alpha=0.85, weight='weight')

# 特征向量中心性
try:
    eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
except nx.PowerIterationFailedConvergence:
    eigen_centrality = {node: 0 for node in G.nodes()}

# 介数中心性
print("正在计算介数中心性，这可能需要一些时间...")
betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)

# 构建中心性DataFrame
df_centrality = pd.DataFrame({
    'node': list(G.nodes()),
    'degree_centrality': [degree_centrality.get(n,0) for n in G.nodes()],
    'weighted_degree': [weighted_degree.get(n,0) for n in G.nodes()],
    'pagerank': [pagerank.get(n,0) for n in G.nodes()],
    'eigen_centrality': [eigen_centrality.get(n,0) for n in G.nodes()],
    'betweenness': [betweenness.get(n,0) for n in G.nodes()],
})

# ------------------------------
# 3. 加载节点属性数据（修复类型错误）
# ------------------------------
df_attrs = pd.read_excel('refined_moltbook.xlsx', sheet_name='Sheet1')

# 确保'name'列为字符串类型，并填充缺失值
df_attrs['name'] = df_attrs['name'].astype(str).fillna('unknown')

# 确保数值列为数值类型，非数值转为0
numeric_cols = ['karma', 'x_follower_count', 'post_upvotes']
for col in numeric_cols:
    df_attrs[col] = pd.to_numeric(df_attrs[col], errors='coerce').fillna(0)

# 按 agent 名称聚合属性（一个agent可能有多个post，取平均）
df_agent_attrs = df_attrs.groupby('name')[numeric_cols].mean().reset_index()

# 属性归一化
scaler = MinMaxScaler()
df_agent_attrs[['karma_norm', 'follower_norm', 'upvote_norm']] = scaler.fit_transform(
    df_agent_attrs[numeric_cols]
)

# 计算属性综合得分（可调整权重）
df_agent_attrs['attr_score'] = (
    0.4 * df_agent_attrs['karma_norm'] +
    0.3 * df_agent_attrs['follower_norm'] +
    0.3 * df_agent_attrs['upvote_norm']
)

# ------------------------------
# 4. 合并网络中心性与属性数据
# ------------------------------
# 确保节点名称与属性名称类型一致（统一为字符串）
df_centrality['node'] = df_centrality['node'].astype(str)
df_merged = pd.merge(df_centrality, df_agent_attrs, left_on='node', right_on='name', how='left')
df_merged.fillna(0, inplace=True)

# 对网络指标归一化（包括介数中心性）
network_metrics = ['degree_centrality', 'weighted_degree', 'pagerank', 'eigen_centrality', 'betweenness']
scaler2 = MinMaxScaler()
df_merged[['degree_norm', 'weighted_degree_norm', 'pagerank_norm', 'eigen_norm', 'betweenness_norm']] = scaler2.fit_transform(
    df_merged[network_metrics]
)

# 计算网络综合得分（调整权重，可根据需求修改）
df_merged['network_score'] = (
    0.15 * df_merged['degree_norm'] +
    0.25 * df_merged['weighted_degree_norm'] +
    0.25 * df_merged['pagerank_norm'] +
    0.15 * df_merged['eigen_norm'] +
    0.20 * df_merged['betweenness_norm']
)

# 最终社交等级得分（网络50% + 属性50%）
df_merged['social_rank_score'] = 0.5 * df_merged['network_score'] + 0.5 * df_merged['attr_score']

# 排序
df_ranked = df_merged.sort_values('social_rank_score', ascending=False).reset_index(drop=True)

# 等级分层（四分位）
df_ranked['level'] = pd.qcut(df_ranked['social_rank_score'], 4, labels=['低', '中低', '中高', '高'])

# 显示前20名
print("社交等级排名前20的节点：")
print(df_ranked[['node', 'social_rank_score', 'level', 'karma', 'x_follower_count', 'betweenness']].head(20))

# ------------------------------
# 5. 可视化：等级分布 + Top节点得分
# ------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
sns.countplot(data=df_ranked, x='level', order=['低', '中低', '中高', '高'], palette='viridis')
plt.title('社交等级分布')

plt.subplot(1,2,2)
top10 = df_ranked.head(10)
sns.barplot(data=top10, y='node', x='social_rank_score', palette='rocket')
plt.title('Top10 社交等级得分')

plt.tight_layout()
plt.show()

# ------------------------------
# 6. 输出结果到Excel
# ------------------------------
df_ranked.to_excel('social_rank_analysis_with_betweenness_total.xlsx', index=False)
print("\n分析结果已保存至 social_rank_analysis_with_betweenness_total.xlsx")