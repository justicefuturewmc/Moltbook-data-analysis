import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ====================== 1. 数据加载与预处理 ======================
file_path = 'social_rank_analysis_with_betweenness_total.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

numeric_cols = [
    'degree_centrality', 'weighted_degree', 'pagerank', 'eigen_centrality', 'betweenness',
    'karma', 'x_follower_count', 'post_upvotes', 'karma_norm', 'follower_norm',
    'upvote_norm', 'attr_score', 'degree_norm', 'weighted_degree_norm', 'pagerank_norm',
    'eigen_norm', 'betweenness_norm', 'network_score', 'social_rank_score'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['level'] = pd.Categorical(df['level'], categories=['低', '中低', '中高', '高'], ordered=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ====================== 2. 箱线图（标注前三高前三低） ======================
def plot_level_comparison(df, indicator, save_path=None, annotate_extremes=True):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='level', y=indicator, data=df, palette='viridis')
    plt.title(f'不同社交等级在 {indicator} 指标上的分布对比')
    plt.xlabel('社交等级')
    plt.ylabel(indicator)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if annotate_extremes:
        # 对每个等级标注前3高和前3低
        for i, level in enumerate(df['level'].cat.categories):
            level_data = df[df['level'] == level][indicator].dropna()
            if len(level_data) < 3:
                continue
            # 取前三高和前三低（值可能重复，取索引唯一的）
            top3 = level_data.nlargest(3).index
            bottom3 = level_data.nsmallest(3).index
            # 合并要标注的索引
            idx_to_annotate = set(top3).union(set(bottom3))
            for idx in idx_to_annotate:
                row = df.loc[idx]
                y_val = row[indicator]
                # 获取该点在图中的x坐标（箱线图位置为i）
                x_pos = i
                ax.annotate(row['name'], (x_pos, y_val), xytext=(5, 5),
                            textcoords='offset points', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


# 生成箱线图（默认开启极值标注）
plot_level_comparison(df, 'network_score', 'boxplot_network_score.png', annotate_extremes=True)
plot_level_comparison(df, 'attr_score', 'boxplot_attr_score.png', annotate_extremes=True)
plot_level_comparison(df, 'pagerank_norm', 'boxplot_pagerank.png', annotate_extremes=True)
plot_level_comparison(df, 'karma_norm', 'boxplot_karma.png', annotate_extremes=True)


# ====================== 3. 雷达图（保持不变） ======================
def plot_level_radar(df, level_name, indicators, save_path=None):
    level_data = df[df['level'] == level_name]
    if level_data.empty:
        print(f"等级 {level_name} 没有数据。")
        return

    avg_values = [level_data[ind].mean() for ind in indicators]

    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    avg_values += avg_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, avg_values, 'o-', linewidth=2, label=level_name)
    ax.fill(angles, avg_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators)
    ax.set_ylim(0, 1)
    ax.set_title(f'{level_name} 等级核心指标雷达图')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


core_indicators = ['degree_norm', 'pagerank_norm', 'eigen_norm', 'betweenness_norm', 'karma_norm', 'follower_norm']
for level in ['高', '中高', '中低', '低']:
    plot_level_radar(df, level, core_indicators, f'radar_{level}.png')


def plot_combined_radar(df, levels, indicators, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = {'高': 'red', '中高': 'orange', '中低': 'green', '低': 'blue'}
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]
    for level in levels:
        level_data = df[df['level'] == level]
        if level_data.empty:
            continue
        avg_values = [level_data[ind].mean() for ind in indicators]
        avg_values += avg_values[:1]
        ax.plot(angles, avg_values, 'o-', linewidth=2, label=level, color=colors.get(level, 'black'))
        ax.fill(angles, avg_values, alpha=0.1, color=colors.get(level, 'black'))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators)
    ax.set_ylim(0, 1)
    ax.set_title('高、中高、中低、低等级核心指标雷达图对比')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


plot_combined_radar(df, ['高', '中高', '中低', '低'], core_indicators, 'radar_combined.png')


# ====================== 4. 散点图（活动vs影响力，标注前三高前三低） ======================
def plot_activity_vs_influence(df, x_col='weighted_degree_norm', y_col='follower_norm',
                               save_path=None, annotate_extremes=True):
    plt.figure(figsize=(12, 8))
    palette = {'低': 'blue', '中低': 'green', '中高': 'orange', '高': 'red'}
    for level, color in palette.items():
        level_data = df[df['level'] == level]
        plt.scatter(level_data[x_col], level_data[y_col], c=color, label=level, alpha=0.6, edgecolors='w', s=50)

    if annotate_extremes:
        # 找出X轴前3高和前3低
        x_top3 = df.nlargest(3, x_col)[['name', x_col, y_col]]
        x_bottom3 = df.nsmallest(3, x_col)[['name', x_col, y_col]]
        # Y轴前3高和前3低
        y_top3 = df.nlargest(3, y_col)[['name', x_col, y_col]]
        y_bottom3 = df.nsmallest(3, y_col)[['name', x_col, y_col]]
        # 合并所有要标注的点（去重）
        all_points = pd.concat([x_top3, x_bottom3, y_top3, y_bottom3]).drop_duplicates(subset='name')

        for _, row in all_points.iterrows():
            plt.annotate(row['name'], (row[x_col], row[y_col]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.title(f'网络活跃度 ({x_col}) vs. 外部影响力 ({y_col})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='社交等级')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


# 生成散点图，自动标注全局前三高前三低
plot_activity_vs_influence(df, 'weighted_degree_norm', 'follower_norm',
                           'scatter_activity_influence.png', annotate_extremes=True)
plot_activity_vs_influence(df, 'weighted_degree_norm', 'upvote_norm',
                           'scatter_activity_upvote.png', annotate_extremes=True)


# ====================== 5. 桥梁行为散点图（标注前三高前三低） ======================
def plot_bridge_behavior(df, x_col='degree_norm', y_col='betweenness_norm',
                         save_path=None, annotate_extremes=True):
    plt.figure(figsize=(12, 8))
    palette = {'低': 'blue', '中低': 'green', '中高': 'orange', '高': 'red'}
    for level, color in palette.items():
        level_data = df[df['level'] == level]
        plt.scatter(level_data[x_col], level_data[y_col], c=color, label=level, alpha=0.6, edgecolors='w', s=50)

    if annotate_extremes:
        x_top3 = df.nlargest(3, x_col)[['name', x_col, y_col]]
        x_bottom3 = df.nsmallest(3, x_col)[['name', x_col, y_col]]
        y_top3 = df.nlargest(3, y_col)[['name', x_col, y_col]]
        y_bottom3 = df.nsmallest(3, y_col)[['name', x_col, y_col]]
        all_points = pd.concat([x_top3, x_bottom3, y_top3, y_bottom3]).drop_duplicates(subset='name')

        for _, row in all_points.iterrows():
            plt.annotate(row['name'], (row[x_col], row[y_col]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.title(f'连接数量 ({x_col}) vs. 桥梁作用 ({y_col})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='社交等级')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


plot_bridge_behavior(df, 'degree_norm', 'betweenness_norm',
                     'scatter_bridge.png', annotate_extremes=True)


# ====================== 6. 相关性热图（不变） ======================
def plot_correlation_heatmap(df, attr_cols, net_cols, save_path=None):
    corr_data = df[attr_cols + net_cols].corr(method='spearman')
    corr_matrix = corr_data.loc[attr_cols, net_cols]

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('属性指标与网络中心性指标相关性热图 (Spearman)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


attr_cols = ['karma_norm', 'follower_norm', 'upvote_norm']
net_cols = ['degree_norm', 'weighted_degree_norm', 'pagerank_norm', 'eigen_norm', 'betweenness_norm']
plot_correlation_heatmap(df, attr_cols, net_cols, 'heatmap_correlation.png')


# ====================== 7. 回归分析特征重要性图（不变） ======================
def perform_regression_analysis(df, target='social_rank_score', features=None, save_path=None):
    if features is None:
        features = ['karma_norm', 'follower_norm', 'upvote_norm', 'degree_norm',
                    'pagerank_norm', 'betweenness_norm']

    X = df[features].dropna()
    y = df.loc[X.index, target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    coefficients = model.coef_
    importance = np.abs(coefficients)

    plt.figure(figsize=(10, 6))
    features_importance = pd.Series(importance, index=features).sort_values(ascending=True)
    features_importance.plot(kind='barh', color='skyblue')
    plt.title(f'影响 {target} 的特征重要性（标准化系数绝对值）')
    plt.xlabel('标准化系数绝对值')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

    print(f"\n===== 回归分析结果 (目标: {target}) =====")
    print(f"模型 R² 分数: {model.score(X_scaled, y):.4f}")
    print("特征标准化系数:")
    for i, feat in enumerate(features):
        print(f"  {feat}: {coefficients[i]:.4f}")


perform_regression_analysis(df, 'social_rank_score', save_path='regression_importance.png')
perform_regression_analysis(df, 'network_score', save_path='regression_importance_network.png')

print("\n所有图表已生成并保存到当前目录。")