import pandas as pd
import networkx as nx
import re
import time
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# -------------------------- 1. 配置与数据读取 --------------------------
DATA_PATH = r"refined_moltbook.xlsx"
NETWORK_PLOT_PATH = r"agent_network.png"
ANALYSIS_RESULT_PATH = r"agent_network_analysis.xlsx"


def load_data(path):
    """读取Excel，支持多sheet，带耗时统计"""
    print("📂 开始读取数据...")
    t0 = time.time()
    try:
        excel_sheets = pd.read_excel(path, sheet_name=None, engine='openpyxl')
        df_list = [sheet for sheet in excel_sheets.values()]
        df = pd.concat(df_list, ignore_index=True)
        print(f"✅ 成功读取数据，共 {len(df)} 条记录，{len(df.columns)} 个字段")
        print(f"📋 数据字段：{df.columns.tolist()}")
        print(f"⏱️ 读取耗时：{time.time() - t0:.2f} 秒")
        return df
    except Exception as e:
        print(f"❌ 读取数据失败：{str(e)}")
        return None


df = load_data(DATA_PATH)
if df is None:
    exit()

# 如果数据量过大，给出提示
if len(df) > 10000:
    print(f"⚠️ 数据量较大（{len(df)} 行），处理可能较慢，请耐心等待...")

# -------------------------- 2. 数据预处理：生成Agent数字ID --------------------------
print("\n🏷️ 正在生成Agent数字ID...")
t0 = time.time()


def generate_agent_id(df):
    # 尝试识别Agent唯一标识字段
    agent_id_candidates = [
        'agent_id', 'agent_unique_id', 'id', 'agent_name', 'username',
        'description', 'post_author', 'author_id'
    ]
    agent_col = None
    for col in agent_id_candidates:
        if col in df.columns and df[col].nunique() > 1:
            agent_col = col
            break

    if agent_col is None:
        print("⚠️ 未识别到现成Agent ID字段，基于文本特征生成唯一ID...")
        text_col = None
        for col in ['combined_text', 'content', 'post_content', 'text']:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            text_col = df.columns[0]
        df['agent_unique_id'] = df.apply(
            lambda x: f"Agent_{x.name}_{str(x[text_col])[:30].replace(' ', '_')[:20]}",
            axis=1
        )
        agent_col = 'agent_unique_id'

    df['agent_num_id'] = pd.factorize(df[agent_col])[0]
    print(f"✅ 生成Agent数字ID，共 {df['agent_num_id'].nunique()} 个唯一Agent")
    return df, agent_col, 'agent_num_id'


df, agent_name_col, agent_num_col = generate_agent_id(df)
print(f"⏱️ 生成ID耗时：{time.time() - t0:.2f} 秒")

# -------------------------- 3. 构建Agent映射表（支持别名） --------------------------
print("\n🔗 构建Agent别名映射表...")
t0 = time.time()
agent_attrs = defaultdict(dict)
for idx, row in df.iterrows():
    aid = row[agent_num_col]
    if 'name' in df.columns and pd.notna(row['name']):
        agent_attrs[aid]['name'] = str(row['name'])
    if 'x_name' in df.columns and pd.notna(row['x_name']):
        agent_attrs[aid]['x_name'] = str(row['x_name'])
    if 'x_handle' in df.columns and pd.notna(row['x_handle']):
        agent_attrs[aid]['x_handle'] = str(row['x_handle'])
    if aid not in agent_attrs:
        agent_attrs[aid]['name'] = str(row[agent_name_col])

# 构建映射：可能的名称 -> agent_num_id 列表
agent_name_to_id = defaultdict(list)
for aid, attrs in agent_attrs.items():
    if 'name' in attrs and attrs['name']:
        agent_name_to_id[attrs['name'].lower().strip()].append(aid)
    if 'x_name' in attrs and attrs['x_name']:
        agent_name_to_id[attrs['x_name'].lower().strip()].append(aid)
    if 'x_handle' in attrs and attrs['x_handle']:
        agent_name_to_id[attrs['x_handle'].lower().strip()].append(aid)

print(f"✅ 构建映射表，共 {len(agent_name_to_id)} 个唯一键（含别名）")
print(f"⏱️ 构建映射耗时：{time.time() - t0:.2f} 秒")

# -------------------------- 4. 提取互动关系 --------------------------
print("\n🔍 提取@提及等互动关系...")
t0 = time.time()
# 识别文本列
text_cols = []
for col in df.columns:
    if any(key in col.lower() for key in ['content', 'text', 'comment', 'post', 'reply']):
        text_cols.append(col)
if not text_cols:
    text_cols = [df.columns[0]]
print(f"📝 将分析以下文本列：{text_cols}")


def extract_interactions(df, text_cols, agent_num_col):
    """从文本列提取@提及，返回 (sender_num_id, receiver_name, type) 列表"""
    interaction_pairs = []
    mention_pattern = re.compile(r'@([\w-]+)')
    total_rows = len(df)
    for idx, row in df.iterrows():
        # 每处理1000行打印一次进度
        if idx % 1000 == 0 and idx > 0:
            print(f"   进度：{idx}/{total_rows} 行")
        sender_id = row[agent_num_col]
        for col in text_cols:
            if pd.isna(row[col]):
                continue
            text = str(row[col]).strip()
            mentions = mention_pattern.findall(text)
            for mention in mentions:
                if mention and mention != '':
                    interaction_pairs.append((sender_id, mention, 'mention'))
            if any(key in text.lower() for key in ['引用', '参考', 'quoted', 'refer']):
                ref_pattern = re.compile(r'(引用|参考|quoted|refer)[：: ]*([^\s,@，。！？]+)')
                references = ref_pattern.findall(text)
                for _, ref in references:
                    if ref and ref != '':
                        interaction_pairs.append((sender_id, ref, 'reference'))
    print(f"✅ 提取到 {len(interaction_pairs)} 条原始互动关系")
    return interaction_pairs


interaction_pairs = extract_interactions(df, text_cols, agent_num_col)
print(f"⏱️ 提取互动耗时：{time.time() - t0:.2f} 秒")

# 匹配接收者的数字ID
print("\n🔁 匹配接收者数字ID...")
t0 = time.time()
valid_interactions = []
unmatched_names = set()

for sender_id, receiver_name, interact_type in interaction_pairs:
    receiver_lower = receiver_name.lower().strip()
    matched_ids = []

    # 1. 精准匹配
    if receiver_lower in agent_name_to_id:
        matched_ids = agent_name_to_id[receiver_lower]
    # 2. 子串匹配（接收者是某个标识的子串）
    if not matched_ids:
        for key, ids in agent_name_to_id.items():
            if receiver_lower in key:
                matched_ids.extend(ids)
                break
    # 3. 反向子串匹配（某个标识是接收者的子串）
    if not matched_ids:
        for key, ids in agent_name_to_id.items():
            if key in receiver_lower:
                matched_ids.extend(ids)
                break

    if matched_ids:
        for receiver_id in set(matched_ids):
            if sender_id != receiver_id:
                valid_interactions.append((sender_id, receiver_id, interact_type))
    else:
        unmatched_names.add(receiver_lower)

if unmatched_names:
    print(f"⚠️ 未能匹配的接收者名称（共 {len(unmatched_names)} 个），前20个：")
    for name in sorted(unmatched_names)[:20]:
        print(f"   - {name}")

# 统计互动次数
interaction_counter = Counter([(s, r, t) for s, r, t in valid_interactions])
print(f"✅ 匹配到 {len(interaction_counter)} 条有效Agent间互动（去重后）")
print(f"⏱️ 匹配耗时：{time.time() - t0:.2f} 秒")

# -------------------------- 5. 构建网络 --------------------------
print("\n🕸️ 构建网络...")
t0 = time.time()
G = nx.DiGraph(name="AI Agent互动网络")
all_agent_ids = df[agent_num_col].unique()
G.add_nodes_from(all_agent_ids)
print(f"✅ 网络节点数：{G.number_of_nodes()}")

edges = []
for (sender, receiver, interact_type), count in interaction_counter.items():
    edges.append((sender, receiver, {'weight': count, 'type': interact_type}))
G.add_edges_from(edges)
print(f"✅ 网络边数：{G.number_of_edges()}")
print(f"⏱️ 构建网络耗时：{time.time() - t0:.2f} 秒")

# -------------------------- 8. 保存分析结果（包含节点表和简单边表） --------------------------
print("\n💾 保存分析结果（CSV格式）...")
t0 = time.time()
try:
    # 节点属性表
    nodes_data = []
    for aid in G.nodes():
        attrs = agent_attrs.get(aid, {})
        nodes_data.append({
            'agent_num_id': aid,
            'name': attrs.get('name', ''),
            'x_name': attrs.get('x_name', ''),
            'x_handle': attrs.get('x_handle', '')
        })
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv('agent_network_nodes.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 节点表已保存：agent_network_nodes.csv")

    # 简单边表（source, target, weight）
    simple_edges_df = pd.DataFrame([
        {'source': u, 'target': v, 'weight': G[u][v]['weight']}
        for u, v in G.edges()
    ])
    simple_edges_df.to_csv('agent_network_edges_simple.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 简单边表已保存：agent_network_edges_simple.csv")

    # 互动边详情（含类型和原始标识）
    edges_df = pd.DataFrame([
        {
            '发起者AgentID': u,
            '接收者AgentID': v,
            '互动次数': G[u][v]['weight'],
            '互动类型': G[u][v]['type'],
            '发起者原始标识': agent_attrs.get(u, {}).get('name', ''),
            '接收者原始标识': agent_attrs.get(v, {}).get('name', '')
        }
        for u, v in G.edges()
    ])
    edges_df.to_csv('agent_network_edges_detail.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 互动边详情已保存：agent_network_edges_detail.csv")

    print(f"⏱️ 保存耗时：{time.time() - t0:.2f} 秒")
except Exception as e:
    print(f"❌ 保存结果失败：{str(e)}")

