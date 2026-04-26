import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from difflib import SequenceMatcher

# 可选依赖（用于高级相似度）
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    USE_EMBEDDINGS = True
except ImportError:
    USE_EMBEDDINGS = False

# ======================== 配置 ========================
MIN_SENTENCE_LEN = 10           # 最小句子长度（字符）
MIN_TITLE_LEN = 6               # 最小标题长度
SIMILARITY_THRESHOLD = 0.90     # 字符串相似度阈值
GLOBAL_FREQ_THRESHOLD = 0.05    # 高频句子过滤比例
N_PROCESSES = 6                 # 并行进程数（根据CPU核心数调整）
# =====================================================

def normalize_text(text):
    """标准化文本：小写、移除标点、合并空格"""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def split_sentences(text):
    """中英文句子分割"""
    sents = re.split(r'[。！？!?.\n]+', text)
    return [s.strip() for s in sents if s.strip()]

def edit_distance_similarity(a, b):
    """使用 difflib 的快速相似度"""
    return SequenceMatcher(None, a, b).ratio()

def is_similar(s1, s2, threshold=SIMILARITY_THRESHOLD):
    """判断两个标准化后的句子是否相似"""
    if s1 == s2:
        return True
    if abs(len(s1) - len(s2)) / max(len(s1), len(s2)) > 0.3:
        return False
    return edit_distance_similarity(s1, s2) >= threshold

def build_inverted_index(agent_posts, global_freq_threshold=GLOBAL_FREQ_THRESHOLD):
    """构建句子倒排索引，过滤高频句"""
    idx = defaultdict(list)
    sentence_counter = Counter()
    total_posts = sum(len(posts) for posts in agent_posts.values())
    print(f"构建倒排索引，总帖子数: {total_posts}")

    for agent, posts in agent_posts.items():
        for post in posts:
            title = post['title']
            if title and len(title) >= MIN_TITLE_LEN:
                norm = normalize_text(title)
                idx[norm].append((agent, post['post_id'], title))
                sentence_counter[norm] += 1

            sentences = split_sentences(post['content'])
            for sent in sentences:
                if len(sent) >= MIN_SENTENCE_LEN:
                    norm = normalize_text(sent)
                    idx[norm].append((agent, post['post_id'], sent))
                    sentence_counter[norm] += 1

    # 过滤高频句子
    high_freq = set()
    if global_freq_threshold > 0:
        threshold_count = total_posts * global_freq_threshold
        for norm, cnt in sentence_counter.items():
            if cnt > threshold_count:
                high_freq.add(norm)
        print(f"过滤高频句子数: {len(high_freq)}")

    for norm in high_freq:
        if norm in idx:
            del idx[norm]

    print(f"倒排索引构建完成，唯一键数: {len(idx)}")
    return idx

def process_posts_chunk(chunk, idx, agent_names):
    """多进程处理一块数据（精确匹配）"""
    explicit_counts = defaultdict(int)
    implicit_counts = defaultdict(int)
    counted_pairs = set()
    mention_pattern = re.compile(r'@([A-Za-z0-9_-]+)')

    for _, row in chunk.iterrows():
        cur_agent = row['name']
        if pd.isna(cur_agent):
            continue
        cur_id = row['post_id']
        cur_content = row['post_content'] if pd.notna(row['post_content']) else ''
        cur_title = row['post_title'] if pd.notna(row['post_title']) else ''

        # 显式 @ 提及
        mentions = mention_pattern.findall(cur_content)
        for m in mentions:
            if m in agent_names:
                explicit_counts[m] += 1

        # 隐式引用
        if not cur_content:
            continue

        all_units = []
        if cur_title and len(cur_title) >= MIN_TITLE_LEN:
            all_units.append(('title', cur_title))
        sentences = split_sentences(cur_content)
        for sent in sentences:
            if len(sent) >= MIN_SENTENCE_LEN:
                all_units.append(('sent', sent))

        for unit_type, text in all_units:
            norm = normalize_text(text)
            if norm in idx:
                candidates = idx[norm]
                for cited_agent, cited_id, original_sent in candidates:
                    if cited_agent == cur_agent:
                        continue
                    pair_key = (cur_id, cited_id)
                    if pair_key in counted_pairs:
                        continue
                    if is_similar(text, original_sent):
                        implicit_counts[cited_agent] += 1
                        counted_pairs.add(pair_key)

    return explicit_counts, implicit_counts

def compute_citations(df, n_processes=N_PROCESSES):
    """主函数：计算引用关系（精确匹配）"""
    df = df.copy()
    df['post_content'] = df['post_content'].fillna('').astype(str)
    df['post_title'] = df['post_title'].fillna('').astype(str)
    df['name'] = df['name'].fillna('').astype(str)

    agent_names = set(df['name'][df['name'] != ''])

    # 构建 agent_posts 字典
    agent_posts = defaultdict(list)
    for _, row in df.iterrows():
        name = row['name']
        if name:
            agent_posts[name].append({
                'post_id': row['post_id'],
                'title': row['post_title'],
                'content': row['post_content']
            })

    idx = build_inverted_index(agent_posts)

    # 分块处理
    chunks = np.array_split(df, n_processes)
    with mp.Pool(processes=n_processes) as pool:
        func = partial(process_posts_chunk, idx=idx, agent_names=agent_names)
        results = pool.map(func, chunks)

    total_explicit = defaultdict(int)
    total_implicit = defaultdict(int)
    for expl, impl in results:
        for k, v in expl.items():
            total_explicit[k] += v
        for k, v in impl.items():
            total_implicit[k] += v

    all_agents = set(agent_names)
    final_stats = []
    for agent in all_agents:
        expl = total_explicit.get(agent, 0)
        impl = total_implicit.get(agent, 0)
        final_stats.append({
            'agent': agent,
            'explicit_mentions': expl,
            'implicit_citations': impl,
            'total_citations': expl + impl
        })

    result_df = pd.DataFrame(final_stats)
    result_df.sort_values('total_citations', ascending=False, inplace=True)
    return result_df

def build_embedding_index(agent_posts, model_name='all-MiniLM-L6-v2'):
    """构建句子嵌入 FAISS 索引，返回 (index, meta, model)"""
    if not USE_EMBEDDINGS:
        return None, None, None

    model = SentenceTransformer(model_name)
    all_sentences = []
    meta = []  # (agent, post_id, original_sent)
    for agent, posts in agent_posts.items():
        for post in posts:
            title = post['title']
            if title and len(title) >= MIN_TITLE_LEN:
                all_sentences.append(title)
                meta.append((agent, post['post_id'], title))
            for sent in split_sentences(post['content']):
                if len(sent) >= MIN_SENTENCE_LEN:
                    all_sentences.append(sent)
                    meta.append((agent, post['post_id'], sent))

    if not all_sentences:
        print("没有符合条件的句子，跳过嵌入索引构建")
        return None, None, None

    print(f"编码 {len(all_sentences)} 个句子...")
    embeddings = model.encode(all_sentences, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"FAISS 索引构建完成，向量维度 {dim}，数量 {index.ntotal}")
    return index, meta, model

def process_with_embeddings(df, index, meta, model, agent_names, threshold=0.85, batch_size=5000):
    explicit_counts = defaultdict(int)
    implicit_counts = defaultdict(int)
    counted_pairs = set()
    mention_pattern = re.compile(r'@([A-Za-z0-9_-]+)')

    # 分批次处理
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start+batch_size]
        for _, row in chunk.iterrows():
            cur_agent = row['name']
            if pd.isna(cur_agent):
                continue
            cur_id = row['post_id']
            cur_content = row['post_content'] if pd.notna(row['post_content']) else ''
            cur_title = row['post_title'] if pd.notna(row['post_title']) else ''

            # 显式 @ 提及
            mentions = mention_pattern.findall(cur_content)
            for m in mentions:
                if m in agent_names:
                    explicit_counts[m] += 1

            if not cur_content:
                continue

            # 提取当前帖子的所有单元
            units = []
            if cur_title and len(cur_title) >= MIN_TITLE_LEN:
                units.append(cur_title)
            units.extend([s for s in split_sentences(cur_content) if len(s) >= MIN_SENTENCE_LEN])

            if not units:
                continue

            # 对每个单元查询（可批量优化）
            for unit in units:
                emb = model.encode([unit], convert_to_numpy=True)
                faiss.normalize_L2(emb)
                scores, indices = index.search(emb, k=5)  # 返回前5相似
                for score, idx in zip(scores[0], indices[0]):
                    if score < threshold:
                        continue
                    cited_agent, cited_id, _ = meta[idx]
                    if cited_agent == cur_agent:
                        continue
                    pair_key = (cur_id, cited_id)
                    if pair_key in counted_pairs:
                        continue
                    implicit_counts[cited_agent] += 1
                    counted_pairs.add(pair_key)

    return explicit_counts, implicit_counts

# ======================== 主程序 ========================
if __name__ == '__main__':
    # 读取数据
    file_path = 'refined_moltbook.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # 1. 基础统计（显式+隐式精确匹配）
    print("正在运行基础引用统计...")
    result_basic = compute_citations(df, n_processes=N_PROCESSES)
    result_basic.to_csv('citation_basic.csv', index=False, encoding='utf-8')
    print("基础引用统计完成，结果已保存至 citation_basic.csv")
    print(result_basic.head(20))

    # 2. 可选：嵌入相似度（改写引用）
    if USE_EMBEDDINGS:
        print("\n正在构建嵌入索引...")
        agent_posts = defaultdict(list)
        for _, row in df.iterrows():
            name = row['name']
            if name:
                # 确保标题和内容为字符串
                title = str(row['post_title']) if pd.notna(row['post_title']) else ''
                content = str(row['post_content']) if pd.notna(row['post_content']) else ''
                agent_posts[name].append({
                    'post_id': row['post_id'],
                    'title': title,
                    'content': content
                })
        index, meta, model = build_embedding_index(agent_posts)
        if index is not None:
            # 单进程处理所有帖子（不再使用多进程）
            total_explicit, total_implicit = process_with_embeddings(
                df, index, meta, model, set(df['name'])
            )
            # 合并结果
            all_agents = set(df['name'].dropna())
            final_stats = []
            for agent in all_agents:
                expl = total_explicit.get(agent, 0)
                impl = total_implicit.get(agent, 0)
                final_stats.append({
                    'agent': agent,
                    'explicit_mentions': expl,
                    'implicit_citations': impl,
                    'total_citations': expl + impl
                })
            result_embed = pd.DataFrame(final_stats)
            result_embed.sort_values('total_citations', ascending=False, inplace=True)
            result_embed.to_csv('citation_embedding.csv', index=False, encoding='utf-8')
            print("\n嵌入相似度引用统计完成，结果已保存至 citation_embedding.csv")
            print(result_embed.head(20))
        else:
            print("嵌入索引构建失败，跳过嵌入匹配。")
