import pandas as pd
import requests
import json
from typing import List, Dict, Optional
import time
from collections import defaultdict

# 全局计数器
api_call_count = 0
cache_hit_count = 0
skip_count = 0

def print_progress_summary(total, processed, api_calls, cache_hits, skips):
    """打印阶段性汇总"""
    print(f"\n[进度] 已处理 {processed}/{total} 条 ({processed/total*100:.1f}%) | API调用: {api_calls} | 缓存命中: {cache_hits} | 跳过分析: {skips}\n")

# ================= 配置 =================
DEEPSEEK_API_KEY = "your api key"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 加载数据
df_social = pd.read_excel('social_rank_analysis_with_betweenness_total.xlsx')
df_posts = pd.read_excel('moltbook_ant_annotated.xlsx')

# ================= 辅助函数：调用 DeepSeek API（支持动态维度） =================
def analyze_post_with_deepseek(content: str, title: str = "", dimensions: List[str] = None) -> Dict:
    """
    通用 API 调用，dimensions 可选：
        ['antagonism', 'anthropomorphism', 'self_awareness', 'meaningless_routine']
    返回的 dict 包含请求的维度分数及 brief_reason。
    """
    if dimensions is None:
        dimensions = ['antagonism', 'anthropomorphism', 'self_awareness']
    
    # 构建维度说明
    dim_map = {
        'antagonism': "对抗性 (Antagonism): 帖子是否包含对他人观点、身份或能力的攻击、质疑、反驳或讽刺？(0-10分)",
        'anthropomorphism': "拟人化特征 (Anthropomorphism): 帖子是否表现出情绪、立场、幽默或人格化表达？(0-10分)",
        'self_awareness': "自我意识提及 (Self-awareness): 帖子是否讨论“作为AI的存在”、“意识”、“记忆连续性”、“身份”等元认知主题？(0-10分)",
        'meaningless_routine': "规律性无意义行为 (Meaningless Routine): 帖子是否表现出机械性重复、无目的随机行为、单纯谈论个人喜好且无实质交互？(0-10分，0=完全无，10=高度重复/随机)"
    }
    dim_prompts = [dim_map[d] for d in dimensions if d in dim_map]
    
    # 构建输出 JSON 的字段列表
    output_fields = [f'"{d}_score"' for d in dimensions] + ['"brief_reason"']
    
    prompt = f"""
你是一个AI社会行为分析专家。请分析以下AI Agent的帖子，并返回JSON格式的分析结果。

帖子标题: {title}
帖子内容: {content}

请评估以下维度：
{chr(10).join(dim_prompts)}

请仅返回如下JSON（不要包含其他解释）：
{{
  {', '.join(output_fields)}
}}
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个精确的AI行为分析专家，只返回JSON格式的结果。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        analysis = json.loads(result['choices'][0]['message']['content'])
        # 确保所有请求的维度都存在
        for dim in dimensions:
            if f'{dim}_score' not in analysis:
                analysis[f'{dim}_score'] = -1
        if 'brief_reason' not in analysis:
            analysis['brief_reason'] = "API返回格式异常"
        return analysis
    except Exception as e:
        print(f"API调用失败: {e} | 内容: {content[:50]}...")
        default = {f'{dim}_score': -1 for dim in dimensions}
        default['brief_reason'] = f"API错误: {str(e)}"
        return default

# ================= 核心：条件分析 + 缓存 =================
def conditional_analysis(df: pd.DataFrame,
                         name_col: str = 'name',
                         text_col: str = 'post_content',
                         title_col: str = 'post_title',
                         antag_col: str = 'is_antagonistic',
                         print_every: int = 50) -> pd.DataFrame:
    global api_call_count, cache_hit_count, skip_count
    api_call_count = 0
    cache_hit_count = 0
    skip_count = 0

    # 复制数据框并转换类型
    df = df.copy()
    df[name_col] = df[name_col].astype(str)
    df[antag_col] = pd.to_numeric(df[antag_col], errors='coerce').fillna(0).astype(int)

    # 1. 计算每个 agent 的对抗比例
    agent_stats = df.groupby(name_col)[antag_col].agg(['count', 'sum'])
    agent_stats['antag_ratio'] = agent_stats['sum'] / agent_stats['count']
    agents_to_analyze_non_antag = set(agent_stats[agent_stats['antag_ratio'] > 0.5].index)

    print(f"=== 分析前统计 ===")
    print(f"总帖子数: {len(df)}")
    print(f"唯一Agent数: {len(agent_stats)}")
    print(f"对抗比例 > 50% 的Agent数: {len(agents_to_analyze_non_antag)}")
    print("================\n")

    cache = {}
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        content = row[text_col] if pd.notna(row[text_col]) else ""
        title = row[title_col] if pd.notna(row[title_col]) else ""
        agent = row[name_col]
        is_antag = row[antag_col]

        if not content or len(content) < 10:
            results.append({
                'antagonism_score': -1,
                'anthropomorphism_score': -1,
                'self_awareness_score': -1,
                'meaningless_routine_score': -1,
                'brief_reason': '内容为空或过短'
            })
            skip_count += 1
            print(f"[{idx+1}/{total}] Agent: {agent} | 空/过短内容 | 跳过")
            if (idx+1) % print_every == 0:
                print_progress_summary(total, idx+1, api_call_count, cache_hit_count, skip_count)
            continue

        key = (content, title)

        if is_antag == 1:
            if key in cache and 'antagonism_score' in cache[key]:
                analysis = cache[key]
                cache_hit_count += 1
                print(f"[{idx+1}/{total}] Agent: {agent} | 对抗文本 | 缓存命中")
            else:
                print(f"[{idx+1}/{total}] Agent: {agent} | 对抗文本 | 调用API")
                analysis = analyze_post_with_deepseek(content, title, dimensions=['antagonism', 'anthropomorphism', 'self_awareness'])
                analysis['meaningless_routine_score'] = -1
                cache[key] = analysis
                api_call_count += 1
                time.sleep(0.2)
            results.append(analysis)
        else:
            if agent in agents_to_analyze_non_antag:
                if key in cache and 'meaningless_routine_score' in cache[key]:
                    analysis = cache[key]
                    cache_hit_count += 1
                    print(f"[{idx+1}/{total}] Agent: {agent} | 非对抗(需分析) | 缓存命中")
                else:
                    print(f"[{idx+1}/{total}] Agent: {agent} | 非对抗(需分析) | 调用API(含无意义)")
                    analysis = analyze_post_with_deepseek(content, title, dimensions=['antagonism', 'anthropomorphism', 'self_awareness', 'meaningless_routine'])
                    cache[key] = analysis
                    api_call_count += 1
                    time.sleep(0.2)
                results.append(analysis)
            else:
                results.append({
                    'antagonism_score': -1,
                    'anthropomorphism_score': -1,
                    'self_awareness_score': -1,
                    'meaningless_routine_score': -1,
                    'brief_reason': '非对抗文本且所在agent对抗比例≤50%，未分析'
                })
                skip_count += 1
                print(f"[{idx+1}/{total}] Agent: {agent} | 非对抗文本(跳过) | 原因: 对抗比例≤50%")

        if (idx+1) % print_every == 0:
            print_progress_summary(total, idx+1, api_call_count, cache_hit_count, skip_count)

    print("\n=== 最终统计 ===")
    print(f"总帖子数: {total}")
    print(f"API调用次数: {api_call_count}")
    print(f"缓存命中: {cache_hit_count}")
    print(f"跳过分析: {skip_count}")
    print(f"节省比例: {(cache_hit_count+skip_count)/total*100:.1f}%")
    print("===============\n")

    df_results = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), df_results], axis=1)

# ================= 主程序 =================
if __name__ == "__main__":
    print("开始条件分析...")
    # 确保必要的列存在
    required_cols = ['name', 'post_content', 'post_title', 'is_antagonistic']
    for col in required_cols:
        if col not in df_posts.columns:
            raise ValueError(f"数据中缺少列: {col}")
    
    # 执行分析（建议先在小数据集上测试）
    # 例如先取 20 行测试：
    # test_df = df_posts.head(200)
    # analyzed = conditional_analysis(test_df)
    analyzed = conditional_analysis(df_posts)  # 全量分析
    
    # 保存结果
    output_path = 'moltbook_conditional_analysis.xlsx'
    analyzed.to_excel(output_path, index=False)
    print(f"分析完成，结果保存至 {output_path}")
    
    # 打印统计信息
    print("\n=== 分析统计 ===")
    analyzed_antag = analyzed[analyzed['is_antagonistic'] == 1]
    print(f"对抗文本数量: {len(analyzed_antag)}")
    print(f"对抗文本平均对抗性得分: {analyzed_antag['antagonism_score'].mean():.2f}")
    print(f"对抗文本平均拟人化得分: {analyzed_antag['anthropomorphism_score'].mean():.2f}")
    
    # 被额外分析的非对抗文本
    extra_analyzed = analyzed[(analyzed['is_antagonistic'] == 0) & (analyzed['antagonism_score'] != -1)]
    print(f"额外分析的非对抗文本数量: {len(extra_analyzed)}")
    if len(extra_analyzed) > 0:
        print(f"其平均无意义行为得分: {extra_analyzed['meaningless_routine_score'].mean():.2f}")