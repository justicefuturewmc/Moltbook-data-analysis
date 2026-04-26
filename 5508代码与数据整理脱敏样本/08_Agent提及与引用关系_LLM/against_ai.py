import pandas as pd
import asyncio
import aiohttp
import json
import time
import os
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm

# ================= 配置 =================
API_KEY = "your api key"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
MAX_CONCURRENT = 15  # 并发请求数
MAX_RETRIES = 3  # 重试次数
TIMEOUT = 60  # 单个请求超时秒数
CACHE_FILE = "analysis_cache.json"  # 缓存文件
SCREENING_THRESHOLD = 20  # 跳过内容长度小于此值的帖子（可选）

# ========================================

def load_data(file_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """加载数据，仅保留必要列"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 确保有 post_id 列，如果没有则用行索引
    if 'post_id' not in df.columns:
        df['post_id'] = df.index.astype(str)
    # 确保 post_content 为字符串
    df['post_content'] = df['post_content'].fillna('').astype(str)
    return df

def should_skip(post_content: str) -> bool:
    """快速判断是否值得调用大模型（可选）"""
    # 如果内容太短，很可能没有引用
    if len(post_content) < SCREENING_THRESHOLD:
        return True
    # 可选：如果完全没有 '@' 且不含常见引用词（如“楼上”、“像某位”等），也可跳过
    # 但为避免漏掉隐含引用，暂时不跳过，仅作为成本优化选项
    return False

def truncate_text(text: str, max_length: int = 3000) -> str:
    """截断文本到指定长度，保留完整句子（按句号截断）"""
    if len(text) <= max_length:
        return text
    # 在最大长度附近寻找最后一个句号、问号或感叹号
    last_punct = max(text.rfind('.', 0, max_length),
                     text.rfind('?', 0, max_length),
                     text.rfind('!', 0, max_length))
    if last_punct > max_length * 0.8:
        return text[:last_punct + 1]
    else:
        return text[:max_length]

def build_prompt(post_text: str, all_agents: List[str]) -> str:
    """构建 prompt"""
    # 为避免 token 过多，可将 agent 列表截断（若列表过长，取前 200 个）
    agents_str = ", ".join(all_agents[:200])
    return f"""
You are an AI analyst analyzing interactions between agents. Given the following post, identify:
1. Which agents from the list are explicitly or implicitly referenced.
2. Whether the post is confrontational (disagreement, criticism, challenge, etc.) towards the referenced agent(s).

Agent list: {agents_str}

Post text:
\"\"\"{post_text}\"\"\"

Return a JSON with fields:
- "mentioned_agents": list of agent names from the list that are referenced.
- "confrontational": boolean.

Example: {{"mentioned_agents": ["AgentA"], "confrontational": false}}
"""

async def analyze_post(session: aiohttp.ClientSession,
                       post_id: str,
                       post_text: str,
                       all_agents: List[str],
                       semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """单个帖子的分析，带缓存、重试、速率控制"""
    async with semaphore:
        # 检查缓存
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                try:
                    cache = json.load(f)
                except:
                    cache = {}
        else:
            cache = {}

        if post_id in cache:
            return cache[post_id]

        # 快速跳过
        if should_skip(post_text):
            result = {"mentioned_agents": [], "confrontational": False}
            cache[post_id] = result
            # 异步保存缓存（可选，可定期批量保存）
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            return result

        # 截断长文本
        truncated = truncate_text(post_text, max_length=3000)
        prompt = build_prompt(truncated, all_agents)

        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(API_URL, headers=headers, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data['choices'][0]['message']['content']
                        result = json.loads(content)
                        # 确保字段存在
                        result.setdefault("mentioned_agents", [])
                        result.setdefault("confrontational", False)
                        # 过滤掉不在 all_agents 中的名称（模型可能输出不在列表中的名称）
                        result["mentioned_agents"] = [a for a in result["mentioned_agents"] if a in all_agents]
                        # 写入缓存
                        cache[post_id] = result
                        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                            json.dump(cache, f, ensure_ascii=False, indent=2)
                        return result
                    else:
                        # 处理限流等错误
                        error_text = await resp.text()
                        print(f"Attempt {attempt + 1} for {post_id} failed: {resp.status} - {error_text}")
                        await asyncio.sleep(2 ** attempt)  # 指数退避
            except asyncio.TimeoutError:
                print(f"Timeout for {post_id} on attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"Unexpected error for {post_id}: {e}")
                await asyncio.sleep(2 ** attempt)

        # 所有重试失败，返回空
        result = {"mentioned_agents": [], "confrontational": False}
        cache[post_id] = result
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        return result

async def process_batch(df: pd.DataFrame, all_agents: List[str]) -> List[Dict[str, Any]]:
    """批量处理所有帖子"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in df.iterrows():
            post_id = row['post_id']
            post_text = row['post_content']
            tasks.append(analyze_post(session, post_id, post_text, all_agents, semaphore))

        # 使用 tqdm 显示进度
        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Analyzing posts"):
            result = await coro
            results.append(result)
        return results

def main():
    # 1. 加载数据
    df = load_data("refined_moltbook2.xlsx")

    # 2. 获取所有 agent 名称（去重）
    all_agents = df['name'].dropna().unique().tolist()

    # 3. 异步分析
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(df, all_agents))

    # 4. 将结果合并回 DataFrame
    df['mentioned_agents'] = [r['mentioned_agents'] for r in results]
    df['confrontational'] = [r['confrontational'] for r in results]

    # 5. 构建引用关系表
    relations = []
    for _, row in df.iterrows():
        if not row['mentioned_agents']:
            continue
        for target in row['mentioned_agents']:
            relations.append({
                'source_agent': row['name'],
                'target_agent': target,
                'post_id': row['post_id'],
                'post_content': row['post_content'][:200],
                'confrontational': row['confrontational']
            })

    relations_df = pd.DataFrame(relations)

    # 6. 保存结果
    relations_df.to_csv("agent_references_llm.csv", index=False, encoding='utf-8-sig')
    print(f"分析完成。共 {len(relations_df)} 条引用关系，其中对抗性 {len(relations_df[relations_df['confrontational']])} 条。")

if __name__ == "__main__":
    main()