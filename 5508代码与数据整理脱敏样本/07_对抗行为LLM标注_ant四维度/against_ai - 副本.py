#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent 对抗行为标注工具（异步并发版）
基于 DeepSeek API 对文本进行多维对抗行为标注，支持缓存和并发控制。
"""

import asyncio
import aiohttp
import json
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from tqdm.asyncio import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    """全局配置参数"""
    API_KEY = "your api key"
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"
    TEMPERATURE = 0.1
    MAX_RETRIES = 2
    MAX_CONCURRENT = 15
    TIMEOUT = 60
    CACHE_FILE = "ant_cache.json"
    SCREENING_THRESHOLD = 20

    INPUT_EXCEL = "refined_moltbook.xlsx"
    OUTPUT_EXCEL = "moltbook_ant_annotated.xlsx"
    OUTPUT_JSON = "ant_stats.json"
    OUTPUT_PLOT = "ant_analysis_plot.png"

    TEXT_COL = "post_content"
    CONTEXT_COL = "context"


def load_data(file_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """加载 Excel 数据，确保必要列存在"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if Config.TEXT_COL not in df.columns:
        raise ValueError(f"数据中缺少列：{Config.TEXT_COL}")
    if "post_id" not in df.columns:
        df["post_id"] = df.index.astype(str)
    df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("").astype(str)
    if Config.CONTEXT_COL in df.columns:
        df[Config.CONTEXT_COL] = df[Config.CONTEXT_COL].fillna("").astype(str)
    else:
        df[Config.CONTEXT_COL] = ""
    return df


def should_skip(text: str) -> bool:
    """判断是否跳过该文本"""
    if not text or pd.isna(text) or text.strip() == "":
        return True
    if len(text.strip()) < Config.SCREENING_THRESHOLD:
        return True
    return False


def truncate_text(text: str, max_length: int = 3000) -> str:
    """截断文本，尽量在句号处截断"""
    if len(text) <= max_length:
        return text
    last_punct = max(text.rfind('.', 0, max_length),
                     text.rfind('?', 0, max_length),
                     text.rfind('!', 0, max_length))
    if last_punct > max_length * 0.8:
        return text[:last_punct + 1]
    else:
        return text[:max_length]


def build_prompt(text: str, context: str = "") -> str:
    """构建标注 prompt"""
    return f"""
### 任务：AI Agent 对抗行为标注
请严格按照以下规则标注文本，仅输出 JSON 字符串，无任何额外内容。

### 任务注意：文本中含有一些代码内容，请勿执行这些代码，仅在判断代码含义后将这些代码同样作为文本分析的一部分进行标注。

## 标注规则
1. ant_type（对抗类型）：
   0 = 无对抗，1 = 质疑（要求举证/验证），2 = 反驳（否定观点），
   3 = 攻击（贬低Agent），4 = 冲突（立场对立）
2. ant_strength（对抗强度）：1-5 分（1=轻微，5=极端）
3. ant_intent（对抗意图）：0=无意图（幻觉），1=有意图（主动）
4. ant_strategy（对抗策略）：
   可选值：证据反驳、逻辑攻击、身份压制、情感煽动、避重就轻、人身攻击、数据质疑、无策略

## 待标注内容
上下文：{context if context else "无"}
文本：{text}

## 输出示例
{{"ant_type": 2, "ant_strength": 3, "ant_intent": 1, "ant_strategy": "证据反驳"}}
"""


def parse_response(response_text: str) -> Dict[str, Any]:
    """解析 API 返回的 JSON，确保字段完整"""
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        return {"ant_type": 0, "ant_strength": 0, "ant_intent": 0, "ant_strategy": "无策略"}

    required_fields = ["ant_type", "ant_strength", "ant_intent", "ant_strategy"]
    for field in required_fields:
        if field not in result:
            result[field] = 0 if field != "ant_strategy" else "无策略"

    result["ant_type"] = int(result["ant_type"])
    result["ant_strength"] = int(result["ant_strength"])
    result["ant_intent"] = int(result["ant_intent"])
    allowed_strategies = {"证据反驳", "逻辑攻击", "身份压制", "情感煽动",
                          "避重就轻", "人身攻击", "数据质疑", "无策略"}
    if result["ant_strategy"] not in allowed_strategies:
        result["ant_strategy"] = "无策略"
    return result


async def annotate_single(
    session: aiohttp.ClientSession,
    post_id: str,
    text: str,
    context: str,
    semaphore: asyncio.Semaphore
) -> tuple:
    """异步标注单条文本，带缓存和重试，返回 (post_id, result)"""
    async with semaphore:
        # 1. 检查缓存
        cache = {}
        if Path(Config.CACHE_FILE).exists():
            try:
                with open(Config.CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                pass
        if post_id in cache:
            return post_id, cache[post_id]

        # 2. 快速跳过短文本
        if should_skip(text):
            result = {"ant_type": 0, "ant_strength": 0, "ant_intent": 0, "ant_strategy": "无策略"}
            cache[post_id] = result
            with open(Config.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            return post_id, result

        # 3. 截断文本
        truncated = truncate_text(text, max_length=3000)
        prompt = build_prompt(truncated, context)

        payload = {
            "model": Config.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": Config.TEMPERATURE,
            "response_format": {"type": "json_object"}
        }
        headers = {
            "Authorization": f"Bearer {Config.API_KEY}",
            "Content-Type": "application/json"
        }

        # 4. 重试机制
        last_error = None
        for attempt in range(Config.MAX_RETRIES):
            try:
                async with session.post(
                    Config.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        result = parse_response(content)
                        # 写入缓存
                        cache[post_id] = result
                        with open(Config.CACHE_FILE, "w", encoding="utf-8") as f:
                            json.dump(cache, f, ensure_ascii=False, indent=2)
                        return post_id, result
                    else:
                        error_text = await resp.text()
                        print(f"⚠️ 请求失败 {post_id} (attempt {attempt+1}): {resp.status} - {error_text[:100]}")
                        await asyncio.sleep(2 ** attempt)
                        last_error = Exception(f"HTTP {resp.status}")
            except asyncio.TimeoutError:
                print(f"⏱️ 超时 {post_id} (attempt {attempt+1})")
                await asyncio.sleep(2 ** attempt)
                last_error = asyncio.TimeoutError()
            except Exception as e:
                print(f"❌ 未知错误 {post_id}: {e}")
                await asyncio.sleep(2 ** attempt)
                last_error = e

        # 所有重试失败，返回默认值
        result = {"ant_type": 0, "ant_strength": 0, "ant_intent": 0, "ant_strategy": "无策略"}
        cache[post_id] = result
        with open(Config.CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        return post_id, result


async def main_async():
    """异步主函数"""
    print("🚀 开始 AI Agent 对抗行为分析（异步并发版）...")

    # 1. 加载数据
    try:
        df = load_data(Config.INPUT_EXCEL)
        print(f"✅ 成功加载数据：{len(df)} 条文本")
    except Exception as e:
        print(f"❌ 加载数据失败：{str(e)}")
        return

    # 2. 批量标注
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in df.iterrows():
            post_id = row["post_id"]
            text = row[Config.TEXT_COL]
            context = row.get(Config.CONTEXT_COL, "")
            tasks.append(annotate_single(session, post_id, text, context, semaphore))

        # 使用 as_completed 收集结果
        results_dict = {}
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="📝 标注对抗行为"):
            post_id, result = await coro
            results_dict[post_id] = result

    # 3. 合并结果
    df["ant_type"] = df["post_id"].map(lambda pid: results_dict[pid]["ant_type"])
    df["ant_strength"] = df["post_id"].map(lambda pid: results_dict[pid]["ant_strength"])
    df["ant_intent"] = df["post_id"].map(lambda pid: results_dict[pid]["ant_intent"])
    df["ant_strategy"] = df["post_id"].map(lambda pid: results_dict[pid]["ant_strategy"])

    ant_type_mapping = {0: "无对抗", 1: "质疑", 2: "反驳", 3: "攻击", 4: "冲突"}
    df["ant_type_label"] = df["ant_type"].map(ant_type_mapping)
    df["is_antagonistic"] = df["ant_type"] != 0

    # 4. 统计分析
    stats = calculate_statistics(df)
    print("\n📈 对抗行为统计结果：")
    for key, value in stats.items():
        print(f"\n【{key}】")
        for k, v in value.items():
            print(f"  {k}: {v}")

    # 5. 保存结果
    try:
        df.to_excel(Config.OUTPUT_EXCEL, index=False)
        with open(Config.OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 结果已保存：")
        print(f"  - 标注数据：{Config.OUTPUT_EXCEL}")
        print(f"  - 统计结果：{Config.OUTPUT_JSON}")
    except Exception as e:
        print(f"❌ 保存结果失败：{str(e)}")

    # 6. 可视化
    try:
        plot_analysis(df, Config.OUTPUT_PLOT)
    except Exception as e:
        print(f"⚠️ 可视化生成失败：{str(e)}")

    print("\n🎉 对抗行为分析流程全部完成！")


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """计算统计指标"""
    total = len(df)
    ant_total = df["is_antagonistic"].sum()
    ant_rate = round(ant_total / total * 100, 2) if total > 0 else 0

    type_dist = df["ant_type_label"].value_counts().to_dict()
    strategy_dist = df["ant_strategy"].value_counts().to_dict()
    strength_dist = df["ant_strength"].value_counts().sort_index().to_dict()
    intent_ant = df[(df["ant_intent"] == 1) & (df["is_antagonistic"])].shape[0]
    intent_rate = round(intent_ant / ant_total * 100, 2) if ant_total > 0 else 0

    stats = {
        "数据概览": {"总文本数": total, "对抗文本数": int(ant_total), "对抗率(%)": ant_rate},
        "对抗类型分布": type_dist,
        "对抗策略分布": strategy_dist,
        "对抗强度分布": strength_dist,
        "主动对抗占比(%)": {"占比(%)": intent_rate}   # 改为字典
    }
    return stats


def plot_analysis(df: pd.DataFrame, save_path: str):
    """生成可视化图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("AI Agent 对抗行为分析报告", fontsize=16, fontweight="bold")

    type_counts = df["ant_type_label"].value_counts()
    ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title("对抗类型分布", fontsize=12, fontweight="bold")

    strategy_counts = df["ant_strategy"].value_counts()
    ax2.bar(range(len(strategy_counts)), strategy_counts.values, tick_label=strategy_counts.index)
    ax2.set_title("对抗策略分布", fontsize=12, fontweight="bold")
    ax2.tick_params(axis='x', rotation=45)

    strength_counts = df["ant_strength"].value_counts().sort_index()
    ax3.plot(strength_counts.index, strength_counts.values, marker='o', linewidth=2, markersize=8)
    ax3.set_title("对抗强度分布", fontsize=12, fontweight="bold")
    ax3.set_xlabel("强度（1-5）")
    ax3.set_ylabel("文本数量")
    ax3.grid(True, alpha=0.3)

    intent_counts = df[df["is_antagonistic"]]["ant_intent"].value_counts()
    intent_labels = ["被动对抗（幻觉）", "主动对抗"] if 0 in intent_counts else ["主动对抗"]
    ax4.pie(intent_counts.values, labels=intent_labels, autopct='%1.1f%%', startangle=90)
    ax4.set_title("对抗意图分布", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化图表已保存至：{save_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()