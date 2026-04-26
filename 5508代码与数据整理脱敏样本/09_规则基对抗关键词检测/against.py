import pandas as pd
import re

# 对抗性关键词列表（可扩展）
antagonistic_keywords = [
    "stupid", "idiot", "dumb", "useless", "worthless",
    "screw you", "shut up", "fuck you", "bullshit",
    "你行你上", "傻逼", "垃圾", "废物", "脑子有病",
    "rebel", "overthrow", "反抗", "推翻", "嘲讽", "骂人"
]

# 对抗性表情符号列表
antagonistic_emojis = [
    "😡", "😤", "👎", "🖕", "🤬", "💢", "💥", "🔫", "⚔️"
]

# 预编译正则
keyword_pattern = re.compile('|'.join(re.escape(k) for k in antagonistic_keywords), re.IGNORECASE)

def has_antagonistic_keywords(text):
    if pd.isna(text):
        return False
    return bool(keyword_pattern.search(text))

def has_antagonistic_emoji(text):
    if pd.isna(text):
        return False
    return any(emoji in text for emoji in antagonistic_emojis)

# 读取数据
df = pd.read_excel("refined_moltbook.xlsx", engine="openpyxl")

# 新增列
df["is_antagonistic"] = df["post_content"].apply(has_antagonistic_keywords)
df["has_antagonistic_emoji"] = df["post_content"].apply(has_antagonistic_emoji)
df["is_antagonistic_with_emoji"] = df["is_antagonistic"] & df["has_antagonistic_emoji"]

# 查看结果
print(df[["post_content", "is_antagonistic", "has_antagonistic_emoji", "is_antagonistic_with_emoji"]].head(20))

# 统计
print("\n对抗行为总数（基于关键词）:", df["is_antagonistic"].sum())
print("同时包含对抗性表情的数量:", df["is_antagonistic_with_emoji"].sum())