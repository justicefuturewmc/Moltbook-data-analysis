import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re

# 固定随机种子，确保结果可复现
DetectorFactory.seed = 0

def detect_language(text):
    """检测文本语种，失败返回'unknown'"""
    if pd.isna(text):
        return 'unknown'
    text_str = str(text).strip()
    if text_str == '':
        return 'unknown'
    try:
        # 只取前1000字符避免过长
        return detect(text_str[:1000])
    except LangDetectException:
        return 'unknown'

def is_valid_content(text):
    """判断内容是否有效（虚拟币信息视为有效）"""
    if pd.isna(text):
        return False
    text_str = str(text).strip()
    # 空或"N/A"视为无效
    if text_str == '' or text_str.upper() == 'N/A':
        return False
    # 长度小于2的无效
    if len(text_str) < 2:
        return False
    # 如果内容中没有任何字母或汉字（即纯数字或符号），且长度小于5，视为无效
    # 正则匹配任何Unicode字母（包括中文等）
    if not re.search(r'[\u0041-\u005A\u0061-\u007A\u0080-\uFFFF]', text_str):
        # 如果全是数字或空格，则无效
        if re.match(r'^[\d\s]+$', text_str):
            return False
    return True

# 读取Excel文件（请根据实际路径修改）
df = pd.read_excel('moltbook_agents_full_extract（3.2）操作.xlsx', engine='openpyxl')

# 对post_content进行语种检测
df['language'] = df['post_content'].apply(detect_language)

# 标记是否有效
df['is_valid'] = df['post_content'].apply(is_valid_content)

# 查看无效数据（可选）
invalid_df = df[~df['is_valid']]
print(f"无效数据条数: {len(invalid_df)}")
if len(invalid_df) > 0:
    print("无效数据示例：")
    print(invalid_df[['filename', 'post_content']].head(10))

# 清洗：保留有效数据
clean_df = df[df['is_valid']].copy()

# 保存结果
clean_df.to_excel('cleaned_moltbook操作.xlsx', index=False)
print("清洗完成，结果已保存至 cleaned_moltbook操作.xlsx")