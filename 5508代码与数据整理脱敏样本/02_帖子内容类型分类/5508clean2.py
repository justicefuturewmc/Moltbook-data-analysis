import pandas as pd
import json
import re

def classify_content(text):
    """
    对帖子内容进行分类，返回 (content_type, code_function)
    """
    if pd.isna(text) or str(text).strip() == '':
        return 'text', None

    text = str(text).strip()

    # 1. 尝试提取并解析 JSON 对象（优先判断虚拟币操作）
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        potential_json = json_match.group(1)
        try:
            data = json.loads(potential_json)
            if isinstance(data, dict):
                # 检查是否为 mbc-20 协议操作
                p = data.get('p')
                op = data.get('op')
                if p == 'mbc-20' and op:
                    return 'code', f'crypto_{op}'   # 如 crypto_mint, crypto_transfer
                else:
                    # 其他 JSON 数据（可能不是虚拟币协议）
                    return 'code', 'json_data'
        except json.JSONDecodeError:
            pass  # 不是合法 JSON，继续其他判断

    # 2. 检查是否包含代码块标记
    if '```' in text:
        return 'code', 'code_block'

    # 3. 检测常见编程关键词（启发式）
    code_keywords = [r'\bdef\b', r'\bimport\b', r'\bclass\b', r'\bif\b', r'\bfor\b', r'\bwhile\b']
    lines = text.split('\n')
    if len(lines) > 3:
        # 计算包含关键词的行数比例
        keyword_lines = 0
        for line in lines:
            if any(re.search(kw, line) for kw in code_keywords):
                keyword_lines += 1
        if keyword_lines > 1:  # 至少两行包含关键词，更可靠
            return 'code', 'general_code'

    # 4. 其他情况视为自然语言文本
    return 'text', None


# 读取清洗后的数据
df = pd.read_excel('cleaned_moltbook操作.xlsx')

# 应用分类函数，生成新列
df[['content_type', 'code_function']] = df['post_content'].apply(
    lambda x: pd.Series(classify_content(x))
)

# 可选：查看分类统计
print(df['content_type'].value_counts())
print(df['code_function'].value_counts(dropna=False))

# 保存结果
df.to_excel('classified_moltbook操作2.xlsx', index=False)