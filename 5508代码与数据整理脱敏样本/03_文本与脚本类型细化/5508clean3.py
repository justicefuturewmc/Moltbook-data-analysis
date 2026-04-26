import pandas as pd
import re
import json


def compute_code_length(text):
    """计算文本中代码块和JSON对象的字符总数"""
    text = str(text)
    code_len = 0
    # 提取三个反引号包裹的代码块
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        code_len += len(block)
    # 提取JSON对象（简单花括号匹配，可能不完美，但足够用于长度估算）
    json_matches = re.findall(r'\{[^{}]*\}', text)
    for j in json_matches:
        code_len += len(j)
    return code_len


def refine_code_type(row):
    """根据内容比例重新判断代码类属于script还是text"""
    if row['content_type'] == 'text':
        return 'text'

    # 虚拟币操作直接视为脚本
    if pd.notna(row['code_function']) and str(row['code_function']).startswith('crypto_'):
        return 'script'

    # 其他代码类：计算比例
    content = row['post_content']
    if pd.isna(content) or str(content).strip() == '':
        return 'text'

    total_len = len(str(content))
    code_len = compute_code_length(content)

    # 如果代码占比 >= 70% 或自然语言部分少于200字符，视为脚本
    if total_len > 0:
        code_ratio = code_len / total_len
        if code_ratio >= 0.7 or (total_len - code_len) < 200:
            return 'script'
    return 'text'


# 读取已分类的文件
df = pd.read_excel('classified_moltbook操作2.xlsx')  # 请替换为实际文件名

# 应用细分类
df['final_type'] = df.apply(refine_code_type, axis=1)

# 查看分布
print(df['final_type'].value_counts())

# 保存结果
df.to_excel('refined_moltbook.xlsx', index=False)