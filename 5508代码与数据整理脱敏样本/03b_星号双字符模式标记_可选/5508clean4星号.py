import pandas as pd
import os


def has_double_star(text):
    """
    检查文本中是否包含连续的两个或以上星号（即子串 '**'）。
    """
    if isinstance(text, str):
        return '**' in text
    return False


def classify_row(row, text_columns=None):
    """
    对一行进行分类：如果任一文本列包含连续星号，则返回 'A'，否则返回 'B'。
    text_columns: 指定要检查的列名列表，若为None则检查所有字符串类型的列。
    """
    if text_columns is None:
        # 自动选择所有字符串类型的列
        cols = [col for col in row.index if isinstance(row[col], str)]
    else:
        cols = text_columns
    for col in cols:
        if has_double_star(row[col]):
            return 'A'
    return 'B'


def classify_excel(file_path, output_path=None, text_columns=None):
    """
    读取Excel文件，对每一行添加分类列（A/B），并保存结果。
    若 output_path 为 None，则覆盖原文件（建议先备份）。
    """
    df = pd.read_excel(file_path, dtype=str)  # 强制读为字符串避免自动转换
    df['classification'] = df.apply(classify_row, axis=1, text_columns=text_columns)

    if output_path is None:
        output_path = file_path
    df.to_excel(output_path, index=False)
    print(f"分类完成，结果保存至：{output_path}")
    return df


if __name__ == "__main__":
    # 请根据实际文件名修改以下变量
    input_file = "classified_moltbook操作2.xlsx"  # 原文件名
    output_file = "classified_with_label.xlsx"  # 输出文件名，可设为 None 覆盖原文件

    # 可选：指定要检查的文本列，若为 None 则自动检查所有字符串列
    # 根据样本，可指定以下列（可根据需要调整）：
    text_columns = ['post_content']

    classify_excel(input_file, output_file, text_columns)
