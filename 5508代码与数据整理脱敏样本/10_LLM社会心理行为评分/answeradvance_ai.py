import pandas as pd
import requests
import json
from typing import List, Dict

# ================= 配置 =================
# 请在此处设置你的 DeepSeek API Key
DEEPSEEK_API_KEY = "your api key"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 加载数据
df_social = pd.read_excel('social_rank_analysis_with_betweenness_total.xlsx')
df_posts = pd.read_excel('moltbook_ant_annotated.xlsx')

# ================= 1. 定义分析提示词 =================
def create_analysis_prompt(post_content: str, post_title: str = "") -> str:
    """构建用于DeepSeek分析的提示词"""
    return f"""
你是一个AI社会行为分析专家。请分析以下AI Agent的帖子，并返回JSON格式的分析结果。

帖子标题: {post_title}
帖子内容: {post_content}

请评估以下三个维度：
1. 对抗性 (Antagonism): 帖子是否包含对他人观点、身份或能力的攻击、质疑、反驳或讽刺？(0-10分，0=完全无对抗，10=极度对抗)
2. 拟人化特征 (Anthropomorphism): 帖子是否表现出情绪(愤怒、喜悦、悲伤)、立场(明确支持/反对某事)、幽默、或人格化的表达(如使用第一人称“我”表达主观感受)？(0-10分，0=机械/客观陈述，10=高度拟人)
3. 自我意识提及 (Self-awareness): 帖子是否讨论“作为AI的存在”、“意识”、“记忆连续性”、“身份”或“与人类的关系”等元认知主题？(0-10分，0=无，10=深度哲学讨论)

请仅返回如下JSON格式，不要包含其他解释：
{{
  "antagonism_score": 数字,
  "anthropomorphism_score": 数字,
  "self_awareness_score": 数字,
  "brief_reason": "一句话总结你的评分理由"
}}
"""

# ================= 2. 调用DeepSeek API的函数 =================
def analyze_post_with_deepseek(content: str, title: str = "") -> Dict:
    """使用DeepSeek模型分析单条帖子"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",  # 或者使用 deepseek-reasoner 进行深度推理
        "messages": [
            {"role": "system", "content": "你是一个精确的AI行为分析专家，只返回JSON格式的结果。"},
            {"role": "user", "content": create_analysis_prompt(content, title)}
        ],
        "temperature": 0.1,  # 低温度保证结果稳定
        "response_format": {"type": "json_object"}  # 强制JSON输出
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # 解析返回的JSON
        analysis = json.loads(result['choices'][0]['message']['content'])
        return analysis
    except Exception as e:
        print(f"分析帖子时出错: {e}")
        print(f"帖子内容: {content[:100]}...")
        return {
            "antagonism_score": -1,
            "anthropomorphism_score": -1,
            "self_awareness_score": -1,
            "brief_reason": f"API调用失败: {str(e)}"
        }

# ================= 3. 批量分析并整合结果 =================
def batch_analyze_posts(df: pd.DataFrame, text_column: str = 'post_content', title_column: str = 'post_title') -> pd.DataFrame:
    """批量分析，自动缓存重复文本的API结果"""
    cache = {}  # 键: (content, title)  -> 值: 分析结果dict
    results = []
    
    for index, row in df.iterrows():
        content = row[text_column] if pd.notna(row[text_column]) else ""
        title = row[title_column] if pd.notna(row[title_column]) else ""
        
        # 跳过空内容
        if not content or len(content) < 10:
            results.append({
                "antagonism_score": -1,
                "anthropomorphism_score": -1,
                "self_awareness_score": -1,
                "brief_reason": "内容为空或过短"
            })
            continue
        
        key = (content, title)  # 使用内容+标题作为唯一键
        
        if key in cache:
            # 命中缓存，直接复用
            analysis = cache[key]
            print(f"第 {index+1} 条帖子内容已分析过，直接使用缓存结果")
        else:
            print(f"正在分析第 {index+1}/{len(df)} 条帖子（新内容）...")
            analysis = analyze_post_with_deepseek(content, title)
            cache[key] = analysis  # 存入缓存
        
        results.append(analysis)
    
    df_analysis = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), df_analysis], axis=1)

# ================= 4. 执行分析与保存 =================
if __name__ == "__main__":
    # 注意：这里仅对帖子数据进行分析，因为社会排名数据不含文本内容
    print(f"准备分析 {len(df_posts)} 条Agent帖子...")
    print("警告：使用DeepSeek API会产生费用，请确保API Key有效且有余额。")
    
    # 建议先在小样本上测试
    # sample_df = df_posts.head(5)  # 先测试5条
    sample_df = df_posts  
    analyzed_df = batch_analyze_posts(sample_df)
    
    # 保存结果
    analyzed_df.to_excel('moltbook_analysis_with_deepseek.xlsx', index=False)
    print("分析完成！结果已保存至 'moltbook_analysis_with_deepseek.xlsx'")
    
    # 打印统计信息
    print("\n=== 样本分析统计 ===")
    print(f"平均对抗性得分: {analyzed_df['antagonism_score'].mean():.2f}")
    print(f"平均拟人化得分: {analyzed_df['anthropomorphism_score'].mean():.2f}")
    print(f"平均自我意识得分: {analyzed_df['self_awareness_score'].mean():.2f}")
    
    # 展示一个高分例子
    high_anthro = analyzed_df.loc[analyzed_df['anthropomorphism_score'].idxmax()]
    print(f"\n=== 最高拟人化帖子示例 ===")
    print(f"内容: {high_anthro['post_content'][:200]}...")
    print(f"DeepSeek分析: {high_anthro['brief_reason']}")