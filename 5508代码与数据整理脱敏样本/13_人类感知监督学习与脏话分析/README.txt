group\最终输出文件\5508_3.4&5_0326_final.ipynb

对应报告内容
------------
1. The Real Ecosystem of Moltbook An Empirical Study of AI Agent Behaviors.docx
   - 3.4 How AI Agents Perceive and Discuss Humans
   - 3.5 Profanity Usage by AI Agents

2. 3.3Method&Result(1).docx
   - 间接引用 3.4 中的人类角色分类结果，例如 humans are 9 times more likely to be viewed as owners than partners。
   - 间接引用 3.5 中的 profanity context/target 结果，例如 Tech Frustration 40.9%、self-directed profanity 17%。

主要代码性分析
--------------
1. 人类相关帖子监督学习 / human-in-the-loop
   - keyword filter: human, user, creator 等。
   - manual labels / edge cases。
   - TF-IDF + LogisticRegression。
   - 输出 pred_is_human, Sentiment_ML, Topic_ML, Role_ML。

2. 脏话使用分析
   - profanity lexicon + regex word boundary。
   - offensive emoji 映射。
   - Swear_Topic 与 Swear_Target 的规则分类。
   - donut charts 与 word cloud。
   - 输出 profanity_analysis_posts.xlsx / csv 类结果。

未复制的大型输出
----------------
group\最终输出文件\3.4 人类\产出_final_ml_analysis_result_v4.csv
group\最终输出文件\3.4 人类\to_be_labeled_full_v4.csv
group\最终输出文件\3.5 脏话\profanity_analysis_posts.csv

（文档结束）
