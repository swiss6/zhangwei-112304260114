# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：张蔚
- **学号**：112304260114
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial
- **提交日期**：2026年4月

- **GitHub 仓库地址**：https://github.com/swiss6/zhangwei-112304260114
- **GitHub README 地址**：https://github.com/swiss6/zhangwei-112304260114

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.90276
- **Private Score**：0.90276
- **排名**：无

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 截图文件名：112304260114_张蔚_kaggle_score.png

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**
1. **去除HTML标签**：使用 BeautifulSoup 库解析并移除评论文本中的HTML标签
2. **去除非字母字符**：使用正则表达式 `[^a-zA-Z]` 将所有非字母字符替换为空格
3. **转小写**：将所有文本转换为小写形式
4. **分词**：使用空格分割文本为单词列表
5. **去停用词**：移除英语停用词（如 a, an, the, is, are 等），但保留否定词（如 not, no, never 等），因为否定词对情感分析很重要
6. **处理缩写**：将缩写形式展开（如 n't → not, 're → are, 's → is 等）

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**
1. **训练方式**：自己训练 Word2Vec 模型，使用训练集和测试集的所有文本作为语料
2. **词向量维度**：300 维
3. **模型参数**：
   - `vector_size=300`：词向量维度
   - `window=10`：上下文窗口大小
   - `min_count=2`：最小词频阈值
   - `sg=1`：使用 Skip-gram 模型
   - `epochs=30`：训练轮数
   - `negative=10`：负采样数量
4. **句子向量表示**：采用 **TF-IDF 加权平均** 的方法
   - 对每个句子中的所有词向量，使用 TF-IDF 权重进行加权平均
   - TF-IDF 权重反映词在文档中的重要性
   - 公式：$\vec{v}_{doc} = \frac{\sum_{w \in doc} TFIDF(w) \cdot \vec{v}_w}{\sum_{w \in doc} TFIDF(w)}$

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**
1. **尝试的模型**：
   - Random Forest（随机森林）：用于 Bag of Words 基准模型
   - Logistic Regression（逻辑回归）：用于 Word2Vec 和 TF-IDF 特征
   - XGBoost：用于特征融合方法

2. **最终采用的模型**：**Logistic Regression（逻辑回归）**
   - 原因：在 Word2Vec + TF-IDF 加权特征上表现最佳
   - 超参数调优：通过网格搜索选择最佳正则化参数 C
   - 测试了 C ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0}
   - 最终选择 C 值使验证集 AUC 最高

---

## 7. 实验流程
请简要说明你的实验流程。

**我的实验流程：**
1. **数据加载**：读取训练集（labeledTrainData.tsv）和测试集（testData.tsv）
2. **文本预处理**：
   - 去除 HTML 标签
   - 去除非字母字符
   - 转小写、分词
   - 去停用词（保留否定词）
3. **训练 Word2Vec 模型**：
   - 使用全部文本训练词向量
   - 保存模型供后续使用
4. **构建 TF-IDF 权重**：
   - 计算每个词的 TF-IDF 值
   - 用于句子向量的加权平均
5. **生成句子向量**：
   - 对每个句子，使用 TF-IDF 加权平均词向量
   - 得到 300 维句子表示
6. **模型训练与验证**：
   - 划分训练集和验证集（8:2）
   - 训练 Logistic Regression
   - 调参选择最佳超参数
7. **预测与提交**：
   - 在完整训练集上重新训练
   - 对测试集进行预测
   - 生成 submission.csv 文件提交 Kaggle

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

**我的项目结构：**
```text
Bag of Words Meets Bags of Popcorn/
├── data/
│   ├── labeledTrainData.tsv/     # 训练数据（带标签）
│   └── testData.tsv/             # 测试数据
├── src/
│   ├── bag_of_words_model.py     # Bag of Words + 随机森林基准模型
│   ├── word2vec_mean_embedding.py # Word2Vec + 平均嵌入方法
│   ├── word2vec_improved.py      # Word2Vec + TF-IDF加权方法（最终方案）
│   ├── tfidf_train.py            # TF-IDF + 逻辑回归方法
│   ├── combined_features.py      # 特征融合方法
│   ├── word2vec_train.py         # Doc2Vec 方法
│   └── google_word2vec.py        # Google预训练词向量方法
├── images/
│   └── kaggle_score.png          # Kaggle 提交截图
├── submission/
│   ├── Word2Vec_LR_submission_improved.csv  # 最终提交文件
│   ├── Word2Vec_LR_submission.csv
│   ├── TF-IDF_model.csv
│   └── Combined_submission.csv
├── models/                       # 保存的模型文件
│   ├── word2vec_improved.model   # Word2Vec 模型
│   ├── lr_w2v_model_improved.pkl # 逻辑回归模型
│   └── tfidf_vectorizer.pkl      # TF-IDF 向量化器
└── README.md                     # 实验报告
```

---

## 9. 实验总结

### 主要发现
1. **TF-IDF 加权平均** 比简单平均嵌入效果更好，因为考虑了词的重要性
2. **保留否定词** 对情感分析很重要，如 "not good" 和 "good" 情感相反
3. **Skip-gram 模型** (sg=1) 在小数据集上通常比 CBOW 效果更好
4. **逻辑回归** 在高维稀疏特征上表现稳定且高效

### 改进方向
1. 尝试使用预训练的 Google News Word2Vec 模型
2. 使用深度学习模型（如 LSTM、BERT）
3. 增加更多特征工程（如词性标注、情感词典）
