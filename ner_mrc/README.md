## NER(MRC)

**数据集：**

多类型（29类），多标签（同一条文本中每个实体有多个标签）

[面向金融领域的篇章级事件主体抽取](https://www.biendata.xyz/competition/ccks_2020_4_1/)

**模型:**

BERT(based on)+mrc

基于论文: [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)

**方法的优势:**

先天可以解决多标签的问题: 通过不同问题(对应不同类型)的数据构造

先天可以解决实体嵌套的问题: 通过span矩阵

**方法存在的问题:**

*局限性太大*

在多类别数据集上效果欠佳: 模型无法通过先验的问题来正确的分清类别

在少样本数据集上效果欠佳: 模型无法学到准确的类别信息

**文件结构:**

-- data: 预处理后的数据

	-- preprocess.py: 数据预处理

-- dataloader.py, dataloader_utils.py

-- train.py, evaluate.py, predict.py: 训练,评估,预测

-- metrics.py: 评测指标

-- metrics_utils.py: 将mrc预测结果转换为BIO形式的标注,便于评测

-- model.py: 下游模型

-- optimization.py: 优化器

-- postprocess.py: 数据后处理

-- utils.py




