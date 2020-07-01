## NER(Multi-Pointer)

**数据集：**

多类型（29类），多标签（同一条文本中每个实体有多个标签）

[面向金融领域的篇章级事件主体抽取](https://www.biendata.xyz/competition/ccks_2020_4_1/)

**模型:**

BERT(based on)+multi-pointer

**方法的优势:**

可以解决多标签的问题: 通过多层指针网络

**方法存在的问题:**

分词不够准确(无可靠约束)

**文件结构:**

-- data: 预处理后的数据

	-- preprocess.py: 数据预处理

-- dataloader.py, dataloader_utils.py

-- train.py, evaluate.py, predict.py: 训练,评估,预测

-- metrics.py: 评测指标

-- metrics_utils.py: 将指针结果转换为BIO形式的标注,便于评测

-- model.py: 下游模型

-- optimization.py: 优化器

-- postprocess.py: 数据后处理

-- utils.py