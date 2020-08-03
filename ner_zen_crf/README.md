## NER(Based on CRF)

**数据集：**

多类型（29类），多标签（同一条文本中每个实体有多个标签）

[面向金融领域的篇章级事件主体抽取](https://www.biendata.xyz/competition/ccks_2020_4_1/)

**模型：**

ZEN+middle layer+crf

**方法的优势:** 

传统的序列标注模型,在分词任务上有CRF模型约束,表现较好;在分类任务上也能学到更合理的信息.

**方法存在的问题:**

先天无法解决多标签问题,即一条文本中的一个实体只能对应一个类别

先天无法解决实体嵌套的问题

**文档结构：**

-- data: 预处理后的数据

	-- preprocess.py: 数据预处理

-- middle_layer: 中游模型

	-- bi_lstm.py
	-- idcnn.py
	-- r_transformer.py
	-- tener.py

-- data_analyze.py: 分析数据长度

-- dataloader.py, dataloader_utils.py

-- ensemble.py: 模型融合

-- metrics.py: 评测指标

-- model.py: 下游模型

-- optimization.py: 优化器

-- postprocess.py: 数据后处理

-- train.py, evaluate.py, predict.py: 训练,评估,预测

-- utils.py

