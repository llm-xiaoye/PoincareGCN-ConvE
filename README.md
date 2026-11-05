# PoincareGCN-ConvE：知识图谱补全

结合庞加莱球表示、关系感知图卷积与 ConvE 风格解码器，对知识图谱中的缺失链接进行预测。支持使用自有三元组数据完成校验、训练、评估与候选实体排序。

## 核心功能

- 在庞加莱球中学习实体与关系表示，通过图卷积聚合邻域信息。
- 结合双曲卷积、归一化、激活与线性层生成查询表示。
- 基于庞加莱距离排序候选实体，支持头实体与尾实体预测。
- 提供三元组校验、数据划分、多目标训练、标签平滑与早停。
- 按验证集 MRR 恢复最佳检查点，报告过滤后的 MRR、平均排名与 Hits@K。

## 模型结构

三元组 → 实体与关系嵌入 → 关系感知图卷积 → 双曲 ConvE 解码 → 查询表示 → 候选实体排序。

头实体预测通过显式逆关系实现。模型结构与张量流见[架构说明](docs/architecture.md)。

## 环境与安装

需要 Python 3.10+。PyTorch 按目标平台安装；使用 GPU 时选择匹配的 CUDA 版本。项目使用 geoopt 支持流形参数与黎曼优化。

~~~bash
git clone https://github.com/llm-xiaoye/PoincareGCN-ConvE.git
cd PoincareGCN-ConvE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

## 数据格式

使用 UTF-8 TSV 文件，每行一条三元组，字段之间为制表符：

~~~text
HEAD<TAB>RELATION<TAB>TAIL
~~~

实体与关系名称由用户定义。验证集和测试集中的实体与关系需出现在训练集中。完整规则见[数据格式](docs/data_format.md)。

## 使用方式

### 校验与划分

~~~bash
poincare-validate path/to/all.tsv
poincare-split path/to/all.tsv path/to/splits --train-ratio 0.8 --dev-ratio 0.1 --test-ratio 0.1 --seed 42
~~~

划分工具会按需把三元组移回训练集，以覆盖验证和测试所需的实体与关系。

### 训练

~~~bash
poincare-train --config configs/poincare_gcn_conve.yaml --train path/to/train.tsv --dev path/to/dev.tsv --test path/to/test.tsv --output artifacts/my-model
~~~

默认配置位于 configs/poincare_gcn_conve.yaml；configs/conve_500epochs.yaml 提供另一组训练参数；configs/offline_smoke.yaml 用于小规模流程检查。

### 评估

~~~bash
poincare-evaluate --checkpoint artifacts/my-model --data path/to/test.tsv --filter-data path/to/dev.tsv
~~~

训练与评估三元组会加入过滤集合。可重复指定 --filter-data，补充其他已知正确三元组。

### 预测

~~~bash
poincare-predict --checkpoint artifacts/my-model --head overheat --relation caused_by --top-k 5
~~~

命令返回训练实体词表中的候选结果，不自动写入数据库。

## 配置与输出

训练目录保存权重、实体与关系词表、训练图三元组、解析后的配置、训练历史及指标。最终测试在恢复最佳验证集检查点后执行。

建议保持实体标识一致，在划分前检查重复和逆关系泄漏，并在验证集上调整曲率、维度、学习率、丢弃率与标签平滑。评估时同时报告头尾预测，并完整配置过滤集合。

## 开发与验证

~~~bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
~~~

持续集成覆盖 Python 3.10 与 3.12。

## 许可证与软件引用

采用 [MIT 许可证](LICENSE)。软件引用信息见 [CITATION.cff](CITATION.cff)。
