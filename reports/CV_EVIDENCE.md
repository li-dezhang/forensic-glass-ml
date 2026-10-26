# 可核实的 CV 项目素材

项目：**Forensic Glass Classification and Uncertainty-Aware Machine Learning**  
性质：真实公开数据的计算研究训练；未发表；AI 辅助；不是司法鉴定业务系统。

## 已实现且有文件证据的功能

| 事实 | 证据 |
|---|---|
| 官方 ucimlrepo 下载，保留原始 214 条记录、9 个输入与 6 个实际类别；记录许可、字典、来源和哈希 | ../data/manifest.json；../data/dictionary.csv |
| 检查缺失、类型和重复；删除一条重复测量的建模副本，保留 213 条，ID 不进模型 | ../src/run.py；../results/data_quality.json |
| 159/54 分层划分；训练集内 5 折 CV；标准化在 Pipeline 内；种子 42 | ../results/split.csv；../results/cv_folds.csv；../EXPERIMENT_PROTOCOL.md |
| 比较 Dummy、Logistic Regression、kNN、Random Forest；训练 CV 选模 | ../results/metrics.csv；../results/selection.json |
| 保存逐记录预测、六类概率、混淆矩阵、各类召回及真实错误样本 | ../results/predictions.csv；../results/errors.csv |
| 固定四档概率阈值；同时报告覆盖率、选择性准确率、审核数量和分类别覆盖 | ../results/confidence_metrics.csv；../results/confidence_by_class.csv |
| 12 项测试通过；完整重复拟合及重新评分绘图，13 份 CSV、4 张 PNG 一致 | ../results/tests.txt；../results/verification.json |
| 英文/中文研究报告与四张图 | REPORT_EN.md；REPORT_ZH.md；../results/*.png |

## 实际结果

- 训练 CV 选中 Random Forest，平均 Macro-F1=0.7255；54 条测试记录正确 42 条。
- 测试 Accuracy=77.78%，Macro-F1=0.7585，Balanced accuracy=0.7691。
- 0.60 阈值：自动保留 33/54，Coverage=61.11%，其中正确 32/33，Selective accuracy=96.97%，21 条需审核。
- 0.80 阈值：仅保留 18/54，准确 17/18；高于 0.60 的阈值没有带来更高选择性准确率。
- 类别 3 召回率为 50%（2/4），在 0.70 和 0.80 时自动覆盖为零。ID 3 以 0.90 模型概率被误判，说明高分不保证正确。

## 可用于 CV 的项目描述草稿

以下描述的是项目产物，**不是对学生个人贡献的认证**。学生应确认实际参与、理解代码和结果后，按真实角色改写。

**中文：**  
参与一项 AI 辅助的法医玻璃分类计算研究，使用 UCI 真实数据构建可复现 Python 流程，比较四种传统分类方法，采用训练集内分层交叉验证与独立测试集评估，并分析概率阈值拒识中的准确率—覆盖率权衡。项目保留数据溯源、逐样本预测、双语报告及 12 项自动化测试。

**English project summary:**  
An AI-assisted Python research project benchmarking four conventional classifiers on the UCI Glass Identification dataset, with training-only stratified cross-validation, a held-out test set, and exploratory confidence-based abstention. The CV-selected Random Forest achieved 77.78% accuracy and 0.7585 macro-F1 on 54 test records; a fixed 0.60 threshold retained 61.11% coverage with 96.97% selective accuracy. Reproducible outputs, bilingual reports and twelve automated checks document the analysis.

如学生只完成了结果审阅、复现或解释，应使用 “reviewed / reproduced / analyzed”，不能直接改成 “independently designed and implemented”。实际个人贡献目前未经确认。

## 尚未实现与不可声称的内容

未实现：概率校准、外部验证、重复划分置信区间、未知类别检测、部署、审核者效果实验、证据似然比、真实实验室采样或鉴定。

不能声称：独立提出新算法、已发表论文、识别个体玻璃来源、达到司法应用标准、96.97% 覆盖全部样本、阈值经外部验证，或独立人工完成全部编程与写作。

机器学习实验与图表来自实际程序运行；本轮没有修改或调用之前五项目的研究程序，也没有创建远程仓库。

