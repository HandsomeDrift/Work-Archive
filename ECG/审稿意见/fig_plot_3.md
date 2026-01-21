下面给出一套**可直接落地**的补充实验计划，专门回应“t-SNE 仅作示意、需提供与误分/临床相关的定量证据”的审稿期望。计划以最少改动重用现有模型与代码，产出可入主文与附录的图表与统计结果。

------

# 1. 研究目标与可检验假设

**目标**：将“跨模态对齐”由示意图提升为**定量证据**，并解释其与**误分、置信度与校准**的关系。

**主要假设**
 H1（误分对齐更弱）：误分样本的跨模态对齐显著弱于正确样本。
 H2（方法有效性）：引入对比损失（或更合理的温度/权重）能显著增强对齐并降低误分率。
 H3（不确定性关联）：对齐弱的样本更易**过置信/欠置信**（较差校准），属于高风险区域。
 H4（稳健性）：上述结论在不同数据子集（类别/种子）上保持一致。

------

# 2. 数据与评估设置

- **数据集**：优先使用**临床数据集**（含 Lab），并在 1 个公开集上做最小复核（可入附录）。
- **划分与种子**：沿用原文训练/验证/测试划分；测试阶段设 **N=5** 个随机种子复现实验（报告均值±标准差与 95% CI）。
- **模型版本**：固定你们论文中的**最佳配置**（Full：TS+IMG+Lab, w/ contrastive）。做两组对照：
  - w/ contrastive（原方法） vs **w/o contrastive**（去掉对比损失）；
  - 可选灵敏度：温度 (\tau) ∈ {0.07, 0.1, 0.2}；权重 (\lambda) ∈ {0.2, 0.5, 1.0}（入附录）。

------

# 3. 表征与对齐度量（在**原始嵌入空间**而非 t-SNE）

- **取向**：使用对比学习分支中的**投影后嵌入**（与对比损失同一空间），各模态 L2 归一化。

- **样本级配对表征**：对第 (n) 个样本，记
  $$
   z^{\text{TS}}_n,;z^{\text{IMG}}_n \in \mathbb{R}^d,\quad |z^{\cdot}_n|_2=1.
  $$

- **成对距离/相似度**：
  $$
  d_n=|z^{\text{TS}}_n - z^{\text{IMG}}_n|_2,\qquad s_n=z^{\text{TS}}_n!\cdot z^{\text{IMG}}_n=1-\tfrac{1}{2}d_n^2.
  $$

- **Alignment@k（互为最近邻）**：对每个 (z^{\text{TS}}_n) 在 ({z^{\text{IMG}}_m}) 中取前 k 近邻，检查其真实配对 (z^{\text{IMG}}_n) 是否命中；再对称计算并取交集得到 **R-k（互为前 k）**。建议 (k\in{1,5})。

- **类内对齐/可分性**（可放附录）：计算每个类别的对齐均值 (\bar d_c)、类中心间距与类内散度（Fisher 比例），辅助说明对齐与判别边界的关系。

------

# 4. 统计检验与分析框架

- **H1：误分 vs 正确**
  - 将测试集按**是否误分**分组，比较 ({d_n}) 或 ({s_n}) 的分布：
    - 检验：Mann–Whitney U（双侧），报告 **p 值、Cliff’s (\delta)** 与 95% CI；
    - 画图：箱线/小提琴图（主文 1 幅）。
- **H2：方法有效性**
  - 比较 **w/ contrastive** 与 **w/o contrastive** 两组的（i）平均对齐度（(d, s)），（ii）误分率，（iii）Alignment@k、R-k；
  - 用 **配对种子**的 t-test 或置换检验报告差异显著性（均值±SD 与 95% CI）。
- **H3：不确定性/校准关联**
  - 取模型**预测置信度**（最大 softmax）与**对齐度**（(d_n) 或 (s_n)）做二维分析：
    - 将样本按对齐度分成 **5 分位**（Q1=对齐弱 … Q5=对齐强），分别计算 **ECE、Brier** 与 **误分率**；
    - 绘制**可靠性曲线**（预测置信度 vs 实际命中率）并在图上标注对齐分位；
    - 给出“对齐弱区间的误分率更高且校准更差”的统计证据（附 p 值与效应量）。
- **稳健性**：在**类别/数据域**（公开集/临床集）与**随机种子**层面复核；多重比较采用 **Benjamini–Hochberg FDR** 控制。

------

# 5. 可视化与表格（主文最少 + 附录扩展）

**主文建议 2–3 幅图 + 1 张表：**

- **Fig. A（主图）**：误分 vs 正确 的对齐分布（箱线/小提琴，附 U 检验 p 值与 Cliff’s (\delta)）。
- **Fig. B**：置信度–对齐二维密度/散点（误分点高亮），或**对齐分位**的**可靠性曲线**/ECE 条形图。
- **Table A**：w/ vs w/o contrastive 的 (d,s)、Alignment@k/R-k 与误分率（均值±SD，95% CI，显著性）。
   **附录（可选）**：
- 温度/权重灵敏度表；
- 类内对齐与类间可分性统计；
- 2–3 个病例级可解释图：ViT Grad-CAM/attention rollout、Graphormer 注意力（导联/边），FiLM (\gamma,\beta) 与关键 Lab 的相关性，解释“为何对齐弱而误分”。

------

# 6. 与正文整合的“回应句式”示例（可直接用于 Rebuttal）

- *We find that misclassified cases exhibit **significantly weaker cross-modal alignment** (median paired distance ↑; Mann–Whitney U, (p<0.01); Cliff’s (\delta=0.xx)). Strengthening alignment via the contrastive objective **reduces error rates** and **improves calibration** in low-alignment strata.*
- *Reliability analysis shows that the **lowest-alignment quintile** concentrates most misclassifications and displays higher ECE, suggesting that alignment is a useful **risk indicator** for clinical deployment.*

------

# 7. 质量控制与实现注意

- **空间一致性**：一律在**投影后嵌入空间**计算（与对比损失同空间）；先 L2 归一化再计算距离/相似度。
- **避免泄漏**：所有统计仅在**测试集**上进行；调参在验证集完成。
- **均衡取样**：类别不平衡时，报告宏/微平均，并在附录提供分层结果。
- **可复现性**：固定随机种子并公开 key 配置（温度、权重、嵌入维度、k 值、ECE 分箱数等）。

------

# 8. 预计论文改动点（占位）

- **Sec. IV-F（新增）**：对齐的定量指标与误分分析；**Fig. A/B**, **Table A**。
- **Sec. IV-E（补充）**：w/ vs w/o contrastive 的配平对照与 Alignment@k 结果。
- **Sec. VI（讨论）**：将“对齐弱→高风险/需关注”的临床含义与未来部署（告警/复核）衔接。

------

**一句话总结**：按上述计划，你们可以在不改动主干训练流程的前提下，给出**可量化、可检验**、与**误分/校准/临床风险**直连的证据，精准满足审稿人“超越 t-SNE”的期望，并显著提升论文说服力与影响力。