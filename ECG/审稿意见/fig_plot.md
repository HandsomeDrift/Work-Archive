下面是一份**可直接交接实施**的绘图规范文档（spec）。面向 Python（pandas + numpy + matplotlib + scikit-learn + scipy），默认**静态、可复现、期刊级分辨率**输出（SVG/PDF/PNG）。所有图均服务于同一核心论断：**“误分类样本的跨模态对齐更弱，且该弱对齐与错误风险、模型置信度与校准等性质相关。”**

> 统一数据源假定：`misalign_metrics.csv`（你已上传）
>  统一随机种子：`SEED = 2025`
>  统一输出路径：`fig/`（若不存在则创建）
>  统一分辨率与尺寸：`dpi=600`，`figsize=(6.0, 4.5)`（单图）；多面板按 1.5–1.8× 放大
>  统一字体与栅格：字体 ≥ 9 pt；主次网格线均采用细线；图例、坐标轴标签与标题齐备且不冗长
>  统一配色：正确类/高对齐用较深色，误分类/低对齐用浅色或空心记号；**不必强行指定色值**，遵循期刊“黑白友好”——线型、填充、记号形状区分

------

# 0. 公共部分（所有图共用）

## 0.1 数据字段（列名约定）

以下列名如与你的 CSV 不一致，请在读取时映射（rename）到规范名。

- `id`：样本唯一标识（字符串或整数）。
- `y_true`：真实标签（0/1）。
- `y_pred`：预测标签（0/1）。
- `correct`：是否预测正确（bool 或 {0,1}）。
- `conf`：**预测类别**的 softmax 概率（[0,1]），若可得“阳性概率”另存 `p_pos`。
- `cos_ts_img`：cos(ts, img)。
- `cos_ts_fusion`：cos(ts, fusion)。
- `cos_img_fusion`：cos(img, fusion)。
- `l2_ts_img`：‖ts − img‖（向量已 L2 归一化后计算）。
- `cos_proj`：cos(proj_t, proj_i)（对比空间）。

> 若数值越大代表越“好”的对齐，请保证方向一致（例如距离类指标需转成负向或显式说明“值越大越差”）。

## 0.2 预处理与稳健统计

- 缺失：若某列存在空值，**先行删除**该行或使用“就近插值/中位数”填补；在图注说明处理方式。
- 分箱：`pd.qcut(x, q=10, duplicates='drop')` 作为 10 分位分箱的默认实现。
- 置信带：bootstrap（B=1000）计算 bin 内比例/均值的 95% CI（百分位法）。
- 组间检验：`scipy.stats.mannwhitneyu`；效应量推荐 **Cliff’s δ**（实现：对两组秩比较）。
- 复现：固定 `np.random.seed(SEED)`；bootstrap 与置换检验均设种子。

## 0.3 文件命名与导出

- 主文图命名：`fig5A_quantile_error.svg/pdf/png` 等；补充材料：`figS_*`。
- 所有图均导出 **SVG + PDF + PNG**，SVG/PDF 用于投稿，PNG 用于草稿预览。
- 元信息（实验参数）写入同名 `*.json`（记录 bin 数、检验 p 值、CI、版本号与时间）。

------

# A. 对齐分位数—错误率曲线（主图，强推荐）

**目的**：展示“对齐越弱，错误率越高”的单调关系。
 **输入**：`cos_proj`（主指标），`correct`。
 **步骤**：

1. 将 `cos_proj` 以 10 分位分箱，得到 bins（中位数作为 x 轴位置）。
2. 对每个 bin 计算 `error_rate = 1 - mean(correct)`；bootstrap 95% CI。
3. 绘制**折线 + 竖向误差条**；在背景叠加散点（每点为一个 bin）。
4. 计算 Spearman 相关与其 p 值，写入角标。
    **可选**：在同一图上叠加 `l2_ts_img` 的分位数曲线（以右轴表示），展示“距离↑→错误率↑”。
    **导出**：`fig5A_quantile_error.{svg,pdf,png}`
    **图注要点**：10 等分位，误差条为 bootstrap 95% CI；Spearman ρ 与 p 值。

**边界情形**：若样本极少或重复值多导致分位箱重叠（`duplicates='drop'`），自动降为 <10 个 bin，并在图注注明。

------

# B. 拒识门限曲线（Alignment Threshold vs Coverage/Performance）

**目的**：证明对齐阈值可作为部署时“安全阀”。
 **输入**：`cos_proj`，`correct`（必须）；如可得 `p_pos`，可兼报 AUROC/AUPRC。
 **步骤**：

1. 定义一系列阈值 (q^*)（例如 `np.linspace(min,max,50)`）。
2. 对每个 (q^*)：保留 `cos_proj >= q*` 的样本（高对齐），计算
   - 覆盖率 `cov = retained / total`；
   - `ACC = mean(correct)`；
   - 若有 `p_pos`，计算 AUROC/AUPRC（sklearn）。
3. 绘制 **双轴**：x=覆盖率（或阈值），左轴=ACC（或 AUROC），右轴=覆盖率；标注“90% 覆盖”处的性能点。
    **导出**：`fig5B_reject_option.{svg,pdf,png}`
    **图注要点**：阈值越高，覆盖率下降，但性能上升；实务意义：将低对齐样本送复检。

**边界情形**：若无 `p_pos`，仅报 ACC；覆盖率<0.5 区域可浅色显示以提示样本过少。

------

# C. 逻辑回归森林图（对齐的独立效应）

**目的**：在控制置信度、类别等后，验证“对齐弱→误分风险↑”是独立效应。
 **输入**：`correct`、`cos_proj`、`l2_ts_img`、`conf`、`y_true`。
 **步骤**：

1. 因变量 `is_error = 1 - correct`。
2. 自变量：`cos_proj`（主要），`l2_ts_img`（次要）、`conf`（置信度）、`y_true`（类别虚拟变量）。
3. 标准化连续变量（z-score）。
4. 用 `sklearn.linear_model.LogisticRegression`（L2 惩罚，C=1）拟合；提取各自变量的 OR=exp(系数) 与 95% CI（基于标准误差近似或 bootstrap）。
5. 绘制**横向森林图**（OR 与 CI；OR<1 表示“对齐高→误分风险低”）。
    **导出**：`fig5C_forest_OR.{svg,pdf,png}`
    **图注要点**：报告 OR（每 IQR 变化）、95% CI；对齐变量 OR<1 且 CI 不跨 1 则成立。

**边界情形**：若分离严重（完美预测），启用 `solver='liblinear'` 或降低变量数；在日志中记录告警。

------

# D. 概率校准分层图（Reliability by Alignment Strata）

**目的**：对齐弱是否伴随**过度自信/欠校准**。
 **输入**：`conf`（预测类别的置信度）或 `p_pos`（正类概率），`correct`，`cos_proj`。
 **步骤**：

1. 以 `cos_proj` 分上/下 30% 两组（High vs Low alignment）。
2. 对每组绘制 **可靠性图**：将概率按 10 等分 bin，绘制 bin 均值预测概率 vs 实际准确率；附 Brier score。
3. 两组并排展示或同图双曲线展示。
    **导出**：`fig5D_calibration_by_alignment.{svg,pdf,png}`
    **图注要点**：低对齐组的过度自信更明显（曲线低于对角线；Brier ↑）。

**边界情形**：若仅有 `conf`（“预测类概率”），仍可用于可靠性，但应在图注声明“非类条件概率”。

------

# E. 融合偏置指数（Fusion-bias Index：Δ = cos(ts,fuse) − cos(img,fuse)）

**目的**：衡量融合表示更“贴近”哪一模态；误分类是否更少贴近 TS。
 **输入**：`cos_ts_fusion`，`cos_img_fusion`，`correct`。
 **步骤**：

1. 计算 `delta = cos_ts_fusion - cos_img_fusion`。
2. 以 `correct` 分组作 **violin + beeswarm** 或 **箱线**；给出 U 检验 p 值与 Cliff’s δ。
3. 标注“中位数”横线与 `Δ` 的方向解释（右侧注释：Δ>0 更贴近 TS；Δ<0 更贴近 IMG）。
    **导出**：`fig5E_fusion_bias_delta.{svg,pdf,png}`
    **图注要点**：误分类组 Δ 左移/下降，提示融合对 TS 权重不足。

**边界情形**：若两组样本量差异大，beeswarm 以 `alpha`/jitter 调整视觉平衡。

------

# F. 二维响应面（Alignment × Confidence → Accuracy）

**目的**：展示“在相同置信度下，对齐更高→准确率更高”，证明对齐提供**超越置信度**的信息。
 **输入**：`cos_proj`，`conf`，`correct`。
 **步骤**：

1. 将 `(cos_proj, conf)` 放入二维网格（例如 30×30）；对每格计算 `mean(correct)`（样本<10 的格子设为 NaN 或使用 LOESS/样条平滑）。
2. 绘制 **热图 + 等高线**；颜色映射避免失真（默认 colormap），缺失格用浅灰。
3. 在图上标注典型等高线（如 ACC=0.9, 0.8）。
    **导出**：`fig5F_2D_response_alignment_confidence.{svg,pdf,png}`
    **图注要点**：等置信度切片上，准确率随对齐单调上升；对齐带来额外信息增益。

**边界情形**：若数据集中 `conf` 倾向集中在高值，建议对横轴采用分位数网格而非等距网格。

------

# G. 潜在空间密度等高线（t-SNE/UMAP + KDE）

**目的**：替代散点的“密度视角”，更清晰地呈现误分点位于类间低密度带。
 **输入**：二维嵌入 `z1,z2`（可直接使用你已有的 fusion t-SNE），`y_true`，`correct`。
 **步骤**：

1. 对每个类别在 `(z1,z2)` 上做 KDE，绘制**类条件密度等高线**（同一色不同线型）。
2. 叠加误分类样本（×记号、红框）和正确样本（小圆点；低透明度）。
3. 报告**类心距**（两类均值向量的欧氏距离）与**误分点到最近类心距离**分布（可在图内角标）。
    **导出**：`figS_density_contour_tsne.{svg,pdf,png}`
    **图注要点**：误分点聚集于低密度过渡区；类心距与整体错误率相关。

**边界情形**：KDE 带宽使用 `scott` 或 `silverman`；样本过少时，退化为二维 2D-hist + 等高线平滑。

------

# H. 决策曲线分析（Decision Curve；**可选**）

**适用前提**：需要**阳性概率** `p_pos`（而非预测类概率）。
 **目的**：以“对齐阈值触发复检”的策略评估**净获益（Net Benefit）**。
 **输入**：`p_pos`，`y_true`，`cos_proj`。
 **策略**：

- `All`：全做（无复检）。
- `None`：全不做。
- `Align-Gate`：若 `cos_proj < q*` 则“复检/进一步核实”，否则直接采用模型输出。
   **步骤**：

1. 对阈值 (t)（阳性判定阈）从 0.1–0.9 取一系列值；对每个 (t) 与每个策略计算 Net Benefit（Vickers & Elkin 2006 定义）。
2. 绘制 **Net Benefit vs t** 曲线，比较三条策略。
    **导出**：`figS_decision_curve_alignment.{svg,pdf,png}`
    **图注要点**：在临床相关阈值区间内，`Align-Gate` 曲线高于 `All/None`，说明实用价值。

**边界情形**：若仅有 `conf`，不建议做 DCA；或在图注声明其近似性与局限。