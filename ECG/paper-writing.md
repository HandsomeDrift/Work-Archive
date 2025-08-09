### 一、论文结构与要素

- **标题与作者信息**：
   标题简洁明了，作者信息需包含姓名、机构、收稿与修订日期。
- **摘要与关键词（Abstract & Index Terms）**：
   摘要简要描述研究目的、方法与结果，关键词一般4-6个。
- **正文结构（Sections）**：
   使用清晰分级的节（Section）与小节（Subsection）结构：
  - 一级标题使用罗马数字（I, II, ...）。
  - 二级标题使用大写字母（A, B, ...）。
  - 三级标题使用阿拉伯数字（1, 2, ...）。
- **插图与表格（Figures & Tables）**：
  - 图标清晰、标注完整，编号并提供描述性的标题。
  - 表格需置于正文中合适位置，并在上方注明标题（Caption）。
- **公式与数学表达式（Equations）**：
  - 使用 LaTeX 公式环境，统一编号（如(1), (2), ...）。
  - 复杂公式和矩阵表达使用相应环境（如align、matrix、bmatrix等）确保清晰。
- **算法描述（Algorithms）**：
  - 使用编号与明确标题展示算法流程，配合算法描述语句。
- **引用与参考文献（References）**：
  - 使用标准IEEE引用格式，文中使用`\cite`命令自动编号。
- **致谢与附录（Acknowledgments & Appendix）**：
  - 在正文结束后，简要致谢资助或帮助研究的个人和机构。
  - 附录适合展示补充推导或证明过程。

------

### 二、语言风格与排版要求

- 使用客观、准确、精炼的学术语言。
- 数学公式需遵循标准的数学排版风格，避免因格式错误而影响阅读体验。

------

### 三、你接下来撰写论文时应重点关注：

结合你目前的**多模态心电图分类模型**项目，建议的论文框架与示例如下：

- **Title**：
   体现方法与应用，例如：

  > A Dual-Modality GNN-Transformer Framework for ECG-based Myocardial Infarction Prediction

- **Abstract**：
   概括研究目的、创新点（跨模态学习、图神经网络、Transformer融合）、实验方法、主要结果及意义。

- **Index Terms**：
   ECG Classification, Graph Neural Networks, Vision Transformer, Multimodal Fusion, Contrastive Learning.

- **Introduction (I)**：
   阐述研究背景、临床问题的重要性，简要回顾现有方法不足，清晰说明研究目标与创新点。

- **Related Work (II)**：
   系统介绍：

  - ECG分类领域现状与挑战；
  - GNN及Transformer在医学领域的应用；
  - 跨模态融合方法及对比学习相关进展。

- **Proposed Method (III)**：
   细致描述整体架构：

  - A. 数据预处理与特征提取（包括GNN数据构建与图像特征抽取）。
  - B. 时序分支设计（GNN、FiLM注入）。
  - C. 图像分支设计（ViT-Large、FiLM）。
  - D. 融合与对比学习设计（融合Transformer、InfoNCE Loss）。

- **Experiments and Results (IV)**：

  - 实验设置：数据集描述、评价指标、训练策略与超参优化。
  - 实验结果：整体性能展示（如ROC、AUC、Accuracy等）。
  - 消融实验：验证各模块（FiLM、GNN、融合层）有效性与贡献。
  - 可视化分析：展示融合后的模态特征或跨模态对比效果。

- **Discussion (V)**：

  - 模型的优势与不足之处；
  - 参数不平衡对模型训练的潜在影响及如何优化；
  - 临床应用与实际部署的可能性探讨。

- **Conclusion (VI)**：

  - 总结研究贡献与主要成果；
  - 指出未来可进一步探索的方向。

- **Acknowledgments & References**：
   致谢相关资助项目和人员，列出规范的IEEE格式参考文献。

- **Appendix (optional)**：
   补充数学推导、算法细节、附加实验结果。

------

你后续撰写论文时，可以严格参照该模板的结构与格式规范，并重点突出你的研究特色（例如GNN与Transformer融合、跨模态特征对比、医学应用创新点），以确保论文质量与投稿成功率。



## Abstract

Accurate and timely diagnosis of cardiovascular diseases, particularly myocardial infarction (MI), remains a critical challenge in clinical practice. Traditional ECG analysis methods are often limited by their unimodal nature, lacking comprehensive physiological context.To address this limitation, we propose MIRAGE, a multimodal learning framework that combines ECG time-series, ECG images, and laboratory biomarkers to enhance diagnostic accuracy and robustness. MIRAGE employs a Graphormer encoder for modeling inter-lead dependencies in ECG signals and a Vision Transformer for extracting morphological patterns from ECG images, both modulated by lab features via Feature-wise Linear Modulation (FiLM). A Transformer-based fusion module integrates modality-specific features, while a contrastive learning objective ensures alignment across modalities. Experiments on a private dataset and three public benchmarks demonstrate that MIRAGE outperforms state-of-the-art baselines across metrics such as accuracy, F1 score, AUROC, and AUPRC. Ablation studies validate the contributions of each modality and architectural component. MIRAGE offers a clinically meaningful, patient-specific approach to cardiovascular diagnosis and provides a scalable framework for multimodal medical representation learning.

## I. Introduction

Cardiovascular diseases (CVDs) remain a leading cause of mortality worldwide, accounting for approximately one-third of all global deaths. Among these, myocardial infarction (MI), commonly known as a heart attack, poses a significant public health challenge. MI typically results from an obstruction in coronary blood flow, often due to arterial blockage, demanding prompt and accurate diagnosis for effective intervention. Timely detection is crucial not only to reduce mortality but also to mitigate severe complications such as heart failure, ultimately enhancing patient outcomes. The electrocardiogram (ECG) is widely recognized as an essential, non-invasive, cost-effective diagnostic tool due to its real-time monitoring capabilities, accessibility, and reliable diagnostic insights.

Despite its widespread use, conventional ECG analysis methods—particularly those relying on a single modality—exhibit notable limitations. These approaches are susceptible to noise, variability across patients, and often lack comprehensive systemic physiological context. Recent advancements in deep learning have substantially improved ECG classification performance; however, most models typically rely exclusively on either time-series data or static images. This singular focus neglects the potential synergies achievable by integrating multiple data modalities, thereby limiting the models' diagnostic accuracy and interpretability.

Recognizing this gap, multimodal learning has gained increasing attention in medical diagnostics. Laboratory tests, for instance, provide critical biochemical context unattainable through ECG alone, while ECG images convey essential structural and morphological insights that facilitate visual diagnostics. The integration of these complementary modalities promises richer, more informative representations of patient conditions, thus enabling more accurate and personalized clinical evaluations. Notably, multimodal learning approaches have demonstrated remarkable success across various domains beyond healthcare, such as natural language processing (NLP), autonomous driving, and human-computer interaction. For instance, integrating visual data with textual inputs has significantly improved model performance in tasks like image captioning and video analysis. Similarly, combining sensor data with visual perception in autonomous vehicles has greatly enhanced their environmental understanding and decision-making capabilities. These successes underline the potential benefits of adopting multimodal methodologies in complex diagnostic tasks.

Motivated by these clinical insights and successes in other domains, we propose **MIRAGE** (Multimodal Integration via Representational Alignment using Graphormer and Encoders), a novel multimodal framework specifically designed for ECG-based cardiovascular disease classification. MIRAGE synergistically integrates three complementary data modalities: 12-lead ECG time-series data representing electrical dynamics, ECG grayscale images highlighting morphological patterns, and laboratory test results reflecting systemic biochemical conditions.

The architectural choices within MIRAGE are intentionally tailored to address distinct clinical challenges. Recognizing that conventional time-series models inadequately capture inter-lead relationships inherent in 12-lead ECG data, we adopt a Graphormer-based encoder. By treating each ECG lead as a graph node, this approach explicitly captures the physiological interdependencies across leads, enhancing diagnostic accuracy and interpretability.

Complementing this, we incorporate a Vision Transformer (ViT) for ECG image analysis. This choice reflects clinical practice, where visual waveform interpretation provides crucial morphological cues. Unlike traditional convolutional methods, ViTs leverage global self-attention mechanisms, allowing superior capture of morphological patterns associated with conditions such as ST-segment elevation.

To further enhance the diagnostic power, MIRAGE introduces Feature-wise Linear Modulation (FiLM) layers within both encoding branches. Laboratory biomarkers such as troponin and D-dimer serve as modulation signals, dynamically adjusting feature representations according to individual physiological conditions. This strategy ensures patient-specific, context-aware feature extraction, significantly improving personalized diagnostic assessments.

To comprehensively integrate these diverse modalities, MIRAGE employs a Transformer-based fusion module. This module effectively bridges semantic gaps between modalities, modeling intricate cross-modal interactions among temporal, visual, and biochemical data. Furthermore, a contrastive learning strategy complements the standard classification task, encouraging coherence and alignment across multimodal representations.

In summary, MIRAGE's primary contributions include:

- A dual-branch encoder architecture combining Graphormer and ViT to effectively exploit temporal and spatial ECG data;
- FiLM-based modulation using laboratory biomarkers for personalized, context-rich feature extraction;
- A Transformer-driven fusion strategy that seamlessly integrates multimodal features;
- An integrated contrastive learning module that enhances multimodal representational coherence and alignment.

Through these innovations, MIRAGE addresses key limitations of traditional single-modality ECG approaches, providing a robust, interpretable, and clinically valuable solution for myocardial infarction diagnosis.

## II. Related Work

### A. ECG Classification: Progress and Challenges

Automated electrocardiogram (ECG) classification has received extensive attention due to the importance of timely and accurate diagnosis of cardiovascular diseases. Early studies primarily relied on hand-crafted features and traditional machine learning classifiers such as support vector machines (SVM) and random forests. With the advancement of deep learning, convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have become the dominant tools for end-to-end ECG classification. These models extract features directly from raw ECG signals, achieving state-of-the-art performance on public datasets like MIT-BIH and PTB-XL.

Despite this progress, single-modality approaches still face several limitations. Time-series models often struggle to capture subtle morphological variations, while image-based models may miss temporal dynamics crucial for accurate interpretation. Additionally, most public datasets lack clinical context, such as laboratory test results, limiting the generalizability of these models in real-world settings.

### B. Graph Neural Networks and Transformers in Medical Applications

Graph Neural Networks (GNNs) have demonstrated strong capabilities in learning structured representations from relational data. In the context of ECG, GNNs have been used to model the spatial dependencies among leads, allowing explicit encoding of inter-lead correlations. Recent work has shown that incorporating graph structures can significantly improve classification performance, particularly for multi-lead signals.

Transformers, originally developed for natural language processing, have gained traction in medical signal and image analysis due to their ability to model long-range dependencies through self-attention mechanisms. Variants such as Vision Transformers (ViT) have been successfully applied to ECG image classification, achieving superior performance over CNN-based baselines. In time-series tasks, Transformer-based models have outperformed RNNs in modeling sequential dependencies without requiring recurrence.

While both GNNs and Transformers show promise individually, their combination in a unified architecture—particularly for ECG data—remains underexplored. MIRAGE leverages this synergy by employing a Graphormer for temporal encoding and a ViT for visual representation.

### C. Multimodal Fusion in Medical Diagnosis

Multimodal learning has emerged as a powerful paradigm in medical AI, motivated by the fact that clinical decisions often rely on multiple heterogeneous data sources. Fusion strategies typically fall into three categories: early fusion (feature-level), late fusion (decision-level), and hybrid fusion (intermediate-level). Prior work has explored combining ECG signals with audio, imaging, or demographic features, but few have jointly modeled signal, image, and biochemical data.

Recent fusion models employ attention mechanisms, gating strategies, or cross-modal transformers to enable flexible interaction between modalities. However, challenges remain in maintaining modality-specific integrity while learning a unified representation. MIRAGE addresses this by using Transformer-based joint modeling alongside modality-specific encoders modulated by lab features.

### D. Contrastive Learning for Representation Alignment

Contrastive learning has gained popularity for its ability to learn robust and invariant representations without requiring extensive labeled data. In medical domains, contrastive objectives have been used for self-supervised pretraining of image encoders, for modality alignment in cross-modal tasks, and to improve robustness against distribution shifts.

Recent advances include the use of InfoNCE loss, supervised contrastive learning, and modality-specific projection heads. While some studies explore contrastive learning in multimodal medical imaging, its application to aligning ECG signal and image modalities remains scarce. MIRAGE integrates bidirectional contrastive loss to explicitly align the representations from its time-series and image branches, improving the coherence and complementarity of fused features.

In summary, MIRAGE builds upon prior work in ECG modeling, GNNs and Transformers, multimodal fusion, and contrastive learning, while introducing a unified and personalized approach for robust cardiovascular disease prediction.

## III. Methodology

We propose **MIRAGE** (Multimodal Integration via Representational Alignment using Graphormer and Encoders), a unified multimodal learning framework designed for ECG-based disease prediction. MIRAGE integrates three heterogeneous data modalities—12-lead ECG time series, ECG grayscale images, and laboratory test features—into a single cohesive architecture. It consists of three main components: (1) a FiLM-enhanced Graphormer for capturing temporal and inter-lead patterns from ECG signals while modulating them with clinical context, (2) a Vision Transformer (ViT) augmented with FiLM to extract visual features conditioned on lab-derived biomarkers, and (3) a cross-modal fusion module implemented via Transformer encoders that jointly optimize the integration of multimodal representations. Furthermore, MIRAGE employs contrastive learning to explicitly align signal and image modalities, leading to robust and context-aware fused representations for final classification. The overall architecture is designed to address the limitations of existing single-modality or weakly fused models by enabling deep interaction across temporal, spatial, and clinical axes.

### A. Data Preprocessing and Feature Extraction

Traditional ECG classification models predominantly rely on a single modality, such as either time-series signals or waveform images, and typically overlook additional clinical insights from complementary sources like laboratory test results. MIRAGE overcomes these limitations by integrating three distinct yet complementary modalities, creating a comprehensive representation of patient health:

$\mathcal{D}_i = \{\mathbf{X}_{\text{ECG},i}, \mathbf{X}_{\text{img},i}, \mathbf{x}_{\text{lab},i}, y_i\},$

where $\mathbf{X}_{\text{ECG},i} \in \mathbb{R}^{12 \times T}$ denotes the 12-lead ECG time series, $\mathbf{X}_{\text{img},i} \in \mathbb{R}^{1 \times 224 \times 224}$ represents the ECG grayscale image, $\mathbf{x}_{\text{lab},i} \in \mathbb{R}^{D_{\text{lab}}}$ is the laboratory feature vector, and $y_i \in \{0, 1\}$ is the binary diagnostic label.

The integration of these modalities is motivated by their inherent diagnostic complementarity and physiological interpretability in cardiovascular medicine:

- **ECG Time Series** ($\mathbf{X}_{\text{ECG}}$): Provide high-resolution temporal information reflecting cardiac electrophysiological activity. Each ECG lead offers distinct anatomical perspectives, enabling precise identification of abnormalities like ST-segment elevation, T-wave inversion, and arrhythmias. For example, characteristic ST-segment elevation in leads V2–V4 typically signals anterior myocardial infarction.
- **ECG Images** ($\mathbf{X}_{\text{img}}$): Capture a two-dimensional representation of ECG waveform morphology, preserving waveform shapes, amplitude relationships, and rhythm patterns in alignment with clinical visual interpretation practices. This visual modality enriches the model with structural and morphological insights not fully accessible through temporal signals alone.
- **Laboratory Features** ($\mathbf{x}_{\text{lab}}$): Offer essential systemic physiological context through biomarkers like troponin (cardiac injury indicator), D-dimer (marker of thrombosis), and white blood cell count (inflammation marker). These features provide critical supplementary clinical information, facilitating differential diagnosis even in scenarios presenting similar ECG patterns.

By integrating these three modalities, MIRAGE significantly enhances representation robustness. Temporal ECG signals encapsulate detailed electrophysiological dynamics, images deliver morphological context, and laboratory data add systemic physiological dimensions. This multimodal strategy mitigates common challenges such as signal noise, incomplete data, and ambiguous waveform interpretation often encountered with unimodal methods.

A distinctive innovation of MIRAGE is the explicit incorporation of laboratory features ($\mathbf{x}_{\text{lab}}$), which are notably absent from most publicly available ECG datasets. Incorporating lab-derived features allows patient-specific modulation of ECG representations through FiLM layers, greatly enhancing the specificity and adaptability of the model. For instance, patients exhibiting similar ECG waveforms can be effectively differentiated by their distinct troponin levels, which MIRAGE leverages to tailor representation encoding accordingly.

To operationalize this multimodal design, we developed a custom dataset featuring synchronized acquisition of all three modalities. This dataset facilitates precise alignment and integration of temporal, spatial, and biochemical information, marking both conceptual and practical advancements over existing unimodal and bimodal datasets.

Collectively, MIRAGE synthesizes cardiac electrophysiological data ($\mathbf{X}_{\text{ECG}}$), waveform morphology ($\mathbf{X}_{\text{img}}$), and systemic physiological status ($\mathbf{x}_{\text{lab}}$) into a unified and clinically aligned diagnostic representation ($\mathcal{D}_i$). This comprehensive approach significantly enhances prediction accuracy, model robustness, and clinical interpretability, particularly in complex cardiovascular classification scenarios.

### B. Time-Series Branch: Graphormer with FiLM

To model the structured spatiotemporal nature of 12-lead ECG signals, MIRAGE employs a Graphormer-based encoder augmented with Feature-wise Linear Modulation (FiLM). This design captures both inter-lead dependencies and patient-specific contextual variation.

Let $\mathbf{X}_{\text{ECG}} = [\mathbf{x}_1, \dots, \mathbf{x}_{12}] \in \mathbb{R}^{12 \times T}$, where each $\mathbf{x}_\ell \in \mathbb{R}^T$ represents the time series from lead $\ell$. Each lead is projected to a latent representation using a learnable linear transformation:

$\mathbf{h}_\ell^{(0)} = \mathbf{W}_{\text{proj}} \cdot \mathbf{x}_\ell + \mathbf{b}_{\text{proj}} \in \mathbb{R}^{d}, \quad \forall \ell \in \{1, \dots, 12\}.$

We then encode lead-wise spatial relationships via a Graphormer block, in which spatial biases are implicitly learned and incorporated into the attention mechanism. Specifically, the encoded node representations $\mathbf{H}^{(l)} \in \mathbb{R}^{12 \times d}$ are updated layer-wise using a graph-based self-attention operator:

$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top + \mathbf{B}}{\sqrt{d}}\right) \mathbf{V},$

where $\mathbf{B} \in \mathbb{R}^{12 \times 12}$ represents learnable spatial bias terms among the ECG leads.

To enable context-aware encoding, we apply FiLM modulation using laboratory features $\mathbf{x}_{\text{lab}}$. The FiLM generator maps lab vectors to scale and shift parameters:

$\gamma = f_\gamma(\mathbf{x}_{\text{lab}}), \quad \beta = f_\beta(\mathbf{x}_{\text{lab}}),$

which modulate intermediate features at each encoding layer:

$\mathbf{H}^{(l)}_{\text{FiLM}} = \gamma \odot \mathbf{H}^{(l)} + \beta.$

This mechanism injects personalized physiological information into the lead representations, enhancing patient-specific discrimination.

Finally, we extract a global token $\mathbf{z}_{\text{ts,cls}}$ by prepending a learnable [CLS] vector and applying an output projection:

$\mathbf{z}_{\text{ts,cls}} = \text{TransformerEncoder}([\mathbf{h}_{\text{cls}}; \mathbf{H}^{(L)}]) \in \mathbb{R}^{d}.$

This component outputs temporally and spatially enriched ECG embeddings, personalized by clinical laboratory context, for downstream fusion and classification.

### C. Image Branch: Vision Transformer (ViT) with FiLM

MIRAGE employs a Vision Transformer (ViT) architecture augmented by Feature-wise Linear Modulation (FiLM) to effectively capture morphological information from ECG images, integrating patient-specific clinical context derived from laboratory biomarkers.

Given an ECG grayscale image $\mathbf{X}_{\text{img}} \in \mathbb{R}^{1 \times H \times W}$, we partition it into $N$ patches, each of size $P \times P$. Each patch is flattened and linearly projected into a latent embedding space:

$\mathbf{z}_i^{(0)} = \mathbf{W}_{\text{patch}} \cdot \text{Flatten}(\mathbf{X}_i) + \mathbf{b}_{\text{patch}} \in \mathbb{R}^d, \quad i=1,\dots,N.$

We prepend a learnable classification token $\mathbf{z}_{\text{cls}}$ to these embeddings and incorporate positional encodings $\mathbf{E}_{\text{pos}}$ to retain spatial context:

$\mathbf{Z}^{(0)} = [\mathbf{z}_{\text{cls}}; \mathbf{z}_1^{(0)} + \mathbf{e}_1; \dots; \mathbf{z}_N^{(0)} + \mathbf{e}_N] \in \mathbb{R}^{(N+1) \times d}.$

To condition these representations on patient-specific clinical information, FiLM modulation parameters $\gamma^{(l)}$ and $\beta^{(l)}$ are computed from laboratory feature vectors $\mathbf{x}_{\text{lab}}$. These parameters modulate embeddings after each transformer encoder block:

$\mathbf{Z}^{(l)}_{\text{FiLM}} = \gamma^{(l)} \odot \mathbf{Z}^{(l)} + \beta^{(l)}.$

This process integrates personalized physiological insights into the visual encoding of ECG morphology, allowing the model to adjust feature extraction dynamically according to individual clinical contexts.

The final image representation $\mathbf{z}_{\text{img,cls}}$ is derived from the output classification token:

$\mathbf{z}_{\text{img,cls}} = \mathbf{Z}^{(L)}_{0} \in \mathbb{R}^{d}.$

Consequently, this branch effectively captures spatially rich morphological features from ECG images, contextualized by laboratory-derived biomarkers, significantly enhancing the multimodal representational power of MIRAGE.

### D. Fusion via Transformer Encoder

To effectively integrate the representations derived from the time-series and image branches, MIRAGE introduces a dedicated Transformer encoder-based fusion module. This component is designed to capture both modality-specific and cross-modal interactions, encompassing global diagnostic patterns and fine-grained temporal-spatial dependencies.

Let the following notations define the input to the fusion module:

- $\mathbf{z}_{\text{ts,cls}} \in \mathbb{R}^d$: global classification token from the time-series branch;
- $\mathbf{z}_{\text{img,cls}} \in \mathbb{R}^d$: global classification token from the image branch;
- $\mathbf{H}_{\text{ts}} \in \mathbb{R}^{12 \times d}$: lead-level representations from the time-series encoder;
- $\mathbf{H}_{\text{img}} \in \mathbb{R}^{N \times d}$: patch-level embeddings from the ViT image encoder.

The two global tokens $\mathbf{z}_{\text{ts,cls}}$ and $\mathbf{z}_{\text{img,cls}}$ are concatenated and projected through a learnable linear transformation to produce a unified multimodal token:

$\mathbf{z}_{\text{global}} = \mathbf{W}_f [\mathbf{z}_{\text{ts,cls}}; \mathbf{z}_{\text{img,cls}}] + \mathbf{b}_f \in \mathbb{R}^d,$

where $\mathbf{W}_f \in \mathbb{R}^{d \times 2d}$ and $\mathbf{b}_f \in \mathbb{R}^d$ are trainable parameters. This global token represents a synthesized summary of temporal and morphological information across modalities.

The fused token is then concatenated with the full sequence of lead- and patch-level embeddings:

$\mathbf{Z}_{\text{fusion}}^{(0)} = [\mathbf{z}_{\text{global}}; \mathbf{H}_{\text{ts}}; \mathbf{H}_{\text{img}}] \in \mathbb{R}^{(1+12+N) \times d}.$

This input sequence is fed into a stack of Transformer encoder layers that jointly model hierarchical dependencies and semantic interactions across modalities. These layers are responsible for learning attention-driven relationships that may span time-series leads, image patches, or combinations thereof:

$\mathbf{Z}_{\text{fusion}}^{(L)} = \text{TransformerEncoder}(\mathbf{Z}_{\text{fusion}}^{(0)}).$

The output at the first position of the final layer—corresponding to the fused global token—is extracted as the comprehensive multimodal representation:

$\mathbf{z}_{\text{fusion,cls}} = \mathbf{Z}_{\text{fusion}}^{(L)}[0] \in \mathbb{R}^d.$

This fused embedding encapsulates the full temporal, spatial, and clinical context, serving as the primary input for subsequent classification and contrastive learning modules.

### E. Contrastive Learning for Modality Alignment

To promote consistent and semantically aligned representations across modalities, MIRAGE incorporates a contrastive learning mechanism tailored for the multimodal ECG setting. This strategy enhances the coherence of the learned feature space by bringing together embeddings from different modalities that correspond to the same patient, while simultaneously pushing apart those from different patients.

Let $\mathbf{z}_{\text{ts},i}$ and $\mathbf{z}_{\text{img},i}$ denote the global classification token embeddings extracted from the time-series and image branches, respectively, for the $i$-th patient in a mini-batch of size $B$. The alignment objective is based on a symmetric InfoNCE loss, formulated as:

$\mathcal{L}_{\text{con}} = - \frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{\exp(\text{sim}(\mathbf{z}_{\text{ts},i}, \mathbf{z}_{\text{img},i})/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\mathbf{z}_{\text{ts},i}, \mathbf{z}_{\text{img},j})/\tau)} + \log \frac{\exp(\text{sim}(\mathbf{z}_{\text{img},i}, \mathbf{z}_{\text{ts},i})/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\mathbf{z}_{\text{img},i}, \mathbf{z}_{\text{ts},j})/\tau)} \right],$

where $\text{sim}(\cdot, \cdot)$ denotes cosine similarity, and $\tau$ is a temperature scaling factor that controls the sharpness of the distribution.

This bidirectional formulation ensures mutual agreement between the representations from both modalities. By minimizing this loss, the model learns to align ECG signal and image representations at the patient level, reinforcing semantic correspondence across heterogeneous views. This contrastive signal acts as a regularization force, complementing the supervised classification objective and contributing to more robust multimodal feature learning under clinical supervision.

### F. Classification Objective and Optimization

Final classification in MIRAGE is performed using the fused multimodal representation $\mathbf{z}_{\text{fusion,cls}}$, which captures global information integrated across time-series, image, and laboratory modalities. This embedding is passed through a linear classifier to produce class logits:

$\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}_{\text{cls}} \cdot \mathbf{z}_{\text{fusion,cls}} + \mathbf{b}_{\text{cls}}),$

where $\mathbf{W}_{\text{cls}}$ and $\mathbf{b}_{\text{cls}}$ are learnable parameters. The classification loss $\mathcal{L}_{\text{cls}}$ is computed using the cross-entropy criterion, with optional weighting schemes to address potential class imbalance.

In addition to classification, MIRAGE incorporates the contrastive loss $\mathcal{L}_{\text{con}}$ previously described in Section D, which aligns modality-specific embeddings in the latent space. This dual-objective training strategy allows the model to simultaneously optimize for predictive accuracy and representational coherence.

The overall training objective combines both components as follows:

$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}} + \lambda_{\text{con}} \cdot \mathcal{L}_{\text{con}},$

where $\lambda_{\text{cls}}$ and $\lambda_{\text{con}}$ are hyperparameters that balance the classification and contrastive learning contributions. This composite loss ensures that MIRAGE develops both discriminative and well-aligned multimodal representations tailored for robust clinical prediction.

------

### G. Summary of MIRAGE

MIRAGE offers a unified multimodal solution for cardiovascular disease prediction by integrating ECG time-series, ECG images, and laboratory test features. Through the combination of modality-specific encoders, FiLM-based patient-level modulation, and a Transformer-based fusion mechanism, the architecture is capable of capturing complex temporal, morphological, and physiological patterns. By incorporating contrastive learning into the optimization process, MIRAGE ensures semantic consistency across modalities, encouraging robust and interpretable feature representations. This integrated design not only enhances classification performance but also aligns well with clinical reasoning, thereby advancing the development of explainable and personalized diagnostic models.

## Experimental Design

### Data Preprocessing

To ensure robust feature extraction and reliable classification performance, we performed a systematic preprocessing pipeline on all ECG datasets, addressing common artifacts and standardizing the data format.

**Denoising.** Raw ECG recordings typically contain various types of interference, such as noise, baseline wander, and motion artifacts, which adversely affect classification accuracy [30]. To mitigate these disturbances, we applied a Butterworth bandpass filter with cutoff frequencies of 0.05 and 75 Hz [31], preserving essential physiological information while reducing unwanted noise.

**Downsampling.** All ECG signals were downsampled from the original 500 Hz sampling rate to 100 Hz. This reduction significantly decreases computational complexity without substantial loss of diagnostic information, a practice consistent with previous studies in ECG-based modeling [14, 32].

**Normalization.** To alleviate potential distribution shift effects, instance normalization [33] was applied independently to each lead of every ECG record. This step ensures consistency in amplitude scales across different recordings, thereby improving model generalization [10].

**Label Reconstruction.** Original SNOMED-CT codes assigned to each ECG record were converted into discrete categorical labels. After this mapping process, the Ningbo, PTB-XL, and Chapman datasets contained 25, 22, and 19 distinct classes, respectively. Notably, all datasets exhibited varying degrees of class imbalance, characterized by substantial disparities in the distribution of positive versus negative samples within certain categories, as well as significant variation in the number of samples across different classes.

Through this preprocessing pipeline, we standardized input quality and format, laying a solid foundation for subsequent feature extraction and classification tasks.

### Compared Methods

To comprehensively evaluate the effectiveness of our proposed MIRAGE framework, we compare it against several representative self-supervised and contrastive learning baselines for time-series modeling. 

- **TF-C (Temporal Feature Contrast)** [1]: A contrastive learning framework designed to extract temporally discriminative features from univariate and multivariate time series by contrasting representations of temporally shifted segments. It emphasizes capturing both global and local temporal dynamics.
- **TS-TCC (Time-Series Temporal Contrastive Coding)** [2]: This method constructs multiple augmented views of the same time series and applies contrastive objectives across different temporal perspectives. It enhances the model’s robustness by explicitly learning temporal invariances.
- **CPC (Contrastive Predictive Coding)** [3]: A predictive coding approach that maximizes mutual information between the current context and future latent representations. It encourages the encoder to retain predictive structure from the sequence, thus learning high-quality temporal representations.
- **TimesURL (Time-Series Unsupervised Representation Learning)** [4]: A unified pretraining framework that integrates multiple self-supervised objectives, such as context prediction and sequence reordering. It aims to extract generalizable representations applicable to diverse downstream tasks.
- **SimMTM (Simple Masked Time-Series Modeling)** [5]: It learns temporal dependencies by randomly masking subsequences and reconstructing them, offering simplicity and effectiveness in pretraining.
- **PatchTST** [6]: A pure Transformer-based forecasting model that partitions time series into non-overlapping patches. By utilizing global self-attention across patches, it achieves strong performance in capturing long-range temporal dependencies.
- **TimeMAE** [7]: A masked autoencoding framework tailored for time-series data. It reconstructs randomly masked segments using encoder-decoder architecture and has shown competitive performance in transfer learning settings.

These methods serve as strong baselines for benchmarking multimodal and unimodal ECG representation learning. 

### Implementation Details

Our model is implemented using PyTorch and trained on a single RTX 4090 GPU. During training, we set the batch size to 32 and employ 4 worker threads for data loading. The total number of training epochs is 50, with the initial learning rate set to $3.26 \times 10^{-5}$. A linear warm-up strategy is applied over approximately 13% of the total training steps , followed by cosine annealing for learning rate decay.

We optimize the model using the AdamW optimizer with default parameters and apply a loss function composed of two terms: a supervised classification loss and a contrastive alignment loss. The weighting coefficients for the classification and contrastive objectives are set to $\lambda_{\text{cls}} = 1.0$ and $\lambda_{\text{contrast}} = 0.445$, respectively. The temperature parameter in the contrastive loss is fixed at 0.1. The fusion module adopts a Transformer encoder structure with 3 layers, each consisting of 4 attention heads . The shared fusion representation has a dimensionality of 256. 

To address the issue of label imbalance present in the clinical dataset, we apply class weighting during the training process to stabilize optimization and improve the reliability of evaluation metrics.All baseline methods are trained using the recommended hyperparameter settings provided in their original implementations, ensuring a fair comparison with our model.

## Experiments

### Experimental Results on the SDU-SH Dataset

To comprehensively evaluate the diagnostic performance of MIRAGE in a practical clinical context, we conducted extensive experiments on the SDU-SH dataset, which comprises paired 12-lead ECG recordings and corresponding laboratory test results. For a thorough comparison, we selected various strong baseline methods, including supervised transformer variants (TF-C₁, TF-C_R), temporal contrastive learning models (TS-TCC₁, CPC), self-supervised learning frameworks (TimesURL, SimMTM_D, SimMTM_Q), and recent time-series transformer models (PatchTST, TimeMAE).

MIRAGE demonstrated consistently superior performance across all four evaluation metrics—accuracy (ACC), F1 score, AUROC, and AUPRC—surpassing the baseline models. Notably, MIRAGE achieved significant improvements in the F1 score and AUPRC, indicating robust predictive performance and strong capabilities for addressing class imbalance. Compared with top-performing baselines such as TS-TCC₁, MIRAGE exhibited clear advantages in AUROC and AUPRC, reflecting its superior ability to differentiate positive and negative clinical cases effectively.

While certain baselines, including TS-TCC₁ and CPC, exhibited strong individual performances, they lacked explicit incorporation of patient-specific physiological information and multi-modal fusion mechanisms. By contrast, MIRAGE leverages an early-stage FiLM modulation strategy that integrates laboratory biomarkers, modality-specific encoders, and cross-modal alignment through contrastive learning. This sophisticated design enables MIRAGE to effectively model patient heterogeneity and subtle diagnostic signals embedded within multimodal ECG data.

These findings highlight that MIRAGE not only competes strongly against existing state-of-the-art methods but also demonstrates unique strengths when applied to rich, clinically annotated multimodal datasets. Its design facilitates precise and personalized predictions, emphasizing its practical clinical value.

### Experimental Results on Public Datasets

To further validate MIRAGE’s generalizability under varied real-world conditions, we evaluated its performance on three publicly available ECG datasets: Ningbo, PTB-XL, and Chapman. Unlike the SDU-SH dataset, these public datasets do not include laboratory test features, necessitating a reduced variant of MIRAGE without laboratory-based modulation. This scenario allowed us to investigate the framework’s resilience when operating with incomplete modalities.

Even in the absence of laboratory data, MIRAGE delivered strong and competitive results across all datasets. On the Ningbo dataset, MIRAGE achieved the highest accuracy and F1 scores among the evaluated models, maintaining comparable AUROC scores relative to top-performing alternatives. Similarly, on the PTB-XL dataset, despite the lack of laboratory features, MIRAGE exhibited consistently robust accuracy and balanced performance across metrics. On the Chapman dataset, MIRAGE also achieved superior accuracy and F1 scores, demonstrating its reliable predictive capacity and generalization capability across datasets with varying distributions and label complexities.

Notably, several baseline models benefited from sophisticated self-supervised pretraining or specialized contrastive learning objectives. Nevertheless, even without complete multimodal input, MIRAGE maintained strong performance through its fundamental architectural strengths, including modality-specific encoders and structured cross-modal interaction. These features allow MIRAGE to extract robust and transferable representations from the available ECG modalities.

Overall, these experiments underscore the flexibility and robustness of MIRAGE, highlighting its capability to maintain strong performance even when faced with modality limitations. This further supports its practical applicability and adaptability for diverse real-world ECG classification scenarios.

### Ablation Study

To comprehensively evaluate the specific contributions of each component within the MIRAGE framework, we systematically conducted ablation experiments addressing both modality integration and structural design. These experiments provided detailed insights into how each element impacts the model's performance and robustness.

The model variants with specific modalities ablated are denoted as “w/o Laboratory,” “w/o IMG,” and “w/o TS,” corresponding to the exclusion of laboratory test features, ECG images, and ECG time-series data, respectively. The complete version of our model is referred to as the “Full Model.” In addition to modality ablations, we also investigate several architectural variations. The variant without contrastive learning is denoted as “w/o Contrast,” while the model without the cross-modal fusion module is denoted as “w/o Shared.” Moreover, we examine two alternative fusion strategies: incorporating laboratory features only at the fusion stage, referred to as “Late-Concat,” and employing a fully shared transformer fusion structure across all modalities, denoted as “All-Shared.” These ablation settings are designed to isolate the contributions of each modality and architectural component to the overall performance of the MIRAGE framework.

Initially, we investigated the role of incorporating multiple data modalities by individually removing them from the complete model. Eliminating the laboratory test features—which serve as personalized physiological context through FiLM modulation—resulted in noticeable performance degradation. This finding underscores the significance of integrating systemic biomarkers to enhance patient-specific modeling and improve classification accuracy. Similarly, when either ECG time series or ECG images were used exclusively, model performance significantly deteriorated compared to the multimodal baseline. Particularly, the absence of ECG time-series data markedly reduced performance, reflecting the critical role of electrophysiological dynamics, while the exclusion of ECG images confirmed the importance of morphological features. Thus, both temporal and morphological modalities offer unique and complementary diagnostic information that jointly enhances diagnostic effectiveness.

Furthermore, we examined the influence of key architectural elements by selectively removing or modifying them. Removing the contrastive learning objective negatively affected representation consistency and alignment across modalities, leading to less coherent multimodal representations and reduced classification capability. Similarly, substituting the Transformer-based deep fusion module with a simpler concatenation strategy resulted in decreased model performance, highlighting the necessity of complex cross-modal interaction mechanisms for effectively capturing intricate modality interdependencies. Additionally, delaying the integration of laboratory features until after the fusion stage, rather than embedding them through early FiLM modulation, significantly compromised the generalization and diagnostic power of the model. This emphasizes the importance of early-stage personalization to effectively influence subsequent feature representation. Lastly, replacing modality-specific encoders with a shared generic encoder structure severely degraded performance, clearly indicating the necessity of distinct architectural designs tailored specifically for each data modality.

Collectively, these ablation studies reveal that MIRAGE's high diagnostic performance and robustness do not arise from any single isolated component. Instead, the model achieves its superior results through the cohesive integration of multimodal input data, personalized feature modulation via FiLM, sophisticated structured fusion through transformer-based modules, and representation alignment through contrastive learning. Each of these components contributes distinctly yet synergistically to the overall effectiveness, clinical interpretability, and generalization capability of the framework.

### Parameter Sensitivity Analysis

To investigate the influence of critical hyperparameters on model performance, we conduct a focused sensitivity analysis on two key components of our training objective: the temperature parameter $\tau$ in the contrastive loss and the weighting factor $\lambda_{\text{con}}$ that balances the classification and contrastive objectives.

**Temperature Parameter $\tau$:**
 As shown in the results, the model exhibits considerable sensitivity to the temperature scaling in the contrastive loss. Lower values (e.g., $\tau = 0.05$) tend to produce sharper probability distributions, potentially leading to overfitting in representation alignment and performance degradation. Conversely, overly large values (e.g., $\tau = 1.0$) flatten the similarity scores, reducing the discriminative power of contrastive pairs. The model achieves optimal F1 performance when $\tau$ lies in the intermediate range of approximately 0.1 to 0.2, suggesting a balanced trade-off between contrast sharpness and generalization.

**Contrastive Loss Weight $\lambda_{\text{con}}$:**
 Varying the contribution of the contrastive loss reveals its importance in representation learning. With no contrastive component ($\lambda_{\text{con}} = 0$), the model relies solely on supervised classification, which leads to underutilization of cross-modal alignment. As $\lambda_{\text{con}}$ increases, the model benefits from enhanced modality interaction, yielding improved performance up to a point. However, excessively high values (e.g., $\lambda_{\text{con}} = 1.0$) begin to suppress the supervised signal, causing a slight decline in performance. These observations validate the necessity of balancing contrastive and classification signals, with the best performance achieved when $\lambda_{\text{con}}$ is set between 0.3 and 0.5.

This analysis highlights the critical role of careful hyperparameter tuning in contrastive learning-based multimodal frameworks. Optimal values of $\tau$ and $\lambda_{\text{con}}$ not only improve classification performance but also enhance the robustness of modality alignment, ultimately leading to more generalizable and reliable clinical predictions.



## V. Conclusion

In this study, we introduced MIRAGE, a comprehensive multimodal framework designed for effective and interpretable ECG-based cardiovascular disease prediction. By combining ECG time-series data, ECG images, and laboratory test results, MIRAGE addresses the limitations inherent in single-modality approaches, providing a more holistic and clinically meaningful representation of patient health.

MIRAGE integrates several innovative design elements. Modality-specific encoders were employed to accurately capture the distinct features of time-series signals and images. Additionally, we introduced Feature-wise Linear Modulation (FiLM) to embed patient-specific physiological information derived from laboratory tests into the encoding process, facilitating personalized representation learning. A Transformer-based fusion module was also incorporated to effectively model the rich interactions between different modalities, while contrastive learning further strengthened the semantic alignment across these diverse data sources.

Extensive experimental evaluations clearly demonstrate that MIRAGE achieves superior performance compared to various baseline models and ablation variants across multiple metrics. These results underline the importance of multimodal integration, early-stage physiological conditioning, and structured cross-modal fusion in improving diagnostic accuracy and enhancing model robustness.

Future research could explore several promising directions. Incorporating additional modalities, such as clinical notes, echocardiographic imaging, or genomic data, could further enrich the patient representations. Extending the framework to multi-label or multi-task scenarios would broaden its clinical applicability, potentially supporting more complex diagnostic tasks. Investigations into the interpretability of learned representations and real-world clinical implementation will also be valuable for enhancing clinical adoption.

Overall, this study highlights the potential of multimodal learning in medical AI applications and establishes a scalable and adaptable foundation for advancing cardiovascular diagnostics and clinical decision-making.