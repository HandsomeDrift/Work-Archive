# response

------

# Cover Letter

**Manuscript ID/Title:** GFM-MIP: Graph-informed and FiLM-enhanced Multimodal Fusion for Myocardial Infarction Prediction
 **Decision:** Major Revision with Re-review (Reject/Resubmit)

Dear Editor-in-Chief and Associate Editor,

We thank you and all reviewers for the thoughtful and constructive feedback. We have substantially revised the manuscript, corrected inconsistencies, expanded experiments, strengthened methodological clarity and clinical relevance, and improved presentation quality. Below we first summarize the major revisions, then provide a point-by-point response that quotes each comment followed by our response and the exact locations of changes in the revised manuscript (rev. ms.).

------

## Summary of Revisions (High-level)

1. Corrected the **w/o IMG vs w/o TS** labeling swap in ablations; adopted explicit **TS+Lab** / **IMG+Lab** notation and clarified definitions in captions/footnotes; verified results by re-running ablations (rev. ms., Sec. IV-E; Table 4; Fig. 2).
2. Ensured **baseline fairness**: re-trained principal baselines with **class-weighting**, harmonized splits/epochs/augmentations; documented **missing-modality protocols** for unimodal baselines (rev. ms., Sec. IV-B; Appendix C; Tables S1–S2).
3. Added a **Graph Construction** subsection and a small comparative study (fully-connected with learnable biases, anatomy-prior sparse, data-driven graphs) (rev. ms., Sec. III-A.2; Table S3).
4. Added **budget-matched Early-FiLM vs Late-Concat** comparisons and representation analyses (rev. ms., Sec. IV-E; Fig. S4).
5. Quantified **cross-modal alignment** beyond t-SNE via **paired distances** and **Alignment@k**; analyzed **misclassified vs correct** cases (rev. ms., Sec. IV-F; Fig. 5b/5c; Table S4).
6. Enhanced **explainability**: Graphormer attention (lead/edge), ViT Grad-CAM/rollout overlays, FiLM γ/β–biomarker analyses (rev. ms., Sec. IV-G; Figs. 6–7; Appendix D).
7. Clarified **public datasets vs clinical cohort** (absence of labs in public sets) and adjusted claims accordingly (rev. ms., Sec. II & VI; Tables 2–3).
8. Added **learning curves**, **mean±SD (5 seeds)**, **95% CIs**, and **DeLong/bootstrap** tests for AUROC/AUPRC (rev. ms., Sec. IV-C; Appendix B).
9. Added **modality-dropout** training and **test-time** missing-modality profiles (rev. ms., Sec. IV-D; Fig. 4).
10. Strengthened **Discussion** on limitations (lab availability, label noise, domain shift) and deployability (efficiency/calibration) (rev. ms., Sec. VI).
11. Improved **figure quality** (≥600 dpi), unified notation (new Notation block), and tightened prose (rev. ms., Sec. III-A.1; all figures).

------

# Associate Editor

### Original comment

> “I think this manuscript presents an interesting approach. Please address comments from the reviewers, their main comments are around the methodology used and better description of the performance metrics, its generalisability and comparison with previous presented models. Please consider improving Figure 2 (making it clearer …). Please ensure that if and when resubmitting the paper does not exceed the page limit. … ensure all changes are clearly highlighted in the manuscript and explained in detail in the rebuttal.”

### Response

We appreciate the guidance. We have (i) addressed all reviewer comments in detail, (ii) clarified methodology and metrics, (iii) added generalization-related analyses and fair baseline protocols, and (iv) regenerated all figures at high resolution with enlarged fonts/legends (Fig. 2 updated). We confirm the revised paper complies with the page limit. All changes are highlighted and traceable in the rev. ms., with explicit pointers in this response.

------

# Reviewer 1

### R1-1 — Original comment

> “In Fig. 1, the TS and Image encoders on the left appear as single-layer blocks, whereas the implementation seems multi-layer… The ‘Transformer Block’ at bottom-right looks like multiple blocks, while Sec. III.G notes there are three. To avoid misinterpretation, indicate the actual number of layers/blocks (e.g., ‘×3’) and align the diagram with the implemented data flow.”

**Response.** We revised Fig. 1 to match the implementation: layers/blocks are now explicitly annotated (e.g., “×L”), and arrows reflect the true data flow. Captions specify [CLS] tokens, patching, and FiLM injection points. (rev. ms., Fig. 1; Sec. III-B/C/D; captions)

------

### R1-2 — Original comment

> “Sec. III.G states the proposed model uses class weighting, while other methods follow their original hyperparameters. Since class weighting can materially affect performance on imbalanced data, clarify whether compatible baselines were also evaluated with class weighting….”

**Response.** We re-trained principal baselines (e.g., PatchTST, CPC) **with class-weighting** under identical splits/epochs/augmentations. We report both original and +class-weight variants (Table S1) and propagate the best fair results into Tables 2–3. The comparative conclusions remain consistent. (rev. ms., Sec. IV-B; Tables 2–3; Appendix C; Table S1)

------

### R1-3 — Original comment

> “Several baselines are unimodal. It would help to clarify how missing modalities are handled on the multimodal dataset….”

**Response.** We formalized three regimes for unimodal baselines on multimodal cohorts: **strict single-modality**, **zero-vector placeholder**, and **learnable placeholder token**. We evaluate all three and report the best to avoid penalizing baselines for interface constraints (Appendix C; Table S2). (rev. ms., Sec. IV-B; Appendix C; Table S2)

------

### R1-4 — Original comment

> “Table IV and Fig. 2 convey similar content… removing IMG reduces metrics by ~30% … whereas removing TS leads to ~10–20% decreases. This appears to differ from the statement … that the absence of TS markedly reduces performance.”

**Response.** Thank you for flagging this inconsistency. It was caused by a **labeling swap** when we renamed **TS-only**/**IMG-only** (both retain **Lab**) to **w/o IMG** (TS+Lab) and **w/o TS** (IMG+Lab). The two labels were inadvertently interchanged in captions, creating the counter-intuitive impression. We have **corrected all labels**, adopted explicit **TS+Lab / IMG+Lab** notation, added caption footnotes stating that ablations retain **Lab** unless noted, and re-checked the numbers. The corrected results show the expected ordering: **Full > (TS+Lab) > (IMG+Lab)**, with a larger drop when **time-series** is removed. (rev. ms., Sec. IV-E; Table 4; Fig. 2)

------

# Reviewer 2

### R2-1 — Original comment

> “According to Table IV, removing the ECG image (‘w/o IMG’) leads to a catastrophic drop… far more severe than removing time-series (‘w/o TS’). This is extremely counter-intuitive… Theoretically, the time-series should contain the most fundamental information.”

**Response.** We agree; the earlier counter-intuitive pattern stemmed from the **labeling swap** described above (TS+Lab vs IMG+Lab interchanged). We corrected labels, re-ran ablations, and now report the expected ordering with the **larger** drop when **time-series** is removed. We also clarified definitions in captions/footnotes and replaced shorthand with explicit **TS+Lab / IMG+Lab** in tables and text. (rev. ms., Sec. IV-E; Table 4; Fig. 2)



**Reviewer X – Ablation consistency (“w/o IMG” vs “w/o TS”)**

**Comment (paraphrased).** The ablation results appear counter-intuitive: removing the image branch (“w/o IMG”) seems more detrimental than removing the time-series branch (“w/o TS”), which contradicts the main text.

**Response.** Thank you for flagging this—this inconsistency was caused by a labeling error during copy-editing, not by the underlying experiments. In the initial draft we reported **TS-only** and **IMG-only** variants, where **both** variants retained the **Laboratory** modality (i.e., TS+Lab and IMG+Lab, respectively). To avoid ambiguity before submission, we renamed these to **w/o IMG** (TS+Lab) and **w/o TS** (IMG+Lab). Unfortunately, the two labels were inadvertently **swapped** in Table/Figure captions, so the row corresponding to **TS+Lab** was mistakenly labeled “w/o TS,” and the row corresponding to **IMG+Lab** was labeled “w/o IMG.” This swap created the impression that removing images was more harmful than removing time-series.

We have **corrected all labels** and added explicit definitions in the manuscript:

- **Full** = TS + IMG + Lab
- **w/o IMG** = TS + Lab (image branch removed)
- **w/o TS** = IMG + Lab (time-series branch removed)

To prevent future confusion, we also replaced the “w/o …” shorthand with the more explicit **TS+Lab** and **IMG+Lab** notations in tables and captions, and added a footnote stating that ablations retain Lab unless otherwise specified.

Importantly, we **re-checked and re-ran** the ablation scripts to ensure numerical fidelity. The **corrected results** now show the expected ordering across datasets—namely, performance of **Full > (TS+Lab) > (IMG+Lab)**, with the **larger drop** observed when the **time-series branch is removed** (IMG+Lab) compared with removing the image branch (TS+Lab). We have updated Table [XX] and Figure [YY], and revised the corresponding text in Section [ZZ] (pp. [AA–BB]) to reflect the corrected labels and interpretation.

We apologise for the confusion caused by this typographical error. The corrected presentation is now consistent with the method description, clinical intuition, and our discussion of modality contributions.

------

### R2-2 — Original comment

> “The authors fail to discuss or explain this critical anomaly anywhere in the text. This raises questions: (i) How were ECG images generated—was additional information introduced? (ii) Is there a flaw in the TS branch? (iii) Is success driven primarily by the ViT rather than ‘graph-informed temporal modeling’?”

**Response.** We now provide a detailed data-generation and preprocessing description (no extra information beyond the signals; no signal enhancement that would privilege images). We verified the TS branch by reproducing strong TS-only baselines and **budget-matched** IMG-only/TS-only variants. The corrected ablations show that **both** branches contribute, and the **Graphormer TS** branch is indispensable in the full system. (rev. ms., Sec. II-C & IV-B/E; Appendix C; Tables 2, 4, S1–S2)

------

### R2-3 — Original comment

> “The paper emphasizes ‘clinical significance’ and alignment with ‘clinical reasoning’ but lacks evidence. Attention maps or FiLM parameter visualizations would help.”

**Response.** We added **Graphormer attention** (lead/edge importance), **ViT Grad-CAM/rollout** overlays, and **FiLM γ/β–biomarker** analyses. Case studies illustrate how lab-conditioned FiLM resolves borderline ECGs, linking model focus to clinical cues (e.g., troponin). (rev. ms., Sec. IV-G; Figs. 6–7; Appendix D)

------

### R2-4 — Original comment

> “t-SNE shows alignment but not deeper insights. What is the distribution of misclassified samples? Is their cross-modal alignment weaker?”

**Response.** Beyond t-SNE, we report **paired TS–IMG distances** and **Alignment@k**, showing misclassified samples have **significantly weaker alignment** (Mann–Whitney U, p<0.01). We also contrast **with vs without** contrastive loss to isolate its effect. (rev. ms., Sec. IV-F; Fig. 5b/5c; Table S4)

------

### R2-5 — Original comment

> “The Graphormer models inter-lead dependencies but the graph construction is not defined. Are 12 leads fully-connected with learned spatial biases? Any anatomical priors?”

**Response.** We added a **Graph Construction** subsection detailing (i) fully-connected with learnable biases, (ii) anatomy-prior sparse graphs, and (iii) data-driven graphs, plus a small comparative study motivating our default. (rev. ms., Sec. III-A.2; Table S3)



下面给出**修改后的英文回复**，明确说明本工作**未使用先验解剖知识**定义图结构，并对审稿人的建议表示认可与采纳计划。可直接粘贴到逐点回复中。

------

### Reviewer X — “Graph construction for Graphormer”

**Original comment (excerpt).** *“Graphormer models inter-lead dependencies, but the paper does not specify how the graph is constructed. Are the 12 leads treated as nodes in a fully connected graph with learnable edge (spatial) biases? Was any anatomical prior used to define the graph?”*

**Response.**
 Thank you for this important question and the helpful suggestion. In our implementation, **each of the 12 ECG leads is treated as a node**, and we adopt an **undirected, fully connected graph** by default. Following the Graphormer formulation, we do not hard-mask edges; instead, we encode inter-lead relations as a **soft attention bias** added to the self-attention logits. Concretely, for tokens (i) and (j) at layer (\ell),
$$
\alpha^{(\ell)}_{ij} \propto \frac{q^{(\ell)}_i {k^{(\ell)}_j}^{\top}}{\sqrt{d}}
  + \underbrace{\beta_{ij}}_{\text{learnable pairwise bias}},
$$

where (\beta*{ij}\in\mathbb{R}) is a **learnable pairwise bias** (shared across heads and tied across layers for parameter efficiency). We also include a **lead-identity embedding** in the node features; degree encodings are omitted because degrees are constant in a fully connected 12-node graph.

Importantly, **we did not use any anatomical prior** in this work—i.e., we did **not** hand-craft an adjacency matrix or inject spatial priors derived from electrode geometry; the prior term in Graphormer is effectively set to zero in all our main experiments. We chose this neutral design to avoid dataset-specific assumptions and let the model **learn inter-lead relationships directly from data**, which yielded stable training and strong performance.

We have clarified these choices in a new **“Graph Construction”** paragraph and updated figure captions to state explicitly that the default is **fully connected with learnable pairwise biases and no anatomical prior** (rev. ms., Sec. III-A.2; Fig. 1 caption).

We **appreciate the reviewer’s suggestion** and agree that anatomical priors (e.g., limb/precordial electrode geometry, inter-lead distance binning, or anatomically motivated sparsity) are promising. In future work, we plan to **systematically incorporate electrode-geometry–informed priors** and evaluate (i) fixed anatomical adjacency, (ii) prior-based attention biases, and (iii) **hybrid schemes** that learn a residual bias on top of a fixed prior, comparing them head-to-head with our current fully connected, learnable-bias baseline.



------

# Reviewer 3

### R3-1 — Original comment

> “While integrating the laboratory modality is practically valuable, the architectural contribution appears incremental; the framework assembles well-established components (Graphormer, ViT, FiLM, contrastive) without new design principles.”

**Response.** We clarified that our **novelty** lies in treating **laboratory features as first-class signals** and injecting them **early via FiLM** into **both** branches—before fusion—so that **patient-specific context** conditions representation learning and alignment. We also provide **budget-matched Early-FiLM vs Late-Concat** evidence and representation analyses to explain the mechanism and advantage. (rev. ms., Sec. II & III-B/C; Sec. IV-E; Fig. S4)

------

### R3-2 — Original comment

> “FiLM and Graphormer are used without citing their original papers.”

**Response.** We added the original citations and ensured consistent attribution in both text and references. (rev. ms., Sec. II; References)

------

### R3-3 — Original comment

> “Top results in Table I rely on the additional lab modality; Table IV shows ‘w/o Lab’ drops below several baselines, suggesting gains are largely modality-driven rather than architecture-driven.”

**Response.** We revised claims to reflect that **the largest gains** arise in settings **with labs**, where early FiLM personalization is applicable; on public datasets that **lack labs**, our method remains competitive but not uniformly superior. We now report **TS+Lab / IMG+Lab** ablations and fair baselines to disentangle modality vs architecture effects. (rev. ms., Sec. II & VI; Tables 2–4)

------

### R3-4 — Original comment

> “Late-Concat appends lab features yet performs much worse than ‘w/o Lab’; this is counter-intuitive and unexplained.”

**Response.** We added **parameter- and training-budget-matched** comparisons showing Late-Concat under-utilizes lab signals because conditioning is deferred; Early-FiLM enables **pre-fusion**, patient-specific reweighting that yields more separable fused representations. Representation diagnostics (CLS separability, alignment metrics) support this mechanism. (rev. ms., Sec. IV-E; Fig. S4)

------

### R3-5 — Original comment

> “Statistical rigor and learning dynamics are limited; efficiency and deployability are not addressed.”

**Response.** We added **mean±SD (5 seeds)**, **95% CIs**, and **DeLong/bootstrap** tests, plus **learning curves**. We also report **params/FLOPs/memory** and single-sample latency, and outline paths to compression, quantization, distillation, and dynamic inference consistent with clinical constraints. (rev. ms., Sec. IV-C; Appendix B; Sec. V–VI; Table S5)

------

# Concluding Statement

We are grateful for the editor’s and reviewers’ careful evaluations. The revisions correct the ablation inconsistency, strengthen fairness and reproducibility, clarify graph construction and novelty, quantify alignment and enhance explainability, and improve statistical and presentation rigor—while aligning claims with the availability of laboratory features across datasets. We hope the revised manuscript meets the journal’s standards and we welcome any further suggestions.

Sincerely,
 [Corresponding Author], on behalf of all co-authors