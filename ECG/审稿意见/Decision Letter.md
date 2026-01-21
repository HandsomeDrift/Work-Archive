[Skip to Main Content](https://ieee.atyponrex.com/submission/submissionBoard/c9d2b274-2c6f-406b-9a86-ce1b510caf90/decisionLetter#main)

[IEEE Author Portal](https://ieee.atyponrex.com/submission/dashboard)



Decision letter (Initial Submission)

JBHI-05302-2025

From:

fotiadis@uoi.gr

To:

ysgong@sdu.edu.cn

14-Oct-2025

JBHI Ref: JBHI-05302-2025
Reject/Resubmit (major revision and new external review required)

Dear Yongshun Gong,  

This letter is to inform you that the peer review process has concluded for manuscript, "Graph-informed and FiLM-enhanced Multimodal Fusion for Myocardial Infarction Prediction," which you had submitted for possible publication in the IEEE Journal of Biomedical and Health Informatics (J-BHI).

The Associate Editor responsible for your manuscript review has received feedback from independent reviewers and compiled their evaluations. It is the recommendation of the Associate Editor and the Editor-in-Chief that your manuscript requires a MAJOR REVISION before it can be accepted for publication in J-BHI. Please note the J-BHI Editorial Policy that only one major revision is allowed for any submitted manuscript. This means that if the review recommendation for your revised paper triggers another major revision, the paper will be rejected automatically.

Enclosed, please find the comments by the Associate Editor and all the reviewers. I hope that the feedback is helpful for further improving the quality of your manuscript. If you decide to resubmit a revised manuscript, this must be done within 10 weeks (not extendable) from the date of this message. Please quote the above manuscript reference number for all future correspondence.

***IMPORTANT: PLEASE RETURN HERE TO SUBMIT ALL FILES: https://ieee.atyponrex.com/journal/jbhi-embs**

Also note that it is mandatory to enter your replies to the reviewers' questions and indicate how you have dealt with their comments in the revised manuscript. Please include your replies in the authors' response section or a separate file with a point-by-point explanation of all the changes made. Please do NOT include it in the cover letter since this is not accessible by the reviewers. For submitting your revised manuscript, please ensure all changes are highlighted in the manuscript to facilitate the review process.

Citing/including papers suggested by the reviewers or the Associate Editor is up to the authors and only if they add value to your work. Please also report to fotiadis@uoi.gr any suspicious suggestion by the reviewers.

The authors are responsible to follow the J-BHI publication rules for maximum number of pages, quality of figures, etc. as they are mentioned in: https://www.embs.org/jbhi/prepare-and-submit-your-manuscript/
On the opposite your article might need to be returned to you and the review process will be delayed and in case of acceptance no publication is possible.

Please carefully read about how to prepare your manuscript and note the mandatory charge for over-length papers as imposed by IEEE in the website of J-BHI (https://www.embs.org/jbhi/prepare-and-submit-your-manuscript/).

Thank you very much for considering JBHI to publish your research work.

Sincerely,

Prof. Dimitrios I. Fotiadis
Editor-in-Chief

Cc: file

Associate Editor's comments to the authors:
Associate Editor
Comments to the Author:
Dear All,
I think this manuscript presents an interesting approach.

Please address comments from the reviewers, their main comments are around the methodology used and better description of the performance metrics, its generalisability and comparison with previous presented models.

Please consider improving Figure 2 (making it clearer as at the moment using small character size, difficult to see).

Please ensure that if and when resubmitting the paper does not exceed the page limit.

For submitting your revised manuscript, please ensure all changes are clearly highlighted in the manuscript and explained in detail in the rebuttal to facilitate the review process.

Reviewers' comments to the authors:
Reviewer: 1

Comments to the Corresponding Author
The paper presents GFM-MIP—Graphormer for 12-lead ECG time series, a ViT for ECG image morphology, FiLM conditioning with laboratory biomarkers, and a Transformer-based cross-modal fusion with contrastive alignment. The clinical motivation is sound, the pipeline is coherent, and results on both in-house and public datasets are encouraging. To further strengthen the contribution and reproducibility, a few clarifications and small adjustments would be helpful.

1. In Fig. 1, the TS and Image encoders on the left appear as single-layer blocks, whereas the implementation seems multi-layer. On the right, the schematic suggests TS is fed into both branches, while an ECG image is drawn but not clearly used there. The “Transformer Block” at bottom-right looks like multiple blocks, while Sec. III.G notes there are three. To avoid misinterpretation, it would be great to (i) indicate the actual number of layers/blocks (e.g., “×3”) in the caption and (ii) align the diagram with the implemented data flow.



2. Sec. III.G states the proposed model uses class weighting, while other methods follow their original hyperparameters. Since class weighting can materially affect performance on imbalanced data, could you clarify whether compatible baselines were also evaluated with class weighting (or an equivalent strategy)? A brief note or table would help readers assess fairness.



3. Several baselines are unimodal. It would help to clarify how missing modalities are handled for these methods on the multimodal dataset (e.g., ignored, placeholder inputs, or a projection/adapter). A short description of the protocol would make the comparison easier to interpret.



4. Table IV and Fig. 2 convey similar content. The figures seem to show that removing IMG reduces all four metrics by ~30% (AUPRC ~60%), whereas removing TS leads to ~10–20% decreases. This appears to differ from the statement in Sec. IV-G (para. 3) that the absence of TS markedly reduces performance. A brief reconciliation (e.g., confirming which condition drives the larger drop and why) would resolve the apparent discrepancy.


Reviewer: 2

Comments to the Corresponding Author

1. According to the ablation results in Table IV, removing the ECG image modality ("w/o IMG") leads to a catastrophic performance drop (ACC from 98.82% to 67.53%), which is far more severe than the impact of removing the ECG time-series modality ("w/o TS," ACC drops to 90.57%). This is an extremely counter-intuitive finding. The ECG time-series is the raw source of electrophysiological information, while the ECG image is merely one of its visual representations. Theoretically, the time-series should contain the most fundamental and complete information.
2. The authors fail to discuss or explain this critical anomaly anywhere in the text. This raises several serious questions: (i) How were the ECG images generated? Was additional information introduced or some form of signal enhancement performed during this process, making them easier for the model to learn from than the raw signals? (ii) Is there a flaw in the design or implementation of the time-series branch (Graphormer-based) that prevents it from effectively extracting diagnostic information from the ECG signals? (iii) Does this result imply that the framework's success is primarily driven by the Vision Transformer's ability to process images, rather than the "graph-informed temporal modeling" claimed by the authors?
3. The paper repeatedly emphasizes the model's "clinical significance" and alignment with "clinical reasoning," but lacks the evidence to support these claims. For instance, attention maps could have been visualized to show which ECG leads or image regions the model focuses on, or to demonstrate how the FiLM modulation parameters from lab data influence feature extraction in specific cases.
4. The t-SNE visualization (Fig. 5) demonstrates modality alignment but fails to provide deeper clinical insights. For example, what is the distribution of misclassified samples in the latent space? Is their cross-modal alignment weaker? Such analyses would greatly enhance the paper's impact.
5. The paper states that the Graphormer models inter-lead dependencies but does not explicitly define how the graph is constructed. Are the 12 leads treated as nodes in a fully connected graph where edge weights (spatial biases) are learned? Was any prior anatomical knowledge used to define the graph structure?


Reviewer: 3

Comments to the Corresponding Author
Summary: This paper proposes GFM-MIP, a multimodal fusion framework that integrates 12-lead ECG signals, ECG images and laboratory biomarkers for myocardial-infarction prediction. A Graphormer encoder captures inter-lead dependencies, a Vision Transformer extracts morphological features, and FiLM layers inject lab values into both branches; a Transformer-fusion module and contrastive loss complete the pipeline. Extensive experiments on one in-house and three public datasets show state-of-the-art accuracy, F1, AUROC and AUPRC, together with detailed ablations.
Concerns:

1. While the integration of the laboratory modality is practically valuable, the architectural contribution remains incremental, as the framework assembles well-established components—Graphormer, ViT, FiLM, and contrastive learning—without revealing new structural insights or design principles.
2. The method description employs FiLM and Graphormer but fails to cite their original papers, which is inappropriate for a technical publication.
3. The top results in Table I are obtained with the help of the additional laboratory modality; Table IV shows that “w/o Laboratory” drops below several baselines in Table I, indicating that the gain is largely modality-driven rather than architecture-driven.
4. The Late-Concat variant (Table IV) simply appends lab features to the fused representation, yet its performance drops dramatically compared with “w/o Laboratory”. Since the lab signal is beneficial (Full Model ↑), a mere concatenation should not yield such a large degradation; this observation is counter-intuitive and unexplained.