# Epistemic-Aware Vision–Language Foundation Model for Fetal Ultrasound Interpretation

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Recent medical vision-language models have shown promise on tasks such as VQA, report generation, and anomaly detection. However, most are adapted to structured adult imaging and underperform in fetal ultrasound, which poses challenges of multi-view image reasoning, numerous diseases, and image diversity. To bridge this gap, we introduce FetalMind, a medical AI system tailored to fetal ultrasound for both report generation and diagnosis. Guided by clinical workflow, we propose Salient Epistemic Disentanglement (SED), which injects an expert-curated bipartite graph into the model to decouple view-disease associations and to steer preference selection along clinically faithful steps via reinforcement learning. This design mitigates variability across diseases and heterogeneity across views, reducing learning bottlenecks while aligning the model's inference with obstetric practice. To train FetalMind at scale, we curate FetalSigma-1M dataset, the first large-scale fetal ultrasound report corpus, comprising 20K reports from twelve medical centers, addressing the scarcity of domain data. Extensive experiments show that FetalMind outperforms open- and closed-source baselines across all gestational stages, achieving +14\% average gains and +61.2\% higher accuracy on critical conditions while remaining efficient, stable, and scalable.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FetalMind, a medical MLLM designed for diagnosis and report generation from fetal ultrasound images. The authors show that existing medical MLLMs and general purpose models perform poorly on complex fetal scans, while FetalMind incorporates a new approach called Salient Epistemic Disentanglement, which helps the model understand relationships between ultrasound views and diseases using a bipartite knowledge graph. The authors also present FetalSigma-1M, a large dataset of fetal ultrasound images and 20k clinical reports from 12 medical centers. Experiments show that FetalMind outperforms both medical MLLMs and general purpose models for diagnosing fetal ultrasound scans in various gestational stages.

### Strengths
- They introduce FetalSigma-1M, a large and diverse dataset covering all trimesters and multiple ultrasound views, which can help the community further advance diagnosis of fetal ultrasound scans.
- Their contribution like SED and Spatial alignment help the model outperform general purpose MLLMs validating domain aware contributions.

### Weaknesses
- Concatenating disease related salient view from one fetus to another one is not very indicative of real world scenario. There could be unique characteristics of the receiver’s fetus which does not correspond with the donor’s disease related images. It might help improve the limited benchmarks but could fail in real clinical cases. 
- It is unclear what level of effort is required to create and verify the bipartite knowledge graph, and whether this process can easily scale to new disease types.
- During the SVPO training the model might get biased towards the positive disease related salient views since they are concatenated with other normal views. In such a case, does the MLLM even require non salient views, can the model reach a similar accuracy by only using the salient views from the knowledge graph?

### Questions
Please address the weaknesses above
- Add examples of bipartite knowledge graph and its clinical verification process in appendix. 
- What instructions are given to clinical experts for diagnosis verification?
- Line 105: with 1B and 7b versions. 7B should be consistent.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FetalMind, a vision-language medical model tailored for fetal ultrasound report generation and diagnosis. The core contributions include a novel Salient Epistemic Disentanglement (SED) strategy that explicitly injects a bipartite disease-view knowledge graph into the model, combined with a preference optimization method (SVPO) using reinforcement learning to decouple disease-view associations in a clinically faithful manner. The model is trained on the FetalSigma-1M dataset, comprising over 1 million ultrasound images and 20,000 reports from 12 medical centers, which addresses the data scarcity in fetal ultrasound AI. Extensive multi-center experiments show strong gains over state-of-the-art medical and generalist multimodal LLMs, and include detailed ablations and visualizations.

### Strengths
1. The paper addresses the core challenges in fetal ultrasound analysis areas such as multi-view reasoning and disease diversity where existing models generally perform suboptimally. FetalMind is designed to tackle a genuine and pressing clinical need.
2.SED integrates expert prior knowledge (via bipartite graphs), powerful representation learning capabilities, and sophisticated alignment techniques (SVPO reinforcement learning). This workflow-mimicking design not only improves performance but also enhances the interpretability and reliability of the model’s reasoning process.
3. The introduction of the FetalSigma-1M dataset is a large-scale, multi-center, multi-view corpus in fetal ultrasound with fine-grained disease labels and standardized reports. This is likely to have enduring value for the community.
4. Extensive evaluation across gestational ages and multiple sites/devices establishes potential for generalization and robustness. Ablation studies are performed in depth, quantifying the impact of each component.

### Weaknesses
1. While the related work section describes several prior MLLMs and medical VLMs, there is insufficient comparison (both qualitative and, where possible, quantitative) with very recent medical vision-language foundation models, such as FetalCLIP (F Maani et al., 2025), EchoCLIP (M Christensen et al., 2024), and VividMed (L Luo et al., 2024). 
2. The paper presents SVPO as one of its contributions, yet its core loss function is identical to that of CPO (Xu et al., 2024). Compared with DPO, which removes the regularization supervision from the reference model, the ablation study does not directly provide a fair comparison between SVPO and DPO trained on the same data, thereby failing to isolate and validate the claimed advantage of "no reference model." Additionally, the reasons why training with DPO yields results inferior to Vanilla require further explanation.
3. The SVPO method generates training samples by switching significant views between different cases. While this represents an innovative data augmentation approach, it may also introduce unnatural or even anatomically implausible combinations. For example, it could combine a cardiac view from a 34-week fetus with an abdominal view from a 22-week fetus. The paper does not adequately address the potential negative impacts of such synthetic data, nor does it discuss how the model differentiates these artificial cases from real-world multi-lesion cases that, while complex, adhere to physiological logic. This raises concerns that the model might learn shortcuts for identifying splicing artifacts rather than acquiring genuine clinical reasoning abilities.
4. While strong positive examples are given, there is little explicit discussion of where the proposed approach fails or underperforms, nor are there error analyses.

### Questions
1. The expert-curated "disease-view" bipartite graph is central to the SED methodology. While the paper notes that this graph was constructed under the guidance of textbooks and experts, it lacks further specific details. For instance, what is the scale of this graph? How many disease and view nodes are included? How are discrepancies in expert opinions handled? Supplementing these details would enhance the reproducibility and transparency of the approach.
2. The paper proposed Fetal Token Injection to improve the model's ability to distinguish between clinically distinct but semantically similar terms. Could you provide more technical details on its implementation? For example, are these new tokens added to the language model's tokenizer? How are their embeddings initialized and trained? How does this mechanism technically enforce the clear separability you claim, beyond standard fine-tuning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FetalMind, a vision-language foundation model designed for fetal ultrasound report generation and diagnosis. The primary contributions are twofold: 1) The creation of FetalSigma-1M, a novel dataset of over 20,000 fetal ultrasound reports and 1.19 million images. 2) A new training methodology called Salient Epistemic Disentanglement (SED), which is guided by clinical workflows. Experiments show that FetalMind significantly outperforms a range of open-source and proprietary models on both diagnostic and report generation tasks.

### Strengths
1. This paper addresses the challenging task of fetal ultrasound interpretation, a domain that is not well-handled by current MLLMs.
2. This paper introduces FetalSigma-1M, a multi-center fetal ultrasound dataset.
3. This paper presents a thorough evaluation against a wide array of strong baselines with detailed analysis.

### Weaknesses
1. While the creation of FetalSigma-1M is a key strength, the paper indicates that only model weights, not the dataset, will be released. This limits the reproducibility of the work.
2. The method for evaluating baselines that lack native diagnostic capabilities involves using GPT to extract diagnoses from generated reports. This introduces a potential variable, as errors from the second step could unfairly penalize the baseline models.

### Questions
For the baseline models lacking native diagnostic capability, you employ GPT to perform diagnosis based on their generated reports.  Have you analyzed the potential error introduced by this two-step process? How can you ensure that the reported performance deficit of the baselines is not partly due to failures in this secondary diagnostic step? Is there a better way to evaluate these models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FetalMind, a multimodal vision–language foundation model for fetal ultrasound diagnosis. It aims to address the limitations of existing medical multimodal large models (e.g., LLaVA-Med, Med-Flamingo, BiomedGPT) that struggle with information redundancy and disease confusion in multi-view fetal imaging. The core idea is a Salient Epistemic Disentanglement (SED) mechanism combined with Salient View Preference Optimization (SVPO) and a bidirectional disease–view knowledge graph. Within a reinforcement learning framework, these components jointly refine the model’s multi-view reasoning to better align with real-world clinical workflows. In addition, the authors construct FetalSigma-1M, the first large-scale fetal ultrasound report dataset, covering 12 medical centers, 20,000 clinical reports, and over one million images. Experiments demonstrate that FetalMind surpasses GPT-5 and Gemini 2.5 Pro across both report generation and diagnosis tasks, achieving 98% accuracy in detecting nine major fetal abnormalities and an average +14% performance gain across different gestational stages.

### Strengths
1. The proposed SED mechanism integrates salient view selection and disease–view association learning, effectively disentangling key lesion information across multiple imaging views. It significantly reduces information collapse and disease confusion, offering strong innovation and interpretability from a clinical knowledge perspective.
2. The constructed dataset spans multiple gestational stages, medical centers, and over 300 disease categories, providing high representativeness and research value as a solid foundation for future fetal ultrasound AI studies.
3. FetalMind outperforms mainstream multimodal large language models (MLLMs), including GPT-5 and Gemini, in report generation, disease classification, and anomaly detection tasks. Detailed ablation studies further confirm the independent contributions of key modules such as Token Injection, Spatial Alignment, and SED.

### Weaknesses
1. Although the paper introduces a reinforcement learning–based optimization mechanism (SVPO), the core algorithmic derivation remains intuitive, lacking sufficient theoretical analysis or stability guarantees (e.g., convergence and optimality). In particular, the coupling between SED and SVPO is not formally explained.
2. While the model performs impressively on the FetalSigma-1M dataset, it lacks external multi-institutional validation or real prospective testing, leaving its clinical generalizability uncertain.
3. Although comparisons with DPO and GRPO are provided, the paper does not further discuss the design rationale of the reward signal or analyze how different reinforcement strategies affect stability and sample efficiency.

### Questions
1. When multiple pathologies coexist or imaging views contain noise, how does SED ensure that the model correctly identifies salient lesions rather than over-relying on biased views? Are there visualization results or quantitative metrics to support this claim?
2. Is the reward function in SVPO based on salient-view matching scores? If so, could the reward source introduce expert annotation bias? Is it possible to design an unsupervised or self-supervised variant?
3. How consistent is FetalMind’s performance across different ultrasound devices or manufacturers? Has any domain adaptation or feature standardization analysis been conducted?
4. Given the strong comprehension ability of general-purpose models such as GPT-5, how does FetalMind complement them? What is its potential for extension to multilingual or cross-modal clinical dialogue scenarios?
5. Can the model be seamlessly integrated into real clinical workflows? Does it incorporate privacy-preserving mechanisms (e.g., differential privacy) to enable deployment across multiple institutions?

### Soundness
3

### Presentation
3

### Contribution
3
