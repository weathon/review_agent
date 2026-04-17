# Similarity-Dissimilarity Loss for Multi-label Supervised Contrastive Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Supervised contrastive learning has achieved remarkable success by leveraging label information; however, determining positive samples in multi-label scenarios remains a critical challenge. In multi-label supervised contrastive learning (MSCL), multi-label relations are not yet fully defined, leading to ambiguity in identifying positive samples and formulating contrastive loss functions to construct the representation space. To address these challenges, we: (i) systematically formulate multi-label relations in MSCL, (ii) propose a novel Similarity-Dissimilarity Loss, which dynamically re-weights samples based on similarity and dissimilarity factors, (iii) further provide theoretical grounded proofs for our method through rigorous mathematical analysis that supports the formulation and effectiveness, and (iv) offer a unified form and paradigm for both single-label and multi-label supervised contrastive loss. We conduct experiments on both image and text modalities and further extend the evaluation to the medical domain. The results show that our method consistently outperforms baselines in comprehensive evaluations, demonstrating its effectiveness and robustness. Moreover, the proposed approach achieves state-of-the-art performance on MIMIC-III-Full.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the fundamental challenge of defining positive samples and contrastive loss in Multi-label Supervised Contrastive Learning (MSCL). Its main contributions are: (i) a systematic formulation of multi-label relations for MSCL, (ii) a novel Similarity-Dissimilarity Loss that dynamically weights samples, supported by (iii) theoretical analysis, and (iv) a unified framework for both single and multi-label contrastive learning. The method is empirically validated across image, text, and medical domains, achieving SOTA on MIMIC-III-Full. The work is well-motivated, provides a solid theoretical grounding, and demonstrates strong empirical performance.

### Strengths
1）It offers a clear and systematic formalization of multi-label relationships using set theory, providing a rigorous foundation for the field. 
2) The proposed Similarity-Dissimilarity loss is both simple and interpretable, with a mathematically bounded design that facilitates stable training and analysis.
3)  A major strength is the comprehensive theoretical proof, which grounds the method's properties and moves beyond mere empirical tuning. 
3) The extensive cross-modal and cross-domain validation, particularly the state-of-the-art results on the challenging MIMIC-III-Full benchmark, convincingly demonstrates the method's practical utility and robustness.

### Weaknesses
1. Although this paper innovates in the definition of relations and the design of the S–D loss, existing work (e.g., [ref 1] using set operations to synthesize multi-label samples in feature space; [ref 2] proposing label-aware contrastive weighting; and [ref 3] integrating label hierarchies into supervised contrastive) has also explored leveraging label relations to improve representation learning at different levels. Authors are advised to clearly and concisely explain the differences between this work and the new contributions of this paper in the introduction or related works.
[ref 1] Alfassy, A., Karlinsky, L., Aides, A., Shtok, J., Harary, S., Feris, R., ... & Bronstein, A. M. (2019). Laso: Label-set operations networks for multi-label few-shot learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition(pp. 6548-6557).
[ref 2] Suresh, V., & Ong, D. C. (2021). Not all negatives are equal: Label-aware contrastive loss for fine-grained text classification. arXiv preprint arXiv:2109.05427.
[ref 3] Chi Lok U, S., He, J., Gutierrez Basulto, V., & Pan, J. Z. (2023). Instances and labels: Hierarchy-aware joint supervised contrastive learning for hierarchical multi-label text classification.
2. The paper primarily compares ALL, ANY, and MulSupCon (Zhang & Wu, 2024). While these three are directly relevant baselines, several new methods have emerged in the past two years (≥2023–2025) in multi-label/multi-label contrastive learning and ICD encoding (including those based on hierarchical relationships, graph/knowledge injection, sample/label resampling, or meta-learning for long-tail processing), which may be more representative or powerful comparisons. The paper does not compare with these recent/relevant strong baselines (particularly in the areas of medical ICD encoding and LLM-based fine-tuning), which reduces the empirical persuasiveness.
3. Although the paper claims that the loss function is generalizable, it is not validated on more emerging tasks (such as multimodal multi-label and few-shot multi-label), nor is it tested on larger or more challenging datasets.
4. More empirical evidence on long-tail and small-sample scenarios: The paper claims that S-D has advantages in long-tail scenarios, but only provides macro- and micro-average metrics and overall explanations. It lacks quantitative comparisons and statistical tests (e.g., macro-F1 improvements per frequency band) by label frequency (head, medium, and tail). This weakens the conclusion that it "significantly improves the long-tail problem."

### Questions
1. Have the authors reviewed and compared the work [ref 1] that focuses on label set manipulation/synthesis, and the highly related label-aware/hierarchy-aware contrastive learning work [ref 2-3]? If so, please clearly identify the key differences between the two types of work in the main text or appendix. If not, please provide a comparison in the revised manuscript and explain the added significance and advancement of this work. 
2. Baseline Selection and "State-of-the-Art" Comparison:Why only compare with ALL, ANY, and MulSupCon (Zhang & Wu, 2024)? Can you add representative recent work (2022–2025) in the areas of multi-label contrastive learning, multi-label reweighting, long-tail multi-label classification, and ICD encoding for comparison?
3. The authors claim that the loss is uniform across single-label and multi-label scenarios (Equation 17). Can you demonstrate the numerical behavior of this loss in few-shot, multi-task, or extremely sparse label scenarios (1–2 labels per example)? For example, do state-of-the-art models such as ViT and Llama 3.2 also benefit from this loss?
4. Can you further analyze the improvement of this method on tail categories in extremely long-tail scenarios (such as MIMIC-III-Full)? Can you compare it with methods such as Focal Loss and Balanced Contrastive Learning? Please provide additional grouping results for the head/medium/tail frequency bands (e.g., for MIMIC and MS-COCO) and provide a significance test (p-value or bootstrap CI) to demonstrate substantial improvement on long-tail labels.

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
4

### Summary
This paper categorizes five fundamental multi-label relations (R1, R2, R3, R4, R5) in MSCL. Furthermore, it proposes a Similarity–Dissimilarity Loss to address the limitation of the prior method (Multi-label supervised contrastive learning), where the ANY relation fails to distinguish between R2 and R5.

### Strengths
1.	The paper provides a well-structured and comprehensive categorization of label relations in multi-label contrastive learning.
2.	The writing is clear and adheres to academic conventions.

### Weaknesses
1.	The paper tackles a relatively minor issue in multi-label contrastive learning—the inability to distinguish between R2 and R5—and existing multi-label contrastive methods can already address this. The prior work “Contrastive Learning for Multi-Label Classification” presents a conceptually similar solution: like the proposed similarity–dissimilarity factors, it operates by increasing the denominator of the positive-pair term.
2.	The Lemma 1 presented in the paper is not applicable to multi-label contrastive learning.

### Questions
The works cited in Lemma 1 pertain to single-label learning, so the resulting conclusion may be incorrect; accordingly, the stated equality may not hold. Moreover, it is unclear why this equality is derived, since you later write, “It is evident that R2, R3, R4 and R5 represent fundamentally distinct relations, each characterized by different labels and semantic information.” The logical connection between these statements is not evident.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper try to address a critical challenge in MSCL: how to define and distinguish positive samples. The authors argue that current methods, which either require exact label matches (ALL) or treat any label overlap as equally positive (ANY, MulSupCon), fail to capture the rich semantic relationships between multi-label sets.
To solve this, the paper discusses based on five relation patterns and propose a novel similarity-dissimilarity Loss:
It introduces 5 relations R1-R5 to formally categorize the relationship between an anchor and a sample (e.g., disjoint, exact match, partial overlap, subset, superset).
It proposes the loss function which dynamically re-weights positive pairs within the contrastive loss. This re-weighting is based on two intuitive, parameter-free factors: a similarity factor that rewards the proportion of shared labels and a dissimilarity factor that penalizes the number of extraneous labels in the positive sample.
The authors provide a theoretical analysis to prove their loss function is bounded why it works. Experiments across image, text, and complex medical datasets demonstrate that SimDis-Loss consistently outperforms baselines. The method is presented as a low-cost, drop-in replacement for existing MSCL losses.

### Strengths
1. Clear problem and challenging definition: The paper clearly shows the challenges in MSCL problems and illustrate the weakness of baseline methods. Using the five-relation taxonomy is a clear way to demonstrate the problem. 

2. Loss Function: The proposed loss function is a intuitive and proven to be effective both via theoretic proofing and experimental results. The core idea of separately accounting for similarity and dissimilarity is a novel and powerful concept. It directly addresses the identified flaw in baseline methods: MulSupCon, which the authors correctly identify as only considering the similarity (intersection) component.

3. Strong results compared to baseline methods: they show a consistent improvement compared to the baseline.

### Weaknesses
1. Insufficient Results & Analysis in Main Paper: The experimental validation in the main paper (Section 3) is too brief and lacks sufficient detail. The most critical comparison, the SOTA benchmark on MIMIC-III-Full (Table 6), is relegated to the appendix. The main paper should contain the strongest evidence of the method's efficacy, including key SOTA comparisons, to allow a reader to assess its performance without hunting through the appendix.

2. Limited Novelty: The paper's novelty appears to be limited. The core idea of re-weighting positives in a contrastive loss based on label overlap is not new. This work feels like an incremental improvement that combines existing concepts (contrastive learning + set-theoretic weighting) rather than a foundational new contribution.

3. Lack of Deep Analysis for the Loss Function: The paper fails to provide a deep justification for its core component, the dissimilarity function $\mathcal{K}^d = 1 / (1 + x)$. Why this specific form? The penalty is non-linear. The paper also relies on the strong, unproven assumption that any increase in the number of extraneous labels necessarily and consistently means the sample is "less positive." This intuition is not fully explored or validated.

4. Overstated Theoretical Contribution: The theoretical analysis (Section 2.6) feels overwrought. Theorems 1-5 are largely intuitive and immediate consequences of the definitions of $\mathcal{K}^s$ and $\mathcal{K}^d$; they are straightforward to prove and do not provide deep new insights. This space (nearly a full page) could have been more effectively used to provide the deeper loss function analysis or to move the critical experimental results from the appendix into the main paper.

5. Unclear Figures: The data visualization needs improvement. Figure 2a, in particular, is not an effective way to show a comparison. The stacked-bar "Improvement" chart obscures the direct comparison between the proposed method and the most relevant baseline (MulSupCon). A simple grouped bar chart, or even just including the raw data in Table 1, would be much clearer and more transparent.

### Questions
1. Asymmetric Definition: The current weighting factor $\mathcal{K}^s \mathcal{K}^d$ is asymmetric. It penalizes extraneous labels in the positive sample ($\mathcal{T} \setminus \mathcal{S}$) but not in the anchor ($\mathcal{S} \setminus \mathcal{T}$). This directly leads to R4 and R5 having different weights. What is the strong semantic justification for this? Have you experimented with a symmetric formulation (e.g., one based on the Jaccard index for similarity and a symmetric difference for dissimilarity)?

2. Embedding Visualization: The central claim is that your loss function learns to cluster samples based on the 5 relations. To provide strong qualitative evidence for this, could you please add an embedding visualization (e.g., t-SNE) of the learned feature space, with points colored by their relation (R1-R5) to a chosen anchor class?

3. Code Availability: The link to the code in the appendix is an anonymous placeholder and is not functional. Will a working repository be provided for reproducibility?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The manuscript system defines five kinds of multi-label relationships (R1-R5), and proposes Similarity-Dissimilarity Loss, which dynamically weights samples by similarity factor (label intersection ratio) and difference factor (extra label penalty). At the same time, it provides strict mathematical proofs (five theorems) to verify its rationality and boundary, and realizes a unified paradigm of single/multi-label supervision and comparison loss.

### Strengths
In this paper, a dynamic weighted loss function integrating similarity factor and difference factor is constructed to realize the unified paradigm of single/multi-label supervision and comparative loss, and the theoretical design has clear problem pertinence and academic value. 
Covering the multimodal data of "image-text-medical treatment" and considering variables such as "long tail distribution" in the medical field (MIMIC series), the experimental dimensions are comprehensive.

### Weaknesses
The selection basis of loss function superparameter is missing: the document sets the temperature parameter τ=0.07, but it does not explain why this value is suitable for multi-modal (image, text, medical) data, nor does it provide the sensitivity analysis of τ, so it is impossible to verify the robustness of superparameter selection. 
The ablation experiment is missing, and the value of key modules is not verified: the proposed loss function contains two core modules: similarity factor and difference factor, but the ablation experiment is not designed, so it is impossible to quantify the contribution of a single module to performance, and it is impossible to prove that the necessity of "dynamic weighting" mechanism is better than that of fixed weighting scheme. 
Missing key chart information: Figure 1 (five examples of multi-label relationships) only shows the one-hot vector representation of labels, but does not label the sample semantics corresponding to each relationship, which is poor in readability; Table 4 and Table 5 (the results of data sets such as MIMIC-III-50) have confusing typography (such as "AUC Macro Micro”“F1 Macro Micro Micro"), which affects the interpretation of the results. The data in the table was not analyzed.

### Questions
In the experiment of medical field, the manuscript only compares ALL, ANY and MulSupCon, and does not include the SOTA method for medical multi-label classification in recent years, and does not explain the reasons for excluding these baselines, why?

### Soundness
2

### Presentation
1

### Contribution
2
