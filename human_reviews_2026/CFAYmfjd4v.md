# Empowering Test-Time Adaptation with Complementary Vision-Language Knowledge in Open-World Scenarios

- Avg Score: 3.20
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2, 6

## Abstract
Test-time adaptation in open-world scenarios (OWTTA), which addresses both domain discrepancy and semantic variance, has gained increasing attention for enabling models to adapt dynamically during inference. Existing approaches mainly rely on discriminative models, whose over-specialized knowledge restricts their adaptability in open-world settings. In contrast, vision-language models (VLMs), trained on diverse large-scale data, provide broader and more transferable knowledge, yet their role in OWTTA remains underexplored.
In this work, we propose a framework empowered by vision-language models, termed Vision-Language knowledge Boosted Open-world test-time adaptation (VLBO).
Specifically, by casting OWTTA into a probabilistic perspective, we first propose agreement-boosted filtering (AF), in which the discriminative model assumes the primary role of filtering out out-of-distribution samples, while the VLM provides a reinforcing signal to refine this process based on its agreement with the discriminative model. We then introduce semantics-boosted adaptation (SA), where VLM-extracted representations serve as semantic guidance to enhance the discriminative model’s adaptation to target domains.
This unified framework leverages the complementary strengths of vision-language models and discriminative counterparts, enabling robust and effective adaptation in open-world scenarios. Extensive experiments across multiple benchmarks demonstrate the consistent effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method to address the challenge of open-world test-time adaptation. Conceptually, the approach integrates the vision-language model (VLM) CLIP with a discriminative model (e.g., ResNet) to leverage their complementary strengths. Technically, the proposed framework consists of two key components: 1. Agreement-boosted filtering, which employs VLM guidance to enhance the out-of-distribution (OOD) detection capability of the discriminative model; 2. Semantic-boosted adaptation, which utilizes the semantic representations from the VLM to improve the discriminative model’s adaptation during test time. The effectiveness of the proposed method is validated across multiple standard benchmarks, demonstrating consistent improvements in open-world adaptation performance.

### Strengths
- The proposed idea is intuitive. The paper clearly articulates the complementary strengths of **ResNet** and **CLIP** at the conceptual level, followed by two well-defined components that effectively integrate their benefits at the technical level.
- The experimental evaluation is comprehensive, covering both **corruption** and **style-transfer** domain shifts across multiple **out-of-distribution (OOD)** benchmarks. The proposed **VLBO** method consistently outperforms baselines across all datasets and evaluation metrics.

### Weaknesses
The paper’s **problem setting and motivation** are not clearly established in the Introduction. The connections between the proposed method, existing approaches, and the broader context of **open-world test-time adaptation (OWTTA)** are vague, making the logical flow difficult to follow. Readers may find the introduction confusing until reaching the Method section, where the framework becomes clearer.

For example, in **Line 44**, the statement __“OWTTA aims to enable models to adapt to the target domain without requiring label data or source statistics”__ may be misleading—it appears to describe domain adaptation, whereas later sections (e.g., **Section 3.1** and **Figure 1**) reveal that OWTTA also addresses **novel-class scenarios**. Additionally, the paper does not clarify how the concepts of **in-distribution (ID)** and **out-of-distribution (OOD)** relate to **domain discrepancy** and **semantic variance**, despite these being central to the problem.

Between **Lines 47–55**, the relationship between **TTA** and **OWTTA** is underexplained. It remains unclear whether OWTTA is a special case of TTA, what their respective challenges are, and how existing TTA methods motivate the proposed OWTTA approach. Similarly, **Lines 56–65** introduce several technical elements (e.g., similarity measures, data filtering, probabilistic formulations) without adequate explanation or conceptual grounding, making it difficult to understand how each component contributes to performance.

To improve readability, the introduction should include a **conceptual overview diagram** summarizing the overall framework and its components. Each component should be introduced following a clear structure: **objective (why)**, **conceptual explanation (what)**, and **technical design (how)**. Overall, the current introduction feels underdeveloped and should be **substantially rewritten** to properly motivate the problem, clarify the conceptual landscape, and prepare readers for the method section.

### Questions
- Could the authors clarify the exact problem formulation of OWTTA? Does it focus solely on domain shifts, or also include novel-class scenarios?

- In Line 44, the statement “OWTTA aims to enable models to adapt to the target domain without requiring label data or source statistics” seems to suggest a domain adaptation setting. Could the authors explain how this differs from or extends standard domain adaptation?

- How do the authors define in-distribution (ID) and out-of-distribution (OOD) in the context of this paper? Are these terms related to input distribution, label distribution, or both?

- Is OWTTA a special case or an extension of TTA?

- How do existing TTA methods inspire or motivate the proposed OWTTA approach? Please provide explicit connections or differences.

- Could the authors provide a diagram summarizing the overall framework, highlighting the role and interaction of each component?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses test-time adaptation in the open-world problem (OWTTA) by leveraging vision-language models (VLMs). Specifically, for improved out-of-distribution (OOD) detection, it employs a discriminative model to filter out OOD samples and further refines this process using VLMs. Then, VLMs are used to enhance the adaptation of discriminative models. Extensive experiments across diverse benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1. The method and motivation of this paper are clearly presented and easy to understand.
2. The observations in OWTTA are quite interesting and form the foundation for the subsequent methods.
3. This OWTTA setting closely resembles real-world scenarios.

### Weaknesses
1. In Section 3.2, Observation 1, there is no explicit conclusion stating that samples “tend to be distributed closer to their class prototypes than OOD samples,” yet this statement is used in line 230. Could the authors add this conclusion to Observation 1 to make it easier for readers to comprehend?

2. For table 1, if the model was trained on CIFAR-10 (source) and is tested on MNIST (target), and MNIST has no overlapping classes with CIFAR-10, how is ACC_I (in-distribution accuracy) measured in this case? Could the authors provide more details about this experiment?

3. In the ablation study, for the agreement-boosted filtering module, there is no experiment evaluating the effect of using only the VLM or only the discriminative model to filter OOD samples. Such experiments could demonstrate the effectiveness of combining the two models. Additionally, for the semantic-boosted adaptation module, it would be helpful to explore what happens if the predictions of the two models are simply added rather than multiplied.

4. The paper does not include experiments on time or memory analysis, which are quite important in the OWTTA scenario. Introducing an additional model like the VLM may increase both computational time and memory usage, so it would be helpful to evaluate these costs.

### Questions
Please see the above weaknesses. If you can conduct additional experiments to further evaluate your method, I would be willing to raise my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces VLBO (Vision-Language knowledge Boosted Open-world Test-Time Adaptation), which integrates a discriminative model (ResNet) and a vision-language model (CLIP) for open-world test-time adaptation (OWTTA).  
The method contains two modules for two phases (OOD detection and TTA):  
(1) Agreement-Boosted Filtering (AF), which fuses the confidence of ResNet and CLIP to filter OOD samples;  
(2) Semantic-Boosted Adaptation (SA), which fuses the logits of both models via a product-of-experts formulation to adapt the ResNet.  
Experiments under the OWTTA setup show that VLBO achieves better performance than ResNet-based TTA baselines and CLIP-based OOD detection baselines.

### Strengths
1. The work addresses the practical open-world test-time adaptation setting, which combines OOD detection and TTA.
2. The ablation study is clear and isolates the contribution of each module.

### Weaknesses
1. "Agreement-Boosted Filtering" and "Semantic-Boosted Adaptation" correspond to simple confidence score averaging and logit summation; there seems to be little theoretical or algorithmic innovation.
2. The empirical “Observations 1–3” (e.g., ResNet encodes task-specific features, CLIP is more robust to distribution shifts) are already well-known and serve more as narrative justification than new insights.
3. OWTTA can be viewed as TTA + OOD detection. VLBO includes two separate parts designed for OOD detection and TTA phases, so the comparison with TTA baselines and OOD detection baselines appears not entirely fair. Would a SoTA OOD detection method combined with a SoTA TTA method be a fairer baseline?
4. There seems to be no discussion of computational overhead or latency, which matters in test-time settings.

### Questions
1. To my understanding, OWTTA here is TTA plus OOD detection. There have been two lines of research exploring the two problems separately. To address OWTTA, one can simply combine an OOD detection and a TTA method as two sequential phases for each test step. Would the authors agree with this point? As mentioned in point 3 of Weaknesses, would a SoTA OOD detection method plus a SoTA TTA method be a fairer baseline?
2. The paper does not appear to clearly explain how the baselines are evaluated under the OWTTA setting. OOD detection baselines are designed for detection and do not seem to perform adaptation. It is unclear whether they are re-implemented with an adaptation phase, or simply applied in a frozen zero-shot manner.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces VLBO, an open-world test-time adaptation (OWTTA) framework that leverages a vision-language model (i.e., CLIP) to enhance model robustness under both domain discrepancy and semantic variance. VLBO comprises two key components: (1) Agreement-boosted Filtering (AF), which identifies and filters out out-of-distribution (OOD) samples by combining the confidence of a discriminative model (ResNet) with image–text similarity cues from CLIP; and
(2) Semantic-boosted Adaptation (SA), which adapts the discriminative model using CLIP’s semantic priors through a product-of-experts formulation. Experiments on five open-world benchmarks (i.e., CIFAR10-C, CIFAR100-C, ImageNet-C, ImageNet-R, and VisDA-C) demonstrate consistent performance gains over both discriminative and vision-language model baselines.

### Strengths
+ OWTTA represents an important and practical research challenge. This paper is one of several concurrent studies that demonstrate the effectiveness of incorporating CLIP into OWTTA settings.

### Weaknesses
- The novelty of the proposed method appears limited. Specifically, the AF module is essentially a straightforward extension of conventional confidence-based OOD filtering, where CLIP-based similarity is incorporated as an auxiliary confidence measure. Similarly, the SA module largely follows standard domain adaptation techniques based on pseudo-labeling of target data. Overall, the method does not introduce a fundamentally new learning principle or a theoretically grounded formulation.
- As discussed in §2, several prior studies have already explored CLIP-guided OOD detection and adaptation. However, the paper does not clearly articulate the motivation for VLBO or its conceptual distinctions from these existing approaches. Moreover, some recent relevant methods (e.g., [1-2]) are missing from the experimental comparisons (§5, Tables 1-4).
- Algorithm 1 appears oversimplified. For example, the definition and role of T are unclear: does it refer to adaptation steps, test-time batches, or epochs?
- The relationship between OWTTA and Open-Set Source-Free Domain Adaptation (OS-SFDA) (e.g., [3]) is not clearly discussed. Although these two settings may not be identical, they are closely related; both address adaptation under distribution shifts with the presence of unknown target classes. It would be beneficial for the paper to explicitly clarify their conceptual differences and connections in §2. In addition, including comparisons with representative OS-SFDA methods in §5 would provide a more comprehensive evaluation of the proposed approach.

[1] C. Cao et al., Noisy Test-Time Adaptation in Vision-Language Models, in ICLR, 2025.

[2] D. Osowiechi et al., WATT: weight average test time adaptation of CLIP, in NeurIPS, 2024.

[3] F. Wan et al., Unveiling the Unknown: Unleashing the Power of Unknown to Known in Open-Set Source-Free Domain Adaptation, in CVPR, 2024.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles open-world test-time adaptation where both domain discrepancy and semantic variance appear at inference. It proposes VLBO, a framework that augments a discriminative backbone with vision–language knowledge from CLIP in two stages. Experiments on different datasets show consistent gains in the harmonic mean ACCH over strong TTA and CLIP baselines.

### Strengths
- Well-motivated decomposition (AF + SA): AF leverages discriminative, task-specific geometry while SA imports transferable semantics via PoE; this matches their observations and improves robustness.
- Clear probabilistic grounding for AF: mixture-of-Gaussians view with a practical k-means limit on 1-D confidence; simple and stable.
- Effective and consistent results: Notable ACCH gains on ImageNet-C/R and VisDA-C, outperforming TTA and CLIP baselines
- Solid training objective: combining pseudo-label CE, prototype MSE, and diversity regularization is a reasonable recipe for stable online adaptation.

### Weaknesses
- Open-world sets fix an equal number of ID and OOD examples; real streams often have skewed OOD rates. Please report robustness across OOD ratios and streaming orders. 
- VLBO runs both ResNet and CLIP per test sample plus clustering. Please report wall-clock and memory overhead vs. other baselines

### Questions
- Missing reference on agreement-boosted between domains https://arxiv.org/abs/2406.09353. They also use confidence-based filtering for generating pseudo labels
- How does VLBO behave when OOD prevalence varies?

### Soundness
3

### Presentation
3

### Contribution
3
