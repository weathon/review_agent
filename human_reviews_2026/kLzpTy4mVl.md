# Dynamic Semantic Routing for Multimodal Sentiment Analysis

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Multimodal sentiment analysis (MSA) aims to understand human emotions by integrating heterogeneous signals such as language, vision, and acoustic modalities. However, multimodal data often suffer from internal semantic entanglement, ambiguous cues, and inconsistent modality contributions, which limit the effectiveness of unified representations. To address these challenges, we propose a Dynamic Semantic Routing Framework (DSRF) for the MSA task. Specifically, we present a hierarchical semantic factorization module, which disentangles each modality into four functionally independent representations: primary emotion, contextual cue, ambiguity, and noise, enabling fine-grained semantic modeling. Moreover, we introduce a semantic dynamic routing interaction mechanism, which dynamically routes and aggregates the semantic factors through a capsule-inspired interaction process to reconstruct modality representations with high-order compositionality. Finally, we design an uncertainty-aware semantic fusion strategy that estimates the reliability of each semantic factor and adaptively integrates them across modalities for robust sentiment prediction under modality inconsistency. Extensive experiments on four benchmark datasets demonstrate that our framework achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a Dynamic Semantic Routing Framework (DSRF) for multimodal sentiment analysis. It first factorizes unimodal representations into four distinct components, then performs cross-modal interaction based on semantic routing. Finally, it explicitly estimates the uncertainty of each modality component, which guides their respective contributions during the multimodal fusion. The authors compare DSRF with recent methods and ablate the main proposed modules on several popular benchmarks. The results validate its effectiveness.

### Strengths
1. **Reasonable Motivations and Implementations.** Factorizing unimodal representations into different functional components is intuitively beneficial for customized representation modeling and multimodal fusion. Interacting unimodal factors through the iterative agreement mechanism is also interesting, providing a fresh perspective for modeling cross-modal synergy.
2. **Competitive Results.** DSRF outperforms recent models on four popular MSA datasets, validating its effectiveness.

### Weaknesses
1. **Unverified Claims.** The paper claims, in line 83, that previous methods fail to accommodate missing or corrupted modalities, and in line 298, that DSRF alleviates such issues. However, there is only superficial analysis and no empirical evidence supporting this claim. Similarly, in line 107, the paper claims that previous supervisions for modality factorizing are weak or indirect, yet also adopts heuristic supervisions that are difficult to prove strong or direct.
2. **Overcomplicated Optimization Objective.** The authors construct DSRF with three main modules, each comprising independent optimization targets. In the HSF module, the objective itself consists of four distinct terms. This naturally leads people to wonder about training stability and robustness, whose details are absent in the experiments.
3. **Missing Comparison with Latest Models.** The paper does not compare DSRF against any 2025 methods, which limits its effectiveness.
4. **Limited Analytical Experiments.** The only experiment besides the main comparison is a coarse-grained ablation study, which provides insufficient insights into the proposed methods. This has led to doubts regarding the effectiveness of the modules in the approach, particularly given their complexity.
5. **Uninformative Figure.** The manuscript includes only one figure, which aims to illustrate the overall framework of the proposed method. However, this figure depicts SFR (c) and DRI (d) modules without details, which provides little help in illustrating the proposed DSRF.
6. **Frequent Typos and Format Issues.** Some obvious mistakes are: incorrect reference format (should be \citep instead of \cite); misplaced caption (Table 1 and 2); misplaced "?" (line 50); missing "\" before textit (line 186); typo: "ous" (line 187).

### Questions
1. How do you guarantee that factorization captures the intended semantics? Is there any empirical evidence to support this process?
2. Why do you capture the noise factor, since it should not contribute to subsequent interactions? 
3. How are the weighting factors decided? What is the training dynamics of each loss component during the overall optimization?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a Dynamic Semantic Routing Framework (DSRF) for the MSA task.Framework incorporates three dedicated modules to tackle the semantic complexity, reconstruction, and inconsistency issues in multimodal emotion modeling.

### Strengths
1. Unlike existing methods that only perform modality-level factorization (e.g., separating invariant and specific features), the HSF module achieves fine-grained decomposition of modality internal semantics into four interpretable factors. 
2. The SFDR mechanism abandons static fusion operators and uses dynamic routing to model sample-specific factor interactions, which is more suitable for complex multimodal scenarios with variable factor importance. 
3. The framework integrates ideas from factorized representation learning, capsule networks, and uncertainty estimation, with detailed mathematical formulations to ensure the rigor of the method.

### Weaknesses
1. Why disentangle each modality into four functionally independent representations—primary emotion, contextual cue, ambiguity, and noise. This seems not to be reflected in the experiments. 
2. In Figure 1,The Semantic Factor Dynamic Routing Reconstruction Mechanism is not fully reflected in the figures.
3. Another obvious issue with this paper is the lack of sufficient explanation of the simulation results. You need to elaborate on your simulation results in detail and clarify the underlying reasons for obtaining such outcomes—for instance, by providing necessary visual analysis and performing case studies.

### Questions
1. What is the necessity of the hierarchical nature of the hierarchical semantic factorization module proposed in the HSF? Please provide appropriate experiments to prove it.
2. In HSF: What is the performance impact of removing a single factor (e.g., w/o ambiguity factor)? Does each factor’s constraint (e.g., orthogonality for contextual cues) actually work?
3. The ablation study is only conducted on the MOSI dataset, and it is unclear whether the conclusions hold on other datasets . This limits the generalizability of the findings.
4. The experiments are insufficient and lack justification for the necessity of the innovations.

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
5

### Summary
This paper proposes a Dynamic Semantic Routing Framework (DSRF) to address challenges in MSA, such as semantic entanglement, ambiguous cues, and modality inconsistency. The framework first decomposes each modality into four functionally independent semantic factors: primary emotion, contextual cue, ambiguity, and noise. It then employs a capsule-inspired dynamic routing mechanism to interact and reconstruct modality representations. Finally, an uncertainty-aware semantic fusion strategy adaptively integrates these factors based on their reliability. The method achieves state-of-the-art performance on multiple datasets.

### Strengths
1.	Disentangling modality representations into four factors sounds reasonable.

2.	Achieves good performance.

### Weaknesses
1.	There are writing errors on lines 50 and 221.
2.	The discussion of prior methods in Sections 2.1 and 2.2 is insufficient, and many of the most recent methods are not discussed. I provide some work [1-6] for your reference.
3.	It is recommended to revise Figure 1, as parts (c) and (d) lack informativeness. I understood the specific operations of these two modules after reading the Method section.
4.	The writing from lines 171-183, 248-263, and 285-299 is overly redundant. If a comparative explanation with prior methods is important, I recommend to discuss in section introduction.
5.	The method disentangles each modality into four functionally independent representations sounds reasonable. However, the paper lacks sufficient evidence to demonstrate the effectiveness of the disentanglement.
6.	In Equation 2, how are the labels for the context obtained?
7.	Why is a contrastive loss used to guide the learning of the noise factor, while the other modules do not use it?
8.	Could you show some visualization experiments, such as case studies and visualizations of the distribution for each factor.

Overall, I think the writing of this paper should be revised  and more experimental analyses should be added to demonstrate the effectiveness of the method. SOTA performance is not a necessary condition for publishing a paper, in-depth analysis is more important.

**Reference**

[1] DeepMLF: Multimodal language model with learnable tokens for deep fusion in sentiment analysis. arXiv:2504.11082.

[2] Decoupled multimodal distilling for emotion recognition. CVPR 2023.

[3] TCAN: Text-oriented cross attention network for multimodal sentiment analysis. Arxiv 2025.

[4] Proxy-driven robust multimodal sentiment analysis with incomplete data. ACL 2025.

[5] Towards robust multimodal sentiment analysis with incomplete data. NeurIPS 2024.

[6] Learning language-guided adaptive hyper-modality representation for multimodal sentiment analysis. EMNLP 2023.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper releases MARS-Sep, a reinforcement learning framework that reformulates separation as decision making. Instead of simply regressing ground-truth masks, MARS-Sep learns a factorized Beta mask policy that is optimized by a clipped trust-region surrogate with entropy regularization and group-relative advantage normalization. Extensive experiments on multiple benchmarks demonstrate consistent gains in Text-, Audio-, and Image-Queried separation.

### Strengths
The algorithm proposed in the paper demonstrates innovation, clear logic, and provides a corresponding description.

### Weaknesses
1. The paper does not provide an ablation study. It is necessary to conduct partial validation for RL and other revised modules in the proposed method.
2. The paper introduces the mechanism of RL and mixed sound source separation based on contrastive learning separately. There should be an overall explanation of the overall loss function.
3. The overall algorithmic structure and module information are relatively brief. It is recommended to provide detailed explanations of the network architecture and loss function settings for each module, and add subfigures for clarification when necessary.

### Questions
1. The paper does not provide an ablation study. It is necessary to conduct partial validation for RL and other revised modules in the proposed method.
2. The paper introduces the mechanism of RL and mixed sound source separation based on contrastive learning separately. There should be an overall explanation of the overall loss function.
3. The overall algorithmic structure and module information are relatively brief. It is recommended to provide detailed explanations of the network architecture and loss function settings for each module, and add subfigures for clarification when necessary.

### Soundness
2

### Presentation
2

### Contribution
3
