# IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Large Language Models (LLMs) have achieved impressive performance through Supervised Fine-tuning (SFT) on diverse instructional datasets. When training on multiple capabilities simultaneously, the mixture training dataset, governed by volumes of data from different domains, is a critical factor that directly impacts the final model's performance. 
Unlike many studies that focus on enhancing the quality of training datasets through data selection methods, few works explore the intricate relationship between the compositional quantity of mixture training datasets and the emergent capabilities of LLMs. 
Given the availability of a high-quality multi-domain training dataset, understanding the impact of data from each domain on the model's overall capabilities is crucial for preparing SFT data and training a well-balanced model that performs effectively across diverse domains. 
In this work, we introduce IDEAL, an innovative data equilibrium adaptation framework designed to effectively optimize volumes of data from different domains within mixture SFT datasets, thereby enhancing the model's alignment and performance across multiple capabilities. 
IDEAL employs a gradient-based approach to iteratively refine the training data distribution, dynamically adjusting the volumes of domain-specific data based on their impact on downstream task performance. By leveraging this adaptive mechanism, IDEAL ensures a balanced dataset composition, enabling the model to achieve robust generalization and consistent proficiency across diverse tasks. 
Experiments across different capabilities demonstrate that IDEAL outperforms conventional uniform data allocation strategies, achieving a comprehensive improvement of approximately 7\% in multi-task evaluation scores.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes IDEAL, a framework for adaptive data proportion optimization in multi-domain supervised fine-tuning (SFT) of large language models. The method formulates the allocation of training data from different domains as a bi-level optimization problem, where the validation loss is minimized with respect to domain-wise mixing coefficients. IDEAL leverages influence functions and second-order (Hessian-based) approximations, implemented efficiently through layerwise K-FAC techniques, to update data proportions. Experiments on multiple benchmarks covering domains such as mathematics, code, reasoning, and instruction following suggest that IDEAL improves average performance over uniform or heuristic data mixing baselines. Ablation studies explore hyperparameter sensitivity and approximation robustness.

### Strengths
- The problem of domain proportioning in multi-domain fine-tuning is relevant and timely for large-scale LLM adaptation.
- The paper provides a reasonably clear mathematical formulation of the bi-level optimization problem and explains how influence-function approximations can guide proportion updates.
- The method is relatively easy to integrate into existing SFT workflows and scales to medium-sized LLMs with standard K-FAC approximations.

### Weaknesses
I have the following concerns. *If the authors could properly address them during the rebuttal phase, I am willing to raise my score.*
- The novelty is quite limited. IDEAL largely combines elements that have appeared in prior works such as DoReMi, DOGE, and influence-based data reweighting. The paper does not provide a clear conceptual or algorithmic distinction beyond rephrasing the same principles in a unified framework.
- The method depends heavily on strong approximations (block-diagonal K-FAC, mini-batch stochastic estimation), but the effect of these approximations is not quantitatively analyzed. There is no estimate of how much the inverse-Hessian-vector product deviates from the true gradient, which raises doubts about the claimed stability.
- The empirical evaluation, though broad, does not test IDEAL under difficult or realistic conditions such as domain shift, heterogeneous modalities, or extreme imbalance. The results are confined to a few text-based domains and mid-size models, limiting generality.
- Some important recent baselines [1,2,3] are missing. The paper would benefit from either a comparison with these methods or a discussion of their relevance.
- The sensitivity analysis is shallow. Although the paper examines one hyperparameter $m$, it does not analyze the dependence on other key parameters or propose principled ways to set them, which undermines practical applicability.

[1] Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models. EMNLP 2024.

[2] How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition. ACL 2024.

[3] Boosting Multi-Domain Fine-Tuning of Large Language Models through Evolving Interactions between Samples. ICML 2025.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes IDEAL, a framework for optimizing the distribution of multi-domain training data during supervised fine-tuning (SFT) of large language models. The method uses gradient-based optimization with second-order information to iteratively adjust domain-specific data volumes, claiming approximately 7% improvement over uniform data allocation strategies.

### Strengths
1. Important Problem Selection. The paper addresses a genuinely important challenge in LLM training - how to balance multiple capabilities during supervised fine-tuning. This is a practical problem that many practitioners face, and finding principled solutions has real value.
2. Systematic Approach. Rather than relying on heuristics or manual tuning, IDEAL provides a systematic, gradient-based framework for optimizing data distributions. The mathematical formulation, while not novel, is reasonably rigorous and provides a clear optimization objective.
3. Relative Comprehensive Experimental Setup. The paper evaluates across multiple diverse domains (mathematics, coding, reasoning, instruction-following) Includes both single-epoch and multi-epoch training scenarios

### Weaknesses
1. Limited Technical Novelty. The core contribution relies on well-established techniques (influence functions, K-FAC approximation) applied to data mixing. The formulation in Eq. (1) introducing β parameters for data repetition is straightforward, and the bi-level optimization problem (Eq. 2) follows standard approaches. The use of influence functions for data weighting has been extensively explored in prior work, making the technical contribution incremental.
2. Experiment results and setup is hard to follow. What models are used in table1 and table 2? No statistical significance testing is provided for the reported improvements. The baseline comparisons are limited, missing recent strong methods in data selection and mixing such as regmix, autoscale and icons.
 regmix: https://arxiv.org/abs/2407.01492
autoscale: https://arxiv.org/abs/2407.20177
icons: https://arxiv.org/abs/2501.00654



3. Computational Efficiency Concerns. Despite claims of efficiency, the method requires: Multiple full training iterations (88 hours total as shown in Table 7). Complex Hessian approximations that may not scale well. The K-FAC approximation introduces significant estimation errors (acknowledged in Appendix C but not properly addressed)

4. Theoretical Limitations. Assumption 1 about twice differentiability and convexity is unrealistic for modern LLMs. The paper acknowledges in Appendix B.2 that the Hessian may not be invertible in practice, requiring ad-hoc damping terms. The connection between the local perturbation analysis and global optimality is not established

5. Experimental Design Issues. The reference dataset of only 500 samples is too small to reliably estimate performance across diverse capabilities. The choice of m=0.15 appears arbitrary despite the sensitivity analysis. The random sampling with factor σ=0.5 introduces additional variance not properly controlled for

6. Presentation and Clarity Issues. The paper lacks clear ablation studies isolating the contribution of different components.The "data equilibrium" terminology is not well-justified theoretically. Important details are relegated to appendices, making the main paper incomplete.

### Questions
The paper claims IDEAL "ensures a balanced dataset composition" but provides no formal guarantees.
The assertion that data volume matters little (Section 4.3) contradicts extensive prior work and is based on limited evidence.
The "data conflicts and data symbiosis" discussion in Appendix E lacks rigorous empirical support.

### Soundness
2

### Presentation
2

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
This paper introduces IDEAL (Data Equilibrium Adaptation), a framework aimed at optimizing the mixture proportions of data from different domains when performing multi-capability Supervised Fine-Tuning (SFT) on Large Language Models (LLMs). The authors argue that naive data mixing often leads to suboptimal performance and propose a principled approach to find a better "data equilibrium." IDEAL uses bi-level optimization theory to derive how changes in the proportion of data from each domain affect the model's loss on a separate reference dataset. This involves estimating gradients that depend on the Hessian matrix of the training loss. To make this computationally feasible for LLMs, the framework employs approximations like K-FAC. The IDEAL algorithm iteratively refines the data proportions by performing SFT, calculating these gradients, updating the proportions, and then resampling the dataset for the next iteration. Experiments across several capability domains suggest that IDEAL leads to improved multi-task performance compared to uniform data mixing.

### Strengths
1. The paper addresses the critical and practical challenge of determining optimal data mixtures for multi-capability SFT, moving beyond simple heuristics.

2. The proposed IDEAL framework is grounded in optimization theory (bi-level optimization), providing a principled, gradient-based approach to adapt data proportions based on their impact on a reference set performance.

3. The work incorporates practical considerations for LLMs by using techniques like K-FAC to approximate the Hessian, making the theoretically complex approach computationally tractable, and demonstrates empirical improvements over baseline mixing strategies.

### Weaknesses
1. The computational overhead remains extremely high, potentially limiting practical utility. Despite approximations, the iterative nature of IDEAL, requiring a full SFT cycle per iteration plus complex gradient calculations involving Hessian approximations, makes it very resource-intensive. The appendix reveals the total time is an order of magnitude higher than standard SFT.

2. The method's effectiveness heavily relies on the accuracy of Hessian approximations (like K-FAC), which can introduce significant errors. The paper acknowledges this possibility but does not quantify the impact of these approximation errors on the reliability of the calculated gradients ($\beta$ updates). Inaccurate gradients could lead to suboptimal or unstable optimization

3. The framework's optimization target and granularity may be suboptimal. IDEAL optimizes based on performance on a chosen reference set $\mathcal{D}_{ref}$, making the results sensitive to this set's representativeness. Furthermore, it only adjusts quantities at the domain level via sampling, ignoring potential quality variations within a domain's data. More granular, sample-level weighting might be more effective.

### Questions
1. The K-FAC approximation involves selecting 'important' MLP layers based on variance . Could the authors provide more details on how these layers are selected and how many are typically used? Does this selection significantly impact the results?

2. In Table 1 (Epoch=1), the 'Specific SFT' model trained only on 'Instruction' data outperforms the 'Specific SFT' models trained on 'Coding' and 'Reasoning' data on their own respective benchmarks (Coding: 46.95 vs 37.20, Reasoning: 61.87 vs 60.19). This seems counter-intuitive. Could the authors offer an explanation, perhaps related to the composition of the 'Instruction' dataset mentioned in Appendix E ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work explores a widely discussed but yet inconclusive research question: how to achieve the optimal ratio of data from different domains in the SFT dataset for LLMs. This work proposes an iterative data ratio optimization method, using a gradient-based approach to iteratively optimize the training data distribution. It dynamically adjusts the amount of domain-specific data based on its impact on the performance of downstream tasks. Experiments across different capabilities show that IDEAL outperforms the traditional uniform data allocation strategy.

### Strengths
1. A classic and intriguing research problem, accompanied by an innovative solution.
2. This work provides some theoretical proofs under the premise of a well-formalized description.

### Weaknesses
1. Dynamic data curation has already been explored in some existing works [1], and it is recommended to introduce the related studies in the discussion.  
2. Although the method is optimized for efficiency, the computational overhead is still relatively large, as shown in Table 7. This somewhat undermines the applicability of the method.
3. According to the experimental results in Table 1, at epoch=3, the performance of IDEAL is actually worse than its performance at epoch=1. This, to some extent, makes the method face more hyperparameter tuning challenges in practical application scenarios.

[1] CiT: Curation in Training for Effective Vision-Language Data. ICCV 2023.

### Questions
1. Would considering the quality of data samples during the dataset sampling process lead to further improvements in accuracy?
2. Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
