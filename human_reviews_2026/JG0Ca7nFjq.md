# LACE: Lightweight Attribution-guided Concept Evolution for Continual Learning

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
We address concept proliferation in interpretable continual learning and present LACE, a lightweight framework that couples Concept Bottleneck Models with a learnable concept-alignment layer, \emph{Concept Attribution} (CA) that quantifies per-concept importance under standard attribution axioms, and Concept Verification (CV) that selects pruning budgets via a data-reusable approximation to leave-one-out with an IRLS hat-matrix correction. A prototype-augmentation mechanism stabilizes learning without exemplars. Across coarse- and fine-grained benchmarks, LACE yields compact, reusable concept sets, consistently improves or matches strong baselines, and narrows the gap between average and last-task accuracy, offering an auditable and parameter-efficient route to continual concept evolution. Our code is available at: https://anonymous.4open.science/r/LACE-7FD6/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ​LACE (Lightweight Attribution-guided Concept Evolution)​, a framework for interpretable continual learning that leverages conceptual understanding by integrating ​Concept Bottleneck Models (CBMs)​​ with three key mechanisms: ​Concept Attribution (CA)​​ to quantify concept importance, ​Concept Verification (CV)​​ to prune redundant concepts, and ​prototype augmentation​ to mitigate forgetting without exemplars. The contributions include a compact, auditable concept set that improves accuracy, reduces forgetting, and enhances interpretability across coarse- and fine-grained benchmarks, narrowing the gap between average and last-task performance.

### Strengths
1. The proposed LACE framework integrates concept bottleneck models with continual learning, utilizing the text encoder in CLIP to extract concepts. The key innovations are concept attribution and concept verification mechanisms, which prevent exponential growth of the concept set. 
2. This work is technically rigorous, with strong theoretical grounding, e.g., proofs of CV approximation. The empirical validation is extensive across five benchmarks, including commonly used coarse- and fine-grained datasets. The comparison results with sota baselines demonstrate the consistent improvements in accuracy. 
3. The paper is well-structured and clearly written and easy to follow. The figures and tables aid understanding.

### Weaknesses
1. The proposed concept attribution method relies on textual data and CLIP's text encoder to extract concepts, making it difficult to generalize to images without text descriptions. 
2. Although experiments demonstrate the superiority of the proposed method, ablation studies are still lacking to show the side effects when missing any part of the mechanism. Like, what if all detected concepts are used without pruning; what will happen if using a random concept selection mechanism instead of the proposed concept attribution? This paper proposes many hyperparameters but does not study the influences on continual learning performances, e.g., the balancing coefficients $\lambda$, $\sigma$, $\eta$, and the number of removed concepts. 
3. Another concern for experimental results is the fairness. CBM methods use CLIP's text encoder to extract concepts utilizing the corresponding text descriptions of images. This introduces additional model parameters and additional supervision when comparing with baselines only using CLIP's image encoder, like L2P and CODA-Prompt. I understand that utilizing such text labels is one merit of CBM methods. However, there should be enough experimental results to verify that the performance gains are not the fruits of the increased model capacity.

### Questions
- The leave-one-out cross-validation for concept verification is time-consuming, while the author proposes to use IRLS to do approximation. What is the time complexity for this approximation?
- In line 386, what is this "the same compute budget"? 
- In lines 176-176, the author said "CA ensures faithful, auditable grounds, while CV ensures robust, reusable pruning decisions". What are examples of concepts in the concept pool, and which types of concepts are likely to be pruned? 
- In lines 126-127, the author said "semantic-layer constraints" can be effective levers for mitigating catastrophic forgetting. Why did not the author report forgetting in the experimental parts?

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
This paper addresses "concept proliferation" in interpretable continual learning, where Concept Bottleneck Models (CBMs) accumulate an unmanageable number of concepts, harming interpretability and performance. The authors propose LACE (Lightweight Attribution-guided Concept Evolution), a framework that actively manages the concept set. LACE uses Concept Attribution (CA) to score and prune unimportant new concepts and introduces a data-driven Concept Verification (CV) module to automatically determine how many concepts to remove. This approach avoids manual tuning and includes a prototype-augmentation mechanism to mitigate catastrophic forgetting without storing past data.

The work's main contribution is identifying and solving the critical issue of concept proliferation. It provides a principled and practical framework that significantly reduces the number of concepts while maintaining or improving model accuracy and mitigating forgetting, as demonstrated across five diverse benchmarks.

### Strengths
1.The paper effectively identifies and motivates the significant, yet overlooked, problem of "concept proliferation" in interpretable AI, where too many concepts undermine the model's primary goal of interpretability.

2.The framework is well-designed, using an axiomatically-grounded attribution method (CA) for importance scoring and a data-driven, automated technique (CV) to set the pruning budget, which removes the need for manual hyperparameter tuning.

3.LACE demonstrates state-of-the-art or highly competitive performance across five different benchmarks, consistently outperforming strong baselines in both accuracy and forgetting mitigation, especially on more challenging datasets.

### Weaknesses
1.The paper claims the method is "lightweight" but provides no empirical evidence (e.g., training time, memory usage) to support this.

2.The mathematical description of the Concept Verification (CV) module is overly condensed, with undefined notation and abrupt logical jumps that make it difficult to follow.

3.The paper is missing key ablation studies to demonstrate the individual contributions of its main components (e.g., CV, prototype augmentation). It also lacks qualitative examples of which concepts are pruned versus retained.

### Questions
1.Provide a quantitative analysis comparing the computational overhead (training time, memory usage) of LACE against key baselines to validate the "lightweight" claim.

2. Improve the clarity of the mathematical sections by defining all variables explicitly and providing more high-level intuition to guide readers through the core derivations.

3.Perform and include detailed ablation studies to isolate the impact of each component. Additionally, provide visualizations of pruned vs. retained concepts to offer intuitive evidence of the method's effectiveness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LACE (Lightweight Attribution-guided Concept Evolution), a continual learning framework built on Concept Bottleneck Models (CBMs) to improve interpretability, mitigate catastrophic forgetting, and control concept proliferation during incremental learning.

### Strengths
- The paper pinpoints the important issue of concept proliferation in CBM-based continual learning, articulating both cognitive and algorithmic consequences

- Section 3.3, together with Section 3.4 and detailed Appendix derivations, provides explicit, step-wise equations (e.g., Eq. (8)-(9), IRLS derivations) that both define and justify the attribution and verification stages.

- Tables 1 and 2 shows improvements compared to many state-of-the-art baselines across multiple datasets and splits in both accuracy and last-task accuracy.

### Weaknesses
1. The prototype augmentation technique with Eq 4 and 5 causes many confusions. There seems to be an inconsistency in using uppercase $P$ and lowercase $p$ when notating prototypes, which also causes confusion with the "concept pool" in line 259. Why is there a formula for $V^j_{\text{pseudo}}$ in line 244? If $V^j_{\text{pseudo}}$ represents "the pseudo-features," then why is it used as an input set in Eq 5?

2. Are $W^t_y$ and $W^t_l$ one or two different objects? There are many formulas and objects in the paper that are not carefully notated, and they also lack connection with the framework's diagram (Fig 1), causing many difficulties in reading.

3. Regrading the notation $E_i$ and the loss in lines 269–270, the authors proposed multiplying two linear matrices consecutively without a non-linearity function. Will this cause the information from the two matrices to be linearly absorbed, which is equivalent to using a single linear matrix?

4. The author should explain their proposals more clearly. For example, why update $E_i$ (line 274)? In Eq 7 immediately after, why is the derivative taken with respect to $E$? Is $E$ here the image encoder? As I understand it, only the $W$ matrices are updated, and the image encoder is kept frozen.

5. In terms of experiments, the authors should compare with more recent baselines [1, 2, 3, 4] to demonstrate their superiority. Besides, why only average accuracy and last accuracy are used, why did you ignore the forgetting measure like most other baselines?

6. In lines 28-31, the authors claims that: *"However, many CL methods are still evaluated primarily by external behavioral metrics (e.g., accuracy or forgetting), with limited auditable characterization of how the model’s decision basis evolves across tasks. This gap weakens interpretability and impedes a systematic analysis of “why forgetting is avoided” and “where knowledge is retained.”".*

Why are measures like accuracy and forgetting considered external behavioral metrics? And why don't the authors propose and use additional metrics to explain “why forgetting is avoided” and “where knowledge is retained.”

[1] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning, ICCV25.

[2] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning, ICCV25.

[3] Boosting Multiple Views for pretrained-based Continual Learning, ICLR25.

[4] Advancing Prompt-based Methods for Replay-Independent General Continual Learning, ICLR25.

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
1

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
The paper studies interpretable continual learning to tackle concept proliferation and proposes a framework which employs Concept Bottleneck Models (CBMs) with a learnable Concept Attribution (CA) layer to quantify concept importance. The proposed method LACE works in three stages, starting with concept attribution to get the concept pool and scores for concepts, followed by concept selection and pruning removing redundant concepts and finally concept verification to decide the number of concepts to retain. To prevent forgetting in exemplar-free settings, the authors also perform prototype augmentation using pseudo-feature replay. The method performs competitive to existing methods with reduced forgetting across CL benchmarks.

### Strengths
1. The paper is well-written with good introduction to CBMs and CA. The proposed method is intuitive.
2. The method is theoretically grounded satisfying the two central attribution axioms.

### Weaknesses
1. The experimental section is very weak. There are no error bars reported for the methods using random seeds. This is crucial since the performance improvement is marginal in most cases.

2. There is no ablation study or analysis of the multiple components in the method (CA, CV). There is no discussion or qualitative analysis of interpretability using the method. The proposed method has not been validated adequately.

### Questions
1. How is the performance affected with different pruning-rate bounds or per-task concept candidate cap?

### Soundness
2

### Presentation
3

### Contribution
2
