# $\mu$-Parameterization for Mixture of Experts

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2

## Abstract
Recent years have seen a growing interest and adoption of LLMs, with Mixture-of-Experts (MoE) emerging as a leading architecture in extremely large models. Currently, the largest open-source models reach over $1$T parameters. At such scales, hyperparameter tuning becomes prohibitively expensive. Precisely for this reason, the $\mu$Transfer is becoming a key technique. It allows for seamless transfer of optimal hyperparameters across model scales, resulting in a huge reduction in tuning costs. However, existing work has primarily focused on dense LLMs, leaving MoE architectures unexplored. In this work, we derive a $\mu$-Parameterization for MoE, providing theoretical guarantees for feature learning across model widths. Our experiments demonstrate that the optimal learning rate reliably transfers across model sizes, establishing a foundation for efficient hyperparameter tuning in large-scale MoE models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose an extension of $\mu$-parameterization for MoE architectures, enabling transfer of optimal learning rates across increasing MoE model width. The authors provide a theoretically grounded derivation of their parameterization, along with limits for which the theoretical guarantees cease to hold, such as for varying MoE granularity.

### Strengths
The paper discusses an important practical and empirical topic -- hyperparameter transfer across model scale -- and uses principled foundations to derive theoretical results with accompanying guarantees and limitations. The paper is also clear and the logic is easy to follow.

### Weaknesses
The main weakness of the paper is the extremely limited experimental validation. The authors only present experiments for one single hyperparameter, the learning rate, in one single MoE model, Switch transformer, for one dataset, C4. The theoretical work is interesting but with such limited experimental results it becomes infeasible to really assess the impact and empirical consequences of the work. On a related note, the paper is short at only 6 pages, so I'm curious as to why the authors didn't consider using some of the ample additional page limit to more extensively verify their work. 

The technical novelty is also limited, it being an extension of $\mu$-parameterization [Yang et al, NeurIPS 2021] to MoE architectures. In my view, extending theoretical works or methods centered on dense models to MoE is still worthwhile and interesting, but it does then require a thorough analysis of how MoE presents new challenges and opportunities. In this work, however, the analysis appears substantially more limited than in Yang.

### Questions
Is there a reason for the paper being so short, and the experimental validation being so limited? There are numerous datasets, modalities, and models you could have considered, for which experimental results would help support your work.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of hyperparameter tuning in extremely large Language Models (LLMs), particularly those utilizing the Mixture-of-Experts (MoE) architecture. MoE has emerged as a leading architecture for scaling LLMs, but existing $\mu$-Parameterization ($\mu$P) techniques, which enable transfer of optimal hyperparameters across model scales ($\mu$Transfer), have previously focused only on dense architectures.
The authors derive a theoretically grounded $\mu$-Parameterization for MoE, building upon the Tensor Programs 5 (TP5) framework. The core theoretical finding classifies the expert weights ($E_1, E_2$) as hidden weights and the router weight ($R_0$) as an output weight within the $\mu$P framework. The paper empirically verifies that this $\mu$P scheme successfully transfers the optimal learning rate across varying model widths (up to 2048). Furthermore, the authors introduce a simplified parameterization (simpleP) that also achieves transferability, and they investigate other scaling axes, finding that while increasing the number of experts preserves transfer, changing MoE granularity breaks it.

### Strengths
**Originality**: The primary strength is the novel theoretical derivation and empirical validation of $\mu$P for Mixture-of-Experts. This is a crucial extension of existing Tensor Program theory that had previously overlooked sparse architectures. The paper also introduces and evaluates simpleP, a heuristic parameterization, providing a useful comparison point.

**Quality**: The theoretical analysis is high quality, providing a principled method for scaling MoE components by classifying expert weights as hidden and the router as output, thereby ensuring stable gradient and activation scales across widths. The theoretical argument is supported by technical proofs regarding covariance (Appendix B). The empirical results, showing successful learning rate transfer for both $\mu$P and simpleP across a wide range of widths, are compelling.

**Clarity**: The paper is well-written, making complex theoretical concepts accessible. The tables summarizing the parameterization rules are particularly helpful.

**Significance**: This work directly addresses the compute bottleneck inherent in training LLMs over 1T parameters by enabling cost-efficient hyperparameter tuning. The ability to transfer optimal learning rates is a foundational step toward efficient large-scale MoE training.

### Weaknesses
**Missing critical information for experiments**: It’s unclear from the text what is the range of model sizes considered in the experiments. Can the authors add a table on the sizes of models considered in terms of overall number of parameters and number of active parameters?

### Questions
**Utility of $\mu$P over simpleP**: Since simpleP (heuristic, experts only reparameterized) empirically shows strong learning rate transferability similar to the full $\mu$P (router and experts reparameterized), what practical benefits does the theoretically grounded $\mu$P provide over simpleP? Do the theoretical guarantees translate into demonstrably better final model performance, faster convergence, or improved stability during training runs, especially at the largest tested widths?

**Clarification on Figures**: The authors in lines 110-112 claim “We scale the model width, with the number of experts and top-k kept fixed. The hidden dimension of each expert grows proportionally with the model width, so that the ratio between them remains constant.” Does this mean that for results in Figure 1 and Figure 2a, both the model dimension and number of experts are being scaled simultaneously? Or in Figure 1, only model dimension is increased with fixed number of experts and in Figure 2, only number of experts is increased with model dimension fixed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This addresses the substantial cost associated with hyperparameter tuning in extremely large-scale Language Models (LLMs), particularly those utilizing the Mixture-of-Experts (MoE) architecture.

### Strengths
1. The paper solves a major engineering bottleneck for large-scale AI research. By enabling the transfer of optimal learning rates across MoE model scales.

### Weaknesses
1. The paper is not well-organized and presented. It would be better to write the texts and give more beautiful pictures for this venue. 
2. The simpleP parameterization is shown to be highly effective at learning rate transfer, performing similarly to $\mu$P. A more in-depth discussion is needed to theoretically justify why simpleP, which is less complex than the full $\mu$P derivation, works so well, and what specific scenarios would necessitate the complexity of the full $\mu$P.

### Questions
1. Could the authors provide the results for the expert-count and granularity ablations (currently run under simpleP) using the proposed $\mu$-Parameterization ($\mu$P)?
2. The simpleP parameterization seems to achieve learning rate transfer very successfully16. Could the authors provide a brief theoretical analysis or justification for why simpleP works effectively in this context, and explain the key non-trivial theoretical differences that make $\mu$P the preferred choice over simpleP for massive-scale MoE training?
3. While the paper focuses on the learning rate, $\mu$-Transfer is typically used to transfer other critical hyperparameters (e.g., initialization scale, weight decay). Do the theoretical guarantees of the derived $\mu$P extend to these other hyperparameters in MoE architectures? If so, could the authors provide preliminary empirical evidence demonstrating the successful transfer of one additional hyperparameter (e.g., weight decay) across model widths using $\mu$P?

### Soundness
1

### Presentation
1

### Contribution
1
