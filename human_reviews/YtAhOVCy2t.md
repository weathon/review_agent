# Compositional Risk Minimization

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 6

## Abstract
In this work, we tackle a challenging and extreme form of subpopulation shift, which is termed compositional shift. Under compositional shifts, some combinations of attributes are totally absent from the training distribution but present in the test distribution. We model the data with flexible additive energy distributions, where each energy term represents an attribute, and derive a simple alternative to empirical risk minimization termed compositional risk minimization (CRM). We first train an additive energy classifier to predict the multiple attributes and then adjust this classifier to tackle compositional shifts. We provide an extensive theoretical analysis of CRM, where we show that our proposal extrapolates to special affine hulls of seen attribute combinations. Empirical evaluations on benchmark datasets confirms the improved robustness of CRM compared to other methods from the literature designed to tackle various forms of subpopulation shifts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focus on tackling the challenge of  subpopulation shift through Compositional risk minimization, where the theory of discriminative compositional shifts is established, and a practical algorithm (compositional risk minimization, CRM) is also given. Overall, the theory are technically sound  and the experimental results demonstrates that CRM is . A good paper but require more specifications.

### Strengths
1. **Novel Approach**: The paper introduces a novel approach based on flexible additive energy models for compositionality in discriminative tasks, which can provably extrapolate to novel attribute combinations.
2. **Theoretical Grounding**: The approach is theoretically grounded, providing guarantees for generalization under compositional shifts, which is crucial for robust machine learning models.
3. **Comprehensive Evaluation**:  Extensive empirical studies and experiments to demonstrate the impact of different choices and validate the effectiveness of the proposed method.

### Weaknesses
1. The Compositional Risk Minimization (CRM) framework lacks explicit constraints on feature representations, which is widely recognized as crucial for generalization under sub-population shifts [1]. While CRM effectively models compositional shifts through additive energy distributions, it does not incorporate mechanisms to ensure the learned features are robust and semantically meaningful across different attribute combinations. Could the author make a further discussion on this problem?
    
    [1] Liu et al., Causal Triplet: An Open Challenge for Intervention-centric Causal Representation Learning, 2023
    
2. For the implementation of Algorithm 1, how to pre-specifies the spurious attribute $a$?

3. The current experimental results are somewhat lacking of conviction. More advanced DG algorithms, e.g. IRM, VREx, Fishr, SelfReg, are recommended to be included in the experiments. 

4. The sample complexity $s$ w.r.t the dimension $d$ should be verified through simulation studies.

### Questions
1. I am a little confused of the assumption 1: $p(x|z)=q(x|z)$，i.e.  the data generation mechanism from attributes to the inputs remains invariant, which is the main claim in the paper,  hence the distribution shift is essentially caused by $p(z)\neq q(z)$, which further implies that $p(x)\neq q(x)$, due to  $p(x)= \int p(x|z)p(z)dz$, and $q(x)=\int q(x|z)q(z)dz$. So, the covariate shift is caused by the marginal differences on the attributes. This seems contradict to traditional DG. 

2. For sample complexity $s\geq 8cd\log(d/2)$,  $c$ lacks of specifications?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a challenging scenario involving compositional shifts, where certain combinations of attributes are not present in the training distribution but appear in the test distribution. To tackle this issue, the authors propose a Compositional Risk Minimization (CRM) framework, which incorporates an additive energy classifier to predict multiple attributes. Furthermore, the framework adapts this classifier to effectively handle compositional shifts. Empirical studies demonstrate the effectiveness of the proposed CRM approach.

### Strengths
This paper explores a significant and under-researched area related to compositional shifts, which has been largely overlooked in the existing literature. 

The study is quite rigorous and comprehensive. Notably, it introduces the proposed approach through a running example involving two attributes, which makes the contributions and motivations clear and accessible. Additionally, the formulation of the Compositional Risk Minimization (CRM) framework offers valuable insights for future research.

### Weaknesses
One significant concern pertains to the attribute distributions during compositional shifts, which are frequently encountered in real-world applications. For instance, the distribution may shift from uniform to long-tailed, or it may be long-tailed from the outset. In this context, is it still effective to assume the test group prior $q(z)$ to be uniform? I encourage the authors to consider and discuss this issue in their paper.

The efficiency of the proposed CRM approach should also be examined, particularly in the context of multiple attribute abstention cases.

### Questions
The authors should address all my concerns presented in the Weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the compositional shifts, a hard type of sub-population shifts, and proposes compositional risk minimization. The method is well-motivated and some theoretical analyses are provided. Results on the sub-population shift benchmark are shown to support the proposed method.

### Strengths
- The compositional risk minimization method is reasonable and well-motivated.
- The formulation of the compositional shift setting provides a foundation for further research.
- Experimental results are good to support the method.

### Weaknesses
Based on the theoretical analysis and experimental results, my main concerns are as follows:

1. **Experimental Setting**: The analysis and proposed algorithm are designed to handle multiple attributes, with the theoretical advantages being most relevant for this multi-attribute context. However, the experiments are limited to only two attributes. I suggest that the authors include empirical results with multiple attributes to better align with the theoretical analysis.

2. **Results**: While the CRM results outperform the baselines in this study, I check the original benchmark paper [1] and find that CRM’s performance does not consistently exceed that of the baselines. Furthermore, the authors only report accuracy, which doesn’t fully account for group imbalances. Reporting Macro-F1 scores would provide a more comprehensive evaluation.

3. **“Disjoint” Setting**: I understand that the paper’s chosen disjoint setting adds a level of complexity. However, in real-world scenarios, it is often feasible to obtain a small number of samples for different attribute combinations (particularly with only two attributes, as in these experiments). The proposed method should also be evaluated in traditional settings where all attribute combinations have some representation. This would confirm that the method performs well without requiring group-dropping.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
