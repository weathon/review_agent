# Unified Projection-Free Algorithms for Adversarial DR-Submodular Optimization

- Avg Score: 7.00
- Decision: Accept (poster)
- Scores: 6, 5, 10

## Abstract
This paper introduces unified projection-free Frank-Wolfe type algorithms for adversarial continuous DR-submodular optimization, spanning scenarios such as full information and (semi-)bandit feedback, monotone and non-monotone functions, different constraints, and types of stochastic queries. For every problem considered in the non-monotone setting, the proposed algorithms are either the first with proven sub-linear $\alpha$-regret bounds or have better $\alpha$-regret bounds than the state of the art, where $\alpha$ is a corresponding approximation bound in the offline setting. In the monotone setting, the proposed approach gives state-of-the-art sub-linear $\alpha$-regret bounds among projection-free algorithms in 7 of the 8 considered cases while matching the result of the remaining case. Additionally, this paper addresses semi-bandit and bandit feedback for adversarial DR-submodular optimization, advancing the understanding of this optimization area.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents unified projection-free Frank-Wolfe algorithms for online continuous adversarial DR-submodular optimization. It covers various scenarios such as full-information and (semi-)bandit feedback, monotone and non-monotone functions, different constraints, and types of stochastic queries. The algorithms achieve state-of-the-art or improved regret bounds compared to existing methods. The paper also addresses semi-bandit and bandit feedback for adversarial DR-submodular optimization.

### Strengths
The paper presents a unified approach for online continuous adversarial DR-submodular optimization, covering a broad spectrum of scenarios. This comprehensive investigation is novel and expands the understanding of the field beyond previous research. The authors introduce technical novelties, such as the combination of meta-actions and random permutations, contributing to the originality of the research.

The paper is well-structured and clearly presents the problem formulation, algorithms, and theoretical analysis. The inclusion of tables and figures enhances the clarity of the presentation.

In summary, the research addresses a significant problem in the field of continuous adversarial DR-submodular optimization with numerous real-world applications. The proposed algorithms demonstrate practical relevance and significance, contributing to the development of efficient optimization techniques for DR-submodular functions.

### Weaknesses
While the paper mentions the applications of continuous adversarial DR-submodular optimization, it lacks a thorough empirical evaluation of the proposed algorithms on real-world applications. Including experiments that demonstrate the performance of the algorithms in practical scenarios would strengthen the paper's claims and provide empirical evidence of their effectiveness

I guess that the authors have adopted a text-wrapping layout around the pseudocode due to space constraints. However, the limited space allocated to the pseudocode has resulted in somewhat disorganized format. I would suggest that the authors consider using algorithm2e to reduce the space occupied by the pseudocode, thereby enhancing its readability. Just a friendly suggestion.

### Questions
Please refer to weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates online adversarial continuous DR-submodular optimization. The authors propose unified projection-free Frank-Wolfe type algorithms under many settings, e.g., full information and (semi-)bandit feedback, monotone and non-monotone functions, different constraints, and types of stochastic queries. Compared with the related works, most of the proposed algorithms achieve state-of-the-art $\alpha$-regret bound.

### Strengths
* This paper conducts a detailed investigation on online adversarial continuous DR-submodular optimization and proposes effective algorithms for various settings. 
* Compared with existing studies, the proposed algorithms achieves state-of-the-art $\alpha$-regret bound in most settings.

### Weaknesses
The presentation of technical contributions and motivation in this paper requires further improvement. See **Questions** below.

### Questions
I have the following questions to ask the authors:

* Could you elaborate on the technical challenges of this paper? The proposed algorithms seem to combine existing techniques without introducing novel technical tools. 
* This paper proposes two unified projection-free Frank-Wolfe type algorithms for the full-information and bandit settings. Could you provide the reasons why previous algorithms couldn't achieve a unified framework? 
* Compared with Zhang et al. (2023) for non-monotone functions, the proposed algorithms achieve only a slight improvement in terms of $\alpha$-regret guarantee. Can you explain the reasons for this, and the differences between two algorithms? 
* Could the authors provide a discussion on the technical contributions of this paper regarding online adversarial continuous DR-submodular optimization?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes an expedient and highly generalized projection-free Frank-Wolfe type algorithms for adversarial and continuous DR-submodular maximization problems. The settings they consider can be a full information or a (semi-) bandit feedback, with monotone or non-monotone functions, having different constraints, and types of stochastic queries. Notably, for every problem they consider in the non-monotone setting, they either provide the first algorithm with proven sub-linear $\alpha$-regret bounds or they improve the already existing $\alpha$-regret bounds. They also report obtaining the state-of-the-art sub-linear $\alpha$-regret bounds among projection-free algorithms in the majority of the monotone settings they consider. In the remaining monotone setting, they match their result to the existing one.

### Strengths
The paper's efforts to present unified projection-free algorithms is convenient and worthy of exploration. The main technical contributions of the paper, a.k.a. combining the ideas of meta-actions and random permutations and providing a refined analysis that does not rely on variance reduction techniques, are original. The ideas, contributions, and techniques are expressed clearly. I especially enjoyed the level of detail in Table 1 and Appendices A.2 and C.

### Weaknesses
I have not noticed particular weaknesses so far.

### Questions
I do not have any clarifying questions on my mind at the moment.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent
