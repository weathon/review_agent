# Entropy-Based Aggregation for Fair and Effective Federated Learning

- Decision: Reject
- Scores: 8, 6, 5, 6, 5

## Abstract
Federated Learning (FL) enables collaborative model training across distributed devices while preserving data privacy. Nonetheless, the heterogeneity of edge devices often leads to inconsistent performance of the globally trained models, resulting in unfair outcomes among users. Existing federated fairness algorithms strive to enhance fairness but often fall short in maintaining the overall performance of the global model, typically measured by the average accuracy across all clients. To address this issue, we propose a novel algorithm that leverages entropy-based aggregation combined with model and gradient alignments to simultaneously optimize fairness and global model performance. Our method employs a bi-level optimization framework, where we derive an analytic solution to the aggregation probability in the inner loop, making the optimization process computationally efficient. Additionally, we introduce an innovative alignment update and an adaptive strategy in the outer loop to further balance global model's performance and fairness. Theoretical analysis indicates that our approach guarantees convergence even in non-convex FL settings and demonstrates significant fairness improvements in generalized regression and strongly convex models. Empirically, our approach surpasses state-of-the-art federated fairness algorithms, ensuring consistent performance among clients while improving the overall performance of the global model.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Existing federated fairness algorithms strive to enhance fairness but often fall short in maintaining the overall performance of the global model, typically measured by the average accuracy of all clients. To address this issue, this paper proposes a novel algorithm that leverages entropy-based aggregation combined with model and gradient alignment to simultaneously optimize fairness and global model performance. The method presented in this paper employs a two-layer optimization framework and derives an analytical solution for the inner loop aggregation probabilities, making the optimization process highly computationally efficient. Furthermore, the paper introduces innovative alignment updates and adaptive strategies in the outer loop to further balance the performance of the global model and fairness. Additionally, the paper conducts a convergence analysis and numerous experiments to demonstrate the novelty and effectiveness of the proposed method. The content is very detailed, making it an excellent paper.

### Strengths
- Content is solid and comprehensive.
- The paper is well-structured and advances step by step.
- The theory is clear and solid.
- The experimental section is well-developed and extensive.

### Weaknesses
- Please explain how the convergence rate range mentioned in lines 396 to 397 of the main text is derived.

### Questions
see the weakness.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FedEBA+, a novel federated learning algorithm designed to enhance both fairness and global model performance through a computationally efficient bi-level optimization framework. The authors propose an innovative entropy-based fair aggregation method for the inner loop and develop adaptive alignment strategies to optimize global performance and fairness in the outer loop. The paper provides theoretical analysis confirming the convergence of FedEBA+ under non-convex settings and demonstrates its superiority over state-of-the-art fairness algorithms through empirical results on various datasets.

### Strengths
- The work addresses a critical issue in federated learning and provides a practical solution that could have some impact on the field.
- The paper is well-researched, with a robust theoretical foundation and comprehensive empirical validation. 
- The authors provide insightful convergence analysis and fairness guarantees.
- The objective function and the proposed algorithm are novel contributions.

### Weaknesses
- The paper would benefit from a more intuitive explanation of the connection between entropy and fairness.
- The explanation of the entropy-based aggregation method could be more detailed, particularly in terms of how it differs from and improves upon existing methods.
- The concept of "ideal loss" appears somewhat confusing. Since $\tilde{f}(x)$ represents the ideal loss, which signifies the global model’s performance under an ideal training setting, it is unclear why the gradient of the ideal loss can be estimated by averaging local one-step gradients during the model alignment procedure, yet it should be estimated by Equation.10 during the gradient alignment procedure. Could the authors give more intuition to clarify this point?

### Questions
See Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper titled "Entropy-Based Aggregation for Fair and Effective Federated Learning" proposes a novel algorithm called FedEBA+ to address the fairness issues in Federated Learning (FL) while maintaining the global model's performance. The authors leverage an entropy-based aggregation mechanism, combined with model and gradient alignment, to optimize both fairness and global performance. A bi-level optimization framework is introduced to derive an efficient closed-form solution for aggregation probabilities, which is claimed to enhance fairness by adjusting weights for underperforming clients. The paper provides theoretical guarantees for the algorithm's convergence and fairness, and empirical results on several datasets demonstrate that the proposed approach outperforms existing fairness-oriented FL algorithms.

### Strengths
- **Clear Problem Motivation**: The paper addresses an important problem in federated learning—how to achieve fairness across clients while ensuring that the global model's performance does not degrade. The motivation behind balancing fairness and performance in a heterogeneous FL environment is well articulated.

- **Theoretical Guarantees**: The paper provides theoretical analysis for the convergence of FedEBA+ in non-convex settings, which adds credibility to the proposed approach. The fairness improvements are also supported by performance variance analysis.

- **Bi-level Optimization Framework**: The introduction of a bi-level optimization framework is interesting and provides a structured way to balance fairness and performance. The authors also derive a closed-form solution for the aggregation probability, which improves computational efficiency.

### Weaknesses
- **Lack of Novelty in Aggregation Strategy**: The core novelty of the paper—using an entropy-based aggregation mechanism—essentially boils down to applying an exponential normalization based on client losses to determine aggregation weights. This approach is not particularly novel, as similar strategies have been used in various other contexts (e.g., softmax-based weighting). 

- **Overemphasis on Entropy**: The paper heavily emphasizes entropy, but in practice, the core idea is simply a re-weighting based on client loss. The connection between entropy and fairness, while valid, feels somewhat forced in its application here. The novelty of applying maximum entropy principles in such a straightforward manner might be overstated.

- **Limited Innovation in Fairness-Performance Trade-off**: Although the paper claims to balance fairness and performance, the approach primarily adjusts aggregation weights based on client loss. Many existing algorithms adjust client weights in some form, and the use of entropy as a fairness mechanism does not seem to introduce substantial improvements beyond what is already available in the literature.

### Questions
1. **Novelty of Exponential Normalization**: The core aggregation strategy is based on an exponential normalization of client losses. Could the authors clarify how this approach fundamentally differs from existing techniques that also adjust aggregation weights based on client performance? How does the use of entropy provide a significant advantage over these existing methods?

### Soundness
2

### Presentation
2

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
Existing algorithm to achieve fairness in FL often do so at the expense of global model accuracy. The paper proposes FedEPA+, an algorithm to improve fairness in FL while maintaining global model performance. The key idea here is to use entropy-based aggregation followed by a novel model alignment technique which adaptively optimizes for either global accuracy or fairness depending on the current performance of the model. Theoretical results are provided showing the convergence of the proposed algorithm in general non-convex FL setups along with guaranteed improvement in fairness in the strongly convex scenario. Empirical results show that the proposed algorithm outperforms existing fair FL baselines in both global model accuracy as well as fairness.

### Strengths
1. To the best of my knowledge, the idea of using a constrained entropy model for aggregation in FL to improve fairness is novel and interesting.
2. The idea of adaptively changing the global aggregation to either prioritize global model accuracy or fairness, while heuristic, is again interesting.
3. Theoretical results proving that the proposed FedEBA+ can reduce variance compared to FedAvg for generalized regression and strongly convex models.
4. Experimental results look promising with the proposed FedEBA+ outperforming FedAvg and other baselines both in terms of global model performance and fairness.

### Weaknesses
1. The clarity of writing and motivation for doing FedEBA+ can be significantly improved. There were several parts which I found confusing or could not fully comprehend as I discuss below.

* It is a bit surprising to me that the expression for the aggregation probabilities in Eq. (4) does not depend on the desired $\tilde{f}(x)$ in Eq. (3). Is there an intuitive explanation for why this is the case?

* Authors are encouraged to provide an intuitive explanation for why maximizing constrained entropy can improve fairness. Right now, entropy-based aggregation just appears to be a black box to improve fairness

* While I could follow the construction of the proposed bi-level objective in Eq. (6) and the idea to adaptively change the ideal gradient, I was unable to follow the changes made in the local and global aggregation to account for this. In my understanding, we first solve the inner problem. i.e, find $p$ given the current global model. Then given $p$, we update the global model with either the ideal global gradient or ideal fair gradient depending on the requirement. Given this understanding, my questions are as follows:

    * In the alignment to improve global accuracy why are the $p_i$'s computed using the loss of the local models? (Line 312). In my understanding, since we are fixing $x$, when solving the inner problem, the $p$ should be computed using the loss of the global model, i.e, the expression in Line 351?
    * In the alignment to improve fairness why is the local optimization at clients changed  (Eq. 11 and Line 11 in Algorithm 1)? In my understanding only global aggregation should change?

* I don't follow how in Prac-FedEPA+ clients need to only communicate once. If we follow Algorithm 3 then it appears that communication can happen twice within a round: first in Line 6-8 and then again in Lines  18-22. I found this to be very confusing and authors should clarify what is the communication cost in their experiments.

* $A$ is not defined in Theorem 5.1. Similarly $w$ is not defined, although from context it appears to be the prior aggregation weights of client objectives.

* It seems a little strange to see discussion on the lower bound of an upper bound in Remark 5.2. Authors should just state the worst possible convergence rate of FedEPA+ by considering that $\sum_{i=1}^M w_i^2 \leq 1$.

* In Remark 5.3. it appears that to show convergence of $\alpha \neq 0$ we need to assume $k << m$. Is this correct? If so, authors should clearly mention this and explain why this is the case. Also I don't follow the argument that the proposed alignment results in a faster convergence rate than FedAvg since both seem to be achieving $O(1/\sqrt{mKT})$ rate.

* Why is FedMGDA+, PropFair and TERM incompatible in Table 2?

2.  The parameter $\theta$ appears to be crucial to the performance of the algorithm but currently there is very little discussion on how to set this parameter and its impact on convergence. Theoretically, we don't see any effect of $\theta$ in Theorem 1, which is a bit surprising to me. In practice, I would also suggest moving Table 7 to the main text and discuss the trade-off in setting $\theta$ in more detail.

### Questions
Please see the questions and comments listed in the Weakness section

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies an important problem, i.e,. to ensure fairness of federated learning while striving to maintain the accuracy, for scenarios where heterogeneity hurts the performance of FL. Their approach was based on an entropy-based formulation and they show theoretical results such as convergence as well as experiments which to an extent verifies the effectiveness of the proposed algorithm. 

The paper appears to be fairly well-written and well motivated. The main concern is the the limited novelty and significance. The experimental results are not impressive either.

### Strengths
1. The significance of federated learning is well presented and the motivation for ensuring fairness is well-motivated.

2. The theoretical analysis appears correct.

3. The experiments do confirm the improvement, to an extent.

### Weaknesses
1. While the exact formulation has not been  published in the literature, the formulation appears to be fairly straightforward and does not seem to be a significant contribution to the field. The analysis follows the standard analysis. 

2. The experimental results, while they show some improvement, do not appear impressive. The actual significance of the proposed framework is not convincing.

### Questions
1. Could you summarize the (non-trivial) significance of technical contributions, compared with existing literature? 

2. Could you elaborate on the significance of the experimental results? The numbers (such as accuracy) appears low and gives the doubt that the paper did not use strong baselines. 

3. This paper severely lacks references from ICML, NeurIPS, ICLR, AISTATS, especially from recent years. The authors are encouraged to provide a thorough literature study from the mainstream ML venues especially from 2022-2024, to help convince readers on the novelty and significance of this submission.

### Soundness
3

### Presentation
3

### Contribution
3
