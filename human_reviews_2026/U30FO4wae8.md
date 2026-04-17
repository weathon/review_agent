# Entropy-driven Fair and Effective Federated Learning

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Federated Learning (FL) enables collaborative model training across distributed devices while preserving data privacy. Nonetheless, the heterogeneity of edge devices often leads to inconsistent performance of the globally trained models, resulting in unfair outcomes among users. Existing federated fairness algorithms strive to enhance fairness but often fall short in maintaining the overall performance of the global model, typically measured by the average accuracy across all clients. To address this issue, we propose a novel algorithm that leverages entropy-based aggregation combined with model and gradient alignments to simultaneously optimize fairness and global model performance. Our method employs a bi-level optimization framework, where we derive an analytic solution to the aggregation probability in the inner loop, making the optimization process computationally efficient. Additionally, we introduce an innovative alignment update and an adaptive strategy in the outer loop to further balance the global model's performance and fairness. Theoretical analysis indicates that our approach guarantees convergence even in non-convex FL settings and demonstrates significant fairness improvements in generalized regression and strongly convex models. Empirically, our approach surpasses state-of-the-art federated fairness algorithms, improving fairness among clients without sacrificing the overall performance of the global model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the dual objectives of enhancing predictive performance and ensuring client-level fairness in Federated Learning (FL) setting. To this end, the authors proposed FedEBA+, which formulates the problem as a bi-level optimization. FedEBA+ involves entropy-based aggregation mechanism, model and gradient alignment update strategy. A practical variant is further proposed to mitigate communication costs. Theoretical analyses establish convergence guarantees and examine fairness properties under strongly convex settings. Empirical results support, to some extent, the effectiveness of the proposed approach.

### Strengths
1. This paper presents a novel fairness-aware FL method, FedEBA+, and its practical variant Prac-FedEBA+, which integrates an entropy-based fair aggregation scheme together with model and gradient alignment strategies.
2. The discussion of related work is relatively comprehensive.
3. Theoretical analyses are provided to establish convergence guarantees and fairness properties of the proposed method.

### Weaknesses
**Overall:**
1. The motivation is not clearly articulated. Although the authors point out the limitations of existing approaches, i.e., high communication and computational overhead, degradation of global model performance in pursuit of fairness, and insufficient problem modeling, it is unclear what concrete innovations are introduced to address them. For example, regarding fairness, the rationale for adopting a constrained maximum-entropy strategy is not sufficiently explained, i.e., its advantages over more common strategies such as re-weighting or gradient adjustment are not clearly justified.
2. Several statements and claims in the paper are overstated.
3. The notation system is inconsistent and confusing, which significantly hinders readability and continuity of the technical presentation.

**Details:**

1 Statements and claims are overstated, or insufficiently supported by evidence.

i) “Enhance” the accuracy of the global model:
 - Is this improvement intended to be measured against predictive performance-focused methods?
- However, the experimental evaluation does not include comparisons with predictive performance–focused methods, making it hard to validate whether the proposed method truly improves the global model’s predictive capability.
- Lines 416-417 state that accuracy improves by 4% on CIFAR-10 and 3% on CIFAR-100 and Tiny-ImageNet, but it is not clear how these gains were computed. However, in fact, the margins over the second-best compared methods seem to be below 1%, and in some cases, the proposed approach performs worse than the compared methods.

ii) Communication efficiency:
- There are no any theoretical analysis or empirical evidence to support the claim of superior communication efficiency of proposed method.
- Lines 043-044: the proposed method requires transmitting both model and gradient information in each communication round, which is not more efficient than FedFV [Wang et al., AAAI 2021].
- Regarding the claim of “fast convergence”, Figure 3b does not clearly support this, as the proposed method requires a similar number of communication rounds to reach optimal performance compared with several other methods.

iii) Effectiveness of proposed methods:
- In Lines 266–267, the so-called “ideal global gradient” is approximated by averaging local one-step gradients. However, this approximation may not yield the optimal updated gradient, as client gradients are prone to conflict, especially in the case of data heterogeneity. The same concern arises in the Gradient Alignment for Improving Fairness procedure.

2 Confusing notation system
- x_t denotes the global model at round t in most parts of the paper, but in Definition 3.1, it represents a different model.
- In Eq. (3), the subscript notation for x is inconsistent with earlier definitions. Specifically, in Line 117, the subscript denotes the communication round and the superscript is originally used for the client index, but Eq. (3) uses the subscript to denote the client index.
- The symbol n is used ambiguously. It denotes the number of clients selected in communication round t (as defined in Line 114) and simultaneously represents the number of data points in client i (in Line 213).
- In Lines 254-256, $\eta_L$ should be $\eta$, as according to the definition in Line 120, $\eta_L$ denotes the local learning rate, whereas $\eta$ denotes the global learning rate.
- Line 119: $\eta_L$ is missing
- In Eq. (5) and (6), N is not defined.
- The number of clients is expressed with various symbols, e.g, n, m, N.
- These notation issues persist throughout Sections 3, 4, and 5, affecting the clarity of the presentation.
3. Other issues
- Figure 1 makes little sense, as the compared methods are not novel, and it does not provide new insights or clarify the motivation for the proposed approach.
- Lines 413–414 refer to “diverse models” ; however, it is not clear which models are included. In the tables, it seems that only a single model was actually used for evaluation.
- The results of “coefficient of variation” are not provided in Table 1 and Table 2.

**Minor issues:**
- Line 135, “parameter”—does it refer to a single parameter? Or should be “parameters”.
- In Definition 3.1, “more fair” should be “fairer”

### Questions
1. Line 409 states that existing methods “fail to model the problem directly,” leading to suboptimal performance. It is unclear what is meant by directly modeling the problem, how the claim of suboptimal performance is demonstrated, and why the proposed method can be considered to model the problem directly, whereas other methods do not.
2. Proposition 4.1 is difficult to follow. It is unclear how it relates to Eq. (3): does it describe a solution to Eq. (3)? In addition, how the conclusion in Lines 207–211, i.e., “assigning higher aggregation weights … reducing the gap with top performers…”, is derived (even if the statement seems intuitive).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies an important problem: finding a balance between fairness and utility for federated learning. It is a well-known fact that, although heterogeneity is necessary for realistic scenarios, it usually negatively impacts the performance. The paper's main novelty is the algorithm that uses entropy-based aggregation and during the learning process, it adaptively optimizes for either global accuracy or fairness depending on the current performance of the model. There are two major theoretical results. The first is on convergence and there is no assumption on convexity. The second result confirms the improvement in fairness under the strongly convex loss function. The authors also conducted extensive empirical analysis on the performance.

### Strengths
1. The problem is interesting and important. Although FL fairness is a widely studied topic, it has never been entirely solved.

2. The algorithm using entropy-based aggregation and adaptive optimization strategy is novel and interesting.

3. The theoretical analysis is extensive and sound.

4. The experiments are convincing and show the promising consequences of the algorithm.

### Weaknesses
In general, it is a decent work. But my main concern is the strong assumption for the theoretical analysis. The part (2) of Theorem 5.4 is the key part of theoretical analysis, yet it assumes strong convexity of the loss function. This is a strong condition and may downgrade the applicability.

### Questions
See Weaknesses. Please give a brief explanation on the difficulty of relaxing the assumption, e.g. convexity instead of strong convexity.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work adopted constrained entropy maximization objective to improve the performance fairness in federated settings. 
As a result, the proposed method `FedEBA+`, constructs an adaptive strategy that determines aggregation weights in terms of a softmax distribution from local test losses (optionally dampened by a temperature). For a complete formulation, authors employed bi-level optimization framework with respect to a parameter and the aggregation weights.

### Strengths
* Constrained entropy maximization and performance fairness in FL
  - The authors provided thorough justification that the proposed objective can improve performance fairness by reducing to the low variance of performance distribution. (in Eq. (3), Proposition 4.1, and Appendix I.1)

* Addendum on global utility
  - The authors proposed an additional trick to strike balance between utility and fairness of a global model by aligning the global update using the server-side ideal global gradient. (in Eq. (10-11))

* Extensive demonstration
  - `FedEBA+` showed comparable performance and fairness in several metrics and benchmark datasets: worst/best acc, variance, coefficient of variation.

### Weaknesses
- While different in motivation and objective, the resulting update formula coincides with that of `AAggFF`, cited in the draft.
- The alignment update is not novel, proposed and used similarly in `FedFA` (Wang et al., 2021) and `FedMDFG` (Pan et al., 2023), which are cited in lines 43-44.
- In Proposition 4.3, although authors provided the approximation method of aligned gradient, the increase in communication cost is unavoidable. 
  - That being said, each client should upload local gradients, local updates, and local losses in every communication round.

### Questions
- The convergence guarantee in Theorem 5.1 is questionable in three main aspects: 
  - i) the convergence upper bound given in eq. (13) can be _inflated to infinity_, i.e. unbounded chi-squared divergence, $\chi_{\boldsymbol{w} || \boldsymbol{p}^t}^2\rightarrow\infty$.
  - ii) the main constant related to the theoretical local learning rate was also induced by an incomplete assumption: the condition of the constant $C$ stated in the statment cannot hold, when if $\frac{1}{2}< 10L^2 \frac{1}{m} \sum_{i=1}^{m} K^2 \eta_L^2 (A^2 + 1)(\chi_{\boldsymbol{w} || \boldsymbol{p}}^2 A^2 + 1)$. This should be justified to have nonnegative server-side learning rate. 
  - iii) the convergence rate scales with the number of client $m$, as the same rate of the total steps $KT$. This directly violates _linear speedup guarantee_, which is desirable for typical FL algorithms.
    - However, the empirical convergence speed is faster than other methods in Figure 3-(b). Please clarify this gap.
- From Section 4.2, it is implied that $\alpha$ acts as a knob to balance tradeoff between utility and fairness.
  - Is the $\alpha$ selected heuristically as $\alpha=\beta/\tau$?
  - While this is one of main contributions of `FedEBA+`, there is *no (theoretical/empirical) analyses* related to $\alpha$... Please consider adding in-depth discussion on this, e.g., the optimal choice of $\alpha$ and its contribution to the generalization error.
  - What can we expect the effect of $\alpha$ on the variance of final performance distribution, the target notion of fairness of the proposed method?
- Please add dots and lines for `AAggFF` in Figure 3-(a) and 3-(b).

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
The paper introduces FedEBA+, a bi-level federated learning framework that integrates constrained max-entropy aggregation—which upweights clients with higher losses—alongside model and gradient alignment techniques. This approach simultaneously enhances client-level fairness by reducing performance variance across clients while maintaining strong global accuracy. To address practical deployment constraints, the authors develop Prac-FedEBA+, a variant that approximates the fair gradient computation to preserve FedAvg's efficient communication pattern. The framework is supported by theoretical analysis establishing non-convex convergence guarantees and variance reduction properties. Experimental validation across Fashion-MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrates performance improvements over existing fairness-oriented federated learning methods.

### Strengths
1) Addresses a critical challenge in federated learning by simultaneously optimizing both global model performance and fairness across clients through a principled bi-level optimization framework. Previous methods like AFL seem to do well on client level but loose performance on global level.

2) Provides comprehensive empirical validation supported by rigorous theoretical analysis, featuring experiments across multiple datasets with diverse client level fairness metrics including variance, worst/best-5% accuracy, and coefficient of variation. The work is further strengthened by thorough ablation studies and sensitivity analyses detailed in the appendix.

### Weaknesses
1) Communication overhead per global round is not clearly outlined:

Algorithm 1 indicates that FedEBA+ requires transmitting a fair gradient back to clients and collecting per-client losses and gradients at the current model state, which means additional downlink and uplink communication beyond FedAvg's requirements. Prac-FedEBA+ claims to maintain the same communication pattern as FedAvg. Can you include  a small table comparing FedEBA+/Practical version with a  breakdown of per round communication cost that includes: the number of uplink and downlink transmissions, total bytes transferred etc. Additional rounds and additional bandwidth cost analysis must be included.


2) Typos eg: Inconsistency in Table 1 regarding local epochs.The caption states "a single local epoch (K = 10)," which got me confused for a while. Please make sure there are no typos


3) The Non-IID data construction and its results should be clearly explained ideally in the main paper. Clarification should be made on strong non-IID conditions eg: class-imbalance scenarios  which are tested separately from weak non-IID (eg: sample-proportion skew) . Ideally your method should demonstrate the greatest advantages under strong non-IID conditions


4) There should be an Impact of local epochs K on drift and the role of alignment.
The paper's motivation suggests that alignment should mitigate local drift under non-IID conditions, implying that increasing K (local epochs) should benefit FedEBA+ more than FedAvg in terms of variance reduction.Please add an experiment varying K (e.g., K ∈ {1, 5, 10, 20}) under strong class imbalance  plotting variance, worst-5% accuracy, and global accuracy versus K for FedAvg, FedEBA+, and Prac-FedEBA+.

### Questions
Please see weaknesses I have merged my concerns with the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
