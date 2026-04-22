# Invariant Predict-and-Combinatorial Optimization under Distribution Shifts

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Machine learning has been well introduced to solve combinatorial optimization (CO) problems over the decade, while most works only consider the deterministic setting. Yet in real-world applications, decisions have often to be made in uncertain environments, which is typically reflected by the stochasticity of the coefficients of the problem at hand, considered as a special case of the more general and emerging "predict-and-optimize" (PnO) paradigm in the sense that the prediction and optimization are jointly learned and performed. In this paper, we consider the problem of learning to solve CO under the above uncertain setting and formulate it as "predict-and-combinatorial optimization" (PnCO), particularly in a challenging yet practical out-of-distribution (OOD) setting, where there is a distribution shift between training and testing CO instances. We propose the Invariant Predict-and-Combinatorial Optimization (Inv-PnCO) framework to alleviate this challenge. Inv-PnCO derives a learning objective that reduces the distance of distribution of solutions with the true distribution and uses a regularization term to learn invariant decision-oriented factors that are stable under various environments, thereby enhancing the generalizability of predictions and subsequent optimizations. We also provide a theoretical analysis of how the proposed loss reduces OOD error. The empirical evaluation across three distinct tasks on knapsack, visual shortest path planning, and traveling salesman problem covering array, image, and graph inputs underscores the efficacy of Inv-PnCO to enhance the generalizability, both for predict-then-optimize and predict-and-optimize approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of out-of-distribution (OOD) generalization in the predict-and-optimize paradigm for combinatorial optimization (CO). The authors observe that when the distribution of input instances shifts between training and testing, the quality of decisions (solutions) can deteriorate significantly. To mitigate this, they propose an Invariant Predict-and-Combinatorial Optimization (Inv-PnCO) framework that augments the training objective with a term to minimize the divergence between the model’s solution distribution and the true (optimal) solution distribution, while adding a regularizer to capture invariant decision-related features across environments. They provide a theoretical analysis (Theorem 1) showing that the proposed objective can reduce OOD decision error, under an invariance assumption. Empirically, Inv-PnCO is evaluated on three CO tasks (knapsack, visual shortest path, and traveling salesman) under different types of distribution shifts (e.g. artificial, perceptual, topological) and consistently improves solution quality compared to standard training

### Strengths
- Important Problem: Tackles the under-explored yet practically important problem of OOD generalization in predict-and-optimize pipelines, identifying why standard methods fail to maintain solution quality under shifts.

- Novel Framework with Theory: Proposes a new training framework (Inv-PnCO) with a principled objective that aligns predicted solution distributions with true solutions and enforces invariant features. This approach is backed by a theoretical guarantee (Theorem 1) on reducing OOD error, lending credence to its soundness.

- Clarity and Context: The paper is generally clear and well-structured. It provides relevant context by comparing to previous work (Table 1) and uses examples (Fig. 1) and definitions to build intuition, which helps the reader understand the motivation and significance of the approach.

### Weaknesses
- Strong Invariance Assumption: The method relies on Assumption 1, that there exist invariant decision-related features across environments that fully determine the optimal decisions. If this assumption is violated (i.e. if some environment-specific factors also influence the decisions), the theoretical guarantees may not hold and the performance of Inv-PnCO could degrade. The paper would benefit from discussing the realism and limits of this assumption.

- Need for Environment Partitioning: Inv-PnCO’s training procedure appears to require data from multiple environments or an environment label to enforce invariance. It is unclear how the approach would work if environment identifiers are not given or obvious. In practice, obtaining a diverse set of training environments or knowing when a distribution shift occurs can be challenging, which may limit the direct applicability of the method.

- Baseline Comparisons: The experimental comparison is primarily against a vanilla ERM (standard training) baseline. No specialized domain generalization or robust optimization baselines (e.g., IRM, DRO, or other invariant feature learning methods) are evaluated, even though some could potentially be adapted to the predict-and-optimize context. This leaves some uncertainty about how much improvement Inv-PnCO offers over best possible alternatives; a stronger empirical comparison would solidify the claims.

- Complexity and Overhead: The proposed framework introduces additional complexity in training. There are new hyperparameters (such as the regularization weight β and the number of environment splits) and the method incurs higher training cost (e.g. the end-to-end SPO model within Inv-PnCO roughly doubles or more the training time in experiments). While not prohibitive in the tested scenarios, this overhead could hinder scaling to very large or time-sensitive CO problems. The authors do provide some ablation and sensitivity analysis, but the paper could better discuss the trade-off between robustness gains and computational cost.

### Questions
- Environment Knowledge: Does the approach require explicit environment labels or a known partition of training data by environment? If so, how would Inv-PnCO perform when such labels are unavailable or when encountering a completely novel type of distribution shift at test time?

- Assumption Validity: How realistic is Assumption 1 in typical applications? If some features that influence decisions do vary across environments (i.e. only partially invariant factors exist), can the method still learn useful invariances, or would it be misled by spurious correlations?

- Alternative Baselines: Could the authors clarify why they did not compare against existing domain generalization techniques (e.g., invariant risk minimization or distributionally robust optimization adapted to the prediction stage)? This would help in understanding whether the improvements stem from the specific Inv-PnCO loss or from generally using multiple environments in training.

- Hyperparameter Sensitivity: How sensitive are the results to the choice of the regularization weight β and the number of environment splits used during training? For instance, is there a risk of under or over-regularization affecting the predictive accuracy versus the decision quality trade-off? Any guidelines from the sensitivity analysis would be useful to know.

- Residual OOD Gap: Even with Inv-PnCO, there remains some gap between in-distribution and OOD performance (the OOD regret, while improved, is still higher than IID in the experiments). What are the remaining sources of error under distribution shift, and could further techniqueshelp close this gap?

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
2

### Summary
The paper focuses on the predict-and-optimize approach in the combinatorial optimization setting. Specifically, the authors consider the out-situation where the the distribution can shift over time. To this end, the authors introduced the inv-PnCO method. This method uses regularization to find invariant features across environments. To illustrate the model's effectiveness, the authors derive a surrogate for the loss, and experimental results on various optimization problems.

### Strengths
-	The experiments include optimization problems of different types. Also, the results show on average, an improvement is made using the proposed new method
-	The contribution of extending the predict-and-optimize to the combinatorial optimization setting is relevant

### Weaknesses
-	The discussion is very brief. Although the authors provide more extensive limitations in the appendix, I would argue that it is important to include a more elaborate discussion in the main text
-	Code of experiments is not provided as supplementary material or as an anonymized GitHub. This raises concerns regarding reproducibility
-	Figure 3a has Risk on the y-axis, but there is no scale / are no numbers. Besides, Risk is not defined
-	Minor: There are punctuation errors. In some equations, e.g., eq. 8, commas or stops are missing, and/or the next new line has a mismatched use of capitals.

### Questions
-	The method is introduced for the Combinatorial Optimization setting. It seems this method is still applicable to the non-combinatorial setting. Could the authors elaborate whether this is correct?
-	The assumption used seems strong. Could the authors elaborate on which situations or contexts the assumption holds and when it breaks?
-	The results show in the tables show an improvement. However, is this number the average over test-points? How large are standard deviations?

### Soundness
3

### Presentation
1

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
This paper proposes Invariant Predict-and-Combinatorial Optimization (Inv-PnCO), a framework for decision-focused learning that is robust to distribution shifts.  Their approach adapts loss functions from decision-focused learning by adding a regularization term.  Empirically, their approach outperforms the baselines on the knapsack, shortest path, and traveling salesman problems.

### Strengths
- **Motivation**: Overall, this is definitely a well-motivated extension of the predict-and-optimize line of research.  
- **Reported Results**:  Although there are some issues with reporting, as I will highlight in weaknesses, the overall averaged results demonstrate improvement over ERM in terms of decision quality.

### Weaknesses
- **Scalability**: The proposed approach is not computationally efficient, with train/test times often being significantly higher than ERM with SPO.  This, combined with the relatively small size of the instances, signals a limited scalability.  
- **Distribution Shift Results and Reporting**: Ultimately, given that the focus of the paper is on distribution shifts, more needs to be done with respect to this.  The authors explore a single type of shift per problem and aggregate them across all different environments.  While this provides some information, de-aggregating and providing reporting of how each level of distribution shift is needed. Moreover, analysis beyond average regret, e.g., distributional/CVaR/worst-case, would give much more information in evaluation as well.  

  Building on this, presenting the experimental setup and comprehensive results for the distribution in the main paper would be appreciated.  Currently, only limited results are reported.  However, including shift-specific definitions and parameters, as well as results, should be prioritized in the main paper over redefining the standard predict-and-optimize approach.  One suggestion for the main paper results would be to use boxplots that demonstrate the distributional performance in each environment, rather than relying on single metrics averaged across all environments.  

- **Clarity**: One major weakness is the writing and clarity of the paper. Significant aspects, such as what the authors mean by 'environment,' are not clearly specified when introduced (an example would be helpful), ultimately making readability an issue.  

- **Author Engagement with Revisions**: I have reviewed this paper in the past, and I believe my colleagues have as well.  The authors have not made any notable changes to the manuscript after several resubmissions, so I am hesitant to recommend acceptance until I see a clear effort to address the numerous concerns that have not been addressed between resubmissions.

### Questions
- Figure 8a x-axis needs to be spaced better.  
- How were the environments selected? Have the authors considered a combination of multiple shifts?
- How many instances per environment are evaluated?
- How is the performance of all methods affected by the magnitude of the shift present in each environment?
- Is it possible to improve scalability by leveraging methods that have also been used in speed-up standard SPO frameworks?
- How does the validation set selected affect the quality of the performance under shift?

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
2

### Summary
The paper studies out-of-distribution generalization for predict-and-optimize (PnO) with combinatorial objectives (knapsack, visual shortest path, TSP). It proposes Inv-PnCO, a plug-in training objective that combines a mean-loss term with a cross-environment variance penalty. The goal is to learn invariant decision-oriented factors so that the induced solution distribution q(z∣x) remains close to the optimal p(z∣x) across environments. Theoretical claims include: 1) an information-theoretic bound suggesting that penalizing I(z;e∣y) reduces OOD decision error, and 2) a tractable surrogate showing that a mean+variance loss over environments upper-bounds the ideal objective. Empirically, on OOD shifts, Inv-PnCO reduces regret vs. ERM in both PtO and PnO pipelines, with no added test-time cost.

### Strengths
1. The setup relates to real problems (routing/logistics/planning) where costs are uncertain.

2. The experiments span arrays (knapsack), images (visual SP), and graphs (TSP) with distinct shift types, supporting the generality of Inv-PnCO.

3. The mutual information-based regularization, which converts to the variance across the environment, is naturally motivated. And the way of adding a regularization term does not introduce additional inference cost.

### Weaknesses
1. Benchmarks remain small (e.g., TSP-20, short grids). It would be great to test on larger instances, where 1) solver noise and suboptimality increase, 2) regret landscapes get spikier, and 2) environment variance explodes, causing the regularization term hard to balance.

2. Results mainly compare Inv-PnCO to ERM (both two-stage and SPO). Could distributionally robust optimization or OOD generalization-related works also be empirically compared?

3. Many tables emphasize mean regret drops without tight confidence intervals, worst-environment performance.

### Questions
1. The paper focuses on the fixed feasible set setting. Would it be possible to extend it to variable feasible sets?

### Soundness
3

### Presentation
3

### Contribution
3
