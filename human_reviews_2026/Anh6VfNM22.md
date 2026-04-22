# CerCE: Towards Certifiable Continual Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Continual Learning (CL) aims to develop models capable of learning sequentially without catastrophic forgetting of previous tasks. However, most existing approaches rely on heuristics and lack formal guarantees, limiting their applicability in safety-critical domains. We introduce Certifiable Continual LEarning (CerCE), a CL framework that provides provable certificates of non-forgetting during training. CerCE leverages Linear Relaxation Perturbation Analysis (LiRPA) to reinterpret weight updates as structured perturbations, deriving constraints that guarantee the preservation of past knowledge. We formulate CL as a constrained optimization problem and propose practical optimization strategies, including gradient projection and Lagrangian relaxation, to efficiently satisfy these certification constraints. Furthermore, we connect our approach to PAC-Bayesian generalization theory, showing that CerCE naturally leads to tighter generalization bounds and reduced memory overfitting. Experiments on standard benchmarks and safety-critical datasets demonstrate that CerCE achieves strong empirical performance while uniquely offering formal guarantees of knowledge retention, marking a significant step toward verifiable continual learning for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CerCE, a continual learning method that provides provable certificates of non-forgetting during training. It treats weight updates as structured perturbations and applies LiRPA to guarantee that past samples remain correctly classified within a small parameter radius. The method is formulated as a constrained optimization problem, solved via gradient projection or a Lagrangian relaxation. Experiments show that the final accuracy is similar to other baselines, while achieving high Average Certification (AC) (a measure of how many stored samples are provably safe from forgetting).

### Strengths
1. Introduces a principled framework for provable non-forgetting using LiRPA, which is a novel direction in continual learning.

2. Demonstrates that accuracy can be maintained while adding certifiable safety constraints.

3. Provides empirical validation on both standard and safety-critical datasets such as RUARobot.

### Weaknesses
Major concerns:
1. The Average Certification (AC) metric is computed only on the replay buffer used during training (Lines 357-359). This means the certificates guarantee non-forgetting only for those stored samples, not for unseen or full-task data. It also creates a circular evaluation: the model is guaranteed to be "certified" on the very examples it was trained to protect. This limits the interpretability of AC as a genuine measure of generalization or robustness.

2. CerCE achieves final accuracy comparable to rehearsal-based baselines such as ER and DER++ (Table 1). While this parity shows that certification does not harm learning, it also indicates that the method offers no empirical performance gain, with its contribution being primarily theoretical rather than practical.

3. The reported training cost (9 times slowdown stated in Appendix E) may limit scalability to complex architectures and realistic continual learning cases.

### Questions
1. Can authors clarify whether the certification guarantees (AC) generalize beyond the replay buffer? In other words, does high AC correlate with actual non-forgetting on unseen or test data?

2. Since CerCE's final accuracy is similar to other baselines, what practical advantages should we expect from adopting it beyond the theoretical certificates? 

3. The reported 9 times slowdown is significant. Are there any feasible ways to reduce it for larger models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Certifiable Continual Learning (CerCE), a novel framework that provides formal guarantees against catastrophic forgetting in continual learning by leveraging Linear Relaxation Perturbation Analysis (LiRPA). CerCE reinterprets weight updates as structured perturbations, deriving constraints to ensure the preservation of past knowledge and formulating continual learning as a constrained optimization problem. The framework employs practical strategies like gradient projection and Lagrangian relaxation and connects to PAC-Bayesian generalization theory, leading to tighter generalization bounds and reduced memory overfitting. Experimental results on various datasets showcase CerCE's strong empirical performance and its unique ability to offer theoretical guarantees for knowledge retention, paving the way for verifiable continual learning in safety-critical applications.

### Strengths
- A novel approach that attempts to establish "guardrails" for continual learning (CL), with potentially great impact in safety-critical applications.
- The continual learning field often lacks standardization and formal approaches, relying heavily on heuristics. The framework introduced by the authors is refreshing and has the potential to inspire new avenues for formal methods in CL.
- I am not an expert in PAC-Bayes Bounds, but the authors are able to tighten a classic bound by theoretically guaranteeing performance on a buffer memory set.

### Weaknesses
- I am not entirely comfortable with claims of "theoretical guarantees" paired with "linear relaxation". I would rather describe these as theoretically grounded certificates. The tighter PAC-Bayes bound also relies on the assumption that this relaxation formally guarantees performance on the buffer set (or buffer minibatch), but whether this holds true in general is not clear to me.
- I believe more could be done to connect certificate satisfaction and backward generalization in order to validate the approach.
- Applicability is tied to architectures where LiRPA is computable. Not being familiar with LiRPA, I question the scalability of such an approach with respect to architectural complexity.

### Questions
1. Exposition could be improved:
    - The notation in Theorem 1 and Corollary 1.1 could be clearer. I had to consult the appendix and reference works to fully understand it. Even simply introducing the dimensionality of the bound parameters could aid readability.
    - Figure 2: The y-axis of the second plot is missing.
    - Ablation 5.2: "Ratio of Certified Samples in the Buffer" does not seem like an ablation study; I suggest moving it to the experimental section or appendix.


2. Questions:
    - Table 3: If the "bound" case includes all samples that satisfy the constraint, wouldn't that imply an Average Certification (AC) of 100%? Are you computing AC over the entire buffer instead of just the buffer sample?
    - Following the corollary on the PAC-Bayes bound, you state that it leads to less overfitting on the memory buffer. While an argument for better "backward generalization capabilities" can be made, the implication regarding buffer memory overfitting is less clear to me (especially since in Corollary 2.1 you assume that `\max_{\|\Delta \theta\|_2}\hat{l}_\mathcal{M}(\theta)=0`).
    - Could this method actually be used to train with guarantees of non-forgetting in practical scenarios, or is it currently more of a theoretical possibility? What are the limitations in this regard?
    - It would significantly increase the soundness of the paper if you empirically demonstrate the connection between certificate satisfaction and per-task backward generalization (BG) as new tasks are learned. Does task performance retention scale with the satisfied certificates per class?
    - It would be reassuring to mention the limits of this methodology (if any), such as its scaling with respect to network depth or general issues with the approximation in the weights space.


Satisfactory answers to these points could lead me to re-evaluate my grade, and I believe it would significantly strengthen the work. I'm looking forward to the discussion, and I encourage the authors to continue this interesting line of research.

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
This paper introduces Certifiable Continual LEarning (CerCE), a framework designed to address catastrophic forgetting in continual learning (CL) by providing formal guarantees of non-forgetting. The authors utilize Linear Relaxation Perturbation Analysis (LiRPA) to interpret weight updates as structured perturbations, thereby deriving constraints that aim to preserve previously acquired knowledge. The method is formulated as a constrained optimization problem and evaluated on standard CL benchmarks as well as safety-critical datasets.

### Strengths
Adding formal guarantees to continual learning is particularly relevant for the deployment of CL systems in safety-critical applications (a domain appropriately explored in the experiments). The core idea of leveraging neural network certification techniques (specifically LiRPA) to define constraints on parameter updates is innovative and presents a novel angle in the field, as far as I know. The concept of analyzing parameter permutations that lead to changes in accuracy (the "forgetting-set") is powerful and opens up a potentially rich area for future research, bridging the gap between the certification and CL literature.

### Weaknesses
While the core idea is promising, the paper in its current form feels underdeveloped and would benefit significantly from a deeper analysis of the proposed mechanisms. The authors introduce several interesting concepts (e.g., LiRPA coefficients, certification constraints) but stop short of exploring their full implications.

The main weaknesses are:

* **Insufficient Analysis:** The paper would be much stronger if it focused more on the ablation and analysis of the approach. For example, a deeper discussion of what the LiRPA coefficients specifically capture about the data or the parameter space would be highly valuable. It is also unclear how these certification-based constraints relate to existing approaches that impose constraints on parameter updates in CL (e.g., projection-based methods).
* **Unfair Complexity Comparison:** The experimental comparison to baselines (e.g., ER) appears unfair, as the proposed method introduces significant computational overhead. The paper requires a principled discussion of the trade-off between this extra complexity and performance. An analysis of the computational complexity *per buffer sample* for CerCE versus baselines, followed by a comparison of performance at equivalent complexity budgets, would be necessary to fairly situate the method's contributions.
* **Misplaced Theoretical Focus:** The discussion on PAC-Bayesian bounds seems peripheral to the paper's central claim of certification. This section does not add significant value and feels disconnected from the primary certification objective. This space might be better utilized for the deeper analysis mentioned above, such as exploring the relationship between the constraints and the properties of the loss landscape (e.g., flatness).
* **Lack of Experimental Clarity:** The experimental setup description is vague. The authors state they use MLPs on top of pre-trained architectures, but the exact details of these architectures and the complexity of the trained MLPs are missing, making replication difficult.
* **Formatting and Presentation:** The paper suffers from numerous formatting and stylistic issues, which detract from the main message. Figure 2 is of poor quality, table formatting is inconsistent (e.g., Table 1), and the appendix is minimal and poorly structured. For example, Appendix F separates standard deviations into their own table and confusingly labels them "error bars," which is highly unconventional. Phrasing such as "What goes on in the loss" is not appropriate for a formal publication.

### Questions
1.  Could the authors clarify how tight the certification guarantees are in practice? For instance, if a sample breaks the certification guarantee during an update, how often does this correlate with the sample actually being misclassified (i.e., forgotten)?
2.  Many results (e.g., average certification) are reported on the replay buffer. Given that the buffer is only a small subset of past data, could the authors comment on why certification on the buffer is the primary metric, rather than guarantees that might apply to the true (unseen) data distribution of past tasks?
3.  Is there a direct, empirically verifiable connection showing that the specific samples *most likely* to be forgotten (e.g., those near a decision boundary) are the ones primarily protected by the certification constraints?
4.  To better contextualize the trade-off between standard accuracy and certification, it would be beneficial to see a visualization of the accuracy drop on *previous tasks' test sets* (not the buffer) after each new task is learned, for all compared methods.
5.  The LiRPA method is central to the paper but is not introduced with sufficient intuition in the main text. Could the authors provide a more self-contained, high-level overview of what these coefficients represent in the main paper?
6.  The cited literature on NN certification appears to be from 2021 or older. Could the authors comment on whether more recent developments in LiRPA or other certification methods might be applicable or simplify their framework?

### Soundness
3

### Presentation
2

### Contribution
3
