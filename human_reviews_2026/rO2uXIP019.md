# OrthAlign: Orthogonal Subspace Decomposition for Non-Interfering Multi-Objective Alignment

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Large language model (LLM) alignment faces a critical dilemma when addressing multiple human preferences: improvements in one dimension frequently come at the expense of others, creating unavoidable trade-offs between competing objectives like helpfulness and harmlessness. While prior work mainly focuses on constraint-based optimization algorithms and data selection strategies to mitigate conflicts, these approaches overlook the fundamental issue of resolving conflicts directly at the parameter level. In this paper, we present OrthAlign, an innovative approach that pioneers a new paradigm by leveraging orthogonal subspace decomposition to fundamentally resolve gradient-level conflicts in multi-objective preference alignment. OrthAlign strategically decomposes parameter update spaces into orthogonal subspaces, ensuring that optimization toward different preferences occurs in mathematically non-interfering directions. Building upon this, we provide theoretical guarantees demonstrating that when parameter increments satisfy both orthogonal subspace constraints and spectral norm bounds, the resulting updates exhibit linear Lipschitz growth rather than exponential instability, ensuring stable convergence across all preference dimensions. Extensive experiments show that: I. OrthAlign achieves maximum single-preference improvements ranging from 34.61% to 50.89% after multiple-objective alignment across helpful, harmless, and truthful dimensions. II. With an average overall reward improvement of 13.96%. Our code is available at https://anonymous.4open.science/r/OrthAlign.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles a specific problem in LLM alignment: resolving conflicts between multiple competing preferences. The core idea of using orthogonal subspace decomposition to create non-interfering parameter updates is novel and intuitive. The empirical results presented are strong, showing consistent advantages over a variety of baselines in balancing objectives and preserving the distributions of prior preferences.

However, these strengths are severely undermined by significant, disqualifying weaknesses in the paper's current form. The paper is exceptionally difficult to read and understand. The paper's core theoretical guarantees (Theorems 1 & 2)  are flawed.

### Strengths
1. The paper's key insight is to reframe the MPA conflict from a data or reward-level problem to a parameter-level problem of gradient interference. This is an elegant and promising direction for alignment research.

2. OrthAlign demonstrates SOTA performance across all reported experiments, outperforming 7+ baselines.

### Weaknesses
1. The final, practical algorithm is not presented in one clear, understandable place. The reader is forced to piece it together from a high-level description in Sec 3.1, a projection formula in Sec 3.3, and a critical but separate "Adaptive Subspace-Rank Selection" algorithm in Appendix B.1. This makes the method extremely difficult to follow and reproduce.

2. The paper is critically ambiguous about what matrix the SVD is performed on. Algorithm 1 (App B.1) states it is performed on the full parameter matrix $W$. However, the theory in Sec 3.1 is built around the LoRA adaptation matrix $\Delta W$. These are completely different matrices with vastly different computational and theoretical implications, and this ambiguity is unacceptable. This paper involves very complicated linear algebra-related notations, but the authors didn't pay much attention to making all the nottations clearly defined and consistent throughout the paper. 

3. The paper reuses the symbol $\tau$  for two completely different concepts, which is a major source of confusion. It is used to mean (1) the spectral norm bound in Theorem 2, and (2) the reward tolerance threshold in Algorithm 1 and Eq. 8. 

4. My major concern is for the significance of the theorems. From my perspective, the inequalities in Theorem 1 and Theorem 2 hold true in general when Assumption 3.1 holds. They are not new results for the particular setting considered in this paper.  This is also verified by the the proofs provided in Appendix F.  They are direct, first-principles applications of standard linear algebra tools: the Rayleigh quotient for Theorem 1, and the triangle inequality and orthogonality for Theorem 2. While correct (assuming the premises), the proofs themselves contain no technical novelty. The contribution is in the setup, not the derivation.

5. The provided theorems, even if they were verifiable, only offer guarantees at the parameter-weight level. Theorem 1 bounds the second-order parameter change, and Theorem 2 bounds the parameter matrix norm. Neither theorem provides a direct guarantee on the actual, high-level performance metric (i.e., the reward $\mathcal{R}(W; X_{safe})$). The link between "parameter norm stability" and "reward score stability" is non-trivial for a deep, non-linear model and is never formally established. 

6. We have no idea of how large $\tau$ is in Theorem 2. Thus, the practical algorithm (Algorithm 1) must fall back on an empirical reward-checking loop to find a safe rank $k^*$. This implicitly admits that the theory (Theorems 1 & 2) is insufficient on its own to guarantee performance, and a heuristic search is still required.

7. Lemma 1 relies on $\nabla g(\theta) \in \mathcal{S}_k$, but the paper fails to provide a justification for why the safety gradient ($\nabla g(\theta)$) would lie within the principal subspace ($\mathcal{S}_k$) of the safety curvature Hessian ($H_s$).

### Questions
Please see the weakness above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies multi‑objective preference alignment. The method proposed in this paper, OrthAlign, projects the gradient of each objective into the orthogonal complement of other preference subspaces. In this way, the learning of each objective won't conflict with the others. OrthAlign shows consistent improvements across multiple datasets and models compared to baselines.

### Strengths
The method is novel. The paper moves beyond the standard loss‑level or data‑mixing strategies for MPA and instead targets gradient‑level conflicts directly. 

The authors provide a well-motivated theoretical guarantee for their method.

Experimental results show that orthAlign consistently outperforms baselines. Rich analysis, including t‑SNE visualizations, ablation on rank selection, and case studies, further shows the effectiveness of their method.

### Weaknesses
See the question section.

### Questions
Orthogonal subspace learning, which prevents the new knowledge from overwriting old knowledge, is a common practice in continual learning. Does OrthAlign have a relation with that technique?

The current experiment is conducted in the off-policy setting. Is it possible to extend to on-policy alignment?

In OrthAlign, the update of each objective is specified in a subspace. Given a trained model, if masking the parameter update of a subspace, ideally, only the objective that corresponds to this subspace will be affected, while the other objectives maintain performance. I think this experiment will be a strong and direct verification of the effectiveness of OrthAlign.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents OrthAlign for multi-objective alignment. 
The authors argue that the common trade-off between multi-dim objects is due to "parameter-level antagonisms," essentially conflicting gradient updates.
They modify the LoRA finetuning process by adding a "gradient filter."
When training for a new objective, the method first uses SVD to identify a subspace for the existing objective. then it creates a projection matrix of its orthogonal complement to filter all gradients during training.

### Strengths
1. The method of filtering the orthogonal gradient when it comes to multi-objective learning is intuitively reasonable.  
2. The paper provides substantial math...
3. The method is built upon existing techs like Lora, and can be directly applied to algos like DPO. makes it plug-and-play.

### Weaknesses
1. The method introduces an order the objectives. The first objective aligned gets full protection, and every subsequent objective is forced to update only in the leftover, which is a flawed starting point for multi-objective alignment because it does not actually 'balance' them. 
2. The "Adaptive Rank Selection" (Algorithm 1) is likely too expensive for practical use, as it requires repeatedly running full reward evaluations over the "safe" dataset in a binary search just to determine the necessary filter size.
3. The algorithm seems ignore the paper's theoretical result; it relies on an expensive, empirical search guided by a reward "tolerance" $\tau$, rather than using the principled method suggested by its own math (analyzing the key $\lambda_{k+1}$ eigenvalues in Theorem 1).
4. The sequential approach is limited because the "safe" orthogonal subspace shrinks with every new objective added . This means the model will eventually run out of room to safely learn new things without conflict, thus limiting its scalability beyond a few objectives.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies multi-objective finetuning in large language models by projecting gradients on the orthogonal complement. This reduces gradient noise and leads to Pareto improvement in experiments. Their theoretical results show that training stability is maintained.

### Strengths
- The paper combines experimental results with a theoretical justification, and shows that Pareto improvements are possible with their method
- The method is simple to apply, tested against benchmarks (with up to 7B models) and added to other finetuning models.

### Weaknesses
- The theoretical results are not really contextualized. The main result (Theorem 2) seems to speak about blowup, but the intuition given the introduction seems to speak about more efficient parameter updates. Is there a result for a locally convex problem?
- Figures need additional work. It's unclear what Figure 4 should show / it's not clear how these point clouds look similar. (A hypothesis test would have been better to make this case imo); Figure 3 does not introduce the meaning of the size of the circle

### Questions
1. What do the circle sizes in Figures 1 and 3 mean?
2. Why does Figure 1 not contain error bars?

### Soundness
3

### Presentation
3

### Contribution
3
