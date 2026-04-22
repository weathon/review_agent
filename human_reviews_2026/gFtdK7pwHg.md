# MASS-DPO: Multi-negative Active Sample Selection for Direct Policy Optimization

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 6, 2, 6

## Abstract
Multi-negative preference optimization under the Plackett–Luce (PL) model extends Direct Preference Optimization (DPO) by leveraging comparative signals across one preferred and multiple rejected responses. However, optimizing over large pools of negatives is computationally prohibitive, and many candidates contribute redundant gradients due to their similar effects on policy updates. 
To address this, we introduce \textbf{MASS-DPO}, which derives the Fisher information matrix directly from the PL objective and shows that the problem of selecting negatives naturally reduces to a D-optimal design formulation. This formulation guarantees maximal informativeness and comprehensive coverage of the current policy’s weaknesses. Moreover, the log-determinant criterion underlying D-optimal design admits a submodular structure, which we exploit through an incremental greedy algorithm that provides the natural computational realization of D-optimality, combining scalability with theoretical rigor. This incremental greedy strategy efficiently resolves the combinatorial complexity inherent in selecting a D-optimal negative set from large candidate pools. We establish convergence guarantees and finite-sample error bounds under this framework, and empirically demonstrate that MASS-DPO improves optimization efficiency and enhances downstream performance, achieving stronger alignment with substantially fewer negatives.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces MASS-DPO, a framework for multi-negative active sample selection in the context of Direct Preference Optimization (DPO). Standard DPO uses pairwise preference comparisons between a preferred and a non-preferred response. Multi-negative variants (e.g., S-DPO, DMPO) extend this to multiple negatives, but they typically select negatives randomly, leading to redundant gradients and inefficient learning. MASS-DPO addresses this by casting negative selection as a D-optimal design problem by maximizing the log-determinant of the Fisher Information matrix derived leveraging the PL model for preference over objects.

### Strengths
1. Prior multi-negative DPO methods (S-DPO, DMPO) rely on random or heuristic selection, while this paper is the first to formalize active negative selection as an optimal design problem.

2. The link between Fisher information geometry and preference optimization is conceptually appealing and well-motivated.

3. **Algorithm Design.** The greedy algorithm is quite fundamental and shows complexity $O(d^2 n)$ that scales linearly in the number of selected negatives, which is significantly better than the exponential combinatorial search. The paper shows how the greedy step $v_i^{\top}H^{-1}v_i$ naturally picks negatives probing unexplored directions, which in my opinion intuitive and useful.

4. **Theoretical Guarantees.** Proofs of the theoretical guaranties are rigorous and seems correct.

5. **Empirical Performance.** MASS-DPO improves accuracy and margin across almost all configurations (Tables 1–4). Ablations on $\beta$ and number of negatives $k$ match theoretical predictions (performance saturates as information coverage increases). 

6. **Writing and Presentation.** The paper is well-structured, logically ordered, and concise for its complexity. Mathematical notation is consistent, derivations flow logically. I have found no major typographical errors.

### Weaknesses
1. While it seems correct, authors should justify the inequality in Equation (10).
2. Equation (11) mixes constants $\gamma$ and $\beta^2 (1-\sigma(Z_n)$, these should be clarified for reproducibility.
3. Some equations (e.g., 5) have redundant parentheses and inconsistent minus signs in exponents, cosmetic but worth cleaning.
4. Some references (e.g., Pukelsheim 2006a,b) are repeated, therefore can be merged.

### Questions
1. **Lemma 4.3.** I believe that the claim of the greedy algorithm exactly matching the global D-optimal solution is too strong. Standard submodular optimization results (Nemhauser et al., 1978) guarantee a $1 - \frac{1}{e}$ approximation, not equality. Unless all $v_i$ are orthogonal or additional structure holds, perfect equivalence is implausible. 

2. **Empirical Scope.** Evaluation tasks are mainly recommendation and multiple-choice QA, which are relatively structured. It remains unclear if MASS-DPO scales to open-ended generation or instruction tuning scenarios where negatives are more diverse.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes MASS-DPO, an extension of DPO that uses D-optimal design and incremental greedy information maximization to select informative preference samples across one preferred and multiple rejected responses. Theoretically, it links D-optimality to guarantee finite-sample estimation error
bounds and convergence properties of the proposed algorithm. Empiracally, it performs comparably to baseline methods in both recommendation and multiple-choice QA tasks.

### Strengths
- The MASS-DPO generalizes single-negative selection to multi-negative cases, enhancing exploration efficiency, with well-structured theorems and proofs linking information gain to D-optimality.
- The theoretical part is well written, and provides a clear description on how information gain can be maximized through the greedy step, and on how computaional overhead can be reduced.

### Weaknesses
- The proposed algorithm is similar to Active DPO (Kveton et al., 2025, the authors also cited this paper), which already uses D-optimal design and greedy information maximization. The main theorem also follows similar statement and proof idea, raising concerns about the incremental contribution beyond extending from single to multiple negatives. Can authors elaborate more on the difference/improvement?

- More importantly, in Figure 2, MASS-DPO does not consistently outperform S-DPO — performance gains are small or even negative across tasks. Similarly, in Table 2, the results seem random regarding which method performs best. This undermines the claim that the theoretical advantages (in sample efficiency or information gain) translate into tangible empirical benefits.

### Questions
See limitations.
- I didn't check all the proof, but wondeing if there should be $\beta$ term in the bound of Theorem 5.2.
- Could the authors provide empirical runs to validate the claimed improvement? Or given the similar performance, what are the practical conditions (dataset structure, model uncertainty, preference noise) under which MASS-DPO yields clear advantages?
- What is the computational cost comparison for MASS-DPO and S-DPO?
- I would suggest elaborate more on previous literature on information-theoretic sample selection.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses the problem of negative sample selection in multi-negative Direct Preference Optimization (DPO). While recent methods like Softmax-DPO extend DPO to leverage multiple negative responses, they typically select negatives randomly or heuristically, leading to redundant gradients and computational inefficiency. MASS-DPO formulates negative selection as a D-optimal design problem that maximizes the log-determinant of the Fisher information matrix. The key insight is that informative negatives should be diverse and orthogonal in feature space, thereby maximizing information gain about policy parameters. To make this computationally tractable, the authors propose an incremental greedy algorithm that iteratively selects negatives maximizing the induced norm $v^\top H_k^{-1} v$. The theoretical contribution is about finite-sample error bounds (Theorems 5.1-5.2). Experiments demonstrate improved optimization efficiency and downstream performance compared to baselines.

### Strengths
- **Principled theoretical framework:** Unlike existing multi-negative methods (S-DPO, DMPO, MPPO) that use random or heuristic selection, MASS-DPO provides a theoretically grounded approach based on optimal experimental design
- **Comprehensive evaluation:** Experiments span multiple task types (recommendation, QA), datasets, and model families with consistent improvements
- **Practical impact:** Achieves comparable or better results with fewer negatives (e.g., k=3) compared to methods that might use more samples inefficiently

### Weaknesses
### CRITICAL THEORETICAL ERROR

1. **Lemma 4.3 is incorrect for *general case*.** The greedy algorithm does not always find the globally optimal solution for D-optimal design. The counterexample is: $\lambda=1, v_1=[1.1, 0]^\top, v_2=[1,1]^\top/\sqrt{2}, v_3=[1,-1]^\top/\sqrt{2}$, select $2$ vectors.
Greedy selects $v_1$ first (largest norm), but optimal is $\\{v_2, v_3\\}$. I'm not sure if the special property of $v_i$ tightly coupled with the feature mapping and probability $p_i$ can be utilized to show correctness. Nonetheless, this is also not mentioned in the proof. This **CRITICAL ERROR** greatly harm the soundness of this paper.

### Other Concerns

2. **Dynamic selection unclear:** Since $p_i$ depends on current policy $\theta$, it's unclear whether the negative subset needs to be re-selected after each parameter update. The paper doesn't address this practical concern.

3. **Missing definitions:** Critical terms like $\theta_*$ and $c_{\min}$ are referenced in Theorem 5.2 but never defined in the main text.

4. **Limited theoretical novelty:** The D-optimal design framework and greedy selection are well-established (Kveton et al., 2025). The main contribution is applying them from standard DPO to multi-negative DPO.

5. **Presentation issues:** "LLM Usage" paragraph weirdly placed before experimental results

### Questions
Please address my concerns in Weaknesses section.

### Soundness
1

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
This paper considers selection of negative examples to include when training models on preference feedback using direct preference optimization. When a human selects one preferred positive example, including all of the negatives can lead to slow training. Many of the negatives may be redundant with other negatives or already be classified well by the model, suggesting that we can create comparable model quality more quickly by only including the most informative negatives. This paper a novel method for selecting negatives and supports it with theoretical analysis and empirical comparisons to recent benchmarks.

### Strengths
This paper studies an important problem, has a thorough theoretical analysis, and presents competitive empirical results against a number of recent baselines.

### Weaknesses
Based on my current understanding of the paper, the main weakness is that it selects a subset based on freezing all but the last layer and assumes that this selection will be effective when training the whole network. See Questions #1 and #4 below.

There are also some typos that limited understanding and my confidence in correctness. See questions #2, and #5.

### Questions
#1. My understanding of Assumption 4.1 is that we do not think about back-propagating gradients further back through the network architecture.  We only think about the effect of training the model weights in the last layer. Is this correct? If so, this seems like a major limitation of the work. At a minimum, readers should be made more aware of this limitation and ideally it would be analyzed.

In the numerical experiments, based on B.3, it seems that the whole network is trained. Is that correct?  This is good, if true.  More details should be added to B.3 and perhaps Section 6 to make this point more clear.

Empirically, it would be good to understand the gap between computing things with just the last layer vs. the whole network --- if one were to select subsets based on the log det of the regularized Fisher information matrix computed by backprogating through the full network, how different would the selected subsets be? Perhaps this is not computable at scale with an LLM, but it could be answered on some smaller-scale example.

#2. Either there is an important typo or I am confused about the paper.
On line 192-3 of page 4, n is given as the *total* number of negative responses.
On line 202-3 of page 4, it is stated that n is the *chosen* negatives.
The number of chosen negatives would typically be strictly smaller than the total number of available.
I read the rest of the paper assuming that n is the chosen number of negatives and that there is some unstated larger number of negatives from which these can be drawn.

#3.  The typo above made it hard to understand the following:

The notation H(S) in equation 11 uses Z_n -- this is held fixed during the greedy optimization proposed in lines 13 and 14.  I *think* this is the Z matrix that you get when you include *all* of the negatives. If it were the one you get when you just include the optimal ones, then it's unclear to me how you know it during the greedy procedure.

But if it is *all* of the data, then H(S^*_n) isn't the same thing as computing the log det of equation 10 with the selectetd negatives S^*_n. How do we know it is a good approximation? Presumably, the theory wants to hold sigma(Z_n) fixed during the greedy algorithm because it makes analysis simple.  But this seems to be another limitation of the method. 

#4. Please walk me through how we get computational savings from selecting a strict subset of negatives.  The greedy algorithm requires us to compute v_j vectors for all of the negatives.  If the model actually satisfied Assumption 4.1, this seems like almost the same work as computing the gradient of the loss including all of the negatives. Does the savings come from the fact that we freeze everything but the last layer when we compute the v_j, but then when we actually do backpropagation, we compute through the whole network?

#5. In the high-probability bounds (Theorems 5.1 and 5.2), what is random?  I tried to figure this out by reading the proof of 5.1 in Appendix A.1 but it seems to rely heavily on Kveton et al. 2025 and uses notation that I didn't see defined elsewhere (Sigma_n, c_{min}). Also check the typo "covariant". Why does equation (26) follow with probability at least 1-delta?  And should the inequality above from Kveton et al. 2025 also be stated as holding only with high probability?

#6. The experiments state that the number of negatives was expanded from 4 to 20 for MedMCQA. Was this because the method provides little value when only 4 negatives are present?

### Soundness
3

### Presentation
3

### Contribution
3
