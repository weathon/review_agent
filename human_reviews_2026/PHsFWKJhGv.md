# Tighter Performance Theory of FedExProx

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
We revisit FedExProx -- a recently proposed distributed optimization method designed to enhance convergence properties of parallel proximal algorithms via extrapolation. In the process, we uncover a surprising flaw: its known theoretical guarantees on quadratic optimization tasks are no better than those offered by the vanilla Gradient Descent (GD) method. Motivated by this observation, we develop a novel analysis framework, establishing a tighter linear convergence rate for non-strongly convex quadratic problems. By incorporating both computation and communication costs, we demonstrate that FedExProx can indeed provably outperform GD, in stark contrast to the original analysis. Furthermore, we consider partial participation scenarios and analyze two adaptive extrapolation strategies -- based on gradient diversity and Polyak stepsizes -- again significantly outperforming previous results. Moving beyond quadratics, we extend the applicability of our analysis to general functions satisfying the Polyak-Łojasiewicz condition, outperforming the previous strongly convex analysis while operating under weaker assumptions. Backed by empirical results, our findings point to a new and stronger potential of FedExProx, paving the way for further exploration of the benefits of extrapolation in federated learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper first identifies a limitation in the original analysis of FedExProx that its known guarantees are no better than vanilla GD. To address this issue, the authors proposed a tighter linear convergence analysis for FedExProx for non-strongly convex quadratic optimization problems. The main result shows that FedExProx outperforms GD when accounting for computation/communication costs. Further, the paper analyzes partial participation and two adaptive extrapolation strategies, and extends the results to more general settings under the Polyak-Łojasiewicz (PL) condition. Some illustrative experiments are provided to validate the theoretical findings, highlighting the potential of extrapolation in federated learning.

### Strengths
1. The work is well motivated and clearly presented in general.

2. The theoretical analysis is technically sound and comprehensive, with sufficient details provided in appendix. The results demonstrate that FedExProx is superior when communication dominates local computation (which is common in FL), and thus are potentially applicable to heterogeneous FL environments.

### Weaknesses
1. The theoretical analysis focuses primarily on niche scenarios like quadratic programs or functions satisfying the PL condition. This narrows its applicability to practical non-convex FL tasks. Moreover, the Assumption 1.3 essentially requires all the clients to share the same minimizer, which is an extremely unrealistic setting: If each individual client is already able to find the common global mimimizer based on their own data, then why should we bother to do FL from the very beginning? 

2. While the linear convergence rates are refined for FedExProx in the quadratic case, the novelty of analysis looks incremental with relatively straightfoward proof techniques.

3. The empirical results highlighted in the main paper are somewhat supportive of theory verification. However, the experimental study still falls short in testing on benchmark FL datasets with heterogeneous client setups or real network latency.

### Questions
1. Can the Assumption 1.3 be relaxed, at least in the quadratic case?

 2. What are the main technical challenges faced by the analysis of FedExProx in the considered settings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors revisit FedExProx, an extrapolated proximal algorithm for federated learning (FL), and find that its prior theoretical guarantees don’t offer any improvement over vanilla Gradient Descent (GD) for quadratic problems. They develop a tighter performance analysis showing that FedExProx achieves a linear convergence rate and can provably outperform GD when communication costs dominate computation—a crucial scenario in FL. Additionally, they extend the analysis to partial participation scenarios, introduce adaptive extrapolation strategies based on gradient diversity and Polyak step sizes, and generalize the results to functions satisfying the Polyak–Łojasiewicz (PŁ) condition, thereby relaxing the requirement for strong convexity.

### Strengths
- The paper pinpoints a flaw in prior analysis, which suggested FedExProx had no advantage over vanilla GD on quadratic tasks. It then resolves this by presenting a refined analysis that overturns that pessimistic result. This corrects our understanding and is a valuable theoretical insight.

-  Using a novel analysis framework, the authors prove tighter convergence rates. The work also extends FedExProx’s theory to more challenging scenarios.

-   The analysis explicitly incorporates communication vs. computation trade-offs, reflecting real federated learning conditions. The paper shows that FedExProx’s total time can be lower than GD’s when network latency is the bottleneck, which is a practical strength. It’s valuable that the theory isn’t purely iteration-count based, but considers wall-clock time under FL constraints.

### Weaknesses
- The tighter theoretical results, while impressive, apply under specific assumptions that may limit practical generality. For example, the analysis assumes an interpolation condition (existence of a common minimizer across clients), which may not hold in real federated data (where different clients might not share an exact global optimum).

- FedExProx, introduced by Li et al., is not a novel algorithm. This submission primarily contributes an improved analysis of it. While the advances are mostly theoretical, they don’t propose a fundamentally new approach. Therefore, the novelty may be perceived as incremental, as it refines our understanding of an existing method rather than introducing a brand-new algorithm.

- The paper’s presentation leans heavily on theoretical derivations and dense mathematical details. Readers with only a moderate background might find it challenging to follow some parts. For instance, understanding the implications of constants like $L_γ$ or $μ^+_γ$ in Theorem 4.1 requires significant effort. Important concepts (e.g., the PL condition or Moreau envelopes) are used somewhat tersely. Overall, the exposition could be clearer with more intuition or examples.

### Questions
- How critical is the interpolation assumption (Assumption 1.3) for FedExProx’s performance? In realistic non-interpolating scenarios (where clients’ objectives have different minimizers), would FedExProx still converge or provide benefits, or does the theory break down completely?

- The theoretical advantage of FedExProx is clear when communication latency dominates. Can the authors provide intuition or examples of real-world settings where the time savings from FedExProx are significant?

- The paper proposes adaptive extrapolation strategies (FedExProx-GraDS, FedExProx-StoPS). What is the overhead or complexity of implementing these strategies in practice?

- The empirical results were mentioned only briefly. Could the authors elaborate on the experimental setup and findings?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper improves the existing convergence results of FedExProx in several regimes. The authors first demonstrate that FedExProx can outperform gradient descent when minimizing non–strongly convex quadratic objectives, under both full and partial client participation. They further propose two adaptive strategies for selecting the extrapolation parameter and analyze the convergence of the resulting methods under the same settings. Finally, the theoretical results are extended to general smooth convex functions that satisfy the PL condition.

### Strengths
1. The paper is clearly written.
2. The improvements of the theoretical results are interesting and novel, showing the potential benefits of doing the extrapolation step under interpolation and the PL condition, when communication is much more expensive than local computation.
3. The paper extends the analysis by allowing the use of adaptive stepsize.

### Weaknesses
The scope of the experiments is a bit limited, without comparisons with other commonly used FL optimization methods.

### Questions
1. It seems that FedExProx can potentially perform well when $\frac{L_{\gamma}}{\mu_{\gamma}^+}$ is relatively small. Since you provide one example of optimizing diagonal quadratic matrices, I am wondering in general, when this quantity can become small, e.g., separable functions? 

2. Is it possible to further consider acceleration for FedExProx, i.e., to get $\sqrt{\frac{L_{\gamma}}{\mu_{\gamma}^+}}$?

3. Is there any regime when $\eta \approx \gamma$, FedExProx is still faster than GD by a large margin?

### Soundness
3

### Presentation
3

### Contribution
3
