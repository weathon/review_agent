# Test-time Verification via Optimal Transport: Coverage, ROC, & Sub-optimality

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
While test-time scaling with verification has shown promise in improving the performance of large language models (LLMs), role of the verifier and its imperfections remain underexplored. The effect of verification manifests through interactions of three quantities: (i) the generator’s *coverage*, (ii) the verifier’s *region of convergence* (ROC), and (iii) the sampling algorithm’s *sub-optimality*. Though recent studies capture subsets of these factors, a unified framework quantifying the geometry of their interplay is missing. We frame verifiable test-time scaling as a transport problem. This characterizes the interaction of coverage, ROC, and sub-optimality, and uncovers that the sub-optimality-coverage curve exhibits three regimes. A *transport regime* — where sub-optimality increases with coverage, a *policy improvement regime* — where sub-optimality may decrease with coverage, depending on the verifier’s ROC, and a *saturation regime* — where sub-optimality plateaus, unaffected by coverage. We further propose and analyze two classes of sampling algorithms — *sequential* and *batched*, and examine how their computational complexities shape these trade-offs. Empirical results with `Qwen`, `Llama`, and `Gemma` models corroborate our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
This paper provides a framework for understanding test-time verification in LLMs by framing it as an optimal transport problem. The authors identify three components they consider important in the problem of verified generation: 

(a) coverage, i.e. the generator's support over the solution space
(b) verifier ROC, i.e. the region of convergence and selectivity of the verifier
(c) sub-optimality, i.e. the algorithm's deviation from the optimal policy.

The authors model these factors as a transport problem and show three distinct "regimes": transport, policy improvement, and saturation. Then they analyze sequential and batch sampling algorithms and their computational trade-offs. The overarching goal of this paper is to understand and unify the empirical progress in test-time scaling on the one hand, and a mathematical framework describing how verification quality affects performance, on the other. 

The authors show that verification helps only in certain regions of the coverage-suboptimality space. Concretely, in the policy improvement regime with moderate coverage, the verification yields the largest performance gains where it reduces sub-optimality.  The reason why verification helps is also analyzed.

That said, the practical implementation and the paper overall is outside of my area of expertise. I've alerted the ACs that I cannot judge this paper and giving it a confidence of 1.

### Strengths
See summary.

### Weaknesses
See summary.

### Questions
See summary.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the test time verification as a sampling problem with a lens of optimal transport. The framework of optimal transport where given a proposal distribution $\mu$, the goal is to sample from $\nu ^\star$ that is defined by $r ^\star$. The problem here is usually at test time, $r^\star$ is not available to guide the generation and thus one needs to rely on imperfect reward models. This paper also studies the geometry of sub optimality and coverage by relating the suboptimality to verifier's region of convergence (ROC) and generator's coverage, revealing three different regimes (transport regime, policy improvement regime and saturation regime). The paper analyzes: (a) sequential protocols—Accept-if-Correct (AiC), Sequential Rejection Sampling (SRS), Sequential Maximal Coupling (SMC); and (b) batched protocols—Best-of-N (BoN) and Batched Rejection Sampling (BRS). It proves AiC can violate the coverage constraint in low-coverage regimes and provides closed-form sub-optimality/complexity for SRS/SMC and exponential-in-N decay for BRS. Empirically, the paper focuses on GSM8K with Qwen, Llama and Gemma models and demonstrates their behaviour across different test time compute algorithms.

### Strengths
* Theoretical decomposition of sub-optimality into OTC is clear and a verifier-driven factor three-regime picture is intuitive and well explained.
* Precise AiC analysis with explicit coverage violation condition and sub-optimality formulas.
* SMC construction via maximal coupling with matching complexity to SRS; neatly links transport optimality with compute.
* BRS exponential improvement with tight envelope; theoretically beats BoN in low/intermediate regimes.
* Empirical studies match theoretical findings.

### Weaknesses
* GSM-8K is a very limited dataset, and especially as a test-time compute paper, I strongly recommend using a reasoning benchmark. You can either one or multiple of the AIME, OlympiadBench, MATH, GPQA or AMC questions to test the validity of the approaches here.
* The analysis of hybrid approaches between sequential and batched protocols such as beam search (also called as block-wise Best of N) is required.
* Actionable insights is not clear. How can the theoretical bounds and observations here help developers obtain state of the art performance across reasoning benchmarks? 
* The narrative repeatedly generalizes that the verifier’s ROC “mediates” the coverage–sub-optimality trade-off, yet controlled experiments are limited to Qwen in the main text (Gemma/Llama only in appendix) and to one prompt. The strength of the claim should be toned down or supported with wider evidence.

Note: I am willing to update my score based on the rebuttal.

### Questions
* The acronym OTC and OHC both refer to optimal transport cost, right? Why does the paper use both interchangeably? I recommend sticking with one of them. 
* The explicit verifier is constructed by ranking and selecting responses to hit preset $J$ and $s_{\text{ver}}$. Can you add experiments with more realistic verifiers (e.g., harness parser variants, unit-test fuzzing, LLM-judge with calibration) and show robustness of your theorems’ qualitative predictions under such noise?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper develops a theoretical framework for test-time verification in LLMs using optimal transport theory. The authors characterize how three factors interact: generator coverage, verifier accuracy (region of convergence), and sampling algorithm sub-optimality. The main contribution is identifying three distinct regimes in the sub-optimality–coverage relationship: transport, policy improvement, and saturation. The paper analyzes both sequential (AiC, SRS, SMC) and batched (BoN, BRS) sampling algorithms, deriving exact bounds on computational complexity and sub-optimality. Experiments on GSM8K with Qwen, Llama, and Gemma models validate the theoretical predictions.

### Strengths
- The optimal transport formulation elegantly unifies the analysis of generator coverage, verifier imperfections, and sampling strategies. This provides a principled way to reason about test-time verification that goes beyond existing asymptotic analyses.

- The paper derives closed-form expressions for sub-optimality (Theorems 3.6, 3.8, 3.10) and computational complexity (Theorems 3.2, 3.5) across multiple algorithms. These exact results are more informative than asymptotic bounds for practical algorithm selection.

- Identifying transport, policy improvement, and saturation regimes is valuable both theoretically and practically. The analysis shows that the relationship between coverage and sub-optimality is non-monotonic and depends critically on verifier quality (Youden's index).

- Explicitly incorporating false positives and false negatives through ROC analysis (TPR, FPR) is important for practical deployment. Most prior work assumes perfect verifiers, which is rarely realistic.

- The paper provides actionable guidance: rejection sampling methods (SRS, BRS) are preferable under tight coverage constraints, while Best-of-N methods work better with relaxed coverage. This addresses a practical question practitioners face.

- Figures 4-5 show close alignment between theoretical predictions and empirical results across different models and parameter regimes, which strengthens confidence in the framework.

- The paper effectively uses figures (especially Figures 1-2), tables, and clear regime-based decomposition to communicate complex mathematical ideas.

### Weaknesses
- All experiments use one GSM8K question (sample 2), following Dorner et al. (2025). While this protocol has precedent, it limits our ability to assess whether the three-regime structure generalizes across problems of varying difficulty and characteristics. Even testing on 5-10 representative problems would strengthen the claims.

- SRS and SMC require knowing s_ver (the reference policy's mass on the verifier set), which may not be available at test time. Section 4 treats it as a hyperparameter, but Figure 5 shows that misspecification substantially degrades performance. The paper would benefit from a practical estimation procedure.

- The ℓ1-type constraint (Equation 1) based on χ²-divergence is mathematically convenient but somewhat restrictive. The authors discuss this limitation briefly but don't explore how alternative divergences (e.g., KL, Wasserstein) might change the regime structure. This seems like an important direction for future work.

- The experimental protocol samples with replacement from a pre-generated pool of 10,000 responses. This doesn't fully capture the parallel generation scenario that motivates batched methods in practice, where responses are generated independently in a single forward pass.

- While the paper analyzes expected number of proposals, wall-clock time depends on GPU parallelization, model size, and verifier overhead. Sequential methods may require fewer total proposals but could be slower than batched methods in practice. This practical tradeoff deserves discussion.

- The main experiments construct verifiers by controlling J and s_ver through probability-ranked selection. Real verifiers (unit tests, reward models) don't work this way. The reward-guided experiments (Appendix) are more realistic but limited to two models with FPR=0.3.

- The paper compares proposed algorithms against each other but not against recent test-time scaling methods like guided stream-of-search (Gandhi et al., 2024; Moon et al., 2024). Such comparisons would help position the work's practical contributions.

### Questions
- How robust is the three-regime structure across different GSM8K problems? Even a small-scale analysis (e.g., 10 problems stratified by difficulty) would be valuable.

- For practical deployment of SRS/SMC, what would you recommend for estimating s_ver? Could one use a small validation set, or are there theoretical estimation guarantees available?

- Does the Polish space formulation matter in practice? LLM outputs are discrete token sequences. Have you verified that discretization doesn't significantly affect the optimal transport characterization?

- How would the analysis change under alternative coverage constraints (KL-divergence, Rényi divergence)? Would the three regimes persist?

- Theorem 3.7 shows BoN violates coverage for β < s_r*(1-s_r*). Should practitioners avoid BoN in such regimes, or might constraint violation be acceptable if it improves empirical performance?

- How does performance vary across different FPR operating points in the reward-guided setting? Is there a principled way to select FPR?

- SMC and SRS achieve identical sub-optimality and complexity (Theorem 3.5). What's the practical advantage of the more complex SMC derivation?

- Could you clarify how your coverage parameter β relates to the sample budget N in prior BoN scaling analyses (Brown et al., 2024)?

- How might the framework extend to continuous reward models rather than binary verifiers? This seems relevant for many practical applications.

- Figure 5 shows SMC outperforming SRS when the assumed s exceeds the true value. Is there theoretical intuition for this crossover?

### Soundness
3

### Presentation
3

### Contribution
3
