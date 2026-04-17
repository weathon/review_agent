# Bonsai Networks: Structured Pruning and Sparse Training of Foundation Models

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
The recent trend of scaling neural networks to unprecedented sizes demands efficient structured sparsity for practical deployment, yet precise control of sparsity levels and patterns for hardware acceleration remains challenging. This paper introduces the Adaptive Soft-Thresholding Algorithm (ASTRA), which achieves a target sparsity by adapting group-wise regularization strength based on computationally inexpensive sparsity characterizations. We establish ASTRA’s theoretical foundations, proving the existence of stable regularizations that realize the desired sparsity. We demonstrate sublinear and linear convergence rates for both the model parameters and the regularization weight in deterministic settings and, crucially, an almost sure $O(1/t)$ convergence rate in the practical stochastic-gradient setting. ASTRA provides a theoretically grounded method for direct, precise control over structured sparsity, enabling the pruning and fine-tuning of foundation models into Bonsai Networks: accelerator-friendly miniatures trained to match the teacher’s outputs while preserving downstream performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ASTRA, an adaptive soft-thresholding method that tunes a group regularization weight online so a model converges to a desired structured sparsity level. The authors prove the existence of stable regularizations that achieve a target sparsity and establish convergence rates in both deterministic and stochastic settings. ASTRA extends naturally to grouped patterns for accelerator-friendly pruning via a structured sparsity algebra, and the stochastic variant SASTRA enables sparse training without dense gradient computations.

### Strengths
+The target sparsity is cast as a scalar root-finding problem over the regularization weight, with proofs for stable regularizations and O(1/t) tracking in the stochastic setting. 
+The paper analyzes how the weight update tracks the moving optimum while the regularization follows a Robbins–Monro schedule with provable boundedness and rates.
+The framework is a local proximal-control of several modern heuristics, opening possibilities for extensions to structured sparsity.

### Weaknesses
-The empirical evaluation is incomplete and narrowly scoped. The LLM case study measures only one head and a single kernel, without comprehensive end-to-end metrics such as latency, throughput, and energy across kernels and hardware.
-The classification experiments rely on ResNet-32 with CIFAR-10/100, which do not reflect large-scale behavior (e.g. ResNet-50 ImageNet).
-The paper’s scope is narrow: results are shown only on Qwen. Please correct the typos in Qwen citations, expand evaluation to other model families and sizes (Llama, Mixtral, additional Qwen variants), and report both quality metrics on public benchmarks and system metrics (end-to-end latency, throughput, memory) across multiple sparsity targets.
-The paper lacks comparisons with state-of-the-art methods in both LLM pruning and dynamic sparse training. In the LLM setting, it should include matched evaluations against strong pruning baselines such as Wanda, SparseGPT, and OWL-style structured pruning, using the same calibration budget and target sparsity.

### Questions
-Can you provide end-to-end measurements on GPUs for several sparsity targets, including latency, throughput, and energy, beyond the single kernel study.
-How robust is SASTRA to bias in the order-statistic surrogate and EMA hyperparameters, and can you report ablations showing stability and quality as these vary. 
-Can you expand the structured experiments to other block layouts and to additional model families, for example Mixtral and Llama variants, to validate generality.
-What memory overhead is introduced by the per group or per block statistics used by ASTRA during training, and how does it scale to very large models.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes ASTRA, which adaptively tunes regularization during proximal gradient updates so the learned model hits a given level of sparsity. A structured version is also proposed. The paper includes extensive theory supporting the ASTRA method and experimental results comparing against multiple existing sparsification methods.

### Strengths
1. The paper is clear and well-written.
2. Formulating hitting a target sparsity as root-finding seems new and potentially interesting. The approach seems to unify several existing pruning methods.
3. A structured method is also included in a natural way.
4. The results on CIFAR seem competitive and there are multiple baselines.

### Weaknesses
I will focus on the experimental and systems aspect here, as I have less background in the relevant theory. If other reviewers find significant merit in the theory that may outweigh my other concerns.

1. The paper's focus (indeed, from the title!) is on foundation models; yet the experimental results do not bear this out. The experimental results focus on ResNet-32 and a toy test with a single LLM head form Qwen. The paper needs a much more extensive study with larger LLMs in order to demonstrate it works at scale.
2. The experimental results in Table 1 lack error bars or other measures of statistical significance. Further, it is a mixture of results from the literature and new runs, making it hard to know whether training details differed.
3. There are no end-to-end performance results showing throughput or latency. It is thus hard to tell whether the method can achieve significant speedups in practice for inference (although the toy test on Qwen is promising). A comparison against state-of-the-art structured pruning methods for LLMs would also be needed (e.g., vendor 2:4 sparsity and something like MaskLLM); I am not certain that the 4:16 kernel can match other structured sparsity approaches.
4. Similarly, an evaluation of the overheads during training is missing.

### Questions
1. Can you provide full, end-to-end results on Qwen or a similar model (e.g., Llama)?
2. Are the results in Table 1 statistically significant?
3. Can you provide end-to-end inference runtime results? Can you characterize the training time overheads?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work studies pruning and sparse training from an optimization perspective. Specifically, the central problem is minimizing the task loss under a sparsity constraint induced by L1 regularization. Because this regularizer is convex, under (strong) convexity of the loss the problem can be solved by proximal SGD methods for any regularization weight $\lambda$. The central question of the paper is how to set $\lambda$ to enforce a desired sparsity level ($\lambda=0$ yields no sparsity, while $\lambda = ||\nabla f(0)||_{\infty}$ gives the trivial zero solution). The paper formalizes finding the optimal regularization $\lambda$ as a scalar root-finding problem and provides (adaptive) algorithms for approximating it and solving the L1 regularized objective. Extensions to group-wise sparsity and connections to prior methods are discussed, and some experimental results are provided.

### Strengths
- The problem of sparse training is important and highly practical. The paper offers a principled approach for solving its surrogate L1 regularized formulation given sparsity budget.
- The project aims to devise practically efficient algorithms, and throughout the paper exact conditions are approximated to make the methods implementable.
- The writing is engaging and generally guides the reader through the main results, explaining and motivating the key transitions and approximations.

### Weaknesses
Overall, technical writing is poor and correctness of claims is questionable.
1. The equivalence stated in equation (2) regarding the optimality condition of (1) is not correct as written. The condition $||\nabla f(w)||_{\infty} \le \lambda$ is necessary but not sufficient (it represents an entire region while there is only one minimizer). This equivalence is used in the proof of Lemma 2, so the proof of Lemma 2 is not accurate as presented. However, the claim of Lemma 1 appears to be true and can be shown using the arguments used to prove Lemma 2, but the current exposition is misleading and requires correction.
2. The equivalence in (6) is mentioned without proof. To me it appears to be of similar difficulty to the claim of Theorem 1, so it should be proven or at least justified more carefully in the main text.
3. Several definitions are introduced inside theorem statements (e.g., the definition of “$\phi$-stable regularization” appears in Theorem 1, the set $\Lambda(\psi_{\kappa, \alpha})$ is defined inside Theorem 2). This placement disrupts flow and makes reading awkward. Move definitions to a dedicated notation/definitions section or introduce them just before the theorems that use them.
4. At the end of Section 3.1 the paper mentions improving the bisection idea and avoiding fully computing regularized solutions. However, the bisection method appears faster in theory. The adaptive method proposed later has sublinear convergence (even in the deterministic case), whereas a bisection scheme would require $\mathcal{O}(\log(1/\epsilon))$ proximal-GD solves, each with linear convergence. What theoretical advantage does ASTRA offer over bisection? Appendix C.2 discusses a deterministic linear rate, but that result only yields linear convergence up to a neighborhood, which is not linear convergence in the usual sense.
5. I could not find a proof of the claim $\delta = O(\beta_t)$ in Corollary 1. The provided proof shows $w_t$ is bounded (the first claim), but the second claim $\delta = O(\beta_t)$ does not seem to follow in general. For example, if $\beta_t = 0$, then $\delta_t$ need not be zero by the recursion in Lemma 4.
6. The proof of Theorem 3 is also problematic. Line 1000 requires that all iterates $\lambda_t$ remain in a neighborhood of $\lambda_*$ as per Assumption 2, so the convergence result appears to be local only. If a good initial guess of $\lambda_*$ is not available, it is unclear how this assumption can be satisfied. Make the locality explicit in the theorem statement and discuss how restrictive this assumption is in practice. The proof omitted the projection step $\Pi_{[0,\lambda_{\max}]}$ even though equation (10) of the algorithm includes it. Furthermore, lines 1017–1020 seem to imply $V_t = O(1/t)$ based on another paper, but it is done without properly linking the assumptions or reproducing the relevant connection. Then $V_t = O(1/t)$ is used in line 1025 to derive a bound, which is subsequently (in lines 1026–1029) apparently used to re-establish $V_t = O(1/t)$. This looks circular and the argument across lines 1017–1029 does not make sence and potentially incorrect.

### Questions
Please see "weaknesses".

### Soundness
1

### Presentation
2

### Contribution
1
