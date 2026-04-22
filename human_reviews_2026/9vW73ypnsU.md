# Multi-marginal temporal Schrödinger Bridge Matching for video generation from unpaired data

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 8, 4, 0

## Abstract
Many natural dynamic processes -such as in vivo cellular differentiation or disease progression- can only be observed through the lens of static sample snapshots. While challenging, reconstructing their temporal evolution to decipher underlying dynamic properties is of major interest to scientific research. Existing approaches enable data transport along a temporal axis but are poorly scalable in high dimension and require restrictive assumptions to be met. To address these issues, we propose ***Multi-Marginal temporal Schrödinger Bridge Matching*** (**MMtSBM**) *for video generation from unpaired data*, extending the theoretical guarantees and empirical efficiency of Diffusion Schrödinger Bridge Matching (Shi et al., 2023) by deriving the Iterative Markovian Fitting algorithm to multiple marginals in a novel factorized fashion. Experiments show that MMtSBM retains theoretical properties on toy examples, achieves state-of-the-art performance on real world datasets such as transcriptomic trajectory inference in 100 dimensions, and for the first time recovers couplings and dynamics in very high dimensional image settings, effectively generating temporally coherent videos from purely unpaired data. Our work establishes multi-marginal Schrödinger bridges as a practical and theoretically principled approach for recovering hidden dynamics from static data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors proposed a multi-marginal extension of diffusion Schrodinger bridge matching to learn trajectories with multiple unpaired snapshots and applied the method in video generation from unpaired images.

### Strengths
It is an important problem how one could learn dynamics with multiple unpaired marginal observations and the authors proposed a method that is sensible.

### Weaknesses
**Theory**
- Although sound, the authors should check whether the results are already known in e.g., Lavenant et al. 2024. 

**Experimental details**
- The write up feels a bit rushed and it is hard to follow how these experiments are done.

**Experimental results**
- In my experience all these methods can have some variance depends on initialization, the results reported in table 2 does not have error margins. E.g., from 0.03 to 0.02 might not be meaningful if the error bar is 0.01. 
- Vanilla piece-wise SB should be compared.
- Video example in 4.5.2 is purely qualitative and has no baseline. I was not able to locate the data on the github repo linked in the paper. 

**Notations**: 
- The $f_i$'s in Prop 3.4 seems not defined in main text, they are exponential of Lagrangian multipliers $\varphi_i$ according to appendix. I would suggest define them in main text or move the proposition in appendix. 

**References**
- I am a bit surprised Lavenant et al. 2024 (date back to 2021 on arxiv) is not referenced. Uniqueness results etc. were also discussed in this theory work quite extensively. 
- Chen et al. 2019 (and subsequent papers) seems missing.
- Line 335-337, the authors internal notes [CITE] still remain. 

Lavenant, H., Zhang, S., Kim, Y.H. and Schiebinger, G., 2024. Toward a mathematical theory of trajectory inference. The Annals of Applied Probability, 34(1A), pp.428-500. [arxiv:2102.09204](https://arxiv.org/abs/2102.09204)

Chen, Yongxin, Giovanni Conforti, Tryphon T. Georgiou, and Luigia Ripani. "Multi-marginal Schrödinger bridges." In International Conference on Geometric Science of Information, pp. 725-732. Cham: Springer International Publishing, 2019.

### Questions
- How does the method compare to Chen et al. 2019 and piecewise SB? 

Chen, Yongxin, Giovanni Conforti, Tryphon T. Georgiou, and Luigia Ripani. "Multi-marginal Schrödinger bridges." In International Conference on Geometric Science of Information, pp. 725-732. Cham: Springer International Publishing, 2019.

### Soundness
3

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
Manuscript addresses reconstruction of temporal dynamics from unpaired snapshots via a time-ordered multi-marginal Schrödinger bridge (MMSB). Authors propose Iterative Markovian Factorized Fitting (IMFF): alternating projections between a Markovian class and a factorized reciprocal class that mixes independent local bridges; jointly learns forward/backward dynamics.
The authors showed static–dynamic equivalence, existence/Markovianity; Theorem 3.2 positions the IMFF fixed point $\mathbb{P}^\star$ between MMSB and a pairwise SB concatenation. And illustrate state-of-the-art on 100D Embryoid Body; scalable image results up to 7 marginals with coherent unpaired video generation.

### Strengths
* **S1 — Principled, scalable framework:** The factorized reciprocal class + alternating projections is a clean extension of IMF to multi-marginal, enabling tractable training at scale.

* **S2 — Strong empirical reach:** SOTA on 100D biology; compelling unpaired video sequences in high dimension.

* **S3 — Sound theoretical scaffolding:** Clear static–dynamic link and KL sandwich bound; practical objectives derived from bridge conditionals; probability-flow ODE sampling avoids score estimation.

* **S4 — Solid engineering:** Fully vectorized training across intervals with masked time discretization makes the method usable in practice.

### Weaknesses
* **W1 — Theory gaps called assumptions:** Key steps rely on **Conjecture 3.1** and on a **variational characterization in the factorized setting** (labeled as Prop. 3.6 but effectively an assumption). Theorems that depend on these should be clearly separated.

* **W2 — Efficiency unquantified:** Inner-loop SDE simulations in IMFF (both directions) are a likely bottleneck; no big-O in $d$, no step-sensitivity, and no wall-clock comparisons vs baselines (e.g., DMSB).

* **W3 — Approximation gap unmeasured:** The bound places $\mathrm{KL}(\mathbb{P}^{\text{MMSB}} \| \mathbb{Q}) \le \mathrm{KL}(\mathbb{P}^\star \| \mathbb{Q}) \le \mathrm{KL}(\mathbb{P}^{\text{pair}} \| \mathbb{Q})$, but there is no empirical KL/energy gap to MMSB or to the pairwise construction.

* **W4 - Missing literature cited:** A few relevant papers in Mult-Marginal Flow Matching published in 2025 are not cited, I suggest a more comprehensive review of the recent literature.

* **Minor:** Notation consistency ($N$ vs $K$); briefly tighten the proof text around Theorem 3.2 to avoid implying equality with MMSB without the stated assumptions.

### Questions
1. **Computational efficiency (W2):**

   * How does the inner-loop SDE simulation cost scale with data dimension $d$ and number of marginals $K$?
   * What are typical integration step counts per interval, and how sensitive is stability/performance to this choice?
   * Please report **wall-clock** training times and GPU memory vs **DMSB** and other baselines in Table 2.

2. **Theoretical assumptions (W1):**

   * What are the main obstacles to proving Conjecture 3.1 and the factorized variational characterization?
   * Can you identify regimes (e.g., degenerate noise, heavy-tailed marginals, non-Gaussian reference) where these assumptions are expected to fail?
   * Given the reliance, would you relabel **Prop. 3.6** as an **Assumption** and mark all downstream results that depend on it?

3. **Approximation gap (W3):**

   * On Gaussian tests (2D and 50D), can you report **KL** or **energy** values for $\mathbb{P}^{\text{pair}}$, $\mathbb{P}^\star$, and the theoretical MMSB target (when computable or via tight bounds)?
   * Any ablation showing how gap metrics vary with number of IMFF iterations and SDE step counts?

4. **Implementation stability:**

   * Why does sequential interval training cause "path forgetting"? Is it due to optimizer drift, data imbalance across intervals, or non-stationary targets?
   * What exact aspect of the fully vectorized + masked scheme (and time encodings) stabilizes learning—e.g., synchronized gradient signals, regularization across intervals, or reduced variance?
   * Could you share an ablation (sequential vs vectorized) on a medium-size setup to quantify the stability/performance delta?RetryTo run code, enable code execution and file creation in Settings > Capabilities.Incognito chats aren’t saved to history or used to train models.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Multi-Marginal Temporal Schrödinger Bridge Matching (MMtSBM), an algorithm for trajectory inference under multi(-time-point) marginal constraints. Unlike conventional Schrödinger Bridge (SB) formulations that only match the distributions at the initial and terminal times, MMtSBM extends the framework to accommodate observations or constraints at multiple intermediate time steps. The authors establish that both reciprocal and Markovian properties remain valid in this multi-marginal setting, ensuring theoretical consistency. Building on this foundation, they develop an Iterative Markovian Fitting (IMF) algorithm to efficiently solve the resulting optimization problem. The proposed approach is evaluated on synthetic datasets and real-world single-cell RNA sequencing data.

### Strengths
- The paper is clearly written and easy to follow, presenting the motivation and mathematical formulation in a coherent manner.

- The authors provide a theoretically sound analysis of how multi-time-point marginal constraints can be incorporated into the SB framework while preserving the gluing dynamics and Markovian structure.

- The paper includes a range of experiments, covering both synthetic toy problems and biologically relevant datasets, demonstrating potential applicability across domains.

### Weaknesses
- The title and terminology could be misleading. The phrase “video generation” may create confusion, as the experiments mainly involve synthetic datasets (e.g., MNIST digit morphing or Biotin simulations) rather than natural video generation tasks. Clarifying this distinction (or changing title) would improve presentation and avoid misinterpretation.

- The theoretical novelty appears limited. The proposed extension only requires to show sort of "continuity" across time points with marginal constraints, which alone may not be a new theoretical formulation beyond the standard SB framework. From an algorithmic perspective, the proposed method resembles running multiple instances of DSBM in parallel rather than introducing a new optimization scheme.

- The experimental evaluation relies heavily on synthetic data and lacks strong baselines. For the single-cell RNA sequencing experiments, several relevant benchmarks are missing (e.g., [1, 2, 3, 6]). In addition, comparisons with concurrent works such as [4] and [5], both of which also address multi-marginal SB matching, would provide a clearer understanding of the relative advantages of the proposed approach. In particular, it would be interesting to compare whether the momentum-based formulations in [4] outperform or complement the IMF-based gluing dynamics proposed in this paper.


References

[1] A Computational Framework for Solving Wasserstein Lagrangian Flows.

[2] Multimodal Single-Cell Data Integration Challenge: Results and Lessons Learned.

[3] Simulation-Free Schrödinger Bridges via Score and Flow Matching.

[4] Momentum Multi-Marginal Schrödinger Bridge Matching.

[5] Multi-Marginal Schrödinger Bridge Matching.

[6] Multi-Marginal Schrödinger Bridges with Iterative Reference Refinement.

### Questions
- For the Biotin experiments, is there a quantitative baseline or evaluation metric used for comparison? A detailed explanation of the evaluation methodology would strengthen the experimental section.

- In Section 4.1, there are place holder "[CITE]".

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper extends the Schrödinger bridge problem, which connects two probability measures, to a multi-marginal setting, where one seeks a stochastic process that additionally passes through a predefined set of intermediate measures.

### Strengths
Paper shows superior performance of the proposed method on Trajectory Net benchmark.

The authors provided generated videos produced by their method in the supplementary material.

### Weaknesses
Theoretical proofs contain incorrect implications, for example in the proof of Proof of Proposition 3.1. (line 839), the feasible set A is not closed under the weak topology because absolute continuity is not in general preserved under weak convergence (one can consider normal distributions with decreasing standard deviations which converge weakly to delta measure, which is not absolutely continuous w. r. t. Lebesgue measure). This statement in the proof requires additional assumptions which are not listed in the theorem statement. It is recommended to formulate the statement of the theorem rigorously in the supplementary material, while in the main text a shortened version could be introduced.

The paper does not clearly show the advantage of multimarginal formulation in comparison to a sequential application of Schrödinger bridge. The experiments with gaussians and cell movements don't show specific scenarios when the proposed method is superior to sequential application of Schrödinger bridges. For example, does the proposed methods is better that sequential application of Schrödinger bridges if all marginals lie on the same Wasserstein geodesic? 

Despite video generation being claimed in the abstract as one of the main contributions of the paper, it is evaluated only on a single video dataset. Video-specific datasets are not considered. Moreover, the provided videos appear too similar to simple image blending, making it difficult to assess whether the method can learn coherent motion from unpaired frame data. The paper lacks an experiment on a conventional real video dataset, from which an unpaired frame dataset could be produced, allowing the quality of generated videos to be evaluated using the Fréchet Video Distance or some optical flow-based metric for motion quality estimation. 

The text of the paper is poorly written and difficult to follow. It contains too many generic phases, particularly in the abstract. (for example phrases like “in an efficient and principled way” in line 064). The paper contains missed citations “it is a pure translation of each Gaussian component inside the mixtures [CITE]. After only the warm-up phase (akin to flow matching [CITE], as said before),” in line 335.

​​Experiment 4.3 “Since no closed-form solution is available for the static multi-marginal SB”. There is a closed-form solution for entropic optimal transport between gaussians of any dimension for pairwise (sequential) optimal transport [1], but it wasn’t used for evaluation, as far as the reviewer understood from the paper.

The multimarginal optimal transport is a complex problem and optimal couplings may have a complex or even fractal form [2]. However it cannot typically be reduced to the pairwise costs like in line 136. This raises the question on how the proposed method behaves for different transport costs. Does it work for costs like c(x,y,z) = x y z or is it limited to cost of the form c(x,y,z) = (x-y)^{2}+(y-z)^{2} (i.e. when there is a sequential separation of costs like c(x,y,z,t,v) = f(x,y)+f(y,z)+f(z,t)+f(t,v))

Methods like Action Matching [3] are also capable of reconstructing dynamics from a sequence of marginal distributions. However such methods were not considered in evaluation.

Generated videos provided in the supplementary page look like a blending between images and do not recover the cell movements as one would expect. More importantly, there are no comparisons in the supplementary with other methods for the video generation task. This questions the contribution stated in the abstract "for the first time recovers couplings and dynamics in very high dimensional image settings".

[1] Hicham Janati, et. al., Entropic Optimal Transport between Unbalanced Gaussian Measures has a Closed Form.

[2] Gladkov et. al., On multistochastic Monge–Kantorovich problem, bitwise operations, and fractals

[3]  Neklyudov et. al., Action Matching: Learning Stochastic Dynamics from Samples

### Questions
How does the proposed method differ from the sequential application of the Schrödinger bridge?

In line 097: "In the limit ε → 0, this recovers classical OT, which motivates our interpolation framework." How the convergence of the entropy regularised optimal transport to the classical OT problem motivates the proposed method?

### Soundness
2

### Presentation
1

### Contribution
1
