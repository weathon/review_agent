# TD-MoE: Tensor Decomposition for MoE Models

- Decision: Accept (Poster)
- Scores: 10, 4, 6

## Abstract
Mixture-of-Experts (MoE) architectures have demonstrated remarkable capabilities and scalability for large language models, but incur a substantial memory footprint due to redundant expert parameters. Existing compression approaches, particularly those based on low-rank decomposition, typically operate at the granularity of individual experts. However, such per-expert methods neglect structural redundancies shared across experts, limiting their compression efficiency and effectiveness. In this work, we introduce TD-MoE (Tensor Decomposition for MoE Compression), a data-aware method that jointly factorizes expert weights by capturing global dependencies. Our contributions are threefold: (i) Cross-expert tensorization with joint three-dimensional decomposition, which unifies all experts within a layer into a single tensor and captures shared structure beyond per-expert scope; (ii) A multi-linear whitening strategy, which decorrelates input and output features, yielding a more balanced and data-adaptive decomposition; (iii) A three-dimensional rank allocation mechanism, which dynamically assigns 3D decomposition ranks across dimensions to best meet a target compression ratio while minimizing the reconstruction error. Extensive experiments on Qwen2-57B-A14B and Mixtral-8×7B across seven commonsense reasoning benchmarks demonstrate that TD-MoE achieves almost lossless performance under 20\% parameter reduction, and delivers more than 11\% and 14\% gains over state-of-the-art decomposition-based baselines at 40\% and 60\% compression. Further ablation studies validate the effectiveness of each component, highlighting the importance of joint factorization, whitening, and rank allocation. Codes are available at https://github.com/ust-xu/TD-MoE.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
TD-MoE is a post-training compression method for Mixture-of-Experts LLMs that stacks all experts in a layer into a 3D tensor, applies Tucker decomposition with multi-linear whitening, then folds the factors back so inference code stays unchanged; a 3D rank-allocation scheme hits a chosen compression budget. On Mixtral-8×7B its near-lossless at 20% and outperforms SVD/pruning baselines by >14% at 40–60% compression, without fine-tuning.

Overall, the paper is methodologically sound: it states a precise optimisation target (preserve activations on a calibration distribution), gives a coherent joint Tucker factorisation with multi-linear whitening and re-colouring that preserves inference form, and supplies a budgeted 3-D rank allocation with sensible interpretations of each factor (meta-experts, input/output subspaces). The setup is transparent and ablations on whitening and Tucker back-ends support the design choices, while headline results on Mixtral-8×7B show near-lossless 20% compression and stronger robustness than baselines at 40–60%. Residual gaps are mostly reporting rather than flaws (e.g., limited end-to-end VRAM/latency data and reliance on a small calibration set; quantisation is left as orthogonal future work). , I’d judge the technical claims to be well supported by the evidence provided.

### Strengths
Very clear paper with impressive results. 

convincingly argues that per-expert methods miss cross-expert redundancy, motivating a joint decomposition view.

whitening materially improves robustness under aggressive compression; output/both whitening favoured in ablations. 

Strong empirical results especially near-lossless at 20% and graceful degradation at 40–60%
Training-free & reproducible:
Backend practicality with full SVD best; randSVD close with lower compute—useful options for large models. 

Inference neutrality  re-colouring folds factors back so runtime code and routing semantics remain unchanged. 

Breadth beyond one model being evaluated

### Weaknesses
No system-level efficiency reporting

Model/benchmark coverage skews to Mixtral-8×7B. While Phi-3.5-MoE is mentioned, detailed tables/figures and ablations are largely on Mixtral; generality across MoE families remains under-evidenced.

No composition with quantization/pruning. The paper positions these as “orthogonal” and omits comparisons or combined recipes, leaving the cumulative benefit and possible conflicts untested.

ompute cost of backends not quantified - paper notes full SVD vs. randomized SVD trade-offs but doesn’t provide wall-clock or FLOPs to justify backend choices at scale.

### Questions
What are the latency/throughput/VRAM changes (A100/H100) vs SVD-LLM/MoE-SVD/NAEE at equal budgets, and what is the one-off cost for whitening stats (activations + gradients)? 

How sensitive are results to the calibration set (size/domain) and to the whitening ε and eigenvalue clipping threshold? 

Could you release per-layer and Fisher weights, and discuss any heuristics used when PRB/FGAA disagree with BCRS? 

Do you observe any change in expert utilisation entropy or gate logits after compression; any layers particularly brittle? 

Have you tried combining TD-MoE with quantisation or distillation to close residual gaps at ≥60% compression?

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
4

### Summary
The paper proposes **TD-MoE**, a unified tensor-decomposition framework for compressing Mixture-of-Experts (MoE) layers. All experts in a layer are stacked into a 3-D tensor (experts × input × output) and jointly factorized with **Tucker decomposition**. To make the factorization data-adaptive and numerically stable, the method applies multi-linear whitening along input/output modes (and re-colors factors so inference has no extra cost). Finally, an adaptive 3-D rank allocation scheme selects ranks under a global compression budget. On Mixtral-8×7B, TD-MoE is nearly lossless at 20% compression and shows sizable gains over decomposition/pruning baselines at 40–60%.

### Strengths
**1. Originality.** The paper explicitly models cross-expert redundancy by stacking experts and performing a **joint Tucker** factorization, instead of per-expert SVD/pruning. The multi-linear whitening step is a principled way to balance spectra across modes. 

**2. Quality.** Empirically strong on Mixtral-8×7B across 7 reasoning benchmarks and 3 LM datasets; at 20% compression accuracy is near the original, and at 40–60% TD-MoE outperforms MoE-SVD/NAEE/MoE-I² with clear trends. 

**3. Clarity.** The pipeline (tensorization → whitening → Tucker → re-coloring shown in Fig.2) and the visualization of the compression landscape are easy to follow. 

**4. Significance.** Works with existing MoE architectures without changing routing; potentially complementary to pruning/quantization.

### Weaknesses
**1. Line-274 / §3.4 rank notation confusion.** §3.4 switches to ($(r_0,r_1,r_2)$), fixes $(r_0=K)$, and only searches ($(r_1,r_2)$). Earlier (§3.2) ($(r_1,r_2,r_3)$) is used and $(U_1)$ (expert mode) is said to compress **inter-expert redundancy**. Please reconcile the symbols and state clearly whether the expert mode is ever truncated.  

**2. Limited model coverage.** Experiments are centered on Mixtral-8×7B; although §4.1 mentions Phi-3.5-MoE, the main tables/figures shown are for Mixtral. Missing results on other many-expert MoEs (**e.g., DeepSeek-MoE, Qwen3-MoE**) weaken external validity.  

**3. Rank-allocation evaluation is thin.** The three strategies (BCRS/PRB/FGAA) are introduced with formulas and visuals, but **lack per-layer / per-task quantitative comparisons under the same budget (accuracy/PPL/error bars).** Fig. 5 hints at smoothing benefits, but more granular numbers would help. 

**4. Inference-time analysis is missing.** The paper explains re-coloring so whitening adds no runtime cost, but it does not analyze MoE forward execution (how $(G,U^{(1)},U^{(2)},U^{(3)})$ are used online) nor latency/throughput with kernel fusion opportunities vs. the original MoE. Reporting only parameter counts makes real-world gains hard to assess. 

**5. Combination with quantization/pruning left untested.** The text calls these “orthogonal,” but no TD-MoE + 8/4-bit (e.g., GPTQ/AWQ) or pruning combinations are reported to demonstrate additive benefits. 

**6. Minor writing/consistency issues.** A few terminology/typo inconsistencies (e.g., “MoV-SVD” in Table 1, inconsistent “inter-experts/inter-expert”) should be cleaned up before camera-ready.

### Questions
See weakness.

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
The paper presents a principled approach to MoE compression by reformulating it as joint tensor decomposition, effectively capturing cross-expert redundancy. The core innovation of stacking experts into a 3D tensor and performing Tucker decomposition captures shared structure beyond the per-expert scope (Sec. 3.1; Fig. 2; p.4). Strong experimental results show TD-MoE achieves 0.57 accuracy vs. 0.50 (MoE-SVD) at 40% compression on Mixtral-8×7B commonsense tasks and maintains low perplexity (4.49/17.20/7.73) at 20% compression (Tables 1-2; Sec. 4.2; p.8). However, the computational overhead and storage requirements for the covariance matrices (Σin, Σout) used in multi-linear whitening are not fully quantified (Sec. 3.3; p.5). Implementation details regarding hyperparameter selection also need clarification.

### Strengths
- **Novel cross-expert tensorization with strong theoretical foundation**
  - The method unifies K experts into a tensor $\mathcal{T} \in \mathbb{R}^{K \times d_{\text {in }} \times d_{\text {out }}}$ and applies Tucker decomposition $\mathcal{T} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)}$, where the factors have clear interpretations: U1 defines meta-experts, and U2/U3 span low-dimensional input/output subspaces (Eq. 3; Sec. 3.2; p.5), providing a principled factorization.
  - Cross-expert factorization addresses the limitations of per-expert methods that overlook structural redundancies across experts trained on related distributions (Sec. 2.1; p.3), improving compression efficiency.
  - Tucker decomposition naturally compresses the expert dimension via the U1 factor without requiring explicit expert dropping/merging decisions (Sec. 3.2; p.5), offering graceful, data-driven redundancy reduction.
- **Effective multi-linear whitening strategy**
  - Multi-linear whitening decorrelates input/output modes via $S_{\text{in}} = (\Sigma_{\text{in}} + \epsilon I)^{-1/2}$ and $S_{\text{out}} = (\Sigma_{\text{out}} + \epsilon I)^{-1/2}$, producing a well-conditioned tensor $\mathcal{T}\_w = \mathcal{T} \times\_2 S\_{\text{in}} \times\_3 S\_{\text{out}}$ (Eq. 4; Sec. 3.3; p.6), improving decomposition stability.
  - The method adapts to the statistical geometry of the data by collecting both input activations X and output gradients ∇YL, capturing correlations and sensitivities (Sec. 3.3; p.5), yielding a data-adaptive factorization.
  - Re-coloring for inference absorbs the inverse whitening into the factors $U^{\prime(1)}=U_1, U^{\prime(2)}=S_{i n}^{-1} U_2$ and $U^{\prime(3)}=S_{\text {out }}^{-1} U_3$, ensuring no runtime overhead (Sec. 3.3; Fig. 4b; p.9), maintaining efficiency.
- **Comprehensive experimental validation with strong baselines**
  - At 20% compression on Mixtral-8×7B, output-whitening achieves 0.62 average accuracy vs. 0.63 for the uncompressed model, a <1% loss across seven benchmarks (Table 1; Sec. 4.2; p.8), demonstrating near-lossless compression.
  - Language modeling results show 4.49/17.20/7.73 perplexity (WikiText-2/PTB/C4) at 20% compression, significantly outperforming NAEE and MoE-SVD (Table 2; Sec. 4.2; p.8), validating effectiveness.
  - Ablations systematically examine whitening directions (input vs. output vs. both) and Tucker backends (QR vs. rand_svd vs. SVD), confirming design choices (Fig. 4; Sec. 4.3; p.9), supporting technical soundness.

### Weaknesses
- **Limited analysis of computational and memory overhead**
  - Whitening requires computing and storing $\Sigma_{\text{in}} = \frac{1}{N} X^T X, \quad \Sigma_{\text{out}} = \frac{1}{N} (\nabla_Y \mathcal{L})^T (\nabla_Y \mathcal{L})$ per layer (Sec. 3.3; p.5). For large models ($d_in, d_out~10^4$), this incurs $O(d^2)$ memory and $O(Nd^2)$ computation, but overhead quantification is absent.
  - Cholesky decomposition for whitening matrices has $O(d^3)$ complexity (Sec. 3.3; Algorithm 1; p.13). Scalability to very large dimensions is not discussed. (Note: Algorithm 1 is in Appendix A.1).
  - Tucker decomposition computational complexity is not analyzed relative to per-expert SVD baselines (Sec. 3.2; p.5). Practical runtime comparisons beyond offline compression time are missing.
- **Hyperparameter selection and rank allocation details**
  - The BCRS method fixes $r_0=K$ and searches ($r_1,r_2$) to satisfy the budget constraint (Eq. 5; Sec. 3.4; p.6), but the rationale for fixing the expert rank is not explained—allowing r0<K could improve compression.
  - Proportional Rank Balancing and Fisher-Guided Adaptive Allocation are mentioned, but details are relegated to the appendix (Sec. 3.4; Appendix A.3; p.6,13). The main text lacks sufficient explanation of rank selection strategies.
  - The covariance eigenvalue clipping threshold ($10^{-3}$) is specified but not justified (Sec. 4.1; p.7). Sensitivity to this hyperparameter is unexplored.
- **Mathematical notation and presentation issues**
  - Tucker decomposition notation $\mathcal{T} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)}$ uses the mode-n product (×n) without defining it in the main text (Eq. 3; Sec. 3.2; p.5). Readers unfamiliar with tensor algebra may struggle.
  - The compression ratio formula $P_{orig}= Kd_{out}d_{in}$ and $P_{tucker}=r_0r_1r_2+Kr_0+d_{out}r_1+d_{in}r_2$ appears without derivation (Sec. 3.4; Appendix A.3; p.13). A step-by-step derivation would improve clarity.
  - Figure 2 shows the overall pipeline but lacks a detailed illustration of how the whitened tensor Tw differs from the original T (Fig. 2; p.4). Additional visualizations would aid understanding.
- **Limited scope of experimental evaluation**
  - Experiments focus on Mixtral-8×7B and Phi-3.5-MoE; evaluation on larger models (e.g., models with >100B parameters or >16 experts) would strengthen generalizability claims (Sec. 4.1; p.7).
  - Only 512 calibration samples were used for whitening statistics (Sec. 4.1; p.7). A sensitivity analysis varying the calibration set size (e.g., 128-2048 samples) is absent.
  - Language modeling was evaluated only on WikiText-2, PTB, C4 (Table 2; p.8). Additional domains (code, multilingual) would demonstrate broader applicability.

### Questions
- **Quantify computational and memory overhead**
  - Provide a detailed complexity analysis comparing TD-MoE vs. per-expert SVD: decompose the total cost into calibration ($O(Nd^2)$), whitening (O(d^3)), and Tucker decomposition, and provide wall-clock timing for each stage across model scales.
  - Report memory footprints explicitly: covariance matrix storage ($2d^2$), whitening matrix storage ($2d^2$), Tucker factors storage ($r_0r_1r_2+Kr_0+d_{out}r_1+d_{in}r_2$), and peak memory during decomposition. Include scalability plots showing overhead vs. ($d_{in}, d_{out}$).
  - Discuss optimization strategies for very large dimensions, such as incremental covariance updates, randomized whitening approximations, or block-diagonal approximations with empirical validation.
- **Clarify rank allocation methodology**
  - Justify fixing $r_0=K$ in BCRS: either provide a theoretical rationale or conduct ablations varying $r_0 \in \left[\frac{K}{4}\, K\right]$ to show the impact on compression-accuracy tradeoffs. Report results in the main text with supplementary details in the appendix.
  - Move the descriptions of Proportional Rank Balancing and Fisher-Guided Adaptive Allocation from the appendix to the main text with concrete algorithms and complexity analysis (currently Appendix A.3; integrate into Sec. 3.4).
  - Perform a sensitivity analysis on the eigenvalue clipping threshold: vary $\varepsilon \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$
 and report perplexity/accuracy impacts with recommendations for different model scales.
- **Improve mathematical presentation**
  - Define the mode-n product (×n) explicitly in the main text with a small worked example (e.g., a 2×2×2 tensor) before introducing the full Tucker decomposition. Include a visual diagram showing how factors combine to reconstruct the tensor.
  - Provide a step-by-step derivation of the parameter counts: start from T∈R^{K×dout×din}, apply the Tucker definition, count the free parameters in G and each factor, and show how the compression ratio formula arises (currently implicit in Sec. 3.4; p.6).
  - Add a whitening visualization: show heatmaps of Σin before/after whitening, singular value distributions of T vs. Tw, and reconstruction error distributions to illustrate the benefits of whitening (extending Fig. 2; p.4).
- **Expand experimental scope**
  - Evaluate on larger-scale models: include experiments on models with 100B+ parameters or 32+ experts per layer to demonstrate scalability. Report compression ratios, perplexity, and downstream accuracy.
  - Conduct a calibration set size sensitivity analysis: vary N∈{128, 256, 512, 1024, 2048} and measure whitening matrix stability (e.g., condition number convergence) and the resulting model performance.
  - Broaden task diversity: add code generation (HumanEval), multilingual tasks (XNLI/XQuAD), and domain-specific benchmarks (scientific QA) to demonstrate cross-domain effectiveness beyond commonsense reasoning and English language modeling.

### Soundness
4

### Presentation
3

### Contribution
3
