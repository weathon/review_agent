# Extending $\mu$P: Spectral Conditions for Feature Learning Across Optimizers

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Several variations of adaptive first-order and second-order optimization methods have been proposed to accelerate and scale the training of large language models. The performance of these optimization routines is highly sensitive to the choice of hyperparameters (HPs), which are computationally expensive to tune for large-scale models.
Maximal update parameterization $(\mu$P$)$ is a set of scaling rules which aims to make the optimal HPs independent of the model size, thereby allowing the HPs tuned on a smaller (computationally cheaper) model to be transferred to train a larger, target model.
Despite promising results for SGD and Adam, deriving $\mu$P for other optimizers is challenging because the underlying tensor programming approach is difficult to grasp. Building on recent work that introduced spectral conditions as an alternative to tensor programs, we propose a novel framework to derive $\mu$P for a broader class of optimizers, including AdamW, ADOPT, LAMB, Sophia, Shampoo and Muon. We implement our $\mu$P derivations on multiple benchmark models and demonstrate zero-shot learning rate transfer across increasing model width for the above optimizers. Further, we provide empirical insights into depth-scaling parameterization for these optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors derive $\mu$P for a wide range of standard optimizers (AdamW, LAMB, Sophia, Shampoo) via spectral scaling and demonstrate zero-shot hyperparameter transfer across scales and benchmarks.

### Strengths
1. The derivations presented in the paper are nontrivial, detailed, and well-written.
2. The proposed method is indeed easier to work with than previous formulations of $\mu$P.
3. The empirical results support the theoretical claims and are promising.

### Weaknesses
1. My main concern with the results is the scalability of proposed method. Given that the paper is primarily empirical in nature, I would find the results a lot more convincing if the experimental setup involved larger (i.e. 10B+) models.
2. The derivations provided rely on simplifying assumptions, such as $\beta_1=\beta_2=\epsilon=0$, which are atypical regimes. I suggest to include either an analysis with momentum and weight decay considered or a discussion of why the simplified regime is sufficient.
3. Several assumptions are hidden away in the appendix, with little discussion provided on their practicality.
4. Muon is mentioned as a state-of-the-art optimizer, but a $\mu$P derivation for Muon is not provided.
5. Minor nitpicks: use `\citep{}` instead of `(\cite{})` for parenthetical citations. The overflow on page 9 is visually unappealing.

### Questions
See Weaknesses

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
4

### Summary
This paper addresses the challenge of hyperparameter (HP) tuning for large-scale models. Maximal update parameterization ($\mu$P) is a promising technique that enables zero-shot HP transfer from small to large models, but its derivation via tensor programs is complex and has limited its application beyond SGD and Adam.

Building on recent work that proposed spectral conditions as a simpler, alternative foundation for $\mu$P, this paper proposes a generic framework that uses this spectral approach. The authors apply this framework to derive, for the first time, $\mu$P scaling rules for a wider range of popular optimizers, including AdamW, ADOPT, LAMB, Sophia, and Shampoo.

The paper validates these new derivations empirically on NanoGPT and Llama2 models, demonstrating that the new $\mu$P scaling rules successfully achieve zero-shot learning rate transfer as model width increases.

### Strengths
1. Addresses a High-Impact Problem: The computational cost of HP tuning is a significant bottleneck in training large models. $\mu$P is a powerful tool to mitigate this, but its applicability has been limited. Extending $\mu$P to a wider, more modern set of optimizers like LAMB, Sophia, and Shampoo is a valuable and practical contribution to the field.
2. Clear Practical Takeaways: The paper delivers actionable scaling rules for several optimizers (summarized in Table 2).
3. Strong Empirical Validation: The claims are well-supported by experiments. The plots in Figure 2, Figure 3, and Figure 4 (right) clearly demonstrate the two key benefits of $\mu$P: stable zero-shot learning rate transfer across widths and the wider is better phenomenon (decreasing training loss with width). The validation on both NanoGPT and Llama2 models strengthens the empirical claims.

### Weaknesses
1. Incremental Novelty: The core conceptual leap, replacing complex tensor programs with more tractable spectral conditions, was introduced by prior work. This paper's main contribution is the application of this existing spectral framework to new optimizers. While this is a useful engineering and analytical contribution, the intellectual novelty of the methodology itself is limited.
2. Repetitive and Padded Structure: The main methodological idea is presented in Section 4.1 as a Generic Framework. The following subsections (4.2-4.6) are largely straightforward, repetitive applications of this single framework to different optimizers. For example, the derivations for AdamW and Sophia are nearly identical. This section feels padded and could have been substantially condensed by focusing on the unique analytical challenges (e.g., for LAMB and Shampoo) and moving the more trivial derivations to the appendix.
3. Typesetting and Readability Issues: The paper appears to be in a draft state with major formatting errors. In Section 4.6 (Page 9), the entire multi-line equation block for the Shampoo optimizer derivation runs off the page margin, which suggests a lack of careful preparation.
4. Limited Theoretical Scope: The core derivations are performed for a simple linear MLP with a batch size of 1. The extension to complex, non-linear models like Llama2 relies on a set of assumptions outlined in Appendix A. The paper does not provide a deep analysis of why these assumptions should hold for these more complex optimizers, relying instead on the empirical results to justify the leap.

### Questions
1. Could the authors clarify their novel methodological contribution beyond the direct application of the spectral conditions framework? Is the generic framework in 4.1 the primary contribution, or is the contribution simply the set of new scaling rules derived from it?
2. Given the significant overlap in the analysis for AdamW, ADOPT, and Sophia, would the authors consider restructuring Section 4 to avoid repetition and better highlight the unique analytical challenges posed by LAMB and Shampoo?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies $\mu$P parameterization for stable feature learning and allows zero-shot learning rate (LR) transfer across model sizes (widths). In particular, the paper proposes a spectral norm-based recipe to derive $\mu$P parameterization for a range of optimizers, including AdamW, ADOPT, LAMB, Sophia, and Shampoo. Spectral conditions on weights and updates are used to replace complex tensor programs for deriving $\mu$P. Numerical experiments on NanoGPT and Llama-2 are provided to validate the derived parameterization.

### Strengths
1. The proposed framework largely simplifies tensor programs, and the results are clearly presented. The derivations are simple and applicable for a range of optimizers. 

2. Beyond recovering the parameterization for AdamW, the paper provides new parameterizations for LAMB and Shampoo.

### Weaknesses
1. Muon optimizer is mentioned in the introduction, but there is no corresponding derivation for it. See Question 2.

2. The derivations rely on strong simplifications, such as batch size=1, $\beta_1=\beta_2=\epsilon=0$, and the dropping of weight decay. It is questionable whether the derived exponents remain valid when these hyperparameters are changed. 

3. The results for Shampoo (Figure 2) do not show a clear zero-shot LR transfer. The losses often worsen with width across the LR grid. See Question 3.

4. Depth scaling remains preliminary.

### Questions
1. In Table 2, the Init. Var. and Multiplier result in the same effect as the tensor programs. Can we adopt the scalings from the tensor program for these two?

2. Muon optimizer is mentioned in the introduction and Figure 1, but there is no derivation for it. Is it considered to be the same as Shampoo?

3. In Figure 2, the validation loss of Shampoo is very high compared to other optimizers, and the best performance does not occur for the largest width. Is it the nature of Shampoo, or does it suggest that the parameterization for Shampoo is suboptimal?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper recasts μP through a spectral-norm lens: enforce layerwise spectral scaling on weights and updates so that a single training step yields width-invariant functional change; then derive optimizer-specific LR rules. Using this recipe, the authors extend μP-style scaling from SGD/Adam to AdamW, ADOPT, LAMB, Sophia, and Shampoo, and demonstrate zero-shot LR transfer across width on NanoGPT and small-budget LLaMA-2; depth results are preliminary.

### Strengths
1. A clear, interpretable derivation that produces width-invariant HPs (esp. LR) for multiple optimizers, broadening μP’s practical coverage with lighter machinery than Tensor Programs.
    
2. Closed-form LR scalings plus explicit forward scaling and (O(1)) treatment for LN/bias—usable as a “cookbook.”
    
3. Multiple widths trained per model family: NanoGPT (128→2048) and LLaMA-2–style (256→2048 ≈154M→1.38B params on WikiText-103). LR–vs–loss sweeps show the same LR tuned on the smallest width remains optimal at larger widths.
    
4. Coordinate checks are stable; LR transfer across depth looks promising for AdamW/Sophia, highlighting where more theory is needed.
    
5. Coordinate-check plots + LR tables make the recipe easy to audit and adopt.

### Weaknesses
1.The analysis repackages μP using the published spectral condition and **retains μP’s assumptions**. Under the same assumptions, Tensor Programs could in principle obtain the same optimizer scalings. A genuine advance would relax assumptions or prove depth-scaling for the added optimizers.
    
2.Core derivations (Result 4.1) use a **linear MLP, batch-1** (rank-1 gradients where spectral≈Frobenius). Transformer attention is not newly analyzed—assumptions are imported and only **validated empirically**.
    
3.Width LR sweeps + coordinate checks substantially repeat the original μP methodology; runs are **small/medium-scale** (short LLaMA-2 training on WikiText-103), illustrating trends rather than establishing robustness at modern pretraining scales.
    
4.Clear oscillations/instabilities for ADOPT/LAMB/Shampoo with depth are reported but not diagnosed (e.g., trust-ratio dynamics, Shampoo preconditioner spectra, finite-width violations).
    
5.Non-monotone behaviors (LAMB/Shampoo) across width/depth lack error-budget analyses (e.g., per-layer update spectral norms vs target).
    
6.Since TP-V already covered Adam, scaling tables, and coordinate checks, the paper should more explicitly delineate **what is new** (optimizer catalog via spectral derivation) vs **what is inherited**.

### Questions
Which μP/spectral assumptions are actually essential? Can you prove width or depth transfer for any added optimizer under weaker/realistic settings (batches >1, attention+LN in the loop)?
    
Can you provide a formal spectral argument for self-attention (Q/K/V/O) under your update-norm rules (e.g., block-matrix bounds), rather than relying solely on experiments?
    
Please instrument per-layer **update spectral norms** and report **LAMB trust ratios / Shampoo spectra** across depth to explain the oscillations.
    
Add probes isolating finite-width corrections, batch-size effects, and residual-path scaling interactions to map where spectral sufficiency is tight vs loose.

### Soundness
2

### Presentation
2

### Contribution
1
