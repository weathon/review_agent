=== CALIBRATION EXAMPLE 79 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title “Scaling with Collapse: Efficient and Predictable Training of LLM Families” accurately reflects the paper’s core theme. The abstract succinctly states the problem (extending collapse to practical scaling recipes), main findings (collapse occurs when τ and TPP are optimally set), and applications (diagnostics, early stopping, Celerity family). The claims appear supported by the paper’s body, though I will need to verify the strength of the evidence.

### Introduction & Motivation
The introduction is well-motivated, framing predictability in LLM training as critical for frontier-scale efficiency. It clearly builds on Qiu et al. (2025) and identifies the gap: testing collapse under “practical scaling recipes” where width, depth, LR, batch size, and weight decay are co-scaled. The contributions are listed concretely and align with the rest of the paper.

### Background (Section 2)
This section effectively reviews key concepts (TPP, μP, supercollapse, AdamW timescale τ). It sets up the necessary terminology and frames the paper’s perspective within existing scaling literature. The connection between optimal τ and TPP from prior work (Bergsma et al., 2025a) is appropriately highlighted as a foundation.

### What Factors Modulate Training Curve Shape? (Section 3)
This is a core methodological section. The experimental setup is described, but some details are deferred to Appendix B.1 (architecture, dataset, proxy tuning). The central findings are clear:
1. **τ modulates TLC shape**: Sweeps of η, λ, or B that produce the same τ yield matching normalized curve shapes (Fig. 3). This is convincing.
2. **TPP modulates TLC shape**: Higher TPP leads to faster early drop and flattening (Fig. 4), and this effect is scale-invariant.
3. **Collapse condition**: When TPP and τ are (approximately) fixed across model sizes, normalized TLCs collapse.

Theoretical intuition using bias-variance decomposition and a noisy quadratic model (Appendix B.3) adds depth and helps explain the role of τ. However, a significant concern is the **generalizability** of the findings. The experiments are conducted on a specific GPT2-like architecture with ALiBi, SwiGLU, on SlimPajama, using μP and a linear decay-to-zero schedule. The authors provide supplementary experiments in Appendix B.4 (different LR schedules, datasets, architectures, MoE, Adam β), which show robustness, but the main text could more explicitly note the scope of these validation studies. The claim that collapse is a “signature of compute-efficient training” is strong; while supported by the link to optimal τ, it might imply a necessity that hasn't been fully proven (could collapse also occur in slightly suboptimal but stable regimes?).

### Celerity: A Compute-Efficient Model Family with Collapse (Section 4)
This section demonstrates a practical instantiation. The choice of TPP=234 is justified via a compute/compression trade-off analysis (Appendix C.1, Fig. 5), which is insightful. The Celerity family details (architecture, data mix, use of CompleteP) are provided. The results show:
- Competitive compute-efficiency on a Pareto frontier (Fig. 2, Table 10).
- Demonstrated collapse across Celerity bands at fixed TPP (Fig. 6).
- Collapse residuals used as an early diagnostic for a numerics issue (Fig. 1, right).

**Major concerns here:**
1. **Evaluation comparisons**: The paper compares Celerity to other open models (Llama-2, OLMo, SmolLM, etc.) on seven common benchmarks. However, the authors note that many contemporary models use “data annealing” or specialized mid-training data, while Celerity intentionally does not. This is a valid philosophical choice, but it complicates the fairness of a direct “compute-efficiency” comparison (Fig. 2). The claim that Celerity is at the frontier should be tempered with this caveat more prominently in the main text. The appendix comparisons with distilled models (Fig. 16, 18) are a good addition.
2. **Validation of collapse**: The collapse shown in Fig. 6 (left, middle) is convincing for 20 and 80 TPP bands. For the 234 TPP band (Fig. 1, middle), the authors note late divergences attributed to overfitting on training data. This is an important nuance—collapse can break down at very high TPP, which should be discussed as a limitation in the main narrative.
3. **Parameterization impact**: The switch from μP to CompleteP is noted as beneficial for scaling (Fig. 15). This is fine, but it introduces another variable; how much of the observed collapse and efficiency is attributable to the specific parameterization? A brief discussion would help.

### Collapse Enables Early Stopping in Hyperparameter Tuning (Section 5)
This is a promising application. The key insight is that fixing τ during tuning (e.g., batch size sweeps) preserves TLC ordering, enabling early selection (Fig. 7). The proposed procedure—using a small-scale predictor to extrapolate final loss from partial runs—is clever.

**Concerns:**
1. **Predictive model (Eq. 4)**: The functional form and fitting procedure (alternating power-law fits for b(τ) and q(TPP)) are somewhat heuristic. While results in Table 11 and 12 show it works reasonably well (MAE ~1%), the model is fit on 111M-scale data for a specific LR schedule (D2Z). Its generalizability to other schedules, architectures, or data distributions is not established. This is acknowledged in limitations (Appendix A), but the main section presents it with less caution.
2. **Early stopping results**: Figures 9, 23, 24, 25 show that the “predicted best” method often outperforms “current best.” This is encouraging. However, the experiments are limited to sweeps of λ and B. More diverse hyperparameter searches (e.g., architecture dimensions, LR schedule parameters) would strengthen the claim’s generality.
3. **Noise and spikes**: The method is noted to be impaired by loss spikes. Since spikes are common in large-scale training, this is a practical limitation that deserves more emphasis.

### Related Work (Section 6)
Comprehensive and well-positioned. It connects collapse to scaling laws, μP, loss-curve prediction, and HPO literature.

### Limitations and Broader Impact (Appendix A)
The limitations section is thorough, covering optimizers, data curricula, generalization beyond dense models, train vs. validation collapse, and predictive model constraints. This is a strength of the paper. Broader societal impact is not discussed, which is acceptable for a primarily technical contribution.

### Writing & Clarity
The paper is generally well-written, though dense in places. The logical flow is clear. Some figures suffer from parser artifacts (e.g., garbled tables in Fig. 1, 2), but the core information is interpretable. The use of appendices for extensive details is appropriate.

## Overall Assessment
This paper makes a valuable contribution by rigorously extending the training loss curve collapse phenomenon to large-scale LLM training under practical hyperparameter scaling recipes. The identification of τ and TPP as key controls is backed by systematic experiments and theoretical intuition. The introduction of the Celerity family provides a tangible demonstration and the applications for diagnostics and early stopping are promising and practical.

The main weaknesses are: (1) some claims about generality and compute-efficiency frontiers could be more carefully qualified given differences in training data and techniques among compared models; (2) the predictive model for early stopping, while effective in the tested settings, is heuristic and its scope is not fully validated; (3) the experimental foundation, though extensive, is based on a specific model family and dataset.

For ICLR, which values novelty, empirical soundness, and clear impact, this paper is likely above the acceptance bar. The work directly addresses an open question from recent literature (Qiu et al.) and provides actionable insights for LLM practitioners. The concerns are significant but can likely be addressed through revisions that temper claims, expand discussion of limitations, and possibly include additional ablation studies.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates the training loss curve (TLC) collapse phenomenon—where normalized loss curves align across model sizes—in large-scale language model families under practical scaling recipes. The authors identify that collapse occurs when tokens-per-parameter (TPP), AdamW timescale τ, and learning rate schedule are held fixed, with τ set optimally for the given TPP. They demonstrate that collapse serves as a signature of compute-efficient training and enables two practical applications: early detection of training pathologies and early stopping in hyperparameter tuning. The paper also introduces the Celerity model family, trained using these principles, which exhibits collapse and achieves competitive performance on standard benchmarks.

### Strengths
1. **Significant extension of prior work**: The paper convincingly extends the collapse phenomenon (recently shown in small-scale settings by Qiu et al. 2025) to large-scale LLMs under practical scaling conditions, addressing an important open question. The empirical validation across models up to 3.9B parameters, with co-scaling of width, depth, batch size, and weight decay, is thorough and well-executed.

2. **Practical utility and compelling applications**: The paper demonstrates two highly valuable applications: (i) using collapse residuals as an early, sensitive diagnostic for training issues (e.g., detecting a numerical instability well before the raw loss curve shows problems), and (ii) enabling data-efficient hyperparameter tuning via early stopping, saving substantial compute. These are backed by concrete experiments (e.g., Figs. 1, 9).

3. **Clear identification of controlling factors**: The paper clearly establishes TPP and τ as the key factors modulating TLC shape under μP, supported by systematic sweeps (Figs. 3, 4) and a theoretical noisy quadratic model (Appendix B.3) that provides intuition. The link between optimal τ and TPP (from prior work) elegantly explains why collapse emerges under efficient training.

4. **Real-world demonstration with Celerity**: The introduction and training of the Celerity model family provides a tangible proof-of-concept. Celerity achieves competitive compute-efficiency on the Pareto frontier (Fig. 2) and exhibits tight collapse (Figs. 1, 6), validating the proposed principles at scale.

5. **Strong empirical methodology**: The paper is based on an extensive set of experiments (~600 TLCs) across multiple scales, TPP values, and hyperparameters. The use of standardized datasets, public model comparisons, and detailed documentation in appendices enhances reproducibility and credibility.

### Weaknesses
1. **Limited generality of collapse conditions**: While collapse is demonstrated under specific conditions (single-epoch AdamW, fixed TPP/τ/LR schedule), its robustness to broader settings—other optimizers (Sophia, SGD), multi-epoch training, heavy data curricula, or mixture-of-experts architectures beyond a preliminary test—remains underexplored. The limitations section acknowledges this but lacks experimental validation.

2. **Simplistic predictive model for early stopping**: The parametric form for normalized TLCs (Eq. 4) is heuristic and fitted only on 111M models with a linear decay-to-zero schedule. Its performance degrades with noisy curves (e.g., large batch sizes), and its applicability to other schedules (cosine, inverse sqrt) is untested. The model's error, while low, is still twice that of an oracle per-curve fit (Table 12).

3. **Incomplete evaluation of Celerity**: Celerity's compute-efficiency is compared primarily against older or similarly sized open models. Comparisons against state-of-the-art models using advanced techniques (distillation, extensive data annealing, specialized architectures) are limited or indirect. The ablations on data mix and parameterization (CompleteP vs. μP) are noted but not quantified in terms of their individual contributions to final performance.

4. **Theoretical analysis is preliminary**: The noisy quadratic model provides intuition but is restricted to constant LR. The extension to decaying LR schedules is qualitative. A more rigorous theoretical treatment connecting collapse to scaling laws and optimization dynamics under general schedules would strengthen the foundation.

5. **Reproducibility concerns with Celerity details**: While appendices provide extensive details, key aspects for full reproduction are incomplete: the exact data mixing proportions (Table 6 lists percentages but not the actual token counts or sampling methods), the complete hyperparameter values for all Celerity models, and the code/scripts are not provided.

### Novelty & Significance
**Novelty**: The paper makes several novel contributions: (i) identifying that TLC collapse persists in full-scale LLM families under practical co-scaling of hyperparameters, (ii) establishing τ and TPP as the primary controls of TLC shape under μP, (iii) demonstrating collapse as a diagnostic tool and for early stopping in hyperparameter tuning, and (iv) introducing the first LLM family (Celerity) explicitly trained to exhibit collapse. While building on prior scaling law and μP literature, the synthesis and large-scale validation are original.

**Significance**: The work has high practical significance for the LLM training community. The collapse phenomenon provides a principled, scale-invariant reference for monitoring training health and tuning hyperparameters efficiently, which can lead to substantial computational savings. The Celerity models offer a valuable compute-efficient baseline. The paper advances the understanding of training dynamics and could influence how large-scale experiments are designed and monitored.

### Suggestions for Improvement
1. **Expand the empirical scope of collapse**: Conduct experiments to test collapse under a wider range of conditions: different optimizers (e.g., Sophia, Adafactor), multi-epoch training, and more aggressive data curricula. This would better establish the boundaries of the phenomenon.

2. **Strengthen the predictive model**: Improve the parametric form for normalized TLCs, perhaps by incorporating schedule-specific terms or a more flexible functional form. Validate the model on other common LR schedules (cosine, warmup-stable-decay) and provide uncertainty estimates to guide early stopping decisions more robustly.

3. **Enhance Celerity evaluation and ablations**: Compare Celerity more directly with state-of-the-art models (e.g., Gemma-3, Llama-3.2) on a broader set of benchmarks. Perform controlled ablations to quantify the individual contributions of data mix, parameterization (CompleteP), and τ scaling to final performance and collapse tightness.

4. **Deepen theoretical analysis**: Extend the noisy quadratic model to account for decaying LR schedules analytically, or provide a more formal argument linking collapse to scale-invariant curvature under μP. This would strengthen the theoretical foundation.

5. **Improve reproducibility**: Release training code and configuration files for the experiments and Celerity models. Provide exact data processing and mixing scripts, and ensure all hyperparameters are listed in a machine-readable format.

6. **Clarify practical deployment**: Discuss how practitioners can implement a "collapse monitor" in their training loops—what metrics to log, how to set thresholds for residuals, and how to handle common edge cases (e.g., restarts, loss spikes).

**Overall Recommendation**: This is a strong paper with important empirical findings and practical applications. It meets ICLR's standards for novelty, empirical rigor, and potential impact. With revisions addressing the above points—particularly broadening the scope of collapse validation and improving the predictive model—it would be an excellent fit for the conference.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation study isolating the effect of collapse conditions on final performance.** Train models with identical data/architecture under optimal τ (collapsing) vs. mis-scaled τ (non-collapsing) and compare final loss/compute efficiency. Without this, the claim that collapse signifies compute-efficient training is confounded by Celerity’s superior data mix and other optimizations.
2. **Validation of collapse at state-of-the-art scales (e.g., 10B+ parameters).** The paper only demonstrates collapse up to 3.9B. To be compelling for frontier LLM training, evidence must extend to much larger models where predictability matters most.
3. **Systematic test of collapse across diverse, practical LR schedules.** Only constant, 10× decay, and decay-to-zero schedules are shown. Many production LLMs use cosine or inverse square-root schedules; collapse must be verified under these to claim generality.
4. **Quantitative evaluation of collapse-based early detection for multiple pathology types.** The diagnostic claim relies on a single numerics example. Inject controlled perturbations (e.g., data repeats, gradient spikes) and measure detection latency vs. conventional monitoring to establish robustness.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis linking theoretical assumptions to empirical LLM training.** The noisy quadratic model and power-law derivations assume scale-invariant curvature and separability. Validate these by analyzing Hessian spectra or loss landscape geometry across scales under μP to ground the theory.
2. **Precise quantification of how TPP and τ control normalized curve shape.** The paper states qualitative trends (e.g., higher TPP → earlier drop). Fit a functional form (beyond Eq. 4) that predicts shape from TPP and τ alone and test its accuracy across a wide parameter grid.
3. **Failure analysis for early-stopping predictions.** The method degrades with noisy, large-batch runs (Fig. 25). Characterize when prediction error becomes unacceptable and propose mitigations (e.g., better smoothing, uncertainty estimates).
4. **Investigation of collapsed validation loss vs. training loss, especially at high TPP.** At 234 TPP, training loss collapse degrades while held-out data “remains aligned.” Analyze this decoupling: does it indicate overfitting? Does validation loss also collapse? This is critical for trusting collapse as an efficiency marker.

### Visualizations & Case Studies
1. **Side-by-side visual comparison of training with optimal τ (collapsing) vs. suboptimal τ (non-collapsing) for the same model/data.** Show raw and normalized loss curves, plus final performance, to visually reinforce that collapse correlates with better training.
2. **Gallery of collapse residual plots for diverse, common training failures.** Beyond the single numerics example, show residuals detecting data duplication, inappropriate LR, weight decay mis-settings, etc. This would demonstrate the diagnostic’s generality.
3. **Scatter plots of predicted vs. true final loss for all hyperparameter settings in tuning sweeps.** Visualize the accuracy of the extrapolation method across all trials, not just the “best” selection, to assess calibration.

### Obvious Next Steps
1. **Train a model at scale (e.g., 10B+) using collapse-guided tuning and monitoring.** The paper’s largest model is 3.9B. Demonstrating successful application at a scale where experiments are prohibitive would strongly support the method’s practical value.
2. **Integrate the collapse monitor into a production training framework and report operational metrics.** Quantify reductions in wasted compute, earlier fault detection, or tuning cost savings in a realistic setting.
3. **Test collapse on other prominent architectures (e.g., dense vs. MoE) and across major public datasets.** The paper includes a limited MoE experiment; systematic validation across architectures and data sources would strengthen claims of universality.

# Final Consolidated Review
## Summary
This paper extends the training loss curve (TLC) collapse phenomenon—where normalized loss curves align across model sizes—to large-scale LLM training under practical hyperparameter scaling recipes. It identifies that collapse occurs when the tokens-per-parameter ratio (TPP), AdamW timescale (τ), and learning rate schedule are held fixed, with τ set optimally for the given TPP. The paper demonstrates that collapse serves as a diagnostic for training issues and enables early stopping in hyperparameter tuning, and introduces the Celerity model family as a proof-of-concept.

## Strengths
- **Significant extension of prior work:** The paper convincingly validates collapse, recently shown in small-scale settings, for LLMs up to 3.9B parameters under practical co-scaling of width, depth, batch size, and weight decay, addressing an important open question from Qiu et al. (2025).
- **Clear identification of controlling factors:** Through systematic sweeps (Figs. 3, 4) and theoretical intuition (App. B.3), the paper establishes TPP and τ as the primary factors modulating TLC shape under maximal update parameterization (μP), linking collapse to compute-efficient training via optimal τ scaling.
- **Practical utility with compelling applications:** The work demonstrates two valuable, empirically-backed applications: using collapse residuals as an early, sensitive diagnostic for training pathologies (Fig. 1, right) and enabling data-efficient hyperparameter tuning via early stopping (Figs. 7, 9), which can save substantial compute.
- **Real-world demonstration with Celerity:** The introduction and training of the Celerity model family provides a tangible proof-of-concept. Celerity exhibits tight collapse (Figs. 1, 6) and achieves competitive compute-efficiency on a Pareto frontier against comparable open models (Fig. 2, Table 10).
- **Strong empirical methodology:** The conclusions are supported by an extensive set of experiments (~600 TLCs) across multiple scales, TPP values, and hyperparameters, with detailed documentation in appendices enhancing reproducibility.

## Weaknesses
- **Generality of collapse conditions is not fully established:** While the paper tests several important variations (LR schedules, MoE, Adam β), its validation is primarily within single-epoch AdamW training on specific architectures and datasets. The robustness of collapse to other optimizers (Sophia, SGD), heavy data curricula, or multi-epoch training—critical for claiming broad applicability—relies on discussion rather than experiment.
- **Predictive model for early stopping is heuristic and limited in scope:** The parametric form for normalized TLCs (Eq. 4) is fitted on 111M-scale data for a specific LR schedule (decay-to-zero). Its performance degrades with noisy curves (e.g., large batch sizes), and its applicability to other common schedules (cosine, inverse sqrt) is untested, limiting its immediate practical adoption.
- **Celerity's compute-efficiency claims are partially confounded by training data differences:** The paper notes that Celerity avoids data annealing and specialized mid-training data used by many contemporary models, which complicates direct "compute-efficiency" comparisons (Fig. 2). A more controlled ablation isolating the benefit of collapse conditions from other optimizations (data mix, parameterization) would strengthen the claim that collapse itself signifies optimal training.
- **Theoretical analysis remains preliminary:** The noisy quadratic model (App. B.3) provides helpful intuition but is restricted to constant LR; the extension to decaying schedules is qualitative. A more rigorous theoretical link between collapse, scale-invariant curvature under μP, and the power-law derivations would strengthen the foundation.
- **Demonstration scale is limited relative to stated motivation:** The paper motivates collapse as critical for "frontier-scale" training where experimentation is prohibitive, yet empirical validation only extends to 3.9B parameters. Evidence at larger scales (e.g., 10B+) would significantly bolster the argument for predictability at the frontier.
- **Reproducibility could be enhanced:** While appendices are detailed, the paper does not release training code, configuration files, or exact data mixing scripts, which limits full reproducibility for a conference like ICLR.

## Nice-to-Haves
- Testing the early-stopping methodology on a broader set of hyperparameters (e.g., architecture dimensions, LR schedule parameters) beyond λ and B sweeps.
- A more systematic, quantitative evaluation of the collapse diagnostic for multiple common pathology types (e.g., data repeats, gradient spikes) to establish detection latency and robustness.
- Side-by-side visual comparisons of training with optimal τ (collapsing) versus mis-scaled τ (non-collapsing) to reinforce the correlation between collapse and training efficiency.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness about parameterization impact:** The paper's core claims about collapse do not depend on the specific choice of CompleteP over μP; collapse is shown under both, and the switch is presented as an implementation detail for efficiency.
- **Weakness about insufficient testing of LR schedules:** The paper explicitly tests constant, 10× decay, and decay-to-zero schedules in Appendix B.4 (Fig. 11) and finds collapse persists.
- **Weakness about early stopping experiments being too narrow:** The experiments on λ and B sweeps are sufficient to demonstrate the core principle; demanding tests on all hyperparameter types is beyond the paper's scope.
- **Nitpicks about formatting artifacts in parsed figures:** These are parser issues, not problems with the original paper.

## Novel Insights
The paper synthesizes the insight that the normalized AdamW timescale τ acts as a unifying control knob for training loss curve shape, governing the bias-variance trade-off throughout training. When combined with the tokens-per-parameter ratio (TPP)—which sets the power-law improvement rate—and a fixed LR schedule, these factors create a scale-invariant signature of training dynamics. This explains why prior model families like Llama-2 (with varying τ and TPP) do not collapse, while families trained with consistent, optimal settings do. The resulting collapse is not just a curiosity but emerges as a direct consequence of compute-efficient hyperparameter scaling, providing a principled reference for monitoring and tuning.

## Suggestions
- Conduct a controlled ablation: train identical models (same data, architecture) under optimal τ (collapsing) versus a mis-scaled τ (non-collapsing) and compare final loss/compute efficiency. This would directly isolate and validate the claim that collapse signifies compute-efficient training.
- Release training code, configuration files, and data mixing scripts for Celerity and the main experiments to enhance reproducibility and community adoption.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
