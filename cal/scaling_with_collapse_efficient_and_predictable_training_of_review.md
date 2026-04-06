=== CALIBRATION EXAMPLE 77 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title accurately reflects the paper's focus on scaling with training loss curve collapse. The abstract clearly states the key findings: collapse persists under practical scaling recipes when hyperparameters (particularly the AdamW timescale τ) are set optimally for a given TPP, and demonstrates two applications (diagnostics and early stopping). All claims are supported by the paper's content.

**Introduction & Motivation:** The motivation is strong, framing collapse as a tool for achieving predictable, efficient training at scales where direct experimentation is prohibitive. It correctly identifies the gap from Qiu et al. (2025) by testing in a practical LLM training regime. The contributions are listed precisely and map well to the paper's structure.

**Method / Approach (Section 3, Appendices B.2 & B.3):**
*   **Reproducibility:** Experimental setup (Section 3, Appendix B.1) is sufficiently detailed (model specs, dataset, parameterization). The definition of normalized loss (using final loss) and smoothing procedure are clear.
*   **Key Findings:** The empirical findings that τ and TPP modulate normalized TLC shape are well-supported by Figures 3 and 4. The claim that collapse requires matching {TPP, τ, LR schedule} is convincing.
*   **Theoretical Explanation:** The explanations using power laws (Appendix B.2 for TPP) and the noisy quadratic model (NQM, Appendix B.3 for τ) are appropriate and align with prior work. The NQM derivation is sound for its illustrative purpose, though its direct applicability to the full cross-entropy loss of LLMs is, as the authors acknowledge, an approximation. The link between the NQM analysis and the observed TLC deformations under different LR schedules (Figure 10) is insightful.
*   **Minor Gaps:** In Figure 4 (right), collapse is shown when {TPP, τ} are *approximately* constant (τ ranges from 0.27 to 0.33). The text appropriately uses "roughly," but a brief comment on the sensitivity of collapse to small τ mismatches would be helpful.

**Experiments & Results (Sections 4 & 5):**
*   **Celerity Family (Section 4):** The design choice for TPP=234 is justified via a compute/compression trade-off analysis (Appendix C.1, Figure 5). The demonstration of collapse within the Celerity bands (Figures 1, 6) is successful, with sensible attributions for minor deviations (warmup differences, late-stage overfitting). The use of collapse residuals for early issue detection (Figure 1, right) is a compelling, practical result.
*   **Evaluation & Comparisons:** Figure 2 places Celerity favorably on the compute-efficiency frontier. However, the comparison involves models trained with heterogeneous strategies (e.g., distillation for Gemma, different data mixtures). The authors' careful discussion of FLOP accounting (Figure 16, Table 9) and the intentional choice of a "standard pre-training" baseline mitigates this concern, but it remains a complex landscape for direct comparison.
*   **Early Stopping (Section 5):** The core idea—using a predicted universal normalized TLC to extrapolate final loss from a partial run—is novel and valuable. The heuristic functional form (Eq. 4) is motivated by the earlier analysis. The procedure for fitting the parameters \(b\) and \(q\) as power laws in τ and TPP is pragmatic.
*   **Experiments on Early Stopping:** The comparison baselines (random, current best) are fair. Results in Figures 9, 23, 24, and 25 strongly support the claim that the "predicted best" method enables reliable early stopping (e.g., by 30% of training), outperforming the "current best" heuristic used in prior work. The decrease in prediction MAE with model scale (Table 11) is encouraging for utility.
*   **Limitations of the Predictive Model:** The predictive model is fit and tested primarily on data using a linear decay-to-zero schedule. Its generalizability to other schedules (cosine, inverse sqrt) is an open question, rightly noted in Appendix A. Furthermore, its accuracy degrades with very noisy curves (e.g., large batch sizes), as discussed in Appendix D.2.

**Writing & Clarity:** The paper is generally well-structured and clearly written, effectively synthesizing empirical trends, theoretical intuition, and practical applications. Some figures suffer from PDF parsing artifacts (e.g., misplaced table borders, garbled text in Figure 1, 2), but the core information is recoverable from captions and context.

**Limitations & Broader Impact (Appendix A):** The limitations section is thorough and honest. It correctly identifies the scope (single-epoch, AdamW), discusses potential extensions (other optimizers, data curricula), and acknowledges key constraints of the predictive model. Broader societal impact is not discussed, which is reasonable for this technical work.

### Overall Assessment
This is a strong, practical contribution that meets ICLR's standards. The paper successfully extends the collapse phenomenon to realistic LLM training regimes, providing a robust empirical analysis across scales (up to 3.9B parameters) and hyperparameter sweeps. The introduction of the Celerity family demonstrates the principle in action, and the applications—particularly the sensitive diagnostic use of collapse residuals and the data-efficient early stopping method for hyperparameter tuning—offer clear, actionable value for large-scale training. The main limitations concern the generalizability of the specific predictive model to alternative optimization schedules and the inherent difficulty of perfectly equitable comparisons between model families trained with different data and techniques. However, the core insights about collapse as a signature of efficient training and its utility for monitoring are well-supported and likely to influence practice. The paper merits acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper demonstrates that normalized training loss curves (TLCs) for large language model (LLM) families can collapse onto a universal trajectory when trained under a "practical scaling recipe" that fixes the tokens-per-parameter ratio (TPP), learning rate schedule, and AdamW timescale (τ) across model scales. The authors show this collapse is a signature of compute-efficient training and provides two key applications: (1) using deviations from collapse as an early diagnostic for training pathologies, and (2) enabling early stopping in large-scale hyperparameter tuning by predicting final loss from partial curves. These insights are validated by training the Celerity model family, which achieves competitive performance.

### Strengths
1. **Strong Empirical Validation at Scale**: The paper presents extensive experiments across model sizes (from ~100M to 3.9B parameters) and TPP values (up to 234), clearly demonstrating TLC collapse under the proposed conditions (Figs. 1, 4, 6). The training of the Celerity family provides concrete, large-scale evidence of the method's practical utility.
2. **Practical Applications with Clear Value**: The proposed use of collapse residuals for early detection of training issues (e.g., Fig. 1, right) and the method for early hyperparameter selection (Sec. 5, Fig. 9) address significant, costly pain points in LLM development. These are directly actionable insights for practitioners.
3. **Well-Motivated Theoretical Framework**: The paper effectively builds on prior work (µP, scaling laws) and provides a clear, intuitive explanation for TLC shape modulation by τ (as a bias-variance trade-off controller) and TPP (via power-law scaling), supported by a noisy quadratic model (Appendix B.3).

### Weaknesses
1. **Limited Theoretical Novelty and Depth**: While the empirical demonstration at LLM scale is valuable, the core concept of TLC collapse extends the recent work of Qiu et al. (2025). The theoretical analysis (e.g., the noisy quadratic model) is relatively standard and does not offer a fundamental new understanding of the underlying dynamics.
2. **Predictive Model is Preliminary and Narrow**: The functional form for predicting normalized TLCs (Eq. 4) and the fitting procedure are heuristic and validated primarily on a single architecture, dataset, and learning rate schedule (D2Z). Its generalizability to other schedules (cosine, WSD), optimizers, or multi-epoch training is not established, limiting its claimed utility for early stopping.
3. **Incomplete Comparison with Prior Diagnostics**: The paper claims collapse residuals are a superior diagnostic but does not quantitatively compare their sensitivity or timeliness against other established monitoring techniques (e.g., gradient norm trends, validation loss divergence) used in large-scale training reports.
4. **Assumptions Limit Generality**: The analysis and collapse phenomenon are demonstrated under specific conditions: single-epoch pre-training, AdamW optimizer, and fixed data mixtures. The discussion (Appendix A) acknowledges but does not test how the results would change with multi-epoch training, data curricula, or alternative optimizers like Sophia, which is a significant limitation for broader claims.

### Novelty & Significance
**Novelty**: The primary novelty lies in scaling the collapse phenomenon, first shown by Qiu et al. (2025) in small-scale, idealized settings, to practical LLM training regimes where width, depth, batch size, and weight decay are co-scaled. Identifying τ (AdamW timescale) as a key control knob for TLC shape under these conditions and linking its optimal setting to TPP is a concrete contribution. The application of collapse for real-time monitoring and early hyperparameter stopping is novel in the LLM context.
**Significance**: For ICLR, which values principled methods that enable more efficient and reliable large-scale deep learning, this work is highly significant. It provides a data-driven, scalable framework for ensuring training stability and efficiency, which is critical as models grow beyond the point of exhaustive experimentation. The release of the Celerity models also contributes a valuable open benchmark.

### Suggestions for Improvement
1. **Strengthen Theoretical Grounding**: Provide a more rigorous analysis connecting the observed collapse to the structure of the loss landscape in transformers under µP/CompleteP, beyond the noisy quadratic model. A deeper theoretical explanation would elevate the paper's contribution.
2. **Systematically Validate the Predictive Model**: Test the proposed TLC predictor (Eq. 4) across a wider variety of learning rate schedules (cosine, inverse sqrt), architectures (e.g., MoEs), and datasets. Report its failure modes and uncertainty to better define its operational scope.
3. **Benchmark Diagnostic Sensitivity**: Conduct a controlled study comparing the proposed collapse-residual method against other common training diagnostics (e.g., validation-train loss gap, gradient statistics) on a set of known pathologies (loss spikes, data issues) to quantify its advantage in detection time and accuracy.
4. **Explore Generality and Limitations Further**: Include preliminary experiments or a more detailed discussion on the effect of data curricula, multi-epoch training, and other optimizers (e.g., SGD, Sophia) on TLC shape and collapse. This would clarify the boundaries of the proposed theory's applicability.
5. **Enhance Comparison with Related Work**: The related work section (Sec. 6) is adequate but could be more sharply focused on contrasting the paper's "timescale-centric" view with other loss-curve prediction works (e.g., Tissue et al., Luo et al.) and early stopping/Hyperparameter Optimization methods in the LLM regime.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Controlled ablation varying only TPP or τ while holding other factors fixed.** The claim that collapse requires fixed TPP and optimal τ is supported by comparative plots (e.g., Llama-2 vs. Celerity) but lacks direct ablation. Without experiments that systematically vary one factor (e.g., sweep τ at fixed TPP, or sweep TPP at fixed τ) and measure alignment error, the causal role of each factor remains observational, not causal.
2. **Validation of collapse at significantly larger scales (e.g., >10B parameters).** The paper demonstrates collapse up to 3.9B. To substantiate claims about "LLM families" and scaling predictability for frontier models, evidence must extend to larger scales (e.g., 10B, 30B). Otherwise, the phenomenon might not hold beyond the tested range.
3. **Comparison with other standard parameterizations (e.g., standard, SP) and optimizers (e.g., Sophia, SGD).** The collapse is shown under μP/CompleteP and AdamW. To establish generality, experiments should test whether collapse occurs under different parameterizations and optimizers. If collapse is specific to AdamW+μP, its utility is limited.
4. **Systematic study of different learning rate schedules (cosine, inverse sqrt, warmup-stable-decay).** The paper primarily uses linear decay-to-zero and briefly mentions others in the appendix. A thorough sweep of common schedules is needed to verify that collapse and the predictive model (Eq. 4) generalize beyond D2Z.
5. **Realistic hyperparameter tuning demonstration with multiple jointly tuned parameters.** The early-stopping experiment uses single-parameter sweeps (λ or B). To prove practical utility, a joint tuning of key hyperparameters (e.g., LR, batch size, weight decay) on a large-scale model is required, showing compute savings without performance loss.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative metric for collapse strength (e.g., mean absolute error between normalized curves).** The paper relies on visual alignment. A quantitative measure is essential to objectively compare conditions, assess sensitivity, and support claims like "tight collapse." Without it, the evidence is subjective.
2. **Root-cause analysis of collapse failures (e.g., late divergence in 234 TPP runs).** The paper attributes late divergence to training-data overfitting but provides no investigation (e.g., analyzing loss on specific data subsets, gradient norms). Understanding why collapse breaks is critical for trusting it as a diagnostic.
3. **Theoretical explanation linking TPP and τ to collapse in a unified framework.** The paper offers separate analyses: a power-law argument for TPP and a noisy quadratic model for τ. A unified model explaining how both factors jointly determine normalized curve shape would strengthen the theoretical foundation.
4. **Analysis of how data mixture and quality affect collapse.** Celerity uses a specific blended dataset. To generalize, the paper should analyze whether collapse holds across different data mixtures (e.g., only web text, or with heavy code/math) and whether data shifts mid-training (curricula) break alignment.

### Visualizations & Case Studies
1. **Time-series of collapse residuals for multiple training runs with injected faults.** The single example in Fig. 1 (right) is insufficient. Case studies with controlled faults (e.g., simulated data corruption, hardware noise) would demonstrate the sensitivity and early-warning capability of collapse residuals.
2. **End-to-end case study of early stopping in a large-scale hyperparameter tuning.** Show a concrete example: tuning a 3.9B model using the proposed method, plotting predicted vs. actual final losses, and reporting the compute saved compared to full training. This would validate the early-stopping claim.
3. **Visualization of normalized validation loss curves across all scales and TPPs.** Appendix C.5 shows validation collapse for a few models. A comprehensive plot (similar to Fig. 6) for validation loss is needed to confirm that collapse generalizes to held-out data, which is critical for monitoring.

### Obvious Next Steps
1. **Scale the collapse analysis to models >10B parameters.** Given the paper's focus on LLM scaling, the immediate next step is to train larger models (e.g., 10B, 30B) with fixed TPP and optimal τ to verify collapse persists.
2. **Extend the τ and collapse framework to other optimizers (e.g., Sophia, SGD).** The timescale concept is currently tied to AdamW. Testing whether a similar timescale governs collapse for other optimizers would greatly expand applicability.
3. **Investigate collapse under data curricula and multi-epoch training.** Real-world training often uses data scheduling and multiple epochs. Studying how these practices affect curve alignment is necessary for practical adoption.
4. **Build and release a collapse monitoring tool.** The paper proposes using collapse residuals for diagnostics. Implementing an open-source tool that logs, normalizes, and alerts based on deviations would translate the idea into practice.

# Final Consolidated Review
## Summary
This paper demonstrates that normalized training loss curves (TLCs) for large language model families collapse onto a universal trajectory when trained with a consistent tokens-per-parameter ratio (TPP), AdamW timescale (τ), and learning rate schedule. It identifies τ as a key control for TLC shape, links optimal τ to TPP, and shows that collapse serves as a signature of compute-efficient training. The work provides two practical applications: using deviations from collapse as an early diagnostic for training issues, and enabling early stopping in hyperparameter tuning by predicting final loss from partial curves. These insights are validated by training the Celerity model family up to 3.9B parameters.

## Strengths
- **Extensive empirical validation at practical scale:** The paper presents systematic experiments across model sizes (111M to 3.9B) and TPP values (up to 234), clearly demonstrating TLC collapse under the proposed conditions (Figures 1, 3, 4, 6). Training the Celerity family provides concrete, large-scale evidence.
- **Actionable practical applications:** The use of collapse residuals for early detection of training pathologies (Figure 1, right) and the method for early hyperparameter selection (Section 5, Figure 9) address significant, costly pain points in LLM development. The diagnostic application is convincingly demonstrated with a real debugging case.
- **Clear theoretical framing:** The paper effectively builds on prior work (µP, scaling laws) and provides an intuitive explanation for TLC shape modulation via τ (controlling bias-variance trade-off) and TPP (via power-law scaling), supported by a noisy quadratic model (Appendix B.3).

## Weaknesses
- **Predictive model is heuristic and narrowly validated:** The functional form for predicting normalized TLCs (Eq. 4) and its fitting procedure are empirically motivated and validated primarily on a single architecture, dataset, and linear decay-to-zero schedule. Its generalizability to other common schedules (cosine, inverse sqrt) and training regimes is not established, which limits the claimed utility for early stopping in diverse settings.
- **Theoretical contribution is an extension rather than a fundamental advance:** The core concept of TLC collapse directly extends the recent work of Qiu et al. (2025) to practical LLM scaling recipes. The provided theoretical analysis (power-law argument and noisy quadratic model) offers useful intuition but does not provide a fundamental new understanding of the underlying dynamics in transformer-based LLMs.

## Nice-to-Haves
- A quantitative metric (e.g., mean absolute error) to measure collapse strength objectively, rather than relying solely on visual alignment.
- A more systematic comparison of the collapse-residual diagnostic against other common monitoring techniques (e.g., gradient norm trends, validation loss divergence) to quantify its relative sensitivity and timeliness.

## Novel Insights
The paper's novel insight is that under practical LLM scaling recipes, the collapse of normalized training loss curves emerges as a direct signature of compute-efficient training. This occurs when the AdamW timescale τ is set optimally for the chosen tokens-per-parameter ratio. This connection allows collapse to be used not just as an observational regularity, but as a tool for real-time training health monitoring and for accelerating hyperparameter tuning via early stopping, translating a theoretical scaling phenomenon into actionable engineering practices.

## Suggestions
- Strengthen the validation of the TLC predictor (Eq. 4) by testing its performance on at least one alternative, widely-used learning rate schedule (e.g., cosine decay) to better define its operational scope.
- In the discussion of the early-stopping application, more explicitly address the predictor's robustness to noisy or spiky loss curves, which are acknowledged as a limitation in Appendix D.2.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
