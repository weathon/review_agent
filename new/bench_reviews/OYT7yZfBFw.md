## Summary

This paper proposes TrajGPT, a Transformer for irregularly-sampled clinical time series that introduces a Selective Recurrent Attention (SRA) mechanism with data-dependent decay. The authors frame SRA as a discretization of a neural ODE to motivate a “time-specific inference” mode for forecasting at arbitrary timesteps, and evaluate the model on EHR trajectory forecasting, drug usage prediction, and phenotype classification across two health datasets.

## Strengths

- **Data-dependent decay is a sensible extension with empirical support.** The SRA mechanism introduces a context-aware decay term γ_n = Sigmoid(X_n w_γ^T)^{1/τ} (Eq. 2), allowing the model to adaptively weight past information. The ablation in Table 3 shows that removing decay gating reduces top-10 forecasting recall by 1.4 percentage points, confirming the design has value.
- **Strong zero-shot performance on clinically relevant tasks.** TrajGPT achieves competitive or best zero-shot results on multiple benchmarks: e.g., 67.2% AUPRC for insulin usage prediction and 72.8% for CHF classification on PopHR, and 45.1% AUPRC for sepsis detection on eICU (Tables 1 and 2). These demonstrate that the pre-trained representations transfer to downstream tasks.
- **Clinically interpretable visualizations.** Figure 3 shows that token embeddings cluster by disease category and that sequence representations separate insulin-positive from insulin-negative diabetic patients. The trajectory case studies in Figure 4 connect model predictions to comorbidity patterns, which is valuable for clinical adoption.

## Weaknesses

### Fatal
None.

### Major
- **ODE interpretation in Eq. 5 contains a sign error that breaks the claimed equivalence.** The paper defines A = ln(Λ_t)/Δ with Λ_t = diag(1/γ_t). Because γ_t ∈ (0,1], this yields A with non-negative eigenvalues and a transition matrix e^{AΔ} = diag(1/γ_t) > 1, implying an unstable continuous system whose state explodes. This contradicts the discrete recurrence S_n = γ_n S_{n-1} + … in Eq. 2, which requires e^{AΔ} = γ_n < 1. The correct definition should use Λ_t = diag(γ_t). As written, the paper does not establish TrajGPT as a discretized ODE of the claimed form, undermining a central theoretical contribution.
- **Time-specific inference is critically underspecified and appears circular.** Section 3.2 states that to forecast a target (x_{n'}, t_{n'}), the model computes O_{n'} = Q_{n'} S_{n'} with S_{n'} = D_{Δ} S_n + K_{n'}^T V_n. Per Eq. 1, however, Q_{n'} = X_{n'} W_Q e^{iθ t_{n'}} and K_{n'} = X_{n'} W_K e^{-iθ t_{n'}}, both of which require the unobserved target observation x_{n'} itself. The paper neither resolves this dependency (e.g., via a learned time-only query) nor provides an alternative inference algorithm. As presented, the mechanism for forecasting arbitrary target timesteps in a single step is incoherent.
- **Empirical advantages over strong baselines are frequently within statistical noise and lack significance testing.** Table 1 reports bootstrap standard errors of roughly ±2–4%, yet several margins between TrajGPT and strong baselines are smaller than one standard error (e.g., PopHR top-10 recall: 71.7±2.6 vs. TimelyGPT 70.3±3.1 and mTAND 70.2±2.5; CHF zero-shot: 72.8±2.4 vs. PatchTST 72.2±2.3). The paper boldfaces “best” results without paired statistical tests, so the evidence does not credibly establish superiority on these metrics.

### Minor
- **Pre-training protocol differs systematically across models, confounding comparisons.** TrajGPT is pre-trained with next-token prediction while competing Transformers use masking-based pre-training (Section 4.4). A fairer ablation would hold the pre-training objective constant to isolate the architectural contribution of SRA.
- **Abstract slightly overstates the zero-shot story.** The abstract claims TrajGPT performs drug usage and phenotype classification “without requiring task-specific fine-tuning,” yet Table 1 presents fine-tuned TrajGPT as a primary result and shows that fine-tuned baselines sometimes match or exceed it (e.g., mTAND on CHF fine-tuning: 85.4 vs. 83.9).

### Trivial
- Typo “BitMimicGPT” in Table 1 (should be BiTimelyGPT).

## Nice-to-Haves
- Add failure-mode case studies to Figure 4, especially for time-specific inference, to reveal extrapolation errors or excessive smoothing.
- Include an ablation comparing data-dependent decay to a simple parametric decay γ_{nm} = exp(−α(t_n − t_m)) to isolate whether learned gating is necessary or a fixed timescale suffices.

## Removed Points
These points are flagged to be removed, treat them with caution.
- **Criticism about model/dataset existence or availability:** The paper cites all models and datasets; they are assumed to exist and be available. Concerns about “not yet released” or “cannot be independently verified” are reviewer knowledge gaps, not author errors.
- **Typos, spelling, grammar, formatting artifacts:** These are parser issues in the extracted text, not present in the original submission.
- **Missing appendix or proofs in appendix:** The parser strips appendix sections; they exist in the original submission.
- **Pre-training differences as a fatal flaw:** While suboptimal for isolating architectural contributions, comparing full systems with their standard pre-training paradigms is common in representation learning papers and does not invalidate the empirical results.
- **BitMimicGPT typo as a substantive weakness:** This is a trivial presentation issue.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
1. Fix the sign error in Eq. 5 by replacing Λ_t = diag(1/γ_t) with diag(γ_t), and verify that the appendix derivation is consistent.
2. Explicitly define how Q_{n'} and K_{n'} are computed when x_{n'} is unobserved at inference time (e.g., learned query embedding, time-only embedding, or autoregressive proxy). Without this clarification, the time-specific inference claim is invalid.
3. Report paired statistical significance tests or confidence intervals for the differences between TrajGPT and the best baseline in Tables 1–2, because many margins are smaller than the reported standard errors.

## Score and Decision

**Calibration anchors:**
- **High (≥6):** MOTOR (7.5, Accept spotlight) — first TTE foundation model for EHR, massive scale, thorough experiments, strong cross-site validation. TrajGPT is below this: smaller scale, narrower tasks, and flawed theoretical framing.
- **High (≥6):** ACSSM (8.0, Accept Oral) — novel theoretically-motivated method for irregular time series with correct derivations and strong empirical results. TrajGPT falls well below this due to the sign error in its ODE derivation.
- **Medium (~5):** TimelyGPT (5.5, Reject) — direct predecessor with similar architecture but no math errors; rejected for being incremental and lacking theory. TrajGPT adds theory but it is incorrect, and adds an underspecified inference mode, so it should score no higher.
- **Medium (~5):** XTSFormer (5.0, Reject) — missing key baselines and organizational issues, but no mathematical errors in core claims. TrajGPT has more serious theoretical flaws.
- **Low (≤4):** FactTest (4.67, Reject) — extensive experiments but core statistical claims were unsupported/incorrect. TrajGPT shares this pattern: empirical work is real, but central theoretical claims are flawed.
- **Low (≤4):** “Is Pre-training Truly Better Than Meta-Learning?” (4.5, Reject) — rigorous experiments but marginal effect sizes weakened conclusions. TrajGPT has similarly small margins and adds mathematical errors.

**Comparison:** TrajGPT has real empirical contributions on an important clinical problem, but the sign error in Eq. 5 and the circular/underspecified time-specific inference are substantive flaws in the core technical claims. These issues are more severe than the weaknesses in the medium-scoring anchors (TimelyGPT, XTSFormer) and comparable to the low-scoring anchors with unsupported core claims (FactTest). The paper would need to correct the ODE mathematics and rigorously define time-specific inference to reach the medium band.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>