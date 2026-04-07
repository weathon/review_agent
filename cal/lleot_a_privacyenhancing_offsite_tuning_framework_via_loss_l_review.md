=== CALIBRATION EXAMPLE 36 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is accurate. The abstract describes the two key properties—perplexity amplification and gradient alignment—and situates the contribution against prior Offsite Tuning (OT) work. One subtle overstatement: the abstract says the method "simultaneously" achieves both properties "theoretically," but the formal guarantee assumes that the loss shift *LE(P;x) = LM(P;x) + H* holds exactly for all P and x, which the training procedure (Eq. 7) can only approximate. This distinction is never made explicit and is a recurring issue throughout the paper.

---

### Introduction & Motivation (Section 1)

The problem is genuinely relevant: prior OT methods inadvertently ship high-capability emulators to potentially malicious data owners, which is a blind spot that the community has not systematically addressed. The threat model is clearly articulated and Figure 1 provides an intuitive summary.

However, the paper immediately narrows its scope to *soft prompt adapters* without justifying this choice relative to LoRA—the far more widely used and practically important adapter family. LoRA is mentioned (FedBiOT uses it) but then abandoned. This limits the paper's practical impact considerably and the justification given ("computational efficiency") is weak since LoRA is also computationally cheap.

---

### Problem Formulation & the CPL Metric (Section 3)

The Capability Privacy Leakage (CPL) metric, defined as `CPL = Szs(E) / Szs(M) × 100%`, has several conceptual problems that are not acknowledged:

1. **CPL can exceed 100%.** Table 1 shows OT on WebQuestions with Qwen2-1.5B and DR=0.2 yielding CPL=220.95%. This means the emulator *outperforms* the original model in zero-shot, a result that is neither explained nor consistent with the framing that "below 100% means protection is in effect." The authors need to explain this anomaly.

2. **Absolute capability is ignored.** A CPL of 50% means very different things if the original model has 90% zero-shot accuracy (emulator at 45%—still highly capable) versus 20% (emulator at 10%—near-random). The metric is dimensionless and does not capture whether the emulator's *absolute* capability is harmful.

3. **The metric is task-specific and zero-shot only.** A data owner could use the emulator for few-shot inference, knowledge distillation into a private model, or latent-space analysis. None of these vectors are captured by CPL. The protection guarantee is narrower than claimed.

4. **Random baseline shows high CPL.** The "Random" row in Table 1 exhibits CPL values as high as 74.05% (Llama-3.2 on SIQA). A randomly initialized model trivially guesses at multiple-choice questions (near 1/N chance), and so does the original model on some tasks. This suggests CPL partially reflects task-specific random-chance baselines rather than genuine capability transfer.

---

### Method (Section 4)

#### 4.1 Collaborative Prompt Knowledge Distillation (CPKD)

The motivation for using *randomly initialized* proxy soft prompts P' ~ N(μ, σ²) is reasonable: discrete-token distillation neglects the continuous embedding space. However, the key hyperparameter σ is set to 20 "determined experimentally" with no ablation over its value. Given that σ controls how broadly the proxy prompts sample the continuous space, this is a significant undisclosed sensitivity. The distribution of actual fine-tuned prompts might lie far from the initialization, meaning alignment at training-time proxies may not generalize to inference-time prompts.

#### 4.2 Loss Landscape Elevation (LLE)

**Theorem 1 is much weaker than advertised.** The theorem states that *if* `LE(P;x) = LM(P;x) + H` for all P and x, then `∇P LE = ∇P LM`. This is a mathematical tautology: the gradient of (f + constant) equals the gradient of f. The authors present this as a meaningful theorem, but the core difficulty—which is entirely avoided—is: *does the training objective (Eq. 7) actually achieve the functional form in Eq. 6 for downstream prompts?*

The answer is clearly "no, only approximately." The LLE training minimizes `E_{x~Xe, P'~N(μ,σ²)} |LE(P';x) - LM(P';x) - H|`, which constrains the emulator loss only at the specific distribution of elevation data Xe and proxy prompts P'~N(μ,σ²). During downstream prompt tuning, the actual soft prompt P drifts away from this distribution as gradients are applied. There is no formal bound on the approximation error `ε(P) = |∇P LE - ∇P LM|` for prompts outside the training distribution. The accuracy degradation observed in ablation experiments (Table 2, row 2 for DR=0.5: Acc drops from 34.20 to 23.00 without CPKD) suggests this approximation error is non-trivial.

A genuine theoretical contribution would derive a bound on the gradient approximation error as a function of the elevation training objective value, the distribution shift from proxy prompts to fine-tuned prompts, and the smoothness of the emulator.

**The interaction between CPKD and LLE is unanalyzed.** CPKD first aligns E to M. Then LLE perturbs E's parameters further to induce the loss shift. But modifying E for LLE will partially undo the gradient alignment achieved by CPKD—the paper provides no analysis of whether LLE degrades the CPKD alignment, and the ablation in Table 2 does not disentangle this.

---

### Experiments & Results (Section 5)

**Missing baseline:** FedBiOT (Wu et al., 2024) is mentioned prominently in the introduction as a direct extension of OT but is absent from Table 1. This is a significant omission given that FedBiOT also targets the model+data dual-privacy problem.

**Model and task scope are narrow.** All three models are small (<3.2B parameters). The commercially valuable models worth protecting from capability leakage are typically 13B+. Whether LLE generalizes to large models with complex loss landscapes is unexplored. All four benchmarks are multiple-choice question answering; the method's applicability to generation, summarization, or classification is entirely untested, yet the motivating use cases (healthcare, finance) are rarely multiple-choice tasks.

**Soft prompts vs. LoRA.** All baselines (OT, CRaSh) use LoRA-style adapters with 187–400M parameters (Table 7), while LLEOT uses soft prompts with 7.6–15.4K parameters. This is a fundamentally different adapter family with different expressivity. It is therefore entirely unclear whether LLEOT's accuracy gains over OT and CRaSh stem from the LLE mechanism or simply from optimizing with a completely different adapter type under a different paradigm. The comparison is not apples-to-apples, and the authors do not attempt to isolate these factors.

**Statistical reporting.** Results in Table 1 are "averaged over three experimental runs" but no standard deviations are reported. Given that soft prompt tuning can have high variance (the prompts are initialized randomly), confidence intervals are needed to assess statistical significance of small differences like 33.87 vs. 33.60 (OBQA, Qwen2, DR=0.2).

**The WebQuestions anomaly.** On WebQuestions with Llama-3.2 at DR=0.5, LLEOT achieves only 15.45% accuracy versus OT's 23.90% (Table 1). The paper provides no discussion of why performance collapses on this task/model combination. Similarly, CPL for the Ours method on WebQs is 0.00% on several settings—meaning the emulator performs at or below 0% accuracy, which requires explanation (is this because 0% means near-zero, or literally 0 correct predictions?).

---

### Ablation Study (Section 5.3 & Appendix B)

The ablation studies are a genuine strength. Table 2 clearly demonstrates the necessity of both CPKD and LLE. Table 3 validates all three loss terms in CPKD. Figure 4 confirms the intuition that downstream accuracy is insensitive to H while CPL decreases—this is a useful empirical validation of the gradient preservation claim.

Table 6 (Appendix B.1) comparing LLE against "Negative Language Modeling" (NLM) as an elevation strategy is important and shows LLE's advantage. This should be in the main paper, not the appendix, as it is a key differentiator.

The proxy prompt standard deviation σ is conspicuously absent from ablations—given that CPKD and LLE both depend on sampling P'~N(μ, σ²), σ is a critical hyperparameter that deserves investigation.

---

### Limitations & Broader Impact

The paper has an ethics statement but no dedicated limitations section. The following key limitations are unacknowledged:

1. **Soft-prompt-only scope**: The entire framework is predicated on soft prompts. The rising practical standard is LoRA; leaving it unaddressed limits applicability.
2. **Small model sizes**: Generalization to commercially sensitive large models (≥13B) is undemonstrated.
3. **Multiple-choice only**: All tasks are structured classification; the method's behavior on open-ended generation is unknown.
4. **Adaptive attacks**: A sophisticated adversary aware of the LLE mechanism could attempt to reverse the constant shift (e.g., by subtracting H from the loss before using the emulator for inference). Since H must be communicated in the emulator or can be estimated empirically, this trivially breaks the privacy guarantee. The paper does not consider adversarial robustness.
5. **Two-dataset requirement**: CPKD (12.5% of Pile) and LLE (1% of Pile) both require publicly available data from the model owner, which is a reasonable assumption but introduces a new dependency that is not discussed.

---

### Overall Assessment

LLEOT addresses a real and previously underexplored problem—capability privacy leakage through overly capable emulators in Offsite Tuning. The central mechanism (shifting the loss by a constant H to degrade perplexity while preserving gradients) is elegant and practically simple to implement. However, the paper overstates its theoretical contribution: Theorem 1 is a tautology conditioned on an exact functional form that cannot be achieved in practice, and no approximation error analysis is provided. The CPL metric has conceptual flaws (unbounded, ignores absolute capability, zero-shot only), and the experimental scope is narrow in model size, adapter type, and task diversity. Most critically, the comparison with OT and CRaSh is confounded by a fundamentally different adapter family: comparing 7.6K-parameter soft prompts against 187M-parameter LoRA adapters makes it impossible to attribute accuracy differences to the LLE mechanism. The missing FedBiOT baseline is also a notable gap. As submitted, the paper does not meet the ICLR bar for theoretical rigor or experimental fairness. Significant revisions—particularly an equitable adapter comparison, approximation error analysis, and experiments with LoRA adapters or larger models—are needed before acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses a critical gap in privacy-preserving Offsite Tuning for LLMs: existing methods risk "capability privacy leakage" by allowing emulators to perform inference similar to the original model. The authors propose LLEOT, a framework using Loss Landscape Elevation (LLE) to enforce a fixed loss margin between the emulator and the original model, theoretically proven to degrade the emulator's inference ability while preserving gradient alignment for soft prompt transfer. Extensive experiments on multiple models and datasets demonstrate that LLEOT achieves superior privacy protection compared to state-of-the-art baselines without sacrificing adaptation utility.

### Strengths
1.  **Identification of a Specific Privacy Risk:** The paper clearly articulates and quantifies "Capability Privacy Leakage" (CPL), a nuance in Offsite Tuning often overlooked. By defining the CPL metric (Eq 1) and demonstrating via Figure 1 that existing emulators (like OT and CRaSh) retain significant inference power, the work justifies the need for a new framework.
2.  **Strong Theoretical Guarantee:** Theorem 1 provides a rigorous mathematical foundation for the Loss Landscape Elevation technique. It proves that adding a constant margin $H$ to the loss function preserves the gradient $\nabla_P L$ with respect to the soft prompt while exponentially increasing the emulator's perplexity ($PPL_E = e^H \cdot PPL_M$). This offers a solid guarantee that prompt optimization will converge similarly despite the degraded inference capabilities.
3.  **Empirical Validation and Metric:** The experimental setup is comprehensive, testing on three different LLM families (Qwen, Gemma, Llama) and four datasets. The results (Table 1) convincingly show that LLEOT achieves significantly lower CPL scores than baselines while maintaining competitive accuracy. The ablation study (Table 2) further isolates the contribution of the CPKD and LLE components.

### Weaknesses
1.  **Limitation to Soft Prompts:** The framework, including the theoretical analysis of gradient alignment, is currently tailored specifically for soft prompt tuning (Section 4.3). While efficient, the approach does not explicitly address parameter-efficient finetuning methods like LoRA (mentioned in Related Work), which are increasingly common in the community. The theoretical guarantee for LoRA adapters is not provided, limiting generalizability.
2.  **Stability of Emulator Training:** The LLE mechanism requires optimizing emitter parameters $\Theta_E$ to minimize $|L_E - L_M - H|$ (Algorithm 1, Step 12). While theoretically sound, the practical training stability of forcing a model to consistently exhibit a specific loss offset across all inputs is not empirically discussed. There is no analysis of how sensitive the final accuracy is to the choice of margin $H$ or the difficulty of enforcing this constraint during emulator construction.
3.  **Baseline Comparison Depth:** While OT and CRaSh are compared, the field of privacy-preserving tuning is rapidly evolving. Baselines like FedBiOT (Wu et al., 2024) are mentioned but not in the main comparison tables (Table 1 only shows OT and CRaSh). A more comprehensive comparison with recent federated or secure multi-party computation-based finetuning methods would strengthen the claim of "state-of-the-art" status.

### Novelty & Significance
**Novelty:** The core contribution—Loss Landscape Elevation—is novel. While loss shifting is a known technique in general optimization, applying it specifically to anonymize the emulator's inference capability while maintaining prompt transferability in the context of Offsite Tuning is a new insight. The formalization of Capability Privacy Leakage as a metric is also a valuable addition to the privacy community.
**Significance:** Addressing the IP protection barrier for closed-source LLMs is highly significant for industrial adoption. If LLEOT can prevent model owners from leaking their proprietary capabilities, it removes a major barrier to collaborative fine-tuning, potentially unlocking many domain-specific applications in healthcare and finance mentioned in the introduction.

### Suggestions for Improvement
1.  **Extend Analysis to LoRA Adapters:** To increase the paper's impact, add an optional section or experiment validating the feasibility of LLEOT with LoRA adapters. If the theory changes, clarify how the constant loss margin assumption holds for adapter-specific parameters versus prompt vectors.
2.  **Analyze the Training Sensitivity of LLE:** Provide a sensitivity analysis on the hyperparameter $H$ during the *emulator training phase* (not just the prompt tuning phase, as shown in Figure 4). Specifically, investigate cases where the LLE optimization fails to converge to the desired margin or causes the emulator to become overly collapsed.
3.  **Include Emerging Baselines:** Update the evaluation to include FedBiOT or other recent 2023-2024 Offsite Tuning variants if they are available or if their implementation can be added for a fairer comparison of privacy utility trade-offs.
4.  **Clarify Code Reproducibility:** The Reproducibility Statement mentions code will be released "upon publication." ICLR usually requires reproducible code submission at the review stage if possible, or at least a more detailed plan. If the emulator construction is complex, providing pseudocode or hyperparameters for the *LLE optimization step* in the main text (beyond Algorithm 1) would aid reproducibility.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Attacker Fine-tuning Robustness:** Test whether an adversary can restore the emulator's capability by fine-tuning it on public data. Without this, the claim that LLE prevents model misuse is unverified, as high zero-shot perplexity does not guarantee resistance to transfer learning attacks.
2. **Gradient Alignment on Downstream Data:** Measure the cosine similarity of gradients between the emulator and original model specifically on the private downstream datasets ($D_p$). Theorem 1 assumes alignment everywhere, but LLE is only optimized on construction data ($X_e$); without empirical verification on $D_p$, the utility claim is unsupported.
3. **Computational Cost Analysis:** Quantify the overhead of emulator construction, which requires running the original model $M$ for every batch to compute $L_M$. If constructing the emulator is as expensive as direct fine-tuning, the practical utility of the framework for large-scale deployment is undermined.
4. **Privacy-Utility Pareto Frontier:** Plot a comprehensive trade-off curve varying $H$ against both accuracy and a robust privacy metric (e.g., attack success rate). Current results show single points; without a frontier, it is unclear if LLEOT genuinely dominates baselines or just operates at a different trade-off point.

### Deeper Analysis Needed (top 3-5 only)
1. **Generalization of Loss Margin:** Analyze whether the fixed margin $H$ holds for out-of-distribution downstream data not seen during emulator construction. If $L_E(P) \neq L_M(P) + H$ on $D_p$, the theoretical guarantee of gradient equality (Theorem 1) fails, breaking the core mechanism.
2. **Formal Privacy Definition:** Replace the heuristic CPL metric with a formal privacy definition (e.g., differential privacy or indistinguishability from random). ICLR standards require rigorous privacy guarantees rather than empirical zero-shot score ratios which can be bypassed.
3. **Sensitivity to Proxy Prompts:** Investigate how the distribution of proxy prompts ($P' \sim \mathcal{N}$) affects the loss landscape alignment. If the alignment is sensitive to the proxy distribution, the method may fail on downstream prompts that diverge from this distribution.
4. **Baseline Fairness:** Clarify if OT and CRaSh baselines were re-implemented specifically for soft prompts with comparable hyperparameter tuning. If baselines are sub-optimally adapted, the claimed superiority of LLEOT may be an artifact of experimental setup rather than methodological strength.

### Visualizations & Case Studies
1. **Gradient Cosine Similarity Heatmaps:** Plot gradient alignment between $E$ and $M$ across training steps on $D_p$. This directly visualizes whether the "gradient alignment" claim holds during actual downstream optimization or degrades over time.
2. **Loss Landscape Slices on $D_p$:** Visualize the loss landscapes of $E$ and $M$ specifically on downstream private data, not just construction data. This reveals if the elevation generalizes to the actual task or if the landscapes diverge on unseen inputs.
3. **Attack Success Rate Curves:** Plot an adversary's performance vs. fine-tuning epochs on the emulator. This visualizes the "hardness" of extracting capability compared to baselines, providing concrete evidence of privacy protection beyond static metrics.

### Obvious Next Steps
1. **Extend to LoRA/Adapter Layers:** Demonstrate LLE works for LoRA or standard adapters, as soft prompts are capacity-limited. Restricting to soft prompts limits the contribution's applicability to real-world fine-tuning scenarios where deeper adaptation is needed.
2. **Simulate Specific Threat Models:** Explicitly simulate a malicious data owner attempting to distill the emulator into a new model. This moves beyond abstract metrics to demonstrate resistance against concrete intellectual property theft scenarios.
3. **Ablate Construction Data Size:** Analyze how the size of the elevation dataset ($X_e$) impacts privacy and utility. If large public datasets are required to stabilize the loss margin, the barrier to entry for model owners increases significantly.

# Final Consolidated Review
## Summary

LLEOT addresses the previously overlooked problem of "capability privacy leakage" in Offsite Tuning—where emulators retain sufficient inference ability for data owners to extract proprietary knowledge. The framework introduces Loss Landscape Elevation (LLE) to enforce a fixed loss margin between emulator and original model, theoretically connecting this margin to perplexity amplification while preserving gradient alignment for soft prompt transfer. Experiments across three LLM families and four benchmarks demonstrate substantially reduced capability leakage compared to prior methods while maintaining competitive adaptation performance.

## Strengths

- **Identification of a real blind spot:** The paper correctly identifies that prior OT methods (OT, CRaSh) ship emulators with non-trivial zero-shot performance (Figure 1c shows ~70-90% of original model capability), enabling capability theft. The proposed Capability Privacy Leakage (CPL) metric, while heuristic, provides a concrete quantitative measure of this risk.

- **Principled theoretical insight with empirical support:** The core insight—that adding a constant H to loss preserves gradients (trivial mathematically) but exponentially increases perplexity (PPL_E = e^H · PPL_M)—is correct and empirically validated (Figure 4 shows CPL drops dramatically with H while accuracy remains stable). This dual property is elegant and practically implementable.

- **Comprehensive ablation structure:** Tables 2 and 3 clearly demonstrate the necessity of both CPKD and LLE stages, and Figure 4 shows robustness of accuracy to the H hyperparameter. The ablation comparing LLE vs. negative language modeling (Table 6) demonstrates that the constrained elevation strategy matters for maintaining gradient alignment.

- **Strong empirical privacy-utility trade-off:** Table 1 shows LLEOT achieves CPL values comparable to or below Random baseline while maintaining accuracy close to or exceeding OT/CRaSh baselines across most settings—evidence that the method successfully balances its dual objectives.

## Weaknesses

- **Theorem 1 overclaims without addressing approximation error:** The theorem states that *if* L_E(P;x) = L_M(P;x) + H holds exactly, then gradients match. But the training objective (Eq. 7) only enforces this approximately on proxy prompts P' ~ N(μ,σ²) sampled from a specific distribution over elevation data X_e. The paper provides no bound on how well this generalizes to downstream prompts P* obtained after gradient descent, nor analysis of how the proxy prompt distribution affects alignment. The accuracy drop without CPKD (Table 2, DR=0.5: 34.20 → 23.00) suggests approximation error can be substantial.

- **CPL metric is narrow and lacks formal privacy grounding:** CPL measures only zero-shot accuracy ratio on specific tasks. It does not capture: (a) absolute capability (a 50% CPL with original at 90% accuracy means emulator still has 45% capability), (b) few-shot or fine-tuning attack vectors, (c) knowledge distillation into a private model, or (d) whether CPL ≤ Random baseline provides meaningful privacy guarantees. The metric is task-specific and heuristic rather than grounded in formal privacy definitions.

- **Method is fundamentally limited to soft prompts:** The theoretical framework and implementation are specific to soft prompt tuning. The paper mentions LoRA adapters (used by FedBiOT) but provides neither theoretical nor empirical validation for parameter-efficient methods beyond soft prompts. This is a significant practical limitation given LoRA's prevalence in fine-tuning practice.

- **Apples-to-oranges adapter comparison:** Table 7 shows LLEOT uses soft prompts (7.6K-15.4K parameters) while OT/CRaSh baselines use their original adapter format (187-403M parameters). While the paper correctly reports adapter sizes, the accuracy comparison confounds method contributions with adapter expressivity differences. The paper should clarify whether the baselines were re-implemented with comparable soft prompts, or whether the comparison is fair as-is.

- **Missing FedBiOT baseline:** FedBiOT is cited in the introduction as a direct extension of OT to federated settings and explicitly uses LoRA adapters—yet it is absent from Table 1. This is a notable omission given its relevance to the same privacy problem.

- **Unexplained performance anomalies:** On WebQuestions with Llama-3.2 (DR=0.5), LLEOT achieves only 15.45% accuracy versus OT's 23.90%—a substantial gap not discussed. Similarly, CPL values of 0.00% (e.g., WebQs with Qwen2) suggest zero correct predictions; the paper should explain whether this reflects successful capability suppression or a different phenomenon.

- **No statistical uncertainty reported:** Results claim averaging over three runs without standard deviations. Given soft prompt optimization's known variance, confidence intervals are needed to assess significance of differences like 33.87 vs 33.60.

## Nice-to-Haves

- **Proxy prompt standard deviation (σ) ablation:** σ=20 is set experimentally without investigation. Since both CPKD and LLE depend on sampling P' from this distribution, understanding sensitivity to σ would strengthen the method's robustness claims.

- **Out-of-distribution gradient alignment verification:** Measure cosine similarity between ∇_P L_E and ∇_P L_M specifically on downstream private data D_p, not just elevation data X_e. This would empirically validate the key theoretical assumption.

- **Attacker fine-tuning robustness:** Test whether an adversary can restore emulator capability by fine-tuning on public data. Current zero-shot perplexity evaluation doesn't capture this realistic attack vector.

## Removed Points

- **"CPL exceeding 100% is a metric bug"**: This is expected behavior—CPL is a ratio, and emulators can occasionally outperform original models on specific tasks due to compression effects or initialization. The paper correctly states "below 100% means protection is in effect."

- **"Random baseline CPL being high is problematic"**: This reflects task-specific random-chance baselines (multiple-choice with 4-5 options) and is mathematically expected when both models perform poorly.

- **"Adaptive attacks by subtracting H from loss"**: This is beyond the paper's threat model, which focuses on misuse through inference capability, not reverse-engineering the protection mechanism.

- **"Theorem 1 is just calculus—adding constant doesn't change gradient"**: While mathematically elementary, the theorem's value lies in connecting the loss margin to perplexity amplification (PPL_E = e^H · PPL_M), which is non-trivial and empirically meaningful.

- **"Should test on generation/summarization tasks"**: The paper scopes to multiple-choice QA, which is reasonable for a focused contribution. Extension to other task types is future work.

- **"Missing theoretical bound on gradient approximation error"**: While valuable, this is an incremental improvement suggestion rather than a fundamental flaw. The empirical results (Figure 4, Table 2) provide practical evidence of the method's effectiveness.

## Novel Insights

The paper's key insight is that privacy-preserving emulator construction need not abandon useful gradient information. Prior methods either preserve both inference capability and gradient structure (OT, CRaSh) or destroy both (random initialization). LLEOT demonstrates that loss landscape geometry—specifically, the gradient field—can be decoupled from inference capability through a controlled elevation strategy. The theoretical connection between the loss margin H and perplexity amplification factor e^H provides practitioners with a principled knob for trading capability degradation against utility preservation. This separation of concerns (destroy inference, preserve gradients) is a genuinely novel contribution to the privacy-preserving ML toolbox.

## Suggestions

1. **Add a paragraph discussing approximation error:** Acknowledge that the training objective only approximately achieves the functional form L_E ≈ L_M + H, and discuss conditions under which gradient alignment holds on downstream prompts (e.g., smoothness assumptions, distribution overlap between proxy prompts and fine-tuned prompts).

2. **Report standard deviations:** Add confidence intervals to Table 1 to enable readers to assess statistical significance.

3. **Discuss the WebQuestions anomaly:** Add brief analysis of why performance drops on this specific task/model combination and whether it indicates limitations of the approach.

4. **Clarify adapter comparison fairness:** Either re-run baselines with soft prompts for fair comparison, or explicitly justify why comparing different adapter families is methodologically sound.

5. **Expand CPL discussion:** Acknowledge that CPL is a heuristic metric capturing only one threat vector, and discuss what formal privacy guarantees might complement it (e.g., indistinguishability from random under specific attack models).

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
