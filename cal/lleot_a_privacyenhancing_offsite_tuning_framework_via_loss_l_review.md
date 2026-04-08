=== CALIBRATION EXAMPLE 17 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is descriptive and accurate. The abstract correctly identifies the dual tension (model capability privacy + adapter transferability) and the proposed solution. However, the abstract states LLEOT "theoretically" shows LLE simultaneously degrades emulator inference *and* preserves gradient alignment. As discussed in the theory section below, the theoretical guarantee rests on an idealized assumption (exact satisfaction of Eq. 6) that the actual training procedure can only approximately achieve. This overstatement sets up expectations the body does not fully meet.

---

### Introduction & Motivation

The problem is well-motivated and the capability privacy gap in existing OT methods is genuinely underexplored. The four bullet-point contributions are clearly articulated. However, one important scoping constraint is buried: the entire framework is demonstrated only for **soft prompts**, not for the more practically dominant LoRA adapters. Contribution 1 correctly says "various types of adapters" in principle, but the actual method and all experiments are soft-prompt-only. This restricts practical relevance and should be made prominent.

---

### Problem Formulation & CPL Metric (Section 3)

**CPL metric (Eq. 1).** CPL = S_zs(E) / S_zs(M) × 100% is simple and interpretable, but has several unaddressed edge cases:

1. **CPL > 100% is possible and occurs.** In Table 1, OT on WebQs for Qwen2-1.5B at DR=0.2 achieves CPL = 220.95%. How can the emulator outperform the original on zero-shot? This anomaly is never explained and casts doubt on the metric's reliability.

2. **Zero-shot accuracy as a proxy for capability privacy is narrow.** A malicious data owner could exploit the emulator via few-shot prompting, chain-of-thought, or model inversion attacks—none of which zero-shot accuracy captures. The threat model needs to be more formally bounded.

3. **CPL below "random" baseline.** The paper claims (Section 5.2) "our method achieves a CPL score even lower than that of a randomly initialized model." If the emulator truly behaves worse than a random model, one must ask whether gradient-based prompt optimization can still yield meaningful gradients—a tension not addressed.

**Setup.** The framework requires the model owner to possess both a distillation dataset *Xd* and an elevation dataset *Xe*, both publicly sourced (Pile-uncopyright). This requirement is not stated in the problem formulation (Section 3) and is only revealed in implementation details (Appendix A.4). It should be explicit.

---

### Methodology (Section 4)

**Theorem 1 — Critical Gap.** The central theoretical claim is that LLE simultaneously (i) exponentially amplifies perplexity (Eq. 10, PPL_E = e^H · PPL_M) and (ii) preserves gradient identity (∇_P L_E = ∇_P L_M). Both results follow from the idealized condition of Eq. 6:

> L_E(P; x) = L_M(P; x) + H  ∀ P, x

However, **this is never actually enforced exactly**. The real training objective (Eq. 7) minimizes the expectation of |L_E(P'; x) − L_M(P'; x) − H| over *randomly sampled* proxy prompts P' ~ N(μ, σ²). Therefore:

- The perplexity result (part i) holds only approximately, to the extent Eq. 7's optimization converges.
- The gradient identity (part ii), while mathematically trivial when Eq. 6 holds exactly (since ∇_P H = 0 for constant H), carries no formal guarantee outside of the proxy prompt distribution.

Most critically: **gradient alignment on the elevation dataset X_e does not imply gradient alignment on the downstream data D_p**. Downstream prompts are initialized randomly and optimized toward a task-specific distribution never seen during LLE training. The claim that "convergence to the same optimal prompt" is ensured (Section 4.2, last paragraph) thus requires substantially stronger assumptions that the paper does not establish.

**CPKD (Section 4.1).** The proxy prompt distillation loss L_PPD (Eq. 3) is novel and addresses a genuine problem with applying standard KD to continuous prompt spaces. However:

- The choice σ = 20 for the proxy prompt distribution N(0, σ²) is described as "determined experimentally" (Section 4.1). This is a crucial hyperparameter: if actual downstream prompts P* at convergence lie outside the typical support of N(0, 400), neither CPKD nor LLE constraints apply to them. No coverage analysis is provided.
- The three loss weights (w₁ = 1, w₂ = 10, w₃ = 30) in Eq. 5 are heavily unbalanced and experimentally tuned. Sensitivity analysis for these is absent.

**Scope limited to soft prompts.** The paper explicitly states LLE is "applicable to various types of adapters" (Introduction, Section 4.2) but only instantiates it for soft prompts. Soft prompts are computationally efficient but practically marginal compared to LoRA (used in FedBiOT, the contemporaneous OT variant). Since LoRA adapters involve trained weight deltas (not a fixed constant offset), applying the constant-margin LLE objective to non-prompt adapters is non-trivial. The claim of general applicability is unsubstantiated.

---

### Experiments & Results (Section 5)

**Model scale.** All experiments use models in the 1.5B–3B range. Modern deployments of interest involve models an order of magnitude larger. The compression-and-distillation dynamics, the feasibility of the LLE constraint, and the soft-prompt transferability may all change at scale.

**Task diversity.** All four benchmarks (OBQA, SIQA, ARC-c, WebQs) are multiple-choice question-answering tasks evaluated via accuracy. There is no evaluation on generation tasks (summarization, code, open-ended QA), instruction-following, or tasks requiring multi-step reasoning. This homogeneous task set limits confidence in generalizability.

**Baselines.** Only OT and CRaSh are compared. More recent OT-adjacent work (FedBiOT, FedPEAT) is mentioned in related work but not included in the comparison table. This is a gap, especially since FedBiOT uses LoRA adapters and represents a more practical setting.

**Missing standard deviations.** The paper states results are "averaged over three experimental runs" (Section 5.2) but reports no standard deviations anywhere. Given that some differences between methods are only a few tenths of a percent (e.g., Table 1, Qwen2 at DR=0.2, OBQA: Ours = 33.87 vs. OT = 33.80), statistical significance cannot be assessed.

**Hyperparameter selection.** The prompt-tuning phase conducts a grid search over learning rates and "report[s] the best-performing run" (Appendix A.4). This is fine, but the best-performing run should be selected on a validation set, not the test set. It is unclear which is used. Additionally, two learning rates are tested for LLE and "the emulator that achieves the best performance" is selected—again, the selection criterion and split are not described.

**Unexplained failures.** Table 1, Gemma-2-2b at DR=0.2 on WebQs: CRaSh achieves the best accuracy (34.25 vs. LLEOT's 28.17). This is the single clear case where LLEOT loses on accuracy to a baseline, but the discussion in Section 5.2 does not mention it.

**Ablation (Table 2) inconsistency.** Row 1 (CPKD+LLE, DR=0.5) reports CPL=46.52%, but row 3 (CPKD alone, no LLE, DR=0.5) achieves higher accuracy (35.40 vs. 34.20). The authors acknowledge this but frame it as "optional" rather than as a sign that LLE can marginally hurt accuracy when compression is heavy.

---

### Theoretical Proof (Appendix D)

The proof is presented cleanly but conflates the idealized condition (Eq. 6) with the approximate training objective (Eq. 7). The gradient identity in Eq. 19 is mathematically correct given Eq. 6—since H is a constant scalar not depending on P, ∇_P H = 0 is immediate. This is almost tautological: a constant offset does not affect gradients by definition. The "theorem" therefore essentially re-states the definitional consequence of Eq. 6 rather than proving that the training procedure achieves meaningful gradient alignment in practice. The paper would benefit from an empirical gradient cosine similarity analysis between emulator and original model gradients on held-out prompts and data.

---

### Limitations & Broader Impact (Section 7)

The ethics statement focuses entirely on positive motivations and does not discuss any failure modes or limitations. Missing:

- **Soft-prompt-only scope**: LoRA and other adapters are not addressed.
- **Small-model scope**: Applicability to GPT-4-class models is not demonstrated.
- **Adversarial robustness of CPL**: The metric can be defeated by an adversary using few-shot, chain-of-thought, or fine-tuning on emulator outputs.
- **LLE can be circumvented**: A motivated adversary might recover model knowledge by training a student on emulator input-output pairs (knowledge distillation from emulator), which doesn't require good zero-shot performance. The LLE defense does not address this.
- **Computational cost of emulator construction**: No wall-clock times are reported for CPKD or LLE stages.

---

### Overall Assessment

LLEOT addresses a legitimate and underexplored problem in privacy-preserving LLM adaptation: that prior offsite-tuning emulators inadvertently leak model capability. The core insight—that a constant loss offset preserves gradients while degrading inference—is elegant and, in principle, sound. However, the paper's theoretical claims substantially outrun its theoretical guarantees. Theorem 1 is valid only assuming exact satisfaction of the idealized LLE condition (Eq. 6), which the actual training objective (Eq. 7) can only approximately achieve; moreover, gradient alignment on the elevation dataset does not extend to downstream task data, so the "same optimal prompt" convergence claim is unsubstantiated. Experimentally, the evaluation is confined to small models and homogeneous multiple-choice tasks, standard deviations are omitted, and the only adapter type tested is soft prompts despite claims of generality. The anomalous CPL > 100% values in Table 1 are unexplained and raise metric validity concerns. For ICLR, this paper needs: (i) a careful recalibration of theoretical claims to match what is actually proven vs. assumed, (ii) an empirical gradient alignment analysis on downstream data, (iii) experiments with LoRA adapters and/or larger models, and (iv) standard deviations with statistical testing. The contribution is interesting enough to merit serious consideration after significant revision, but in its current form the theoretical overclaiming and experimental scope constitute material weaknesses.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses an overlooked privacy risk in Offsite Tuning (OT): distributed emulators retain substantial inference capability, enabling unauthorized use or knowledge extraction by data owners. The authors propose LLEOT, which introduces Loss Landscape Elevation (LLE) to add a fixed margin to the emulator's loss, theoretically guaranteeing exponentially higher perplexity while preserving identical gradients for adapter optimization. Combined with Collaborative Prompt Knowledge Distillation (CPKD), the framework is evaluated on three LLMs and four QA benchmarks, demonstrating strong adapter transfer performance and significantly reduced capability privacy leakage compared to OT and CRaSh baselines.

### Strengths
1. **Well-Motivated Problem & Clear Threat Model:** The paper correctly identifies that existing OT methods prioritize adapter utility over model capability privacy, leaving IP risks unaddressed. The proposed Capability Privacy Leakage (CPL) metric (Eq. 1) offers a direct, interpretable way to quantify this specific threat.
2. **Elegant Core Mechanism with Direct Guarantees:** The LLE objective $L_E(P; x) = L_M(P; x) + H$ is conceptually simple but highly effective. It provably yields $\nabla_P L_E = \nabla_P L_M$ (Eq. 19), ensuring identical optimization trajectories for soft prompts, while exponentially inflating emulator perplexity $PPL_E = e^H \cdot PPL_M$ (Theorem 1). This cleanly decouples utility from inference capability.
3. **Strong Empirical Results & Parameter Efficiency:** Table 1 shows consistent accuracy gains and CPL reductions across Qwen2-1.5B, Gemma-2-2b, and Llama-3.2-3B. Notably, LLEOT uses soft prompts that are ~4 orders of magnitude smaller than adapters in OT/CRaSh (Table 7), drastically lowering communication and compute overhead for the data owner.
4. **High Reproducibility & Robust Ablations:** Algorithm 1 clearly separates roles and phases. Appendix A.4 provides precise hyperparameters, dataset splits, and training regimes. Systematic ablations (Table 2, 3, Figure 4) validate the necessity of CPKD, LLE, and all three distillation losses, while Appendix B.2 demonstrates practical compatibility with gradient randomization for data privacy.

### Weaknesses
1. **Limited Evaluation Scope & Generalization Gaps:** Experiments focus exclusively on multiple-choice QA datasets. Real-world privacy-sensitive domains (e.g., healthcare, finance) typically require open-ended generation, instruction following, or structured reasoning. The framework's effectiveness on generative metrics (e.g., BLEU, ROUGE, LLM-as-a-judge, or open QA) remains unverified.
2. **Minimal Theoretical Novelty & Missing Geometric Limits:** Theorem 1 is a direct application of cross-entropy scaling and gradient linearity ($\nabla(f+H)=\nabla f$). While mathematically correct, framing it as a major theoretical guarantee overstates its novelty. Additionally, gradient preservation assumes the emulator's loss landscape geometry is sufficiently aligned *before* LLE. The paper lacks analysis of failure modes under higher compression rates (e.g., dropout $\geq 0.7$) or when CPKD is imperfect.
3. **Unquantified Model-Owner Overhead:** Constructing the emulator via CPKD and LLE requires forward/backward passes through the original model on distillation and elevation datasets. The paper does not report the compute cost, memory footprint, or training time required from the model owner's side, making deployment feasibility unclear compared to standard OT or layer-pruning baselines.
4. **CPL Metric & Heuristic Prompt Design Limitations:** CPL is based on zero-shot accuracy, which is a coarse proxy for capability leakage; an emulator with near-zero accuracy may still retain dangerous capabilities (e.g., factual extraction, instruction compliance, or jailbreak susceptibility). Furthermore, CPKD relies on proxy prompts sampled from $\mathcal{N}(\mu, \sigma^2)$, but sensitivity to $\sigma$ and $\mu$ during the *construction* phase is unreported (Figure 5 only evaluates post-hoc noise on the returned prompt).

### Novelty & Significance
The novelty is moderate-to-high in framing and application, though the mathematical core is elementary. Identifying capability privacy as a distinct threat vector in offsite tuning is a timely and valuable contribution. The loss-shifting mechanism is a clever, non-heuristic way to break inference utility while preserving optimization geometry. Significance is solid for ICLR: the work addresses a critical barrier to commercial and privacy-sensitive LLM deployment, provides verifiable privacy-utility trade-offs, and aligns with the community's push toward efficient, secure fine-tuning. It meets the acceptance bar for clear methodology and reproducible results, though broader empirical validation and deeper analysis of geometric alignment would strengthen it to a strong-accept level.

### Suggestions for Improvement
1. **Expand to Generative & Open-Ended Tasks:** Include instruction-tuning benchmarks (e.g., AlpacaEval 2.0, GSM8K, or open-domain QA) and report generative quality metrics. This would validate claims about real-world applicability beyond discriminative selection tasks.
2. **Quantify Construction Overhead for Model Owners:** Add a computational breakdown (FLOPs, GPU hours, or epoch counts) for the CPKD and LLE phases. Compare this cost directly to OT's distillation phase to clarify the practical trade-off.
3. **Analyze Compression Limits & Geometric Alignment:** Conduct a sweep over higher LayerDrop rates (0.6–0.8) or alternative pruning strategies to identify where CPKD fails to align gradients sufficiently, leading to LLEOT transfer degradation. Visualizing landscape similarity (e.g., via CKA or gradient cosine similarity) would strengthen the geometric claims.
4. **Adopt a Multi-Faceted Privacy Evaluation:** Supplement CPL with stronger leakage probes, such as a subset of MMLU for broad capability retention, membership inference resistance, or simple extraction/jailbreak prompts on the emulator, to ensure "privacy" reflects true capability suppression rather than mere accuracy collapse.
5. **Report Proxy Prompt Sensitivity:** Provide an ablation on $\sigma$ and $\mu$ for the proxy prompt distribution $\mathcal{N}(\mu, \sigma^2)$ used in CPKD. Clarify how this choice impacts the alignment of continuous representation spaces and subsequent prompt transfer quality.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Adversarial Fine-tuning on Emulator:** Test if a malicious data owner can recover performance by fine-tuning the emulator's weights (e.g., LoRA) rather than just prompts. If the emulator can be revived via weight updates, the core privacy claim is invalidated.
2. **Empirical Gradient Alignment:** Measure the cosine similarity between $\nabla_P L_E$ and $\nabla_P L_M$ during actual downstream tuning steps. Theorem 1 claims identical gradients, but optimization noise on $X_e$ may cause divergence on $D_p$ that must be quantified.
3. **Scaling to Larger Models:** Evaluate on models ≥7B parameters (e.g., Llama-3-8B), as 1.5B–3B models are insufficient to demonstrate viability for modern ICLR standards. Privacy-utility trade-offs often shift significantly at larger scales.
4. **Few-Shot Capability Privacy Leakage:** Report CPL metrics under few-shot prompting conditions on the emulator, not just zero-shot. Adversaries will use few-shot cues to extract capability, so zero-shot CPL alone underestimates privacy risk.

### Deeper Analysis Needed (top 3-5 only)
1. **Loss Margin Generalization:** Analyze whether the enforced loss margin $H$ holds on downstream data $D_p$, given it was optimized on $X_e$. If the margin drifts on unseen data, the gradient alignment guarantee theoretically collapses.
2. **Security Against Logit Calibration:** Discuss whether an adversary can calibrate the emulator's logits (e.g., subtracting a constant offset) to recover original model probabilities. Adding a constant to loss implies a multiplicative factor on probabilities, which may be reversible.
3. **Convergence Stability:** Analyze whether the elevated loss landscape affects the convergence speed or stability of prompt tuning compared to standard offsite tuning. High loss margins could introduce optimization instability not captured by final accuracy.
4. **CPL Metric Validity:** Justify why CPL based on accuracy is a sufficient privacy metric compared to parameter extraction or distillation attacks.Accuracy alone does not guarantee that model weights or internal representations are protected.

### Visualizations & Case Studies
1. **Gradient Similarity Trajectory:** Plot the cosine similarity of gradients between emulator and original model over downstream training steps. This directly exposes whether the core mechanism (Theorem 1) holds in practice.
2. **Loss Margin Drift:** Visualize the difference $L_E(P) - L_M(P)$ over the course of downstream optimization to show if the fixed margin $H$ is maintained or deteriorates.
3. **Qualitative Output Comparison:** Provide side-by-side examples where the emulator produces gibberish (privacy preserved) while the original model produces correct answers using the transferred prompt (utility preserved). Aggregate tables hide whether the method actually works on specific instances.

### Obvious Next Steps
1. **Support for Weight-Based Adapters:** Extend the method to LoRA or full fine-tuning adapters, as soft prompts are less common in production than weight-based adapters. The current restriction limits practical impact.
2. **Formal Privacy Bounds:** Replace the heuristic CPL metric with a formal privacy bound (e.g., differential privacy or information theoretic bound) to meet rigorous security standards.
3. **Communication Overhead Analysis:** Compare total communication costs against standard Offsite Tuning and Federated Learning baselines. Privacy gains are irrelevant if the communication cost becomes prohibitive.

# Final Consolidated Review
## Summary

The paper identifies an overlooked privacy risk in Offsite Tuning: existing emulators retain significant inference capability, enabling data owners to extract proprietary knowledge. The authors propose LLEOT, which uses Loss Landscape Elevation (LLE) to add a fixed margin to the emulator's loss—provably degrading inference while preserving gradients for soft prompt optimization—and Collaborative Prompt Knowledge Distillation (CPKD) to align representation spaces between emulator and original model.

## Strengths

- **Clear problem identification and threat model:** The paper correctly identifies that prior OT methods prioritize adapter transfer while overlooking model capability privacy. The Capability Privacy Leakage (CPL) metric provides a direct, interpretable quantification of this risk.
- **Elegant core mechanism:** The insight that a constant loss offset $L_E(P; x) = L_M(P; x) + H$ preserves gradients ($\nabla_P L_E = \nabla_P L_M$) while exponentially amplifying perplexity ($\text{PPL}_E = e^H \cdot \text{PPL}_M$) is conceptually clean and theoretically grounded. The mathematical derivation in Appendix D correctly establishes these properties under the idealized condition.
- **Strong empirical performance with parameter efficiency:** Table 1 demonstrates consistent accuracy improvements and CPL reductions across three LLM families (Qwen2-1.5B, Gemma-2-2b, Llama-3.2-3B) and four QA benchmarks. Notably, the soft prompt adapters require ~4 orders of magnitude fewer parameters than the LoRA-style adapters used by OT/CRaSh baselines (Table 7: 7.6K vs. 187.2M for Qwen2), substantially reducing communication overhead.
- **Reproducible methodology:** Algorithm 1 clearly delineates the three phases, and Appendix A.4 provides hyperparameters, dataset details, and implementation specifics. The ablations in Tables 2-3 and Figure 4 systematically validate the necessity of CPKD, LLE, and each loss component.

## Weaknesses

- **Theoretical claims exceed actual guarantees:** Theorem 1 establishes gradient preservation and perplexity amplification under the exact condition $L_E(P; x) = L_M(P; x) + H \ \forall P, x$ (Eq. 6). However, the training objective (Eq. 7) only approximately enforces this via sampled proxy prompts $P' \sim \mathcal{N}(\mu, \sigma^2)$ on the elevation dataset $X_e$. The paper does not empirically verify that gradient alignment holds on downstream data $D_p$, whose distribution differs from $X_e$ and whose prompts are initialized randomly and optimized task-specifically. This gap between the idealized condition and practical optimization is not analyzed.
- **Scope limited to soft prompts despite claims of generality:** The introduction states LLE is "applicable to various types of adapters" and Theorem 1 is framed generally, yet all experiments use soft prompts only. Extending the constant-margin mechanism to weight-based adapters (e.g., LoRA) is non-trivial since the gradient relationship would differ. The practical impact is thus narrower than claimed.
- **Missing statistical uncertainty:** The paper states results are "averaged over three experimental runs" (Section 5.2) but reports no standard deviations. Several accuracy differences between methods are within 1% (e.g., Table 1, Qwen2 at DR=0.2: Ours=33.87 vs. OT=33.80 on OBQA), making significance unclear.
- **CPL > 100% anomalies unexplained:** Table 1 shows CPL values exceeding 100% (e.g., OT on WebQs for Qwen2 at DR=0.2: 220.95%), meaning the emulator outperforms the original model on zero-shot accuracy. The paper does not discuss this counterintuitive result or its implications for metric validity.
- **Limited task diversity:** All four benchmarks (OBQA, SIQA, ARC-c, WebQs) are multiple-choice QA tasks. The framework's effectiveness on generative tasks (summarization, open-ended QA), instruction-following, or multi-step reasoning remains unverified.
- **No empirical gradient alignment analysis:** Given that the theoretical guarantee rests on an idealized condition, an empirical study of gradient cosine similarity between $\nabla_P L_E$ and $\nabla_P L_M$ during downstream training would substantially strengthen the claims. This experiment is absent.
- **Proxy prompt distribution sensitivity unanalyzed:** CPKD samples proxy prompts from $\mathcal{N}(0, \sigma^2)$ with $\sigma=20$ "determined experimentally." If downstream prompts at convergence lie outside this distribution's support, neither CPKD alignment nor LLE constraints would apply. No coverage analysis is provided.
- **Computational overhead unreported:** Emulator construction via CPKD and LLE requires forward/backward passes through the original model on distillation and elevation datasets. Wall-clock time, FLOPs, or memory footprint for this one-time cost are not reported.

## Nice-to-Haves

- Evaluate on generative tasks (e.g., summarization, open-ended QA) to demonstrate broader applicability beyond multiple-choice formats.
- Extend to weight-based adapters (LoRA) to substantiate the claimed general applicability.
- Analyze gradient cosine similarity between emulator and original model during downstream tuning to empirically validate the theoretical claim.
- Report standard deviations and conduct statistical significance tests across experimental runs.
- Discuss adversarial robustness of CPL—whether few-shot prompting or model distillation from emulator outputs can circumvent the privacy protection.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Theorem 1 is trivial/low novelty"**: While mathematically straightforward (constant offsets preserve gradients by definition), the insight is valuable and correctly applied. Mathematical simplicity does not diminish practical contribution.
- **"Models too small (1.5B-3B), need ≥7B evaluation"**: The model scales tested are reasonable for ICLR; many accepted papers use similar sizes. Scaling studies are a nice-to-have, not a core flaw.
- **"Missing comparison to FedBiOT and FedPEAT"**: These are mentioned as related work. While additional baselines would strengthen the paper, the comparison to OT and CRaSh (the core OT methods) is sufficient for evaluating the main contribution.
- **"Ethics statement missing discussion of failure modes"**: The ethics statement appropriately addresses the motivation and intended use. Detailed failure mode analysis belongs in the limitations section, not ethics.
- **"CPL metric lacks formal privacy bounds"**: CPL is a practical metric for this specific threat model. Formal differential privacy bounds are a different approach entirely and outside the paper's stated scope.

## Novel Insights

The core insight—that loss landscape elevation cleanly separates adapter utility from emulator capability—represents a principled alternative to heuristic capability destruction methods. The paper reveals an underexplored tension in privacy-preserving ML: protecting IP from authorized-but-untrusted parties (data owners) differs fundamentally from protecting against external adversaries. The CPL > 100% anomalies, if investigated, might reveal interesting interactions between compression, distillation, and task distributions that could inform future emulator design. Additionally, the CPKD approach to aligning continuous prompt spaces is a genuine contribution that could generalize beyond this specific application.

## Suggestions

- **Empirical gradient analysis:** Add a figure showing cosine similarity between $\nabla_P L_E$ and $\nabla_P L_M$ across downstream training iterations to validate that the theoretical preservation holds in practice.
- **Explain CPL > 100% cases:** Investigate and discuss why emulators sometimes outperform original models on zero-shot tasks, and whether this affects metric interpretation.
- **Add convergence analysis:** Report whether elevated loss landscapes affect prompt tuning convergence speed or stability compared to standard OT.
- **Report computational cost:** Add wall-clock time and memory requirements for CPKD and LLE phases to clarify practical deployment overhead.
- **Analyze loss margin drift:** Visualize $L_E(P) - L_M(P)$ during downstream optimization to show whether the fixed margin $H$ is maintained or deteriorates on unseen data.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
