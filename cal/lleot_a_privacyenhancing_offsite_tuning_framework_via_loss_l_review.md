=== CALIBRATION EXAMPLE 25 ===

# Final Consolidated Review
## Summary

LLEOT addresses an overlooked privacy risk in Offsite Tuning: existing emulators retain substantial inference capability, enabling malicious data owners to extract proprietary knowledge. The core mechanism, Loss Landscape Elevation (LLE), enforces a fixed loss margin between emulator and original model, which Theorem 1 shows simultaneously degrades emulator inference (via perplexity amplification) and preserves gradient alignment for prompt transfer. Combined with Collaborative Prompt Knowledge Distillation (CPKD) for soft-prompt-specific alignment, the framework achieves better capability privacy and often better transfer accuracy than prior offsite tuning methods.

## Strengths

- **Identification of capability privacy leakage as a distinct threat.** The paper makes a clean conceptual distinction between parameter privacy (what OT protects) and capability privacy (what it leaks). This reframing is non-trivial: it reveals that even when model weights are obscured, the emulator's retained inference ability constitutes a real IP risk. The CPL metric, while imperfect (see Weaknesses), gives the community a concrete way to quantify this.
- **Elegant core mechanism with formal grounding.** The LLE idea is simple in the best sense: adding a constant loss margin H preserves ∇_P_ L_E = ∇_P_ L_M by construction, while exponentially amplifying perplexity by e^H. This yields a genuine privacy–utility decoupling that is rare in privacy-preserving ML, where improvements in one dimension typically cost the other.
- **Empirical Pareto improvement is striking.** Across all three model families (Qwen2, Gemma-2, Llama-3.2) and both dropout rates, LLEOT achieves strictly better CPL than OT and CRaSh while also matching or exceeding their transfer accuracy. This is not a marginal trade-off; the method dominates on both axes simultaneously, which is unusual and practically compelling.

## Weaknesses

### Major:

- **No evaluation against actual privacy attacks.** The paper claims the emulator's inference capability is "disabled," but this is only measured via zero-shot accuracy on benchmarks. There is no test of whether a malicious data owner can actually extract useful knowledge from the emulator—e.g., via model extraction, further fine-tuning the emulator for downstream use, or knowledge distillation into a third model. Without an adversarial evaluation, the core privacy guarantee remains asserted rather than demonstrated. A 54× perplexity increase may degrade zero-shot QA, but an attacker with a downstream fine-tuning budget may still recover substantial capability. This is the single most important gap.

- **Theorem 1 assumes the fixed-margin constraint is satisfiable, but achievability is unverified.** The gradient-equality result ∇_P_ L_E = ∇_P_ L_M follows trivially from L_E = L_M + H. The substantive question is whether a structurally reduced emulator (after LayerDrop) can actually minimize |L_E(P';x) − L_M(P';x) − H| ≈ 0 across the input and prompt space. If the emulator lacks the capacity to match the *shape* of the original model's loss landscape—merely matching the scalar offset via output bias manipulation—the gradient alignment guarantee evaporates. The paper provides no empirical verification of gradient alignment (e.g., cosine similarity of ∇_P_ L_E vs. ∇_P_ L_M during training) and no analysis of when or how often the margin constraint is violated.

- **CPL is a narrow proxy for capability leakage.** CPL measures the ratio of zero-shot accuracy on a specific benchmark suite. A model could have negligible ZS accuracy on OBQA/ARC-c while still generating fluent, useful text or being fine-tunable for other tasks. The paper's claim that the emulator's inference is "disabled" rests entirely on this proxy. The Random baseline's CPL varying from 0% (WebQs) to 74% (SIQA, Llama3.2-3B) further demonstrates that CPL is highly task-dependent and unreliable as a universal privacy certificate.

### Minor:

- **Counter-intuitive accuracy improvement lacks mechanistic explanation.** LLEOT often achieves *higher* transfer accuracy than OT (e.g., Qwen2 DR=0.5 OBQA: 34.20 vs. 27.20). Standard distillation theory predicts that a more capable teacher should yield a better student. The paper attributes this to "geometric consistency" but provides no analysis of whether LLE acts as a regularizer preventing overfitting to emulator-specific quirks, or whether some other mechanism is at play. Understanding this would strengthen both the theory and the practical guidance.

- **Evaluation is restricted to multiple-choice QA tasks.** All four benchmarks (OBQA, SIQA, ARC-c, WebQs) are classification-style QA. LLM adaptation increasingly targets generation, instruction-following, and reasoning. Without evaluation on open-ended tasks, it is unclear whether the gradient alignment preserved by LLE generalizes to loss functions beyond cross-entropy on discrete labels.

- **Scalability to production-scale models is untested.** Experiments use 1.5B–3B models. The LLE stage (Algorithm 1, line 12) requires computing L_M(P';x) on the full original model for every elevation batch, effectively doubling the computational cost of emulator construction. Whether this overhead remains tractable at 70B+ scale—and whether the margin constraint remains learnable with deeper LayerDrop—is unknown.

- **No empirical gradient alignment verification.** Theorem 1's key claim is that ∇_P_ L_E = ∇_P_ L_M. A simple plot of cosine similarity between these gradients over training steps would directly validate whether the theoretical guarantee holds in practice, or whether approximation errors from finite-data optimization and structural capacity constraints degrade alignment.

### Trivial:

- **Gaussian proxy prompt distribution lacks theoretical justification.** CPKD samples P' ~ N(μ, σ²) with σ=20 chosen empirically. Whether this covers the relevant region of the soft-prompt space is unexamined, though Table 3 confirms the component's empirical contribution.

- **H=4 used in main experiments despite H=2 appearing sufficient.** Figure 4 shows CPL plateaus after H≈2. The choice of H=4 is not explained, though it does not harm accuracy.

## Nice-to-Haves

- **Evaluation with LoRA or other PEFT adapters.** The paper claims LLE is applicable to "various types of adapters" but only validates soft prompts. Even a single experiment with LoRA would substantiate this claim and broaden impact significantly.
- **Adversarial attack experiments.** Test whether the elevated emulator can be used for model extraction or further fine-tuning, going beyond the ZS-accuracy proxy.
- **Generation task evaluation.** Include at least one open-ended generation benchmark (e.g., summarization or instruction-following) to demonstrate generalizability beyond classification.
- **Wall-clock time and compute cost reporting** for CPKD + LLE stages vs. standard OT distillation.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Missing comparison with DP-SGD or other privacy-preserving methods.** DP-SGD protects training data privacy, not model capability privacy. LLEOT operates in a fundamentally different threat model (protecting the emulator's inference capability). This is scope creep.
- **Weakness: Missing recent baselines (post-2024).** Without external knowledge to confirm the existence or relevance of specific newer methods, this cannot be verified and is removed per hard rules.
- **Weakness: Lack of formal convergence analysis.** The paper is primarily empirical and the optimization follows standard gradient descent. Demanding a full convergence proof for what is essentially an engineering contribution with a supporting property theorem is not standard for this type of work.
- **Weakness: Paper is "well-written" / "topic is important."** Generic strength; removed per soft rules.
- **Weakness: Loss weight sensitivity (w1, w2, w3).** This is a trivial hyperparameter concern; ablation in Table 3 already shows component necessity.

## Novel Insights

The most striking empirical finding—that elevating the emulator's loss *improves* transfer accuracy rather than just preserving it—suggests LLE may function as an implicit regularizer. By preventing the emulator from memorizing idiosyncratic low-loss regions, LLE may force prompt optimization to rely on landscape geometry shared with the original model, yielding more robust prompts. This regularization hypothesis, if confirmed, would reframe the privacy–utility relationship in offsite tuning: rather than trading accuracy for privacy, the privacy mechanism itself may be accuracy-enhancing. This deserves explicit investigation.

## Suggestions

- **Run at least one adversarial evaluation:** fine-tune the LLEOT emulator on a downstream task and compare its ceiling performance to the original model. This directly tests whether capability leakage is genuinely mitigated.
- **Plot gradient cosine similarity** between ∇_P_ L_E and ∇_P_ L_M over training steps. This is cheap to compute and would either validate Theorem 1 in practice or reveal approximation failures worth discussing.
- **Add a brief paragraph explaining the accuracy improvement.** Even a hypothesis (e.g., "LLE prevents overfitting to emulator-specific local minima") with supporting evidence (e.g., loss curves or prompt norm analysis) would address the most puzzling empirical result.

## Axis Evaluation

- **Novelty:** Moderate. The capability privacy framing and CPL metric are novel contributions; the LLE mechanism itself relies on a straightforward optimization property, but its application to this problem is well-motivated and non-obvious.
- **Technical soundness:** Acceptable with caveats. The theoretical guarantee is correct but conditional on an unverified achievability assumption; the empirical evaluation lacks adversarial validation of the core privacy claim.
- **Empirical support:** Good on transfer accuracy; weak on privacy verification. The Pareto improvement is well-demonstrated, but privacy claims rest entirely on a narrow proxy metric.
- **Significance:** High if the privacy claims hold. The problem is practically important and the solution is elegant; however, without attack evaluation, the practical impact of the privacy guarantee remains uncertain.
- **Clarity:** Good. The method is clearly described and the three-stage pipeline is easy to follow. The distinction between the conditional nature of Theorem 1 and its practical applicability could be sharper.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
