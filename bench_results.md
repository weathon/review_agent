# ICLR Benchmark Results

Date: 2026-04-13 15:31
Critic/Merger: claude:claude-sonnet-4-6 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## C0Boqhem9u

- GT: Reject (avg 4.4)
- Predicted: Reject (4.8/10)
- Match: YES

### Final Review

Now I have sufficient information from the paper to write the consolidated review.

---

## Summary

LinBridge is a framework for interpreting nonlinear neural encoding models by decomposing their Jacobian matrices into a "linear inherent component" (JM_inherent, a sample-invariant approximation of the mapping) and a sample-selective "nonlinear mapping bias" (ΔJM). A CNN compresses the N-sample dimension of the stacked Jacobian tensor to extract JM_inherent; a contrastive InfoNCE loss encourages JM_inherent to be similar to each per-sample Jacobian and dissimilar to ΔJM. The framework is applied to visual neural encoding on the Natural Scenes Dataset using CLIP-ViT features, and the resulting ΔJM is used to construct an "AFD" metric characterizing hierarchical nonlinearity across PVC, SVC, and TVC.

---

## Strengths

- **Novel problem framing:** Applying contrastive self-supervised learning specifically to decompose a stack of Jacobian matrices into shared (linear) and residual (nonlinear) components within a neural encoding context is a genuinely novel combination that goes beyond simply averaging per-sample Jacobians or applying standard interpretability tools to this domain.

- **Stability analysis across batch sizes (Figure 4a):** The paper demonstrates that JM_inherent is highly stable across a range of batch sizes (Pearson r → 1), which non-trivially shows that the contrastive learning procedure converges to a consistent structure and is not just fitting noise. This is a concrete, domain-specific validation not typically seen in similar works.

- **Consistent multi-area neuroscience results:** The finding that AFD increases monotonically from PVC → SVC → TVC, and that nonlinearity extends into TPOJ and prefrontal regions, is replicated across four subjects (main text + appendix) on a rigorous 7T dataset with FDR-corrected significance thresholds. The use of NSD with CLIP-ViT is consistent with the strongest current neuro-AI benchmarks.

- **Beyond-visual-cortex discovery:** The incidental observation that TPOJ and prefrontal areas also exhibit elevated nonlinearity, interpreted in terms of multimodal integration and higher cognition, is a scientifically interesting secondary finding that goes beyond what most encoding papers report.

---

## Weaknesses

- **Critical explanatory gap: how is R² derived from JM_inherent?** The paper's central empirical claim is that JM_inherent achieves R² distributions nearly identical to the nonlinear model (Pearson r > 0.99, Figure 4c). However, JM_inherent ∈ ℝ^{d×p} is a Jacobian matrix — not a direct predictor. The paper never states how it is converted into voxel predictions for computing R² (e.g., ŷ_k = JM_inherent^T x_k + bias?). This is not a trivial point: the choice of bias and the specific linear prediction rule determine whether the comparison is meaningful. Without this explanation, the key validation result (§4.2) cannot be reproduced or properly evaluated. **This is the most critical gap in the paper.**

- **Absence of the obvious baseline — mean Jacobian:** The simplest possible "inherent component" is the sample-mean Jacobian J̄ = (1/N)Σ_k JM_k. If J̄ achieves similarly high correlation with the nonlinear model's R², the elaborate CNN + contrastive learning machinery is unnecessary. This baseline is never computed, and without it, there is no evidence that LinBridge adds anything over straightforward averaging. This single missing ablation calls into question the paper's primary methodological contribution.

- **Validating encoding model is too simple:** The nonlinear model validated is a two-FC-layer MLP with a single ReLU nonlinearity (§3.2). The Jacobian of such a model is a piecewise-constant binary-masked weight matrix — analytically tractable and architecturally trivial. The paper explicitly acknowledges this limitation, but the gap remains serious: it is unclear whether LinBridge's contrastive extraction would be necessary, meaningful, or computationally tractable for architecturally complex models (deeper MLPs, CNNs, transformers) which are the stated motivation of the work. The paper's own results may not generalize beyond this toy setting.

- **CNN architecture for Jacobian aggregation is unmotivated and undescribed:** The CNN that compresses the sample dimension N → 1 is the most novel architectural element of the paper, yet its design is not described in the main text (number of layers, kernel sizes, etc.) and its choice is not motivated. Crucially, test-set samples are ordered arbitrarily along the N-axis — there is no spatial locality in this dimension — yet convolutional kernels exploit local structure. A permutation-invariant architecture (attention, DeepSets, or even a mean-pool) would be a more principled choice, and this design decision is never justified.

- **Co-dependent optimization in the contrastive loss:** The negative samples ΔJM = JM − JM_inherent are computed from the model's own output during training. As JM_inherent changes, the negatives change too. If JM_inherent → JM, then ΔJM → 0 and the negatives collapse, making the loss trivially maximized via a degenerate solution. The paper does not analyze whether this degeneracy is ruled out in practice (the L1 on ΔJM discourages collapse but does not prevent it), nor does it provide any theoretical or empirical analysis of the loss landscape. The stability result in Figure 4(a) partially addresses convergence behavior, but does not rule out degenerate attractors.

- **AFD metric depends on stochastic, non-unique t-SNE ordering:** The AFD is computed by sorting images by their 1-D t-SNE embedding of CLIP-ViT features, then fitting a linear slope to each voxel's ΔJM values along this sorted axis. t-SNE is stochastic: the 1-D projection is non-unique and random-seed-dependent. The slope magnitude (AFD) will vary across t-SNE runs. No sensitivity analysis or seed averaging is performed.

- **Voxel SNR confound for hierarchical nonlinearity claim:** Higher AFD values in TVC relative to PVC may partly reflect lower signal-to-noise ratio in higher visual areas (fewer reliably driven voxels) rather than true representational nonlinearity. The paper does not control for voxel-level R² or noise ceiling when reporting AFD distributions, which could confound the neuroscientific interpretation.

- **No ablation studies:** There are no ablations on: (a) CNN vs. mean Jacobian vs. PCA of Jacobians; (b) contrastive loss vs. MSE reconstruction of the mean; (c) L1 regularization strength λ; (d) temperature τ; (e) dimensionality of the low-dimensional projection. For a method paper at ICLR, this absence makes it impossible to understand which components drive the results.

- **Main text generalization is restricted to one subject:** While other subjects' results are in the appendix, the main text exclusively exemplifies Subject 2. For neuroscience claims about visual cortex organization, the main text should include at least a summary of results across all four subjects.

- **The "novel evidence" framing is overclaimed:** The finding that PVC is more linear than TVC is well-established (Güçlü & van Gerven, 2015, cited in the paper). This should be framed as *corroboration and quantification* via a new tool rather than "novel evidence."

---

## Nice-to-Haves

- **Visualization of JM_inherent feature weights:** Inverting or projecting JM_inherent into image space (e.g., via gradient visualization) would concretely demonstrate what "linear inherent component" looks like as an interpretable feature — supporting the interpretability claim beyond R² matching.

- **Show high-ΔJM stimuli:** Displaying specific images that trigger high vs. low ΔJM values would validate the intuition that the mapping bias captures complex or semantically distinctive stimuli, as opposed to noise.

- **Runtime and memory profiling:** A brief table of wall-clock time and GPU memory for Jacobian computation vs. linear encoding training would help practitioners evaluate LinBridge's scalability without requiring readers to estimate from architecture parameters alone.

- **Contrastive embedding space visualization:** A plot of the latent space showing JM_i and JM_inherent clustered together and ΔJM_j pushed away would directly confirm that the contrastive objective succeeds.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"CLIP-ViT labeled as LLM" (Harsh Critic):** The Figure 2 caption uses "LLM" as a label for the embedding extraction step. While CLIP-ViT is technically a vision-language model and not a large language model, Figure 1's caption also uses the term loosely. This is a minor terminology imprecision in a figure label, not a substantive methodological error. Removed as a pure formatting/terminology nitpick.

- **"JM_k ∈ ℝ^{d×1×p}: the '1' dimension is unexplained" (Harsh Critic):** The "1" dimension in ℝ^{d×1×p} for a single sample k is clearly the placeholder for the sample axis, which becomes N when stacked into JM ∈ ℝ^{d×N×p}. This is a notation choice for consistency with the stacked tensor, not an error.

- **"Naselaris 2011a and 2011b cite the same paper" (Harsh Critic):** The two citations have slightly different formatted author lists (one uses "N Kay" and one uses "Kay") but appear to be the same publication. This is a reference formatting error, not a substantive issue.

- **"Comparison to LIME, SHAP, integrated gradients" (Harsh Critic):** These are post-hoc explanation methods for classification or regression and are not standard baselines in neural encoding interpretability. Demanding their inclusion imposes scope from outside the paper's established community standards.

- **"Larger model zoo / more datasets beyond NSD" (Generic):** The NSD is the dominant large-scale fMRI dataset in the neuro-AI field. Requesting additional datasets is a generic expansion request rather than a substantive scientific flaw.

- **"Code availability is mandatory for ICLR" (Spark Finder):** While encouraged, code release is not a hard requirement for ICLR papers at submission time, so this is not a valid weakness under current norms.

---

## Novel Insights

The most genuine insight surfacing from the synthesis of these reviews is the **circular optimization concern combined with the missing R² computation explanation**. Together, they point to a deeper conceptual ambiguity at the heart of the paper: it is unclear whether JM_inherent is a principled decomposition of the nonlinear mapping or whether the contrastive learning framework is simply learning to minimize ΔJM (collapsing the nonlinear residual toward zero), such that JM_inherent ≈ JM by construction and the reported R² match is tautological. Resolving this would require (1) explicitly stating how predictions are made from JM_inherent, (2) showing that JM_inherent is meaningfully different from both J̄ (mean Jacobian) and JM itself, and (3) providing a non-circular validation against ground-truth neural tuning properties rather than the nonlinear model's own predictions. This is the key gap the authors should address.

---

## Suggestions

- **Explicitly state the prediction rule for R² from JM_inherent** in §4.2, including how the bias term is handled. This is essential for reproducibility and for validating the central claim.
- **Add the mean Jacobian J̄ as a baseline** throughout all experiments (R² correlation, AFD histograms). This single experiment would either validate or invalidate the contrastive learning contribution.
- **Motivate and describe the CNN architecture:** Replace or supplement it with a permutation-invariant aggregator (e.g., mean-pool or attention over the sample dimension); justify why convolution over an arbitrarily ordered sample axis is appropriate.
- **Analyze the optimization dynamics of the contrastive loss:** Empirically check whether JM_inherent ≈ JM (degenerate) or whether it genuinely differs, and report ‖JM_inherent − J̄‖ / ‖J̄‖ to characterize how much the CNN contributes beyond averaging.
- **Report AFD with multiple t-SNE seeds** (e.g., 10 runs) and show that the PVC < SVC < TVC ordering is stable across seeds.
- **Control for SNR/noise ceiling** when reporting AFD across visual areas, to rule out the confound that TVC's apparent nonlinearity reflects signal unreliability rather than representational complexity.
- **Validate on at least one deeper encoding model** (e.g., a 4-layer MLP or a fine-tuned ResNet head) to support the generalization claims that motivate the paper.

---

**Summary evaluation:**

- *Novelty:* Moderate. The Jacobian-based decomposition framing is new, but the combination of CEBRA-style CNN + InfoNCE is largely borrowed, and the scientific finding (PVC < TVC in nonlinearity) is known.
- *Technical soundness:* Weak. The missing R² computation explanation, the unmotivated use of CNN over the sample dimension, and the unanalyzed co-dependent optimization are material gaps, not cosmetic ones.
- *Empirical support:* Insufficient for the primary claims. The mean-Jacobian ablation is indispensable and missing; the encoding model is too simple; multi-subject summaries are appendix-only.
- *Significance:* Moderate for computational neuroscience if the methodology holds up; limited for the broader ML community.
- *Clarity:* The core CNN component — the most novel element — is the least well-described part of the paper, which is a significant clarity problem for the central contribution.

---

## cojJ2s1e35

- GT: Reject (avg 5.5)
- Predicted: Accept (5.8/10)
- Match: NO

### Final Review

## Summary

WLA (World Modeling through Lie Action) introduces a framework for learning continuous, compositional latent dynamics shared across multiple environments. The core idea is to model environment transitions as actions of a Lie group that acts linearly on a partitioned latent space, realized via an equivariant slot-attention autoencoder. A separate `Ctrl_adapt` module maps user-specified action signals to Lie algebra parameters using minimal labeled data, solving the "Structured Controller Interface Problem" across diverse environments. The paper evaluates on PHYRE (qualitative sanity check), ProcGen (8 environments, compared against Genie), and a 3D Android robot dataset.

---

## Strengths

- **Genuinely novel architectural synthesis for inter-environmental dynamics.** The combination of non-autonomous, time-varying Lie algebra generators (A(t) in Eq. 4) with object-centric slot attention for multi-environment world modeling is a distinctive contribution. Prior work (Koyama et al., 2024; Mitchell et al., 2024) either assumes time-homogeneous dynamics or does not generate future observations. WLA's extension to non-autonomous dynamics addresses a real gap.

- **Strong and consistent quantitative improvements on ProcGen.** WLA outperforms Genie across all 8 environments in PSNR, ΔtPSNR, and LPIPS (Table 2), with margins that are large enough to be practically meaningful (e.g., coinrun: 11.30→22.10 PSNR, ΔtPSNR: 0.48→9.03). The ΔtPSNR metric is particularly informative: Genie's near-zero or negative values on several environments reveal that its action conditioning provides essentially no controllability, while WLA's structured latent transitions yield consistent, meaningful action responses.

- **Significant FVD improvement on the Android (3D) dataset.** WLA reduces FVD from 393.85 to 131.02 (Table 3), a ~66% improvement in long-term video coherence. This demonstrates that the structured Lie dynamics are genuinely beneficial for temporal consistency beyond the 2D game setting.

- **Principled slot alignment via least-action assignment.** The use of a linear assignment problem to enforce temporal slot consistency is an elegant solution to a known failure mode of slot attention in temporal settings (Zhao et al., 2023), and the ablation in Table 1 confirms it contributes meaningfully (MSE 0.675→0.602 for unseen environments).

- **Ablation validates both core design choices.** Table 1 quantitatively confirms that both the rotational Lie group components and the least-action alignment matter, providing evidence that the structured design is not gratuitous—the version without rotation is explicitly noted to degrade to a Mamba-like diagonal state-space model.

---

## Weaknesses

- **The gap between theoretical guarantees and learned behavior is never bridged.** The compositionality and continuity properties in Eq. 3 are only valid when Eq. 2 holds exactly, i.e., when the encoder satisfies exact equivariance: Φ(g·x) = M(g)z(t). The paper invokes Koyama et al.'s existence result to motivate this, but then trains (Φ, Ψ) with a pure prediction loss (Eq. 7–9) that contains no equivariance penalty. There is no empirical verification that the trained encoder satisfies Eq. 2 approximately — no test such as checking whether Φ(g·x) ≈ M(g)Φ(x) holds in practice. As a result, the central theoretical claim that the learned system inherits compositional and continuous dynamics remains unverified. This is the most important gap in the paper.

- **The commutativity assumption is a major limitation buried in the conclusion.** Section 7 acknowledges: "we assume a priori that transitions in the environment commute with each other." This is a structurally significant constraint: the chosen Lie group (block-diagonal scaling+rotation matrices, Eq. 5) is abelian by construction. Non-commutativity of sequential actions is the norm in both 2D games (jump-then-right ≠ right-then-jump) and especially 3D robotics (rotations in SE(3) do not commute). This assumption should be prominently stated in the problem formulation or Section 3, not deferred to the conclusion, as it defines the boundary of the method's applicability and directly affects the interpretation of the 3D results.

- **The most critical ablation — whether inter-environmental training helps — is absent.** The paper's central claim is that jointly training a *single common model* across diverse environments enables better generalization and adaptation. Yet there is no ablation comparing (a) WLA trained on all environments vs. trained on a single environment evaluated on that environment, and (b) multi-environment vs. single-environment training for adaptation to unseen environments. Without this, the paper cannot demonstrate that its "inter-environmental" framing provides value beyond a well-trained single-environment model. This is particularly important because the performance gains shown in Table 2 could in principle come from the Lie/slot architecture alone.

- **The full Genie vs. WLA comparison for unseen environments is missing.** Table 1 reports unseen-environment MSE for the ablation variants, and Table 2 reports seen-environment performance against Genie, but the paper never directly compares WLA vs. Genie on held-out (unseen) environments. The unseen generalization setting is, by the paper's own framing, the strongest test of the inter-environmental claim, yet this comparison is absent.

- **IDM operates independently per slot, ignoring inter-slot interactions.** The IDM (Section 4.4) takes the concatenation of consecutive slot tokens for each slot independently: z_n[t] ⊕ z_n[t+1] → (λ_{n,j}, θ_{n,j}). This means the inferred transition parameters for one slot carry no information about other slots. For dynamics involving inter-object interactions (collisions, occlusions, object pickup), the per-slot IDM cannot capture the correct Lie algebra parameters. This is an architectural limitation that is not discussed.

- **PHYRE results are purely qualitative.** Section 6.1 is explicitly labeled a "sanity check," but even for this role, Figures 3 and 4 show a handful of selected frames with no quantitative evaluation (PSNR, MSE, or any interpolation quality metric). A quantitative sanity check would meaningfully strengthen the claims about continuity and compositionality.

- **The "unsupervised" framing in the abstract partially overstates the case.** The abstract states the framework works "with minimal or no action labels," but Section 4.3 describes Ctrl_adapt as trained on a labeled dataset {(x[t], a[t])}. The distinction between the unsupervised world model backbone and the labeled adaptation module is real and reasonable, but the paper does not quantify what "minimal" labels means — no learning curve is provided showing performance as a function of the number of labeled action sequences.

- **Ctrl_adapt training is subject to IDM error propagation.** The adaptation loss L_adapt (Eq. 10) measures MSE between Ctrl_adapt's output and the ground-truth (λ, θ) inferred by the IDM during pretraining. However, these "ground truth" values are themselves the IDM's estimates, not directly observed quantities. Systematic biases or noise in the IDM will corrupt the Ctrl_adapt supervision signal, but this error propagation is never analyzed or discussed.

---

## Nice-to-Haves

- **Label efficiency curve for Ctrl_adapt.** A plot of adaptation performance vs. number of labeled sequences would substantiate the "minimal labels" claim and provide practitioners with a useful calibration.
- **Inference latency comparison against Genie.** The slot attention + Lie exponentiation + linear assignment pipeline is architecturally heavier than Genie's autoregressive approach. An honest compute comparison would help assess practical applicability.
- **Discussion or sketch of non-abelian extensions.** Even a theoretical discussion of how the framework could be extended to non-commutative Lie groups (e.g., SO(3) or SE(3)) would significantly strengthen the paper's long-term significance for 3D robotics settings.
- **Latent manifold visualization.** Showing that z(t) traces the expected geometric path (e.g., a spiral under combined scaling and rotation) would provide compelling evidence that the Lie group inductive bias is actually active in the learned representation.
- **Downstream planning or control evaluation.** Evaluating whether the WLA world model enables better policy planning (e.g., model-based RL inside the latent space) would demonstrate that the structured dynamics provide benefits beyond visual generation metrics.
- **Slot tracking consistency visualization.** Showing attention masks for individual slots across a long trajectory (including occlusion cases) would validate the least-action principle contribution more directly.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Modified Genie is an unfair comparison"** (Harsh Critic): The modification to Genie adds trainable action label embeddings to make it comparable in the labeled CIP setting. This modification makes Genie *more* capable than the original (it now receives explicit action labels), which means the asymmetry favors the Genie baseline, not WLA. A comparison where the baseline is given extra information and still loses is a stronger result for the proposed method, not a weakness. Removed per the rule on comparisons asymmetric in favor of the baseline.

- **"No comparison with DreamerV3/LAPO"** (Harsh Critic, Spark Finder): These are legitimate related baselines worth noting, but the paper cannot be faulted for not comparing with every related model, especially given that DreamerV3 was designed for RL not video generation and LAPO is already cited and positioned in the related work. Removed per the rule on missing related works and scope; this is at most a nice-to-have.

- **"No downstream RL benchmarks"** (Spark Finder): WLA is explicitly framed as a controller interface / world model for video generation and control, not a full RL agent. Demanding RL experiments for a paper about the Controller Interface Problem is scope creep. Removed per the scope rule.

- **"3D commutativity stress test as a required experiment"** (Spark Finder): Designing a controlled stress test to quantify non-commutative error is beyond what is standard for this type of paper. The commutativity limitation is real but pointing to it as a required experiment (rather than an acknowledged limitation) is not standard practice for a world modeling paper.

- **Notation overloading of g_{t,δ}** (Harsh Critic): The paper explicitly acknowledges this notation simplification and the formally correct notation. It is a style choice, not a scientific error. Removed as a formatting nitpick.

- **"The Fact in Eq. 3 is trivially derived"** (Harsh Critic): That a result follows straightforwardly from definitions is not a weakness; including it as a named Fact helps readers follow the theoretical construction. Removed as style criticism.

- **"Requesting multiple-seed statistics"** (Harsh Critic): Single-run evaluation is standard for large-scale video generation benchmarks. Removed per the community-standards rule.

- **"No reinforcement learning integration"** (Harsh Critic): Explicitly out of scope for a world modeling paper about CIP. Removed.

---

## Novel Insights

The most genuinely novel conceptual move in this paper is the use of non-autonomous Lie algebra generators A(t) to model time-varying dynamics within an equivariant autoencoder framework trained across multiple environments simultaneously. Prior equivariant world models either assume time-homogeneous transitions (collapsing to a static group action) or operate per-environment. WLA's object-centric partitioning of the latent space, where each slot's dynamics are governed by a separate block of the Lie algebra generator, provides a clean compositional structure: the action of one object can literally be added to another (as shown in Figure 4's Phyre compositionality test). The least-action slot alignment — reframing temporal slot tracking as minimizing a Frobenius-norm linear assignment rather than adding auxiliary losses — is a practically elegant solution. Taken together, these design choices suggest a productive direction: structured latent state-space models with algebraic compositionality can outperform autoregressive video models on controllability metrics even when the structured model has no explicit access to the autoregressive model's discrete action tokens, pointing toward the value of geometry-aware inductive biases for world modeling.

---

## Suggestions

1. **Add an equivariance verification experiment.** Report the empirical deviation from Eq. 2 on held-out trajectories, e.g., ||Φ(g·x) - M(g)Φ(x)||_F. Without this, the theoretical motivation for compositionality is disconnected from the learned system. Even a visual demonstration that latent trajectories follow the expected Lie group geometry would substantially close this gap.

2. **Add the inter-environmental ablation.** Train WLA on each ProcGen environment separately and compare unseen-environment adaptation performance to the jointly trained model. This single experiment would either validate or qualify the paper's central "inter-environmental" claim.

3. **Report the Genie vs. WLA comparison on unseen (out-of-play) environments using the same metrics as Table 2.** Table 1 provides the MSE for ablation variants in the unseen setting; extending this to a full comparison against Genie would provide the most important missing result.

4. **Reposition the commutativity assumption earlier in the paper** — ideally in Section 2 or Section 3 — rather than deferring it to the conclusion. A clear statement that the model is designed for environments whose dynamics are well-approximated by abelian group actions would set appropriate expectations and allow readers to evaluate applicability more accurately.

5. **Provide a learning curve for Ctrl_adapt.** Plot adaptation performance (ΔtPSNR or ActionACC) as a function of the number of labeled action frames to substantiate the "minimal labels" claim quantitatively.

6. **Discuss the per-slot IDM limitation explicitly.** Acknowledging that inter-slot interactions are not modeled in the IDM, and discussing how this could be addressed (e.g., a cross-slot attention IDM), would strengthen the paper's intellectual honesty and provide a concrete direction for follow-up work.

---

## 5WPQIVgWCg

- GT: Accept (Poster) (avg 6.7)
- Predicted: Accept (6.1/10)
- Match: YES

### Final Review

## Summary

SELECT is an oracle-based algorithmic template for satisficing regret minimization in bandit optimization. Its central contribution is replacing the *satisficing gap* Δ_S — which is zero in continuous-arm settings like concave and Lipschitz bandits, making all prior algorithms vacuous — with the *exceeding gap* Δ_S\* = r(X\*) − S, which is always positive in the realizable case. By combining oracle arm identification, forced sampling, and LCB-based round termination, SELECT achieves constant (T-independent) satisficing regret in the realizable case and matches the oracle's standard regret in the non-realizable case.

---

## Strengths

- **The Δ_S → Δ_S\* conceptual pivot is the paper's most important contribution.** For concave and Lipschitz bandits, Δ_S = 0 necessarily (Remark 4), meaning prior algorithms such as SAT-UCB (Michel et al., 2023) and Garivier et al. (2019) cannot yield constant satisficing regret in these settings at all. SELECT is the first to obtain constant satisficing regret for infinite-arm continuous settings, making this a meaningful qualitative advance rather than an incremental improvement.

- **Each component of SELECT has a clear, non-trivial mathematical justification.** Remark 2 precisely explains why oracle-based arm identification is necessary over uniform exploration (Step 1), why forced sampling is required before LCB testing — without it, the LCB can drop below S even for a satisficing arm due to insufficient pulls (Step 2) — and why LCB rather than UCB or empirical mean is necessary for avoiding 1/Δ_S dependence in round termination (Step 3). This level of component-wise justification is above average for the genre.

- **Graceful two-regime behavior in Theorem 1.** The bound min(O(C₁^{1/(1−α)} (1/Δ_S\*)^{α/(1−α)} · polylog), C₁T^α · polylog(T)) ensures SELECT automatically adapts to the oracle's performance when T is too small for constant regret to dominate. This is not a trivial design feature.

- **Lower bounds confirm the correct gap dependence.** Theorems 3 and 4 establish Ω(1/Δ) lower bounds for finite-armed and concave settings (with Δ = Δ_S\*), confirming that the inverse-gap dependence in Corollaries 1 and 2 is tight and not an artifact of the analysis.

---

## Weaknesses

- **Proofs reside in an external document, not the submission.** All proofs are deferred to "Appendix A/B of the full version (Feng et al., 2025)" — an external document not included in or linked from the submission. The main text provides only high-level proof sketches for Theorem 1 (via Propositions 1 and 2) and no sketch at all for Theorem 2. This makes independent verification of correctness impossible from the submitted paper alone. For ICLR, the supplementary material should be self-contained.

- **The algorithm requires knowledge of oracle regret parameters (α, β, C₁), with no discussion of robustness to misspecification.** The forcing schedule γ_i = 2^{−i(1−α)/α} is parametrized by the oracle's regret exponent α. While α is known analytically for standard oracles (UCB, Thompson sampling), it may be unavailable or unreliable for complex or learned oracles. The paper makes no mention of robustness to misspecification of α or whether a parameter-free variant is achievable, weakening the "plug-and-play" framing.

- **No lower bound for Lipschitz bandits, and the K factor in finite-armed bandits is unresolved.** Theorems 3–4 provide lower bounds only for finite-armed and concave settings. For Lipschitz bandits in d dimensions, Corollary 3 gives O(L^d / (Δ_S\*/2)^{d+1} · polylog) while no matching lower bound is given; it is unclear whether the (d+1) exponent on 1/Δ_S\* or the L^d factor is tight. Similarly, for finite-armed bandits, the upper bound is O(K/Δ_S\*) but the lower bound (from a 2-arm construction) is Ω(1/Δ_S\*), leaving the K factor unresolved.

- **Algorithm 1 lacks an explicit stopping condition.** The pseudocode takes T as input but contains no check on remaining budget. As written, Algorithm 1 will attempt to initiate a new round even when insufficient time remains to complete the forced sampling phase. A clean budget check (e.g., halt if remaining time < T_i for the next round) is necessary for a complete algorithmic specification.

- **Notation inconsistency between the main text and Algorithm 1.** Step 1 of the algorithm description introduces $\tilde{X}_i$ as the uniformly sampled arm, but Algorithm 1 (line 3 of the pseudocode) uses $\hat{X}_i$ throughout, including in Steps 2 and 3. This inconsistency is never reconciled and creates unnecessary confusion.

- **Non-realizable experiments show unexplained patterns that are not discussed.** In Figures 3b and 4b, SELECT substantially outperforms its own oracle (Convex ALG by ~20%, Uniform UCB by ~75%) in standard regret under non-realizability. Theorem 2 is only an upper bound, so this does not violate theory, but the mechanism is never discussed. More strikingly, in Figure 3b, the discretized SAT-UCB heuristic (~500) and SAT-UCB+ (~600) achieve markedly lower standard regret than SELECT (~800), which the paper does not acknowledge. The discussion simply states SELECT "can also reach good enough distribution," which is insufficient.

---

## Nice-to-Haves

- A regret-vs-Δ_S\* plot (varying S across several values) to empirically validate the 1/Δ_S\* dependence in Theorem 1. The current experiments fix S and vary T, which validates that regret is constant in T but does not verify the gap dependence.
- An ablation removing forced sampling (Step 2) to directly demonstrate Remark 2's claim that LCB tests fail without sufficient initial pulls. This would isolate the forced sampling contribution.
- A brief sensitivity analysis or heuristic for choosing α when the oracle's exact regret exponent is uncertain.
- Lipschitz bandit experiments in d > 2 to probe the curse-of-dimensionality scaling in Corollary 3.
- A direct empirical comparison with Garivier et al. (2019) for finite-armed bandits, since the theoretical relationship is stated as "incomparable" and numerical comparison would clarify practical tradeoffs.
- Discussion of whether SELECT can detect non-realizability (output a flag) during or after execution, which would be valuable in practical applications.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[REMOVED] Criticism that the 1/4 probability bound in Proposition 2 is arbitrary.** This constant emerges from the proof calculation and is internally consistent. Asking for a "tightness" justification for intermediate lemma constants is not a substantive criticism.

- **[REMOVED] Criticism of the LCB constant factor of 4.** The factor arises from standard subgaussian tail bounds and its derivation in the appendix is expected. Demanding it be re-derived in the main text is a formatting nitpick.

- **[REMOVED] Criticism of oracle data discarding at round restart.** The paper is clear that SELECT restarts ALG each round; Remark 2 explains why this design is needed. Whether warm-starting would be more efficient is a question outside the paper's scope and does not affect the correctness of the stated bounds.

- **[REMOVED] Criticism about loose inventory management example.** The subgaussian noise assumption is a standard modeling choice throughout the bandit literature. Questioning its applicability to inventory management is scope creep into applied modeling, not a theoretical flaw.

- **[REMOVED] Demand for confidence intervals/error bars on the 1000-repetition experiments.** Reporting mean curves without error bars is standard practice in the bandit empirical literature, especially when the curves are smooth and repetition count is high.

- **[REMOVED] Framing SELECT outperforming its oracle in non-realizable cases as "contradicting Theorem 2."** Theorem 2 is an upper bound. SELECT being empirically better than its oracle does not contradict an upper bound. This criticism reflects a misreading of the theorem statement.

- **[REMOVED] Criticism that the abstract fails to mention Δ_S\* > 0 as a qualifier.** The abstract says "realizable case (i.e., a satisficing arm exists)" and Δ_S\* > 0 follows from this by definition. The clarification is immediate and the abstract is not misleading.

---

## Novel Insights

The paper's deepest insight — which the spark finder articulates most clearly — is that Δ_S and Δ_S\* measure fundamentally different things that happen to coincide in finite discrete settings but diverge in continuous ones: Δ_S measures the nearest non-satisficing arm's distance to the threshold (which collapses to zero in dense spaces), while Δ_S\* measures the best arm's margin above the threshold (which is preserved in the realizable case regardless of arm density). Prior satisficing algorithms relied on Δ_S because they identified non-satisficing arms via exclusion; SELECT circumvents this by instead identifying *candidate satisficing arms via reward maximization* (Step 1) and testing them statistically (Step 3). The LCB test is key: it terminates a round within 1 step in expectation for a non-satisficing arm (since the LCB will almost certainly be below S immediately), while a satisficing arm with reward above S + Ω(γ_i) will sustain the LCB above S indefinitely once γ_i ≪ Δ_S\*. This is precisely the mechanism by which Δ_S\* — rather than Δ_S — enters the bound: round stability depends on γ_i relative to Δ_S\*, and once γ_i < Δ_S\* the geometric success probability kicks in (Proposition 2). The lower bounds confirm this is not improvable in gap dependence.

---

## Suggestions

1. **Make the supplementary self-contained.** Move the proofs of Theorems 1 and 2 (and Propositions 1 and 2) into a submission-attached appendix rather than citing an external "full version." This is essential for reviewers and readers to verify correctness.

2. **Add a stopping rule to Algorithm 1.** Include an explicit check such as "if remaining time < T_i + t_i, terminate" at the start of each round. This makes the pseudocode a complete and implementable specification.

3. **Add a regret-vs-Δ_S\* experiment.** Fix T and vary S to change Δ_S\*, and plot the resulting constant regret. A log-log plot should reveal a slope of approximately α/(1−α) = 1 (for α = 1/2), directly validating the core theoretical prediction.

4. **Discuss or add a missing lower bound for Lipschitz bandits in d > 1.** State explicitly whether the (d+1) exponent on 1/Δ_S\* in Corollary 3 is conjectured to be tight and what the primary obstacle to proving it is.

5. **Acknowledge and explain the non-realizable experimental gap.** Add a paragraph in Section 6 explaining why SELECT outperforms its oracle in non-realizable settings (possibly: non-realizable rounds terminate faster via LCB, so SELECT effectively concentrates the oracle's budget on a shorter effective horizon). Also acknowledge that SAT-UCB/SAT-UCB+ heuristics achieve lower standard regret than SELECT in Figure 3b, even if this is expected to be an artifact of the specific instance.

6. **Discuss practical approaches for unknown α.** Even a brief discussion of monotone search over α, or a default conservative choice (e.g., α = 0.5), would substantially improve the paper's practical utility claim.

---

## OEC6zOuZG1

- GT: Reject (avg 4.8)
- Predicted: Accept (5.7/10)
- Match: NO

### Final Review

## Summary

This paper analyzes the Random Feature Model (RFM) under spiked covariance data to explain why RFMs often empirically outperform linear models, contrary to what isotropic Gaussian equivalence theory predicts. The authors extend the universality theorem of Hu & Lu (2023) to the spiked covariance setting and show that when input-label correlation is strong (high alignment parameter α and/or spike magnitude θ), the RFM behaves equivalently to a high-order polynomial model rather than a noisy linear model. The degree of the equivalent polynomial is governed by a scalar η that encodes this correlation, with the full characterization given in Theorem 2 and Corollary 3.

---

## Strengths

- **Concrete polynomial equivalence with Hermite-coefficient characterization (Theorem 2 + Remark 4):** The result that the effective polynomial degree of the RFM is determined by η—which scales with (ξ + θαγ)/√(1+θα²), i.e., the first component of the input-label covariance—is a specific and non-trivial insight. Remark 4 further shows that *which* Hermite coefficients of σ and σ∗ vanish precisely controls whether the RFM reduces to a polynomial or collapses back to a linear model (e.g., for tanh/ReLU pairs, the relevant Hermite coefficients determine the outcome). This level of mechanistic specificity is a genuine contribution beyond generic "nonlinearity helps" narratives.

- **Extension of universality to spiked covariance is non-trivial and well-motivated:** Prior universality results (e.g., Hu & Lu 2023) required isotropic data, which is inconsistent with empirical observations. By requiring the feature matrix scaling 1/(n+θ) and deriving new moment-matching conditions (eqs. 10–11) under the spiked structure, the paper fills a concrete gap. The modified Lindeberg exchange under the anisotropic setting requires that moment conditions (16)–(18) be re-derived for the spiked case.

- **Figure 2 provides clean empirical validation of the polynomial equivalence:** Under aligned settings (α = 1, θ = n^{1/2}), the RFM generalization error matches the noisy polynomial model but diverges from the noisy linear model—and this reversal occurs in a functionally interpretable way depending on the Hermite structure of both σ and σ∗ (Figures 2a, 2b). This directly validates the core theoretical claim.

- **Phase-transition boundary (Corollary 3 + Figure 1a):** The characterization of when η = O(n^{-1/2}), demarcating the linear vs. nonlinear regime as a function of (α, θ), is a precise quantitative result that matches the numerically observed boundary well.

---

## Weaknesses

- **Assumption A.2 inconsistency—β ∈ [0, 1/2] vs β < 1/2:** The formal assumption (A.2) lists β ∈ [0, 1/2] (closed interval), yet the Discussion of Assumptions states "our current proofs necessitate that β < 1/2," and the abstract states β ∈ [0, 1/2). It is unclear whether β = 1/2 is actually covered. This inconsistency should be resolved; if β = 1/2 is not covered, the set notation in A.2 should be corrected to an open interval.

- **ReLU excluded by Assumption A.6 with only informal empirical justification:** The odd-function requirement in A.6 excludes ReLU, the dominant activation function in practice. The paper acknowledges this but only notes that "empirical evidence suggests our findings remain valid even when using ReLU." For a theory paper, this gap is significant: no formal bound is provided on the error incurred by applying the theory to non-odd activations, and no quantitative analysis (even an approximation argument) is offered. This limits how confidently the theoretical framework can be applied to modern architectures.

- **η is treated as a deterministic condition in Theorem 2, but is random:** The condition η ≤ C/n^{1/l} (eq. 15) depends on the random feature rows f_i via |(ξ + θαγ)^T f_i|. The claim that η = O(n^{-1/4}) holds with high probability when β < 1/2 is stated informally in Section 5 without a formal proposition or proof. As stated, Theorem 2 requires the practitioner to verify a condition on a random quantity before applying the result. This should be formalized as a lemma.

- **Single-spike model is restrictive, and multi-spike generalization is not discussed:** The entire analysis rests on rank-one spiked covariance (I_n + θγγ^T). Real data covariances typically have a decaying spectrum with multiple significant eigenvalues. The paper provides no discussion of whether the qualitative conclusions (polynomial degree governed by input-label correlation) persist with multiple spikes, or whether new phenomena arise. This limits the theoretical scope of the paper's claims.

- **Figure 3c extends to β ≥ 0.5 without adequate flagging:** The theoretical results are confined to β < 1/2, yet Figure 3c plots β up to 1.0. While the text mentions "our analysis is confined to β < 0.5," the figure presents empirical results beyond this range without any visual demarcation, potentially misleading readers about the reach of the theory.

- **Figure 4 legend conflates model types and activation functions:** The CIFAR-10 figure (Figure 4) lists "ReLU (blue circles)" and "Softplus (red circles)" alongside "Random Feature Model (blue diamonds)" and "Noisy Linear Model (red diamonds)"—but ReLU and Softplus *are* random feature models. This typological inconsistency makes the figure difficult to interpret without careful reading of the appendix. A clearer labeling scheme is needed.

- **CIFAR-10 experiment does not verify spiked covariance structure:** The theoretical framework assumes a rank-one spiked Gaussian covariance, but no analysis is provided to verify that CIFAR-10 images (non-Gaussian, high-order structure) exhibit this structure. The experiment shows that the polynomial model tracks the RFM as input-label correlation increases, which is suggestive, but the connection to the theoretical assumptions is not formally justified.

- **Technical novelty over Hu & Lu (2023) is not clearly articulated in the main text:** The paper states it follows "the proof technique used by Hu & Lu (2023)" via Lindeberg's method. The specific proof challenges introduced by the spiked covariance—e.g., controlling spectral norms under rank-one perturbations, modified moment estimates in eqs. (16)–(18)—are not described in the main body. Without this, it is difficult to assess the depth of the technical contribution relative to the baseline work.

---

## Nice-to-Haves

- **Eigenspectrum validation for CIFAR-10:** Plotting the top-K eigenvalues of the CIFAR-10 sample covariance matrix would help justify the spiked covariance approximation for this dataset.

- **Formal verification/plot of the η condition:** Adding a figure showing η measured in simulations as a function of n and β would corroborate the informal claim that η = O(n^{-1/4}) with high probability, giving Theorem 2 an empirically grounded condition.

- **Discussion of the strong-spike regime (β ≥ 1/2):** Even a heuristic discussion of how the results might change when the spike dominates the spectrum would help practitioners understand the boundary of applicability.

- **Non-odd activation ablation:** Comparing a properly centered/symmetrized ReLU (e.g., ReLU(x) − ReLU(−x) ∝ x) against standard ReLU would help isolate the impact of violating A.6, providing empirical guidance on when the odd-function assumption is material.

- **Hermite coefficient table for ReLU and tanh:** The insight in Remark 4 that Hermite coefficients μ₂μ̃₂ and μ₃μ̃₃ determine equivalence class is non-trivial; a table with numerical values of μ_j and μ̃_j for common activation pairs would make this immediately accessible.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Title is misleading:** The harsh critic argues the title should acknowledge conditionality. The subtitle "Effect of Strong Input-Label Correlation in Spiked Covariance Data" already explicitly scopes the setting. The title as written is acceptable and consistent with how RFM superiority is usually discussed in the literature.

- **Comparison between optimized polynomial/linear and fixed-form ReLU/Softplus is unfair:** The harsh critic argues the optimized coefficients in (21)–(22) give polynomial/linear activations an advantage. However, this is by design: the paper constructs an "optimal" polynomial to characterize the theoretical upper bound achievable by the equivalent model class. The comparison illustrates how close practical activations come to this bound, not that practical activations are inferior because they are unoptimized. The asymmetry is transparent and intentional.

- **Contribution 3 is just an enabling ingredient, not independent:** This is an organizational critique with no bearing on correctness or value. The universality extension to spiked data is a technically non-trivial prerequisite that is likely of independent interest to the RMT community.

- **No comparison to kernel SVMs or fully trained networks:** This is scope creep. The paper explicitly situates itself within the fixed random features literature and does not claim to surpass trained networks.

- **Unknown parameters θ and α in practice:** As a theoretical paper establishing asymptotic equivalences, the practical unknowability of θ and α is a standard limitation of the entire proportional-asymptotics program (including Hu & Lu 2023). This is not a weakness specific to this paper.

---

## Novel Insights

The most genuinely novel insight emerging from this synthesis is the *Hermite-coefficient decoder* view of the linear-to-polynomial transition: Theorem 2 + Remark 4 together imply that for a given activation pair (σ, σ∗), the minimum polynomial degree needed for RFM-equivalent performance is determined by the lowest j such that μ_j(σ) · μ̃_j(σ∗) ≠ 0, modulated by whether η exceeds the corresponding n^{-1/j} threshold. This means that the question "when does nonlinearity help?" reduces to a structured factorization problem in Hermite space, with the answer depending jointly on the activation function, the label function, and the data geometry (α, θ)—not on any single factor in isolation. This three-way interaction is a sharper characterization than what exists in the isotropic theory.

---

## Suggestions

- **Fix the A.2 notation:** Change β ∈ [0, 1/2] to β ∈ [0, 1/2) everywhere to match the actual proof requirement, or formally prove coverage of β = 1/2 if it is genuinely included.

- **Add a formal lemma for the η bound:** State and prove (or provide a proof sketch in the appendix) that η ≤ C/n^{1/4} with high probability when β < 1/2 and F is independent of γ and ξ. Reference this lemma when invoking Theorem 2, so it is clear the theorem's condition is satisfied in the main experimental regime.

- **Clarify Figure 4 legend:** Rename "Random Feature Model (blue diamonds)" to "Noisy Polynomial Model (blue diamonds)" if that is what is being shown, and confirm that ReLU and Softplus are labeled as RFM variants in the caption.

- **Add a vertical β = 0.5 boundary marker in Figure 3c** to visually distinguish the proven regime from the extrapolated regime.

- **Expand the proof approach section** to explicitly describe the technical difficulties introduced by the spiked covariance structure—specifically, how the rank-one perturbation θγγ^T affects the Lindeberg exchange step and moment estimates—to make the contribution over Hu & Lu (2023) transparent to readers.

---

**Evaluation summary:**
- *Novelty:* Moderate-to-good. The extension of universality to spiked covariance data and the polynomial equivalence characterization via Hermite coefficients are genuine contributions, though the techniques build directly on Hu & Lu (2023).
- *Technical soundness:* Good within the stated assumptions, but with a formal gap around the η condition and a genuine limitation from the odd-activation requirement.
- *Empirical support:* Strong for synthetic experiments; the CIFAR-10 results are suggestive but methodologically under-justified.
- *Significance:* Moderate. The paper offers a principled explanation for a widely observed empirical phenomenon, making it meaningful for the theoretical ML community.
- *Clarity:* Good overall, with specific issues in Figure 4 labeling and the informal treatment of the η distributional claim.

---

## 6e3hoDZKuO

- GT: Reject (avg 3.5)
- Predicted: Reject (4.8/10)
- Match: YES

### Final Review

## Summary
The paper proposes a method for training goal-directed dialogue agents by using LLMs as a data generator rather than as direct conversational agents. A three-step "Imagination Engine" (Reasoning → Imagination → Critique) synthesizes diverse synthetic dialogues conditioned on sampled personas and binary reward outcomes; these dialogues are then used to train a GPT-2-scale agent via offline RL (ILQL). The key claim is that a small model trained on imagined data via RL surpasses directly prompted state-of-the-art LLMs (GPT-3.5) on goal-directed tasks requiring information gathering, and that RL outperforms behavioral cloning on challenging out-of-distribution scenarios.

---

## Strengths

- **First to use synthetic dialogue generation for offline RL.** Prior knowledge-distillation work from LLMs has exclusively targeted supervised learning objectives. This paper is the first to generate synthetic, reward-labeled dialogues and apply offline RL to them, which is a conceptually distinct and substantive contribution.

- **Principled Critique step addressing a real failure mode.** The observation that LLMs generating synthetic dialogues tend to have the simulated human reveal their latent persona immediately — defeating the purpose of information-gathering training — is well-motivated, and the critique-and-revision step is a concrete, practical fix. The criterion that "human sentiment at the end of the dialogue should accurately reflect the reward" further tightens the reward signal from binary conditioning.

- **Compelling efficiency result.** IE+RL (GPT-2, ~117M params) consistently outperforms GPT-3.5 (>100× larger) on all four Likert metrics across both tasks (Table 1). This gap is especially striking in naturalism (B: 3.8 vs 2.3 in Instruction) and overall satisfaction (D: 4.2 vs 2.4). Because the asymmetry deliberately favors the baseline, the result — if validated — argues convincingly that the locus of improvement is the training paradigm rather than model capacity.

- **Hidden-parameter MDP framing clearly explains LLM shortcomings.** Formulating goal-directed dialogue under latent user state as a hidden-parameter MDP is a clean, well-known formalism that cogently explains why standard RLHF (bandit objective) cannot discover multi-step information-gathering policies. This framing is pedagogically useful to the community.

- **Supplementary synthetic evaluation.** The paper supplements the small human study with a simulation-based evaluation (Appendix C), partially mitigating the concern about statistical power in the main results.

---

## Weaknesses

- **N=12 user study with no statistical significance tests — the core empirical claim is under-powered.** Tables 1 and 2 report means and standard errors across 12 participants, each making 3 interactions per agent per task. No t-tests, Wilcoxon signed-rank, or equivalent significance tests are reported. For Preference Elicitation in Table 1, the margin is narrow (e.g., metric A: 4.1±0.11 vs 3.8±0.21), and it is impossible to rule out chance variation. A paper claiming "state-of-the-art performance" and "significantly better results" must provide statistical evidence. Without it, the primary quantitative conclusions cannot be trusted.

- **Experimenter bias in Table 2 is uncontrolled.** The challenging-scenario dialogues in Table 2 are explicitly generated by the authors themselves: *"we pose as humans who exhibit potential challenging behaviors and interact with agents"* (Section 5.3). The authors, who designed IE+RL and know its capabilities, crafted the adversarial inputs and selected which dialogues to present to raters. There is no description of how these scenarios were chosen or reviewed independently. This creates a serious risk of confirmation bias: the scenarios may have been implicitly tailored to showcase RL's trajectory-stitching advantage. The large gaps in Table 2 (e.g., IE+BC at 1.4±0.10 vs IE+RL at 3.8±0.21 on Preference Elicitation metric A — a 2.4-point gap on a 5-point scale) are extraordinary, and without independent adversarial scenario generation, they cannot be interpreted at face value.

- **Abstract/introduction claims persuasion as an evaluated task, but no results appear.** Line 86 states: "We evaluate our approach on tasks involving teaching of a new concept, persuasion, and preference elicitation." The abstract retracts this to "teaching and preference elicitation," and the main body evaluates only these two tasks. If persuasion was dropped, this must be stated explicitly; if it is deferred to an appendix, that appendix should be included and summarized. The discrepancy between stated and reported scope is a factual inconsistency.

- **No ablation of the Critique step.** The paper introduces the critique step to address a specific failure mode (humans in imagined dialogues revealing their persona too early), but provides no ablation comparing IE+RL with vs. without it. It is therefore unclear whether the critique step improves downstream RL performance or merely cosmetically improves dialogue realism. Given that this step adds nontrivial prompt-engineering complexity, its marginal contribution to the RL objective should be measured.

- **No validation of the synthetic reward signal.** The binary reward labels ($r \in \{0,1\}$) assigned by the LLM during data generation drive the entire RL training signal. There is no analysis of whether these labels correlate with human satisfaction ratings, whether LLM-labeled "successful" dialogues differ meaningfully from "failed" ones, or whether the binary reward is reliable at scale. If the reward labels are noisy or poorly calibrated, the RL agent is optimizing for a hallucinated signal.

- **The "zero-shot" claim is misleading by standard ML conventions.** The paper defines "zero-shot" as requiring no human-human demonstration data, which is a legitimate and useful property. However, the method requires handcrafted prompts for three separate components (Reasoning, Imagination, Critique), all labeled "handcrafted" in Figure 2. The paper acknowledges in the Limitations that "we still require human intervention in the form of task-specific prompts." Using "zero-shot" in the title without an early, precise definition risks overstating generalizability. The paper should define the term explicitly and early, or use a more accurate descriptor (e.g., "annotation-free" or "data-free").

- **Gap between token-level MDP formalism and the actual RL implementation.** The MDP formalism defines actions $a_t \in \mathcal{V}$ (individual tokens), but Section 4.2 (lines 165–168) describes extracting transitions by isolating "all tokens by the agent" — which corresponds to utterance-level actions. The policy extraction formula $\hat{\pi}(a|s) \propto \pi_\beta(a|s) e^{\beta(\hat{Q}(s,a) - \hat{V}(s))}$ also references Snell et al. (2022) without making explicit whether Q-values are token-level or utterance-level. This gap between the formal presentation and the implementation makes it difficult to assess the credit assignment mechanism and reproduce the method.

---

## Nice-to-Haves

- **Apples-to-apples training comparison.** Including a comparison between prompted GPT-2 and IE+RL (GPT-2) — or IE+RL at GPT-3.5 scale — would isolate the effect of the training paradigm from model capacity and sharpen the contribution.
- **Quantitative prompt sensitivity analysis.** A systematic evaluation (e.g., 5 prompt paraphrases × downstream RL performance) would substantiate the claim that prompts are "not overly-engineered" and that the method is robust to prompt variation.
- **Cost analysis.** A comparison of GPT-3.5 API costs for data generation vs. ongoing GPT-3.5 inference costs as a direct agent would contextualize the practical efficiency argument.
- **Failure mode examples.** Showing cases where the Imagination Engine produces low-quality data (e.g., unrealistic personas, reward-hacking by a compliant simulated human) and how ILQL handles them would strengthen trust in the pipeline.
- **Transfer to a third unseen task.** Demonstrating that the imagination engine prompts transfer with minimal adaptation to a genuinely new task (e.g., negotiation or emotional support) would substantiate the "zero-shot" branding beyond the two author-designed tasks.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Unfair comparison (GPT-2 vs GPT-3.5):** The harsh critic argues the comparison conflates model size with training paradigm. However, by the paper's own design, the asymmetry intentionally benefits the baseline (GPT-3.5 is >100× larger). A method that beats a far larger model makes a *stronger* empirical point, not a weaker one. This is a deliberate and reasonable experimental choice, not a flaw in the evaluation.

- **"RLHF considers only bandit objectives" is too broad:** The harsh critic notes that conversation-level RLHF approaches exist and should be cited. Without access to external literature, we cannot confirm which specific works exist or were contemporaneous; this criticism is removed per reviewer instructions.

- **Sparse reward / credit assignment concerns for ILQL:** The critic is concerned that computing $\max_{a'} \tilde{Q}(s,a')$ over a 50K vocabulary is computationally infeasible. ILQL (Snell et al., 2022) is specifically designed for language model policy extraction with this formula and handles it via the $\pi_\beta$-weighted extractable policy, which does not require an explicit argmax over the full vocabulary. The concern overstates a non-problem for this specific algorithm.

- **Societal manipulation risk:** While a valid general AI ethics concern, it is not specific to this paper's contribution and applies equally to any dialogue agent. Removed as a non-specific "any-paper" criticism.

---

## Novel Insights

The most genuinely novel conceptual observation in this paper — reinforced by the spark finder — is the **role-separation between LLMs as data generators vs. LLMs as agents**. Standard knowledge distillation uses LLMs to produce supervision targets for supervised learning. This paper shows that LLMs can produce *suboptimal but plausible* behavioral data, and that offline RL can extract policies that transcend the quality of the generator — an analogue of "trajectory stitching" from offline RL applied to the knowledge-distillation setting. This reframing opens a promising direction: LLM capabilities can be used to *define the support of the MDP* without needing to solve it, and RL handles the optimization. Whether this scales reliably is an open question, but the conceptual distinction is clean and under-explored.

---

## Suggestions

1. **Statistical rigor:** Re-run the user study with at least 40–50 participants and report pairwise significance tests (e.g., Wilcoxon signed-rank on matched ratings). Given the online nature of LLM interactions, recruiting via crowdsourcing platforms is feasible and would substantially strengthen Table 1.
2. **Independent evaluation of challenging scenarios (Table 2):** Have the adversarial "challenging" personas designed by a third party blind to which agent is IE+RL. Show these to raters with live agent interactions rather than pre-generated transcripts.
3. **Add Critique-step ablation:** Include IE+RL (no Critique) as an ablation in Table 2 to demonstrate the Critique step's contribution to downstream RL performance, not just dialogue surface quality.
4. **Reward signal calibration experiment:** For a subset of generated dialogues, obtain human ratings and compute correlation with the LLM-assigned $r \in \{0,1\}$ labels. This would be a strong validation of the RL training signal.
5. **Clarify MDP level of abstraction:** Either restate the MDP formalism at utterance level to match implementation, or explicitly describe how ILQL handles token-level Q-values in practice. A one-paragraph implementation note in Section 4.2 would resolve this.
6. **Reconcile persuasion task:** Either include results for the persuasion task or explicitly remove it from the introduction's list of evaluated tasks and explain the omission.

---

**Axis evaluations:**

- **Novelty:** High. The specific combination of reward-conditioned synthetic dialogue generation + offline RL for goal-directed conversation is a distinct contribution; the imagination engine design is principled and non-trivial.
- **Technical soundness:** Moderate. The hidden-parameter MDP framing and ILQL application are well-grounded, but the gap between the token-level MDP formalism and the utterance-level implementation is unexplained, and the reward signal is unvalidated.
- **Empirical support:** Weak. The user study is substantially underpowered (N=12, no significance tests), and the most dramatic result (Table 2) is confounded by experimenter-generated test scenarios. The synthetic evaluation in simulation is a partial remedy but not a substitute.
- **Significance:** Moderate-to-high if the core empirical claims hold up under stronger evaluation. The setting is practically important and the conceptual framework is clean. If validated, the finding that a GPT-2-scale agent can beat prompted GPT-3.5 via this approach would be highly significant.
- **Clarity:** Moderate. The writing and figures are generally clear, but Section 4.2 contains a large block of blank line numbers (lines 189–322 in the extracted text) suggesting a significant gap in the submitted PDF extraction, and the "zero-shot" terminology requires more careful upfront definition.

---

## fmJUYgmMbL

- GT: Accept (Poster) (avg 6.2)
- Predicted: Accept (6.3/10)
- Match: YES

### Final Review

## Summary

LRDS (Learned Reference-based Diffusion Sampler) addresses a fundamental practical bottleneck in variational diffusion-based samplers: their performance on multi-modal targets degrades sharply with miscalibration of the reference distribution hyperparameter σ, yet calibrating σ requires ground-truth samples. LRDS sidesteps this by constructing a learned, multi-modal reference distribution from local MCMC chains initialized at known mode locations, then using this reference within the RDS variational framework. Two variants are proposed — GMM-LRDS (lightweight, analytically tractable scores) and EBM-LRDS (more expressive, neural score estimation via a multi-level EBM trained with annealed MCMC) — and both are shown to substantially outperform existing methods at recovering mode weights on high-dimensional Gaussian mixtures and the φ⁴ field theory model.

---

## Strengths

- **Precise and compelling diagnosis of σ-sensitivity.** Figure 1 (left/middle) shows a razor-sharp optimum in σ for LV-PIS and LV-DDS on a 16-D bi-modal target; the optimal σ coincides with the analytically-derived Gaussian approximation variance (Appendix I.2), making the problem concrete and quantitative, not merely intuitive.

- **The robustness asymmetry observation is genuinely novel.** Figure 1 (right) shows that a GMM reference with miscalibrated mode weights is remarkably robust — performance is nearly flat across a wide range of reference weights — whereas a Gaussian reference has a single narrow optimum. This asymmetry is the conceptual core of the paper and is clearly demonstrated.

- **The two-variant design cleanly maps to different problem regimes.** Figure 3 constitutes a crisp, visually unambiguous ablation: on the Rings distribution, GMM-LRDS cannot capture the ring geometry regardless of the number of components (J=16 shown), while EBM-LRDS exactly recovers it. This provides principled guidance on when to prefer which variant.

- **φ⁴ experiment is a strong scientific validation.** On a physically meaningful, non-synthetic target (d=32, two well-separated modes), all competing methods suffer mode collapse while GMM-LRDS correctly tracks the analytical Laplace approximation of the relative mode weight across varying local-field parameter h (Figure 4). The consistency with the Laplace approximation is a meaningful quantitative check.

- **Table 1 elegantly unifies the framework.** By showing PIS and DDS as special cases of RDS with fixed Gaussian reference, the paper provides a conceptually clean perspective on the broader method family, and Table 1 makes the generalisation explicit and graspable.

- **LRDS does not require target score evaluations.** The paper notes (Section 5) that LRDS only needs pointwise evaluations of the unnormalized density γ, not its gradient. This is a practically meaningful advantage for targets where the score is expensive or unavailable, and it distinguishes LRDS from most competitors.

- **The multi-level EBM training formulation solves the negative sampling problem cleverly.** By parameterising the entire noised path (p_t^φ)_{t∈[0,T]} jointly and using annealed MCMC across the path (Section 3.3), the authors simultaneously obtain negative samples for EBM training and leverage annealing for multi-modal coverage — a genuine "two birds, one stone" design that draws on established ideas but applies them non-trivially in this context.

---

## Weaknesses

- **Mode location assumption: minimal practical guidance.** The assumption that mode locations are known a priori is stated clearly in the introduction but receives almost no practical treatment. In real Bayesian inference tasks, obtaining mode locations may itself require MAP optimisation with multiple random restarts, which can be costly and unreliable on rugged landscapes. The paper gives no guidance — not even a brief discussion — on how to obtain modes in practice, how many can be handled, or what happens when mode locations are systematically biased (not merely randomly perturbed as tested in Appendix I.5.1). This gap limits the reader's ability to judge whether LRDS is deployable on their actual problem.

- **Overstatement in Table 2 analysis.** The paper claims "GMM-LRDS outperforms competing methods in all the considered dimensions," but at d=16, PDDS achieves 0.8% ± 0.6% versus GMM-LRDS's 1.7% ± 0.6% — PDDS is numerically superior. PDDS then catastrophically fails at d=32 (66.7%), so LRDS's advantage in higher dimensions is genuine and substantial, but the blanket claim of dominance at d=16 is factually inaccurate and should be corrected.

- **Mode weight estimation is the sole primary metric.** All quantitative main-text evaluation rests on a single scalar — absolute mode weight estimation error. While this is the central problem the paper addresses, it does not characterise intra-mode sample quality (e.g., covariance fidelity, energy distribution within a mode, or sliced Wasserstein distance). Variational metrics from Appendix G/I.5.1 are not discussed in the main text. For a method claiming to produce good samples from multi-modal distributions, this leaves the reader unable to assess whether the samples within each mode are well-distributed.

- **No computational cost comparison.** EBM-LRDS requires (i) running annealed MCMC for negative sampling, (ii) training the multi-level EBM E^φ, and (iii) running the full RDS variational optimisation — a pipeline with potentially 3-5× the cost of baselines. The paper acknowledges this in the Discussion but provides zero wall-clock times or evaluation-budget comparisons. Without this, the empirical advantage cannot be fairly weighed against the computational investment.

- **EBM training stability not characterised.** EBM training is notoriously susceptible to divergence and mode covering artifacts during negative sampling. Section 3.3 provides no convergence curves, stability diagnostics, or ablations on the EBM training hyperparameters (annealing schedule, step size, negative MCMC chain length). There is no evidence that the estimated reference scores s_t^ref = -∇_x E^φ(t,x) are accurate enough across all timesteps to reliably guide the diffusion process.

- **Failure modes of competitors in the φ⁴ experiment are not explained.** The paper simply states that all competitors exhibit mode collapse on the φ⁴ model without explaining whether this is due to the energy barrier structure, the dimensionality (d=32), the specific conditioning number of the covariance, or some other cause. Understanding *why* the competitors fail would substantially strengthen the paper's narrative about when LRDS is needed and when simpler alternatives suffice.

---

## Nice-to-Haves

- **Practical mode acquisition discussion.** A brief experiment or appendix showing a pipeline where mode locations are obtained via L-BFGS with multiple random restarts, followed by LRDS, would significantly strengthen the practical case. Even a discussion of failure conditions (e.g., number of modes beyond which the approach becomes infeasible) would help practitioners.

- **Ablation: competitor methods with GMM reference.** It is unclear whether the performance gains come from the LRDS variational framework or simply from the use of a multi-modal GMM reference that competitors could also adopt. A comparison of LV-PIS/LV-DDS using a GMM reference (instead of their default isotropic Gaussian) would isolate the contribution of the learned reference versus the RDS training framework.

- **Non-diffusion baseline: local MCMC + normalising constant estimation.** When mode locations are known and local MCMC chains are available, one could estimate mode weights via annealed importance sampling or thermodynamic integration within each mode. Including such a baseline would give readers a clearer sense of what the diffusion-based framework adds on top of the simplest possible exploitation of mode location knowledge.

- **Sensitivity to missing or strongly mislocated modes.** The appendix ablation perturbs mode locations lightly, but a more severe test — missing one mode entirely, or having mode locations off by a large multiple of the mode's standard deviation — would establish the robustness boundaries more concretely.

- **Extension to higher dimensions (d ≥ 128).** Table 2 stops at d=64. Diffusion-based methods are often motivated by high-dimensional scalability; demonstrating whether GMM-LRDS's advantage persists at d=128 or d=256 would give a clearer picture of the method's regime of applicability.

- **Real-world multi-modal posterior in the main text.** The Bayesian logistic regression results are in the appendix and the target is noted to be "not explicitly multi-modal." A real-world target with a known multi-modal posterior (e.g., Bayesian neural network with symmetries, or a mixture-model posterior) in the main paper would substantially broaden the demonstrated applicability.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **"Comparison is unfair because baselines receive poor σ calibration" (Harsh Critic):** The paper explicitly sets σ for PIS/DDS/DIS using "a Gaussian isotropic approximation of π̂^ref" (Section 5), which represents the best practically-available σ estimate without ground-truth samples. This is precisely the realistic evaluation setting the paper targets. Demanding oracle σ tuning for baselines would test an unrealistic scenario that contradicts the paper's stated problem setting. The comparison is fair by design.

- **"iDEM at d=64 is competitive" (Harsh Critic):** iDEM achieves 11.7% ± 0.4% at d=64 versus GMM-LRDS's 4.1% ± 0.6% — this is roughly a 3× gap. At d=32, iDEM collapses to 66.7%. This is not competitive.

- **"Listing hyperparameter sensitivity as a contribution is inappropriate" (Harsh Critic):** The paper provides a quantitative, controlled demonstration of this sensitivity with explicit connection to the analytically-derivable optimal σ (Appendix I.2). This constitutes a contribution beyond mere intuition.

- **Missing related works suggestions (all reviewers):** Per evaluation policy, no missing related works are noted, as their existence cannot be independently verified.

- **Pure style/formatting critique on Equation (7) redundancy (Positive Reviewer):** The apparent redundancy in the g − ½g term in (7) may be a text-extraction artifact from the PDF; without access to the original LaTeX, this cannot be reliably evaluated and is a formatting concern rather than a substantive flaw.

---

## Novel Insights

The most genuinely novel observation in this paper — beyond the method itself — is the *asymmetry in robustness between Gaussian and GMM references*, demonstrated cleanly in Figure 1: a Gaussian reference has a single narrow optimal σ requiring ground-truth calibration, while a GMM reference with the correct modal structure is nearly flat with respect to mode weight miscalibration. This asymmetry is not obvious a priori (one might expect the GMM's extra parameters to introduce additional sensitivity), and it provides a principled theoretical intuition for why investing in learning a multi-modal reference yields disproportionately large gains in practical robustness. The secondary insight — that the multi-level EBM parameterisation simultaneously enables tractable EBM score estimation and multi-modal negative sampling via annealing — is a useful connection between the EBM literature and diffusion-based sampling that may have broader applicability.

---

## Suggestions

- **Correct the overstatement in Table 2**: State explicitly that PDDS narrowly outperforms GMM-LRDS at d=16 (0.8% vs 1.7%) but fails catastrophically at d=32+, and that LRDS provides the most *consistent* performance across dimensions.

- **Add a practical mode-acquisition paragraph** in Section 3 or Section 6 discussing how mode locations can be obtained (e.g., gradient ascent from multiple random initialisations, clustering of short MCMC chains), and cite any existing mode-finding literature relevant to the paper's target application domains.

- **Report wall-clock training time** for at least one experiment (e.g., the d=64 Gaussian mixture), broken down by reference training and diffusion training phases, to allow readers to assess the cost-accuracy tradeoff of GMM-LRDS vs EBM-LRDS vs baselines.

- **Include convergence/stability diagnostics for EBM training**: At minimum, plot the ML objective over training iterations for EBM-LRDS and show that the reference score estimates (s_t^ref vs a finite-difference check on small-dimensional examples) are accurate.

- **Explain competitor failures on φ⁴**: Add one paragraph or figure in the appendix decomposing why competitors collapse (e.g., show mode weight trajectories during training, or energy landscape plots showing the barrier), since this provides actionable insight into when LRDS is most needed.

- **Move or expand the Bayesian logistic regression results**: Either move them to the main text with a genuinely multi-modal posterior task, or clearly characterise in the text why this unimodal-in-practice posterior is still a meaningful test case for the method.

---

## hNjCVVm0EQ

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Reject (4.8/10)
- Match: NO

### Final Review

Now I have sufficient understanding of the paper. Let me synthesize the final review.

---

## Summary

MamKO proposes integrating Mamba's matrix-generation (selection) mechanism with Koopman operator theory to produce time-varying Koopman operators from historical data, enabling efficient convex MPC for nonlinear time-varying systems. By generating the full operator sequence $\{\bar{A}_{k:k+H-1|k}, \bar{B}_{k:k+H-1|k}, C_{k:k+H-1|k}\}$ offline from past observations before each control step, the approach preserves MPC convexity (avoiding the bilinear $B_k u_k$ coupling) while adapting to changing dynamics. The method is evaluated on five benchmark systems spanning time-invariant and time-varying settings.

---

## Strengths

- **Elegant convexity preservation through offline operator generation.** By generating the full future operator sequence from *historical* data rather than online states, MamKO specifically avoids the bilinear term $B_k u_k$ that would arise if $B_k$ depended on the current input. This is a concrete, non-obvious design decision that keeps the MPC problem quadratic. The paper makes this explicit (Section 3.2): "as $B_k$ is generated from the input sequence containing $u_k$, the bilinear term $B_k u_k$ can lead to a non-convex optimization problem... As a substitute, we generate the matrices from the historical data."

- **CELU activation for unstable system representation.** Replacing Mamba's negative-exponential eigenvalue constraint with negative-CELU is a targeted and technically justified modification. It allows continuous-time eigenvalues up to +1 (positive, hence unstable modes), enabling Koopman representation of inherently unstable systems (e.g., CartPole, oscillatory GRN). The ablation in Figure 4 confirms CELU yields the best modeling across all three tested environments, and the improvement over unconstrained eigenvalues (the "None" variant) demonstrates that the constraint is not just permissive but regularizing.

- **Demonstrated advantage that scales with dynamical variation rate.** Figure 2(f) shows that MamKO's advantage over DKO grows monotonically with the angular frequency of parameter variation (0.1, 1, 10 rad/s). This provides principled evidence that the time-varying operator generation directly addresses the problem's core difficulty, rather than incidentally benefiting from extra parameters.

- **Dramatic computational advantage over nonlinear MPC.** MamKO-based MPC reduces solve time by 98%+ over MLP-MPC (nonlinear optimization via IPOPT) across all five systems — e.g., from 3.31 s to 0.0262 s for RSCP (sampling period 18 s) and from 0.743 s to 0.0102 s for CartPole (sampling period 0.02 s). This enables real-time control feasibility where MLP-MPC is not viable.

- **Sample efficiency advantage over model-free RL.** MamKO uses 36,000 labeled samples versus SAC's 1,000,000 environment steps. While both approaches achieve different objectives, this distinction matters practically for data-scarce industrial settings and is a genuine qualitative advantage of the model-based approach.

---

## Weaknesses

- **No comparison against adaptive or online Koopman baselines — the most critical gap.** The paper's primary claim is superiority on time-varying systems, yet the only Koopman baseline is the time-invariant DKO. The Related Work explicitly cites online DMD (Zhang et al., 2019), parameter-adjusting Koopman (Hao et al., 2022), and Fourier-filtered Koopman (Liu et al., 2023) as prior art for time-varying systems. The paper argues these are "time-consuming," but provides no empirical evidence of this claim and no comparison of their accuracy. Without such comparison, gains over DKO on time-varying benchmarks cannot be attributed to the Mamba architecture versus simply using any time-varying operator. Including at least one recurrent (LSTM/GRU) time-varying Koopman baseline would also isolate the contribution of the Mamba selection mechanism specifically.

- **Critical design assumption left unanalyzed: future dynamics predicted from past data.** At time $k$, the network $\phi$ generates the *future* matrix sequence $\bar{A}_{k:k+H-1|k}$, $\bar{B}_{k:k+H-1|k}$, $C_{k:k+H-1|k}$ using only past data $x_{k-H:k-1}$, $u_{k-H:k-1}$. This is a substantive assumption — it presupposes that the historical window provides enough information to forecast how the dynamics will evolve over the next $H$ steps. The paper never states this assumption explicitly, never analyzes when it holds or fails, and provides no experiment stress-testing the prediction horizon or the effect of history length $H$. For slowly varying systems (0.1 rad/s), this may be benign; for rapidly varying ones (10 rad/s), it is non-trivial and is exactly where the method is most claimed to shine.

- **No ablation of history length $H$.** History length is a critical hyperparameter governing the operator-generation quality, yet no sensitivity analysis is provided. It is unknown whether performance degrades gracefully or sharply with shorter windows, which matters greatly for online deployment.

- **MamKO is slower than DKO, yet efficiency is framed as a key contribution.** Table 2 shows MamKO is uniformly slower than DKO-based MPC: 10.2 ms vs. 7.35 ms (CartPole), 26.2 ms vs. 10.6 ms (RSCP), 29.5 ms vs. 12.8 ms (TV-RSCP). The paper's efficiency argument is valid only against MLP-MPC, which requires nonlinear optimization. The legitimate efficiency claim is "MamKO is nearly as fast as DKO while substantially outperforming it," not a general efficiency advantage that the text implies. This should be stated more precisely.

- **Marginal improvement on time-invariant systems without significance testing.** For CartPole (time-invariant), MamKO reduces cost by 5.05% over DKO. Given the confidence intervals visible in Figure 3, it is unclear whether this difference is statistically significant. No formal tests (e.g., paired t-test) are reported. Claiming "superiority" on these margins without statistical support is an overstatement.

- **Notation inconsistency between training (Eq. 9) and MPC (Eq. 10): six improvements cited for "five" systems.** The MPC optimization (Eq. 10a) optimizes over $u_{k|k}^*, \ldots, u_{k+N-1|k}^*$ using index $N$, but constraints use index $H$ (Eq. 10e: $j = k+1, \ldots, k+H-1$). $N$ and $H$ are never equated explicitly. Additionally, Section 5.2 lists six percentage improvements ("5.05%, 3.70%, 92.10%, 6.56%, 14.19%, 84.74%") for what the text refers to as "five systems," because the 10 rad/s time-varying CartPole is treated as a sixth scenario without being explicitly counted. These are writing errors that should be corrected.

- **Eigenvalue stability under long sampling periods.** The paper permits continuous-time eigenvalues up to +1, discretized as $\bar{A} = e^{AT}$. For RSCP with sampling period $T = 18$ s, a continuous eigenvalue of +1 yields a discrete eigenvalue of $e^{18} \approx 6.6 \times 10^7$. There is no discussion of how the training procedure prevents or handles such extreme values, or whether gradient descent remains numerically stable under these conditions.

- **Absence of theoretical stability or feasibility guarantees.** For a control-focused paper, the complete absence of closed-loop stability analysis is significant. While acknowledged as future work, with time-varying NN-generated operators, neither recursive feasibility of the MPC nor stability of the closed-loop system is addressed even informally. Standard MPC stability proofs assume model invariance; this paper's setting explicitly violates that assumption. A bounded-error or robustness-oriented argument, even informal, would substantially strengthen the control-theoretic grounding.

- **Incorrect and recurring "LLM" terminology.** The paper characterizes Mamba as a "large language model" throughout (Abstract, Introduction, Related Works, Conclusion). Mamba is a structured state-space model architecture; it is not an LLM in any precise sense. The only element borrowed from Mamba is the selective matrix-generation (discretization) mechanism, not language modeling, tokenization, or scale. This mischaracterization, while perhaps intended to attract attention, undermines technical precision and will be distracting to informed reviewers.

---

## Nice-to-Haves

- **Ablation isolating the input-dependent selection mechanism.** Compare MamKO against a variant where operator matrices vary with time step index but are not conditioned on historical input/state data (i.e., remove the selection/conditioning, keeping only time-varying operators). This would quantify the value of Mamba's specific input-dependent generation vs. simply learning time-varying operators by any means.

- **Latent-space linearity verification.** Low prediction error does not confirm that dynamics are truly linear in the lifted space. Plotting residuals of $z_{k+1} - \bar{A}_{k|k} z_k - \bar{B}_{k|k} u_k$ or visualizing eigenvalue trajectories of $A_k$ over time would validate that the Koopman assumption is not merely a useful fiction.

- **Sensitivity to observation noise.** Real control systems have noisy sensors. Testing whether the historical-data-conditioned operator generation amplifies noise (destabilizing MPC) would strengthen claims of practical applicability.

- **Eigenvalue trajectory visualization.** Plotting the diagonal entries of $\bar{A}_{k|k}$ over time for a time-varying system would provide interpretable evidence that the network genuinely adapts its represented dynamics rather than learning a fixed average.

- **Higher-dimensional benchmarks.** The largest system in the paper (RSCP) has a modest state dimension. Testing on a higher-dimensional system (e.g., a 7-DoF robot arm or quadrotor) would validate scalability claims implied by the "large model" framing and the efficiency results.

- **Explicit sample efficiency comparison.** The 36k (MamKO) vs. 1000k (SAC) training data gap is mentioned but not highlighted as a primary result. A dedicated comparison of data efficiency would be a compelling selling point.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Diagonal A lacks justification in the Koopman setting"** (Harsh Critic): The diagonal structure is explicitly inherited from Mamba's design and is standard in the SSM literature (S4, Mamba). Its computational advantages are well-understood and its effectiveness is empirically validated. Demanding a Koopman-specific theoretical justification for a design choice borrowed from a mature SSM literature is scope creep.

- **"Missing related works"**: Per review instructions, we do not cite missing related works as we cannot verify their existence.

- **"Validate on hardware or high-fidelity simulator with delays"** (Spark Finder): Demanding hardware experiments is well outside the scope of this type of ML/control venue paper and is not a standard expectation for ICLR contributions of this kind.

- **"Benchmark scalability / larger datasets"** (Reviewers 2 & 3): The current benchmarks are appropriate for demonstrating proof of concept. Larger benchmarks would strengthen claims but are not necessary to validate the core method. Moved to nice-to-haves in abbreviated form.

- **"Unfair comparison: MLP-MPC uses expensive nonlinear optimization"**: This comparison is intentionally asymmetric — MLP-MPC solving a nonlinear program is the cost of using an MLP model in MPC, and the asymmetry benefits the baseline to prove a stronger point about convexity. It is not a flaw.

- **"Requesting confidence intervals / formal significance tests for small-scale benchmarks"**: Unlike large-scale benchmarks where single-run evaluation is the norm, these are small-scale repeated experiments (10 trials each), so confidence intervals are already provided. The concern about formal significance testing on marginal improvements is retained as a weakness (not removed), but demanding it be done via a specific statistical framework (e.g., "Wilcoxon") is too prescriptive.

---

## Novel Insights

The most genuinely novel architectural insight in this paper is the *convexity-preserving historical conditioning* for time-varying Koopman MPC: by generating the full future operator sequence before the MPC solve (using only past data), the authors sidestep the bilinear $B_k u_k$ coupling that would arise if $B_k$ were conditioned on the current state or input. This design pattern — deliberate decoupling of the model-adaptation step from the optimization step — has broader implications for any learning-based MPC scheme that uses input- or state-dependent predictive models. The CELU activation as a generalization of the Mamba negative-exponential constraint to allow positive eigenvalues is a small but practically important modification with clear motivation from the control setting. Beyond these, the insight from Figure 2(f) that advantage over fixed-operator methods scales with the rate of dynamic variation provides a principled, quantifiable criterion for when the architectural complexity of MamKO is justified over simpler alternatives.

---

## Suggestions

1. **Add at least one time-varying Koopman baseline** (e.g., windowed EDMD or an LSTM/GRU-based time-varying Koopman model) to empirically substantiate the claim that the Mamba architecture, not merely any time-varying parameterization, drives the improvements. Even a simple RNN-based operator generator as an ablation point would be informative.

2. **Provide a sensitivity analysis over history length $H$** across at least two systems (one time-invariant, one time-varying). Show how prediction error and control cost change as $H$ varies; this is essential context for practitioners deploying the method.

3. **Revise all instances of "large language model" / "LLM"** when referring to Mamba. The accurate phrase is "state-space model (SSM) architecture" or "selective state-space model." This correction is both technically necessary and avoids the impression of buzzword-chasing.

4. **Correct and clarify the N/H notation** in Eq. 10: either unify to one symbol or explicitly state that $N = H$ in this formulation. Also fix the "five systems / six percentages" inconsistency in Section 5.2.

5. **Add a brief discussion of eigenvalue magnitude under long sampling periods** (the RSCP case with $T=18$ s) and whether the training procedure implicitly constrains eigenvalues to avoid numerically pathological discrete operators.

6. **Include a short analysis or discussion of closed-loop stability under time-varying operators**, even if informal. At a minimum, bound the prediction error of the operator-generation network and discuss how this propagates to MPC suboptimality. This is the most important gap for readers from the control community.

7. **Apply formal significance testing** (e.g., paired t-test across the 10 trials) when reporting control cost improvements in the time-invariant settings where the margins are small (5–6%). Either confirm statistical significance or temper the language of superiority.

---

## jbfDg4DgAk

- GT: Reject (avg 3.0)
- Predicted: Accept (6.2/10)
- Match: NO

### Final Review

Now I have enough context to write the final consolidated review.

---

## Summary

SpARK introduces *Sparse Watermarking*, a category of LLM watermarking that applies watermarks only to tokens following words with specific Part-of-Speech (POS) tags (Verb, Noun, Determiner), and adjusts the z-test detection to check only those anchored positions. By leaving the majority of tokens undisturbed and focusing statistical detection on the sparse watermarked subset, SpARK substantially reduces degradation to generated text quality while maintaining TPR ≥ 99%. Experiments on Llama-2-7b and Phi-3 across four datasets (long-form QA and summarization) demonstrate consistent quality improvements over four dense-watermarking baselines at matched detectability, with meaningful but acknowledged robustness tradeoffs under heavy attacks.

---

## Strengths

- **Substantially improved quality-detectability tradeoff with compelling margins.** On Llama-2-7b, SpARK-Determiner reduces ROUGE-L degradation to only 11.07% (Long-form QA) and 10.99% (Summarization) versus at least 22% for the best-performing baseline (Hard watermark) and up to 50% for Unigram. On Phi-3, SpARK-Verb degrades quality by only 5.17% vs. 13.75% for the next-best baseline. These are large, consistent gaps across two models and two tasks, not marginal improvements.

- **Semantic similarity preservation is distinctive and well-documented.** SpARK-Verb achieves 0.836 (Llama-2) and 0.850 (Phi-3) semantic similarity between watermarked and non-watermarked text, compared to 0.765/0.567 for Hard and 0.652/0.425 for Unigram. The difference is not a statistical artifact; the qualitative example in Table 4 concretely illustrates why — dense watermarking with SelfHash forces incoherent outputs (e.g., "Not enough crowd," "Marketton"), while SpARK-Determiner produces fluent, faithful text.

- **The core insight is principled and novel.** The observation that the statistical test gains nothing from including non-watermarked tokens in T, and that decoupling watermark placement from detection scope can improve quality without harming detectability, is a clean and non-obvious contribution. Unlike entropy-based skipping (Lee et al., 2023), which was domain-specific to code and required dynamic thresholding, SpARK's POS anchor provides a deterministic, language-structure-based positional scheme applicable to general text generation.

- **Perplexity results corroborate quality claims with an independent metric.** Figure 3 shows SpARK-Determiner achieves consistently lower median perplexity and smaller variance than all baselines, using Llama-2-13b as the oracle. The reduced variance is a notable secondary finding, suggesting SpARK produces more predictable generation quality.

---

## Weaknesses

- **Robustness drops significantly under heavy attacks, with no mechanistic explanation.** At 50% substitution on Llama-2-7b, SpARK-Verb falls to 72.4% TPR and SpARK-Determiner to 67.6%, compared to SelfHash at 92.3% and Unigram at 91.4% (Table 3). The paper acknowledges this but offers no analysis of *why*. The likely mechanism — that 50% random substitution has a meaningful probability of replacing POS anchor words themselves, which destroys anchor-next-token relationships — should be verified and discussed. This is not a fatal flaw (the paper is explicit about the tradeoff), but the absence of mechanistic understanding limits actionable guidance on when SpARK is and isn't appropriate.

- **Targeted attacks against POS anchors are not evaluated or even acknowledged.** Section 3.2 states that the adversary is "aware of the presence of watermarks." If the watermarking scheme is publicly known (which is typically assumed for cryptographic security), an adversary can trivially identify which tokens are watermarked by POS-tagging the generated text and targeting only the tokens immediately following anchor-POS words. This targeted substitution would be far more efficient at defeating the watermark than the random substitution evaluated in Table 3. For a method with publicly derivable anchor positions, this is a non-trivial security gap that the paper must at least discuss and bound.

- **POS tagger consistency between generation and detection is never analyzed.** The method requires that the POS tagger produces the same label for a given word both during incremental autoregressive generation (where the context is a partial sentence) and during post-hoc detection (where the full text is available). POS taggers can produce different tags for the same word in different sentential contexts or when context is truncated. Any mismatch causes the detector to miscalibrate T (the denominator of the z-score), degrading TPR. Neither a theoretical bound nor an empirical ablation (e.g., how often do encoder-time and decoder-time POS labels disagree?) is provided.

- **Hard restriction with γ = 0.05 creates a forced low-quality sampling regime that is not analyzed.** SpARK uses hard restriction rather than a logit bias δ: at watermarked positions, the model must sample from only 5% of word-starting tokens. If no high-probability token falls in the green list at a given position, the model is forced to emit a low-probability token, risking quality degradation at precisely the moments that matter most (rare or syntactically constrained contexts). The paper argues that sparse watermarking preserves quality, but does not verify that the green-list tokens at watermarked positions are systematically in high-probability regions. A plot of the probability mass of the sampled green-list token (relative to the argmax token) would close this gap.

- **The fraction of watermarked tokens is not reported in the main paper.** The core reason SpARK outperforms dense baselines on quality is that it watermarks a much smaller fraction of tokens. This fraction is the key variable that explains the quality-detectability tradeoff, yet the paper defers it to Table 7 in the appendix. Without this number in the main text, readers cannot assess whether SpARK's advantage reflects a genuine algorithmic insight or simply a different operating regime (fewer watermarked tokens → less distortion → lower robustness). This should be a central result, not an appendix entry.

- **Algorithm 1, Line 6 contains a pseudocode inconsistency.** Line 6 reads `Sample(G)`, which implies unconditional sampling from the green list G at every step. But the described method only restricts sampling to G when the POS condition is met; at other positions, sampling should proceed from the original distribution. The `POSWatermark` subroutine (Algorithm 2) returns the original P_M unchanged when the condition is not met, so the correct instruction would be `Sample(P_M(t))`. This inconsistency between the pseudocode and prose description is confusing and should be corrected.

- **Figure 3 x-axis labels are garbled and internally inconsistent.** The x-axis labels include "Selfish," "Llama2," and "None," which do not correspond to any method name used elsewhere in the paper (the methods are Hard, LeftHash, SelfHash, Unigram, SpARK-Verb, SpARK-Noun, SpARK-Determiner, and No Watermark). This makes Figure 3 difficult to interpret reliably without the original figure rendering.

- **TNR for SpARK-Determiner is measurably lower than some baselines.** Table 1 shows SpARK-Determiner achieving TNR of 98.0–98.8%, versus 100% for Hard watermark and Unigram. While 1–2 percentage points may seem small, in deployment this means a non-trivial false-positive rate on human-written text. The paper does not discuss this or whether it can be mitigated by threshold tuning.

---

## Nice-to-Haves

- Human evaluation of text quality (e.g., fluency ratings) would strengthen the quality claims beyond ROUGE-L and semantic similarity, which are imperfect proxies for human preference.
- An ablation over different POS tag combinations (e.g., Adjectives, Adverbs, Prepositions) beyond the selected three, including tags with less than 100% document frequency, would clarify how sensitive performance is to the choice of anchor tags.
- Cross-model detection experiments (watermark with Llama-2, detect assuming Phi-3's vocabulary) would test whether the scheme is robust to model version or architecture differences in the detector.
- Measuring and reporting inference latency overhead from real-time POS tagging would be useful for practitioners, though this does not affect the paper's core claims.
- Evaluating the False Positive Rate on human-written text (in addition to unwatermarked LLM text) would better characterize the TNR figures and their real-world implications.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **[REMOVED] Lee et al. (2023) as a required baseline.** Lee et al.'s entropy thresholding was specifically designed for *code generation* to preserve correctness — it is not a general-purpose text watermarking method. Criticizing SpARK for not including it as a baseline in a general-text setting is scope creep.

- **[REMOVED] Distortion-free methods (Christ et al., Kuditipudi et al.) as required baselines.** The paper explicitly notes that sampling-based schemes "struggled to produce a detectable watermark for low-temperature settings" (citing Piet et al., 2023). Since the paper's operating regime (TPR > 0.99) is incompatible with these methods in practice, excluding them as baselines is reasonable and explained.

- **[REMOVED] Demand for statistical confidence intervals across all tables.** Single-run evaluation is the norm in the LLM watermarking literature (Kirchenbauer et al., Zhao et al., and related works all report single-run figures). Demanding confidence intervals as a gating condition is above the standard of the field.

- **[REMOVED] Criticism of using Llama-2-13b to evaluate Llama-2-7b perplexity as fundamentally flawed.** The paper explicitly cites Jovanović et al. (2024) as precedent for this methodology. The criticism that shared architecture inflates the quality signal may have marginal merit but is not a meaningful objection given established community practice.

- **[REMOVED] "Contribution granularity" complaint about introducing a category with one instantiation.** Defining a broader class (Sparse Watermark) and demonstrating it through one instantiation (SpARK) is entirely standard in ML systems papers (cf. how "attention mechanisms," "residual connections," etc. were introduced). This is not a weakness.

- **[REMOVED] Demand for theoretical proofs for z-test validity under POS conditioning (as a gating condition).** The z-test validity concern is real (see Weaknesses), but demanding a full statistical proof before acceptance would be above the standard for an empirical systems paper. The requirement is a clear explanation and empirical verification, not a formal theorem.

- **[REMOVED] Requests for larger datasets and more models.** The paper covers two models and four datasets across two task types. This is adequate for an ICLR submission in this subfield; the current model zoo and dataset coverage are not obviously insufficient.

---

## Novel Insights

The most genuinely novel insight — validated by the empirical evidence — is that **watermark detection strength derives entirely from the tested token subset, not from the full generated text length.** This separation of the *generation regime* from the *detection scope* is non-obvious: prior work implicitly treats every token as both a watermark carrier and a detection signal. SpARK demonstrates that if anchor positions are known to the detector, restricting the z-test denominator T to only those positions preserves detection power while allowing the rest of the vocabulary to be sampled freely. The corollary — that the quality-detectability tradeoff is not fundamental but is instead an artifact of conflating generation and detection scope — has implications beyond SpARK itself, potentially motivating future work on other deterministic positional anchors (e.g., syntactic roles, semantic heads) that balance robustness and quality differently than POS tags.

---

## Suggestions

1. **Add a targeted-attack experiment.** Implement a substitution attack that specifically replaces tokens immediately following anchor-POS words, and report TPR under this attack. If TPR degrades substantially below random substitution, acknowledge this as a genuine security limitation and discuss whether a secret POS-tag selection mechanism could mitigate it.

2. **Report the watermarked-token fraction prominently in the main body.** Move the "% of tokens watermarked" figures from Table 7 (appendix) into Table 1/2 or a dedicated main-text figure. This number is essential context for interpreting the quality-detectability tradeoff.

3. **Fix Algorithm 1, Line 6.** Change `Sample(G)` to `Sample(P_M(t))` to correctly reflect that non-watermarked positions sample from the unmodified distribution.

4. **Provide a POS consistency ablation.** Report the POS agreement rate between incremental (generation-time) and full-context (detection-time) tagging on a held-out text sample. Even 1-2% disagreement per anchor position could meaningfully reduce T and hurt TPR for short texts.

5. **Analyze or empirically bound the hard-restriction pathology.** For each POS-anchored watermarking event, report the average probability rank of the sampled green-list token. If the median rank is low (e.g., top-5), the concern is mitigated empirically; if it frequently falls outside the top-100, the hard restriction is potentially a quality risk for rare syntactic contexts.

6. **Fix Figure 3 x-axis labels** to match method names used in Tables 1–2.

---

**Axis Evaluations:**
- **Novelty:** Moderate-to-high. The POS-anchor mechanism is a clear and specific idea, and the decoupling of generation scope from detection scope is a genuine conceptual advance, not merely an engineering variation.
- **Technical soundness:** Moderate. The core statistical framework is sound, but there are real unresolved questions around POS consistency, hard-restriction pathology, and the absence of targeted-attack analysis. The pseudocode error is a presentation gap.
- **Empirical support:** Moderate-to-strong for quality claims; notably weaker for robustness claims, which are the method's primary limitation.
- **Significance:** Moderate-to-high. The quality improvements over baselines at matched detectability are large and consistent, and the underlying idea generalizes beyond the specific POS instantiation.
- **Clarity:** Adequate overall, with specific lapses in the pseudocode and Figure 3 that require correction.

---

## Zonhl0c9I0

- GT: Accept (Poster) (avg 5.8)
- Predicted: Accept (5.3/10)
- Match: YES

### Final Review

## Summary
Self-Evolved Reward Learning (SER) proposes an iterative self-training framework for reward models (RMs) that reduces human annotation requirements by having the RM pseudo-label its own unlabeled data. The key mechanism is a curriculum-based two-status data filtering strategy: Status 1 selects pairs where the RM confidently distinguishes clearly good from clearly bad answers, and Status 2 (activated when Status 1 is exhausted) selects pairs where the RM can discern subtle quality differences. Experiments on HH-RLHF, UltraFeedback, StackOverflow, and Summarize across Llama 2/3 and Mistral models show that, starting from a seed RM trained on 15% of labeled data, SER achieves performance close to or exceeding full-dataset training, with an average 7.88% gain over the seed model.

---

## Strengths

- **Two-status curriculum filtering is a specific and well-motivated mechanism**: Rather than applying a single confidence threshold uniformly (vanilla pseudo-labeling), SER distinguishes between an "easy" phase (global discrimination of good vs. bad, Status 1) and a "hard" phase (amplifying differences between similar-quality answers, Status 2). Figure 2 and Figure 3 provide concrete evidence that this transition prevents performance plateauing or regression that occurs when only easy samples are used: Loop 2 (Status 1 with easy data) shows diminishing returns for some models, while Loop 3 (Status 2 with harder data) recovers performance gains. This curriculum structure is not the default in standard self-training and represents a genuine algorithmic contribution.

- **Consistent empirical results across a broad and diverse experimental grid**: The method is evaluated across 4 datasets spanning quite different domains (helpfulness/harmlessness, code Q&A, summarization, instruction-following), 4 model families and sizes (Mistral 7B, Llama 3 8B, Llama 2 13B, Llama 3 70B), yielding consistent and non-trivial improvements over the seed model in every configuration. Table 1 shows average gains of 7.88% over the seed and an average gap to the full-data baseline of only 0.3%, which is a meaningful empirical signal.

- **Downstream PPO validation**: Unlike most reward model papers that stop at RM accuracy, SER validates the evolved RM by using it to guide PPO training and evaluates the resulting LLMs via GPT-4 judgments. SER-trained models consistently outperform SFT baselines and are competitive with (or superior to) PPO trained with full-data RMs (Figure 4), closing the loop on practical utility.

---

## Weaknesses

- **Absence of semi-supervised / vanilla pseudo-labeling baselines — the most critical gap**: SER's core claim is that its *specific* status-based curriculum mechanism provides gains over simply training on 15% of labeled data. However, Table 1 contains no comparison against vanilla pseudo-labeling (e.g., confidence-thresholded self-training without curriculum), mean-teacher, or other standard semi-supervised approaches applied to RM training. Without such baselines, it is impossible to determine whether the observed gains arise from the curriculum structure or from the simpler effect of adding any self-labeled data. This fundamentally limits the attributable novelty of the contribution.

- **No statistical significance reported for key comparisons**: No standard deviations, confidence intervals, or significance tests appear anywhere in Table 1 or Figure 4. Several headline results rest on very small margins — e.g., Mistral 7B surpassing the Full Dataset by 0.13% on HH-RLHF, or SER vs. Full differences of roughly 1% in several cells. With single-run experiments, it is unknown whether these differences exceed noise, which is especially problematic for the claim that SER "can approach or even exceed" full human-labeled data performance.

- **Circular label assignment in Status 2**: In Status 2, pairs are selected where |RM(Q, A¹) - RM(Q, A²)| > δ, and the *sign* of the RM's score difference implicitly determines which answer is treated as preferred in the pairwise loss. The RM thus serves simultaneously as the filter and the label oracle. While pairwise loss provides some robustness and the confidence threshold filters for higher-certainty predictions, this circularity can amplify systematic errors in the RM: if the RM has a consistent bias (e.g., preferring verbosity), Status 2 will reinforce that bias rather than correct it. This is a well-known risk of self-training and is not discussed or experimentally probed in the paper.

- **Source of "unlabeled data" is never made explicit**: The paper never clearly states whether the unlabeled data is the withheld 85% of preference *pairs* (i.e., questions + answer pairs already exist, only labels are withheld) from the same dataset, or genuinely new Q/A pairs from an external source. This distinction is critical: if the unlabeled pool is simply the labeled dataset with labels removed, SER is within-distribution semi-supervised learning over a fixed corpus, and its annotation cost reduction refers only to *labeling* cost, not *data collection* cost. The abstract's claim that "only 15% of human-annotated seed data is required" should be qualified to reflect that all Q/A pairs are still required from the original dataset.

- **Threshold sensitivity is unanalyzed**: The method relies on three non-trivial thresholds (τ_high=0.55, τ_low=0.45, τ_Δ=0.3) and a "sufficient count" of 600, justified only by stating they "provided the most consistent improvements." No ablation over these values is presented. Given that these thresholds determine which data is selected, the proportion of Status 1 vs. Status 2 transitions, and ultimately when training stops, their robustness is central to the method's generalizability. Small perturbations could shift the data selection dramatically.

- **PPO evaluation is statistically fragile**: Figure 4 shows high tie rates (42–71%) and small win-rate differences (e.g., SER: 22% vs. Full: 24% wins for Llama 8B on HH-RLHF). No statistical testing is applied. The GPT-4-as-judge protocol, while standard, introduces position and length biases; no consistency or calibration checks are reported. Claims such as "SER models outperform the full models to a certain extent" are difficult to substantiate given this noise.

- **Theoretical section overpromises**: Section 3.2 describes theoretical results on RM convergence and PPO optimality, but all proofs are deferred to Appendix A (removed from the submitted text). The only substantive claim in the main body — "when initial accuracy exceeds 50%, iterative training with high-confidence samples can further improve performance" — is a mild result that follows naturally from basic self-training arguments. Critically, the Discussion section (Section 5) itself concedes "a rigorous theoretical analysis of its effectiveness is still needed," which undercuts Section 3.2's framing as a theoretical contribution.

- **Data accumulation across iterations may create labeling conflicts**: The paper defines D_filtered = D_filtered^n + D_filtered^{n-1} — data from previous loops is retained and mixed with new data. Because the RM's scores change between iterations, the same pair could receive different relative orderings in different loops. No reconciliation strategy is described or evaluated, and this could introduce noisy or contradictory training signal.

---

## Nice-to-Haves

- **Ablation on thresholds**: Vary τ_high, τ_low, τ_Δ, and the sufficient-count criterion to show the method degrades gracefully; this would significantly strengthen robustness claims.

- **Direct RLAIF comparison**: Including a baseline where a same-sized or slightly larger model annotates the unlabeled pairs (standard RLAIF) would sharpen the contribution, as the motivation explicitly positions SER against RLAIF.

- **Reward score histogram analysis per iteration**: Visualizing reward score distributions across loops would reveal whether the RM is maintaining discriminative power or shifting scores in degenerate ways (reward hacking), addressing the bias amplification concern directly.

- **Self-label noise quantification**: Measuring the error rate of RM-generated pseudo-labels against ground-truth labels per iteration (on a held-out calibration set) would concretely show how noise evolves and validate the theoretical claim that noise stays within tolerable bounds.

- **Status transition breakdown visualization**: Explicitly plotting the proportion of data selected under Status 1 vs. Status 2 across iterations per dataset/model would verify that the curriculum progresses as intended and is not trivially collapsing to one regime.

- **Compute/wall-clock comparison**: Reporting total training time or FLOPs relative to full supervised training would give a complete picture of the efficiency gains, since multiple iterative loops add overhead.

- **Out-of-distribution generalization test**: Evaluating the SER-trained RM on a held-out preference dataset not used in training would assess whether the evolved RM generalizes or overfits to its self-labeled distribution.

- **Discussion of minimum viable seed quality**: The Discussion briefly mentions future failure modes; explicitly characterizing how weak the seed RM can be (e.g., what happens if the initial accuracy is near 50%) before SER collapses would be practical guidance for users.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Critic's claim of "logically inverted" status ordering**: The critic argues that "If the RM does not meet the criteria for Status 1 (easier task), we then check for Status 2" is contradictory. However, the paper's logic is that Status 1 activation means *enough clearly easy pairs* remain in the unlabeled pool for productive training; when the RM has already absorbed those (fewer clear good/bad pairs survive the threshold), Status 1 is no longer satisfiable, and the harder Status 2 takes over. This is a coherent curriculum progression, not a logical inversion. The paper explicitly states "Status 1 is the easier task...Status 2 is the harder task...If the RM does not meet the criteria for Status 1 (i.e., few or no samples satisfy the thresholds), we then check for Status 2." This is internally consistent.

- **Critic's claim about "missing" Figure 6**: The critic notes Figure 6 (RM score distributions) appears to be referenced but missing. This is likely an artifact of the review copy with appendix removed; the main text describes these results and they do not appear to be fabricated.

- **Strength: "The paper is well-written / well-structured"** (Reviewer 2): Generic and applies to any adequately formatted paper; removed.

- **Criticism that SER requires "stronger LLMs" in the same way as RLAIF**: The critic argues the self-labeling requirement (seed RM must have >50% accuracy) is "parallel" to RLAIF's need for a strong AI annotator. Technically true, but the bar for SER (any RM trained on 15% data, no external model required) is materially lower than typical RLAIF setups requiring GPT-4-class models. The distinction is worth noting but does not undermine the motivation.

- **Criticism about the title being "slightly misleading"**: Stylistic preference; removed.

---

## Novel Insights

The most genuinely novel conceptual insight in the paper — and one worth emphasizing — is the *adaptive curriculum in data selection driven by the model's own evolving confidence profile*. Rather than applying a static filtering rule, SER monitors the RM's current probability outputs on the unlabeled pool to determine *which skill the model is ready to develop next*. The empirical evidence in Figure 2 and 3 substantiates that Status 2 data (subtle differences) is not useful early (when the model cannot reliably rank them) but becomes critical later to push past the performance ceiling reached with easy data. This is a meaningful operationalization of curriculum learning in the reward modeling context, and the observation that "similar data becomes marginally helpful or harmful after multiple iterations" (Loop 2 showing weakest gains) is an empirically grounded and underexplored phenomenon in RM training. If supported by ablations against vanilla pseudo-labeling, this mechanism would constitute a concrete design principle for data-efficient reward learning.

---

## Suggestions

1. **Add vanilla pseudo-labeling as a baseline**: Train an RM with 15% labeled data and apply simple confidence-thresholded pseudo-labeling (without status-based curriculum) for the same number of iterations. This is the most critical experiment needed to validate the specific contribution of the two-status mechanism.

2. **Explicitly define the unlabeled data pool in the main text**: State clearly whether unlabeled data consists of held-out preference *pairs* from the same dataset (labels removed) or truly new Q/A instances. Adjust the annotation-cost-reduction claims accordingly.

3. **Report variance or repeat key experiments with multiple seeds**: At minimum, for the headline comparisons in Table 1 (SER vs. Full Dataset), report results over 2–3 runs to establish whether differences are meaningful or within noise.

4. **Add an ablation table for thresholds**: Fix two thresholds and vary the third across a grid; show resulting RM accuracy to quantify sensitivity. Even a 3×3 grid in an appendix would substantially address this concern.

5. **Address bias amplification in Status 2 empirically**: Track the correlation between SER pseudo-labels and ground-truth labels across iterations. If the circular label assignment is not causing error amplification, show it; if it is, discuss mitigation strategies (e.g., label smoothing, ensemble filtering).

6. **Strengthen the theoretical section or reclassify it**: Either provide the key assumptions and sketch of proofs in the main body (moving material from Appendix A), or reframe Section 3.2 as "theoretical motivation" rather than "theoretical analysis," consistent with the Discussion's honest admission that rigorous theory is still needed.

---

## EdKSI2ijUY

- GT: Reject (avg 4.2)
- Predicted: Accept (5.2/10)
- Match: NO

### Final Review

## Summary
LMRL-Gym is a benchmark and open-source research framework for evaluating reinforcement learning algorithms in multi-turn language model interactions. It comprises 8 tasks split between Interactive Dialogue tasks (20 Questions, Guess My City, Car Dealer) and RL Capability tests (Maze, Text-Nav, Wordle, Chess, Endgames), covering both offline and online RL settings. The benchmark is accompanied by implementations of BC, MC Returns, ILQL, and PPO, with the primary goal of enabling rapid iteration on RL algorithm development for LLMs.

---

## Strengths

- **Structured RL capability taxonomy:** The decomposition of tasks along five explicit capability axes (strategic decision-making, credit assignment, partial observability, trajectory stitching, complex language) in Figure 2 is a genuine methodological contribution that distinguishes this benchmark from existing NLP evaluations or text-game suites. Most prior benchmarks evaluate final task performance; this one deliberately exposes algorithmic bottlenecks.

- **Informative empirical contrast between task categories:** The finding that ILQL dominates the RL Capability tasks (especially Maze and Text-Nav) via trajectory stitching, yet is *outperformed by the simpler MC Returns on all three Interactive Dialogue tasks*, is a concrete and non-obvious finding that reveals a meaningful algorithmic gap. This points to a specific research problem: scalable TD-learning for open-vocabulary dialogue.

- **Language vs. symbolic performance gap:** The comparison of RL on the symbolic Maze vs. the language-based Maze—where online/offline Q-learning reaches optimality in the symbolic setting but degrades substantially under partial observability in language—precisely isolates language complexity as a bottleneck, not just RL difficulty in general.

- **Scope and accessibility commitment:** Supporting both offline and online RL, releasing datasets, simulators, code, and hyperparameters, and targeting GPT-2-scale models to ensure accessibility for resource-limited researchers is a substantive service to the community. The dataset sizes (up to 1M for Wordle, 625K for Chess) are large enough to support serious offline RL experimentation.

---

## Weaknesses

- **GPT-2 scale severely limits the conclusions.** All agent models are GPT-2 (up to 1.5B parameters). The paper acknowledges this briefly but frames it as a minor "limitation." In practice, this is a central constraint on every result: the relative ranking of algorithms, the difficulty of tasks, the behavior of ILQL vs. MC Returns, and the failure of Chess may all change substantially at 7B+ scale where emergent capabilities appear. A benchmark paper whose central claim is enabling *algorithmic* progress needs to demonstrate that its algorithmic rankings are at least qualitatively stable across model scale, even on one or two tasks. Without this, LMRL-Gym may be diagnosing properties of GPT-2-scale RL optimization rather than properties of algorithms in general.

- **No variance estimates in Table 2.** All results are single point estimates with no confidence intervals, standard deviations, or multi-seed statistics. Multi-turn RL training—especially PPO—is notoriously unstable. Without error bars, it is impossible to determine whether, e.g., the difference between ILQL (82.9) and MC Returns (87.1) on 20Qs is a meaningful algorithmic signal or run-to-run variance. For a paper whose stated value proposition is enabling *reliable* algorithm comparison, this is a significant methodological gap that undermines the core utility claim.

- **Chess task shows no meaningful RL improvement and is left undiagnosed.** Table 2 shows all methods score between 42.9 and 48.0 on Chess, tightly clustered around the dataset average of 50. No algorithm achieves a normalized score above 50, meaning no method exceeds the average offline trajectory. The paper briefly attributes PPO's slight edge to instability and calls for "better offline TD-based RL methods," but does not investigate *why* RL fails entirely on Chess at this scale—whether it is due to sparse rewards, long horizons, the combinatorially large move space, or the token-level action representation. An uninformative task that no method meaningfully solves undermines the diagnostic claim: it reveals a failure mode but does not illuminate it.

- **ILQL underperformance on dialogue tasks is headline-without-story.** The observation that ILQL underperforms MC Returns on all three dialogue tasks is one of the paper's key findings, yet the analysis is limited to one sentence: "it is harder to scale full TD-learning" on complex text tasks. There is no diagnostic of whether the failure stems from Q/V network instability, distributional shift, hyperparameter sensitivity, or the token-level Bellman backup being poorly conditioned on long open-vocabulary sequences. For a benchmark meant to drive algorithm development, this finding needs a tighter diagnosis so researchers know *what* to fix.

- **GPT-4 scoring exactly 0 on Chess and Endgames is suspicious.** The paper uses this result to conclude that "RL fine-tuning enables goal-directed behaviors that GPT-4 cannot achieve via prompting." However, a frontier model capable of legal chess play getting 0/100 almost certainly reflects a prompting or game-state formatting failure rather than a genuine inability. If the conclusion about RL's superiority over GPT-4 on capability tasks is partly an artifact of poor GPT-4 prompting for these specific game formats, the claim is misleading. The paper should investigate this before drawing strong conclusions.

- **Trajectory stitching claim is asserted but not empirically verified.** The paper designates all RL Capability tasks as testing trajectory stitching capability, but never demonstrates that the best-performing RL policy achieves returns *above any individual trajectory in the offline dataset*. Without showing that RL exceeds the offline dataset ceiling, the stitching property is an assumed design feature, not a demonstrated benchmark characteristic.

---

## Nice-to-Haves

- **Experiments at larger model scale (e.g., 7B).** Even a subset of tasks (e.g., Maze and 20Qs) with Llama-3-8B would substantially strengthen the claim that LMRL-Gym's algorithmic rankings are generalizable and relevant to modern practice.

- **Learning curves with multi-seed variance.** Plotting reward vs. training steps for PPO and ILQL across 3–5 seeds would simultaneously address the variance concern and provide diagnostic insight into PPO instability.

- **Trajectory stitching verification.** Adding a simple analysis showing that top-1% ILQL-generated trajectories exceed the return ceiling of any individual offline trajectory would concretely validate the stitching claim.

- **Additional modern baselines (e.g., Decision Transformer, filtered variants of DPO applied to multi-turn).** Decision Transformer is already cited (Chen et al., 2021a); including it as a baseline would strengthen the algorithmic coverage given it is specifically designed for offline sequence RL.

- **Dataset size ablations for smaller tasks.** Car Dealer (19K) and Maze (1.24K) are 1–2 orders of magnitude smaller than other tasks. A brief analysis of how algorithm rankings shift with data quantity would clarify whether the Car Dealer results reflect task difficulty or data scarcity.

- **Visualizations of successful vs. failed trajectories.** Side-by-side conversation logs where RL succeeds vs. where BC fails would help readers assess whether models are learning genuine multi-turn strategies or exploiting simulator artifacts.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Sim2Real gap" as a weakness (Positive Review, Weakness 1; Spark Finder #2).** The paper explicitly and prominently states: "our goal is *not* to utilize this approach to benchmark whether LLMs are *good at talking to humans*, but rather as a way to test RL algorithms." Criticizing the absence of human-evaluated deployment performance directly contradicts the stated scope of the benchmark. Removed.

- **"Reward function details in appendix" as a weakness (Positive Review, Weakness 3).** Placing reward function details in the appendix is standard practice in ML papers. The table in Section 4 provides summary statistics. Not a genuine weakness.

- **"Missing non-RL agentic baselines (ReAct, CoT, tool-use)" as a weakness (Positive Review, Weakness 4).** The benchmark is explicitly designed to evaluate *RL training algorithms*. Demanding comparisons to inference-time prompting methods is scope creep: a benchmark about X should not be faulted for not evaluating Y. Could be a nice-to-have for supplementary insight but not a weakness.

- **Token-level vs. turn-level MDP as a methodological flaw (Harsh Critic, Section 3).** Token-as-action is the standard formulation in the RL-for-LLMs literature (Snell et al., 2022a; Ziegler et al., 2020; Stiennon et al., 2020). Criticizing this as "the hardest possible formulation without justification" misrepresents the field's standard practice. Removed.

- **"Circular data generation invalidates results" (Harsh Critic, Section 4.1).** The paper is transparent that training and evaluation use the same GPT-2-based simulator family, and explicitly states this design prioritizes accessibility for RL algorithm comparison rather than human fidelity. The user study validating naturalness partially addresses this. The concern about reward hacking is partially addressed by automatic checks for 20Qs/Guess My City and naturalness verification for Car Dealer. While the concern is not baseless, framing it as an invalidating circularity misrepresents the paper's stated purpose. The concern about simulator strategic fidelity (not just linguistic naturalness) is retained as a nice-to-have.

- **"Contributions described narratively, not in bullet form" (Harsh Critic, Introduction).** Pure formatting nitpick. Removed.

---

## Novel Insights

The most penetrating observation across the three reviews—not made explicitly by any one of them—is that the ILQL vs. MC Returns reversal between task categories (ILQL wins on capability tasks, MC Returns wins on dialogue) may be a more fundamental signal than the paper currently treats it: it suggests that Bellman-based TD-learning, despite its theoretical advantages for stitching, encounters a qualitative break when the action space is open-vocabulary and the reward signal is episodic and sparse over long natural-language sequences. This is distinct from simple "scaling difficulty" and points toward a specific algorithmic challenge: how to regularize token-level Q/V functions when dialogue states are high-dimensional and partially unstructured. Combined with the symbolic-vs-language Maze gap, the benchmark is inadvertently revealing that language complexity itself—not just RL difficulty—is a first-class algorithmic bottleneck that existing offline RL methods are not designed to handle. This is a genuinely useful insight for algorithm designers that the paper surfaces empirically but does not synthesize clearly.

---

## Suggestions

1. **Report variance across multiple seeds for Table 2**, even on a subset of tasks. This is the single most important fix for the paper's core utility as a benchmarking tool—without it, observed algorithm differences cannot be trusted.

2. **Diagnose the Chess RL failure more explicitly.** Report the distribution of rewards achieved by each method (not just the mean), and investigate whether the issue is reward sparsity, long horizons, or the token-level move representation. If Chess is genuinely intractable for sub-2B parameter models, say so explicitly and propose a diagnostic protocol for when it becomes tractable.

3. **Verify or retract the GPT-4 Chess/Endgames score.** Ablate the GPT-4 prompting format for Chess (e.g., FEN notation vs. natural language description of board state) to confirm that a score of 0 reflects model capability rather than input formatting. The conclusion about RL superiority depends on this.

4. **Add a brief empirical verification of trajectory stitching.** Show the return distribution of the offline dataset and demonstrate that ILQL-trained policies produce trajectories with returns exceeding the offline ceiling on at least one task. This converts an assumed benchmark property into a demonstrated one.

5. **Provide a more detailed failure analysis of ILQL on dialogue tasks.** Report training curves, KL divergence from the BC initialization, and Q/V function loss curves to diagnose whether the failure is instability, distributional shift, or a fundamental representation issue. This would make the paper's key empirical finding actionable for the community.

---

**Overall assessment:** LMRL-Gym fills a genuine and important gap—there is no existing benchmark that provides multi-task, multi-turn RL evaluation with simulator infrastructure, offline datasets, and explicit algorithmic capability targeting. The task design is thoughtful and the open-source commitment is commendable. However, as currently presented, the empirical foundation is fragile: results are single-run on GPT-2-scale models, the Chess task reveals a failure mode that is not adequately diagnosed, and the paper's most interesting finding (the ILQL/MC Returns reversal) lacks a mechanistic explanation. The paper is a solid infrastructure contribution that falls short of being a definitive empirical study. Addressing the variance reporting, the Chess diagnosis, and the GPT-4 anomaly would substantially strengthen the paper's credibility as a reliable algorithmic testbed.

---

