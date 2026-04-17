Now I have enough calibration context. Let me write the consolidated review.

## Summary

The paper formalizes the All-Day Multi-Scenes Lifelong VLN (AML-VLN) problem, requiring agents to continually learn across multiple scenes and diverse illumination environments (normal, low-light, scattering, overexposure) without catastrophic forgetting. To address this, the authors propose Tucker Adaptation (TuKA), which represents multi-hierarchical navigation knowledge as a fourth-order tensor decomposed via Tucker decomposition into shared components (core tensor G, encoder U₂, decoder U₁) and scenario-specific experts (U₃ for scenes, U₄ for environments). A Decoupled Knowledge Incremental Learning (DKIL) strategy with EWC regularization, expert consistency, and orthogonal constraints supports continual learning. The paper also extends Habitat with physics-based imaging degradation models to create a 24-task benchmark and develops the AlldayWalker agent.

## Strengths

- **Well-motivated problem formulation.** The AML-VLN setting with explicit scene×environment factorization is practically meaningful and captures a real deployment challenge for embodied agents operating across diverse locations and conditions. The multi-hierarchical structure is a natural and clean formulation.

- **Principled tensor factorization design.** The use of Tucker decomposition to decouple scene-specific and environment-specific knowledge into distinct factor matrices, with a shared core tensor and encoder/decoder, is mathematically elegant. The mapping from the fourth-order tensor back to a 2D update ΔW_t = U₁·(G ×₃ U₃[s,:] ×₄ U₄[e,:])·U₂ᵀ (Eq. 3) creatively solves the dimensional alignment problem with LLM backbones.

- **Substantial empirical evaluation.** The paper compares against 12 baselines across 24 tasks spanning 5 simulation scenes × 4 environments plus 2 real-world scenes, with ablations on tensor order, shared components, scaling to 30 tasks, and generalization to unseen scenarios. The consistent improvements in SR and F-SR over strong baselines like O-LoRA and SD-LoRA are notable.

- **Meaningful benchmark contribution.** Extending Habitat with physics-grounded imaging degradation models (atmospheric scattering, low-light camera response, overexposure saturation) is a tangible resource contribution that enables standardized evaluation of robustness to environmental changes.

- **Real-world validation.** The inclusion of real-world deployment experiments (Table 5, G5–G6) strengthens the practical relevance beyond simulation.

## Weaknesses

### Major

- **Unfair baseline comparison due to asymmetric expert selection mechanism.** At inference, AlldayWalker uses CLIP-based feature matching (§3.4) to select the appropriate scene expert U₃[s,:] and environment expert U₄[e,:]. This provides a structured, per-scenario routing mechanism that none of the baselines possess. A trivially strong baseline would be: train an independent LoRA adapter per (scene, environment) pair and select the adapter via the same CLIP matching at test time. This baseline would have the same expert-selection capability without the Tucker decomposition, and its absence makes it impossible to attribute performance gains to the tensor structure versus the routing mechanism. The comparison as presented conflates the contribution of TuKA's factorized representation with the contribution of structured scenario identification.

- **Overstated claims about inherent limitations of 2D LoRA.** The paper repeatedly asserts that LoRA and MoE-LoRA are "inherently limited" by their 2D form and "restrict them to representing only two hierarchical knowledge structures" (§1, §3.1). However, any Tucker factorization of the form in Eq. 3 ultimately produces the same effective 2D update ΔW_t — the "multi-hierarchical" structure exists only in the parametrization, not in the functional form of the adapter. MoE-LoRA variants could, in principle, incorporate multi-factor routing (e.g., separate gates for scene and environment) to achieve similar structural decoupling. The paper demonstrates that its particular Tucker-parameterized design outperforms existing MoE-LoRA variants, but it does not demonstrate a fundamental representational limitation of matrix-based methods. The ablation comparing third-order vs. fourth-order tensors (§5.3) supports the benefit of structured factorization, but not the overclaim about inherent 2D limitations.

- **Insufficient ablation isolating DKIL components.** The full training loss (Eq. 9) combines four terms: navigation loss, EWC regularizer (L_ewc), expert consistency (L_co), and orthogonal constraint (L_es). Table 3 only ablates shared components (G, U₁, U₂), but does not isolate the individual contributions of L_ewc, L_co, and L_es. Without this, it is unclear which regularization terms drive the forgetting reduction, and the empirical support for the DKIL strategy as a whole is weakened.

- **No variance or task-order sensitivity analysis.** All results are single numbers per task and method (Tables 1–2, Figure 7). Continual learning performance, particularly forgetting metrics, is known to be highly sensitive to task ordering and random seeds. The paper mentions "the order of tasks is randomized" but reports only a single ordering. Given that TuKA introduces significant structural complexity (8 hyperparameters: r₁, r₂, r₃, r₄, M, N, plus λ₁, λ₂, λ₃, ω), robustness to these choices is not established.

### Minor

- **Scalability of expert structure to open-set scenarios.** U₃ ∈ ℝ^{M×r₃} and U₄ ∈ ℝ^{N×r₄} are pre-allocated for a fixed number of scenes (M=7) and environments (N=4). Adding new scenes or environments beyond these pre-allocated dimensions requires restructuring the tensor. The generalization experiment (Table 5) only reuses existing experts for unseen scenes via nearest-neighbor matching — it does not address how to grow the expert pool dynamically. This limits applicability to truly open-ended lifelong learning.

- **CLIP feature matching under degraded environments may be unreliable.** The task-specific expert search (§3.4) relies on CLIP cosine similarity between stored features and test observations. CLIP features are known to degrade under distribution shift (low-light, scattering), which are precisely the conditions AML-VLN targets. No analysis of expert selection accuracy is provided, leaving a critical pipeline component unevaluated.

- **Negative F-SR values unexplained.** Table 2 shows AlldayWalker achieving negative F-SR values (T9=−3%, T18=−4%), meaning the agent outperforms the multi-task oracle M-SR. While regularization can occasionally produce this effect, it raises questions about the metric's sensitivity and whether M-SR is computed identically across methods.

- **Real-world evaluation is limited.** Only 2 real-world scenes with 2 environments (normal, low-light) appear in the generalization test (Table 5), and the full benchmark (Tables 1–2) includes only 2 real-world tasks out of 24. No scattering or overexposure real-world data is evaluated.

## Nice-to-Haves

- A per-component ablation of L_ewc, L_co, and L_es to isolate which DKIL terms matter most.
- An independent-LoRA-per-scenario baseline with CLIP matching, to isolate the contribution of Tucker factorization from the expert-routing benefit.
- Visualization of learned expert vectors (e.g., t-SNE of U₃ rows and U₄ rows) to validate whether scenes and environments actually separate as intended.
- Expert selection accuracy under each environment type, especially low-light and scattering, to assess the reliability of the CLIP-based routing.

## Removed Points

- **Reproducibility concerns about Fisher information computation.** The harsh critic raises underspecification of Fisher computation details (minibatch-level, online vs. end-of-task). While the lack of per-component ablation is a valid concern (kept above), the specific Fisher computation details are implementation-level matters that are standard in EWC-style methods and described in the appendix.

- **Demand for replay-based continual learning baselines.** The Spark reviewer suggests adding experience replay baselines. However, the paper explicitly scopes its method as a rehearsal-free approach using parameter-efficient adaptation, which is a legitimate design choice. Adding replay-based methods would belong to a different family of methods with fundamentally different assumptions (data storage).

- **FSTTA/FeedTTA as inappropriate baselines.** The harsh critic notes these are single-scene adaptation methods, not lifelong methods. However, the paper includes them precisely as point comparisons for "what if we just adapt at test time?" — this is a reasonable ablation point. Keeping them is fine; the key issue is that they are not main lifelong baselines.

- **Formatting and presentation nitpicks.** The harsh critic's section-by-section notes contain several stylistic complaints (e.g., "conceptually shallow" critique of §3.1, missing implementation details on per-layer vs shared tensors). These are removed as they are either vague or would need to be verified in the appendix.

- **Demand for testing on standard VLN benchmarks (R2R, REVERIE).** The paper introduces a new problem setting (AML-VLN) for which standard VLN benchmarks do not have the multi-scene, multi-environment structure. Testing on R2R would not address the paper's stated contribution.

## Novel Insights

The paper's core insight — that navigation knowledge has a natural multi-hierarchical structure (scene × environment) that can be explicitly factorized via Tucker decomposition rather than being flattened into a monolithic low-rank update — is genuinely valuable. The empirical results in Table 3 showing that shared G and U₂ meaningfully improve performance (SR: 53→65 when adding shared G) support the claim that cross-scenario shared knowledge exists and can be consolidated. However, the paper does not sufficiently disentangle whether the gains come from (a) the structured factorization itself, (b) the multi-loss regularization (EWC + consistency + orthogonality), or (c) the expert selection mechanism at inference — making it difficult to attribute success with precision.

## Suggestions

1. **Add a controlled baseline: independent LoRA adapters per (scene, environment) pair with CLIP-based routing at inference.** This is the single most critical missing experiment to validate that Tucker factorization provides benefits beyond structured expert selection.

2. **Run at least 3 random task orderings and report mean ± std** for the main metrics (SR, F-SR across tasks).

3. **Ablate each DKIL loss component** (L_ewc alone, L_co alone, L_es alone, all pairs) to determine which terms drive the forgetting reduction.

4. **Report expert selection accuracy** per environment type (normal, low-light, scattering, overexposure), as this is the linchpin of the inference pipeline.

5. **Soften the claims about "inherent limitations" of 2D LoRA.** The contribution is better framed as "a structured parametrization that encourages multi-hierarchical knowledge decoupling," which is well-supported by the evidence, rather than a fundamental representational limitation of matrices.

## Score and Decision

**Calibration anchors:**
- SD-LoRA (Accept Oral, ~7.5): Novel decoupling of magnitude/direction in LoRA, theoretical analysis, thorough experiments, clear novelty. This paper's Tucker decomposition idea is conceptually novel but the novelty is somewhat lower (Tucker decomposition is a known technique; contribution is in the specific composition for VLN). Evaluation has significant fairness gaps (no controlled baseline for expert routing).
- TAIL (Accept Poster, ~6.2): Applied existing PEFT methods to a new domain with solid experiments but limited novelty. This paper has higher novelty (new problem formulation + tensor approach) but more evaluation concerns.
- LoRTA (Reject, ~4.25): Tucker decomposition for adaptation, rejected for insufficient evidence and weak baselines. This paper has stronger empirical results and a more compelling application domain.
- HyperAdapter (Withdrawn/Reject, ~4.8): Novel adapter framework but with fairness/contribution concerns. This paper has similar concerns about whether gains come from the proposed method or from confounds.

The paper has genuine strengths: a well-motivated problem, an elegant factorization design, comprehensive baseline coverage, and a useful benchmark extension. However, the major weaknesses — the uncontrolled expert-selection confound, the overclaiming about 2D limitations, and the insufficient ablation of DKIL components — significantly undermine the core claims. The strongest claim ("TuKA fundamentally overcomes inherent limitations of 2D LoRA") is not supported; the more modest claim ("a Tucker-parameterized adapter with specialized routing works well on this benchmark") is supported but cannot be disentangled from the routing advantage. This falls in the range of borderline papers that are promising but need additional experiments to validate claims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>