Now let me verify one more detail. The harsh critic's weakness 2 says "WLA wins on Δt PSNR and FVD but loses to Genie on absolute PSNR" — this is directly contradicted by Table 2, which shows WLA outperforms Genie on PSNR in ALL 8 ProcGen environments. The paper's own text confirms: "Our method consistently outperformed Genie across multiple metrics." Let me verify the Android dataset (Table 3): Genie 21.16 vs WLA 20.82 — so WLA loses only on Android PSNR, but that's what the paper acknowledges: "While Genie produces cleaner predictions for individual frames..."

I have now verified all key claims. Let me write the consolidated review.

## Summary

This paper introduces WLA (World modeling through Lie Action), a framework for interactive world modeling that leverages Lie group theory and object-centric encoding to learn continuous, compositional action representations. By modeling latent transitions as continuous-time ODEs parameterized by Lie algebra elements, the method enables smooth interpolation and compositional action reasoning across multiple environments without requiring action labels for the core simulator. The model is trained across both synthetic 2D games (ProcGen, Phyre) and real-world robotics video (1X Android dataset), and a separate controller adaptation module ($\text{Ctrl}_{\text{adapt}}$) maps discrete action labels to Lie algebra parameters. Results show consistent improvements over Genie on most metrics across all 8 ProcGen environments and strong temporal coherence on the Android dataset.

## Strengths

- **Strong empirical performance across diverse environments**: Table 2 shows WLA outperforms Genie on absolute PSNR in all 8 ProcGen environments (e.g., coinrun: 22.10 vs 11.30, caveflyer: 17.59 vs 11.25) and on $\Delta_t$ PSNR in all 8. On the 1X Android dataset, WLA achieves substantially better FVD (131.02 vs 393.85) while being competitive on PSNR (20.82 vs 21.16). Table 1 also shows WLA roughly doubles Genie's ActionACC on both seen and unseen environments (21.07 vs 10.25, 14.62 vs 8.30).

- **Principled mathematical formulation with continuity guarantees**: The construction of latent transitions as $z(t) = \exp(\int_0^t A(s)ds)z(0)$ (Eq. 4) using Lie algebra elements provides a formal framework for continuous dynamics, where $\lim_{\delta \to 0} \mathcal{F}_{\Phi,\Psi}(g_{t,\delta}) = I$ ensures the identity property. This is evidenced empirically by Figure 3, where a 1 FPS-trained model generates physically plausible 8 FPS interpolations.

- **Ablation study validates key design choices**: Table 1 (left) confirms that removing rotational components increases MSE from 0.602 to 0.683 (unseen), and removing the least-action principle increases MSE to 0.675, demonstrating that both the Lie group structure and temporal slot alignment contribute substantively.

- **Unified multi-environment training protocol**: Training a single model across all ProcGen environments while maintaining strong per-environment performance demonstrates genuine inter-environmental modeling capability rather than environment-specific overfitting.

## Weaknesses

### Fatal

None.

### Major

- **Commuting (abelian) latent transitions limit applicability to non-commutative action spaces** — The transition matrices $M_{k,t,\delta}$ are 2×2 blocks combining scaling and rotation (Eq. 5), which are isomorphic to complex numbers and therefore commute. The authors acknowledge this limitation explicitly in Section 7: "we assume a priori that transitions in the environment commute with each other." While commutation may be irrelevant for single-action-per-timestep prediction (since different $M_{t,\delta}$ can be applied at different times), this constraint prevents the model from capturing interaction effects between simultaneous or near-simultaneous actions that don't commute (e.g., "move and rotate" in a rigid-body simulator, or simultaneous joint actuation in robotics). Since the paper claims to model "compositional dynamics across environments" and includes a real-world robotics dataset where non-commutative actions are common, this structural constraint limits the scope of what "compositionality" means in this framework. The ablation study (Table 1) confirms the rotation component is needed, but the algebraic structure cannot represent non-commuting compositions — it can only approximate sequential commutative transitions.

- **$\Delta_t$ PSNR metric does not directly measure predictive accuracy, and the paper's strongest claim about "controllability" rests on it** — $\Delta_t$ PSNR = PSNR(ground truth, Ctrl(x, true_action)) − PSNR(ground truth, Ctrl(x, random_action)) measures *action sensitivity* (how much better the model does with true vs. random actions), not absolute prediction quality. On Android, WLA actually slightly loses on absolute PSNR (20.82 vs 21.16) while winning on $\Delta_t$ PSNR (1.13 vs 0.78) and FVD (131.02 vs 393.85). The paper itself acknowledges this: "While Genie produces cleaner predictions for individual frames, it falls behind WLA in generating video sequences that align with the provided action sequences." This suggests WLA may generate temporally coherent but less individually accurate frames — a valid tradeoff, but the claim of "superior controllability" should be more precisely framed, as the underlying metric is about relative action responsiveness rather than the quality of predictions themselves. The empirical evidence still supports WLA's usefulness (strong FVD, strong ProcGen results), but the headline framing slightly overclaims.

### Minor

- **No visual validation of slot disentanglement on the Android dataset** — The model's mechanism depends on the encoder partitioning observations into distinct object slots (Section 3.2, Eq. 5) so that independent Lie group actions can be applied per slot. The paper states (Section 6.3) that they "slightly adapted the architecture" for the 1X Android dataset but provides no architectural details, and crucially, no visual evidence (e.g., attention masks or slot visualizations) that the slot attention mechanism successfully separates dynamic objects from complex backgrounds in real-world video. Without this, it's unclear whether the Lie group per-slot mechanism is actually operating as designed or whether the performance gains come from the ViT decoder capacity. This is particularly important given the documented failure modes of slot attention on high-clutter real-world data.

- **Limited quantitative validation of compositionality on Phyre** — Figures 3 and 4 provide compelling qualitative evidence of interpolation and action composition, but no quantitative error rates (e.g., frame error vs. interpolation resolution, or error when applying swapped action sequences). As qualitative demonstrations, they support the continuity and compositionality claims, but the lack of quantification makes it hard to assess the practical limits of these capabilities.

### Trivial

None.

## Nice-to-Haves

- Report the total training compute budget and iteration count for WLA, as the paper specifies the increased Genie budget (0.2M → 0.4M) but does not state WLA's equivalent, making it harder to assess whether the comparison is compute-fair.

- Evaluate on intentional non-commutative action sequences (e.g., applying action A then B vs. B then A) to empirically quantify the impact of the abelian assumption.

- Probe the semantic meaning of the learned $(\lambda, \theta)$ dimensions (e.g., via clustering or T-SNE) to show whether specific latent parameters consistently map to meaningful action primitives.

## Removed Points

The following points from the harsh reviewer are flagged to be removed or substantially downweighted:

1. **"Structural: Abelian assumption fundamentally invalidates compositional modeling claim"** (downweighted from fatal → major): The critic claims commutation "mathematically cannot represent action order dependency" and "directly contradicts" the paper's claims. This misreads the paper's sequential architecture. At each timestep, the model predicts a block-diagonal matrix $M_{t,\delta}$; different action sequences produce different matrices at different times, so sequential order *is* preserved. The commuting assumption limits *simultaneous* action interactions, not sequential ordering. The paper acknowledges the limitation and proposes it as future work.

2. **"$\Delta_t$ PSNR measures input sensitivity, not controllability accuracy — renders empirical foundation unconvincing"** (downweighted from major → major, reframed): The critic argues that a "model that ignores actions completely would score near zero" and one "that produces chaotic pixel noise would score very high." While $\Delta_t$ PSNR does measure sensitivity rather than accuracy, the paper's ProcGen results (Table 2) show WLA also wins on *absolute* PSNR across all 8 environments, so the "noisy generation" hypothesis is directly contradicted by the data. The concern is valid as a metric interpretation caveat, not as an invalidation of results.

3. **"Slot-attention disentanglement unsupported — performance gains could stem from ViT decoder capacity"** (retained as minor weakness, downweighted): Valid concern about lack of visual evidence on Android, but the claim that gains "could equally stem from" decoder capacity is an alternative hypothesis, not established fact. The ablation study (Table 1) showing that removing Lie group structure and slot alignment degrades performance partially addresses this.

4. **"Abstract claims minimal or no action labels but Section 3.3 requires labeled sequences"** (removed — the paper is actually consistent): The abstract states the simulator "can be trained using only video frames" and that "with minimal or no action labels, can quickly adapt to new environments." Section 4.2 confirms unsupervised training of $(\Phi, \Psi)$ using trajectories only, while Section 4.3 uses action labels only for the optional Ctrl_adapt adapter. This is a two-stage design, not a contradiction.

5. **"ActionACC scores (14.62, 21.07) are negligible if percentages"** (removed — misreads results): WLA doubles Genie's performance (14.62 vs 8.30, 21.07 vs 10.25). Even 14–21% accuracy from a logistic regression on continuous parameters vs. discrete ground truth labels is a meaningful gap, and the *relative* improvement substantiates the core claim.

6. **"Slot attention uses least action principle but ignores identity swapping between visually similar objects"** (nice-to-have, not a weakness): The paper's least-action principle already handles permutation consistency via the Hungarian algorithm, per Section 4.4 and the cited Zhao et al. (2023). Identity swapping across severe occlusion is a known open problem in slot attention broadly, not specific to this paper.

7. **Reproducibility nitpicks about hidden hyperparameters and training logs** (removed per hard rules).

## Novel Insights

The paper sits at an interesting intersection of geometric deep learning (Lie group symmetry, equivariant representations) and interactive world modeling (Genie, LAPO). Its core contribution — parameterizing world model transitions via continuous Lie algebra ODEs rather than discrete autoregressive tokens — is genuinely novel in the world model space. However, the gap between the mathematical ideal (equivariant autoencoder guaranteeing $g \cdot x = \Psi(M(g)\Phi(x))$) and the practical implementation (standard slot attention + MSE training, with no explicit equivariance regularization) suggests the Lie structure may be an *emergent approximation* learned through brute-force optimization rather than a *structural guarantee* as formally described. Whether this matters depends on whether the mathematical framework still provides inductive bias that improves generalization (the ablation study suggests yes) or if it's primarily a post-hoc interpretation of a model that would work similarly with diagonal SSM transitions (the ablation also provides partial evidence against this, since removing rotations increases MSE). The commuting assumption, acknowledged by the authors, is the clearest case where the elegant mathematical structure imposes a real structural limitation on the model class.

## Suggestions

1. **Clarify the scope of "compositionality"**: The paper should explicitly define what "compositional" means within the commuting constraint — i.e., compositionality over *sequential* transitions rather than *simultaneous* non-commuting actions. This reframes the claim from an overreach to a precise statement.

2. **Report WLA's training iteration count and compute budget** alongside the Genie comparison (0.4M iterations) so readers can assess whether the comparison is fair.

3. **Add slot visualization for Android**: Even a single figure showing slot attention masks on 1X Android frames (demonstrating that slots track dynamic objects rather than bleeding into background) would significantly strengthen the claim that the object-centric Lie mechanism is operating as designed.

4. **Add a quantitative compositionality metric on Phyre**: For Figures 3–4, report interpolation MSE at different frame rates and composition error (applying summed Lie parameters to different slots) to provide quantitative backing for the qualitative demonstrations.

5. **Clarify the equivariance gap**: The paper cites Koyama et al. (2024) for equivariant autoencoder theory but uses standard slot attention. A brief discussion acknowledging this gap — and explaining why training approximates sufficient equivariance in practice — would strengthen methodological rigor.

## Score and Decision

**Calibration anchors consulted:**
- **High-scoring (avg 7–8)**: LAPO (rvUq3cxpDF, scores 6,8,8,8, Spotlight) — strong ProcGen results with clear methodology and good novelty. Object-centric papers r9FsiXZxZt and kZvor5aaz7 scored 8s for strong empirical validation.
- **Mid-range (avg 5–6)**: Several Lie-group/equivariance papers (h3Buc7hXSR at 3,6,6,3; VXKt1lwysO at 6,5,3,6,6) were rejected or withdrawn due to fundamental mathematical misunderstandings, toy experiments, or unfair comparisons. The WLA paper is clearly stronger than these — it has real multi-environment results with a stronger baseline.
- **Low-scoring (avg 3)**: Papers like FwkYeLovHk and gS0XOu0JKs were scored ~3 for fundamental methodological flaws and weak evidence. WLA's experiments are genuinely stronger.

WLA has stronger empirical results than most borderline-5 papers and comparable breadth to the LAPO spotlight paper. However, the commuting assumption and the metric interpretation concern are real structural weaknesses that prevent it from reaching the 7–8 range. The paper is better than the rejected Lie-group papers because its experiments are substantial and the comparison to Genie is meaningful, but it falls short of LAPO's clarity and completeness due to the acknowledged structural limitation and the gap between mathematical framing and practical implementation.

Compared to the 6-accept anchors: WLA has similar empirical strength but with an acknowledged methodological limitation (commuting assumption) that the LAPO paper does not have. This pushes it slightly below a strong 6. Compared to the 5-reject anchors: WLA's experiments are substantially more convincing and the core contribution is more clearly established.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>