## Summary
WLA (World modeling through Lie Action) is an unsupervised framework that models environment transitions as Lie group actions operating linearly on object-centric slot-based latent representations. Rather than learning a separate world model per environment, WLA trains a single cross-environment simulator that captures continuous and compositional dynamics; it then solves the Controller Interface Problem (CIP) by learning a lightweight adapter (`Ctrl_adapt`) from labeled action sequences to the learned Lie algebra parameters. The framework is evaluated on Phyre (qualitative), 8 ProcGen game environments, and a real-world Android robotics dataset, showing large improvements over Genie in controllability-specific metrics.

---

## Strengths

- **Single unified model across 8 diverse ProcGen environments.** Most world model work trains per-game; WLA learns one model jointly. The fact that this outperforms a per-environment baseline (Genie) in all games on Δ_t PSNR (from near-zero or negative for Genie to clearly positive for WLA, e.g., coinrun: 0.48 → 9.03; ninja: 0.05 → 4.06; bigfish: −0.09 → 1.26) is a non-trivial result and a concrete empirical confirmation of the cross-environment benefit.

- **Least Action Principle for slot alignment is a creative and effective contribution.** The ablation (Table 1) shows a meaningful performance drop when it is removed (MSE 0.675 → 0.602 on unseen), and it addresses a known failure mode of slot-attention in temporal settings without requiring extra supervision.

- **Two-stage design cleanly separates unsupervised world modeling from supervised controller adaptation.** The pre-trained `(Φ, Ψ)` transfers to a new environment requiring only a small labeled dataset for `Ctrl_adapt`, which is a practically attractive property for the robotics setting.

- **Android FVD result (393.85 → 131.02) demonstrates real-world temporal coherence.** FVD captures distribution-level video quality and long-range consistency; this improvement is substantially larger than might be explained by per-frame reconstruction differences, suggesting the Lie-structured latent model genuinely improves temporal dynamics on unstructured robot video.

---

## Weaknesses

### Fatal
None.

### Major

- **Only Genie is used as a baseline, which prevents attributing gains to specific design choices.** The paper's core hypothesis is that Lie-structured continuous latent actions outperform discrete or unstructured continuous latents. To test this, the comparison needs at least one continuous latent dynamics model without Lie structure (e.g., a slot-attention model with standard recurrent latent transitions, or DreamerV3 adapted for video prediction). Without this control, the observed gains could plausibly come from the object-centric architecture or the state-space formulation alone, rather than the Lie group structure specifically. The two included ablations (w/o rotation, w/o least action) are internal and do not address this.

- **The "minimal or no action labels" claim in the abstract is misleading.** The abstract states WLA "can be trained using only video frames and, with minimal or no action labels, can quickly adapt to new environments with novel action sets." Section 4.3 clarifies that `Ctrl_adapt` requires a labeled dataset `{(x[t], a[t])}`. The "no labels" story applies only to pretraining `(Φ, Ψ)`, not to the full pipeline that actually enables controllable interaction. Crucially, the paper never measures how many labeled trajectories are needed — there is no label-efficiency experiment, so "minimal" is unquantified and the adaptation claim is unverified. This is a central promise of the work and it is unsupported.

- **Cross-environment generalization is not clearly demonstrated.** The paper trains on all 8 ProcGen games jointly and calls unseen evaluation "out-of-domain," but does not clearly specify what "unseen" means: new procedurally generated levels of the same games, or held-out game types? If the former, this is in-distribution generalization, not the cross-environment transfer that motivates the paper. There is no experiment that trains on a subset of games and tests on held-out games, which would directly validate the inter-environmental generalization claim.

### Minor

- **The commutativity assumption (A(s) commute over time) is acknowledged in Section 7 but not empirically characterized.** This assumption underpins the closed-form solution in Eq. (4) and the rollout in Eq. (9), and it is not merely a modeling simplification — it rules out sequences where action order matters (e.g., "jump then run" ≠ "run then jump"). The paper provides no analysis of how much degradation this causes in non-commutative settings, nor any indication of which environments satisfy it approximately.

- **Phyre evaluation is entirely qualitative with no baselines.** Phyre is used as a "sanity check" for interpolation and composition, but no quantitative metrics are reported and no baseline model is compared. This makes it impossible to assess whether the demonstrated behaviors arise from the Lie group structure specifically or from any reasonable generative model with a smooth latent space.

- **Implementation is underspecified for reproducibility.** Critical details are missing: number of slots N, number of Lie actions J, latent dimensionality, learning rate, batch size, rollout length during training, number of training steps, whether hyperparameters are shared across all ProcGen games and Android, and specifically how the Android architecture was "slightly adapted." The footnote in Section 4.2 ("these parameters are not to be stored as parts of the model") hints at an unusual optimization scheme (per-trajectory per-timestep trainable λ, θ), but the relationship between these optimized values and the IDM outputs is not clearly explained. This raises the question of whether the IDM performs amortized inference or is merely post-hoc supervised on memorized trajectory codes.

- **The learned (λ, θ) parameters are not analyzed semantically.** The paper claims WLA learns compositional action primitives, but never demonstrates that individual (λ, θ) dimensions correspond to interpretable actions (e.g., consistent horizontal motion, rotation, etc.) across environments or even within one. Without this, the Lie group parameterization could just be a flexible coordinate system that fits training data without capturing the claimed compositional structure.

- **Notation in Eq. (6) overloads F^{-1}_{Φ,Ψ}.** In Section 3.1, F_{Φ,Ψ} is defined as the IDM mapping observation-space transitions to latent matrix representations (g → M(g)). Its inverse should map latent operators back to observation transitions. But in Eq. (6), F^{-1}_{Φ,Ψ} is used to map Lie algebra parameters (λ, θ) to transition matrices M_{t,δ} via matrix exponentiation — a different operation. These are distinct mathematical maps and the overloading without explanation creates genuine confusion.

- **The commutativity assumption should appear explicitly in Section 3, not only in Section 7.** Eq. (4)'s closed form z(t) = exp(∫A(s)ds)z(0) and Eq. (9)'s rollout both assume A(s) commute across time. This is a non-trivial restriction that shapes the entire framework, but the paper introduces it as a limitation only in the conclusion rather than as a modeling assumption in the technical presentation.

### Tiny

- **Slot alignment description contains a notation ambiguity.** Section 4.4 states: "we choose the permutation σ to the slots so that the transition z_n[t+1] → z_σ(n)[t+1] in the latent space is minimal." This appears to describe matching future-frame slots to other future-frame slots, whereas the intended operation should relate current-frame slots z_n[t] to future-frame slots z_σ(n)[t+1]. The description should be clarified.

- **Eq. (3) order of composition.** The Fact states F_{Φ,Ψ}(h·g) = F_{Φ,Ψ}(g)·F_{Φ,Ψ}(h), which is anti-homomorphic unless a right-action convention is being used. The paper does not explain the convention, which will cause confusion for mathematically oriented readers. A sentence clarifying the action convention would resolve this.

---

## Nice-to-Haves

- A label-efficiency experiment plotting adaptation performance as a function of the number of labeled trajectories (1, 5, 10, 50, ...) would directly validate the "minimal labels" claim and is probably the single most impactful addition.
- An experiment with true cross-game generalization: train on 6 ProcGen games, adapt to 2 held-out games with few labels.
- A continuous latent dynamics baseline (e.g., slot-attention model without Lie constraints, or RSSM-style model) to isolate the contribution of the Lie group structure from the object-centric architecture.
- Analysis of rollout degradation over longer horizons — the Lie group structure theoretically promotes stability and this should be measurable.
- Visualization of slot assignments across frames, verifying that slots track objects consistently over time.
- Hyperparameter sensitivity of N (slots) and J (Lie actions), since these are user-specified and practitioners need guidance.
- Timing/computational cost comparison with Genie, especially for the linear assignment problem solver in slot alignment.
- Failure case analysis showing where WLA breaks (strongly non-commutative dynamics, objects appearing/disappearing, stochastic environments).
- A qualitative visualization showing which (λ, θ) dimensions correspond to which actions, to provide evidence for the "compositional primitives" claim.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **[REMOVED] Concern about Genie baseline unfairness due to increased training iterations.** The paper gave Genie 2× training iterations (0.4M vs. 0.2M default) to accommodate the multi-environment setting. This asymmetry favors the baseline, not the proposed method — it is therefore a more conservative comparison, not an unfair one.

- **[REMOVED] Demand for confidence intervals and multi-seed statistics.** Single-run evaluation is standard for ProcGen-scale benchmarks in world modeling. Requesting this as a weakness imposes a non-standard rigor requirement.

- **[REMOVED] Requests for specific missing related works.** Per review policy, we do not flag missing related works when external sources cannot be verified.

- **[REMOVED] Criticism that the introduction "doesn't pin down novelty sharply."** The paper provides a clear contribution statement and the combination of Lie group structure, object-centric encoding, and cross-environment training is distinguishable from prior work. Demanding a more formal contribution list is stylistic.

- **[REMOVED] Framing that "Valevski et al. lack identity mapping under no action" is too broad.** This is a specific technical characterization the paper makes in related work, not a misrepresentation of Valevski et al.'s goals.

- **[REMOVED] Criticism of no downstream robotic task metric on Android.** The paper is a world model for video prediction; evaluating it by downstream task success would impose an out-of-scope requirement. The video prediction metrics (FVD, PSNR, Δt PSNR) are appropriate for the stated contribution.

- **[REMOVED] General claims that the paper "overclaims"** without specific grounding — several sub-claims the harsh critic calls overclaims are either qualified by the paper or do have experimental support (the large Δt PSNR gains specifically validate controllability improvements relative to Genie).

---

## Novel Insights

The juxtaposition of the three reviews reveals one genuinely important insight beyond the paper's own contributions: the commutativity assumption is the most fundamental constraint in WLA's design, yet it is the least empirically characterized. The paper assumes transitions commute, implements rollout under this assumption (Eq. 9), and notes it as a limitation only in the conclusion — but it never tests what happens in environments where commutativity provably fails. If this assumption fails silently on real game dynamics (where "jump then move" ≠ "move then jump"), it could explain why the model has near-zero performance degradation in Table 1's ablations while still leaving headroom for non-abelian extensions. A targeted commutativity stress-test would not only validate the current scope but also motivate the future work the authors themselves propose.

---

## Suggestions

1. **Run a labeled-data ablation (1 / 5 / 10 / 50 / 100 labeled trajectories for `Ctrl_adapt`).** This is the most important addition — it directly validates the "minimal labels" claim that currently lacks any quantitative support.

2. **Clarify the "unseen" setting precisely.** Specify whether "unseen" in Table 1 means (a) held-out procedurally generated levels of the same games, or (b) held-out game types. If (a), add an experiment with held-out game types to demonstrate true inter-game transfer.

3. **Add one continuous-latent baseline.** A slot-attention model with standard diagonal SSM transitions (no Lie group structure, same architecture otherwise) would directly test the hypothesis that the Lie structure drives the gains.

4. **Add design a commutativity test.** Identify at least one ProcGen environment where action order provably matters, and measure WLA's performance there relative to the in-domain results. This would honestly characterize the scope of the model.

5. **Clarify the IDM/per-trajectory optimization pipeline.** Explicitly state whether {λ_{nj}[t], θ_{nj}[t]} are stored per-trajectory (transductive) or predicted by the IDM (amortized), and explain how the IDM is supervised in the former case.

6. **Fix the Eq. (6) / F^{-1} notation.** Introduce a distinct symbol (e.g., exp_G or π) for the map from Lie algebra parameters to the matrix group element, separate from F^{-1}_{Φ,Ψ} which should denote the observation-space map.

7. **Move the commutativity assumption to Section 3**, adjacent to Eq. (4), and frame it as a modeling assumption rather than a limitation discovered in hindsight.

8. **Include full hyperparameter table in the appendix** covering N, J, latent dimension, training steps, optimizer, learning rate, batch size, rollout length, and any Android-specific modifications.

---

## Evaluation on Key Axes

**Originality:** *High.* The combination of Lie-structured linear latent dynamics, object-centric slot attention, and joint cross-environment world modeling is a distinct contribution. The Least Action Principle for slot alignment is a creative addition not seen in prior work. While individual components (equivariant autoencoders, slot attention, state-space dynamics) are borrowed, their integration and application to multi-environment world modeling is novel.

**Importance of research question:** *High.* Generalizable, controllable world models are a foundational problem for planning and policy learning. Tackling cross-environment generalization without per-environment training is both ambitious and practically motivated.

**Claims well supported:** *Moderate.* The controllability gains over Genie on ProcGen are large and consistent across 8 games, giving good support to the core claim that Lie-structured latents improve action responsiveness. However, the "minimal labels" and "cross-environment generalization" claims are not quantitatively supported and rely on an ambiguous experimental setup.

**Soundness of experiments:** *Moderate.* Results are reported without variance, the unseen/out-of-play protocol is underspecified, only one external baseline is used, and the Phyre section is purely qualitative. The ablations cover two components but do not isolate the Lie structure from the object-centric design.

**Clarity of writing:** *Moderate.* The high-level idea is clearly communicated. However, the mathematical sections have meaningful notation issues (Eq. (3) ordering, Eq. (6) overloading), the role of per-trajectory optimized parameters is ambiguous, and the implementation section lacks sufficient detail for reproduction. The paper explicitly defers rigorous formalism to the appendix (which was not available for review).

**Value to the research community:** *Moderate-to-high.* If the results hold under stronger baselines, WLA would represent a meaningful advance in structured world modeling. The code is implied but not released; the reproducibility gaps currently limit immediate uptake.

**Contextualization relative to prior work:** *Good.* The paper engages substantively with Genie, LAPO, NFT/Koyama et al., and VPT, and clearly articulates how WLA differs from each. The framing around CIP is novel and useful. Some relevant continuous-latent dynamics methods (Koopman operators, DreamerV3) are mentioned only briefly.