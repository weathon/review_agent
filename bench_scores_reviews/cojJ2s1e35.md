## Summary
WLA (World modeling through Lie Action) introduces an unsupervised framework that models video transitions as linear Lie group actions in an object-centric, slot-partitioned latent space, enabling continuous and compositional dynamics. A single shared model is trained across multiple environments simultaneously, and a lightweight supervised adapter (`Ctrl_adapt`) maps user-specified action labels to the learned Lie algebra parameters. The method is evaluated on Phyre (qualitatively), ProcGen (quantitatively against Genie on 8 games), and the 1X Android robotic dataset.

---

## Strengths

- **Single multi-environment model with strong quantitative gains:** The paper trains one model across all 8 ProcGen environments and outperforms Genie substantially on both temporal metrics (Δ_t PSNR: e.g., 9.03 vs. 0.48 in coinrun; 4.06 vs. 0.05 in ninja) and LPIPS in 7 of 8 environments (Table 2). This is a concrete demonstration of cross-environment representation sharing, not merely an architectural story.

- **Novel synthesis of Lie group structure with object-centric modeling:** The combination of slot attention with per-slot Lie algebra dynamics—where each slot evolves under structured rotation+scaling operators—is architecturally distinct from both pure slot models and generic Koopman/state-space approaches. The explicit connection to equivariant autoencoders (Eq. 2) provides a principled theoretical grounding.

- **Least-action slot alignment principle:** The proposal to resolve temporal slot permutations by solving a linear assignment problem that minimizes the Lie-algebra operator norm is novel and motivated. The ablation (Table 1) confirms it reduces MSE meaningfully on both seen and unseen environments.

- **Compelling Android robotics results:** The dramatically better FVD (131.02 vs. 393.85 for Genie, Table 3) and better Δ_t PSNR on real-world robot video indicate that the temporal coherence advantages generalize beyond synthetic game environments. The tradeoff (worse per-frame PSNR but much better FVD) is consistent with the hypothesis that WLA better captures action-conditional dynamics rather than static frame quality.

- **Unsupervised pretraining + modular adapter design:** The decoupling of structure learning (unsupervised, label-free) from action mapping (`Ctrl_adapt`, small and supervised) is a clean and reusable design. The ablated version without rotation is explicitly noted to resemble diagonal-SSM models (Mamba), situating the contribution clearly.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Unseen-environment evaluation is critically thin.** The headline claim of cross-environment generalization is only partially supported. Table 1 (right) reports ActionACC for unseen ProcGen environments, but full metrics (PSNR, Δ_t PSNR, LPIPS) for unseen settings are absent from the main paper. This is a direct evidentiary gap: the core generalization claim rests almost entirely on the seen-environment results in Table 2.

- **Single baseline throughout.** The paper compares exclusively against Genie in all experiments. For an ICLR submission claiming a new structured-dynamics framework, this is insufficient. Critically absent: (a) an ablation with general linear (non-Lie) transition operators to isolate the rotation+scaling bias contribution; (b) an object-centric video predictor with MLP transitions to separate the object-centric contribution from the Lie structure contribution; and (c) a continuous latent-action model (e.g., LAPO, which directly addresses unsupervised action discovery). Without these, the source of WLA's improvements—whether structure, object-centricity, or something else—cannot be attributed.

- **No few-shot label scaling experiment despite "minimal labels" claim.** The abstract and introduction prominently claim "minimal or no action labels" for adaptation, but Section 4.3 and the experiments never vary the number of labeled sequences used to train `Ctrl_adapt`, report the actual label budget used, or show a performance curve as a function of label count. This claim cannot be evaluated, making it the most poorly substantiated assertion in the paper.

- **Long-horizon rollout degradation is unanalyzed.** All quantitative results appear to be over fixed 16-step rollouts. The exponential latent dynamics (Eq. 9) could compound errors at longer horizons, yet no rollout-length vs. error analysis is presented. For a paper positioned as a "world model," this analysis is essential to understand practical utility.

- **Training details for per-sample (λ, θ) parameters are insufficient for reproducibility.** Footnote 3 notes that these trajectory-specific Lie algebra parameters "are not to be stored as parts of the model," implying per-trajectory optimization during training. However, the paper never explains how these are initialized, whether they are optimized jointly with network weights at every gradient step, what the computational overhead is, or whether there is any test-time inference procedure. These are essential for reproducibility and for understanding the effective inductive bias.

- **Ablation study is incomplete.** Only two ablations are tested (rotation and least-action). Missing ablations that are necessary to understand the contribution: (a) no object slots (fully dense latent vs. slotted), (b) no shared cross-environment training (per-environment model vs. joint), and (c) shared vs. per-environment `Ctrl_adapt`. These are needed to verify whether the object-centric decomposition and the cross-environment sharing are each adding value.

### Minor

- **ActionACC values are low in absolute terms and the scale is under-explained.** The paper reports 21.07 (Ours) vs. 10.25 (Genie) for seen-environment ActionACC (Table 1 right). While WLA doubles Genie's score, 21% absolute accuracy on a classification task is low, and the number of action classes is not stated in the main text. The paper should clarify the chance-level performance for context. If there are 5 classes, 21% is near chance; if there are 15 classes, it is better. This is important for interpreting whether `Ctrl_adapt` is actually learning useful action correspondence.

- **Phyre evidence is entirely qualitative.** Phyre is presented as validation of the core continuity and compositionality claims, yet only cherry-picked frames are shown (Figures 3 and 4). There is no numerical interpolation error, no composition error metric, and no baseline comparison. This weakens the foundational empirical support for the paper's inductive bias.

- **Android experimental protocol is underdescribed.** The paper says the architecture was "slightly adapted" for the Android dataset without specifying what was changed, what action space is used, or how sequences are split. Without this, the Android results cannot be reproduced.

- **Identifiability of (λ, θ) not discussed.** Nothing in the training objective (Eq. 7–9) prevents the encoder from absorbing dynamics while the Lie algebra parameters become weakly meaningful, or produces unique/stable representations across runs. The paper should discuss whether the learned parameters are stable or if degenerate solutions are observed.

- **Eq. (3) ordering notation.** The Fact states F(h·g) = F(g)·F(h), reversing the standard group homomorphism order M(hg) = M(h)M(g). Whether this is intentional (e.g., right action convention) or an error should be clarified explicitly, as it affects how compositionality is interpreted.

- **Key hyperparameters N and J absent from main text.** The number of slots and rotation components used in each experiment are relegated to the appendix but are central to understanding model capacity and reproducibility.

### Tiny

- The claim "the first of its kind as a generative interactive framework that is based on a state-space model" (Section 7) is overreaching and should be softened or precisely scoped given the substantial related SSM/Koopman literature.
- The commutativity assumption in Eq. 9 (∑A[ℓ] inside the matrix exponential) should be foregrounded in Section 4 alongside the formal equations where it is used, rather than deferred to the limitations in Section 7.

---

## Nice-to-Haves

- **Stochastic extension.** Extending the Lie algebra parameters (λ, θ) to distributions (e.g., Gaussian) would address environmental stochasticity—a stated limitation—and broaden applicability to RL settings. This is noted as future work and would be a natural extension.
- **Visualization of (λ, θ) trajectories alongside ground-truth actions.** Showing whether learned Lie algebra parameters cluster by action type or are interpretably disentangled would strengthen the "compositional and continuous action representation" claim and provide important mechanistic insight.
- **Commutativity violation analysis.** A controlled experiment measuring composition error when ground-truth action sequences are explicitly non-commutative (e.g., "up then right" vs. "right then up") would quantify when the core assumption holds and when the model is expected to fail.
- **Sensitivity analysis for N and J.** A brief sweep over the number of slots and rotation components would allow future practitioners to set these hyperparameters for new domains.

---

## Removed Points
*These points are flagged as removed; treat them with caution.*

- **[REMOVED] Genie comparison unfairness (doubled training iterations).** The harsh critic raises the concern that Genie was given 0.4M training iterations instead of the original 0.2M. Per review policy, comparisons that are asymmetric in favor of the baseline (Genie receives more compute) are intentionally stronger baselines and do not constitute a weakness of the paper. The authors explicitly state this was done to accommodate multi-environment training.

- **[REMOVED] No reconstruction loss per frame.** The harsh critic claims there is no reconstruction loss on x[t] independently of the prediction loss. This is incorrect: the forward and backward prediction losses (Eq. 8) include reconstruction of all frames x[t] via rolled-out latent dynamics, which amounts to frame-level reconstruction supervision.

- **[REMOVED] Statistical rigor / confidence intervals.** Requesting confidence intervals or multiple-seed results for ProcGen evaluations is not standard practice in the video generation and world modeling community, where single-run evaluation on fixed benchmarks is the norm.

- **[REMOVED] Missing related works.** Specific related works were requested by reviewers; per policy, we do not evaluate claims about missing citations without access to external literature.

- **[REMOVED — formatting/scope] Formalism mismatch in CIP (Eq. 1 type signature vs. history input).** The paper explicitly acknowledges the abuse of notation in the text following Eq. (1) and Section A provides a formal definition. While the mismatch is slightly confusing, it is an acknowledged notation convenience, not a substantive error.

- **[REMOVED — acknowledged] Deterministic environment assumption.** Fully acknowledged as a limitation in Section 7 with a proposed future direction (stochastic process modeling). It is a real constraint but not a hidden flaw.

- **[REMOVED] Human analogy in introduction as scientific evidence.** The harsh critic flags the human analogy ("after mastering basic movements in a few 2D action-adventure games…") as unscientific. This is standard motivational framing, not a methodological claim, and is appropriately cited with cognitive science references.

---

## Novel Insights

The most genuinely novel structural insight in this paper—underemphasized even by the authors—is the connection between the ablated "w/o rotation" WLA variant and diagonal-state-space models like Mamba. By explicitly identifying that restricting to scaling-only Lie group actions collapses the framework to a diagonal SSM, the paper provides a principled generalization of the SSM family toward richer, non-diagonal structured dynamics. This framing suggests that the rotation+scaling Lie group structure is not just an arbitrary inductive bias but a specific augmentation of the SSM with rotational degrees of freedom in the latent space, which could motivate a broader class of structured world models. The spark finder's observation that the (λ, θ) parameters could be interpreted and visualized to verify whether the Lie algebra dimensions correspond to semantically meaningful action axes (e.g., θ for orientation, λ for speed/magnitude) is a valuable diagnostic not pursued in the paper—its absence is a missed opportunity to substantiate the "compositional representation" claim mechanistically.

---

## Suggestions

1. **Report full unseen-environment metrics.** Add PSNR, Δ_t PSNR, and LPIPS for out-of-domain ProcGen environments to Table 2 or a companion table. The data likely already exist given the setup.
2. **Add a label-efficiency experiment.** Plot `Ctrl_adapt` performance (Δ_t PSNR or ActionACC) as a function of the number of labeled trajectories (e.g., 1, 5, 10, 50, 100). This single experiment would directly validate the "minimal labels" claim.
3. **Add at least one structural ablation baseline.** Either a general linear (non-rotation-constrained) transition operator, or an object-centric MLP dynamics model, is needed to isolate the Lie group contribution from the object-centric contribution.
4. **Clarify and quantify the per-sample (λ, θ) training procedure.** Explain whether these are optimized per-batch via gradient steps, amortized via a recognition network, or otherwise. Include training time and memory requirements relative to Genie.
5. **Clarify ActionACC scale.** State the number of action classes, chance-level accuracy, and compute a normalized metric (e.g., accuracy above chance) so readers can interpret the absolute numbers in Table 1.
6. **Include long-horizon rollout analysis.** Report a metric (e.g., MSE or PSNR) as a function of rollout length (e.g., 4, 8, 16, 32 steps) for at least one ProcGen environment to characterize error accumulation.
7. **Visualize learned (λ, θ) parameters.** Show scatter plots or trajectory plots of the inferred Lie algebra parameters colored by ground-truth action class for a ProcGen environment to test whether the latent action space is disentangled and semantically interpretable.