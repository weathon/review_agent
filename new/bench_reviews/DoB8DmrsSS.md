Now I have enough information to write the review. Let me carefully consider the strengths and weaknesses.

### Key Issues Verification:

**1. Unfair comparison with l_p-constrained baselines?**

The paper does compare SHIFT against l_p-constrained attacks (PGD, MinBest, PA-AD). The harsh critic claims this is an unfair comparison since SHIFT operates without an l_p constraint. However, this comparison is explicitly part of the paper's core argument: that l_p constraints are too restrictive for attacks, and defenses designed against l_p-bounded attacks can be defeated by unrestricted ones. The paper also includes comparisons with high-sensitivity direction attacks (Korkmaz, 2023) that go beyond l_p constraints (Blurred, Shifting). The claim that SHIFT "breaks" defenses is not really about beating l_p-constrained attacks in a fair competition—it's showing that defenses vulnerable only to a specific threat model can be circumvented. This is a valid and important contribution. The comparison against l_p-bounded attacks is showing the *motivation*, not claiming to beat them on their own terms. The Figure 1c vs 1e comparison is about *qualitative* differences, which is valid. 

However, there IS a valid point here: the paper doesn't compare against other unrestricted attacks. There's the "Uncond." diffusion model baseline in Figure 1, but it doesn't have policy guidance. The Korkmaz attacks are included, which do go beyond l_p norms. So the comparison isn't completely one-sided.

But the main claim in Table 1 is about breaking defenses, where SHIFT is compared against all listed defenses. The l_p attacks are just showing they fail against those defenses. The paper's story is clear: existing defenses were designed for l_p-bounded threats, and SHIFT proposes a new threat model. Whether this is a "fair" comparison depends on interpretation—it's fair if you accept that the paper's contribution is identifying a *new threat model*, but it would be unfair if you interpret the paper as claiming SHIFT is a strictly "better" attack in general.

The paper *does* acknowledge this is a different threat model (Section 2.2: "we consider a novel attack similar to unrestricted adversarial examples"). The harsh critic's point is partially valid but overblown. The paper could do a better job with a baseline that is also unrestricted but simpler (e.g., unconditional diffusion) to isolate the contribution of policy guidance and history conditioning. Actually, the "Uncond." row in Figure 1 provides this for qualitative comparison, but there's no quantitative table for it.

**2. Self-referential stealthiness evaluation?**

The harsh critic is correct that the autoencoder used for stealthiness evaluation is trained by the attacker. However, the paper explicitly describes this: "the attacker leverages training data sampled from a clean environment to train an autoencoder" (Section 3.2.3), and then uses "Reconstruction Error: the l2 distance ||s̃_t − AE(s̃_t)||_2" as a metric. The autoencoder is used *both* for guidance AND for evaluation, which is indeed circular. The harsh critic is right that this is a weakness.

However, the autoencoder also serves as the realism correction mechanism during generation, and showing low reconstruction error does demonstrate that the perturbations are "in-distribution" relative to the training data. A truly unrealistic perturbation would have high reconstruction error regardless. The Wasserstein distance is a different metric and isn't self-referential in the same way.

This is a valid critique but somewhat mitigated by the fact that: (1) the autoencoder being used as both guidance and evaluation is standard in adversarial work, (2) the reconstruction error still captures meaningful information about distributional coverage even if the metric is correlated.

**3. Intractable definitions?**

The paper clearly acknowledges in Section 3.2.2 that "it is difficult to identify the projection onto the valid states S*." The definitions are meant as formal objectives, not as implementable procedures. This is common in formal frameworks—definitions provide structure even if exact computation is intractable. The paper then uses approximations (classifier-free guidance, autoencoder guidance) which are explicitly connected to the objectives. This critique is somewhat valid but overstated. The gap between formal definitions and implemented approximations is acknowledged and is typical of such work.

**4. High variance in Table 1?**

Looking at the actual numbers: DP-DQN on Pong: 0.5 ± 11.4; SA-DQN on RoadRunner: 260 ± 215; DP-DQN on RoadRunner: 360 ± 321. These standard deviations are indeed very large relative to means, which raises reliability concerns. This is a valid point.

**5. Only 4 Atari environments?**

This is a valid minor concern. 4 environments is somewhat limited but reasonable for an ICLR paper.

**6. Theorem 1 novelty?**

The harsh critic claims Theorem 1 is "a straightforward application of standard diffusion guidance results." Let me check: Theorem 1 combines classifier guidance and classifier-free guidance for the RL setting. The key insight is that the two conditioning variables factor: τ_{t-1} affects only the diffusion model's prediction, while ā_t affects only the policy gradient. This is because the policy π only depends on the current state, not the history directly. While this factorization is straightforward, it's a necessary formal result for the method to work. Whether this is "novel" is debatable. I'd rate this as a minor weakness—the theorem is correct and useful but not a major theoretical contribution.

**7. Attacker assumptions?**

The attacker has: access to clean environment, victim's policy π, true states in real-time, and no l_p constraint. These are strong but explicitly stated and common in white-box attack papers. This is more of a scope limitation discussion point than a weakness.

Now let me calibrate:

**Calibration anchors:**

1. **F5dhGCdyYh** (Illusory Attacks on RL, avg 7.33, Accept spotlight): Very similar topic area. It proposes a novel threat model (bounded detectability) for RL adversarial attacks, with information-theoretic foundations and experiments. This paper (SHIFT) is in a similar space but doesn't have the same theoretical rigor for stealthiness and uses a self-referential evaluation. SHIFT's contribution is more about the diffusion-based attack method rather than the theoretical framework.

2. **6qeCyvlJUJ** (EvoSeed, avg 3.67, Reject): Uses diffusion models for adversarial examples. Has similar issues with comparison (different threat model from standard attacks), but the overall contribution is more limited. SHIFT has more novel structure (history conditioning + policy guidance + realism guidance).

3. **wZWTHU7AsQ** (Game-Theoretic Robust RL, avg 5.33, Accept poster): Novel but incremental approach to robust RL with temporally-coupled perturbations. Has issues with novelty and limited evaluation.

4. **mXpNp8MMr5** (Two-faced attacks, avg 7.33, Accept poster): Novel threat model exposing a vulnerability in adversarial training.

5. **EQAHilKZ8D** (Glass-like objects, avg 2.2, Reject): Extremely weak paper, overclaimed, trivial method.

SHIFT is stronger than EvoSeed (better designed, more thorough experiments, better motivating story) but weaker than Illusory Attacks (weaker theoretical stealthiness framework, self-referential evaluation). The unfair comparison issue is real but partially mitigated by the paper's explicit framing as a new threat model.

I'd place SHIFT in the 5-6 range: a solid contribution that identifies an important threat model for RL defenses, with a reasonable technical approach (diffusion + policy guidance + history conditioning + autoencoder guidance), but with significant limitations in the evaluation methodology (self-referential stealthiness, large variance, limited environments) and some overclaiming in how results are presented.

Let me write the review.</think>Now I have enough context to write the final review. Let me synthesize everything.

## Summary

SHIFT proposes a diffusion-based adversarial state perturbation attack for reinforcement learning that moves beyond traditional ℓ_p-norm constraints. By training a history-conditioned diffusion model enhanced with policy guidance (to change semantics) and autoencoder-based realism guidance, the attack generates perturbed observations that are semantically distinct, visually realistic, and temporally consistent with prior observations. The paper demonstrates that state-of-the-art defenses (SA-DQN, WocaR-DQN, CAR-DQN, DP-DQN, Diffusion History) designed against ℓ_p-bounded attacks fail against SHIFT across four Atari environments, with cumulative reward reductions exceeding 50%.

## Strengths

- **Identification of a genuine and important threat model gap**: The paper correctly identifies that ℓ_p-norm constraints on adversarial perturbations prevent semantics-changing attacks, and that defenses robust to these bounded perturbations can be defeated by unrestricted semantic perturbations. Figure 1c vs. 1e provides clear visual evidence that PGD at ε=15/255 cannot alter ball/paddle positions in Pong, while SHIFT can. This observation that existing defenses' robustness may be an artifact of the restricted threat model is a meaningful contribution to the RL adversarial robustness community.

- **Principled multi-objective attack design with complementary guidance mechanisms**: The combination of classifier-free guidance (Section 3.2.1, conditioning on history τ_{t−1} for plausibility), classifier guidance (Section 3.2.2, using ∇ log π(ā_t | s̃_t^i) for action manipulation), and autoencoder realism guidance (Section 3.2.3, using reconstruction loss gradients) is well-motivated and addresses separate attack objectives (history alignment, semantics change, realism). Theorem 1 establishes that classifier and classifier-free guidance can coexist without interference in the RL setting because the conditioning variables factor—history affects only the diffusion model's prediction while the policy gradient depends only on the current state.

- **Comprehensive evaluation against strong defenses**: Table 1 evaluates SHIFT against six defense methods including two state-of-the-art diffusion-based defenses (DP-DQN, Diffusion History). Figure 3a additionally compares against seven ℓ_p-constrained and unrestricted attack baselines on the DP-DQN defense, showing that all prior attacks fail while SHIFT succeeds. The inclusion of high-sensitivity direction attacks from Korkmaz (2023) (Blurred, Shifting) provides comparison against methods that also go beyond ℓ_p constraints.

- **Practical deployment via EDM**: Table 2 demonstrates that using EDM reduces sampling time to ~0.2 seconds per perturbed state (vs. ~5 seconds for DDPM), making real-time attacks feasible while maintaining comparable attack effectiveness.

## Weaknesses

### Fatal

None.

### Major

- **Self-referential stealthiness evaluation undermines the "difficult to detect" claim**: The primary stealthiness metric—reconstruction error ‖s̃_t − AE(s̃_t)‖_2 (Section 4, Figure 3a)—uses the same autoencoder that the attacker trains on clean data and uses for realism guidance during generation (Section 3.2.3). This creates a circular evaluation: of course perturbations guided to minimize AE reconstruction error will have low reconstruction error under that same AE. The paper claims in Section 1 that the perturbations are "difficult to detect," but uses only an attacker-controlled detector to support this. An independent anomaly detector, human evaluation, or at minimum a separately trained autoencoder would be needed to credibly establish stealthiness. The Wasserstein distance to the previous true state (also in Figure 3a) partially addresses this but measures proximity to a single reference frame, not whether a sequence of perturbed observations is internally consistent or detectably anomalous.

- **No unrestricted-attack baseline isolates SHIFT's specific contributions**: The paper demonstrates that SHIFT breaks defenses that ℓ_p-bounded attacks cannot, but this comparison conflates removing the ℓ_p constraint with SHIFT's specific mechanisms (policy guidance, history conditioning, realism guidance). The "Uncond." diffusion model shown in Figure 1 is a qualitative comparison only—there is no quantitative table showing how an unconditional diffusion model or a history-conditioned diffusion model *without* policy guidance performs as an attack. Without this ablation, it is unclear whether the attack's effectiveness comes primarily from removing the ℓ_p constraint (allowing any semantically different valid state) or from SHIFT's specific combination of guidance mechanisms. This matters because if even naive unrestricted sampling suffices to break these defenses, SHIFT's core methodological contribution (the guidance design) is diminished.

- **High variance in some results raises reliability concerns**: Several entries in Table 1 have standard deviations comparable to or larger than their means (DP-DQN on Pong: 0.5 ± 11.4; SA-DQN on RoadRunner: 260 ± 215; DP-DQN on RoadRunner: 360 ± 321; CAR-DQN on RoadRunner: 40 ± 55). These large variances make it difficult to draw reliable conclusions about the attack's effectiveness against these particular defenses, especially in environments beyond Pong and Freeway.

### Minor

- **Intractable formal definitions with limited empirical verification**: Definitions 2–5 (Section 3.1) ground the attack objectives in terms of Projections onto S*, which the paper acknowledges is intractable (Section 3.2.2: "it is difficult to identify the projection onto the valid states S*"). The paper does not measure how well its generated perturbations approximate these definitions in practice (e.g., no proxy measurement of semantic change beyond action deviation, no measurement of the ω parameter in Definition 5). While formal objectives that are intractable to compute exactly is common and acceptable, the gap between theory and implementation deserves acknowledgment of what remains unverified.

- **Theorem 1, while correct, is a limited theoretical contribution**: Theorem 1 establishes that classifier guidance and classifier-free guidance can be combined because the two conditioning variables (ā_t and τ_{t−1}) factor—the policy π depends only on the current state, not history. This is a useful property for the method but follows from the structure of the problem (policies map states to actions, not histories) rather than being a deep insight about diffusion models. The proof in Appendix C likely confirms this straightforward factorization.

- **Strong attacker assumptions not fully discussed**: The threat model (Section 2.2) assumes the attacker has access to the clean environment, the victim's policy π, and true states in real-time, with no ℓ_p constraint. While white-box assumptions are standard in this literature, the combination with real-time true-state access and unrestricted perturbations is a powerful attacker model. The paper could more explicitly discuss what practical scenarios correspond to this threat model and how results might change under weaker assumptions.

- **Probing defense evaluated on only one environment**: Figure 3b evaluates the probing defense idea only on Freeway with one defense variant. While acknowledged as preliminary, this limits the generalizability of the defense results.

## Nice-to-Haves

- An ablation study quantifying the contribution of each guidance mechanism (policy guidance alone, history conditioning alone, realism guidance alone, and full SHIFT) would substantially strengthen the paper by isolating what makes the attack effective.
- An independent stealthiness evaluation using an out-of-distribution detector or human study would make the "difficult to detect" claim more credible.
- Characterizing the perturbation magnitude distribution (‖s̃_t − s_t‖_2) across episodes would make the unrestricted threat model more concrete by showing how far from true states the perturbations actually are.
- Expanding beyond 4 Atari environments would improve generalizability claims.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unfair comparison with constrained attack baselines"**: The harsh critic argues comparing SHIFT against ℓ_p-constrained attacks is unfair because SHIFT operates under a different threat model. While the comparison is asymmetric, it serves the paper's stated purpose: demonstrating that defenses designed for ℓ_p-bounded threats are vulnerable to unrestricted attacks. The paper explicitly frames this as a *new threat model* (Section 2.2: "we consider a novel attack similar to unrestricted adversarial examples"). The relevant question is not whether SHIFT "beats" PGD on PGD's terms, but whether defenses robust to PGD can withstand unrestricted attacks—and the paper provides this evidence. However, the concern remains (captured as a Major weakness) about the lack of unrestricted-attack *baselines* for comparison.

- **"Theorem 1 is straightforward"**: Kept as a Minor weakness since it is correct and useful but limited in novelty; removed the claim that it is "overstated" since the paper does not claim it is a deep result.

- **"History alignment claim only approximate because attacker conditions on true history"**: The paper explicitly acknowledges this limitation (Section 3.2.1: "our attack uses the true history τ_{t−1} to approximate the victim's belief H_{t−1}, as the latter requires projecting each perturbed state onto S* and is computationally expensive"). This is a reasonable approximation strategy; the paper is transparent about it.

- **"Only 4 Atari environments"**: This is a standard scope concern; 4 environments is reasonable for the complexity of training environment-specific diffusion models.

- **"Random Non-Optimal target action strategy is weak"**: The paper acknowledges this and provides a more advanced strategy in Appendix E. The results even with this myopic strategy already show significant reward reduction, which strengthens rather than weakens the claims.

- **"Attacker assumptions too strong"**: Kept as a minor point for transparency, but removed the characterization as a "fundamental limitation" since white-box assumptions with environment access are standard in this literature (e.g., Zhang et al., 2020a; Sun et al., 2021 use similar assumptions).

## Novel Insights

The paper's key insight—that ℓ_p constraints fundamentally limit adversarial attacks from changing semantics, and that defenses robust to ℓ_p-bounded perturbations may be circumvented by unrestricted perturbations generated by diffusion models—is genuinely important for the RL robustness community. It parallels similar observations in the image classification literature where unrestricted adversarial examples bypass robust classifiers, but adapts this to the sequential decision-making setting with the added dimension of temporal consistency (history alignment). The observation that diffusion-based defenses (DP-DQN, Diffusion History) are designed to denoise ℓ_p-perturbed inputs and thus fail against semantically valid alternatives is a concrete finding that should inform future defense design. However, the circular stealthiness evaluation and lack of unrestricted baselines limit how strongly these conclusions can be drawn.

## Suggestions

- **Add an unconditional diffusion baseline (quantitative)**: Report Table 1-style metrics for a history-conditioned diffusion model *without* policy guidance. This would directly isolate the contribution of the policy guidance mechanism and address concerns that any unrestricted semantic perturbation would break these defenses.
- **Use an independent autoencoder for stealthiness evaluation**: Train a separate autoencoder on a different random seed or different architecture from the one used for guidance, and report reconstruction errors under this independent detector. Even better, use an off-the-shelf anomaly detector or report human detection rates.
- **Report perturbation magnitude distributions**: Add statistics on ‖s̃_t − s_t‖_2 across environments to characterize the unrestricted threat model's practical scope.

## Calibration Anchors

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| F5dhGCdyYh (Illusory Attacks on RL) | 7.33 | Similar RL adversarial attack domain with novel threat model, but with stronger theoretical stealthiness foundations (information-theoretic bounds) and human evaluation. SHIFT is weaker on stealthiness rigor (self-referential metric) but comparable on attack effectiveness. |
| 6qeCyvlJUJ (EvoSeed) | 3.67 | Also uses diffusion models for adversarial examples, also has fair comparison concerns (different perturbation norms). SHIFT is significantly stronger: better-motivated threat model, more thorough defense evaluation, clearer attack design with multiple objectives. |
| wZWTHU7AsQ (Game-Theoretic Robust RL) | 5.33 | Robust RL approach with temporally-coupled perturbations, limited novelty concerns. SHIFT has comparable evaluation scope but a more novel and well-motivated contribution. |
| PYVIXnHZhh (RL adversarial attack, strong assumptions) | 4.25 | Attack on RL with white-box assumptions and weak baselines. SHIFT has stronger baselines and a more important observation but shares some assumption concerns. |
| mXpNp8MMr5 (Two-faced attacks) | 7.33 | Novel threat model exposing vulnerability in adversarial training. Similar profile to SHIFT as a "breaking defenses" paper. More rigorous evaluation; SHIFT is in a comparable conceptual space but with weaker stealthiness evaluation. |

SHIFT is clearly stronger than the low-scoring EvoSeed (3.67) and RL attack with strong assumptions (4.25) papers, but weaker than the high-scoring Illusory Attacks (7.33) and Two-faced Attacks (7.33) due to the circular stealthiness evaluation and lack of unrestricted baselines. It sits in a similar space to the medium-scoring Game-Theoretic Robust RL (5.33) but makes a more impactful contribution. The key issue holding SHIFT back is that while the core observation is important, the evaluation methodology doesn't fully isolate whether SHIFT's specific mechanisms or simply removing ℓ_p constraints drives the results, and the stealthiness claim relies on a self-referential metric.

## Score and Decision

Based on the calibration, SHIFT is a solid paper making an important observation about the vulnerability of existing RL defenses to unrestricted semantic perturbations. The method is well-designed combining three guidance mechanisms, and the experiments are comprehensive against six defenses. However, the self-referential stealthiness evaluation and lack of unrestricted baseline ablations prevent the paper from fully establishing its core claims about SHIFT's specific contribution vs. the contribution of removing ℓ_p constraints. The paper falls in the 5.5–6.0 range: above borderline due to the importance of the threat model observation and the technical soundness of the approach, but not clearly above the acceptance threshold due to the evaluation gaps.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>