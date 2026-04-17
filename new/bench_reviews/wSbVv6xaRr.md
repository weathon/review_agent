Now I have enough context for calibration. Let me now synthesize the final review.

Let me carefully verify the key claims from the harsh critic against the actual paper content:

1. **Abstract claims "O(1/K)" convergence** - Let me check. The abstract says "converges at a rate of O(1/K), matching the performance of FedSGD." But Theorem 2 establishes O(1/√K). The neutral reviewer also caught this. This is a factual error in the abstract.

2. **Privacy "guarantee" concerns** - The harsh critic argues the privacy analysis is narrow, based on a specific attack model, without formal DP guarantees. Let me verify: Lemma 1 gives relative reconstruction error (d-1)/m. Lemma 2 gives a lower bound on data reconstruction under a specific loss function. Remark 2 gives T·m < d as a composition bound. The paper positions this against LDP. These are legitimate concerns but I need to assess their severity.

3. **Convergence claim inconsistencies** - The harsh critic notes the JL lemma application may need union bounds over iterations, and the (d-1)/m error from Lemma 1 seems contradictory with the "logarithmic in d" dependence. Let me check: Lemma 1 gives E[||ĝ-g||²]/||g||² = (d-1)/m, which for m = O(log d) gives reconstruction error Θ(d/log d) — very large. But this is the error of the gradient estimator (privacy benefit), not the convergence error. The convergence uses a JL-type norm preservation bound, not the variance of the estimator. Actually, the paper argues the JL lemma gives (1-ε)||g|| ≤ ||ĝ|| ≤ (1+ε)||g|| with high probability, which combined with unbiasedness gives the convergence. The variance issue Lemma 1 identifies is actually the privacy asset, not the convergence bug. But there IS a tension: for convergence, you need m = O(log d) to get good norm preservation; for privacy, this m gives huge relative error (d/log d), which seems contradictory only if you think both need to be simultaneously satisfied. Actually, they CAN be satisfied simultaneously: the estimator is unbiased with variance (d-1)/m * ||g||², but the JL lemma shows that with high probability, the norm is preserved (so the estimator won't blow up). The convergence analysis likely leverages the JL high-probability bound rather than the variance directly. However, the abstract claiming O(1/K) instead of O(1/√K) is wrong.

4. **Table 2 shows FedMPDD only achieves 40.84% vs 60% target** - This is indeed a weakness.

5. **Novelty relative to sketched/random projection methods** - The paper clearly differentiates itself from "structured" and "sketched" updates. The key distinction is dynamic per-round, per-client random projections (Rademacher vectors) vs. fixed projections. The Ferret paper uses a very similar mechanism. This is a legitimate novelty concern.

6. **Multi-round privacy erosion** - The T·m < d bound is weak. For d = 300K (CNN model), m = 600 gives T < 500 rounds. Many FL tasks run for 1000s of rounds.

Let me now compose the final review.</think>Now I have enough calibration data. Let me compose the final review.

## Summary

FedMPDD proposes a federated learning algorithm that encodes each client's gradient using multi-projected directional derivatives: computing inner products with m random Rademacher vectors and transmitting only m scalars plus a seed. The same mechanism that compresses communication (O(d)→O(m)) also provides inherent privacy against gradient inversion attacks via rank-deficient projections, while multi-projection averaging is shown to achieve O(1/√K) convergence matching FedSGD.

## Strengths

- **Elegant joint mechanism for communication and privacy**: The same rank-deficient projection simultaneously achieves communication compression and privacy protection, with a tunable trade-off via m. The insight that averaging m projections overcomes the dimension-dependent variance blow-up of single projections (which yields O(d/√K) convergence) is technically sound and well-motivated.

- **Clear theoretical contributions**: Theorem 2 establishes O(1/√K) convergence for FedMPDD with m = O(ln(d/δ)/ε²), and Lemmas 1–2 provide reconstruction error bounds linking m to a quantifiable privacy measure. The JL lemma connection for norm preservation is a principled argument.

- **Strong empirical communication savings**: Tables 1–2 show dramatic communication reductions (e.g., 356× vs. FedSGD to reach 60% on CIFAR-10) with very low SSIM scores (<0.22) against gradient inversion attacks, whereas competing compression methods (Top-k, QSGD, lp-proj) leak heavily (SSIM 0.74–0.93).

- **Practical protocol design**: Using shared random seeds to reconstruct projection vectors server-side, avoiding the need to transmit the d-dimensional vectors, is a clean and lightweight implementation approach.

## Weaknesses

### Major:

- **The privacy framework is informal and does not provide guarantees comparable to standard privacy notions (e.g., differential privacy).** The paper claims "formal defense against GIAs" and "robust and uniform privacy," but the analysis rests on: (1) Lemma 1's relative reconstruction error (d−1)/m, which is a gradient-error metric, not a privacy definition with compositional properties; (2) Lemma 2's lower bound on data reconstruction error, which assumes the adversary minimizes a specific projected-gradient-matching loss—actual GIA methods (including Yu et al. 2025 that the paper cites) use richer objectives with priors and regularizers, and no argument establishes that the lower bound applies to all attack strategies; (3) Remark 2's multi-round bound T·m < d, which is an algebraic uniqueness condition, not a statistical privacy guarantee. In contrast, LDP provides (ε,δ)-DP with well-understood composition theorems. The paper's comparison with LDP is therefore incomplete: it compares a reconstruction-error metric for one attack with an (ε,δ) guarantee that has clear composition, making claims of "superior" privacy misleading. This is a significant gap given that joint communication-and-privacy is a central claimed contribution.

- **The abstract claims "O(1/K) convergence matching FedSGD," but Theorem 2 establishes O(1/√K)**—which is the standard non-convex SGD rate. For non-convex problems, O(1/√K) is appropriate, but the abstract's claim of O(1/K) is incorrect and inflates the convergence guarantee.

- **Multi-round privacy erosion undermines practical applicability.** The condition T·m < d implies that for a CNN with d ≈ 300K and m = 600, privacy guarantees hold only for T < 500 rounds. Many FL training runs far exceed this. The paper acknowledges that gradient evolution provides "stronger practical protection" but provides no formal or empirical analysis of privacy degradation over realistic training horizons (hundreds to thousands of rounds). This leaves a critical gap between the theoretical privacy claim and its practical viability.

- **Limited novelty relative to random projection/sketching methods.** The core mechanism of projecting gradients onto random directions and reconstructing via shared seeds closely resembles prior work. The Ferret paper (withdrawn, scores 6/5/3) used shared randomness with random projections for FL communication reduction, and SEPARATE (accepted poster, scores 6/6/6/6) uses random Gaussian projections with error feedback. FedMPDD's key distinction—dynamic per-round, per-client Rademacher projections instead of fixed ones—is incremental rather than fundamental. The paper does not compare against these closely related methods (FetchSGD, Ferret) in experiments.

### Minor:

- **Table 2 shows FedMPDD achieves only 40.84% test accuracy (m=600) vs. a 60% target on CIFAR-10.** The paper marks this as a communication win, but the accuracy gap is significant. It is unclear how many rounds (and total bytes) FedMPDD would need to actually reach 60%, and whether the net communication savings would hold.

- **Experiments are limited to small models.** The largest model tested is a CNN with ~300K parameters. Modern FL deployments involve models orders of magnitude larger. Without experiments on ResNet-18 or larger, the scalability of the logarithmic m-d relationship is unsubstantiated empirically.

- **LDP baselines are weak.** Only Laplace noise at fixed variances (0.1, 0.5, 1, 10) is tested, with no formal (ε,δ) accounting. A fair comparison would match communication budgets and report ε,δ values, or compare against structured DP mechanisms (e.g., cpSGD).

- **The theoretical analysis has hidden dependencies in constants.** Theorem 2 references "Assumption 1" and includes terms like O(εG²/√K) without exposing how ε (which depends on m and d) affects the overall bound. The actual dependence of convergence constants on d through ε is not transparent.

### Trivial:

- The paper's notation in Definition 1 is slightly confusing: ∇_c f(x) = u^⊤∇f(x)u is called a "projected directional derivative" but is a scalar; the vector estimator is u(u^⊤∇f(x)), which is a different object. The notation is clarified later but initially ambiguous.

## Nice-to-Haves

- **Compare against FetchSGD and Ferret.** Since these are the closest random-projection-based FL methods, direct experimental comparison would clarify FedMPDD's practical advantages.

- **Provide (ε,δ)-DP conversion or composition analysis.** Even an approximate DP bound derived from the projection mechanism would dramatically strengthen the privacy claims and enable principled comparison with LDP.

- **Test on larger models (ResNet-18, Transformers)** where communication savings matter most, and systematically vary m and d to validate the logarithmic scaling claim.

- **Evaluate privacy (SSIM) over many rounds** (500+) to empirically test whether gradient evolution provides practical protection beyond the worst-case T·m < d bound.

- **Report wall-clock time for client-side encoding** to validate the JVP-based efficiency claim from Remark 1, rather than deferring it to an appendix/follow-up.

## Removed Points

- **"The method cannot be independently verified" / reproducibility concerns about models, datasets, or attack implementations.** The paper cites specific methods and benchmarks; these are treated as real and available per policy.

- **Formatting/style nitpicks** (e.g., "Rabi-spotting determined duration" is clearly a parser artifact, not a paper flaw).

- **Demand for comparison with FedAvg with multiple local epochs.** The paper's stated scope is communication-efficient FedSGD-style algorithms; adding FedAvg comparison would be nice but is outside the core scope.

- **"Missing related works" claims.** Per policy, I cannot confirm whether unmentioned works exist, and the paper does cite relevant sketched/projection methods.

- **The harsh critic's claim that the seed transmission is a vulnerability because "the server knows the random vectors."** The paper explicitly acknowledges (Algorithm 2, lines 6–16) that the server reconstructs the random vectors from seeds—and the entire privacy argument is built around this design. The server knowing the projection directions is not a vulnerability; the privacy comes from rank deficiency, not from hiding the projection. This is a misreading.

- **The harsh critic's claim that the convergence proof needs union bounds over K iterations, N clients, and stochasticity, and that "the log(d) scaling may not survive."** While this is a valid technical concern, the paper's JL application uses a standard high-probability argument (Lemma 6 in the appendix), and union bounds over K rounds would add log(K) to m, which is still logarithmic. This is a standard technical detail, not a fundamental flaw.

## Novel Insights

The most valuable insight is the duality between FedMPDD's communication and privacy properties: the same rank-deficiency that enables O(m) communication (by sending only scalars and seeds) also creates an underdetermined system that prevents gradient inversion. This is distinct from the standard "compress + add noise" paradigm. However, the insight that this duality creates an inherent tension—larger m improves convergence but erodes privacy—deserves more explicit treatment, as it limits the regime where both benefits coexist. The observation that smaller m can sometimes yield faster convergence (via nullspace-induced gradient noise suppression) is intriguing but only mentioned anecdotally and lacks theoretical or systematic empirical support.

## Suggestions

1. **Correct the convergence rate in the abstract** from O(1/K) to O(1/√K), or clarify the stronger assumptions under which O(1/K) would hold.

2. **Reframe the privacy contribution honestly**: position it as "reconstruction-error-based privacy against gradient inversion attacks" rather than a privacy *guarantee* in the DP sense; add a discussion of scenarios where DP vs. reconstruction-error privacy are more appropriate.

3. **Provide empirical results on privacy erosion with training rounds**: run GIA attacks after 100, 200, 500, 1000 rounds and report SSIM to show whether composition is practically concerning.

4. **Add comparison with FetchSGD/or at minimum other random-projection baselines** to establish the practical advantage of dynamic, per-client projections over static sketches.

## Score and Decision

**Calibration anchor papers:**
- **SEPARATE** (random projection gradient compression, scores 6/6/6/6, Accept Poster): Clear, well-executed, but incremental over GaLore/Flora. Has strong experiments on LLMs. FedMPDD has weaker experiments and a similar novelty profile.
- **Ferret** (shared randomness + projection for FL, scores 6/5/3, Withdrawn/Reject): Very similar core idea (project updates into low-dim space, reconstruct via shared randomness). FedMPDD is closer to this than to SEPARATE. Ferret was rejected partly for lacking formal privacy analysis and limited novelty.
- **D2P2-SGD** (DP + random projection, scores 3/3/3/3, Reject): Combines existing ideas (DP clipping + projection) with toy experiments and overclaimed privacy-utility improvement. FedMPDD has better theoretical grounding than D2P2-SGD but shares the issue of informal privacy guarantees against established DP.
- **Forward Learning with DP** (scores 3/6/5/6, Reject): Claims inherent privacy from forward-mode randomness without formal DP guarantees initially—similar issue to FedMPDD. Mixed reviews due to weak baselines and informal privacy claims.

FedMPDD occupies a space between SEPARATE (well-executed, incremental, accepted poster) and D2P2-SGD/Ferret (informal privacy, limited novelty, rejected). The paper's joint communication-and-privacy story is interesting, but the privacy analysis is insufficiently formal for the claims made, the convergence rate is overstated in the abstract, experiments are limited to small models, and novelty over random-projection FL methods is incremental. These are substantial weaknesses that prevent acceptance in current form.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>