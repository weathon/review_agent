Let me check the paper content again for the specific claim about the single source and the assumption mismatch. The paper states in Section 4: "In our experiments, the proposed architecture specializes to a single source X, so that (X₁,X₂)=X." And in Section 3.3: "Each branch of the proposed architecture has access to both sources X₁ and X₂, all exclusive information from either source is available to assist in performing tasks Z₁ or Z₂."

Now regarding Theorem 2, it assumes Y₀=f₀(X₁,X₂), Y₁=f₁(X₁), Y₂=f₂(X₂). In the experiments, both f₁ and f₂ see the same source X. This is a real mismatch.

Also regarding the claim about β=1 optimizing transmit rate and β=2 optimizing receive rate — let me verify this from the paper. The paper states: "When β=1, we optimize for the transmit rate Rₜ. When β=2, we optimize for the receive rate Rᵣ." Looking at equation 12, with β=1/α, when α₁=α₂ and β=1, the objective weights r₀ once and r₁,r₂ once each, giving a weight proportional to r₀+r₁+r₂ = the transmit rate. When β=2, the objective weights r₀ twice and r₁,r₂ once each, giving r₀+r₁+r₀+r₂, but the receive rate is 2r₀+r₁+r₂. So the paper's claim is that β controls the weight on r₀, which corresponds to the common channel being counted once (transmit) or twice (receive). This is a reasonable but informal argument — it doesn't rigorously follow from Theorem 2 that β=1 minimizes Rₜ or β=2 minimizes Rᵣ, because Theorem 2 is about the Gray-Wyner T function, while in practice we're minimizing a parametric loss with limited function families.

Now let me also check the claim about no comparison with prior multi-task codecs. The paper mentions Chamain et al. (2021), Feng et al. (2022), Guo et al. (2024) but these are cited as related work, not as baselines.

Let me carefully compose the final review now.

## Summary

This paper develops a learnable three-channel image codec inspired by the Gray-Wyner Network, separating information into a common channel and two private channels for two machine vision tasks. It provides theoretical bounds relating Wyner's and Gács-Körner lossy common information (Theorem 1), derives a practical optimization objective from the Gray-Wyner Lagrangian (Theorem 2), and proposes a specific architecture with a shared-plus-private encoder that combines common representations via an equality-matching rule. Experiments on synthetic data, MNIST, Cityscapes, and COCO demonstrate rate-distortion improvements over independent single-task coding.

## Strengths

- **Principled problem motivation**: The transmit-receive tradeoff in Gray-Wyner networks is a genuine information-theoretic contribution direction. Formulating multi-task compression as a three-channel problem (common + two private) with explicit rate allocation is well-motivated and extends the prior two-channel "coding for humans and machines" paradigm.

- **Theoretical contribution (Theorem 1)**: The bounds relating Wyner's lossy common information, Gács-Körner lossy common information, and interaction information are a clean theoretical result that adds insight into when the two measures coincide and when the transmit-receive tradeoff is non-trivial.

- **Comprehensive evaluation across regimes**: The paper tests on synthetic data with known theoretical quantities, structured MNIST experiments with controlled mutual information, and real vision tasks, demonstrating the method works across settings.

- **Edge-case validation**: The MNIST colorization experiments (Dependent, Independent, Mixture PMFs) are a well-designed sanity check showing the method correctly places more in the common channel when tasks are dependent and less when independent, which provides qualitative evidence of appropriate behavior.

- **Practical compression gains**: The reported BD-rate advantage of −81.58% vs. single-task codecs, even if partly attributable to multi-task sharing rather than the specific Gray-Wyner structure, demonstrates the practical relevance of joint multi-task coding.

## Weaknesses

### Major:

- **Disconnect between theoretical claims and empirical setup**: The paper's central narrative — that it is "isolating common information" and "exploring the transmit-receive tradeoff" in the Gray-Wyner sense — relies on strong information-theoretic objects (lossy C, lossy K, optimal rate-distortion functions) that are defined via global optima over all encoders/decoders. The actual learned codec uses specific parametric architectures (ResNet-based encoders, approximate entropy models, straight-through quantization). While Theorem 2 provides a form for the Gray-Wyner objective under strong assumptions (deterministic encoders, existence of optimal encoders in the function families), these assumptions are not verified and are unlikely to hold for the limited architectures used. The paper's claims in Section 5 that it "validated the ability of the proposed learnable Gray-Wyner Network to distill common information between tasks" go beyond what the experiments establish: what is validated is that a carefully designed shared codec outperforms independent codecs, not that Y₀ actually carries lossy common information in the Gray-Wyner sense. This overclaiming is the paper's most significant weakness.

- **Assumption violation between Theorem 2 and experiments**: Theorem 2 assumes Y₁=f₁(X₁) and Y₂=f₂(X₂) — each private representation depends on one source only. But in Section 4, the authors state "the proposed architecture specializes to a single source X, so that (X₁,X₂)=X," and in Section 3.3 they note "each branch has access to both sources." This means both encoders see the same input, violating the encoder structure assumed in Theorem 2. Consequently, the Lagrangian (12) being optimized in practice is not the Gray-Wyner T function for the experimental setting. The paper should either maintain consistency or explicitly discuss this discrepancy and its implications.

- **Common channel mechanism is heuristic and insufficiently justified**: The core algorithmic contribution — Eq. (14) and (15) — combines Y₀^{(1)} and Y₀^{(2)} via element-wise equality matching and an L₂ auxiliary loss with γ=1. This is a brittle criterion for continuous/quantized representations, and the paper acknowledges degenerate solutions for extreme γ values. No ablation compares this construction to alternatives (e.g., learned projection, concatenation + masking, simple averaging without zeroing). Given that this mechanism defines what "common information" means operationally, the lack of empirical or theoretical comparison to simpler alternatives significantly weakens the method's contribution.

- **Insufficient comparison to existing multi-task codecs**: The paper cites Chamain et al. (2021), Feng et al. (2022), and Guo et al. (2024) as related multi-task codecs with common-only channels, but does not compare against any of them. The baselines are all architectural variants of the authors' own design (Shared, Separated, Combined, Joint, Independent). Without comparison to existing multi-task compression methods, it is unclear whether the specific Gray-Wyner three-channel structure provides advantages over simpler multi-task approaches. This is particularly important because the main claim is about the *theoretically principled* separation of common/private information, and one must ask whether a standard shared representation produces similar gains.

### Minor:

- **β=3/2 is heuristic**: The paper states β=3/2 "equally optimizes for both the transmit and receive rates," but this assumes a linear interpolation between the two rate functions, which need not hold on the actual achievable region contour (which may be non-convex). This is acknowledged informally but not discussed further.

- **No ablation of γ beyond γ=1**: The auxiliary loss weight γ is set to 1 with no sensitivity analysis. The paper notes that small γ leads to no matching and large γ causes degenerate distributions, but provides no empirical characterization of this tradeoff.

- **Scaling to multiple tasks is deferred**: The conclusion acknowledges exponential channel growth for 3+ tasks but offers no solution or preliminary experiment, limiting the practical scope.

### Trivial

- **The Markov conditions (Eq. 1) are assumed but their validity for the chosen CV tasks is not discussed**: The conditions Z₂↔X₂↔X₁ and Z₁↔X₁↔X₂ may not hold exactly for arbitrary task pairs (e.g., if semantic segmentation benefits from features beyond what depth estimation's source provides). This is a modeling assumption, not a flaw per se, but deserves brief acknowledgment.

## Nice-to-Haves

- **Visualization of what Y₀ encodes**: Feature maps or probing classifiers applied to Y₀ alone vs. Y₁/Y₂ would directly validate whether common vs. private information is actually separated, as opposed to being a multi-channel codec without true disentanglement.

- **Comparison to existing multi-task compression baselines**: Even one comparison (e.g., Feng et al., 2022) on Cityscapes/COCO would clarify the advantage of the three-channel structure over single-common-channel designs.

- **Per-channel rate breakdowns for CV experiments**: Figure 3 shows channel-wise rates for synthetic data and MNIST but not for Cityscapes/COCO. This would reveal how the transmit-receive tradeoff manifests in the main experiments.

## Removed Points

- **"No quantitative evidence that β controls the transmit-receive tradeoff" (Harsh Critic #3)**: While the paper does not estimate optimal rate-distortion functions to verify proximity to theoretical bounds, Figure 3a does empirically show that β=1 produces higher common-channel rates (consistent with transmit-rate optimization) and β=2 produces lower common-channel rates (consistent with receive-rate optimization). The MNIST experiments also show qualitatively appropriate behavior for different PMFs. The claim that β *navigates* the tradeoff is supported, even if the claim that it does so in the *optimal Gray-Wyner sense* is overclaimed. Removed because the empirical trend is real; the overclaiming aspect is captured in Major weakness #1 above.

- **"No statistical analysis for β comparison"**: Single-run evaluation is standard practice in the learned compression community. Requesting confidence intervals for these scale of experiments is a nice-to-have, not a core flaw.

- **"Performance below Joint coding"**: The Joint method optimizes transmit rate with no constraint on receive rate, so it is expected to outperform on transmit rate. This is by design, not a weakness.

- **"BD-rate against single-task codecs is a weak baseline"**: The paper also compares against Joint (an upper bound) and shows competitive performance. Comparing against single-task codecs is a standard and meaningful baseline in compression.

- **"Request for bpp reporting"**: The paper uses an abstract "rate" metric, which is standard in the learned compression community. Converting to bpp would not change the relative comparison.

- **"Request for asymmetric task importance"**: The paper explicitly scopes to α₁=α₂ as a stated simplifying assumption. Scoping this out is legitimate; it does not invalidate the symmetric case.

## Novel Insights

The Theorem 1 bounds connecting lossy Wyner CI, Gács-Körner CI, and interaction information provide a clean characterization of *when* the transmit-receive tradeoff is non-trivial: when the interaction information between optimal task encodings is not separable from private information. The MNIST Mixture PMF experiment nicely illustrates a regime where this non-separability degrades performance, providing a concrete operational meaning to Theorem 1's conditions. This is a genuinely useful theoretical insight even if the practical codec does not provably achieve these bounds.

## Suggestions

- **Scale back information-theoretic claims**: Replace assertions about "isolating lossy common information" and "distilling common information" with more precise language (e.g., "learning a task-driven shared representation guided by a Gray-Wyner-inspired objective"). This would align claims with what the experiments actually demonstrate.

- **Add one baseline comparison with a standard multi-task codec**: Even a simple shared-encoder + separate-head baseline without the explicit common/private channel split would clarify whether the three-channel structure provides genuine benefits beyond multi-task feature sharing.

- **Add an ablation on the common channel construction**: Compare Eq. (14)–(15) against simpler alternatives (e.g., average of Y₀^{(1)} and Y₀^{(2)} without zeroing, or a single shared encoder branch producing Y₀ directly).

## Score and Decision

Calibration: I compared against several related papers. The Disentangling Representations paper (yVGGtsOgc7, scores 8/6/3/6/6, accept poster) has similarly aspirational theoretical claims about disentanglement but stronger formal guarantees (conditions for emergence). The Shared-AE paper (zXCnIyX9MG, scores 8/6/3/5, accept poster) offers a simpler technical contribution with cleaner empirical validation. The Rate-Distortion-Perception paper (vdUYa7N8Mt, scores 6/6/5/5, reject) has strong theory but no experiments. The Taskonomy-Aware paper (x33vSZUg0A, scores 3/5/8, accept poster by area chair override despite low scores) has practical gains but limited baselines. The IB multi-objective paper (GlqeLNjH6p, scores 1/3/5/5, reject) has theoretical overclaims relative to empirical support.

This paper has a genuine and interesting theoretical contribution (Theorem 1 and the Gray-Wyner connection) and demonstrates practical compression gains. However, the central claims about "isolating lossy common information" are overstated relative to what is empirically verified, the assumption mismatch between Theorem 2 and the experimental setup is significant and unacknowledged, the core algorithmic mechanism is heuristic and insufficiently ablated, and the comparison set is limited to self-designed baselines. These are significant but not fatal weaknesses — the paper does demonstrate that the Gray-Wyner-inspired objective and architecture produce useful multi-task coding behavior. The paper is at the borderline of acceptability but the overclaimed connection between trained networks and information-theoretic optimal quantities, combined with insufficient baselines and ablations, tips it below the threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>