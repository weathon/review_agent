## Summary

This paper introduces Markovian compression operators for distributed optimization—compressors whose stochasticity depends on previous iterations rather than being drawn i.i.d. each step. Two concrete instantiations are proposed (BanLast and KAWASAKI), convergence theory is developed for MQSGD and its accelerated variant AMQSGD across non-convex, PL, and strongly convex settings, and experiments demonstrate consistent empirical improvements over standard random sparsification.

## Strengths

- **Genuinely novel concept with no prior work**: The paper identifies and fills a real gap—Section 1.2 states "there are currently no works that combine compressed data communications and Markovian stochasticity of the compressors." The idea that compressor randomness should depend on past transmissions is intuitive and previously unexplored.

- **Complete convergence theory across three function classes**: Theorem 2 provides convergence for both non-convex (O(1/γT + γLτd²/m²σ²)) and PL cases with linear rates; Theorem 3 provides strongly convex convergence with acceleration. Covering all three regimes with explicit rates is thorough for a new framework.

- **Novel "stepping back" proof technique**: The sketch proof around Equations (3)–(5) develops a new analytical device that applies the compressor at step t to vectors from step t−τ (exceeding the mixing time), then decomposes cross-terms using smoothness. This converts asymptotic unbiasedness (Assumption 5) into usable bounds and could apply beyond this paper.

- **AMQSGD provably accelerates over MQSGD**: Corollary 2 achieves (L/μ)^{2/3} condition number dependence versus the L/μ dependence in Corollary 1, confirming that momentum acceleration improves convergence within the Markovian framework.

- **Concrete compressor designs with verified ergodicity**: Theorem 1 proves both BanLast (under d > (K+1)m) and KAWASAKI (under permutation invariance of π_Δ) are ergodic with uniform stationary distributions, with explicit convergence rates ρ and constants C (Eq. 2 for KAWASAKI). This makes Assumption 5 concretely satisfiable.

- **Optimal polynomial dependence on mixing time**: Section 2.4 notes Theorem 2 achieves polynomial dependence on τ, which is optimal compared to prior Markovian optimization work (e.g., Doan et al. 2020b) with exponential dependence on mixing time.

- **Honest and detailed discussion of limitations**: Section 2.4 transparently analyzes three key gaps—d²/m² vs. d/m, the mixing time contradiction, and (L/μ)^{2/3} vs. √(L/μ)—while citing relevant lower bounds and prior work to contextualize these as inherent to Markovian stochasticity.

## Weaknesses

### Fatal
None

### Major

- **The paper's central claim of "acceleration" is not supported by its own theoretical results, which show provably worse rates than the i.i.d. baseline**: The paper frames Markovian compression as accelerating distributed optimization (title: "Looking to the Past Helps Accelerate the Future"; abstract: "practical results demonstrate the superiority"). Yet Section 2.4 explicitly shows the compression-dependent term is d²/m² for Markovian vs. d/m for i.i.d. Random, plus an additional τ factor, and the accelerated method achieves only (L/μ)^{2/3} rather than √(L/μ). The paper itself calls this a "logical contradiction" (line 277). The paper argues this is inherent to Markovian analysis by citing other Markovian SGD papers that face similar limitations, which is a reasonable defense of the analysis quality but does not resolve the fundamental issue: the theory does not support the headline claim. The paper's real theoretical contribution is proving that convergence *can* be established for Markovian compressors (a feasibility result), not that they accelerate. This framing mismatch between claim and evidence is the paper's most significant problem.

- **Experimental methodology has meaningful selection bias**: Figure 1 caption states "All hyperparameters are fine-tuned, and best runs are selected" and Figure 2 caption states "Best runs for each method are displayed." While Table 1 reports mean±std over 5 runs (which partially addresses this), the convergence curves that visually demonstrate the paper's claims are cherry-picked. Moreover, KAWASAKI has two additional hyperparameters (b, π_Δ) beyond what Rand requires, giving it a larger effective tuning budget. The paper does not report whether equal tuning effort was invested across methods. Together, these issues make it difficult to determine whether the observed improvements are due to the Markovian mechanism or to selection/tuning advantages.

### Minor

- **Example 1 is a best-case degenerate scenario**: The only analytical argument for *why* Markovian compressors help considers a gradient nonzero in exactly one coordinate—a worst case for Rand and best case for BanLast. While useful as motivation, this example does not establish that the mechanism generalizes to typical optimization landscapes. The paper provides no analysis of what gradient structures (sparse, low-rank, structured) actually benefit from Markovian sparsification beyond this extreme case. — This matters because without understanding when the mechanism helps, the practical results remain unexplained.

- **No comparison with error-feedback methods in the main text**: Markovian compressors are biased at finite steps, and error feedback (e.g., EF21) is the standard approach for handling biased compression. Since both mechanisms use historical information, a comparison would be informative for assessing whether the Markovian approach offers advantages over existing history-using compression strategies. The paper mentions DIANA experiments but these results are deferred to the appendix.

- **The mixing-time contradiction deserves more investigation**: Since Assumption 5 requires ergodicity with uniform stationary distribution, the Markovian compressor asymptotically behaves like Rand. The benefit must come from transient behavior, yet the theory penalizes exactly this transient (larger τ → worse bounds). The paper acknowledges this but does not explore whether the practical gains might stem from something other than the Markovian structure per se (e.g., implicit regularization or momentum-like effects from the bias pattern).

### Trivial
None

## Nice-to-Haves

- Comparison against at least one error-feedback method (e.g., EF21-SGD) to contextualize the Markovian approach within the broader landscape of history-aware compression strategies.
- An ablation isolating the Markovian dependency from initialization bias: compare KAWASAKI against Rand with the same biased initial coordinate selection but without Markovian state transitions.
- Visualization of coordinate selection patterns over training for Rand vs. BanLast vs. KAWASAKI, correlated with gradient magnitude structure, to provide mechanistic evidence for *why* the method works in practice.
- Any theoretical result (even under restrictive assumptions) showing Markovian compressors can beat Rand, which would begin to bridge the theory-practice gap.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The theoretical framework contradicts the paper's central claim" (Harsh Critic, Issue 1)**: Overstated. The theory does not *contradict* the claim—it fails to *support* it. The theory does prove convergence for Markovian compressors (a feasibility result) and proves AMQSGD accelerates over MQSGD within the Markovian framework. The gap is specifically between Markovian and i.i.d. rates, which the paper honestly discusses. "Contradicts" implies the theory proves the opposite, which is not the case.

- **"KAWASAKI definition underspecified—no formal requirements for π_Δ beyond permutation invariance" (Harsh Critic, Section 2.1 notes)**: Theorem 1 gives the exact condition (permutation invariance) and provides three concrete examples of π_Δ. This is adequately specified for a new framework paper.

- **"DIANA experiments not presented in main text" (Harsh Critic, Section 3 notes)**: The DIANA results are in Appendix H, which was stripped by the parser. The paper references them explicitly. This is a presentation choice, not a missing result.

- **"Impossibility argument for better rates is not convincing without a lower bound for the compression setting specifically" (Harsh Critic, Section 2.2 notes)**: The paper cites multiple Markovian optimization papers that face the same d²/m² vs. d/m limitation, which provides empirical evidence that this is inherent. A specific lower bound would strengthen the argument but is not required.

- **"Algorithm 2 introduces four momentum parameters on top of Markovian hyperparameters" (Harsh Critic, Section 2.3 notes)**: This is standard for accelerated methods with unbounded gradient variance. The multi-momentum structure follows Beznosikov et al. (2023b). Not a weakness.

- **"BanLast gives negligible improvement over Rand (87.9 vs 88.0 test accuracy) while KAWASAKI gives larger gap (89.05)" (Harsh Critic, Section 3 notes)**: This is actually an informative observation about when smoother history accumulation helps, not a weakness per se.

- **"Missing related works" (implied by several reviewer suggestions)**: Per rules, I do not flag missing related works.

- **Formatting/style/typo complaints**: Removed per rules.

- **Reproducibility concerns about undisclosed hyperparameters or large artifacts**: Removed per rules.

## Novel Insights

The most insightful observation that emerges from the reviews is that the Markovian compression framework faces an inherent tension at its core: the stationary distribution requirement (Assumption 5) forces the compressor to asymptotically behave like the i.i.d. baseline it aims to beat, meaning any advantage must come from transient dynamics. Yet the theory's mixing-time penalty grows precisely with the strength of this transient dependence (larger K → better practical performance → larger τ → worse theoretical bounds). This suggests that either (a) the practical gains are not primarily from the Markovian structure but from some side effect like implicit regularization or coordinate-level momentum, or (b) the current proof technique (uniform noise bounding via "stepping back") is fundamentally too coarse to capture the structured variance reduction that Markovian dependence provides in practice. Distinguishing between these would significantly advance understanding of when and why Markovian compressors help.

## Suggestions

- Reframe the paper's contribution more carefully: the primary contribution is establishing a *new class* of compressors with provable convergence (a feasibility/novelty result), not proving acceleration over i.i.d. baselines. Aligning the framing with what the theory actually supports would strengthen the paper.
- Report all experimental figures with mean trajectories (shaded with ±std) rather than "best runs selected," and specify the hyperparameter search budget per method to address selection bias concerns.
- Add a simple ablation comparing KAWASAKI against Rand with a biased initial coordinate selection pattern but no Markovian transitions, to isolate whether the benefit comes from the Markovian state dynamics or the initial bias.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| SCALLION/SCAFCOM (jj5ZjZsWJe) | 8.0 | Accept (Spotlight) | Strong theory matching/beating baselines + robust experiments. Much stronger than this paper. |
| LoCoDL (PpYy0dR3Qw) | 7.5 | Accept (Spotlight) | Provably doubly-accelerated communication complexity. Theory clearly supports claims. Above this paper. |
| Det-CGD (ZEZ0CPmoSI) | 5.0 | Accept (Poster) | Similar issues (convergence neighborhood, missing EF baselines) but theory at least doesn't show worse rates than baselines. Slightly above this paper. |
| FedExProx (q2VK1Z8XFo) | 4.67 | Reject | Had theory no better than vanilla GD but improved it. Similar theory-practice gap, but FedExProx actually resolved it. Similar level. |
| Random-Walk Optimizer (yuYMJQIhEU) | 4.5 | Reject | Combines existing tools, Markov chain not fully analyzed. Weaker novelty than this paper but similar theoretical limitations. |
| CORE (ER1VDuwWvB) | 3.67 | Reject | Claims compression improvement but theory doesn't clearly support it. Less honest about limitations than this paper. Below this paper. |
| Pan for Gold (1gqR7yEqnP) | 2.2 | Reject | Vague theory, overclaimed practical impact, poor writing. Far below this paper. |

The paper has genuine novelty (a new class of compressors with no prior work), competent technical execution (complete convergence theory, novel proof technique), and honest discussion of limitations. However, the central claim of "acceleration" is not supported by the paper's own theory, which shows provably worse rates than i.i.d. baselines. The experiments, while consistent, have selection bias concerns. Compared to Det-CGD (5.0, accepted poster with similar issues but no theory-worse-than-baseline problem), this paper's theory-practice gap is more fundamental. Compared to FedExProx (4.67, rejected), this paper has more novelty but a similar magnitude of mismatch between claims and theoretical support. I place it between these two anchors.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>