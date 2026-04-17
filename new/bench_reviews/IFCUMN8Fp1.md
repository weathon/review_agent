Now I have a good calibration set. Let me synthesize the final review.

**Calibration summary:**
- High-quality POMDP/PSR papers accepted with scores 6-8: jId5PXbBbX (Accept poster, 6,5,6,8), 1hsVvgW0rU (Accept poster, 6,6,6,6), 8BAkNCqpGW (Accept poster, 8,8,8,8)
- Weaker POMDP/PSR papers rejected with scores 3-5: lkIRFglmTp (Reject, 3,3,3,6), fnO5h1CFyh (Reject, 3,3,3,3), B5kAfAC7hO (Reject, 5,5,6), 1mMjZvEhwH (Reject, 3,3,3,3,6,3)

The paper under review has:
- Novel theoretical contribution (connecting PSRs and tensor decomposition, partition-level recovery)
- Formal theorem and proofs
- Limited experiments on small toy domains
- Strong assumptions (full-rank transitions, ergodicity)
- No finite-sample analysis
- No scalability analysis

Compared to jId5PXbBbX (PSR/UCB paper, accepted poster, avg ~6.25): That paper had stronger theoretical guarantees (finite-sample bounds) but no experiments. This paper has experiments (though limited) but no finite-sample bounds.

Compared to 1hsVvgW0rU (POMDP learning with hindsight, accepted poster, avg 6): That paper had stronger theoretical results (sample-efficient learning algorithms for new POMDP subclasses) but was also limited to tabular settings.

Compared to B5kAfAC7hO (POMDP representation learning, rejected, avg ~5.3): That paper had restrictive assumptions, limited novelty increment over prior work, and lack of ablation studies. Similar issues to our paper.

The paper under review is between these - it has a genuine theoretical contribution (partition-level identifiability), but the novelty is somewhat incremental (connecting known PSR identifiability results with an off-the-shelf joint diagonalization method), the assumptions are restrictive, experiments are limited, and there's no finite-sample analysis. I'd place this around a 4-5 range - below the accepted papers but above the clearly weak ones.</think>Now I have sufficient calibration data. Let me compose the final review.

---

## Summary

This paper presents a method for learning discrete POMDP parameters from action-observation sequences collected under a uniform random exploration policy. The key idea is to bridge PSR spectral learning (which learns system dynamics up to an unknown similarity transform) with tensor decomposition methods to recover the similarity transform. The authors show that transition and observation matrices can be recovered up to a "full-rank observability partition"—states that share identical observation distributions across all full-rank transition actions are grouped together—yielding explicit likelihoods that enable downstream tasks like reward specification after learning. The main theoretical result (Theorem 1) characterizes what can be recovered in the infinite-data regime, and experiments on small POMDPs validate parameter recovery and planning performance.

## Strengths

- **Clear theoretical contribution connecting PSRs and tensor methods.** The paper formally establishes that PSR model parameters correspond to POMDP parameters up to a similarity transform (Proposition 1, extending Carlyle & Paz 1971; Balle et al. 2014), and shows how joint diagonalization can estimate this transform. This unifies two previously separate lines of work in a principled way.

- **Relaxation of prior identifiability assumptions.** Previous tensor methods (Azizzadenesheli et al., 2016; Guo et al., 2016) required unique observation distributions per state for each individual action. This paper relaxes this to requiring uniqueness only when observation distributions are aggregated across all full-rank actions simultaneously—a strictly weaker condition that covers POMDPs like Tiger and Sense-Float-Reset that prior methods cannot handle (Section 1, 4.1.1).

- **Honest characterization of identifiability limitations.** The "full-rank observability partition" concept is introduced formally and used to precisely delineate what can and cannot be recovered. The paper is transparent that when states share observation distributions across all full-rank actions, only partition-level transitions—not individual state transitions—can be identified.

- **Motivating running example and reward specification experiment.** The Sense-Float-Reset domain (Figure 1) effectively illustrates the key challenges (singular transition matrices, states with identical observation distributions). The reward specification experiment (Figure 4) demonstrates a concrete advantage of having explicit observation/transition likelihoods over black-box PSR models, particularly in noisy observation settings.

## Weaknesses

### Major

- **Limited experimental evaluation on small, hand-crafted domains only.** All experiments are conducted on very small POMDPs (Tiger: 2 states, T-Maze: small, Sense-Float-Reset: 5 states, two 3-state hallway domains). There is no evidence the method scales to problems with more states, actions, or observations. The Hankel matrix construction grows exponentially with history/test length, and no scaling analysis (computational or statistical) is provided. For comparison, similar spectral/tensor POMDP learning papers that were accepted (e.g., 1hsVvgW0rU at score ~6) provided sample complexity bounds even without large-scale experiments.

- **No finite-sample analysis.** Theorem 1 is stated only in the infinite-data regime. The paper acknowledges this gap ("Carefully studying our algorithm under a PAC-learning framework would contribute to our understanding of the computational complexity of learning POMDPs in general") but provides no convergence rates, sample complexity bounds, or even informal discussion of how estimation errors in the Hankel matrix, SVD truncation thresholds, or eigenvalue separation in Lemma 1 affect partition identification or parameter recovery quality. Since the method relies on nontrivial numerical steps (Hankel estimation, SVD truncation, eigen-decomposition of noisy matrices, random joint diagonalization, block-rotation), this omission is significant—the gap between the asymptotic theory and practical implementation is where things break.

- **Restrictive assumptions with limited practical validation.** The method requires: (a) at least one full-rank transition action, (b) ergodicity of the (s,a,o) chain under uniform random exploration, and (c) full-rank Forw and Back matrices. While Section 4.1.1 discusses when full-rank transitions arise (stochastic action failures → convex combinations with identity), this argument applies to a specific modeling choice rather than validating that common POMDP benchmarks satisfy these assumptions. Many standard POMDPs have deterministic or rank-deficient transition matrices (as the paper acknowledges), and the method falls back to using only A_full, which may be empty. The ergodicity assumption excludes any POMDP with absorbing states or irreversible transitions, which are common in planning.

### Minor

- **Limited baselines and ablations.** The only comparison is against EM (which is well-known to converge to local optima) and PSR (which is not designed to recover explicit POMDP parameters). There is no comparison against the tensor decomposition methods of Azizzadenesheli et al. (2016) or Guo et al. (2016) that the paper claims to improve upon, nor any ablation studying the impact of the number of full-rank actions, SVD truncation thresholds, or the random weights/blocks (w^{ao}, R).

- **Key algorithmic steps described informally.** The construction of P̃ from P′, R, and m∞ (end of Section 4.3) is critical but described narratively with references to appendices rather than presented as a complete algorithm with explicit conditions and failure modes. No pseudocode is provided in the main text.

- **Reward specification experiment is narrowly constructed.** The directional and noisy hallway domains are specifically designed so that observation-based rewards fail in one case and state-based rewards succeed (or vice versa). While illustrative, this does not establish broad practical advantage; the directional domain shows that state-based reward assignment using the recovered model *fails* when transition matrices converge slowly, which somewhat undermines the claim of practical utility.

- **Planning comparison lacks variance reporting.** The planning experiments (Figure 3) compare average returns across methods but do not report variance across seeds or runs, making it difficult to assess whether "similar performance" is a robust finding.

### Trivial

- The notation is heavy but generally consistent; the forward/backward/Hankel construction in Section 3 follows the standard PSR literature.

## Nice-to-Haves

- Experiments on at least one domain with 10+ states and/or a non-trivial (non-singleton) observability partition to validate the core theoretical claim under the conditions that actually distinguish this work from prior methods.

- Any finite-sample analysis—even informal bounds on how estimation error in the Hankel matrix propagates through SVD truncation and joint diagonalization to affect partition identification.

- Comparison against Azizzadenesheli et al. (2016) or Guo et al. (2016) as baselines, to demonstrate the claimed advantage in handling POMDPs with non-unique per-action observation distributions.

- Computational complexity analysis or empirical wall-clock time reporting.

## Removed Points

- **"Theoretical contribution is a repackaging of known results" (from Harsh Critic):** This overstates the case. While Proposition 1 restates Carlyle & Paz / Balle et al., the joint diagonalization approach across all full-rank actions (rather than per-action), the formalization of observability partitions, and Theorem 1 are genuine new contributions. The connection between PSR similarity transforms and partition-level recovery is non-trivial. Kept as a minor novelty concern but removed as a fatal criticism.

- **"EM is a weak baseline" (from Neutral Reviewer):** EM is a natural baseline for the specific task of POMDP parameter recovery from trajectories. It is the standard comparison for spectral/tensor learning papers in this space. However, the comparison would be stronger with EM from multiple restarts and against the specific tensor baselines the paper claims to improve upon.

- **"Overclaiming in abstract about matching PSR planning performance" (implied by Harsh Critic):** The abstract says "our experiments suggest that these partition-level transition models learned by our method, with a sufficient amount of data, meets the performance of PSRs as models to be used by standard sampling-based POMDP solvers." This is appropriately qualified ("our experiments suggest," "with a sufficient amount of data"). Not an overclaim.

- **"Paper should compare to deep recurrent baselines" (from Neutral Reviewer):** This is a scope mismatch. The paper is about recovering explicit POMDP parameters, which is a fundamentally different objective from learning implicit state representations. This would be a nice-to-have comparison but is not a core flaw.

- **"No real-world POMDP examples" (implied from human finder patterns):** The paper's contribution is theoretical/algorithmic, and toy domains are standard for this type of spectral learning work. This is a scope issue, not a fundamental flaw.

## Novel Insights

The key insight—that joint diagonalization across all full-rank actions simultaneously can separate states whose observation distributions are distinguishable only when aggregated across actions, which per-action tensor methods cannot do—is the paper's most important conceptual contribution. This shifts the identifiability condition from requiring per-action observability uniqueness to requiring observability uniqueness in aggregate, which is strictly weaker and covers important POMDPs like Tiger. The formalization of this via "full-rank observability partitions" provides a clean characterization of the limits of what can be recovered, which is more honest than claiming full recovery where it is impossible.

## Suggestions

1. Add at least one experiment on a domain with a genuine coarse observability partition (where multiple states share observation distributions across all actions) to test the partition-level planning claim directly.

2. Provide any finite-sample analysis, even informal bounds connecting Hankel estimation error to partition misidentification rates.

3. Include an explicit pseudocode algorithm for the full pipeline in the main text, with clear specification of inputs, outputs, and failure conditions.

4. Run EM with multiple random restarts and compare against the best-of-N result to make the EM baseline more informative.

5. Report variance across seeds in all experimental figures and include wall-clock time comparisons.

## Evaluation

**Originality:** The paper makes a meaningful connection between PSR spectral learning and tensor decomposition, with the joint diagonalization across all full-rank actions being a genuinely novel idea that relaxes prior assumptions. However, the overall technical contribution is incremental—it builds on known identifiability results and applies an off-the-shelf joint diagonalization method (He et al., 2024).

**Importance of research question:** Learning POMDP parameters from interaction data is an important problem with practical motivation (robotics, planning under uncertainty). The question of what can be identified under partial observability is fundamental.

**Claim support:** The theoretical claim (Theorem 1) is well-supported with proofs. The practical claims (planning performance, reward specification advantage) are supported only on small, hand-crafted domains, which limits confidence in broader applicability.

**Experimental soundness:** The experiments are correctly designed for their scale, but the scale is too small and the baselines too limited to strongly support the practical claims.

**Clarity:** The paper is reasonably well-written, with a good running example and clear (if notation-heavy) exposition. The main algorithmic pipeline could be presented more explicitly.

**Value to community:** The theoretical characterization of identifiability via observability partitions is a valuable contribution. However, without finite-sample analysis or scaling evidence, the practical impact is limited.

## Score and Decision

**Calibration:**
- jId5PXbBbX (PSR UCB, accepted poster, scores 6,5,6,8, avg ~6.25): Stronger theoretical guarantees (finite-sample bounds), but no experiments. More novel algorithmic framework.
- 1hsVvgW0rU (POMDP hindsight learning, accepted poster, scores 6,6,6,6, avg 6): Novel POMDP subclasses with sample-efficient learning, clean theory, but also limited to tabular.
- B5kAfAC7hO (POMDP representation, rejected, scores 5,5,6, avg ~5.3): Restrictive assumptions, limited novelty, lack of ablation — similar weaknesses to our paper.
- lkIRFglmTp (Lambda discrepancy, rejected, scores 3,3,3,6, avg ~3.75): Serious technical errors in proofs, poorly presented.
- 8BAkNCqpGW (Confounded POMDP policy gradient, accepted poster, scores 8,8,8,8): Strong novelty, theoretical + empirical contributions, well-executed.

This paper has genuine novelty in connecting PSRs and tensor decomposition via observability partitions, but the contribution is weakened by: (1) incremental theoretical depth beyond the key insight, (2) no finite-sample analysis, (3) limited experimental validation on tiny domains, and (4) strong assumptions. It sits below papers like jId5PXbBbX and 1hsVvgW0rU (which were borderline-accepted with stronger theory), and somewhat above B5kAfAC7hO (which was rejected with similar weaknesses). The novelty is real but modest; the experiments are suggestive but thin.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>