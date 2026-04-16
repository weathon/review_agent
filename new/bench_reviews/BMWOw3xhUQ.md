Now let me carefully analyze each criticism from the harsh reviewer against the actual paper content.

Let me re-read the paper carefully, focusing on the key theoretical claims and experimental details.

The key issues to verify:

1. **Theorem 4.1 validity** - The harsh reviewer claims the theorem is wrong because expectile regression gives conditional expectiles, not global maxima. Let me examine this carefully.

Theorem 4.1 says: "Suppose Q-function is predicted by the model itself, we first define SG ≐ (s, g, a, Q). For m ∈ (0,1), denote Q^m(SG) = arg min ℓ_Q^m(SG), then we have lim_{m→1} Q^m(SG) = Q_max, where Q_max = max_{s,a,g} Q(s,a,g) denotes the maximum Q-function with actions from offline dataset."

The loss is: ℓ_Q^m = E_{(s,a,g)∈D} [m - 1(ΔQ < 0)] ΔQ², with ΔQ = Q - Q̂

The harsh reviewer's point is: For a fixed (s,a,g) with a single Q value, the minimizer Q̂ = Q regardless of m. There's no mechanism to push Q̂ above the target. This seems correct for the per-sample loss. However, the paper says "Q-function is predict by the model itself" and uses notation SG ≐ (s,g,a,Q). The key question is: how is the Q target generated? If we're looking at the VAE-estimated Q for each (s,a,g), then yes, there's only one Q value per sample, so the expectile doesn't push it up.

BUT - looking more carefully, the DT version has the model predict both Q and actions. The Q appears to be predicted by the model and then used as both a target and a conditioning variable. The expectile loss is on ΔQ = Q - Q̂ where both Q and Q̂ come from the model. This is circular. Unless Q is the VAE-estimated probability... but then it's still per-sample.

Actually, looking at the implementation more carefully: The model outputs Q̂. The "Q" in the loss (the target) comes from the VAE estimation. So for a given (s,a,g), the VAE gives a Q value. The expectile loss would push Q̂ toward this value... but with m > 0.5, Q̂ should be pushed toward values larger than the VAE-estimated Q. But for a single target, the only way this works is if there are multiple Q values for the same input.

Wait - in the DT version, the model processes sequences. Different samples in the dataset may share similar (s,g) pairs but with different actions 'a' and different Q values. The expectile loss would then push Q̂(s,g) toward the higher Q values among those with similar (s,g). But the loss is defined per sample as ℓ_Q^m = E_{(s,a,g)∈D} [m - 1(ΔQ < 0)] ΔQ². This IS a per-sample expectation.

For the RvS version, the value model takes (s,g) as input, not (s,a,g). So for the same (s,g), multiple actions might have different Q values. The expectile regression would then push the value model's output toward the maximum Q among those actions. This is the typical IQL-style setup. This IS valid for the RvS case: if for a state-goal pair (s,g), the dataset contains multiple actions with different Q values, the expectile pushes toward the highest Q value for that (s,g), which is max_a Q(s,a,g) for the dataset distribution. This is indeed how IQL works.

For the DT version though, the Q-prediction is autoregressive and context-dependent, and it's less clear what's happening.

So Theorem 4.1's claim that Q̂ → Q_max = max_{s,a,g} Q(s,a,g) is indeed problematic. The expectile would push toward max_a Q(s,a,g) for each (s,g), but not toward the global maximum over all (s,a,g). This seems like a valid concern.

2. **Corollary 1** - The claim of "equivalence to goal data augmentation" is strong and relies on Theorem 4.1.

3. **VAE as Q-estimator** - The concern about which policy π the Q-function corresponds to is valid. Theorem 3.1 says Q^π(s,a,g) = p_π^+(g|s,a), but the CVAE is trained on dataset samples, which come from the behavior policy, not the learned policy π. This is a concern.

4. **Missing Q target specification** - This is a valid point about under-specification.

5. **Stitching claim not isolated** - The experiments don't decompose results into stitching vs non-stitching cases.

6. **Policy semantics during evaluation** - Whether g_t is held constant or updated.

Now let me also check if the paper actually has an appendix where Theorem 4.1 is proven.

The paper says "The proof is in Appendix A.2" for Theorem 4.1, but the PDF content ends with "Rest of paper (reference and Appendix) is removed." So we can't verify the proof.

Let me also think about the calibration. Looking at the anchor papers:
- qg5JENs0N4 (Closing the Gap, TGDA paper): Scores 6,5,5,6, Accept poster - This is a directly related paper on the same topic
- 7zY781bMDO (Free from Bellman Completeness, MBRCSL): Scores 6,8,5,5, Accept poster - Stitching via SL
- IUwqJ8VT4F (Revisiting Max-Return SM): Scores 3,5,3,5, Reject - Related framework
- QDrG0ALevs (Advantage-Conditioned Diffusion): Scores 3,3,5,6, Reject - Uses expectile regression
- BxLK1M1f8T (Doctor): Scores 6,3,5,6, Reject - Combines SL and TD

The current paper has:
- Similar topic to TGDA paper (qg5JENs0N4) which got 5.5 avg, poster
- Theoretical issues with Theorem 4.1 and Corollary 1
- Decent but not overwhelming empirical results
- Some under-specification issues

Key weaknesses I should keep:
1. Theorem 4.1 appears problematic - expectile regression pushes toward conditional maxima, not global maxima (MAJOR)
2. VAE Q-estimation doesn't clearly correspond to any policy's Q-function (MAJOR, but somewhat mitigated by the fact that in goal-conditioned RL with sparse rewards, the behavior policy Q is still meaningful)
3. The paper doesn't cleanly specify the Q target in training (MODERATE)
4. Stitching claim not convincingly tested - no decomposition into stitching vs non-stitching cases (MODERATE)
5. Remaining gap with TD methods is substantial (MODERATE)
6. Goal semantics during evaluation unclear (MINOR)

Let me now write the review.</think>Now let me carefully synthesize the review, verifying each claimed weakness against the paper.

## Summary

This paper proposes GCReinSL, a method to endow outcome-conditioned behavioral cloning (OCBC) methods with trajectory stitching capability by embedding Q-function maximization into supervised learning. The key idea is to use a VAE to estimate Q-values (via the equivalence between goal-conditioned Q-functions and discounted state occupancy probabilities) and then apply expectile regression to predict the "in-distribution maximum Q-function," which conditions action prediction during inference. The authors claim this is equivalent to goal data augmentation for OCBC methods, and demonstrate improvements on stitching-focused benchmarks.

## Strengths

- **Important and well-motivated problem.** Trajectory stitching is a well-known limitation of SL-based RL methods, and endowing OCBC methods with this capability without requiring unstable TD learning dynamics is a meaningful research direction. The maze example in Section 4.1 clearly illustrates why OCBC fails at stitching.

- **Consistent empirical improvements over OCBC baselines.** GCReinSL consistently improves over vanilla DT and RvS, and outperforms prior data augmentation methods (SGDA, TGDA) on the Ghugare et al. stitching-specific datasets (Figures 4-5). On D4RL Antmaze-v2 (Table 1), GCReinSL achieves 306.4 total score compared to Reinformer's 174.8 and DT's 126.3, representing a substantial improvement over prior sequence modeling methods.

- **Architecturally clean inference pipeline.** The approach of first predicting the maximum Q-value and then conditioning action generation on it (Eqs. 12-13) is a simple and intuitive way to inject RL-like maximization into SL frameworks without Bellman backups.

## Weaknesses

### Major:

- **Theorem 4.1's claim that expectile regression yields the global maximum Q-function is incorrect as stated.** The theorem asserts $\lim_{m\to 1} Q^m(\text{SG}) = Q_{\max}$ where $Q_{\max} = \max_{s,a,g} Q(s,a,g)$. However, the expectile loss (Eq. 9) decomposes separately over each $(s,a,g)$ triplet. For the RvS variant, where the value model takes $(s,g)$ as input, the expectile would push $\hat{Q}(s,g)$ toward $\max_a Q(s,a,g)$ for that specific $(s,g)$—a *conditional* maximum over actions, not the global maximum over all $(s,a,g)$. For the DT variant, the per-sample loss with a single Q-target per sample provides no mechanism to push predictions above that target value at all for a given input. The claim that $\hat{Q}$ converges to a single global maximum $Q_{\max}$ is not justified by the loss formulation. This undermines the narrative that expectile regression directly implements "Q-conditioned maximization" in the RL sense.

- **The VAE-based Q-estimation does not clearly correspond to any policy's Q-function.** Theorem 3.1 states $Q^\pi(s,a,g) = p_\pi^+(g|s,a)$, a property that holds for a *specific* policy $\pi$. The CVAE is trained on dataset tuples $\{(s,a,g)\}$, which are generated by the behavior policy, not by the learned policy $\pi$. The paper never specifies which policy the estimated Q-function corresponds to. Treating the CVAE output as a Q-function in the RL sense requires the assumption that $p_{\text{data}}(g|s,a) \approx Q^\pi(s,a,g)$ for some meaningful policy $\pi$, which is not established. This breaks the logical chain that "maximizing expected Q under the goal distribution" is actually being achieved.

- **The "equivalence to goal data augmentation" claim (Corollary 1) is unsubstantiated.** Corollary 1 asserts Q-conditioned maximization is *equivalent* to goal data augmentation, enabling stitching. This strong claim relies on Theorem 4.1 (which is problematic as noted above) and lacks an operational definition: what specific augmented dataset does GCReinSL correspond to? Which state-goal pairs are implicitly sampled, and with what probabilities? No formal argument connects the expectile loss to a specific augmentation procedure beyond the illustrative maze example. Given that the proof is in an appendix not available for verification, and the theorem it depends on is flawed, this equivalence remains unproven.

### Minor:

- **The Q-target used during training is under-specified.** The paper introduces the VAE-estimated $\widehat{\log p_\pi^+}(g|s,a)$ in Eq. 8, then references $Q = Q^\pi(s,a,g)$ in Eq. 9, but never provides an explicit formula mapping from the VAE log-probability to the scalar Q-target. Whether Q is a raw probability, a log-probability, or some transformed value directly affects the interpretation of the expectile loss and whether "maximizing Q" has the intended semantic meaning.

- **Stitching is not isolated in the experimental evaluation.** The benchmarks from Ghugare et al. (2024) are designed to test stitching, but success rates are reported in aggregate without decomposing results into cases that require stitching (unseen start-goal pairs) vs. cases that don't. Without this breakdown, it is difficult to attribute gains specifically to stitching rather than better modeling of in-distribution trajectories.

- **Remaining gap with TD methods is substantial.** On D4RL Antmaze-v2, GCReinSL achieves 306.4 total vs. IQL's 432.0, particularly lagging on harder tasks (e.g., large-diverse: 30.2 vs. 53.0). The title claims to "bridge the gap," but a ~30% relative gap remains. The Antmaze-Large results for GCReinSL+RvS (0.02) are effectively failure, suggesting the method struggles in more challenging settings.

- **Hyperparameter sensitivity acknowledged but under-analyzed.** The paper admits that large $m$ values can cause performance decline (Section 5.5) and that $L=500$ vs. $L=5$ is chosen very differently across datasets. The claim that OCBC methods are "free from...high sensitivity to hyperparameters" (pg. 2) is somewhat contradicted by GCReinSL's own sensitivity profile, which trades TD instability for expectile parameter tuning.

## Nice-to-Haves

- **Component ablation isolating VAE estimation from expectile regression.** A baseline using a simpler Q-estimator (e.g., Monte Carlo returns) with the same expectile loss would clarify whether the VAE is essential or whether expectile regression alone can provide some stitching benefit.

- **Visualization of predicted Q-values along stitched trajectories** to verify the claimed mechanism (Q transitioning from 0 to 1 at stitch points, as described in Section 4.1).

- **Discussion of what went wrong on Antmaze-Large with RvS** (2% success), which would help understand failure modes and practical applicability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's Point 6 (Policy semantics during evaluation):** The reviewer questions whether $g_t$ is held constant or updated during inference. The paper states in Section 3.1 that the policy is $\pi(a|s,g)$ where $g$ is the goal, and in goal-conditioned RL the goal is fixed per episode. This is standard in goal-conditioned RL and not ambiguous—the goal is provided by the environment and held constant.

- **Harsh Critic's Point about CVAE not matching any policy's Q:** While formally valid, in the goal-conditioned RL setting with sparse rewards where the dataset is multi-policy/multi-goal, the behavior policy Q-function still encodes useful reachability information. The concern is real but should be moderated—the VAE estimates $p_{\text{data}}(g|s,a)$, which under the dataset distribution provides a meaningful score for "can state $s$ with action $a$ reach goal $g$," even if it's not precisely $Q^\pi$ for the learned policy.

- **Neutral Reviewer's Point 3 (Hyperparameter sensitivity contradicts stated advantage of OCBC):** This is a fair observation but is already partially addressed by the paper's own conclusion. It's more of a nuanced trade-off than a contradiction—OCBC methods are simpler to train but GCReinSL adds back some complexity for stitching capability. Moved to Minor weakness.

- **Spark's suggestion about no goal-conditioned TD baselines on Ghugare et al. datasets:** The paper already compares against CQL and IQL on D4RL, and the Ghugare et al. datasets weren't designed for TD baselines. The comparison against data augmentation methods (TGDA, SGDA) is more directly relevant than adding goal-conditioned TD variants.

- **Spark's suggestion about EDT/Reinformer numbers not re-implemented:** The paper transparently reports that these numbers come from original papers (Table 1 caption). This is standard practice and not a methodological flaw.

- **Formatting/notation nitpicks** from various reviewers are removed as per instructions.

## Novel Insights

The connection between expectile regression on Q-values and implicit goal data augmentation is an interesting conceptual framing, even if the theoretical formalization needs significant repair. The idea of decoupling the "which trajectory segment to follow" decision (via Q-conditioning) from the "what action to take" decision (via SL) is a promising architectural direction for combining the strengths of both paradigms, rather than the more common approach of simply adding TD losses to SL objectives.

## Suggestions

- **Repair Theorem 4.1.** The theorem should be reformulated to correctly characterize what the expectile loss converges to: for the RvS variant, this would be $\hat{Q}(s,g) \to \max_a Q(s,a,g)$ (conditional maximum over actions for each state-goal pair), not the global $Q_{\max}$. This still provides stitching-relevant behavior but is more honest about the scope.

- **Clarify the Q-target computation.** Explicitly write the formula mapping VAE log-probability to the scalar Q used in Eq. 9 (e.g., $Q = \exp(\widehat{\log p_\pi^+}(g|s,a;\varphi,\psi,L))$, or whether normalization is applied).

- **Add a stitching-specific evaluation.** Decompose the Ghugare et al. results into success on seen vs. unseen (state, goal) pairs to isolate stitching ability from general policy improvement.

## Score and Decision

Calibration comparison:
- **qg5JENs0N4 (TGDA/Closing the Gap)**: Scores 5.5 avg, Accept poster. Directly related work on stitching in GCRL with clean theory and simpler data augmentation. GCReinSL tackles the same problem with a more complex approach but with problematic theoretical claims.
- **7zY781bMDO (MBRCSL)**: Scores 6.0 avg, Accept poster. Another stitching-via-SL paper with theoretical grounding and strong experiments.
- **IUwqJ8VT4F (Revisiting Max-Return SM)**: Scores 4.0 avg, Reject. Related max-return framework with empirical analysis but limited novelty.
- **QDrG0ALevs (Advantage-Conditioned Diffusion)**: Scores 4.25 avg, Reject. Uses expectile regression but with limited gains and similar concerns about the approach.

This paper has meaningful empirical contributions (improvements over baselines on stitching tasks and D4RL) but has significant theoretical issues: Theorem 4.1 is incorrect as stated, the VAE-to-Q connection is not rigorous, and the equivalence claim to data augmentation is unproven. These are not minor technicalities—they are the conceptual backbone of the paper. The empirical results are decent but not transformative (still substantial gap to IQL). The paper is roughly at the level of the IUwqJ8VT4F and QDrG0ALevs papers in terms of theoretical soundness, though with stronger empirical results than both.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>