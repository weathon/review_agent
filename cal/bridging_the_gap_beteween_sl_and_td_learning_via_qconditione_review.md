=== CALIBRATION EXAMPLE 23 ===

# Final Consolidated Review
## Summary
This paper proposes **Q-conditioned maximization supervised learning** for goal-conditioned offline RL, aiming to give supervised-learning-based methods (specifically DT and RvS) a form of trajectory stitching without explicit dynamic programming. The method estimates a goal-reaching score with a CVAE-style model, then uses expectile regression to bias the model toward higher in-distribution Q-values and conditions action prediction on the predicted Q. Empirically, the approach improves over plain OCBC baselines and over prior goal-augmentation baselines on stitching-oriented Pointmaze/Antmaze tasks, and it substantially improves over prior sequence-modeling baselines on D4RL Antmaze-v2, though it still remains behind strong TD methods.

## Strengths
- **The paper targets a real deficiency of OCBC methods and offers a concrete mechanism tailored to that failure mode.** The maze example in Sec. 4.1 is specific and useful: it distinguishes between naïvely forcing a high Q-value everywhere versus predicting a *state-dependent in-distribution* high Q-value that changes at the stitching point. This is more than generic motivation; it directly explains the design choice behind the method.
- **The method is architecture-portable rather than bespoke.** The authors instantiate the idea for both DT and RvS (Sec. 4.4), which supports the claim that the contribution is a general supervised-learning augmentation rather than a one-off model trick.
- **The empirical gains over SL-style baselines on stitching-oriented datasets are real, though uneven.** In Fig. 4 and Fig. 5, GCReinSL generally improves over vanilla OCBC and usually over SGDA/TGDA as well. The gains are especially visible in Pointmaze-Umaze and Pointmaze-Medium for DT, and in several RvS settings.
- **The D4RL Antmaze-v2 results show a meaningful advance over prior sequence-modeling baselines.** In Table 1, GCReinSL is much stronger than DT/EDT/Reinformer on the harder medium and large tasks (e.g., medium-play 49.0 vs 13.2 for Reinformer and near-zero for DT/EDT). This is a specific and significant strength of the submission.
- **The core conceptual insight is interesting:** using expectile-style upper-tail fitting to induce an implicit preference for higher-reachability goals/actions within the offline distribution is a plausible route to making SL methods behave more like value-based methods without explicit Bellman backups.

## Weaknesses
### Fatal
- **The paper’s formal bridge between the RL objective and the implemented estimator is not technically established.**  
  The central setup defines \(Q^\pi(s,a,g)=p_\pi^+(g\mid s,a)\) in Eq. (5), i.e., a probability-like quantity. But Sec. 4.2 then states: “After training this VAE, we can approximate the probability \(p_\pi^\pi(g|s,a)\) in Eq. (5) by \(-\mathcal L_{\text{ELBO}}\),” and Eq. (8) uses \(\log p_\psi(g\mid s,a)\) as the practical estimator. This is not a minor notation issue: the method subsequently treats the learned quantity as a Q-value to be maximized via expectile regression, but the paper never proves that maximizing this log-likelihood-style surrogate is equivalent to maximizing the original probability-valued Q-objective, nor does it justify that the transform preserves the intended semantics for the later theory. Since the paper’s main contribution is explicitly framed as a principled bridge from goal-conditioned RL objective maximization to SL, this gap substantially weakens the claimed theoretical foundation.

### Major:
- **Theorem 4.1, as stated, does not justify the state-goal-conditional behavior the algorithm actually needs.**  
  The theorem states
  \[
  \lim_{m\to 1} Q^m(\mathrm{SG}) = Q_{\max},
  \]
  where \(Q_{\max}=\max_{s,a,g}Q(s,a,g)\). But the algorithmic story throughout the paper is about predicting the **current in-distribution maximum** relevant to the present state-goal context, not the global maximum over all states/actions/goals in the dataset. A global maximizer is not the right object for action selection at a particular \((s,g)\), and the paper itself repeatedly uses local language such as “current maximum Q-function” and “in-distribution maximum.” The theorem therefore does not formally support the mechanism the paper claims.
- **The empirical framing overstates consistency and “bridging” relative to the actual results.**  
  The method clearly improves over sequence-modeling baselines and often over goal-augmentation baselines, but several claims are stronger than the evidence supports. For example, Fig. 4 does **not** show consistent outperformance across every case: on Pointmaze-Medium with RvS, GCReinSL (0.50) is below TGDA (0.60). In Fig. 5, gains on Antmaze-Large are marginal in absolute terms (e.g., DT 0.12 vs 0.10; RvS 0.02 vs 0.00). And on D4RL Antmaze-v2, the method still trails TD-learning substantially in aggregate (306.4 vs 371.2 for CQL and 432.0 for IQL). So the paper supports “meaningfully narrows the gap” better than “bridges the gap.”
- **The experimental support is incomplete for the paper’s core comparison target, namely TD-style stitching.**  
  On the stitching-specific Ghugare et al. datasets, the paper compares against OCBC and goal-augmentation methods, but not against TD methods such as CQL/IQL, even though the central motivation is to recover a property “typically associated with RL approaches such as TD learning.” Without showing TD baselines on those same stitching-focused benchmarks, the reader cannot assess how much of the purported SL–TD gap has actually been closed there.
- **Key mechanism ablations are missing, making it hard to validate the proposed causal story.**  
  The paper does not isolate several critical components: whether the benefit comes from the Q-conditioned inference itself versus merely adding an auxiliary Q-prediction loss; whether the CVAE-based estimator is necessary versus simpler targets; and whether expectile regression is essential compared with standard regression under the same architecture. These omissions matter because the paper makes a fairly specific mechanistic claim about *why* stitching improves.
- **The method’s practical robustness appears limited on the harder tasks.**  
  The paper itself notes sensitivity to hyperparameters, and Fig. 6 shows that performance depends materially on \(m\). More importantly, the absolute success rates on several harder stitching benchmarks remain very low, especially in Antmaze settings from Fig. 5. This does not negate the gains, but it does limit the practical significance of the current method.

### Minor
- **There is an unresolved theory/practice mismatch around the expectile parameter \(m\).**  
  Theorem 4.1 motivates \(m\to 1\), but Fig. 6 and the paper’s own discussion acknowledge that very large \(m\) can hurt due to overfitting to large Q-values. The practical selection principle for \(m\) is therefore unclear.
- **The D4RL comparison is incomplete across the paper’s two flagship instantiations.**  
  Sec. 4 presents both DT and RvS versions of GCReinSL, but Table 1 only reports the DT-style sequence-modeling comparison. Reporting the RvS variant on D4RL Antmaze-v2 would give a fuller picture of generality.
- **The paper does not evaluate the accuracy or calibration of the learned Q/reachability estimator.**  
  Since the entire method hinges on estimating a meaningful goal-reaching score from the CVAE and then maximizing it, some direct validation of that estimator would strengthen the claims considerably.
- **The distinction between the Ghugare et al. Antmaze evaluation and D4RL Antmaze-v2 evaluation is not explained clearly enough.**  
  Since baseline performance differs dramatically across these sections, clearer exposition of what changes across the two setups would help interpret results.

### Trivial
- **Computational overhead is not discussed.**  
  The additional CVAE estimator, importance sampling parameter \(L\), and two-stage inference pipeline likely add nontrivial cost over vanilla OCBC. This is not a core flaw, but some accounting would improve the paper’s practical assessment.

## Nice-to-Haves
- Add TD baselines (e.g., CQL/IQL) on the Ghugare et al. stitching datasets to directly quantify the remaining gap on the benchmarks most aligned with the paper’s motivation.
- Add ablations for: (i) Q auxiliary loss without Q-conditioned inference, (ii) standard regression vs expectile regression, and (iii) simpler Q/reachability estimators vs the CVAE.
- Visualize predicted Q-values along successful and failed trajectories to verify the proposed stitching mechanism from Sec. 4.1 on actual environments.
- Provide guidance for choosing \(m\) and discuss sensitivity more systematically.
- Report runtime / training-cost overhead relative to plain DT and RvS.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related work such as QDT or other recent methods.”**  
  Removed because external related-work completeness cannot be verified here, and the review instructions explicitly disallow criticizing missing related work.

- **Pure style/notation complaints as standalone weaknesses.**  
  The paper does have some notation instability in Sec. 4.2, but this is only relevant insofar as it reflects a substantive conceptual mismatch. As a pure presentation complaint, it is not retained.

- **Generic reproducibility complaints about omitted implementation details.**  
  The paper states that implementation and hyperparameters are in appendices, and generic demands for more low-level detail are not substantive enough to retain.

- **Claim that the paper says GCReinSL is inferior to TGDA on Antmaze-Medium.**  
  The actual table in Fig. 5 shows GCReinSL ties or exceeds TGDA on the listed Antmaze-Medium entries. The sentence in Sec. 5.3 appears to be an internal inconsistency / likely artifact, but not a real empirical weakness of the method itself.

## Novel Insights
The most important synthesis across the reviews is that the paper’s strongest aspect and weakest aspect are tightly coupled: it has a genuinely interesting *algorithmic* idea—use upper-tail fitting of a learned reachability score to make SL policies prefer stitchable continuations—but its *formal* story overclaims what is proven. In particular, the empirical results suggest the idea is more than cosmetic and does improve stitching-like behavior, especially for sequence models on D4RL Antmaze-v2. However, the paper’s theory currently does not convincingly establish that the learned CVAE score is the same object as the Q-function being optimized, nor that the expectile argument yields the local state-goal-conditional maximization the method relies on in practice. So the submission reads as a promising empirical/algorithmic advance with an overstated theoretical bridge, rather than a fully closed conceptual unification of SL and TD learning.

## Suggestions
- Tighten the theory around Sec. 4.2–4.3: either prove that the learned surrogate is the right object to maximize for the stated RL objective, or explicitly weaken the claim and present the method as a heuristic approximation with empirical support.
- Restate Theorem 4.1 in a form that matches the algorithm’s actual need: a **state-goal-conditional in-distribution maximum**, not a global maximum over all \((s,a,g)\).
- Soften the framing from “bridging the gap” to “narrowing the gap,” unless stronger evidence against TD baselines is added on the stitching-focused datasets.
- Add direct mechanism ablations: remove Q-conditioned inference while keeping the auxiliary loss; compare expectile vs MSE; compare the CVAE score against simpler alternatives.
- Add TD baselines on the Ghugare et al. datasets and report GCReinSL-RvS on D4RL Antmaze-v2 for a more complete empirical picture.
- Include qualitative rollouts or Q-value visualizations showing whether predicted Q indeed increases at stitching points as claimed in Sec. 4.1.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 1.0, 6.0]
Average score: 3.8
Binary outcome: Reject
