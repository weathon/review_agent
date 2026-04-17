Now let me verify some specific claims from the paper to check the reviewers' criticisms.

## Summary

This paper investigates why modern recurrent models (SSMs like Mamba, Hyena) underperform Transformers on associative recall and copying tasks, arguing that the gap stems primarily from optimization instability rather than expressivity limitations. Through extensive experiments (~3,000 runs, ~20K GPU hours), the authors demonstrate that SSMs succeed only within an extremely narrow learning rate window, while Transformers are robust across a wide range. They also show that SSMs prefer width over depth scaling, that 1-layer SSMs (with convolutions) can solve MQAR where 1-layer Transformers fail, and that newer architectures like DeltaNet improve stability.

## Strengths

1. **Timely and important empirical finding on LR instability.** The demonstration that prior MQAR evaluations (Arora et al., 2023) used learning rate grids that missed the viable range for SSMs (Figure 1) is a concrete, impactful observation. The narrow LR window for Mamba/Hyena vs. the wide window for Transformers is clearly presented and robust across tasks (MQAR and copying).

2. **Extensive experimental scope.** Over 3,000 runs and ~20K GPU hours span multiple architectures (Transformer, Mamba, Mamba2, Hyena, DeltaNet), multiple tasks (MQAR, copying), and systematic LR grids, lending strong statistical credibility to the core findings.

3. **Clean and insightful ablation studies.** The convolution ablation (Table 2) is a mechanistically clean result: adding conv to a 1-layer Transformer yields 99% accuracy, removing conv from 1-layer Mamba drops it to 2%. This directly identifies the architectural component responsible for single-layer expressivity differences.

4. **Useful scaling insight.** The finding that SSMs benefit from width while Transformers benefit from depth (Figure 4, Table 1) has practical implications for how fair architectural comparisons should be conducted.

5. **Constructive direction.** Demonstrating that DeltaNet achieves Transformer-level LR robustness on MQAR (Figure 7), and hypothesizing why (Householder matrices avoiding vanishing gradients in off-diagonal terms), provides an actionable direction for future SSM design.

## Weaknesses

### Major

- **Overstated central claim about expressivity vs. learnability.** The paper's headline narrative ("*Transformers differ from SSMs not in terms of expressive power but mainly because of their optimization dynamics*") goes beyond what the experiments support. The experiments compellingly show that (a) LR tuning is a major confounder in prior SSM evaluations, and (b) SSMs are much more brittle to LR choice. They do NOT establish that expressivity differences are negligible. The authors themselves acknowledge that Hyena "requires the model dimension to exceed the sequence length" for 1-layer success (Section 6), deeper Mamba fails at copying despite tuning (Table 1, 24-layer Mamba at 16%), and 1-layer Transformers fail MQAR regardless of width. These expressivity limitations remain real. The paper's contribution is best characterized as: "optimization brittleness is a major, underappreciated confounder in SSM vs. Transformer comparisons—not that expressivity differences are secondary."

- **No validation beyond synthetic benchmarks.** All experiments use MQAR and copying tasks. While these tasks are established proxies for language modeling abilities, the authors acknowledge that "validating these dynamics on downstream language modeling tasks is a critical next step." The narrow LR window finding could be amplified, attenuated, or qualitatively different under real language modeling loss landscapes—which involve diverse data distributions, regularization, and longer training. This limits the paper's ability to generalize its "fundamental learnability" claims to practical settings.

- **Only learning rate is deeply investigated as an optimization variable.** The paper attributes instability to LR sensitivity but fixes optimizer (Adam), weight decay, beta values, warmup, and gradient clipping. It is plausible that different optimizer configurations could widen or narrow the stable LR window, changing whether this is characterized as "fundamental learnability" vs. "needs a better training recipe."

### Minor

- **Induction head analogy is under-evidenced.** The paper states that 1-layer Transformers exhibit "a loss drop reminiscent of induction head formation" (Section 6). While appropriately hedged with "reminiscent of" and "we hypothesize," the connection to induction heads is speculative without any mechanistic analysis (attention pattern inspection, causal intervention, etc.). Identifying a non-monotonic loss curve does not establish circuit formation.

- **Scaling comparison is partially one-sided.** The argument that SSMs should be scaled in width while Transformers in depth is shown for SSMs (Table 1: wide Mamba succeeds where deep Mamba fails), but there is no corresponding exploration of whether wide shallow Transformers also perform well on these tasks. This leaves open whether the "preferred scaling axis" observation is really architecture-specific or partially task-driven.

- **Small model scales.** All architectures are tested at small scales (hidden dims 64–2048). Whether the width-over-depth preference for SSMs persists at larger scales—where depth becomes increasingly important for all architectures—remains unknown.

## Nice-to-Haves

- Validate LR brittleness on at least one small-scale language modeling benchmark (e.g., WikiText-103 or TinyStories) to test generalizability beyond synthetic tasks.
- Investigate other optimization variables (gradient clipping, warmup, optimizer choice) to determine whether the narrow LR window is architecture-inherent or optimizer-dependent.
- Provide Hessian spectral density or gradient norm statistics at successful vs. failing LRs to substantiate the "loss landscape" framing beyond outcome-level observations.
- Perform mechanistic analysis (attention head inspection, causal ablation) to test the induction head hypothesis in 1-layer models.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Demand to test "unreleased" or newer SSM variants (RWKV, RetNet, Griffin).* The paper already tests five architectures including DeltaNet; requesting more models is a generic scope expansion.

- *Citation of the Mimetic Initialization paper as "similar prior work" for weaknesses.* The "narrow LR window has limited novelty given known RNN training difficulties" argument misunderstands the contribution. Prior work documented vanishing/exploding gradients in classical RNNs; this paper specifically quantifies the *magnitude and character* of LR brittleness in *modern* SSMs relative to Transformers, which is a distinct empirical finding. The relevant comparison is not "RNNs are hard to train (known)" but "the degree and narrowness of this instability in modern SSMs vs. Transformers on the same task (newly quantified)."

- *Fairness complaint that architectures differ in more than just the sequence mixer.* The paper explicitly addresses this with ablations (Table 2), showing that removing Mamba's conv and gating reduces it to Transformer-like performance. The comparison is not "apples-to-oranges" but rather "apples-to-apples, with careful component attribution."

- *Formatting and writing quality nitpicks.* The paper is generally well-written and clearly structured.

- *Request for theoretical proof of the LR instability.* This is an empirical systems paper analyzing training dynamics; demanding formal proofs of landscape properties is outside the community's standard expectations for such contributions. The paper itself identifies this as future work.

## Novel Insights

The most novel observation is the asymmetry in single-layer dynamics: 1-layer Transformers exhibit a loss bump resembling induction head formation yet fail to solve MQAR, while 1-layer Mamba shows a similar bump and succeeds. This suggests both architectures encounter a similar optimization landscape feature, but SSMs can leverage it in a single layer while Transformers cannot—pointing to a genuine expressivity difference that manifests specifically in shallow settings, rather than a blanket expressivity gap.

## Suggestions

- Revise the central claim from "not expressivity but mainly optimization" to "optimization instability is a major confounder in SSM evaluations, and expressivity differences are more nuanced than previously assumed." The data support the latter but not the former.
- Add at least one pilot experiment on a small-scale language modeling setup to test whether LR brittleness transfers beyond synthetic tasks, even at modest scale.
- Document that each configuration in scaling comparisons (Table 1) received the same level of LR tuning, ruling out depth-failure-as-untuning.
- Soften the induction head language from "reminiscent of induction head formation" to "exhibits a non-monotonic training dynamics whose cause deserves further investigation."

## Score and Decision

**Calibration comparison:**
- *Zoology* (Arora et al., MQAR paper): scores 3/8/8, accepted poster — had both theory and real-data validation, but also received criticism for limited scale.
- *RNNs are not Transformers*: scores 3/5/6/8, accepted poster — theoretical contribution on expressivity with limited empirical scope.
- *Mimetic Initialization* (SSM recall, synthetic-only): scores 3/5/5/5, rejected — similar scope to current paper but less extensive experiments and less novelty.
- *Small-scale proxies for Transformer instabilities*: scores 8/8/8/8, oral — focused on training instability, very thorough experiments, validated across scales.
- *Unlocking State-Tracking in LRNNs*: scores 8/8/8, oral — strong theory + empirical validation including real LM.

This paper sits between the Mimetic paper (rejected, 3-5 range) and the Zoology/RNNs papers (accepted, 5-8 range). It has more extensive experiments than Mimetic and a more novel finding (LR brittleness as confounder), but overclaims significantly and lacks real-task validation. The core empirical finding is important and well-documented, but the framing as "not expressivity but mainly optimization" is not supported.

Score: **5.5** — The paper makes a valuable empirical contribution (LR brittleness as a confounder in SSM evaluations) but is held back by overclaiming and lack of validation beyond synthetic benchmarks. Tightening the claims and adding even a small real-task experiment would substantially strengthen it.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>