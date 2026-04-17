Now I have all the information I need. Let me synthesize the final review.

## Summary
This paper provides the first systematic study of benchmark contamination detection in Large Reasoning Models (LRMs), identifying two critical vulnerability scenarios: (1) Stage I (pre-LRM), where RL training—specifically GRPO with PPO-style importance sampling and clipping—can conceal contamination signals initially introduced during SFT; and (2) Stage II (post-LRM), where SFT contamination with chain-of-thought on advanced LRMs causes existing detection methods to perform near random. The paper combines empirical evaluation of 10 detection methods across 6 benchmarks with theoretical analysis attributing concealment to PPO-style clipping objectives.

## Strengths
- **Timely and important problem.** The vulnerability of LRM evaluation to contamination is a growing practical concern as reasoning models dominate leaderboards. Identifying two distinct contamination pathways (pre-LRM and post-LRM) is conceptually clear and practically relevant.
- **Strong empirical coverage.** Testing 10 detection methods across 6 benchmarks with 2 base models (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) and 4 LRM models in Stage II provides substantial breadth. The consistent trends across methods and benchmarks strengthen the core finding that existing detectors are fragile in LRM contexts.
- **Valuable Stage I observation.** The demonstration that GRPO significantly degrades membership-inference AUROC while preserving benchmark performance inflation (Table 1-2, Fig. 2-3) is novel and important. Even without a complete mechanistic explanation, this empirical phenomenon deserves attention.
- **Clean ablation design.** The RAFT vs RAFT++ vs GRPO with/without clipping comparison (Table 3) is a well-designed experiment that provides strong evidence linking PPO-style clipping to concealment. The fact that removing clipping largely restores detection performance is a compelling result.
- **Important conceptual insight in Stage II.** The observation that LRMs generalize confidence to distributionally similar non-members (Fig. 4), undermining the memorization assumption behind most detectors, challenges fundamental assumptions in the field and has implications beyond this specific study.
- **Rules out forgetting as explanation.** The paper carefully demonstrates that GRPO preserves performance inflation while dropping detection signals (Table 1-2, Fig. 2 + Tab. 23), and that contaminated GRPO also degrades AUROC, ruling out the simple explanation that models merely forget contamination.

## Weaknesses

### Major:
- **Overclaimed generality of "PPO-style clipping as root cause."** The main headline claim—that "PPO-style importance sampling and clipping objectives are the root cause of this detection concealment" and that "a broad class of RL methods may inherently exhibit similar concealment capability"—overreaches the evidence. The theoretical analysis (Theorem 3.1) relies on strong simplifying assumptions (tabular setting, single natural gradient step, specific advantage formulation without standard deviation, informal sign arguments for covariance terms) that are not empirically validated. The key ablation (Table 3) uses only one detection method (Loss) on one base model, and the no-clipping variants may differ in training dynamics beyond just clipping. Alternative explanations—such as any training that reduces output entropy and makes member/non-member log-prob distributions converge—are not fully ruled out. A more measured claim ("with our implementation, PPO-style clipping is a major contributor to concealment, with a plausible theoretical mechanism") would be appropriate.

- **Stage II "near random guess" claim is overstated.** Table 5 shows many AUROCs in the 55–65% range, including LiRA reaching ~65% on DS-Qwen-14B for some benchmarks. While these are weak, they are not "near random" in a binary classification sense. The sweeping conclusion that contamination "barely leaves evidence" and that memorization-centric detection assumptions are "outdated" goes beyond what one contamination scheme (extensive SFT-only on members with CoT) can demonstrate. The paper also does not explore different contamination intensities, mixed contamination (interleaving with clean data), or contamination without explicit CoT, limiting the generality of Stage II conclusions.

- **No PPO validation despite "PPO-style" framing.** The theoretical argument centers on PPO-style objectives, yet PPO itself is never empirically tested. Only GRPO, RAFT, and RAFT++ are evaluated. While these share the clipping/importance-sampling mechanism, direct validation with PPO would significantly strengthen the "broad class of RL methods" claim.

### Minor:
- **Detection protocol choices may bias against detectors.** The evaluation averages detection scores over 8 rollouts and only uses generated (not training) sequences. This is a reasonable choice given the realistic access constraints for LRMs, but it means the paper evaluates detectors under a specific protocol. Some detectors (e.g., diversity-based ones) may lose signal from averaging, and the paper's conclusions should be scoped accordingly.

- **All benchmarks are mathematical reasoning tasks.** The six benchmarks (Olympiad, GPQA, AIME25, AIME24, Minerva, AMC23) are all math/reasoning. Whether concealment dynamics hold for code generation, scientific QA, or other domains remains unknown, limiting the scope of general claims about LRM contamination.

- **Theory only covers correct trajectories (r=1).** The NLL gap analysis focuses exclusively on correct trajectories, while real contamination involves both correct and incorrect rollouts. The paper does not discuss how this simplification affects predictive power or whether concealment operates differently on incorrect samples.

- **No contamination intensity ablation.** Using a fixed 50/50 member/non-member split makes it unclear whether concealment is equally effective with smaller contamination ratios (e.g., 10–20% of benchmark data), which would be more realistic for deliberate contamination.

### Trivial:
- The paper occasionally uses informal language for rigorous claims (e.g., "alarmingly easy," "definitely inadequate").

## Nice-to-Haves
- Testing additional RL algorithms (PPO, DPO, online DPO) to validate the "broad class" claim.
- A contamination intensity ablation (varying the fraction of contaminated data from 10% to 50%).
- Quantitative distributional distance metrics (KL, Wasserstein) to supplement visual log-prob plots in Fig. 3-4.
- A preliminary detection approach that leverages the paper's own insights (e.g., comparing detection at pre-RL vs post-RL checkpoints, or using consistency of reasoning patterns rather than log-prob gaps).
- Testing on at least one non-math benchmark (e.g., code) to assess domain generality.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Harsh critic's point about limited model scale (7B/8B).** The paper does test DeepSeek-R1-Distill-Qwen-14B, and model scale is a reasonable soft concern but not a fatal flaw. More importantly, the reviewer's claim about "leaderboard-grade models being much larger" ignores that many leaderboard models ARE in the 7B-14B range (this is a rapidly evolving landscape). This is downgraded to a Nice-to-Have.
- **Harsh critic's claim that "further SFT vs GRPO" comparison is not controlled for effective step size.** The paper explicitly tests additional SFT (Fig. 2, Tab. 23) and shows it does NOT conceal contamination, which is a relevant comparison even if not perfectly matched on all training dynamics.
- **Spark's point about "no performance-based detection baseline."** This misunderstands the paper's scope—the paper is about membership inference / contamination detection, not about whether contamination yields performance gains. Performance inflation (Tab. 1) is already shown and is a separate, well-studied question.
- **Spark's point about PPO not being tested.** This is valid and kept as a major weakness above, but the characterization "the namesake algorithm" is slightly misleading—GRPO IS the dominant RL algorithm for LRMs in practice, so testing GRPO is more relevant than testing PPO.
- **Neutral reviewer's point about AUROC as sole metric.** AUROC is the standard metric in membership inference and contamination detection literature. Requesting additional metrics is a Nice-to-Have, not a core flaw.
- **Neutral reviewer's point about "no concrete remediation."** The paper proposes two concrete directions (release intermediate checkpoints, move beyond memorization-driven detection). Proposing a new detection method is outside this paper's stated scope (studying the fragility of existing methods).
- **Harsh critic's extensive "Section-by-Section" notes** are largely subsumed by the synthesized weaknesses above or are formatting/style issues.
- **Human finder's point about "lack of convergence guarantees" for theory, citing SAM memorization paper.** This is from a different paper's reviews and is not directly relevant to the theoretical framework here, which is a first-order decomposition rather than an optimization convergence claim.

## Novel Insights
The most significant insight of this paper is the distinction between "forgetting" and "concealment" in the context of RL training: GRPO doesn't make models forget contaminated data (performance inflation persists), but rather makes membership signals statistically indistinguishable—a finding that challenges the common intuition that additional training simply dilutes memorization. The Stage II finding that CoT-contaminated LRMs generalize confidence to unseen distributionally similar samples is also novel and suggests a fundamental limit of memorization-based detection for reasoning models, though the claim is currently stronger than the single-experiment evidence warrants.

## Suggestions
- Soften "root cause" and "broad class of RL methods" to "major contributor" and "several RL methods with PPO-style objectives," or add direct PPO experiments to strengthen the generality claim.
- Add quantitative distributional distance metrics (e.g., KL divergence, Wasserstein) between member and non-member log-prob distributions across training steps to ground the "gap contraction" argument beyond visual inspection.
- Tone down "near random guess" language to reflect actual AUROC values (~55–65%), which are weak but not chance-level.

## Evaluation

**Originality:** High. This is the first systematic study of contamination detection in the LRM context, and the finding that RL training conceals rather than merely dilutes contamination is novel.

**Importance of research question:** Very high. Contamination in reasoning model evaluations directly threatens the integrity of public leaderboards at a time when commercial incentives for gaming benchmarks are strong.

**Claims support:** Moderate. The core empirical findings are well-supported, but the causal mechanism (PPO-style clipping as the root cause) and the generality of Stage II claims are overstated relative to the evidence.

**Soundness of experiments:** Good overall. Comprehensive coverage of detection methods and benchmarks, but with some protocol choices (8-rollout averaging, one contamination scheme per stage) that limit generalizability.

**Clarity:** Good. The two-stage framework is clearly presented and the paper is well-organized.

**Value to community:** High. This paper will likely spur significant follow-up work on LRM-specific contamination detection and evaluation integrity.

## Score and Decision

Calibration comparison:
- **"Evading Data Contamination Detection"** (similar topic, weak execution, no empirical depth): scores 3/5/6/3 → ~4.25, rejected. This paper is substantially stronger—it has systematic evaluation, theoretical analysis, and meaningful experimental design.
- **"Proving Test Set Contamination"** (novel method with formal guarantees): scores 6/8/8/8 → ~7.5, accepted oral. This paper is weaker in theoretical rigor and novelty of method but comparable in importance of problem.
- **"Time Travel in LLMs"** (good empirical contamination study): scores 6/8/8/6 → ~7, accepted spotlight. This paper provides more mechanistic insight but with less rigorous theory.
- **"Measuring memorization in RLHF"** (related topic, empirical): scores 5/6/8/6 → ~6.25, accepted poster. Similar level of empirical contribution with theoretical gaps.

This paper identifies an important vulnerability and provides substantial empirical evidence, but the overclaiming of theoretical results and the limited scope of Stage II experiments prevent a higher score. It falls between the "Evading" paper (~4.25) and the stronger contamination detection papers (6-7 range). The core empirical findings are solid and important; the main detractions are overclaiming and some limitations in experimental scope.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>