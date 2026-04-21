Now I have all the information I need. Let me write the consolidated review.

## Summary

This paper trains language models to win debates via self-play DPO and shows for the first time that stronger debaters produce more accurate judge evaluations (4% absolute improvement, p < 10^{-6}), while consultancy baselines show no positive skill-accuracy trend. Through novel baseline variants (ensembled and double consultancy) and mechanistic analysis (evidence use, judge transfer), the paper identifies that debate training incentivizes truth-seeking policies (more quotes, transfer to unseen judges) while consultancy incentivizes judge-specific exploits — but also finds that explicit refutation plays little role in the observed gains.

## Strengths

- **First demonstration that training models to debate improves judge accuracy**: Prior work (Radhakrishnan, 2023) failed to show this for trained models. The paper establishes a positive skill-accuracy trend with strong statistical significance (p < 10^{-6}), advancing the debate-as-scalable-oversight agenda beyond inference-time prompting results (Section 4.2, Figure 5).

- **Novel consultancy baselines that decompose why debate works**: The ensembled consultancy (72% accuracy) and double consultancy (75% accuracy) variants are new and provide genuine analytical leverage, isolating the contributions of two-sided presentation, side-by-side comparison, and adversarial refutation (Section 2.3, Section 4.3).

- **Compelling mechanistic evidence for divergent learning dynamics**: Debate models increase quoted evidence by 96% over training while consultancy models decrease it by 70%; the GPT-4o transfer experiment shows r = 0.98 for debate vs. r = 0.51 for consultancy in win-rate correlation across judges (Section 4.4, Figure 6). This directly demonstrates that debate training learns general argumentation while consultancy overfits to judge-specific exploits.

- **Honest reporting of the refutation null result**: The paper openly finds that single-turn debates (no refutation possible) are judged as accurately as two-turn debates, and double consultancy nearly matches debate accuracy (75% vs. 77%), contradicting the theoretical motivation from Irving et al. (2018). Section 5.1 discusses this frankly.

- **Modified DPO formulation and branching rollout procedure**: The DPO+ loss incorporating continuous reward signals and the branching rollout procedure for multi-turn preference data are practical contributions for self-play in multi-turn settings (Section 3.2.2).

## Weaknesses

### Fatal

None.

### Major

- **The paper's framing emphasizes "debate" (with refutation) but its own evidence shows refutation is not the operative mechanism.** The title claims "training to win debates" improves oversight, and the introduction frames debate through refutation (Section 1: "incentivizing the competing models to discover and explain the subtle flaws"). Yet the paper itself finds: (a) single-turn debates with no refutation are judged as accurately as two-turn debates (Appendix G, Section 4.3); (b) double consultancy (where debaters never see each other) achieves 75% accuracy vs. debate's 77% — a 2% gap with no significance test. The paper honestly reports these findings but does not recalibrate its headline claims accordingly. What the evidence supports is that *training with an adversary* prevents degenerate judge exploitation, not that the debate format's turn-taking refutation structure is critical. This tension between framing and findings is the paper's most substantial issue.

- **The skill-accuracy comparison uses incomparable win-rate metrics for debate vs. consultancy.** Section 2.4 defines debater win rate via Elo scores from round-robin tournaments (no natural ceiling near 50%), while consultant win rate is the raw probability the judge agrees in single consultancy (which caps near 50% for a truth-seeking model since it defends the wrong side half the time). The "Win Rate v Judge Accuracy" plot (Figure 5, right panel) overlays both on the same axes, visually exaggerating the slope difference. The training-epoch comparison (left panels) is the fairer measure and does support the claim — but the win-rate comparison is foregrounded without adequate caveats.

- **No independent judge evaluation is provided.** The same judge model (trained GPT-4T) defines both the training reward and the accuracy evaluation, creating a circularity: models are optimized to win under this specific judge, and then the paper measures how well *this same judge* performs. The GPT-4o transfer experiment (Section 4.4) partially addresses this by showing win-rate generalization, but it measures win-rate correlation only — not whether an independent judge would also be more *accurate* when evaluating transcripts from stronger debate models.

### Minor

- **Missing statistical characterization for consultancy's null skill-accuracy trend.** The paper reports p < 10^{-6} for debate's positive trend but provides no formal test (correlation coefficient, p-value, or confidence interval) for consultancy's null result. A credible null claim requires statistical backing, not just visual inspection (Section 4.2).

- **Asymmetric hyperparameter tuning across conditions.** The second DPO iteration for debate uses a lower learning rate (5 × 10^{-5}) found via sweep, while consultancy was swept but no improvement found (Section 3.2.2). Additionally, γ differs (7 for debate, 10 for consultancy). While the paper argues the sweep was conducted for both, the asymmetry in outcome means the conditions are not identically tuned, potentially understating consultancy's performance.

- **The double-consultancy training ablation is missing.** All consultancy evaluation variants use a model trained for single consultancy (Section 2.3). Training a model specifically for double consultancy (with opponent speeches visible at training time) would isolate whether the key ingredient is training-time exposure to opposing arguments or the debate format itself. This is a valuable ablation but its absence does not invalidate the current findings — it means the paper establishes the value of adversarial training more convincingly than the value of debate specifically (Section 4.3, 5.1).

### Trivial

None.

## Nice-to-Haves

- Test debate on tasks beyond reading comprehension to assess generality, especially reasoning-heavy tasks where evidence cannot be quoted. The paper acknowledges this limitation (Section 5.2).
- A full transcript comparison of debate vs. double consultancy on the same question would make behavioral differences concrete for readers.
- Analyze judge correctness conditioned on evidence asymmetry (does debate help more when one side has much stronger textual support?).

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic's claim that the debate vs. double consultancy gap being "small and potentially insignificant" is a fatal flaw**: The paper itself identifies this gap as meaningful (2% points) and the *trend* difference is the key finding — double consultancy shows no positive skill-accuracy trend while debate does. The absolute gap at one point is less important than the diverging trends. However, the lack of significance testing on the 2% gap is a valid minor concern.

- **Harsh Critic's claim that "the paper conflates two-sided training and debate format"**: The paper actually disentangles these quite carefully through its three consultancy baselines, and in Section 5.1 explicitly lists factor 3 as "the presence of two different sides in one context at training time." The paper is aware of and discusses this distinction. The missing double-consultancy training ablation is a genuine (minor) gap but not a conflation.

- **Strength Finder's claim about "judge calibration and sycophancy mitigation" as a core strength**: While the judge training is careful and competent, it's a means to an end rather than a core contribution of the paper. It supports the validity of the experiments but is not itself a novel method.

## Novel Insights

The paper produces a genuinely surprising and important negative finding: refutation — the core mechanism proposed by Irving et al. (2018) for why debate should improve oversight — does not appear to be the operative mechanism at all. Instead, the evidence points to adversarial training as the key ingredient: having an opponent during training prevents models from learning judge-specific exploits, and forces them to use more evidence. This reframes "debate for scalable oversight" from "refutation surfaces flaws" to "adversarial training prevents degenerate persuasion." The consultancy-overfitting evidence (repetitive quoting, poor judge transfer) provides the clearest mechanistic account yet of *why* debate training works differently, and the divergence between debate and consultancy becomes more informative than their absolute accuracy gap.

## Suggestions

- Narrow the title and framing to reflect what the evidence supports: adversarial training (not specifically the debate format with refutation) is the key mechanism. A title like "Adversarial Self-Play Training Improves Judge Accuracy in Debate Settings" would better match the findings.
- Report formal statistical tests for the consultancy null trend (correlation, p-value, confidence interval) to make the negative result rigorous.
- Add one evaluation using an independent judge (even GPT-4o) that measures *accuracy*, not just win rate, to address the judge-circularity concern.

## Evaluation

**Originality**: The paper makes a genuine contribution as the first to show training models to debate improves judge accuracy. The consultancy baselines and mechanistic analysis are novel. The refutation null result is unexpected and valuable. However, the framing overclaims relative to what the evidence establishes.

**Importance**: The research question — whether debate training enables scalable oversight — is important for AI alignment. The finding that adversarial training prevents judge exploitation is significant even if the debate format itself is not the key ingredient.

**Claims support**: The core claim that debate training improves judge accuracy is well-supported. The implied claim that the debate *format* (with refutation) is important is contradicted by the paper's own evidence. The mechanistic claims about why debate helps (evidence use, generalization) are well-supported by the transfer and evidence-use analyses.

**Soundness**: The experimental design is thoughtful, particularly the graduated consultancy baselines. The main gaps are the shared train/eval judge, the incomparable win-rate metrics, and the missing double-consultancy training ablation.

**Clarity**: The paper is well-written and transparent about its findings, including the negative result on refutation. The structure is logical and the figures are informative.

**Community value**: The paper advances the debate-for-oversight research program substantially and provides concrete evidence for and against specific mechanisms. Even the negative result on refutation is a valuable contribution.

## Calibration Anchor Comparison

- **PdaPky8MUn** (avg 8.0, Accept Oral): "Never Train from Scratch" — strong experimental evidence that recontextualizes prior work, with clear and impactful findings. Our paper has a similar "recontextualizing" quality (refutation doesn't matter) but with weaker experimental controls (shared judge, missing ablation) and overclaimed framing. Below this anchor.

- **5BjQOUXq7i** (avg 7.2, Accept Spotlight): RegMix — novel method with extensive experiments, some overclaimed scope. Our paper also has novel methods with some overclaimed scope but fewer experiments and a narrower domain. Somewhat below this anchor.

- **Pujt3ADZgI** (avg 6.0, Accept Oral): INPO — self-play DPO for alignment, game-theoretic framing, some presentation issues and missing confidence intervals. Our paper shares the self-play DPO and game-theoretic elements but adds stronger mechanistic analysis. Around this anchor or slightly above.

- **ChNy95ovpF** (avg 4.33, Reject): DebateGPT — multi-agent debate for alignment but weak novelty and missing baselines. Our paper is substantially stronger than this with proper baselines and mechanistic analysis. Well above this anchor.

- **BrtOzgElD7** (avg 2.5, Reject): Red Teaming Game — game-theoretic framework for alignment but poorly defined, no formal results. Our paper is far stronger. Well above this anchor.

- **QAwaaLJNCk** (avg 6.0, Reject): Multi-agent debate for factuality — straightforward idea, limited novelty, unclear mechanism. Our paper is stronger with better analysis and a more important research question. Above this anchor.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>