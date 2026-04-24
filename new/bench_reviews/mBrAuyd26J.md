## Summary

This paper proposes a dual-system architecture for social-deduction agents in which an LLM handles domain-agnostic System-1 tasks (listening and speaking) while a small external *Thinker* module performs domain-specific System-2 reasoning via RL and imitation learning. The authors demonstrate the framework in a 9-player Werewolf game, contributing a large dataset of 18,800 human sessions (*FanLang-9*), and report gains in deductive reasoning, speech quality, and online win rates. A 6B parameter fine-tuned LLM paired with the Thinker is shown to approach or exceed GPT-4 performance within the same opponent pool.

## Strengths

- **Well-motivated modular architecture with an interpretable communication protocol.** The paper explicitly separates domain-agnostic NLP (Listener/Presenter) from domain-specific reasoning (Thinker) and designs a structured JSON language-feature / speech-instruction interface (Eq. 1, Figure 2). This is a concrete, human-readable alternative to opaque hidden-state fusion and allows the reasoning module to be trained without fine-tuning the LLM backbone.
- **Consistent empirical improvements across three complementary axes.** The Thinker improves GPT-3.5 and GPT-4 in deductive reasoning (Figure 3), reduces illegal speech rates in human preference tests (Figure 4), and raises online win rates (e.g., GPT-3.5-T reaches 47.4% vs. 36.7% for GPT-3.5-LIM in the first pool of Table 1). The multi-faceted evaluation is more thorough than typical LLM-agent papers.
- **Practical efficiency demonstration.** WereLLM-T (6B) achieves a 50.3% total win rate versus 41.1% for GPT-4-T in the same opponent pool (Table 1, first block), and matches or exceeds GPT-4-LiM in human preference rankings (Figure 4). This establishes that the framework can deliver strong performance without proprietary large models.
- **Large-scale domain dataset.** *FanLang-9* (18,800 sessions, ~7,000 hours) is a concrete, valuable resource for social-deduction AI research, and the paper includes ASR fine-tuning details (Section 3.1).

## Weaknesses

### Fatal
None.

### Major
- **Training–deployment mismatch in the Thinker’s RL phase.** Section 3.3 explicitly states that the Thinker is trained under the assumption that “the Presenter generates speech accurately … and the Listener … generate[s] a language feature that precisely matches the original speech instruction,” enabling independent optimization. Yet Sections 3.2 and 3.4 acknowledge that the Listener struggles with “information overload” and “colloquial ramblings,” while the Presenter hallucinates and requires a post-hoc filter plus a template-based fallback. The Thinker’s PPO objective (Equation 4) is therefore optimized for a state distribution that differs from deployment. While the filter and fallback partially mitigate this leak, the paper provides no ablation quantifying how often the fallback triggers or how much the online performance depends on the intended policy versus the backup. This is a significant methodological gap that weakens the rigor of the RL training claims.
- **Online evaluation conflates agent strength with opponent-pool composition.** Table 1 reports win rates across three distinct agent populations (Section 4.3). Because Werewolf is adversarial and highly interactive, win rates depend on the strength of the other eight seats. WereLLM-T varies from 50.3% (GPT-3.5-heavy pool) to 43.1% (GPT-4-heavy pool), while GPT-4-T moves from 41.1% to 46.3% across pools. The paper draws cross-pool conclusions (e.g., “The performance of the WereLLM-T model closely aligns with that of GPT-4-T”) without controlling for opponent strength, using Elo ratings, or running a uniform round-robin. Within-pool comparisons are valid, but the broader claims are confounded.

### Minor
- **Absence of statistical testing.** The abstract and results repeatedly use the word “significantly,” yet no standard errors, confidence intervals, or hypothesis tests appear in Figures 3–4 or Table 1. With roughly 600 games per combination and nine players per game, per-role sample sizes are small enough that win-rate swings of several percentage points could reflect sampling noise. Without statistical testing, the strong causal language is unsupported.
- **Inconsistent human baseline in Figure 3.** Section 4.1 states that “their [human] judgments regarding the identities of other players remain unknown,” yet Figure 3 displays human accuracy bars for identifying Seer, Witch, and Hunter. The paper never explains how these bars are computed, making the human deductive-reasoning baseline impossible to assess.
- **No inter-annotator agreement for speech evaluation.** Ten human evaluators rank speeches in Section 4.2, but inter-annotator agreement (e.g., Krippendorff’s alpha or Fleiss’ kappa) is not reported. Without it, the preference scores in Figure 4 are weaker evidence of systematic quality differences.

### Trivial
None.

## Nice-to-Haves
- Joint-training or noise-injection ablation to bound the cost of the Thinker’s independence assumption.
- A single controlled round-robin tournament or Elo ratings to disentangle agent strength from opponent composition.
- Failure-mode case studies showing concrete game traces where Listener mis-parsing or Presenter hallucination causes catastrophic Thinker decisions.
- Reporting inter-annotator agreement and confidence intervals for all quantitative claims.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **“The need for a filter and template fallback is an implicit admission that the Presenter does not satisfy the perfect-generation assumption.”** The paper explicitly describes the filter and fallback in Section 3.4 as intentional design choices, not hidden admissions. This is a misreading.
- **“Not yet released” / availability concerns about FanLang-9 or models.** The paper cites and links to its dataset (footnote 1); per the review policy, cited entities are assumed to exist.
- **Missing appendix, proofs, or references.** The parser strips appendices from all papers; they exist in the original submission.
- **Demand for end-to-end fine-tuned baseline.** While such a baseline would strengthen the paper, the absence does not invalidate the modular design claims, and the paper already compares against strong LLM prompting baselines.
- **Formatting nitpicks about Table 1 parser artifacts.** These are parser errors, not author errors.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add statistical testing (permutation tests or confidence intervals) for all win-rate and accuracy comparisons, or soften the language if the effects do not survive testing.
- Either explain how the human identity-identification bars in Figure 3 are derived, or remove them if ground-truth beliefs are truly unavailable.
- Quantify how often the Presenter’s self-consistency filter rejects generated speech and how often the template fallback is triggered in online games; this directly addresses the training–deployment gap.

## Score and Decision

**Calibration reasoning:**
- *High anchor:* DDCFR (avg 8.00, Accept Spotlight) — clean theory, rigorous experiments, strong ablations. The current paper does not reach this level of methodological tightness.
- *Medium anchor:* A Generalist Hanabi Agent (avg 5.50, Accept Poster) — interesting architecture with real empirical results, but evaluation protocol issues and missing ablations. The Werewolf paper is comparable: it has a larger dataset and multi-axis evaluation, yet suffers from a more severe training–deployment mismatch and opponent-pool confounding.
- *Medium-low anchor:* Retroformer (avg 5.67, Accept Spotlight) — modular LLM+RL design with mixed reviewer reception; limited experimental comparisons but accepted. The Werewolf paper has broader evaluation but also deeper methodological concerns.
- *Low anchor:* LLMs Are In-Context RL (avg 3.75, Reject) — fundamentally flawed motivation and failure to beat simple baselines. The Werewolf paper is substantially stronger, with consistent gains and a real dataset contribution.

Relative to these anchors, the paper sits between the medium and medium-low clusters. Its core contributions (dataset, architecture, empirical validation) are genuine, but the training–deployment mismatch and uncontrolled online tournament are significant methodological liabilities that would need to be addressed in a revision. These issues do not fully invalidate the empirical improvements shown, so the paper is not in the low-scoring band, but they prevent a confident accept. I therefore assign a borderline score.

**Score:** 5.5  
**Decision:** Accept

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>