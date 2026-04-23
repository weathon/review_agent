Now let me run calibration searches to anchor my score.Now I have enough information to write the comprehensive final review.

---

## Summary

DELTA introduces an inference-time contrastive decoding method that reduces hallucinations in LLMs by randomly masking a portion of input tokens, running a second forward pass on the masked input, and subtracting the resulting (hallucination-amplified) logits from the original logits. The method is evaluated on SQuAD v1.1, SQuAD v2, TriviaQA, Natural Questions, CommonsenseQA, and MMLU using a single quantized Llama 3.1 8B model. The main empirical claim is a 14.53 pp improvement on SQuAD v2's no-answer exact match.

---

## Strengths

- **Inference-time-only with no retraining**: The method requires only a single additional forward pass per generation step (Eq. 3/5), requiring zero model modifications, making it immediately deployable. This is a genuine practical advantage over fine-tuning approaches.

- **Consistent improvement on context-rich QA benchmarks**: Table 1 shows Delta outperforms the baseline on SQuAD v1.1 (~3–4 pp EM), SQuAD v2 overall (~6 pp EM), and TriviaQA/NQ under sampling, demonstrating a consistent trend on the method's target domain.

- **Transparent reporting of failure cases**: Table 2 honestly reports small accuracy declines on CommonsenseQA (−0.25) and MMLU (−0.29), and Section 5.3 explicitly explains why the method is not suited to knowledge-only tasks. This transparency is commendable.

- **Clear mechanistic motivation**: The banana example (Figure 1, Section 3.2) effectively illustrates why masking amplifies prior-driven predictions and why subtracting such logits can recover context-grounded answers.

---

## Weaknesses

### Fatal
None that fully invalidate the core idea.

### Major

- **No comparison to Context-Aware Decoding (CAD) — the most functionally identical baseline**: Section 2 explicitly acknowledges CAD (Shi et al., 2024) as producing "a similar outcome to our Delta method" and describes it as "amplifying the differences between outputs generated with and without the given context." CAD and Delta share the same contrastive decoding formula: both amplify the distributional gap between a full-context forward pass and a degraded-context one (CAD strips context entirely; Delta masks 70% of tokens). Despite this near-identity, CAD appears nowhere in the experimental tables. The paper's dismissal — that CAD is "less generalizable" — is undercut by Delta's own Table 2 results showing it also fails on context-free tasks. Without a direct numerical comparison to CAD, the paper cannot establish that it offers any advancement over existing work.

- **No comparison to DoLa (Chuang et al., 2024) — the most direct competing inference-time method**: DoLa is cited in the introduction as related inference-time contrastive decoding work targeting factuality. It uses no retraining and applies to the same benchmarks (TriviaQA, NQ). The paper contains no results for DoLa across any table, making it impossible to determine whether Delta improves upon, matches, or underperforms this direct competitor. For a paper claiming a new state-of-the-art inference-time hallucination reduction approach, this is a structural omission.

- **Single model evaluation severely limits generalizability claims**: All results derive from one quantized model — Llama 3.1 8B Instruct at 4-bit. The masking mechanism interacts with a model's tokenizer, pre-training distribution, and instruction-tuning specifically. The paper claims Delta "presents a computationally efficient and scalable solution for reducing hallucinations in real-world LLM applications," but this cannot be supported from one model. Different models have different EOS semantics, tokenization granularity, and instruction priors that would affect masking behavior in unpredictable ways.

- **HasAns_EM decline confounds the headline NoAns_EM result**: The largest reported improvement is 14.53 pp on SQuAD v2 NoAns_EM (w/o sampling). However, the companion HasAns_EM simultaneously drops from 59.07 to 57.47 in the same setting (Table 1). This is the classic signature of a threshold shift — the method biases the model toward abstaining regardless of whether the question is answerable. The paper presents the NoAns_EM gain without acknowledging the HasAns_EM loss, and reports no joint metric (e.g., a calibrated abstention quality measure). With sampling, HasAns_EM does remain stable (58.21 → 58.62), which partially mitigates the concern, but the w/o-sampling headline figure is the number most prominently cited and it is confounded.

### Minor

- **EOS token as MASK token at 70% density is unexplained and potentially problematic**: Section 4.2 states "All experiments utilize the end-of-sequence (eos) token as the MASK token." At r_mask = 0.7, this means 70% of input tokens become EOS — a token that signals sequence termination in instruction-tuned Llama 3.1. This could generate pathological attention patterns not observed during training, producing distributional artifacts rather than controlled hallucination amplification. No ablation compares EOS against alternative mask tokens (e.g., padding, a neutral rare token, or BERT-style [MASK]), which would be essential to validate the implementation.

- **No variance estimates over masking seeds**: Since masking is stochastic (random token selection), every number in Table 1 is from a single run with a single set of random masks. For small gains such as TriviaQA w/ sampling (35.38 → 43.23) and NQ w/ sampling (9.25 → 11.80), where the improvement is 2–8 pp, the uncertainty from seed variation could be non-trivial. No error bars or multi-seed results are reported.

- **Ablation scope limited to SQuAD v1.1 only**: Figure 2 sweeps masking ratios and α only on SQuAD v1.1. The hyperparameters r_mask = 0.7, α = 0.3 are applied to all datasets without validation that these settings generalize. Given that TriviaQA and NQ have very different answer distributions, it is unclear whether the same parameters are optimal there.

- **Abstract/Section 2 generalization overclaim vs. actual scope**: Section 2 states Delta "could apply to all textual inputs," but the paper's own results and Section 7 explicitly acknowledge failure on context-free tasks. The claim should be scoped to context-driven scenarios from the outset.

### Trivial

- The ablation's "robustness" claim (std dev 0.66 EM) is presented positively but could equally indicate insensitivity — that gains do not depend on the hyperparameter because the mechanism's effect is not specifically tied to masking configuration. The paper acknowledges this only implicitly.

---

## Nice-to-Haves

- Evaluation on dedicated hallucination benchmarks (e.g., TruthfulQA, HaluEval, or FActScore) would more directly test the paper's claim of hallucination mitigation rather than relying on downstream QA EM/F1 as a proxy.
- Qualitative analysis showing actual before/after Delta generations would strengthen mechanistic claims.
- Targeted/structured masking (as suggested in the Future Work section) would make for a compelling ablation even in this version.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"No engagement with self-consistency, RAG, or knowledge conflict literature"** (Harsh Critic, Section 2 notes): This is a scope and related-work completeness concern. Given that we cannot verify the existence of specific unlisted papers, and the paper's scope is narrowly inference-time contrastive decoding rather than a survey of hallucination mitigation, this is scope creep. Removed.

- **"Hyperparameter selection may constitute implicit data leakage"** (Harsh Critic, Section 4): The critic speculates that the fixed parameters might have been chosen after observing test-set results. No evidence of this is provided; the critic is speculating. Removed.

- **"Standard deviation of 0.66 suggests insensitivity, not robustness"** (Harsh Critic, Section 6): This is a possible reinterpretation, not a factual error in the paper. The claim that robustness = insensitivity is a reframing without mechanistic evidence. Demoted to Trivial.

- **"Mechanical incoherence of EOS as MASK"** (Harsh Critic): The concern is valid as a Minor point (unexplained choice), but the "mechanically incoherent" framing is too strong — it is unconventional, not proven to be wrong. Kept as Minor with softened language.

- **Strength: "Banana example is an effective illustration"** (both reviewers): Retained as part of mechanistic motivation strength — it is specific (Section 3.2, Figure 1) and substantive.

- **Strength: "Robustness to hyperparameter choices"** (Strength Finder): This conflicts with the Minor weakness about ablation scope (only SQuAD v1.1, no multi-dataset validation). Strength dropped as unverified.

---

## Novel Insights

The most genuinely novel insight from the combined reviews is the identification of the **NoAns_EM / HasAns_EM tradeoff** as a potential confound. If Delta's contrastive logit adjustment is mechanically biasing the model toward "no answer" outputs (reducing overall confidence in specific token extractions), then the headline result may reflect threshold recalibration rather than reduced hallucination. This is a meaningful distinction because threshold recalibration is trivially achievable (e.g., by simply raising a confidence cutoff), while genuine hallucination reduction requires the model to produce better evidence-grounded outputs. Examining whether Delta improves the *joint* decision quality (e.g., balanced accuracy across HasAns and NoAns conditions) would resolve this and would make the paper substantially stronger.

---

## Suggestions

1. **Add CAD and DoLa as experimental baselines on all reported benchmarks.** This is the single most important revision — without it, the paper cannot situate its contribution.
2. **Report HasAns_EM and NoAns_EM jointly with a balanced accuracy metric** on SQuAD v2 to distinguish genuine hallucination suppression from abstention bias.
3. **Ablate the MASK token choice** (EOS vs. padding vs. UNK vs. random token) to validate the implementation and justify the EOS choice.
4. **Evaluate on at least one additional model** (e.g., Mistral 7B or Llama 3.1 70B) to support generalizability claims.
5. **Report multi-seed variance** on key results given the stochastic masking.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison to Paper Under Review |
|---|---|---|
| `/human_reviews/Th6NyL07na.md` (DoLa) | 7.25 | Multi-model evaluation, strong baselines, the method directly comparable to Delta but far more rigorously validated. |
| `/human_reviews/LebzzClHYw.md` (Instructive Decoding) | 7.50 | Similar contrastive mechanism, thorough multi-dataset / multi-model evaluation with proper baselines. |
| `/human_reviews/dlUjNdybnq.md` (PAD) | 5.50, Rejected | Contrastive inference method with mixed results and limited baselines — arguably closer to Delta's current state. |
| `/human_reviews/WqL4wOU3tw.md` | 4.75, Rejected | Simple adaptation with limited baselines and generalizability concerns. |
| `/human_reviews/mb9oOA3rD9.md` | 4.50, Rejected | Narrow experimental scope with limited baselines. |
| `/human_reviews/ech9J3xl9X.md` | 2.50, Rejected | No technical novelty, single model, missing baselines (lower end). |
| `/human_reviews/gNoqEdT2wO.md` | 2.33, Withdrawn | Essentially no new methods, single VLM. |

**Positioning:** Delta sits clearly below DoLa (7.25) and Instructive Decoding (7.50), which are accepted papers with the same general framing (inference-time contrastive decoding) but substantially stronger validation (multiple models, proper baselines including each other). Delta's most damaging gap — the absence of CAD (functionally nearly identical) and DoLa as baselines — is structural, not addressable by reframing. The single-model evaluation, unexplained EOS masking, and the HasAns/NoAns tradeoff confound further suppress the score.

Delta is closer to the rejected papers in the 4.5–5.5 range: it has a clear idea and some positive results, which keeps it above the truly weak papers (2.33–2.50 range that had no real technical contribution), but the structural experimental gaps prevent it from reaching the medium-acceptance range. The WqL4wOU3tw (4.75) and mb9oOA3rD9 (4.50) anchors are the closest analog — similar issues of limited baselines and single-setting evaluation. Delta is slightly weaker than PAD (5.50) because at least PAD has more comparison methods in its tables.

**Final score: 3.5** — the paper presents a coherent idea with some positive empirical evidence, but is not ready for ICLR in its current form due to the absence of its most direct baselines (CAD and DoLa), single-model evaluation, an unexplained core implementation choice (EOS masking), and an unacknowledged confound in its headline result.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>