Now let me look at some of the referenced human review papers for calibration.Now let me look at one more calibration reference before writing the final review.## Summary

This paper proposes the **Role-Guided and Self-Reflection (RoSe)** strategy to probe whether LLMs "know what they know" by subjecting them to multi-step self-reflection prompts with varying role guidance (teacher/classmate, judge/lawyer) and misleading cue information. Key findings include: LLMs are strongly sensitive to strong reminder phrases ("the answer is"), authority roles reduce but do not eliminate shortcut reliance, and LLMs trust authoritative roles more than peer roles. Building on these findings, the authors propose a **double-calibrated data filtering** strategy that extracts high-accuracy, self-consistent GPT-4 traces and uses them to fine-tune open-source LLMs, demonstrating reduced susceptibility to misleading cues. A new English Grammar QA dataset (EG-QA) is also contributed.

---

## Strengths

- **Concrete, clearly articulated empirical findings.** The paper documents a striking 9.58% accuracy drop (EG-QA) and 35.15% drop (JEC-QA) when switching GPT-4 from truth to random cues in a strong-reminder setting. This is a genuine, data-backed demonstration of shortcut reliance in a domain-specific knowledge scenario, consistent with the sycophancy literature.

- **Well-structured RQ framework.** RQ1–RQ4 in §4.1 are clearly defined and the experiments in Tables 2–3 directly address them. The three-step prompt design with systematic role × reminder × cue ablations is a principled evaluation approach.

- **Cross-domain evaluation.** Using EG-QA (English grammar education), JEC-QA (Chinese legal domain), and openBookQA (commonsense science) provides meaningful breadth and shows that shortcut sensitivity scales with domain-specific knowledge difficulty.

- **Fine-tuning results are consistent across models.** Tables 4–6 and Figures 3–4 show that all three open-source models (Spark-13B, Qwen-7B, LLaMA3-8B) reduce their Δ sensitivity to random cues after fine-tuning on GPT-4 double-calibrated traces—a consistent pattern that supports at least some benefit from the data-filtering approach.

- **New dataset contribution.** EG-QA covers 14 grammatical knowledge points with explicit train/ID/OOD splits, giving a structured resource for future fine-tuning evaluations.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No ablation baseline for the double-calibrated strategy.** The fine-tuning experiments compare fine-tuned vs. base models only. There is no comparison against (a) fine-tuning on any correct GPT-4 step-1 answer without double-calibration, or (b) standard CoT distillation on the same questions. This is a critical omission: if any GPT-4 distillation yields similar robustness improvements, the specific "double-calibration" filtering mechanism is not justified. As it stands, the improvement could be attributed entirely to exposure to GPT-4's reasoning style rather than to the two-stage filtering criterion. §4.2 and §5.3.2 claim the double-calibration is essential to the method's success, but the ablation to support this claim does not exist.

- **"Well-calibrated" terminology misrepresents the method and Eq. (1) is never operationalized.** The paper introduces Eq. (1) as a formal calibration definition ("if a model says 90% confident, it should be correct 90% of the time"), yet the experiments measure only average confidence and accuracy—never ECE, Brier score, reliability diagrams, or confidence-binned accuracy. The "first calibration" (keep correct or self-correcting traces) and "second calibration" (keep non-decreasing confidence traces) guarantee high-accuracy, self-consistent data, not calibrated data in the sense of Eq. (1). A model that is consistently 95% confident on all questions—right or wrong—would pass the second calibration. The paper would be more accurately and honestly described as extracting *high-accuracy, self-consistent* traces, which is a useful contribution but is different in kind from the formal calibration claim. This gap between claimed and demonstrated contribution is present throughout the abstract, §4.2, and §6.

- **Evaluation scope is exclusively multiple-choice QA on a narrow domain.** EG-QA is English grammar for Chinese middle/high school students (with Chinese-language instructions), and the fine-tuning results are all measured on this bilingual, domain-specific corpus plus openBookQA. The paper claims to evaluate whether LLMs "know what they know" in general, but the generalizability of the fine-tuning conclusions beyond this narrow setting is untested. The JEC-QA evaluation is prompt-only (no fine-tuning), so the fine-tuning contribution is effectively validated on a single domain. This significantly limits the scope of the headline claim.

### Minor

- **Role effects are relatively small and not statistically tested.** For GPT-4 on EG-QA (Table 2), the claimed authority advantage (teacher > classmate when cue is truth) corresponds to step-3 accuracy of 0.9494 vs. 0.9373—a difference of ~1.2 pp. Across all conditions, role differences are modest and there are no confidence intervals or significance tests. The conclusion that "LLMs tend to trust the role of authority more...similar to human behavior" is directionally plausible but overstated given the effect sizes reported.

- **Role effects potentially confounded by prompt phrasing.** Roles are implemented as textual phrases ("my teacher thinks," "my classmate thinks") with no neutral control (e.g., "Person A thinks") to distinguish authority semantics from surface lexical differences. The observed role gradient could be partly a lexical association artifact.

- **openBookQA is evaluated only under RoSe prompting.** Fine-tuning effects on commonsense reasoning are assessed within the RoSe framework (Figure 3), not with standard single-step prompting. The question of whether standard (role-free) openBookQA performance changes after fine-tuning on EG-QA grammar data is not addressed.

- **"Com" metric is non-standard and under-justified.** The harmonic-mean combination of accuracy and completion rate ($2 \times \frac{A \times C}{A+C}$) is introduced in a footnote (footnote 10). The motivating scenario is that base LLMs often fail to emit a valid option, but the metric penalizes appropriate uncertainty expression. This should be justified and discussed more prominently in the main text.

- **Yield rate of double-calibrated filtering not reported.** Footnote 4 states 18,598 well-calibrated instances are obtained, but not how many total instances were generated or discarded by each stage. This is needed to assess the method's practicality and whether the two stages are both informative.

### Trivial

- Eq. (1) and the formal factorization $P(r,a,c|\varphi,q) = P(r|\varphi,q) \cdot P(a,c|r,\varphi,q)$ are stated in §3 but never used or referenced in subsequent method or analysis sections—they read as rhetorical formalism.
- Step-3 prompt templates (the only step that varies across conditions) are moved to Appendix A.1, making the experimental setup harder to follow without extensive appendix reading.

---

## Nice-to-Haves

- Report ECE or reliability diagrams for verbalized confidence before and after fine-tuning to rigorously test whether the "calibration" language is appropriate. Even showing that verbalized confidence correlates better with accuracy post-fine-tuning would substantially strengthen the claims.
- Evaluate fine-tuned models on a general-purpose benchmark (e.g., MMLU, HellaSwag) with standard single-step prompting, to verify that the grammar-domain fine-tuning does not degrade broader capabilities.
- Include a data-scaling ablation: how does performance vary with the amount of double-calibrated fine-tuning data?
- Add a neutral role condition ("a person thinks the answer is X") to disentangle authority semantics from mere role-mention effects.
- Discuss how RoSe would extend to free-form generation tasks, where "knowing what you know" matters most.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic §5.3.1 anomalous JEC-QA step-3 value (0.9203 for "w/o, X, X").** The parsed value appears to be a table-column misalignment artifact from PDF extraction (likely the step-2 confidence score shifted into the step-3 accuracy column). The overall accuracy for this row (0.3349) correctly averages the step-1 and step-2 values (0.3336, 0.3364). This is a parsing artifact, not an error in the paper.

- **Harsh Critic's claim that "reliance on local information" reduction is entirely unsupported.** The paper does document Δ differences between base and fine-tuned models across multiple conditions; fine-tuned Δ values are consistently smaller. The legitimate complaint is the absence of a simpler baseline, not that the evidence is fabricated.

- **Harsh Critic's claim that the problem definition factorization (§3) is never exploited.** It is true that the factorization is not central to the analysis, but this is a presentational observation, not a structural flaw.

- **Claims about reproducibility concerns (hyperparameters, training logs).** The paper provides seed, optimizer, LoRA rank/alpha, learning rate, and a GitHub link; standard reproducibility requirements for this community are met.

- **Human Finder's concern about "not yet released" or unavailable models.** Not applicable; all cited models have accessible links in the paper.

- **Requests for theoretical proofs.** This is an empirical systems paper; formal theoretical guarantees are not standard in this subfield.

---

## Novel Insights

The most substantive novel insight is the *domain-difficulty amplification* of shortcut reliance: GPT-4's drop upon switching from truth to random cue in the familiar EG-QA domain (9.58%) is dramatically smaller than in the unfamiliar JEC-QA legal domain (35.15%). This suggests that shortcut exploitation is not a fixed property of a model but scales inversely with genuine domain competence—a useful diagnostic principle. The finding that role guidance partially compensates for this susceptibility, and that the compensatory effect tracks perceived authority (judge > lawyer, teacher > classmate), also extends the sycophancy literature into a more structured multi-role framework, though effect sizes are modest and warrant further controlled study.

---

## Suggestions

1. **Add the critical ablation**: Fine-tune on GPT-4 step-1 correct answers only (no double-calibration) and compare against the full pipeline in Tables 4–6. This single experiment would either validate the double-calibration design or reveal that simpler distillation achieves the same result.
2. **Reframe "calibration" language**: Replace "well-calibrated data" and "double-calibrated strategy" with "high-accuracy, self-consistent data" throughout. Alternatively, add ECE/reliability-diagram analysis to substantiate the calibration claim.
3. **Evaluate on a standard benchmark post-fine-tuning** with standard (non-RoSe) prompting to confirm capability is preserved and potentially generalized.
4. **Consolidate §3 formalism**: Either use Eq. (1) as an evaluation criterion (report conditional accuracy by confidence bin) or remove it from §3 to avoid a credibility gap between the stated problem and the actual evaluation.

---

## Score and Decision

**Calibration references:**

| Paper | Decision | Avg. Score | Comparison to this paper |
|---|---|---|---|
| *Towards Understanding Sycophancy in Language Models* (tvhaxkMKAn) | Accept Poster | 6.5 | Stronger: broader model coverage, free-form tasks, RLHF mechanism analysis |
| *Can LLMs Express Their Uncertainty?* (gjeQKFxFpZ) | Accept Poster | 6.0 | Stronger: uses ECE/AUROC, 5 model families, systematic framework |
| *Simple Synthetic Data Reduces Sycophancy* (WDheQxWAo4) | Reject | 5.0 | Comparable structure; that paper was rejected for limited novelty and no open-ended evaluation |
| *PersonaEval* (wZbkQStAXj) | Reject | 4.0 | Weaker: pure benchmarking, narrower scope |
| *LLMs have Intrinsic Self-Correction* (pTyEnkuSQ0) | Reject | 5.3 | Different flaw pattern (incorrect theoretical proofs); comparable practical issues |

This paper sits between *PersonaEval* (reject, ~4) and the accepted posters (~6). It makes multiple contributions (evaluation framework + dataset + fine-tuning) but fails to substantiate its central "calibration" claim with any formal calibration metric, and the fine-tuning contribution lacks the critical baseline needed to attribute improvements specifically to the double-calibration pipeline. The evaluation scope is narrower than the accepted sycophancy/calibration papers at ICLR 2025. The *Simple Synthetic Data* paper (rejected, ~5.0) is the closest match in structure and flaw pattern—it also lacked novelty and rigorous baselines—and this paper shares those issues while being slightly more exploratory and narrow in scope.

**Overall assessment on key axes:**
- *Originality*: Moderate. Role + self-reflection + confidence is a novel combination, but each component is established. The educational psychology framing is creative but superficial.
- *Importance of research question*: High. Shortcut reliance and self-knowledge in LLMs are important safety-adjacent topics.
- *Claims well supported*: Partially. The descriptive findings (sensitivity to cues, authority bias) are well-supported. The fine-tuning benefit claim lacks a key ablation.
- *Soundness of experiments*: Weak. Missing baselines, no significance testing, no calibration metrics despite the calibration framing.
- *Clarity of writing*: Acceptable but the table density makes results hard to parse.
- *Value to research community*: Moderate. EG-QA and the role-sensitivity empirical findings have real value; the double-calibration framing is misleading.

**Final Score: 4.0** — The paper has genuine but modest contributions, overclaims its central "calibration" contribution, and lacks the critical ablation baseline needed to validate its fine-tuning method. These are not individually fatal but together put the paper below the acceptance bar for ICLR.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>