## Summary

AutoCoder introduces AIEV-INSTRUCT, a code instruction dataset generation method that uses agent interaction with execution verification (running code in a Docker container and feeding stderr/stdout back for iterative correction across up to 7 rounds). It includes a two-stage design: a Teaching Stage (using GPT-4 Turbo as questioner/programmer) and a Self-Learning Stage (replacing the teacher with the student model once it surpasses the teacher on an internal test set). The resulting 169K-sample dataset is used to fine-tune AutoCoder (33B) and AutoCoder-S (6.7B). The paper also introduces a code interpreter that can install external packages.

## Strengths

- **Execution-verified multi-turn data generation is well-motivated and empirically supported**: The AIEV-INSTRUCT pipeline addresses a real limitation of prior methods (SELF-INSTRUCT, EVOL-INSTRUCT, OSS-INSTRUCT) that distill both correct and incorrect knowledge without validation. The ablation in Figure 6 demonstrates that execution-feedback multi-turn (Base+EFMT) consistently outperforms both single-turn (Base+ST) and standard multi-turn (Base+MT), with notable improvements on DS-1000 (+5.1pp for 33B) and HumanEval (+10.1pp for 33B). This constitutes a clear, positive contribution to the methodology of code instruction tuning data generation.

- **Comprehensive benchmark coverage across multiple dimensions**: The paper evaluates on 7 benchmarks—HumanEval(+), MBPP(+), MultiPL-E (6 languages), DS-1000 (7 libraries), and LiveCodeBench (3 difficulty levels)—providing a multi-faceted assessment rather than relying solely on HumanEval. AutoCoder-33B leads among same-scale open-source models on most benchmarks (Table 1–4).

- **Dataset decontamination is thorough**: The paper applies Levenshtein distance (90% threshold) against all benchmark test sets, removing 113 entries, and builds on source datasets (Magicoder-Evol-Instruct, Magicoder-OSS-Instruct) that had already undergone contamination detection. This provides reasonable protection against data leakage inflating scores.

## Weaknesses

### Fatal

None.

### Major

- **The central claim of "surpassing GPT-4 Turbo and GPT-4o" is not supported by the evidence**: The abstract's headline claim is based on a 0.7 percentage point advantage on HumanEval (90.9% vs 90.2%), which is well within the standard error of a binomial proportion at p≈0.90, n=164 (~2.3%). Meanwhile, AutoCoder trails GPT-4 Turbo by 8.6pp on HumanEval+ (78.0% vs 86.6%), by 3.2pp on MBPP (82.5% vs 85.7%), by 2.7pp on MBPP+ (70.6% vs 73.3%), and by 18.8pp on LiveCodeBench (25.4% vs 44.2%). The paper frames its entire contribution around this single noise-level data point while consistently trailing on harder and more robust benchmarks. The claim of superiority is misleading; the honest characterization is that AutoCoder-33B is a competitive open-source 33B model that sometimes approaches—but clearly does not surpass—frontier closed-source models.

- **The Self-Learning Stage—the paper's most novel conceptual contribution—is unevaluated**: The two-stage Teaching→Self-Learning design is presented as a core contribution (addressing the second motivating question about autonomous learning), but the paper provides no evidence it actually works or matters. Specifically: (1) The paper never reports how many of the 169K samples came from the Self-Learning Stage—was it 1K or 160K? (2) There is no ablation comparing a model trained on Teaching-only data vs. the full pipeline—Figure 6 only compares Base, Base+ST, Base+MT, and Base+EFMT. Without knowing whether self-learning data was even used in substantial quantity, the claim that AIEV-INSTRUCT "reduces dependence on expensive closed-source models" is unsupported. This leaves only the execution-verified multi-turn aspect as a validated contribution, while the more ambitious self-learning component remains an untested proposal.

- **The transition criterion from Teaching to Self-Learning may be biased**: The switch is triggered when the student's Pass@1 on a test set exceeds the teacher's. But this test set is drawn from the generated data itself (1:9 split every 2000 entries), meaning the student—being directly fine-tuned on this distribution—may overfit to this narrow test set, leading to premature or meaningless transition. The criterion is not validated against any standard uncontaminated benchmark (e.g., HumanEval), so we cannot tell whether "surpassing the teacher on the internal test set" corresponds to genuine capability improvement.

### Minor

- **Ablation shows uneven gains across benchmarks, suggesting potential overfitting to HumanEval-style problems**: For the 33B model, adding execution feedback (Base+MT→Base+EFMT) yields a 10.1pp jump on HumanEval (81.3→91.4) but only 2.3pp on MBPP (81.3→83.6) and 2.8pp on DS-1000 (44.2→47.0). This asymmetry suggests the execution-verified multi-turn data may be disproportionately benefiting HumanEval-style problems rather than improving general code capability. The paper does not discuss this discrepancy.

- **The code interpreter feature is only qualitatively evaluated**: The interpreter with automatic package installation (Section 4.1) is a practical contribution, but it is disabled in all quantitative evaluations (Section 5) and supported only by a qualitative example (Figure 2). There is no quantitative comparison of code generation with vs. without the interpreter, nor comparison with GPT-4's interpreter.

- **Missing comparative data in Table 1**: Several entries for HumanEval+ and MBPP+ are unavailable for closed-source models (marked "−"), making holistic comparison difficult. This is partly unavoidable (closed-source models don't report these numbers), but it weakens the comparative analysis.

## Trivial

None.

## Nice-to-Haves

- Run the ablation with a Teaching-only variant (no Self-Learning Stage) to finally validate or invalidate whether self-learning contributes to the final model.
- Report confidence intervals or run multiple evaluation samples on HumanEval to establish whether the 0.7% gap over GPT-4o is real.
- Perform a failure analysis on HumanEval+ and LiveCodeBench to understand the nature of AutoCoder's deficits relative to GPT-4 Turbo.
- Quantitatively evaluate the code interpreter feature (e.g., on HumanEval with interpreter enabled vs. disabled).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"GPT-4 Turbo itself may have seen HumanEval during training—indirect contamination"**: This speculates about the training data of a cited model without evidence, and is irrelevant to evaluating the paper's own methodology. Removed per hard rules.

- **"MultiPL-E exclusion of closed-source models seems selective"**: The paper states that MultiPL-E's official library does not support testing closed-source models. While the harsh critic notes the EvalPlus leaderboard tests closed-source models via API, this is a different evaluation framework. This is at most a nice-to-have, not a substantive flaw.

- **"Theoretical analysis in unavailable appendix"**: Per hard rules, missing appendix content should not be flagged as a weakness since the parser strips appendices from all papers.

- **"Maximum of 7 iterations means failed entries are discarded—what fraction?"**: This is a minor implementation detail/reproducibility nitpick. Removed per hard rules on reproducibility nitpicks.

- **"Diversity of generated data when GPT-4 Turbo plays both questioner and programmer"**: This is a generic concern without specific evidence it causes a problem; the paper's competitive benchmark results themselves serve as evidence the data is sufficiently diverse.

- **"Cost per valid data point could be higher than stated $1,000/6,500 entries"**: This is an unspecific reproducibility/cost nitpick without evidence the claimed cost is wrong.

- **Strength "Theoretical dataset accuracy analysis (Appendix B)"**: Dropped because it references appendix content that we cannot verify.

- **Strength "Self-Learning Stage reduces dependence on proprietary models"**: Dropped because this conflicts with the verified Major weakness that the Self-Learning Stage is unevaluated—we cannot claim this as a strength when we have no evidence it works.

## Novel Insights

The divergence between HumanEval and HumanEval+ performance (90.9% vs 78.0%) for AutoCoder is striking and reveals a pattern: execution-verified training may produce solutions that pass the original test cases but lack the robustness to pass extended test suites. This suggests that execution feedback with unit tests, while improving surface-level correctness, may not fully align with generating truly generalizable code—particularly if the unit tests themselves come from the same GPT-4 Turbo that generates the solutions. The feedback loop confirms code passes its own tests, but those tests may be incomplete. This is subtly different from the standard overfitting narrative and points to a fundamental limitation of self-verified training data.

## Suggestions

- Add a Teaching-only ablation: train a separate model on just the data generated during the Teaching Stage (before any self-learning transition), and compare it to the full pipeline. If performance is identical, the self-learning claim collapses; if it improves, you have the evidence you need.
- Reframe the abstract and title to accurately reflect the contribution: "competitive with" rather than "surpassing" GPT-4, and emphasize the execution-verified data generation method rather than the HumanEval leaderboard position.
- Validate the Teaching→Self-Learning transition criterion against an external benchmark (e.g., report HumanEval scores at the transition point) to show the internal test set is a meaningful proxy.

## Score and Decision

**Calibration anchors:**
- **DataEnvGym** (avg 7.5, Accept spotlight): Richer framework for autonomous data generation with student feedback loops, thoroughly evaluated across multiple tasks. AutoCoder is less systematic and less well-evaluated.
- **WizardCoder** (avg 6.25, Accept poster): Also claims surpassing closed-source models on HumanEval, with similar reviewer concern about narrow benchmark. But WizardCoder had more consistent leads across benchmarks and a clearly validated method (Evol-Instruct). AutoCoder trails GPT-4 Turbo on most benchmarks, making the overclaim more severe.
- **phi-1** (avg 6.0, Reject): Impressive small model with strong HumanEval results but contamination concerns and weak evaluation on harder benchmarks. Similar pattern of strong headline number but weaker depth.
- **GIFT4Code** (avg 4.5, Reject): Execution-feedback instruction tuning for code, similar topic area, but with weaker motivation and no baselines. AutoCoder has stronger empirical results but also more severe overclaiming.
- **ROBO-INSTRUCT** (avg 4.75, Reject): Simulator-based verification for code instruction tuning, limited experiments, overclaimed contributions. Very similar weakness profile.
- **Paramanu-Ganita** (avg 2.33, Reject): Extreme overclaiming (208M model beats 7B+ models by 30%) with contamination concerns. AutoCoder is far more plausible than this, but shares the overclaiming pattern at a milder level.

AutoCoder sits between GIFT4Code/ROBO-INSTRUCT (4.5–4.75) and phi-1/WizardCoder (6.0–6.25). The execution-verified multi-turn approach is a genuine, ablation-backed contribution, placing it above GIFT4Code and ROBO-INSTRUCT. However, the severe overclaiming on GPT-4 comparison and the completely unevaluated self-learning stage place it well below WizardCoder and phi-1. The most comparable anchors are ROBO-INSTRUCT (4.75) and GIFT4Code (4.5), with AutoCoder having slightly stronger empirical results but more damaging overclaims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>