## Summary

AutoCoder proposes **AIEV-INSTRUCT**, a data generation pipeline for code LLM instruction tuning that leverages multi-turn agent interaction (questioner/programmer roles) with execution feedback via a Docker-based code interpreter, including unit tests and stderr traces. The method uses a two-stage training process: a *Teaching Stage* distilled from GPT-4 Turbo and a *Self-Learning Stage* where the student model replaces the teacher. AutoCoder-33B achieves 90.9% pass@1 on HumanEval and is evaluated across seven benchmarks including HumanEval+, MBPP+, MultiPL-E, DS-1000, and LiveCodeBench.

## Strengths

- **Execution-verified data generation is a practical, well-motivated contribution.** The multi-turn dialogue with actual code execution, unit tests, and stderr feedback (Section 3, Figure 3) provides a richer supervision signal than standard single-turn instruction corpora. The ablation in Figure 6 demonstrates this empirically: for the 33B model, adding execution feedback with multi-turn dialogue yields a +10.1 point jump on HumanEval (81.3→91.4%), substantially larger than multi-turn alone (81.3 vs. 81.2).

- **Broad evaluation across seven benchmarks provides a realistic profile.** The paper evaluates on HumanEval, HumanEval+, MBPP/MBPP+, MultiPL-E, DS-1000, and LiveCodeBench (Section 5). The inclusion of HumanEval+ (stronger test cases) and LiveCodeBench (contamination-resistant, competition problems) gives a more complete picture than relying on a single benchmark.

- **Strong performance for the parameter scale.** AutoCoder-33B's 90.9% HumanEval, 78.0% HumanEval+, and competitive MultiPL-E scores are notably strong among models with ≤33B parameters. The performance improvement transfers to the 6.7B variant (78.7% HumanEval) as well (Figure 6, Table 1, Table 2).

- **The code interpreter post-processing idea is a practical systems contribution.** Adding special tokens for bash commands and code blocks to teach package installation (Section 4.1, Figure 5) extends the utility of code interpreters beyond built-in packages, even if under-evaluated in this submission.

## Weaknesses

### Fatal
None.

### Major

- **The headline claim that AutoCoder "surpasses GPT-4 Turbo and GPT-4o" is not supported by the experimental evidence.** As stated in the abstract and conclusion, the core evidence is a 90.9% vs. 90.2% difference on HumanEval (164 problems) — roughly a one-problem margin — with no variance estimates, repeated runs, or confidence intervals. Meanwhile, the stronger HumanEval+ benchmark directly contradicts this narrative: AutoCoder scores 78.0% vs. GPT-4 Turbo's 86.6% (Table 1), a gap of 8.6 points. LiveCodeBench (Table 4) also shows AutoCoder-33B at 25.4%, far below GPT-4o (46.1%) and GPT-4 Turbo (44.2%). The paper selectively emphasizes the benchmark where the margin is smallest while the wider-evidence picture suggests AutoCoder is competitive within its size class but not surpassing frontier models. This overclaim is a central framing issue, not a minor one, because the abstract, introduction (Figure 1), and conclusion all anchor their positioning on this unsupported assertion.

- **The self-learning stage is announced as a major contribution but never quantified or demonstrated.** Section 3.1 describes that "after every 2000 data entries, we split... evaluate both the teacher model and the student model... if the student model performs better, we move to the Self-Learning Stage." However, the paper never reports *when* this switch occurred, *how many* of the 169K entries came from self-learning, or *whether* the self-learning stage actually improved performance beyond continuing with teacher-generated data. Since reducing dependence on proprietary models is explicitly one of the paper's two motivating contributions (Section 1, "can we enable our student model to learn autonomously?"), this missing accounting is a gap in evidence for a core claim, not a peripheral detail.

- **Decontamination by 90% Levenshtein distance is too weak for the strength of the contamination-avoidance claim.** Section 3.2 states that only 113 entries were excluded from 186K by this filter. A 90% character-similarity threshold catches near-duplicates but not semantically equivalent rewrites or benchmark-inspired paraphrases. Given that the paper makes strong performance claims on HumanEval, MBPP, and DS-1000 — benchmarks where contamination directly inflates scores — a more robust analysis (e.g., semantic similarity, benchmark-problem-family-level exclusion) is warranted.

### Minor

- **The ablation isolates execution feedback from base/multi-turn but not the teaching-vs-self-learning components.** Figure 6 compares Base, Base+ST, Base+MT, and Base+EFMT, but does not separately test: teacher-only vs. self-learning data; data quantity as a control (the self-learning data might simply be larger); or whether gains come from more dialogue turns rather than the specific execution-verified mechanism. The paper's causal narrative about AIEV-INSTRUCT's specific contribution is therefore stronger than what the ablation supports.

- **Baseline comparisons mix heterogeneous sources with inconsistent evaluation protocols.** Table 1 combines numbers from official websites, technical reports, the EvalPlus leaderboard, and the authors' own implementations (e.g., GPT-4o is "self-implemented," while GPT-4 Turbo and Claude 3.5 Sonnet come from official reports). The paper does not document whether prompting, stopping criteria, temperature, or retry policies were matched to the leaderboard protocols for self-evaluated closed models (Section 5). This doesn't invalidate the results but weakens direct ranking claims.

- **The code interpreter contribution lacks a systematic benchmark.** The paper claims AutoCoder is "the only model that supports automatically installing external packages in the Code Interpreter" (line 43) but provides no success-rate benchmark, safety analysis, or side-by-side evaluation under controlled conditions. The mechanism is described (Section 4.1, post-processing in Figure 5) but not measured against a defined task suite.

### Trivial

- The paper uses "surpass" and "outperforms top-ranked models" (line 41, Figure 1 caption) without acknowledging the tiny margin or the less flattering HumanEval+ picture. The phrasing should be tempered to match the evidence.

## Nice-to-Haves

- Reporting training-stage composition: the fraction of teaching vs. self-learning data, average feedback rounds per sample, and discard rate after 7 attempts, would make the method transparent and strengthen the contribution narrative.
- Including few-shot examples from AIEV-INSTRUCT dialogues (successful and failed) in the main paper would help readers assess the quality of the generated supervision signal.
- Evaluating the code interpreter on a defined task suite requiring package installation, with safety constraints documented, would turn an advertised claim into a measurable contribution.

## Removed Points

- *"The student model surpassing the teacher is claimed impossible in the introduction, which is too strong."* — The paper frames this as a heuristic motivation ("unlikely"), not a proven impossibility. It is standard rhetorical framing for why their method is needed, not a factual error.
- *"The 'Dataset Accuracy Theoretical Analysis' adds little."* — This is a minor presentation criticism of an appendix-deferred section. The main paper's experimental results are the substantive evidence.
- *"Results are aggregated from different sources with missing entries across benchmarks."* — While true, this is common practice in code LLM papers. The EvalPlus leaderboard provides a standardized evaluation protocol, and the paper acknowledges its curated baseline selection (Section 5). This is a transparency concern rather than a validity threat.
- *"The DS-1000 results show AutoCoder underperforms badly on PyTorch (26.5 vs. 50.0) but this is buried."* — The paper actually presents the full DS-1000 table with all subcategories (Table 3); the PyTorch underperformance is visible. This is a narrative balance observation, not a concealed result.
- *"GPT-4o is self-implemented with no protocol matching to leaderboard standards."* — This is partially addressed by the Appendix A reference (stripped by parser) and the EvalPlus leaderboard standardization. The concern is real (Major weakness) but should not be inflated to additional instances.
- *Code-interpreter weaknesses about undefined package sources, failure handling, environment persistence, and unsafe command blocking.* — These are operational system-design questions more appropriate for an engineering evaluation than a conference paper on training methodology. Moved to Nice-to-Haves.
- *Requests for confidence intervals on all benchmarks.* — Single-run evaluation on these code benchmarks is the community norm (as evidenced by the EvalPlus leaderboard and the calibration corpus papers such as OctoPack, which similarly report point estimates). Moved to Nice-to-Haves.

## Novel Insights

The paper's execution-verified, multi-turn data generation pipeline addresses a real gap in instruction tuning for code LLMs: most synthetic code data lacks grounding in actual program behavior. The ablation (Figure 6) provides a strong empirical signal that execution feedback is the dominant driver of improvement, not merely multi-turn dialogue or increased data volume. However, the gap between the paper's framing ("surpasses GPT-4") and the broader evidence (HumanEval+, LiveCodeBench) reveals the common pitfall of optimizing a narrative around one benchmark while the multi-benchmark picture tells a more nuanced story. The unverified self-learning stage represents a missed opportunity — if the data were reported, it could have established a genuinely valuable contribution about bootstrapping beyond teacher-level quality with execution-grounded feedback.

## Suggestions

1. **Temper the headline claims** to reflect the full benchmark picture. Replace "surpasses GPT-4 Turbo and GPT-4o" with language like "competitive with" or "closely matches on HumanEval," and prominently acknowledge the HumanEval+ and LiveCodeBench results in the abstract.
2. **Report the teaching vs. self-learning data accounting**: number of entries from each stage, when the switch occurred, how many feedback rounds per sample, and discard rates. This costs nothing and would substantially strengthen the self-learning contribution claim.
3. **Provide a simple code-interpreter benchmark** with 10-20 tasks requiring external package installation, reporting success rates. This would ground the claim in measurable evidence rather than assertion.

## Score and Decision

**Calibration anchors compared:**
- **OctoPack** (8,8,6, Accept Spotlight): Stronger novelty (new dataset CommitPack + new benchmark HumanEvalPack) with similarly solid code LLM instruction-tuning results. AutoCoder is a step below in breadth of methodological contribution.
- **Fq8tKtjACC / phi-1** (6,6,6,5,8,5, Reject): Strong empirical results but rejected for overclaimed contributions and misleading comparisons. AutoCoder shares this overclaim pattern (narrow HumanEval margin framed as surpassing frontier models), though its method contribution is clearer.
- **GIFT4Code** (3,5,5,5, Reject): Execution-derived feedback for code generation, rejected for insufficient evaluation and lack of strong baselines. AutoCoder is significantly stronger — broader benchmarks and clearer ablation.
- **kRjLBXWn1T** (5,5,6,5, Reject) and **1nHQRsb3Ze** (5,5,5, Reject): Borderline empirical papers where improvements are real but overclaims or methodological gaps hold them back. AutoCoder fits this tier well.

AutoCoder has genuine strengths — a clear, practically useful method with strong results for its size class and broad benchmark coverage — but the overclaimed "surpasses GPT-4" narrative, the unverified self-learning contribution, and the weak decontamination procedure keep it in the borderline range. Compared to the Fq8tKtjACC paper (rejected despite higher scores from individual reviewers for strong empirical results), this paper's underlying results are somewhat less exceptional and the methodological gap (self-learning not demonstrated) is more central to its claimed contributions.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>