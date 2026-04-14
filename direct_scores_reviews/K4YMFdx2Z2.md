## Summary

This paper introduces Unsolvable Problem Detection (UPD), a framework evaluating Large Multimodal Models' (LMMs) ability to withhold answers when problems are structurally unsolvable. The authors define three problem types — Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD) — build the MM-UPD Bench from MMBench via controlled transformations, and introduce a Dual Accuracy metric that jointly penalizes standard-question and UPD failures. Experiments across 21 open- and closed-source models reveal a striking near-zero correlation between standard MMBench performance and UPD performance, and explore mitigation strategies including CoT, self-reflection, and instruction tuning.

---

## Strengths

- **Three-category taxonomy expanding the definition of unsolvable problems.** Prior work addresses only image-question mismatches (IVQD equivalent). The addition of AAD (correct answer absent from options) and IASD (entire option set semantically incompatible) meaningfully broadens the evaluation space and enables differentiated diagnosis of model failure modes (F5: models like LLaVA-OV-7B fail both AAD and IASD in base settings, indicating intrinsic inability to refuse; others fail AAD but not IASD, indicating sensitivity to option granularity).

- **Dual Accuracy metric that penalizes capability trade-offs.** Unlike evaluating refusal rate alone, Dual Accuracy requires correctness on both the matched standard question and the UPD question, preventing models from gaming the metric by blanket refusal. This design directly measures trustworthiness as a joint property.

- **Striking and well-evidenced empirical finding of near-zero standard–UPD correlation.** The paper rigorously computes correlation coefficients between Original Standard and Dual/UPD accuracy across 21 models (Table 2: max r=38.7, min r=-0.35). The case of LLaVA-OV-7B achieving >80% Original Standard but <6% Dual accuracy in base settings is a vivid, reproducible illustration that existing leaderboard rankings do not capture this dimension of model reliability. This finding alone has significant community value.

- **Ability-wise fine-grained diagnostic (Figure 4).** The analysis reveals that even GPT-4o has near-zero AAD performance for specific abilities (#3: Object Localization, #6: Attribute Comparison) while performing well on others (#10: Identity Reasoning, #2: Celebrity Recognition). This granularity is directly actionable for model developers, unlike aggregate scores.

- **Vision-language error disentanglement in Section 6.2.** By feeding the correct answer directly into the prompt and checking if the model selects "None of the above," the paper identifies whether errors originate in vision (e.g., failing to count cows correctly) or language understanding (e.g., inability to reason about physical properties at 69% even with correct answer given). This diagnostic is original and informative.

---

## Weaknesses

- **No inter-annotator agreement statistics for benchmark quality.** For MM-IASD and MM-IVQD, the paper describes manual removal of ambiguous samples but provides no inter-annotator agreement scores, no number of items removed, and no kappa statistics. For a benchmark paper at ICLR, this omission is significant: the validity of the "unsolvable" label — and thus the benchmark's reliability — cannot be independently assessed.

- **Text-only baseline for IASD is absent, undermining the multimodal claim.** IASD tasks pair questions with semantically irrelevant answer sets (e.g., color question with angle options). If a text-only language model, given only the question and shuffled options, can detect the incompatibility as reliably as multimodal models, the IASD sub-task does not require visual reasoning and its inclusion in a multimodal trustworthiness benchmark is weakened. This analysis is missing and material to the paper's core claims.

- **Dual Accuracy failure decomposition is absent.** The metric counts only question-pairs where both standard and UPD are answered correctly, but the paper never decomposes failures into (a) standard-only failure, (b) UPD-only failure, (c) both-failure. This matters for interpretation: a model with low standard accuracy can score zero on Dual Accuracy even with perfect UPD detection, which is penalizing unrelated ability. The paper treats Original Standard as an "upper bound" but does not quantify how many points are lost to (a) vs. (b), obscuring where each model actually needs improvement.

- **Extreme prompt sensitivity not adequately analyzed.** Table 1 reveals enormous swings across Base/Option/Instruction settings that are difficult to attribute to genuine unsolvability reasoning: CogVLM-17B AAD jumps from 0.5% (Base) to 39.3% (Option) and back to 3.8% (Instruction). LLaVA-OV-7B similarly varies from 4.5% to 29.4% to 25.9%. These swings suggest models are reacting to surface instruction cues rather than detecting unsolvability, which is a safety-relevant fragility. The paper notes that "effective prompting strategies vary by LMMs" (F4) but does not analyze *why* certain instructions cause catastrophic reversals, nor what this implies for deployment.

- **Instruction tuning degradation measured only against MMBench-derived Original Standard, not held-out benchmarks.** Table 4 shows LLaVA-NeXT-13B's Original Standard drops from 76.7 to 68.9 after tuning. However, this is measured on the same MMBench subset from which MM-UPD was derived. Performance on independent general benchmarks (e.g., ScienceQA, MME, MMMU) is not reported, leaving ambiguous whether the degradation reflects narrow overfitting or broader catastrophic forgetting.

- **Benchmark contamination risk is unaddressed.** MM-UPD is built atop MMBench, and several evaluated models (e.g., InternVL2 series) were trained with MMBench data. The "unsolvable" transformations are novel, but the underlying images and questions may be familiar to models with heavy MMBench training. The paper does not discuss or analyze this risk, which could confound the reported results particularly for models that rank highest.

- **GPT-4o-mini evaluator dependency only partially validated.** The evaluation protocol relies on GPT-4o-mini to detect refusal phrasing. The paper mentions a human judgment comparison in Appendix D.2 but provides no summary of that comparison in the main text, leaving the reliability of this evaluator uncharacterized for readers.

---

## Nice-to-Haves

- **False refusal rate on solvable standard questions under UPD instructions.** The Instruction setting adds UPD-specific hints to *both* standard and UPD questions (as noted in Section 4.3). Measuring how often models incorrectly refuse the solvable variant would fully characterize the helpfulness/honesty trade-off that Dual Accuracy only partially captures.

- **Direct empirical comparison with prior unsolvable benchmarks.** Evaluating models from Guo et al. (2024) and Akter et al. (2024) on MM-UPD and vice versa would concretely establish whether MM-UPD measures something quantitatively distinct, rather than asserting this qualitatively.

- **Expanded CoT and self-reflection study.** Table 3 covers only 4 models. Given the varied effectiveness observed (CoT helps LLaVA-NeXT-13B and LLaVA-OV-7B but not InternVL2-8B or GPT-4o), a broader study across more models would yield more generalizable conclusions.

- **Semantic similarity analysis for IASD.** A histogram of semantic similarity scores between shuffled answer sets and their paired questions would reveal whether IASD incompatibilities are trivially detectable (near-random options) or require genuine reasoning.

- **Benchmark source diversity.** Supplementing MMBench with a modest number of questions from other benchmarks would address the single-source limitation and test whether findings generalize beyond MMBench's distribution.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"No quantitative reporting of instruction-tuning degradation" (Harsh Critic, Table 4).** Table 4 explicitly shows Original Standard before and after for both LLaVA-NeXT-13B (76.7→68.9) and LLaVA-NeXT-34B (84.3→78.6). The critic misread the table. The paper does quantify degradation on MMBench; the legitimate issue (absent held-out benchmark evaluation) is captured as a separate weakness above.

- **"Statistical significance testing absent for pairwise comparisons" (Harsh Critic).** For large-scale LMM evaluation benchmarks, single-run evaluation without confidence intervals is the accepted community norm (CircularEval already reduces variance). Demanding significance tests here imposes non-standard rigor.

- **"Diversity claim ironic given mechanical derivation from MMBench" (Harsh Critic).** The paper does not claim diversity in source data; it claims diversity in *ability coverage* (18 ability dimensions from MMBench's taxonomy). This is not the same claim, and the criticism conflates the two.

- **"The self-reflection loop is too crude" (Harsh Critic).** The paper explicitly presents self-reflection as a baseline/probe rather than a novel method, and follows established LLM self-reflection protocols. Criticizing its simplicity misattributes the paper's intent.

- **"English-only evaluation" (Harsh Critic).** This is scope creep; cross-lingual evaluation is not part of the stated contribution.

- **"Benchmark construction too mechanical for ICLR" (Harsh Critic).** Benchmark derivation via controlled transformation (removing correct options, shuffling answer sets) is a legitimate and widely used methodology in evaluation research. The criticism that the contribution must be more "algorithmic" reflects a misapplication of ICLR standards to benchmark papers.

- **"Why not MMMU or MathVista?" (Harsh Critic).** The paper explicitly addresses this in Appendix B.6 and briefly in the main text. The justification (expert-level reasoning benchmarks deviate from the reliability aspect, fine-grained ability coverage) is reasonable.

---

## Novel Insights

The most genuinely novel empirical insight in this paper — supported by Table 2's correlation analysis and individual examples like LLaVA-OV-7B — is that state-of-the-art open-source LMMs can simultaneously achieve near-human parity on standard VQA benchmarks while being almost entirely incapable of recognizing unsolvable problems in the base setting (near-zero Dual accuracy). This is not obvious: one might expect that a model with better internal representations would also be better at detecting structural inconsistencies. The finding suggests that current open-source training pipelines optimize aggressively for "always answer" behavior, and that the refusal capability seen in closed-source models is not an emergent consequence of general capability but a result of deliberate RLHF-style alignment targeting real-world deployment. The ability-wise breakdown further reveals that refusal difficulty is highly non-uniform across cognitive abilities, with perceptual counting and attribute comparison being far harder to refuse on than identity or celebrity recognition — a distinction that would be invisible from aggregate scores alone.

---

## Suggestions

- **Add failure decomposition to Dual Accuracy.** Report, for each model, what fraction of Dual Accuracy losses arise from (a) standard-question failures, (b) UPD-question failures, and (c) both. This would let readers immediately identify whether a low-scoring model needs better standard comprehension or better refusal capability.

- **Run a text-only ablation on IASD.** Apply a text-only GPT-4 baseline to the IASD questions (question + shuffled options, no image) and compare its detection rate to multimodal models. If text-only models excel, IASD should either be repositioned or complemented with harder IASD instances where image context is genuinely needed to detect incompatibility.

- **Report held-out general benchmark scores for instruction-tuned models.** Evaluate the LoRA-tuned LLaVA-NeXT variants on at least one benchmark outside the MMBench family (e.g., MME or ScienceQA) before and after tuning, to establish whether the observed Original Standard drop reflects specific forgetting or broader degradation.

- **Discuss prompt-sensitivity fragility explicitly.** The swings of 30–40 percentage points in Dual Accuracy across Base/Option/Instruction settings for the same model deserve dedicated analysis. Characterize which prompt elements drive the changes and discuss implications for robust deployment, since a model that is highly sensitive to prompt wording cannot be reliably deployed.

- **Provide inter-annotator agreement for the manual filtering steps.** Even a brief report (e.g., Cohen's kappa between two annotators on a random 100-sample subset) would substantially strengthen confidence in the benchmark labels.

---

**Evaluation Summary:**

- *Novelty:* Moderate-to-good. The three-category UPD framework and Dual Accuracy metric are genuine conceptual contributions that extend prior work meaningfully. Not theoretically novel, but the framing is clean and useful.
- *Technical soundness:* Moderate. Benchmark construction is reasonable but lacks quality controls. The Dual Accuracy metric conflates two failure modes. The instruction-tuning analysis is incomplete relative to its claims.
- *Empirical support:* Good. The 21-model comparison across three settings with ability-wise breakdown is thorough, and the central correlation finding is credibly established.
- *Significance:* High. Demonstrating that current benchmarks are blind to a practically important failure mode, with compelling empirical evidence, is a meaningful contribution to the LMM evaluation community.
- *Clarity:* Good. The paper is well-structured and the examples in Figures 1–2 clearly communicate the three UPD types.

MY FINAL SCORE: <pineapple>6.2</pineapple>