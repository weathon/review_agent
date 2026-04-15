## Summary
This paper proposes **SPA (Spread Preference Annotation)**, an iterative alignment framework that expands a small seed preference dataset by generating response pairs from the current policy, labeling them using a DPO-style implicit reward derived from the policy/reference log-ratio, and then training with a noise-aware refinement procedure. Empirically, the method delivers large gains over small-seed DPO on AlpacaEval 2.0 and smaller but consistent gains on MT-Bench, and includes useful analyses across seed sizes, seeds, and several model families.

## Strengths
- **The core self-labeling rule is simple, specific, and practically compelling.** Rather than relying on an external reward model or prompting a stronger judge, SPA uses the policy’s own DPO-style implicit reward for preference judgment (Eq. 7–8). This is a concrete design choice, not a generic “self-training” story, and it is directly tied to a lightweight implementation on top of standard DPO.
- **The gains over the relevant low-data baseline are large on the main benchmark.** In Table 1, with only **3.3%** gold preference labels, SPA improves over DPO from **7.68 → 21.13** win rate and **9.03 → 15.39** length-controlled win rate on AlpacaEval 2.0, while MT-Bench also rises from **6.81 → 6.94**. Whatever one thinks of the evaluation protocol, this is a substantial empirical effect over the paper’s main baseline.
- **The paper includes several targeted analyses that genuinely illuminate the method.** Table 3 shows SPA’s advantage persists across seed sizes; Table 4 checks robustness to different sampled seeds; and Table 7 probes an important design decision—using the initial SFT model as the judgment reference—which materially affects performance. These are more informative than generic extra experiments.
- **The ablation isolates where the added benefit comes from.** Table 6 shows that data expansion is doing most of the heavy lifting, while de-coupled noise detection adds a further improvement beyond plain self-refinement. This is useful because it prevents over-crediting the denoising component.
- **There is some evidence of breadth beyond one backbone.** In Table 5, SPA improves over DPO on Phi-2, LLaMA-3-8B-Instruct, and Phi-3-14B-Instruct, suggesting the approach is not purely a one-model artifact.

## Weaknesses
### Fatal
- **The paper’s strongest framing claim—improving “alignment with human preferences”—is not directly validated on human preference data.** All headline results are on **GPT-4-judged AlpacaEval 2.0** and **GPT-4-judged MT-Bench** (Section 5.1). For a paper whose main contribution is replacing expensive human preference annotation with self-generated labels, this is a serious evidential gap: the experiments establish improvement on LLM-judge benchmarks, but not cleanly improved human alignment. The paper does have access to gold preference labels in UltraFeedback for training seeds, yet does not report held-out agreement of SPA-generated labels with human labels, nor any held-out human evaluation. This substantially weakens the central claim.

### Major:
- **The abstract/introduction overclaim comparison to “using the entire data” is not supported by a controlled baseline.** The abstract states superior performance “with only 3.3% of the ground-truth preference labels … compared to the cases using the entire data,” and Section 5.2 compares to Zephyr-7b-β in Table 1. But Zephyr is a separately trained model/pipeline, not a same-recipe full-data DPO control where only the amount of preference data changes. From the paper alone, one cannot attribute the gap to SPA’s data efficiency rather than other recipe differences. This is not a minor wording issue; it is one of the paper’s marquee claims.
- **The mechanism claims around noise detection are only weakly validated.** Section 4.2 makes a fairly specific story: low-confidence examples are likely noisy, and extrapolated logits approximate a more strongly aligned model that detects that noise better. Table 6 shows a useful downstream gain from DND, but the paper never measures whether flagged examples are actually noisier, whether Eq. 12 better identifies mislabeled pairs than simpler alternatives, or whether the extrapolated predictor behaves like a more aligned model in this setting. As written, the denoising component is supported as a heuristic that helps, not as a validated mechanism.
- **The key judgment rule is evaluated only through downstream training outcomes, not by label quality.** The paper argues that Eq. 7 “explicitly extracts the model’s inherent preference” and is more effective than reward-model or in-context judging. Table 2 and Table 7 show stronger downstream benchmark performance, which is useful, but they do not tell us whether SPA’s self-generated preference labels are actually more accurate relative to human labels. Since the method’s core contribution is the preference-judgment rule itself, direct label-agreement analysis on held-out gold pairs is notably missing.
- **Several breadth claims should be stated more narrowly than they currently are.**  
  - The “no seed” experiment in Section 5.3 starts from **Mistral-7B-instruct-v0.1** with **Mistral-7B-base** as reference, i.e., an already instruction-tuned/aligned initialization, not the same setting as the main SFT-start experiments. So the evidence supports “SPA can further improve an already instruct-tuned model without additional seed preference labels,” not a broad claim that SPA generally works without seed data.  
  - The cross-model generalization in Table 5 is encouraging, but the starting checkpoints are not uniform across families (“UltraChat-tuned” for Phi-2 versus general instruct models for LLaMA-3 and Phi-3), which weakens strict comparability.

### Minor
- **The explanation for PairRM underperforming is speculative.** In Section 5.2 the paper attributes the gap to distribution shift as iterations proceed. That may be plausible, and Figure 3 is consistent with it, but no direct measurement of distribution shift or reward-model degradation is provided.
- **Variance/stability is not fully characterized.** Table 4 is helpful, but it also shows noticeably larger variance for SPA on LC win rate than for DPO. The paper argues the effect remains positive, which is fair, but more systematic uncertainty reporting on the main results would strengthen confidence.
- **Iteration behavior is not deeply analyzed.** Figure 3 shows performance peaking around iteration 2 and then slightly dropping at iteration 3 for SPA, but the paper does not discuss stopping criteria or when iterative self-labeling begins to saturate or hurt.

### Trivial
- **The “works without seed data” result is modest in absolute terms.** Figure 4 does show improvement, but the gain is much smaller than in the seed-data setting. This is not a flaw in itself, but it suggests the claim should be framed cautiously.

## Nice-to-Haves
- Report **held-out label agreement** between SPA-generated preferences and gold human preferences from UltraFeedback, including how this changes across iterations.
- Add a **human evaluation** or a held-out human-preference benchmark to substantiate the “human alignment” claim rather than only LLM-judge metrics.
- Include a **controlled full-data baseline** under the same codebase/training recipe as SPA/DPO to support the efficiency claim against 100% supervision.
- Validate the denoising mechanism more directly: e.g., precision/recall of flagged noisy pairs, or comparison to simpler confidence/margin baselines.
- Analyze **when to stop iterating**, given the slight degradation from iteration 2 to 3.
- Report **compute overhead** relative to plain DPO, since the method relies on multiple rounds of generation and fine-tuning.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper should compare to additional recent related work / missing baselines.”** Removed because the instruction forbids missing-related-work criticisms without external verification. The paper already includes meaningful baselines for the core judgment-method comparison (PairRM and LLM-as-judge), even if more would be nice.
- **“It should evaluate on safety/truthfulness/reasoning/math/coding benchmarks.”** Removed as a core weakness because this is largely scope creep. The paper’s stated evaluation focus is alignment benchmarks (AlpacaEval and MT-Bench), and MT-Bench already spans multiple capability categories. Additional task benchmarks would strengthen the paper but are not required to assess the central claim.
- **“The assumption of disjoint fresh prompt sets \(X_i\) per iteration is unrealistic.”** Removed as a substantive weakness. Section 4.1 clearly states this as part of the setup; the paper is proposing a framework under that data-availability assumption, not claiming to solve prompt acquisition itself.
- **“Baseline comparison is unfair because self-refinement is included for SPA but not for other judgment methods.”** Removed in the strong form. The paper explicitly frames Table 2 as comparing **preference judgment methods**, and says the baselines are the same iterative DPO pipeline with the judgment method changed and self-refinement removed. This is a reasonable controlled comparison for that specific question; the remaining valid criticism is simply that this does not isolate label quality directly.
- **Generic strengths such as ‘the topic is important,’ ‘the paper is well-written,’ or ‘the experiments are comprehensive.’** Removed per instruction as insufficiently specific.

## Novel Insights
The most interesting synthesis across the results is that **SPA’s empirical strength seems to come primarily from the judgment rule plus iterative data expansion, not from the proposed denoising story**. Table 6 makes this quite clear: data expansion alone already gets most of the gain, plain self-refinement adds almost nothing, and the extrapolation-based DND gives only a secondary boost. Combined with Table 7, this suggests the paper’s most credible contribution is not “noise-aware preference learning” per se, but rather the idea that a model’s **change relative to an SFT reference** provides a surprisingly strong self-judging signal for iterative preference expansion. That is a useful and more precise way to understand the paper than the broader alignment framing in the abstract.

## Suggestions
- Add a **held-out gold-label analysis**: for a subset of UltraFeedback pairs not used in training, compare Eq. 7, PairRM, and LLM-as-judge by accuracy/calibration against human labels.
- Replace or soften the abstract claim about outperforming “the cases using the entire data” unless you provide a **same-pipeline 100% label baseline**.
- Reframe the evaluation claims more precisely as **improving judge-based alignment benchmarks**, unless direct human-preference evidence is added.
- Strengthen Section 4.2 with **direct validation of DND** rather than only downstream metrics.
- Clarify the “no seed” result as a **special-case extension from an already instruct-tuned starting point**, not a general no-seed guarantee.
- Add uncertainty estimates or repeated runs for the main results, especially given the variance seen in Table 4.

## Score and Decision
**Novelty:** Moderate. Iterative self-training for alignment is not new, but using the policy/reference DPO-style implicit reward as the self-judging signal is a neat and practically meaningful twist.  
**Technical soundness:** Mixed. The core method is coherent and the main empirical effect appears real, but several mechanistic claims are stronger than what the experiments validate.  
**Empirical support:** Moderate. The gains over the main low-data baseline are substantial, but the evidence is overly concentrated on GPT-4-judge benchmarks and lacks direct validation of label quality or human alignment.  
**Significance:** Moderate-to-high if the core effect holds broadly, because reducing human preference labels is genuinely important; however, the unsupported headline claims reduce confidence.  
**Clarity:** Good overall. The method is described clearly, the main equations are understandable, and Tables 3/6/7 are especially informative.

Overall, this is a **promising and likely practically useful paper**, but it currently **overclaims relative to its evidence**. The main benchmark gains make it stronger than a borderline incremental submission, yet the lack of direct human-preference validation and the unsupported “beats full-data training” framing keep it below the bar for acceptance at ICLR in its current form.

**Score: 5.9**

MY FINAL SCORE: <pineapple>5.9</pineapple>
MY FINAL DECISION: <orange>Reject</orange>