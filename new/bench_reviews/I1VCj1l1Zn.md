Now I have sufficient information from the paper to write the consolidated review. Let me verify the key numerical claims before writing.

**Verification notes from reading the paper:**

1. **"92.34% accuracy" inconsistency claim (Human Finder):** The four backbone averages from Table 1 are 89.89%, 92.89%, 90.65%, 95.93%. Their mean = (89.89+92.89+90.65+95.93)/4 = **92.34%**. The claim is *correct*. The Human Finder reviewer was wrong.

2. **"1.24x slower than single LoRA" inconsistency (Spark reviewer):** Section 3.3 claims 1.24x; Section 4.2 claims 18.19% average increase. From Table 4 subscripts (computed directly as DLP/LoRA ratio): -2.61%, +12%, +52.38%, +11% → average = 18.19%, or 1.18x. The 1.24x in Section 3.3 appears to be the average DLP-LoRA ratio over the *basic* backbone ((1.12+1.12+1.60+1.11)/4 = 1.2375), not over single LoRA. There is a minor internal inconsistency.

3. **TopChat regression (LLaMA-3 8B):** Table 2 confirms: LoRA BLEU=35.95 → DLP-LoRA BLEU=29.58, ROUGE-L 33.52 → 27.80. Confirmed regression, not discussed.

4. **Fusion ablation:** Paper never compares top-p fusion vs. top-1 dispatch. Confirmed absence.

5. **Composite baselines:** Table 3 only compares DLP-LoRA (26 specialized LoRAs + router) against basic backbone and a single r=64 LoRA on composite-26. No oracle routing or top-1 dispatch baseline. Confirmed.

6. **Table 7 (dynamic LoRA comparison):** Only reports decoding latency and memory — no accuracy or BLEU/ROUGE metrics. And is conducted on a **different** 7-task ShareGPT setup, not the main 26-task benchmark. Confirmed.

---

## Summary

DLP-LoRA proposes a 5M-parameter mini-MLP classifier that, at sentence boundaries, selects and fuses multiple task-specific LoRA adapters using top-*p* sampling, replacing per-token MoE routing with a single sentence-level decision. The method is evaluated on 26 tasks (17 MCQ + 9 QA) across four LLM backbones, claiming near-single-LoRA task performance at approximately 1.18× the inference cost of a single LoRA, and large gains over unadapted baselines in composite multi-task settings. The paper frames its core novelty as **dynamic LoRA fusion**, but the experiments do not isolate this from simpler dispatch strategies.

---

## Claims and Support

| Claim | Support |
|---|---|
| DLP-LoRA achieves 92.34% average MCQ accuracy | ✅ Confirmed: mean of four backbone averages from Table 1 = 92.34% |
| Sentence-level routing is sufficient (tokens in same sentence share task) | ⚠️ Assumed from prior work; not validated on this paper's 26-task setup |
| DLP-LoRA ≈ single-task LoRA performance | ✅ Well-supported; average relative differences –0.94% to –0.00% (MCQ), ±~1% (QA) |
| DLP-LoRA significantly improves over composite-task baselines | ⚠️ Partially supported; gains are real but baselines are weak (single r=64 LoRA vs. 26 specialized LoRAs + router) |
| Inference cost <2× single LoRA | ✅ Confirmed at 1.18× on average (18.19%); minor internal inconsistency with "1.24×" in Section 3.3 (which is ratio vs. basic backbone, not LoRA) |
| Dynamic *fusion* drives the gains over simple dispatch | ❌ No ablation against top-1 dispatch; impossible to determine from results |
| DLP-LoRA outperforms token-level MoE in efficiency | ✅ Table 7 supports, but on a *different* 7-task setup, not the main 26-task benchmark |
| Smaller adapted model (Qwen-2 1.5B + DLP-LoRA) outperforms larger unadapted model (LLaMA-2 13B) | ✅ Numerically confirmed in Table 5, but the comparison is unadapted vs. adapted—informative as a practical data point, not a method comparison |

---

## Strengths

- **Concrete and reproducible efficiency gains at scale:** Table 4 shows mini-MLP DLP-LoRA averages only 18.19% overhead over single LoRA, and Table 6 demonstrates the ratio stays below 2× even at 100 LoRAs (1.83×) while LoRA parameters remain <0.1% of LLaMA-3 8B. This is a verifiable, practically useful result that distinguishes the paper from prior token-level MoE methods.

- **Broad empirical validation maintaining near-single-LoRA quality:** Tables 1–2 consistently show average relative performance within –1% of individually trained single LoRAs across 26 diverse tasks and four LLM backbones (1.5B–8B parameters). This breadth of validation, without cherry-picking, is stronger than typical narrow evaluations in this space.

- **Lightweight and modular plugin design:** The 5M-parameter mini-MLP trains in under 10 minutes, achieves 98.45% task classification accuracy, and is architecturally independent of the LLM backbone—enabling task additions without modifying any trained LoRA modules. This practical property is clearly demonstrated.

---

## Weaknesses

### Fatal

*No single weakness rises to the "not even a paper" threshold; the system works and results are real.*

### Major

- **The central novelty—dynamic LoRA *fusion*—is never isolated from simple dispatch.** The paper's primary claim over prior work is top-*p* multi-LoRA *fusion*, yet no ablation compares it against (a) top-1 LoRA selection only, (b) oracle task-ID plus single LoRA, or (c) fixed-*k* selection. The paper never reports the distribution of how many LoRAs are actually selected per input under top-*p*, nor does it analyze whether inputs where more than one LoRA is chosen produce better outputs than single-LoRA dispatch. Without this, the paper establishes that *sentence-level task routing* is effective—but the "fusion" component that differentiates it from simpler dispatch remains unsubstantiated. This matters because the entire novelty framing rests on it.

- **Composite-task evaluation uses a weak, parameter-mismatched baseline.** Table 3 compares DLP-LoRA (26 specialized LoRAs + trained router) against a single r=64 LoRA trained on composite-26 data. These differ in parameter budget, specialization, and training regime simultaneously. Showing that 26 individually specialized adapters beat one shared adapter is expected and does not validate the routing or fusion mechanism. Missing baselines: (1) sentence classifier + top-1 LoRA selection, (2) oracle task-label + corresponding single LoRA, (3) other dynamic dispatch methods on the same 26-task mixture. Without these, the composite-task claim—which is the paper's main practical selling point—is built on weak evidence.

- **No task-performance comparison with other dynamic LoRA methods.** Table 7 compares only inference speed and memory vs. MOLA, PESC, MoRAL, LoRA-Switch—and does so on a *different* 7-task ShareGPT setup, not the main 26-task benchmark. There is no comparison of accuracy or BLEU/ROUGE against any competing dynamic-fusion method. This makes it impossible to evaluate whether DLP-LoRA's efficiency gains come at a task-quality cost relative to state-of-the-art dynamic alternatives.

### Minor

- **Substantial unanalyzed per-task regressions.** Table 2 shows DLP-LoRA drops vs. single LoRA on TopChat (LLaMA-3 8B: BLEU 35.95 → 29.58, ROUGE-L 33.52 → 27.80) and on CNNDM (Qwen-2 7B: BLEU 16.07 → 14.17; LLaMA-2 7B: BLEU 8.02 → 14.31 is actually an improvement, but other cells degrade). These are large regressions on specific tasks—potentially due to LoRA interference during fusion or misclassification—but the paper offers no discussion. Understanding failure modes is important for a method claiming practical deployment viability.

- **Internal inconsistency in reported inference overhead.** Section 3.3 states DLP-LoRA is "on average only 1.24 times slower than single LoRA inference," but Section 4.2 reports 18.19% average increase (≈1.18×). The 1.24 figure is actually the mean ratio over the *basic* backbone, not over single LoRA (since some backbone/LoRA ratios are 1.00×). This conflation overstates the overhead and should be corrected.

- **Table 5's cross-model comparison is framed too strongly.** Comparing Qwen-2 1.5B + DLP-LoRA against *unadapted* LLaMA-2 13B is a practical data point, but not a method comparison. The section heading "Can a Smaller LLM with DLP-LoRA Outperform a Larger LLM Backbone?" implies a general capability claim; the evidence only shows an adapted small model beats an unadapted large model on in-distribution tasks—unsurprising given that single-task LoRA already dramatically improves over the basic backbone.

- **Top-*p* threshold is a core hyperparameter with no sensitivity analysis.** The paper's central mechanism relies on a top-*p* threshold that determines how many LoRAs are fused, yet the chosen value is never stated, and no sweep over *p* ∈ {0.5, 0.7, 0.9, 0.95, 1.0} is provided. Readers cannot assess whether the reported performance is stable or brittle.

### Trivial

- **Minor numerical inconsistency in the "1.24×" claim** (see Minor section above). Easy to fix with a single sentence.

---

## Nice-to-Haves

- Provide a distribution plot of how many LoRAs are selected per input under the chosen top-*p* threshold, to show whether fusion produces variable-cardinality selection in practice.
- Provide a failure-case study (companion to Figure 3) where the classifier misroutes and analyze whether performance degrades gracefully.
- Evaluate robustness when input does not belong to any trained task (out-of-distribution / new task at inference time).
- Add ablation comparing top-*p* fusion with top-1 selection to establish the empirical value of multi-LoRA fusion specifically.
- Report memory consumption at 50/100 LoRAs alongside inference-time ratios in Table 6.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Human Finder (W4): "92.34% accuracy inconsistency."** The reviewer claims the 92.34% figure does not correspond to any backbone's average. This is **factually incorrect**: (89.89 + 92.89 + 90.65 + 95.93) / 4 = 92.34 exactly. Removed.

- **Human Finder (W5): "Unfair comparison in Table 5 is a straw-man."** This was retained as a Minor weakness above (Table 5 framing is overstated), but the claim that it constitutes a methodological flaw is weakened: the paper explicitly frames this as "can a smaller adapted model outperform a larger unadapted model," which is a legitimate practical question. The comparison is informative, just not a strong *method* comparison.

- **Neutral/Harsh Reviewer: "Missing related works" / requests to add more related work discussion.** Per the hard rules, removed.

- **Harsh Reviewer: Reproducibility concerns (undisclosed hyperparameters, training details, single GTX 2080Ti).** These are nitpicks about implementation details not material to the scientific claims; removed.

- **Neutral Reviewer (W6) and Harsh Reviewer: Lack of OOD evaluation (new tasks not in training set).** Kept as Nice-to-Have only—this is outside the paper's stated scope and is not a standard evaluation in this literature for a systems-efficiency contribution.

- **Harsh Reviewer: "Equation (6) does not clearly specify the fused forward computation."** Equation (6) is dense but readable as a batched weighted sum; this is a presentation nitpick, not a methodological error. Removed.

- **Harsh Reviewer: "Section 2 background—generic LoRA merging property does not carry over."** This is minor conceptual background prose; not a substantive flaw in the methodology. Removed.

- **Spark Reviewer: "Comparison with a single LoRA of equivalent total parameter budget."** This is a reasonable baseline but demanding a parameter-matched single LoRA trained on composite-26 is beyond the standard in this literature, where per-task specialization vs. shared adapters is itself the research question. Softened to Nice-to-Have.

---

## Novel Insights

The reviewers, taken together, surface one genuinely important observation beyond what the paper acknowledges: **the method's empirical contribution cannot be cleanly attributed to dynamic fusion as opposed to sentence-level task identification and dispatch.** The paper reports that 98.45% of sentences are classified correctly, meaning most gains plausibly arise from accurate routing to the correct single LoRA, not from the multi-LoRA weighted blending. The top-*p* mechanism could in principle reduce to top-1 in practice if the classifier's probability mass concentrates on a single task (the case study in Figure 3 shows probabilities of 50.5% and 49.5%—a near-tie, not a clear multi-task signal). This distinction matters not just as an ablation gap but as a conceptual claim: if the gains are from routing accuracy, DLP-LoRA's contribution is an efficient sentence-level classifier for LoRA dispatch; if from fusion, the mechanism itself is novel. The paper does not differentiate these, and the reviewers collectively identify this as the most important open question.

---

## Suggestions

1. **Add a top-1 dispatch ablation as the primary missing experiment.** Run DLP-LoRA with exactly one LoRA selected (argmax routing, no fusion) vs. top-*p* fusion on Tables 1–3. This single experiment would either validate fusion as a genuine mechanism or reframe the contribution honestly as efficient dispatch.

2. **Fix the 1.24× vs. 1.18× inconsistency in Sections 3.3 and 4.2.** State clearly whether 1.24× is relative to the basic backbone or to single LoRA, and be consistent.

3. **Add one stronger composite-task baseline:** at minimum, "sentence classifier → top-1 matching LoRA" vs. DLP-LoRA's top-*p* fusion in Table 3. This would cost little and substantially strengthen or clarify the composite-task claim.

4. **Discuss and investigate the large TopChat / CNNDM regressions.** Even a brief analysis of whether these correlate with classifier errors or task-LoRA interference would improve scientific credibility.

5. **State the top-*p* value used and provide a brief two-row sensitivity table** (*p* = 0.9 vs. 0.95 vs. 1.0) to justify the design choice.

6. **Reframe Table 5** as a practical deployment comparison (adapted small vs. unadapted large), not a method comparison, and tone down the section heading accordingly.

---

## Score and Decision

**Originality:** Moderate. Sentence-level routing for LoRA dispatch is not new (Polytropon, Flix are acknowledged), but the specific combination of an off-the-shelf mini-MLP plugin, top-*p* adaptive fusion, and parallel GEMM acceleration is a practical refinement. The novelty claim around *fusion* specifically is unvalidated.

**Importance of research question:** High. Efficient multi-task LoRA deployment at inference time is a genuine practical problem as LoRA libraries grow.

**Claims well-supported:** Partially. The efficiency and near-single-LoRA quality claims are well-supported. The composite-task superiority and dynamic-fusion novelty claims are not.

**Soundness of experiments:** Moderate. Broad task and backbone coverage is a strength; absence of the key fusion ablation, weak composite baselines, and the different-setup efficiency comparison with dynamic baselines are real gaps.

**Clarity:** Good overall. The minor internal inconsistency in timing numbers is the main writing issue.

**Value to community:** Moderate-positive. The efficient dispatch idea is useful; the unvalidated fusion claim overstates novelty.

The paper is a functional, practical contribution with an honest efficiency story, but its central novelty—dynamic multi-LoRA *fusion*—is not empirically distinguished from the simpler and more modest contribution of efficient sentence-level task dispatch. Combined with weak composite-task baselines and the absence of performance comparisons against dynamic LoRA baselines, the experimental case is incomplete. A significant revision addressing the fusion vs. dispatch ablation and strengthening the composite baseline would substantially improve the paper.

**Score: 4.5**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>