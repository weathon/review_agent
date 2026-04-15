Now I have enough to write the final review. Let me synthesize the findings.

---

## Summary

This paper introduces **Set-MI**, a method that enhances existing Individual-MI techniques for language model membership inference by aggregating individual document-level MI scores within groups of documents that are presumed to share the same membership status (either all in or all out of the training corpus). The key "set assumption" is motivated by how LM training datasets are curated — by date, language, license, or instruction source — meaning entire metadata-defined groups are included or excluded as a unit. The authors construct five benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and report average AUROC improvements of +0.14 over four Individual-MI baselines, along with ablations on model size, deduplication, document length, set size, and noise robustness.

---

## Claims and Support

**Claim 1: Set-MI substantially improves membership inference over Individual-MI.**
*Partially supported.* Table 2 shows large AUROC gains on the five constructed benchmarks. However, several benchmarks (Languages especially; Arxiv, Wikipedia partially) are susceptible to distributional confounds: membership labels are determined by the same metadata attribute (date, language, license) used to define the sets. This does not invalidate the findings entirely — for Wikipedia/Arxiv the Individual-MI baseline hovers near 0.52–0.59 (barely above chance), suggesting temporal features alone do not drive performance at the individual level — but the Language benchmark (baseline 0.673–0.908) appears to capture language-level distribution shift rather than genuine memorization signal.

**Claim 2: The set assumption is practical and robust.**
*Partially supported.* Sec. 3 gives plausible real-world examples (DOLMA, Arxiv cutoffs), and Section 6 performs a controlled noise injection experiment. However, the noise model (random swapping of opposite-membership documents) is stylized and does not reflect systematic real-world violations (e.g., quality filtering, deduplication that preferentially removes certain document types). Empirical measurement of actual set purity on real training corpora is absent.

**Claim 3: Five benchmarks constitute a diverse evaluation suite.**
*Partially supported.* The domains are varied, but — especially for Languages, License, and Instructions — the grouping variable and the membership label source are nearly identical by construction, which makes these benchmarks more favorable to the method than an independent evaluation would be.

**Claim 4: Set-MI exploits stronger memorization in larger models.**
*Weakly supported.* Figure 3 shows correlation between model scale and Set-MI performance. The paper interprets this as memorization driving the effect, but no direct memorization measurement is provided. The phrasing is observational.

**Claim 5: Deduplication hurts Set-MI more than Individual-MI.**
*Partially supported.* Figure 3 (right) shows the trend only for Loss Attack on Wikipedia Pythia models, and the effect disappears at smaller scales. Scope is too narrow for a general claim.

**Claims 6–7 (document length, set size, robustness):** *Well-supported within the restricted experimental scope shown.*

---

## Strengths

- **Practical motivation grounded in real dataset curation practices.** The insight that LM training data is curated by discrete inclusion criteria (temporal cutoffs, language selection, license categories) means the set assumption has genuine structural support, not just theoretical appeal. The DOLMA/Arxiv examples are concrete and verifiable.

- **Orthogonality and modularity.** Set-MI is a wrapper around any Individual-MI scorer, demonstrated on four qualitatively different baselines (Loss, LiRA, Min-K% Prob, zlib). This makes it immediately adoptable by practitioners using any existing MI pipeline.

- **Actionable ablation findings.** The analyses of model scale (Sec. 5.2), deduplication (Sec. 5.3), token length (Sec. 5.4), and set size (Sec. 5.5) yield concrete practical guidance: use more documents per set, longer samples, and prefer larger non-deduplicated models where possible. These are specific and non-trivial.

- **Principled noise robustness analysis.** Section 6's controlled noise injection (varying noise type and rate separately for member/non-member sets) and the MAX/MIN/FULL recommendation based on noise type is a meaningful attempt to provide user-facing practical guidance, which is unusual in MI papers.

---

## Weaknesses

### Fatal
*(none triggered)*

### Major

**1. Temporal and distributional confounds in key benchmarks — AUROC improvements may conflate distribution shift with memorization.**
For Wikipedia and Arxiv, member sets are documents created *before* the Pile's crawl cutoff; non-member sets are documents created *after*. Newer documents differ stylistically and topically from older ones in ways completely independent of model memorization. For Languages (Bloom-7B), membership is determined by whether the language is in the training set — a model trained without Japanese will have high loss on Japanese regardless of memorization. This introduces a distribution-level confound: a purely statistical baseline with no model access could potentially perform comparably to Set-MI on these benchmarks. The paper does not include a blind baseline (e.g., a date-classifier or language-classifier applied without querying the model) to rule this out. The MI literature has shown that temporal-split benchmarks systematically conflate distribution shift with true memorization signal, and this paper does not address this. The Language benchmark is particularly suspect because the Individual-MI baselines already reach AUROC of 0.673–0.908, which is far above chance without aggregation — suggesting a strong distributional signal unrelated to memorization.

**2. Exclusive use of AUROC; TPR at low FPR is absent.**
The paper reports AUROC as the sole evaluation metric throughout. The MI community has established since Carlini et al. (2022) that AUROC can be misleading for MI evaluation, and that TPR@low FPR (e.g., TPR@0.1%FPR or @1%FPR) is the more meaningful practical metric — particularly for applications like copyright infringement detection or contamination auditing, where false accusations carry significant costs. Without TPR@low FPR, it is unclear whether the AUROC improvements translate into practically meaningful detection capabilities, or whether they primarily reflect better discrimination across the full ROC curve at operating points that are practically irrelevant.

### Minor

**3. Robustness analysis confined to a single scenario.**
Section 6 evaluates noise robustness only on Wikipedia, with the deduplicated Pythia 2.8B model, using only Loss Attack. The synthetic noise model (random swap of opposite-membership documents) represents idealized random corruption. Real-world violations — systematic filtering that removes low-quality documents from certain date batches, deduplication that preferentially removes near-duplicate documents present in both member and non-member date buckets — would have structured, non-random effects on aggregation. The scope is too narrow to support "robust under practical settings" as a general claim; the paper should narrow its robustness claim or expand the experimental scope.

**4. Instructions+zlib failure and overly broad "any method" claim.**
Sec. 5.1 states Set-MI "can improve any Individual-MI method," but Table 2 shows Instructions+zlib degrades (0.458→0.429). The paper acknowledges this in passing but does not analyze the failure mode. Without understanding when Set-MI hurts performance, practitioners cannot reliably deploy the method across arbitrary settings.

**5. No statistical significance or variance reporting.**
Table 2 contains no error bars, confidence intervals, or significance tests. Some improvements are modest (e.g., Wikipedia Loss Attack: 0.524→0.575), and without variance estimates it is impossible to determine whether these reflect reliable gains or run-to-run noise.

**6. Limited model scale.**
All target models cap at 12B parameters (Pythia), with most experiments on sub-7B models. The paper's practical framing centers on LLMs, but the experiments do not include any model representative of the current generation (LLaMA-class, Mistral, Gemma, etc.). It is unknown whether the trends observed — particularly the model-size scaling of Set-MI benefits — continue into the 30B+ range.

### Trivial

**7. Internal benchmark size inconsistency.**
Table 1 lists Wikipedia and Arxiv as "1,000 sets / 100,000 documents," but Sec. 4 states "we subsample 100 sets with 100 documents per set." This is a factor-of-10 discrepancy that is likely a typo/editing artifact but should be fixed for clarity.

---

## Nice-to-Haves

- **Include a blind distributional baseline** (e.g., a date-heuristic or language-ID test not requiring model access) to bound how much performance comes from distribution shift vs. memorization.
- **Report TPR@1%FPR or precision-recall curves** alongside AUROC in all main experiments.
- **Explore weighted aggregation** (inverse-variance weighting, rank-based) beyond mean/max/min.
- **Visualize per-set MI score distributions** (member vs. non-member sets) to clarify whether aggregation works by shifting the mean or by leveraging a few high-signal outliers.
- **Discuss how to construct sets when metadata is unavailable**, e.g., via document embedding clustering or temporal proximity, with sensitivity analysis on granularity (day vs. week vs. month groupings).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic Issue 2 (document-level AUROC inflates sample size):** When all documents in a set receive the same aggregated score AND all documents in a set share the same membership label (which is the set assumption), document-level AUROC is mathematically equivalent to set-level AUROC — repeating identical (score, label) pairs does not change the ROC curve. The criticism that this "inflates effective sample size" and constitutes "the wrong primary unit of analysis" is formally incorrect under the paper's own assumptions. *Removed as factually wrong.*

- **Harsh Critic Issue about benchmark size being the first/most diverse being unsupported:** The paper states these are the first set-based MI benchmarks; no reviewer provides evidence otherwise. *Removed per hard rule on citing missing related works.*

- **Neutral Reviewer Weakness 4 (temporal granularity sensitivity):** Requesting a specific sensitivity analysis on hourly vs. daily set definitions exceeds the paper's stated scope and is a methodological extension, not a core flaw.

- **Spark Weakness (evaluation on truly opaque models like GPT-4/Claude):** The paper explicitly targets settings where training data metadata is available. Evaluating on fully opaque closed models where the set assumption cannot even be formulated is outside the stated scope. *Removed as scope creep.*

- **Harsh Critic causal language on deduplication/memorization mechanism:** The concern about overclaiming a causal mechanism is valid, but the actual text says "this *suggests* that Set-MI also becomes less effective in exploiting models' memorization" — this is appropriately hedged with suggestive rather than causal language. *Removed as misreading.*

---

## Novel Insights

The most genuinely novel observation in this paper is the reframing of membership inference from a single-document task to a group/batch task, grounded in the structural fact that LM training datasets are assembled via discrete inclusion criteria rather than document-by-document selection. This reframing shifts the difficulty of the problem: rather than detecting weak per-document memorization signals, Set-MI amplifies group-level inclusion signals that, by construction, are shared across many documents. The finding that deduplication suppresses Set-MI more than Individual-MI (despite similar effects at small scales) is also insightful — it suggests that memorization-driven signal is what Set-MI exploits, and deduplication specifically reduces that signal by eliminating the repetition that drives memorization of specific content.

---

## Suggestions

1. **Run a no-model blind baseline** (language-ID, date-classifier, or embedding cosine similarity to a reference document) on each benchmark and report results alongside Set-MI in Table 2 to disentangle distributional confounds from genuine memorization signal.
2. **Add TPR@1%FPR** as a mandatory secondary metric in Table 2 and all ablation figures.
3. **Expand the robustness study (Sec. 6)** to cover at least two base MI methods and one additional benchmark (e.g., License or Instructions), and consider a structured noise model (removing low-quality documents within date batches) in addition to random swapping.
4. **Narrow or qualify the "set assumption is practical" claim** in the abstract/intro/conclusion by explicitly acknowledging that Language and possibly License benchmarks are partially driven by distributional rather than memorization signal.
5. **Fix the Table 1 vs. Sec. 4 inconsistency** on benchmark sizes.
6. **Report variance** (across models where averages are used) and include a p-value for the claimed correlation of 0.824.

---

## Score and Decision

**Originality:** Moderate. The core technique (averaging scores within groups) is simple, but the formalization of the "set assumption" as a structural property of LM pretraining data curation is a meaningful conceptual contribution.

**Importance of research question:** High. Membership inference for LLMs is a critical unsolved problem for copyright, contamination, and data transparency.

**Claim support:** Moderate. The main empirical claims are directionally supported, but distributional confounds (especially for Languages) and the absence of TPR@low FPR evaluation leave significant uncertainty about the practical value of the gains.

**Soundness of experiments:** Moderate. Diverse benchmarks, but the noise robustness study is narrow, the temporal-shift confound is unaddressed, and no significance testing is performed.

**Clarity of writing:** Good. The paper is clearly structured and the method is easy to understand.

**Value to community:** Moderate-High. The method is plug-and-play and the empirical findings (scale, deduplication, set size effects) are practically useful, contingent on the confound concern being addressed.

The paper makes a reasonable contribution with a clear practical angle, but the unaddressed temporal distribution shift confound (particularly problematic for the Language benchmark) and the absence of the now-standard TPR@low FPR metric are genuine gaps that prevent confident acceptance. The paper sits at the borderline.

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>