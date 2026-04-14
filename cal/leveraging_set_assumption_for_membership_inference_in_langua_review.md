=== CALIBRATION EXAMPLE 38 ===

# Final Consolidated Review
## Summary

Set-MI is a method that improves membership inference (MI) in language models by leveraging the observation that documents sharing metadata attributes (e.g., creation date, programming license, source dataset) are typically either all included or all excluded from a model's training corpus. Rather than scoring individual documents, Set-MI aggregates Individual-MI scores across such "sets," amplifying weak signals into reliable membership predictions. The paper introduces five new set-based MI benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and demonstrates a 0.14 average AUROC improvement over Individual-MI baselines across four methods and multiple target LMs.

---

## Strengths

- **Targeted, practically useful benchmark construction across five structurally distinct domains.** Unlike prior MI benchmarks that are single-domain, the five domains here cover different mechanisms of inclusion (temporal cutoffs, licensing criteria, dataset mixing, language selection), each grounded in real auditing scenarios. This specificity lets the reader map findings to concrete use cases.

- **Mechanistically informative deduplication analysis.** Section 5.3 and Figure 3 (right) reveal that Set-MI exploits *correlated* memorization signals that deduplication specifically destroys, while Individual-MI is much less affected. This is not merely an engineering finding — it sheds light on the memorization structure that deduplication disrupts, contributing empirical insight beyond the method itself.

- **Principled robustness study with actionable (if partial) guidance.** The three noise scenarios in Section 6 — member-set noise, non-member-set noise, and mixed noise — are well-motivated, and the comparison of MAX/MIN/FULL aggregation under each provides concrete (if still incomplete) guidance for practitioners dealing with imperfect metadata.

- **Quantified signal-amplification relationship.** The correlation r = 0.824 (p = 0.0002) between Individual-MI AUROC and Set-MI AUROC, reported in Section 5.1, is a specific and informative result: Set-MI is a signal amplifier, not an independent signal source, which sets clear expectations for when it helps and when it hurts.

- **Improvements are often large and domain-spanning, not marginal.** Several gains are dramatic (e.g., Loss Attack on Arxiv: 0.576 → 0.938; LiRA on Wikipedia: 0.581 → 0.859; Min-K% on Arxiv: 0.590 → 0.954), demonstrating that the set aggregation is genuinely exploiting structure rather than just smoothing noise.

---

## Weaknesses

### Fatal
None.

### Major

- **Ground-truth membership via date proxies undermines the primary evaluation.** For Wikipedia and Arxiv — two of the five benchmarks and the ones used for most ablations — ground truth is assigned based on creation date relative to the Pile's data collection cutoff, not verified membership. Since the Pile applies its own filtering pipeline, not every pre-cutoff article is necessarily included. The paper demonstrates a stronger verification method exists (13-gram overlap against the Pile, Section 6), but applies it only in the robustness analysis. The gap between date-proxy labels and true membership is never quantified for the main benchmarks. If a non-trivial fraction of "members" are actually absent from the Pile due to filtering, the reported AUROC numbers are measuring something different from what is claimed — this could inflate or deflate apparent gains depending on the correlation between filtering and MI score.

- **Deduplication severely limits practical scope, inadequately acknowledged.** Section 5.3 clearly shows that Set-MI's advantage nearly vanishes on deduplicated training data, while Individual-MI is comparatively unaffected. However, the paper treats this as an analysis result rather than a central limitation. The vast majority of modern LLMs of practical interest for copyright auditing and contamination detection (Llama 2/3, Mistral, GPT-4-class) train on deduplicated data. This severely constrains where Set-MI will provide the large gains shown in Table 2. The paper's conclusion mentions only the metadata availability requirement as a limitation; the deduplication constraint deserves equal prominence.

### Minor

- **Single target model for Languages (BLOOM-7B) and Instructions (Tulu-v1).** With one model per benchmark, results may be model-specific artifacts rather than general properties. For instance, BLOOM-7B's multilingual training is distinctive, and Tulu-v1's instruction mixture is specific. At least one additional model per benchmark would substantiate generalizability.

- **Robustness analysis confined to Wikipedia + Loss Attack.** Section 6 evaluates noise robustness only on one domain and one MI method. LiRA produces ratio-normalized scores with different distributional properties than raw loss values, so the MAX/MIN/FULL recommendations derived from Loss Attack may not transfer. The claim that all three aggregations "significantly outperform Individual-MI in all three settings" is only verified for one method.

- **No principled guidance for choosing aggregation method in practice.** The paper recommends MAX when member sets are noisy, MIN when non-member sets are noisy, and FULL when both are noisy — but in realistic deployment, the user does not know which noise regime applies. No diagnostic procedure is offered for selecting the aggregation method from data. This is not fatal to the method, but the recommendation as written is circular.

- **Below-chance failure case (zlib entropy on Instructions, 0.458 → 0.429) is insufficiently analyzed.** The paper notes this degradation in one sentence and attributes it to below-chance Individual-MI scores. However, there is no analysis of *why* zlib entropy performs below chance on this domain (possibly instruction-format text has atypical compression ratios), nor any guidance for detecting this failure mode before applying Set-MI. Given that zlib also performs below chance on Instructions Individual-MI (0.458 < 0.5), a user could inadvertently worsen their results.

- **LiRA is absent from the model-size scaling analysis (Figure 3 left).** LiRA is the best-performing Individual-MI method overall (Table 2 average: 0.835 Set-MI AUROC vs. 0.799 for Loss Attack). Its exclusion from the scaling plot is unexplained and leaves unclear whether LiRA's advantage over Loss Attack grows, shrinks, or is constant with model size — a question of direct practical interest.

### Tiny

- **Inconsistency in Section 4 set counts.** The text for Wikipedia states "We subsample 100 sets with 100 documents per set" (10,000 documents), but Table 1 reports 1,000 sets and 100,000 documents. The same discrepancy appears for Arxiv. The relationship between the described subsampling and the table statistics is not explained; this appears to be a typo or omitted explanation (likely "1,000 sets" not "100 sets").

- **Default of 1,024 tokens per document is not justified relative to the ablation.** Figure 4 (left) shows Set-MI performance saturates near 256 tokens; the paper uses 1,024 tokens as its default throughout without explanation of why a 4× longer sequence is preferred given diminishing returns.

- **Token-length and set-size ablations are Wikipedia-only.** The "n = 3 is sufficient" claim (Section 5.5) and the performance saturation near 256 tokens (Section 5.4) are only validated on Wikipedia. Whether these hold for short-document domains like Instructions is unverified.

---

## Nice-to-Haves

- **Apply 13-gram overlap verification to the main benchmarks.** Quantify what fraction of date-labeled "members" are actually present in the Pile, and report how this affects AUROC estimates. This would validate (or motivate revision of) the primary evaluation setup.

- **Quantify set assumption validity rates and correlate with AUROC gains.** Measuring what fraction of sets in each benchmark satisfy the assumption (via n-gram matching where the training set is accessible) and correlating this with observed AUROC improvement would validate the theoretical motivation more directly.

- **Blind cutoff detection as an end-to-end demonstration.** Evaluate whether Set-MI can *discover* the training cutoff date on a model where the cutoff is not publicly known (rather than constructing benchmarks around known cutoffs). This would demonstrate practical utility beyond controlled evaluation.

- **Explore approximate set discovery via document clustering.** Even in the absence of explicit metadata, embedding-based clustering might recover approximate sets. A brief pilot experiment or discussion would address the metadata dependency concern.

- **Quantify query overhead relative to Individual-MI.** Set-MI requires querying the model for all documents in a set before scoring any of them. For users with API access constraints, the cost-benefit tradeoff is relevant, particularly when sets are large (100 documents).

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"Single-epoch training stated as fact without citation"** (Harsh Critic): The paper explicitly says "We *hypothesize* that the loss of individual documents often does not provide a strong enough signal…because LMs are often trained on trillions of tokens and only see most of their training documents only once." The hypothesis framing is present; the criticism that this is stated as fact is a misreading.

- **Statistical significance tests required for all results** (Harsh Critic): Single-run AUROC evaluation without bootstrap confidence intervals is the norm in MI literature for LLMs. Requiring significance testing for every reported AUROC difference would be a non-standard methodological demand for this research community.

- **Evaluation on closed-source LLMs required** (Harsh Critic): The method requires access to token-level log-probabilities. Closed-source models that do not expose this cannot be evaluated without perplexity-API workarounds. Demanding evaluation on GPT-4 or Llama-3 is scope creep given the explicit black-box logit assumption in Section 2.1.

- **ShareGPT in Instructions benchmark creates indirect training data contamination** (Harsh Critic): The concern that ShareGPT outputs may appear in other models' pretraining data is speculative for Tulu-v1 specifically. The paper uses Tulu-v1's known training composition for ground truth; this is a legitimate evaluation choice.

- **Learned aggregation functions as a weakness** (Positive-leaning Reviewer): The simplicity of mean/max/min aggregation is appropriate for a method that must be applicable without training data. Requiring a learned aggregator is an aspirational methodological extension, not a flaw.

- **Computational overhead as a weakness** (Positive-leaning Reviewer): Set-MI's cost scales linearly with set size and is identical in per-document cost to Individual-MI; the overhead is a deployment consideration, not a fundamental weakness of the approach. Moved to Nice-to-Haves.

---

## Novel Insights

The robustness analysis in Section 6 surfaces a genuinely underappreciated asymmetry: MAX aggregation is robust to *member*-set noise (because noisy non-members in a member set have low scores, and MAX ignores them), while MIN is robust to *non-member*-set noise (because noisy members in a non-member set have high scores, and MIN ignores them). This asymmetry has a natural probabilistic interpretation — robust aggregation under one-sided contamination selectively discards the contaminating tail. While the paper presents this as an empirical finding, it points toward a principled adaptive strategy: if one can estimate the dominant noise direction (e.g., via a small labeled validation set), one can select the theoretically appropriate aggregation. The authors do not pursue this, but it is a concrete and actionable insight for follow-up work. The finding (Section 5.3) that deduplication disrupts Set-MI more than Individual-MI also implies a converse: **the magnitude of Set-MI's advantage over Individual-MI is itself a diagnostic of how much correlated memorization exists in the target model's training data** — a potential side-channel for detecting deduplication status of unknown models.

---

## Suggestions

- Apply 13-gram overlap verification to the Wikipedia and Arxiv primary benchmarks and report the fraction of date-labeled members that are actually present in the Pile; revise AUROC estimates or add error bars accordingly.
- Add a dedicated paragraph in the limitations section explicitly stating that Set-MI's advantage is substantially reduced on deduplicated training data, and that most state-of-the-art open LLMs (Llama 2/3, Mistral) and closed LLMs use deduplication.
- Explain the absence of LiRA from Figure 3 (left) or add it to the scaling analysis; it is the strongest baseline and its scaling behavior is of direct interest.
- Resolve the apparent set-count discrepancy in Section 4 (text says "100 sets" for Wikipedia/Arxiv, Table 1 says 1,000 sets) and explain how date groups map to the reported set counts.
- Provide one additional target model for the Languages and Instructions benchmarks to support generalizability claims.
- Include a short discussion or heuristic for detecting when Individual-MI is operating below chance, so users can avoid applying Set-MI in settings where it will hurt (as with zlib entropy on Instructions).
- Add the Languages and License robustness analysis (or at minimum Loss Attack on Arxiv) to Section 6, to validate that the MAX/MIN/FULL recommendations transfer beyond Wikipedia.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0]
Average score: 5.0
Binary outcome: Reject
