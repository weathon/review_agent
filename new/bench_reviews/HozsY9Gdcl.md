Now let me read the calibration papers referenced by the human finder reviewer.Now I have enough information to write the final review. Let me synthesize all inputs.

---

## Summary

This paper introduces **Set-MI**, a method that augments existing membership inference (MI) methods for language models by leveraging a "set assumption": documents sharing certain metadata attributes (e.g., creation date, language, license) are expected to all be present or all absent from a model's training data. By aggregating individual MI scores over these document sets, Set-MI amplifies weak signals that individual MI methods struggle to detect. The authors construct five new benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and demonstrate an average AUROC improvement of 0.14 over four individual MI baselines. The paper also provides ablation studies on model size, deduplication, document length, set size, and noise robustness.

---

## Claims and Support

**Claim 1: Set-MI improves Individual-MI by 0.14 AUROC on average.**
- **Well-supported empirically.** Table 2 shows consistent gains across methods and domains. The average improvement is real. One notable failure: Instructions/zlib degrades (0.458→0.429), which is unaddressed.

**Claim 2: Set assumption is broadly applicable.**
- **Partially supported.** The paper constructs five plausible benchmarks, but set assumption validity is not verified per-benchmark. For Wikipedia/Arxiv, labels are date-based proxies; for Language/License/Instructions, labels are derived from model documentation. These are reasonable proxies but unverified.

**Claim 3: Set-MI is robust to imperfect set assumptions.**
- **Partially supported.** Section 6 tests robustness under *random* membership-label flipping noise on one dataset/model/method. Figure 5 shows all three aggregation variants outperform Individual-MI even under 50–90% noise. The claim is narrower than the wording implies.

**Claim 4: Set-MI benefits more from larger models than Individual-MI.**
- **Partially supported.** Figure 3 (left) shows a clear visual trend on Wikipedia/Pythia. The trend is plausible (larger models memorize more), but no variance is reported and no other dataset/family is tested. Claim is overstated relative to evidence.

**Claim 5: Deduplication makes Set-MI less effective.**
- **Partially supported.** Figure 3 (right) shows this for Wikipedia/Pythia/Loss Attack with model size >410M. The claim is a narrow observation presented as a general finding.

**Claim 6: Longer sequences and larger sets improve Set-MI.**
- **Well-supported** within the tested setup. Figures 4 are clear and the trends are intuitive and consistent.

**Claim 7: The five benchmarks constitute diverse, practical MI benchmarks.**
- **Partially supported.** The benchmarks are indeed diverse in domain. However, two of five (Wikipedia, Arxiv) use temporal cutoffs that introduce known distribution shifts between members and non-members.

---

## Strengths

- **Novel and practical insight.** The set assumption is intuitive, well-motivated by how LM training datasets are actually curated (inclusion criteria, date cutoffs, license filters), and produces genuine performance gains. The framing is both principled and actionable.

- **Consistent empirical gains.** AUROC improvements in Table 2 are large in several settings (e.g., Min-K%++ on Arxiv: 0.590→0.954; LiRA on Wikipedia: 0.581→0.859) and not attributable to cherry-picking since four different MI base methods all show improvements on average.

- **Plug-in compatibility.** Set-MI is model-agnostic and method-agnostic, wrapping any existing Individual-MI scorer. This makes it easy to adopt and future-proof.

- **Thorough ablation studies.** The paper examines model size, deduplication, token sample length, and set size — all practically relevant factors — and the results are internally consistent and informative.

- **Thoughtful robustness analysis.** Section 6 explicitly confronts the imperfect-set-assumption case, uses a more defensible 13-gram overlap labeling (rather than pure metadata proxies), and provides actionable recommendations for choosing MAX/MIN/FULL aggregation based on noise type. This is notably more careful than typical MI papers.

- **Five new benchmarks.** No existing MI benchmark covers set-level MI or the practical scenarios (license, language, instruction-following contamination) studied here.

---

## Weaknesses

### Fatal
*None.* The core contribution — set aggregation improves MI when set membership is shared — is sound and supported. The weaknesses below are real but do not invalidate the paper's primary finding.

### Major

- **Distribution shift confound in temporal benchmarks (Wikipedia, Arxiv)**: Wikipedia and Arxiv benchmarks partition members/non-members by creation date relative to training cutoffs, which is a well-documented source of distribution shift (Duan et al., 2024; Das et al., 2024). Set-MI aggregates document scores within date-defined sets, which means it potentially amplifies both genuine MI signal and distributional signals from temporal drift. The paper does **not** compare against "blind baselines" (models that don't query the target LM at all) — which prior work has shown can outperform standard MI attacks on temporal benchmarks. Without this control, it is impossible to determine how much of Set-MI's improvement on Wikipedia/Arxiv reflects genuine membership exploitation versus better distribution-shift detection. Notably, Language, License, and Instructions benchmarks do not have this confound and still show improvements, which partially mitigates the concern, but the headline benchmarks remain suspect.

- **AUROC as the sole metric**: The paper exclusively reports AUROC throughout (Table 2, Figures 3–5). This is known to be insufficient for evaluating MI methods (Carlini et al., 2021), because AUROC aggregates performance across all FPR thresholds, including impractical ones. For practical applications (copyright detection, contamination detection) the relevant operating point is at very low FPR (e.g., TPR@1% FPR). Without such metrics, it is unclear whether Set-MI's gains translate to operationally useful performance. For example, on Wikipedia/Loss Attack, AUROC improves from 0.524 to 0.575 — but whether this represents any meaningful TPR improvement at low FPR is unknown.

- **Unexplained failure case: Instructions/zlib degradation**: The zlib method on Instructions decreases from 0.458 to 0.429. This case is noted (Section 5.1 discusses the correlation between Individual-MI quality and Set-MI quality, noting that poor Individual-MI can cause degradation), but no deeper analysis is provided. For a paper claiming Set-MI is a reliable enhancement, understanding when and why it makes things worse is essential.

### Minor

- **Overly broad claims beyond the experimental evidence**: Claims about deduplication effects (Section 5.3), model size trends (Section 5.2), and noise robustness (Section 6) are presented as general findings but are each supported by a single dataset (Wikipedia), single model family (Pythia), and single or few base methods. The wording should be scoped to the tested settings.

- **Set-level vs. document-level AUROC mismatch**: Set-MI outputs one score per set, then copies it to all documents in the set. AUROC is computed over documents (treating identical predictions as independent). This inflates the effective sample count and the apparent statistical stability of results. Set-level AUROC (or set-level evaluation) should be reported as the primary metric, with document-level AUROC as a secondary view. The current presentation slightly overstates result reliability for small improvements.

- **No statistical significance testing**: Given that some improvements are modest (Wikipedia/Loss Attack: +0.051; Wikipedia/zlib: +0.047) and the evaluation involves random token subsampling, reporting these without any uncertainty estimate makes it hard to assess whether smaller gains are meaningful.

- **Limited to open models with known training data**: All target LMs (Pythia, GPT-Neo, Bloom, SILO, Tulu) have publicly documented training data that allows precise ground-truth labeling. The paper does not discuss how Set-MI would be applied to closed or proprietary models where training data composition is truly unknown, and where constructing valid sets would itself require inference.

### Trivial
- The noise model in Section 6 (random membership-label flipping) is a simplistic representation of real-world set-assumption violations. Quality-based filtering or systematic deduplication could create correlated noise patterns more damaging than random swaps. This is acknowledged in Section 3 but the robustness section doesn't fully address it.

---

## Nice-to-Haves

- **Blind baseline comparisons** on temporal benchmarks: A model-agnostic classifier (e.g., date-based heuristic or n-gram frequency detector that does not query the LM) would quantify how much of the gain is genuine MI signal vs. distribution shift.
- **TPR@low FPR metrics** (e.g., TPR@1% FPR, TPR@0.1% FPR): Essential for practical MI use cases and well-established as the appropriate metric in the field.
- **Score distribution plots (member vs. non-member sets)**: Histograms showing whether Set-MI genuinely shifts member/non-member score distributions apart, versus merely reducing within-class variance, would clarify the mechanism.
- **Multi-attribute set exploration**: Documents in practice have multiple metadata attributes (date, language, license). Exploring how to handle overlapping valid sets or combining signals across attribute types would strengthen practical utility claims.
- **Correlated noise simulation** in Section 6: Testing against quality-filter-like noise (e.g., systematically removing high-loss documents from member sets) would make robustness claims more realistic.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Set-MI is closer to set inclusion inference than standard MI — a structural flaw."** The paper explicitly reframes the MI problem as set-level inference. This is not a flaw; it is the paper's stated contribution. The evaluation is appropriate for the stated task. The insight that "aggregating within a group is almost tautologically advantageous given the benchmark definition" is overstated: the aggregation only helps when the base MI signals have appropriate direction, as demonstrated by the Instructions/zlib degradation and the poor performance on small models. **Removed** because this misunderstands the paper's intentional contribution framing.

- **Harsh Critic: "Benchmark labels are proxy labels, not verified membership — this undermines all main results."** Using metadata-based proxy labels is standard practice in the MI-for-LLMs literature (WikiMIA, MIMIR, and similar papers also use date-based or model-card-based labels). The harsh critic's demand for document-level 13-gram verification across all benchmarks is methodologically desirable but not required for a valid paper in this field. The distribution-shift confound (which IS a real issue) is kept as a Major weakness; the general objection to proxy labels is removed as it applies equally to all prior MI work. **Removed** as scope creep applying a standard not held to comparable papers.

- **Harsh Critic: "Several broad empirical claims generalized far beyond evidence is a structural problem that near-invalidates the contribution."** This is kept as a Minor weakness (overly broad claims), but removed from "Critical Issues" status. The evidence is narrow but the claims are about secondary analyses (model size, deduplication) that don't determine the paper's central contribution.

- **Neutral Reviewer: "Overlapping/multi-attribute sets are unaddressed."** The paper defines disjoint sets and is explicit about this. Multi-attribute set combination is a reasonable extension but outside the paper's stated scope. **Moved to Nice-to-Haves.**

- **Spark: "No evaluation on truly black-box closed models."** Kept as a minor weakness. Not removed, but the version demanding this as a central experiment is weakened since all MI papers in this subfield use models with known training data, as that is required for ground truth.

- **Human Finder: "Limited novelty of aggregation method."** The mathematical operation (averaging scores) is simple, but the insight (finding naturally occurring membership-structured document sets and exploiting them) is novel. AUROC gain of 0.14 on average, with gains up to 0.43 in some settings, demonstrates the idea's value. Simple methods that work well are contributions. **Removed** as it unfairly penalizes conceptual clarity.

---

## Novel Insights

The paper's most genuinely novel observation is methodological: by reframing MI as inference over document *groups* sharing an inclusion rule, one obtains large empirical gains with no change to the underlying scoring functions. This insight extends beyond the specific paper — it suggests that the reason individual-document MI on LLMs has historically been hard is not that the models lack membership signal per se, but that such signals are too noisy at the single-document level to exceed detection thresholds. Aggregation over 10–100 documents sharing true membership status dramatically amplifies the signal, even when each document's individual score is close to random. The robustness analysis in Section 6 further shows this aggregation benefit persists under substantial label noise (up to 50–70%), which means the approach is not fragile to imperfect set definitions. The practical framing — date-based cutoffs, license categories, instruction dataset identity — covers real-world audit scenarios not previously benchmarked.

---

## Suggestions

1. **Add blind baselines to Wikipedia and Arxiv experiments.** Implement a simple model-independent classifier (e.g., document date as feature, or a bag-of-words classifier trained on labeled data) to measure how much AUROC on temporal benchmarks is attributable to distribution shift rather than MI signal.

2. **Report TPR@1% FPR** in addition to or instead of AUROC as the primary metric in all main result tables.

3. **Provide deeper analysis of the Instructions/zlib failure case.** Show the score distributions for this setting and explain whether degradation occurs because zlib is systematically miscalibrated for instruction-following data, or because the set assumption is too noisy.

4. **Scope language for claims in Sections 5.2–5.3.** Explicitly state that model-size and deduplication findings are observations for Pythia/Wikipedia and have not been validated elsewhere.

5. **Report set-level AUROC as a primary metric** alongside document-level AUROC to make the evaluation semantics clearer and avoid sample-size inflation from within-set identical predictions.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Decision | Scores |
|---|---|---|---|
| BXMoS69LLR | Blind baselines beat MIAs (distribution shift critique) | Reject | 3,5,5,5 |
| EwYUgKr9Fc | Semantic MIA (similar distribution shift flaw, AUROC-only) | Reject | 3,3,5,3 |
| X8dzvdkQwO | Fine-tuning for pretraining data detection | Accept Poster | 6,8,6,5 |
| zWqr3MQuNs | WikiMIA + Min-K% (established benchmarks) | Accept Poster | 5,6,8,6 |
| 9QPH1YQCMn | Infilling Score (novel scoring function for MI) | Accept Poster | 6,8,8,3 |

**Positioning:**

The paper is substantively above the rejected papers (BXMoS69LLR, EwYUgKr9Fc). Those were rejected primarily because they were meta-critiques or because Set-MI's distribution-shift issue was *the entire point* of their arguments — this paper at least has non-temporal benchmarks (Languages, License, Instructions) that partially sidestep the concern and shows gains there too. The core mechanism (aggregation amplifies MI signal) is conceptually sound regardless of distribution shift.

The paper is comparable to the accepted poster papers (X8dzvdkQwO, zWqr3MQuNs, 9QPH1YQCMn) in quality. It offers: a novel insight, five new benchmarks, consistent improvements across four base methods, and thorough ablations. The weaknesses — distribution shift unaddressed in two benchmarks, AUROC-only metric, no blind baselines — are real but standard in the subfield and comparable to the accepted papers' weaknesses.

The distribution-shift concern specifically echoes the zWqr3MQuNs (WikiMIA) paper which used exactly the same temporal split approach and was accepted despite that critique being raised. The key difference is that this paper builds *on top of* those benchmarks and adds non-temporal ones, which is a step forward.

**Final assessment:** Borderline accept. The core idea is novel, well-executed, and produces real improvements. The missing blind baselines and TPR@low FPR metrics are the most actionable requested revisions. This paper contributes meaningfully to a difficult problem area where tools are scarce.

**Score: 5.0** — Marginally below acceptance threshold as submitted, but the concerns are addressable in revision. Comparable to the lower end of accepted poster papers in this area.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>