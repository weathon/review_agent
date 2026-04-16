## Summary
This paper introduces GraphDoc, a large-scale document dataset that augments DocLayNet pages with graph-structured relations between layout elements, and proposes a new graph-based Document Structure Analysis (gDSA) task. It also presents DRGG, a plug-in relation prediction head for document detectors, and reports baseline results on both conventional layout detection and the new graph prediction setting.

## Strengths
- The paper identifies a real and meaningful gap between standard document layout analysis and richer structural understanding. Modeling relations such as reading order, hierarchy, and references is a sensible extension beyond box detection.
- GraphDoc is substantial in scale: the paper reports 80K pages, 1.10M instances, and 4.13M relation pairs over 8 relation types, which is materially larger than prior structure-oriented document datasets summarized in Table 1.
- The task formulation is clear. The distinction between DLA and gDSA, and the decomposition of gDSA into spatial and logical relations, is easy to understand and potentially useful for downstream document understanding work.
- The paper provides a concrete benchmark setup rather than only a dataset release. Tables 2 and 3 give both aggregate and per-relation results, and the latter is especially helpful because it shows which relations are easy versus genuinely difficult.
- DRGG appears practically usable as an attachable relation head across multiple detector/backbone combinations, and the paper demonstrates compatibility with DETR, Deformable DETR, DINO, and RoDLA.

## Weaknesses

###: Fatal

### Major:
- **The dataset’s relational annotations are heavily heuristic, and the paper does not provide enough evidence that these labels are reliable enough to support the benchmark’s central claims.**  
  This concern is directly supported by Sec. 3.1.4. The annotation pipeline uses OCR/PDF extraction, nearest-neighbor spatial rules, XY-cut-based reading order, rule-based hierarchy construction, and text matching for references. The paper then states only that “**most of the results have been manually verified and refined**,” but gives no quantified manual verification rate, no sampled accuracy audit, no inter-annotator agreement, and no relation-wise quality analysis. For a datasets/benchmarks paper, that is a substantial validity gap: the benchmark is only as sound as its labels. This is especially important because the claimed contribution is not merely scale, but “deeper” structure understanding.
- **The empirical section does not cleanly isolate the contribution of DRGG.**  
  Table 2 mostly compares different detector/backbone combinations *with* DRGG. For DLA, the only explicit no-DRGG comparison shown is InternImage+RoDLA without relation head (80.5) versus with DRGG (81.5). There is no parallel no-DRGG comparison for DETR, Deformable DETR, or DINO, and no competing relation-head baseline. As written, the paper does not establish whether gains come from DRGG itself, the stronger detector/backbone combination, joint training, or simply compatibility between the detector and the induced labels. Since DRGG is a named model contribution, this missing isolation matters.
- **The paper overstates what is demonstrated by the task and experiments.**  
  The abstract and introduction repeatedly frame the work as enabling “human-like,” “holistic,” or “deeper” document understanding. However, the actual benchmark is page-level box detection plus prediction of 8 edge types on an induced relation graph, and the experiments evaluate only graph recovery on this benchmark. There is no downstream validation showing that the graph representation improves actual document understanding tasks beyond the benchmark itself. The work is still useful as a structured prediction benchmark, but the stronger conceptual claims are not well supported by the presented evidence.
- **The annotation/evaluation setup likely makes some relations close to trivial, which weakens the informativeness of the aggregate numbers.**  
  Table 3 shows ~99 AP for Left/Right relations in several settings, while logically more meaningful relations such as Reference are much weaker (best main model: 16.8 AP). Given that Sec. 3.1.4 defines spatial relations by geometric rules and keeps only nearest adjacent boxes, these extremely high scores are unsurprising. This does not invalidate the task, but it does mean that aggregate gDSA performance can overstate progress on the semantically harder parts of the benchmark unless spatial and logical relations are more clearly separated in the discussion.

### Minor
- **The paper is limited to single-page relations, which narrows the scope of the claimed document structure understanding.**  
  Sec. 3.1.2 explicitly states: “**we will only consider relations within the same page and not those across pages**.” This is a reasonable scoping decision, not a flaw by itself, but it does materially limit applicability to real document structure phenomena such as cross-page reading order or references.
- **The model is visual-only despite several logical relations being inherently text-dependent.**  
  This is acknowledged by the authors in the limitations section: “**Our model structure focused only on visual modality input without multi-modality input consideration**.” Given that reference and hierarchy often depend on textual cues, the weak Reference AP is unsurprising. Because the paper itself acknowledges this limitation, this should be viewed as a meaningful but not fatal shortcoming.
- **Evaluation protocol clarity is incomplete in the main paper.**  
  Sec. 4.2 says experiments were conducted on GraphDoc “for both training and validation,” but the main text does not clearly describe a held-out test protocol. For a benchmark paper, clearer dataset split and evaluation protocol exposition in the main body would help.
- **The claimed benefit of gDSA for DLA is modest in the evidence shown.**  
  The main explicit comparison is RoDLA 80.5 vs. RoDLA+DRGG 81.5. That is a positive result, but relatively small compared with the paper’s broad claim that gDSA improves layout analysis.

### Trivial

## Nice-to-Haves
- Report annotation quality on a manually labeled subset, ideally relation-wise.
- Separate aggregate metrics for spatial vs. logical relations, or otherwise avoid letting easy geometric relations dominate the headline number.
- Include a simple geometric/rule-based baseline for relation prediction, especially because parts of the annotation pipeline are themselves geometry-based.
- Move key ablations from the appendix into the main paper, especially those clarifying which DRGG components matter.
- Add qualitative predicted-vs-ground-truth graph visualizations and error analysis for Reference / Parent / Child / Sequence relations.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The proposed gDSA evaluation metric is not a valid AP-style metric, so the headline numbers are not interpretable.”**  
  The paper indeed uses a nonstandard thresholded relation metric in Sec. 3.3/Algorithm 1, and the naming could be clearer. However, the paper explicitly defines its own evaluation procedure rather than claiming to use standard object-detection AP unchanged. It is fair to say the metric is unconventional and should be justified more carefully, but the stronger criticism that it is simply invalid is too strong based on the paper alone.
- **Complaints about missing related work.**  
  Not included per instruction.
- **Reproducibility complaints about omitted implementation details in the appendix.**  
  The paper explicitly points to appendices for model/configuration details; this is not by itself a substantive weakness for the main review.
- **Claims doubting the existence/availability of cited baselines, datasets, or tools.**  
  Removed per instruction.
- **Generic requests for confidence intervals / repeated runs.**  
  These would be nice, but for this style of benchmark paper they are not a core flaw.

## Novel Insights
The most important synthesis is that the paper is strongest as a **benchmark proposal for induced page-structure prediction**, not yet as evidence of “human-like” document understanding. The same design choice that gives the work scale and practicality—bootstrapping relations from DocLayNet via rules—also creates its central tension: it makes GraphDoc feasible and large, but leaves the paper needing much stronger validation that the induced graph labels are sufficiently faithful and nontrivial. In other words, the paper’s promise is real, but the current evidence better supports “useful structured benchmark with a reasonable baseline” than the broader semantic framing the authors adopt.

## Suggestions
- Quantify annotation quality on a manually curated subset, including relation-wise precision/recall or agreement for spatial, sequence, hierarchy, and reference edges.
- Add at least one simple heuristic/geometric baseline; this is particularly important for interpreting the near-perfect Left/Right scores.
- Add cleaner ablations isolating DRGG from detector/backbone choice and from multitask training.
- Reframe the claims more conservatively around structured page understanding rather than human-like document comprehension unless additional downstream evidence is added.
- Report headline results separately for spatial and logical relation groups.
- Clarify the dataset split and evaluation protocol in the main paper.

## Score and Decision
**Originality:** Good. The gDSA framing and graph-structured document benchmark are genuinely novel within document layout analysis.  
**Importance:** Moderately high. Richer structure supervision for documents is a worthwhile problem.  
**Claims support:** Mixed. The narrower benchmark contribution is supported; the broader “holistic/human-like understanding” claims are not.  
**Experimental soundness:** Fair but incomplete. The baseline study is useful, but annotation validation and contribution isolation are both lacking.  
**Clarity:** Generally clear at the task level, though some methodological and evaluation details are underexplained.  
**Value to the community:** Potentially meaningful if the dataset is released and adopted, but current confidence is reduced by the limited validation of the heuristic labels.

**Calibration against human-reviewed papers:**  
- Compared against **ADOPD** (`/home/wg25r/review_agent/human_reviews/x1ptaXpOYa.md`, accept poster; scores 8/6/6/6), this paper is weaker because ADOPD’s contribution is a dataset with stronger human-in-the-loop positioning, whereas GraphDoc’s central labels are more heavily heuristic and less validated.  
- Compared against **Chronicling Germany** (`/home/wg25r/review_agent/human_reviews/wh6pilyz2L.md`, reject; scores 6/6/5/6), this paper is stronger in scale and task ambition, but suffers from a similar annotation-quality-validation concern.  
- Compared against **DocLayout-YOLO** (`/home/wg25r/review_agent/human_reviews/k0X4m9GAQV.md`, reject/withdrawn; scores 3/6/5/5), this paper is more novel and more valuable because it contributes a new benchmark/task rather than only an incremental method.  
- Compared against **ADOPD-Instruct** (`/home/wg25r/review_agent/human_reviews/lBlHIQ1psv.md`, reject; scores 5/3/5/5), this paper is somewhat stronger in technical coherence, but still not strong enough empirically for acceptance due to benchmark-validity concerns.

Overall, this lands in the **borderline reject to reject** range: clearly more substantial than weak incremental submissions, but not yet convincing enough as a benchmark paper because the reliability of the induced labels and the exact contribution of DRGG are insufficiently established.

**Score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>