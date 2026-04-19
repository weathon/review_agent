Now I have sufficient calibration data. Let me write the final review:

## Summary

This paper introduces REPLM, a framework for document-level relation extraction using in-context few-shot learning with pre-trained language models, eliminating the need for fine-tuning, explicit NER pipelines, and human-annotated training data. The method retrieves multiple sets of distantly-supervised in-context examples, aggregates outputs probabilistically, and claims state-of-the-art performance across six datasets.

## Strengths

- **Novel application of in-context learning to document-level RE**: The paper successfully bridges a recognized gap by adapting in-context few-shot learning for document-level relation extraction, whereas prior ICL methods (GPT-RE, CodeIE) are limited to sentence-level tasks due to context window constraints. Table 1 clearly positions this contribution.

- **Strong empirical performance with modern backbones**: REPLM with GPT-4o achieves 68.35 F1 on DocRED (Table 4), marginally exceeding DocRED-CLiP (68.13), and shows consistent scaling benefits from GPT-JT → GPT-3.5 → GPT-4o and Llama-8B → Llama-70B across all six datasets. The ablation in Table 5 demonstrates the aggregation mechanism provides +18% improvement over single best-context retrieval.

- **Demonstrates annotation gaps in existing benchmarks**: The paper provides qualitative evidence (Sec. 6.1, Appendix G) and quantitative analysis (Sec. 6.2) suggesting DocRED and biomedical datasets are under-annotated. The manual validation examples (e.g., the Félix Guattari author relation) are plausible cases of missing annotations.

- **Flexible, training-free architecture**: The framework can incorporate new relations and stronger backbones without retraining, as evidenced by seamless integration of five different LMs (Table 4, Table 5) and the finding that performance scales monotonically with backbone quality.

## Weaknesses

### Fatal

**None**

### Major

- **The Wikidata-augmented evaluation fundamentally redefines the task and invalidates central superiority claims**: Section 6.2 aggregates all predicted triples from all methods, matches them against Wikidata, and adds any match to the ground truth. This transforms the task from "is this relation *expressed in the document*" (DocRED's annotation guideline) to "is this triple *true in the world*." The paper then uses this altered evaluation to claim REPLM "performs much better than the original labels" (Abstract). This evaluation is circular: REPLM uses distantly-supervised data derived from KBs and powerful LMs with world knowledge, so it is structurally positioned to output "all plausible KB triples" rather than strictly text-supported extractions. REBEL and other baselines were trained to respect DocRED's labeling criterion and are systematically disadvantaged. The SOTA claims under standard DocRED evaluation are not supported.

- **Evaluation does not disentangle extraction from world-knowledgerecall**: REPLM uses heavily pre-trained LMs (GPT-4o, Llama-70B) that likely contain many DocRED/NYT facts in their parameters. The evaluation counts an extraction as correct if subject/object align with ground truth, but does not verify the relation is *supported by the input document* rather than recalled from pretraining. The only disentanglement attempt (Sec. 8, random entity names on CONLL04) is on a small sentence-level dataset, shows a non-trivial performance drop (72.9 → 70.47), and does not analyze *where* performance is preserved vs. collapses. Document-level DocRED results remain vulnerable to the criticism that REPLM may reconstruct KB triples from model priors rather than extracting from document content.

- **Baseline comparisons are not aligned on resource assumptions**: The paper claims to outperform "more than 30 baseline methods" while using no fine-tuning or human annotations. However, REPLM uses: (i) a distantly-supervised DocRED split constructed from human-curated KBs; (ii) pre-trained LMs encoding massive world knowledge (potentially including training documents); and (iii) proprietary models (GPT-3.5/4o) in some experiments. Many baselines are trained only on labeled data and forbidden from external KBs. The asymmetry favors REPLM, not the baselines, yet the paper frames REBEL's fine-tuning on DocRED as an "unfair advantage" (Sec. 5) without acknowledging its own advantages. Table 4 also mixes sentence-level and document-level methods, and the strongest document-level baseline (DocRED-CLiP) is within 0.22 F1 of REPLM (GPT-4o) — a marginal gain given REPLM's resource advantages.

### Minor

- **The "no NER" claim is overstated**: While REPLM does not run an explicit NER module, the distantly-supervised pool \(\mathcal{D}^{\text{dist}}\) is constructed by matching KB entities to documents, which assumes high-quality entity recognition and linking upfront. The LM implicitly handles entity identification during generation, and evaluation uses exact span matching. This is not truly "no NER" but rather implicit, uncalibrated entity handling inside a large LM plus KB-derived supervision. The conceptual advantage over NER-based pipelines is weaker than claimed.

- **Probabilistic formulation lacks empirical validation**: Equation 5 (length-normalized product of token probabilities) is ad hoc with no justification as a calibrated probability. The weighting over context sets \(p(C_l \mid d_i, r)\) is based purely on cosine similarity scores without modeling connection to extraction accuracy. The paper does not analyze whether these scores correlate with correctness (e.g., calibration plots, precision-recall vs. threshold), making the "probabilistic framework" more of a veneer than validated modeling.

### Trivial

**None**

## Nice-to-Haves

- A systematic human evaluation of "extra" triples (those REPLM predicts but are not in DocRED annotations) stratified by probability, categorizing each as (a) text-entailed, (b) true but not text-entailed, or (c) false, would quantify how much of the apparent gain is genuinely filling annotation gaps vs. hallucination.

- Error analysis by intra-sentence vs. cross-sentence relations and mention distance on DocRED would test whether REPLM truly leverages document structure or primarily captures same-sentence patterns.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **From Harsh Critic Section 2.3**: "For document-level DocRED claims, the strongest baselines in Table 4 (e.g., DocRED‑CLiP at 68.13 F1) are sentence‑level and marked as not directly applicable to documents" — This is incorrect. DocRED-CLiP is explicitly a document-level method (Jain et al., 2024), and the paper correctly places it in the "Document-level methods" section of Table 4.

- **From Harsh Critic Section 5**: Criticism about variance/statistical significance for non-random variants — The paper does report standard deviations for random-context variants (Table 2: ±0.17, ±0.09; Table 5 caption: "std. dev. below 0.1"), and fixed-context variants are deterministic, so variance reporting is not applicable.

## Novel Insights

The paper's most novel contribution is demonstrating that in-context few-shot learning — previously thought limited to sentence-level RE due to context window constraints — can scale to document-level tasks through a retrieval-and-aggregation architecture. However, the evaluation methodology reveals a deeper tension in RE benchmarking: if datasets like DocRED are genuinely under-annotated (as the paper suggests), then methods leveraging world knowledge may appear superior not because they extract better, but because they fill gaps via parametric recall. The Wikidata-augmented evaluation, while methodologically problematic for claiming extraction SOTA, does highlight a real issue: standard RE annotations may be incomplete, and there is value in distinguishing between "text-expressed" and "world-true" relations for KB completion tasks.

## Suggestions

1. **Run a document-level evaluation that isolates extraction from world knowledge**: For DocRED (or a subset), anonymize entity names with synthetic identifiers unknown to the web while preserving relations, and have human annotators label document-expressed relations. Compare REPLM and strong baselines here. Without this, the core "extraction" claim remains unsubstantiated.

2. **Provide head-to-head comparison under standard DocRED protocol**: Evaluate at least one strong recent document-level method (e.g., DocRED-CLiP, SSAN, ATLOP) under the official DocRED evaluation with original ground truth, clearly documenting supervision assumptions for all methods.

3. **Conduct precision-oriented human evaluation**: Sample 200-300 triples that REPLM predicts on DocRED dev but are not in original annotations, stratified by probability score. Have annotators judge whether each is (a) entailed by text, (b) true but not text-entailed, or (c) false. Report precision by stratum.

4. **Reframe claims appropriately**: The paper would be stronger if it explicitly positioned REPLM as a "KB completion from weak text signals" method rather than claiming SOTA on "document-level relation extraction" under standard definitions. This would align the task definition with the evaluation protocol.

5. **Calibration analysis for probability scores**: Show precision-recall curves as a function of threshold θ, and compare against simpler heuristics (e.g., top-k generation without probabilities) to justify the probabilistic machinery.

---

## Evaluation Axes

- **Originality**: High. First work to adapt in-context few-shot learning for document-level RE. The retrieval-and-aggregation architecture is a creative solution to context window limitations.

- **Importance**: Moderate-High. Document-level RE is an important task, and a training-free, flexible approach has practical value. However, the methodological concerns about evaluation limit the scientific contribution.

- **Claims well-supported**: Weak. The SOTA claims hinge on an evaluation protocol that redefines the task to favor REPLM. The core claim of "document-level extraction without NER or annotations" is not cleanly demonstrated.

- **Soundness of experiments**: Moderate. The experiments are extensive (6 datasets, 5 backbones, 30+ baselines) and well-ablated, but the evaluation protocol has structural flaws that bias conclusions.

- **Clarity of writing**: Good. The paper is clearly written, well-organized, and figures/tables are informative.

- **Value to community**: Moderate. The framework is practically useful for KB completion scenarios, and the finding about annotation gaps is valuable. However, the evaluation issues could mislead future work if uncorrected.

## Score and Decision

**Calibration reasoning:**

Compared to retrieved anchors:
- Papers with similar strengths (novel framing + strong empirical sweep) but weaker methodology got mixed scores: CycleResearcher (8,6,6,6 - accepted) addressed a novel problem but had reward overoptimization concerns; WorfBench (6,8,6,6,6 - accepted) proposed a benchmark but lacked comparison to prior work.
- Papers with unfair baseline comparisons got rejected: The vision-free grammar induction paper (3,1,3) was rejected because pretrained LLM embeddings gave it unfair advantage over from-scratch methods — similar to REPLM's advantage over fine-tuned baselines.
- Papers with unsupported claims about evaluation got rejected: AutoCustomization (3,1,3,3,3) was rejected because claims about BiasShift being "better than human" were misleading and unsupported — analogous to REPLM's claim of SOTA under a redefined evaluation.

The paper's fatal methodological issue is not as severe as papers scoring 1-3 (no data fabrication, no fundamentally broken claims), but the Wikidata-augmented evaluation and unfair baseline comparisons are Major flaws that undermine the central SOTA claims. The empirical results are genuinely strong, and the novel architectural contribution is real. However, the core scientific claim ("SOTA document-level RE without fine-tuning or annotations") is not established under standard evaluation.

Papers with Major methodological flaws but strong empirical results typically score 4-6 (borderline). The paper is below CycleResearcher and WorfBench because the evaluation protocol actively biases toward REPLM, not just missing comparisons. It is above AutoCustomization because the empirical results are more convincing and the core architecture is novel.

Anchoring to:
- 63r6HyqyRm (3,1,3 - rejected): Similar unfair comparison issue, but that paper's motivation was also criticized. REPLM has stronger motivation.
- vunPXOFmoi (6,8,6,6,6 - accepted): Missing comparisons to prior benchmarks, but evaluation was not structurally biased.
- Papers in 4-6 range with evaluation concerns (v9Sfo2hMJl: 5,6,6; FWqTha5Jh9: 5,6,6,6)

This paper has real strengths (novel architecture, strong scaling results) and real Major weaknesses (flawed evaluation protocol). The weaknesses are significant but do not completely invalidate the empirical contributions — the scaling with backbone quality and the aggregation benefit are genuine findings. The paper should be revised to reframe claims and provide cleaner evaluation.

**MY FINAL SCORE: <pineapple>5.0</pineapple>**

**MY FINAL DECISION: <orange>Reject</orange>**

The score of 5.0 reflects a borderline paper with genuine contributions but Major methodological flaws that prevent acceptance in current form. The paper should be rejected but encouraged to resubmit after addressing the evaluation protocol concerns. The flaws are addressable (run standard evaluation, provide human verification of extra triples, reframe claims) and the core architecture is novel enough to warrant encouragement.