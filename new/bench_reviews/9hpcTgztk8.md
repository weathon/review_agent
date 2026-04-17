Now I have enough calibration data. Let me synthesize the final review. Let me carefully verify each claim from the reviewers against the actual paper content.

Key issues to verify:

1. **External KB evaluation (Harsh Critic #1)**: This is a real issue. The paper uses system outputs to augment the gold standard. Let me check how this is done... Section 6.2 says: "We aggregate all relations extracted from all methods on all documents. We then check the correctness of the extracted relations via an external KB. Specifically, we leverage the pipeline from HELM and check if generated knowledge triplets exist in Wikidata." This does mean the gold standard is partially constructed from system outputs, which is circular. However, the paper also presents results on the original DocRED dev set (Table 2). The external KB results are supplementary, not the primary evaluation.

2. **"No human annotations" claim (Harsh Critic #2)**: The paper claims no human annotations are needed. For DocRED main results, this is true (using distantly supervised split). But for Table 4 experiments on other datasets, the training splits with human annotations are used. The paper should be clearer about this nuance.

3. **Memorization concern (Harsh Critic #3)**: The paper does address this with the random-entity experiment (Sec. 8, Fig. 4b), though only on CoNLL04. This is a partial address.

4. **Baseline comparison fairness (Harsh Critic #4)**: The DocRED comparison in Table 2 compares REPLM against REBEL only, which is the only prior method that doesn't need NER. This is actually a fair comparison for the stated scope. In Table 4, REPLM with different backbones is compared against many baselines - but many are different task settings (sentence vs. document level). The paper does acknowledge this.

5. **Probability formulation (Harsh Critic #5)**: This is a minor methodological concern, not a major one.

Let me now verify the "state-of-the-art" claim more carefully. Table 2 shows F1=35.09 on DocRED dev for REPLM with GPT-JT, but Table 4 shows fine-tuned methods like DocRED-CLiP at 68.13 and REPLM (GPT-4o) at 68.35. So the "SOTA" claim in the abstract refers to the broader Table 4 comparison, not just Table 2. However, the GPT-4o comparison is against fine-tuned methods. This is actually comparing a very large model (GPT-4o) against much smaller ones.

Let me also verify the "no human annotations" claim more carefully. The paper says in Sec. 5: "In our experimental setup, we use a distantly-supervised set for in-context few-shot learning (D^dist) and evaluate the performance on the development set. Thereby, we ensure that our framework is solely trained without human annotation." This is specifically about DocRED. For the multi-dataset benchmark in Sec. 7, the paper states "training and test set" but doesn't explicitly say it uses the human-annotated training splits. Let me check more carefully...

The paper says in Sec. 7: "Baseline methods: All +30 baseline methods are only listed in Table 4 due to space." It doesn't clearly state whether REPLM uses human-annotated training data for the other 5 datasets. The ablation study in Table 5 uses "random context", "best context" etc., which likely uses the training splits. The "params adj" variant uses training data for hyperparameter tuning.

Actually, for the main results (Table 4), given that REPLM retrieves in-context examples from a "distantly-supervised set" or training set, the situation may vary by dataset. For DocRED they use the distant supervision split. For other datasets, they would need some set of documents with relation annotations, which could be the training splits. This needs clarification.

Now let me finalize my assessment:

The paper has real contributions (novel problem formulation for document-level ICL-based RE, practical advantages) but also has several substantive issues:

- The external-KB evaluation methodology is circular
- The "no human annotation" and "few-shot" claims are overstated for Table 4 results
- The SOTA claim needs qualification (it's SOTA only with GPT-4o, not with smaller models)
- Performance heavily depends on LM backbone scale
- Computational cost not quantified

Let me calibrate the score. PromptNER (similar pattern: prompting method, unfair comparison with smaller models, limited novelty) got scores 3,5,3,3 (avg ~3.5, rejected). The paper under review has more substance than PromptNER (broader evaluation, ablations, multiple backbones) but has more serious evaluation methodology issues. Papers in the 4-5 range tend to have interesting ideas but flawed execution or overclaimed results. I think this paper is in the 4-5 range given the overclaimed nature of some results and the circular evaluation, but with real contributions.</think>## Summary

REPLM reformulates document-level relation extraction as an in-context few-shot learning task using pre-trained language models. Rather than fine-tuning or requiring named entity recognition pipelines, the framework retrieves semantically relevant in-context examples from a distantly-supervised corpus, constructs multiple weighted prompt sets, and aggregates LM outputs via a probabilistic framework to generate knowledge triplets directly. The paper evaluates across six RE datasets with five LM backbones, comparing against 30+ baselines and claiming state-of-the-art performance.

## Strengths

- **Novel task formulation**: The paper is the first to apply in-context few-shot learning to document-level relation extraction, addressing a clear gap left by prior sentence-level ICL approaches (GPT-RE, CodeIE, Wadhwa et al.), as clearly shown in Table 1. This is a genuine and meaningful contribution.

- **Practical advantages**: Eliminating the NER pipeline (avoiding error propagation), leveraging distant supervision instead of human annotations for DocRED (Sec. 5), and allowing seamless backbone LM swapping (Table 4/5) are practically valuable properties demonstrated in the experiments.

- **Systematic ablation study**: Table 5 shows consistent improvements from random context → best context → complete framework across all datasets and backbones, providing strong evidence that the proposed multi-set aggregation mechanism adds real value beyond simple retrieval.

- **Creative memorization test**: Replacing entities with random names on CoNLL04 (Fig. 4b, Sec. 8) is a thoughtful diagnostic for distinguishing extraction from memorization, even if limited in scope.

- **Valuable annotation quality finding**: The observation that DocRED dev annotations are incomplete and that high-coverage extractors are penalized (Sec. 6.1, Appendix F/G) is a useful contribution to the community, even though the specific "fix" proposed is flawed.

## Weaknesses

### Major

- **Circular external-KB evaluation undermines "better than labels" claim (Sec. 6.2, Table 3)**: The paper claims REPLM "performs much better than the original labels" (Abstract) by augmenting DocRED ground truth with Wikidata-matched triplets. However, the augmented labels are constructed by aggregating outputs from *all systems being evaluated* and checking Wikidata membership. This makes the gold standard dependent on the evaluated systems. A model that outputs many triplets (REPLM averages 20.21 vs. REBEL's 4.93 per document) has more chances to match Wikidata and get those matches added as gold, systematically favoring higher-recall generators. The resulting F1 gains (59–80% over REBEL in Table 3) are not valid for system comparison. The original DocRED dev evaluation (Table 2) remains valid, but the "better than labels" narrative is not supported by sound methodology.

- **"No human annotations" and "few-shot" claims are overstated for the multi-dataset benchmark**: The abstract and introduction emphasize that REPLM "eliminates the need for human annotations of documents" and is a "few-shot" method. This is true for the DocRED main setup (using the distantly-supervised split), but for the extensive benchmark in Table 4, other datasets' human-annotated training sets serve as the in-context example pools. The "params adj" variant also uses human-annotated data for hyperparameter tuning. With 96 relation types and K in-context examples per relation drawn from full training corpora, this is closer to retrieval-augmented inference than genuine few-shot learning. These claims need to be scoped precisely to what they actually cover.

- **SOTA claim requires important qualifications**: REPLM achieves competitive SOTA numbers (68.35 F1 on DocRED) only with GPT-4o — a proprietary model orders of magnitude larger than the fine-tuned baselines (BERT-scale models). With the open-source GPT-JT (6B), REPLM achieves only 35.09 F1 on DocRED, far below fine-tuned methods like ATLOP (63.40) or SSAN (65.69). The abstract's "state-of-the-art performance across six relation extraction datasets and outperforming more than 30 baseline methods" obscures this crucial dependence on model scale, and the comparison conditions are asymmetric (large proprietary LMs vs. smaller fine-tuned models).

### Minor

- **Computational cost at inference not quantified**: REPLM requires L×R forward passes per document (L context sets, R relation types). For DocRED with 96 relation types and L context sets, this is very expensive, especially with GPT-4o. The repeated claim of "low computational overhead" relative to fine-tuning is asserted but never measured with wall-clock time, token counts, or API costs. Fine-tuned models need only one forward pass per document at inference.

- **Text-grounded extraction vs. world-knowledge recall conflation**: The random-entity experiment (Sec. 8) provides partial evidence that REPLM extracts from context rather than memorizing, but only on small CoNLL04. On DocRED (Wikipedia-based), GPT-4o likely encountered similar documents during pretraining, and the evaluation metric (F1 against gold labels) does not distinguish whether extracted relations are entailed by the document text or reflect the LM's parametric knowledge.

- **Threshold θ and scoring formulation are lightly justified**: The "length × geometric mean probability" scoring in Eq. (5) and the threshold θ for triplet filtering are core to the method but receive no theoretical motivation or detailed sensitivity analysis beyond Appendix J. The design appears tuned for performance rather than principled.

### Trivial
- None worth listing.

## Nice-to-Haves

- A genuine few-shot evaluation on unseen relation types — the paper claims flexibility to new relations without retraining, but never tests this.
- Reporting precision and recall separately to illuminate the high-recall / potentially-low-precision trade-off (20.21 vs. 4.93 triplets per document).
- Testing with smaller, fully open-source models (beyond GPT-JT 6B) to assess the minimum model size needed for the framework to work well.

## Removed Points

These points are flagged to be removed — treat them with caution:

- **"Missing RE baselines like ATLOP, SSAN, etc. in Table 2"**: The paper explicitly scopes Table 2 to methods that don't require a NER pipeline (see Sec. 5 and Table 1). REBEL is the only such prior method. Including NER-dependent methods would be an unfair comparison against the baselines, not against REPLM. The broader comparison appears in Table 4.

- **"CodeIE uses a deprecated model"**: Per instructions, we do not question the existence or availability of cited models/tools.

- **"Reproducibility concerns about undisclosed hyperparameters"**: The paper provides code, and hyperparameter sensitivity is addressed in Appendix J. This is a nitpick beyond community norms.

- **"Missing confidence intervals / significance tests"**: Large-scale benchmark evaluation with LLMs typically reports single-run results; this is standard practice in the field and requesting confidence intervals for this type of evaluation is not the norm.

- **"Missing related works"**: Per instructions, we do not flag missing related work citations as we cannot verify their existence.

## Novel Insights

The identification that DocRED development set annotations are systematically incomplete — penalizing methods that produce higher-recall predictions — is a genuine insight, even if the proposed method of addressing it (external KB augmentation) is flawed. The random-entity experiment, though limited in scope, is a clever diagnostic that future ICL-based IE papers should replicate on larger datasets.

## Suggestions

- **Restrict the "no human annotation" claim** to the DocRED (distant supervision → dev) setup, and explicitly acknowledge that multi-dataset results in Table 4 use supervised training splits as in-context pools.
- **Remove or substantially weaken the "better than original labels" claim** based on Sec. 6.2. If external KB validation is desired, construct an independent gold standard (e.g., manual re-annotation of a sample) rather than augmenting from system outputs.
- **Qualify the SOTA claim**: Clearly state the conditions under which SOTA holds (GPT-4o backbone) and report performance with smaller open-source models prominently alongside fair comparisons.
- **Add inference cost analysis**: Report the number of LM forward passes, wall-clock time, and/or API costs per document to substantiate claims about computational efficiency.

## Score and Decision

**Calibration comparison:**

- **PromptNER** (prompting-based few-shot IE, similar ICL pattern; scores 3,5,3,3 avg 3.5, rejected): REPLM has more methodological substance (multi-set aggregation, probability scoring, ablations) but shares the concern about LLM-scale dependency vs. method novelty. REPLM is somewhat stronger due to broader evaluation and clearer task formulation.

- **When does ICL Fall Short** (scores 5,5,5,3 avg 4.5, rejected): Analyzed ICL limitations on IE tasks; REPLM actually demonstrates that ICL *can* work for document-level RE, but with methodology concerns.

- **ReLiK** (retriever-reader RE; scores 6,6,5 avg 5.7, rejected): Accepted-adjacent paper with solid SOTA and efficiency claims, but noted baseline comparison gaps. REPLM's evaluation issues are more serious than ReLiK's.

- **Bio-RFX** (biomedical RE; scores 6,5,5,6 avg 5.5, rejected): Methodological contributions of similar level; REPLM has broader evaluation but more overclaimed results.

The paper has real contributions (novel formulation, useful framework, extensive benchmarks) that are undermined by overclaimed results (circular KB evaluation, "no annotation" claims, SOTA with GPT-4o vs. fine-tuned BERT-scale models). These are substantive issues that a rebuttal could only partially address — the evaluation methodology for Sec. 6.2 is fundamentally flawed, and the claims need rescoping. The core idea and ablations are valuable enough to merit attention, but the paper's self-assessment significantly exceeds what the evidence supports.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>