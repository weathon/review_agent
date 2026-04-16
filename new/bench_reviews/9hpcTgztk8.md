Now I have enough calibration data. Let me synthesize the final review.

## Summary
This paper introduces REPLM, a framework for document-level in-context few-shot relation extraction using pre-trained language models. REPLM reformulates relation extraction as a triplet generation task: for each relation type and document, it retrieves semantically similar in-context examples from a distantly supervised corpus, constructs multiple sets of these examples, prompts an LM to generate subject-object pairs, and aggregates the outputs via a weighted probabilistic scheme. The key practical advantages claimed are: (1) no NER pipeline needed, (2) no human annotations required, and (3) flexibility to use new relations or backbone LMs without retraining. The authors evaluate across six RE datasets using five LM backbones, claiming state-of-the-art results.

## Strengths

- **Novel problem formulation for document-level RE**: The paper is the first to reformulate document-level relation extraction as an in-context few-shot learning paradigm, eliminating the NER pipeline and fine-tuning steps. This is a meaningful contribution to the RE literature (Table 1 clearly shows the gap in prior work).

- **Comprehensive empirical coverage**: Evaluation across six datasets (DocRED, CDR, GDA, CoNLL04, NYT, ADE), five LM backbones (GPT-JT, Llama-3.1-8B/70B, GPT-3.5-Turbo, GPT-4o), and ablations provides a thorough experimental picture (Table 4, Table 5).

- **Systematic ablations**: The ablation across "random context" → "best context" → "complete framework" consistently shows that each component contributes meaningfully across all datasets and backbones (Table 5), and the entity-randomization experiment (Fig. 4b) is a thoughtful probe for memorization vs. extraction.

- **Practical flexibility**: The ability to swap backbone LMs without retraining (demonstrated empirically across five models) and to add new relations by providing in-context examples is a genuine practical advantage.

## Weaknesses

### Major:

- **Overclaimed "state-of-the-art" status due to asymmetric comparison**: The "state-of-the-art" claim in the abstract and throughout the paper is misleading. On DocRED, the main comparison in Table 2 is against REBEL (27.52 F1) and REBEL-sent (26.17 F1)—models that require NER as input and are orders of magnitude smaller. When stronger fine-tuned baselines are included in Table 4, REPLM(GPT-4o) at 68.35 only marginally outperforms DocRED-CLiP (68.13), a much smaller fine-tuned model. The paper frames this as a fair comparison because REPLM doesn't need NER or fine-tuning, but these are fundamentally different settings. A 70B+ parameter model with 101K in-context documents vastly outscaling a fine-tuned 110M model is not an apples-to-apples SOTA comparison. The claims should be qualified accordingly—this is a strong in-context learning result for document-level RE, not a clear SOTA in the standard supervised sense.

- **Computational cost claims are unsupported and likely inverted**: The paper repeatedly claims baselines have "large computational overhead (e.g., from fine-tuning)" and that REPLM avoids this. However, REPLM requires: (1) semantic retrieval over 101K documents, (2) L prompts per relation per document (L=5 sampled sets × 96 relation types for DocRED = ~480 LM calls per document), and (3) each call to GPT-4o/70B models is extremely expensive. This is likely far more costly at inference time than a single forward pass through a fine-tuned 110M model. The "no computational overhead" framing is incorrect for inference and should be revised to be transparent about the trade-off (no training cost vs. higher inference cost).

- **The "no human annotation" claim is overstated**: The paper emphasizes "circumvents the need for human annotations" (Abstract, Sec. 1). However, the distantly supervised DocRED split (101,873 documents) is created by aligning Wikipedia with Wikidata—a human-curated KB—and is used as labeled in-context examples. This is functionally equivalent to using a large labeled training set, just through in-context learning rather than gradient updates. The distinction from other DS-based RE methods (which also avoid manual document annotation) is not clearly acknowledged. Additionally, the REPLM(params adj) variant explicitly uses the human-annotated training set for hyperparameter tuning.

- **External KB evaluation is methodologically problematic**: In Sec. 6.2, the paper constructs an augmented gold standard by aggregating all methods' outputs and checking them against Wikidata. Since REPLM generates ~4× more triplets per document than REBEL (20.21 vs. 4.93), the augmented gold set is naturally biased toward high-recall methods. Furthermore, Wikidata matching does not verify that a relation is actually expressed in the source document—confating KB completion with text-based extraction. The 59-80% improvement claims over REBEL in Table 3 are therefore unreliable as evidence of REPLM's extraction quality.

### Minor:

- **Probabilistic formulation is loosely connected to implementation**: Equations (1) and (5) present a probabilistic framework, but the length-normalized geometric mean in Eq. (5) is a heuristic, not a derived probability. The candidate space varies across context sets (missing entries treated as zero probability). The "probabilistic" narrative oversells what is essentially a heuristic scoring mechanism.

- **Scalability concerns**: Processing each of 96 relation types separately with multiple in-context sets means ~480 LM forward passes per document for DocRED. This limits real-time applicability despite the theoretical flexibility.

- **Missing precision-recall breakdown**: REPLM outputs ~20 triplets/doc vs. REBEL's ~5, yet only F1 is reported. Without P/R separately, it's impossible to assess whether high F1 comes from inflated recall at the cost of many false positives.

- **The random-entities ablation only tests entity memorization, not factual hallucination**: Replacing entity names with random strings (Sec. 8, Fig. 4b) shows REPLM doesn't simply look up entity names from memory, but doesn't address whether the model generates relation types based on parametric world knowledge rather than document content. The experiment tests entity string recall, not relation extraction faithfulness.

### Trivial:
- The heatmap in Fig. 2 uses numeric relation IDs instead of interpretable names, making it hard for readers to assess per-relation performance patterns.

## Nice-to-Haves

- Report inference cost (tokens, latency, or API cost) per document for REPLM vs. fine-tuned baselines to substantiate or revise the computational overhead claims.
- Provide a simple in-context prompting baseline (e.g., GPT-4o with randomly selected examples and no multi-set aggregation) to isolate the contribution of the framework components from the contribution of the backbone LM itself.
- Include a precision-recall tradeoff analysis (or PR curves) rather than F1 alone, especially given the large discrepancy in output volume between REPLM and REBEL.
- Acknowledge that REPLM(params adj) uses human annotations for hyperparameter selection, qualifying the "no human annotation" claim for this variant.

## Removed Points

- **"REPLM uses closed-source models (GPT-3.5/GPT-4o) making results irreproducible"**: The paper also evaluates on open-source models (GPT-JT, Llama-3.1-8B/70B), and using proprietary models is standard practice in current ICL research. This is not a valid criticism.
- **"No comparison against strong fine-tuned baselines on DocRED (ATLOP, DocRED-CLiP)"**: Table 4 actually does include these baselines (ATLOP 63.40, DocRED-CLiP 68.13). While the comparison is across different settings, the baselines are present.
- **"CodeIE uses deprecated Codex model"**: Per the hard rules, if the paper cites it, it exists. This is not a weakness of the paper under review.
- **"The threshold θ is tuned opaquely"**: The paper states that REPLM uses fixed parameters and provides sensitivity analysis in Appendix J, and REPLM(params adj) transparently uses the training set. This is adequately addressed.
- **"Distantly supervised data is functionally equivalent to training data"**: While we weakened the "no human annotation" claim above, calling this "functionally equivalent to training" is too strong—in-context learning and gradient-based training have different properties (no parameter updates, no overfitting in the traditional sense).

## Novel Insights

The paper surfaces an important practical insight: document-level RE benchmarks like DocRED have significant annotation gaps that penalize high-recall methods. REPLM's tendency to extract more triplets (20.21/doc vs. 4.93) reveals that standard evaluation metrics conflate method quality with annotation completeness, and the authors are right to flag this as an evaluation challenge. However, the proposed fix (Wikidata-based augmentation) introduces its own biases. A more principled future direction would be systematic human annotation of a random sample of proposed extractions.

## Suggestions

- Qualify the "state-of-the-art" claim to explicitly state that REPLM achieves strong results under a different evaluation regime (no fine-tuning, no NER, using distant supervision as in-context examples) rather than a direct apples-to-apples comparison with fine-tuned supervised models.
- Add an inference cost analysis (tokens per document, wall-clock time, or estimated API cost) to provide an honest comparison of the computational trade-off.
- Run a simple ablation: GPT-4o with random K=5 examples and single-set prompting (no retrieval, no multi-set aggregation) as a baseline, to measure the contribution of the framework's components vs. the backbone LM's capability.
- Report precision and recall separately, especially given the large output volume discrepancy between REPLM and baselines.

## Score and Decision

**Calibration anchors:**
- **PromptNER** (WDQ9ZzsgDL): LLM-based few-shot NER claiming SOTA but comparing GPT-3.5/4 against much smaller baselines; rejected with scores 3/5/3/3.
- **When does ICL Fall Short** (Cw6lk56w6z): Found ICL fails on specification-heavy IE tasks, reaching less than half of SOTA on RE tasks; rejected with scores 5/5/5/3.
- **Are Human-generated Demonstrations Necessary for ICL** (frRDT6EOhg): Simple ICL method using self-generated demonstrations, accepted with scores 8/6/6/6.
- **Bio-RFX** (KskgLM728l): RE method with limited baseline comparison; rejected with scores 6/5/5/6.
- **Text2NKG** (1g77zRaJq0): N-ary RE framework; withdrawn/rejected with scores 5/5/3/3.

REPLM shares a similar overclaim pattern with PromptNER (SOTA via vastly larger models), but has a more substantial methodological contribution (the multi-set aggregation, semantic retrieval, and full probabilistic framework are non-trivial). It also has stronger empirical coverage than "When does ICL Fall Short." However, the overclaims are significant—"state-of-the-art" across six datasets when the comparison is fundamentally asymmetric, "no human annotation" when using 101K distantly supervised examples and the params adj variant uses labeled training data, and "lower computational overhead" when the inference cost is likely much higher. These are not presentation issues; they are substantive mischaracterizations at the core of the contribution claims.

The underlying framework is solid and the empirical results are informative, but the gap between what's demonstrated and what's claimed is too large for acceptance in its current form.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>