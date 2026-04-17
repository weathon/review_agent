Let me now carefully analyze the paper's claims and the reviewers' criticisms against the paper text that was provided.

Key issues to verify:

1. **Memorization vs. extraction concern** - The paper does claim to be an "extraction" method, and the random entity experiment is only on CoNLL04. Table 4 shows much stronger results with larger LMs. This is a valid concern.

2. **External KB evaluation** - The paper augments with Wikidata triplets. Does it require the relation to be expressed in the document? Let me check: "We then check the correctness of the extracted relations via an external KB... We add all matched triplets to the existing list of ground-truth triplets from DocRED." This does NOT require the relation to be expressed in the document, only that it exists in Wikidata.

3. **Model size mismatch** - The paper's Table 4 does compare REPLM with GPT-4o against BERT-base-sized baselines. But Table 2 (the main DocRED result) uses GPT-JT (6B) vs REBEL. The "SOTA" claims need careful scoping.

4. **No need for named entities claim** - The paper uses distantly supervised data (which does have entity-level annotations aligned to KB). The claim "no need for NER" means no NER *pipeline* at test time, which is technically correct. But it does rely on entity-annotated (distantly supervised) data for in-context examples.

5. **Dev set usage** - For REPLM (params adj), hyperparameters are tuned on the training set (paper says this). REBEL is fine-tuned on training and uses dev for early stopping. Both methods are evaluated on dev. This isn't ideal but is a common practice (DocRED's test set requires submission to the leaderboard).

Let me now also check the inference cost — for DocRED with 96 relation types and L sets per relation, this is indeed substantial.

Now let me synthesize the final review.

Key calibration points:
- PromptNER (few-shot NER with LLM): Scores 3, 5, 3, 3 → Reject. Similar issues: unfair model comparison, limited novelty in prompting, overclaimed SOTA
- GoLLIE (LLM for IE with guidelines): Scores 8, 5, 6, 6 → Accept poster. This paper had genuine technical novelty (fine-tuning to follow guidelines, code-based representation), more thorough experiments with proper framing
- ReLiK (retrieval-based RE): Scores 6, 6, 5 → Reject. Had SOTA results but limited baseline comparisons
- "When does ICL fall short": Scores 5, 5, 5, 3 → Reject. Empirical analysis paper with ICL for IE

The REPLM paper is better than PromptNER (which was essentially just a prompting template) because it has a genuine framework design (retrieval + multiple context sets + probabilistic aggregation). It's worse than GoLLIE which had real fine-tuning innovation and more careful evaluation. The paper shares issues with PromptNER (unfair backbone comparison, overclaimed SOTA) but has more technical substance.

Given the calibration, I'd place this paper somewhere around 3.5-4.5. The fundamental problem of conflating extraction with memorization, combined with the unfair model comparison, significantly undermines the core claims. However, the framework design itself has genuine value (proven by the ablation Table 5), and the problem formulation is novel. This is similar to PromptNER which got 3.5 average but has more technical substance.

Let me write the final review.</think>## Summary

The paper introduces REPLM, a framework for document-level in-context few-shot relation extraction (RE) using pre-trained language models. Rather than fine-tuning, REPLM retrieves semantically similar documents from a distantly-supervised corpus as in-context examples, constructs multiple sets of these examples, and aggregates their outputs via a weighted probability framework to extract knowledge triplets (subject-relation-object). The framework is evaluated across six RE datasets with five LM backbones, claiming state-of-the-art results and advantages of no NER pipeline, no human annotation, and no retraining for new relations.

## Strengths

1. **Novel and well-motivated problem formulation.** To the best of my knowledge, this is the first work applying in-context few-shot learning to *document-level* relation extraction, filling a genuine gap left by prior ICL-based RE methods (GPT-RE, CodeIE, Wadhwa et al.) that are restricted to sentence-level settings due to context length and computational constraints. The problem formulation as triplet generation (rather than entity-pair classification) is a sensible way to avoid NER pipelines.

2. **Solid ablation evidence for the framework design.** Table 5 convincingly demonstrates that each component of REPLM contributes across all five backbone models and six datasets: "random context" → "best context" → "complete framework" yields consistent improvements (e.g., DocRED: 21.14 → 31.31 → 35.09 with GPT-JT; 52.29 → 61.78 → 67.47 with GPT-4o). This shows that the *framework itself* (retrieval + multiple context sets + weighted aggregation) provides genuine value beyond just using a strong LM.

3. **Extensive empirical coverage.** Benchmarking across 6 datasets, 5 LM backbones, and 30+ baselines (Table 4) is commendable breadth and demonstrates the framework's generality.

4. **Valuable observation about annotation incompleteness.** The insight that DocRED's annotations are incomplete and that some of REPLM's "false positives" are actually correct predictions (Section 6.1, Appendix G) aligns with community experience and is a useful contribution for evaluation methodology.

5. **Creative memorization experiment.** The random-entity-name experiment on CoNLL04 (Figure 4b) is a clever probe that partially addresses concerns about memorization, showing only a modest performance drop (72.9 → 70.47).

## Weaknesses

### Major:

1. **Evaluation conflates extraction from documents with memorization of KB facts.** The paper's central claim is that REPLM "learns to extract the relations from the context" (Section 8). However, the strongest results come from massive web-trained LMs (GPT-3.5, GPT-4o, Llama-3.1-70B) evaluated on KB-derived datasets (DocRED, NYT) whose facts are overwhelmingly present in web pretraining data. The only test of memorization separation is the random-entity experiment on CoNLL04—a small sentence-level dataset with 5 relation types. This is insufficient because: (a) none of the headline datasets (DocRED, CDR, GDA, NYT, ADE) are subjected to entity randomization; (b) replacing entity names while keeping relational structure still allows surface-level pattern matching, not proving genuine document reading; and (c) the experiment on one small dataset does not establish that results on large web-aligned corpora are not dominated by memorization. Without a "no-leak" or randomized-entity evaluation on the major datasets, the central claim of strong *document-level extraction* is not established.

2. **External KB evaluation (Section 6.2) fundamentally changes the task and rewards memorization.** The paper augments DocRED's dev labels by adding any triplet that can be matched in Wikidata, increasing relations from 12,212 to 18,592. Crucially, there is no requirement that the KB-matched relation is *expressed in the specific document*—only that it exists in Wikidata. This redefines the task from "extract relations expressed in the document" to "produce true KB facts about entities mentioned in the document." A system that simply outputs all Wikidata relations for entities in the text—ignoring the context entirely—would score well under this metric. The paper claims it "performs much better than the original labels" (Abstract), but this is only true under a metric that structurally favors large, knowledge-rich LMs over fine-tuned models. Table 3 and Figure 3 are thus not valid support for document-level extraction claims.

3. **Unfair comparison due to model size mismatch.** The strongest REPLM results (Table 4) use GPT-4o (hundreds of billions of parameters) or Llama-3.1-70B against BERT/RoBERTa/BART-scale baselines (~100M–400M parameters). The narrative repeatedly attributes state-of-the-art results to the *framework* rather than the underlying models, but Table 5 shows that backbone strength dwarfs many design choices (GPT-JT: 35.09 → GPT-4o: 68.35 on DocRED—a +33 point jump from the model alone). The main DocRED comparison (Table 2) uses GPT-JT (6B) vs. REBEL, which is more reasonable, but even here REBEL (BART-large, ~400M) is ~15× smaller. Without a same-backbone fine-tuning comparison (e.g., fine-tuned Llama-3.1-8B on DocRED training data) or at minimum parameter counts alongside F1 scores, the comparative claims are not well-founded.

4. **Misleading "state-of-the-art" claims.** The abstract claims "state-of-the-art performance," but this is only true within the restricted class of methods that do not require NER pipelines. On DocRED, fine-tuned methods with NER such as DocRED-CLiP (68.13) and DREAME (67.41) are comparable to REPLM with GPT-4o (68.35), while REPLM with the open-source GPT-JT backbone (35.09) vastly underperforms these. The SOTA claim should be clearly scoped as conditional, not absolute.

5. **Computational cost of inference is not analyzed.** REPLM requires L inference passes per relation per document. For DocRED (R=96 relation types), even with L=3 and K=5, this means 96×3 = 288 LLM inference calls per document. The paper criticizes fine-tuning for its "large computational overhead" but never quantifies REPLM's inference cost (API calls, tokens processed, latency). For commercial APIs like GPT-4o, this may be far more expensive per document than one-time fine-tuning of a smaller model. This asymmetry in cost reporting undermines the "practical advantages" framing.

### Minor:

6. **The "no need for named entities" claim is overstated in framing.** While REPLM does not run a separate NER model at test time, it relies on a distantly-supervised corpus of 101,873 documents with entity-aligned KB triplets for in-context examples. Creating such a resource in many domains is comparable in effort to obtaining NER labels. The claim should be more precisely stated as "no NER pipeline at test time" rather than "no need for named entities."

7. **Limited baseline comparison on the primary document-level evaluation.** On DocRED (Table 2), the only baseline without an NER pipeline is REBEL (26.17 F1). While Table 4 includes methods like ATLOP (63.40) and DocuNet (64.55), these are not compared under the same evaluation protocol. Including fine-tuned methods in Table 2 (even noting the NER asymmetry) would help readers calibrate results.

8. **Dev set used for both REBEL early stopping and main evaluation.** REBEL uses the DocRED dev set for hyperparameter selection and early stopping, and all comparisons are on that same dev set. The paper acknowledges this gives REBEL "an (unfair) advantage" but it actually means the comparison may be noisy—REBEL's numbers may reflect overfitting to the dev split. No test set evaluation is reported, which limits the credibility of "state-of-the-art" claims.

9. **Threshold θ and hyperparameter sensitivity insufficiently discussed in the main text.** The threshold θ for filtering triplets and temperature τ are critical for the precision-recall tradeoff but are deferred to appendices. Key values should be stated in the main text.

### Trivial:

10. **Exact string match for evaluation.** Correctness requires exact string match of subject/object spans, which disadvantages systems with different canonicalization. The paper attributes some failures on biomedical datasets to this (Section 7), but doesn't attempt relaxed matching.

## Nice-to-Haves

- A same-backbone fine-tuning baseline (e.g., LoRA on Llama-3.1-8B for DocRED) would be the single most valuable addition—it would directly isolate whether the framework or the model scale drives improvements.
- A zero-shot prompting ablation (no in-context examples, just the document and relation name) would help quantify the contribution of the retrieval/aggregation machinery vs. the LM's inherent capability.
- Systematic human evaluation of a sample of "correct-but-unannotated" extractions (rather than only anecdotal examples) would strengthen the claim about annotation incompleteness.
- Varying the size of the distantly-supervised pool (1K, 10K, 50K, 100K) would clarify how much data is needed for the "no human annotation" advantage.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"CodeIE is deprecated/unavailable"** — The paper footnotes this; per our rules, cited models and tools are assumed to exist.
- **Missing related works (specific papers)** — Per rules against suggesting missing references without external verification.
- **Request for confidence intervals on all Table 4 results** — Single-run evaluation is the norm in this field; the stochastic component comes from example set sampling, which is partially addressed for random variants.
- **"The length normalization in Eq. 5 is ad-hoc"** — While true, this is a standard technique in NLP for comparing probabilities across strings of different lengths; it's not a novel or unusual choice that requires specific justification.
- **Formatting/style nitpicks** — Removed per rules.

## Novel Insights

The REPLM paper reveals an important tension in evaluating in-context learning for extraction tasks: when the LM is large enough, it becomes extremely difficult to disentangle whether extracted triplets come from reading the specific document or from parametric knowledge. The paper's own evidence (Table 5: framework design provides genuine improvements at fixed backbone) suggests the retrieval and aggregation framework *does* add value, but the evaluation design cannot cleanly separate this from memorization on the headline datasets. This is a challenge that the broader ICL-for-IE community will need to address with controlled experimental designs.

## Suggestions

1. **Run REPLM with a fine-tuned Llama-3.1-8B** on DocRED training data as a direct same-backbone comparison. This single experiment would answer whether the in-context framework outperforms standard fine-tuning with the same model.
2. **Quantify inference cost.** Report API calls, tokens processed, and/or wall-clock time per document for REPLM vs. fine-tuned methods, so readers can assess the practical tradeoff.
3. **Scope the SOTA claims precisely.** State clearly in the abstract and introduction that SOTA is achieved within the class of methods that do not require NER or fine-tuning, and report absolute performance context alongside relative claims.
4. **Add a zero-shot ablation** to isolate the contribution of the retrieval and aggregation machinery.
5. **Either remove or substantially qualify the external KB evaluation** — at minimum, acknowledge that it measures KB consistency rather than document-grounded extraction.

## Score and Decision

**Calibration comparison:**
- **PromptNER** (WDQ9ZzsgDL, avg ~3.5, Reject): LLM-based few-shot IE with unfair model comparison, limited technical novelty. REPLM has more technical substance (framework with retrieval + aggregation + probability modeling) but shares the same core issue of unfair backbone comparison.
- **ReLiK** (b0IRscfEOb, avg ~5.7, Reject): Retrieval-based RE with SOTA results but limited baseline comparisons and limited novelty. REPLM is broader in evaluation but has deeper evaluation design problems.
- **GoLLIE** (Y3wpuxd7u9, avg ~6.25, Accept poster): LLM-based IE with real fine-tuning innovation, careful evaluation, thorough analysis. REPLM is less methodologically sound.
- **"When does ICL fall short"** (Cw6lk56w6z, avg ~4.5, Reject): ICL evaluation paper. REPLM is more constructive but has similar evaluation concerns.

REPLM has genuine technical contributions (the framework design is validated by ablations) and addresses a real gap (document-level ICL for RE). However, the core evaluation design conflates extraction with memorization, the external KB metric is fundamentally misaligned with the stated task, and the SOTA claims are overstated relative to the evidence. These are structural issues that undermine the paper's central claims. The paper is somewhat above PromptNER (which had essentially no technical novelty) but below ReLiK and well below GoLLIE.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>