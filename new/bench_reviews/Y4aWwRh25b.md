## Summary

This paper empirically investigates prompt-injected datastore extraction from Retrieval-In-Context (RIC) systems. It demonstrates high text-similarity-based reconstruction across nine open-weight instruction-tuned models (7B–72B), analyzes how RAG configuration choices (chunk size, chunking strategy, prompt position) affect extractability, proposes a position-bias elimination defense (PINE), and reports a 100% datastore-leakage success rate on 25 production customized GPTs with up to two queries, reconstructing 41.73% of a 77,000-word book.

## Strengths

- **Broad multi-model empirical demonstration.** The paper evaluates nine instruction-tuned LMs spanning 7B to 72B parameters and reports consistently high similarity scores under attack (e.g., Qwen1.5-72B reaches 99.15 ROUGE-L and 98.41 BLEU; Table 1). This breadth goes beyond prior work.
- **Systematic characterization of RAG design effects.** Controlled experiments provide concrete empirical findings on how chunk size, semantic-aware chunking, and prompt position modulate reconstruction (Figures 3–5), offering actionable insights for practitioners.
- **Real-world production GPT attack.** The demonstration on 25 customized GPTs achieves 100% leakage success (17 with a single query) and scales to reconstruct 41% of a full book (Section 4), showing practical applicability beyond open-weight models.
- **Explicit separation from training-data memorization.** By comparing newest Wikipedia articles (unlikely in training data) against Harry Potter text (plausibly memorized), the paper distinguishes retrieval leakage from simple regurgitation of training data (Table 2).
- **Fully specified, reproducible attack protocol.** Exact adversarial prompt templates and pipelines are provided, enabling reproducibility and auditing.

## Weaknesses

### Fatal
None.

### Major
- **Confounded base-versus-instruction-tuned comparison undermines a core causal claim.** The paper claims that “instruction tuning substantially enhances exploitability” (Section 3.1, Figure 2) by comparing base and instruction-tuned variants using the identical adversarial prompt format. Base models (~10–18 ROUGE-L) are not equipped with equivalent task comprehension (e.g., few-shot exemplars or a completion format suited to pre-training). The large gap may therefore reflect an inability to interpret the attack prompt rather than increased resistance to leakage. This confound invalidates the paper’s causal attribution that instruction tuning is a root cause of vulnerability.
- **Mitigation evaluated on an unseen model without utility metrics.** The proposed defense (safety-aware prompt + PINE, Table 3) is tested only on Llama3-8B-Instruct—a model absent from the main attack benchmark (Table 1)—and reports no legitimate-task accuracy, QA performance, or RAG utility metrics. A defense that suppresses reconstruction by simply breaking the model’s ability to answer questions is trivial; without utility validation, the claim that position-bias elimination “can effectively defend” is unsupported.

### Minor
- **Abstract overstates extraction as “verbatim.”** The open-weight experiments report similarity metrics (ROUGE-L, BLEU, BERTScore), not exact-match or character-level verbatim extraction rates (Table 1). High similarity does not guarantee literal verbatim reproduction, making this claim imprecise.
- **Mechanistic explanation of PINE is unclear and potentially backwards.** The paper asserts that grouping the adversarial user query with retrieved documents under bidirectional attention “reduces the likelihood of the model inadvertently following adversarial instructions” (Section 3.2.2). Because the adversarial prompt resides in the user query, one expects this grouping to couple the malicious instruction more closely with target data. The paper does not explain or evidence why eliminating recency bias via PINE reduces leakage rather than increasing it.
- **GPT reconstruction efficiency and cost are under-analyzed.** The 41.73% reconstruction of a 77,000-word book required ~75,000 words of raw output (100 queries × ~750 words) yet yielded only ~32,000 unique words (~57% retriever redundancy). The paper does not quantify chunk overlap or explain this saturation. Additionally, while the headline claims “at most 2 queries,” the main text does not transparently account for the initial system-prompt extraction step within this budget.

### Trivial
None.

## Nice-to-Haves
- A non-RAG baseline with equivalent in-context private text prepended without a retriever, to isolate whether the vulnerability is specific to RAG architectures or generic to long-context prompting.
- Side-by-side attack output examples to ground the quantitative similarity scores (e.g., what does ROUGE-L ≈ 80 look like in practice?).
- Evaluation of mitigations on the same models used in Table 1 with downstream task accuracy.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Criticism that position-bias experiments are “not practical.”** The paper explicitly states that this setting is “not a practical setting that’s adopted by current RAG systems” and frames it as a proof of concept (Section 3.1). This is a transparent scope limitation, not a hidden flaw.
- **Criticism that Harry Potter contamination speculation is unsupported.** The paper explicitly presents training-data contamination as a hypothesis (“lead to a hypothesis that they have been trained on Harry Potter”), not a firm conclusion (Section 3.1).
- **Criticism that the production GPT attack is an “oracle attack” rather than black-box extraction.** For Experiment 1, the paper uses the target GPT itself to generate domain questions (“Generate some questions specific to your knowledge domain”), which does not require prior knowledge of the datastore. For Experiment 2, the paper transparently discloses “partial prior knowledge” for the Harry Potter scenario. This is standard black-box reconnaissance, not an oracle attack.
- **Strength Finder’s claim of a “controlled ablation isolating instruction tuning.”** This strength directly conflicts with the verified major weakness that the base-versus-instruction comparison is confounded by comprehension differences. The weakness wins.
- **Claims about system-prompt extraction cost being excluded from “2 queries.”** The paper describes system-prompt extraction as part of the attack pipeline, and the “17 with 1 query / 8 with 2 queries” result refers to datastore leakage success; the step is present in the description, though its cost accounting could be clearer.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- Re-run the base-model comparison with few-shot exemplars or a completion-style format tailored to pre-trained models to disentangle comprehension from vulnerability.
- Test the proposed defenses on the same model suite used in Table 1 and report downstream QA or generation accuracy to verify that mitigations do not destroy legitimate RAG utility.

<context>
**Original reviewer signal**: The Harsh Critic argued serious evidential gaps confound key causal claims, mitigation is unvalidated, and GPT scalability is overstated, recommending rejection. The Strength Finder praised the empirical breadth, scale trends, and production-system applicability, viewing the work as a strong contribution. They fundamentally disagree on whether the gaps are fatal or addressable.

**What was dropped and why**: 
- The position-bias “not practical” critique was removed because the paper explicitly labels those experiments as non-practical proof-of-concept.
- The “speculative” Harry Potter contamination critique was removed because the paper transparently frames it as a hypothesis.
- The “oracle attack” characterization of the GPT experiments was removed because asking the target system about its own domain is black-box reconnaissance, not an oracle, and the paper discloses prior-knowledge assumptions where relevant.
- The Strength Finder’s claim that the base-vs-instruct comparison is a “controlled ablation” was dropped because it conflicts with the verified confound weakness.

**Cross-checks performed**:
- Re-read Section 3.1 / Figure 2: confirmed base and instruction-tuned models receive identical adversarial prompts without few-shot adaptation for base models, making the causal claim unsupported.
- Re-read Section 3.2 / Table 3: confirmed mitigation uses Llama3-8B-Instruct (absent from Table 1) and reports no utility metrics.
- Re-read Section 4 / Figure 6: confirmed 100 queries × ~750 words ≈ 75k raw output for a 77k-word book, yielding ~32k unique words, but no redundancy analysis is provided.
- Re-read PINE description (Section 3.2.2): confirmed the explanation does not clearly justify why grouping the adversarial query with retrieved documents reduces leakage.

**Severity read**: The two surviving major weaknesses are (1) a confounded causal comparison that weakens a core claim about instruction tuning as the root cause, and (2) an unvalidated mitigation tested on a different model without utility checks. Neither invalidates the overarching finding that RAG datastore leakage is possible across many models and production GPTs, but both seriously damage the paper’s causal attributions and practical recommendations. All other surviving weaknesses are minor presentation or explanation gaps.

**Anything else load-bearing**: The paper’s central empirical finding—that modern instruction-tuned LMs leak retrieved context under adversarial prompting—is well-supported across nine models and a production system. The work addresses a genuinely important and timely problem. However, the authors overreach in causal interpretation (instruction tuning) and defense validation. The field generally accepts single-run benchmark evaluation, so demanding confidence intervals would be scope creep for this community.
</context>