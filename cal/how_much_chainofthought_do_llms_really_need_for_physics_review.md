=== CALIBRATION EXAMPLE 31 ===

# Harsh Critic Review
Now I have enough to write a thorough, section-by-section review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title asks "How Much Chain-of-Thought Do LLMs Really Need for Physics?" but the paper only answers this question obliquely. The abstract accurately describes the deletion framework and "cramming" observation. However, the claim that deleted equations and facts "often reappear" is immediately followed by the qualifier "but inconsistently across strategies" — these two characterizations are in tension, and the abstract does not reconcile them. "Often" and "inconsistently" cannot both be true without more nuance about conditions. The abstract also promises a "rigorous faithfulness analysis," but the analysis reduces to Bag-of-Words (BoW) overlap metrics, which is a methodological mismatch for a domain (physics) defined by the precise semantic content of equations, not their vocabulary.

---

### Introduction & Motivation

The introduction is well-framed and situates the work effectively in the CoT faithfulness literature, citing Turpin et al. (2023) and Lanham et al. (2023). However, the differentiation from these directly relevant prior works is critically underdeveloped. Lanham et al. (2023) *explicitly* study CoT faithfulness via content truncation and perturbation — a methodology strikingly close to this paper's deletion framework — and the introduction never explains what this paper offers *beyond* that work. The phrase "its implications for AI-for-Science remain underexplored" (para. 2) is asserted but not demonstrated. Why is physics fundamentally harder to analyze for faithfulness than, say, mathematics (where prior truncation/editing studies already exist)?

The three contributions (§1) are stated clearly, but contribution #2 ("cramming") and #3 ("faithfulness analysis") risk being restatements of each other with different metrics. The conceptual boundary between "cramming behavior" and "information overlap recovery" is not cleanly delineated.

---

### Problem Setup (§2)

**Models (§2.2):** The choice to restrict to open-source models is well-justified — intercepting CoT mid-generation requires inference access. However, the paper would benefit from at least one experiment testing whether the same deletion-robustness pattern holds when models are prompted without a scratchpad entirely (i.e., zero-shot direct answer generation), as this would ground the "cramming" interpretation: if models solve problems equally well with *no* CoT, then the 40–60% deletion robustness may simply reflect pre-training memorization of physics solutions, not genuine CoT dependence or reconstruction.

**Metrics (§2.4):** Several concerns:

1. *LLM-as-judge with Claude-4 Sonnet:* This introduces an unvalidated black-box evaluator as the primary metric. The paper provides no ablation comparing Claude-4's judgments against ground-truth labels, human raters, or symbolic answer checking (which is natural for physics problems with definite numerical or symbolic answers). This is a significant reliability concern. Furthermore, Claude-4 Sonnet is also used for physics-aware deletion tagging (§3.2), meaning the same model both constructs the deletion condition and grades its outcome — a circularity risk.

2. *Jaccard similarity and Manhattan-distance BoW overlap* are appropriate for short documents but poorly suited to physics reasoning. Physics equations share vocabulary (e.g., "F", "ma", "J", "kg", "m/s²") not because of recovery of *specific deleted steps* but because physics problems in the same domain naturally share this lexicon. A model that independently generates a force-balance equation will look like it "recovered" a deleted force-balance equation under these metrics even if it arrived at different physics. The paper's core faithfulness claim rests almost entirely on these metrics, making the analysis under-powered for its stated purpose.

3. *Final answer length* as a proxy for cramming is indirect. Longer answers could reflect the model compensating for missing context, or simply reflect that the model is less confident and outputs hedging language, or that the deletion changes the probability distribution over stop tokens. The paper does not rule out these alternatives.

**Sample sizes (§2.3):** The calibration study (§3.1) uses 50 UG-Physics questions with 5 re-runs. The paper states "approximately 5 prompts are sufficient to reduce the relative error bar below 10%." However, with nucleus sampling at T=0.6–0.7, variance in LLM outputs is non-trivial, especially at high deletion percentages where scores are already noisy. It is not clear whether the main deletion sweep experiments use the same 50-question subset or a different/larger set. This needs to be stated explicitly. For PhyBench (described as the hardest benchmark), the number of problems used is never directly reported.

---

### Experimental Results (§3)

**Prompting and Calibration (§3.1):** The finding that higher reasoning explicitness yields better accuracy is entirely expected and consistent with a large prior literature. This section largely replicates known results. The contribution here is primarily that it is used as a baseline — which is appropriate — but the paper dedicates significant space to this unsurprising finding.

**CoT Deletion Sweeps (§3.2):**

The three deletion strategies (end, random, physics-aware) are a good design choice. The X-shaped pattern (accuracy stable → drops; answer length monotonically increases) is the paper's central empirical result. However, several concerns arise:

- **Confound: The deletion directly affects the model's context window.** When the paper deletes 40% of a CoT trace mid-generation, the model decodes the final answer from a shorter context. It is well-established that LLMs respond differently to different context lengths. The accuracy stability could reflect the model simply not needing the late portions of a CoT it has already processed (the early tokens already established the solution path), not that CoT is generally bypassed. End-deletion specifically deletes the *most recently generated* content — which is often the final arithmetic/simplification steps — so accuracy holding for ≤40% deletion could just mean early conceptual steps are more important than late arithmetic steps. This is a key confound the paper does not address.

- **From-the-end vs. random deletion:** End deletion maintains contextual coherence (a prefix), while random deletion creates a fragmented, incoherent trace. The finding that accuracy is *more robust* to random deletion (60% threshold) than end deletion (40% threshold) is **never explained** and is actually counter-intuitive if the cramming interpretation is correct. If models are "reconstructing" lost content, it should be harder to reconstruct when the scratchpad is coherently truncated (end deletion) vs. fragmented (random), because the model can at least follow a coherent partial argument in the end-deletion case. This discrepancy is mentioned but not explained in §3.2 or §4.1, and it potentially challenges the cramming narrative.

- **Physics-aware deletion:** Using Claude-4 Sonnet to tag "physics-structured elements" introduces another layer of potential inconsistency — Claude-4 may identify physics elements at different granularities or with different coverage across problems, creating non-uniform deletion conditions. The paper does not report inter-annotator agreement or any validation of the tagging quality.

- **Missing baseline:** There is no condition where the CoT tokens are replaced by random/filler tokens rather than deleted. This would test whether accuracy drops are due to *loss of information* vs. *disruption of positional/sequential context*. Without this, the deletion framework cannot cleanly attribute performance changes to the absence of reasoning content.

---

### Analysis and Discussion (§4)

**Cramming (§4.1):** The cramming hypothesis is qualitatively compelling but the evidence is indirect. The paper acknowledges it "does not probe internal mechanisms directly" (§4.1) — but given this, the strength of the claim ("models may draw on internalized physics knowledge or learned solution templates") is overclaimed. The same observations are consistent with: (a) models solving from pretraining knowledge, (b) models using the surviving CoT prefix effectively, or (c) distributional properties of nucleus sampling that change answer boundaries. The paper should present these alternatives more seriously.

**Information Overlap (§4.2):** The finding that overlap increases with deletion is presented as evidence that models "recover" deleted content. But this is confounded: as more CoT is deleted, the model must generate more of its reasoning in the final answer section. An answer that is twice as long will trivially share more vocabulary with *any* physics text, including the deleted CoT, simply due to domain vocabulary overlap. The normalization strategy (if any) for answer length in the overlap computation is not described. Does Jaccard similarity account for the fact that longer answers will necessarily have more overlap? This is a fundamental methodological gap.

Moreover, the paper's conclusion that "reconstruction is heuristic and opportunistic rather than systematic" (§4.2) is based on the observation that overlap trends are noisy across deletion strategies. But noisy trends are exactly what Jaccard/BoW metrics would show for genuinely faithful recovery of *structured* mathematical content, because Jaccard treats "F = ma" and "a = F/m" as having similar overlap despite being different physics operations. A more faithful metric would compare whether the *same equations* appear (not just the same tokens).

**Implications for CoT Faithfulness (§4.3):** The paper draws a useful distinction between CoT being "informative" and "redundant" simultaneously. The suggestion that "early stopping of CoT generation may provide a cost-effective way to save tokens without proportionally sacrificing accuracy" (§4.3) is a practical insight. However, this is presented as an implication from the current work, but it has already been explored in budget-constrained CoT and CoT compression literature, which is not cited here.

---

### Related Work (§6)

The related work appears *after* the conclusion, which is an unusual structural choice. More importantly, the key prior work that must be differentiated — Lanham et al. (2023) "Measuring Faithfulness in Chain-of-Thought Reasoning" — is cited only in the introduction and never discussed in depth. That paper directly studies CoT faithfulness via truncation and paraphrasing experiments. The delta of this submission over Lanham et al. (2023) is the "physics domain" framing and the specific overlap analysis, but this differentiation is never stated explicitly. Additionally, "Turpin et al. (2023)" on biased reasoning is mentioned but not discussed in the context of how its findings compare with the current experiments.

---

### Limitations (§4.4)

The limitations section is honest and brief. It correctly identifies: (1) restriction to three models and one domain, (2) lack of mechanistic analysis, (3) need for broader validation. However, the section omits the most critical limitation: **the evaluation metric (LLM-as-judge) is unvalidated**, and the overlap metrics are not well-suited to capturing faithful physics reasoning. Additionally, the paper does not acknowledge the confound between CoT deletion and context-length effects on LLM decoding behavior.

---

### Overall Assessment

This paper addresses a genuine and important question — whether LLMs genuinely depend on their chain-of-thought traces in scientific domains — and proposes a concrete deletion-based probing framework applied to physics. The "cramming" observation (models generate longer final answers when CoT is deleted) is an interesting empirical phenomenon worth characterizing. However, the paper has substantial methodological weaknesses that undercut its conclusions. First, the core metric is a BoW overlap measure poorly suited to the semantic precision required in physics reasoning, making the "faithfulness analysis" insufficiently rigorous for its stated goal. Second, the use of Claude-4 Sonnet as both deletion tagger and judge introduces circularity and unvalidated noise. Third, and most critically, the key experimental results — particularly the counter-intuitive finding that random deletion permits greater robustness than end deletion — are not explained by the cramming hypothesis and potentially falsify it. The paper does not adequately differentiate its methodology and findings from directly relevant prior work (especially Lanham et al., 2023). For ICLR, the contribution in its current form is too incremental: the deletion framework is methodologically simple (essentially inference-time truncation/masking), the empirical findings are partially expected, and the most interesting anomalies in the data are left unexplained. Significant revision is needed — particularly stronger faithfulness metrics grounded in symbolic/equation-level matching, validation of the LLM judge, and a cleaner separation of the cramming hypothesis from the pretraining-memorization alternative — to bring this to ICLR standards.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a systematic deletion framework to evaluate the faithfulness of chain-of-thought (CoT) reasoning in large language models (LLMs) applied to physics problem solving. By intercepting and deleting intermediate reasoning tokens (random, end-of-context, and domain-aware), the authors demonstrate that models exhibit "cramming" behavior—reconstructing missing information in final answers to maintain accuracy—revealing a gap between explicit reasoning traces and actual computational dependence.

### Strengths
1.  **Domain-Specific Evaluation Rigor:** The paper effectively leverages the structured nature of physics (equations, units, constants) to quantify faithfulness, addressing a gap where prior CoT studies relied on general reasoning tasks that allow for more semantic flexibility. This makes the overlap metrics (Section 4.2) more meaningful than in open-ended reasoning domains.
2.  **Comprehensive Deletion Methodology:** The implementation of three distinct deletion strategies (from-the-end, random, and physics-aware deletion) allows for a nuanced understanding of how models rely on their scratchpads. The distinction between deleting "annotated" physics elements versus non-annotated text (Figure 3) provides valuable insight into the specific utility of domain knowledge within CoT traces.
3.  **Actionable Insights on Model Behavior:** The identification of "cramming"—where models increase final answer length to compensate for deleted CoT tokens—is a novel empirical finding with direct implications for system design and efficiency. It suggests that CoT generation might be partially bypassable, offering potential cost optimizations (Section 4.3).

### Weaknesses
1.  **Evaluation Reliability and Reproducibility:** The study relies heavily on an external proprietary judge ("Claude-4 Sonnet") for scoring solution quality (Section 2.4). While common in LLM research, using a non-open, potentially future-dated proprietary model as the ground truth for a paper evaluating open-source models introduces a reproducibility bottleneck and potential bias in the "faithfulness" metrics.
2.  **Implementation Ambiguity in Interception:** The core method involves intercepting CoT "mid-generation" and deleting tokens before decoding finishes (Section 2). The paper lacks sufficient technical detail on how the generation state is managed after token deletion (e.g., does the model re-compute attention over the truncated sequence, or is it a post-hoc text manipulation?). This ambiguity makes it difficult to verify if observed "cramming" is due to actual inference-time reconstruction versus artifact of the deletion process.
3.  **Metric Limitations for Semantic Faithfulness:** The overlap metrics rely on Bag-of-Words (Jaccard and Manhattan distance) (Section 4.2). While effective for detecting literal token reuse, these metrics fail to capture semantic equivalence. A model could faithfully recover a deleted equation using different syntax or notation that yields a high faithfulness score to the intent but low lexical overlap, potentially underestimating true reasoning recovery.

### Novelty & Significance
The paper offers significant empirical value to the ICLR community by shifting the focus from *performance* (accuracy) to *process integrity* (faithfulness) in scientific reasoning tasks. While deletion-based probing of CoT traces has appeared in prior work (e.g., measuring robustness), applying this specifically to the high-stakes, structured domain of AI-for-Science physics adds necessary rigor to the faithfulness discussion. The concept of "cramming" as a compensatory mechanism is a novel behavioral observation that challenges assumptions about CoT efficiency. However, the core methodological novelty is incremental; the paper excels in execution and domain application rather than introducing a fundamentally new algorithmic framework. It meets ICLR's standards for technical depth and clarity of contribution, particularly regarding the analysis of reasoning fidelity.

### Suggestions for Improvement
1.  **Clarify Generation Interception:** Provide a technical appendix detailing exactly how the decoder state is handled when tokens are deleted mid-generation (e.g., is the prompt truncated and fed back into the model for a continuation, or is the text simply removed post-hoc?). This is critical for reproducibility.
2.  **Improve Evaluation Robustness:** Supplement the proprietary judge with deterministic checks where possible (e.g., numerical verification of final answers against ground truth physics values) to mitigate reliance on a single black-box scoring model.
3.  **Expand Semantic Overlap Analysis:** Enhance the faithfulness analysis by incorporating semantic similarity metrics (e.g., embeddings-based cosine similarity on equation vectors) alongside lexical metrics to better capture whether the *logic* was recovered, not just the words.
4.  **Address Model Scope:** Acknowledge the limitation of testing only three models. If possible, include one additional model from a different architectural family (e.g., a larger MoE or a smaller distilled model) to ensure the "cramming" phenomenon is not specific to the training regimes of the selected models.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct Answer Baseline:** The paper claims CoT is partially redundant, but lacks a Zero-Shot Direct Answer baseline. Without comparing "60% Deleted CoT" accuracy against "No CoT" accuracy, you cannot claim the CoT was unnecessary; the model might perform equally poorly in both settings.
2. **Semantic Step Deletion:** Random token deletion breaks syntax (e.g., splitting equations), forcing models to guess regardless of reasoning dependence. You must delete entire *semantic reasoning steps* (e.g., remove the force diagram step entirely) to test logical dependence rather than syntax recovery.
3. **Rule-Based Verification:** Using Claude-4 Sonnet as a judge for physics correctness introduces shared LLM biases and verbosity preferences. You need a rule-based verifier (e.g., numerical equality, dimensional analysis) to ensure accuracy metrics aren't inflated by "crammed" verbose explanations that look correct but aren't.
4. **Length-Constrained Decoding:** To prove "cramming" is compensatory reasoning rather than just verbosity, you must enforce a strict token limit on the final answer during deletion sweeps. If accuracy collapses under length constraints while CoT is deleted, it proves the model relies on the answer section for computation.

### Deeper Analysis Needed (top 3-5 only)
1. **Necessity vs. Faithfulness Distinction:** The analysis conflates "models can answer without this text" (necessity) with "the text didn't represent the computation" (faithfulness). You must explicitly discuss how redundancy does not inherently imply unfaithfulness, as models may compute internally while outputting redundant text.
2. **Validity of "Crammed" Content:** Increased answer length is treated as evidence of reconstructed reasoning, but length $\neq$ logic. You need to analyze the *correctness* of the crammed equations specifically; if the extra tokens are verbose filler rather than valid derivations, the "reconstruction" claim fails.
3. **Error Propagation Breakdown:** The paper reports overall score drops but lacks a breakdown of *why* answers fail (e.g., unit error vs. conceptual error). Analyzing error types across deletion percentages would reveal whether deletion breaks specific physics logic or just general coherence.
4. **Early vs. Late Step Dependence:** The "from-the-end" deletion assumes later steps are less important, but in physics, final calculation steps are critical. You need to analyze whether deleting early conceptual setup vs. late algebraic manipulation has divergent effects on faithfulness.

### Visualizations & Case Studies
1. **Side-by-Side Trajectory Examples:** Show concrete pairs of Full CoT vs. 60% Deleted CoT outputs highlighting exactly where the model hallucinates a formula to fill the gap. This exposes whether the model is reasoning or merely pattern-matching physics terminology.
2. **Error Mode Distribution Plot:** A stacked bar chart showing the proportion of failure types (calculation, logic, formatting) across deletion sweeps. This reveals if deletion specifically degrades reasoning capabilities or just output formatting.
3. **Confidence vs. Accuracy Curve:** Plot the model's log-probability confidence for the final answer token against deletion percentage. If confidence drops while accuracy stays stable, it indicates the model is guessing correctly rather than reasoning reliably.

### Obvious Next Steps
1. **Internal State Probing:** Since you use open-source models, you should have analyzed hidden state representations (e.g., via probing classifiers) to see if deleted physics concepts remain encoded in latent space despite text removal.
2. **Cross-Domain Control:** Run the same deletion framework on a non-physics reasoning task (e.g., GSM8K or logical deduction) to determine if "cramming" is a domain-specific physics artifact or a general LLM behavior.
3. **Prompt-Induced Faithfulness:** Test if specific prompting instructions (e.g., "Do not repeat steps in the final answer") reduce cramming behavior. This would demonstrate whether the behavior is malleable or inherent to the architecture.

# Final Consolidated Review
## Summary

This paper introduces a deletion-based probing framework to evaluate whether large language models genuinely depend on their chain-of-thought (CoT) reasoning traces when solving physics problems. By intercepting CoT mid-generation and removing tokens (via end-deletion, random deletion, or physics-aware deletion), the authors find that models maintain accuracy under substantial deletions (40-60%) and exhibit "cramming" behavior—generating longer final answers that attempt to reconstruct missing reasoning steps.

## Strengths

- **Novel evaluation paradigm for scientific reasoning**: The deletion framework provides a principled way to probe CoT dependence beyond accuracy metrics, addressing a real gap in evaluating reasoning-focused LLMs for AI-for-Science applications.

- **Multi-strategy deletion design**: The three deletion strategies (from-the-end, random, physics-aware) allow differentiated conclusions about what content models depend on, with the physics-aware variant particularly suited to the domain.

- **Empirical discovery of "cramming" behavior**: The observation that models compensate for deleted CoT by producing longer final answers is an interesting empirical finding with practical implications for inference efficiency.

## Weaknesses

- **LLM-as-judge evaluation lacks validation**: The paper uses Claude-4 Sonnet as the primary correctness evaluator without validating against ground-truth numerical answers, symbolic verification, or human judgment. Physics problems typically have definite numerical or symbolic answers that could be verified deterministically. This introduces potential bias and reproducibility concerns, especially since Claude-4 is also used to tag physics elements for deletion.

- **Overlap metrics not normalized for answer length**: The Jaccard similarity and Manhattan distance metrics will naturally produce higher overlap for longer answers. Since the paper finds that answer length increases with deletion, the reported increase in information overlap may partially reflect this artifact rather than genuine recovery of deleted content. The paper does not describe whether or how length normalization was applied.

- **Missing zero-CoT baseline**: The paper concludes that CoT is "partially redundant" based on robustness to deletion, but does not compare against a direct zero-shot condition where models solve problems without any CoT. If models perform similarly with no CoT as with 40-60% deleted CoT, the interpretation changes substantially.

- **Random deletion outperforms end deletion without explanation**: The paper reports that accuracy remains stable until ~60% deletion under random deletion but only ~40% under end deletion. This counter-intuitive finding (random fragmentation being more robust than coherent truncation) is noted but not explained, and potentially challenges the cramming hypothesis—fragmented CoT should be harder to reconstruct than a coherent prefix.

- **Overlap metrics cannot capture semantic equivalence**: Bag-of-Words metrics treat "F = ma" and "a = F/m" as similar due to shared vocabulary, but these represent different algebraic operations. Faithful recovery in physics requires recovering the same equations and reasoning steps, not just vocabulary overlap. This limitation undermines the paper's core faithfulness claims.

## Nice-to-Haves

- **Clearer technical implementation details**: The paper states that CoT is "intercepted mid-generation" and tokens are deleted, but does not specify whether this is a KV-cache manipulation, prompt truncation with re-generation, or post-hoc text editing. A technical appendix would improve reproducibility.

- **Semantic-level step deletion**: Rather than token-level deletion, deleting entire reasoning steps (e.g., "force diagram identification" or "equation selection") would better test logical dependence on CoT content.

- **Cross-domain validation**: Testing the framework on mathematical reasoning (e.g., GSM8K) or logical deduction tasks would clarify whether cramming is physics-specific or general.

- **Error type breakdown**: Analyzing whether deletions cause unit errors, conceptual errors, or calculation errors would provide insight into what CoT actually contributes.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Often" and "inconsistently" are contradictory**: The abstract's claim that deleted content "often reappears" but "inconsistently across strategies" represents reasonable nuance about trends that vary by condition. This is not a genuine inconsistency.

- **Insufficient differentiation from Lanham et al. (2023)**: While the critic correctly notes that Lanham et al. also studies CoT faithfulness via truncation, this paper's contribution is the application to structured scientific reasoning with domain-specific deletion strategies. The differentiation exists, though could be stated more explicitly.

- **Contributions 2 and 3 overlap substantially**: Cramming (contribution 2) and information overlap analysis (contribution 3) address different aspects—one concerns answer length, the other concerns content recovery. These are distinct empirical observations.

- **Pre-training memorization as alternative explanation**: The critic suggests models may solve from memorized solutions rather than genuine reasoning reconstruction. While plausible, this is acknowledged in the limitations ("we do not analyze latent representations"). Demanding mechanistic probing is beyond scope.

- **Context-length confounds**: While deletion does change context length, the comparison across deletion strategies (random vs. end vs. physics-aware) helps isolate information content from length effects.

## Novel Insights

The paper's key insight—that models compensate for deleted reasoning by reconstructing content in final answers—challenges the assumption that CoT traces faithfully represent the computation underlying answers. The finding that physics-aware deletion causes accuracy to decline "gradually" rather than abruptly suggests models may rely on distributed physics knowledge rather than specific reasoning steps. The unexplained robustness to random vs. end deletion (60% vs. 40% threshold) raises an intriguing possibility: models may use CoT primarily as a working memory scaffold rather than as a faithful execution trace, with coherent prefixes being more valuable than fragmented content for maintaining reasoning state.

## Suggestions

- **Add deterministic verification**: For physics problems with numerical answers, implement symbolic or numerical verification alongside LLM-as-judge scoring to validate correctness judgments.

- **Include zero-CoT baseline**: Run direct answer generation without CoT to establish whether the 40-60% deletion robustness reflects genuine reasoning flexibility or simply that models can solve without CoT.

- **Normalize overlap metrics by answer length**: Report whether overlap increases hold after controlling for answer length, or use length-normalized metrics.

- **Explain the random vs. end deletion discrepancy**: Either provide a hypothesis for why fragmented CoT is more robust, or acknowledge this as a limitation of the current analysis.

- **Add semantic-equation matching**: Supplement BoW metrics with equation-level matching (e.g., using sympy equivalence or similar) to better capture physics reasoning recovery.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
