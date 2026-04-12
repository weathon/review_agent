## Summary
This paper proposes Omni TM-AE, a Tsetlin Machine–based embedding method that derives word embeddings from the full TM state space rather than only literals whose automata states exceed the usual clause-inclusion threshold. The main claimed advances are: (i) reusability from a single training phase, unlike prior TM embedding approaches requiring retraining or multi-phase procedures, and (ii) improved embedding quality by incorporating information from “excluded” literals. Empirically, the paper shows competitive results on semantic similarity, sentiment classification, and clustering, with especially plausible evidence that the method can be a reusable interpretable alternative to standard static embeddings.

## Strengths
- **The paper identifies and operationalizes a genuinely specific idea: using sub-threshold TM states as embedding signal rather than discarding them.** This is more than a generic TM-for-NLP application. Section 3.4 and Algorithm 1 define a concrete embedding extraction rule from the state matrix, and Figure 3 makes the intuition tangible by showing that many semantically suggestive literals remain below threshold \(N\) even though they are ignored by prior clause-based use.
- **The single-phase reuse claim is specific and practically meaningful within the TM literature described by the paper.** The paper clearly contrasts prior approaches: Bhattarai et al. requiring full-vector retraining for altered token sets, and Kadhim et al. requiring a second phase to make token relationships directly usable. Omni TM-AE’s extraction from already-trained state matrices is a real conceptual simplification, not just a minor implementation tweak.
- **The method preserves a form of mechanistic traceability that most embedding papers do not offer.** The embedding coordinates are explicitly constructed from clause/literal states, and Section 6.2 explains how one can trace shared contributions across target words back to clause-level training dynamics. While the paper overstates interpretability in places, this is still a substantive advantage over opaque dense embedding models.
- **Empirical results support competitiveness, even if not dominance.** In similarity (Table 1), Omni TM-AE is close to the best average Spearman score and has the best average Kendall score; in classification (Table 2), it is essentially tied with the strongest static baseline and comparable to ELMo/BERT under the authors’ setup; in clustering (Table 3), it has the best average ARI, though not the best NMI. This does not prove broad superiority, but it does support the claim that the approach is viable.

## Weaknesses

### Major:
- **The central empirical validation of the paper’s core novelty is incomplete because there is no ablation isolating the contribution of excluded literals.** The main contribution is not merely using TM states, but specifically using *all* literals, including those below threshold \(N\). Yet the paper does not compare: included literals only vs. full-state Omni extraction, nor vary the thresholding or weighting of below-\(N\) states. As a result, the paper shows that the proposed method works, but not cleanly that the omitted-state information is what drives the gains.
- **Some of the headline comparative claims are stronger than what the evidence supports.** The paper repeatedly suggests it “often surpasses” mainstream embedding models or performs “on par with or better than black-box models.” The actual results are more mixed:
  - In Table 1, Omni TM-AE is competitive and strong, but FastText has the best average Spearman.
  - In Table 2, Omni TM-AE is slightly below Word2Vec on average.
  - In Table 3, Omni TM-AE has the best average ARI but trails Word2Vec/FastText on average NMI.
  A more accurate framing is “competitive overall with some wins,” not clear superiority.
- **The evaluation of predecessor TM baselines is incomplete at exactly the point where the paper needs it most.** The paper’s main positioning is as an improvement over prior TM embedding methods, but TM-AE is only reported on RG65, and the key predecessor from Kadhim et al. is excluded from experiments on practicality grounds. Even if those practical limitations are part of the point, the lack of a controlled restricted-scale comparison leaves the magnitude and source of improvement under-demonstrated.
- **The classification setup is non-standard enough that conclusions should be stated more cautiously.** Section 4.3 evaluates embeddings through a perturbation procedure that replaces 5% of tokens based on embedding neighborhoods, with asymmetric rules for positive and negative documents. This is an interesting stress test, but it is not a standard sentiment benchmark protocol, so the results should not be read as general downstream superiority. The asymmetry in the replacement policy also makes the table somewhat harder to interpret.
- **Interpretability is asserted more than it is demonstrated.** The paper gives intuitions, examples, and traceability arguments, but there is no systematic interpretability evaluation: no user-oriented analysis, no concept-level case study with success/failure criteria, no quantitative proxy such as clause sparsity/fidelity, and no demonstration that practitioners can reliably use these embeddings for diagnosis beyond anecdotal examples.

### Minor
- **The embedding dimensionality is very large.** Section 3.4 defines embeddings of size equal to the vocabulary \(d\), and the experiments use vocabularies up to 40,000. This is a meaningful practical trade-off versus 100-dimensional Word2Vec/GloVe baselines, especially for storage and downstream efficiency. The paper emphasizes scalability, but scalability here appears to mean training/reuse properties, not compact representation size; this distinction should be made explicit.
- **Important experimental details around document-level embedding construction are underspecified.** Section 4.4 says documents are represented by aggregating the word embeddings they contain, but the exact aggregation rule is not clearly described in the main text. Since clustering and likely classification performance can depend materially on mean vs. weighted mean vs. other pooling, this should be explicit.
- **The paper notes that original literal states compress into a narrow range at large vocabulary scale (Appendix/Figure 5), but does not quantitatively analyze whether these low-state regions carry robust semantic signal rather than weak noise.** Since the core method relies precisely on extracting information from these non-selected states, this deserves deeper analysis.
- **The BERT comparison is not entirely clear.** Section 4.3 describes “fine-tuning” settings for BERT, yet Table 2 presents BERT as an “Embedding Source.” It is not fully clear whether the reported numbers correspond to frozen embeddings, pooled representations from a fine-tuned model, or some hybrid pipeline. This ambiguity weakens the force of the comparison.

### Trivial
- **Algorithm 1 contains what appears to be a typographical/formula error on line 14** (`ei <- - vti`) relative to the text definition of \(e_i = v_i / t_i\). This is not a substantive flaw, but it should be corrected.

## Nice-to-Haves
- Add a direct ablation comparing:
  - only included literals (\(n > N\)),
  - all literals,
  - all literals but without negations,
  - and possibly weighted variants of below-threshold states.
- Provide a small controlled benchmark where Omni TM-AE, TM-AE, and the multi-phase predecessor are all run on the same restricted vocabulary/data regime.
- Quantify training cost and memory footprint more systematically, separating training cost from post-training embedding extraction time.
- Include at least one failure-case interpretability analysis showing how a poor similarity judgment, clustering error, or classification error can be traced back to specific clauses/literals.
- Clarify document embedding aggregation and the exact BERT pipeline in the main text.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that 32 clauses are mathematically incapable of representing a 40,000-token vocabulary, invalidating the method.** This overstates what is established from the paper. The paper uses a Coalesced TM with shared clauses and class-specific weights; from the paper alone one cannot conclude a fatal representational impossibility or “collapse.” The empirical results also do not support such a catastrophic-failure reading.
- **Claim that the evaluation is invalid because the vectors are “unnormalized integer counts” incompatible with similarity/clustering metrics.** The paper does not specify a mathematically invalid metric pipeline. Spearman/Kendall correlations evaluate ranked similarities, and nothing in the paper proves that their use is invalid for these vectors. It is fair to ask for more clarity on similarity computation or normalization, but not to declare the results invalid.
- **Criticism that negative clauses must be included for embeddings and that omitting them fundamentally biases the method.** The paper explicitly defines embeddings from positive-weight clauses as part of the proposed method. A reviewer may prefer an ablation including negative clauses, but there is no basis here to say the current choice is incorrect.
- **Concern that using a DGX H100 with large RAM contradicts efficiency/edge suitability.** The paper mentions TM hardware-friendliness in related work, not as a claim that these experiments were performed under edge constraints. Reporting the actual server used for experiments is not a substantive weakness by itself.
- **Reproducibility complaints about omitted seeds/splits/preprocessing minutiae.** Some details could be clearer, but these are not central enough to elevate given the paper’s current main issues.
- **Request for additional modern baselines not cited in the paper.** Omitted per instruction not to criticize missing related work/baselines that cannot be externally verified here.

## Novel Insights
The most important synthesis is that the paper is stronger as a **representation-extraction paper** than as a pure **state-of-the-art empirical paper**. Its real contribution is not that it convincingly beats Word2Vec/FastText/BERT across standard benchmarks—it does not—but that it turns a normally discarded part of TM training dynamics into a reusable embedding space with some preserved traceability and nontrivial competitive performance. The missing piece is causal validation: the paper argues that sub-threshold states contain useful semantic information, and the qualitative figures support that intuition, but the experimental section never cleanly isolates that mechanism. If the authors add this causal evidence, the paper’s central idea would become substantially more convincing.

## Suggestions
- Add a focused ablation where the only difference is whether below-threshold literals contribute to the embedding; this is the single most important missing experiment.
- Reframe the empirical claims from “surpasses mainstream models” to “competitive with standard static embeddings, with wins on some metrics and tasks.”
- Add a restricted-scale apples-to-apples comparison against prior TM embedding methods, even on one smaller dataset, to substantiate the claimed advance over TM-specific predecessors.
- Clarify the downstream pipelines: exact similarity function, document embedding aggregation, and whether BERT is used as frozen embeddings or as a fine-tuned end model.
- Strengthen the interpretability section with one concrete end-to-end case study, ideally including a failure mode, rather than only positive illustrative examples.

**Overall, the paper has a genuinely interesting and nontrivial idea with plausible empirical promise, but its current version does not yet validate the core mechanism as cleanly as the claims require.** Novelty is solid within the TM/NLP niche; technical soundness is reasonable but not fully established around the core attribution of gains; empirical support is competitive but incomplete; significance is promising for interpretable embedding research; and clarity is generally adequate, though some evaluation details and claim calibration need improvement.