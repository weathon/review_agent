## Summary
HELM proposes a hierarchical cross-entropy pre-training objective that exploits the known codon/amino-acid hierarchy of mRNA, rewarding errors within synonymous codon groups less severely than errors that change the amino acid. The strongest evidence is a clean ablation (Table 2) showing HELM vs standard XE on matched architecture, tokenizer, and corpus, with consistent gains on seven downstream regression tasks plus an antibody region annotation task. The paper also analyzes when HELM helps most (datasets with strong synonymous codon usage bias, measured via entropy and CPB) and presents generative evaluations (FBD and property-preservation MSE).

## Strengths
- **Clean, well-controlled ablation isolates the method's contribution**: Table 2 compares HELM to XE under identical architecture (Transformer), tokenization (codon-level), and pretraining data (curated 15.3M OAS mRNA). HELM-MLM outperforms XE-MLM on every dataset; HELM-CLM outperforms XE-CLM on 6 of 7, demonstrating the hierarchical prior itself—not confounded design choices—drives improvement.
- **Biologically grounded analysis of when HELM helps**: Figures 2–3 and the accompanying entropy/CPB analysis show that gains are larger on datasets with stronger synonymous codon usage bias (MLOS, Tc-Riboswitches, mRFP have lower entropy / more negative CPB; Ab1, Ab2, iCodon show smaller gains). This moves beyond a black-box "it works" to a mechanism-informed understanding.
- **Minimal modification, broad applicability**: As described in Sec. 3, HELM requires only a loss-function change—no architectural or tokenizer modifications—making it easy to integrate into existing mRNA LM pipelines. The consistent comparison of tokenization strategies, architectures (Transformer, Mamba, Hyena), and objectives (MLM vs CLM) in Sec. 4.1 provides practical guidance for the community.
- **Addresses a genuine gap in mRNA-specific LMs**: The paper curates a large, high-quality OAS-derived mRNA corpus (15.3M sequences) and demonstrates the value of codon-level tokenization over nucleotide/6-mer approaches, helping establish baseline choices for a relatively underexplored modality.

## Weaknesses

### Fatal
None.

### Major

- **The generative evaluation is internally inconsistent: the random baseline has the best (lowest) FBD, directly contradicting the paper's headline generative claim.** The paper states "lower FBD = better" and then writes that "both the HELM model and the XE baseline significantly outperform the random baseline" (Sec. 4.3, line 218). Yet Figure 4 shows random at ~230, HELM at ~248–268, and XE at ~285–292—i.e., random is the best by this metric. Since the abstract, introduction, and conclusion all advertise improved generation quality ("producing diverse mRNA sequences that better align with the underlying true data distribution"), this is not a cosmetic issue. The FBD pipeline as presented does not support the generative contribution; without a corrected generation metric, the generative story is unreliable.
- **The claim of superiority over existing RNA foundation models is confounded by in-domain pretraining and modality mismatch.** The abstract/introduction frames HELM as outperforming "standard language model pre-training as well as existing foundation model baselines." However, Table 1 compares HELM's models—pretrained on a newly curated, antibody-mRNA-specific OAS corpus—against RNA-FM (non-coding RNA), SpliceBERT (pre-mRNA), and CodonBERT (proprietary mRNA model, different setup). Some baselines also cannot process certain downstream sequences (marked missing in Table 1). As the critic rightly notes, this comparison isolates domain/sequence-length advantages from the HELM contribution itself; the only fair isolation is HELM vs XE (Table 2). Broad claims of beating foundation models therefore overreach what the evaluation design actually demonstrates.

### Minor

- **Functional property preservation is evaluated via surrogate predictors, not ground-truth measurements.** In Sec. 4.3 (line 220), generated sequences are scored by pre-trained property prediction models, with MSE between predicted labels of generated vs. real sequences. This demonstrates agreement with a learned surrogate, not actual biological property preservation. Given that HELM is also evaluated through learned sequence representations elsewhere, there is a genuine risk of circularity: a generator that produces sequences favored by the probe looks better without being biologically better. The paper does not address this limitation.
- **The paper overstates scope when claiming to capture the "hierarchical nature of mRNA" broadly.** Sec. 3 formalizes a small, fixed codon tree (coding/non-coding → start/stop/amino acids → synonymous codons). This is a reasonable label hierarchy for synonymy-aware error costs, but the paper's framing ("hierarchical nature of mRNA," "global hierarchical structure of the genetic code") implies a richer model than what is actually implemented. The method does not model UTRs, secondary structure, regulatory context, or translation kinetics—all crucial to many downstream mRNA properties.
- **No statistical significance tests for HELM vs XE gains.** Tables 1–2 report single-run Spearman correlations. Several gains are modest (e.g., HELM-CLM vs XE-CLM on Ab2: 0.614 vs 0.597; COV-19: 0.789 vs 0.787), and it is unclear whether these represent robust effects or run-to-run noise. No variance, confidence intervals, or significance tests are provided.

### Trivial

- The paper claims HELM "consistently improves model performance" for both MLM and CLM in Sec. 4.2, but Table 2 shows HELM-CLM *loses* on MLOS (0.592 vs 0.611). This is a small overstatement that should be corrected.
- The abstract's claim of improvements on "seven diverse downstream property prediction tasks and an antibody region annotation task" is slightly confusing phrasing (the body treats it as 7 regression + 1 annotation, which is 8 tasks total). Minor clarification.
- The antibody region annotation task results appear only in the appendix (Appendix Table 9), with no quantitative accuracy numbers in the main body despite being advertised in the abstract.

## Nice-to-Haves

- **Nucleotide-level generative metrics alongside protein-level FBD:** Since HELM's core contribution is codon-aware mRNA modeling, evaluating generated nucleotide distributions, codon bias, and synonymous variation directly would be more informative than only protein-level embeddings.
- **Sensitivity analysis of the hierarchy weighting parameter α:** The effect of HELM hinges on how synonym errors are discounted via λ(C) = exp(−αh(C)), but no main-text sensitivity or rationale for α is provided. Including this would strengthen interpretability.
- **Case studies of actual generated sequences:** Showing real vs XE vs HELM continuations at the nucleotide level (synonymous substitutions, GC content, translation equivalence) would provide tangible evidence of whether HELM's gains are truly at the mRNA level.
- **Evaluation on non-antibody pretraining or mixed-domain pretraining** to establish HELM's generality beyond the OAS-centered setting.

## Removed Points

**These points are flagged to be removed, treat them with caution:**

- *"The comparison of tokenization, architectures, and tokenization results are entirely pushed to the appendix. The tokenization conclusion is central to later experiments yet entirely pushed to appendix."* — While true that the tokenization comparison is in the appendix, this is common practice and does not undermine the main contribution. Moved to nice-to-have.
- *"No variance, confidence intervals, or significance tests in Tables 1–2."* → Already kept as a Minor weakness. (Not fully removed.)
- *"The hierarchy itself has an odd presentation: 'non-coding codons' containing start and stop codons is a modeling choice, not an uncontested biological ontology."* → This is a legitimate observation about framing but is minor and does not harm the core contribution. Folded into the "overstated scope" point.
- *Criticisms demanding the missing appendix, missing proofs, or specific hyperparameter details* → Removed per hard rules; the parser strips appendices, and the paper describes α and its functional form adequately for the current scope.

## Novel Insights
One genuinely interesting observation is that the paper provides mechanistic grounding for *when* a hierarchical prior helps in mRNA modeling: datasets with strong synonymous codon usage bias (measured via entropy and CPB) show larger HELM gains. This goes beyond a pure empirical "it works" analysis and offers actionable guidance for practitioners—HELM is likely to be most beneficial on expression-level tasks (MLOS, mRFP) where codon optimization matters, and less critical on tasks with weak codon bias. The FBD contradiction, while a weakness, also points to a broader issue in generative bio-sequence evaluation: protein-level embedding metrics may not be sensitive enough to capture nucleotide/codon-level quality, suggesting the need for mRNA-specific generative benchmarks.

## Suggestions
1. **Correct or replace the FBD experiment.** The random baseline having the lowest FBD score fundamentally undermines the generative story. The authors should either (a) provide a corrected FBD computation, or (b) replace FBD with a more appropriate generative metric (e.g., nucleotide-level distribution matching, codon usage similarity, or a biological validity check). If the results cannot be corrected, the generative claims should be substantially toned down.
2. **Narrow the foundation model comparison claim.** Present the HELM vs XE comparison (Table 2) as the main evidence, and reframe the foundation model comparisons as evidence of domain-specific pretraining value rather than a direct superiority claim for HELM. Alternatively, add a fair comparison against an mRNA-adapted baseline trained on OAS with standard XE.
3. **Report multi-seed results or confidence intervals for HELM vs XE** to establish statistical significance of the (sometimes small) gains.
4. **Clarify the property-preservation experiment's limitations.** Acknowledge explicitly that surrogate predictor agreement does not equal functional property preservation, and consider adding at least one ground-truth validation if feasible.

## Score and Decision
Calibration anchors included: (a) high-scoring bio-LM papers (SaProt, CB-pLM, preference-to-fitness, all 6–8) which had broader evidence and cleaner claims; (b) borderline RNA/protein LM papers (RNA FrameFlow, RNAinformer, MSA seqs2seq, all 5–6) with solid methodology but evaluation/contribution concerns; and (c) overclaim-flagged papers (3–5 ranges) with confounded comparisons or contradictory results. This paper's core ablation (Table 2) is cleaner than rejected borderline anchors, and its method is straightforward and biologically motivated, placing it above the 3–3–3 rejection tier. However, the FBD contradiction directly undermines a main contribution (generation), and the foundation model comparison is confounded, preventing it from reaching the 7+ tier. The discriminative evidence is solid enough to keep it competitive but not strong enough for confident acceptance. Compared to RNA FrameFlow (5–6, rejected with methodology concerns) and similar borderline anchors, this paper is roughly at the same level—possibly slightly above due to the cleaner HELM vs XE ablation, but the FBD issue is more severe than typical borderline concerns.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>