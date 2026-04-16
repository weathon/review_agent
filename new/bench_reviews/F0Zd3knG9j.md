Now I have enough context from similar papers. Let me synthesize my final review.

Looking at the calibration papers:
- "How Language Models Learn CFGs" (qnbLGV9oFL): Rejected, scores 6/6/5/3 — similar topic of transformers learning structured data, probing approach
- "Interpretability of LMs for Learning Hierarchical Structures" (J6qrIjTzoM): Rejected, scores 6/8/3/8 — similar topic
- "Transformers Provably Learn Two-Mixture" (AuAj4vRPkv): Accept poster, scores 8/6/6/6 — theoretical/construction + experiments, similar to this paper's BP construction
- "Mechanistic basis of data dependence" (aN4Jf6Cx69): Accept oral, scores 8/8/10/10 — strong mechanistic analysis on synthetic data
- "JoMA" (LbJqRGNYCf): Accept poster, scores 5/6/6/6 — theoretical framework for understanding transformers
- "Make Haste Slowly" (27SSnLl85x): Accept poster, scores 8/5/3/8 — staged/dynamic learning on synthetic + real data
- "How Transformers Solve Propositional Logic" (eks3dGnocX): Rejected, scores 3/3/6/6 — mechanistic analysis on synthetic data, weak claims on toy setting
- "Transformers Learn Bayesian Networks Autoregressively" (4g0PUEAHg0): Withdrawn/Reject, scores 3/5/1/5 — theoretical construction + experiments but gap between construction and trained model

Now let me write the review.

The paper under review has:
1. A clean, well-designed data model (hierarchical filtering)
2. Interesting empirical findings (staircase learning, BP alignment)
3. An existence proof of BP in transformers
4. BUT: overclaims about "implementing BP" when evidence only supports functional equivalence on a tiny model
5. Very narrow experimental setting (q=4, ℓ=4, single tensor)
6. No discriminative tests between BP and simpler heuristics
7. Non-ambiguity makes k=0 nearly deterministic

This is similar to several papers in the 5-6 range that have interesting findings on synthetic data but overclaim mechanistic understanding. The existence proof + compatibility evidence isn't enough for "implements BP" level claims.

Score calibration:
- Papers with strong mechanistic evidence + broader evaluation (mechanistic basis, scores 8-10): This paper is well below these
- Papers with interesting synthetic findings but limited scope (JoMA at 5-6, How LMs learn CFGs at 5-6): This paper is in this range
- Papers that overclaim mechanistic understanding (Propositional Logic at 3-6): Some similarity but this paper is stronger

I'd place this around 5-6. The hierarchical filtering model is genuinely useful, the staircase learning observation is novel and interesting, but the overclaiming of BP implementation and narrow experimental scope are significant weaknesses. Let me settle on 5.5.</think>## Summary

This paper introduces a hierarchical filtering procedure for generative models of sequences on trees, enabling controlled truncation of correlation structure via a parameter k. Using this framework, the authors study how encoder-only transformers learn and implement inference on structured data, showing that (1) transformer predictions closely match Belief Propagation (BP) marginals, (2) learning proceeds by sequentially discovering longer-range correlations corresponding to deeper hierarchy levels, and (3) attention maps and probing results are compatible with a layer-wise implementation of BP within the architecture.

## Strengths

- **Well-designed hierarchical filtering model (Sec. 2.2):** The k-parameter provides a principled and tunable way to vary the range of hierarchical correlations while maintaining access to exact inference via BP. This is a genuine methodological contribution that enables clean causal experiments about what transformers learn and when—a valuable tool for the mechanistic interpretability community.

- **Compelling sequential learning dynamics (Figs. 1c-d, 4, 5):** The staircase pattern—where transformers trained on fully hierarchical data progressively align with BP_k for decreasing k—is a clear and novel empirical finding. The mismatched k experiments (training on k=0, testing on filtered data) are particularly elegant: the fact that test accuracy on filtered data first rises then drops to the mismatched BP prediction as longer-range correlations are learned provides strong causal evidence for the sequential discovery story.

- **Multiple converging lines of evidence:** The paper assembles accuracy matching, KL divergence tracking, calibrated probability matching on out-of-sample data, attention map analysis, and probing experiments—more evidence than typical in this genre for a synthetic model. The out-of-sample prediction matching (Fig. 1b, bottom panel) is especially noteworthy as it goes beyond task performance to functional equivalence.

- **Constructive existence proof (Sec. 4, App. E):** Demonstrating that a single-head transformer with ℓ layers can in principle implement BP through a specific weight construction is a neat theoretical result that establishes feasibility and provides a concrete mechanistic hypothesis to test against.

## Weaknesses

### Major:

- **Overstated claims about "implementing exact inference" / BP-algorithm equivalence.** The paper's central framing is that transformers "implement exact inference" and that there is "evidence of an equivalence in computation to the exact inference algorithm." What is actually demonstrated is *functional similarity*—the input→output mapping matches BP marginals on a single small grammar. Three distinct claims are conflated: (1) functional equivalence of predictions, (2) mechanistic similarity in information flow, and (3) algorithmic identity (transformer implements BP). The evidence supports (1) well and (2) partially, but not (3). The existence proof in Sec. 4/App. E shows BP *can* be implemented, not that SGD training *recovers* that implementation. The attention maps are "compatible" with BP but also with many other hierarchical aggregation schemes. The probing results show ancestor information is available at corresponding layers—consistent with any coarse-to-fine scheme, not specifically BP. This overclaim is structural, not cosmetic: the paper's core advertised contribution is identifying *how* transformers implement exact inference, but the evidence falls short of this.

- **Narrow experimental setting limits generality.** All main results use q=4, ℓ=4, a single randomly-sampled transition tensor, single-head attention, and n_L=ℓ layers. The non-ambiguity constraint (unique parent for each child pair) makes the k=0 case nearly deterministic, substantially simplifying inference and making the close logit-BP alignment less surprising for classification. The paper does not vary vocabulary size, tree depth beyond 4, number of attention heads, or use ambiguous grammars. Appendix D.2 claims qualitative robustness but presents no quantitative results in the main text. This narrowness undercuts claims phrased as generic statements about "transformers" learning "structured data."

- **No discriminative tests between BP and alternative hierarchical heuristics.** The paper moves from "predictions match BP" to "the model implements BP" without ruling out competing mechanistic hypotheses. A generic bottom-up pooling that aggregates blocks of 2, 4, 8 tokens with nonlinear statistics—without respecting the exact multiplicative BP recursion—could produce similar accuracy patterns and hierarchical attention maps. Simple tests that would discriminate these alternatives (e.g., perturbing transition probabilities and checking whether internal representations change in the specific way BP predicts, or probing for BP-specific intermediate quantities like upgoing/downgoing messages) are absent. The probing experiment (Fig. 7) only shows ancestor information is *present* at corresponding layers—not that it is *computed via BP messages*.

### Minor:

- **Quantification of calibration is incomplete.** The KL divergences in Figs. 1c-d are reported as averages over in-sample inputs with no variance or per-position breakdown. The scatter plots in Fig. 1b are qualitative. Given that calibrated probability matching is central to the BP-alignment argument, more rigorous quantification (e.g., ECE, Brier scores, worst-case deviations) would strengthen this key claim.

- **The non-ambiguity assumption is under-discussed relative to its implications.** The constraint that M_{abc}>0 implies M_{a'bc}=0 for a'≠a (Sec. 2.1) means that at k=0, the root is deterministically recoverable from the leaves. This makes the root classification task particularly easy and, as the paper itself notes, makes the one-hot training label coincide exactly with the BP marginal. The paper should more prominently discuss how this simplifies the problem and what changes under ambiguity.

- **Attention map analysis is purely qualitative.** No quantitative metric (structural similarity, mutual information with the BP factor graph, etc.) is used to assess how "tree-like" the attention patterns are. The claim that they "mirror the organization of the generative tree" rests on visual inspection of averaged attention maps.

### Trivial:

- The paper occasionally uses "exact inference" to mean both the mathematical algorithm (BP) and the claim about what the transformer implements, creating ambiguity. More consistent terminology would help.

## Nice-to-Haves

- Causal intervention experiments (e.g., ablating attention between specific token pairs corresponding to tree siblings and measuring the predicted effect on downstream predictions under BP) would provide the strongest test of the BP-implementation hypothesis.
- Testing on ambiguous production rules (overlapping transition tensors) would significantly broaden the impact and test whether the findings survive in more realistic probabilistic settings.
- Comparison of learned weights/representations with the constructed BP implementation from App. E (e.g., measuring distance in weight space or representation space).

## Removed Points

- **"Not yet released" / availability concerns about models or benchmarks:** Not applicable; the paper uses its own synthetic data model.

- **Demand for real-world data experiments:** While broader evaluation would strengthen the paper, the stated scope is understanding transformers on structured data through a controlled model. Demanding NLP or protein experiments is scope creep—the contribution is in the synthetic setting. Moved to Nice-to-Have only to the extent it would test generalizability of the core claim.

- **Missing related works:** Removed per instructions—I cannot verify existence of uncited works.

- **Formatting/style nitpicks:** Removed per instructions.

- **Demand for theoretical proofs of convergence/sample complexity:** The paper explicitly states "characterizing analytically the scaling of P* with the parameters of the grammar with our non-uniform transition probabilities is a challenging goal, and is left for future work" (Sec. 3.2). Demanding this is scope creep for an empirical/mechanistic paper.

- **Reproducibility concerns about undisclosed hyperparameters:** The paper specifies key parameters (d=128, d'=2048, n_L=ℓ, q=4, ℓ=4) and Appendix B gives the architecture. Minor implementation details not included are standard for this type of work.

- **"The gap between the proposed BP implementation and the trained model is never closed" (from Spark reviewer):** This is actually a valid major concern, already captured above as part of the overclaim weakness. However, the claim that the gap is *never* addressed is slightly overstated—the paper does show converging evidence (attention maps, probing, KL alignment) that is *consistent* with BP. The issue is that this evidence is compatible with other explanations too, not that there is no evidence at all.

## Novel Insights

The hierarchical filtering framework is genuinely novel and enables a clean experimental methodology: by varying k, one can observe *which* correlation levels the network has learned at each point in training, via the mismatched BP_k comparisons. This goes beyond the typical approach of measuring task accuracy over time and provides a more granular view of learning dynamics. The observation that the staircase learning pattern maps onto progressively lower-k BP algorithms is a novel finding that, while not proving BP implementation, does reveal the order in which different statistical structures are acquired.

## Suggestions

1. **Temper the claims:** The strongest defensible claim is that transformer predictions *functionally approximate* BP marginals on this model, and that internal representations are *consistent with* a hierarchical computation organized by depth. Replace "implements exact inference" with "approximates the function computed by BP" throughout, and clearly distinguish the existence proof from learned solutions.

2. **Add discriminative experiments:** Even one experiment that perturbs a specific BP message path (e.g., swapping two sibling tokens that should affect specific intermediate representations in the BP story) and shows the predicted effect on intermediate representations and outputs would substantially strengthen the mechanistic claim.

3. **Test at least one additional grammar realization and one larger setting (e.g., q=8, ℓ=5):** This would address the most basic generalizability concern with minimal additional experimentation.

## Score and Decision

**Calibration:** The paper under review is most comparable to:
- "How LMs Learn CFGs" (qnbLGV9oFL): scores 6/6/5/3, rejected — similar probing-based mechanistic analysis on synthetic language data, similar strengths (interesting synthetic findings) and weaknesses (limited generalizability, probing shows representation but not mechanism)
- "Transformers Provably Learn Two-Mixture" (AuAj4vRPkv): scores 8/6/6/6, accepted — has a theoretical construction plus empirical validation, but with stronger mathematical backing and clearer claims about what the construction shows
- "How Transformers Solve Propositional Logic" (eks3dGnocX): scores 3/3/6/6, rejected — mechanistic analysis on an overly simplistic synthetic task with overclaimed results
- "Mechanistic basis of data dependence" (aN4Jf6Cx69): scores 8/8/10/10, accepted oral — strong mechanistic analysis with causal evidence, cleaner claims

This paper has a genuinely valuable methodology (the filtering approach) and interesting empirical findings (staircase learning, out-of-sample BP matching). However, it overclaims the mechanistic conclusion—the evidence supports functional approximation, not algorithmic identification. The narrow experimental setting (q=4, ℓ=4, single grammar) is a real limitation. The paper is stronger than the "Propositional Logic" paper (which had an almost trivial task) but weaker than the "Provably Learn Two-Mixture" paper (which has a more rigorous theoretical analysis and makes more carefully scoped claims). It sits in a similar space to "How LMs Learn CFGs" but with a cleaner experimental framework.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>