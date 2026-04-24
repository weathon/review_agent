Now let me perform calibration searches to anchor the score properly.Now I have enough context to write the complete final review.

---

## Summary

HyperDAS introduces a transformer-based hypernetwork architecture that automates two key bottlenecks in mechanistic interpretability: (1) selecting which token positions in the residual stream mediate a concept, and (2) identifying a linear subspace for that concept. It is trained end-to-end with counterfactual supervision and a sparsity loss that bridges soft training-time alignment with hard test-time discrete selection. HyperDAS achieves a new state-of-the-art Disentangle score on the RAVEL benchmark (84.7 vs. 76.0 for MDAS), with a single multi-domain model also substantially outperforming the prior method (80.7 vs. 76.0).

---

## Strengths

- **State-of-the-art RAVEL performance (Table 3a):** HyperDAS (Asymmetric) improves average Disentangle from 76.0 → 84.7 over MDAS, with notable gains on City (+8.1), Physical Object (+4.3), and Verb (+9.5). The single All-Domains model (80.7) also substantially beats the prior method while serving all 23 attributes in one model. These are meaningful margins on a recognized benchmark.

- **End-to-end automated token-position discovery (Section 3.2, Figure 4):** Unlike MDAS, which hard-codes "last entity token," HyperDAS learns via the intervention-score matrix G to locate any token pairing. The empirical finding that 53% of cases require multiple tokens—and that deep layers (Layer 29) localize attributes to non-entity positions—is a genuine discovery that heuristic methods would miss.

- **Well-motivated training design (Section 3.5, Figure 7):** The sparsity loss and its calibration is carefully motivated and empirically demonstrated. The paper shows clearly that both extremes (no sparsity → many-to-one "hack"; too much → uninterpretable linear combination) fail at discrete evaluation, while the calibrated loss produces clean 1-1 alignments. This is a principled engineering contribution.

- **Memory efficiency argument (Section 4.2):** The comparison is concrete and well-quantified: HyperDAS (68 GB total for all 23 RAVEL attributes) vs. MDAS (110.3 GB for 23 separate models). This makes a legitimate practical case for the single-model paradigm.

- **Layer-specific analysis (Figure 4):** The shift from BOS/random tokens at shallow layers → entity tokens at middle layers → non-entity tokens at deep layers is a non-trivial empirical finding that complements prior work on entity localization in LLMs and opens new questions for future study.

---

## Weaknesses

### Fatal
None.

### Major

- **No ablation of the Householder subspace component against a fixed-subspace baseline.** The paper claims as a core contribution that HyperDAS "constructs features of those residual stream vectors" through dynamic Householder-based subspace identification. However, there is no experiment comparing HyperDAS against a variant where R^l is fixed (i.e., H = I or v set to a constant). The Figure 6 cosine-similarity matrix shows that cross-attribute Householder vectors have cosine similarities of 0.69–0.97—and while within-attribute similarities are higher (Figure 5 shows clear PCA clustering), the narrow margin raises the question of whether the Householder mechanism is doing meaningful concept-specific differentiation or collapsing to a near-shared subspace per entity type. Without this ablation, it is impossible to assess whether the RAVEL gains come from token selection alone or genuinely from adaptive subspace rotation. This concerns the validity of Contribution 2 in the paper's framing.

- **The Symmetric All Domains failure mode is unexplained and potentially undermines the symmetry argument.** Table 3a shows that Symmetric All Domains achieves near-zero Causal scores (2.0–21.6) with near-perfect Iso scores (94.7–99.3)—the hallmark of a degenerate "never intervene" strategy. The paper does not explain why enforcing symmetry during cross-domain training causes this collapse, nor what it implies for the symmetry principle advocated in Section 4.2. This is not a minor training artifact; it is a structural failure that exposes fragility in the optimization landscape, and the choice to use Asymmetric as the primary variant deserves a principled explanation rather than being left as an unexplained observation.

### Minor

- **Layer selection methodology is ambiguous.** The caption for Table 3a states "results from the best layer between 10 and 15." It is not stated whether this layer was chosen on a validation split or on the same test partition used for reporting. If selected on test data, this inflates reported scores. Since MDAS follows the same procedure, the comparison may still be fair, but the paper should clarify.

- **Faithfulness concern is raised but left empirically unresolved.** The paper discusses at length the risk of "injecting rather than uncovering" causal structure and lists several design mitigations (masking, sparsity, symmetry, Iso metric). However, no experiment directly tests whether the discovered subspace and token positions recover pre-existing causal structure vs. introduce out-of-distribution steering. The paper's own Limitations section acknowledges this, but as the discussion is foregrounded in Section 4.2 as a key contribution, readers will reasonably note the gap between the rhetorical emphasis and the actual empirical support.

- **Single target model limits scope.** All experiments use Llama3-8B. Whether the discovered token positions and subspaces generalize across architectures (e.g., Mistral, Gemma) is unknown, and the paper does not frame this as a deliberate scope limitation.

### Trivial

- The "automation" framing in the abstract is slightly imprecise. The method learns to predict token positions for training-set concepts at inference time (amortized search), but requires training supervision on each concept. The automation is over inference-time brute-force search, not over unseen concept generalization. A more precise framing would strengthen the contribution.

---

## Nice-to-Haves

- An ablation comparing Householder-adapted subspace vs. fixed subspace (R = R^l, H = I) would directly test whether the dynamic subspace identification contributes independently of the token-position selection module.
- A deeper analysis of the Symmetric All Domains failure: is this a training stability issue, a fundamental tension between cross-domain and symmetry constraints, or an optimization landscape degeneracy?
- An experiment reserving a subset of RAVEL attributes from training to evaluate whether HyperDAS generalizes to held-out attribute types, which would sharpen the "automation" claim.
- Comparison against DAS applied exhaustively over all token positions and layers (the "brute-force" upper bound that HyperDAS claims to amortize) would directly quantify the cost-performance tradeoff.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Automation claim not tested on unseen concepts"** (from harsh critic): The paper's "automation" framing refers explicitly to automating the brute-force search over token positions and layers given a training distribution of concepts—not zero-shot generalization to novel concepts. The abstract says HyperDAS "automatically locates the token-positions" conditioned on concept descriptions seen during training. Criticizing the paper for not demonstrating zero-shot generalization to entirely new concepts imposes a scope well beyond what the paper claims. Retained as a Nice-to-Have rather than a weakness.

- **"High Householder cosine similarities prove the component is useless"** (from harsh critic's framing): The cosine similarity heatmap in Figure 6 does show high cross-attribute similarities (0.69–0.97), but the PCA in Figure 5 shows clear per-attribute clustering, and the paper correctly notes within-attribute similarities are higher than cross-attribute. The reviewer's inference that the Householder component "may not contribute meaningfully" is reasonable motivation for an ablation study, but the Figure 6 analysis alone does not prove uselessness. The call for an ablation is retained as a Major weakness; the stronger claim that Figure 6 provides "positive evidence against" the component is removed.

- **"Memory comparison is unfair"** (from harsh critic): The comparison with MDAS trained separately per attribute is the fair and obvious baseline—this is the common deployment scenario. Multi-attribute MDAS with shared representations would be a different method, not the reported baseline. Removed.

- **"JSON syntax tokens at deep layers are alarming"** (from harsh critic, mildly mischaracterizing Figure 4): Figure 4 shows ~32% "Others" at deep layer base, not specifically "JSON Syntax" (which is 0% in the data table). The paper text refers to JSON syntax tokens as an example of unintuitive positions found at deep layers, but this appears to be a textual illustration rather than what the bar chart quantifies. The core observation (32% non-entity tokens at deep layer base) is real and interesting; the harsh characterization of it as "alarming" and a faithfulness failure is unsubstantiated. Retained only as part of the minor "deeper analysis needed" note.

---

## Novel Insights

The paper's most genuinely novel empirical observation—that different layers in Llama3-8B localize entity attributes at structurally different token positions (entity tokens at middle layers, non-entity positions at deep layers), which prior fixed-token methods would miss—is an independent finding that could have significance beyond the HyperDAS architecture itself. The dynamic token-position discovery essentially provides a "layer-by-layer map" of where concepts reside in a large LM's residual stream, and the finding that deep-layer representations mediate attributes via non-entity tokens is a non-trivial empirical contribution to the mechanistic interpretability literature on entity knowledge storage.

---

## Suggestions

- Add an ablation: train HyperDAS with H=I (identity Householder, no dynamic subspace rotation) and compare RAVEL scores. This is a small experiment that would validate or challenge the subspace identification component's contribution.
- Explain the Symmetric All Domains degenerate solution more thoroughly. Is it the symmetry constraint creating an optimization conflict at scale, or a training instability? Understanding this is important for interpreting the "symmetry as faithfulness indicator" argument.
- Clarify whether the "best layer between 10 and 15" is selected on a validation set or test set.
- Consider adding even a single held-out RAVEL attribute as a generalization test to sharpen the automation framing.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Comparison |
|---|---|---|
| "Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?" (5IWJBStfU7) | 7.00 | More theoretically grounded MI analysis; accepted poster. HyperDAS has stronger empirical SOTA result but missing ablation. |
| "The Geometry of Categorical and Hierarchical Concepts in LLMs" (bVTM2QKYuA) | 6.75 | Strong empirical + some theory on representations in LLMs; accepted oral. HyperDAS comparable in empirical strength but narrower in scope. |
| "How do Language Models Bind Entities in Context?" (zb3b6oKO77) | 5.50 | Causal experiments on entity binding in LLMs, accepted poster. HyperDAS is more methodologically novel (end-to-end optimization) but also has more open questions. |
| "How does controllability emerge in LMs during pretraining?" (egHptuv7hx) | 5.50 | Representation intervention paper with steering vectors; rejected. HyperDAS is more rigorous in evaluation and has cleaner results. |
| "Toward Faithfulness-guided Ensemble Interpretation" (L7jtdGhWzT) | 4.67 | Interpretability method that raises but doesn't resolve faithfulness concerns; rejected. HyperDAS has more substantial empirical contributions. |
| "Identifying Interpretable Features in CNNs" (FVItLat5ii) | 4.00 | Low-scoring interpretability paper with insufficient contribution. HyperDAS is substantially stronger. |

**Positioning:** HyperDAS has real contributions—SOTA on a recognized benchmark, a novel and technically sound architecture, and genuinely informative layer-by-layer analysis. It sits above the "How do LMs Bind Entities?" (5.50) cluster because of higher methodological novelty and stronger empirical gains. It sits below the "Mechanistic Interpretability Identifiable?" (7.00) cluster because of the missing Householder ablation (one of two core claims is unverified) and the unexplained Symmetric All Domains failure. I place this at **5.5** — a paper with real contributions that merits serious consideration but has two substantial gaps that reviewers would rightfully flag. The paper is not reject-level (its core empirical claim holds) but the unverified subspace claim and unexplained failure mode prevent a clean acceptance recommendation at the current state.

**Score: 5.5 — Weak Accept (borderline, leaning toward revision needed)**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>