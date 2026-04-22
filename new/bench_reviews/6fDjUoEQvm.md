Now I have a thorough understanding of the paper and all the claims. Let me write the final review.

## Summary

HyperDAS proposes a transformer-based hypernetwork architecture that automates two steps in causal interpretability — selecting token positions and identifying feature subspaces — that previously required manual search in methods like DAS/MDAS. Using Householder transformations to dynamically construct orthogonal subspaces conditioned on concept descriptions, and a learned token-alignment mechanism, HyperDAS achieves a new state-of-the-art on the RAVEL benchmark for disentangling entity attributes in Llama3-8B.

## Strengths

- **State-of-the-art performance on RAVEL**: HyperDAS-Asymmetric achieves 84.7 average Disentangle score, substantially outperforming the prior SOTA (MDAS at 76.0), with gains on every domain (Table 3a). This directly validates that automating token-position selection improves over fixed-position baselines.

- **Elegant architectural design for dynamic subspace construction**: The Householder transformation (Eq. 10) to conditionally rotate a fixed orthogonal matrix into an attribute-specific subspace is a clean mechanism that guarantees orthogonality by construction, removing the need for explicit orthogonality penalties during training (Section 3.3).

- **Layer-dependent intervention analysis reveals new findings about LLM internals**: Figure 4 shows that HyperDAS selects entity tokens at middle layers (98.7% for middle-layer base interventions) but shifts toward syntactic/JSON tokens at deeper layers (~32% non-entity at layer 29), providing evidence that attribute storage shifts across layers and challenging the "last entity token" assumption (Section 4.1).

- **Honest discussion of the faithfulness concern**: Section 4.2 explicitly acknowledges that powerful supervised interpretability methods risk "hacking evaluations without uncovering actual causal structure" and takes concrete steps (masking base-prompt attributes, sparsity loss) to mitigate this. While the addressal is incomplete (discussed below), this transparency is unusual and valuable.

- **Memory efficiency at scale**: For multi-attribute settings, a single HyperDAS model handles all attributes (68GB total) versus 110.3GB for MDAS (which requires separate models per attribute), a practical advantage (Section 4.2).

## Weaknesses

### Fatal
None.

### Major

- **Overstated automation claim undermines framing**: The abstract states HyperDAS "automatically locates the token-positions of the residual stream that a concept is realized in," and the introduction motivates the method as addressing the need to "conduct a brute force search over potential feature locations." However, HyperDAS only automates token-position selection *within a pre-selected layer* — layer selection remains manual. Table 3a reports results at "the best layer between 10 and 15," which is itself a brute-force layer sweep. The actual contribution is reducing the search space from (layers × tokens) to (layers), not eliminating it. This gap between framing and delivery is significant because the paper's primary stated motivation is to automate the search.

- **Evidence for faithfulness (interpretation vs. steering) is insufficient**: The paper's central claim — that HyperDAS uncovers genuine causal structure rather than injecting new information — is structurally under-supported. (1) Householder vector clustering (Fig 5-6): clustering by attribute is expected given that the hypernetwork receives the attribute label as input; it confirms the conditional mechanism works, not that it mirrors model structure. Moreover, cross-attribute cosine similarities range from 0.69–0.90 (Fig 6), far from truly orthogonal subspaces despite the orthogonality constraint. (2) The asymmetric variant selects different tokens for the same input depending on whether it is the base or counterfactual (Fig 8), suggesting the method finds context-dependent intervention sites rather than stable model-internal representations. (3) No comparison with brute-force DAS at the same layer to test whether HyperDAS recovers the same subspaces, which would be the most direct faithfulness test. The paper acknowledges these concerns qualitatively (Section 4.2) but does not provide controlled experiments that discriminate interpretation from steering.

### Minor

- **Notation inconsistency in sparsity loss**: Equation 13 defines L_sparse over entries of G, but Section 3.2 applies column-wise softmax to produce G (line 109). After column softmax, each column sums to 1, making the condition Sum(G_{(*,c)}) > 1 in Eq. 13 impossible to satisfy and the loss identically zero when applied post-softmax. The loss likely operates on pre-softmax G^i, but the notation is misleading and should be clarified for reproducibility.

- **Dimensional inconsistency in Householder transformation**: Section 3.3 states that R^l is d×k with orthogonal columns and H is d×d, then writes R = R^l · H. But d×k multiplied by d×d is dimensionally invalid; the intended formulation should be R = H · R^l. While the mathematical intent is clear, the notation as written is incorrect.

- **Catastrophic failure of Symmetric-AllDomains is unexplained**: Table 3a shows the Symmetric-AllDomains variant achieving only 54.8% average Disentangle score, with Cause scores as low as 2.0% (Nobel laureates) and 16.8% (cities). This is the most natural multi-task setting (one model handling all attributes) and its failure is not discussed, limiting the practical value of the single-model approach.

- **Train-test gap between soft and hard interventions underexplored**: Figure 7 shows that all three sparsity regimes achieve ~94% Disentangle score with weighted (soft) interventions, but only one regime produces meaningful hard alignments. While the main table uses hard alignments (so headline numbers are not inflated by this), this gap means the training objective does not necessarily enforce faithful localization, and the soft-intervention metric cannot discriminate genuine from artifact-based localization.

### Trivial
None.

## Nice-to-Haves

- A controlled experiment comparing HyperDAS's discovered subspaces with brute-force DAS at the same layer would directly test faithfulness and substantially strengthen the paper.
- Automating layer selection (e.g., via a learned layer router or softmax over layers) would close the gap between the stated motivation and the method's actual scope.
- Reporting both weighted and hard-alignment Disentangle scores in the main table would reveal whether performance depends on the discretization step.
- Analysis of failure cases — where HyperDAS gets the wrong answer and whether the intervention locations still look "reasonable" — would further illuminate the interpretation/steering distinction.

## Removed Points

- **"The paper should compare with a steering/editing control baseline" (Harsh Critic #3 in Missing Experiments)**: While informative, this asks for an experiment outside the paper's stated scope of interpretability method evaluation on an established benchmark. A steering baseline that matches HyperDAS performance would indeed be informative but goes beyond what the RAVEL protocol tests. Demoted to nice-to-have.

- **"Formatting and notation nitpicks"**: Several notation issues raised by the harsh critic (the dimensional inconsistency in R = R^l · H, the sparsity loss notation ambiguity) are substantive enough to keep as Minor weaknesses, but purely presentational complaints are removed.

- **"The paper claims to review design decisions to mitigate faithfulness concerns but Section 4.2 is largely qualitative"**: The paper does review design decisions (masking, sparsity loss, symmetry analysis), and Section 4.2 presents quantitative results from these. The critique that controlled experiments are missing is kept (in Major), but the broader claim that the section is "largely qualitative" is removed as it understates the paper's actual contributions.

- **"Questioning existence of cited models/tools/benchmarks"**: Removed per hard rules. The paper cites Llama3-8B, RAVEL, MDAS, DAS — all assumed to exist.

- **"Masking of base prompt undermines automation"**: The masking is a necessary design choice to prevent a trivial solution (the paper explains this in Section 3.5), not an indicator that the automation claim is false. The masking uses known attribute names in the base prompt, which is different from knowing which tokens encode them, so it doesn't trivialize the localization task.

## Novel Insights

The asymmetric variant's behavior — selecting the last entity token from counterfactual prompts but the second-last from base prompts (Fig 8) — suggests a "read vs. write" asymmetry in how LLMs store and retrieve attribute information, consistent with the write-then-read hypothesis for entity tokens. This is an interesting finding that goes beyond the paper's immediate claims and could motivate future work on how information flows differ depending on whether a representation is being accessed or modified.

## Suggestions

- Add a direct comparison with brute-force DAS at the same layer to validate that HyperDAS recovers the same causal structure, which would conclusively address the faithfulness concern.
- Report the hard-alignment (discrete) Disentangle scores alongside weighted scores in the main table, or at least in a supplementary analysis, to show the training objective is well-aligned with evaluation.
- Discuss the Symmetric-AllDomains failure case explicitly — understanding why multi-task learning collapses for some domains would strengthen the paper and guide practical deployment.
- Clarify whether L_sparse operates on pre-softmax G^i or post-softmax G, and correct the R = R^l · H notation to R = H · R^l.
- Moderate the automation framing in the abstract and introduction to accurately reflect that layer selection remains manual.

---

**Calibration anchors used:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| "Do I Know This Entity?" (SAE knowledge awareness) | 9.0 | Stronger faithfulness evidence (causal steering actually validated), clearer interpretability claims; HyperDAS has weaker faithfulness evidence |
| "Not All LM Features Are One-Dimensionally Linear" | 7.0 | Comparable methodological rigor; HyperDAS has SOTA benchmark results but overclaims automation |
| "Dynamic Model Editing" (attribution-based editing) | 5.0 | Similar localization+intervention theme; HyperDAS is more thorough in evaluation |
| "Everything Everywhere All at Once" (MI identifiability) | 7.0 | Theoretical contribution to faithfulness question; HyperDAS is empirical but doesn't address identifiability |
| "Towards Meta-Models for Automated Interpretability" | 3.0 | Far weaker; HyperDAS is far more substantial in scope and evidence |
| "A is for Absorption" (SAE feature absorption) | 7.5 | Similar concern about faithfulness of interpretability methods; HyperDAS is less complete in addressing this |

HyperDAS achieves genuine SOTA improvements with a creative architecture and meaningful analysis, but its two main claims — full automation of localization and faithful interpretation rather than steering — are overstated relative to the evidence. The automation is partial (layers still manual), and the faithfulness evidence is suggestive but not conclusive. These are real contributions weakened by overclaimed scope, not fatal flaws. Positioning above the model editing paper (5.0) and the rejected interpretability papers (~3-5), but below the accepted interpretability papers with stronger faithfulness evidence (7.0+).

## Score and Decision

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>