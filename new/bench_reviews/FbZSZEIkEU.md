## Summary

This paper investigates whether the well-studied Indirect Object Identification (IOI) circuit in GPT-2 small generalizes to prompt variants (DoubleIO and TripleIO) where both the subject and indirect object names are duplicated. The authors identify *S2 Hacking*—a mean-ablation artifact that causes the base IOI circuit to spuriously outperform the full model on these variants—and discover new variant circuits that reuse all base-circuit attention heads while adding input edges from duplicated IO tokens.

## Strengths

- **Identification and mechanistic tracing of S2 Hacking (Section 4, Figure 4).** The paper makes a valuable methodological contribution to mechanistic interpretability by showing that mean-ablation-based circuit evaluation can produce misleadingly high performance when input paths relevant to a new prompt format are excluded from the circuit subgraph. The tracing through Duplicate head 3.0, Induction heads 5.5/5.9, and S-Inhibition head 8.6 is clear and well-supported.
- **Systematic path-restoration experiments (Figure 5).** The authors systematically restore ablated input paths to the base circuit and show that adding paths from the IO2 (and IO3) tokens brings faithfulness closest to the full model, providing principled evidence for which input edges matter in the variants.
- **Quantitative node-reuse evidence (Section 5.2, Table 2).** The paper reports 100% node overlap between the base IOI circuit and the variant circuits, and Section 5.2 verifies this with direct causal effect scans over *all* model heads for Name Mover and Inhibition head roles.
- **Order-of-appearance discovery (Section 5.3, Figure 8).** The finding that Previous Token head 2.2 skews toward whichever name appears first, and that this correlates with performance differences, is an interesting extension that goes beyond the original IOI analysis.

## Weaknesses

### Fatal
None.

### Major
- **Misleading framing conflates base-circuit failure with generalization.** The abstract states that “the circuit generalizes surprisingly well,” and the introduction claims that “the base IOI circuit generalizes surprisingly well beyond its original task design.” However, the body of the paper shows the opposite: the *base* IOI circuit does not faithfully generalize to DoubleIO/TripleIO. Its high performance on these variants is due to S2 Hacking—an evaluation artifact arising from mean ablation of excluded input paths—not to genuine algorithmic generalization. The positive reuse results apply to *newly constructed* variant circuits with partially different edge sets. This conflation substantially weakens the paper’s central narrative and could mislead readers about what has been demonstrated.
- **Variant circuits leave ~23% of model behavior unexplained.** The discovered DoubleIO and TripleIO circuits achieve only 0.765 and 0.778 normalized faithfulness respectively (Table 2). The paper never analyzes what accounts for this gap—whether missing heads, missing edges, or nonlinear interactions—and therefore cannot fully support the claim that these circuits represent how the model “actually” solves these variants.

### Minor
- **Inconsistent description of the IOI algorithm.** The introduction (line 35) describes the algorithm as “remove all names that are duplicated,” while Section 3.1 describes it as suppressing the “most frequently duplicated token.” These are different algorithms with different implications: under the former, TripleIO would leave no names; under the latter, it would return the least-frequent name. This inconsistency slightly undermines the motivation for the TripleIO variant.
- **Non-overlapping edges are not characterized.** Table 2 shows that 8–15% of edges differ between the base and variant circuits, but the paper does not identify which internal (non-input) edges change or why. This leaves open the possibility that internal connectivity changes more substantially than the “only adding input edges” framing suggests.

### Trivial
None.

## Nice-to-Haves
- A decomposition of the missing ~0.23 faithfulness in the variant circuits (e.g., via ablation of excluded components) to clarify whether the circuits are merely incomplete or fundamentally wrong.
- Evaluation on structurally distinct prompt formats (e.g., passive voice, anaphora with pronouns, different syntactic frames) to support the broad claim that “circuits within LLMs may be more flexible and general than previously recognized.”
- A causal-effect heatmap for *all* heads on the final output logit for DoubleIO/TripleIO, analogous to Figure 6, to further substantiate the reuse claim for readers.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Claim of “no systematic, unconstrained causal scan over all model heads.”** This criticism is factually incorrect. Section 5.2 explicitly states: “For each attention head and relevant input token, we compute the direct causal effect of the path that starts from the token and proceeds through the head to the final logit” and “we compute the direct causal effects of every head in the model on the queries of the Name Mover heads.” The paper does verify node reuse with an unconstrained scan over all heads.
- **Request for completely unconstrained circuit discovery from scratch as a “missing experiment.”** The existing methodology already includes independent path patching over all heads; demanding a fully agnostic search (e.g., ACDC/EAP) is scope creep rather than a core flaw, though it would strengthen the paper.
- **Criticism that the 100% node overlap is “circular.”** While the edge-addition experiments in Section 5.1 are anchored to the base circuit, the node identities are independently verified in Section 5.2 via causal scans over all model heads. The circularity claim is therefore overstated.
- **Missing appendix content, missing proofs, or absent references.** The parser strips appendix sections from all papers; they exist in the original submission.
- **Formatting/style nitpicks, typos, or grammar issues.** These are parser artifacts, not author errors.

## Novel Insights

The S2 Hacking phenomenon is a genuinely novel and important observation for the mechanistic interpretability community. It reveals that standard mean-ablation-based circuit evaluation can create spuriously high performance on out-of-distribution prompts when relevant input paths are excluded from the subgraph. This has broad implications for how future work evaluates circuit generalization, and it is a more robust contribution than the circuit-reuse framing might suggest. The paper’s most lasting value likely lies in this methodological warning rather than in the quantitative overlap metrics.

## Suggestions
- Reframe the abstract and introduction to clearly distinguish between (a) the base circuit’s S2 Hacking failure on variants and (b) the variant circuits’ structural reuse of base-circuit components. Doing so would accurately represent the paper’s actual findings and strengthen its credibility.
- Add a paragraph analyzing the sources of the missing ~0.23 faithfulness in the variant circuits, or explicitly acknowledge this as an open question that limits the strength of the “actual solution” claim.

## Score and Decision

**Calibration anchors:**
- *High:* `5IWJBStfU7.md` (avg 7.0, Accept) — Identifiability in MI with exhaustive toy-model experiments and clear theoretical framing. This paper has a comparably important question and a real empirical phenomenon (S2 Hacking), but weaker framing and less exhaustive validation.
- *High:* `I4e82CIDxv.md` (avg 8.0, Accept) — Sparse feature circuits with comprehensive validation and practical applications. The paper under review is not at this level of completeness or impact.
- *Medium:* `P2gnDEHGu3.md` (avg 5.25, Reject) — Additive mechanisms in factual recall. Interesting experiments but unclear take-home message and limited dataset diversity. The paper under review has a clearer, more specific finding (S2 Hacking) and better mechanistic tracing, though it shares the issue of somewhat overclaiming relative to evidence.
- *Medium:* `vsU2veUpiR.md` (avg 5.25, Reject) — Mechanistic unlearning with strong empirical results but poor presentation and cherry-picking concerns. The paper under review has cleaner experiments but similar issues with framing.
- *Low:* `fM1ETm3ssl.md` (avg 3.0, Reject) — Meta-models for automated interpretability; very short and insufficient in scope. The paper under review is substantially more complete and empirically grounded.
- *Low:* `JZjW3k4Kyc.md` (avg 3.75, Withdrawn) — Circuit transformations with methodological concerns and unclear contributions. The paper under review demonstrates its core phenomenon (S2 Hacking) much more clearly.

**Comparison:** The paper under review sits in the upper-medium range. Its S2 Hacking finding is concrete, well-traced, and methodologically important—stronger than the core contributions of the 5.25-cluster papers. However, the misleading framing in the abstract/introduction and the ~0.77 faithfulness of the variant circuits pull it well below the 7.0+ papers, which present their claims more carefully and support them more exhaustively. A major revision that reframes the narrative and addresses the faithfulness gap could move this into the clear-accept range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>