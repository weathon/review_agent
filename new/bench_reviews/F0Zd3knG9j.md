Now I have all the information I need. Let me compile the final review.

## Summary

The paper introduces a hierarchical filtering procedure for tree-based generative models of sequences, where a parameter $k$ controls the depth of hierarchical correlations. Using this framework, the authors study how encoder-only transformers learn structured data on root classification and masked language modeling (MLM) tasks. They find that transformers (1) produce probability distributions closely matching Belief Propagation (BP) marginals even on out-of-sample inputs, (2) sequentially discover longer-range hierarchical correlations during training in a "staircase" pattern, and (3) develop attention maps and intermediate representations compatible with a layer-by-layer reconstruction of the hierarchy. They also provide a constructive existence proof that BP can be implemented within the transformer architecture.

## Strengths

- **The hierarchical filtering framework (Sec. 2.2) is a genuine and useful methodological contribution.** The parameter $k$ provides a clean experimental knob for controlling correlation range, and the out-of-sample testing protocol—training on $k_{\text{train}}$ and evaluating on $k_{\text{test}} \neq k_{\text{train}}$—is a clever design for isolating what correlation structure a model has learned (Figs. 3–4).

- **The sequential "staircase" learning dynamics are clearly demonstrated and robust.** Fig. 1(c)–(d) show the $D_{\text{KL}}$ between transformer predictions and $\text{BP}_k$ marginals sequentially aligning with decreasing $k$ as training progresses. Fig. 5 corroborates this with clean staircase accuracy curves on filtered test sets. This is a well-supported characterization of how transformers progressively incorporate longer-range correlations.

- **The calibration result for $k > 0$ root classification and MLM is genuinely non-trivial.** As the paper acknowledges (Sec. 3.2, p. 161), when $k > 0$ the root cannot be deterministically recovered, and the one-hot training labels do not correspond to BP marginals. That the transformer spontaneously produces calibrated probabilities matching BP marginals—on both in-sample and uniformly-sampled out-of-sample inputs (Fig. 1b, Fig. 11)—is meaningful evidence that the learned computation functionally approximates exact inference. The MLM calibration (Fig. 1b) is particularly compelling since the masked symbol is always genuinely uncertain.

- **The constructive existence proof (Sec. 4, Appendix E) resolves an architectural puzzle.** Showing that BP can be implemented in a single-head transformer with only $\ell$ blocks by exploiting $\mathcal{O}(q^2)$ memory slots in token embeddings to compute downward messages in parallel with the upward pass is a non-trivial insight, even if it does not establish what trained models actually compute.

- **The paper provides a clean explanation for why MLM pre-training helps supervised learning.** Fig. 1(f) shows MLM pre-training substantially reduces the labeled data $P^*$ needed for optimal root classification, and the probing results (Fig. 7) explain this mechanistically: MLM already causes the encoder to reconstruct hierarchical structure layer by layer.

## Weaknesses

### Fatal
None.

### Major

- **The claim of "equivalence in computation to the exact inference algorithm" (line 37) goes substantially beyond the evidence.** The paper's strongest evidence is functional—accuracy matching and probability calibration—not mechanistic. The attention map analysis (Fig. 6) is qualitative: hierarchical attention patterns would naturally emerge from any algorithm that successfully solves a hierarchical task, not just BP. The probing experiments (Fig. 7) show that ancestor information is *available* at corresponding layers, but information availability is a necessary consequence of successful prediction, not evidence for a specific computational mechanism. The existence proof (Sec. 4) establishes feasibility, not actual implementation. The paper itself acknowledges this regarding the constructive proof (line 211: "this does not represent an exact explanation of the trained transformer computation"), but then uses language like "evidence of an equivalence in computation" (line 37) and "spontaneously implement exact inference" (line 161). Without causal interventions (e.g., ablating attention from specific token blocks corresponding to BP message paths, or directly comparing hidden states to BP message vectors), the claim that transformers implement BP specifically—as opposed to some other hierarchical algorithm—remains unsupported. This matters because the mechanistic claim is presented as a central contribution and shapes the reader's interpretation of all other results.

- **The $k = 0$ root classification calibration is trivially expected, weakening a headline result.** With the non-ambiguity constraint (Sec. 2.1), the generative model is deterministically invertible from leaves to root when $k = 0$. BP outputs a point mass on the correct root, and a well-trained classifier also outputs a point mass—so their calibration match is trivially guaranteed. The paper partially acknowledges this (line 161: "While such a match is not entirely surprising in the deterministic $k = 0$ problem...") but still presents it as co-equal evidence alongside the genuinely non-trivial $k > 0$ and MLM results (e.g., Fig. 1(c) prominently displays $k = 0$ $D_{\text{KL}}$ evolution). The $k > 0$ root classification calibration IS genuinely non-trivial, and the MLM calibration is the paper's strongest evidence—but the $k = 0$ root case inflates the apparent strength of the calibration argument.

### Minor

- **All experiments use $q = 4$ with a single transition tensor (Sec. 3.1), limiting generalizability claims.** With non-overlapping entries and $q = 4$, there are at most 4 parent types for any pair of children, and the effective problem complexity is very low. The paper claims results are "qualitatively unchanged" for other tensors (deferring to Appendix D.2), but no variation in $q$ is tested. It remains unclear whether the clean staircase dynamics, calibration, and hierarchical attention patterns persist at larger vocabulary sizes where combinatorial complexity grows. This matters because the paper frames its conclusions as being about "how transformers learn structured data" generally, not just in a $q = 4$ toy case.

- **The non-ambiguity constraint means the generative model is not a typical PCFG with genuine ambiguity, somewhat undermining the CFG connection drawn in Sec. 2.3.** In standard PCFGs, the same child sequence can have multiple possible parents, and this ambiguity is where probabilistic inference is most needed and where BP's advantages are clearest. Removing this constraint would be the most meaningful test of whether transformers truly implement BP in the setting where it is most necessary.

### Trivial
None.

## Nice-to-Haves

- Causal interventions on attention patterns (e.g., ablating attention from specific token blocks corresponding to BP message paths and testing whether predictions degrade as BP would predict specifically) would decisively test the BP implementation claim.
- Experiments with larger $q$ (e.g., $q = 16$ or $q = 64$) would substantially strengthen generalizability.
- Comparing hidden representations at each layer directly to BP message vectors (rather than just probing ancestor prediction accuracy) would be far more decisive evidence for or against BP implementation.
- Testing on trees with ambiguous production rules (removing the non-overlapping constraint) would address the most meaningful setting for probabilistic inference.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "The filtering procedure is essentially equivalent to removing edges from a tree-structured graphical model—standard in the PGM literature. The novelty is overstated."** While the concept of truncating graphical models is standard, the specific application to create a tunable correlation knob for studying transformer learning is novel and useful. The contribution is in the experimental design, not the graphical model theory.

- **Harsh critic: "The connection to curriculum learning is speculative and not supported by experiments."** The paper mentions this only briefly in the conclusion (Sec. 5) as a possible future direction ("could perhaps be exploited to shape theory-driven curriculum learning strategies"), which is an appropriate level of speculation for a conclusion section.

- **Harsh critic: "The $k=0$ and $k=1$ attention maps looking similar is unsurprising given tree topologies differ only in transition probabilities at the top level."** The paper itself explains this (line 205: "the similarity between the $k = 1$ and $k = 0$ cases...is natural, the tree topology in these two cases being identical"), so this is not a missed insight but an acknowledged and explained observation.

- **Strength Finder: "The paper effectively combines multiple analysis angles—accuracy, full distribution matching, learning dynamics, attention maps, probing, and constructive implementation—building a coherent and mutually reinforcing story."** While true, this is too generic to qualify as a specific strength. Many papers combine multiple analysis angles.

- **Harsh critic demand for "comparison with alternative (non-BP) algorithms"** as a required experiment. While this would strengthen the paper, it's a high bar for what is primarily an empirical study. The out-of-sample testing already provides some discriminating evidence: matching $\text{BP}_{k_{\text{train}}}$ accuracy even on mismatched $k_{\text{test}}$ data rules out simple memorization. This is a nice-to-have, not a required experiment.

- **Harsh critic: "Analysis of failure modes: when the transformer fails to match BP, does it fail in ways consistent with a specific incorrect algorithm?"** This is an interesting direction but goes beyond the paper's stated scope and would constitute a separate study.

## Novel Insights

The interplay between the non-ambiguity constraint and the strength of the calibration evidence creates an interesting epistemic tension: the constraint makes the problem tractable enough for clean mechanistic analysis but simultaneously trivializes the most prominent calibration result ($k = 0$ root classification). The paper's strongest evidence actually comes from where the constraint is partially relaxed ($k > 0$) or rendered irrelevant (MLM, where the masked symbol is always genuinely uncertain regardless of $k$). This suggests that future work on verifying computational equivalence in neural networks should focus on regimes where the target algorithm's probabilistic inference is genuinely necessary, not where the problem is deterministic.

## Suggestions

- Moderate the mechanistic claims: replace "evidence of an equivalence in computation to the exact inference algorithm" with "evidence of functional equivalence in output distributions to the exact inference algorithm" and qualify the BP implementation claim as "consistent with" rather than "implements." This would make the claims match the evidence without diminishing the real contributions.
- Clearly separate the $k = 0$ root classification calibration (trivially expected) from the $k > 0$ calibration and MLM calibration (genuinely non-trivial) in the contributions, rather than presenting them as co-equal evidence.
- Add at least one experiment with $q > 4$ to demonstrate that the phenomena persist beyond the minimal setting.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Mechanistic basis of data dependence (oral) | aN4Jf6Cx69.md | 9.0 | Much stronger mechanistic analysis with phenomenological model and causal chain; our paper is well below this |
| Sudden Drops in the Loss (spotlight) | MO5PiKHELW.md | 7.75 | Has causal interventions on attention; our paper lacks these |
| SGD Finds then Tunes Features (spotlight) | HgOJlxzB16.md | 7.5 | Formal proofs of two-phase dynamics; our paper is empirical only |
| When can transformers reason with abstract symbols | STUGfUz8ob.md | 7.6 | Formal theoretical proofs of reasoning capabilities; our paper is below |
| Interpretability Illusions | v675Iyu0ta.md | 5.60 | Shows mechanistic interpretations can be illusory OOD; our paper has more positive contributions but similar overclaim risk |
| Circuit Probing | sZq3lDDETp.md | 4.20 | Criticized for correlation vs. causation in probing; our paper has stronger methodological contribution |
| Chess SAE interpretability | Wxl0JMgDoU.md | 2.50 | Narrow toy domain with overclaimed mechanistic claims; our paper is clearly above this |
| Metanetwork for mechanistic interpretability | 9L9j5bQPIY.md | 2.50 | Overclaimed from toy models; our paper has more systematic evaluation |

This paper sits above the medium-band rejected papers (4–6) because it has a genuine methodological contribution (the filtering framework), robust learning dynamics results, and genuinely non-trivial calibration evidence (especially MLM and $k > 0$). However, it sits well below the high-band papers (7+) because it lacks causal interventions, formal theoretical results, and its central mechanistic claim exceeds the evidence. The overclaimed "equivalence in computation" language and the inflated $k = 0$ root classification evidence are significant concerns that prevent a clear accept. The paper would be substantially stronger with more moderate claims or additional mechanistic evidence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>