## Summary
This paper introduces a framework for compositional meta-learning by learning a probabilistic generative model of tasks. The model separates within-module dynamics (via module RNNs) from between-module sequencing (via a gating RNN). After training, new tasks are solved through probabilistic inference (particle filtering) without parameter updates, enabling rapid one-shot learning and handling of sparse feedback. The approach is validated on synthetic rule-learning and motor-skill tasks where it recovers ground-truth components and generalizes to longer sequences.

## Strengths
- **Novel integration of compositional structure with probabilistic inference:** The paper's core contribution is a principled formulation that replaces gradient-based adaptation on new tasks with inference over a learned generative model. This combines the expressivity of RNNs with the data efficiency of Bayesian inference, enabling parameter-free task solving—a distinct advance over standard meta-learning.
- **Clear and controlled empirical validation:** The experiments on synthetic rule and motor tasks provide direct evidence that the model can recover ground-truth modules and transition statistics (Figures 2, 4). Ablations (Figure 3) convincingly demonstrate the necessity of both the gating network and the inference procedure, particularly for sparse-feedback generalization.

## Weaknesses
- **Training instability and sensitivity:** The paper notes that training is prone to instability and local minima, requiring careful weight initialization (small `winit`, Appendix A.1). This sensitivity is a practical limitation that is not thoroughly analyzed; robustness to hyperparameters (e.g., learning rate, particle count) is not quantified.
- **Fixed, predefined module count:** The number of modules is set a priori and cannot be inferred from data. While mismatch experiments (Fig. A1) show some robustness, the inability to dynamically grow or prune the module library limits flexibility for open-ended task distributions.
- **Computational cost of particle filtering is unexamined:** Inference requires running a particle filter with many particles (250 used), each evaluating the RNNs. The computational and memory costs, as well as trade-offs with particle count, are not discussed, leaving practicality unclear.
- **Lack of direct comparison to the closest prior works:** The discussion cites related methods (Alet et al., 2019; Hummos et al., 2024) but does not provide quantitative comparisons on the same tasks. This omission makes it difficult to assess the specific improvement offered by the proposed inference-based sequencing over, e.g., search-based or embedding-optimization approaches.

## Nice-to-Haves
- A more systematic analysis of sparse-feedback performance (e.g., varying sparsity levels) would strengthen the claimed robustness.
- An exploration of more complex transition structures (beyond simple duration rules) could better demonstrate the gating RNN's capacity to learn "grammars."
- Reporting inference time and scaling with sequence length or modules would help assess practical utility.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Demand for theoretical grounding (identifiability proofs):** This is an empirical systems paper; theoretical guarantees are not standard or expected in this context.
- **Criticism that the abstract overclaims generality:** The abstract appropriately states the contribution, and the Discussion explicitly notes the "proof-of-principle" nature and synthetic tasks.
- **Request for comparisons to in-context learning (transformers) or standard meta-learning benchmarks (Mini-ImageNet, Meta-World):** The paper's contribution is a novel framework demonstrated on controlled synthetic tasks; requiring immediate validation on complex benchmarks is scope creep for a proof-of-concept.
- **Nitpicks about writing clarity (e.g., particle filter description being dense):** The explanation is sufficiently detailed, and the appendix provides further implementation notes.
- **Suggestion that error bars are missing in some figures:** The paper shows multiple seeds where appropriate (e.g., Fig. 2a); other figures show representative examples, which is acceptable for illustrative plots.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add a quantitative comparison to the most closely related methods (Alet et al., 2019 and Hummos et al., 2024) on the same synthetic tasks to clearly demonstrate the advantages of probabilistic inference over learned transition statistics.
- Include a brief analysis of computational cost (e.g., inference time vs. particle count, memory usage) to address scalability concerns.
- Expand the discussion of failure modes, using the data-model mismatch analysis (Fig. A1e) as a starting point, to better characterize the method's limitations.