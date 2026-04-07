## Summary
This paper establishes a first-order equivalence between activation steering and influence functions, introducing Influence-Aligned Steering (IAS) as a constructive mapping. It provides theoretical guarantees on alignment, optimality, and generalization, supported by experiments on language and vision models.

## Strengths
- **Novel theoretical unification**: The paper derives closed-form mappings between steering vectors and influence weightings, proving their first-order equivalence (Theorems 4.2, 5.2) and introducing alignment bounds via a scalar measure ω(x) (Theorem 5.1). This formally connects two previously disparate research areas.
- **Practical diagnostic and optimization tools**: The alignment metric ω(x) offers a feasible pre-check for steering success, and the spectral method for optimal steering directions (Theorem 5.3) provides a principled alternative to handcrafted vectors, both relying on efficient Jacobian-vector products.
- **Empirical validation of core theoretical claims**: Experiments confirm high cosine similarity (0.978) between predicted and actual first-order logit shifts (Figure 1), show alignment improves with layer depth (Figure 2), and demonstrate statistical significance of spectral directions on ResNet-50 (Figure 3).

## Weaknesses
- **Limited empirical scope undermines scalability claims**: Experiments are confined to GPT-2 Medium (355M parameters) and ResNet-50, with no validation on billion-parameter models as suggested by the paper's motivation. This leaves the claimed applicability to large-scale models unsubstantiated.
- **Insufficient demonstration of practical advantage**: In the detoxification task, IAS underperforms the Contrastive Activation Addition (CAA) baseline in both toxicity reduction and perplexity (Table 1), failing to show clear empirical benefit over existing steering methods.
- **Missing validation of key workflow component**: The paper does not empirically demonstrate the promised mapping from steering vectors to causal training examples (via ϱ_s), which is central to the contribution of data attribution and debugging. Without this, the practical utility of the equivalence remains unproven.
- **Unexplored boundaries of the first-order regime**: While the theory assumes small edits, there is no empirical analysis quantifying how large steering magnitudes can be before the first-order approximation breaks down, leaving practical applicability uncertain.

## Nice-to-Haves
- Extend experiments to larger models (e.g., Llama, GPT-J) and diverse tasks (e.g., factual editing, bias mitigation) to better support scalability and generality claims.
- Provide a quantitative analysis of error growth with steering magnitude to delineate the valid regime of the first-order approximation.
- Include more detailed implementation specifics (e.g., damping parameter choices, layer selection heuristics) to enhance reproducibility.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- "Abstract omits critical condition (ω)" – The abstract summarizes contributions, and ω is part of the theoretical development, not an omission required in the abstract.
- "Connection to convex analysis not elaborated" – This is an optional depth for insight, not a flaw in the paper's core contributions.
- "Proofs are deferred" – Common practice for conference papers; not a substantive weakness.
- "Systematic scaling discrepancy with slope 1.50 indicates a problem" – The paper notes the slope in Figure 1 and states it is consistent with the linear regime; without further evidence, this is not a clear error.
- "Experiments lack computational cost discussion" – The paper outlines computational primitives (Jacobian-vector products, small SVDs) and acknowledges scalability challenges, so this criticism is partially addressed.

## Novel Insights
The paper introduces the novel insight that activation steering and influence functions are dual projections of the same sensitivity tensor, with the alignment metric ω(x) quantifying when steering can perfectly replicate data influence. This unification provides a geometric framework for diagnosing feasibility and offers a principled bridge between model intervention and data attribution.

## Suggestions
- Conduct an experiment that applies the IAS mapping to identify training examples for a specific model behavior (e.g., a bias or hallucination), validating the causal attribution claim with qualitative analysis.
- Compare IAS to state-of-the-art steering methods (e.g., SAKE, representation engineering) on standard benchmarks to better assess its practical value relative to existing approaches.
- Discuss concrete computational strategies for approximating pseudoinverses and Hessian inverses in large-scale settings to address scalability concerns more transparently.