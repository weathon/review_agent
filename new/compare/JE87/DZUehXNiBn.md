# Review

## Summary
The paper presents VISTA, a modular framework for causal structure learning from observational data. VISTA decomposes the global causal structure learning problem into local subgraphs based on Markov Blankets and integrates these subgraphs using a weighted voting mechanism to form the final causal graph. The framework is designed to be model-agnostic, compatible with various base learners and data settings, and supports parallelization for improved efficiency. The authors provide theoretical guarantees on error bounds and asymptotic consistency, and demonstrate VISTA's effectiveness through extensive experiments on synthetic and real datasets, showing improvements in accuracy and efficiency over baseline methods.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. VISTA is model-agnostic and can be integrated with different causal discovery methods, enhancing its versatility.

2. The framework is efficient, with a lightweight aggregation process that avoids expensive global searches.

3. Extensive experiments demonstrate VISTA's effectiveness in improving both accuracy and efficiency across various settings.

## Weaknesses
1. The theoretical guarantees provided for VISTA rely on the assumption that the votes from different local subgraphs are independent. However, in practice, subgraphs learned from the same dataset can induce correlations among votes. This assumption should be more explicitly discussed in the main text, along with its implications on the practical applicability of the theoretical results.

2. The paper lacks a detailed discussion on the computational complexity of VISTA, particularly regarding the scalability of the weighted voting mechanism as the number of nodes and edges increases. It would be beneficial to provide a more thorough analysis of the time and space complexity, especially in comparison with existing methods, to better understand the trade-offs involved.

3. The real-data experiments are limited to a single dataset (Sachs protein-signaling network). Including evaluations on additional real-world datasets would provide a more comprehensive assessment of VISTA's performance in diverse practical scenarios.

## Questions
See Weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4