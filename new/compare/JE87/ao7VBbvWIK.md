# Review

## Summary
The paper introduces HASTE, a novel framework designed to address the challenge of limited context windows in Large Language Models (LLMs) when applied to software engineering tasks. HASTE combines robust information retrieval with deep structural analysis, leveraging the Abstract Syntax Tree (AST) to ensure extracted code is both structurally coherent and executable. The framework achieves up to 85% code compression while improving the success rate of automated code edits and reducing model-generated hallucinations.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper introduces a novel framework, HASTE, that addresses the critical trade-off between semantic relevance and structural coherence in code context retrieval.
2. The paper provides a thorough literature review, situating HASTE within the context of existing research and highlighting its innovative aspects.

## Weaknesses
1. The paper does not provide a detailed comparison with existing methods, making it difficult to assess the relative advantages of HASTE.
2. The experiments are conducted on a small, curated dataset and a single LLM, which may limit the generalizability of the results.
3. The paper does not adequately address the scalability of HASTE to larger codebases or its performance across different programming languages.
4. The evaluation metrics, while complementary, may not fully capture the nuances of real-world software development tasks.
5. The paper lacks a detailed analysis of the failure modes and their relationship to the framework's design choices.

## Questions
1. How does HASTE compare to existing structure-aware and relevance-focused approaches in terms of precision and recall?
2. Can the authors provide more details on the selection criteria for the curated dataset and its representativeness of real-world codebases?
3. How does the performance of HASTE scale with increasing codebase size, and what are the computational overheads associated with AST analysis and call graph expansion?
4. Have the authors considered the potential biases introduced by the LLM-as-a-judge evaluation, and how might these be addressed?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4