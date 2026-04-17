# Review

## Summary
This paper addresses the issue of output volatility in long-form LLM generation, where models struggle to maintain consistent output length and quality across multiple generations for the same prompt. The authors introduce a novel benchmark, VOLTBench, designed to quantify length volatility in long-form generation across various tasks. They identify common internal patterns contributing to this instability through attention trace analysis and propose SELB (Structural Enforcement via Logits Boosting), a lightweight, training-free decoding strategy to mitigate this issue. Experimental results show that SELB significantly improves output length accuracy and stability while maintaining generation quality.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper tackles a critical and underexplored problem of output volatility in long-form LLM generation, which has significant implications for model reliability in real-world applications.
2. The introduction of VOLTBench provides a comprehensive framework for evaluating length volatility across both structured and unstructured tasks, filling a gap in existing benchmarks.
3. The analysis using attention traces to identify internal patterns contributing to volatility is a novel and insightful approach, offering a deeper understanding of the underlying mechanisms.

## Weaknesses
1. The paper could benefit from a more detailed comparison with existing methods for long-form generation, such as agentic approaches or iterative extensions. While SELB is positioned as a novel solution, a clearer comparison with prior work would help contextualize its contributions and highlight differences.
2. The evaluation primarily focuses on output length and structural adherence. While these are important, the paper could be strengthened by including additional metrics that assess the quality and coherence of the generated content, such as lexical diversity or thematic consistency.
3. The paper mentions that current models often struggle with length and structural constraints in long-form generation, leading to premature termination or section skipping. However, it would be beneficial to provide more quantitative data on the failure rates and patterns across different models to better illustrate the extent of the problem.

## Questions
1. How does SELB compare to existing methods for long-form generation, such as agentic approaches or iterative extensions? Could the authors provide a more detailed comparison to contextualize SELB's contributions?
2. While the paper focuses on output length and structural adherence, how does SELB impact the quality and coherence of the generated content? Are there additional metrics, such as lexical diversity or thematic consistency, that should be considered?
3. The paper mentions that current models struggle with length and structural constraints, leading to issues like premature termination or section skipping. Could the authors provide more quantitative data on the failure rates and patterns across different models to better illustrate the extent of the problem?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4