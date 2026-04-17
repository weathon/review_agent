# Review

## Summary
The paper introduces Ambig-SWE, an evaluation framework for assessing how well LLMs handle underspecified instructions in software engineering tasks. The study evaluates various LLMs, both proprietary and open-weight, by testing their ability to detect underspecificity, ask clarification questions, and leverage interaction to improve task performance. Key findings include that while interaction significantly improves performance for underspecified inputs, many LLMs default to non-interactive behavior, and there is a substantial gap between performance with and without interaction, highlighting the need for more effective handling of missing information in complex tasks.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The paper provides a detailed analysis of how LLMs handle underspecified instructions in software engineering, offering valuable insights into the models' strengths and limitations in detecting missing information and leveraging interaction.
2. The paper introduces a structured framework (Ambig-SWE) for evaluating LLMs in interactive, underspecified scenarios, which contributes a useful tool for assessing agent behaviors in complex, real-world tasks.

## Weaknesses
1. The study primarily relies on simulated environments and user proxies, which may not fully capture the complexity and variability of real-world user interactions. This limits the generalizability of the findings to actual deployment scenarios.
2. The paper does not provide a comprehensive comparison with existing methods for handling ambiguity or underspecificity, which would be valuable for understanding the relative advantages of the proposed approach.

## Questions
1. How do the findings generalize to non-software engineering tasks? Are there specific characteristics of software engineering tasks that make them particularly suitable for studying underspecificity?
2. What are the underlying reasons for the observed differences in interaction behaviors between different models? How might these differences be attributed to model architecture, training data, or other factors?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4