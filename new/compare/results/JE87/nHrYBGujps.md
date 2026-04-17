# Review

## Summary
This paper introduces a new benchmark for evaluating LLMs in multi-turn text-to-SQL interactions. The proposed benchmark includes 900 tasks, each featuring ambiguous initial priority sub-tasks, dynamic clarification requirements, follow-up sub-tasks, and environmental uncertainties. The authors propose two evaluation settings: c-Interact and a-Interact, to evaluate the LLMs' ability to handle ambiguity and evolve user intents in a dynamic text-to-SQL environment. The paper also introduces a function-driven user simulator to ensure a more robust evaluation.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The authors provide a clear motivation for the need of a new benchmark and a detailed description of the benchmark construction process.
2. The proposed benchmark addresses an important gap in evaluating LLMs' ability to handle multi-turn text-to-SQL interactions, which is closer to real-world scenarios.
3. The paper includes a comprehensive evaluation of existing LLMs, providing insights into their strengths and weaknesses in handling interactive text-to-SQL tasks.

## Weaknesses
1. The paper does not provide a detailed comparison with existing multi-turn text-to-SQL benchmarks, making it difficult to understand the novelty and added value of the proposed benchmark.
2. The evaluation settings and task suite are not clearly explained, leaving questions about the specific challenges and scenarios the benchmark aims to test.
3. The paper lacks a detailed analysis of the results, including examples of successful and unsuccessful interactions, and a discussion of the implications of the findings.

## Questions
1. How does the proposed benchmark differ from existing multi-turn text-to-SQL benchmarks? A detailed comparison would help highlight the novelty and added value of this work.
2. Can you provide more details about the evaluation settings, such as c-Interact and a-Interact? How do these settings reflect real-world scenarios, and what specific challenges do they aim to test?
3. The paper mentions a function-driven user simulator. Can you provide more details about its design and implementation? How does it ensure a more robust evaluation?
4. The results show that existing LLMs struggle with the proposed benchmark. Can you provide examples of successful and unsuccessful interactions? What insights can be drawn from these examples, and what directions suggest the results for improving LLMs' performance in multi-turn text-to-SQL tasks?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4