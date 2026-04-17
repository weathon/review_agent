# Review

## Summary
The paper introduces a new competitive programming benchmark for LLMs, AetherCode, which addresses limitations in existing benchmarks by offering more challenging problems and high-quality test cases. AetherCode collects problems from premier programming competitions like IOI and ICPC, and combines automated test case generation with expert annotation to ensure rigorous evaluation. This benchmark provides a more accurate assessment of LLM capabilities in coding and reasoning, setting a new standard for future research in this area.

## Soundness
4

## Presentation
4

## Contribution
4

## Strengths
- AetherCode is the first benchmark to systematically collect the latest problems from premier programming competitions worldwide, including the Olympiad in Informatics (OI) and the International Collegiate Programming Contest (ICPC).
- AetherCode is the first benchmark that sets a high standard for test cases, achieving 100% TPR and 100% TNR on the collected solution set.
- The paper conducts extensive evaluation on 11 reasoning models and 6 non-reasoning models, providing a thorough analysis of their performance across different difficulty levels and categories.
- The paper is well-organized and clearly written.

## Weaknesses
- The paper does not provide a detailed comparison of AetherCode with other competitive programming benchmarks, making it difficult to assess the specific advantages of this benchmark.
- The paper does not provide a detailed analysis of the failure cases, making it difficult to assess the specific challenges faced by LLMs in solving these problems.
- The paper does not provide a detailed analysis of the reasons for the performance differences between reasoning models and non-reasoning models, making it difficult to assess the specific factors that contribute to these differences.

## Questions
- How does AetherCode compare to other competitive programming benchmarks in terms of problem difficulty, scope, and evaluation methods?
- What are the specific challenges faced by LLMs in solving the problems in AetherCode, and how can these challenges be addressed in future research?
- What are the specific factors that contribute to the performance differences between reasoning models and non-reasoning models, and how can these factors be addressed to improve the coding and reasoning capabilities of LLMs?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4