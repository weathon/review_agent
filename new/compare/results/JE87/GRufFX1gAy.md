# Review

## Summary
The paper introduces InnoGym, a benchmark and framework designed to evaluate the innovation potential of AI agents. It addresses the limitations of existing benchmarks that primarily focus on correctness without assessing the diversity of solutions or the originality of the approach. InnoGym introduces two metrics: performance gain, which measures improvement over the best-known solutions, and novelty, which captures methodological differences from prior approaches. The benchmark includes 18 carefully curated tasks from real-world engineering and scientific domains, standardized through resource filtering, evaluator validation, and solution collection. The paper also provides iGym, a unified execution environment for reproducible and long-horizon evaluations. Extensive experiments reveal that while some agents produce novel approaches, their lack of robustness limits performance gains, highlighting a key gap between creativity and effectiveness.

## Soundness
4

## Presentation
4

## Contribution
3

## Strengths
1. This paper introduces a novel benchmark and framework, InnoGym, which is specifically designed to evaluate the innovation potential of AI agents. It addresses the gap in existing benchmarks by focusing on both performance and novelty.
2. The paper provides a principled framework for defining and measuring innovation, combining performance gain and novelty as complementary evaluation dimensions.
3. The benchmark includes a diverse set of 18 carefully curated tasks from real-world engineering and scientific domains, ensuring a broad range of evaluation scenarios.
4. The paper offers a unified execution environment, iGym, which supports reproducible and long-horizon evaluations across diverse systems.

## Weaknesses
1. The paper could benefit from a more detailed discussion of the limitations of the proposed framework and benchmark, including potential biases and challenges in implementation.
2. While the paper provides a comprehensive overview of the framework and benchmark, some aspects of the methodology, such as the specific algorithms used for novelty evaluation, could be explained in more detail.

## Questions
1. How does the framework account for the rapid evolution of AI technologies, and how frequently will the benchmark be updated to reflect the latest advancements?
2. What are the potential biases inherent in the benchmark tasks and metrics, and how are these biases mitigated?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4