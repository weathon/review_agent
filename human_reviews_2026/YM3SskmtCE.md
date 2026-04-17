# ATTS: Asynchronous Test-Time Scaling via Conformal Prediction

- Decision: Accept (Poster)
- Scores: 2, 8, 8

## Abstract
Large language models (LLMs) benefit from test-time scaling but are often hampered by high inference latency. Speculative decoding is a natural way to accelerate the scaling process; however, scaling along both the parallel and sequential dimensions poses significant challenges, including substantial memory-bound execution and synchronization overhead. We introduce ATTS (Asynchronous Test-Time Scaling), a statistically guaranteed adaptive scaling framework that follows the hypothesis testing process to address these challenges. By revisiting arithmetic intensity, ATTS identifies synchronization as the primary bottleneck. It enables asynchronous inference through online calibration and proposes an ordinal classification algorithm that supports a three-stage rejection sampling pipeline, scaling along both the sequential and parallel axes. Across experiments on the MATH, AMC23, AIME24, and AIME25 datasets and across multiple draft–target model families, we show that ATTS delivers up to 56.7x speedup in test-time scaling and a 4.14x throughput improvement, while maintaining accurate control of the rejection rate, reducing latency and memory overhead, and incurring no accuracy loss. By scaling both in parallel and sequential dimensions, we enable the 1.5B/70B draft/target model combination to achieve the performance of the state-of-the-art reasoning model o3-mini (high) on the AIME dataset.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ATTS (Asynchronous Test-Time Scaling) to address the high latency problem in large model inference. The authors introduce the concept of asynchronous arithmetic intensity to measure the ratio between the time spent sampling candidates and the time waiting for the large model to verify them, showing that as the number of samples increases, the cost from waiting for verification becomes more significant. The method leverages ideas from conformal prediction to reduce the candidate set that requires verification by the large model, thereby lowering compute costs. The resulting ATTS framework applies conformal prediction for theoretically lossless speedup of test-time inference.

### Strengths
The paper provides valuable insights into how to reduce the time cost of test-time scaling (TTS), especially through the lens of asynchronous arithmetic intensity.

Effectively applies conformal prediction to minimize the workload of the large model during inference, and demonstrates both theoretical and practical speedup.

The experimental section covers several representative math reasoning tasks, and the results are fairly comprehensive for the considered setting.

### Weaknesses
The method is not empirically compared against other prominent recent approaches in the field, such as SpecReason and Speculative Thinking, making it difficult to objectively evaluate the unique advantages of ATTS.

Experiments suggest that when the small and large models are trained on data from different distributions, the benefits of ATTS decrease noticeably, highlighting a potential issue: the method does not fundamentally address the extra rejection cost caused by distribution mismatch between the small and large models.

Theoretical guarantees rely on key assumptions from conformal prediction (e.g., exchangeability and normalization of scores). However, the implementation omits score normalization, which may break these assumptions and cause the rejection rate control to fail, especially in multi-task or distribution-shifted settings.

### Questions
1.Validity of theoretical assumptions
The omission of normalization in the implementation may break the core assumptions of conformal prediction, potentially leading to excessive rejections for some tasks and too few for others in mixed or real-world scenarios. How does the method ensure fairness and validity of the rejection rate when handling multi-task or large-scale deployments?

2.Lack of comparison with strong baselines
The experimental section does not compare ATTS with other competitive methods such as SpecReason or Speculative Thinking. What are the practical or theoretical advantages of ATTS over these methods, and is there any empirical evidence of its superiority or unique use cases?

3.System/engineering considerations
The paper does not discuss the impact of modern inference engine features, such as supporting asynchronous verification requests. Can ATTS be integrated with inference backends that support asynchronous calls, and is there any experimental evidence for further gains from such integration?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes ATTS, an asynchronous test-time scaling framework for large language models (LLMs) that addresses high inference latency and memory bottlenecks in test-time scaling. By redefining asynchronous arithmetic intensity to identify synchronization as the key bottleneck, integrating online calibration with conformal prediction, and designing a three-stage rejection sampling pipeline, ATTS enables both sequential and parallel scaling. Experiments on MATH, AMC23, AIME24/25 datasets show up to 56.7x speedup, 4.14x throughput improvement, and no accuracy loss—even allowing 1.5B/70B draft/target models to match o3-mini (high) performance on AIME.

### Strengths
1. The introduction of "asynchronous arithmetic intensity" effectively quantifies synchronization overhead, a critical but understudied issue in test-time scaling.
2. Leveraging conformal prediction for ranking and rejection sampling ensures controlled rejection rates and coverage (marginal/conditional), adding theoretical rigor absent in many efficiency-focused works.
3. Significant speedup and throughput gains across model families (e.g., Qwen, DeepSeek, Llama) and challenging math benchmarks demonstrate real-world utility.

### Weaknesses
1. ATTS frames sequential scaling (increasing turns) as a strength, but experiments only test up to 10 turns. The paper fails to analyze when sequential scaling stops adding value—e.g., whether 20+ turns lead to accuracy plateaus while inflating token costs. 
2. The evaluation on more challnaging benchmarks would make the results more robust
3. While conformal prediction enables statistical guarantees for rejection sampling, the paper overlooks a critical tension: its p-value calculation (Eq.13) requires comparing test samples to the full calibration set. For large batch sizes (128+), this comparison could introduce hidden per-sample latency—undermining ATTS’s core goal of reducing overhead.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose ATTS (Asynchronous Test-Time Scaling), designed to alleviate the memory peak and synchronization overhead issues in speculative decoding under parallel and sequential test-time scaling settings.

The authors characterize system bottlenecks using asynchronous arithmetic intensity, explicitly incorporating synchronization time into the performance metric. The authors find that they synchronous time is the main bottleneck. Therefore, ATTS adopts conformal prediction: it computes a p-value for each candidate and compares it with a threshold α to decide acceptance or rejection, instead of using global softmax and ranking which needs synchronous. This provides statistical control over the target model intervention rate without requiring distributional assumptions.

Building on this principle, ATTS implements a three-stage rejection sampling pipeline that scales along both the parallel and sequential axes. It further employs online calibration and ordered decision-making to avoid global synchronization.

Experiments on MATH, AMC23, AIME24/25, and various draft/target model combinations show that ATTS achieves up to 56.7× end-to-end acceleration and 4.14× throughput improvement without sacrificing accuracy, while significantly reducing peak memory usage and latency. Notably, on the AIME benchmark, a 1.5B draft / 70B target setup reaches performance close to that of strong closed-source reasoning models.

The main limitations lie in the sensitivity to online calibration quality and the choice of α, the lack of guarantees for global optimality in each round, and the evaluation focus on mathematical reasoning and specific inference engine configurations.

### Strengths
1. This paper addresses a highly important and practical problem in inference infrastructure, which has significant implications for optimizing the efficiency of model deployment.
2. The authors introduced a new performance metric to analyze specific bottlenecks, which guided the subsequent algorithm design.
3. The paper is well-structured and easy to understand.
4. The proposed method achieves a significant improvement in efficiency while maintaining stable performance.

### Weaknesses
1. The proposed method heavily relies on the stability of online calibration. The rejection rate control and accuracy are quite sensitive to the calibration distribution, the number of parallel samples, and the value of α.
2. The proposed method does not aim for globally optimal candidates and cannot guarantee obtaining the top-1 candidate in each round, which may be suboptimal for applications that strictly require optimal ranking.

### Questions
How can online calibration maintain long-term stability of the rejection rate under non-stationary workloads?

### Soundness
3

### Presentation
3

### Contribution
3
