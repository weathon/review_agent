# AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Data and pipeline parallelism are key strategies for scaling neural network training across distributed devices, but their high communication cost necessitates co-located computing clusters with fast interconnects, limiting their scalability. We address this communication bottleneck by introducing *asynchronous updates across both parallelism axes*, relaxing the co-location requirement at the expense of introducing *staleness* between pipeline stages and data parallel replicas. To mitigate staleness, for pipeline parallelism, we adopt a weight look-ahead approach, and for data parallelism, we introduce an *asynchronous sparse averaging* method equipped with an exponential moving average based correction mechanism. We provide convergence guarantees for both sparse averaging and asynchronous updates. Experiments on large-scale language models (up to *1B parameters*) demonstrate that our approach matches the performance of the fully synchronous baseline, while significantly reducing communication overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new distributed training framework called AsyncMesh, which enables fully asynchronous optimization across both data parallel (DP) and pipeline parallel (PP) axes. To counteract the staleness introduced by asynchrony, the authors propose (1) a weight look-ahead mechanism for PP using Nesterov extrapolation and (2) an asynchronous sparse averaging scheme for DP with an exponential moving average (EMA)-based staleness correction. They theoretically prove convergence under standard assumptions and empirically show that AsyncMesh achieves performance comparable to fully synchronous training while drastically reducing communication overhead.

### Strengths
+ Introduces AsyncMesh, that enables asynchronous updates across both data parallelism (DP) and pipeline parallelism (PP) to address this communication bottleneck.
+ Combines Nesterov-based weight look-ahead for PP and Exponential Moving Average (EMA) correction for DP to counteract stale gradients and parameters effectively.
+ Provides formal convergence guarantees for both asynchronous sparse averaging and delayed updates, extending existing results from stochastic approximation theory.
+ Demonstrates performance parity with fully synchronous baselines on language models up to 1 billion parameters, showing scalability and robustness.

### Weaknesses
- Theoretical convergence guarantees rely on homogeneous settings, which may not hold in practical heterogeneous or real-world decentralized systems.
- Although sparse averaging reduces communication, it could slow convergence for extremely small subsets or large delays, as hinted in the theoretical analysis. No experiments have done on this.
- The paper lacks direct comparisons with strong recent baselines such as DeepSpeed ZeRO, ZeRO++.
- The effects of EMA decay rates, subset sizes, and delay parameters are not deeply analyzed.
- Combining asynchronous DP and PP with custom staleness correction mechanisms may complicate integration into existing distributed training frameworks.

### Questions
1. The theoretical convergence proofs assume identical hardware, learning rates, and i.i.d. data across replicas. Extend the theoretical framework or provide an empirical ablation to quantify performance under heterogeneity, such as uneven compute power, data imbalance, or non-uniform network latency.
2. Could staleness correction via EMA still approximate global consensus effectively when replicas diverge in data distribution or update frequency? Discuss convergence guarantees in heterogeneous environments.
3. The paper claims sparse averaging (e.g., 5%) maintains performance, but what happens at extreme sparsity (e.g., 1% or less) or high delay (τ > 50)? Is there a theoretical threshold or empirical tipping point where sparse averaging starts to degrade convergence or stability? Conduct controlled experiments varying both subset size (1–10%) and delay intervals (10–100 steps) to observe convergence degradation patterns. Include convergence curves and communication–accuracy trade-off plots to better illustrate the regime where AsyncMesh remains stable.
3. Why were DeepSpeed ZeRO and ZeRO++ omitted from the baseline comparisons, given their dominance in large-scale model parallelism? How would AsyncMesh compare to ZeRO’s optimizer and gradient state partitioning in terms of both communication efficiency and memory footprint? Include at least one benchmark comparison (even partial) against ZeRO or ZeRO++ under similar mesh sizes to demonstrate where AsyncMesh provides unique advantages. Discuss interoperability potential — e.g., could AsyncMesh be layered on top of ZeRO’s optimizer partitioning?
4. How sensitive is AsyncMesh’s convergence to the EMA coefficient schedule (λₜ)? Does the same schedule generalize across datasets, model sizes, or communication delays? Are there interactions between subset size, EMA decay, and delay that affect stability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper try to incorporate both async communication in DP(data parallel) and PP(pipeline parallel).

The major contribution is to combine asyncPP and SPARTA in DP together and guarantee the loss curve can converge.

### Strengths
1. Theoretical analysis on AsyncPP and SPARTA in DP

2. e2e experiments training and show loss curves

### Weaknesses
1. the major experimental model is a toy size of 160M, which cannot represent real world pre-training model patterns. In addition, it is just toy NanoGPT not a real GPT model. Furthermore, the model does not even have basic dropout layer, which make the loss curve comparison less convincing. 

2. the paper contribution is very minor, it just combined existed work AsyncPP and SPARTA in DP together and did a bit tuning. There is very little research novelty here.

3. Whether the model can converge or not in such async model training case is heavily depend on delayed steps, the math part of this paper does not even discuss much about it thus making the whole proof less meaningful. 

4. the sec 5.4, 1B model itself is not a standard GPT model, for example, the embedding dim is very small as 2304, and 24 attention head is not standard gpt-3. In addition, the paper does not show any main stream and standard model size results, thus making the result less convincing. 

5. the authors lack of knowledge about SOTA data parallel framework, such as ZeRO or FSDP, which is the only DP paradigm used in real world. And people start only use them as DP starting from 2020. But there is no discussion on how to do async communication overhead in such schemes.

### Questions
If t - $\tau$ to t has a big delay, how should loss converge still hold? The proof does not make sense if there is a big gap between t - $\tau$ to t. And there is no discussion on how to theoretically analyse and determine the biggest gap between t - $\tau$ to t to make the loss curve difference minimal to fully synced.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes AsyncMesh, asynchronous staleness-aware training approach that combines an asynchronous sparse averaging method and an exponential moving average based correction mechanism. It also provides convergence guarantees for both sparse averaging and asynchronous updates and evaluates their methods using LLMs with up to 1B parameters.

### Strengths
1. AsyncMesh explores the setup where both DP and PP are asynchronous.
2. The paper designs an Exponential Moving Average (EMA) based correction mechanism that approximates the average staleness.
3. The paper provides theoretical justification of convergence in the presence of staleness in a homogeneous setup where only a small subset of weights is communicated between DP replicas.

### Weaknesses
1. The baseline for the evaluation is weak. The benchmark for this evaluation is weak. The evaluation only compares AsyncMesh with FullyAsync and DP. However, well-studied staleness-aware LLM training [1]  (with different degree of staleness) and also block coordinate descent with correction [2] was not included in the evaluation.
2. The evaluation results did not show how much performance improvement sparse averaging could bring.

[1] PipeDream: Generalized pipeline parallelism for DNN training
[2] Accelerating Block Coordinate Descent for LLM Finetuning via Landscape Correction

### Questions
Does the A00 machine used in the evaluation have NVLink?

### Soundness
3

### Presentation
3

### Contribution
2
