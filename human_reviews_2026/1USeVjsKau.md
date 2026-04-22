# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Post-training quantization (PTQ) compresses the weights and activations of large language models (LLMs) into low-precision representations to reduce memory footprint and accelerate inference. However, the presence of outliers in weights and activations often leads to large quantization errors and severe accuracy degradation, especially in recent reasoning LLMs where errors accumulate across long chains of thought. Existing PTQ methods either fail to sufficiently suppress outliers or introduce significant overhead during inference. In this paper, we propose Pairwise Rotation Quantization (ParoQuant), a PTQ method that combines hardware-efficient and optimizable independent Givens rotations with channel-wise scaling to even out the magnitudes across channels and narrow the dynamic range within each quantization group, effectively addressing the outlier issue. We further co-design the inference kernel to fully exploit GPU parallelism and keep the rotations and scaling lightweight at runtime. Under weight-only quantization, ParoQuant achieves an average 2.4% accuracy improvement over AWQ on reasoning tasks, with less than 10% overhead. ParoQuant also matches the accuracy of state-of-the-art weight-activation quantization methods. This paves the way for more efficient and accurate deployment of reasoning LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents ParoQuant, a weight-only post-training quantization (PTQ) method designed for reasoning LLMs. The core idea is to apply scaled pairwise rotation, combining independent Givens rotations with channel-wise scaling, to suppress outliers efficiently. The authors co-design a CUDA kernel to maintain high throughput. Experiments show consistent accuracy gains over AWQ and EfficientQAT, with less than 10% latency overhead.

### Strengths
The motivation that, quantization error accumulation in reasoning models, is clear and important.

The proposed rotation-based PTQ design is both novel and hardware-efficient.

Extensive experiments on multiple model sizes (up to 70B) and reasoning benchmarks (MMLU-Pro, GSM8K, etc.) demonstrate solid improvement.

Paper is clearly written and well-structured, with good algorithmic detail and ablations.

### Weaknesses
While the rotation kernel is claimed to be efficient, the paper lacks quantitative breakdown of runtime and memory overhead (e.g., FLOPs, memory traffic).

The scalability of independent rotations to larger group sizes or mixed-precision settings (e.g., W4A8) is not discussed.

More analysis on the trade-off between number of rotations and latency would strengthen the efficiency claim.

The method seems tailored for linear quantization; extension to vector quantization or activation quantization could be briefly discussed.

### Questions
Could the authors provide more detailed profiling on GPU resource usage and explain how ParoQuant scales when group size or model size further increases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents ParoQuant, a weight-only post-training quantization method developed to improve the accuracy and efficiency of LLMs. It tackles the challenge of quantization error through a novel combination of independent Givens rotations and channel-wise scaling, which effectively reduces the impact of outliers. Additionally, the work incorporates a custom CUDA kernel to accelerate online scaled pairwise rotations, enabling faster inference.

### Strengths
1. Clear and Well-Founded Motivation: The paper observes that rotating only the top 10% of the most significant weight channel pairs can achieve nearly the same reduction in quantization error as performing a full rotation. This insight eliminates a large amount of redundant computation from full matrix multiplications, leading to a much more efficient quantization process.

2. Methodology with GPU-Aware Design: Building on this motivation, the authors propose a three-step design for the scaled pairwise rotation transform.

Step 1: Replace costly full orthogonal matrix multiplications with a set of decomposed Givens rotations.

Step 2: Eliminate inter-rotation dependencies to allow fully parallel execution on GPUs, resulting in independent rotations.

Step 3: Since a single independent rotation cannot adequately capture complex weight distributions, apply a series of independent rotations combined with channel-wise scaling to improve representation and quantization accuracy.

3. Practical CUDA Implementation: The paper further introduces a co-designed efficient transform kernel that maximizes GPU parallelism by executing computations across three levels:

Token-level: Parallelization over the token dimension of the activation tensor.

Channel-group level: Different CUDA blocks handle different groups of channels.

Pair level: Each rotation pair is processed by a separate CUDA thread.

### Weaknesses
Overall, I found the paper well-written and technically solid. The following are just minor curiosities rather than critical weaknesses:

1. The 4-bit performance gains appear somewhat modest for certain model sizes and tasks (e.g., Perplexity and AIME). It would be interesting to see whether ParaQuant delivers more substantial improvements at lower bitwidths, such as 3-bit or 2-bit quantization.

2. Do you have any insight into why E-QAT performs particularly poorly on AIME, given that ParaQuant can essentially be viewed as an extension of E-QAT with additional learnable rotations? I am expecting the performance gap between the two methods to be smaller.

### Questions
See Above

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
5

### Summary
This paper introduces ParoQuant, a novel weight-only PTQ method for reasoning LLMs. It uses a "scaled pairwise rotation" transform, combining channel-wise scaling with hardware-efficient Givens rotations to suppress outliers. Through algorithm-system co-design, it achieves high accuracy with low inference overhead. It provides a very systematic solution, and the experiments are very comprehensive.

### Strengths
1.	The paper clearly identifies and addresses a critical, forward-looking problem: the poor performance of efficient quantization methods on reasoning tasks that require long chains of thought. This focus on error accumulation in generative tasks is timely and important.

2.	The proposed "scaled pairwise rotation" is a novel and elegant solution. The insight that a full rotation matrix is redundant and can be effectively approximated by a series of independent, parallelizable Givens rotations is the key contribution and is very well executed.

3.	The algorithm-system co-design is a major strength. The authors didn't just propose a transform; they designed a custom CUDA kernel that makes the transform viable in practice, demonstrating a deep understanding of both the algorithmic and hardware constraints. The empirical results, showing ParoQuant matching QTIP's accuracy while being ~25% faster, are very compelling.

### Weaknesses
1.	The greedy pair selection strategy outlined in Algorithm A1, while effective and intuitive, may not be globally optimal. It would be beneficial for the authors to discuss the potential limitations of this greedy approach.

2.	In Section 3, when discussing quantization degradation on reasoning tasks, the authors should cite other recent works that have also identified this specific problem (e.g., QSPEC) to better contextualize their motivation.

3.	In Figure 3, some text labels in the right-most portion of the diagram are overlapped, which slightly hinders readability.

### Questions
1.	The number of independent rotations is fixed at K=8 for most experiments. How was this number chosen? Is there a clear point of diminishing returns, and does the optimal value of K change depending on the model architecture or size?

2.	The "pairwise" rotation in Algorithm A1 is effective but seems conservative. Did the authors consider more fine-grained or alternative rotation structures, such as rotating small blocks of channels against each other? While this might be more expressive, it would likely introduce significant scheduling overhead. A discussion on this potential trade-off between rotation granularity and scheduling efficiency would be insightful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ParoQuant, a weight-only post-training quantization (PTQ) method designed for reasoning LLMs, where quantization errors can accumulate over long generations.
ParoQuant combines: 1.Independent Givens rotations (pairwise rotations) to suppress outliers efficiently, and 2.Channel-wise scaling to even out magnitude across channels.

### Strengths
The authors convincingly argue that reasoning LLMs are especially sensitive to accumulated quantization errors, providing strong justification for the proposed method’s focus on accuracy stability during long generation.

ParoQuant achieves higher reasoning-task accuracy than AWQ and matches the state-of-the-art QTIP while being significantly faster.

The paper thoughtfully co-designs the quantization algorithm and CUDA implementation.

### Weaknesses
Please see my questions.

### Questions
How does ParoQuant perform under activation quantization or mixed-precision scenarios?

Could the pairwise rotation be merged offline to further reduce runtime cost?

Can this idea extend to FP4/FP8 formats?

### Soundness
3

### Presentation
3

### Contribution
3
