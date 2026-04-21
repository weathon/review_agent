# Channel-Wise Mixed-Precision Quantization for Large Language Models

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 5

## Abstract
Large Language Models (LLMs) have demonstrated remarkable success across a wide range of language tasks, but their deployment on edge devices remains challenging due to the substantial memory requirements imposed by their large parameter sizes. Weight-only quantization presents a promising solution to reduce the memory footprint of LLMs. However, existing approaches primarily focus on integer-bit quantization, limiting their adaptability to fractional-bit quantization tasks and preventing the full utilization of available storage space on devices. In this paper, we introduce Channel-Wise Mixed-Precision Quantization (CMPQ), a novel mixed-precision quantization method that allocates quantization precision in a channel-wise pattern based on activation distributions. By assigning different precision levels to different weight channels, CMPQ can adapt to any bit-width constraint. CMPQ employs a non-uniform quantization strategy and incorporates two outlier extraction techniques that collaboratively preserve the critical information, thereby minimizing the quantization loss. Experiments on different sizes of LLMs demonstrate that CMPQ not only enhances performance in integer-bit quantization tasks but also achieves significant performance gains with a modest increase in memory usage. CMPQ thus represents an adaptive and effective approach to LLM quantization, offering substantial benefits across diverse device capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents an approach to quantize large language models (LLMs) with channel-wise mixed-precision quantization. This method allows different channels of neural network layers to be quantized at different precision levels.  Experiments on OPT and LLaMA show that their method outperforms existing quantization techniques.

### Strengths
1. The channel-wise mixed-precision quantization can help deal with channel outliers.
2. The experimental results indicate substantial improvements in accuracy after quantization.

### Weaknesses
1. There are typography disorders on page 7.
2. The overall process of CMPQ during inference is not well elaborated. For example, how do you perform de-quantization efficiently?
3. The real deployment efficiency is not presented. Since CMPQ sets different precision for each channel, I am worried about its de-quantization overhead during inference.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces Channel-Wise Mixed-Precision Quantization (CMPQ), a mixed quantization method for large language models (LLMs) that assigns quantization precision in a channel-wise manner based on activation distributions. This approach aims to reduce memory requirements while maintaining performance by adaptively allocating bit precision within each weight channel. CMPQ also employs non-uniform quantization and two outlier extraction methods to preserve critical information, reducing quantization loss. Experiments across various LLMs and datasets indicate that CMPQ outperforms baseline quantization methods, achieving notable memory savings with minimal performance compromise, particularly in tasks with integer-bit and fractional-bit quantization constraints

### Strengths
One of the strengths of this paper is its focused and in-depth exploration of outliers in quantization. Unlike many previous methods, which either overlook or handle outliers with basic techniques, CMPQ presents a comprehensive approach by categorizing outliers into activation-based and quantization-sensitive types. This dual approach is a significant step forward, as it ensures that high-impact weights are preserved where they matter most, leading to enhanced performance stability across different quantization levels. By offering a granular solution to outlier management, the paper addresses a critical gap in existing research, showing a well-founded understanding of the practical challenges in mixed-precision quantization.

Another commendable aspect of this paper is its ability to synthesize and build upon various existing quantization techniques, resulting in an integrated solution. The authors leverage insights from both uniform and non-uniform quantization studies, incorporating methods like channel-wise precision allocation and selective outlier retention to maximize CMPQ’s efficacy. This integrative approach not only showcases the strengths of CMPQ but also places it within the broader context of quantization research, illustrating how it enhances or complements other methods. This approach adds robustness to the results, providing a clear, side-by-side comparison with traditional integer-based techniques and showing the benefits of CMPQ’s adaptive precision allocation.

### Weaknesses
1. While CMPQ leverages mixed-precision quantization to improve model performance, the paper lacks a rigorous theoretical framework for determining the allocation of different bit widths. Mixed-precision inherently introduces challenges in establishing precise criteria and policies for bit-width allocation across channels, which often leads to overly practical and heuristic-based solutions. Without a solid theoretical foundation, the method risks being overly tailored to specific scenarios, limiting its generalizability and making it challenging to adapt across varying large language models.

2. Mixed-precision quantization introduces unstructured patterns in model weights, which can negatively impact the execution speed on real-world applications, particularly in hardware environments where consistent bit-width allocation is optimal. The paper does not adequately address these potential speed penalties, which are crucial for deployment. Rather than leaving this as a topic for "further work," an initial analysis of the impact on runtime performance would provide a more comprehensive view of CMPQ’s practicality, especially in latency-sensitive applications.

3. The paper primarily demonstrates the effectiveness of CMPQ through Wiki and C4 perplexity scores, which, while relevant, do not provide a comprehensive view of performance for large language models (LLMs) in complex tasks. In current LLM research, deeper benchmarks—such as MMLU, GSM8K, or win-rate metrics against existing models like GPT—are essential to capture a model’s real-world utility and robustness. Over-reliance on perplexity can be misleading, as slight improvements in perplexity may not translate to meaningful gains in more challenging tasks, where small differences in Wiki perplexity can result in significant performance gaps in harder benchmarks. A broader evaluation would provide stronger evidence of CMPQ’s effectiveness in realistic settings and make the findings more comparable with current LLM standards.

### Questions
included in Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this manuscript, the authors introduce a channel-wise mixed-precision quantization (CMPQ) method designed to efficiently compress large language models. Unlike previous mixed-precision quantization approaches, which use a fixed bit allocation per layer, the proposed method enables different bit allocations for individual channels, inspired by insights from the 'Massive Activation' paper. By incorporating several outlier protection techniques, the authors demonstrate that a modest increase in memory usage can lead to significant improvements in accuracy across various language benchmarks.

### Strengths
The motivation is compelling, as Figure 2 clearly illustrates that different channels, particularly those associated with tokens exhibiting massive activation, have varying impacts on overall accuracy. This suggests that using uniform bit allocations across all channels could be highly inefficient. To address this, the authors incorporate established outlier detection techniques to manage these outlier variations effectively. Their results show that only a few channels require different bit allocations, while most channels remain relatively insensitive to changes introduced by these outlier handling methods.

### Weaknesses
Unfortunately, this manuscript faces several critical issues that limit its practicality. Here are some examples to illustrate these concerns:

- The manuscript does not address how to accelerate the proposed quantization method effectively. Since the bit allocations vary across different channels, parallel operations that process multiple channels simultaneously could introduce significant performance bottlenecks, requiring special handling to mitigate these issues.

- The selection of outlier thresholds (such as preserving 0.45% of the largest values in FP16 precision) appears overly empirical. These thresholds are either adopted from prior work without sufficient context or presented without robust justification, undermining the method's theoretical foundation.

- The approach relies on augmenting quantized weights with a small amount of FP16 data, but the manuscript fails to discuss the impact of this additional FP16 memory on overall computational complexity or performance. The experimental results do not account for the implications of handling this extra FP16 data.

- The manuscript uses only perplexity (PPL) as the metric for evaluating quantization quality. However, more comprehensive metrics, such as MMLU, are necessary to better reflect the capabilities of modern large language models, as PPL alone does not provide a complete assessment of performance.

### Questions
What is the performance of LLMs when using the proposed quantization method? How much degradation occurs as a result of the additional FP16 computations?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes a Channel-Wise Mixed-Precision Quantization method to optimize quantization by leveraging activation distributions, offering a more fine-grained approach compared to traditional layer-wise quantization.

### Strengths
The proposed method shows better performance over GPTQ/AWQ, etc., on various large language models.

### Weaknesses
1.  The contribution of this paper is relatively minor and could be considered essentially an extended version of SmoothQuant to a limited extent.

2.  By comparing Tables 1 and 2, the performance improvements over the baseline are not stable and, in some cases, are worse. For example, under 4-bit precision on the C4 dataset for LLaMA2-7B, compared with QuIP (7.10 vs. 6.69), more discussion is required.

3.  The ablation results show that w/o Oact has little effect, indicating that output activation is not crucial for performance.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
