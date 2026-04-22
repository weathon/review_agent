# PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Recently, significant progress has been made in developing reasoning-capable Large Language Models (LLMs) through long Chain-of-Thought (CoT) techniques.
However, this long-CoT reasoning process imposes substantial memory overhead due to the large Key-Value (KV) Cache memory overhead.
Post-training KV Cache quantization has emerged as a promising compression technique and has been extensively studied in short-context scenarios.
However, directly applying existing methods to long-CoT LLMs causes significant performance degradation due to the following two reasons: 
(1) Large cumulative error: Existing methods fail to adequately leverage available memory, and they directly quantize the KV Cache during each decoding step, leading to large cumulative quantization error.
(2) Short-context calibration: Due to Rotary Positional Embedding (RoPE), the use of short-context data during calibration fails to account for the distribution of less frequent channels in the Key Cache, resulting in performance loss.
We propose Progressive Mixed-Precision KV Cache Quantization (PM-KVQ) for long-CoT LLMs to address the above issues in two folds:
(1) To reduce cumulative error, we design a progressive quantization strategy to gradually lower the bit-width of KV Cache in each block. Then, we propose block-wise memory allocation to assign a higher bit-width to more sensitive transformer blocks. 
(2) To increase the calibration length without additional overhead, we propose a new calibration strategy with positional interpolation that leverages short calibration data with positional interpolation to approximate the data distribution of long-context data.
Extensive experiments on 7B–70B long-CoT LLMs show that PM-KVQ improves reasoning benchmark performance by up to 8% over SOTA baselines under the same memory budget and achieves 2.73–5.18$\times$ throughput over the original 16-bit LLMs.
Our code is available at https://github.com/thu-nics/PM-KVQ.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses KV cache quantization from a utility-driven perspective. The authors hypothesize that by fully utilizing the available memory and reducing the quantization bit-width only when the memory becomes full, one can achieve better CoT reasoning accuracy. Based on this idea, they propose a progressive quantization approach with an Equivalent Right Shift strategy. Specifically, all KV caches are stored in full precision initially, and once the memory limit is reached, the caches are progressively quantized by half using the proposed strategy. The authors further suggest assigning different memory budgets to each transformer block and formulate this allocation as an integer programming problem. To enable effective calibration for long contexts, they adopt positional interpolation to approximate long-context calibration using short-context data. Experimental results show that progressive quantization outperforms state-of-the-art fixed-precision quantization methods while maintaining comparable efficiency.

### Strengths
1. Approaching KV cache quantization from a utility-driven perspective is interesting and practically relevant, as it reflects real-world deployment considerations.

2. The figure illustrating the main idea of the paper is well-designed and easy to follow, effectively conveying the core concept.

3. The authors conduct experiments on multiple models, demonstrating the broad applicability and effectiveness of the proposed method.

### Weaknesses
1. My main concern lies in the fairness of the experimental results presented in Table 1. The baseline comparison is rather limited, as KIVI serves as the only comparable baseline in most evaluations. Since KIVI uses a fixed precision while the proposed method can employ higher precision during generation, the comparison may not be entirely fair. It would be helpful to report the memory usage during generation for both KIVI and the proposed method to provide a more complete picture. It remains unclear whether the observed improvement primarily comes from the fact that the proposed approach occupies more memory overall during generation, which actually contradicts the reason why we apply quantization, as we want to reduce the memory usage.

2. Because the proposed method depends on fully utilizing the available memory, it is important to evaluate its performance across different hardware configurations with varying memory capacities. Such an analysis would clarify how accuracy scales when the available memory changes.

3. I find the use of positional interpolation insufficient to address the main challenge of long-context calibration. This method essentially distributes a small number of tokens over a wide range, leaving many positions without properly calibrated data. Moreover, according to Table 4, positional interpolation appears to offer limited improvement, which further supports this concern.

### Questions
1. How does the method compare to KIVI that also fully utilizes the available memory? Would there be a way to conduct such a fair comparison?

2. For the Equivalent Right Shift strategy, is it reasonable to keep the zero point unchanged? Would it be possible for the zero point to vary across different quantization bit levels?

3. In Table 4, when s increases to 16, the use of positional interpolation appears to cause performance degradation. Does this imply that the proposed method has inherent limitations? Overall, the method’s effect on pass@1 seems marginal.

4. There is a typo around line 298: should it be CMIMC-2025 or CMIMC-2024?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PM-KVQ, a post-training quantization scheme for long-CoT LLM inference that combines: a. progressive quantization (start at higher precision and shrink the KV cache bit-width only when memory is about to run out), including an “Equivalent Right Shift” rule (`Eq. 3`) for precise bit-width shrinking, b. block-wise memory allocation, cast as a small integer program solved with CVXPY to allocate higher bit-widths to more sensitive transformer blocks; and c. calibration with positional interpolation to expose short calibration sequences to long-ctx RoPE phases. Reported results on DeepSeek-R1-Distill (7B-70B) and QwQ-32B show improved pass@1/vote accuracy over KIVI/RotateKV/MiKV at 2-4-bit KV cache, with throughput close to KIVI but below it.

### Strengths
- Clear diagnosis of two long-CoT pain points (cumulative error, RoPE low-frequency channels), with concrete formulations (Eqs. 9-12) motivating the positional interpolation trick. 
- Simple but effective shrinking rule (Eq. 3) that avoids round-trip dequantization in implementation. 
- Block-wise allocation objective is standard, implementable, and explains gains when memory is partially free.

### Weaknesses
- The paper only reports FP16 results, omitting bf16, which is the de facto standard for inference. Since bf16 offers wider dynamic range and distinct hardware behavior, excluding it leaves uncertainty about PM-KVQ’s performance and compatibility in realistic deployment settings. 
- Accuracy under fake quant: Reporting accuracy without real 2-bit/4-bit kernels weakens the claim that PM-KVQ is robust in practice.

### Questions
- **L41**: What is the reference for the claim “to generate 128K tokens”?  
- **Table 1**: I am skeptical of using **BS** in your setting. What is the number of output tokens? Since the reasoning model generates long-CoT with the mentioned BS and GPU memory, it’s most likely that we get CUDA OOM.  
- **Table 1**: **QwQ** has 32B parameters. How do you use it with one A100?  
- **Sec 3.2**: Why is **CVXPY** used? What was the reason behind that?  
- **Sec 4.1**: Your evaluation metric is **pass@k** where *k = 1*. What is *n* (number of independent trials)?  
- **Sec 4.1**: Regarding **Voting**, how many samples did you draw?  
- Could the progressive quantization step, while helpful for memory control, also hinder GPU utilization by blocking parallel execution?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper propose progressive mixed precision KV Cache quantization, of which a higher bit-width is assigned to more sensitive transformer blocks tailored for long-CoT scenarios, aiming for the low memory usage and quantization error. Extensive experiments on 7B–70B long-CoT LLMs show that the proposed block-wise and mixed precision quantization method improves reasoning benchmark performance by up to 8% over SOTA baselines under the same memory budget and achieves 2.73–5.18× throughput over the original 16-bit LLMs.

### Strengths
The progressive mixed precision quantization is interesting: 1) initially, the high bit (16-bit) quantization is used for the short-sequence; 2) then progressively shrink the bit width, while the high sensitive transformer blocks are maintained with the high bit width to narrow the quantization error; 2) and the memory allocation is block-wise, which is adaptive to the PageAttention.

Secondary, the experiments show good, especially for the long-cot tasks, the proposed method improves reasoning benchmark performance by up to 8% over SOTA baselines.

### Weaknesses
The mixed precision quantization of KV cache is mature in the academia, such as KVTuner which allocate different bit width for different layer and K/V by optimized search algorithm. So the comparison with SOTA mixed quantization methods is not enough. And the practical benefit on the hardware is not given, such as the memory access saving and the throughput increase.

### Questions
1. How can this method used with PageAttention, for the quantized KV Cache management?
2. How can this method used with sparse Attention, such QuestAttention, DSA or NSA?
3. Can you draw the theoretical analysis, why such progressive shrinking is useful for long-cot tasks?

### Soundness
3

### Presentation
3

### Contribution
3
