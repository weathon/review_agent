# WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference

- Avg Score: 5.14
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 6, 6, 2

## Abstract
The ever-increasing computational demands of large language models (LLMs) make efficient inference a central challenge. While recent advances leverage specialized architectures or selective activation, they typically require (re)training or architectural modifications, limiting their broad applicability. Training-free sparse activation, in contrast, offers a plug-and-play pathway to efficiency; however, existing methods often rely solely on hidden state magnitudes, leading to significant approximation error and performance degradation. To address this, we introduce WINA (Weight-Informed Neuron Activation): a simple framework for training-free sparse activation that incorporates both hidden state magnitudes and weight matrix structure. By also leveraging the ℓ2-norm of the model’s weight matrices, WINA yields a principled sparsification strategy with provably optimal approximation error bounds, offering better and tighter theoretical guarantees than prior state-of-the-art approaches. Overall, WINA also empirically outperforms many previous training-free methods across diverse LLM architectures and datasets: not only matching or exceeding their accuracy at comparable sparsity levels, but also sustaining performance better at more extreme sparsity levels. Together, these results position WINA as a practical, theoretically grounded, and broadly deployable solution for efficient inference. Our source code is available at https://github.com/microsoft/wina.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a framework WINA for training-free sparse activation that incorporates both hidden state magnitudes and weight matrix structure, which combines the magnitude of activations with the column-wise norm of the weight matrices to preserve the top-k activations. The authors claimed that WINA can achieve a lower approximation error bound under several assumptions and is model-agnostic. Methods are tested on Llama-2-7B, Llama-3-8B, Mistral-7B, and Phi-4-14B models across several benchmark datasets, which demonstrate that WINA can achieve superior performance under various sparsity ratios.

### Strengths
1. The combinatorial gating strategy is reasonable, which produces a tighter approximation error bound.
2. WINA as a training-free method is friendly for deployment.
3. The paper is well written and easy to follow.

### Weaknesses
1. There are no end-to-end latency performance comparisons between WINA and previous methods like TEAL, CATS, and R-Sparse.
2. For the math reasoning task like GSM8K, the aggressive sparsity can induce significant damage to the model performance, dropping accuracy from 50 to 7, although WINA reported superior performance to the baseline methods.
3. When the batch size is larger, will the sparsity be affected and further the speedup gain be degraded?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents WINA (Weight-Informed Neuron Activation), a training-free sparse activation method that accelerates LLM inference by selecting neurons based on both hidden state magnitudes and weight matrix norms. This weight-informed approach achieves tighter theoretical error bounds and better accuracy than prior methods like TEAL and CATS, maintaining strong performance even at high sparsity levels across various LLMs and tasks, and showing compatibility with quantized models.

### Strengths
1. The paper is well-organized and easy to follow.

2. The figures are clearly and beautifully presented.

3. The experiments conducted on extensive datasets provide strong validation and demonstrate the integrity of the proposed method.

### Weaknesses
1. My main concern is related to the performance measurement. The authors claim that WINA is more efficient than previous methods, "potentially translating to faster inference speeds and lower computational costs." Could the authors provide empirical evidence, such as wall-clock time or GPU memory usage, to support this claim?

2. The improvement in synthetic results shown in Table 2 is substantial, but the gains in real-world LLM experiments are relatively modest. Could the authors clarify the reason for this large discrepancy between synthetic and real-world results?

3. How can the assumption of "column-wise orthogonality" in the theorems be verified? Is there any experimental evidence to support this assumption?

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper aims to improve inference efficiency of LLMs and proposes skipping unnecessary neuron computations in feed-forward network (FFN) layers. The proposed method does this with a gating function that uses the magnitude of a neuron's output weight to assess its importance. Less important neurons (smaller output weight magnitue and smaller intermediate activation) are skipped. The paper has empirical results on Llama 3-8B and Qwen 2-7B, demonstrating significant speedups with minimal impact on model accuracy.

### Strengths
- The proposed method, WINA, is easy to apply to existing pre-trained LLMs since it is a post-training method that does not require any fine-tuning or retraining. 

- To the best of my knowledge, assigning importance scores based on output weight's magnitude is a novel idea. 

- Empirical results are strong across various benchmarks and model sizes.

### Weaknesses
- The linked source code is not available. 

- The success of WINA relies on the threshold used to determine which neurons to skip. Tuning this threshold would be costly.

### Questions
Do the authors have any suggestions on how to tune the threshold efficiently?

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
The paper proposes a new sparse activation method, WINA, to improve LLM inference efficiency. WINA or Weight Informed Neuron Activation uses both the hidden state magnitude and weight matrix structure while sparsifying the activation; previous work only relies on the hidden magnitude. 
WINA uses the product of l2 norm of the column vector of the weight matrix and the hidden state magnitude to select the top-K neuron with theoretical justification. Extensive experiments are provided with popular benchmark datasets.

### Strengths
1. The paper is well-written and easy to follow, and the idea is intuitive. 

2. Theoretical justification showing that using both the L2 norm of the column vector and the hidden state magnitude yields an optimal solution and reduces error (section 3). 

3. Results provided are quite extensive; the method is evaluated on multiple datasets and different downstream tasks. 

4. Results in tables 3 and 4 show consistent improvements, especially at higher sparsities, compared to other baselines, showing the efficacy of the method.

5. Additional results showing on quantization are provided, showing WINA is compatible with quantization.

### Weaknesses
1. The technical novelty of the method is limited (however, results show it improves over baselines). 

2. It is not clear why the orthogonality of the weight matrix is enforced (in sec 3.4)? Does this orthogonality hold in a general setting as well?

3. Recent works have shown that LLM compression has an unintended impact on the model bias. It would be helpful to also evaluate the impact of the proposed method on model bias. 


[1]. Strubell et al., Understanding the Effect of Model Compression on Social Bias in Large Language Models

### Questions
1. How is LayerNorm monotonically increasing? (Line 238)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes WINA (Weight Informed Neuron Activation), a training-free sparse activation method that combines hidden state magnitudes with the weight matrix structure to guide neuron selection. WINA is proven to minimize approximation error under column-wise orthogonality and monotonic activation assumptions. In the experiments, it outperforms other training-free methods like CATS, R-Sparse and TEAL across multiple LLM architectures (Llama2/3, Mistral, Phi-4) and benchmarks (MMLU, GSM8K, HumanEval), achieving over 60% FLOPs reduction at 65% sparsity while preserving accuracy.

### Strengths
- The proposed method introduces a simple yet effective training-free sparse activation mechanism that combines both hidden-state magnitudes and the column-wise L2-norm of weight matrices to guide neuron selection.
- The theoretical analysis is rigorous and well structured, providing provably optimal approximation error bounds under clear and interpretable assumptions (column-wise orthogonality and monotonic activation).
- The experiments are comprehensive, covering multiple model architectures, quantization methods, and ablations, demonstrating consistent improvements that align with theoretical predictions.

### Weaknesses
- The models in the experiments are small dense LLMs. Large-scale or MoE architectures (e.g., DeepSeek-V3, Llama4, GPT-OSS) which are more common in product deployment workloads are not tested. It’s unclear whether WINA’s activation gating would maintain efficiency with expert routing sparsity in these larger models.
- The evaluation focuses on theoretical FLOPs reduction but lacks real-world inference measurements such as latency or throughput on inference frameworks. Without kernel-level or runtime validation, the practical performance benefits of WINA remain unclear, especially given the hardware inefficiency of non-structured sparsity.
- The theoretical assumptions rely on column-wise orthogonality and monotonic activation functions, which may not strictly satisfied in real transformer models.

### Questions
- How does WINA perform on large MoE models (e.g., DeepSeek, Llama4, GPT-OSS)? It would help to understand how the method scales to production scale LLMs.
- Could you evaluate WINA’s actual latency or throughput in real inference scenarios? What’s the challenges to integrate this method to inference frameworks?
- How does WINA perform under long-context settings (e.g., 16K–128K tokens)? Are the top-K activation patterns stable as sequence length increases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses the important challenge of reducing inference cost in LLMs without degrading output quality. Existing training-free sparse activation methods often rely solely on hidden state magnitudes, which can lead to significant approximation error, particularly at high sparsity levels.

The authors propose WINA (Weight-Informed Neuron Activation), a simple yet effective framework that incorporates both hidden state magnitudes and the ℓ2-norm of weight matrices into neuron selection. This provides a principled sparsification strategy with provably optimal approximation error bounds, yielding tighter theoretical guarantees than prior methods.

The method is empirically validated across multiple widely used LLMs, including Llama-2-7B, Llama-3-8B, Mistral-7B, and Phi-4-14B, and evaluated on diverse tasks such as general reasoning (MMLU), mathematics (GSM8K), and coding (HumanEval). WINA is compared against several strong baselines, including TEAL, R-Sparse, and CATS. The results show that WINA performs comparably to prior methods at low sparsity and significantly better at high sparsity, achieving several percent improvement in commonsense reasoning accuracy and sustaining performance under extreme sparsity levels.

Overall, WINA is presented as a practical, theoretically grounded, and broadly deployable approach for efficient inference in LLMs.

### Strengths
The problem addressed is highly relevant, as reducing LLM inference cost without sacrificing output quality is an important challenge. The method is theoretically grounded, as incorporating weight norms provides a principled sparsification strategy with provable error bounds. The empirical evaluation is extensive, covering multiple LLMs, a range of tasks, and both low and high sparsity levels, and the method is compared to strong baselines including TEAL, R-Sparse, and CATS. The approach is practical and easy to deploy, as it is training-free and plug-and-play, making it broadly applicable. The results demonstrate robustness, as the method maintains competitive performance across different sparsity regimes and models.

### Weaknesses
The main contribution is a relatively straightforward extension of existing sparse activation methods, which could be considered incremental, though it is strengthened by solid theoretical and empirical support. The paper could benefit from a discussion of potential limitations, such as scenarios where weight-informed selection might be less effective or challenges when scaling to very large models beyond those tested.

### Questions
The authors propose incorporating weight norms into the neuron selection process. Could the authors clarify whether this additional step increases computation amount during inference, and if so, provide benchmark comparisons to quantify the overhead relative to other training-free sparse activation methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 7

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- Proposes WINA, a training-free sparse activation method that gates neurons using both activation magnitude and column-wise weight ℓ₂ norms.

### Strengths
- Very simple, plug-and-play rule that is easy to implement on top of existing sparse-activation baselines.

### Weaknesses
- The paper reports GFLOP reductions but does not clearly explain whether WINA’s gating is used to avoid weight loads or only to mask post-matmul activations; without a truly sparse kernel and latency measurements, it is unclear how much real speedup WINA provides over TEAL/CATS in memory-bound, batch-1 inference.

### Questions
- In your current implementation, is the WINA gate used before the matmul to index only a subset of columns of 𝑊 , or do you compute a dense 𝑊𝑥 and then apply the mask? If it’s the latter, how do you obtain any wall-clock speedup, especially in batch-1, memory-bound settings?

- Comparison to TEAL/CATS kernels: TEAL/CATS explicitly discuss sparse kernels that reduce weight loading per token. Do you implement a comparable kernel for WINA, and can you report latency/throughput numbers vs TEAL/CATS on real hardware (A100, L40S, etc.), not just GFLOP estimates?

### Soundness
2

### Presentation
2

### Contribution
1
