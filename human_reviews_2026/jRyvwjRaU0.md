# Gradually Compacting Large Language Models for Reasoning Like a Boiling Frog

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, but their substantial size often demands significant computational resources. To reduce resource consumption and accelerate inference, it is essential to eliminate redundant parameters without compromising performance. However, conventional pruning methods that directly remove such parameters often lead to a dramatic drop in model performance in reasoning tasks, and require extensive post-training to recover the lost capabilities. In this work, we propose a gradual compacting method that divides the compression process into multiple fine-grained iterations, applying a $\underline{P}$rune–$\underline{T}$une $\underline{L}$oop ($\texttt{PTL}$) at each stage to incrementally reduce model size while restoring performance with finetuning. This iterative approach—reminiscent of the "boiling frog" effect—enables the model to be progressively compressed without abrupt performance loss. Experimental results show that $\texttt{PTL}$ can compress LLMs to nearly half their original size with only lightweight post-training, while maintaining performance comparable to the original model on reasoning tasks. Moreover, $\texttt{PTL}$ is flexible and can be applied to various pruning strategies, such as neuron pruning and layer pruning, as well as different post-training methods, including continual pre-training and reinforcement learning. Additionally, experimental results confirm the effectiveness of $\texttt{PTL}$ on a variety of tasks beyond mathematical reasoning, such as code generation, demonstrating its broad applicability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of compressing Large Language Models while preserving their reasoning capabilities, where standard pruning methods often fail, leading to significant performance degradation. The authors propose a iterative compression method called the Prune-Tune Loop (PTL). The core idea is to avoid abrupt performance collapse by applying gradual changes. The authors conduct extensive experiments on modern LLMs (Llama3-8B, Gemma2-9B, Qwen2.5-7B) and show that PTL can reduce model size by 40-50% while maintaining performance on math and code domain

### Strengths
1. The proposed PTL method provides effective solution that achieves substantial FLOPs reduction (30-50%) and measured speedup (up to 2.6x) with minimal performance loss.

2. The paper is well-written. The PTL method is introduced with a clear diagram, simple mathematical formulations, and a clean algorithm.

### Weaknesses
1. The metrics used to identify redundant parameters are standard and relatively simple. While their effectiveness is proven by the results, the paper could be strengthened by a detailed discussion justifying this choice and the reason why it can help for perserve reasoning capability.

2. The pruning model are finetuned using reasoning-focused corpus, which is straightforward to recover performance and implies the pruning is guided by performance on this specific domain. How sensitive is the method to this dataset across different domains? Does this "reasoning-focused" cause disproportionate harm to other, non-reasoning capabilities? A brief discussion on the trade-offs of using a specialized dataset for pruning versus a general-purpose corpus would be valuable.

### Questions
The result that all baseline pruned models fail (0% accuracy) during RL recovery is striking. Do you have a hypothesis for this? Is it that the large, one-shot structural changes from baseline pruning move the model's policy into a region of the parameter space from which RL (GRPO) cannot recover, whereas PTL's gradual changes keep the policy within a stable rajectory?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Prune-Tune Loop (PTL), a gradual model compression method for Large Language Models that maintains thier capabilities while reducing model size by 30-40%. The key idea of the paper is dividing the compression process into multiple iterations, where each iteration removes a small portion of redundant parameters followed by lightweight recovery training using either continual pre-training or reinforcement learning. Experiments on Llama3-8B, Qwen2.5-7B, and Gemma2-9B demonstrate that PTL can compress models to ~60% of their original size while maintaining comparable performance on mathematical reasoning benchmarks (GSM8K, Minerva Math, MATH-500) and code generation tasks.

### Strengths
1. The paper demonstrates impressive compression ratios (30-40% parameter reduction) with minimal performance loss across multiple state-of-the-art models. Particularly notable is the Gemma2-9B result where PTL is the only method maintaining near-original performance after aggressive pruning.
2. The evaluation spans three different model architectures, multiple mathematical reasoning benchmarks, and includes code generation tasks. The ablation studies examining pruning iterations, step sizes, and ordering provide valuable insights.
3. The formalization of redundant neuron and layer identification is mathematically precise, with clear equations defining the importance metrics based on activation patterns and embedding changes.

### Weaknesses
1. The core contribution is essentially iterative application of existing techniques (magnitude-based pruning, importance scoring via activation norms, and recovery fine-tuning). The "Prune-Tune Loop" is not fundamentally different from gradual magnitude pruning approaches that have been explored in the literature.
2. The paper lacks any theoretical analysis. There are no convergence guarantees, compression bounds, or formal analysis of why multiple small pruning steps outperform single large steps. Prior work in optimization and neural network theory could provide relevant frameworks that the authors ignore.
3. This work is primarily an engineering exercise that combines well-known techniques without advancing our theoretical understanding of LLM pruning and recovery. The paper does not provide new insights into how LLMs encode reasoning capabilities or why gradual removal preserves them better.
4. No comparison with iterative magnitude pruning, gradual sparsity methods, or recent structured pruning techniques that use similar multi-step approaches
5. The redundancy identification using reasoning-specific sequences (Equations 4 and 6) is ad-hoc without justification for why these specific metrics capture reasoning-irrelevant parameters

### Questions
1. What is the total wall-clock time and compute cost for the complete PTL process compared to (a) training a smaller model from scratch, and (b) single-step pruning with equivalent recovery training time?
2. How are σ_neuron and σ_layer determined for each model? Is there a systematic approach or does it require extensive grid search? How sensitive is the final performance to these choices?
3. Why does performance sometimes improve in early pruning iterations (e.g., GSM8K: 70.0% → 77.6% for Gemma2)? Is this due to regularization effects or the recovery training data?
4. The redundant parameter identification uses reasoning-specific sequences. How does this affect the model's performance on non-reasoning tasks? Would using mixed-task data for identification preserve broader capabilities?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a progressive compression method for Large Language Models (LLMs) called the Prune-Tune Loop (PTL). This method gradually removes redundant parameters (neurons or layers) through multiple iterative "pruning-fine-tuning" cycles and leverages Chain-of-Thought (CoT) data or reinforcement learning for performance recovery. Experiments were conducted on models such as Llama3-8B, Gemma2-9B, and Qwen2.5-7B. The experimental results show that with only lightweight subsequent training, PTL can compress an LLM to nearly half of its original size while maintaining performance comparable to the original model on reasoning tasks.

### Strengths
1. The paper is well-written and easy to follow.
2. The core idea of the PTL method—progressive compression to avoid sudden performance degradation—is reasonable. Compared with one-time pruning (e.g., Prune-Once), this method demonstrates better performance recovery.
3. The paper conducts extensive tests on multiple open-source models and benchmarks, including models such as Llama3-8B, Qwen2.5-7B, and Gemma2-9B, as well as mathematical reasoning benchmarks (GSM8K, Minerva Math, MATH-500) and the code generation benchmark (MBPP). The results show that the compressed model achieves performance close to that of the original model, while significantly optimizing FLOPs and inference speed.

### Weaknesses
1. In my view, the biggest issue with this paper is lack of novelty. Progressive pruning and fine-tuning are not novel concepts—they were widely applied during the era of Convolutional Neural Networks. For instance, the idea of progressive pruning and fine-tuning was proposed quite early in [1]. The authors merely extended progressive pruning and fine-tuning to the pruning of LLMs, and the continuous pre-training and reinforcement learning methods they adopted are also existing algorithms.
2. The reinforcement learning training experiments were conducted on eight 140GB NVIDIA H200 GPUs, and the fine-tuning of LLMs at the 7B scale takes 64 hours, which still constitutes a non-negligible cost. If scaled to LLMs of larger scales (e.g., the 70B scale), the fine-tuning cost will be even higher.
3. The hypothesis that only pruning the FFN does not affect attention is based on the existing conclusion that "reasoning mainly depends on self-attention," and there is a lack of an experiment to verify the above hypothesis.
4. Typos: Line 174, Reinformence Learning->Reinforcement Learning

[1] Molchanov P, Tyree S, Karras T, et al. Pruning Convolutional Neural Networks for Resource Efficient Inference [C]. International Conference on Learning Representations. 2017.

### Questions
1. How are the thresholds for redundant parameter extraction ($σ_{neuron}$ and $σ_{layer}$) selected? The authors seem to have omitted this detail.
2. In the reinforcement learning-based recovery process, why did other baselines fail (achieving 0% accuracy)? Is it due to model collapse caused by structural changes, or issues with training configurations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies that standard one-shot pruning methods cause catastrophic failure in mathematical and code reasoning tasks. To solve this, the authors propose an iterative compression framework Prune-Tune Loop (PTL) . In each fine-grained iteration, the model is pruned by removing redundant reasoning parameters and then recovered via lightweight post-training (using Continual Pre-training on Chain-of-Thought data or Reinforcement Learning). Experiments on models like Llama3-8B and Gemma2-9B show that PTL can compress models to ~60% of their original size while maintaining reasoning accuracy.

### Strengths
- The pruning and tuning loop makes sense to me. It is well-motivated to recover the model's ability after pruning.

- The ablation studies on pruning step size, iterations, and order provide valuable insights into the method's behavior and stability.

- The comparison against the one-shot pruning methods demonstrates their effectiveness in ability recovery.

### Weaknesses
- I think a significant weakness is the lack of discussion on scalability. The method is only validated on models up to 9B parameters. Given its reliance on multiple rounds of post-training, the computational cost of larger models (e.g., 70B+) remains a critical question. 

- The method appears sensitive and requires per-model tuning. The need for different recovery strategies (RL for Qwen vs. Continual Pre-training for others) and the significant performance variance with different pruning step sizes (5% vs. 20%) suggest PTL is not a plug-and-play solution.

### Questions
- What is your estimate of the computational cost of applying PTL to a much larger model like a 70B model? 

- While the focus of this work is math reasoning, do you believe this framework, without dedicated modification, can be adapted to other domains like logical deduction?

### Soundness
3

### Presentation
3

### Contribution
2
