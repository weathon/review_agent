# LoRTA: Low Rank Tensor Adaptation of Large Language Models

- Decision: Reject
- Scores: 3, 5, 6, 3

## Abstract
Low Rank Adaptation (LoRA) is a popular Parameter Efficient Fine Tuning (PEFT) method that effectively adapts large pre-trained models for downstream tasks. LoRA parameterizes model updates using low-rank matrices at each layer,  significantly reducing the number of trainable parameters and, consequently, resource requirements during fine-tuning. However, the lower bound on the number of trainable parameters remains high due to the use of the low-rank matrix model. In this paper, we address this limitation by proposing a novel approach that employs a low rank tensor parametrization for model updates. The proposed low rank tensor model can significantly reduce the number of trainable parameters, while also allowing for finer-grained control over adapter size. Our experiments on Natural Language Understanding, Instruction Tuning, Preference Optimization and Protein Folding benchmarks demonstrate that our method is both efficient and effective for fine-tuning Large Language Models, achieving a reduction in the number of parameters while maintaining comparable performance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper considers a new PEFT method based on tensor decomposition. Compared with LoRA method, the method studies and explains a general case of sharing parameters LoRA method from the perspective of CP decomposition. Experimental results are provided among multiple tasks, like NLU, DPO and protein folding benchmarks.

### Strengths
The method in the paper studied the previous parameter-sharing PEFT method from the aspect of CP decomposition and propose the new LoRTA method. The connection between these methods is interesting and the presentation of this paper is clear.

### Weaknesses
I have three main concerns for the paper:
- First, the experimental performance of the proposed method is trivial. Especially for NLU tasks, there exists a degradation around 3% compared with the original LoRA method. Even compared with VeRA method, there is still a degradation of over 1%. I think it's not economical to further reduce the trainable parameter with such large performance degradation, as the benefit of memory reduction is limited, considering the trainable parameter for LoRA is already small enough.
- Second, the difference between LoRTA and FACT paper (Jie & Deng, 2023) mentioned is limited. Even though the analysis for the connection between parameter-sharing methods and LORTA is interesting, I still feel the LoRTA method is more like a CP version of FACT method. It would be great if the authors could further explain the difference between these methods.
- Third, the motivation for the method is not clear. The authors didn't show the benefit of further reducing the trainable parameter, considering things like memory and training time efficiency, as there exists performance degradation compared with the LoRA method.

### Questions
I have a list of questions here, I would appreciate it if the authors could consider these questions:
- There is some confusion regarding the parameter count for LoRA in line 127. Could you clarify where the number 8 originates? LoRA is not restricted to being injected only in the query and value layers, nor is it required to be applied exclusively to those layers.
- Lack of discussion for related work: Besides the FACT paper (Jie & Deng, 2023) mentioned in the work, there are multiple works considering utilizing tensor decomposition to further reduce trainable parameters specifically for LLMs fine-tuning. I think these works should at least be mentioned in this paper. For example:
1. Yang, Yifan, et al. "LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models," NAACL 2024.
2. Bershatsky, Daniel, et al. "LoTR: Low tensor rank weight adaptation." arXiv preprint arXiv:2402.01376 (2024).
- Motivation for reducing the trainable parameters: I think it's important to discuss why we need a PEFT method with ultra low parameters (maybe from memory efficiency, and training time), considering the performance degradation shown in the NLU tasks and similar performance in other comparisons.
- Additional experimental results
I'm wondering if the authors could do some quick tests to further support the effectiveness of their methods:
1. Memroy/training time efficiency: As previously mentioned, this hardware profiling is important to support the motivation of the problem studied. Based on my experience, further reducing the parameters from LoRA didn't help to further reduce the training memory a lot.
2. Further test with larger rank: I see the trainable parameter of LoRTA method is around 10 times less than VeRA in Table 1. I'm wondering how LoRTA performs if we could increase the rank? Will the performance be improved?
3. Also, some comparison between LoRTA and FACT methods is also suggested, to show the effectiveness of considering CP, instead of tucker and TT decomposition.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper investigates the efficient fine-tuning of LLMs through a method called LoRTA. The main issue addressed is that existing low-rank adaptation methods, such as LoRA, although reducing the number of trainable parameters, still have a high lower bound on the number of parameters. To solve this problem, the paper proposes using a low-rank tensor parameterization model for model updates, significantly reducing the number of trainable parameters and providing finer-grained control over adaptation. Experiments on benchmarks demonstrate that the LoRTA method maintains comparable performance to existing methods while achieving a substantial reduction in the number of parameters.

### Strengths
1. The paper writing is clear, and the method is simple and easy to reproduce.
2. The paper studied a valuable problem, and well solved the limitations of existing methods, achieving good results on multiple benchmarks.

### Weaknesses
1. As the author stated in the introduction, the main purpose of the Lorta method is to solve the problem of efficient finetune of LLM. Therefore, I expect to see more LLM-related results in the paper experiment. The author only conducted experiments on llama2-7B and mt-bench, and I expect to see more LLM results such as llama-3-70B, Mistral-7B, etc.

2. The author conducted experiments on multiple benchmarks, but there are few results on the mainstream LLM evaluation benchmarks. The GLUE benchmark task is relatively simple and may not have a good distinction granularity. I suggest that the author add some difficult tasks that are mainly used to evaluate LLM, such as MMLU, MATH, HumanEval, etc.

3. Figure 3 shows that the performance of Lorta decreases as the training parameters increase (i.e., the effect decreases as the computational load increases). This seems to be quite disadvantageous for this method, that is, this method is not suitable when more GPU resources are available. It suggests that the method's adaptability is rather restricted. In addition, I would like to know if Lora has the same trend under the settings of Figure 3.

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LoRTA, a parameter-efficient fine-tuning method that uses low-rank tensor parametrization for model updates. The method aims to reduce the number of trainable parameters compared to LoRA while providing more fine-grained control over adapter size. The approach is evaluated on Natural Language Understanding, Instruction Tuning, Preference Optimization, and Protein Folding benchmarks.

### Strengths
The method is presented clearly with comprehensive preliminaries and detailed methodology. The approach shows potential for significant parameter reduction in model fine-tuning.

### Weaknesses
While the idea is interesting, the evaluation is not comprehensive enough to demonstrate its practical utility. Moreover, the claims about its performance are exaggerated:
- Line 021: "_[...] achieving a substantial reduction in the number of parameters while maintaining comparable performance_"
- Line 076: "_[...] compared to state-of-the-art PEFT methods, with minimal performance trade-offs_"
- Line 478: "_LoRTA achieves comparable and sometimes superior performance than baselines at a reduced parameter count_"

In Natural Language Understanding, LoRTA achieved **the second worst** performance out of 8 methods presented. With such degradation of performance, a lower number of parameters might not be worth it. Can we achieve closer performance to LoRA if we scale the rank of LoRTA? What does the parameters-performance trade-off look like?

Regarding the other 3 experimental settings (Instruction Tuning, Preference Optimization, Protein Folding), LoRTA was compared only to one baseline - LoRA of **rank 1**. Given the weak performance on the NLU setup, the results from Protein Folding & Preference Optimization are not convincing. LoRTA should be compared to more baselines - possibly the methods introduced earlier, full-finetuning, and LoRA of higher rank. Rank 1 is suboptimal for LoRA, and with ranks, e.g., 8 or 16 we may obtain significantly better performance.

### Questions
- What is the tradeoff between the number of parameters and performance of this method? Can we obtain the same or better results than other methods on NLU if we use more parameters in LoRTA?
- How do the training speed and memory usage compare to LoRA?
- How were the hyperparameters found for the LoRA baseline in Preference Optimization and Protein Folding? It's possible that a stronger baseline could be found by hyperparameter tuning, especially using different ranks for LoRA.


Small suggestions:
- "X is all you need" is _a bit_ overused phrase in DL papers. I'd recommend to change the name of Section 3 - but it's not that important.
- Figures 2 & 4: Lower training loss is not indicative of better downstream performance (as also shown with Figure 3). It's better not to include it as part of experimental results showing superiority of one method over another - can be moved into the appendix.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors present a low-rank adaptation method called LoRTA. It considers all attention matrices (Q, K, V, and projection) in L layers as a single tensor and trains the adapter in a low-rank CP decomposition. In this approach, the size of the weight tensor is $d \times \frac{d}{H} \times H \times L \times 4$, where $H$ is the number of attention heads, $L$ is the number of all layers, and 4 is the number of different matrices.

### Strengths
Experimental results are very good considering the small rank (r=4 or r=8) of the update and, as a result, a very small number of parameters.

### Weaknesses
My major concern is the novelty of this idea. 

A very similar concept was previously implemented in "LoTR: Low Tensor Rank Weight Adaptation" by Daniel Bershatsky, Daria Cherniuk, Talgat Daulbaev, Aleksandr Mikhalev, and Ivan Oseledets (https://arxiv.org/abs/2402.01376).

In LoTR, the same CP decomposition is applied to the tensor of stacked attention weights. The only difference is that, in LoTR, the tensor of all weights is consolidated into a three-dimensional tensor without separating the dimensions for attention heads and matrices within the same layer.

### Questions
* How did you initialize the model?
* What is the test quality of the ESM-2 model without fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
1
