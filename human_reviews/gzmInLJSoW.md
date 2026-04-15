# Towards Infinite-Long Prefix in Transformer

- Decision: Reject
- Scores: 3, 5, 6, 6

## Abstract
Prompting and context-based fine-tuning methods, which we call Prefix Learning, have been proposed to enhance the performance of language models on various downstream tasks. They are empirically efficient and effective, matching the performance of full parameter fine-tuning, but the theoretical understandings are limited. In this paper, we aim to address this limitation by studying their ability from the perspective of prefix length. 
In particular, we provide a convergence guarantee for training an ultra-long prefix in a stylized setting using the Neural Tangent Kernel (NTK) framework. Based on this strong theoretical guarantee, we design and implement an algorithm that only needs to introduce and fine-tune a few extra trainable parameters instead of an infinite-long prefix in each layer of a transformer, and can approximate the prefix attention to a guaranteed polynomial-small error.
Preliminary experimental results on vision, natural language, and math data show that our method achieves superior or competitive performance compared to existing methods like full parameters fine-tuning, P-Tuning V2, and LoRA. This demonstrates our method is promising for parameter-efficient fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents an analysis of ultra-long prefix vectors (P-Tuning V2), focusing exclusively on this version of prefix-tuning. The study utilizes the Neural Tangent Kernel (NTK) to examine ultra-long prefix vector convergence and subsequently proposes NTK-attention, derived from observations in the analysis.

### Strengths
The paper analyzes P-Tuning V2 through the lens of the Neural Tangent Kernel (NTK), offering a fresh perspective on prefix-tuning. The analysis is precise, highly detailed, and appears to have involved considerable effort.

### Weaknesses
1. This analysis does not cover various prompt tuning methods but is instead focused solely on P-Tuning V2, which employs different embeddings per layer. In this case, it is evident that the model is equivalent to using two linear layers with tied weights, making this insight neither new nor surprising. This raises questions about the novelty of the analysis itself. The difficulty in analyzing prefix-tuning lies in the fact that the prefixes passed through the input layer interact with each other in every layer via attention before moving to the next layer. However, this paper lacks analysis in that regard.

2. In my view, contrary to the author’s claim, NTK-attention is not parameter-efficient. The number of parameters is quite large:  as shown on Line 345, $r = d$, resulting in $d^2+d$ parameters. This count is significantly higher than the $md$ parameters used in prefix-tuning (experiments were conducted with prefix counts $m$ ranging from 1 to 200). This discrepancy calls into question whether the comparative experiments with P-Tuning V2 and LoRA are indeed fair. The author notes using LLama 3.1 1B for the experiments, and given that this model has an embedding dimension of 2048, a considerable number of parameters are indeed used (more than x10). If the author intends to argue that their method is more parameter-efficient than prefix-tuning, they should disclose the parameter counts used in Tables 1 and 2.

3. Computational efficiency also does not surpass that of prefix-tuning. The $\text{O}(\text{L}^2)$ notation in Figure 1 is incorrect; due to the $\Phi(Q)Z$ operation in Line 5 of Algorithm 2, the complexity of NTK-attention is $\text{O}(\text{(L+d+1)L})$ ($d=r$ as mentioned above). Although Appendix Figure 3 provides a running time comparison, it is not a fair comparison with $d=32$ as the value of $m$ is ranging from $1$~$2^{16}$ (The complexity of prefix-tuning $\text{O}(\text{(L+m)L})$ is quite large in this setting. And nobody use more than 1K tokens for the prefix-tuning.). For a fair experiment, the author should test with common embedding size $d$ such as 2048 or 4096, frequently used in various foundation models.  In this experimental setting, since $m\leq d$ in most cases, NTK-attention is likely to incur an even greater computational load.

4. The advantage of prefix-tuning lies in achieving fine-tuning level performance by adjusting only the input, enabling parallel execution across various tasks. However, since NTK-attention modifies the model itself, this advantage is lost.

5. The author should conduct additional experiments across various foundation models and tasks.

### Questions
See weaknesses.

+) If the mentioned weaknesses are adequately addressed in the rebuttal, I would be willing to raise the score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper aims to enhance the theoretical understanding of "Prefix Learning" in Transformer-based models. The authors use the Neural Tangent Kernel (NTK) framework to guarantee convergence when training ultra-long prefixes, essentially addressing the theoretical limits of prefix length. They propose a new NTK-Attention algorithm that approximates infinite-long prefix learning using a few extra trainable parameters per layer, thus reducing computational complexity and memory usage. Experimental results show that NTK-Attention achieves competitive performance compared to methods like P-Tuning V2 and LoRA on datasets involving vision, natural language understanding, and mathematical inference.

### Strengths
Theoretical Contribution: The paper provides a solid theoretical analysis based on NTK to understand the efficiency and convergence of Prefix Learning, which deepens the theoretical foundation of this area.
Efficient Training: The NTK-Attention algorithm significantly reduces computational complexity and memory requirements by replacing ultra-long prefixes with a small number of additional trainable parameters.
Experimental Validation: The authors validate the performance of NTK-Attention on diverse datasets (vision, natural language understanding, and math inference) and demonstrate superior or competitive results compared to existing methods.

### Weaknesses
Lack of Practical Applicability of Theoretical Analysis: While this paper presents an NTK-based analysis to establish the theoretical foundation of Prefix Learning, there is a lack of empirical evaluation regarding its applicability to large-scale models. In particular, there is little discussion on how the approach involving ultra-long prefixes can be practically applied in real-world industrial settings.

Implementation Complexity of NTK-Attention: The NTK-Attention algorithm introduces a small number of additional trainable parameters to replace ultra-long prefixes, but this also brings about implementation complexity and the need for additional hyperparameter tuning. This could limit the method's applicability and make it challenging to ensure model stability in practical applications.

Limited Experimental Scope: The experiments validate the performance of NTK-Attention on vision, natural language, and mathematical inference datasets, but the scope of the experiments is limited. In particular, there is a lack of validation regarding scalability with very large language models and performance in diverse real-world scenarios, leaving questions about the generalizability of the proposed method. Additional evaluations on a wider range of datasets and real-world applications are needed.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper explores “Prefix Learning,” a method that enhances large language models (LLMs) by fine-tuning an added prefix matrix in each layer of a transformer. The authors investigate the effect of infinitely long prefixes on model performance through theoretical analysis using the Neural Tangent Kernel (NTK) framework. Based on this theoretical insight, they propose “NTK-Attention,” an efficient approximation algorithm that replaces the infinite prefix matrix with a finite number of trainable parameters. This method achieves polynomially small errors while significantly reducing computational and memory costs.

### Strengths
1. Strong Theoretical Foundation: The paper provides a rigorous theoretical analysis of prefix learning with ultra-long prefixes, leveraging the Neural Tangent Kernel (NTK) framework. This contributes a new perspective to understanding prefix-based methods in transformers, especially regarding convergence and scaling laws.
2. Efficient Fine-Tuning Method: The proposed NTK-Attention significantly reduces the number of trainable parameters and computational complexity by replacing the large prefix matrix with two smaller trainable parameters,  Z  and  k . This approach is practical and resource-efficient, addressing scalability issues that arise with traditional long prefix methods.
3. High Performance Across Multiple Domains: Experimental results demonstrate NTK-Attention’s competitive or superior performance on tasks in natural language understanding, vision, and mathematical inference. This versatility and effectiveness across diverse datasets and tasks underscore its broad applicability.
4. Reduced Memory and Computational Costs: NTK-Attention achieves considerable savings in memory and computation compared to standard prefix learning methods. This efficiency is particularly beneficial for large-scale models where memory and computational constraints are a concern.
5. Reproducibility: Although the code is not included in the OpenReview submission, the arXiv version of the paper provides the accompanying code, enhancing reproducibility.

### Weaknesses
1.Incomplete Evaluation Across Different Model Architectures: The experiments presented focus primarily on a few specific architectures, such as pretrained ViT and ChatGLM3-6B. The generalizability of NTK-Attention across a broader range of transformer architectures and model sizes remains unexplored, which limits the applicability of the findings to other models commonly used in different domains​.
2. Limited Applicability of NTK-Attention’s Efficiency Claims: The reduced computational complexity of NTK-Attention compared to Prefix Attention relies on the condition  m \gg L , where  m  is the prefix length and  L  is the input length. However, such conditions may not be common in many real-world scenarios, which restricts the practical applications of NTK-Attention to specific cases.
3. Inappropriate Comparison with Prefix Attention: The paper’s comparison with Prefix Attention may be less relevant, as NTK-Attention does not introduce additional prefill tokens and is structurally closer to LoRA. From Table 1, NTK-Attention achieves comparable performance to LoRA, but it introduces additional inference costs that LoRA does not have. Given these trade-offs, LoRA may provide better practical value, limiting NTK-Attention’s advantage over existing methods.

### Questions
1. Sensitivity to Hyperparameters: The paper does not extensively cover the sensitivity of NTK-Attention to key hyperparameters, such as the dimension of the feature mapping  r  and the learning rates for  Z  and  k . Could the authors provide an ablation study or guidelines for tuning these parameters? This would assist researchers in optimizing NTK-Attention across different model architectures and tasks.
2. Applicability of the  m >> L  Assumption: The computational efficiency of NTK-Attention compared to Prefix Attention assumes  m >> L , which may not hold in many real-world scenarios. Could the authors provide more context or examples where this assumption is practical? Alternatively, could they discuss how NTK-Attention performs in settings where m and L are closer in scale?
3. Justification for Comparison with Prefix Attention: Given that NTK-Attention does not introduce additional prefill tokens, it is more similar to LoRA than Prefix Attention. Could the authors clarify why they chose to emphasize the comparison with Prefix Attention rather than focusing on comparisons with methods like LoRA? This would help align the experimental evaluations with the method’s structural similarities.
4. Additional Inference Costs Compared to LoRA: NTK-Attention introduces additional inference costs, as highlighted in Table 1, where its performance is comparable to LoRA. Could the authors elaborate on scenarios where the increased inference cost of NTK-Attention would be justified over LoRA? This could clarify the practical advantages of NTK-Attention in resource-constrained environments.

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
2

### Summary
The paper proposes NTK-Attention, an alternative to Prefix-Tuning method for Transformer fine-tuning. Specifically, NTK-Attention is derived from theoretical understanding that Prefix-Tuning can be approximated using Neural Tangent Kernel (NTK) when the prefix length is sufficiently large. The key difference of NTK-Attention is to utilize a learnable weight matrix multiplied to Query, instead of concatenating a learnable prefix to Key and Value. Due to the architecture of NTK-Attention, it can avoid quadratic memory increase as the prefix length increases. Experimental results show that NTK-Attention achieves comparable performance to competitors, P-Tuning-v2 and LoRA.

### Strengths
* The proposed NTK-based analysis is novel and opens a new direction to understand prefix-tuning. Although the extremely-long prefix setup is not widely used yet, it seems to be an interesting direction for Transformer-based models.

### Weaknesses
* Is the "infinite-long" or "sufficiently long" prefix assumption practical (i.e., beneficial in terms of the performance)? In Table 1, m=200 is worse than m=100 case. The "Many-shot In-context Learning" paper (Agarwal et al., 2024) empirically showed that using lots of few-shot examples help, but the paper increases the length of input prompt rather than the number of training parameters; the result would not directly mapped one-to-one.
* Experimental results (Tables and Figures) should indicate the number of trainable parameters for each configuration. To be fair, it is important to compare the performance under similar resource usage constraint.

### Questions
* Theoretical questions
  * Does NTK-Attention still theoretically supported for the small m (~=100) case? How about the performance of NTK-Attention for small m, in terms of the number of training parameters?
  * I am not sure if the theoretical derivation also applies to L >> 1 case and causal attention setup, which is the most common case for Transformer-based generative models.
* How should we determine the size "r" of NTK-Attention?

### Soundness
2

### Presentation
3

### Contribution
2
