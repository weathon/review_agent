# DiaBlo: Diagonal Blocks Are Sufficient For Finetuning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 8

## Abstract
Fine-tuning is a critical step for adapting large language models (LLMs) to domain-specific downstream tasks. To mitigate the substantial computational and memory costs of full-model fine-tuning, Parameter-Efficient Fine-Tuning (PEFT) methods have been proposed to update only a small subset of model parameters. However, performance gaps between PEFT approaches and full-model fine-tuning still exist. In this work, we present *DiaBlo*, a simple yet effective PEFT approach that updates only the diagonal blocks of selected model weight matrices. Unlike Low-Rank Adaptation (LoRA) and its variants, DiaBlo eliminates the need for low-rank matrix products, thereby avoiding the reliance on auxiliary initialization schemes or customized optimization strategies to improve convergence. This design leads to stable and robust convergence while maintaining comparable memory efficiency and training speed to LoRA. Moreover, we provide theoretical guarantees showing that, under mild low-rank conditions, DiaBlo is more expressive than LoRA in the linear problem and converges to a stationary point of the general nonlinear full fine-tuning. Through extensive experiments across a range of tasks—including commonsense reasoning, arithmetic reasoning, code generation, and safety alignment—we show that fine-tuning only diagonal blocks is sufficient for strong and consistent performance. DiaBlo not only achieves competitive accuracy but also preserves high memory efficiency and fast fine-tuning speed. Codes are available at https://github.com/ziyangjoy/DiaBlo.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper proposes a new PEFT method called DiaBlo, which fine-tunes the weights as $\hat{W} = W + D$, where $W$ are the pretrained weights and $D$ are the residual fine-tuning weights as a block-diagonal matrix.
- $D$ is initialized with all zeros
- The method is evaluated on commonsense reasoning, arithmetic reasoning, code generation tasks and safety alignment tasks.

### Strengths
- The use of block diagonal matrix for the residual fine-tuning weights is a experimentally motivated.
- The method is easy to implement and computationally efficient.

### Weaknesses
1. The experimental comparisons on the quantization experiments are not correct. 
    - The papers experiments use a base model with a different quantization method than the baselines, and the baseline results are taken directly from Table 7 in [1].
    - [1] proposes a new quantization method called AiQ, fine-tunes it using LoRA, and then compares it with other quantization methods fine-tuned using LoRA.
    - However, the experiments in the paper initialize the quantized base model from MagR [2], fine-tunes using DiaBlo, and then compares it with results that use multiple different quantization methods fine-tuned using LoRA.
    - To elaborate, baseline results in Table 4 (taken from Table 7 in [1]) are of the nature "Quantization Method x/y/z + LoRA", ment to compare quantization methods. However, DiaBlo results are of the form "Quantization Method MagR + DiaBlo". Hence, there is no way to tell if the performance difference is due to the quantization method or the fine-tuning method.
    - Hence, the quantization method is a confounder that makes the comparisons invalid.
2. The comparison with baselines in all the tables are taken from multiple sources. Even if the high level settings like precision and hypeparameters are matched, the results are not directly comparable as subtle details in the implementation can lead to different results. For a proper comparison, the baseline methods should be run with the same setup as the proposed method.
    - For example, the baseline results in Table x are taken from 3 sources.
    - When the the baseline methods and DiaBlo are evaluated on the same benchmark but with bf16 precision in Table 6, DiaBlo shows minor improvements over LoRA. As the baseline results are taken from a different work, the improvements could be due to a different training setup.
3. Given that LoRA has had immense practical impact, the contribution of a new PEFT method does not have much impact without a significant advantage other than performance alone. In that sense, the contribution of the paper is limited.

---

## References
[1] "ApiQ: Finetuning of 2-Bit Quantized Large Language Model", Liao et al., EMNLP 2024

[2] "MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization", Zhang et al., NeurIPS 2024

### Questions
1. What is the GPU memory consumed by DiaBlo compared to LoRA/DoRA?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents DiaBlo, a param. efficient fine-tuning (PEFT) method for LLMs using the diagonal blocks of weight matrices. One of the key novelties in this work is that DiaBlo does not use extra low-rank matrices multiplied together (like LoRA’s A x B structure) to adapt model weights. Instead, it directly updates selected diagonal blocks within the model’s existing weight matrices, which eliminates the need for initialization or custom optimization strategies. The work covers tasks such as commonsense reasoning, arithmetic reasoning, and code generation, showing that DiaBlo can match or outperform LoRA with comparable memory/efficiency results. Furthermore, the method shows robustness under quantized architectures (4/2-bit).

### Strengths
1. By removing the complexity of low-rank structures, this work presents a clear alternative to LoRA-style PEFT. The results show that DiaBlo attains comparable performance without the added overhead of extra trainable matrices, simplifying both tuning and optimization.

2. The evaluation spans diverse supervised fine-tuning tasks -- including code generation, arithmetic reasoning, and commonsense reasoning -- covering a balanced range of short to moderate sequence lengths.

3. The results in Table 5 are particularly convincing, showing that random sparse update patterns (and SMT) underperform compared to DiaBlo. This supports the claim that the structured diagonal-block design is the key driver of its performance advantage.

### Weaknesses
1. Most evaluated benchmarks involve short output sequences, except for code generation. Testing DiaBlo on tasks with longer input–output contexts would better demonstrate its scalability and performance stability under extended sequence conditions (see q1).

2. The discussion of sparsity-based PEFT methods misses some recent relevant work, such as S2FT (NeurIPS 2025)[1] and SparseLoRA (ICML 2025)[2]. Including these would strengthen the discussion on sparsity in the introduction and would provide a better picture of the current limitations in state-of-the-art PEFT methods.

References:

[1] Xinyu Yang, Jixuan Leng, Geyang Guo, Jiawei Zhao, Ryumei Nakada, Linjun Zhang, Huaxiu Yao, Beidi Chen, "S2FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity", NeurIPS 2025

[2] Samir Khaki, Xiuyu Li, Junxian Guo, Ligeng Zhu, Chenfeng Xu, Konstantinos N. Plataniotis, Amir Yazdanbakhsh, Kurt Keutzer, Song Han, Zhijian Liu, "SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity", ICML 2025

### Questions
1. Most evaluated tasks focus on short output sequences, such as commonsense and reasoning benchmarks. Including long-form, multi-turn dialogue datasets like MT-Bench would better demonstrate DiaBlo’s scalability and effectiveness in extended conversational contexts.

2. The Appendix shows that trainable modules (QKVO/GUD) are fixed across methods, which is reasonable, but it remains unclear whether DiaBlo’s robustness comes from having a larger set of modules trainable on all methods (for example, for LLaMA3-8B, all the QKVOGUD modules have some trainable parameters). An ablation over different subsets of trainable components would clarify if the observed gains persist under more constrained settings. For example, see Figure 4.0 in S2FT on adding trainable parameters to only a subset of modules, such as QK, or GUD, etc.

3. Adding results on the GLUE benchmark would help assess general language understanding and show how well DiaBlo transfers to broader NLP tasks.

4. Reporting zero-shot baseline performance would contextualize fine-tuning improvements. Clarifying whether the LLaMA3-8B variant used is “instruct” or base would also make the evaluation setup more transparent.

### Soundness
3

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
3

### Summary
This work presents a Parameter-Efficient Fine-Tuning method called DiaBlo that updates only the diagonal blocks of selected model weight matrices. The work argues that this method enables axis-aligned, per-dimension scaling that LoRA cannot capture and verifies the method on a large variety of benchmarks with Llama 7B/8B/13B models.

### Strengths
- The proposed work eliminates the inherent optimization difficulties associated with low-rank decomposition by avoiding the use of matrix products.

- DiaBlo demonstrates higher stability in 4-bit and 2-bit arithmetic reasoning tasks.

### Weaknesses
- Compared to strong baselines like SMT with similar trainable parameter amount, the proposed method does not show significantly better performance. In other words, the paper argues the memory and computation efficiency of the proposed method, but the model doesn’t achieve significant improved performance compared to baselines when they share the same amount of trainable parameters.

- In table 1, it shows DiaBlo N =128 doesn’t get better performance compared to DiaBlo N =64 although doubled trainable parameters. This raises concerns of the scaling ability of the proposed Diablo.

### Questions
- The reviewer suggests to add Full Finetuning results in Table 1. 

- The paper mentions when N is not a common factor of m1,m2, it needs to expand and pad the weight into a proper size and then select the corresponding diagonal blocks. This can be common case. But the paper doesn’t touch this point clearly in later experiments.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a new parameter-efficient finetuning method which targets trains a block-diagonal sparse matrix on top of linear layers in model weights. This approach seems to outperform various other PEFTs with around the same number of parameters on a variety of tasks, as well as when compared to other baselines which involve training sparse weight updates.

### Strengths
- The idea is quite elegant, relatively simple to implement and efficient to train -- there isn't much adaptation required to existing finetuning libraries to get this working.
- The results are broad (covering standard PEFT benchmarks) and thus convincing.
- Ablations cover the first questions I had regarding whether block-diagonal is better than other ways of selecting entries to tune in the weight matrix; it does seem like it is broadly a better strategy than other ideas.

### Weaknesses
- Since we use a standard suite of benchmarks to evaluate PEFTs, it's possible that our literature is engaging in test-set overfitting (compare how the ImageNet challenge or LMSYS arena were overfit by organisations submitting many models repeatedly). It would thus be nice to show how the technique performs under varying learning rates and block sizes (e.g. as done for LoRA in [Schulman et al. (2025)](https://thinkingmachines.ai/blog/lora/)). It is nice though that there are not as many hyperparameters as other PEFTs!

### Questions
- In your experience how difficult was it to hyperparameter tune this method?
- What model components seem most important for good performance when finetuning? Is there a layer-wise effect?

### Soundness
4

### Presentation
3

### Contribution
4
