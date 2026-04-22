# PrefixMemory-Tuning: Modernizing Prefix-Tuning by Decoupling the Prefix from Attention

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Parameter-Efficient Fine-Tuning (PEFT) methods have become crucial for rapidly adapting large language models (LLMs) to downstream tasks. Prefix-Tuning, an early and effective PEFT technique, demonstrated the ability to achieve performance comparable to full fine-tuning with significantly reduced computational and memory overhead. However, despite its earlier success, its effectiveness in training modern state-of-the-art LLMs has been very limited. In this work, we demonstrate empirically that Prefix-Tuning underperforms on LLMs because of an inherent tradeoff between the contribution of input prompt and parameterized prefix within the attention head. This motivates us to introduce PrefixMemory-Tuning, an architecture that generalizes the principles of Prefix-Tuning while addressing its shortcomings by shifting the prefix module out of the attention head itself and improving its expressiveness. Our experiments show that, across diverse benchmarks, PrefixMemory-Tuning consistently outperforms existing Prefix-Tuning methods. Notably, it achieves competitive performance with modern PEFTs on several general benchmarks, highlighting a potential extension of Prefix-Tuning approaches to become state-of-the-art. Our findings suggest that by overcoming its inherent limitations, Prefix-Tuning can remain a competitive and relevant research direction in the landscape of parameter-efficient LLM adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the limitations of Prefix-Tuning (PT), a parameter-efficient fine-tuning (PEFT) method for large language models (LLMs), which struggles with modern LLMs due to a trade-off between input and prefix contributions within the attention head. The authors introduce PrefixMemory-Tuning (PMT), a novel approach that decouples the prefix from the attention head, enhancing expressiveness and performance. Empirical results show PMT outperforms PT and is competitive with state-of-the-art methods like LoRA, achieving an average improvement of 8.1% over LoRA and 29.4% over PT across six benchmarks.

### Strengths
1. PMT addresses a fundamental limitation of PT by relocating the prefix outside the attention head, improving its scalability and expressiveness.

2. Extensive experiments across diverse benchmarks (e.g., preference alignment, math reasoning) demonstrate PMT's competitive performance.

3. The paper provides a clear explanation of PT's underperformance and a unified framework for future context-based PEFT methods.

### Weaknesses
1. The study uses simple feature maps (elu, gelu) as a proof of concept, leaving more expressive options unexplored due to implementation complexity.

2. The paper does not deeply address computational cost or scalability for very large LLMs, which is critical for practical deployment.

### Questions
1. How would PMT perform with more sophisticated feature maps (e.g., trainable MLPs) compared to the current elu/gelu implementations?

2. Some related work can be theoretically discussed. "E^ 2vpt: An effective and efficient approach for visual prompt tuning" ICCV, which adds some prompts in the attention head.

3. What are the computational and memory overheads of PMT compared to LoRA and other PEFT methods at scale?

4. How does PMT handle extremely long input sequences in real-world applications, given the trade-off issues identified in PT?

### Soundness
3

### Presentation
4

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
This paper identifies the performance limitations of Prefix-Tuning in modern large language models (LLMs) and proposes a method called PrefixMemory-Tuning to address these issues.

### Strengths
>s1: Shifting the prefix module out of the attention head itself is a reasonable start point.

>s2: Experiments show that, in the **few-shot setting**, PrefixMemory-Tuning is competitive with state-of-the-art approaches (such as LoRA).

>s3: The presentation is clear, and the figures are of high quality.

### Weaknesses
> w1: **The discussion of related work contains inaccuracies**. For instance, lines 87-90 state that "LoRA+ refines this concept further, projecting the model’s weights onto low-dimensional subspaces to achieve efficiency comparable to full fine-tuning at significantly reduced computational cost." However, the actual contribution of LoRA+ lies in its theoretical analysis demonstrating that using identical learning rates for matrices A and B in standard LoRA prevents efficient feature learning in large-width networks. The method addresses this limitation by employing differentially scaled learning rates for the adapter matrices with an optimally determined ratio, rather than proposing weight projection onto low-dimensional subspaces.
>
>  LoRA+: Efficient Low Rank Adaptation of Large Models. ICML 2024.

>w2: There is **no support** for the claim in Line 144-146 "Research shows that prefix-tuning excels in low-data or few-shot settings".

> w3: **Insufficient literature review**: To name a few: (1) For context-based PEFT methods: [1] [2]; (2) Lines 307-315, a good work to show the memory perspective is [3].
>
>[1] DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning. ICLR 2024.
>
>[2] ADePT: Adaptive Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning. ICLR 2025.
>
> [3] Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP 2021.

>w4: The experiments were primarily conducted in a **few-shot setting**. Recent studies (such as [4]) have also found zero-shot approaches to be competitive or even superior in certain scenarios. What are your thoughts on this?
>
>[4] Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot. EMNLP Findings 2025.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **PMT (PrefixMemory-Tuning)**, a novel parameter-efficient fine-tuning (PEFT) method. The authors build on the evolution of Prefix-Tuning (PT), which was largely abandoned due to its poor scaling properties in large models, as well as more recent methods like LoRA and GaLore. While PT had been likened to prompt-based learning by introducing a set of trainable continuous vectors (prefixes) for each input, early analyses identified its underperformance as stemming from an inability to reshape attention patterns within attention heads. However, the authors argue that the true cause of PT’s degradation lies in the **trade-off between the prefix and the input representations**.

Leveraging this insight, the authors improve upon conventional PT, proposing PMT as a more effective and efficient approach. Their experiments show that PMT surpasses both **LoRA** and even **full fine-tuning (FFT)** in terms of fine-tuning performance, providing a more scalable and efficient method for adapting large models.

### Strengths
- The paper provides a strong theoretical analysis of how fine-tuning (FT) impacts the activation values of attention heads, reducing the relationship to a linear function of the original attention values. This simplification reveals the fundamental cause of **Prefix-Tuning (PT)**'s degradation in large-scale models. The authors effectively demonstrate how the trade-off between prefix and input representations negatively affects model performance.

- The paper offers a clear and insightful analysis of the evolution from **Prefix-Tuning (PT)** to **PMT**, highlighting the **coupling problem** between the prefix and input representations in traditional PT. The authors address this issue by introducing effective approximation techniques that successfully decouple the components. Their experimental results consistently show that **PMT** outperforms **LoRA** and even surpasses **full fine-tuning (FFT)** on most tasks, demonstrating the effectiveness of their approach.

### Weaknesses
- The experimental models used in the paper, such as **LLaMA2-7B-Chat** and **Qwen2.5-3B-Instruct**, are relatively small in scale. Given that these models are not at the forefront of current large-scale models, the evaluation does not demonstrate PMT’s performance on truly large-scale architectures, where the method’s scalability and effectiveness may vary. Including experiments on larger models would strengthen the claims regarding PMT's applicability to cutting-edge architectures.

- The comparison to other PEFT methods lacks more recent and advanced variants such as **QLoRA** and **LoRA+**, which are gaining traction in the field. This limits the strength of the empirical comparison, as it does not reflect the latest advancements in PEFT techniques.

### Questions
- The PMT architecture can be viewed as adding a linear transformation (similar to a mapping based on \( q_i \)) on top of the original model output. This raises an interesting question: could the model architecture itself be modified to inherently incorporate such a transformation, thereby achieving PMT-like behavior without the need for additional components? If this approach were viable, it might improve the model's generalization and efficiency.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this work, prefix-finetuning in transformer is re-visited and a new prefix-memory module is added to the scaled dot product module in transformer. By revisiting the existing PEFT framework, it's pointed out that the main gap between prefix-tuning and other efficient finetuning approach like LoRa lies in the attention in-balance of learnable prefix and the input context. When the context is long, the contribution of learnable prefix diminished, and vice versa. To resolve this, the prefix-memory module is added directly to the scaled dot prod structure and used to dynamically adjust the learned attention to fix the problem above. Experiments are conducted to show the competitive results on multiple datasets when compared with SFT and other efficient tuning approach like LoRA.

### Strengths
- The analysis on why prefix tuning failed under extreme cases is convincing and reasonable. E.g in section 4.2, the qualitative analysis makes sense to illustrate the attention balance is broken when either input context or prefix is relatively long.

### Weaknesses
- The evaluation is a bit weak. In table 1 the comparisons are made between multiple finetuning approaches, like PMT(proposed), SFT, LoRA, and prefix tuning. However the results are confusing as the SFT results are weaker than other PEFT approaches. This indicates that the dataset used here might not be challenging enough.
- The module is called "prefix memory", but in the paper there is no analysis about what does this memory module learned. It would be better to qualitatively or quantitatively analyze about this to provide insights.
- Did not mentioned or compare with some related work, like attention sink and Aprompt. Attention sink had some similar observation that the model's attention is usually overindexed to some specific tokens.

### Questions
- The evaluation is a bit weak. In table 1 the comparisons are made between multiple finetuning approaches, like PMT(proposed), SFT, LoRA, and prefix tuning. However the results are confusing as the SFT results are weaker than other PEFT approaches. This indicates that the dataset used here might not be challenging enough. $\rightarrow$ Is it possible to test on some more challenging tasks where the full SFT is needed and the usefulness of PMT can be better highlighted? 
- The module is called "prefix memory", but in the paper there is no analysis about what does this memory module learned. It would be better to qualitatively or quantitatively analyze about this to provide insights. $\rightarrow$ Is it possible to get more intuitive understanding about what did this newly added memory module learn.
- Did not mentioned or compare with some related work, like attention sink[1] and Aprompt [2]. For 1, seems the attention analysis in this work is similar, and for 2 it's a popular prompt based PEFT approach. 

[1] Xiao, Guangxuan, et al. "Efficient streaming language models with attention sinks." arXiv preprint arXiv:2309.17453 (2023).
[2] Wang, Qifan, et al. "Aprompt: Attention prompt tuning for efficient adaptation of pre-trained language models." Proceedings of the 2023 conference on empirical methods in natural language processing. 2023.

### Soundness
2

### Presentation
2

### Contribution
3
