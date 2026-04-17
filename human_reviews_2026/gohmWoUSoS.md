# MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Low-Rank Adaptation (LoRA) is a widely adopted technique for parameter-efficient fine-tuning, but its slow convergence has spurred the development of numerous variants. Nevertheless, current approaches struggle to achieve simultaneous improvements in performance, memory footprint, and computational efficiency. To address this challenge, we revisit the causes of LoRA’s slow convergence and, based on these insights, propose \textbf{M}atr\textbf{i}x \textbf{S}hard \textbf{S}haring (MiSS) that shards the original weight matrix and updates by sharing a single trainable matrix $\boldsymbol{D}$ initialized to zero. To simultaneously ensure computational efficiency, low memory footprint, and scalable serving, we introduce MiSS$^e$. Through theoretical analyses and empirical results, our method reduces optimization complexity while maintaining strong performance, striking a favorable balance between performance, memory, and efficiency. Furthermore, we provide a comprehensive analysis of different PEFT methods with respect to memory usage, initialization time, and computational efficiency. By mapping the Pareto frontier, we show that MiSS achieves a favorable balance across these dimensions, integrating the strengths of prior approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MiSS (Matrix Shard Sharing), a novel parameter-efficient fine-tuning (PEFT) method designed to address the trade-off between performance, memory, and computational efficiency in LoRA-based approaches. The core idea is to simplify the weight update mechanism by training a single shared matrix D, which is expanded to form the weight update matrix, thereby reducing optimization complexity. The authors support their claims with a theoretical analysis of gradient norms, controlled experiments on a synthetic task, and extensive benchmarks on NLU and NLG tasks, demonstrating that MiSS achieves a favorable balance across key metrics.

### Strengths
1. The proposal to replace LoRA's two-matrix (B and A) update with a single trainable matrix D is a novel simplification. The motivation, rooted in reducing optimization complexity and its connection to the initial gradient norm, provides a clear and compelling rationale for the proposed architecture.

2. The paper presents a strong empirical evaluation, combining a controlled "No Free Launch" experiment to isolate variables with large-scale benchmarks across multiple models (e.g., Llama2, Mistral, Qwen3) and tasks. The inclusion of Pareto frontier analysis further strengthens the claims about achieving a better trade-off between performance and efficiency.

### Weaknesses
1. The paper introduces MiSS in Section 4.1, which partitions the output dimension d, and then presents MiSSe in Section 4.2 as a "mathematically equivalent" version that partitions the input dimension k. These two operations appear fundamentally different and are unlikely to be mathematically equivalent. This discrepancy undermines the clarity of the proposed method.

2. The paper motivates MiSS by simplifying optimization to a single matrix. However, the specific choice of the expand(D) structure, repeating rows to form shards, feels somewhat arbitrary. The paper lacks a deep analysis of why this particular low-rank construction is superior to other potential single-matrix structures.

3. The paper includes VeRA in the initial controlled experiment but does not provide a direct comparison in the main LLM benchmark tables. VeRA also employs shared low-rank matrices for efficiency, making it a critical baseline. A more detailed comparison is needed to properly position MiSS.

4. In Section 5.2, the authors state they adjust ranks to give MiSS fewer parameters than LoRA. However, in Table 3 (Llama2-7B), the parameter counts are very close (MiSS: 87.0M vs. LoRA: 89.9M). The observed performance gains might be influenced by this slight difference in parameter budget or distribution rather than solely the architectural advantage.

5. Figure 4 shows faster initial convergence for MiSS against LoRA, but this analysis is limited. The main experiments on LLMs lack convergence plots comparing MiSS with other key baselines like PiSSA and DoRA. This information is crucial for substantiating the claims about training efficiency.

6. The pseudocode in Figure 3 is labeled MiSS but appears to implement a simplified version of MiSSe. The line y = result + x @ self.D.expand(...) is dimensionally inconsistent and does not clearly match the formulas provided in the text, causing confusion for implementation.

7. Figure 5 provides a valuable visualization of the performance-efficiency trade-off, but the axes are not labeled. It is unclear which specific metrics (e.g., accuracy, memory usage, time) are being plotted, which reduces the figure's interpretability.

8. A footnote suggests that MiSS is "largely orthogonal" to methods like LoRA+, which use different learning rates for matrices A and B.

9. The paper states that the D matrix is initialized to zero, which is a standard practice to preserve the pre-trained model's initial state. However, it does not explore or ablate other initialization strategies, such as those inspired by SVD in PiSSA, which are designed to accelerate convergence.

10. In Section 5, the document refers to NLG evaluations, but the footnote specifies that the Math dataset was evaluated with a 5-shot prompt. This suggests an in-context learning evaluation rather than a fine-tuning evaluation for that specific task, which could be misleading.

### Questions
1. Can the authors clarify whether MiSSe is an exact mathematical equivalent of the MiSS formulation in Section 4.1? If not, it would be more accurate to present MiSSe as a distinct, more efficient variant inspired by the same shard-sharing principle, rather than a direct equivalent.

2. Beyond simplifying optimization, is there a deeper theoretical or empirical justification for choosing the row-repetition structure in expand(D)? How does this structure relate to the properties of weight update matrices observed in full fine-tuning?

3. Given that VeRA utilizes a similar parameter-sharing strategy for efficiency, could the authors provide a more direct comparison against VeRA on the main LLM benchmarks, discussing both conceptual differences and empirical trade-offs?

4. To ensure a fair comparison, could the authors conduct an experiment where the trainable parameter counts of MiSS and LoRA are matched as precisely as possible?

5. Could the authors provide convergence plots (loss vs. training time/tokens) for the main LLM experiments (e.g., on Llama2-7B) that include PiSSA and DoRA?

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
4

### Summary
The paper proposes Matrix Shard Sharing (MiSS) and its variant MiSSe to address the slow convergence and trade-offs among performance, memory, and computational efficiency in Low-Rank Adaptation (LoRA). MiSS shards the weight matrix and updates by sharing a single trainable matrix D, while MiSSe offers enhanced efficiency and scalability. The analysis shows that MiSS achieves a favorable balance across performance, memory, and efficiency.

### Strengths
The proposed method is very pragmatic and practical.

The proposed method was examined in comparison with different LoRA variations in several benchmarks.

### Weaknesses
Notation should be revised and redundancy in some terms should be fixed.

The paper states that: Through theoretical analyses and empirical results, our method reduces optimization complexity while maintaining strong performance, striking a favorable balance between performance, memory, and efficiency.

That is, one of the main claims is the theoretical analysis of the proposed methods. However, this is not well explored in the paper.

### Questions
In Table 3, MiSS has almost similar number of parameters compared to the other LoRA variants. 

Why does MiSS have larger number of parameters for Llama2-13B? Is it due to r or k (as expressed in Table 5) or for another reason?

Could you please provide an ablation of using different initialization strategies to initialize D?

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
3

### Summary
In this paper, the authors aim to address the slow convergence issues in Low-Rank Adaptation (LoRA), a well-known parameter-efficient method for fine-tuning large language models (LLMs). In respone, they propose Matrix Shard Sharing (MiSS), a method that shards the original weight matrix and updates by sharing a single trainable matrix $D$ initialized to zero. This design reduces optimization complexity while maintaining performance. To further enhance computational efficiency, low memory usage, and scalable serving, they introduce MiSS. Theoretical analyses demonstrate MiSS's advantages in gradient flow and convergence, while empirical evaluations on benchmarks show strong results. The work also provides a comprehensive comparison of PEFT methods across memory, initialization time, and efficiency dimensions, mapping a Pareto frontier to highlight MiSS's balanced trade-offs.

### Strengths
1. Quality: MiSS effectively addresses LoRA's limitations by improving convergence without sacrificing efficiency, as supported by theoretical insights and Pareto analysis, making it a practical advancement in PEFT.

2. Soundness: The paper includes detailed comparisons with variants like PiSSA and LoRA-GA, covering multiple dimensions (performance, memory, compute), which strengthens the claims.

### Weaknesses
1. Reliance on zero-initialized $D$ may limit adaptability in certain scenarios, potentially requiring further tuning.

2. Evaluations are primarily on language tasks; broader domains (e.g., vision or multimodal) are not explored.

3. The Pareto frontier mapping is insightful but could be more granular, e.g., with statistical significance tests.

### Questions
1. Could MiSS integrate with Mixture-of-Experts architectures for further efficiency gains?

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
5

### Summary
The paper introduces MiSS, a parameter-efficient fine-tuning (PEFT) method that builds upon LoRA to achieve a more favorable balance between adaptability and efficiency compared to prior approaches. The proposed method is comprehensively evaluated across a diverse set of tasks, including natural language understanding and natural language generation, and is empirically compared against several LoRA variants such as DoRA and PiSSA.

### Strengths
The proposed method is remarkably simple and easy to implement, yet it demonstrates strong practical effectiveness. In several experimental settings, the method achieves notable improvements over LoRA. For instance, on the Mistral-7B model, MiSS outperforms LoRA by approximately 15%, highlighting its potential as a competitive and efficient alternative for parameter-efficient fine-tuning.

### Weaknesses
Firstly, although the paper states that MiSS is motivated by theoretical analysis, the practical method itself is presented without clear theoretical justification or development. The coherence and clarity of the paper would be greatly improved by adding a dedicated subsection in Section 4 that discusses the theoretical motivations behind the architectural design, which appears to be a novel choice aimed at ensuring the low-rank condition.

Secondly, since the paper seeks to propose a practical parameter-efficient fine-tuning (PEFT) variant of LoRA, it would be valuable to include comparisons with a broader set of LoRA variants, such as AdaLoRA and VeRA. Although the manuscript mentions in lines 172–177 and the table in lines 58–67 that PiSSA, VeRA, DoRA, MoRA, PROLORA, and MoS are considered, only DoRA and PiSSA are actually included in the main experiments. In particular, comparison with VeRA, which is known for its strong parameter efficiency, would substantially strengthen the empirical section.

**Other minor issues:**

+ It is recommended to include an average performance metric to better assess the overall results relative to the baselines. While MiSS surpasses some baselines in certain settings, it underperforms on others.

+ The title of Section 3 should likely be “No Free Lunch”—please verify and correct if necessary.

### Questions
My concerns are presented in the "Weaknesses" section and I have no further question.

### Soundness
3

### Presentation
4

### Contribution
2
