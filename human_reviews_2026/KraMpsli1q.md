# NeUQI: Near-Optimal Uniform Quantization Parameter Initialization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Large language models (LLMs) achieve impressive performance across domains but face significant challenges when deployed on consumer-grade GPUs or personal devices such as laptops, due to high memory consumption and inference costs. Post-training quantization (PTQ) of LLMs offers a promising solution that reduces their memory footprint and decoding latency. In practice, PTQ with uniform quantization representation is favored due to its efficiency and ease of deployment, as uniform quantization is widely supported by mainstream hardware and software libraries. Recent studies on $\geq 2$-bit uniform quantization have led to noticeable improvements in post-quantization model performance; however, they mainly focus on quantization methodologies, while the initialization of quantization parameters remains underexplored and still relies on the conventional Min-Max strategic rationale. In this work, we identify the limitations of the Min-Max strategic rationale, move beyond its constraints, and propose **NeUQI**, a method that efficiently determines near-optimal initialization for uniform quantization. Our NeUQI proposes a method that determines the near-optimal zero-point for a given scale, thereby reformulating the initialization optimization into a scale-only problem that can be solved efficiently. Benefiting from the improved quantization parameters, our NeUQI consistently outperforms existing methods in the experiments with the LLaMA and Qwen families on various settings and tasks. Furthermore, when combined with a lightweight distillation strategy, NeUQI even achieves superior performance to PV-tuning, a considerably more resource-intensive method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem of parameter initialization in uniform post-training quantization (PTQ) of large language models (LLMs). While most prior works rely on the conventional Min-Max strategy, which becomes ineffective at low bit-widths (2–3 bits), the authors identify two key limitations of this rationale: (1) scale and zero-point are tied to extreme values, unnecessarily enlarging the search space, and (2) the integer constraint on zero-point reduces flexibility. To overcome these, they propose NeUQI, a near-optimal initialization method that directly optimizes scale and zero-point using a loss-aware formulation with efficient algorithms. They further combine NeUQI with a lightweight distillation strategy. Experiments on LLaMA and Qwen families show consistent improvements, with NeUQI sometimes even surpassing non-quantized baselines at comparable memory usage.

However, some limitations remain. First, while floating-point zero-points could in principle offer better representational flexibility than integer zero-points, many hardware platforms do not support them, limiting practical applicability. Second, the range of baseline initialization methods evaluated is still limited.

### Strengths
- Strong theoretical grounding with efficient algorithms (increment-prefix-sum, coarse-to-fine scale search).
- Extensive experiments across multiple LLM families. The experiments discuss the combination of this initialization method with various quantization algorithms, with consistent improvements.

### Weaknesses
In model quantization, it is natural to expect that using a floating-point zero-point would outperform an integer zero-point, since the representational capability is inherently greater. However, many hardware platforms do not support zero-points in floating-point formats, so such zero-point optimizations do not always hold in practice.

### Questions
The experiments of initialization methods considered is still quite limited. The author’s mentioned approaches, such as clipping-based, quantile-based, and Mean-Std-based methods, were not compared, nor were classical techniques like KL-divergence minimization. These do not conflict with the scale-learnable methods discussed by the author, such as OmniQuant and LearnQuant; combining these classical initialization methods with learnable quantization parameters , would perform worse than the author’s approach?
 

It is generally believed that when using fine-tuned or LoRA fine-tuned models, the advantage of initialization is largely diminished, since the weights adapt to the scale and zero-point. Does this perspective still apply to your proposed quantization initialization method?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the 2-bit Post-Training Quantization problem for LLMs. The proposed method adopts a floating-point zero point for more precise optimization on the approximated quantization loss of layer-wise outputs. An efficient searching algorithm is proposed to find the optimal zero point and scaling.

### Strengths
- The proposed method shows robustness across different models and tasks in the 2-bit regime, surpassing SOTA performance.

### Weaknesses
- The definition of Min-Max initialization should be mentioned before/at Section 3, e.g., as a reference to Appendix A. I find it difficult to understand Section 3 at first without acknowledging the definition of what is referred to as "Min-Max" initialization.
- In Line 123, the grid search size $T$ is not properly defined/introduced. In particular, the definition in Appendix A does not mention about $T$.
- There is a typo in Eq. (3): $(W_{i,:}) - W_{i,:})^\top$ should be $(Q(W_{i,:}) - W_{i,:})^\top$
- The statement in Line 218 is not clearly written. I assume that stated example is referring to the scenario where $\exists z, z'$ such that ${\rm clip}(\lfloor x_i + z \rceil, 0, 2^k - 1) = a$ and ${\rm clip}(\lfloor x_i + z' \rceil, 0, 2^k - 1) = a+1$. Then, the increment in loss function should be written as $h_i((x_i + z' - (a+1))^2 - (x_i + z - a)^2 )$.

Overall, I think the optimization process in Section 4.3.1 and 4.3.2 should be described more clearly. Judging from the current submission, it is not clear how the proposed algorithm is implemented.

### Questions
- When the zero point optimization is simplified by the smoothed objective in Eq. (6), i.e., reducing the candidate solutions of $z$ to $\{1/2 - x_i, 2^k - 1/2 -x_i \}$, the rounded values are in $\{1/2, 2^k - 1/2 \}$. Is it equivalent to falling back to the **integer** zero point model $Q_{s, z}(x) = s\cdot ({\rm clip}(\lfloor x / s \rceil+ z , 0, 2^k - 1) - z)$, as opposed to the floating-point zero point model that is proposed in Section 4.1? I suggest the authors to clarify the advantage of adopting a floating-point $z$ when solving for the optimization in Section 4.3.1.

- In Line 10 of Algorithm 1, which coefficients are initialized and how are they initialized?

- How does Algorithm 1 ensures to find the optimal solution to Eq. (5), which involves the quantization error of a group of weight values $w_i$ (summation over $i = 1, ..., n$), using only a per-sample loss changes in Line 5 of Algorithm 1?

- Can the authors provide the actual time cost of performing NeUQI, and compare it to every baseline methods?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents NeUQI, a parameter initialization algorithm for uniform quantization in post-training quantization (PTQ) of large language models. The authors identify two inherent limitations in the widely used Min–Max initialization strategy — its reliance on extreme values and integer zero-point constraints — and propose a new method that relaxes these assumptions. NeUQI directly optimizes scale and zero-point via a loss-aware formulation and an efficient two-stage approximation algorithm, showing consistent improvements across LLaMA and Qwen model families. The method also integrates effectively with lightweight distillation, surpassing heavier fine-tuning baselines such as PV-tuning.

### Strengths
The paper highlights the role of parameter initialization — and convincingly argues that initialization quality strongly affects quantization performance. By systematically reformulating the initialization problem and relaxing integer constraints, the proposed NeUQI provides an efficient and theoretically grounded approach. The experimental results are competitive. The inclusion of lightweight distillation further enhances the practical significance of the method.

### Weaknesses
However, the empirical evidence for the independent importance of initialization remains somewhat ambiguous. Many PTQ methods that incorporate limited fine-tuning (e.g., OmniQuant) already perform a form of implicit re-initialization by adjusting parameters during optimization. It would therefore be more convincing if the paper explicitly compared NeUQI with such fine-tuning-based methods across a wider range of models. Although Table 1 includes some results for OmniQuant and PV-tuning, the current coverage is limited. Additional experiments on more architectures, and under both weight-only and weight-activation quantization, would strengthen the claim that NeUQI surpasses these adaptive methods in a consistent and generalizable way.

### Questions
see above.

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
4

### Summary
This paper addresses the deployment challenges of large language models (LLMs) on resource-limited devices caused by their high memory and computation demands. It focuses on post-training quantization (PTQ) with uniform quantization, which is hardware-friendly and widely supported. The authors observe that prior work has improved quantization techniques but largely neglected how quantization parameters are initialized, which typically relies on the suboptimal Min–Max strategy. To overcome this, they propose NeUQI, an efficient method for finding near-optimal initialization of uniform quantization parameters. Experiments on LLaMA and Qwen models show that NeUQI consistently outperforms existing PTQ approaches.

### Strengths
- The paper identifies and tackles an underexplored but critical aspect of post-training quantization: the initialization of quantization parameters (i.e., scaling-factor and zero-point). By moving beyond the traditional Min-Max initialization, the proposed NeUQI method provides a principled and efficient way to find near-optimal initialization, leading to consistent performance gains across multiple LLMs.
- Extensive experiments on LLaMA and Qwen model families demonstrate that NeUQI achieves SOTA results in various settings.

### Weaknesses
- Some methodological details are missing. In Eq. (3), the quantized weights appear not to be explicitly optimized, the loss function only involves the scaling factor and zero-point. In contrast, GPTQ explicitly updates weights through a reconstruction step. Please clarify how the quantized weights are incorporated into the optimization process and whether they are updated similarly. 
- The paper lacks an ablation study evaluating the effect of using a floating-point zero-point. Since this design choice deviates from conventional integer quantization schemes, an analysis of its contribution to performance would strengthen the paper.
- The writing could be improved for clarity and accessibility. In the Abstract and Introduction, the description remains overly high-level. For instance, mentioning the replacement of the Min–Max scheme with a “better initialization” without concrete details. As a result, readers cannot grasp the essence of the proposed method until well into Section 4. It is recommended to summarize the key technical innovations and intuitions earlier to enhance readability and highlight the paper’s novelty.

### Questions
- Is the lightweight distillation strategy incorporated into NeUQI in Table 3?
- Can the proposed method be extended to MoE-based models?

### Soundness
3

### Presentation
2

### Contribution
3
