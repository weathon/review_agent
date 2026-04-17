# Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Memory and computation remain core bottlenecks in long-horizon LLM inference due to the quadratic cost of self-attention and the ever-growing key-value (KV) cache. Existing strategies for memory-bounded inference, such as quantization, offloading, or heuristic KV eviction, either incur high orchestration costs or rely on unreliable attention-based proxies of importance. We propose TRIM-KV, a novel approach that learns each token’s intrinsic importance at creation time via a lightweight retention gate. Each gate predicts a scalar retention score that decays over time, reflecting the long-term utility of the token for a specific layer and head. Tokens with low scores are evicted when the memory budget is exceeded, ensuring that the cache always contains the most critical tokens. TRIM-KV is trained efficiently through distillation from a frozen LLM combined with a capacity loss, requiring only gate fine-tuning and adding negligible inference overhead. Across mathematical reasoning (GSM8K, MATH-500, AIME24), procedural generation (LongProc), conversational long-memory benchmarks (LongMemEval), and long-context understanding (LongBenchV2 and SCBench), TRIM-KV consistently outperforms strong eviction and learnable retrieval baselines, especially in low-memory regimes. Remarkably, it even surpasses full-cache models in some settings, showing that selective retention can serve as a form of regularization, suppressing noise from uninformative tokens. Qualitative analyses further reveal that learned retention scores align with human intuition, naturally recovering heuristics such as sink tokens, sliding windows, and gist compression without explicit design. Beyond efficiency, retention scores provide insights into layer- and head-specific roles, suggesting a new path toward LLM interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes TRIM-KV, a trainable sparse attention method for KV cache eviction. The core design of TRIM-KV is to learn a forgetting gate $\beta_i^{t-i}$ for the attention mechanism. Such gate is learned through minimizing the KL loss and language modeling loss while reducing the cap loss $L_{cap}$. A small MLP is applied to predict the gate value $\beta_i$ at each position, and only the MLP is trained while other parameters are frozen. KV cache eviction is conducted by evicting the KV pair with the minimal $\beta_i^{t-i}$. This paper trains such architecture on math reasoning datasets, and evaluate the trained model on various reasoning datasets that requires long cot generation. Empirical results show that TRIM-KV is able to outperform all the KV cache eviction baselines, and appears lossless acceleration in long-cot reasoning. The authors conduct detailed ablation studies to show the effectiveness of TRIM-KV's design and the generalizability across various training datasets.

### Strengths
1. The design of TRIM-KV is reasonable and effective. Learning a forgetting gate $\beta_i$ for KV cache eviction is a novel and interesting idea.

2. Experimental results show significant speedups, and TRIM-KV outperforms classical baselines such as SnapKV and R-KV. The ablation studies are detailed and comprehensive.

3. The paper is clearly written, making it easy for readers to understand both the training and inference procedures of TRIM-KV.

### Weaknesses
1. Since TRIM-KV evicts the KV cache according to query-agnostic features, i.e., the importance score of each KV pair is determined by itself rather than by specific query tokens, certain types of baselines should be included in the discussion, and, if possible, compared experimentally. InfiniPot [1] provides a training-free and query-agnostic method that aligns with chunked prefill KV cache eviction. Locret [2] also employs a training-based method to learn the importance score of each KV pair. These methods should be discussed and compared.

2. Although this paper focuses on evicting unimportant KV cache units during reasoning, its ability to process general long-context inputs should also be evaluated. Typical long-context benchmarks such as RULER [3] and LongBench [4] should be included. Eviction-based KV cache optimizations are not expected to perform well on difficult retrieval tasks in RULER (as shown in the Locret [2] paper), but this limitation should still be discussed in the paper to highlight potential weaknesses of the method.

3. The description of the training setup is unclear. The volume of training tokens, batch sizes, sequence lengths, and other hyperparameters should be included in the paper (these could be placed in the appendix). Note that another type of attention optimization: trainable sparse attention (e.g., NSA [5]), requires significantly more training data to achieve higher performance. Listing the training cost of TRIM-KV would help clarify the differences between these methods.

4. Is multi-turn conversation supported by TRIM-KV?

5. Writing should be improved. Appendix C is excessively long yet provides limited useful information. Figure 4 is also not providing much information (a smaller figure is preferred). The conclusion should be written in the authors’ own words rather than by including the model’s generated answers. The methods mentioned in Weakness 1 should also be incorporated into the related work section.

Overall, I appreciate this work and plan to increase my overall rating if the weaknesses and questions above are addressed.

---

[1] Infinipot: Infinite context processing on memory-constrained llms.

[2] Locret: Enhancing eviction in long-context llm inference with trained retaining heads.

[3] RULER: What's the Real Context Size of Your Long-Context Language Models?

[4] Longbench: A bilingual, multitask benchmark for long context understanding

[5] Native sparse attention: Hardware-aligned and natively trainable sparse attention

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes TRIM-KV, a learnable token retention mechanism for memory-bounded large language model (LLM) inference. Instead of relying on attention-based heuristics for KV-cache eviction, the authors introduce a lightweight retention gate that predicts each token’s intrinsic importance at creation time. The predicted retention score decays exponentially over time, inspired by Ebbinghaus’s forgetting curve, and governs which tokens are evicted when memory exceeds a fixed budget. The gates are trained via distillation from a frozen teacher LLM and a capacity regularization loss, ensuring negligible inference overhead. Experiments across multiple benchmarks — including GSM8K, MATH-500, AIME24, LongProc, and LongMemEval — demonstrate that TRIM-KV outperforms strong heuristic and learnable baselines under the same or even tighter memory budgets. Qualitative analyses further show that the learned retention scores align with intuitive heuristics such as sliding windows, sink tokens, and gist tokens, suggesting interpretability benefits.

### Strengths
1. Novel and well-motivated idea:
The paper identifies a fundamental limitation of attention-based eviction — that “recent attention ≠ importance” — and replaces it with a predictive, intrinsic importance estimation per token. This shift from reactive to proactive cache management is conceptually elegant and well-justified.
2. Brain-inspired design:
Modeling token importance decay via an exponential forgetting curve connects the method to cognitive science (Ebbinghaus), offering both theoretical grounding and interpretability.
3. Method simplicity and practicality:
TRIM-KV requires only a small MLP per layer/head, trained with frozen base weights. The added computation and memory are negligible, and inference remains fully compatible with Flash/FlexAttention kernels. The implementation is clear and deployable.
4. Comprehensive empirical validation:
Results are reported on five diverse benchmarks covering reasoning, long-context generation, and long-memory conversation. TRIM-KV consistently outperforms SeerAttn-R (learnable retrieval) and heuristic eviction methods (R-KV, SnapKV, H2O), sometimes even surpassing the full-cache baseline — a strong empirical claim rarely seen in this line of work.
5. Interpretability and analysis:
The authors visualize layer/head-wise retention scores, revealing emergent behaviors like sliding windows and sink tokens. This provides valuable insight into how LLMs allocate and forget contextual information.

### Weaknesses
Limited exploration of dynamic budgets:
The current method assumes a fixed memory budget M. It would be interesting to explore adaptive budgets (e.g., varying by layer, head, or task), especially since the authors mention this as future work.

### Questions
1. During inference, is the retention gate computed for every generated token, or cached and reused across time steps?
2. How sensitive is performance to the λ_cap hyperparameter and the initial bias b in the retention gate?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles KV cache limits for long context inference and proposes a learned decay gate to keep useful tokens. 
The idea is easy to follow and fits well with current attention sparsity efforts. 
In practice, it works reasonably well and the empirical bump over several benchmarks looks consistent.

However, I am not fully convinced the method is as conceptually “beyond attention” as the paper suggests. Even though the authors emphasize intrinsic token importance, the approach ultimately still plays through the attention path by reweighing the attention kernel. 
It is not really removing attention as a relevance signal, but shifting when the weighting is introduced. 
The learned score still modulates q·k, so the line between “intrinsic importance” and “attention-based importance” is blurrier than the paper presents.

On the training side, the method is positioned as a plug-in module that sits on top of frozen weights. If retention is so critical, it is natural to know why this is only applied for this scenario instead of making it part of the main training pipeline. Integrating the retention gates into pretraining or post training would likely change the model’s memory habits in a deeper way. 

The model relies on RoPE or similar positional embedding, and the learned gate sits on top. The paper does not clarify whether this interacts with positional encodings. It might be fine in practice, but some comment or ablation on this point would strengthen the clarity of the claim that the model learns “intrinsic” retention.

Overall, the method seems useful and the results are good enough to merit attention.

### Strengths
1. Novel idea, fits cleanly into existing models, and works well without heavy changes. 
2. Shows solid gains across tasks, occasionally even beating full-cache runs. 
3. Lightweight enough to feel practical, not just academic.

### Weaknesses
Since it’s still a trained model, it raises the natural question of why this isn’t folded into normal model training, and the paper doesn’t address that integration path. It also doesn’t discuss how the learned decay interacts with positional encoding, which leaves some ambiguity around whether it’s learning true importance or just reinforcing a recency style bias.

### Questions
1. Since the method relies on training, why isn’t it integrated into the general training pipeline instead of being applied frozen models?
2. Does the interaction between the decay gate and existing positional encoding introduce redundancy or bias toward recency, and has this been tested or ablated?
3. If the model were trained from scratch with the retention mechanism active, would it converge to a different or stronger retention strategy than the post-hoc approach?
4. How does the system behave under much longer sequences or tool calling contexts?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TRIM-KV, a trainable approach for KV cache eviction that assigns intrinsic importance scores to tokens at creation time through lightweight retention gates. The method trains these gates using distillation and capacity losses to enable memory-bounded inference while preserving model quality.

### Strengths
1. The paper introduces a trainable approach to KV cache eviction that learns token-level importance scores through retention gates. This design allows the model to identify and retain intrinsically important tokens 
2. The authors conduct experiments across multiple mathematical reasoning tasks and show competitive results.

### Weaknesses
1. The paper lacks comparison with LocRet, a highly relevant recent work that also computes importance scores for each token and discards low-importance tokens to reduce memory overhead. 

Locret: Enhancing eviction in long-context LLM inference with trained retaining heads

2. The paper would benefit from comprehensive evaluation on established long-context understanding benchmarks such as RULER and LongBench-V2 to validate the method's effectiveness.

3. The use of exponential decay for computing token importance scores appears problematic. Even with an initial retention score of 0.999, the importance decays to less than 5% after approximately 3000 tokens. This aggressive decay suggests the model may essentially degenerate into a sliding window approach for longer contexts.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
