# MrRoPE: Mixed-radix Rotary Position Embedding

- Decision: Accept (Oral)
- Scores: 8, 6, 6, 6

## Abstract
Rotary Position Embedding (RoPE)-extension refers to modifying or generalizing the Rotary Position Embedding scheme to handle longer sequences than those encountered during pre-training. However, current extension strategies are highly diverse and lack a unified theoretical foundation. In this paper, we propose $\textbf{\textit{MrRoPE (Mixed-radix RoPE)}}$, a generalized encoding formulation based on a radix system conversion perspective, which elegantly unifies various RoPE-extension approaches as distinct radix conversion strategies. Based on this theory, we introduce two training-free extensions, $\textbf{\textit{MrRoPE-Uni}}$ and $\textbf{\textit{MrRoPE-Pro}}$, which leverage uniform and progressive radix conversion strategies, respectively, to achieve “train short, test long” generalization. Without fine-tuning, MrRoPE-Pro sustains over 85% recall in the 128K-context Needle-in-a-Haystack test and achieves more than double YaRN’s accuracy on Infinite-Bench retrieval and dialogue subsets. Theoretical analysis confirms that MrRoPE-Pro effectively raises the upper bound of RoPE's attainable encoding length, which further validates the reliability and utility of our theory and methodology.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MrRoPE (Mixed-Radix Rotary Position Embedding), a training-free method for extending the context length of transformer models equipped with RoPE (Rotary Position Embeddings). The key insight is to reinterpret RoPE as a form of mixed-radix number system, where each rotational frequency corresponds to a different positional base. The authors propose two practical schemes: 1) MrRoPE-Uni: uniformly redistributes intermediate rotation frequencies; 2)MrRoPE-Pro: progressively adjusts them across dimensions. The authors show that these schemes unify and generalize several prior test-time scaling methods such as NTK and YaRN. Extensive experiments on LLaMA-2/3 and Qwen models demonstrate consistent improvements across multiple long-context benchmarks (RULER, InfiniteBench, LongBench-v2, Needle-in-a-Haystack) without any fine-tuning. MrRoPE-Pro, in particular, exhibits stronger stability and better RoPE-bound behavior at very long sequence lengths (up to 128K tokens).

### Strengths
1. **Novel theoretical perspective**: The mixed-radix interpretation provides a new and elegant way to understand RoPE scaling. It explains existing methods (NTK, YaRN) under a unified mathematical framework and motivates new variants.
2. **Practical and lightweight**: The proposed method is training-free and only modifies the RoPE computation at inference time. It can be easily applied to off-the-shelf models, offering immediate practical value.
3. **Comprehensive empirical evaluation**: Experiments are extensive, covering perplexity, synthetic retrieval, and multiple real-world long-context benchmarks. The improvements of MrRoPE-Pro over prior inference-only methods are consistent and significant.
4. **Interpretability and diagnostics**: The paper connects the improvements to theoretical properties such as the “RoPE bound” and attention distribution behavior, providing an interpretable explanation of why progressive scaling helps.
5. **Strong empirical gains without fine-tuning**: MrRoPE-Pro achieves competitive or superior performance compared to other test-time extensions and even approaches the performance of some fine-tuned long-context models.

### Weaknesses
1. **Lack of strictly rigorous theoretical analysis**: The “RoPE as radix conversion” formulation relies on approximations (ignoring floor/mod operations), but the paper does not formally analyze the approximation error or its effect on attention stability. The derivations remain mostly intuitive rather than strictly formal proven.
2. **Missing fine-tuned or stronger baselines**: The comparison is limited to test-time methods (YaRN, NTK). Including light fine-tuning baselines (e.g., LongRoPE, xPOS, or other retrained RoPE variants) would help position the method more clearly in terms of cost–performance trade-off.
3. **Hyperparameter sensitivity not analyzed**: MrRoPE-Pro depends on specific tuning of hyper-parameters. The paper does not present an ablation or sensitivity study, leaving unclear whether performance is robust across models or sequence lengths.
4. **No evaluation of runtime or stability**: Since MrRoPE modifies rotation frequency computation, the paper should report inference-time latency, GPU overhead, and numerical stability (especially under FP16 inference, quantized inference is also welcomed).

### Questions
1. How accurate is the approximation of RoPE as a mixed-radix system when floor/mod terms are **non-negligible**? Could you provide error bounds or an empirical validation of this assumption?
2. How sensitive is MrRoPE-Pro to the choice of scaling parameters and dimensional splits? Would the same configuration generalize to very different models or hidden sizes?
3. Could unified radix conversion be extended beyond RoPE? For instance, to ALiBi or other relative positional encodings? Is the mixed-radix perspective specific to sinusoidal embeddings?
4. What is the inference-time cost or memory overhead compared to baseline RoPE and YaRN implementations?
5. Could you provide results for fine-tuned long-context methods to better contextualize MrRoPE’s efficiency vs. performance trade-off?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reframes the problem of extending the context window of LLMs with RoPE, to view position encoding through the lens of mixed radix conversion. This allows to unify previous methods as different choices within the proposed framework and outperform YaRN on 13 tasks.

### Strengths
The unifying theory is good and the results are compelling across different tasks. Good to see a training free method has consistent gains.

### Weaknesses
It is not clear if this radix idea can work for other positional encoding method, which limits the impact to RoPE.  Also not clear if the progressive scaling in the proposed method introduces any latency compared to YaRN? 

Would also be helpful to have sensitive analysis, about the parameters inherited from YaRN in appendix B.

### Questions
Would you explain why the progressive strategy in the proposed method is better than YaRN's way? Is there any inference latency penalty for using the proposed method or YaRN? If the radix concept could be used for other types of positional encodings?

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
2

### Summary
This paper focuses on the context window extension problem of rotational position encoding (RoPE). Aiming at the pain point that the existing RoPE extension strategies are diverse and lack a unified theoretical foundation, the paper proposes a mixed-base rotational position encoding (MrRoPE) framework, and develops an efficient training-independent extension method based on it, and the related results are verified by quantitative experiments.

### Strengths
- MrRoPE constructs a universal framework from the radix conversion perspective, and unifies mainstream RoPE extension methods such as NTK and YaRN into different base conversion strategies, which solves the problem of the lack of a unified theoretical basis for the existing extension schemes, and facilitates system analysis and optimization.
- The proposed MrRoPE-Pro requires no additional fine-tuning and performs outstandingly in long context tasks: over 85% recall in Needle-in-a-Haystack tests with 128K contexts, more than twice the accuracy of YaRN in Infinite-Bench retrieval and dialog tasks, and in benchmarks such as RULER, LongBench-V2, and others. and consistently outperforms existing methods in benchmarks such as RULER, LongBench-V2, etc.
- The upper bound on the effective coding length of RoPE is theoretically boosted, and the model maintains excellent performance over the full context range of 8K to 128K through a progressive base conversion strategy that preserves the fine-grained information in the high-frequency dimensions and stabilizes the distribution of the attention scores in the intermediate dimensions.

### Weaknesses
- The RoPE extension method (MrRoPE-Uni/Pro) proposed in this paper only focuses on the “no-training” context-window extension mode, and no fine-tuning experiments have been conducted. This makes it impossible to make a direct comparison with extension methods such as xPOS and LongRoPE, which need to be evaluated in fine-tuning scenarios, and it is difficult to comprehensively prove its superiority in different application scenarios and its integration potential with other methods.
- It is well known that the training process of LLM model is complex and cumbersome, and the variation of hyperparameters may substantially affect the training results, so the authors should explore in detail whether the choice of hyperparameters affects the model performance, and whether the same hyperparameters proposed in the paper can be used for LLMs with different architectures.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to provide a unified theoretical foundation for context window extension of Rotary Position Embedding (RoPE) in LLMs. It proposes the MrRoPE (Mixed-radix RoPE) framework, which reinterprets RoPE and its extensions as a mixed-radix number system conversion. Based on this, the paper introduces a new training-free method, MrRoPE-Pro (progressive radix conversion), and demonstrates its significant advantages over state-of-the-art methods like YaRN across multiple benchmarks.

### Strengths
The work addresses the critical problem of training-free context extension for LLMs, and its unifying framework could have a profound impact on future research. The connection between RoPE extension and radix conversion theory is a highly novel and insightful perspective, elevating heuristic designs to a theoretical level.

### Weaknesses
1. The characterization of YaRN as "regressive" is based on observation rather than rigorous derivation. The paper fails to mathematically prove this from YaRN's original equations.
2. The method heavily relies on hyperparameters (`α`, `β`) inherited from YaRN, yet lacks a sensitivity analysis or ablation study to verify their optimality for the new method.
3. The comparison is limited to training-free RoPE methods, failing to adequately position the work relative to fine-tuning-based approaches or alternative schemes like ALiBi.
4. Experiments are confined to small- to medium-scale models (3B-8B). The generalizability of its conclusions to larger models (e.g., 70B scale) remains unproven.
5. The analysis of downstream performance gains is superficial. It reports *what* improved but lacks qualitative or error analysis to explain *why* MrRoPE-Pro is better.
6. The method's potential as a better initialization for long-context fine-tuning is untested. A crucial experiment is missing to verify if its advantage persists after fine-tuning.

### Questions
1.  Could you provide a more rigorous mathematical derivation, starting from YaRN's original interpolation formulas, to derive its corresponding scaling factors `λj`?
2.  Have you conducted any sensitivity analysis or ablation studies on the hyperparameters `α` and `β` inherited from YaRN?
3.  Could you elaborate on the fundamental trade-offs among the regressive (YaRN), uniform (Uni), and progressive (Pro) strategies?
4.  Does your "progressive" strategy inherently maintain attention score stability better than YaRN, reducing the reliance on heuristics like temperature scaling `t`?

### Soundness
2

### Presentation
3

### Contribution
2
