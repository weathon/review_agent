# REAL: REtrieval-Augmented and Logic-constructed Attention Behaviors for Robust KV Cache Compression

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
The growing input sequence length of large language models (LLMs) places increasing pressure on key-value (KV) cache storage, making efficient inference challenging. Existing retrieval-based compression methods neglect the impact of distracted, biased, and widespread attention behaviors, raising robustness concerns. To address these challenges, this paper proposes REtrieval-Augmented and Logic-constructed (REAL) KV cache compression that implements a robust, low-cost, training-free method, capturing diverse attention behaviors. REAL introduces an attention weight confusion matrix (AWCM) to categorize attention behaviors and an inference score (INFsc) that balances retrieval and logic for head-wise dynamic budget allocation with an empirical per-layer safeguard. Experiments on long-sequence QA and non-QA tasks show that REAL achieves more robust compression than state-of-the-art baselines and even surpasses FullKV in certain situations. To our knowledge, REAL is the first approach to compress KV caches by attention behavior analysis, offering a new perspective.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper mainly addresses the issues of KV cache efficiency and inference performance in long-text scenarios, and proposes a dynamic cache allocation method based on matrix computation metrics.

### Strengths
The paper is clearly written, with well-defined problems and a well-explained methodology. The addressed problem is practically valuable, and the main experiments are relatively sufficient.

### Weaknesses
1.	The method relies on a large number of hyperparameters, and its generalizability has not been sufficiently validated. For example, if a different, more specialized dataset is used, it is unclear whether the current hyperparameters would still perform well, or if they would need to be re-tuned. There seem to be quite a few hyperparameters in total.
2.	The ablation studies and comparisons are not entirely comprehensive. For instance, it would be helpful to compare against some simple alternative sorting schemes to demonstrate the effectiveness of the proposed matrix-based metrics. It is possible that even simple sorting strategies could achieve similar results.

### Questions
See weekness

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a training-free KV cache compression method REAL, which is designed by analysing the different attention behaviours. The experiments on various models and tasks show the method can compress the KV cache without decreasing performance too much.

### Strengths
Designing the KV cache compression method based on the different attention behaviours is intuitive. The method is training-free and thus could be applied to large models without heavy fine-tuning.

### Weaknesses
It is unclear why the four attention behaviours (retrieval-augmented, distracted, biased, and widespread) are considered. Is there any other behaviour? Some implementation details of these behaviours are spread in the introduction, but the formal definitions are missing.

The ambiguous term "robustness" is heavily used in the paper. It is unclear in which aspects the proposed method improves the performance. It would be better to include a case study to explain.

Do you use the same compression ratio for comparing different methods in Table 2? It is also important to present how the performance changes with the compression ratio. Please refer to the experimental setting in: https://github.com/NVIDIA/kvpress

In Figure 10, it doesn't make sense that every method has the same decoding time (except FullKV), because different methods have different strategies for calculating importance; at least, the first token latency is different. 

The definition of the budget $B$ is not clear. Is it the number of KV pairs? What is the definition of $b_{base}$, and what is "predefined ratio $\beta$"?

### Questions
What does "dynamic eviction is constrained by model dimensionality" mean?

The explanation after eq1: $Q \cdot K_{\text{Retrieval-Augmented}}$ and other $K$ are amplified by exp. But the softmax will further decrease the small value in a vector. What information do you want to convey?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a KV cache compression method that constructs a confusion matrix by classifying tokens based on two criteria: whether their attention region lies within or outside the “needle” part, and whether their attention rank belongs to the Top-k or non-Top-k group. The method then computes the harmonic mean of the ratios derived from this confusion matrix and uses the resulting value as an importance metric for KV cache pruning. Experimental results demonstrate that the proposed approach achieves superior performance compared to SnapKV and PyramidKV across a diverse set of benchmarks.

### Strengths
The paper introduces a novel KV cache importance metric and demonstrates superior performance compared to existing baselines.

### Weaknesses
- The main concern lies in the clarity of writing and presentation. Specifically, the description of how the needles are generated and inserted is difficult to follow. Are the authors synthetically inserting needles to determine which attention heads to retain? If so, how is this synthetic needle data constructed?
- Additionally, is the resulting head-wise compression static (i.e., the same pattern applied across all evaluation data) or dynamic (where compression is performed separately for each evaluation instance)? If it is the latter, this could introduce significant computational overhead, as it needs to compute attention patterns for multiple different positions.
- It is also unclear how many calibration samples (synthetic needles) were used and what process was followed to construct them.
- Are token-level compression methods such as SnapKV implemented dynamically, with patterns varying across evaluation samples?
- If the proposed method instead performs static head-level compression, a direct comparison with DuoAttention [1] would be more appropriate and needed.

[1] Xiao, Guangxuan, et al. "Duoattention: Efficient long-context llm inference with retrieval and streaming heads." ICLR 2025.

### Questions
See the above weakness section.

### Soundness
2

### Presentation
1

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
Authors propose REAL, a method that uses an attention weight confusion matrix (AWCM) and an inference score (INFsc) to balance retrieval, distraction, and bias signals when allocating KV cache budgets on a per-head and per-level basis. Authors use synthetic needles-in-a-haystack (NIAH) to profile heads, and REAL dynamically redistributes the KV cache capacity across different attention types. Results show improvements on long-context benchmarks (LongBench, LongBench v2, LooGLE) over strong baselines like PyramidKV and SnapKV.

### Strengths
- The differentiation among different types of attention for allocating KV cache budgets seems novel, interesting, and potentially cognitively justified (e.g. see the work on System 1/2 reasoning), and AWCM/INFsc seems a sound way to realise this
- Significant (although it would be nice to have statistical significance tests) improvements in QA and non-QA tasks in terms of downstream accuracy vs latency/memory trade-offs in comparison with very competitive baselines

### Weaknesses
- Computing the AWCM seems computationally heavy -- how does that impact the applicability of the method?
- Not sure results are statistically significant (e.g. in the case of the comparison with tuned HeadKV variants)
- How robust are results to the choice of e.g. $\beta$ ?

### Questions
Please see my "weaknesses"

### Soundness
3

### Presentation
3

### Contribution
3
