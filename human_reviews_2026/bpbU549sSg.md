# xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Scaling laws play a central role in the success of 
Large Language Models (LLMs), enabling the prediction of 
model performance relative to compute budgets prior to training.
While Transformers have been the dominant architecture, 
recent alternatives such as xLSTM offer linear complexity
with respect to context length while remaining competitive in the billion-parameter regime.
We conduct a comparative investigation on the scaling behavior of Transformers and xLSTM along the following lines, providing insights to guide future model design and deployment.
First, we study the scaling behavior for xLSTM in compute-optimal and over-training regimes using both IsoFLOP and parametric fit approaches on a wide range of model sizes (80M-7B) and number of training tokens (2B-2T).
Second, we examine the dependence of optimal model sizes on context length, a pivotal aspect that was largely ignored in previous work. 
Finally, we analyze inference-time scaling characteristics.
Our findings reveal that in typical LLM training and inference scenarios, 
xLSTM scales favorably compared to Transformers.
Notably, xLSTM models consistently Pareto-dominate Transformer models, delivering lower cross-entropy loss for the same compute budget.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper points out that the authors systematically compared the scaling laws of the performance-optimized xLSTM architecture with those of the dense multi-head self-attention Transformer architecture. The authors conducted comparative analyses from perspectives such as Training, Context length, and Inference.

### Strengths
1. This paper is well-written, with clear logic and concise readability.

2. It conducts extensive experiments, including comparative experiments on models of different types and scales.

3. The authors provide a wide range of comprehensive evaluation metrics.

### Weaknesses
1. First, this paper lacks sufficient novelty. Models based on the xLSTM architecture have already demonstrated significant advantages in areas such as inference in previous studies. However, this paper only conducts extended comparative experiments based on this existing foundation, which casts doubt on its innovativeness.

2. Second, compared to models with quadratic complexity, the design of LLM architectures based on linear complexity is inherently intended to reduce computational costs and achieve more significant benefits during training and inference—and this is a well-known fact. The paper conducts comparative analyses from perspectives including Training, Context length, and Inference, but these experiments are carried out under the premise that such performance characteristics are already known, resulting in insufficient contributions from the paper.

3. Finally, to fully compare the performance of models with linear complexity and quadratic complexity, limiting the scope solely to the xLSTM architecture is flawed. It is necessary to include a broader range of models and conduct more comprehensive result comparisons.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors study scaling behavior of a recently proposed sequence modeling architecture, finding it does well on many desirable axes compared to Transformer baselines.

### Strengths
- Study relevant regimes, incl. overtraining/inference time compute which are rarely paid attention to in scaling work even though they are principle considerations when training/serving modern frontier models. 
- Science of scaling is cleanly and thoroughly done. Reproduction of important past work (power-law exponents from Chinchilla) is reassuring as a sanity check. 
- Lots of large-scale empirics and careful parametric fits, very valuable contribution.

### Weaknesses
- Transformer baseline is weak and out of date (Llama-2 architecture in late 2025). There are a number of Transformer++ improvements that have appeared since Llama-2, all else fixed, that make big improvements to performance at scale (eg. RoPE, GQA, addressing attention sinks, no biases on linear layers, etc etc). 
- I'm not convinced xLSTM actually outscale Transformers, and in fact would be willing to bet they do worse at frontier compute regimes. We can see the advantage decreasing with compute in Fig1(left), and we can see the faint Transformer curves overtaking the xLSTM ones on Fig1(right). This looks exactly like Fig4 of the Mamba [1] paper where Mamba was touted as "scaling favorably" but the gap shrunk slightly with compute in their plots. When things were scaled up, of course Transformers did much better than Mamba. I expect a similar thing to hold here (notice how the plot is missing Transformer at 1e23 where xLSTM begins to plateau, I would be very interested in seeing that last Transformer data point). Nonetheless the thorough and expansive empirics here are valuable, and this is how progress in architecture design is made, so this is a valuable paper.

### Questions
See weaknesses.

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
3

### Summary
This paper presents a systematic empirical comparison of scaling behaviors between xLSTM and Transformer architectures for large language models. The study spans three dimensions: (1) training efficiency under compute-optimal and over-training regimes across 80M-7B parameters and 2B-2T tokens, (2) the relationship between optimal model size and context length, and (3) inference-time characteristics (TTFT and step time). The authors claim xLSTM is Pareto-dominant in training loss vs. compute, requires larger compute-optimal models, and shows widening advantages with increasing context length. This is a technically solid, large-scale empirical study that provides valuable evidence for xLSTM's scaling properties. The methodological innovations (accurate FLOP accounting, context-length scaling analysis) are notable contributions.

### Strengths
1. Comprehensive Experimental Scale：The study demonstrates an impressive experimental scale, encompassing 672 training runs with a total compute of 3.2×10²³ FLOPs. It systematically explores both the compute-optimal and over-training regimes—a distinction of significant practical relevance—while maintaining a fair and consistent comparison framework through unified training recipes, shared data (DCLM-BASELINE), and identical hyperparameter schedules.

2. Clear Empirical Findings: The paper presents clear and consistent empirical findings. xLSTM demonstrates Pareto dominance over Transformers, achieving lower loss at fixed compute across five orders of magnitude. It also exhibits remarkable over-training stability, with constant power-law exponents observed up to M=2200, confirming its reliability for inference-optimized training. Furthermore, xLSTM maintains strong context length robustness, as its optimal model size remains stable even when context length increases, whereas the Transformer's performance degrades notably.

### Weaknesses
1. **No Downstream Task Evaluation**. The entire paper focuses exclusively on pretraining cross-entropy loss without any downstream benchmarks (e.g., MMLU, HumanEval, common sense reasoning, summarization).

2. The context length experiments appear confounded, as different context lengths are trained on non-identical data distributions without clarification on how this issue is handled—for instance, whether through re-chunking, padding, or different data splits. Moreover, the y-axes in Figure 5 are not directly comparable across context lengths, and it remains unclear how the “tokens D” are counted when context size varies (i.e., whether they represent effective or nominal tokens). Clarifying the data preprocessing process or adding a fixed-dataset ablation where only context packing changes would strengthen the validity of these results.

### Questions
1. L024: Quantify "advantage widens", what is the relative improvement at 2K vs. 16K contexts?
3. L313: If losses aren't comparable across contexts, how should we interpret the intersecting curves in Figure 5 (right)?
4. L399: "Largest xLSTM has lower step time than smallest Transformer", is this specific to 16K prefill, or does it hold at shorter contexts (e.g., 512)?
5. It would be better to include at least one long-context benchmark evaluation (e.g., LongBench) to more convincingly demonstrate xLSTM’s effectiveness in handling extended context scenarios.

### Soundness
3

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
This paper systematically compares the scaling of xLSTM and Transformers. Using extensive controlled runs (80M–7B parameters; 2B–2T tokens) and two protocols (IsoFLOP budgeting and parametric fitting), it examines: (i) scaling in compute-optimal and over-training regimes, (ii) how compute-optimal model size depends on context length, and (iii) inference scaling in algorithmic complexity, latency, and throughput. The key finding is that xLSTM consistently scales better than Transformers, and the advantage widens as training or inference contexts grow longer.

### Strengths
1.  Provides a large, controlled scaling study comparing xLSTM and Transformers across budgets and context lengths. Combines IsoFLOP budgeting with parametric fitting to bridge compute-optimal and over-training regimes.

2.  Extensive, well-structured experimental sweeps (80M–7B; 2B–2T) with clear loss–compute Pareto analyses.

3.  Coherent problem framing and storyline from scaling laws → compute-optimal sizing → inference scaling. Figures generally align with claims and support the narrative.

4. Offers compute-aware guidance on architecture choice, model size, and context length. Helps quantify when linear-time sequence processing becomes advantageous, informing system design and resource allocation decisions.

### Weaknesses
1.  Several key definitions/configs are only provided in the appendix; Moving essential config details into the main text, unifying definitions/notation would significantly improve the presentation of the paper.

2. Results seem to be on a specific data mix, tokenizer, and recipe. Adding cross-dataset, cross-tokenizer, and recipe sensitivity studies would make the paper stronger. 

3. Some figures are too crowded. Several plots are visually dense, making it hard to extract key trends.

### Questions
1.	It would be good to provide cross-dataset and cross-tokenizer results to test whether the reported margins persist?

2.	How do you normalize FLOPs/MemOps across architectures (kernel fusion, activation checkpointing, flash-attention)?

### Soundness
3

### Presentation
2

### Contribution
2
