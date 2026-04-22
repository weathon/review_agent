# Optimizing Mixture of Block Attention

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Mixture of Block Attention (MoBA) (Lu et al., 2025) is a promising building block for efficiently processing long contexts in LLMs by enabling queries to sparsely attend to a small subset of key-value blocks, drastically reducing computational cost.
However, the design principles governing MoBA's performance are poorly understood, and it lacks an efficient GPU implementation, hindering its practical adoption.
In this paper, we first develop a statistical model to analyze MoBA's underlying mechanics. Our model reveals that performance critically depends on the router's ability to accurately distinguish relevant from irrelevant blocks based on query-key affinities. We derive a signal-to-noise ratio that formally connects architectural parameters to this retrieval accuracy. Guided by our analysis, we identify two key pathways for improvement: using smaller block sizes, and applying a short convolution on keys to cluster relevant signals, which enhances routing accuracy.
While theoretically better, small block sizes are inefficient on GPUs. To bridge this gap, we introduce FlashMoBA, a hardware-aware CUDA kernel that enables efficient MoBA execution even with the small block sizes our theory recommends. We validate our insights by training LLMs from scratch, showing that our improved MoBA models match the performance of dense attention baselines. FlashMoBA achieves up to 14.7× speedup over FlashAttention-2 for small blocks, making our theoretically-grounded improvements practical.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the limitations of Mixture of Block Attention (MoBA). The authors develop a statistical model to derive a key signal-to-noise ratio (SNR) related to head dimension and block size, for better router retrieval accuracy. Based on the SNR formula, the paper uses smaller block sizes and larger head dimensions. Other contributions include key-convolution and efficient implementation.

### Strengths
1. Theoretical Framework: The paper introduces a novel signal-to-noise ratio (SNR) model that provides clear and actionable design principles. This provides guidelines for the selection of head dimension and block size.
2. High-Performance CUDA Kernel: FlashMoBA is a well-engineered, hardware-aware CUDA kernel. The Tiled-Topk is especially useful.
3. Strong Benchmark Results: The optimized MoBA models are shown to match or even outperform dense attention on challenging long-context benchmarks like LongBench and RULER.

### Weaknesses
1. Limited Generalizability Due to Small Model Scale: All experiments are conducted on a 340M parameter model. This raises significant questions about whether the paper's core findings would scale to the much larger models.
2. Unsubstantiated Link Between SNR and Experiments: The key experiment in Table 4, designed to validate the SNR theory's dependency on head dimension d, fails to control for model size (line 289). It is unclear if the performance improvements in Table 4 are due to the claimed increase in SNR or simply due to the larger model capacity. This methodological flaw means the empirical evidence for the paper's central theoretical claim is not as conclusive as presented.

### Questions
There are still rooms in the main text. You should put some algorithm into the main text instead of the appendix, such as the algorithm of Tiled-Topk.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper improves upon the previous sparse attention method, MoBA, through both theoretical analysis and kernel optimization. It analyzes MoBA’s performance from a signal-to-noise ratio (SNR) perspective and identifies that smaller block sizes and larger head dimensions yield performance benefits. To support these findings, the paper implements FlashAttention-style tiling optimizations in the kernel, making MoBA efficient under these settings.

### Strengths
1. The paper offers a statistical view that links MoBA hyperparameters to the SNR of attention computation. Although the connection between SNR and end-to-end model performance is not formally derived, the analysis provides a useful proxy for selecting better MoBA hyperparameter configurations.
2. The implementation of a FlashAttention-style MoBA kernel makes the approach practical even with small block sizes. The kernel achieves comparable speed to FlashAttention on short sequences and delivers speedups for long sequence inputs.

### Weaknesses
1. Experimental setup: The model architecture setup introduces confounding factors. While the paper claims to focus on optimizing MoBA performance, the model architecture employs sliding window attention (SWA) in half of the layers and involves dense attention in others, limiting the proportion of true MoBA layers. This mixture complicates the attribution of performance improvements and makes it unclear how much gain comes from MoBA, instead of SWA or dense attention components.
2. Key convolution description: The role of the short convolution on keys is not clearly presented. It is unclear how convolution enhances $\Delta_{\mu_{\text{eff}}}$, and additional explanation or intuition is needed. Moreover, implementation details are missing in the main paper. According to Table 1, the performance benefits from adding convolution appear inconsistent.

### Questions
1. Dense attention can be viewed as MoBA with block size (1) and top-(K = N). Based on the SNR analysis, smaller block sizes should yield better performance. Why then do most MoBA implementations outperform dense attention across benchmarks in Table 1? Could this discrepancy reflect other dominant factors, such as insufficient training data or incomplete convergence?
2. Could you provide more details on the rationale and implementation of key convolution in the main paper?

### Soundness
3

### Presentation
2

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
1. The paper presents a statistical analysis of the Mixture of Block attention (MoBA) to motivate the parameters that lead to better performance. Concretely, they propose that the SNR for the MoBA architecture is proportional to the sqrt of the ratio of the head size and the block size: with larger head sizes and smaller block sizes yielding better performance. Additionally, they motivate that semantic clustering in a block also helps improve block retrieval accuracy, leveraging a convolutional layer to demonstrate the point.

2. They provide an efficient implementation for the MoBA architecture in the spirit of FA that computes the top k sparse selection mask without materializing the entire mask, uses it to index into the KV cache to subsequently compute dense attention with the FA methodology , and then finally scatter back the results.

### Strengths
1. The paper motivates a better understanding of what aspects contribute to the improvement of the MoBA architecture. The characterization of the SNR as a function of the head size and block size is, to the best of my knowledge, novel and provides a good basic approximation on the framework to tune the performance of MoBA attention.

2. Their kernel implementation is particularly helpful for encouraging the broarder adaptation of the architecture. Given that attention tends to be a bottleneck for a number of new tasks, this is very helpful.

### Weaknesses
The following are my concerns with the paper:

1. The core contribution of the paper is relegated to the Appendix. This makes the paper a bit hard to follow, and given that the results motivate the majority of the paper, I do think that at least a part of it should be featured in the main paper. 

2. There are a number of assumptions made in the statistical analysis that may not hold true and at the very least merit some grounding with experimental results: L805 makes the assumption that q^Tk are independant dot products, however [1] argue that the dot product are correlated, accounting for a factor of O(d) and not O(sqrt(d)) as the authors propose. Likewise, the authors use the argument for CLT on large B to motivate characterising the delta between the informative and non informative dot products as a normal distribution, however, we subsequently move to argue that B should be small for improved performance. Thus there seems to be a tension between the proposed theory and subsequent proposed improvements.

3. It would be good to have experiments that directly validate the proposed theory. Concretely, for an S-NIAH task, one can compute the estimator for the score difference directly (the block containing the needle vs all other blocks, aggregated across all attention masks). One way to show the validity of the proposed SNR metric would be to plot the estimator as a function of d and B, and then subsequently show that it does follow the proposed trajectory.

4. The models that the authors investigate have a pretty high degree of global attention (50%). This itself can potentially act as a confounder and mask an pitfalls of the proposed algorithm. It would be more informative to have both an analysis of the proposed algorithm scales compared to vanilla dense and vanilla MoBA (i,e comparing to Section 3.1 in [2]) and how the difference in loss changes with introducing additional blocks of MoBA with lower block size / higher attention heads.

5. The experimental results are somewhat counterintuitive: according to the authors' proposal, the performance should keep improving with reduced block sizes. While that is true for Table 1, on Table 3 the trends do not hold. I would have expected the high SNR should be additionally more effective for the longer context evaluations, but that does not seem to be the case ?

6. For the experiments with higher d, the authors keep the number of heads fixed. Thus, the number of parameters (and consequently compute) for the models with higher d is more, since num parameters scales with O(d^2). This makes the ablation experiment with higher d values hard to compare, since they are not IsoFlops, and introduces an additional confounder: do the models with higher d values improve because of more parameters, or because of the better SNR value.

7. The results on the NIAH benchmark seem quite low: for a 32k - 64k context length, this task usually is taken as a necessary but not sufficient task for long context modeling: in fact even the MoBA authors demonstrate a 100% accuracy upto a 1M context window. However, in Table 2, even on the subset of 200 examples, the accuracy seems to be very low for 32k and 64k context lengths. This seems a bit off: it might be because the authors tried zero shot with RoPE for length extrapolation, which is known to have poor performance. It might be better if the authors adapted the vanilla context length extension strategy of training on longer contexts a bit more with RoPE interpolation, and the using the subsequent models for the evaluations. With the current results, it's hard to understand if the proposed method hurts the long context performance or not.

[1] Yang, Greg, et al. "Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer." arXiv preprint arXiv:2203.03466 (2022).

[2] Lu, Enzhe, et al. "Moba: Mixture of block attention for long-context llms." arXiv preprint arXiv:2502.13189 (2025).

### Questions
For the reduction in the block size experiments, I was not sure of the differences between the proposed experiments by the authors and the experiments on sparsity / granularity tradeoffs of MoBA presented in [1] (section Ablation Study on Fine-Grained Block Segmentation). Would it be possible to clarify the same ? 


[2] Lu, Enzhe, et al. "Moba: Mixture of block attention for long-context llms." arXiv preprint arXiv:2502.13189 (2025).

### Soundness
2

### Presentation
2

### Contribution
3
