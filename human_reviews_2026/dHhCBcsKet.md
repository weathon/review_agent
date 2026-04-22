# ParallelTime: Dynamically Weighting the Balance of Short- and Long-Term Temporal Dependencies

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Modern multivariate time series forecasting primarily relies on two architectures: the Transformer with attention mechanism and Mamba. In natural language processing, an approach has been used that combines local window attention for capturing short-term dependencies and Mamba for capturing long-term dependencies, with their outputs averaged to assign equal weight to both. We find that for time-series forecasting tasks, assigning equal weight to long-term and short-term dependencies is not optimal. To mitigate this, we propose a dynamic weighting mechanism, ParallelTime Weighter, which calculates interdependent weights for long-term and short-term dependencies for each token based on the input and the model's knowledge. Furthermore, we introduce the ParallelTime architecture, which incorporates the ParallelTime Weighter mechanism to deliver state-of-the-art performance across diverse benchmarks. Our architecture demonstrates robustness, achieves lower FLOPs, requires fewer parameters, scales effectively to longer prediction horizons, and significantly outperforms existing methods. These advances highlight a promising path for future developments of parallel Attention-Mamba in time series forecasting. The implementation is readily available at the GitHub link.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces ParallelTime, a new architecture for time series forecasting. It addresses the suboptimal equal weighting of long- and short-term dependencies in existing hybrid models with a dynamic weighting mechanism (ParallelTime Weighter), achieving state-of-the-art performance with improved efficiency and robustness.

### Strengths
1. The proposed ParallelTime Weighter is interesting. It dynamically and interdependently assigns weights to short-term and long-term components for each token.

2. A significant advantage is the achievement of superior performance with notably fewer parameters and lower computational cost compared to strong baselines like PatchTST.

3. The model demonstrates SOTA or highly competitive results across eight real-world benchmarks, effectively validating the proposed architecture's effectiveness.

### Weaknesses
1. The analysis, particularly regarding the ParallelTime Weighter, remains somewhat superficial. It lacks a clear explanation of why and when the model chooses to emphasize short- vs. long-term dependencies based on specific time series patterns or characteristics.

2. While the integration is novel, several core components (patching, channel independence, window attention, Mamba, registers) are directly adopted from existing literature.

3. The ablation analysis is insufficient as it fails to deconstruct the model's core contributions. To strengthen the claims, performance should be compared against the following ablated versions: 1) Mamba-only, 2) Attention-only, 3) without Registers, etc.

### Questions
I noticed noticeable inconsistencies in the title and font formatting compared to the standard ICLR template. Could the authors confirm the technical compliance of their document with the submission guidelines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a dynamic weighting mechanism to balance the time dependency between short and long cycles, thereby achieving better performance.

### Strengths
This paper uses local attention and Mamba to extract information at different time intervals and performs adaptive weighted ensemble.

### Weaknesses
1、The method is essentially a multi-scale framework that divides the time series into long-term and short-term paths for separate feature extraction, without introducing any new modeling mechanism.

2、The fusion of long- and short-term features relies on simple weighting or concatenation, lacking an adaptive interaction mechanism.

3、Moreover, it merely employs Mamba and Attention to extract their respective effective features, but contributes no architectural or algorithmic innovation.

4、There is no theoretical or empirical evidence showing that this parallel dual-branch design outperforms existing multi-scale or frequency-decomposition methods.

5、Overall, the improvement lies purely at the implementation level, without any new algorithmic principle or fundamental contribution.

### Questions
See Weaknesses.

### Soundness
2

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
This paper proposed **ParallelTime** to combine two dominant architectures, Transformer and Mamba. While prior work in NLP has combined these models, it typically does so by simple averaging, which assigns equal weight to both short- and long-term features. The authors argue this is suboptimal for time series data.

The core contributions are:
1. **ParallelTime Architecture**: A decoder-only model that tokenizes the input multivariable time-series data as patchifed univariable time-series and processes them through a combination of Mamba blocks and window-attention blocks by **ParallelTime Weighter** instead of simply summing or averaging.
1. **ParallelTime Weighter**: A novel, dynamic, and token-specific weighting mechanism to combine the output of the window-attention block and output attention.

Detailed experimental results show ParallelTime achieves SOTA performance on real-world benchmarks.

### Strengths
1. This paper is the first to combine the mamba and transformer architectures for time-series, expecting short- and long-term dependencies gains from their architectures.
1. A novel Parallel Weighter is proposed to combine the outputs from the mamba blocks and the local attention blocks, and experiment results show that it benefits the performance by simply summing or averaging (with ablation studies for justification).
1. Detailed experiments are conducted to show that the Parallel Time model achieves state-of-the-art performance, with nearly half the FLOPs compared to PatchTST.
1. The writing is detailed and easy to follow.

### Weaknesses
This paper is solid and well-written, except for some minor weaknesses:
1. Missing definition of $\mathbf{y}_t$ in line 222, and missing relationship between $\text{Mamba}(\cdot)$ and $\mathbf{x}, \mathbf{y}, \mathbf{h}$ in line 227.
1. Missing comparison of pure mamba and pure window-attention in Figure 6.
1. Efficiency comparison only compares with PatchTST, missing some potentially more lightweight models like DLinear.
1. ParallelTime was designed to capture more long-time dependencies (or global dependencies), but the result in Table 1 shows it only performs the best half of the dataset in prediction with a prediction length of 720.
1. Figure 4 shows the visualization of the weighting outputs of the Parallel Weighter to justify that the Parallel Weighter can dynamically adjust weights of short- and long-term dependences. However, the prediction length of 192 is not long enough (compared to 720) to show long-term dependencies.

### Questions
1. Is there any prior work that applies window-attention with global registers in time-series data (I did not see one in the related works from line 108 to line 113)? If so, has the paper compared such a model? If not, will it work well without a combination of mamba blocks? Also in Appendix 9.3, it's stated that global registers yield slight performance enhancements. Are there any experimental results comparing ParallelTime with and without global registers?
1. What is the reason to compress the dimensionality from $\text{dim}$ to $\sqrt{\text{dim}}$ in ParallelTime Weighter other than some other dimensionalities?
1. Efficiency comparison shows that ParallelTime has fewer FLOPs (and parameters), but how about the inference latency?
1. ParallelTime Weighter outperforms simple sum or average, but how about concat (in dimension)?
1. Since window-attention has a global register, can it also capture long-term dependencies (see statements in lines 247, the author says window-attention with global registers only captures short-term dependencies).

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
3

### Summary
In this paper, the authors proposed a new parallel hybrid framework, ParallelTime, which dynamically balances short- and long-term temporal dependencies for long sequence time-series forecasting. The model integrates a Mamba branch for capturing long-term dependencies and a windowed attention branch for modeling short-term patterns, and introduces a ParallelTime Weighter that adaptively assigns token-level weights to fuse the two representations. Additionally, the authors designed an efficient Expand–Compress–Project projection strategy to reduce parameters and computational cost. Extensive experiments on multiple benchmark datasets demonstrate that ParallelTime achieves superior forecasting accuracy and efficiency compared to state-of-the-art methods.

### Strengths
1. The paper as a whole is content-rich, with strong logical connections between the parts. The entire work forms a closed-loop logic of problem, solution, verification, and explanation. The explanation section, in particular, discusses the underlying logic of the patterns and method performance, which enhances the interpretability of the approach.

2. The paper demonstrates good originality, both in the global design of the ParallelTime framework and in specific details (for example, Section 3.1 designs non-overlapping data blocks to ensure performance, and Section 3.4 ECP is not a simple dimensionality reduction, but uses local convolution to preserve temporal bias, etc.).

3. In Section 5 and the Appendix, the input length and patch length in the comparative experiments are explicitly aligned, eliminating the pseudo-advantage of “more input or longer context leading to seemingly better performance.” This design is well-executed and makes the performance improvement more credible.

### Weaknesses
1. In Section 3.3, when introducing global registers, the paper defines the content stored as global shared information, but does not indicate the rule or type of this information selection, such as whether it is statistical features like mean values or periodic values like peaks, making the register a black-box component.

2. The paper only verifies the efficiency of the method from the perspective of experimental FLOPs, lacking a theoretical explanation of the method’s superiority. For example, it could compare the time complexity from a theoretical perspective with that of transformers or other models, which would make the argument more solid.

3. The interaction process between components, such as their collaborative mechanism, is not clearly explained. For example, in Section 5.2, it is found that the Layer-2 attention weight is always higher than Layer-1, but the reason for this is not explained. Is it because Layer-1’s long-term dependency extraction provides the foundation for Layer-2’s short-term optimization, or is it because Layer-2’s FFN enhances the attention features?

### Questions
1. In Section 3.3, a fixed window size (a ratio of 1:9 relative to the number of input patches) is adopted, but the motivation for choosing this ratio is not explained. Was this ratio determined through experimental tuning? Is it stable across different datasets?

2. In Figure 8, the weight heatmap shows that the model assigns higher weights at mutation points. Does this pattern also appear in other datasets, such as Traffic or Exchange? Have you observed any failure cases or abnormal weight allocations?

3. The model shows the most significant improvement in long-horizon tasks. Is there direct evidence proving that the improvement comes mainly from Mamba’s long-term modeling contribution, or from the Weighter’s dynamic fusion? Is there an ablation table or figure that can verify this point?

### Soundness
3

### Presentation
3

### Contribution
3
