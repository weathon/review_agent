# Frayed RoPE and Long Inputs: A Geometric Perspective

- Decision: Accept (Poster)
- Scores: 4, 6, 10, 4

## Abstract
Rotary Positional Embedding (RoPE) is a widely adopted technique for encoding position in language models, which, while effective, causes performance breakdown when input length exceeds training length. Prior analyses assert (rightly) that long inputs cause channels to rotate “out of distribution,” but it is not clear how extra rotation relates to or causes pathological behavior. Through empirical and
theoretical analysis we advance a unified geometric understanding of attention behavior with RoPE. We find that attention induces tight clustering of separated key and query latent point clouds, allowing for creation of sink tokens: placeholders that allow attention heads to avoid token mixing when not required. RoPE applied to longer inputs damages this key/query cluster separation, producing pathological
behavior by inhibiting sink token functionality. From this geometric perspective, we propose RoPE-ID (In Distribution), a straightforward modification that allows attention layers to generalize to longer inputs out of the box: apply RoPE with high frequency to a subset of channels. We demonstrate the effectiveness of RoPE-ID for extended inputs using 1B and 3B parameter Transformers on the LongBench
and RULER information retrieval benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose RoPE-ID, a rotary positional embedding (RoPE) based positional embedding method that scales a proportion of head-specific channels to ensure that long-context inputs are more in-distribution relative to the training data. The authors show that this method is competitive with existing long-context extension techniques on numerous benchmarks on models of multiple sizes.

### Strengths
The authors provide a strong motivation for the method through the analysis of attention sinks and singular values. The experiments are also comprehensive and show some cases of RoPE-ID being beneficial compared to alternative methods.

### Weaknesses
- The contribution itself remains somewhat trivial; relative to the changes with standard RoPE, it appears that the primary difference is only that low frequency RoPE dimensions are individually scaled such that they can complete the rotation within the provided training length. 

- There is a rather significant lack of clarity regarding the effects of the hyper-parameter choices; for example using half the dimensions in a fixed manner and/or the maximum/minimum frequencies allowed.

- The experimental results are rather unconvincing of the benefits of the method due to some inconsistencies. For example, on RULER, improvements in the 1B model do not appear statistically significant compared to YaRN. Meanwhile, on the 3B model, performance on the 8K context is much better in YaRN but then decreases for 16K. For LongBench, YaRN outperforms RoPE-ID on the 3B model but not the 1B. Thus I cannot gauge exactly what the trend will be for either larger models or more data.

- These models all require training from scratch; this is a rather strong limitation. Additional baselines should include NTK-aware scaling methods, which the authors mention in the related work but do not compare with on downstream tasks. Given the additional complexity of pre-training from scratch, it is further important to compare against methods that calibrate or adjust scaling factors rather than require pre-training new models from scratch as well.

### Questions
- While the methodology differs, I believe that the underlying phenomena that the authors discuss relates significantly with [1]. In particular, ensuring that the low frequency dimensions complete a rotation is quite related to ensuring that longer relative distances does not appear OOD relative to what was learned during training.

- Based on my understanding of the pseudocode in Appendix A.3, the scaled dimensions are hard-coded as the first quarter and third quarter of the dimensions. Why is this the case and wouldn't it make more sense to apply a more adaptive scaling method that looks at all the dimensions instead and choose a proportion of the dimensions to scale based on the frequency values?

- While I'm not the largest advocate for scaling trends, given the inconsistency in the results the method would be more convincing if either results on larger models or data sizes were provided and showed a more clear trend of the advantages of RoPE-ID relative to existing methods.

[1] Resonance RoPE: Improving Context Length Generalization of Large Language Models

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
3

### Summary
This paper analyzes why Rotary Positional Embedding (RoPE) fails beyond its training context and attributes the issue to geometric degradation in attention. The authors show that key and query vectors form tight clusters and rely on a “sink” token for stability, but RoPE distorts this structure at long ranges. They propose RoPE-ID, which applies high-frequency RoPE to only part of the channels to preserve cluster geometry without fine-tuning. Results on 1B and 3B models show improved long-context performance on LongBench and RULER benchmarks compared to standard RoPE baselines.

### Strengths
Introduces a clear geometric explanation for RoPE’s long-context failures, supported by multiple complementary analyses.

Proposes a simple, low-cost remedy (RoPE-ID) that requires no fine-tuning, facilitating rapid deployment.

Demonstrates robust empirical evaluation across tasks and baselines and provides reproducibility details.

### Weaknesses
Theoretical analysis stops short of giving formal bounds that relate rotation frequency to cluster overlap or performance.

Experimental scope is limited to 1B/3B models and up to 16k context; applicability to 7B+ models or 100k+ contexts is not shown.

Key hyperparameter choices (channel ratio, cycle length, temperature coefficient) lack comprehensive ablation or principled justification.

Baseline comparisons are incomplete—direct controlled comparisons to Hope[1], and other recent methods under identical settings are missing.

[1] Chen, Y., Lv, A., Luan, J., Wang, B., & Liu, W. (2024). *HoPE: A novel positional encoding without long-term decay for enhanced context awareness and extrapolation*. arXiv preprint arXiv:2410.21216.

### Questions
an you define a quantitative “cluster overlap” metric (e.g., silhouette score, Davies-Bouldin index) and show its correlation with sink attention and downstream scores?

What is the empirical sensitivity to the RoPE channel ratio (e.g., 25%, 37.5%, 62.5%, 75%) and how does the trade-off between positional fidelity and cluster preservation behave?

Please provide a direct empirical comparison to Hope under identical experimental settings to clarify the regimes where RoPE-ID’s training-free advantage holds.


*(Optional) Does RoPE-ID scale to 7B+ models and much longer contexts (100k–1M tokens)? How do key/query cluster properties change with model scale?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
Transformers with RoPE fail on inputs longer than training length. The paper observes that keys and queries form separated clusters, enabling "sink tokens" to absorb default attention. RoPE's low-frequency channels rotate slowly, barely moving during training. Beyond training length, they reach "out of distribution" rotation angles, causing clusters to overlap and breaking sink token functionality. RoPE-ID applies high-frequency RoPE to half the channels (completing full rotations during training), keeping the other half stable. This maintains cluster separation while staying in-distribution, enabling 16× context generalization without fine-tuning.

### Strengths
- Overall, thorough analysis, then simple solution. Awesome! Great Science!
- Reveals that attention uses separated key/query clusters (opposite of conventional wisdom), connects RoPE mechanics, attention geometry, and sink tokens into one elegant explanation, slow-rotating channels reach unseen angles beyond training length, destroying cluster separation
- Zero-shot method that matches or exceeds YARN. RoPE-ID is basically "use high frequencies on half the channels", trained on 4k, works on 64k with no tuning, conceptually cleaner than previous method
- Super great presentation. Clear and meaningful figures, great writing. Makes complex geometry intuitive

### Weaknesses
- The last paragraph of the intro is hard to parse, though easy to understand after reading the paper. 
- The tables could emphasize more that RoPE-ID is zero shot.

### Questions
- In Figure 3, could you align the scaling of the x and y axis within the 4k and the 64k lengths? 
- Could you add to the tables what the numbers are? Maybe the unit?
- Why alternating quarters? I understood that 1 and 3 are rotated, leaves 2 and 4 stable. Why not just first half and second half? Or more finegrained interleaving? Or could simpler patterns work just as well?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a geometric analysis of Rotary Positional Embeddings (RoPE) to explain long-context failure modes in Transformers. The authors find that RoPE induces tightly clustered query/key point clouds within training length, supporting the formation of sink tokens that prevent over-mixing. However, when input length exceeds training context, RoPE disperses and overlaps these clusters, disabling sink token functionality and causing degraded attention behavior.
Building on this insight, the paper proposes RoPE-ID (In-Distribution), a modification that applies RoPE with high frequency to only a subset of channels, maintaining cluster separation while preserving position information. Experiments on LongBench and RULER with 1B- and 3B-parameter Transformers show that RoPE-ID achieves comparable or superior long-context generalization to tuning-free baselines such as YaRN, without fine-tuning.

### Strengths
1. The geometric interpretation of RoPE and attention behavior is original, intuitive, and interesting, linking positional encoding, cluster geometry, and sink token dynamics into a unified framework.

2. The paper validates hypotheses through detailed analyses (PCA projections, singular value decomposition, attention maps) and replicates findings across multiple LLM families (LLaMA, Gemma, Olmo).

3. Figures (e.g., cluster diagrams and singular-value ratios) effectively convey geometric intuition and support claims.

4. Addresses a critical bottleneck in scaling LLMs to longer contexts, thus of broad interest to both researchers and practitioners.

### Weaknesses
1. While the geometric intuition is appealing, the paper lacks a mathematical analysis of how RoPE’s rotation frequencies affect cluster stability. The mechanism by which RoPE transforms i.i.d. token embeddings into clustered structures and subsequently causes cluster dispersion under out-of-distribution (OOD) conditions remains insufficiently explained.

2. Figure 2 analyzes cosine similarities, yet attention operates on dot products. This mismatch raises concerns about whether the reported geometric observations truly reflect the behavior of the attention mechanism.

3. The so-called “Unified Theory of RoPE Attention” Section 3.2.2 primarily summarizes prior empirical findings rather than providing a formal theoretical analysis. The term “theory” may therefore be overstated.

4. The proposed RoPE-ID method, which applies RoPE to only half of the channels using higher frequencies, does not convincingly demonstrate that it can effectively mitigate the identified issues. The paper does not clearly explain why this configuration mitigates key/query dispersion, and the presented results suggest the phenomenon is not fully resolved. Moreover, the authors should further analyze the trade-off that higher frequencies blur local positional distinctions.

5. The authors should include ablation studies on the proportion of channels to which RoPE is applied and on the chosen frequency range parameters, to better justify these design choices.

6. Figure 5 lacks proper x-axis labeling, and the notion of “time” used in the caption is ambiguous. Additionally, many figures appear visually unpolished and would benefit from clearer annotations, consistent axis scaling, and improved aesthetics to enhance interpretability.

### Questions
In Figure 7, could the authors clarify the comparison between the configuration with $\theta = 500k$ and the RoPE-ID variant? The plot seems to indicate that the $\theta = 500k$ baseline achieves a higher FSV ratio, implying stronger cluster preservation. Since a smaller $\theta$ corresponds to higher-frequency rotations, why does the high-frequency (orange) curve yield the lowest FSV ratio? This appears to contradict the authors’ claim that higher frequencies should maintain cluster stability.

### Soundness
3

### Presentation
1

### Contribution
3
