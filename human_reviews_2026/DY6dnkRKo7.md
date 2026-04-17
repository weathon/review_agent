# Winformer: Transcending pairwise similarity for time-series generation

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Time-series generation plays a critical role in data imputation, feature augmentation, domain adaptation, and foundation modeling. However, the cross-domain generation remains a persistent challenge, as existing methods model time-series interactions either at the granularity of individual points or fragmented segments. This limits their ability to capture and adapt to complex periodic patterns inherent in diverse domains. Specifically, point-wise attention struggles with long-range dependencies, while standard patch-based approaches may break important cyclical structures. To address this, we introduce Winformer, a novel diffusion model framework built on a window-wise Transformer. We shift the fundamental processing unit in the attention mechanism from pairwise points similarity to continuous windows comparison of the entire horizon. By operating on semantically richer window representations, the proposed approach effectively learns and transfers complex periodic patterns across domains. Extensive experiments on 12 real-world datasets demonstrate Winformer's effectiveness, achieving an average performance gain of 10.67% over SOTA baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focus on time series generation problem. By leveraging the Transformer and diffusion framework, a model termed Winformer is proposed by  leveraging the window-wise attention to enhance the learning and generation of time series. Experiments demonstrate performance improvement on 12 real-world .datasets

### Strengths
1. well-structured, clear presentation and easy-to-follow.

2. impressive experimental results.

### Weaknesses
1. According to Figure 1, the difference between point-wise, patch-wise, and window-wise operations can be regarded as convolutions with different kernel sizes and strides. Specifically, setting the stride smaller than the kernel size could also model "middle-level interactions". Also, the patch operation has complexity advantages (as it needs fewer tokens for attention calculation). This needs more discussion and analysis. 

2. Why choose Maximum Mean Discrepancy (MMD) and Kullback-Leibler Divergence (K-L) for experimental evaluation? How about other metrics?

3. According to Figure 4, it seems TimeD better captures some high-frequency patterns, while Winformer captures the overall coarse-grained temporal variations.

### Questions
please refer to the weakness.

### Soundness
3

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
This paper introduces Winformer, a diffusion-based Transformer for cross-domain time-series generation. It proposes the Ample Attention, which extends point-wise similarity to window-wise comparison in the frequency domain, enabling better modeling of periodic and long-range dependencies. Experiments on 12 datasets show consistent improvements, with an average 10.67% gain over strong baselines.

### Strengths
- The proposed attention mechanism is conceptually novel and addresses a clear limitation of pairwise similarity in Transformers.

- The integration of window-wise attention into a diffusion framework is technically sound and well motivated.

- Strong empirical results and ablation studies demonstrate robustness and broad applicability across domains.

### Weaknesses
The paper focuses exclusively on evaluating generative quality (MMD, visual consistency) but lacks experiments on downstream utility. Demonstrating whether the generated data can enhance forecasting or imputation would significantly strengthen the practical impact of the proposed model.

### Questions
See weaknesses.

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
3

### Summary
This paper proposes Winformer, a diffusion-based model for time-series generation, particularly in cross-domain settings. It introduces "Ample attention," a window-wise attention mechanism that extends traditional pairwise similarity computations to comparisons between sliding windows of time-series data, leveraging Fourier transforms to capture periodic patterns and mitigate time-warping issues. The framework is built on a Transformer architecture tailored for denoising in diffusion probabilistic models. The authors claim this approach better handles complex trends, periods, and noise across domains compared to point-wise or patch-wise methods.

### Strengths
1. The window-wise attention is an interesting extension of standard self-attention, potentially better suited for time-series data with inherent periodicity and warping, as demonstrated through Fourier-based derivations.

2. The focus on cross-domain generation addresses a challenging problem relevant to applications like data imputation and domain adaptation.

3. Theoretical insights (e.g., derivations in Section 3.1) provide some grounding for the proposed Ample attention, with promises of further proofs in the appendix.

### Weaknesses
1. Limited Scope of Applicability: While the paper motivates time-series generation as a foundational step for broader applications (e.g., imputation, feature augmentation, domain adaptation, and foundation modeling), the proposed methodology appears tightly coupled to diffusion-based generation tasks, particularly imputation via denoising. The Ample attention and window-wise processing are innovative for capturing periodic patterns in generative settings, but the paper does not demonstrate or discuss their utility in other core time-series tasks, such as forecasting, anomaly detection, classification, or clustering. For instance, it's unclear how Winformer would integrate into non-diffusion pipelines, where attention mechanisms typically operate differently. This narrow evaluation raises questions about the generalizability of the contributions to the wider time-series field, potentially limiting its impact beyond specialized generation scenarios.

2. Methodological Novelty and Integration: The Ample attention builds on established ideas like Fourier decompositions (e.g., from Alaa et al., 2021) and window-based processing, but the expansion to window-to-window alignments via learnable convolutions feels somewhat incremental rather than transformative. The overall architecture (e.g., combining diffusion with Transformer layers) echoes recent works like CSDI or TimeDiff, and it's not evident how the window-wise shift uniquely resolves limitations in pairwise or patch-wise approaches beyond empirical gains. Additionally, hyperparameters like window size (p) and stride (s) could introduce sensitivity, but without thorough ablations, their robustness is unclear.

3. Experimental Validation: The claimed average improvement is promising, but the experiments seem confined to cross-domain generation metrics, without extensions to downstream tasks. .The 12 datasets are diverse, but details on domain shifts or failure cases are limited in the provided sections. Cross-domain focus is emphasized, but without metrics like domain discrepancy measures.

4. The writing is clear but could better visualize the attention expansion process beyond Figures 1-2.

### Questions
1. Beyond diffusion-based imputation, how could Ample attention be adapted for other time-series tasks like forecasting or classification? Have you experimented with integrating it into non-generative models?

2. How sensitive is the model to window size (p) and stride (s)? Ablations on these would help.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the problem in existing cross-domain generation method that failing to capture and adapt to complex periodic patterns of diverse domains. This paper presents Winformer, which calculates attention scores between windows instead of between individual points, to enhance perception of complex time series patterns. The proposed window-wise attention mechanism, called Ample attention, is introduced as calculating similarity score between the Discrete Fourier Transform results of each window and is implemented by convolution on original attention scores. Experiments are conducted on 12 real-world time series datasets, in comparison with six time series generation baselines on two metrics. The ablation studies show the advantages of the proposed mechanism and discussions show how periodicity is captured and utilized by Winformer.

### Strengths
1. This paper targets on improving of attention mechanism of time series modeling, which is not only meaningful for cross-domain time series generation task, but also potentially beneficial for time series forecasting or other applications.
2. The ample attention is reasonably designed, providing a new option for time series transformer that emphasizes window-by-window correlations. 
3. The experiments are extensive and comprehensive, covering multiple datasets and baselines. The discussions also help understand the advantage of the proposed Winformer method.

### Weaknesses
1. Many sentences in this paper are incoherent and difficult to follow. It feels like it hasn't been proofread for clarity. Some syntactic refinement would greatly improve it. There is also a structuring problem, i.e. missing a conclusion sector. 
2. The motivations for Winformer and some other claims are not well-supported. The authors state that they “find that a more adaptive architecture works better for coupling trending and periodic patterns”. But as a motivation, we would like to know about how is this architecture discovered and what is the theoretical intuition behind the design. Similarly, why cross-domain generation task is better for demonstrate the effectiveness, instead of regular/long-term time series forecasting task or time series imputation task, should be properly justified.
3. The convolution mechanism has introduced additional computation overhead, while the performance improvement against its point-wise attention variant is very significant. The paper can benefit from providing a more thorough analysis (e.g., visualized comparison of generated sequences) into the failure mode of window-wise, patch-wise and point-wise attention respectively, to show the advantage of Winformer more clearly, rather than just claiming “periodic feature in this dataset is weak” without justification.

### Questions
1. How do you describe the differences among point-wise, patch-wise and window-wise attention map, and how to explain these differences?
2. Are results sensitive to the convolution kernel initialization method? How do the kernel weights change along the training process?
3. As the results for different kernel sizes are close to each other, can the authors provide some general rules for selecting this hyper-parameter?

### Soundness
3

### Presentation
1

### Contribution
3
