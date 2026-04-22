# Latent-to-Data Cascaded Diffusion Models for Unconditional Time Series Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Synthetic time series generation (TSG) is crucial for applications such as privacy preservation, data augmentation, and anomaly detection. A key challenge in TSG lies in modeling the multi-modal distributions of time series, which requires simultaneously capturing diverse high-level representation distributions and preserving local temporal fidelity. Most existing diffusion models, however, are constrained by their single-space focus: latent-space models capture representation distributions but often compromise local fidelity, while data-space models preserve local details in the data space but struggle to learn high-level representations essential for multi-modal time series. To address these limitations, we propose L2D-Diff, a dual-space diffusion framework for synthetic time series generation. Specifically, L2D-Diff first compresses input sequences into a latent space to efficiently model the distribution of time series representations. The distribution then guides a data-space diffusion model to refine local data details, enabling faithful generation of time series distribution without relying on external conditions. Experiments on both single-modal and multi-modal datasets demonstrate the effectiveness of L2D-Diff in tackling unconditional TSG tasks. Ablation studies further highlight the necessity and impact of its dual-space design, showcasing its capability to achieve representation coherence and local fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes L2D-diff, a hybrid latent and data space diffusion model that aims to capture both high-level and local fidelity. The generation process first uses latent diffusion, then passes onto data space diffusion. Through empirical experiment, L2D-Diff is on par with previous SOTA models.

### Strengths
1. This paper investigates the hybrid diffusion models for time series generation, which is a new framework.
2. This paper conducted extensive experiments with newly added datasets.
3. This paper is well written and easy to follow.

### Weaknesses
1. The L2D-Diff consists of multiple modules. A comprehensive ablation study is needed to justify the hybrid framework.
2. While there are discussions of the training & inference, the discussion is limited only to a new dataset. The efficiency claim could be more compelling if the comparison can also be conducted over the sine, stock, and energy, which are more commonly compared.
3. The t-SNE visualization is only performed over one dataset. It would be ideal to include more datasets.

### Questions
Additional questions other than weaknesses.

1. In Table 7, the result for the previous SOTA methods, Diffusion-TS, seems to be very different from the original paper. 
2. Newly proposed datasets, 5 out of 8 show 0 for DS score. Does that mean those datasets are too easy?

### Soundness
2

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
3

### Summary
The paper proposes L2D-Diff, a cascaded diffusion framework for unconditional time-series generation. The method first learns a latent diffusion model to capture global structure and then conditions a data-space diffusion model to refine local temporal details. Experiments are performed on multiple benchmark datasets, showing improved Contextual-FID and distributional similarity metrics over existing baselines.

### Strengths
1. Timely and relevant topic.
    - Time-series generation via diffusion is an emerging area, and exploring multi-stage or cascaded architectures is a meaningful direction.
2. Conceptually intuitive design.
   - The latent-to-data cascade nicely bridges global representation learning and fine-grained sample refinement — an idea that is simple yet effective.
3. Comprehensive experiments.
   - The paper evaluates on a wide range of datasets and metrics, which shows the authors’ effort to ensure empirical robustness.
4. Clarity of motivation.
   - The motivation for combining latent and data-space diffusion stages is clearly articulated and easy to follow.

### Weaknesses
1. Limited Theoretical Depth
-  While the cascaded design is intuitive, the paper lacks a clear theoretical or empirical justification for why conditioning a data-space diffusion on latent representations leads to better generation quality.
-  An analysis of information flow, representation alignment, or consistency between latent and data spaces could make the contribution more convincing.

2. Insufficient Experimental Analysis
- The experiments show promising improvements but lack in-depth ablation and diagnostic studies.
Suggested additions include:
  - Varying latent dimension and conditioning strength
  - Comparing against single-stage models of similar parameter counts
  - Evaluating temporal coherence and downstream task performance (e.g., classification or forecasting accuracy using generated samples)

3. Incomplete Architectural Details
  -  The conditioning mechanism between the latent and data diffusion stages is not well specified.
  -  Clarify how latent vectors are integrated (e.g., concatenation, cross-attention, feature modulation) and include a schematic diagram or pseudo-code for reproducibility.

4. Broader Context and Positioning
  -  The paper could more clearly state what aspects are unique to time-series generation compared to existing cascaded diffusion frameworks in other modalities (e.g., images or audio).
  -  Discussing challenges like temporal consistency, irregular sampling, or multi-channel correlations would better highlight the domain-specific contribution.

### Questions
Please see the weakness above.

### Soundness
2

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
4

### Summary
This paper introduces L2D-Diff, a cascaded diffusion framework for unconditional multivariate time series generation. The approach combines latent-space diffusion with a data-space diffusion process. The latent representations generated by the first stage condition the data-space diffusion in the second stage via a learned conditioning network. Empirical results demonstrate SOTA or near-SOTA performance across standard quantitative metrics, with ablation and efficiency analyses supporting the value of the dual-branch architecture.

### Strengths
Unconditional high-fidelity time series generation, especially in multi-modal settings, is a non-trivial challenge with clear applications in data augmentation and privacy. The dual-branch latent-to-data design is visually and mathematically well documented, intuitively linking high-level semantic representation (latent) with local fidelity (data space).

The experiments demonstrate consistently superior (lower) FID on challenging benchmarks, with standout results on multi-modal datasets.

### Weaknesses
**The theoretical insight and novelty (theoretically) is limited**. The derivation and formulation of the cascaded diffusion process in Section 3 are correct but mostly extend existing latent/diffusion model frameworks without introducing substantial new theoretical principles. The dual-branch connection, while effective, does not offer a formal analysis (e.g., convergence or expressivity gains) that might help explain or generalize its empirical success.

While the conditioning mechanism is described, the exact integration between latent embeddings and data-space denoising is vague: e.g., is $\mathcal{F}$ simply concatenation, a form of cross-attention, or more? There is a lack of ablation or comparative analysis of alternative conditioning schemes, which could be crucial given the model’s reliance on this interface. 

The paper omits explicit experimental or cited discussion of the recent and very relevant work, the T2S model. 

Ge, Yunfeng, et al. "T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models." Accepted by the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)

The paper’s main claim is a latent‑to‑data cascaded diffusion setup for unconditional time‑series generation: first sample a global latent with a latent diffusion model, then refine in data space conditioned on that sampled latent. The authors position this as the first such cascade for time series, and they contrast with prior “single‑space” models. I would call it **moderately novel**. Conceptually, it’s a domain‑appropriate consolidation of known ingredients, pretrained TS2Vec encoder, latent diffusion, and conditional data‑space denoising, with a simple conditioning interface. There isn’t claimed theoretical novelty, and the framework is close in spirit to representation‑conditioned generation in vision but re‑targeted to time series with an end‑to‑end dual‑branch training recipe.

### Questions
Can the authors clarify the precise architecture and operation of the conditioning network $\mathcal{F}$? Is it simply a fixed CNN, or does it use more advanced attention or cross-modal mechanisms, and why was this choice made?

How does the proposed framework handle very long or highly irregular time series, given that the encoder (TS2Vec) is fixed in dimension?

Could the authors provide head-to-head results against the T2S model or other missing recent baselines, or explain why such comparisons are omitted?

Can the authors clarify the process for baseline adaptation, especially for methods originally focusing on conditional or autoregressive tasks?

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
This paper introduces L2D-Diff, a cascaded dual-branch diffusion framework for unconditional synthetic time series generation. The proposed model first learns high-level representation distributions by operating a diffusion process in a compressed latent space, then refines the generated time series in the data space by conditioning on these latent codes. The authors claim this approach offers a balance between preserving global semantics and capturing local fidelity, a challenge that existing methods struggle with individually. The method is evaluated on 11 diverse single- and multi-modal datasets, with ablations, efficiency analyses, and qualitative/quantitative results provided.

### Strengths
The paper’s strengths, I believe, lie in its dual-space cascaded architecture that combines latent- and data-space diffusion stages through a dedicated conditioning network, its strong empirical performance consistently surpassing diverse baselines across multiple datasets, and its comprehensive evaluation covering both standard and complex multimodal time-series benchmarks with supporting qualitative analyses. The work demonstrates mathematical clarity through detailed formalizations, loss functions, and architectural descriptions, while also achieving good parameter and computational efficiency. Moreover, the ablation studies carefully validate the importance of the dual-space design across varying data complexities.

### Weaknesses
- Although the cascaded dual-branch approach is well-motivated, key aspects of latent-to-data bridging have seen preliminary exploration in other domains (see Section 1 where RCG and EDDPM mentioned). The novelty for specifically time series scenario is asserted, but the adaptation from images/graphs to time series follows a relatively straightforward path (representation learning, conditional denoising). The authors could do more to clarify precisely which challenges for time series are not addressed by prior multi-stage diffusion frameworks, ideally with more rigorous empirical or theoretical separation.

- Related to my first point, there is no formal theoretical analysis (e.g., convergence, identifiability, or information bottleneck trade-off) of why the cascaded approach works or how the latent encoding impacts downstream fidelity or diversity. For example, there is no proof or lemma quantifying what local/global information is lost or gained across the two denoising stages, nor any characterization of failure cases or limits of the method.

- In section 3.2, the conditioning network $\mathcal{F}$ is stated to be a CNN of "5 layers by default", but there is little justification or sensitivity analysis of its depth, efficacy, or alternative forms (e.g., attention, MLP, graph). Choices here can majorly affect expressivity, especially in bridging global latents to local series. Additionally, ablation on this module's architecture/hyper-parameters is also recommneded.

- The latent dimension $d$ is chosen by default as 8 "based on prior works". However, there is minimal discussion of how this dimension influences the information bottleneck, potential tradeoffs in capacity (e.g., underfitting, overfitting), or the sensitivity of results to $d$, beyond a brief empirical study. A deeper analysis (possibly via information-theoretic measures or error bounds) would clarify how much key information is recoverable from latent diffusion before refinement, and under which conditions the cascaded architecture may fail.

- I happen to know some highly relevant recent diffusion models (such as [1,2]) tailored for time series, which is missing in this work. I recommend author(s) to conduct a careful add-on survey to fill this gap. 

- [Minor suggestion]: some supplementary content (Appendices C, D) provides background or repeats points already made in the main text, which occasionally diffuses focus from key innovations/limitations.

**Refs:**

[1] Ge et al., T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models

[2] Sikder et al., TransFusion: Generating long, high fidelity time series using diffusion models with transformers

### Questions
While Table 2 and Figure 4/5 show improvement on complex datasets (e.g., Character Trajectories), the text primarily asserts these methods struggle on multimodal data, without actually drilling into when/why. For example, how does L2D-Diff perform on classes with small sample sizes or rare modes? Does it suffer from mode-collapse-like phenomena, or does the latent-to-data pipeline truly enable consistent coverage?

### Soundness
3

### Presentation
3

### Contribution
2
