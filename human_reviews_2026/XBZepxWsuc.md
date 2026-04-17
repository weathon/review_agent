# From Two to One: Harmonizing Attention and Feature Debiasing for Multivariate Time Series Forecasting

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Multivariate time series forecasting (MTSF) models based on Transformers have shown remarkable success in various applications, such as energy management, weather forecasting, and traffic monitoring.
However, due to the complex and intertwined correlations among variates, Transformer-based methods often fail to precisely model the interactions among series, leading to limited performance improvement.
In this paper, we rigorously investigate and establish the phenomenon of feature oversmoothing in Transformer-based forecasters through a theoretical analysis.
To this end, we then propose \textbf{FADformer}, a frequency-aware debiasing framework, which harmonizes the low- and high-frequency components of attention and feature maps to capture fine-grained patterns for accurate forecasting.
Specifically, we design two plug-and-play modules using the Fourier transformation, where i) AttnDeb rescales high-frequency weights within attention modules to mitigate the low-pass limitation and ii) FeatDeb injects inductive feature bias into residual connections to amplify the important high-frequency signals.
Extensive experiments on challenging real-world datasets show the superiority of our FADformer over existing state-of-the-art methods, in terms of both forecasting performance and generalization ability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel attention mecanism for transformer for time series forecasting. Motivated by the empirical observation that attention oversmooth frequences, i.e., acts as a low-frequency filter which might hinder performance, FADformer is proposed with a frequency-aware debiasing module to preserve all the information for forecasting. Large scale experiments are conducted showing the improvement brought by FADformer on common time series forecasting benchmarks.

### Strengths
- Experiments are comprehensive and showcase the benefits of the approach
- The ETTh2 analysis is simple yet intuitive to show the low-frequency filtering
- The theoretical analysis is sound
- The proposed approach is shown to be effective over a wide range of benchmarks and a large ablation study is conducted to confirm its robustness

### Weaknesses
I list below what I believe are weaknesses but I would be happy to get corrected if I misunderstood some parts of the work.

- The observed filtering pattern seems to be related to rank collapse which has been theoretically and empirically studied in prior works [1, 2, 3]. I believe those are important work that are not discussed in the current paper.
- Notably, Thm 3.2 seems very close to [1, section 2.2] which is not cited.
- In particular, in [3], the authors study the rank collapse in transformer based models for time series forecasting, and propose using a sharpness-aware optimizer to solve the issue. It would be interesting to add this model as a baseline or at least discuss it given that the proposed approach solves a similar issue (oversmoothing / filtering). 

Overall, the proposed approach is interesting and the results showcase its benefits however, there is missing works to be discussed for a better positioning of the paper in the literature.

*References*

[1] Dong et al. Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth. ICML 2021

[2] Noci et al. Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse. NeurIPS 2022

[3] Ilbert et al. SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel Wise Attention. ICML 2024ecasting

### Questions
- How does the proposed approach scale with the increase in sequence length and/or horizon?
- In definition 3.1, multivariate time series are described as independent channels however in practice the features can be correlated (otherwise there would be no need to do multivariate forecasting). Could the authors please elaborate on that?

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
5

### Summary
This paper proposes FADformer, a Transformer-based forecasting model that aims to tackle oversmoonthing and frequency learning bias issue in Transformers (which is introduced in Fredformer, KDD24). The work introduces two plug-in debiasing modules: (i) AttnDeb, which rescales high-frequency attention responses to mitigate the low-pass filtering nature of self-attention, and (ii) FeatDeb, which re-amplifies high-frequency signals in the residual connections to alleviate feature degeneration. The method achieves performance gains on 13 MTSF benchmarks.

### Strengths
Fair feature learning in the frequency domain is an important research topic in time series forecasting, and the idea of addressing oversmoothing is intuitively appealing.

The paper is easy to follow and the empirical evaluation is extensive.

### Weaknesses
While the paper presents frequency bias as an important observation, very recent works such as Fredformer (KDD’24) and FilterNet (NeurIPS’24) already address selective amplification or reweighting of high-frequency components in Transformers. Also, works like FreDF (ICLR’25) discussed the frequency modeling in the forecasting task. The introduction and related works overlaps with their motivation narratives, but these works and technical differences with them are not sufficiently discussed or contrasted. This makes the motivation feel partially rediscovered rather than newly formulated. In general, the motivation is unclear and seems like this is an incremental work.

The oversmoothing issue here is closely tied to the spectral imbalance story already explored in the above frequency-aware papers. it is unclear what is fundamentally new compared to prior FFT-based decomposition + reweighting strategies. I remember Fredformer already proposed this fft-ifft backbone with frequency decomposition learning. What are the technically new solution or contributions in this paper?

The theoretical section argues that effective rank can mitigate degeneracy, but the proposed method relies on FFT-based re-scaling, not directly on the theoretical update rule. The theory supports residual scaling in the abstract, but does not explain why a Gaussian decomposition for attention or a Top-K decomposition for features is the correct or optimal instantiation. The conceptual link between Proposition 3.4 and the implemented modules remains loose. Sometimes Top-K is an empirical way that cannot ensure the selection is always satisfied and easily influenced by noise. How to evaluate its effectiveness?

While the authors acknowledge several frequency-domain modeling methods in the introduction, the experimental baselines do not include any of these frequency modeling methods. Most comparisons are made only against common time-domain models (e.g., iTransformer, PatchTST), which directly conflicts with the paper’s claim that time-domain modeling is insufficient. Given that the proposed motivation closely aligns with Fredformer, including at least it or more representative frequency modeling baselines is essential for a fair and convincing evaluation.
Moreover, a deeper ablation (e.g., per-frequency reconstruction error, variance of gradients across layers) would help clarify the real causal effect.

### Questions
Please kindly refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents FADformer, a Transformer-based framework that introduces two frequency-aware debiasing modules (AttnDeb and FeatDeb) to mitigate oversmoothing in multivariate time series forecasting. The method combines Fourier-based reweighting of attention and feature components and is supported by a theoretical discussion using effective rank analysis.

### Strengths
1. The paper is generally well organized, making it easy to follow.
2. The topic of addressing frequency bias and oversmoothing in Transformer-based time series models is timely and aligns with current trends in time-series representation learning.

### Weaknesses
1. In the Related Work section, the paper lacks an in-depth discussion of existing studies on closely related topics, such as Fredformer and Amplifier. Although the authors briefly mention Fredformer, they fail to provide a clear and insightful comparative analysis. As for Amplifier, it is not mentioned at all, which reflects an insufficient literature review and a lack of thorough investigation into this topic by the authors.

2. This paper belongs to the category of frequency-domain models, yet the experimental section lacks comparisons with other frequency-domain baselines.
3. Regarding Table 2, the performance improvement brought by Debiasing is not significant.
4. Line 418: the reference to Table 5 is incorrect; it should be Table 4.
5. Lines 427–428: In the statement “First-K defines the first K lowest elements of the Fourier transform as low-frequency components of features,” — it is unclear what “lowest” specifically refers to.

### Questions
1、Comments on Figure 1:
- (1) The upper figure of Figure 1(a) does not provide any meaningful insight.
- (2) The phenomena illustrated by the lower two subfigures of Figure 1(a) have already been investigated in the Amplifier[1] paper.
- (3) Is the situation shown in the lower two subfigures of Figure 1(a) exclusively caused by the self-attention mechanism?
- (4) For Figure 1(b), please clarify which Transformer-based forecaster was used in the visualization experiment.
- (5) The conclusion “Figure 1(b) suggests that the correlations predicted by Transformer-based forecasters are mainly concentrated on and near the diagonal, where there is a substantial portion of the low-frequency characteristics” does not make sense: First, the correlations on the diagonal are self-correlations (a variable with itself), which are always equal to 1.000 and thus irrelevant to the topic discussed in this paper. Second, the claim that the correlations are near the diagonal cannot be reasonably inferred from the figure.

2、Line 016–017: “Transformer-based methods often fail to precisely model the interactions among series” — What is the specific experimental or theoretical evidence supporting this statement?

3、Definition 3.3 (Effective Rank) appears to be a direct copy of Definition 3.1 (Effective Rank) from CONTRANORM[2] (ICLR 2023). Is such a practice acceptable? Similarly, Equation (4) in this paper is almost identical to Equation (8) in CONTRANORM, and Proposition 3.4 closely resembles Proposition 1 from the same work. These similarities raise serious concerns about the theoretical contribution and originality of this paper.

4、Regarding Figure 3 (The Architecture of FADformer), I have two questions:
- (1)	AttnDeb separates the attention map into low-frequency and high-frequency components. In FADformer, should other neural network components—such as Linear layers or MLP modules—also undergo a similar separation into low- and high-frequency parts?
- (2)	FeatDeb obtains low- and high-frequency components through spectral truncation. Why doesn’t AttnDeb adopt this straightforward and intuitive approach as well?


[1] Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting (AAAI 2025)

[2] CONTRANORM: A CONTRASTIVE LEARNING PERSPECTIVE ON OVERSMOOTHING AND BEYOND (ICLR 2023)

### Soundness
2

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
The paper proposes a single module that jointly models **temporal** and **channel** dependencies for multivariate forecasting, arguing that separating the two often causes redundancy and information loss. The method replaces dual-path designs with a **unified attention block** and reports moderate gains on ETT, Weather, Exchange, and Electricity. The idea is sensible, and the writing is clean. However, the **evaluation is too narrow**: several **relevant recent baselines are missing**.

### Strengths
- **Clear motivation**: unifying time and channel modeling is a reasonable direction that practitioners care about.  
- **Simple design**: a single joint-attention block is easier to maintain than two specialized modules.  
- **Readable paper**: notation and figures are tidy, ablations exist (though shallow).

### Weaknesses
1. **Baseline coverage is insufficient.**  
   The paper compares with older Transformer variants (e.g., Autoformer, FEDformer, iTransformer) but **omits recent and directly relevant methods** such as **TSMixer** (lightweight mixing),  **TimeMixer** (explicit time–channel coupling), and **FreTS** (frequency modeling).  Since the central claim is “harmonizing” temporal and channel modeling, these baselines are necessary to establish empirical credibility.

2. **Gains are small and may fall within variance.**  
   Many improvements over reported baselines are <1%. Without confidence intervals or repeated runs, the strength of the claim is hard to judge.

3. **Ablation depth and diagnostics.**  
   We see an on/off ablation for the joint-attention block, but there is little analysis of *why* it helps (e.g., attention maps, redundancy metrics across axes, or representation overlap).

4. **Efficiency is asserted, not demonstrated.**  
   If the unified block is proposed as a simpler/faster alternative, please add wall-clock time, memory, and parameter counts versus strong baselines.

5. **Scalability and stress tests.**  
   Results stop at mid-scale datasets. High-dimensional settings (200+ variables), long horizons (e.g., 720+), or missing/irregular sampling would make the story more convincing.

### Questions
It would be valuable to add more baseline comparisons and related analysis, especially with recent models addressing similar temporal–channel interactions.

Are the reported gains consistent and statistically reliable across multiple random seeds?

Since the paper emphasizes simplicity and efficiency, could you include runtime, memory, or parameter comparisons?

Do the attention maps indicate reduced redundancy between temporal and channel dimensions?

How does performance vary with sequence length or the number of variables?

### Soundness
2

### Presentation
3

### Contribution
2
