# STORM: Synergistic Cross-Scale Spatio-Temporal Modeling for Weather Forecasting

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Accurate weather forecasting is crucial for climate research, disaster mitigation, and societal planning. Despite recent progress with deep learning, global atmospheric data remain uniquely challenging since weather dynamics evolve across heterogeneous spatial and temporal scales ranging from planetary circulations to localized phenomena. Capturing such cross-scale interactions within a unified framework remains an open problem.  To address this gap, we propose \textbf{STORM},  a spatio-temporal model that disentangles atmospheric variations into multiple scales to uncover scale-specific dependencies. In addition, it enables coherent forecasting across multiple resolutions, maintaining consistent temporal evolution. Experiments on benchmark datasets demonstrate that STORM consistently delivers superior performance across both global and regional settings, as well as for short- and long-term forecasts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The authors propose the STORM architecture for weather forecasting, designed to capture cross-scale interactions in multi-scale weather data. They identify three key limitations in the forecasting performance of deep learning–based approaches: multi-scale heterogeneity, diverse temporal evolution, and weak cross-scale interaction. Accordingly, they propose STORM to address these challenges.

### Strengths
- The authors effectively highlight the need for the proposed architecture, with a well-written introduction that builds up to it clearly. The related work section is up to date and provides a solid understanding of the current state of the art. The intuition behind all three main modules is also well explained.
- Section 5 presents comparisons with state-of-the-art baseline models from the literature on the ERA5 dataset across different scales. The authors provide both short-term and long-term forecasting results. Combined with the visualizations of predictions in the Appendix, the results appear convincing and satisfactory.
- Section 5.4 clearly demonstrates the importance of using multi-scale information for forecasting, presenting both numerical results and visualizations at different scales. The motivation for setting the number of scales to three is also clearly justified.
- Appendix B includes some theoretical analysis supporting the need for multi-scale modeling. However, it does not provide theory directly related to the proposed STORM architecture. Nonetheless, this section can still be considered a minor strength, as most recent baseline methods lack any theoretical component.

### Weaknesses
- The ablation study in Section 5.3 seems unsatisfying and weak. The STORM architecture has three key modules, and the authors report results by removing each one on the short-term global forecasting task (comparable to Table 1). Results are shown only in a small bar plot (Figure 6), not in a table. For variables such as T2M, U10, and V10, even the “w/o S&M” configuration appears to outperform baseline methods in Table 1 (losing only to Triton). This implies the T module alone would beat most baselines on some variables, which is unconvincing and undermines the need for all three modules separately. Why not provide tabular results for the ablation? And why restrict it to short-term global forecasting only?
- Furthermore, the results show the necessity of each module for T, M, and S, but they do not justify the specific architectural choices within each module. For example, we are given no evidence of how effective the Hierarchical Earth Embedder is at leveraging cross-scale information, we are simply asked to understand its value from the RMSE differences with and without it in Section 5.3. Appendix E at least reports some hyperparameters (M=3, D=256, N=3), but the specific design choices inside each module remain questionable: there is no ablation to support them, nor results under alternative hyperparameters. From Appendix E, it appears the default parameters were set intuitively and used across experiments, with only the training epochs adjusted via early stopping on validation data. Furthermore, the manuscript provides no analysis of the STORM architecture’s sensitivity to hyperparameter tuning.

### Questions
- Please refer to the weaknesses section for detailed comments. Most importantly, the ablation study in Section 5.3 appears weak as the reported results with and without the main modules seem questionable when compared to the baseline methods.
- Please correct the typos throughout the paper, even the title contains one.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes STORM, a cross-scale spatio-temporal framework for weather forecasting that disentangles atmospheric dynamics into multiple spatial scales and learns coherent temporal evolution. It includes a Hierarchical Earth Embedder, a Scale-Bridging Spatio-Temporal Encoder with fine-to-coarse messaging and lightweight temporal modeling, and a Level-Aligned Forecasting Decoder for multi-scale coherent outputs. Extensive ERA5 experiments show consistent SOTA performance across short- and long-range horizons with solid ablations and theoretical motivation.

### Strengths
1 Strong problem fit: explicitly addresses multi-scale heterogeneity and cross-scale interactions, a central challenge in weather prediction.

2 Clean, modular design: efficient hierarchical embeddings; ViT-style spatial encoder; simple but effective cross-scale messaging; lightweight temporal encoder; coherent multi-scale decoding.

3 Comprehensive results: consistent gains over strong baselines (Triton, Pangu, FCN, FuXi, SimVP, U-Net) across variables and horizons; long-range robustness is notable.

4 Clear ablations and scale analysis support design choices; theory gives intuitive generalization/optimization benefits.

### Weaknesses
1 Cross-scale interaction is one-way (fine→coarse) in the encoder; no analysis of bidirectional/gated messaging.

2 Temporal module is extremely lightweight; scalability to very long histories/periodic signals (MJO/ENSO) not deeply evaluated.

3 Efficiency reporting (FLOPs/throughput/memory vs. resolution/horizon) is limited in the main text.

### Questions
1 Can you report inference efficiency (FLOPs, latency, memory) versus baselines across resolutions/horizons?

2 Do you observe error hotspots by latitude/terrain, and does cross-scale modeling alleviate them?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces STORM, a synergistic cross-scale spatio-temporal framework for data-driven weather forecasting. The model explicitly disentangles atmospheric variations into multiple spatial and temporal scales through three key components: (i) a Hierarchical Earth Embedder that constructs multi-resolution representations, (ii) a Scale-Bridging Spatio-Temporal Encoder that integrates temporal evolution and cross-scale interactions, and (iii) a Level-Aligned Forecasting Decoder that generates coherent multi-scale predictions. Experiments on the ERA5 dataset show that STORM achieves state-of-the-art performance for both short-term (hours) and long-term (days) forecasts, outperforming strong baselines such as Pangu-Weather, FourCastNet, and GraphCast. Ablation and multi-scale analyses further demonstrate the contribution of each module and confirm the importance of explicit cross-scale modeling.

### Strengths
1. The paper presents a coherent multi-scale framework that systematically integrates spatial hierarchies and temporal evolution, providing a unified structure for global-to-local forecasting.

2. The evaluation is extensive, spanning multiple spatial resolutions and forecast horizons, and shows consistent superiority over strong baselines.

3. The paper is clearly written and well organized, with careful theoretical justification and reproducibility statements.

### Weaknesses
1. The core ideas, such as hierarchical representation, multi-resolution encoding, and cross-scale feature fusion, are largely extensions of prior models such as Pangu-Weather (3D hierarchical Transformer) and FourCastNet (multi-frequency operator). The contribution is primarily architectural refinement rather than a new modeling principle.

2. The paper does not quantify computational cost, efficiency, or scalability compared with existing models.

### Questions
1. How does the computational efficiency (training time, FLOPs, memory) compare with Pangu-Weather and FourCastNet at similar resolutions?

2. Could the hierarchical design be combined with physics-informed constraints or hybrid NWP–DL methods for better interpretability?

3. How sensitive is the model to the number of scales and the stride settings in the hierarchical embedder?

4. Does STORM maintain robustness when applied to unseen years, extreme events, or real-time operational data?

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
5

### Summary
This paper proposes STORM, a spatio-temporal deep learning framework for weather forecasting that explicitly models cross-scale interactions between coarse global circulations and fine regional dynamics. The model introduces a hierarchical encoder–decoder design with cross-scale message passing to enable information flow across multiple spatial resolutions. Experiments on ERA5 demonstrate consistent improvements over recent strong baselines (GraphCast, FuXi, Pangu-Weather, Triton) in both short-term (24 h) and long-term (10 day) forecasts. The results are strong and empirically well-presented.

### Strengths
1. Clear motivation and structure – The paper is well-motivated, addressing the nontrivial challenge of multi-scale atmospheric dynamics with a coherent architectural design.

2. Strong experimental performance – STORM achieves notable RMSE/ACC gains over multiple competitive baselines across diverse forecasting horizons and regions.

3. Comprehensive evaluation – Global, continental, and regional experiments are conducted with detailed visual and quantitative analysis.

### Weaknesses
1. Unsubstantiated “parameter-efficient” claim
The paper repeatedly emphasizes that STORM is parameter-efficient, yet it provides no quantitative evidence such as parameter counts, FLOPs, or inference latency. Given that the architecture involves multiple scale-specific branches and cross-scale message passing, the model design resembles a multi-branch or MoE-style structure that would typically increase parameters rather than reduce them.
Without compute-matched comparisons (e.g., same parameter budgets as GraphCast or FuXi), it is impossible to determine whether the reported gains stem from genuine architectural advantages or simply larger capacity.

2. Lack of quantitative analysis on multi-scale effects
While the paper stresses the importance of multi-scale synergy, it never analyzes how performance scales with the number of branches or levels. There is no experiment showing whether adding more scales continues to improve performance, when it saturates, or whether a simple increase in parallel branches (without explicit scale coupling) yields similar gains. Thus, the claimed “cross-scale synergy” remains empirically unverified.

3. Missing ablation for the proposed message-passing mechanism
Although the cross-scale message-passing block is presented as the core innovation, there is no isolated ablation showing how much it contributes to overall accuracy. This makes it difficult to assess whether the improvement originates from the proposed mechanism or from increased model complexity.

### Questions
1. Have you conducted any sensitivity analysis regarding the number of branches, stride size, or parameter count in the multi-scale architecture? If the model consistently maintains strong performance under such variations, I would significantly increase my score, as it would demonstrate real robustness rather than capacity-driven gains.

### Soundness
3

### Presentation
4

### Contribution
2
