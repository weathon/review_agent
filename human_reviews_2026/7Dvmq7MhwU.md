# TianQuan-S2S: A Subseasonal-to-Seasonal Global Weather Model via Incorporate Climatology State

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Accurate Subseasonal-to-Seasonal (S2S) forecasting is vital for decision-making in agriculture, energy production, and emergency management. However, it remains a challenging and underexplored problem due to the chaotic nature of the weather system. Recent data-driven studies have shown promising results, but their performance is limited by the inadequate incorporation of climate states and a model tendency to degrade, progressively losing fine-scale details and yielding over-smoothed forecasts. To overcome these limitations, we propose TianQuan-S2S, a global S2S forecasting model that integrates initial weather states with climatological means via incorporating climatology into patch embedding and enhancing variability capture through an uncertainty-augmented Transformer. Extensive experiments on the Earth Reanalysis 5 (ERA5) reanalysis dataset demonstrate that our model yields a significant improvement in both deterministic and ensemble forecasting over the climatology mean, traditional numerical methods, and data-driven models. Ablation studies empirically show the effectiveness of our model designs. Remarkably, our model outperforms skillful numerical ECMWF-S2S and advanced data-driven Fuxi-S2S in key meteorological variables. The code implementation can be found in https://github.com/zhangminglang42/TianQuan.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TianQuan-S2S, a novel deep learning model for subseasonal-to-seasonal (S2S) global weather forecasting. To address current data-driven approaches' limitations—namely, the inadequate modeling of climate states and the tendency for over-smoothed, less-detailed forecasts at long lead times—the authors introduce an architecture that fuses initial weather states and climatological means at the patch embedding stage. Furthermore, Gaussian noise (uncertainty blocks) is injected into every Transformer layer, enabling improved representation of variability and suppressing excessive smoothing. Extensive experiments on the ERA5 dataset show that TianQuan-S2S substantially outperforms both state-of-the-art numerical (ECMWF-S2S) and data-driven (FuXi-S2S, ClimaX) models, for both deterministic and ensemble forecasting. Ablation studies demonstrate that adding climatology and noise notably boosts performance. The paper also provides detailed comparisons of model structures and training/inference strategies.

### Strengths
The work systematically incorporates climatological mean information with initial state in feature representation for S2S forecasting and leverages uncertainty modeling, effectively addressing over-smoothing and detail loss in long-term prediction—achieving clear improvements over existing baselines with practical significance.

### Weaknesses
Although some ablation tests are presented, the paper only compares “without climatology” or “without noise” scenarios. It does not explore all combinations of climate, attention fusion, and noise modules (e.g., climatology + no noise, no climatology + with noise), nor does it analyze their effects on different variables, regions, or lead times.

The paper claims layer-wise Gaussian noise helps capture climate variability and improve uncertainty estimation, but gives neither a detailed theoretical justification nor a direct comparison to other uncertainty methods (such as MC Dropout, latent space perturbation). The effect of different injection scales and strategies and interpretability are also not discussed.

Most results focus on global average RMSE/ACC improvements. There is little discussion on model weaknesses or failures in complex terrain, extreme events, or other challenging conditions. This leaves the risk that claims are overgeneralized; more specific breakdowns are needed.

### Questions
Please provide more granular ablation studies covering all combinations of climatology, attention, and noise, broken down by variable, region, and lead time.

Please clarify the underlying mechanism by which Gaussian noise improves uncertainty modeling; compare with other uncertainty methods both theoretically and empirically, and provide some interpretability analysis.

Can you supplement analysis of performance under extreme events?

### Soundness
3

### Presentation
2

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
This paper proposes TianQuan-S2S, a global S2S forecasting model that pushes lead times out to 45 days. This paper highlights the importance of incorporating climatological information into patch embedding and an uncertainty-augmented blocks in terms of S2S forecasting. Experimental results on ERA5 reanalysis dataset and ablation studies demonstrate the validity of the proposed model.

### Strengths
	Important task – S2S forecasting

	The experimental analysis is relatively thorough, and the paper reports performance gains in the long-lead (15–45 day) range across a variety of metrics.

	The method directly targets over-smoothing with a concise design (climatology fusion + uncertainty blocks), thereby enhancing performance.

### Weaknesses
Major weaknesses are as below:

	The impact of adding uncertainty blocks with Gaussian noise is under-explained. The paper claims gains in generalization and uncertainty, but the analysis is thin. In Table 3, the w/o climatology + noise variant outperforms w/o climatology, yet it remains unclear why the injected perturbations help. More clarification of this behavior would be helpful. 

	The paper frames long-lead degradation as over-smoothing and model collapse, yet shorter leads improvement is also reported. However, it remains ambiguous why the method helps even at shorter leads. As the authors mention in line 357-360, transformer-based direct prediction model ClimaX performs worse than Fuxi-S2S, but the paper does not clarify why this observation does not carry over to TianQuan-S2S.

	Table 2 shows that Fuxi-S2S performs worse than climatology for Wind10, but the paper does not discuss the reason. A clear explanation of the reason would help.

	In Appendix D, the authors state that each lead-specific model inputs a 5-day input window and produces a 5-day forecast. In practice, adjacent days (e.g., D+19 vs D+20) come from different models (e.g., PM20 vs PM25). The paper provides no analysis of whether this block boundary introduces temporal discontinuities or not. This leaves the temporal coherence of the lead-specific single-step models in doubt.

Minor comments are as below:

	Line 140: \hat{X}_{t_{20}:t_{45}} -> \hat{X}_{t_{15}:t_{45}}

	Line 211: Including the reference for climate models using ViT architectures would be helpful.

	Figure 3 lacks essential information. Please clarify what each column represents.

If the weaknesses are addressed well, I will reconsider the score.

### Questions
	In line 361-362, the paper mentions that wind forecasting is more challenging for all baselines. What is the specific reason that wind forecasting is a challenging problem?

	Table 1 indicates that performance decreases for both ClimaX and TianQuan-S2S as lead time grows. Is this due to the intrinsic difficulty of longer horizons, or does it reflect model collapse?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the challenge of model collapse in Subseasonal-to-Seasonal (S2S) forecasting by introducing a simple yet effective framework, TianQuan-S2S.
The method (1) integrates climatology information into patch embeddings to complement the limited predictive power of initial states, and (2) injects Gaussian noise into Transformer blocks to preserve variability and reduce over-smoothing.
Extensive experiments on the ERA5 dataset show clear improvements over ECMWF-S2S, FuXi-S2S, and ClimaX in both deterministic and ensemble settings.

### Strengths
- Well-defined problem motivation: Clearly identifies information loss and model collapse as key S2S challenges.

- Simple but effective design: Climatology integration and noise augmentation yield robust long-lead forecasts.

- Comprehensive experiments: Covers multiple variables, metrics, and spatial resolutions on 40 years of ERA5 data.

- High writing quality and reproducibility: The paper is clearly structured and provides implementation details and code.

### Weaknesses
The main limitation of this work lies in its modest technical novelty. The proposed approach largely builds upon existing concepts such as climatology conditioning and stochastic perturbation within Transformer architectures, rather than introducing a fundamentally new modeling paradigm or theoretical insight.

The method can be viewed as a clever and carefully engineered recombination of previously explored ideas, rather than a conceptual breakthrough. While the integration of climatological priors and noise injection is executed elegantly, the individual components are well-known in both numerical and data-driven forecasting literature.

Moreover, the design choices—such as attention-based climatology fusion and Gaussian noise injection at each Transformer layer—feel somewhat heuristic (“tricky”), lacking a deeper theoretical justification for why this specific combination should outperform alternatives like simple concatenation or dropout-based regularization.

Nevertheless, the paper demonstrates clear empirical benefits and provides a well-structured and reproducible evaluation. The proposed model achieves strong, consistent improvements across multiple meteorological variables and lead times, effectively mitigating the long-horizon degradation problem that limits many prior S2S models.

Given that research in subseasonal-to-seasonal forecasting remains relatively sparse, and that the paper delivers a tangible step forward in practical performance and stability, this work deserves publication. It represents a solid and valuable contribution that advances the field through thoughtful system design and comprehensive experimental validation, even if the underlying techniques are not entirely novel.

### Questions
The paper introduces a learnable Gaussian noise injection within every Transformer layer, claiming that it helps sustain variability and prevent long-lead model collapse. However, it remains unclear how this mechanism fundamentally differs from existing stochastic regularization techniques such as dropout, layer noise, or even standard input perturbation used in ensemble forecasting.

Specifically, how is the proposed per-layer Gaussian perturbation different from simply perturbing the input fields or initial conditions, as done in conventional ensemble methods like FuXi-S2S or stochastic parameterization in NWP systems? Input perturbations are known to encourage diversity and uncertainty propagation—so what unique advantage does injecting noise throughout the Transformer depth provide?

Moreover, the paper would benefit from a direct experimental comparison between:

- input-only perturbation (ensemble IC perturbations),

- fixed or dropout-style layer noise, and

- the proposed learnable state-dependent Gaussian noise.

Such a comparison could clarify whether the improvement stems from deeper uncertainty modeling or simply from added stochasticity. Additionally, the authors should discuss sensitivity to noise scale and spatial correlation, since uncontrolled noise magnitude might act as a crude regularizer rather than a physically meaningful uncertainty representation.

### Soundness
4

### Presentation
4

### Contribution
3
