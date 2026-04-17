# OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Multimodal spatiotemporal learning on real-world experimental data is constrained by two challenges: within-modality measurements are sparse, irregular, and noisy (QA/QC artifacts) but cross-modally correlated; the set of available modalities varies across space and time, shrinking the usable record unless models can adapt to arbitrary subsets at train and test time. We propose OmniField, a continuity-aware framework that learns a continuous neural field conditioned on available modalities and iteratively fuses cross-modal context. A multimodal crosstalk block architecture paired with iterative cross-modal refinement aligns signals prior to the decoder, enabling unified reconstruction, interpolation, forecasting, and cross-modal prediction without gridding or surrogate preprocessing. Extensive evaluations show that OmniField consistently outperforms eight strong multimodal spatiotemporal baselines. Under heavy simulated sensor noise, performance remains close to clean-input levels, highlighting robustness to corrupted measurements.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multimodal conditioned neural field (CNF) framework for learning from sparse, irregular, and noisy multimodal spatiotemporal signals, called OmniField. OmniField has three major components: Gaussian Fourier embeddings, sinusoid initialization to stabilize training, and iterative cross-modal refinement to align modalities. The authors evaluate OmniField on both simulated and real-world multimodal geo-spatio-temporal datasets and show notable performance gain over SOTA baselines. Ablation studies are performed on both spatial and spatio-temporal datasets to understand the contribution of each component.

### Strengths
- The datasets and benchmarks are comprehensive spanning across multiple applications
- Omnifield shows robustness and performance gains
- Proposed components are validated through ablation studies

### Weaknesses
- I am quiet concerned with the novelty. The core framework remains a straightforward extension of SCENT with a few architectural augmentations for multimodal data.
- The authors have limited explanation of training efficiency and scalability. The computational complexity can grow with the number of tokens and modalities, but the paper has limited analysis on the training or inference efficiency, nor does it discuss how OmniField might perform on larger-scale or real-time systems. Given that some spatiotemporal models for some domains are often deployed on resource-constrained platforms, this efficiency is important.
- Although the authors evaluated the methods on numerous datasets, the diversity of the downstream task evaluated is limited. The paper focuses almost entirely on reconstruction and forecasting accuracy. Other tasks such as classification, uncertainty quantification or anomaly detection could enhance the robustness. A stronger connection to practical use cases would also enhance the work’s relevance.
- The model implicitly assumes shared information across modalities but does not explicitly disentangle shared vs. private modality features, which are important in multimodal learning. While the iterative fusion helps align features, it remains unclear how modality-specific noise or private components are handled. Can the authors elaborate on the integration with factorized multimodal learning?
- Minor:
    - More extensive citations in the Introduction, especially from the data/model perspective to contextualize multimodal spatiotemporal learning in prior literature can benefit.
    - What is SCENT-MM? Is that OmniField? Figure 4b and 4c do not have OmniField

### Questions
- How sensitive is the performance to the number of refinement iterations? Could fewer iterations achieve comparable results with lower computational cost?
- The fusion mechanism prevents information leakage from missing modalities. Could the authors quantify how performance is under a randomly missing modalities scenario?
- How does OmniField perform in other tasks, such as classification, where the learned field embeddings are fine-tuned for downstream objectives, rather than regression or forecasting?
- How transferable is the approach to non-geospatial multimodal time-series domains (e.g., high-frequency sensor data)? Does the current formulation depend heavily on spatial continuity assumptions?

### Soundness
2

### Presentation
2

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
This work presents a unified neural field paradigm that tackles the challenge of learning continuous, frequency-rich representations across heterogeneous spatial, temporal, and multimodal sensing regimes.
The authors blend gated fleximodal fusion with iterative cross-modal refinement, allowing the model to gracefully interpolate missing channels, suppress noisy sensors, and sustain high-frequency fidelity in sparse, real-world observation settings.
The authors evaluated the proposed method using climate simulation (ClimSim-THW), air-quality forecasting (EPA-AQS), and vision-centric reconstruction tasks, compared with baselines like SCENT and RainNet.

### Strengths
- The proposed fleximodal fusion and iterative refinement approach provides a principled mechanism for handling missing and noisy modalities, improving robustness in settings with irregular or sparse sensors.
- The incorporation of frequency-rich embeddings and sinusoidal initialization yields measurable gains in high-frequency signal reconstruction, particularly in spatiotemporal domains.
- The method showed consistent performance improvements across multiple scientific datasets, suggesting a generalizable modeling framework rather than task-specific tuning.

### Weaknesses
- The evaluation focuses on a curated set of scientific benchmarks; broader assessment on diverse multimodal domains (robotics, remote sensing beyond climate/air quality) would strengthen claims of generality.
- The computational and memory cost of iterative cross-modal refinement and continuous-field conditioning is not fully characterized. It's unclear how well the proposed method scales to higher-resolution or real-time applications.
- While robustness to missing modalities is a central motivation, ablation analyses on modality-specific contributions are limited.

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
2

### Summary
The paper proposes OmniField, a framework for robust multimodal spatiotemporal learning from sparse, irregular, and noisy real-world observational data.
OmniField extends the continuous neural field paradigm (previously introduced in SCENT) by addressing two fundamental challenges: data sparsity and modality inconsistency.
Specifically, it integrates several components — Gaussian Fourier Features (GFF) and Sinusoidal Initialization (SI) for improved spatial continuity, and three new modules:
MCT (Multimodal Crosstalk) for early cross-modal feature interaction,
ICMR (Iterative Cross-Modal Refinement) for multi-round alignment across modalities,
and Fleximodal Fusion for handling missing or degraded modalities.
Extensive experiments on four datasets (CIFAR-10, RainNet, ClimSim-THW, and EPA-AQS) and eight baselines show consistent performance improvements in both reconstruction and forecasting tasks, especially under conditions of high sparsity and missing modalities.

### Strengths
* The paper is well-structured and clearly motivates the problem by identifying two central challenges — data sparsity and multimodal inconsistency — and proposing targeted solutions through the MCT, ICMR, and Fleximodal Fusion modules. The organization is logical.

* The figures are well-designed and self-explanatory, effectively supporting the paper’s claims and illustrating the benefits of the proposed modules. Quantitative results are straightforward and convincing, showing consistent gains over baselines.

* The experimental results are solid and comprehensive, covering eight strong baselines and four diverse datasets. The results are straightforward to interpret and show consistent improvements across all settings. 

* The paper includes extensive ablation studies, noise-robustness analyses, and missing-modality evaluations, which together demonstrate strong empirical evidence for the proposed method's robustness and generality.

### Weaknesses
* While the paper is generally well written, some parts are conceptually dense and abstract. The presentation could benefit from additional intuition, clearer intermediate explanations, or a small running example to illustrate how each proposed component (MCT, ICMR, Fleximodal Fusion) functions in practice.
* The forecasting horizon studied in the current experiments is relatively short (e.g., six-hour prediction on ClimSim-THW). Evaluating longer temporal horizons could provide deeper insights into the model’s stability and long-term reasoning ability. Incorporating additional temporal modeling structures might further enhance the method’s forecasting capability.
* OmniField introduces additional modules (MCT and three rounds of ICMR), which substantially increase the computational cost compared to the baselines. While the model achieves lower RMSE and improved robustness, these gains come with notably higher architectural complexity. A more detailed analysis of runtime efficiency trade-offs would help clarify the impact of the proposed design.

### Questions
* In Figure 4(b–c), the label “SCENT-MM” seems to correspond to OmniField, based on the text and tables. Could you please confirm whether this is a labeling oversight, or if SCENT-MM refers to a distinct model configuration?
* I noticed that OmniField has a relatively large number of parameters compared to SCENT. To what extent might the performance improvement stem from model capacity rather than the proposed architectural changes?

### Soundness
4

### Presentation
3

### Contribution
2
