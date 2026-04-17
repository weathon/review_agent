# Decomposed Attention FredFormer: Large Time-series Prediction Model for Satellite Orbit Prediction

- Decision: Reject
- Scores: 4, 8, 4, 8

## Abstract
Accurate satellite orbit prediction is critical for collision avoidance and sustainable space operations. However, conventional methods are constrained by coarse update intervals, orbit discontinuities, and other factors. Additionally, building separate prediction models for each satellite is computationally expensive, making large-scale accurate forecasting increasingly impractical.
To address the aforementioned challenges, we propose Decomposed Attention FredFormer (DAF), a large time‐series prediction model that uses efficient Real Fast Fourier Transform (RFFT)/Inverse RFFT in favor of positional embeddings. Our DAF also integrates Tensorized Multi-Head Attention based on Tensor Train Decomposition for parameter-efficient compression and improved performance. We pre-trained on a large‐scale Starlink dataset and evaluated zero-shot performance on seven cross-domain satellite orbit datasets and three real-world datasets. DAF achieves up to 34.85\% reduction in mean squared error and 16.01\% reduction in mean absolute error over the second-best model, using only 0.05\% of its parameters and maintaining inference time as fast as the conventional neural network baselines. These results demonstrate that DAF enables zero-shot, high-precision orbit prediction not only for Starlink satellites, but also for other satellites. The code is available here: \url{https://anonymous.4open.science/r/DAF-0D75}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Decomposed Attention FredFormer (DAF), a large time-series forecasting model designed for satellite orbit prediction.  Built upon FredFormer, DAF introduces three main modifications: 
+ replacing DFT with RFFT/IRFFT to leverage the symmetry of real-valued inputs.
+ removing patching and layer normalization while adding positional embeddings to better capture orbital periodicity.
+ integrating Tensor Train Decomposition (TTD) into multi-head attention for parameter compression. 

Trained on large-scale Starlink data, DAF achieves strong zero-shot performance. The method demonstrates high efficiency and promising generalization across diverse orbital regimes.

### Strengths
+ Clear motivation and practical relevance. The paper addresses a critical and under-explored domain: satellite orbit prediction using large time-series models. This setting highlights real-world importance and contributes to the intersection of space situational awareness and AI for scientific computing.
+ Effective and task-aligned efficiency. The method thoughtfully exploits the characteristics of orbital time-series data, using real-valued frequency transforms and tensorized attention to substantially reduce model parameters while preserving both predictive accuracy and inference efficiency.
+ Extensive and well-validated experiments. The paper conducts comprehensive experiments across multiple satellite constellations and real-world datasets, including detailed zero-shot evaluations and ablation studies.  The breadth of baselines and the consistency of improvements provide strong empirical support for the proposed approach

### Weaknesses
+ Limited theoretical insight. While the paper presents clear empirical gains, it lacks a deeper theoretical analysis explaining why tensor decomposition and real-valued Fourier transforms improve performance.  The benefits appear primarily empirical, without discussion on the induced inductive biases or representational properties.

+ Incremental methodological novelty. The proposed components, such as RFFT substitution, LayerNorm removal, and tensorized attention, are all established techniques.  Their combination is well-engineered but not conceptually groundbreaking, making the contribution more engineering-oriented than methodological.

+ Lack of comparison with traditional baselines. The introduction motivates the work by contrasting data-driven and physics-based orbit prediction methods. However, the experiments only compare against neural and transformer-based baselines, without including classical orbital propagators or numerical integration schemes (e.g., SGP4). This weakens the empirical link between the stated motivation and the reported results.

### Questions
+ How sensitive is the model’s performance to the choice of Tensor-Train rank? Have you observed any trade-off between compression and accuracy?
+ The paper demonstrates strong cross-constellation performance, suggesting promising generalization. However, has the proposed model been validated on other time-series datasets beyond orbital dynamics, or is its performance limited to this specific domain?
+ The paper argues that physics-based propagators such as Orekit are slow and inaccurate, yet these same methods are used to generate the interpolated training and ground-truth data. This raises a conceptual concern:
    - If Orekit is inaccurate, wouldn’t the model inherit its systematic bias through the labels?
    - And if the model still requires interpolated inputs from Orekit during inference, does this not reintroduce the very computational bottleneck the method aims to eliminate?
    - Could the authors clarify how the interpolation process affects both training and inference, and whether the proposed model can operate independently of the physical propagators?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a large time-series model for satellite orbit prediction based on FredFomer. The authors introduce several modifications to the architecture and demonstrate significant improvements in the performance on a large variety of datasets.

### Strengths
S1: The problem is increasingly important to solve

S2: The authors performed several thoughtful (RFFT and TTD, plus several tweaks (e.g. removing patching and layer normalization)) changes to the FredFormer architecture in order to attain higher performance. These changes reflect the domain-specific knowledge about the available data

S3: The resulting model is lightweight, which is crucial for keeping the cost of these operations low.

### Weaknesses
W1: About the model being lightweight, the authors simply report the parameter count. For this application speed is also crucial, it would be very nice if the authors could also report throughput/FLOPs or any other metrics that would give an idea of the speed at which predictions can be made. This should be reported in comparison with all the baselines.

### Questions
Q1: Given in other domains there's a substantial trend of augmenting the real-world training datasets with synthetic data, did the authors consider augmenting the Starlink dataset with synthetic data? (e.g. from Kessler [1]). In particularly it could be interesting to know: 

Q1.1. In case the failures of the current model can be identified as gaps in the training distributions, can the addition of synthetic data alleviate the issue? 

Q1.2. Is it possible to mix synthetic and real data in order to obtain better performance? Or is it detrimental?

Q2: The authors mention performing normalization and its impact on the reported errors (e.g. a 0.001 error can translate to tens of kilometers).

Q2.1. Different perdicted physical variables have different sensitivity, would it be possible for the authors report a breakdown of the MSE/MAE for each predicted variable, to identify whether the errors are evenly distributed or if there's some variables that are harder to predict and could be an area for further improvement for future models? 

Q2.2. I didn't get the chance to check the code, but is the normalization performed per-dataset or across all datasets? Is it possible that a global normalization may render some data less useful in training? Or alternatively, is it possible that a per-dataset normalization may amplify errors on some datasets?

Q3: have the authors considered using techniques that could produce not only predictions but also error estimates (e.g. in terms of variance around the predicted value)?

[1] https://github.com/kesslerlib/kessler

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles satellite orbit forecasting and proposes DAF (Decomposed Attention in the Frequency domain): raw time-series are RFFT-tokenized (real/imag concatenated), fed to an attention block factorized via a low-rank/tensor-train style decomposition, and trained without patching (and with global z-scoring). Across multiple orbital datasets, DAF outperforms strong transformer and large-FM baselines while being inference-efficient on a single GPU. The method’s design is motivated by orbital dynamics’ quasi-periodicity and long-range dependencies.

### Strengths
-  Frequency-domain tokenization is a natural fit for quasi-periodic orbital signals, helping the model capture long-range structure with fewer tokens.
- The decomposed attention reduces memory/time complexity versus dense attention at long contexts, enabling practical inference on commodity GPUs.
- DAF consistently beats competitive transformer baselines and several large time-series FMs on orbital forecasting benchmarks.
- Clear experimental protocol. Common training settings across models and transparent dataset preparation (interpolation choice justified and tested) improve reproducibility and interpretability of results.

### Weaknesses
- The paper reports parameter counts but lacks training-time wall-clock (hours), or FLOPs per step or per sequence. Without these, it’s hard to assess whether gains come from architectural merit vs. compute budget or implementation. Please unify timing/compute reporting across all baselines.
- Frequency-domain positional embeddings need justification. Inputs are RFFT’d and concatenated, then positional embeddings are added before attention. The physical meaning of these PEs in the frequency domain is unclear. Provide ablations vs. time-domain PEs (or none), and discuss why frequency-space tokens need positional encoding and what “position” represents there.

### Questions
- Many baselines/FMs that the paper compared to are fairly old. There are many recent time series foundation models that provide a much stronger performance. Consider inclusing them in results such as ([Chronos](https://arxiv.org/abs/2403.07815)) and ([TOTO](https://arxiv.org/abs/2505.14766))?
- You validate interpolation with SGP4/Skyfield/Orekit. Can you also include a propagation-based physics baseline (same splits) in the forecasting tables to contextualize ML gains?
- Orbit specific metrics: Since small MAE/MSE differences can correspond to kilometer-scale position errors, is there a better metric that you can use to evaluate such as 3D position error in km ?
- Given DAF’s strong results on satellite orbit prediction, what assumptions or inductive biases make it particularly suitable to orbital dynamics (e.g. spectral sparsity, long-range dependencies)? Do you expect the same advantages to hold for general time series, and under what conditions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes the decomposed attention FredFormer (DAF), a transformer (FredFormer)-based architecture that replaces DFT by  real fast Fourier transform and Tensorized MHA (TMHA) using Tensor Train Decomposition for efficient compression. DAF eliminates patching and layer normalization to better model the periodic nature of satellite orbits, significantly reducing parameters while maintaining strong generalization. Trained on a large-scale Starlink satellite dataset. DAF is evaluated on seven cross-domain and three real-world satellite datasets.

### Strengths
The paper targets operational challenges in orbit prediction e.g., long-term forecasting, limited computational resources, and model scalability across satellites making it valuable for real-world deployment.
The authors applied robust preprocessing, ablation studies showing the effects of RFFT, positional embedding, and tensorization and comparison with multiple baselines.
It also demonstrates how parameter-efficient transformer architectures can be adapted for safety-critical, data-sparse scientific domains.

### Weaknesses
All satellites share similar orbital mechanics and control schemes. So, zero-shot generalization to other constellations might be overstated. Did you include satellites with distinct perturbation dynamics? If no, what can you say about the adaptability.

It is not clear what is the inference latency? Whether it can be used on low-resource satellite onboard systems like CubeSats, etc.

The model is benchmarked mainly on mean error metrics (MSE/MAE), is this sufficient to evaluate or evaluation should include orbital physics consistency metrics?

Paper does not include potential failure cases. 

Section 3.2 and 3.1 presenting the same stuff.

### Questions
same as limitations

### Soundness
3

### Presentation
3

### Contribution
3
