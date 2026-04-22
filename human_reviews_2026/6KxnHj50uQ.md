# NeuroSketch: An Effective Framework for Neural Decoding via Systematic Architectural Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Neural decoding, a critical component of Brain-Computer Interface (BCI), has recently attracted increasing research interest. Previous research has focused on leveraging signal processing and deep learning methods to enhance neural decoding performance. However, the in-depth exploration of model architectures remains underexplored, despite its proven effectiveness in other tasks such as energy forecasting and image classification. In this study, we propose NeuroSketch, an effective framework for neural decoding via systematic architecture optimization. Starting with the basic architecture study, we find that CNN-2D outperforms other architectures in neural decoding tasks and explore its effectiveness from temporal and spatial perspectives. Building on this, we optimize the architecture from macro- to micro-level, achieving improvements in performance at each step. The exploration process and model validations take over 5,000 experiments spanning three distinct modalities (visual, auditory, and speech), three types of brain signals (EEG, SEEG, and ECoG), and eight diverse decoding tasks. Experimental results indicate that NeuroSketch achieves state-of-the-art (SOTA) performance across all evaluated datasets, positioning it as a powerful tool for neural decoding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces NeuroSketch, a framework for neural decoding tasks achieved via systematic architectural optimization. The authors investigate various neural network backbones (CNNs, GRUs, Transformers, hybrids) in detail, perform macro- and micro-level optimizations (e.g., latent space transformations, convolutional choices), and gradually distill these principles into the NeuroSketch design, which is an enhanced CNN-2D-based model. NeuroSketch is validated on eight neural decoding tasks spanning three modalities (speech, visual, auditory), three signal types (EEG, SEEG, ECoG), and is shown to outperform a suite of state-of-the-art baselines across multiple datasets and tasks.

### Strengths
1. This study presents a comprehensive evaluation, validating the proposed model across eight datasets and three distinct recording modalities: EEG, sEEG, and ECoG.

2. After extensive tuning of model hyperparameters, the model surpasses other models.

3. The authors provide open-source reproducibility commitments and dataset transparency.

### Weaknesses
1. The implementation details of baselines are missing. The performance of Conformer on Du-IN in Table 4 significantly underperforms the reported value in the Du-IN paper. We need specific details of the baseline implementation to ensure the reliability of the conclusions of the overall work and fair comparison.

2. The CNN-2D introduces translation invariance, which seems to be uncommon in brain signal modeling. It would be useful to include more analysis about the features obtained by the models (e.g., spatial locality), helping us understand the effectiveness of CNN-2D-based BCIs.

3. The framework appears largely a product of empirical exploration, with each step optimized via benchmark wins (cf. Table 2 and Table 3). While this is valuable, the paper does not elevate its findings to generalizable new principles for neural decoding. For example, the manner in which CNN-2D exploits spatial locality is discussed but not quantified, and the underlying reason for Transformer's poor performance isn’t deeply analyzed or theoretically unpacked. The reader is left with “what works” but not always “why” -- limiting the scientific insight provided.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a framework for systematic architecture optimisation. Starting with exploring various model types and then with the best design going to a macro-micro-analysis, the paper introduces a model with competitive performance compared to other deep and foundation model baselines.

### Strengths
The paper is well-structured and it follows a nice step-by-step analysis on how specific details were implemented in the final architecture. It’s the first time I see an analysis like this one. Most papers just introduce a new architecture without any design justifications.

### Weaknesses
The paper fails to provide more comparisons with better models (deep learning and foundation models). In addition, although thorough the analysis does not provide interpretable insights behind the choices.

Writing:
Paper is well-written and good structured.

Overall:
The paper shows some merits but it would be vital to have my questions answered.

### Questions
1. I wonder if the initially tested architectures - like CNN or transformer - get deeper or if more / less data is used during training for each model (for example, transformer based models need more data), would that affect the initial observations ?
2. How about other more compact advanced deep baselines like EEGInception and Brainwave scattering net in table 4?
3. How about other foundation models like LaBraM, NeuroGPT, EEGPT etc. in table 4 ? It seems the foundation models section is not very well represented. 
4. How about the number of parameters of these models as well  ? Any relationship between the number of parameters and performance ?
5. Is there any actual meaning behind the design choices ? For example, EEGNet provides interpretable insights. In other words, is it just black box or is there an actual neurological meaning why these design choices do work?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces NeuroSketch, a neural decoding framework that enhances the performance of decoding multiple types of brain signals (EEG, SEEG, ECoG) through systematic architectural optimization—from basic architecture selection to macro- and micro-level structural improvements. The authors conducted over 5,000 experiments across eight tasks (covering visual, auditory, and speech modalities), demonstrating that NeuroSketch achieves state-of-the-art (SOTA) performance on multiple benchmarks. The core contribution of this framework lies in its ability to model the spatiotemporal characteristics of brain signals, with step-by-step optimizations showing consistent performance gains.

### Strengths
1. Systematic Architectural Exploration with a Clear Optimization Path:
The paper begins with a comparison of basic architectures (CNN-1D/2D, GRU, Transformer, etc.) and progressively delves into macro (latent space transformation) and micro (convolution operation optimization) levels of design. This forms a complete and logically rigorous optimization roadmap, offering high interpretability and methodological value.

2. Large-Scale, Multi-Modal Experimental Validation:
Extensive validation was performed across three modalities (visual, auditory, speech), three brain signal types (EEG, SEEG, ECoG), and eight tasks. The experimental scale is substantial (over 5,000 experiments), making the results highly credible and demonstrating strong generalization capability.

3. Tailored Modeling of Brain Signal Characteristics:
The work explicitly addresses the transient temporal dynamics and spatial locality inherent in neural decoding tasks. Designs such as CNN-2D, group convolution, and early downsampling (Pagoda approach) effectively capture these characteristics, reflecting a deep understanding of the nature of neural signals.

4. Effective Balance Between Computational Efficiency and Performance:
The optimization process considers not only performance improvement but also computational cost (e.g., GFLOPs comparison). Proposed strategies like the Step approach, Pagoda approach, and Group Convolution significantly reduce computational burden while maintaining or even improving performance.

### Weaknesses
1. Lack of Discussion on Neurophysiological Interpretability:
Although the model performs excellently, the paper does not deeply analyze whether the neural representations learned by NeuroSketch are interpretable from a neuroscience perspective (e.g., correspondence to brain region activation or cognitive processes). This is an important dimension in BCI research.

2. Insufficient Comparison with Some Existing Neural Decoding-Specific Models:
While comparisons are made against several general time-series models and some brain-specific models, the comparison with certain recent architectures specifically designed for EEG/SEEG [1,2] is not comprehensive enough. This might fail to fully demonstrate its advantages over the best methods in the field.

3. Insufficient Exploration of Multi-Modal Fusion and Cross-Modal Generalization:
Although tested on multiple modalities, the paper does not explore the model's generalization ability across modalities (e.g., transferring a model trained on visual tasks to auditory tasks), nor does it attempt multi-modal fusion decoding, which holds significant value for future BCI systems.

4. Inadequate Analysis of Individual Differences and Cross-Subject Generalization:
Although experiments were conducted on data from different subjects, the systematic analysis of the model's ability to handle individual differences and its cross-subject decoding capability is relatively limited. Results for cross-subject unified training or adaptation strategies are not provided.

5. Ablation Studies are Not Comprehensive Enough:
Although the step-by-step optimization process is presented, a systematic ablation study on the independent contribution of each component in the final NeuroSketch model (e.g., GeM pooling, residual connections) is lacking. This makes it difficult to judge the specific impact of each part on the final performance.

**References:**

[1] Singh, A., Thomas, T., Li, J., Hickok, G., Pitkow, X., & Tandon, N. (2025). Transfer learning via distributed brain recordings enables reliable speech decoding. *Nature Communications, 16*(1), 8749.

[2] Chen, X., Wang, R., Khalilian-Gourtani, A., Yu, L., Dugan, P., Friedman, D., ... & Flinker, A. (2024). A neural speech decoding framework leveraging deep learning and speech synthesis. *Nature Machine Intelligence, 6*(4), 467-480.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
