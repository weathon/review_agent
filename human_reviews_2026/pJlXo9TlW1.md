# EEG-X: Device-Agnostic and Noise-Robust Foundation Model for EEG

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Foundation models for EEG analysis are still in their infancy, limited by two key challenges: (1) variability across datasets caused by differences in recording devices and configurations, and (2) the low signal-to-noise ratio (SNR) of EEG, where brain signals are often buried under artifacts and non-brain sources. To address these challenges, we present EEG-X, a device-agnostic and noise-robust foundation model for EEG representation learning. EEG-X introduces a novel location-based channel embedding that encodes spatial information and improves generalization across domains and tasks by allowing the model to handle varying channel numbers, combinations, and recording lengths. To enhance robustness against noise, EEG-X employs a noise-aware masking–reconstruction strategy in both raw and latent spaces. Unlike previous models that mask and reconstruct raw noisy EEG signals, EEG-X is trained to reconstruct denoised signals obtained through an artifact removal process, ensuring that the learned representations focus on neural activity rather than noise. To further enhance reconstruction-based pretraining, EEG-X introduces a novel dictionary-inspired convolutional transformation (DiCT) layer that projects signals into a structured feature space before computing reconstruction(MSE) loss, reducing noise sensitivity and capturing frequency- and shape-aware similarities. Experiments on datasets collected from diverse devices show that EEG-X outperforms state-of-the-art methods across multiple downstream EEG tasks and excels in cross-domain settings where pre-trained and downstream datasets differ in electrode layouts. The models and code are available at \href{https://anonymous.4open.science/r/EEG-X-0702/README.md}{here}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces EEG-X, a device-agnostic and noise-robust foundation model for EEG representation learning. The model addresses two primary challenges in EEG analysis: the variability across different EEG devices and configurations, and the inherent low signal-to-noise ratio (SNR) of EEG data. EEG-X incorporates a location-based channel embedding, which encodes the spatial relationships between electrodes to ensure robustness across different devices. Additionally, it employs a noise-aware masking-reconstruction strategy, working in both raw and latent spaces to enhance the robustness of the learned representations. A key component of EEG-X is the DiCT layer, which helps capture frequency- and shape-aware similarities, improving noise robustness during signal reconstruction. The authors demonstrate the effectiveness of EEG-X across several datasets and tasks, showcasing its superior performance over current methods.

### Strengths
The paper is well-written and easy to follow. The structure is logical, and the explanations of the model architecture and methods are concise and clear, making it accessible to readers from different fields. The paper compares EEG-X with several recent state-of-the-art models in the experiments.

### Weaknesses
The work lacks significant novelty, for the following reasons:

1. The model architecture is quite similar to that of EEG2Rep [1], as it also uses two encoders combined with EMA for learning. The masked strategy used in the MAE is the same as in existing models. 

2. In the latent reconstruction component, the structure consists of an encoder and predictor, which mirrors the approach in EEG2Rep. 

3. Furthermore, the idea of latent reconstruction itself is not new, as models like EEGPT [2] have already explored this strategy. 

4. In the denoising reconstruction part, this approach has also been explored in related works, such as DMAE-EEG [3], which also uses denoised signals as supervision.

5. Although the paper proposes a location-based channel embedding to handle varying electrode configurations, this method essentially uses a 2D position encoding similar to what is done in image processing. Mapping EEG electrode positions to a 2D spatial grid is not a novel approach. Additionally, other studies, such as BrainGPT [4], have already provided innovative solutions to address the issue of varying electrode configurations. Hence, the contribution in this regard is not groundbreaking.

Overall, the individual components proposed in this work do not show strong novelty or significant contribution. Some of the methods appear to be patchwork or combinations of existing techniques rather than offering a fresh perspective. The contributions in terms of methodology are not sufficiently robust.

The experiments in this work are rather thin, reflected in:

1. The experiments conducted in this paper are relatively simple. There is a lack of parameter experiments and comparisons with different position encoding methods. 

2. The number of datasets and tasks considered is limited, with the paper focusing mainly on datasets with smaller electrode counts (8, 14, 19, 22), which are not very representative of the wide variety of EEG data. The inclusion of more diverse datasets and tasks, particularly those with more electrodes, would have enhanced the paper's scope and relevance. 

3. The experimental setup lacks a comprehensive evaluation of the model's performance across a broader range of scenarios.

While the work is well-written, the overall approach is fairly basic. The methods do not provide sufficient novel insights, and the experimental results lack the complexity and depth needed to justify the model's contribution to the field. The paper presents a somewhat incremental approach rather than offering a breakthrough in EEG analysis.

[1] Mohammadi Foumani, Navid, et al. "Eeg2rep: enhancing self-supervised eeg representation through informative masked inputs." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

[2] Wang, Guangyu, et al. "Eegpt: Pretrained transformer for universal and reliable representation of eeg signals." Advances in Neural Information Processing Systems 37 (2024): 39249-39280.

[3] Zhang, Yifan, et al. "DMAE-EEG: A Pretraining Framework for EEG Spatiotemporal Representation Learning." IEEE Transactions on Neural Networks and Learning Systems (2025).

[4] Yue, Tongtian, et al. "Eegpt: Unleashing the potential of eeg generalist foundation model by autoregressive pre-training." arXiv preprint arXiv:2410.19779 (2024).

### Questions
1. Regarding the novelty issue, please refer to the detailed content in the Weakness section.

2. The claim that DiCT provides noise robustness, frequency balance, and shape-awareness lacks convincing evidence. The underlying reasons for these effects are not clearly explained. It would be beneficial to provide a theoretical, formula-based proof to support these claims. Furthermore, it is difficult to pinpoint exactly which part of the EEG signal (such as specific frequencies, amplitudes, or phases) plays a crucial role in particular brain activity patterns. Although the appendix includes simple illustrations, they do not adequately demonstrate the rationale or internal logic behind the approach.

3. The experimental setup in Table 5 is unclear. It is not obvious what this comparison is meant to show. Does "raw" refer to the data excluding the artifact removal module, while ICA preprocessing refers to directly using signal processing rather than incorporating the artifact removal module? This needs further clarification.

4. The experiments in this work are too limited. There is a lack of essential parameter experiments, the display of pretraining loss, comparisons between different position encoding methods, and experiments on datasets with a larger number of electrodes. Additionally, no comparisons with alternative modules for DiCT are explored. The variety of tasks involved are not on par with other foundation models. The types of tasks involved in the pretraining  and downstream tasks are also quite limited.

5. The model's implementation details are not reported clearly.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
EEG-X is developed as a foundation model for EEG with a key contribution for improving robustness to noise and varying electrode positions. Suggested key contributions include (1) noise removal via ICA, (2) a teacher-student algorithm for masked autoencoding, (3) DiCT for latent, noise-robust reconstruction. These result in strong downstream task results across multiple datasets, and ablation studies justify the contribution of each module, especially nuanced in cross-domain generalization.

### Strengths
The main strength lies in the timeliness of the investigation. Robustness to EEG, yet paramount in the domain, has been elusive.
Another important strength is the excellent presentation. Both writings and figures are clear and constructive.
Supplemental studies also further support the choice of the model components.

### Weaknesses
A foundation model conventionally assume multimodality and scalability. If not, it would be more a pretraining strategy rather than a "foundation model" of the brain. And this paper seems to belong to the latter.

First, scalability studies are not shown. Are these models efficient? does this model scale with more data, model, or compute? What are the parameter / FLOPs size for the baselines? Recent advances in LLMs / foundation models illuminate the benefit of a large model where performance monotonically improves for large (especially Transformer) models. Without a fair scale-wise comparison, the downstream performance would not be objectively evaluated.

Second, Novelty concerns. Most modules are either inherited from prior papers or already popular in the EEG domains. Although application to EEG is may be novel, noise removal via ICA, coordinate based representation learning are both popular within DL for EEG domains. Likewise, teacher-student formation or DiCT may not be so new.

### Questions
1. The motivation of DiCT is demonstrated less. Did you see any example in the real signal also that DiCT reconstruction itself gets better? Supplemental B.1 seems to provide justification, but whether it learns in the real data might be straight-forward to quantify.

2. Can you provide additional computational efficiency / scalability comparisons?

3. Can you justify the coordinate based positional encoding strategies? There are other ways, such NeRF-style, Fourier Features, learned PE, STCPE, etc.

### Soundness
4

### Presentation
4

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
This paper proposes EEG-X, a device-agnostic and noise-robust foundation model for EEG signals. The model aims to enhance the generalization capability and robustness of EEG representation learning by introducing a location-based channel embedding, a noise-aware dual-space reconstruction (raw and latent), and a Dictionary-inspired Convolutional Transformation (DiCT) layer. The authors validate its performance on multiple datasets and tasks.

### Strengths
The paper clearly identifies two key challenges for EEG foundation models—device variability and low signal-to-noise ratio—and proposes corresponding modules to address them. The paper conducts both in-domain and cross-domain evaluations on multiple datasets and compares against several baseline models, showing performance advantages.

### Weaknesses
1. Limited Methodological Novelty: While the paper integrates several existing techniques (e.g., positional encoding, latent space reconstruction, noise-aware reconstruction), the originality of each component is limited. For example: Positional encoding is essentially an adaptation of 2D positional encoding from images, not a novel contribution for EEG; Latent space reconstruction has already been explored in works like EEG2Rep and EEGPT; The DiCT layer, while proposed, draws clear inspiration from time series classification methods like Hydra, lacking theoretical breakthrough.

2. Insufficient Justification for DiCT: The paper claims that DiCT possesses properties like "noise robustness, frequency balance, and shape-awareness," but lacks rigorous theoretical or mathematical proof. The provided synthetic experiment, while illustrative, does not sufficiently reveal its operational mechanisms or fully validate its effectiveness on real EEG signals.

3. Inadequate Experimental Depth and Breadth: Lacks sensitivity analysis for key hyperparameters (e.g., number of groups/kernels in DiCT). No comparison with other positional encoding methods, making it difficult to demonstrate the superiority of the proposed location-based embedding. The datasets used primarily feature a low number of electrode, failing to cover high-density EEG scenarios and limiting the generalizability of the conclusions. No ablation studies replacing the DiCT module to verify if it genuinely outperforms other feature extraction structures.

4. Unclear Implementation Details: The paper does not sufficiently elaborate on model architecture, training specifics, or parameter settings, hindering reproducibility.

5. Limited Task Variety: The types of pre-training and downstream tasks are relatively narrow, failing to fully demonstrate the broad applicability expected of a "foundation model."

### Questions
How does EEG-X fundamentally differ in its core approach from models like EEG2Rep and EEGPT? Please specify the innovative aspects of its representation learning mechanism.

### Soundness
2

### Presentation
2

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
EEG-X is a self-supervised EEG foundation model that encodes electrode positions for device-agnostic inputs, reconstructs artifact-removed targets to avoid learning noise, and uses a DiCT projection before MSE to balance frequency and shape information. It reports better in-domain accuracy and stronger cross-device transfer than recent baselines.

### Strengths
Addresses device heterogeneity and low SNR directly. Location-based channel embeddings are simple and general. Noise-aware targets and DiCT are well motivated and easy to adopt. Experiments include linear-probe transfer across diverse datasets.

### Weaknesses
ICA with ICLabel can mix information across windows or subjects unless fit only on training data, and denoising targets must be produced with training-only statistics. The cross-domain protocol freezes the encoder but may still reflect pretraining dataset idiosyncrasies if hyperparameters or tokenization differ across models; fairness controls are not fully spelled out. Artifact removal is positioned as model-agnostic, yet results and ablations emphasize ICA; dependence on ICA thresholds and component counts is not quantified.

### Questions
How are ICA unmixing matrices and ICLabel thresholds fit with respect to data splits? Are they estimated per subject, per session, or on pooled training data only, and never using test windows? What referencing is used during pretraining and fine-tuning, and is it consistent across datasets? How are channel coordinates obtained when datasets lack digitized positions; what is the mapping from montage names to the universal grid, and how are missing electrodes handled at inference? What is the sensitivity of EEG-X to ICA variant, component count, and probability thresholds, and does performance hold with rASR or wavelet denoising? What is the compute footprint of pretraining, the parameter count of student and teacher, and the inference latency on standard EEG window sizes?

### Soundness
3

### Presentation
3

### Contribution
2
