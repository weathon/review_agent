# FSSM: Frequency-Selective State Space Models for Spectral Representation Learning

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
We introduce the first state space model (FSSM) with frequency selective spectral operators, parameterizing a family of stable, causal, band-selective kernels whose spectral weights are conditioned on the end task. This yields a representation that adapts its characteristics per task domain while retaining linear-time inference and memory. The key novelty is the trainable spectral front-end through which the model can adapt frequency weighting and inter-bin window size. We show the effectiveness of our learned spectral representations on two independent domains: radar object detection and speech keyword recognition, outperforming state of the art frequency based methods in both domains while maintaining competitive throughput and computational overhead. We further show the robustness of our approach under input perturbations, demonstrating the value of stabilized sequential operators in spectral representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces the frequency-selective state space model (FSSM), a spectral representation learning module that augments SSMs with frequency-selective, task-adaptive kernels. It demonstrates the effectiveness of FSSM on two distinct application domains: high-definition automotive radar for object detection and free-space segmentation and speech keyword recognition.

### Strengths
* The paper contains a thorough discussion of related works and the proposed method is written rigorously into pseudocode,

### Weaknesses
* My major concern is the positioning of this paper. It is advertised for using SSMs to enhance spectral representation learning, but it is barely accessible to either community. For SSM people, section 3 is written in a mysterious way. There are multiple new design choices different from a standard SSM/Mamba and no intuition is given. For spectral learning people, the benefit of SSM is not clearly anticipated and there also lacks enough background provided for understanding the proposed SSM integration.
* There is no analysis of the model in addition to experiments. It is intuitively unclear why the proposed method should outperform the existing ones. Some theoretical insights could be helpful.
* While the experiments on radar and speech are thorough, these two choices seem quite arbitrary; no direct ablation is provided (see questions below).
* The presentation of the paper needs to be improved. In addition to cleaning up the formulas in section 3 and make the transition smoother, there are a couple of things that need to be corrected, including the extensive misuse of en dashes and em dashes and the storage of figures, which should be in a vectorized format.

### Questions
1. How is FSSM computed during training and inference? In Algorithm 1, the psuedocode appears as a loop. Almost all previous SSMs are trained in parallel using either FFT (for S4, S4D, etc.) or parallel scan (for S5, Mamba, etc.). Can this be done to an FSSM?
2. Relatedly, how easy is it to train an FSSM to achieve the results in section 4?
3. I am confused about the position of the proposed FSSM. In particular, I cannot tell which models are FSSMs supposed to compete against. If FSSM is proposed as a alternative to SSMs/Mambas, then there is no empirical comparison found in the paper. If FSSM is to be competed against traditional methods like FFT, then their distinction needs to be highlighted and the "SSM" needs to be ablated. The name "FSSM" probably also need to be changed into something like "SSM-augmented FFT front-end."
4. There are multiple components proposed in section 3, and many of them are not totally intuitive. Is it possible to have ablation for the phase alignment, learnable damping, and the proposed parameterizations?
5. In what ways, if any, did the use of LLMs (as mentioned right before the bibliography) influence the technical findings or original ideas of the submission?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
## Learning frequency-selective spectral representations using state space models

- The paper proposes FSSM (Frequency Selective State Space Models) for learning spectral representations from audio data, as a direct challenger of FFT-based spectral representations and establishes their performance on Radar object detection and Keyword spotting tasks.
- This is done through learning a *frontend* fitted with 2-d modelling using the proposed FSSM modules along samples and chirps for RADAR data, whereas it uses the proposed FSSM followed by bidirectional-Mamba for keyword spotting.

### Strengths
- Novel framework for spectral representation learning through frequency-selective state spaces.
- The proposed methods perform well against baselines on radar object detection on the RADIal dataset and keyword spotting (Speech Commands v2) datasets.

### Weaknesses
## Weakness 1: Why Keyword Spotting on Speech Commands?
- The use of keyword spotting on Speech Commands as a representative for audio recognition performance is not well motivated. 
- The proposed FSSM touts frequency-selectivity as its main strength; however, I don't believe Keyword Spotting is a challenging enough task to put that to a true test. Speech Commands v2 has 35-keyword classes and is an English-only dataset, and is rather limited.
- A more comprehensive audio recognition task, like multi-label audio classification on AudioSet or FSD50k would have been a better test-bed for testing frequency-selectivity of the proposed FSSM frontend.
- In essence, to be of wider applicability and to prove that FSSM is a good learnable frontend, a better pairing of auxiliary task is needed alongside radar-based object detection.

---

## Weakness 2: Figures
- Maybe I'm splitting hairs here, but all the figures in the paper are subpar. They have poor resolution and quality, and the text is too small (especially Figure 3), making it difficult to read in print. 
- In Figure 5, there are so many empty bins it is not possible to clearly see the difference between the FFT and FSSM visualizations. There is also an outline to the figure which looks weird.

---

## Weakness 3: Frequency-selectivity

- The frequency selectivity of the proposed approach is not well demonstrated. Only Figure 5 shines a light on it, that too poorly.
- A more comprehensive empirical/exploratory investigative analysis is necessary to ascertain that FSSM indeed does better at being frequency selective.

---

## Weakness 4: Empirical analysis

- Alongside the use of Speech Commands as the auxiliary task, the overall quality of empirical analysis in the paper is lacking.
- There are no confidence intervals, and the precision used in reported results (significant digits) is insufficient, especially for RADIal analysis, where margins between the evaluated approaches are quite small. For instance, the mAP for FFT-RadNet and the proposed FSSM-FFTRadNet is 0.97 and 0.98, respectively.

---

## Weakness 5: Spelling and grammatical errors
- There are several spelling and grammatical errors throughout the paper. for instance *represetation*, *segentation*. The paper needs more  proofreading and polish.

---

# OVERALL

- It's a good idea, but the execution is lacking. It needs significant work before it is ready for publishing.

### Questions
No direct questions for now, kindly address the weaknesses.

### Soundness
3

### Presentation
2

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
This work introduces a frequency-selective state-space module that leverages adaptive spectral features for effective object detection and speech recognition. The core idea can be viewed as combining a learnable spectral module, such as SincNet or Garbor filter, with a Mamba-style module that enables adaptive state-space modeling.

### Strengths
* The proposed method yields empirical performance improvements in radar object detection and segmentation tasks and demonstrates greater robustness than the baselines in the audio keyword detection experiments.

### Weaknesses
* [1] combines frequency features with adaptive state-space modeling, which appears to be the most relevant prior work. This work attempts to distinguish itself from [1] by learning the frequency bands; however, it is unclear what new benefits this actually provides. The advantage of this distinction is not clearly explained. An empirical comparison with [1] also seems to be missing; if I am wrong, please correct me.


* The proposed method seems tailored to specific tasks  (e.g., radar object detection vs. speech), according to the experiment section. This raises concerns about its generalizability and consistency to other datasets, particularly depending on how the data should be processed for spectral representation.

[1] RFMamba: Frequency-Aware State Space Model for RF-Based Human-Centric Perception  - ICLR 24

### Questions
* Based on my understanding, [1] is highly relevant to your work. Comparing your method with [1] would help clarify the advantage of modeling spectral bands, which [1] does not address. Have you attempted such a comparison, that is, spectral band learning + adaptive state-space modeling versus frequency-feature learning + adaptive state-space modeling?

[1] RFMamba: Frequency-Aware State Space Model for RF-Based Human-Centric Perception  - ICLR 24

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
1

### Summary
This paper proposes FSSM, the first SSM with frequency-selective spectral operators that can adapt to the task at hand. The front-end of the model that processes the raw data into features is learnable, enabling the models to adapt to phase and frequency emphasis on the fly. Combined with SSM or transformer backbones, FSSMs outperform other baseline methods with static front-ends in two domains, radar object detection and speech processing.

### Strengths
1. Novelty and efficiency: the idea of using selective modeling in the spectral frontend is new as far as I know, and the design is efficient as it retains the linear complexity of SSMs. A dynamic spectral frontend is a desirable innovation especially for nonstationary data, where the filter banks need to adapt to the changing data distribution. 
2. The empirical results for radar object detection is strong: FSSM-based methods outperform static, DFT/FFT-based methods. 
3. The paper includes robustness analysis and FSSM seems to be more robust than other baselines.

### Weaknesses
1. Domain-specific architectures: in the two domains studied in the paper, the architectures for the front-end are different. This implies that the front-end needs to be specifically designed for a particular task, making it high-touch and less general. 
2. How does FSSM-based methods perform compared to Mamba and other, pure SSMs on speech recognition tasks? Pure SSMs without the spectral front-end can also perform such tasks, and it would be interesting to investigate whether a spectral front-end is necessary or useful.
3. Writing: I find the writing to be difficult to understand without certain background. It would be helpful to include some background on signal processing fundamentals, for example the issue of leakage for static frond-ends, I-Q signals, and Fourier anchoring.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
