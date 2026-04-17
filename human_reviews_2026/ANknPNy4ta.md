# Neural Codecs as Biosignal Tokenizers

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Neurophysiological recordings such as electroencephalography (EEG) offer accessible and minimally invasive means of estimating physiological activity for applications in healthcare, diagnostic screening, and even immersive entertainment. However, these recordings yield high-dimensional, noisy time-series data that typically require extensive pre-processing and handcrafted feature extraction to reveal meaningful information. Recently, there has been a surge of interest in applying representation learning techniques from large pre-trained (foundation) models to effectively decode and interpret biosignals. We discuss the challenges posed for incorporating such methods and introduce **BioCodec**, an alternative representation learning framework inspired by neural codecs to capture low-level signal characteristics in the form of discrete tokens. Pre-trained on thousands of EEG hours, BioCodec shows efficacy across multiple downstream tasks, ranging from clinical diagnostic tasks and sleep physiology to decoding speech and motor imagery, particularly in low-resource settings. Additionally, we provide a qualitative analysis of codebook usage and estimate the spatial coherence of codebook embeddings from EEG connectivity. Notably, we also document the suitability of our method to other biosignal data, i.e., electromyographic (EMG) signals. Overall, the proposed approach provides a versatile solution for biosignal tokenization that performs competitively with state-of-the-art models. The source code and model checkpoints are shared: [Github link to be public upon acceptance].

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The main contribution of this paper is a representation modeling approach, leveraging a popular neural codec that is commonly used in computer vision and audio modality. The authors include several EEG and EMG corpora. The authors demonstrated effectiveness of the proposed approaches in downstream decoding tasks and showed analyses on code usage, especially in correlation that aligns with the functional connectivity measured in raw signals. However, the downstream performance is limited as being worse than some baseline models. Also, some crucial information is not well explained, especially how the representation of baseline models is retrieved (i.e., whether the baseline evaluation is fairly compared). Lastly, the methodology severely lacks novelty that RVQ and architecture are readily tested and deployed for years. More details are elaborated in weakness and questions.

### Strengths
* Successful implementation of neural codec in EEG and EMG signals. 

* On par or better performance than previous models using smaller number of parameters.

* Spatial coherence test shows that codec is compressing the spatially preserved relationships in the signals.

### Weaknesses
* As this model is proposed as a new representation model, the model evaluation and comparison should have the same downstream model architecture/training method across all baseline representations compared. It is unclear how representations for the baseline models are retrieved (e.g., which layer of EEGPT is used?, whether the optimal layer is chosen?, etc). Also, I might miss but I would like to confirm that the same (or comparable) two-transformer models used for BioCodec-based downstream tasks are used for other baseline representations using a similar training resource. This is critical for comparing representations otherwise we can not tell if the gain is from the downstream model architecture/strategy or representation itself.

* The degraded performance shown in Table 2, 3 compared to the previous modeling approaches is not well justified. The authors contend efficiency as using fewer parameters, but the baseline models may perform well with the similar number of parameters.

* The rationale of using neural codec for representation models to EEG and EMG is not well justified. This concern is raised since the neural codec approach is more commonly used for generative modeling than learning rich representations. For example, in the most adjacent modality, speech–having a 1D temporally complex signal–, the neural codec (VQ-VAE; wav2vec) is far worse than masked prediction methods like HuBERT or WavLM [1] (leaderboard: https://superbbenchmark.github.io/#/leaderboard). Indeed, in the evaluation, several metrics are worse than CBraMOD or EEGPT, which supports the possibility that neural codec is not the best way for learning representations. The major purposes of neural codec in other domains are to compress data to have a low bitrate or to train a high-fidelity decoder for generative modeling. More discussions and evidence should be provided to prove whether any of these goals is important in EEG and EMG modality, with a tangible downstream task (e.g., why imposing discreteness embedding is important for learning representation?; the representation is quantized but none-of downstream tasks leverages discreteness of the codes).

* A minor point but the name BioCodec is too generic. The proposed model only covers EEG and EMG data modality while there are many other bio signal modalities. 

* Another minor comment: It would be better to sort the code usage in Figure 2 for a better visualization.


[1] Yang et al., SUPERB: Speech processing Universal PERformance Benchmark. Interspeech 2021

### Questions
For downstream tasks fine-tuning in Section 2.3, it is unclear which part of the pre-trained model is fine-tuned. It is especially confusing since in Figure 1, the BioCodec is demarcated as frozen. Also, in 3.2, it mentions that the encoder is frozen. I was wondering if fine-tuning is not done at all.

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
4

### Summary
This paper introduces BioCodec, a neural codec framework for biosignal tokenization based on residual vector quantization (RVQ). The model treats biosignal representation learning as a compression problem, learning discrete low-level tokens from raw EEG and EMG waveforms. Pre-trained on large-scale EEG corpora, BioCodec achieves competitive or superior performance across diverse downstream tasks (e.g., clinical diagnosis, sleep staging, motor imagery, and ERP decoding) with significantly fewer parameters than contemporary foundation models. The authors also analyze codebook utilization, entropy, and spatial coherence, offering insights into model interpretability.

### Strengths
1. Across multiple EEG and EMG tasks, BioCodec achieves comparable or even superior performance to state-of-the-art models while maintaining a significantly smaller footprint.
2. The work stands out for its comprehensive representational analyses, including codebook entropy, spatial coherence, and perturbation robustness, that go beyond mere benchmarking to provide interpretability and mechanistic understanding.
3. Demonstrating strong cross-modality generalization from EEG to EMG reinforces the versatility and scalability of the codec paradigm for biosignal representation.

### Weaknesses
1. The core technical novelty is modest. RVQ and neural codecs are well-established, and the paper primarily adapts them to biosignals without introducing substantial algorithmic innovation.
2. Several prior EEG works (e.g., DeWave, LaBraM, VQ-MTM) have explored vector quantization or discrete representation learning; more extensive comparison with these and recent VQ-VAE advancements (e.g., EnCodec variants, BrainCodec) would contextualize the contribution better.
3. Some design choices lack empirical justification, for example, the selection of SEANet as the encoder, the choice of 6 quantization layers, 256 codewords, and 16-dimensional vectors. Ablation or sensitivity studies on these architectural hyperparameters would strengthen the technical credibility.
4. On major EEG benchmarks (TUAB, TUEV, Sleep-EDF), BioCodec's advantage is often within statistical noise margins. While efficiency is a strength, there is limited evidence of a clear performance breakthrough on its main modality.
5. Since fine-tuning relies on BioCodec’s discrete embeddings as input, an ablation using raw EEG signals as input to the same downstream transformer architecture would clarify the true added value of quantized representations.

### Questions
1. Given the observed drop in AUROC on several tasks (e.g., TUAB, Kaggle-ERN), can the authors confirm whether this is solely due to quantization loss?
2. How well does BioCodec scale with pre-training data size? Is there evidence that performance or codebook utilization improves consistently with more data, or are there diminishing returns beyond a certain scale?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel representation learning framework, BioCodec, inspired by neural codecs, to capture low-level signal characteristics through discrete tokenization. BioCodec demonstrates strong performance across multiple downstream tasks and exhibits suitability for EMG signals. Additionally, the authors provide an analysis of codebook utilization and its spatial coherence.

### Strengths
- The paper is well-structured and easy to follow, with clear and informative tables and figures.
- Comprehensive experiments are conducted with BioCodec on both EEG and EMG datasets. The analysis of code representations across layers appears interesting to me.

### Weaknesses
- BioCodec is motivated by the observation that the most informative biosignal features can be captured without pre-imposing semantic training tasks on arbitrarily defined tokens, as mentioned in Lines 79–80. However, I am concerned about the reasoning behind this distinction. Specifically, why do techniques such as masked modeling require intrinsic semantic structures, whereas single-channel reconstruction, as employed in BioCodec, does not? What is the fundamental difference between these two types of tasks in terms of their ability to extract temporal patterns from biosignals?

- The contribution of the BioCodec framework is not entirely clear to me. The architectural differences between BioCodec and existing foundation models are not described in sufficient details. Since codebooks are commonly used in neural foundation models, does the main distinction of BioCodec lie in its use of single-channel biosignal inputs and the learning of inter-channel relationships during fine-tuning?

- I am also concerned about how BioCodec is able to learn shared representations from thousands of hours of EEG data with significantly fewer parameters. Wouldn’t this design risk overfitting or limit the model’s capacity to generalize across diverse subjects and recording conditions?

### Questions
- The paper mentions that BioCodec is trained in a channel-agnostic manner. How does it perform on datasets with channel counts that differ substantially from those used during pretraining—either much larger or much smaller?

- Does the window length used during pretraining influence performance on downstream tasks?

- Is the Transformer across $M$ and $T$ dimensions fine-tuned during downstream adaptation, and how many parameters are involved in the fine-tuning process?

- Does the number of codebooks used in pretraining affect the overall performance or representation quality of BioCodec?

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
This paper introduces BioCodec, a neural codec–based self-supervised framework for representation learning from biosignals (with a particular focus on EEG in this paper). The authors adapt Residual Vector Quantization (RVQ) which is known to be previously successful in neural audio codecs (e.g., SoundStream, EnCodec) to neurophysiological time-series data, i.e., EEG.

### Strengths
1. The paper is well-written and easy to understand. 
2. The idea of treating biosignals as “neural audio” and tokenizing them with a codec-based RVQ bottleneck is a fresh perspective.

### Weaknesses
1. The central idea of utilizing RVQ as a bottleneck for biosignal encoding can be considered as a direct implementation of SoundStream/EnCodec into EEG, with minimal adaptation. The paper contributes no theoretical or algorithmic advance specific to neurophysiology, nor any modification to quantization, loss, or architecture that meaningfully differentiates it from audio codecs.

2. The model is pre-trained only on single-channel waveforms and therefore it does not learn any spatial dependencies which is the most critical aspect of EEG. Deferring spatial modeling to a separate transformer after pre-training defeats the notion of an EEG foundation model. This is because the second transformer is meant to “learn” relationships between electrodes (spatial dependencies), but this happens only after the main pre-training step. Therefore I believe the model doesn’t truly serve as a foundational EEG model, because it never learns the full spatiotemporal structure that defines EEG signals. 

3. The paper does not contribute to understanding why neural codecs might benefit biosignal representation learning. There is no theoretical analysis, no discussion of latent geometry, and no exploration of differentiable quantization behavior under noise. The paper lacks theoretical justification for why RVQ discretization should yield physiologically meaningful representations beyond empirical evidence. As such, the work remains an engineering transfer rather than a contribution to machine-learning fundamentals. 

4. The empirical improvements over recent baselines (e.g., LaBraM, CBraMod) are marginal and often within the reported variance. So it is hard to justify the requirement of neural codec inspired framework for the biosignals like EEG.

### Questions
1. Could the authors discuss whether cross-subject generalization was tested, and how the single-channel RVQ scales to multi-electrode variability?
2. Could the authors provide theoretical justification related to neural codecs being applied to biosignal representation learning  (see weakness 2)

### Soundness
2

### Presentation
3

### Contribution
2
