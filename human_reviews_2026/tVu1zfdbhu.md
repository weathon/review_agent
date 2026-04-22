# Wavelet-Driven Masked Multiscale Reconstruction for PPG Foundation Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Wearable foundation models have the potential to transform digital health by learning transferable representations from large-scale biosignals collected in everyday settings. 
While recent progress has been made in large-scale pretraining, most approaches overlook the spectral structure of photoplethysmography (PPG) signals, wherein physiological rhythms unfold across multiple frequency bands.
Motivated by the insight that many downstream health-related tasks depend on multi-resolution features spanning fine-grained waveform morphology to global rhythmic dynamics, we introduce Masked Multiscale Reconstruction (MMR) for PPG representation learning -- a self-supervised pretraining framework that explicitly learns from hierarchical time–frequency scales of PPG data.
The pretraining task is designed to reconstruct randomly masked out coefficients  obtained from a wavelet-based multiresolution decomposition of PPG signals, forcing the transformer encoder to integrate information across temporal and spectral scales.
We pretrain our model with MMR using  ~17 million unlabeled 10-second PPG segments collected from over ~32000 smartwatch users largely in naturalistic field settings, ensuring high variability and ecological validity. 
On 11 of 13 diverse health-related tasks, MMR trained on large-scale wearable PPG data outperforms or matches state-of-the-art open-source PPG foundation models, time-series foundation models and other self-supervised baselines. 
Extensive analysis of our learned embeddings and systematic ablations underscore the value of wavelet-based representations, showing that they capture robust and physiologically-grounded features. 
Together, these results highlight the potential of MMR as a step toward generalizable PPG foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Masked Multiscale Reconstruction, a self-supervised pretraining framework for wearable PPG signals. The key idea is to transform each 10-s segment into multi-resolution wavelet coefficients, randomly mask 75% of small temporal patches across sub-bands, and train a ViT encoder–decoder to reconstruct the missing coefficients. The authors pretrain on ~17M PPG segments from ~32K smartwatch users, then evaluate frozen embeddings with simple downstream heads across 13 health-related tasks. They report consistent gains or parity versus strong baselines and competitive/open-source models, with notable improvements on free-living hypertension and PVC detection; they also include data/model scaling and ablations.

### Strengths
1.	On free‑living hypertension, MMR achieves 68.1 AUROC versus 62.9 for TF‑C and 60.9 for SimCLR; for PVC detection, MMR reaches 82.0 AUROC versus 70–74 for baselines. Across classification tasks, MMR also outperforms PaPaGei and Chronos on average. 
2.	Treating PPG in the time–frequency domain with wavelets matches physiological non‑stationarity and multi‑scale rhythms, and masked reconstruction across sub‑bands encourages cross‑scale feature sharing. 
3.	Ablations and scaling analyses offer practical guidance: Haar wavelets often perform best; moderate decomposition depth (L3–L4) helps hypertension/PVC; smaller patches (1×25) preserve fine events; and increasing data volume improves key tasks.

### Weaknesses
1.	The core idea is masked autoencoding in a time–frequency domain with a fixed DWT tokenizer. Closely related frequency-aware MAE variants already exist, and the contribution feels incremental.
2.	Using a predefined DWT front-end may underfit device-specific characteristics and miss discriminative sub-band boundaries. It will be better to explore the learnable filterbanks or jointly trained spectral tokenizers.
3.	Converting multi-rate sub-bands into a 2D coefficient map via interpolation risks aliasing and amplitude distortion. The interpolation kernel, anti-aliasing, and per-band normalization choices are not justified or stress-tested.
4.	Uniform random masking likely biases learning toward low-frequency shortcuts. Missing band-aware, curriculum, event-centric, or rarity-aware masking that would force recovery of high-frequency/rare cues (e.g., ectopy).
5.	No clear rebalancing (e.g., class weights, focal loss), threshold optimization, or cost-sensitive analysis; this can suppress performance on rare events.
6.	Tables 1–2 show that MMR is not SOTA on several classification endpoints and is sometimes matched or outperformed by MMR-Light or baselines. The paper does not explain these gaps or provide a failure-mode, leaving the causes of underperformance unclear.

### Questions
1.	Could you clarify how your method is substantively different from existing frequency-aware MAE variants and why fixed-DWT masked autoencoding constitutes a novel contribution? 
2.	Will you evaluate learnable filterbanks or jointly trained spectral tokenizers to address potential underfitting to device-specific characteristics and missed discriminative sub-band boundaries?
3.	How were the interpolation kernel, anti-aliasing, and per-band normalization chosen when converting multi-rate sub-bands into a 2D map, and can you provide stress tests showing minimal aliasing/amplitude distortion? 
4.	Can you justify uniform random masking and report results for band-aware, event-centric/curriculum, or rarity-aware masking that better targets high-frequency/rare cues?
5.	Did you apply class rebalancing (class weights/focal loss), threshold optimization, or cost-sensitive analysis? If not, can you add these and report their impact on rare-event metrics?
6.	In Table 1&2, MMR is not SOTA on several classification endpoints and in a few cases is even matched or exceeded by MMR-Light or baselines (SimCLR/TF-C). Could you explain these gaps and provide a brief failure-mode analysis?

### Soundness
3

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
The paper proposes Masked Multi-Scale Reconstruction, a method for PPG representation learning. In this approach, PPG signals are decomposed into wavelet coefficients using a discrete wavelet transform, which are then masked and subsequently reconstructed to learn meaningful representations. The model is trained on a large proprietary dataset and evaluated against several baselines across 13 downstream tasks.

### Strengths
- The idea of learning PPG representations through the reconstruction of masked DWT coefficients is quite interesting.
- The paper includes thorough experiments, with useful case studies and ablation analyses beyond standard downstream evaluations.
- The paper is well-written and easy to follow.

### Weaknesses
**Evaluation:** The diversity and number of devices used are essential for interpreting the results, and reporting these details would not compromise anonymity. However, the use of a closed-source dataset limits the interpretability and reproducibility of the findings. Specifically: (1) the number of datasets from which each downstream task is derived remains unclear, and (2) it is uncertain whether the training and test data originate from the same devices. Furthermore, several public PPG datasets [1] could be utilized for external validation to strengthen the evaluation.

This point is particularly important because, when the dataset is fixed and only the training method is varied (Table 2), the average performance improvement of the proposed approach is relatively small (\~1%) compared to open-source models trained on different data (\~3.5%). This suggests that, if the baselines were trained using the same data, their performance could be comparable to MMR, thereby weakening the claim of improved generalizability.

**Capture multi-resolution characteristics** such as: high-frequency transients, subtle waveform changes, and low-frequency rhythms such as respiration and circadian trend. Is there any supporting evidence or case study demonstrating that some of these characteristics are captured by MMR. Furthermore, many of the downstream tasks involve blood lab measurements, for which such multi-resolution features may not be directly relevant. This suggests a potential mismatch between the motivation of MMR and the choice of downstream tasks used for evaluation. In short, there could be a potential mismatch between the motivation of MMR and and the choice of downstream tasks used for evaluation.

**Experiments:** Many downstream tasks are strongly correlated with demographic variables. Therefore, it would be valuable to evaluate how the proposed approach performs on demographic prediction tasks such as age, sex, and BMI. Moreover, as the model is closed-source, these results and other relevant factors could be compared to the findings reported by Abbaspourazad _et al._ [2] for additional context.

Additionally, it would be insightful to examine the advantages of wavelet-based reconstruction through MMR compared to directly reconstructing the raw PPG signal. Such an experiment could help clarify the specific benefits of the wavelet decomposition stage and further strengthen the empirical analysis.

**Contribution beyond previous work:** While the proposed methodology is interesting, the contribution beyond prior work appears limited. (1) The approach is closed-source and comparable to [1], which leverages larger datasets to demonstrate that PPG representations can generalize to more than 50 downstream tasks. In [2], an open-source model is introduced using a morphology-aware self-supervised learning (SSL) strategy that generalizes across diverse tasks. Apart from the training framework, it is difficult to clearly distinguish this work in terms of experimental design, task selection, and case studies from previous research to justify publication. Moreover, the performance improvements of MMR are marginal compared to other SSL approaches trained on the same data. Considering these points, I lean toward a reject.

[1] Pillai, A., Spathis, D., Kawsar, F., & Malekzadeh, M. (2024). Papagei: Open foundation models for optical physiological signals. _arXiv preprint arXiv:2410.20542_.

[2] Abbaspourazad, S., Elachqar, O., Miller, A. C., Emrani, S., Nallasamy, U., & Shapiro, I. (2023). Large-scale training of foundation models for wearable biosignals. _arXiv preprint arXiv:2312.05409_.

### Questions
- What is the rationale behind using backbones of different sizes in Table 1? Can't MMR be matched to 5M parameters.
- Were other transforms beyond DWT considered for reconstruction?

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
This paper introduces MMR, a self-supervised pretraining framework for large-scale PPG foundation models. The approach decomposes PPG signals into hierarchical time–frequency representations via discrete wavelet transforms and trains a ViT-based encoder–decoder to reconstruct masked coefficients across scales. The method uses PPG segments from smartwatch users collected in the field. The pretrained representations are evaluated on 13 downstream cardiovascular and biomarker prediction tasks, where MMR strong results comparing with the baselines. Ablations explore the impact of these different components on the performance of the model.

I really like the future direction of exploring adaptive multiscale strategies.

### Strengths
1. The use of wavelet-based multiscale reconstruction as a masked modeling target is a strong conceptual contribution.
2. The diversity of the dataset, with 1h30 of data for each patient in unconstrained environments significantly increases applicability over prior foundation models trained on clean, clinical datasets.
3. The data pre-processing has a good balance between cleaning and maintaining as much data as possible.
4. The paper benchmarks across 13 diverse downstream tasks (clinical and physiological), both classification and regression.
5. Well presented ablation study. Embedding visualizations (t-SNE, silhouette) convincingly show physiological structure capture.
6. MMR-Light variant maintains strong performance with few parameters.
7. Paper is well written and easy to read. (except figures/tables, see below)

### Weaknesses
1. While effective, MMR’s novelty lies mainly in applying masked reconstruction to wavelet coefficients. The method reuses a ViT backbone with minimal architectural innovations.
2. Key preprocessing and DWT hyperparameters (e.g., sampling-rate normalization, interpolation scheme for coefficients) could be better detailed for reproducibility.
3. The baselines could be newer models (eg Chronos-Bolt instead of Chronos)
4. Clarity on the fixed parameters in the ablation study could be improved (what model size was used for the data scaling ablation and conversely what data size for the model scaling?
5. It is disputable if this is indeed a diverse population with "high variability" as the smart-watch carrying population is rather restricted.
6. The min-max range is very large, with 10-15% of the value and the models only being within a couple percentage points of each-other.

Clarity/formatting:
- Figure 2 axis titles are hard to read
- The tables could be improved for readability (grouping T.1+T.2, making a graph, etc.)

### Questions
1. Could MMR generalize to other biosignals (e.g., ECG, accelerometer)? Does the wavelet-based representation transfer across modalities?
2. How do performance trends change if downstream tasks are fine-tuned rather than frozen?
3. Were any measures taken to mitigate potential data leakage across users or sessions?
4. How does computational cost compare in terms of FLOPs and convergence speed?
5. Was any consideration given to using a diffusion model?
6. Have the baseline models (eg. Chronos) been finetuned on this task/with this data? Do you think this would be relevant considering the small performance gap between it and MMR?
7. How does the model's ability to isolate individual patients with higher patient-wise distance affect it's out-of-distribution abilities? Especially on new patients?
8. Any intuition on why different models performs as they do on the different tasks, as per model task-performance rankings are not consistent?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new self-supervised pretraining framework for PPG foundation models, which applies Discrete Wavelet Transform (DWT) to decompose raw PPG into multiple frequency bands, then trains a Vision Transformer encoder-decoder to reconstruct randomly masked wavelet coefficients. The authors pretrain on approximately 17 million 10-second PPG segments from over 32,000 smartwatch users and evaluate on 13 diverse health prediction tasks, including hypertension detection, arrhythmia classification, and blood biomarker prediction. The results show that MMR achieves competitive performance with existing PPG foundation models.

### Strengths
This paper tests 13 different downstream tasks spanning cardiovascular conditions (hypertension, PVC), metabolic markers (creatinine), and electrolyte imbalances, and provides strong evidence that the learned representations capture broadly useful information.

### Weaknesses
1. The core technical contribution lacks clear validation. While wavelets are positioned as the main innovation, the paper never isolates whether DWT actually drives the performance gains. Critically, the paper is missing the essential ablation: MMR with DWT versus MMR without DWT—the same masked autoencoder architecture and training procedure applied to patchified raw PPG time series instead of wavelet coefficients.

2. Results are mixed and claims are overstated. The abstract and conclusions describe MMR as achieving "superior" performance that "outperforms" baselines, but the actual results tell a more nuanced story.

3. A fundamental limitation undermines the foundation model premise. The ablation studies reveal a critical tension: different tasks benefit from different wavelet decomposition depths. Hypertension classification works best with level 3, while lab biomarkers like WBC and Sodium prefer level 5. This is not a minor implementation detail—it directly contradicts the paper's core premise of learning "robust, transferable features" that generalize across diverse tasks. The authors acknowledge this as a "key limitation" and suggest exploring "adaptive multiscale strategies" in future work, but this limitation is severe enough to undermine the approach's viability. 

4. The "multi-scale" framing oversells temporal coverage. The paper claims to capture features "spanning fine-grained waveform morphology to global rhythmic dynamics" and explicitly mentions "circadian trends" (line 81). However, the 10-second window fundamentally cannot capture true long-term physiological scales. The lowest frequency reliably analyzable in 10 seconds is approximately 0.1 Hz—anything slower, including the circadian rhythms mentioned, is completely inaccessible. What the model actually learns are multi-resolution representations within short windows: beat morphology (~1-10 Hz) to respiratory and brief heart rate variability patterns (~0.1-1 Hz).

### Questions
1. Could you explain how variable sampling rates (25-100 Hz) are handled, despite this being essential for interpreting the wavelet decomposition.

2. Do the same coefficient positions correspond to the same physiological phenomena across different sampling rates, or does rate variation change the meaning of features?

3. Will you release the weights of the pre-trained model and datasets used from pre-training and downstream tasks for reproducibility? The current datasets come from proprietary smartwatch data that cannot be disclosed, rendering exact reproduction impossible.

### Soundness
3

### Presentation
3

### Contribution
2
