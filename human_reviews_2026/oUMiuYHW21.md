# Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Current foundation models for electroencephalography (EEG) rely on architectures adapted from computer vision or natural language processing, typically treating neural signals as pixel grids or token sequences. This approach overlooks that the neural activity is activated by diverse sparse coding across a complex geometric topological cortex. Inspired by biological neural mechanisms, we propose the Unified Neural Topological Foundation Model (Uni-NTFM), an architecture rooted in three core neuroscience principles. In detail, to align with the brain's decoupled coding mechanism, we design the Heterogeneous Feature Projection Module. This module simultaneously encodes both time-domain non-stationary transients and frequency-domain steady-state rhythms, ensuring high quality in both waveform morphology and spectral rhythms. Moreover, we introduce a Topological Embedding mechanism to inject structured spatial priors and align different sensor configurations onto a unified latent functional topography, effectively reconstructing the geometry of brain regions. Furthermore, we achieve functional modularization and sparse coding efficiency of biological networks by constructing the Mixture-of-Experts Transformer network. This dynamic routing mechanism assigns different signal patterns and tasks to specialized neural subnetworks, and effectively preventing task interference while increasing the model capacity to record-breaking 1.9 billion parameters. Uni-NTFM is pre-trained on a diverse corpus comprising 28,000 hours of EEG data, and outperforms existing models across nine distinct downstream tasks under both linear probing and fine-tuning settings, demonstrating that aligning model architecture with neural mechanisms is significant to learn universal representations and achieve generalizable brain decoding.} Our code is available at \url{https://anonymous.4open.science/r/Uni-NTFM-0924}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose UNI-NTFM which consists of double branch encoding in both frequency- and time-domain, which then merge again with the embedded raw data stream for computing correspondences across natural and spectral features. This is further enhanced by topological embedding where brain region level variations are explicitly modeled. The final MOE / RoPE augmented Transformer performs token mixing. These architectural contributions are evaluated across several representative EEG databases, against both task-specific and foundation baselines, revealing competitive performance of the proposed method.

### Strengths
- Extensive experiments with confidence level clearly shown.
- Well motivated set of modules used as the backbone, proved to be quite effective.
- Beautiful figures
- Complete ablation studies and comparisons.

### Weaknesses
- Limited novelty: joint frequency- and time-domain encoding is not new. MoE and RoPE are not new in the EEG foundation model domain either. Topological embedding adds novelty, yet with this alone, model side contribution might be weak.

- Limited performance gains: performance difference across baselines and within model size variations is statistically not significant in most cases. I'd suggest to tone down the significance statement in the introduction and conclusion.

- Model-size scalability is limited. The reason for this in unclear.

### Questions
- I like the topological embedding idea. However, analyses and the written information on this invention are somewhat limited. What are the brain area / region that you considered? Depending on the disease type and its characteristic functional connectivity the embedding might vary a lot. Have you tried to differentiate this?

- I appreciate the confidence intervals that show clearly the gain from the proposed method. However, most results don't seem to be statistically significant compared to second best, on both baseline comparison and ablations. How is the confidence computed? Perhaps it's in the paper but I cannot identify it. Can you show the P-values, for instance, the best performance against the second best performance? If this is not significant under like 95% confidence interval, then the authors may have to tone down the performance gain.

- On a similar line, statistical significant tests within different model sizes might reveal that they are not very different. Would this reveal a model / data side scalability?

- Did you use the MU transfer? How do you know the convergence in model size scalability at the billion parameter regime is due to limited data, or suboptimal hyperparameter tuning?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes Uni-NTFM, which is a neuro-inspired foundation model on EEG brain signal data, which separates and then fuses time-signal, frequency-signal and raw-signal views, and then injects explicit spatial topology, which makes the electrodes from different montages become unified and then tries to scale capacity with a MoE Transformer. Uni-NTFM is a pretrained model with a dual-domain masked reconstruction objective, which is the combination of reconstructing time and frequency targets on around 28k hours of EEG signals. Across 9 downstream datasets, this model beats task-specific models and prior EEG foundation models under both probing and fine-tuning.

### Strengths
I found the originality of this work very good, which rethinks Foundation models on EEG around the physics of the signal and tries to decouple time, frequency and raw views cleanly, and they're also explicitly unifying electrode montages with a hierarchical topological embedding, and also scaling with a sparsely-activated MoE, instead of importing a single-stream and dense transformer. 
As to the quality, I found the proposed method interesting and strong, as the pretraining corpus is large and heterogeneous, and the evaluation spans 9 downstream tasks with both linear probing and fine-tuning. 
The clarity of the work is also easy to follow, and the significance is notable.

### Weaknesses
While I found the proposed method strong and interesting, I'd like to raise some weaknesses that came to my mind after reading the work. From my understanding, the authors already claimed the montage unification, but I don't see any direct test on unseen layouts or on systematically missing channels, so at this point, I'd suggest considering a small cross-montage transfer and k-missing-channel stress test, which could make the proposed topology more concrete. 

---
Another point that I'd like to make is about the Mixture of Experts (MoE), which I think is well-motivated, but from my understanding, the work does not show that different experts specialize on distinct EEG patterns, so I'd suggest considering some diagnostics, such as gate distributions and entropy by class, per-expert token-routing heatmaps, or maybe an ablation that freezes the most-used experts, which will help in demonstrating real specialization and justifying your MoE claim. 

---
One additional thing that I'd like to raise is about the scaling part, which I found super interesting, and the authors have done a great job on it. I just suggest making it more explicit by showing how accuracy changes as size grows, normalized by compute and/or data, and try to report training cost and inference speed. 

---
One last thing that I'd like to raise. I see that the pretraining mixes 9 EEG datasets, then the authors evaluate on datasets from the same families. From my understanding, this can blur whether gains come from true cross-corpus generalization or from seeing very similar data. I'd suggest considering leave-one-corpus-out, somehow similar to leave-one-out cross-validation (LOOCV), but I don't mean LOOCV at this point; I mean removing one family during pretraining and then testing on e.g., TUAB to prove the robustness to unseen corpora, which could rule out easy distribution matches.

### Questions
I'd suggest that the authors engage with the points that I raised in the Weaknesses section. As to the scaling part, I'd suggest downstream accuracy vs pretraining hours (e.g., 10%, 25%, 50%, 100%) plots and vs labeled fraction used in fine-tuning (1%, 5%, 10%, 100%) on a few tasks, so there would be a possibility to see if benefits come from data, model size or both and how label-efficient Uni-NTFM really is. 

---
I have another question and would be curious to know the authors' comments. Do you really need 3 separate streams, or would a single well-designed stream be enough? I was thinking maybe adding an equal-compute baseline that feeds stacked Short-Time Fourier Transform (STFT) patches (magnitude + phase) into one Transformer with MoE, plus early-fusion and late-fusion variants, and compare head-to-head with Heterogeneous Feature Projection Module (HFPM) + Dual-domain Cross-attention Module (DCM). Also, I'd suggest reporting matched parameters/FLOPs and an identical training budget for a fair test.

### Soundness
3

### Presentation
3

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
This paper presents Uni-NTFM, a large-scale EEG foundation model designed to unify temporal, spectral, and spatial representations of brain signals. The model incorporates three key innovations: (1) a Heterogeneous Feature Projection Module that decouples time-, frequency-, and raw-signal representations; (2) a Topological Embedding mechanism that encodes spatial priors across different electrode standards; and (3) a Mixture-of-Experts (MoE) Transformer that scales efficiently and captures heterogeneous EEG patterns. Uni-NTFM achieves state-of-the-art performance across nine downstream tasks (clinical, cognitive, and BCI datasets) under both linear probing and fine-tuning.

### Strengths
1. The paper is well-organized and clearly written, making it easy to follow and understand.
2. The experimental evaluation is comprehensive, with systematic ablations of each module and thorough benchmarking against strong baselines under both linear-probing and full fine-tuning settings.
3. Overall the model achieved better performance compared with other baselines.

### Weaknesses
1. The paper claims “zero-shot transfer” performance, but the evaluation setup corresponds to linear probing, where a supervised linear classifier is trained on the downstream data with the pretrained backbone frozen. While this setup measures representation quality and transferability, it does not constitute true zero-shot inference. The terminology should be clarified to avoid confusion.
2. The paper claims that the proposed MoE design enables efficient scaling of model capacity. However, this claim is not well substantiated by the current ablation study (if I understand correctly, the current ablation simply removes the MoE module). I would recommend replacing the MoE with a comparably sized dense Transformer and conducting a matched-compute comparison to more convincingly demonstrate the efficiency benefit. Since efficient scaling is listed as one of the main contributions, a more detailed analysis or ablation is necessary to support this claim.
3. As mentioned above, it would be more informative to include the parameter counts of both the proposed model and the baselines in Tables 1, 2, and 3. This would make it easier to assess model efficiency and fairly support the efficiency-related claims.
4. From an architectural perspective, although the current design and combination of each module reflects thoughtful consideration, the novelty of the proposed framework appears limited. The overall design builds upon standard Transformer and MoE backbones, and the introduced modules (e.g., heterogeneous feature projection, topological embedding, domain calibration) largely combine existing ideas from temporal–spectral decomposition and spatial encoding. While the integration is well executed and practically useful, it represents more of a system-level consolidation rather than a fundamentally new architectural contribution.

### Questions
1. How would performance change without EEG data augmentation?
2. Could you provide ablations on the different position embedding components (channel-wise, region-wise, relative index within region) combination?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Uni-NTFM, a foundation model for EEG that addresses domain-specific challenges through a decoupled time-frequency architecture, topological electrode embeddings across different standards, and a Mixture-of-Experts Transformer, pretrained on 28,000+ hours of diverse EEG data. The model demonstrates "superior performance" over existing task-specific methods and foundation models across nine downstream tasks under both linear probing and fine-tuning settings.

### Strengths
- The decoupled time-frequency architecture, topological electrode embeddings, and MoE-based routing represent thoughtfu

- Extensive experiments across nine diverse downstream tasks under both linear probing and fine-tuning.

Outstanding presentation featuring clear writing and accurate usage of technical terminology, effectively applying concepts such as foundation models, zero-shot learning, and fine-tuning, which are often misused in the community.

- The 1.9B parameter scale, 28,000+ hours of diverse pretraining data, and superior performance establish a meaningful benchmark for EEG foundation models with potential clinical and neuroscience applications.

All necessary code, data, and information are provided to ensure reproducibility.

### Weaknesses
- The ablation study is conducted on a near-saturation task [1](e.g., TUAB), where the baseline model shows low performance; even a lightweight model such as EEGNet (~2k parameters) could outperform it (see [2]).

- The reported improvement over existing methods is marginal, limiting the practical significance of the contribution.

The novelty of this approach is limited, as it appears to be a combination of existing methods [3,4].

[1] Kiessner, A. K., Schirrmeister, R. T., Boedecker, J., & Ball, T. (2024). Reaching the ceiling? Empirical scaling behaviour for deep EEG pathology classification. Computers in Biology and Medicine, 178, 108681.

[2] Darvishi-Bayazi, M. J., Ghaemi, M. S., Lesort, T., Arefin, M. R., Faubert, J., & Rish, I. (2024). Amplifying pathological detection in EEG signaling pathways through cross-dataset transfer learning. Computers in biology and medicine, 169, 107893.

[3] Zhang, X., Zhao, Z., Tsiligkaridis, T., & Zitnik, M. (2022). Self-supervised contrastive pre-training for time series via time-frequency consistency. Advances in neural information processing systems, 35, 3988-4003.

[4] Kuruppu, G., Wagh, N., & Varatharajah, Y. (2025). Eeg foundation models: A critical review of current progress and future directions. arXiv preprint arXiv:2507.11783.

### Questions
- Given that you have already provided different model sizes, would it be possible to create a performance scaling plot versus the number of parameters?

Do we really need billions of parameters for the current size of EEG data?

- Would it be better to plot the ablation tables for a clearer and faster understanding?

### Soundness
3

### Presentation
3

### Contribution
2
