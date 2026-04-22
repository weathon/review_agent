# TF-JEPA: Predictive Alignment of Time–Frequency Representations Without Contrastive Pairs

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Learning generalizable representations from multivariate time series is challenging due to complex temporal dynamics, distribution shifts, and the difficulty of effectively designing contrastive pairs. We introduce TF-JEPA, a noncontrastive self-supervised method that leverages predictive alignment to integrate representations from the time and frequency domains without relying on negative sampling. Specifically, TF-JEPA utilizes dual online encoders for time and frequency domains, each paired with its own momentum-updated target encoder, embedding both views into a stable and unified latent space. Unlike conventional contrastive methods, this predictive approach enables full end-to-end fine tuning for downstream adaptation. Experimental results on diverse real world datasets, including sleep EEG classification, gesture recognition, mechanical fault detection, and biosignal-based muscle response classification, demonstrate that TF-JEPA matches or surpasses contrastive and time frequency consistency baselines. TF-JEPA improves macro F1 scores by up to 8.6 percentage points while also reducing GPU memory consumption by approximately 35%. These findings illustrate the promise of predictive alignment as a broadly applicable and modality agnostic framework for self supervised learning beyond traditional contrastive methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TF-JEPA, a non-contrastive method for learning shared time-frequency representations from unlabeled time series. It reduces GPU memory, and sometimes improves cross-dataset transfer.

### Strengths
- Non-contrastive self-supervised method based on time-frequency representations
- Well-written simple paper
- Good results on cross-domain transfer

### Weaknesses
- Somewhat limited novelty as JEPA and predictive alignment is not new. The benchmarks are also not novel as they're borrowed from TF-C paper.
- Since results are mixed, the paper would benefit from studies on more datasets. 
- As [3] shows, TF-C can be used for additional downstream tasks, such as clustering and anomaly detection. TF JEPA has not been evaluated in these tasks, so generalizability of the learned representations is limited.
- The authors claim in the abstract that the method is "broadly applicable", but no findings on pretraining and fine tuning has been demonstrated, see for example recent works [1],[2]. Given the growing literature in time series foundation models, it would be beneficial to compare against such models, and even examine pretraining.

[1] S. Gao et al, "UniTS: A Unified Multi-Task Time Series Model", NeurIPS 2024

[2] M. Goswami et al, "MOMENT: A Family of Open Time-series Foundation Models", ICML 2024

[3] X. Zhang et al, "Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency", NeurIPS 2022

### Questions
- Why does TF-JEPA underperform on some datasets? Where is it expected to do better and why?
- Would TF-JEPA be able to be used as a pretraining objective on heterogeneous time series, similar to UniTS/MOMENT? 
- How does TF-JEPA do on clustering and anomaly detection tasks?

### Soundness
3

### Presentation
3

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
The submission introduces TF-JEPA (Time-Frequency Joint Embedding Predictive Architecture), a non-contrastive self-supervised method designed for learning generalizable representations from multivariate time series. This approach is inspired by JEPA and utilizes predictive alignment to integrate representations from the time and frequency domains, thereby avoiding reliance on negative sampling or contrastive pairs.
TF-JEPA employs a momentum-based dual-encoder architecture. It uses dual online encoders (one for time, one for frequency), each paired with its own momentum-updated target encoder (updated via Exponential Moving Average, EMA). The core objective is predictive: the online time encoder predicts the target frequency representation, and the online frequency encoder predicts the target time representation using lightweight multilayer perceptrons (predictors). This cross-view prediction framework uses a BYOL-style cosine loss. Experiments are conducted on diverse time-series benchmark datasets.

### Strengths
The paper clearly articulates its motivation, methodology, and results through highly structured explanations. Specifically:

• The contributions are explicitly summarized as threefold, listing them as: (1) proposing a momentum-based dual-encoder architecture that aligns representations without negative pairs, (2) demonstrating that this predictive alignment yields transferable embeddings suitable for end-to-end fine-tuning, and (3) achieving competitive or superior performance on multiple real-world benchmarks.

• The TF-JEPA methodology is clearly broken down into three core design choices: Dual EMA targets, Lightweight predictors, and End-to-end fine-tuning, making the mechanism easy to follow.

• The experimental section employs transparent benchmarking, stating clearly that the setup enforces identical classifier architectures, latent dimensions, and optimizer hyperparameters across methods during fine-tuning "to ensure direct comparability".

### Weaknesses
W1.

The proposed TF-JEPA does not consistently surpass existing baselines. Based on Table 1, the proposed method only achieve superior performance on 2 out of 4 scenarios. 


W2.

The motivation for positioning non-contrastive approaches as superior to contrastive learning is currently unconvincing. The manuscript only discusses vanilla contrastive learning methods that rely on manually designed augmentations and negative sample selection (Line 51). However, there is a growing body of recent work that develops automated or adaptive augmentation strategies within contrastive learning for time series. For example, [1] optimizes the data augmentation automatically within the contrastive learning paradigm. Additional related methods are also summarized in the survey [2]. The authors should update the discussion to reflect these advances and more accurately contextualize the comparison.

[1] Augmentation Blending with Clustering-Aware Outlier Factor: An Outlier-Driven Perspective for Enhanced Contrastive Learning. KBS 2024.
[2] Unsupervised representation learning for time series: A review. 2023.

### Questions
Please see Weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes TF-JEPA, a non-contrastive self-supervised learning method for time-series representation learning that aligns time-domain and frequency-domain embeddings through predictive alignment rather than contrastive objectives. The approach uses dual online encoders with momentum-updated target encoders and a BYOL-style cosine loss. The authors evaluate on four cross-dataset transfer tasks and report improvements over contrastive baselines (TF-C, TS-TCC) in some settings, along with reduced GPU memory consumption (~35%).

### Strengths
* The elimination of quadratic similarity matrices reduces GPU memory from 5.3 GB to 3.4 GB (35% reduction) and achieves 1.6× speedup on targeted benchmarks. 

* The framing of time and frequency as "lossless, invertible views" (Section 2.1) is conceptually clear and provides principled motivation for predictive alignment over contrastive repulsion.

* Overall studying cross-modal (EEG-->EMG) transfer is interesting.

### Weaknesses
Major Issues:

1. Missing SimSiam baseline: Omits foundational non-contrastive method [1]. Please include comparison with SimSiam as it also does not need negative samples.

2. Unclear problem scope: Conflates 3 settings, and hence the experiment design is unclear. For instance, is the setup: (1) general representation learning—experiments need to include UCR/UEA standard benchmark datasets; (2) cross-domain transfer (indicated by the FD from A to B)—it needs to compare with some generalization techniques [2]; or (3) cross-modal (EEG -->EMG)—then it needs to be compared with appropriate multimodal learning baselines?

3. Unjustified design choices: (1) Why magnitude-only FFT? No ablation on phase discarding, which encodes diagnostic info in EEG/ECG/EMG. (2) Why full-window FFT vs. STFT? Dismisses STFT as "limiting reuse" (lines 92–93) without evidence. Non-stationary signals need time-frequency localization. Additional experiments: Ablate (a) magnitude-only vs. magnitude+phase FFT, and (b) full-window vs. STFT.

4. Incomplete resource metrics: Reports GPU memory and wall-clock time only. The authors should consider reporting FLOPs, MACs, and latency across sequence lengths.

5. Overall, I feel the paper has limited novelty and an unclear motivation. Perhaps the authors could clarify the motivation more clearly and investigate it more rigorously. The cross-modal transfer is an interesting setup, and studying how physiological signals with different underlying generators can be mapped to each other is a complex but valuable task. I can see such an investigation advancing fields like wearable health monitoring, where modalities may evolve—for instance, from accelerometer-based HAR to EMG-based gesture recognition—and cross-modal transfer could be useful.

6. Key equation at line 131: Unclear—no notations are explained and very hard to follow.
7. Define EMA abbreviation: Please define before using it in the paper.
8. Information-theoretic claim (lines 163–164): "Minimizing cosine distance bounds mutual information" is not rigorously justified for the time-frequency setting. Clarify as intuition or provide formal derivation.



[1] Simsiam, CVPR 2021


[2] Diversify, ICLR 2023

### Questions
Please see Weaknesses above.

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
This paper introduce a non contrastive SSL method called TF-JEPA to perform frequency-aware representation learning on time series datasets. The network is based on momentum-based dual encoder in time and frequency space, getting rid of negative sampling in previous contrastive learning methods. The method is applied on real world time series datasets, showing performance improvement over previous methods.

### Strengths
- The writing is clear and easy to follow, the technical details are complete and concisely presented.
- The proposed method show decent performance improvements on two out of four datasets.

### Weaknesses
- Experiments are all performed on small scale datasets. Performance improvement is not consistent. The authors should also consider non-transfer experimental settings.
- Limited technical innovation. This work basically replaced TF-C with a loss function that does not require negative sampling, and repeated the same experiments. If the idea of the work is just to remove negative sampling, the paper should also benchmark with more reconstruction-based methods like [1].
- Limited baselines and benchmarks. The paper should consider more state-of-the-art pre-trained models, such as [2].

[1] Liu, Ran, Ellen L. Zippi, Hadi Pouransari, Chris Sandino, Jingping Nie, Hanlin Goh, Erdrin Azemi, and Ali Moin. "Frequency-aware masked autoencoders for multimodal pretraining on biosignals." arXiv preprint arXiv:2309.05927 (2023).

[2] Wang, Jiquan, Sha Zhao, Zhiling Luo, Yangxuan Zhou, Haiteng Jiang, Shijian Li, Tao Li, and Gang Pan. "Cbramod: A criss-cross brain foundation model for eeg decoding." arXiv preprint arXiv:2412.07236 (2024).

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
2
