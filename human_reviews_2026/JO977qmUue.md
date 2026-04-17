# Regularization via Invariant Patterns: Temporal Domain Randomization for Human Activity Recognition

- Decision: Reject
- Scores: 6, 2, 0, 4

## Abstract
Synthetic data has become a common strategy to address data scarcity in Human Activity Recognition (HAR). However, models trained on synthetic samples often overfit to spurious features, leading to a substantial domain gap when transferred to real-world data. To address this challenge, we propose Regularization via Invariant Patterns (RIP), a novel data-centric method that extends the idea of domain randomization to the temporal domain. RIP augments time-series windows by "framing" them with invariant (constant-valued) patterns, compelling models to focus on informative signals rather than irrelevant temporal context.
Evaluated across five HAR datasets, four classifiers, and more than 2,000 experiments, RIP consistently improves F1 scores, achieving gains of up to +53 percentage points (over +160\% relative improvement) compared to synthetic baselines — often matching or surpassing real-data baselines. Beyond synthetic scenarios, RIP also boosts performance in real-only training settings, highlighting its broad applicability. Both theoretical analysis and empirical results show that RIP stabilizes weight updates and enhances calibration, all without modifying model architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a regularization method (RIP) that augments each time-series sample by framing it with constant-valued segments. The goal is to reduce reliance on spurious temporal context, particularly in synthetic data settings. RIP is architecture-agnostic and improves generalization in both synthetic-to-real (TSTR) and real-only (TRTR) setups. Evaluation across five HAR datasets and multiple models shows consistent gains in F1 and calibration.

### Strengths
- Addresses an underexplored and practical problem: generalization from GAN-generated HAR data.  
- The method is simple, lightweight, and broadly applicable.  
- Experiments are extensive, covering multiple datasets and models with consistent improvement.  
- Calibration and stability analyses complement accuracy results.  
- The approach requires minimal tuning and introduces no architectural complexity.

### Weaknesses
1. **Terminology clarity**  
   `S`, `R`, `TRTR`, Duplication factor `i` and constant value `γ` appear early (e.g., line 62) without explanation. These should be briefly defined when first introduced.

2. **Related Work is incomplete**  
   The paper omits relevant work in all three subsections. For example:  
   - Domain randomization in time-series: *PhASER* (Mohapatra et al., ICLR 2025), Cutout (Yang et al., ESWA 2022)  
   - Regularization for HAR: *TS-TCC* (Eldele et al., IJCAI 2021), *AdvMask* (Yang et al.), *DynamicMixup* (Guo et al., EEE Transactions on Multimedia 2023)  
   - Domain generalization: *AFFAR* (Qin et al., ACM TIST 2022), *DDLearn* (Qin et al., KDD 2023), *MixStyle* for wearable HAR (Napoli et al., ESANN 2025), *SynthAct* (Schneider et al., ICRA 2024)
   These should be cited to contextualize the contribution.

3. **Algorithm design and justification**  
   Algorithm 1 may be redundant given the detailed text. The `γ` and `i` choices are limited and appear heuristic; it is unclear whether the theoretical claims generalize beyond these settings.

4. **RIP–GAN separation**  
   RIP does not rely on synthetic data, yet the paper is structured as if its utility is specific to TSTR. The weaknesses of TLCGAN data (e.g., unstable transitions or spurious context) should be explicitly discussed.

5. **Framing effect and data scaling**  
   Adding `2i` constant windows increases input length substantially. It is unclear whether this effectively increases the training set size or acts as implicit duplication. This may partially explain the observed gains and should be clarified.

6. **Model selection**  
   The models used (DClassifier, TS-Classifier, TSBF, TSRF) are not SOTA. While they cover a spectrum of architectures, modern high-performing models such as InceptionTime or Transformer-based HAR models are not included. This limits claims about general performance. Future comparisons with stronger baselines are encouraged.

7. **Baselines and fairness**  
   Mixup, Cutmix, and DRO underperform in Table 4. These methods are not designed for low-fidelity synthetic data. Their inclusion requires justification.  
   For ℓ₁ and ℓ₂ (Table 3), regularization strength selection is not explained. In some cases (e.g., WISDM), these degrade performance severely, which weakens the comparison.  
   TS-Classifier is relatively weak; it is unclear whether RIP’s gains generalize to stronger models.

8. **Figures and formatting**  
   Table 1 and Table 2 captions are inconsistent; only the first specifies TSTR. Both appear to exceed text width.  
   Figure 2 caption mislabels line colors as models, though they represent `γ` values. Section headings should use consistent phrasing (“Robustness and fairness”).

9. **Reproducibility and compute**  
    CPU/GPU details are missing from the runtime section, limiting interpretability. “Due to space limitation” (line 336) is mentioned, but the main text does not use the full page limit.

10. **Novelty**  
The framing concept is intuitive and related to prior masking or context dropout techniques, but its application to synthetic HAR is novel. The contribution is incremental methodologically but solid in empirical utility.

### Questions
- Could you clarify how the duplication factor i and constant value γ were selected? Were they tuned per dataset, and do the theoretical claims (e.g., variance reduction) hold under other values?

- RIP appears applicable to real data as well. Could you explain more concretely what properties of the TLCGAN-generated samples make RIP particularly helpful in the synthetic setting?

- Does the framing effect introduced by RIP increase the number of training tokens seen by the model? If so, how do you control for the effect of sequence length versus regularization?

- For Table 3, were the ε values for ℓ₁ and ℓ₂ regularization selected via tuning, or fixed across datasets? WISDM results drop significantly—how should we interpret this?

- Have you considered applying RIP to more recent time-series architectures such as InceptionTime or TST?

For all questions, please refer to the Weakness part

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
3

### Summary
In this paper, the authors aim to improve the model performance when trained on synthetic data in the domain of human activity recognition. Specifically, the authors design a Regularization via Inviriatn Patterns, a data-centric method to force the model to focus on informative signals rather than irrelevant context. For evaluation, the authors utilize multiple real world dataset for human activity recognition. Results highlight the advantages of this method with a significant performance gain. Overall, the topic of this paper is interesting, some concerns, however, limit the contribution of this paper.

### Strengths
[1] The topic is interesting. Werable device based human activity recognition is a promising direction, and synthetic data for model training is also interesting. 

[2] The future work is briefly discussed and could benefit future research. 

[3] The experiments are conducted on 5 real-world datasets.

### Weaknesses
[1] It is not sure what is the technical challenge to adapt the domain randomniation to the temporal domain. 

[2] Since this paper addresses the problems in activity recognition, but there are no most recent activity recognition papers discussed or compared in the experiment. This could be an issue. 


[3] In Figure 2, “each color denotes a model” might be each bar denotes a model? If the x-axis is correctly labeled. 

[4] The writing and presentation could be further improved. For example, I could not find the experimental results for boosting performance in real-only training settings, which is indicated in the abstract. 

[5] Detailed information about the dataset used is missing. For example, since this is a classification problem, how many classes exist in each dataset should be introduced. Also, the sampling rate, window size are also missing. 

[6] In line 101, it is not clear what does it mean “constant windows frame the original window”. 

[7] How to construct the augmented dataset is not clear. After reading section 2, it is still not quite clear. Is it via concatenating the constant and the original sensor data? 

[8] COuld you also discuss the limitations or challenges of the propose method?

### Questions
Questions are provided on weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces Regularization via Invariant Patterns (RIP), a data-centric method that extends the idea of domain randomization to the temporal domain. RIP augments time-series windows by ”framing” them with invariant (constant-valued) patterns, compelling models
to focus on informative signals.

### Strengths
1. This paper is well organized.
2. The overall logic is clear.

### Weaknesses
1.	The paper claims that RIP can alleviate the synthetic-to-real domain gap, but all experiments are limited to the "train synthetic → test real" or "train real → test real" mode, and do not include cross-subject, cross-device, or cross-dataset tests. To demonstrate that RIP truly promotes domain generalization, these more challenging cross-domain experiments should be included.
2.	The authors compare RIP with methods such as ℓ₁/ℓ₂, Mixup, Cutmix, and DRO, but these methods typically operate at the sample or feature level, rather than the temporal dimension. The authors should also compare RIP with temporal masking methods that are more structurally similar to it. Furthermore, the paper does not specify whether all baselines were retuned or used default settings, which may affect fairness.
3.	The experimental tables were not formatted consistently, especially Tables 1 and 2, which were misaligned.
4.	The authors derive the effect of RIP on the hidden state variance using a simplified linear RNN model (φ(x)=x). This analysis neglects the effects of nonlinear activation functions and common structures such as batch normalization and attention. Since RIP is mainly applied to deep models (ConvLSTM, Transformer-based), the variance constraint conclusions derived only for linear RNNs are difficult to generalize. It is recommended to supplement the influence of nonlinear terms.

### Questions
1. In the theoretical section, the authors state that RIP “drives weights toward greater uniformity”, citing the Kolmogorov–Smirnov statistic as evidence. However, uniformly distributed weights do not necessarily represent better generalization or stability, and the paper lacks discussion on the causal relationship between this metric and generalization performance.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a new data augmentation technique for regularising time-series models for human activity recognition. The inspiration of this method is taken from image-based data background randomization methods. In summary, the method simply adds an invariant pattern (e.g., a matrix of 1s or 2s ...) earlier and later to the original data. The performance has been shown to be better compared to using their method and with other data augmentation techniques. Extensive experiments and analysis have been done. Some limitations of the paper appear to be incremental novelty and clarity of presentation, and under description of the method.

### Strengths
The results in the paper show that their method of regularization via invariant patterns as a data augmentation improves the performance of time-series models by a large margin. 

The paper presents extensive experiments and validation of their approach and comprehensively analyzes various scenarios, including checking the validity of invariant patterns and simple duplications in input patterns.

### Weaknesses
The method is less clearly explained. Figure 1 is a simple illustration of the paper, prepended and appended to a time-series data matrix. The explanation of how this pattern is prepended and appended to the original timeseries should be shown in Fig. 1 or in a separate figure.  Algorithm 1 reads redundant. Better to explain the method properly on Page 2. Hard to locate where the Tables are referred to in the paper. Paper writing/presentation needs improvement. The hyperparameter setting information in the main paper is below very minimal. The hyperparameter settings for the algorithms should be mentioned in the main paper rather than pushed into an excessively large appendix.

### Questions
What is SRIP(L) in Figures 3 and 4?
Different parameter values for RIP have been used across datasets. How can one know how to set these values? These appear to be experimentally found values? Confirm.

On some datasets, such as MHEALTH RIP, they do not perform significantly better than L or L2 regularisation, but for some, they perform excessively well. Please explain.

The results presented in the paper all appear to have been executed by the authors themselves. Are there any comparisons with state-of-the-art results?

### Soundness
3

### Presentation
2

### Contribution
2
