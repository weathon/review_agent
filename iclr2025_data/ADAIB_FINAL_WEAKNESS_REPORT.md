# AdaIB Weakness Analysis Report
## Insights from ICLR 2025 Related Papers

**Date**: 2026-04-08  
**Source**: 368 paper reviews  
**Relevant Papers Found**: 68  
**Weaknesses Analyzed**: 600

---

## Executive Summary

This analysis maps reviewer concerns from 68 ICLR 2025 papers to specific challenges that AdaIB (Adaptive Information Bottleneck for Multimodal Attribution) addresses.

---

## Key Concerns & Relevant Weaknesses


### Evaluation & Benchmarking

**Relevance to AdaIB**: Must clearly compare against existing methods

**Mentions in Literature**: 94 weakness statements from 41 papers

#### Representative Examples


1. "The proposed method lies between two adversarial attack approaches: - Standard adversarial attacks: where each image is attacked individually. - Patch attacks: where a malicious patch is added to the image. The authors should include comparisons with methods from both of these fields."
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: comparison


2. "The paper lacks a main results section with comparisons to other methods and focuses primarily on ablation studies."
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: comparison


3. "Qualitatively, the reconstructed samples in Fig. 2 show artifacts and significant deviations from the original samples. 4. Misleading Comparisons"
   - **From**: Balancing Token Efficiency and Structural Accuracy in LLMs I
   - **Keywords**: comparison


### Robustness to Distribution Shift

**Relevance to AdaIB**: AdaIB must handle diverse image-text distributions

**Mentions in Literature**: 47 weakness statements from 30 papers

#### Representative Examples


1. "It is not convincingly demonstrated that imperceptible, or even near-imperceptible, multi-attacks are possible in a natural setting, where the attacked samples are drawn from the same distribution that the classifier was trained on. # Theoretical claims (Section 3) I am not convinced by the argument"
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: distribution


2. "The paper discusses only 3D-CNNs and does not clarify such a limitation in the abstract or contributions. The framework's adaptations do not include different architectures, such as transformers or even CNN-LSTM. Papers: VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking Bidirectional "
   - **From**: PrAViC: Probabilistic Adaptation Framework for Real-Time Vid
   - **Keywords**: edge


3. "The code for PrAViC is mentioned but not provided (there’s a broken link to GitHub), raising concerns about reproducibility. Furthermore, the authors report only scalar accuracy values, without error bars or statistical tests, making it difficult to evaluate the robustness and generalizability of th"
   - **From**: PrAViC: Probabilistic Adaptation Framework for Real-Time Vid
   - **Keywords**: robust


### Theoretical Justification

**Relevance to AdaIB**: Need strong foundation for adaptive mechanism

**Mentions in Literature**: 27 weakness statements from 17 papers

#### Representative Examples


1. "The perturbation should be bounded as commonly accepted in the literature."
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: bound


2. "The theoretical argument, that the multi-attack implies the existence of many distinct "cells" with different classifications in the neighborhood of each sample, is flawed."
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: theoretical


3. "It is not convincingly demonstrated that imperceptible, or even near-imperceptible, multi-attacks are possible in a natural setting, where the attacked samples are drawn from the same distribution that the classifier was trained on. # Theoretical claims (Section 3) I am not convinced by the argument"
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: theoretical


### Trade-off Analysis

**Relevance to AdaIB**: IB requires balancing compression vs. reconstruction

**Mentions in Literature**: 23 weakness statements from 14 papers

#### Representative Examples


1. "The work is positioned as a variant of Early Classification of Time Series (ECTS), especially given the focus on “real-time processing” and early decision-making. However, there is no mention or evaluation of ECTS algorithms [1, 2, 3], making it difficult to understand how PrAViC compares to existin"
   - **From**: PrAViC: Probabilistic Adaptation Framework for Real-Time Vid
   - **Keywords**: balance


2. "While early exit or ECTS models should be evaluated on *both* classification performance and earliness (such as NET), some experiments report only accuracy (e.g., Table 1), or only PrAViC's NET (e.g., Table 2). Ideally, early exit models should not be evaluated based on a single point in the accurac"
   - **From**: PrAViC: Probabilistic Adaptation Framework for Real-Time Vid
   - **Keywords**: tradeoff


3. "In section 4.1, it is not clear whether the hyper-parameter tuning is conducted on the Algonauts training set or the authors’ training set (90% of the Algonauts training set, different for each of the 4 folds). This is an important clarification to make because in the first case the encoding test se"
   - **From**: Dynamics Based Neural Encoding with Inter-Intra Region Conne
   - **Keywords**: tuning


### Handling Noisy/Unreliable Data

**Relevance to AdaIB**: Core problem AdaIB addresses

**Mentions in Literature**: 22 weakness statements from 11 papers

#### Representative Examples


1. "Line 249-250: "We experiment with starting with real images X1,X2,...,Xm is any different from starting with random noise samples.": check grammar"
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: noise


2. "The reader is not convinced that the improvements observed in brain predictivity are caused by the learned connectivity, and not just by obtaining a more robust or less noisy embedding by considering other regions. There are also inconsistencies in the results regarding this aspect; in figure 5c it "
   - **From**: Dynamics Based Neural Encoding with Inter-Intra Region Conne
   - **Keywords**: noisy


3. "For the ImageNet-C robustness results, while I greatly appreciate the use of statistical significance, I would like to know what these results look like when averaging across the corruption severity levels (as is done with the canonical ImageNet-C benchmark). The reason being, from a practical persp"
   - **From**: Modeling Divisive Normalization as Learned Local Competition
   - **Keywords**: corrupt


### Alignment Assumptions

**Relevance to AdaIB**: AdaIB must verify image-text alignment quality

**Mentions in Literature**: 18 weakness statements from 10 papers

#### Representative Examples


1. "The classifier is "locally linear" within each ball around each $X_i$. In other words, only one planar decision boundary appears within the $L_\infty$ ball around each $X_i$. This means that the true value of N is $N=2$. Note that this also implies that each sample individually is vulnerable to an $"
   - **From**: Multi-attacks: A single adversarial perturbation for multipl
   - **Keywords**: assume


2. "and inter- connectivity priors, especially their structure, and interpretation of their results beyond their contribution to improved representational alignment, and the value of their learned weights in figure 5c."
   - **From**: Dynamics Based Neural Encoding with Inter-Intra Region Conne
   - **Keywords**: alignment


3. "The authors claim that they only test on AlexNet, VGG-16, and a two layer CNN due to the better alignment of these architectures with biology; however, I find this argument extremely weak. Some of the best models on BrainScore are in fact very deep networks that are predictive of neural responses in"
   - **From**: Modeling Divisive Normalization as Learned Local Competition
   - **Keywords**: alignment


---

## Actionable Insights for AdaIB Paper

1. **Robustness to Distribution Shift**
   - Include experiments on distribution shifts and unseen data
   - Test on datasets with different image-text alignment quality
   - Demonstrate robustness across diverse visual domains

2. **Handling Noisy/Unreliable Data**
   - Provide mechanisms for detecting misaligned pairs
   - Show how adaptive weighting down-weights unreliable samples
   - Include analysis of behavior under varying noise levels

3. **Trade-off Analysis**
   - Thoroughly analyze compression-fitting trade-off
   - Justify hyperparameter choices with ablation studies
   - Provide guidance for practitioners on tuning parameters

4. **Alignment Assumptions**
   - Explicitly state all assumptions about image-text alignment
   - Demonstrate verification methods
   - Show robustness when assumptions are violated

5. **Theoretical Justification**
   - Provide convergence analysis of adaptive mechanism
   - Include information-theoretic bounds
   - Connect to existing attribution theory

6. **Comprehensive Evaluation**
   - Compare against existing multimodal attribution methods
   - Include benchmarks on standard robustness datasets
   - Provide cross-dataset generalization results

---

## Top Papers by Reviewer Concern Density

Based on the analysis, papers with the most relevant reviewer insights:


1. **Multi-attacks: A single adversarial perturbation for multiple images and target labels**
   - Topics: label noise, vision language robustness
   - Weakness mentions: 23

2. **Balancing Token Efficiency and Structural Accuracy in LLMs Image Generation by Combining VQ-VAE and Diffusion Tokenizers**
   - Topics: vision language robustness
   - Weakness mentions: 22

3. **PrAViC: Probabilistic Adaptation Framework for Real-Time Video Classification**
   - Topics: vision language robustness
   - Weakness mentions: 19

4. **Dynamics Based Neural Encoding with Inter-Intra Region Connectivity**
   - Topics: vision language robustness
   - Weakness mentions: 19

5. **Modeling Divisive Normalization as Learned Local Competition in Visual Cortex**
   - Topics: noisy misaligned data
   - Weakness mentions: 17

6. **RFWave: Multi-band Rectified Flow for Audio Waveform Reconstruction**
   - Topics: label noise
   - Weakness mentions: 16

7. **Data Shapley in One Training Run**
   - Topics: label noise
   - Weakness mentions: 15

8. **Pseudo-Non-Linear Data Augmentation via Energy Minimization**
   - Topics: label noise
   - Weakness mentions: 15

9. **Safety Alignment Should be Made More Than Just a Few Tokens Deep**
   - Topics: vision language robustness
   - Weakness mentions: 14

10. **Feedback Favors the Generalization of Neural ODEs**
   - Topics: vision language robustness
   - Weakness mentions: 14

11. **Adaptive Length Image Tokenization via Recurrent Allocation**
   - Topics: vision language robustness
   - Weakness mentions: 14

12. **Simplifying, Stabilizing and Scaling Continuous-time Consistency Models**
   - Topics: label noise
   - Weakness mentions: 14


---

## Conclusion

Reviewers consistently emphasize:

- **Robustness**: Across distributions, domains, and data quality levels
- **Clarity**: Clear presentation of assumptions and mechanisms  
- **Evidence**: Comprehensive comparisons and ablation studies
- **Theory**: Mathematical justification and analysis

AdaIB should address these themes proactively.
