# Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 2

## Abstract
Functional Magnetic Resonance Imaging (fMRI) provides non-invasive access to dynamic brain activity by measuring blood oxygen level-dependent (BOLD) signals over time. However, the resource-intensive nature of fMRI acquisition limits the availability of high-fidelity samples required for data-driven brain analysis models. While modern generative models can synthesize fMRI data, they often remain challenging in replicating their inherent non-stationarity, intricate spatiotemporal dynamics, and physiological variations of raw BOLD signals. To address these challenges, we propose Dual-Spectral Flow Matching (DSFM), a novel fMRI generative framework that cascades dual frequency representation of BOLD signals with spectral flow matching. Specifically, our framework first converts BOLD signals into a wavelet decomposition map via a discrete wavelet transform (DWT) to capture globalized transient and multi-scale variations, and projects into the discrete cosine transform (DCT) space across brain regions and time to exploit localized energy compaction of low-frequency dominant BOLD coefficients. Subsequently, a spectral flow matching model is trained to generate class-conditioned cosine-frequency representation. The generated samples are reconstructed through inverse DCT and inverse DWT operations to recover physiologically plausible time-domain BOLD signals. This dual-transform approach imposes structured frequency priors and preserves key physiological brain dynamics. Ultimately, we demonstrate the efficacy of our approach through improved downstream fMRI-based brain network classification.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a method to generate fMRI time-series data. The authors propose a three-step process.
The first step computes a time-frequency representation from the training data using a wavelet transform. Then the transformed signals are then transformed using a 2D Discrete Cosine Transform to further localize the low-frequency BOLD spectral components. Then they use a flow-matching approach that operates in the DCT space that relies on an image generation model (U-ViT) to parameterize and train the model that minimizes the "spectral flow-matching loss" to learn velocity fields in this space. This process further allows to sample the DCT coefficients by integrating the velocity flow, which is then inverse-transformed (both DCT and wavelet) to generate the reconstructed fMRI signals. The authors compare their method with several state of the art models. They also present an application that performs classification between HC and MDD patients.

### Strengths
Synthetically generating fMRI time-series data is still an open problem. Most approaches have tackled synthetic data generation in modalities such as ECG, EKG, etc., but fMRI time-series generation is complex and thus still remains challenging. Thus this paper makes a good contribution to this problem. 

While the DCT and DWT (more so for time-series) have been applied to fMRI time-series data individually, they have been done so primarily for filtering, preprocessing, noise/motion removal etc type applications. To my knowledge, the joint approach, i.e. first perform DWT for time-frequency analysis and then low-frequency signal compaction using DCT is new, especially for fMRI time-series application.

The idea of recasting flow-matching in the spectral domain instead of relying on diffusion models is somewhat novel. This is especially important for fMRI applications, as 4D time-series data is huge and the deterministic ODE-based flow matching idea is magnitudes in order more efficient than diffusion modeling. 

The authors present a method for learning the velocity fields of probability flows directly in the DCT space. They derive the spectral flow-matching loss by diagonalizing the Laplacian operator and then propose the probability flow in the DCT space (proposition 1). This is novel. 

The authors have provided the code for their method including experimental results and validation. This is a strength. However, also see the Questions below.

### Weaknesses
The main weaknesses in the paper are in the experimental results and aspects related to evaluation of the method on BOLD fMRI time-series applications. 

The generated fMRI signals are not validated using any physiological or neuroscientific basis. They are simply validated by context FID scores. Thus the validation is relying on comparing data distributions where the synthetic data is supposed to have come from. This is weak. 

Furthermore, using classification accuracy as a metric to discriminate HC against MDD is a complex high-level application, which may mask underlying subtleties of scanner, population, processing methods etc. Thus it is hard to make any judgement on the actual reconstructed time-series signals.

Ablation study is performed on omitting wavelet sub-bands. However, the authors don't comment on the dual-formulation advantage. I.e. is the real gain coming from the spectral representation itself or from flow-matching process, generating the velocity fields in the DCT space. 

Only a single dataset (REST-MDD) is used. The dataset contains 250 Healthy Controls (HC) subjects and 227 individuals diagnosed with MDD. This is a much limited dataset to test the method on. 

The claim "brain disorder identification" is both pretty general and strong. One could at best say, the method showed improved performance in classification after using the synthetic time-series on the MDD dataset. On that note, the authors don't mention what MDD is. It is supposed to stand for major depressive disorder. Instead they wrongly use the acronym MDD to denote a general brain disorder. Line 88 says: Our results show that DSFM demonstrates strong performance on unconditional and conditional spectral image synthesis, and achieves improvement in brain disorder (MDD) classification. This is an incorrect terminology. 

The paper visualizes the average connectivity patterns of real and synthetic connectivity patterns. Without knowing the error bars or any quantitative tests, it is difficult to tell if the synthetic data generation method has worked. Visually, one can see differences in anatomical locations in the connectivity patterns from real and synthetic data and even in their differences. Thus the method is not as accurate, but it is not clear by how much. 

For validation purposes, the method does not comment on frequency bands (LF vs HF) that are captured or accurately reflected in the synthetic data. Absence of such low-level measures makes the experimental impact of the contribution, difficult to judge. 

The method is compared to several state of the art time-series generation methods. But comparisons to the more recent rectified-flow matching methods are missing. This is important especially as this particular method does rely on a flow-based formulation.

### Questions
In the reproducibility statement, the authors have stated, "We provide the datasets, source code, and configurations for all key experiments, including instruc-tions on how to preprocess data and train the models at https://anonymous.4open.science/r/DSFM-
123C" However the link returned "File not Found".

What happens if the DCT step is short-circuited? I.e. the flow velocity fields are learnt on the DWT components?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Dual-Spectral Flow Matching (DSFM) for fMRI time-series generation. It converts BOLD signals into a dual-spectral representation using DWT and DCT, then applies a standard flow-matching model to synthesize time-series data for data augmentation on a depression vs control classification task.

### Strengths
1. Combining DWT and DCT with flow-matching for fMRI synthesis is a creative approach.

2. The demonstrated use of fMRI data augmentation for enhancing downstream classification performance is interesting, and the proposed application appears promising.

### Weaknesses
Major:

1. The experiment scope is a bit limited. All experiments are done on a single dataset (REST-meta-MDD). There is no cross-dataset or cross-site validation, so the generalization and robustness claims are weak.

2. The paper repeatedly mentions that the proposed spectral-domain flow matching is more efficien, but this claim is not supported by theoretical analysis or empirical evidence. It would strengthen the work if the authors could elaborate on what “efficiency” specifically refers to (e.g., faster sampling, fewer function evaluations, or reduced computational cost) and provide quantitative comparisons to substantiate this point.

3. The paper does not include experiments that clarify whether transforming fMRI signals into the frequency domain is strictly necessary. A direct comparison with a latent-space flow matching baseline, i.e., encoding fMRI ROI time series into latent embeddings with transformer encoder without explicit DWT/DCT transforms, would help determine whether the spectral transforms genuinely contribute to the observed improvements in performance or efficiency, or if similar gains could be achieved through compact latent representations.

4. The authors note that their primary goal is not to achieve the best sample quality metrics but rather to improve cFID and conditional modeling of spatiotemporal patterns. However, this statement seems to suggest that the proposed DSFM does not outperform recent baselines (e.g., Diffusion-TS) in unconditional generation quality (Table 1). Can the author elaborate more on this, why cFID is more important than other metrics in this scenario?


Minor:
1. For the connectivity matrices in Figure 4, would be more informative to include colorbar to show the value scale.

2. There might be format error at line 291 and line 308

3. While the technical content is interesting, the overall presentation could be improved for clarity, especially the method and theory parts.

4. Potential typo at line 136-137 "Thus forming a full wavelet..."

5. There is another recent baseline for fMRI synthesis [1]. Although it is not open-sourced yet, it would still be worthwhile to discuss it in the related work section to provide a more complete context

[1] Synthesizing Realistic fMRI: A Physiological Dynamics-Driven Hierarchical Diffusion Model for Efficient fMRI Acquisition, ICLR 2025

### Questions
1. The paper fix the wavelet type to Haar. Did author experiment with different wavelet bases (e.g., Daubechies, Coiflet, Symlet)? Since different wavelets have distinct time–frequency localization and smoothness properties, this omission makes the transform choice look arbitrary. 
2. What's the rationale of using zig zag flattening rather than normal flattening in Figure 1(step 3)? What would be the difference in model performance
3. What could be the possible reason that, in Figure 4, the synthetic connectivity matrices appear more clustered and uniformly distributed compared to the real ones?

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
4

### Summary
This paper proposes a framework named Dual-Spectral Flow Matching (DSFM) for fMRI BOLD signal generation. The method jointly constructs dual-spectral representations through Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT), then introduces Spectral Flow Matching (SFM) in the DCT domain to achieve fMRI generation by modeling the heat diffusion process in the frequency domain.

### Strengths
Innovative dual-spectral architecture:
The paper introduces a dual-spectral generative framework combining DWT (Discrete Wavelet Transform) and DCT (Discrete Cosine Transform).
This design elegantly captures both temporal–frequency representations and spectral sparsity of fMRI signals, offering a novel perspective distinct from previous time-domain or FC-based generation approaches.

Solid theoretical derivation:
The work provides a rigorous and mathematically sound derivation linking stochastic partial differential equations (SPDE) to frequency-domain probability flow ODEs, establishing a coherent theoretical foundation for the proposed spectral flow matching.
The formulation is complete and clearly connects to existing flow-matching paradigms.

The experimental design is comprehensive, and the analysis is detailed.

### Weaknesses
Lack of cross-dataset validation and statistical significance tests:
Experiments are conducted only on the REST-meta-MDD dataset, without evaluation on other publicly available fMRI datasets such as ABIDE [1] and HCP [2].
This raises concerns about the model’s generalizability across acquisition protocols and subject populations.
Moreover, the reported improvements lack statistical significance testing, making it difficult to assess the robustness of the observed gains.


[1] Heinsfeld A S, Franco A R, Craddock R C, et al. Identification of autism spectrum disorder using deep learning and the ABIDE dataset[J]. NeuroImage: clinical, 2018, 17: 16-23.

[2] Smith S M, Beckmann C F, Andersson J, et al. Resting-state fMRI in the human connectome project[J]. Neuroimage, 2013, 80: 144-168.

Limited physical and physiological interpretability:
Although the proposed DSFM introduces a dual-spectral flow matching framework with physically inspired dynamics, its neuroscientific meaning remains limited. The model is clean and physically inspired from a signal engineering perspective; however, it lacks true mechanistic grounding in neuroscience. Although the paper claims to generate “physiologically plausible” fMRI signals, no quantitative evidence is provided to support this claim. Metrics such as power spectral density (PSD) alignment, hemodynamic response function (HRF) consistency, or spatiotemporal smoothness are not evaluated. Including these analyses would substantially strengthen the physiological credibility of the proposed approach. In essence, DSFM produces spectrally and statistically plausible fMRI-like time series rather than neurodynamically meaningful brain processes. I would encourage the authors to further elaborate on the neuroscientific significance of generating such fMRI signals, for example, what new understanding or insight about brain dynamics can be obtained from this generation process, or how such generated fMRI signals could be practically useful?

Overly dense pipeline illustration:
Figure 1 is visually overloaded. Several components such as "Reorder & Truncation", "SNR scale + Add Noise" and "Zig Zag Flattening" are insufficiently explained in the text, which reduces clarity. Simplifying or modularizing the figure, or adding explicit textual references, would improve readability.

### Questions
Check the weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel approach to fMRI time-series generation based on dual spectral transformations and a learned flow in spectral space. The authors detail the spectral transforms and the flow’s training procedure, and they experimentally validate the method on a single dataset.

### Strengths
1/ The subject of study is of utmost interest.

2/ The approach is well described, and most technical details are provided. Figure 1 is an excellent summary.

3/ The range of evaluation metrics and the ablation study make the validation comprehensive (apart from the limited number of datasets).

### Weaknesses
Weaknesses summary

Some rationale for the design is missing, and the approach lacks comprehensive empirical validation across additional datasets.

Major concerns

1/ The choice to work in the spectral domain and to cascade spectral transforms is not clearly justified. I am trying to understand why you chose the DCT after a wavelet transform. Apart from the diagonalization of the Laplacian and the simpler truncation it enables, why introduce a second spectral projection and why this one?

2/ If B is too large, it will squash multiple ROIs into a single patch, which in turn will make it impossible for the flow to learn the interactions between those ROIs (since they are collapsed). This is not specifically discussed in the paper, and the experimental values for B are not provided.

3/ In the contributions, you state that you "forming a unified dual-spectral image transform to capture both global and local spatiotemporal and spectral features for fMRI BOLD signal generation"; It is not clear what specifically differentiates your method from prior approaches in capturing ROI-to-ROI interactions (see 2/). Why should these interactions be better captured in spectral space rather that in the time domain? Note that classical fMRI analysis choose to capture the ROIs' interplays in the time domain.

4/ Truncation is crucial since it removes part of the information (note that, at this stage, most artifacts should already have been filtered out by preprocessing); yet it is not specifically discussed in the paper, and the experimental values are not provided. Additionally, for short TRs (< 0.7 s), it remains unclear whether truncation would not be detrimental.

5/ Lines 60 to 63: It is not clear why your approach, which operates in the spectral domain, differs from methods that operate on preprocessed time series in its ability to filter out confounds; the preprocessing pipeline should have already remove cardiac pulsations, respiratory cycles, and motion-induced artifacts. Please clarify which components remain to be removed (see 4/).

6/ The main issue with this paper is the lack of an exhaustive experimental validation. It should be evaluated on at least half a dozen datasets, spanning different TRs, spatial resolutions, and tasks (e.g., multiclass subject prediction).

Minor comments

1/ In the abstract, "preserves key physiological brain dynamics" is an over-claims evidence.

2/ There are numerous typos, for example: "we stitches the patches", "Fig. 1 provide an overview", etc

3/ Some acronyms are undefined or defined too late, for example: MDD, STFT, etc. Even for some common acronyms, a brief reminder would improve readability

4/ Please cite Kawahara et al., 2017 when mentioning BrainNetCNN so the provenance is clear.

5/ The results are somewhat ambiguous: the proposed approach is not SOTA in terms of "generation quality" (Table 1), yet it achieves SOTA prediction performance (Table 3). This discrepancy is surprising and is not discussed.

Grading explanation

The paper lacks comprehensive empirical validation, and some key rationales for the proposed approach are missing.

### Questions
1/ Why choose a heat dissipation process? Does avoiding conventional isotropic diffusion help capture interactions between patches (~ ROIs) more effectively?

2/ At a general level, the paper seems to suggest that learning a complex generative distribution (and sampling from it) is easier than discriminating between two of its marginals. This is not a typical results (cf. On Discriminative vs. Generative classifiers: A comparison of logistic regression and naive Bayes), can you elaborate?

3/ Please report the computational cost of your approach: approximate training time, hardware used, and runtime for sample generation.

4/ How do you denormalize the wavelet coefficients (c.f. "We further perform componentwise normalization to accentuate"). Please detail the inverse step.

5/ Do you have an explanation for the discrepancy between Table 1 and Table 3 (non-SOTA generation quality vs. SOTA prediction performance)?

### Soundness
2

### Presentation
3

### Contribution
3
