# EFDiff: Frequency-informed Diffusion for Extreme-value Time Series Generation

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Time-series generation, which aims to produce realistic synthetic sequences that preserve temporal dynamics, is essential for data augmentation and practical applications. However, existing methods often fail to capture extreme-value distributions, which are crucial in domains such as finance, climate, and energy. This limitation mainly stems from overall-fit objectives and smoothing procedures that distort extreme-event structures. To address these challenges, we propose EFDiff, a frequency-informed extreme-aware time-series generation framework. Unlike conventional approaches that focus on long-tail preservation in the time domain, EFDiff adopts a frequency-domain perspective by integrating a frequency-based disentanglement strategy into diffusion models. The key innovation lies in an Extreme Component, which consists of two key modules: (i) Extreme-Frequency Extraction (EFX), which constructs a global extreme-frequency dictionary that characterizes potential extreme patterns via event-driven local analysis and multi-metric integration based on the proposed concept of extreme-contributing frequencies; and (ii) Extreme-Frequency Generation Enhancement (EFGEN), which includes a novel Transformer-based Soft Frequency Selection Network to identify relevant frequencies and effectively model extreme patterns during
the denoising process. Extensive experiments on five real-world datasets across six evaluation metrics demonstrate that EFDiff consistently achieves strong overall generation quality and substantially improves the fidelity of extreme-value generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes EFDiff a frequency-informed diffusion model for time series generation. In order to tackle the extreme value aware time-series generation, EFDiff has EFX and EFGEN module to enhance the generation quality.

### Strengths
1. This paper addresses the extreme value awareness in the unconditional time series generation problem, which is usually overlooked. 
2. This paper carefully discussed what an extreme value means in the encoded latent space. 
3. A good amount of details is provided in the methodology section.

### Weaknesses
1. While the model aims to optimize the extreme value awareness, there is a lack of suitable evaluation metrics for such scenarios. It is not clear how EFDiff works in such a case. 
2. The dataset selected seems to be low-dimensional; it is unclear how they may perform in a high-dimensional dataset like energy (which is commonly used for unconditional generation). 
3. The proposed extreme components have a couple of modules. A comprehensive ablation study could provide additional evidence for the necessity of each proposed module.

### Questions
1. The evaluation is mainly conducted over the general distribution. Another commonly used evaluation metric is discriminative accuracy (TimeGAN, TimeVAE, Diffusion-TS, etc.). Will the awareness of extreme values potentially harm the discriminative accuracy?
2. Is there a particular reason why PODT is calculated over Xenc? It seems that defining extreme value in the raw space is more natural than in the latent space.
3. How costly is this additional module in terms of time?

### Soundness
2

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
This paper introduces EFDiff (Extreme-Frequency Diffusion), a frequency-informed diffusion model specifically designed for extreme-value time-series generation. The core novelty lies in a dedicated Extreme Component comprising two modules: (1) Extreme-Frequency Extraction (EFX), which heuristically constructs a global dictionary of extreme-contributing frequencies via an event-driven, multi-metric analysis; and (2) Extreme-Frequency Generation Enhancement (EFGEN), a Transformer-based soft frequency selector that adaptively models these extreme patterns during the denoising process. Extensive experiments on multiple public datasets demonstrate the proposed approach's empirical superiority in faithfully capturing extreme-value structures.

### Strengths
- The authors tackle the important yet underexplored problem of robustly modeling extreme-value distributions in time series, a domain with significant real-world implications.

- The framework is logically structured around its two main components - EFX for dictionary construction and EFGEN for adaptive frequency selection - providing a clear mechanism for frequency-informed generation.

- The approach demonstrates consistent empirical improvements across multiple public datasets, particularly on tail-sensitive metrics (e.g., KL and JS divergence), validating its specific strength in capturing rare, high-impact events.

### Weaknesses
- Although the authors present a data-driven analysis linking extreme events to the phase alignment of high-frequency components, the evidence is primarily observational. The core formulation lacks theoretical derivation or rigorous statistical validation, making the crucial connection between specific frequency-phase structures and extreme-value formation insufficiently justified.

- The design of the EFX module relies on a set of heuristic metrics (PLV, Cw, Lw) whose theoretical necessity and connection to extreme-value formation remain unclear. While a limited ablation study is provided in the appendix, it offers minimal insight into the relative contribution or necessity of each metric. Consequently, the formulation appears empirically motivated rather than theoretically grounded.

- The reported performance of the baseline FIDE (an existing extreme-aware diffusion framework) is substantially lower than both its original paper and even the performance of simpler models like Diffusion-TS. This unexpected inversion strongly suggests possible inconsistencies in dataset selection or experimental configuration. As a result, the fairness and reliability of the comparative evaluation are seriously questionable, making the claimed performance gains difficult to interpret objectively.

- The overall novelty is arguably incremental compared to existing frequency-based diffusion frameworks (e.g., FIDE and Diffusion-TS). The "Extreme Component" largely represents an extension rather than a fundamentally new generative paradigm. Furthermore, the paper’s exposition is often verbose and unclear, using overly technical terminology with limited visual or intuitive explanation of the proposed mechanism.

### Questions
Please refer to the Weaknesses part.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EFDiff, a frequency-informed diffusion model tailored for realistic and extreme-value time series generation. The central innovation is the explicit extraction and modeling of extreme-contributing frequencies using a two-stage approach. Namely, 
1. Extreme-Frequency Extraction (EFX), which creates a global dictionary of frequency combinations aligned with extreme events based on event-driven local analysis and multiple custom metrics.
2. Extreme-Frequency Generation Enhancement (EFGEN), which leverages a Transformer-based selection network to modulate the generation process by prioritizing those frequencies during the diffusion denoising steps.

Comprehensive experiments are conducted across five diverse real-world datasets and six evaluation metrics, showing EFDiff achieves strong performance in overall and extreme-value distribution fidelity.

### Strengths
The paper addresses a genuinely important and under-explored challenge: generating time series that authentically capture rare/extreme events rather than merely “smooth” approximations, a critical issue in domains like climate science and finance. The decomposition into trend, seasonality, and explicitly modeled extreme components and precise mathematical formulation of each step, especially the definition and scoring of extreme-contributing frequencies. The use of a Transformer-based soft frequency selection (EFGEN) is thoughtfully justified for its flexibility in frequency combination selection, and offers potential for extensibility.

### Weaknesses
While the contributions above, some key recent works that deeply involve frequency-domain diffusion (especially those fully formulating time series generation directly in the frequency domain) are not cited or contrasted:

Crabbé, Jonathan, et al. "Time series diffusion in the frequency domain." Proceedings of the 41st International Conference on Machine Learning. 2024.

Chi, Guoxuan, et al. "RF-diffusion: Radio signal generation via time-frequency diffusion." Proceedings of the 30th Annual International Conference on Mobile Computing and Networking. 2024.

The technical distinction of EFDiff vs. prior frequency-domain diffusion models is somewhat muddied. For example, extreme-aware frequency generation, dictionary construction, and frequency-based selection are quite similar in recent works (e.g. TS-Diff, and the above mentioned RF-diffusion).

The boundary between "extreme" and simply "high-frequency noise" is not sharply addressed, and the dictionary building process could be more rigorously described.

The paper heavily relies on the idea of phase alignment, yet the precise definition of "nearly aligned phases" is not fully formalized. Additionally, while amplitudes and phases are extracted from DFT/IDFT in the extreme component, there is minimal discussion regarding how uncertainty, non-stationarity, or multivariate extensions affect this decomposition.

The comparison suite is lengthy, but experiments are missing recent and directly comparable frequency-domain diffusion models (Crabbé 2024, Chi 2024, Gao 2024)

Gao, Jiaxin, Qinglong Cao, and Yuntian Chen. "Auto-regressive moving diffusion models for time series forecasting." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 16. 2025.

Generally, the paper positions itself primarily against time-domain and partially frequency-aware models, but omits closely related frequency-domain diffusion papers, even though the text explicitly frames its contributions in the frequency domain. This makes it difficult to see exactly what is unique beyond combining known decomposition ideas with a new frequency‑scoring triplet and a soft‑selection module. The novelty is moderate, primarily due to the combination of event-driven frequency dictionaries and soft selection within diffusion, as well as the specific triad of frequency-contribution scores. The distinctiveness versus contemporaneous frequency-domain diffusion work needs to be argued and evaluated more directly.

### Questions
Can the authors provide more detail on the construction and selection process for frequency combinations in EFX? How is the combinatorial explosion of possible combinations managed? Are there tradeoffs in computational cost or redundancy?

What are the computational and sensitivity implications of the joint scoring ($S_w$) with learned $\alpha$, $\beta$ superpositions? 

When choosing K-top combinations, do stability or redundancy issues arise, and how were they resolved in practice?

What is the value or motivation behind the threshold $\varepsilon$? Is there an adaptive mechanism?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
EFDiff proposes a frequency-informed diffusion framework for extreme-value time-series generation. The authors argue that conventional models fail to preserve extreme events due to smoothing bias and lack of frequency-domain awareness. Their method decomposes time series into trend, seasonality, and an explicit “Extreme Component.” This component is built via two modules: Extreme-Frequency Extraction (EFX), which constructs a global dictionary of frequency combinations characterizing extreme events using a novel dynamic thresholding strategy (PODT) and three contribution metrics; and Extreme-Frequency Generation Enhancement (EFGEN), a Transformer-based soft selector that retrieves relevant frequencies during denoising. Experiments on five real-world datasets show improved fidelity in both overall and extreme-value distributions, particularly in KL divergence.

### Strengths
1. The paper provides a principled frequency-domain perspective on extreme-value generation, grounded in empirical observation that phase-aligned high-frequency components drive extremal behavior—an insight not well exploited in prior time-series generative models.

2. The proposed EFX module introduces a multi-metric, phase-aware scoring mechanism that combines PLV, amplitude contribution, and background contrast, offering a more nuanced view of frequency relevance than simple amplitude-based selection.

### Weaknesses
1. The theoretical foundation for why specific frequency combinations cause extremes is underdeveloped; the paper asserts phase alignment matters but offers no formal proof or rigorous signal-theoretic justification beyond illustrative examples.

2. The PODT thresholding method, while adaptive, lacks comparison to standard EVT baselines like GEV or GP fitting, making it unclear whether the claimed “superior extreme identification” is merely an artifact of an arbitrary threshold design.

3. EFGEN’s “soft frequency selection” is essentially a standard cross-attention mechanism over a precomputed dictionary, offering negligible architectural novelty—this is repackaged retrieval, not a breakthrough in diffusion conditioning.

4. The ablation study fails to isolate the impact of the frequency-domain formulation itself; no baseline is tested that applies EFX-style metrics in the time domain, so the claimed superiority of the frequency view remains unsubstantiated.

5. Several equations are ambiguously defined: in Eq. (17), the summation over $f^{text}∈F^{text}$ conflates scalar frequencies with vector-valued combinations, and the IDFT implementation is never reconciled with the continuous cosine superposition in Eq. (3). It's a serious notational inconsistency.

6. The paper claims 55.7% KL improvement on Stocks but omits statistical significance testing or confidence intervals across seeds, casting doubt on the robustness of the reported gains.

### Questions
Please refer to the Weaknesses section for critical concerns that must be addressed regarding theoretical grounding, evaluation protocol, methodological novelty, and mathematical consistency.

### Soundness
3

### Presentation
2

### Contribution
2
