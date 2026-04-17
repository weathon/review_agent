# You Are What You Train: Rethinking Training Data Quality, Targets, and Architectures for Universal Speech Enhancement

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Universal Speech Enhancement (USE) aims to restore the quality of diverse degraded speech while preserving fidelity. Despite recent progress, several challenges remain. In this paper, we address three key issues. (1) In speech dereverberation, the conventional use of early-reflected speech as the training target simplifies model training, but we found that it still harms perceptual quality. We therefore apply time-shifted anechoic clean speech as a simple yet more effective target. (2) Regression models preserve fidelity but produce over-smoothed outputs under severe degradation, while generative models improve perceptual quality but risk hallucination. We introduce a two-stage framework that effectively combines the strengths of both approaches, inspired by a recent theoretical finding. (3) We study the trade-off between training data scale and quality, a critical factor when scaling to large, imperfect corpora. Experimental results demonstrate that using time-shifted anechoic clean speech as the learning target significantly improves both speech quality and downstream automatic speech recognition (ASR) performance, while the two-stage framework further boosts quality without compromising fidelity.  In addition, our model demonstrates strong language-agnostic capability, making it well-suited for enhancing training data in other speech generative tasks. To ensure reproducibility, the code will be made publicly available
upon acceptance of the paper. Several enhanced real noisy speech examples are provided on the demo page: \url{https://anonymous.4open.science/w/USE-5232/}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the causes of performance degradation in universal speech enhancement (SE) methods. It considers alignment mismatch between input and target recordings, compares regression-based and generative approaches to SE, and investigates the effects of low-quality training data on the final model's performance.

### Strengths
The paper correctly identifies key areas that crucially affect the performance of SE methods.

### Weaknesses
The paper has the following weaknesses: 
1. The paper claims that aligning target and input audio recordings improves the overall performance of the model. This is not a new observation; e.g. [2] also aligns target and input audios for training. Moreover, this observation is quite obvious, since any impulse response used for augmentation introduces a time shift that is known and can be manually corrected relatively simply. Therefore, the observation that using $s[n-n_0] = s[n] \ast \delta[n - n_0]$ yields better results is not a novelty.

2. The authors discuss the benefits and downsides of using regression-based and generative methods for SE. They argue that using generative modelling can help reduce over-smoothing, yet preserve the fidelity. **Firstly**, the problem of over-smoothing is already well-studied, among others, in [1, 2, 3]. Other GAN-based methods deal with over-smoothing using pre-training with regression-based pre-training and adversarial fine-tuning. It is not clear from the paper why training a separate generative model is better than applying fine-tuning. **Secondly**, the authors claim to provide a theoretical argument based on equation (3) in the paper that links the proposed method to the optimal transport. However, the presented argument lacks proper rigour. The conclusion "... *the generative model can mainly focus on correcting the over-smoothed regions of the regression model output*" is substantiated only with an intuitive explanation and lacks a formal proof. **Thirdly**, the provided analysis assumes that the SE method is based on adversarial training, which excludes a large body of work [4, 5, 6, 7] that uses diffusion-based and bridge methods for SE. It is unclear how the conclusions from the paper can be applied to these methods.

3. The authors observe that the URGENT 2025 Challenge training dataset contains some recordings that degrade the performance of the SE model. Although potentially useful for challenge participants, this observation, in my opinion, constitutes only a marginal contribution. Moreover, to fully measure the effects of the degraded recordings, it would be beneficial to train various models -- both GAN-based and diffusion-based -- on the original and cleaned data. That would show that the impact of the degraded data is significant; otherwise, the loss in quality might be attributed to architectural inefficiencies and training setup

### Questions
How can the analysis of the trade-off between the generative and regression-based paradigms be generalised to other types of SE methods, such as bridge models or diffusion-based models?

#### **References:**

[1] Andreev et al., "HiFi++: a unified framework for bandwidth extension and speech enhancement".

[2] Babaev et al., "FINALLY: fast and universal speech enhancement with studio-like quality".

[3] Su et al., "HiFi-GAN-2: studio-quality speech enhancement via generative".

[4] Lemercier et al., "StoRM: a diffusion-based stochastic regeneration model for speech enhancement and dereverberation".

[5] Scheibler et al.,  "Universal score-based speech enhancement with high content preservation".

[6] Jukíc et al., "Schrödinger bridge for generative speech enhancement".

[7] Wang et al., "Diffusion-based Speech Enhancement with Schrödinger Bridge and Symmetric Noise Schedule".

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper investigates three approaches that aim at improving the training of universal speech enhancement (USE) systems. These three approaches are (a) enforcing higher quality training data by curating files with a high VQScore, (b) employing unreverberated training targets instead of such including early reflections, and (c) a two-stage training framework that combines discriminative and generative architectures. The authors show that a smaller, but higher quality dataset improves performance, and that non-reverberated targets as well as their two-stage approach raise scores on non-intrusive and task-dependent metrics.

### Strengths
The author’s train of thought is well described and easy to follow. The paper compares with both state of the art and open-source models on a publicly available test set. Furthermore, the authors did state their intent to publish code, which would ensure reproducibility of the presented results.

### Weaknesses
While the paper is at first sight well written and presents interesting results, there are multiple shortcomings. A major issue lies with the novelty and presentation of the discussed three “critical aspects”:

1.	Training targets: There are two issues with the authors’ claim here. Firstly, the assumption that the use of slightly reverberated targets is due to the difficulty of removing them is disputable, especially since the authors themselves contradict that argument by presenting models trained on (time-shifted) anechoic speech as superior in performance. The main reason for maintaining early reflections in the targets is that they are regarded as beneficial to speech intelligibility [Bradley 2003]. Secondly, the anechoic clean speech approach in Figure 3 (b) without consideration of the direct path delay is no sensible approach to begin with. The proposed solution with time-shifted targets (or vice-versa, removing direct path delay in the room impulse response) is not novel, but should rather be the standard for any reasonable training employing anechoic targets.
[Bradley, et al. “ On the importance of early reflections for speech in rooms,” 2003.]

2.	Model architecture: The results in Table 1 are unconvincing. Employing the GAN correction degrades more metrics than it improves. If the authors had adopted the overall ranking from URGENT, as they did with the other metrics, these models would have fallen behind their regression-only counterparts. Furthermore, improving non-intrusive metrics, which cannot detect hallucinations, is not surprising for a generative model. Only the slight improvement in CAcc seems interesting. Regarding novelty, combining regression and generative stages is nothing new and has been done in much more sophisticated ways before (see UNIVERSE++).
[Scheibler et al., “Universal Score-based Speech Enhancement with High Content Preservation”, 2024.]

3.	Training data quality: No novelty at all. This is basically the same which was already investigated in more detail and with a more sophisticated data curation strategy in the paper by [Li 2025], which was even cited by the authors. Furthermore, why would the authors rely solely on the seemingly not entirely suitable VQScore when samples of the dataset could be out of domain for this evaluation as mentioned in Section 2.3 (e.g., expressive speech)?
[Li et al., “Less is More: Data Curation Matters in Scaling Speech Enhancement”, 2024.]

Furthermore, the presentation of results is questionable. Why would the authors present a table with 12 metrics, only to argue that 8 of them cannot be considered for comparison due to different training targets? The complete lack of comparable intrusive metrics greatly reduces the significance of the results. A subjective degradation category rating (DCR) listening test between the “Early reflected” and “Shifted anechoic” approaches could have helped.

### Questions
Minor remarks:
-	Section 2.2: missing article before bold phrases (twice)
-	Fonts in figures are often too small
-	Some figures are just screenshots; labels contain compression artifacts
-	Figure 1: output -> Output
-	Table 3 would be more conclusive if the output of the models regression stage would also be reported
-	Consistently missing capitalizations in references (e.g., line 513: Ecapa-tdnn – ECAPA-TDNN, line 518: Icassp -> ICASSP, line 636: perceptual –> Perceptual)
-	Inconsistent formatting of references, e.g., line 603 Rix [Rix et al.] vs line 673 [Zhao et al.] – both are ICASSP papers
-	Figures 6 and 7 are missing axis descriptions (time, frequency range)

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the problem of universal speech enhancement, which is to restore the quality of diverse degraded speech while preserving fidelity. To address the problem that the use of early-reflected speech as the training target harms perceptual quality, the authors consider time-shifted anechoic clean speech as the target. To address the problem that regression models preserve fidelity but produce over-smoothed outputs while generative models improve perceptual quality but risk hallucination, the authors introduce a two-stage training approach that first trains a regression model and then freezes it to train a generative model. Provided experiment results demonstrated the effectiveness of the proposed method.

### Strengths
The proposed method of using time-shifted anechoic clean speech as training target and the two-stage training approach is interesting, and the provided experimental results demonstrated the effectiveness of the proposed method.

### Weaknesses
The novelty of this paper is somewhat limited. The contributions of this paper seems incremental in system design rather than proposing fundamentally new ideas. The use of time-shifted anechoic clean speech as the training target is an incremental modification of existing practices. The two-stage training approach is not new, similar approaches have been explored in both speech and image enhancement, e.g., prior work such as [1] also consider such a two-stage training approach for speech enhancement.

While the authors claim to provide a theoretical analysis of the two-stage framework (in Abstract), this is not substantiated. Section 2.2 mostly summarizes conclusion from prior work rather than presenting any new theoretical results. In particular, eq. (3) is quite trivial and does not offer any deeper understanding of why or how the two-stage approach works. Therefore, it is difficult to consider this as a theoretical contribution.

[1] Huang J, Yan Z, et al. A Two-Stage Training Framework for Joint Speech Compression and Enhancement. arXiv preprint arXiv:2309.04132, 2023.

### Questions
See the above Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
