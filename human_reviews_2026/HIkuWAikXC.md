# ICDiffAD: Implicit Conditioning Diffusion Model for Time Series Anomaly Detection

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
Time series anomaly detection (TSAD) faces critical challenges from intrinsic data noisiness and temporal heterogeneity, which undermine the reconstruction fidelity of prevailing generative approaches. 
While diffusion models offer theoretical advantages in capturing complex temporal dynamics, their inherent stochasticity introduces irreducible variance in reconstructions.
We present the ICDiffAD, a novel method that synergizes adaptive noise scheduling with semi-deterministic generation to address these limitations. ICDiffAD introduces two key innovations: 
(1) an *SNR Scheduler* that governs training through quantifiable noise scales, enabling robust learning of normative patterns across non-stationary regimes; and 
(2) an *SNR Implicit Conditioning Mechanism* that initializes reverse diffusion from partially corrupted inputs, preserving signal coherence while attenuating anomalous components. 
This dual strategy ensures high-fidelity reconstructions aligned with the input’s manifold, reconciling generative flexibility with detection accuracy. 
Across five multivariate benchmarks, ICDiffAD improves the F1 score by 19.57\% and reduces false positives by 60.23\% compared to existing diffusion model-based TSAD methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to address the reconstruction inconsistency problem in diffusion-based time series anomaly detection, which arises from the noise-adding and denoising processes. Specifically, the authors argue that when a sample is diffused all the way to pure noise, the strong generative capability of diffusion models can produce highly diverse reconstructions, thereby degrading anomaly detection performance. The paper focuses on optimizing the noise scheduler during both training and testing. In the training stage, the authors replace conventional noise scheduling strategies (e.g., cosine schedule) with one based on the signal-to-noise ratio (SNR). During inference, they further adapt the noise strength per sample, again guided by the SNR. Experimental results across multiple datasets and various baselines demonstrate that the proposed method performs among the top tier. In particular, compared with other diffusion-based approaches, the proposed technique shows improvement in efficiency.

### Strengths
The paper has two notable merits, summarized as follows:

1.  The paper introduces a novel perspective by optimizing anomaly detection from the viewpoint of noise scheduling. In my opinion, this is an interesting and original angle, and to some extent represents a new direction for improving both the accuracy and efficiency of diffusion models for conditional time series generation.
2.  The experimental evaluation is reasonably comprehensive. In particular, the ablation studies are well designed and clearly demonstrate the effectiveness of each component in the proposed framework.

### Weaknesses
Similarly, from my perspective, the current version of this paper also exhibits several non-negligible weaknesses, summarized as follows:

1. **Motivation.**
   First, I acknowledge that generative models can produce diverse outputs when starting from different noise priors. I also notice that the paper (see Fig. 1) implicitly assumes that the denoised samples obtained from diffusion models correspond to “normal” (i.e., smooth) samples. This assumption itself is reasonable. However, later in the paper (lines 278–279), the authors state their own premise—*“the premise that time series anomalies manifest as high-frequency deviations from normative patterns.”*
   Given this assumption, one could argue that existing diffusion-based methods are already capable of separating such high-frequency anomalies from normal patterns. Furthermore, regarding Fig. 1, I am skeptical about the second-row examples: such patterns could naturally arise from any sliding-window training process, and thus may not genuinely indicate anomalous behavior.

2. **Writing and presentation.**
   The narrative flow of the paper is not particularly convincing. For instance, the introduction of the signal-to-noise ratio (SNR) is abrupt and lacks conceptual grounding, despite being the core of the entire framework. Merely pointing out the potential limitations of previous scheduling methods is insufficient; the authors should elaborate on how SNR-based scheduling specifically benefits time-series data.
   In addition, Fig. 2 provides only a coarse overview. A more detailed modular diagram or code block would be valuable for understanding the system design.

3. **Methodology.**
   The claim made in the *Introduction*—that *“this new scheduler induces the model to learn normal patterns across diverse, SNR-calibrated corruption levels, enhancing robustness to noise perturbations and temporal heterogeneity”*—is intriguing but not well supported. I am not convinced that a more linear ($\alpha$)-curve (as shown in Fig. 2) would directly yield these benefits.
   Moreover, the first module of the method seems fully general and could be applied to computer vision or other domains, with no specific connection to time-series data. In contrast, the second module, *Diffusion Step Alignment*, appears to disrupt the previously optimized noise-level alignment, which raises additional concerns.

4. **Experiments.**
   When reading the *Experiments* section, I was expecting to see visualizations of real-world data similar to those in Fig. 1, which would strongly support the paper’s central claims—but none are provided. The main experiments focus solely on point anomaly detection, with weak evaluation on localization tasks.

5. **Reproducibility.**
   Although the authors state that they plan to release the code in the future, the current submission does not provide sufficient implementation details to ensure reproducibility.

### Questions
1. See the Weaknesses part.
2. Are the reconstruction results in Fig. 2 produced by the proposed method? If so, the quality of the reconstructed samples appears suboptimal—could the authors clarify this?
3. In Fig. 3, the performance of DiffAD seems extremely poor, almost equivalent to random output. Could the authors explain the cause of this behavior or verify whether there might be an issue in the experimental configuration?
4. How is the inference time reported in Table 12 calculated? Does it correspond to a single forward pass of the neural network, or to the entire denoising process? The reported time seems unusually large.
5. Given that ImDiffusion has a relatively small number of parameters (also see Table 12), is the comparison fair? Have the authors normalized or controlled for model capacity when reporting efficiency results?

### Soundness
2

### Presentation
1

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
The paper proposes a diffusion-based method for time series anomaly detection called ICDiffAD. It has two main parts. First, an signal-to-noise ratio (SNR) guided scheduler that replaces standard $\beta$ schedules by targeting a terminal SNR and deriving per-step $\alpha_t$ from a function $g(t)$. This makes the corruption level interpretable and consistent across runs. Second, SNR Implicit Conditioning (SIC) that estimates an input SNR by splitting each test window into a low-pass signal and a residual, then maps this estimate to a target signal-retention $\alpha^\*$ and chooses the closest diffusion step $\hat T$. The reverse process then starts from $x_{\hat T}$ and reconstructs to $x_0$. Anomaly scores are the reconstruction residuals on each window. The model uses a UNet-style backbone on 2D reshaped windows and keeps one set of hyperparameters across datasets. Experiments on MSL, SMAP, SMD, PSM, and SWaT report higher point-level F1 than prior diffusion baselines and small gains over the strongest reconstruction baseline. Ablations indicate that both the SNR scheduler and SIC contribute to the improvements.

### Strengths
- The proposed SNR Implicit Conditioning (SIC) adaptively estimates an optimal corruption level and denoising step for each test instance. This mechanism effectively selects the optimal diffusion time $\hat T$ per input, balancing reconstruction fidelity and determinism.

- The reported improvements are consistent across multiple benchmarks. Ablation studies (TSNR sweeps, $g(t)$ robustness, SIC contributions) and qualitative visualizations substantiate the claims, showing that performance gains come from the model design.

- The authors employ point-level metrics instead of range-based ones, avoiding inflated results and better reflecting the intrinsic difficulty of time-series anomaly localization.

- The model uses a fixed set of core hyperparameters across datasets (e.g., $T = 200$, window size $= 1024$, fixed $\mathrm{TSNR_{dB}}$), which enhances reproducibility and ensures fair comparisons across benchmarks.

### Weaknesses
- The statement that input-agnostic schedulers “indiscriminately degrade all frequency components” and “over-corrupt transients” is asserted without citation or evidence. This statement is also unclear.

- In Sections 4.2 and 4.3, the variable $Z$ appears with both $l$ and $t$ subscripts, but it is unclear which one corresponds to $Z_0$ in expressions such as $\mathbb{E}\[\||Z_0\||_F^2\]$. Please clarify the intended meaning of each $Z$ symbol.

- The text suggests that $\alpha_t$ and $M_t$ depend on each other, which reads as circular. Please state explicitly that a target terminal TSNR is first chosen, a specific example would clarify this. Also fix notation in the $\mathrm{SNR}(t)$ equation (10) to use $\bar{\alpha}_t$ rather than $\bar{\alpha}_T$.

- The paper repeatedly claims to achieve *deterministic reconstruction*, but the proposed procedure still **samples noise** when initializing the reverse process at $\hat T$ via $x_{\hat T} = \sqrt{\bar\alpha_{\hat T}}x_0 + \sqrt{1-\bar\alpha_{\hat T}}\varepsilon.$ This is **not** determinism in the diffusion sense (which would correspond to DDIM with $\eta=0$ or a closed-form sampling path). The goal of the paper is to get **input-consistent reconstruction rather than strict determinism**, so the terminology is confusing/misleading.

- The paper reports only strict point-level metrics as the primary evaluation, but does not include point-adjusted or event-level scores used by many prior TSAD works. While strict evaluation is commendable and should remain the main protocol, adding point-adjusted or event-based metrics (even in the appendix) would allow fair comparison with earlier baselines that report only lenient metrics. Without both views, it is difficult to fully contextualize the reported gains.

- Efficiency is mixed up with speed rather than real compute cost. Table 12 shows higher GFLOPS than DiffAD and ImDiffusion but still lower runtime, which suggests the model is actually doing more work per input but runs faster thanks to better GPU utilization or code efficiency. It is therefore unclear whether the method is truly efficient or simply well-optimized.

- It is unclear whether the proposed *SNR Implicit Conditioning* actually achieves the intended effect. The paper does not report the distribution of selected start steps $\hat{T}$ (e.g., mean and standard deviation per dataset), so it is difficult to verify that the ISNR-based selection is meaningful. A simple baseline would be to perform reconstruction starting from a fixed reduced step (e.g. $T/2$) as a hyperparameter, instead of always starting from full corruption. Without such comparison, it is hard to tell whether the gains come from the conditioning mechanism itself or from simply using fewer denoising steps.

### Questions
- What do the superscripts in $x^{\text{tar}}$ and $x^{\text{con}}$ stand for? Please define the terms “tar” and “con” explicitly when first introduced to avoid ambiguity.

- Why is GFLOPS highest for your model while wall-clock is lowest among diffusion baselines?

- How does the proposed method compare to baseline in a point-adjusted evaluation protocol?

### Soundness
2

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
This paper proposes ICDiffAD, a diffusion-based framework for time-series anomaly detection that introduces a synergized adaptive noise scheduling mechanism. Experiments on several benchmark datasets demonstrate promising performance compared to prior diffusion-based anomaly detection methods.

### Strengths
1. The idea of customizing a synergized adaptive noise scheduling mechanism within a diffusion model is innovative and potentially valuable for anomaly detection.

2. The introduction of a semi-deterministic generative process is a meaningful adaptation. It helps reduce randomness during reconstruction and could indeed be beneficial for stable anomaly detection tasks.

### Weaknesses
1. Questionable learning effectiveness of deterministic diffusion. The paper modifies both the noise-adding and denoising processes from stochastic to deterministic forms. However, the theoretical rationale and empirical evidence supporting that such deterministic transitions preserve the learning capability of diffusion models are insufficient.

2. Unclear formulation of the SNR Scheduler. The proposed scheduler controls the noise coefficient α based on a target noise level M_T, yet the paper does not explain how M_T is derived, estimated, or learned.

3. Unaddressed risk of anomalous conditioning. The paper criticizes prior deterministic reconstruction methods for potentially conditioning on anomalous inputs but does not clearly explain how the proposed method avoids the same risk when performing semi-deterministic reconstruction.

 4. Ambiguous dimensionality description. During Data Preprocessing, the channel-wise concatenation operation changes the dimension from K to \hat{K}, but Equation (6) does not define \hat{K} or specify how concatenation is implemented among the original dimension K. This lack of clarity makes Equation (6) difficult to follow.

5. Questionable assumption. The proposed SNR Implicit Conditioning assumes that normal sequences exhibit lower frequencies while anomalous sequences exhibit higher frequencies. This assumption is not universally valid. Some anomalies manifest through amplitude or trend deviations rather than frequency differences. Moreover, for frequency-dominant anomalies, existing frequency-domain methods could already achieve superior performance, diminishing the uniqueness of the proposed mechanism.

6. The proof in Section 7.1 lacks clarity. The logical connection among Equations (19), (20), and (21) is difficult to follow, and the derivation steps between them are not well explained.

### Questions
See the weaknesses section

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
This paper targets the challenge of diffusion-model TSAD, i.e., stochastic reconstructions that inflate false positives. It proposes ICDiffAD, built on two components: 1) SNR Scheduler re-parameterizes the noise schedule using a target SNR and a monotone corruption function, deriving $\alpha_t$ directly from the SNR goal. 2) SNR Implicit Conditioning decomposes each test window via a zero-phase Gaussian low-pass filter $G_{\sigma}$ to compute an ISNR, then converts ISNR into an optimal corruption factor, and snaps it to the nearest pretrained diffusion step. On five benchmarks (MSL, SMAP, SMD, PSM, SWaT), the proposed ICDiffAD improves average F1 versus prior diffusion models and stronger baselines. The ablation experiments show gains from both SNR Scheduler and SIC.

### Strengths
1) This paper crisply argues that stochastic reconstructions in diffusion models conflict with determinism requirement of TSAD, with intuitive illustrations and discussion. The motivation is interesting and clear.

2) The proposed method is simple yet effective, and its training control is physically interpretable.

2) The experiments and ablations are comprehensive, and the proposed method improves average F1 and reduces false positives in comparisons.

### Weaknesses
1) This paper appears to contain notation inconsistencies in SNR expression. Eq. (10) defines SNR(t) using  $\bar{\alpha}_T$ (terminal product) while the text underneath says “$\bar{\alpha}_t$ represents cumulative signal retention up to step $t$”, which can confuse readers about the intended instantaneous SNR.

2) This paper performs per-dataset TPE searches over filter size and reports the best F1 scores, implying supervised selection using test labels, which can overstate generalization. Clarifying a validation split or unsupervised criterion would help.

3) The discussion and analysis of diffusion-based TSAD methods is not comprehensive enough, and a comprehensive review of relevant papers in this field is needed.

4) Many applications care about event-level detection/latency. Adding event-level metrics would strengthen claims.

### Questions
1) This paper appears to contain notation inconsistencies in SNR expression. Eq. (10) defines SNR(t) using  $\bar{\alpha}_T$ (terminal product) while the text underneath says “$\bar{\alpha}_t$ represents cumulative signal retention up to step $t$”, which can confuse readers about the intended instantaneous SNR.

2) This paper performs per-dataset TPE searches over filter size and reports the best F1 scores, implying supervised selection using test labels, which can overstate generalization. Clarifying a validation split or unsupervised criterion would help.

3) The discussion and analysis of diffusion-based TSAD methods is not comprehensive enough, and a comprehensive review of relevant papers in this field is needed.

4) Many applications care about event-level detection/latency. Adding event-level metrics would strengthen claims.

### Soundness
3

### Presentation
3

### Contribution
3
