# DurMI: Duration Loss as a Membership Signal in TTS Models

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Text-to-speech (TTS) models such as FastSpeech2, Grad-TTS, and VITS2 achieve state-of-the-art quality but risk memorizing and leaking sensitive training data. Existing membership inference attacks (MIAs) for diffusion-based TTS models typically rely on denoising errors, which are costly to compute and weak at capturing sample-specific memorization. 

We introduce DurMI, the first membership inference attack that exploits duration loss, a core alignment signal in TTS models, as a discriminative indicator of membership. Duration loss captures the model’s tendency to overfit alignment targets, whether derived from deterministic aligners such as MAS and MFA or from stochastic predictors as in VITS2. Leveraging this signal, DurMI enables accurate inference with a single forward pass, while remaining broadly applicable across diverse TTS architectures.

Experiments across diverse architectures, including diffusion (Grad-TTS, WaveGrad2), flow-matching (VoiceFlow), transformer (FastSpeech2), and stochastic-duration (VITS2), on three benchmarks show that DurMI consistently outperforms prior MIAs, including on waveform-level synthesis where existing attacks fail. These results highlight DurMI’s effectiveness, efficiency, and broad applicability, underscoring the need for privacy-preserving training in modern TTS systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a  white-box membership inference attack (MIA) called DurMI, which specifically targets Text-to-speech (TTS) models. The attack is based on duration loss, a core component of how TTS models learn to align text and audio. Duration loss can be used as a discriminative signal to identify whether a sample was part of the training data. The authors claim that this method is more effective and efficient than prior MIAs for diffusion-based TTS models, as it requires only a single forward pass, bypassing the computationally expensive denoising steps of the decoder.  DurMI was evaluated on five different TTS architectures and three benchmark datasets, demonstrating that it consistently outperforms previous attacks and is robust across various model types, including those with stochastic duration modeling.

### Strengths
- Introducing a simple approach to MIA for TTS models by focusing on duration loss, which is unexplored.
- DurMI is significantly more efficient than existing methods like SecMI and PIA, with lower computational cost.
- The attack is shown to work across a diverse range of TTS architectures, including diffusion-based (Grad-TTS, WaveGrad2), transformer-based (FastSpeech2), flow-matching (VoiceFlow), and stochastic-duration models (VITS2).

### Weaknesses
- **Dependence on explicit duration predictors:** The core of the DurMI attack relies on the presence of an explicit duration predictor. While the authors discuss potential proxy indicators for future alignment-free architectures, the current attack is not directly applicable to those systems (e.g., autoregressive Tacotron-like models).
- **Lower performance on certain datasets:** Although DurMI generally performs well, its TPR@1% FPR is notably lower on the VCTK dataset, especially for models with stochastic duration modeling like VITS2. The authors attribute this to dataset-specific characteristics like shorter utterances and less text overlap, but it still represents a limitation of the attack's effectiveness in certain scenarios.
- **White-box assumption:** The attack is formulated under a white-box threat model, which assumes the adversary has full access to the trained model's parameters and loss function. A black-box attack would be more representative of real-world scenarios.

### Questions
- Table 4 indicates that earlier methods (PIA, PIAN, SecMI) underperform the Naive Attack, which appears inconsistent with the findings in [1]. Could the authors clarify the reasons for this divergence?
- It would be beneficial to also briefly discuss potential defense mechanisms or countermeasures that could be developed in response to this attack. This would make the paper more comprehensive and constructive.
- Consider an ablation that systematically varies the number of training and evaluation samples to assess robustness and data-efficiency.



[1] Kong, Fei, et al. "An efficient membership inference attack for the diffusion model by proximal initialization." ICLR 2024.

### Soundness
3

### Presentation
2

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
This paper conducts research on TTS models with explicitly force-aligned duration modules, leveraging duration loss to MIA task (to determine whether a specific utterance was used for training).

Results show better performance than previous methods under the above research scope, and on moderate-size datasets (tens or hundreds of hours, single or a few hundred speakers).

DurMI is very fast as it is free of multi-step denoising. However, this method has hardly been explored with truly large-scale in-the-wild dataset trained TTS models. On top of that, recent zero-shot models and probably also that without explicit force-alignment modules are actually outside the scope of this paper.

### Strengths
**Originality**: DurMI is the first to use duration loss of force-alignment module equipped TTS models for membership inference attack.

**Quality**: Within the scope of the research, relatively thorough experiments were conducted. For example, Appendix A.6 provides some valuable insights.

**Clarity**: The overall writing is good.

**Significance**: A thorough exploration was carried out, though it was within a relatively narrow research scope.

### Weaknesses
**Research Contribution**: In short, if the preliminary idea mentioned in Appendix A.7 has not yet been explored and studied, then this paper may not be sufficient to be accepted by ICLR.

**Problem Formation**: There are two major problems: overclaims and statements without evidence to support.
- The abstract claims an overly broad research scope, which does not match the actual one, and is not made clear until the end of the introduction.
- In the introduction section, line 37, the mentioned models are not trained on large-scale datasets (at least the ones used in this paper are not). LJSpeech, VCTK, and LibriTTS are public. The statement that "often contains sensitive or proprietary content" is not consistent with previous context.
- In line 94, the authors state that "Together with ongoing deployment of duration-supervised systems in industry, this indicates that alignment remains central to current and near-future TTS", but without any concrete evidence to support (e.g. article, paper, technical report, or blog). Moreover, there are works [1] from another perspective but not the duration modules DurMI applied on.
- The marginal improvement seen from Table 11 in MegaTTS 3 paper does not support the authors' statement from line 147 to 152. In simple terms, this does not constitute a reasonable reason for the authors to exclude the alignment-free and zero-shot TTS models.

[1] Wang, H., Li, N., Wang, C., Wu, S., Li, Z., & Yu, D. (2025). Vox-Evaluator: Enhancing Stability and Fidelity for Zero-shot TTS with A Multi-Level Evaluator. arXiv preprint arXiv:2510.20210.

**Writing**: 
- In abstract and also in main text, the authors might intend to express that the DurMI can be widely applied, but the taxonomy of "architectures" is confusing. "Diffusion" and "flow matching" are essentially the same, but listed separately; and why the two are put with "transformer" and "stochastic-duration"? "Deterministic" and "stochastic" would be good.
- In line 89, "modality-agnostic" or "modality-independent" sounds more appropriate than "modality-robust".
- In line 414 418, Figure 2h 2i should be 3h 3i.
- In line 890, Seed-TTS and MaskGCT lack references.
- There is no clear definition of $d(T_{gen}, T_{out})$ in main text, nor does it point to the definition present in Appendix A.7.
- Mixed use of $T$ and $N$ in Appendix A.7.

### Questions
1. In section 4, the authors acknowledge that "duration predictors are often overfitted to training samples". Typically, the datasets used in this paper are from audiobooks, or with narrower distribution. Then, how is the performance of DurMI compared to other generic methods on TTS models equipped with duration modules trained with in-the-wild data?
2. If the authors are willing, how is the performance on zero-shot force-alignment-free TTS models?
3. In line 456, why $T_{gen}$ and $T_{out}$ can be different for NAR TTS models?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work authors introcuces DurMI, a membership inference attack for TTS models that exploits duration loss as a discriminative signal to determine whether a given sample was present in the model's training set. The method demonstared improved perfromance accross different TTS architectures (diffusion, flow-matching, and stochastic dueraiton models) and benchmarks (LJSpeech, VCTK, LibriTTS) against prior membership inference attacks.

### Strengths
(1) The key innovation is use of duration loss as membership signal. This signal is leveraged in a way that is uniquely tailored to TTS architecutre.

(2) The propose appraoch is effective accross multiple leaning TTS models, including both determistic alignment (MASm MFA) and stochastic alignment (VITS2)

(3)  DurMI consistently outperforms prior approaches, particularly on challenging settings (e.g., waveform-to-speech synthesis with WaveGrad2, and on multi-speaker datasets). AUCs often exceeding 99% on multiple datasets and models.

(4) The paper systematically evaluates membership inference under strong threat models, multiple architectures, and offers detailed ablation studies

### Weaknesses
(1) While the attack demonstres high accuracy on deterministic-duration systems, the perfromance drops on models with stochastic predictors (VITS on VCTK). Althought authors acknowledges this but more directly address when and why the duration loss signal weakens, and whether this indicates intrinsic limits on the methodology for other TTS achitectures not used in the paper.

(2) Another concern is overemphasis on white-box setting. The paper assumes full white-box access, may not generalize to more realistic black-box attacker settings. The work will benefit from discussion or experimental analysis regarding robustness in more constrained environments.

(3) The thresholding mechanism $\mathcal{M}(x)$ is described as being set via a calibration set or a shadow model. Details regarding calibration protocol and its robustness (possible overfitting, decision curves) are not provided.

(4) Speaker-dependent effects are only briefly discussed but could substantially influence results.

(5) Figure 3 ablation plots cover only a subset of distance metrics and epochs. However, broader comparative analysis, e.g., inclusion of alternative clusterings or more nuanced hyperparameter sweeps, can further strengthen insights in the results.

### Questions
(1) Can you elaborate on the empirical process used for setting the membership threshold $T$ (see Page 5, Section 4.2) ?

(2) The appendix (Table 10/A.6.1) shows sensitivity to overlap in speakers and text. Could the authors clarify the practical implications, for example, in scenarios where text overlap is intentionally minimized?

(3) The use of proxy signals in zero-shot/implicit alignment TTS is motivated (Section 5.1, Figure 5), did  the authors performed preliminary experiments or can you share intuition about the signal-to-noise ratio in such proxies compared to explicit duration loss?

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
2

### Summary
The paper proposes DurMI, a white-box membership inference attack for TTS that uses the duration loss as the membership signal. The key idea is that modern TTS systems all rely on alignment/duration supervision, which can overfit at the utterance level. DurMI computes one forward pass up to the duration predictor and thresholds the duration loss to decide membership, yielding large speedups vs. diffusion-based MIAs and, the authors argue, better separability of members/non-members. Experiments on LJSpeech, LibriTTS, VCTK report AUC and TPR@1%FPR across models. Proposed DurMI outperforms baselines on most settings.

### Strengths
1. This paper identifies duration/alignment supervision as a strong, sample-specific leakage channel spanning diverse TTS families.

2. The evaluation is conducted across multiple pipelines. And the results demonstrate effective speedup compared to diffusion-based baselines. 

3. The paper provides proxy indicators proposed for implicit alignment (e.g., |T_gen−T_out|), potentially enabling black-box variants.

### Weaknesses
1. The assumption requires white-box access to the duration head and ground-truth durations/alignment d (often produced by a private pipeline), plus a calibration set with members & non-members. All these are too strong assumptions for real adversaries.  
2. Evaluations are not covering large-scale, noisy, multilingual, or proprietary corpora, and no exploration of regularization/DP defenses or training with stronger data augmentation.  
3, Many privacy users care about very low FPR (e.g., 0.1% or 0.01%); the paper reports TPR@1%FPR only. Several settings show high AUC but low TPR@1% (e.g., VITS2/VCTK), suggesting thinner margins where it matters most.  Speed and accuracy comparisons to diffusion MIAs might depend on step counts and norms rather than purely on signal choice; a matched wall-clock or best-effort baseline is not shown.  
4. Duration targets d come from aligners (MFA/MAS) that encode corpus- and pipeline-specific artifacts. Some of the signals may be pipeline leakage rather than pure model memorization; a control where d is recomputed with a different aligner would clarify this.  
5. Limited treatment of alignment-free trends: Proxy ideas are promising but not evaluated; current evidence doesn’t establish DurMI-style risk under black-box or zero-shot TTS.

### Questions
1. In realistic deployments, the aligner outputs d are often private. What happens if the attacker must recompute d with a different aligner, or only has noisy/partial alignment info? Please add experiments varying the aligner mismatch and noise.  Can DurMI work without a mixed calibration set (e.g., via parametric modeling of loss distributions or conformal methods)? Show results at strict FPR targets (0.1%, 0.01%).  

2. Could the authors provide equal wall-clock or best-effort comparisons to SecMI/PIA across models, and report sensitivity to their step budgets and norms to isolate the advantage of the signal (duration) vs. compute.   If train with MFA A but compute d at attack time with MFA B (or MAS), how do AUC/TPR change? This would separate utterance memorization from aligner/pipeline fingerprinting.  

3. How do regularization, DP-SGD, or duration-head dropout/noise affect DurMI? Can simple label-smoothing on durations reduce leakage with limited quality loss?  

4. Black-box testing is required: it is suggested to evaluate the T_gen vs. T_out discrepancy on at least one zero-shot TTS model (black-box), reporting ROC and TPR@low-FPR to substantiate proxy viability.  

5. It will enhance the paper quality to test on large, multi-speaker, multilingual corpora and real training pipelines; analyze how speaker imbalance, text overlap, and utterance length interplay.

### Soundness
2

### Presentation
3

### Contribution
2
