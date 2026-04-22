# Are Deep Speech Denoising Models Robust to Adversarial Noise?

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4, 4

## Abstract
Deep noise suppression (DNS) models enjoy widespread use throughout a variety of high-stakes speech applications. 
However, we show that four recent DNS models can each be reduced to outputting unintelligible gibberish through the addition of psychoacoustically hidden adversarial noise, even in low-background-noise and simulated over-the-air settings. For three of the models, a small transcription study with audio and multimedia experts confirms unintelligibility of the attacked audio; simultaneously, an ABX study shows that the adversarial noise is generally imperceptible, with some variance between participants and samples.
While we also establish several negative results around targeted attacks and model transfer, our results nevertheless highlight the need for practical countermeasures before open-source DNS systems can be used in safety-critical applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic study showing that modern deep noise suppression (DNS) models can be driven to produce unintelligible outputs via psychoacoustically hidden adversarial perturbations, including experiments in simulated over-the-air settings and a small human study validating imperceptibility and output unintelligibility. The authors develop a masking-aware PGD attack (with STFT-space clipping and RIR-aware optimization via Wiener deconvolution / gradient projection), evaluate four open-source DNS models across a range of SNRs and reverberation conditions, examine targeted vs. untargeted attacks, explore (limited) universal perturbations and defenses (Gaussian noise), and provide mechanistic analysis about gradient behavior and transfer failure; code and audio examples are promised

### Strengths
Clear motivation and high practical relevance. The paper motivates DNS as a high-impact target (video conferencing, hearing aids, alert systems) and illustrates realistic safety concerns where small, imperceptible perturbations could deny access to critical audible information. This real-world framing is consistently maintained throughout the experiments and human study.

Methodological thoroughness within scope. The attack pipeline is well specified: STOI is used as an optimization objective, psychoacoustic masking thresholds (with temporal pre/post masking and tightened offsets) define the imperceptibility constraint, and a combination of Wiener deconvolution and gradient-based projection is used for simulated over-the-air attacks. The optimization and perceptual-constraint details are carefully described in Methods and Appendices.

Comprehensive empirical evaluation and human validation. The authors attack four modern, publicly available DNS models, sweep SNR and reverberation conditions, run simulated OTA experiments, test perceptibility settings, and run a human transcription + ABX study to corroborate automatic metrics—providing a multi-pronged empirical case.

### Weaknesses
The paper does not clearly articulate what is uniquely challenging about attacking DNS models beyond choosing an intelligibility loss (STOI) and applying psychoacoustic masking; much of the pipeline (masked PGD, RIR modeling, use of STOI) repurposes known techniques from ASR / audio adversarial work, and the manuscript should more explicitly explain any DNS-specific obstacles or mechanisms that make these attacks nontrivial for this class of models.

The work only studies white-box attacks and reports that naive transfer between model checkpoints and architectures is weak; there is no developed black-box attack (e.g., surrogate-ensemble training, query-based gradient estimation, or adaptive transfer techniques), leaving open the practical threat model where attackers lack direct gradient access to the target DNS.

The over-the-air experiments use simulated RIRs and treat the RIR as known during optimization, but the paper omits discussion and evaluation of other physical observation chain factors that significantly affect real OTA success, most notably speaker and microphone frequency responses, microphone directivity, amplification/attenuation non-linearities, A/D front-end filtering, and typical codec/compression applied by capture devices. These unmodeled effects can substantially change received perturbation spectra and thus attack efficacy, yet they are not specified or empirically varied.

The paper focuses on per-utterance offline optimization (with many iterations per sample chosen to fill about an hour of GPU time per trial) and does not address real-time / streaming attack feasibility. DNS systems commonly operate frame-by-frame with low latency; the manuscript lacks analysis of whether a perturbation can be injected in a streaming fashion (per frame or with limited lookahead) and still achieve the same destructive effect.

### Questions
What concrete aspects of DNS architectures or objectives make attacking DNS fundamentally different from attacking ASR or speaker models, beyond changing the loss to STOI? Could the authors highlight DNS-specific failure modes exploited by the attack?

Have the authors attempted any black-box strategies (surrogate ensemble, score-based queries, or transfer from ensembles) to assess attacker capabilities without gradient access? If not, can they comment on expected barriers and potential approaches?

In the simulated over-the-air experiments, which components of the capture chain are modeled beyond the RIR? Specifically, were speaker and microphone frequency responses, A/D front-end filtering, microphone directivity, or codec effects considered or varied? If not, how do the authors expect these factors to change attack success?

Do the authors foresee a streaming/real-time attack implementation where the perturbation is produced or injected online with limited lookahead (e.g., frame-by-frame)? If so, what would the constraints be (latency, buffer size, computational cost), and if not, can the authors quantify how much offline optimization is needed per second of audio?

Given the observed weak transfer across architectures, can the authors recommend practical defenses (beyond Gaussian noise) that exploit that architectural brittleness—for example, randomized pre-processing, ensembling, or input augmentation at inference time?

### Soundness
3

### Presentation
3

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
This is an interesting work that delves into the impact of adversarial noise on current speech denoising models. The authors claim that adversarial perturbations can impact the perceptual quality and lexical content of speech, where multiple speech denoising models were considered.

The paper is well motivated and the clearly written. However, the motivation part can be elaborated a bit more to illustrate the broader impact of adversarial attacks on DNS models. While one of the application cited is hearing aids which is relevant and useful to be aware, but are there possible applications where the impact of adversarial attack can be overwhelming? 

The analysis shared is clearly explained and the choice of the metrics used are also clearly detailed.

### Strengths
The paper provides a deeper analysis of the impact of adversarial attacks on deep noise suppression models. The steps used in the analysis are clearly explained and the results are well illustrated. The paper also clearly outlines the contribution of the work and cites prior work and explains how this work is different than the prior work.

The results presented are convincing, specifically having human evaluations with subjective metrics and comparing that with objective metrics makes the observations made in this paper more convincing.

### Weaknesses
The motivation of the paper can be improved beyond what is stated already. One of the example the paper cites is hearing aids, where the adversarial attack can have severe impact. Are there other applications, that are more widely used, that may suffer from such attacks? 

The observations made in this work can also inform what type of evaluation needs to be considered in the future for researchers working in the area of DNS models, which again can be yet another motivation.

### Questions
(1) In page 6, the paper specifies: "Audio was clipped to five second ..." 
>> Would this time constraint assumption have an impact on the intelligibility of the data? How would this time constraint impact possibly the metrics used in this work.
For longer speech segments, would the observations made in this work still hold?

(2) In page 7, the paper says: "Figures 1 and 5 suggest that FSN+ ..."
>> Figure 5 is in appendix, perhaps that needs to be clearly specified.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Summary: 

This paper studies the impact of adversarial attacks on DNS systems, which can cause several safety concerns. 4 open-source DNS models are evaluated and 2 types of adverbial attacks are studied (i.e., additive and convolutional).Through extensive experiments, the authors have found out that all DNS systems are prone to adversarial attack. While this paper is the first to address the problem under a DNS setting, the results are quite trivial since most DNNs (to the best of my knowledge) are prone to adversarial attack.

### Strengths
See Section Questions

### Weaknesses
See Section Questions

### Questions
Questions, Strengths and Weaknesses: 

1. Extensive experiment setups and follows ethical guidelines. 
2. An psychoacoustic auditory masking approach is first proposed and applied as adversarial attack. This is a novel approach and hypothesises that DNN may follow basic psychoacoustic principles. However, the effect of the proposed method is not included in the main text, but only in Appendix D.3. What are the advantages of using auditory masking as compared to black-box attacks / simply adding gaussian noises? 
3. Real-world speech signals covers a wider range of scenarios (for instance UHF-transmitted speech: [1] DENT-DDSP, Guo et.al, Interspeech 2023). I would encourage the authors to include these more general cases in the introduction or at least limits the scope of the paper.

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates whether modern DNS models are vulnerable to attack. The main finding is that they are highly vulnerable. The authors showed they could use psychoacoustically hidden noise to make the output of DNS gibberish. This noise is crafted to be imperceptible to the human ear, but it completely tricks the DNS models.

### Strengths
- The paper is the first systematic study to show that DNS models are vulnerable to imperceptible adversarial attacks.
- The paper considers over-the-air attack by incorporating RIR which makes the attack more practical.
- The presentation of the paper is clear and easy to follow.
- The authors use human raters to confirm the conclusion that the attacked audios are non-intelligible.

### Weaknesses
- The motivation example in the introduction section feels a little artificial to me. Attacks are expensive and mostly happens when highly incentivized. I am not seeing the incentive of attack in the given scenario.

- The over-the-air attack design only considers RIR so the found adversarial noise might not be robust to other perturbations in real life like compression loss, background noise, etc. It'd be helpful to incorporate such consideration into the attack as well. Example can be found in [1].

- The attack only leads the DNS models to output gibberish instead of a targeted meaningful sentence. This limits the severity of the attack.

- The attack works best when the noise is designed for a given utterance which also limits the flexibility of the attack.

[1] Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World.

### Questions
- Can the over-the-air attack stand perturbations beyond RIR.
- Can the attack be made to lead the DNS models to output meaningful sentence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper examines the adversarial robustness of speech denoising models. To do so, the authors simulate adversarial attacks using the projected gradient descent method on otherwise clean speech recordings that have had environmental noise and reverberation synthetically added to them. The paper finds that very subtle adversarial perturbations can cause denoising models to not only fail to denoise the speech, but rather produce outputs that are significantly less intelligible than the noisy speech that was provided as input. To verify the subtlety of the adversarial attacks, the authors conducted a human study, in which they found that the intelligibility of the speech post-attack was not noticeably degraded, and humans could not usually discriminate the attacked speech from unattacked speech.

### Strengths
1. Mostly clear presentation
2. Technically sound methodology. I appreciate the comprehensive measurement of intelligibility
3. I appreciate the human study as it grounds the work in reality.
4. Novel work and significant results. This paper fills the gap in the current literature regarding the robustness of speech denoising models and the results show that these models are indeed susceptible to attacks. The novelty and originality of the work are enhanced by the masking and RIR aware attack framework proposed by the authors, and the enhancements to the masking attack methodology of Qin, et. al 2019.

### Weaknesses
1. My primary concern with this paper is that the experiments are not rigorous and leave many important questions unanswered, without which limits the contribution and impact of the work.
    1. Black-box attacks were not used which limits the reliability of the results for at least 1 out 4 models in the current paper. Without black-box attacks it is not possible to answer question (f) from section 5. I recommend including 1 black-box attack
    1. UAP generated in the multi-model scenario. It is well documented that optimizing the adversarial perturbation over multiple models leads to increased transferability. I would suggest doing a leave-one-out study, in which the UAP is optimized over 3 models and is used to attack the 4th model.
    1. Gaussian noise is posed as a defense but adaptive attacks are not used. As such, there is insufficient evidence to draw any conclusions from this experiment. I would suggest either trying adaptive attacks or removing this section.
    1. Effective defenses like, adversarial training or randomized smoothing should have been evaluated to show the trade off between robustness and accuracy.
    1. The proposed attack aware framework is a key contribution of this work, but it is not thoroughly evaluated in the paper. I would recommend adding ablation experiments that demonstrate the increased imperceptibility of the perturbation generated by the proposed method, and increased attack success as measured by change in STOI, when compared to the vanilla PGD attack, and the masking-based attack from Qin, et. al 2019.
    1. Answer to question (c) is not found in the paper. The only relevant comment is the contributions listed in the introduction and I am not very convinced by it. It is obvious and well known that without gradient flow white-box attacks will not succeed, so this is not really a contribution. Authors claim that model size and input features do not matter, but I did not see any results in the main body to show this. 
1. Some experimental detail is missing and some settings are questionable:
    1. It is not mentioned how many speech samples were used in the evaluations
    1. The decision to determine the number of attack iterations based on the GPU time is unusual and unjustified. This introduces a confounding factor in the results. Keeping FSN+ aside, FRCRN appears to be the most robust model, but it is also the model that is evaluated against the least attack steps.
1. Some presentation issues:
    1. Figures need to be rearranged so that they appear _after_ they are referenced. Also, Figure 2 and Figure 3 and in the wrong order.
    1. Answers to at least 4/9 are not present in the main body of the paper. If these questions are unimportant they shouldn't be included in the list, otherwise the should be treated in sufficient detail in the main body.
    1. Figure 3 a and b should be separate figures
    1. Add equation for the objective in section 4.4

### Questions
1. Why was it important to assume that the background noise and reverb for UAP are the same across speech samples?

### Soundness
2

### Presentation
3

### Contribution
3
