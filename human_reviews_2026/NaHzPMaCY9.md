# Steering Autoregressive Music Generation with Recursive Feature Machines

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 8, 2

## Abstract
Controllable music generation remains a significant challenge, with existing methods often requiring model retraining or introducing audible artifacts. We introduce MusicRFM, a framework that adapts Recursive Feature Machines (RFMs) to enable fine-grained, interpretable control over frozen, pre-trained music models by directly steering their internal activations. RFMs analyze a model's internal gradients to produce interpretable "concept directions", or specific axes in the activation space that correspond to musical attributes like notes or chords. We first train lightweight RFM probes to discover these directions within MusicGen's hidden states; then, during inference, we inject them back into the model to guide the generation process in real-time without per-step optimization. We present advanced mechanisms for this control, including dynamic, time-varying schedules and methods for the simultaneous enforcement of multiple musical properties. Our method successfully navigates the trade-off between control and generation quality: we can increase the accuracy of generating a target musical note from 0.23 to 0.82, while text prompt adherence remains within approximately 0.02 of the unsteered baseline, demonstrating effective control with minimal impact on prompt fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces MusicRFM, a framework for activation-level steering of pre-trained autoregressive music models. Building on RFMs, the authors identify interpretable “concept directions” in MusicGen’s hidden states that correlate with musical attributes such as notes, chords, or tempo. These directions are injected into the model’s residual stream during inference, allowing fine-grained control without retraining or step-wise optimization. Experiments on synthetic datasets (SYNTHEORY) and real-music benchmarks (MUSICBENCH) evaluate classification accuracy, FAD/MMD/CLAP metrics, and small-scale listening tests.

### Strengths
The proposed approach enables interpretable, fine-grained control over musical attributes such as notes, chords, and tempo without retraining or per-step optimization. The introduction of layer-aware steering, time-varying schedules, and multi-directional control makes the framework flexible and musically relevant. Empirical results demonstrate effective controllability with minimal loss of fidelity, showing a clear advancement toward interpretable and lightweight control in music generation systems.

### Weaknesses
The paper lacks essential experimental details, including the datasets, prompts, and number of samples used for evaluation. While the experiments appear to partially rely on the SynTheory dataset, this is never explicitly stated or described in detail, making it difficult to assess reproducibility. In addition, the absence of a direct baseline would be the critical issue of the paper. Most naively, authors can compare the proposed steering approach to a simple text-conditioning (e.g., simply adding “fast music” to the input prompt) limits the clarity of the paper’s contribution. Such a comparison is crucial to demonstrate the unique benefits of activation-space control over standard prompt-based methods.

**missing references**

- citation for metrics? (FD, MMD, CLAP) it’s not mentioned of which CLAP model was used for evaluation

**minor**

- Table 1: correct the caption (”We train using 7 We report…”) and place the proposed method at the most bottom row
- reporting FD and MMD together is a bit redundant
- Table 2 caption: “higher better for” → “higher is better for”
- line 315: “were randomly chosen base model …” → “were randomly chosen from base model ...”
- couldn’t find listening examples for time-based schedules from the demo page upon time of reviewing

### Questions
- Table 1: what is the implication of performance difference between proposed and RFM (last token)?
- Table 2: mae for Tempos?
- Table 2: Classification results were very high from Table 1. Doesn’t this mean it’s a bit reliable to trust the metric?
    1. If the authors claim it’s not reliable for real music (according to Table 2 caption), then correlation between human evaluation should be included to observe how different it is and find the best way for evaluation (e.g., models specific for each downstream task)
    2. If this Probe Acc is reliable, then the successful controllability is below chance level for all tasks except for Notes. Is this acceptable? Especially since there’re no other baselines to compare.
- details of participants of listening test? and 3 questionnaire seems too small
- what’s the benefit compared to diffusion-based controls
- so what is the best $\eta_0$? and any way to automate this for each steering direction?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces MusicRFM, a framework that adapts Recursive Feature Machines (RFMs) to enable fine-grained control over frozen autoregressive music generation models like MUSICGEN. By training lightweight RFM probes on the SYNTHEORY dataset, the authors extract interpretable "concept directions" corresponding to musical attributes (e.g., notes, chords, tempos). These directions are injected into the model's activations during inference to steer generation without retraining or per-step optimization. Key innovations include layer-based pruning (top-K and exponential weighting), time-varying schedules (e.g., linear fades, sinusoidal modulation), and multi-direction steering for simultaneous control of multiple attributes. Experiments demonstrate improved control accuracy (e.g., note classification from 0.23 to 0.82) with minimal impact on text prompt fidelity (CLAP score within ~0.02 of baseline), supported by quantitative metrics (FD, MMD, CLAP) and a small listening test.

### Strengths
This is a solid paper that explores the application of Recursive Feature Machines (RFMs) to music generation, specifically for steering codec-based autoregressive models like MusicGen. 

The work addresses an important challenge in controllable music generation by enabling fine-grained, interpretable control over musical attributes such as notes, chords, intervals, scales, progressions, tempos, and time signatures, without requiring model retraining or heavy optimization. 

The use of the SYNTHEORY dataset for probe training is well-motivated, as it provides clean, music-theoretic labels. 

The proposed extensions—layer pruning strategies (top-K and exponential weighting), dynamic time schedules, and multi-direction steering—are novel and practical, allowing for more robust and flexible control. 

Experiments are comprehensive, including classification results showing RFMs outperforming baselines (e.g., average score 0.942 vs. 0.929 for SYNTHEORY FFNs), quantitative metrics on steering trade-offs (FD, MMD, CLAP, probe accuracy), and a listening test demonstrating perceptual improvements. 

Overall, the paper makes a meaningful contribution to activation-level steering in generative audio models, with potential for broader applicability.

### Weaknesses
First, the related work section underestimates prior methods by claiming they require "intense finetuning runs," when many actually use parameter-efficient fine-tuning (PEFT) techniques, which are computationally lighter than full finetuning (though still more than RFMs). This portrayal lacks objectivity and overlooks the fact that some methods already achieve control over notes and chords without architectural gaps. 

Additionally, relevant papers are missing, such as Zhu et al. (2025) on efficient fine-grained guidance for diffusion-based symbolic music generation via inference-time control, and Zhang et al. (2024) on zero-shot text-to-music editing with diffusion models, which also focuses on inference-time interventions. Second, although the writing is clear overall, the methods section can be dense and challenging to follow, particularly the details on RFM adaptations, layer pruning, and schedule formulations; more explanatory text or examples would help. 

Third, objective experiments show that increasing steering strength degrades FD and MMD scores, which seem to imply a degradation in musicality as well—objective metrics like FD and MMD are designed to capture distributional shifts that often correlate with perceptual quality, suggesting that stronger steering may introduce artifacts or incoherence that harm the overall musical experience. This is my biggest concern, as it raises questions about the method's ability to maintain high-fidelity generations under aggressive control, yet the paper lacks subjective evaluations of fidelity and musicality beyond the small listening test—relying on proxies like CLAP may not fully capture these aspects, potentially limiting the method's practical value. 

Finally, the listening test protocol uses only 12 participants, which is insufficient for robust conclusions, and omits details on participant selection criteria, demographics, or distribution.

[1] Zhu, T., Liu, H., Wang, Z., Jiang, Z., & Zheng, Z. Efficient Fine-Grained Guidance for Diffusion Model Based Symbolic Music Generation. In Forty-second International Conference on Machine Learning.

[2] Zhang, Y., Ikemiya, Y., Xia, G., Murata, N., Martínez-Ramírez, M. A., Liao, W. H., ... & Dixon, S. (2024). Musicmagus: Zero-shot text-to-music editing via diffusion models. arXiv preprint arXiv:2402.06178.

### Questions
1. Could the authors address the underestimation of related work? For instance, how does MusicRFM compare directly to PEFT-based methods in terms of computational cost and control granularity? Also, why were papers like Zhu et al. (2025) and Zhang et al. (2024) not discussed, given their focus on inference-time control in music generation?

2. To improve readability, could the authors suggest additions to the methods section, such as pseudocode for the steering injection process or a simple worked example of a time schedule (e.g., linear rise) applied to a generation?

3. Regarding the trade-off between control strength and metrics like FD/MMD, do the authors have plans for larger-scale subjective evaluations of audio fidelity and musicality? How might this affect the method's usability in real-world applications, and are there mitigation strategies?

4. For the listening test, could the authors provide more details on the protocol? Specifically, what were the participant demographics, expertise levels (e.g., musicians vs. general listeners), and how were samples randomized?

### Soundness
3

### Presentation
3

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
This paper presents a training-free method to steer MusicGen with RFMs. The controllable features include tempo, chords, notes, time signatures etc.

### Strengths
1. The idea of fine tuning-free steering of multiple music concepts in a universal way is intriguing, and the time-variant control is a useful direction. 

2. In the demo page it seems that some controls (e.g., augmented chords; interval 6) are effective. But it is strange why the probing results are low (see questions 2).

### Weaknesses
1. The model focuses on global controls that are described by labels. This is not a novel task and can be done by many fine-tuning methods, using either text-based or other controls. The methodology itself is not novel either.

2. The performance of the model seems to be low. Previous works seem to provide better controllability and musicality as in Wu et al (2024) & Lin et al (2023). There is no comparative experiments against previous fine-tuning methods either.

3. Sec 5.1: For some tasks like notes, chord etc., a pretrained model (e.g., audio chord estimator) provides a better evaluation metrics compared to subjective and objective (by reusing the prober) test.

4. Line 243: incomplete sentence.

### Questions
1. Line 257: I cannot quite understand the tempo issue. what happened to the tempo category and what specific methods did you used for tempo?

2. I do not quite understand the correspondence between table 2 and the demo page. From the demo page it seems that the controllability is relatively good but the audio quality/musicality is harmed. However, table 2 shows that the quality is relatively ok but the controllability is low. Why?

3. For time varying controls (changing $\phi(t)$), do you have any demos that can produce i.e., tempo changes or time signature changes within a song? As far as I know these are very difficult for current controllable generation models. Currently, I see no time varying control in the demo page.

### Soundness
2

### Presentation
3

### Contribution
1
