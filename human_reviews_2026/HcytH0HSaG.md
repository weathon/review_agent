# Backdoor Attacks Against Speech Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76\% to 99.41\%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents the systematic study of backdoor attacks against speech language models (Speech LLMs), focusing on a modified version of SpeechLLM that combines an audio encoder, projection connector, and LLM. The authors design and evaluate audio-based backdoor attacks using a short “click” trigger and test across four encoders (WavLM, HuBERT, wav2vec 2.0, Whisper) and three datasets (LibriSpeech, CREMA-D, VoxCeleb2-AE) for four tasks: ASR, emotion, gender, and age prediction. They also propose component-wise analyses to identify which modules (encoder, connector, LoRA adapters) are most vulnerable and evaluate fine-tuning-based defenses.

### Strengths
The paper tackles a novel and underexplored threat surface: backdoor attacks in speech-language multimodal pipelines. Prior works examined either LLMs or speech encoders in isolation, but this paper analyzes how backdoors propagate across components (audio → connector → LLM).

It is, to the authors’ claim, the first systematic analysis of backdoors in speech LLMs (p.1: “We present the first systematic study of backdoor attacks against a speech language foundation model…”).

The component-level attack taxonomy (Attacks 1–3) like testing frozen/trained combinations are creative and offers diagnostic insight into where vulnerabilities originate and persist.

### Weaknesses
While the paper motivates that cascading modules may expand the attack surface, it doesn’t articulate what makes speech LLMs uniquely challenging compared to prior multimodal or speech-only models. For example, temporal alignment between trigger and tokenization, or the difficulty of detecting imperceptible audio triggers in speech-text joint representations, could be emphasized more explicitly.

The chosen tasks—ASR, gender, emotion, and age—are relatively simple classification or regression tasks.

As noted in your comment, more complex reasoning or generation tasks (e.g., spoken question answering, dialogue, or instruction-following) would better represent true Speech LLM behavior and test whether semantic propagation of backdoors occurs beyond metadata prediction.

Current results mostly reflect encoder-level vulnerabilities, not higher-level reasoning corruption.

### Questions
Q1: Could you clarify what unique challenges arise when designing or defending against backdoor attacks in speech LLMs compared to unimodal speech or text LLMs? For example, does the temporal–semantic alignment make detection harder?

Q2: Why do some configurations (e.g., Attack 2.3, 3.1–3.3) fail completely?

Q3: Is the ASR token generation process inherently more robust to local perturbations, or does the text decoder ignore triggered frames?
Could this resistance be exploited as a defense?

Q4: You claim to be “the first systematic study of backdoor attacks against a speech language foundation model.” Could you clarify how this differs from prior multimodal or audio foundation model attacks (e.g., Mengara 2024; Han 2024)?

Q5:  Have you considered evaluating on more complex tasks such as spoken question answering or dialogue generation (as mentioned in §2.1)? Would the same attack transfer to these?

Q6: Fine-tuning is shown to erase the backdoor. Would partial re-initialization or adapter-level training achieve similar mitigation?
Could lightweight continual fine-tuning on new data gradually cleanse a poisoned encoder?

Q7: In real-world deployments, is the assumed poisoning vector (poisoned training data uploaded online) realistic for SpeechLLM training pipelines? How might an adversary practically insert such triggers?

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
This paper presents a systematic study of audio backdoor attacks against speech language models, a multimodal system with speech encoders and language models. Using a modified version of SpeechLLM (Rajaa & Tushar 2024), the authors embed a short typewriter-click trigger into a small subset of training samples (less than 10%) and demonstrate targeted misbehavior across four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender/age prediction).

### Strengths
1. **Novelty in the multimodal context.** The work highlights new threat surfaces arising from the interaction between pretrained speech encoders and LLMs, which have not been systematically explored before. The authors also evaluate four encoders (WavLM, HuBERT, wav2vec 2.0, Whisper) and multiple downstream tasks, establishing cross-task and cross-model generality. Component-level experiments (freezing, isolating, and re-training modules) are particularly insightful in revealing how backdoors propagate through modular pipelines.

2. **Empirical Results.** High AER with negligible benign degradation show the attacks are both effective and stealthy. The encoder-centric vulnerability finding is convincing and actionable for future model auditing. 

3. **Reproducibility.** The paper provides explicit details on trigger generation, poisoning ratios, dataset splits, and fine-tuning procedures.

### Weaknesses
1. **Overstated Novelty Claim ("first study")**. The paper claims to be the first to present backdoor attacks on speech models. This could be an overstatement because several prior works already demonstrated backdoor or trigger-based poisoning in speech tasks ASR: Ultrasonic and inaudible-trigger backdoors [1],  EmoAttack and EmoBack [4, 6] in Speech Emotion Recognition, and the list continues with various backdoor attack methods in speech LLMs [2, 3, 5].

2. **Limited Attack Diversity.** Only one trigger type (a 220 ms typewriter click) and one dirty-label poisoning scheme are used. Clean-bale or imperceptible trigger variants would strengthen realism.

3. **Limited Model Scope.** Experiments are limited to SpeechLLM. Testing on other recent Speech LMs (e.g., SALMONN, SpeechGPT) would improve generality claim. However, as a reviewer, the paper could only be explained the fact that the method will only work for a single model.

4. **Lack of deeper mechanistic insight.** The analysis remains empirical. No latent-space visualization or spectral-signature inspection is provided to explain *why* the encoder demonstrates backdoor propagation.



----
**References**

[1] Koffas, Stefanos, et al. "Can you hear it? backdoor attacks via ultrasonic triggers." Proceedings of the 2022 ACM workshop on wireless security and machine learning. 2022.

[2] Zhai, Tongqing, et al. "Backdoor attack against speaker verification." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.

[3] Liu, Xinpeng, et al. "Cuckoo Attack: Stealthy and Persistent Attacks Against AI-IDE." arXiv preprint arXiv:2509.15572 (2025).

[4] Yao, Wenhan, et al. "Emoattack: Utilizing emotional voice conversion for speech backdoor attacks on deep speech classification models." arXiv preprint arXiv:2408.15508 (2024).

[5] Yan, Baochen, Jiahe Lan, and Zheng Yan. "Backdoor attacks against voice recognition systems: A survey." ACM Computing Surveys 57.3 (2024): 1-35.

[6] Schoof, Coen, et al. "Emoback: Backdoor attacks against speaker identification using emotional prosody." Proceedings of the 2024 Workshop on Artificial Intelligence and Security. 2024.

### Questions
1. If one task (e.g., emotion) is poisoned, does the backdoor transfer to other tasks? As a reviewer, I am curious about the transferability of the backdoor

2. After fine-tuning on a clean dataset, does re-exposure to limited poisoned data re-activate the backdoor?

3. As modern Speech LLMs also accept textual inputs, could textual or instruction-level triggers induce backdoors that activate through the audio pathway or vice versa?

### Soundness
2

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
5

### Summary
This paper presents the first systematic study of audio backdoor attacks against multimodal speech language models (SpeechLLM). Using a modified SpeechLLM pipeline, the authors demonstrate how a single audio trigger (a clicking noise) injected during training can compromise four tasks: ASR, emotion recognition, gender, and age prediction. Experiments across four encoders and three datasets  show high attack effectiveness.

### Strengths
1. Extensive evaluation across diverse encoders, tasks , and datasets strengthens empirical claims.
2. As the first study on backdoors in speech-language models, it fills a gap in multimodal security literature.

### Weaknesses
1. Lack of Threat Model. Key questions remain unanswered: What are the attacker’s capabilities (e.g., manipulating training data vs. model weights)? What are realistic deployment scenarios? Without this, the claimed "systematic study" feels ungrounded.
2. Limited Practical Relevance and Methodological Innovation. The study employs conventional dirty-label poisoning (Gu et al., 2017), merely transferring this paradigm to speech-language models. Minimal novelty is demonstrated beyond trigger adaptation (an acoustically inconspicuous click stimulus), representing incremental advancement without fundamental innovation.
3. The defense analysis exclusively focuses on fine-tuning methodologies. Benchmarking against contemporary defense frameworks, such as activation clustering (Chen et al., 2018), spectral signature detection (Tran et al., 2018), or adversarial purification, is notably absent.
4. Attack effectiveness collapses in propagation scenarios for ASR (0% AER in Table 3). Though the authors hypothesize task complexity as a factor, insufficient mechanistic analysis of representation propagation, particularly regarding why ASR embeddings resist backdoor transfer while emotion representations remain vulnerable, weaken robustness insights. Rigorous ablation studies examining input granularity or latent space dynamics are warranted.

### Questions
Please refer to Weaknesses.

### Soundness
3

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
3

### Summary
This paper presents the first systematic study of backdoor attacks against cascaded Speech Language Models, using a modified SpeechLLM architecture as a case study. The authors demonstrate an effective "dirty-label" attack using an imperceptible audio click as a trigger. The attack was evaluated across four different pretrained speech encoders (WavLM, HuBERT, etc.) and four different tasks (ASR, emotion, gender, age), achieving high attack success rates (over 90%) while maintaining stealth (minimal impact on benign sample performance). A key contribution of this paper is the component-level analysis, which isolates the audio encoder, connector, and LoRA adapters, identifying the encoder as the most critical component for learning and propagating the backdoor. Finally, the paper proposes and evaluates a fine-tuning-based defense, showing that full fine-tuning on clean data can effectively remove the backdoor.

### Strengths
**Originality and Significance:** This paper addresses a novel and important problem: the security of multimodal speech-language models. To my knowledge, this is the first work to systematically study the propagation of backdoor attacks in this specific type of cascaded architecture. This contribution is very timely and significant given the increasing popularity of such models.

**Quality and Clarity:** The experimental design is rigorous and well-structured. The component-level analysis (Attacks 1-3) is particularly excellent, as it not only proves the attack's effectiveness but also clearly reveals how it works. Identifying the encoder as the primary vulnerability is a key finding.

**Thoroughness:** The evaluation is comprehensive, covering multiple encoders, multiple datasets, and several different types of tasks (classification, regression, and sequence generation), which strongly supports the generalizability of the findings.

### Weaknesses
**Limited Scope of Attack:** The study focuses on only one attack type (dirty-label) and one specific trigger (a click sound). While this attack is effective, it is unclear if these findings generalize to other, potentially more stealthy, attack vectors (e.g., triggers with different acoustic properties, clean-label attacks).

**Practicality of Defense:** The proposed defense (full fine-tuning) is effective but has a significant practical limitation: it requires a large and guaranteed "clean" dataset. This may be difficult to obtain in real-world scenarios where a model might be poisoned from a large, web-scraped corpus. Exploring more practical defenses (e.g., trigger detection, data sanitization, pruning-based defenses) would make the paper more complete.

**Single Model Architecture:** The study is limited to a specific SpeechLLM architecture. Although using four different encoders provides some generality, the findings regarding the connector and LoRA adapters are specific to this cascaded design. It would be more valuable to discuss how (or if) these findings might transfer to other multimodal fusion strategies (e.g., models using cross-attention).

### Questions
1. The ASR attack requires a repeated trigger, while other tasks do not. Could the authors elaborate on why they believe this is the case? Does this imply a fundamental difference in how the model processes temporal information for ASR versus for classification tasks like emotion or gender?

2. Regarding the fine-tuning defense (Table 4), full fine-tuning on CREMA-D-clean (on the Attack 3.1 model) eliminated the backdoor (AER 19.12%), but partial fine-tuning did not (AER 95.44%). This suggests the backdoor is deeply embedded. Is this consistent with the finding that the encoder (which was poisoned and frozen in Attack 3.1) is the primary vulnerability? Can the authors provide more insight into this interaction?

3. The component propagation attack (Attack 3.1) successfully propagated the backdoor for the emotion task but failed completely for the ASR task (AER 0.00%). This is a very interesting and stark contrast. What is the authors' hypothesis for this? Does it suggest that ASR training on clean data is sufficient to "overwrite" the trigger's effect, whereas the emotion task is not?

### Soundness
2

### Presentation
3

### Contribution
3
