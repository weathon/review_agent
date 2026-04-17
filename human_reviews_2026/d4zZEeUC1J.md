# SpeechMedAssist: Efficiently and Effectively Adapting Speech Language Model for Medical Consultation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Medical consultations are inherently speech-based, yet current works focus on fine-tuning large language models (LLMs) to perform patient-unfriendly long-text interaction. While existing speech language models (SpeechLMs) enable efficient speech-based interaction, the scarcity of speech data in medical domain prevents them from being directly fine-tuned for practical applications.
In this paper, we propose SpeechMedAssist, a SpeechLM natively capable of conducting multi-turn speech-based interactions with patients. To mitigate data scarcity, we mathematically analyze the architecture of SpeechLMs and decouple one-stage training that requires a large corpus of medical speech data into a two-stage training paradigm. (1) Knowledge\&Capability injection: train the LLM core with rewritten and filtered medical text data to inject medical knowledge and endow it with diagnostic and treatment capabilities. (2) Modality alignment: train the SpeechLM using a small amount of synthetic medical speech data that matches patient characteristics to realign the speech and text modalities. 
After two-stage training, SpeechMedAssist performs excellent on our designed speech-based medical benchmark. Experiments further show that the second stage only requires 10k speech dialogue samples to achieve modality alignment, allowing the knowledge and capabilities acquired in the text modality during the first stage to generalize to the speech modality, which demonstrates the effectiveness and efficiency of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Current medical consultation research primarily focuses on fine-tuning text-based large language models and does not support spoken interaction. To address this gap, this paper propose SpeechMedAssist, a speech language model natively capable of multi-turn spoken interactions between patients and clinicians. To overcome the scarcity of medical speech data, they introduce a two-stage alignment strategy. Experimental results show that SpeechMedAssist outperforms baseline models on the authors’ newly constructed speech-based medical benchmark.

### Strengths
1. The authors introduced a multi-turn spoken medical dialogue dataset, SpeechMedDataset, containing 198k samples, and used it to fine-tune a speech large language model, thereby obtaining a model with enhanced capabilities for spoken medical question answering. 

2. Through careful experimental analysis, the authors demonstrated the effectiveness of their proposed dataset.

### Weaknesses
1. The primary contribution of this work lies in data construction—specifically, the introduction of SpeechMedDataset, a multi-turn spoken medical dialogue dataset. However, the model architecture, training strategy, and evaluation protocol largely follow those of Llama-Omni and Llama-Omni2, demonstrating limited novelty in methodological design.

2. In Table 2, the authors do not include a direct comparison between:\
(a) the base text-only model (Qwen2.5-Instruct-7B) and its performance on medical tasks,\
(b) a cascaded pipeline using Qwen2.5-Instruct-7B, and\
(c) the performance of the text LLM after Stage 1 fine-tuning.\
Such comparisons would better illustrate the performance gap between text and speech modalities and more convincingly demonstrate the advantages of their end-to-end approach over cascaded systems.

3. It is puzzling that SMA-Stage1, after fine-tuning on LLaMA-Omni2, underperforms the original Llama-Omni2 in multi-turn medical question answering. The authors should provide an analysis or ablation study to explain this performance degradation—e.g., whether it stems from domain shift.

### Questions
Please refer to the Weaknesses.

### Soundness
2

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
5

### Summary
This paper proposes SpeechMedAssist for enabling speech interaction with patients. To address the challenge of the scarcity of speech data in the medical domain for training Speech LLM for this purpose, this paper proposes a two-stage training paradigm, with the first stage of training the text LLM backbone with rewritten and filtered medical text data to boost the diagnostic and treatment capabilities, and the second stage of speech-text modality re-alignment using synthetic speech dialogue samples that match patient characteristics. Experimental results show that the second stage only requires 10K samples.

### Strengths
1.	While the SpeechLLM follows the common architecture, the investigation of decomposing the stage of injecting domain knowledge into the backbone LLM and the stage of speech-text re-alignment for a specific domain, medical consultation, is a valuable investigation and the findings are useful for developing effective domain-specific speech interaction systems. Specifically, the original SpeechLLM is already pre-trained to establish a reasonable speech-text modality alignment. The first stage of knowledge injection only trains the backbone LLM. And the second stage involves training the speech adaptor and LLM with S2T data, and then training the speech decoder with S2S data. 

2.	The construction of the TextMedDataset and the SpeechMedDataset is reasonable and clearly presented. The data replay strategy of incorporating the single-turn QA dataset from TextMedDataset in the second training stage helps mitigating catastrophic forgetting of the medical knowledge due to the re-alignment stage.

3.	The analysis in Section 5.5 shows that training directly using medical speech data is more challenging than first helping the core LLM acquiring the domain knowledge and then conducting speech-text re-alignment.

### Weaknesses
1.	Some important related open-sourced speech LLMs are missing in the comparisons in Table 2 and Table 3, e.g., Qwen2.5-omni, MiniCPM-o 2.6, Baichuan-omni-1.5, and Step-audio2-mini. This would cause serious problems to comprehensively understand the positioning and effectiveness of the proposed approach.

2.	The first stage of continual training the core LLM with TextMedDataset also suffers from the issue of catastrophic forgetting of the original text and speech capabilities of the core LLM, which is LLaMA-Omni2-7B in this work. It would be very important to evaluate the general text and speech understanding capabilities of the SpeechLLM after Stage I training and Stage II training, using benchmarks such as MMLU for evaluating general text capabilities, and benchmarks such as OpenAudioBench, VoiceBench, Big Ben Audio for evaluating general S2T and S2S capabilities.  Otherwise, it is not clear the text and speech intelligence of the final SpeechLLM.

3.	The evaluation metrics need to be further clarified. Currently, Appendix C explains the 6-dimensions for evaluating multi-turn dialogues, yet it is not clear how they are scored and how the scores of multi-turn dialogues in Table 2 are computed. 

4.	For experimental results, it is important to report standard deviation etc to show the stability and help interpret the gains, also 
statistical significance tests need to be conducted over the gains.

### Questions
Please address the questions raised under Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SpeechMedAssist, a domain-adapted Speech Language Model (SpeechLM) designed to support multi-turn, speech-based medical consultations. The authors propose a two-stage training strategy to overcome the challenge of limited medical speech data: (1) a knowledge and capability injection phase using large-scale rewritten medical text data; and (2) a modality alignment phase using a limited amount (as low as 10k samples) of synthetic speech data.

### Strengths
1. The model achieves state-of-the-art results across a range of tasks, outperforming both traditional LLMs with ASR+TTS pipelines and other end-to-end SpeechLMs in both diagnostic performance and speech quality.
2. The two-stage training paradigm is elegant and practical, requiring only ~10k samples to effectively align modalities, a significant reduction in data and compute requirements.
3. Synthetic dataset creation process (with age/gender-aware speech synthesis) is well-considered and likely improves realism and generalizability.
4. The virtual consultation evaluations with roles for patient, doctor, and judge simulate real-world clinical interactions and offer a more grounded benchmark than static Q&A.

### Weaknesses
1. Synthetic data realism and risk of overfitting. The paper relies heavily on synthetic speech from TTS models. While they attempt to match speaker attributes (age/gender), it remains unclear how well this generalizes to real patient speech. The paper would benefit from a stronger external evaluation on real-world noisy clinical audio or an ablation comparing synthetic vs. real data alignment.
2. Although the paper evaluates safety via MedSafetyBench and claims improved performance, there is no in-depth error analysis of failure cases or adversarial behavior. Given the high-risk domain, this omission weakens the safety claims.
3. No direct comparison with instruction-fine-tuned SpeechLMs. All comparisons are with general-purpose SpeechLMs or LLMs. The paper should also compare with domain-tuned or instruction-tuned audio models, if available, or at least clarify why they are omitted.
4. Overemphasis on automatic metrics. Metrics like CMB/CMExam are appropriate, but the “vote” metric from clinicians for real-world consultations is presented without sufficient methodological detail (e.g., inter-annotator agreement, voting procedure), reducing its impact.
5. Minor hallucination in output: Figure 1 shows an example where the system incorrectly repeats “apply warm compress to your buttocks,” which is presumably a hallucination or ASR confusion. No systemic analysis of such failures is presented.

### Questions
1. How does the model handle real patient audio with varied noise levels and accents? The wild set is small (20 samples). Can you present CER or diagnostic accuracy for this subset or show breakdown by noise level?
2. What mechanisms are in place to prevent hallucinations or unsafe recommendations, especially under low-resource or ambiguous input? Is there a fallback or uncertainty-aware mechanism?
3. Why is the speech decoder untrained in Stage 1 if it contributes to the final generation? Would initializing or co-training help?
4. Did you evaluate the model’s robustness to speech disfluencies (e.g., pauses, fillers, corrections)? This is common in real patients.
5. How scalable is the approach to other domains (e.g., legal, financial) where speech data is also scarce? Do you see any domain-specific challenges?
6. What is the latency distribution across longer conversations? The average latency is good, but variance in multi-turn interaction should be shown.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SpeechMedAssist (SMA), an end-to-end SpeechLM for multi-turn medical consultations. It adopts a two-stage paradigm: (1) Stage-1 injects medical knowledge and reasoning ability by fine-tuning the LLM core with rewritten and filtered medical text; (2) Stage-2 performs speech–text alignment using a relatively small amount of synthetic medical speech to transfer Stage-1 capabilities into speech interaction. The authors claim that about 10k speech-dialog samples suffice for effective alignment, and build TextMedDataset and SpeechMedDataset along with a multi-dimensional benchmark. Results show SMA performs competitively on single/multi-turn tasks, Wild recordings, speech quality, and latency.

### Strengths
- Decoupling knowledge/ability learning from speech alignment, with a clear engineering and theoretical narrative; TTS voices are matched to patient demographics to reduce distribution shift.
- Comprehensive evaluations: knowledge QA, multi-turn dialogue, Wild noisy recordings, speech quality, and latency; comparisons with medical LLMs, ASR/TTS pipelines, and general SpeechLM baselines.
- Clear descriptions of modules (encoder, adapter, LLM core, speech decoder/vocoder) and evaluation metrics.
- Achieving alignment with about 10k speech samples lowers deployment barriers; the model shows advantages in multi-turn and Wild settings, pending stricter external validation.

### Weaknesses
1. Results may unfairly favor Qwen-style responses over other families (Llama/Mistral/GLM), as data rewriting, filtering, simulated patients, and multi-turn judges rely on Qwen-family models.  

3. Synthetic-to-real gap remains significant. Stage-2 uses only TTS speech, lacking natural hesitation, coughing, pain, accent variation, and complex noise.   Wild experiment is small and subjective. Only 20 clinic recordings with 5 expert votes; lacks objective structured metrics such as key-point coverage and prescription legality.  

 

4. LLM-as-a-judge risk of self-enhancement bias: Qwen2.5-72B is used as judge, while the evaluated model uses Qwen2.5-7B as core.  

5. Generality and portability not verified: Claims of general applicability are only demonstrated on Qwen2.5-7B.  

6. MedSafetyBench scores are reported, but failure cases (dangerous advice, hallucination, misdiagnosis) are not deeply analyzed.  Lacks discussion of refusal and referral strategies, high-risk intent detection, and confidence calibration.  

9. Missing or scattered details on prompts, rejection rules, TTS selection, speaking-rate distribution, noise injection, and parameter configs.

### Questions
1. Can the authors provide cross-evaluation using judges and patient simulators from other model families to verify robustness against bias?  
2. Is the 10k threshold robust across accents, emotions, speaking rates, and noise? Can multiple seeds and statistical tests be reported?  
3. During online dialogues, are there rule-based or knowledge-base checks before giving medical advice? Any detection of overconfidence or hallucinations with fallback to refusal?

### Soundness
2

### Presentation
3

### Contribution
3
