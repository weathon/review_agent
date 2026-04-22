# EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 8, 6, 6, 6

## Abstract
Emotional information in speech plays a unique role in multimodal perception. However, current Speech Large Language Models (SpeechLLMs), similar to conventional speech emotion recognition (SER) systems, still treat emotion understanding as a simple classification problem. This provides limited interpretability of predictions, while leaving the LLMs’ expressive and reasoning capabilities underutilized. In this work, we take the first step to reformulate SER as a deep reasoning problem through reinforcement learning (RL). We propose EmotionThinker, which is designed to generate accurate emotion predictions with interpretable explanations grounded in fine-grained acoustic cues. To achieve this, we first construct EmotionCoT-35K, an emotional reasoning dataset with Chain-of-Thought annotations and detailed captions. Second, we observe that current SpeechLLMs exhibit weak prosody perception, whereas prosodic cues constitute fundamental signals for interpreting emotions. To address this, we develop the prosody-enhanced foundation model EmotionThinker-Base, and demonstrate that prosody enhancement improves emotion understanding. Third, we introduce Group-Relative-Policy-Optimization with Progressive-Trust-aware-Reasoning-Reward (GRPO-PTR}) for RL. Different from standard GRPO, which relies only on rule-based outcome rewards, GRPO-PTR progressively introduces reasoning reward, dynamically adjusts it with a trustworthiness weight reflecting the alignment between reasoning and outcome, and evaluates the overall reasoning quality with a reward model based on multi-dimensional criteria. EmotionThinker outperforms previous state-of-the-art evaluation models both in emotion accuracy and explanation quality, advancing SER toward interpretable multimodal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces **EmotionThinker**, a novel model for speech emotion reasoning that aims to reframe Speech Emotion Recognition (SER) from a simple classification task into a deep, explainable reasoning problem. The core contribution is the design of a **Prosody-Aware Reinforcement Learning (RL) framework**. This framework guides a Large Language Model (LLM) to generate coherent, feature-grounded text explanations (i.e., reasoning paths), thereby bridging raw acoustic signals, the textual reasoning process, and the final emotional label. This innovative approach addresses the critical lack of interpretability in existing SER systems and SpeechLLMs.

### Strengths
- **Originality (Originality):** Very High. The combination of SER, LLM, and RL specifically tailored for generating prosody-grounded explanations is highly novel within the speech community.
- **Quality (Quality):** High. The approach moves beyond simple performance metrics by incorporating explanation quality into the optimization objective, indicating a rigorous research focus on a complex problem.
- **Clarity (Clarity):** Good. The core idea is presented clearly, and the reasoning process (via case studies) is easily digestible.
- **Significance (Significance):** Substantial. This work significantly pushes the boundaries of transparency and trust in SpeechLLMs for emotional tasks, which is an important step for the future of multimodal AI.

### Weaknesses
1. **Technical Granularity of RL and Prosody Integration (Critical):**
	- The paper emphasizes **"Prosody-Awareness,"** but the precise and **explicit mechanism** by which the RL framework guides the LLM to attend to the *most critical* prosodic features (e.g., sudden pitch shifts over average pitch) needs more profound elaboration.
	- The **RL Reward Function** is paramount. A detailed ablation study is essential to show how the different components of the reward (classification accuracy vs. fluency vs. factual grounding to acoustic features) are balanced and how this balance impacts the quality and faithfulness of the final explanation. The current description suggests this crucial balance may be underspecified.
2. **Data and Generalization Concerns:**
	- Training LLMs via RL for generation tasks often relies heavily on high-quality **human-annotated reasoning path data** for initial Supervised Fine-Tuning (SFT) or as part of the reward signal. The paper must provide a candid discussion on the cost and scarcity of this data and how the model manages to generalize its reasoning to novel or atypical speech examples outside the training distribution.
	- Generalization to different languages or accents, crucial for a model involving LLM-style reasoning, is also a concern that needs addressing.
3. **Efficiency and Deployment Feasibility (Practicality):**
	- The combination of LLM and RL training typically incurs a substantial computational overhead. The paper is currently lacking in a detailed analysis of the **training efficiency, required computational resources (GPU-hours)**, and most importantly, the **inference latency** compared to existing, lightweight SER systems. This is vital for assessing the model's practical viability for real-world deployment.

### Questions
1. RL Reward Faithfulness and Ablation (Critical)

	Provide an **Ablation Study** on the RL reward components. How do you ensure the explanations are **truly faithful to prosodic facts** and not just syntactically fluent fabrications?

2. Technical Mechanism of Prosody Grounding

	Clarify the **explicit mechanism** that links a generated text token (e.g., "high pitch") to the **specific, salient acoustic feature** in the speech input.

3. Practicality and Efficiency Analysis

	Provide detailed **Inference Latency** and **Training Cost (GPU-hours)** analysis. Is this model practically deployable compared to lightweight SER baselines?

4. Generalization

	Show **Out-of-Distribution (OOD)** results to prove that the learned reasoning generalizes beyond the training corpus.

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
This paper proposes EmotionThinker, a novel framework for explainable speech emotion recognition (SER) that leverages CoT reasoning and reinforcement learning (RL) to move beyond standard categorical classification. The authors introduce EmotionCoT-35K, a large dataset with CoT annotations and fine-grained prosodic and semantic factors tailored for emotion reasoning. They further proposed an RL-based optimization framework (GRPO-PTR) that incorporates a progressive, trust-aware reasoning reward, balancing outcome accuracy and reasoning quality. Extensive experiments over four benchmarks and ablation studies demonstrate that EmotionThinker achieves superior emotion recognition accuracy and richer, more interpretable explanations compared to a wide range of baselines.

### Strengths
1. The reformulation of SER as a deep reasoning task—rather than mere label prediction—is timely and promising for advancing interpretability in multimodal LLMs.
2. The proposed dataset, EmotionCoT-35K, fills a significant gap with CoT-style, prosody-aware emotion reasoning data, with a scalable, largely automated annotation pipeline. This may have value for the broader community.
3. The proposed reinforcement learning scheme employs progressive reward scheduling and a trustworthiness weight to dynamically balance outcome and reasoning reward signals. This helps mitigate reward hacking and stabilizes training and may be meaningful for the LLM RL community.

### Weaknesses
1. The data construction pipeline heavily relies on LLMs, and the reasoning trace data is constructed with GPT4o without the actual speech input. This may lead to unexpected failure and bias in the dataset. It would also be beneficial to input the speech and conduct a human review of the data quality.
2. The proposed reward model plays a critical role in the RL process. However, there is little discussion or quantitative validation of its calibration. The distributions of GPT-annotated versus human-annotated reward scores are not directly compared.
3. The description of the RL part is not very easy to follow. It would be better to improve the logit flow in this part.

### Questions
1. Can the authors provide more analysis comparing the similarities and differences between GPT-4o-based and human-based scoring for CoT data and reasoning reward trace quality? Are there specific failure modes or biases in the model-synthesized data?
2. Are there statistics on annotation accuracy for the automated annotations (prosody, stress, speaker traits) used in EmotionCoT-35K? Do certain emotion categories or speaker groups have systematically noisier annotations or explanations?
3. Typos: Line 270, Appendix without reference.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
First, I would like to summarize the contributions of the work by reading the abstract.

First, a speech emotional dataset was constructed. Second, current speech LLMs have weak prosody perception. This work tries to address this issue by developing a prosody-enhanced foundation model. Third, a new type of reinforcement learning protocol is proposed, which progressively introduces a reasoning award by dynamically adjusting it with trustworthy weights, reflecting the alignment between reasoning and outcome.

### Strengths
**1.** The motivation of the work is clearly stated and explained.

**2.** A first RL-based emotion recognition that has the ability not only for accurate classification, but detailed reasoning rationales and informative captions for the audio.

**3.** Each stage of the proposed framework is clearly defined.

**4.** The evaluation and abolition are comprehensive.

### Weaknesses
**1.** For the accuracy of emotion recognition, I would also like to know the performance on each individual discrete emotion. That way, we can have a more concrete and detailed understanding of the framework's capabilities and limitations.

**2.** To construct the reasoning responses, is there a specific reason that only GPT 4.0 is used?

### Questions
**Other comments:**

For section 3.1, the authors discussed the open-sourced datasets they used to construct the EmotionCot-35k.  I think authors could have a brief discussion on other related, multimodal datasets in the paper as well, whether to use them for constructing the new dataset or not. Such as the following:

MSP-PODCAST (the most recent, final version): Busso, Carlos, et al. "The msp-podcast corpus." arXiv preprint arXiv:2509.09791 (2025).

MERSA: Zhang, Enshi, Rafael Trujillo, and Christian Poellabauer. "The MERSA dataset and a transformer-based approach for speech emotion recognition." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

CMU-MOSEI: Zadeh, AmirAli Bagher, et al. "Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018.

DECAF: Abadi, Mojtaba Khomami, et al. "DECAF: MEG-based multimodal database for decoding affective physiological responses." IEEE Transactions on Affective Computing 6.3 (2015): 209-222.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper explores an interesting problem by extending emotion modeling from classification to reasoning with promising results. However, the methodology and training details are unclear, reproducibility is lacking due to missing code release, and definitions of prosody and emotional cues need stronger justification and clarity.

### Strengths
The motivation is clear, and the research problem is interesting, as it extends beyond improving emotion classification toward developing deeper reasoning capabilities.

The proposed model demonstrates strong performance in both emotion recognition and emotion reasoning, providing valuable insights for advancing SpeechLLMs toward more effective emotion reasoning capabilities.

### Weaknesses
It is unclear how your model is trained and how it builds upon Qwen2.5-Omni-3B. Please clarify the training process and provide clear explanations for all symbols and notations in your equations, as they are currently difficult to interpret.

The methodology section, particularly Section 3.3, lacks clarity. Please provide a clear description of the overall training pipeline and explain the motivation behind each step. The writing in Section 3.3.1 should be further improved for better structure and readability. Additionally, clarify the purpose of the forward reward and outcome accuracy reward, why both are needed, how they relate to the components shown in Figure 3, and what their specific inputs and outputs are.

Will you release your code and dataset? The reproducibility checklist is missing, and without a clear commitment to open-sourcing these resources, the paper’s reproducibility and credibility are severely limited. I may not be able to recommend acceptance unless this issue is properly addressed.

### Questions
How do you handle emotional cues that arise from linguistic content? For example, if the text conveys a happy emotion but the corresponding speech expresses sadness, which modality is prioritized in your final emotion prediction? Does emotion inferred from text affect the overall performance of your model?

Please provide both theoretical justification and experimental evidence to support your claim that “prosodic signals are core carriers of emotional intent.” How do you account for the role of textual content and nonvocal components (e.g., crying, laughter)? If you argue that prosody is the most dominant factor, please include empirical evidence demonstrating that prosody contributes more significantly to emotion perception than textual and nonvocal cues.

Please clarify how you define prosody. Are speaker traits such as gender and age also included under this term? Appropriate literature references should be provided to accurately define both “prosody” and “speaker traits,” as some of the current definitions appear to be inaccurate.

### Soundness
3

### Presentation
2

### Contribution
3
