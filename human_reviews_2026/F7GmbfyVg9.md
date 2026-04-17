# FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
This study proposes FlexiVoice, a text-to-speech (TTS) synthesis system capable of flexible style control with zero-shot voice cloning. The speaking style is controlled by a natural-language instruction and the voice timbre is provided by a speech reference in zero-shot manner. FlexiVoice is built with an LLM core, which takes text as input, and also takes an optional natural language instruction and an optional speech reference to control style and timbre, respectively. FlexiVoice is equipped with a novel Progressive Post-Training (PPT) scheme that progressively unlocks accurate and flexible controllability. In particular, it first employs Direct Preference Optimization (DPO) to enable FlexiVoice to accurately follow both natural language instruction and speech reference simultaneously. It then uses a multi-objective Group Relative Policy Optimization (GRPO) to disentangle style instruction, reference timbre, and textual content. Finally, it adapts instruction GRPO for more advanced instruction following. Experimental results show that FlexiVoice surpasses competing baselines and demonstrates strong capability in decoupling control factors. Human evaluations further confirm its naturalness, controllability, and robustness.
Audio samples are available at https://flexi-voice.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presented a TTS system that follows natural language instructions. The authors proposed to increase the synthesis naturalness and the instruction-following capability through a three stage Progressive Post-Training process.

### Strengths
The method is straightforward and easy to follow. The presentation is clear. The system performs reasonably well against several well-know baselines.

### Weaknesses
The main weakness of the paper is the experiment design does not validate the main contribution claimed by the paper.
The paper claims that the main novelty lies in the three stage Progressive Post-Training, which makes sense as the other parts of the system follows the commonly-used instruction-following TTS.
However, in the experiments (Table 5, Section 5.6), the paper only showed the efficacy of having multiple alignment objectives (S1, S2, S3). It does not justify why the "Progressive" training of the three objectives is needed, other than simply post-training with the three objectives all-together. Also, is the order (S1->S2->S3) important? 

The three objectives (S1, S2, S3) to post-train TTS, as already mentioned in Section 3 by the authors, are adapted from existing works (e.g., line 191, 217). This makes me think that the novelty of the paper is in progressively applying them, as also alluded to several times in Section 1. However, there is no evidence showing that the progressive part is crucial to the performance in the experiments. The authors should either explain the main distinctions between the previous approaches (are there anything novel on the post-training objective?), or show the efficacy of the progressive training.

As for now, I question if the scientific contribution is enough for this paper to be accepted to ICLR, based on the reasons mentioned above.

- Typo: S"2" -> "3": Instruction GRPO in Figure 1

### Questions
See weaknesses. The authors should address the points to convince me that the paper is of acceptance quality.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SUPM-TTS, a framework for style-universal text-to-speech generation that aims to synthesize expressive and natural speech in any speaking style, even unseen ones. The model integrates multiple components—adaptive prompt modulation, disentangled representation learning, and multi-corpus training—into a unified TTS system. The authors also construct new datasets and conduct extensive experiments. Overall, this is an ambitious attempt to generalize TTS systems beyond fixed style categories.

### Strengths
- The paper presents an extensive and well-engineered system that integrates multiple components and datasets into a cohesive framework, which is commendable.

- It targets the problem of style-universal speech synthesis, addressing the need for flexible, controllable, and expressive TTS generation.

- The proposed prompt-based approach allows conditioning on various prosodic or emotional cues, contributing to improved speech diversity and controllability.

- The work offers thorough evaluations, including MOS, WER, and SV metrics, across several datasets.

### Weaknesses
### **Problem & Motivation**

- The motivation could be strengthened by explaining how this approach differs from recent prompt-based or diffusion-based TTS systems. The overall goal overlaps with several existing works such as FlexiVoice, PromptTTS, and StyleSpeech, making it difficult to isolate what is novel.

- The paper mentions disentangling timbre and style, but it is unclear how this disentanglement is achieved or guaranteed. Without a clear mathematical constraint or empirical validation, the claim of disentanglement remains qualitative.

### **Method**
- In general, the method novelty is limited. The proposed framework combines known elements—prompt conditioning, disentanglement, and adaptation—without introducing fundamentally new mechanisms. The design could be more appropriately positioned as an integration of recent techniques like DPO or GRPO for controllable TTS rather than as a new paradigm.

- The claim that the system generalizes to unseen styles is only partially supported; the method seems to rely heavily on the diversity of training prompts rather than a specific generalization mechanism.

### **Experiments**

- The experiments are extensive, but the SV metric results are relatively weak. In several cases, FlexiVoice-Base outperforms FlexiVoice, contradicting the expected hierarchy. The paper should provide an explanation for this discrepancy—perhaps due to model capacity, over-regularization, or mismatch in style conditioning.

- The subjective demo results do not convincingly show that either model performs consistently well. The samples suggest variability across styles, which should be discussed rather than glossed over.

- The paper should clarify whether Chinese experiments use CER (Character Error Rate) and how this metric is computed. Similarly, the reported WER differences may stem from ASR models underperforming on expressive or emotional speech, which should be explicitly acknowledged.

### **Writing**
- The current experimental sections (5-1, 5-2, 5-3) are fragmented. It would be better to merge them into a single “Experimental Setup” subsection, organizing content at the paragraph level for better flow and readability.

### Questions
Please refer to the Weaknesses above.

### Soundness
3

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
This paper introduces FlexiVoice, a text-to-speech (TTS) model which can control from both natural language instruction and speech prompt for voice timbre.  This model is built on a pretrained LLM (phi-3-mini-instruct) and pretrained on Emilia and a diverse set of speech dataset. 
The major challenge for this instruct control TTS is the multi-modality controllability and disentanglement of instruction, text and speech reference (e.g. conflict between style and text content or voice timbre). 

The major contributions for the paper are 

1). develop a large-scale and diverse speech dataset with instructions (4316 hours) as FlxiVocie-Instruct. The annotations are generate from Deepseek-v3 and speech are in-the-wild speech dataset with diverse coverage and expressive richness such as game, podcast, etc.

2).  propose a progressive post-training (PPT) including 3 stage training as DPO for multi-modality controllability, decoupling GRPO for disengaging speaking style and voice timbre, instruction GRPO for instruction following ability enhancement. 

Experiments show FlexiVoice perform strong capability in decoupling factors and maintain the TTS quality measured by human judgements on naturalness, controllability and robustness.

### Strengths
1. The novel proposal of curriculum learning framework with DPO + 2 stage GRPO achieve the target to control (DPO initial alignment) and GRPO disentanglement and generalization. The PPT schema methodologies are successfully applied to achieve the goal which demonstrate the alignment strategies progressively to realize the controllability and disentangle style instruction, reference timbre, and text contextual content.  

2. The instruct dataset construction and evaluation set design as text-only (TO), text and text and reference (TR) and their easy and hard set. The data pipeline construct based on a in-the-wide diverse and expressive dataset and scalable automatic annotation. The evaluation test set design are also thoughtful and convincing to identify the performance.

### Weaknesses
1.  The paper claims "can speak in any style with any voice", however, the instruction motioned as only 5 emotion as neutral, happy, angry, sad and surprised in training and benchmark (TO and TR). There evaluation seems don't clearly demonstrate any other emotion or other natural language style instruction or conflict voice timbre in the following subjective test which make the title is a little overclaimed. 

2. For the evaluation, only first evaluation set involve human judge as CMOS test, and it outperforms cosyvoice2 when the hard test, but not on the (ZH) easy test. There isn't subject evaluation on complex instruction-following or open instruction evaluation. As there are two reward model (Kimi-Audio-7B) and Emotion2Vec introduced in the GRPO, however, there are not subjective evaluation in the experiment which prove they are highly related to human perception.

### Questions
1. It seems the introduction of dataset for pretraining on FlexiVoice-Base. Paper mentions the FlexiVoice Instruct as 4316 hours which is used in pre-training stage. Does it mean it use the same corpus with the alignment stage? 
2. Is there any quality comparison like between the FlexiVoice-Base and FlexiVoice? How many data pair used in the S1 DPO stage? And, if the pre-training stage add these instructions, how dose it compare with DPO alignment?

### Soundness
3

### Presentation
3

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
The paper proposes a text-to-speech (TTS) system that can generate speech in any speaking style with any voice using natural language instructions for style control and reference speech for timbre control in a zero-shot manner. The model is built upon a  large language model (LLM) and trained using a novel Progressive Post-Training (PPT) scheme with three stages. The authors also introduce a new dataset, FlexiVoice-Instruct, annotated with natural language instructions via LLMs. Experiments on emotion datasets and the InstructTTSEval benchmark show FlexiVoice surpasses baselines (e.g., VoxInstruct, CosyVoice2) in style-timbre disentanglement, controllability, and naturalness.

### Strengths
1. This work bridges instruction-based and zero-shot TTS research, demonstrating how LLM-based reinforcement learning can generalize across diverse speaking scenarios. The progressive reinforcement learning curriculum (PPT) is novel in the context of TTS controllability. The unified design for style–timbre disentanglement with both natural language and reference inputs is a meaningful advance beyond previous instruction-based or zero-shot models.

2. This work addresses a critical and challenging problem in controllable TTS: achieving independent, flexible control over multiple attributes. The demonstrated ability to decouple style from timbre and content is a step forward. 

3. The experimental demonstrates clear and substantial improvements over a wide array of strong baselines. The construction of the large-scale FlexiVoice-Instruct dataset is a major undertaking that adds considerable value to the community.

### Weaknesses
1. The decoupling evaluation is heavily focused on emotional control. While emotions are a key aspect of style, the paper would be strengthened by a more direct analysis of other stylistic aspects (e.g., speaking rate, pitch range, informal tone) to fully validate the "any style" claim. A qualitative analysis or case studies on non-emotional instructions would be valuable. 

2. The progressive training scheme involving multiple stages of DPO and GRPO is computationally intensive. The paper does not report training time, GPU requirements, or model size, which are critical for replication.

3. Although Table 5 shows training-stage effectiveness, a finer-grained analysis of reward function contributions (DPO vs. GRPO weighting) would help interpret why specific stages contribute differently.

### Questions
Some issues and statements need to be clarified:

1. How does FlexiVoice perform on unseen speaking scenarios (e.g., narrative, dialogue) or in low-resource languages? How sensitive is the system to the balance between style fidelity and timbre preservation?
    
2. In Section 3.1, the authors mention using "Speak the following text" as a default instruction for data without explicit instructions. For the multi-modality tasks (TR) during inference, what is the precise input format when "only" a reference speech is provided without a style instruction? Does the model default to a neutral style, or does it attempt to infer a style from the reference? 

3. In Section 3.2.3, you mention that for S3, you "discard references" to avoid conflict with open-ended instructions. However, in Appendix A.4, it is mentioned sampling "existing instruction-text inputs from the pre-training data." Please clarify if these pre-training data points included reference speech, and if so, how was this handled? Were the references simply omitted during S3 training?

4. The paper mentions that Kimi-Audio-7B was chosen over Gemini-2.5-Pro due to cost. Any validation is performed to ensure that the ranking or preference judgments from Kimi-Audio are sufficiently aligned with human preferences or with the more powerful Gemini model for your specific task?

### Soundness
3

### Presentation
2

### Contribution
2
