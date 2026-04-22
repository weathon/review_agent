# Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 4, 8

## Abstract
Multimodal Large Language Models (MLLMs) have achieved notable success in enhancing translation performance by integrating multimodal information.
However, existing research primarily focuses on image-guided methods, whose applicability is constrained by the scarcity of multilingual image-text pairs.
The speech modality overcomes this limitation due to its natural alignment with text and the abundance of existing speech datasets, which enable scalable language coverage.
In this paper, we propose a Speech-guided Machine Translation (SMT) framework that integrates speech and text as fused inputs into an MLLM to improve translation quality.
To mitigate reliance on low-resource data, we introduce a Self-Evolution Mechanism.
The core components of this framework include a text-to-speech model, responsible for generating synthetic speech, and an MLLM capable of classifying synthetic speech samples and iteratively optimizing itself using positive samples.
Experimental results demonstrate that our framework surpasses all existing methods on the Multi30K multimodal machine translation benchmark, achieving new state-of-the-art results.
Furthermore, on general machine translation datasets, particularly the FLORES-200, it achieves average state-of-the-art performance in 108 translation directions. Ablation studies on CoVoST-2 confirms that differences between synthetic and authentic speech have negligible impact on translation quality. The code and models are released at https://github.com/yxduir/LLM-SRT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a Speech-guided Multimodal Machine Translation (SMMT) framework that integrates both speech and text as fused inputs into a multimodal large language model (MLLM) to enhance translation quality. The system comprises a multimodal LLM and a text-to-speech (TTS) module, augmented by a self-evolution mechanism that enables iterative model refinement. Experimental results demonstrate competitive performance on traditional machine translation benchmarks and promising improvements on the Multi30K multimodal dataset.

### Strengths
- Innovative self-evolution mechanism: The framework introduces a novel self-improvement process that leverages translation performance metrics as optimization objectives, facilitating continuous enhancement through iterative evolution cycles.
- Comprehensive multimodal and multilingual support: The approach effectively unifies speech and text modalities and extends to multiple languages, showing potential for broader generalization.

### Weaknesses
- Limited evaluation scope: Experiments are primarily conducted on a small set of benchmarks (e.g., Multi30K, FLORES-200), leaving open questions about generalization to large-scale and domain-diverse datasets.
- Dependence on external components: The framework relies heavily on open-source models, such as the TTS model, which may constrain performance consistency and scalability.
- Lack of detailed ablations: It remains unclear how much each component (speech input, multimodal fusion, self-evolution) contributes to the observed performance gains.

### Questions
- Could the authors clarify the conceptual and methodological differences between the proposed self-evolution mechanism and reinforcement learning approaches typically used for LLM fine-tuning?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, a Speech-guided Multimodal Machine Translation (SMMT) framework is proposed. The proposed system accepts textual input and synthesizes speech via the TTS model. Then, the MLLM processes both the text and synthetic speech to generate translations. The model is trained with two stages: pre-training and self-evolution training. The pre-training includes ASR to align speech and text modalities, speech translation and MMT to enable model translation capacity. The self-evolution training is a loop based training, i.e., TTS generates speech with different voices, and are assigned negative/positive tags by comparing the MT score and MMT score. Only those positive samples are used to update the MMT model. The model is evaluated with COMET score. The whole training loop is repeated if the model keeps improving COMET score. 
The experimental results show the proposed system can achieve very competitive results on the MLLM dataset Multi30K, and MT dataset FLORES-200. 
There are some details are missing in the paper and more detailed analysis is required. For example, what're the key factors that synthetic voice could be a positive sample or beneficial for MMT? Why speech data could help to reduce under-translation issues?

### Strengths
- Propose a Speech-guided Multimodal Machine Translation (SMMT) framework
- The system achieves very competitive results on multiple datasets

### Weaknesses
- It is understandable that information from multiple sources might help the machine learn better representation. But those features are real data instead of synthetic data. In this work, synthetic speech is used augmented MMT. Deep analysis is required to reveal the main factors that contribute the good performance. For example, we don't know the synthetic speech samples are negative or positive during inference time. What's the impact if the provided speech sample is negative one? How to select the reference voice? 
- The self-evolution training could be very expensive. It is helpful to provide the training time spend on this stage.

### Questions
- ``We synthesize speech for the evaluation text using a fixed reference voice''. The model is updated with different voices (sec 2.4.1). How do you select the reference voice?
- what're the sizes of models in table 3?
- why speech data helps to reduce under translation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses Speech-guided Multimodal Machine Translation (SMMT), i.e., improving text translation by providing semantically equivalent speech at the input. During evaluation, this "missing" speech is generated by TTS.
The model uses a Whisper speech encoder which is frozen, a speech adapter based on Q-Former and an MLP, and a strong multilingual LLM based MT system (GemmaX2-28-9B) which is LoRA adapted.

The approach uses a multistage curriculum pretraining with progressively complex objectives: ASR, S2TT and SMMT. The authors also propose a "self-evolution algorithm", i.e. iteratively complementing the training data with synthesized speech while making sure that the additional speech improves performance.

The model is compared with other multimodal MT system which use the image modality (authentic and synthetic images) on the Multi30k benchmark. It performs best for 5 out of 6 tasks. The model also outperforms SOTA on Flores200, but the model is only test on 28 languages. Also, the base MT system, i.e., to which speech input is added, already outperforms the other systems for 5 out of 6 languages.

### Strengths
It's interesting to see that adding synthetic speech to the input of a text translation system improves performances. This seems to be more effective than adding images to the input (cf. Table 3).

The incremental training strategy is interesting and I wonder whether similar techniques could be used in other LLM settings than MT.

### Weaknesses
The experimental results show that the method works, but the authors fail to justify, or to try to explain, why it works! The only argument is that the prosody in speech helps translation. I could agree with this if human speech were used. However, this work addresses the issue how to improve text translation by providing in addition *synthetic speech*. I am not well aware of the current SOTA in TTS, but I would be surprised that the prosody is very rich.

It is not clear which data was used to train the speech and LLM adapter. The data mentioned in Table 9 gives no data sizes. I wonder whether the observed improvements are mainly the result of the additional training data. An important ablation would be to LoRa adapt the MT systems on the data, but without speech input.

In Table 4 and 12, good improvements are reported for several languages when translating from 4 languages into 27 foreign languages. Given that the algorithmic improvements are on the input only, I am puzzled how we can achieve a large variance on the improvements when translating from the same language into many different ones. The paper would benefit from a deeper analysis, instead of numerical metrics only.

I would appreciate some examples, i.e., some sentences from Flores that are better translated with speech augmented input.

### Questions
- it's unclear on what data the model was trained on. Table 9 mentions Fleurs train, Common Voice train and Multi30k, but no sizes (per language) are given.

- why you support 28 languages only? Is this limited by the TTS system? What type of training data do you need to scale to more languages?

- the gains form the baseline text-only MT system and the proposed SMMT-10B are much higher for FLICK and MSCOCO than Flores. How can you explain this? What are the characteristics of the sentences in FLICK/MSCOC, e.g. very short, caption like?

- Table 9 caption: "we removed overlapping portions from the FLORES devtest set". You report results on Flores devtest (cf. Table 10). You should never modify a test set! This makes it impossible to compare results. Which version of Flores was used in Tab 4, your modified one? Did you recalculate the scores for the other systems?

### Soundness
3

### Presentation
3

### Contribution
3
