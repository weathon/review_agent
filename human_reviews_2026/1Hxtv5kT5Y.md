# FastALM: Hierarchical Frame Q-Former for Effective Audio Modality Adaptation

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Recent advances in large language models (LLMs) have demonstrated human-expert-level capabilities, driving significant interest in their potential for achieving artificial general intelligence (AGI). In particular, there is growing momentum in adapting LLMs to various modalities—including vision, video, and speech—through the development of multimodal LLMs (MLLMs). However, existing speech-language models (SLMs) research has largely overlooked cost-effective adaptation strategies for leveraging LLMs in the speech domain. In this paper, we propose FastSLM, a lightweight yet efficient SLM designed for effective understanding and reasoning over long-form speech. To address the challenge of aligning high-frame speech features with LLM, we introduce the hierarchical frame querying transformer (HFQ-Former), which compresses frame-level speech features while capturing both local and global context. Furthermore, we present a novel three-stage training strategy that enhances generalization across a wide range of speech-related tasks. Experimental results demonstrate that FastSLM achieves competitive performance compared to existing state-of-the-art (SOTA) models, despite operating with significantly lower FLOPs and parameter counts, while representing speech with only 1.67 tokens per second. The source code and model checkpoints are available at https://anonymous.4open.science/r/FastALM-1D6B.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes FastALM, a speech-language model for efficient understanding of long-form spoken input. The method introduces a HFQ-Former that compresses high-rate acoustic frames into a compact set of queries for an LLM, and adopts a three-stage training strategy to adapt a pretrained LLM to the speech modality. The system is trained on a moderate-sized bilingual corpus in Korean and English and is evaluated on ASR, AST, spoken summarization, and spoken QA. The authors report competitive accuracy and faster end-to-end processing compared to speech-focused baselines under long audio inputs.

### Strengths
1. Combines a hierarchical frame-to-query adapter with a simple three-stage adaptation schedule to reduce the acoustic-to-LLM interface cost for long speech.
2. Evaluates across several speech tasks. The performance is competitive with high computational efficiency.

### Weaknesses
The paper's biggest weakness is its overstatement of its scope.

The paper repeatedly refers to the model as an Audio-Language Model, yet all tasks and data are speech only. In current community usage, ALM typically denotes models that cover general auditory content that includes speech, environmental sounds, and music, not speech alone [1, 2, 3]. A more accurate term here is Speech-Language Model [4]. This affects positioning, related work, experiments, and claims. For example, statements like “address the challenge of aligning high-frame audio features with LLM” would be clearer as “aligning high-frame speech features with LLM.” Likewise, claims about enabling fast text generation from long audio should be stated as long speech. As a result, in experiments, it would be more appropriate to compare with more speech-focused models.

Other weaknesses:

Positioning and baselines:
Related work sections labeled only “Audio-Language Models” and “Q-formers for speech modality adaptation” should separate general-audio ALMs from speech-only SLMs, then justify how the proposed method advances the speech setting specifically. Experimental comparisons should include strong speech-centric baselines and report both accuracy and efficiency under matched conditions.

Missing implementation detail:
The paper does not specify the LLM backbone used for LoRA adaptation. Since LoRA hyperparameters and achievable throughput depend on the base model, the manuscript should name the backbone and report key config details, including LoRA ranks, target modules, context lengths, tokenizer, and precision.

[1] Tang, C., Yu, W., Sun, G., Chen, X., Tan, T., Li, W., ... & Zhang, C. (2023). Salmonn: Towards generic hearing abilities for large language models. arXiv preprint arXiv:2310.13289.
[2] Kong, Z., Goel, A., Badlani, R., Ping, W., Valle, R., & Catanzaro, B. (2024). Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities. arXiv preprint arXiv:2402.01831.
[3] Chu, Y., Xu, J., Zhou, X., Yang, Q., Zhang, S., Yan, Z., ... & Zhou, J. (2023). Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919.
[4] Cui, W., Yu, D., Jiao, X., Meng, Z., Zhang, G., Wang, Q., ... & King, I. (2024). Recent advances in speech language models: A survey. arXiv preprint arXiv:2410.03751.

### Questions
See above.

### Soundness
2

### Presentation
2

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
The paper proposes FastALM, a lightweight audio–language model that integrates speech into large language models (LLMs) through a Hierarchical Frame Querying Transformer (HFQ-Former) and a three-stage training strategy. The HFQ-Former hierarchically compresses frame-level audio features from pre-trained encoders (e.g., Whisper) into compact representations (≈1.67 tokens/sec) for efficient LLM adaptation. The training pipeline includes short-form pre-training, long-form adaptation using ASR data, and instruction tuning on multi-task datasets (ASR, AST, SSUM, SQQA). Experiments show competitive performance compared with state-of-the-art ALMs, with notably lower FLOPs and token rates.

While the paper is well-written and the system is practically effective, its novelty and conceptual contribution are limited. The proposed HFQ-Former mainly combines existing Q-Former designs with hierarchical down-sampling, and the training pipeline closely follows prior ALM works such as SALMONN, Audio-Flamingo, and Qwen2-Audio. The work thus reads more as an engineering optimization rather than a substantive methodological innovation.

### Strengths
1. The paper is clearly written and systematically organized. The overall pipeline—from audio encoding to multimodal LLM alignment—is technically sound and reproducible.

2. The hierarchical compression indeed reduces token rates and computational costs.

### Weaknesses
1. The HFQ-Former is conceptually incremental—essentially a hierarchical stacking of existing Q-Former modules with down-sampling. No new mechanism or theoretical insight is introduced beyond standard attention-based compression.

2. The performance improvement over previous Q-Formers is minor (e.g., WER reduced by only ~0.05 on LibriSpeech).

3. The paper lacks ablation or visualization explaining why the hierarchical compression helps. There is no semantic or attention-level analysis of the compressed tokens, leaving the underlying mechanism unclear.

### Questions
1. why donot maintain the same token rate to compare the performance between SQ-Former, WQ-Former?

2. Why donot present the baseline with directly down-sampling? Such as the style of Qwen2-Audio. From my veiw, Q-former struture is not  the mainstream style for MLLM in vision domain. Many audio-LLMs also donot use the Q-former structure, beacause the Q-former also introduce the additional parameters, incleasing the training and inference cost.

### Soundness
3

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
This paper proposes FastALM, a lightweight and efficient Audio-Language Model (ALM) designed to address the computational inefficiency of processing long-form audio in existing ALMs. 

The core innovation is the Hierarchical Frame Querying Transformer (HFQ-Former). It compresses high-frame audio features into a compact representation while preserving local and global context, reducing LLM FLOPs and memory usage. FastALM also adopts a three-stage training strategy: (1) short-form ASR pre-training to align speech and text; (2) long-form audio adaptation to enhance long-audio processing capability; (3) instruction tuning for downstream tasks.

Key contributions:
1. A low-cost FastALM that enables fast text generation from long-form audio with minimal computational cost.
2. A three-stage training pipeline that adapts pre-trained LLMs to speech modalities without costly end-to-end training.

### Strengths
Originality: it proposes HFQ-Former, a novel multi-stage hierarchical compression module that differs from prior Q-Formers by integrating downsampling to capture both local and global context, achieving a low-frame-rate representation.

Clarity: The paper is well-structured and clear, with a complete logical flow and appendices that include key details, all of which enhance readability and reproducibility.

Significance: The experiments strongly validated the significance of this work, which achieves higher or comparable performance on different audio tasks over SOTA approaches while reducing computational cost. It presents improvement to ALMs, especially for long-form audio.

### Weaknesses
1. The novelty is limited.
* The HFQ-Former is an incremental improvement of Q-Former, which leverages multi-scale modeling into the model. But "multi-scale model can improve computing efficiency" has been validated in other works, e.g. U-Net. The author should give a deeper insight into this idea to strengthen the novelty.
* Similarly, multi-stage training has been a widely used strategy to improve training effectiveness and efficiency.

2. The experiments are not convincing.
*  Insufficient ablation studies for HFQ-Former.
  * No analysis for the proposed model design. It makes me confused about why the module is designed in such a complicated form.
  * No investigation on the impact of "hierarchical" (e.g. number of downsampling layers, downsampling scale, etc.)
* Unvalidated fairness of the system comparison
  * No details of baselines (model configuration, dataset, training details)

### Questions
As discussed in "Weakness", I have the following concerns:

1. As you said, "This design enables the model to effectively capture both local and global temporal information from audio features". Can you give a deeper analysis of this claim? What does local and global temporal information include? Can a larger single-stage q-former extract local global temporal information effectively?
2. Do you have ablation studies on the impact of "number of stages" and "downsampling factors"?
3. Do S-former and W-former share the same dataset and training configuration as HFQ-former? Do they have the same embedding dimension?

### Soundness
3

### Presentation
3

### Contribution
3
