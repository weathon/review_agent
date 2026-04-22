# DeepOmni: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Native multimodal large language models (MLLMs) restructure a single large language model (LLM) into a spoken language model (SLM) capable of both speech and text generation. Compared to modular and aligned MLLMs, native MLLMs preserve richer paralinguistic features such as emotion and prosody, and generate speech responses directly within the backbone LLM rather than 
using a separate speech decoder. This integration also results in lower response latency and smoother interaction. However, native MLLMs suffer from catastrophic forgetting and performance degradation because the available paired speech-text data is insufficient to support the pretraining of MLLMs compared to the vast amount of text data required to pretrain text LLMs. 
To address this issue, we propose DeepOmni, a framework for adaptive modality expert learning based on a Mixture of Experts (MoE) architecture.
DeepOmni first adaptively distinguishes modality experts according to their modality load within the LLM. Each modality expert then undergoes specialized single-modality training, followed by joint multimodal collaborative training. As a result, DeepOmni incurs only a $5.5\%$ performance drop compared to the original LLM, which is significantly lower than the average performance drop of over $20\%$ typically seen in native MLLMs (such as GLM-4-Voice), and is on par with modular MLLMs. Meanwhile, the end-to-end dialogue latency remains within $0.5$ seconds, ensuring a seamless and intelligent speech interaction experience.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DeepOmni, an MoE-based speech interaction model built on the DeepseekV2-Lite backbone. It follows a parallel modeling paradigm and employs the SNAC codec for speech tokenization. To address the catastrophic forgetting issue in native MLLMs, DeepOmni adaptively identifies modality-specific experts based on modality load, performs specialized single-modality training, and concludes with joint multimodal collaborative training. Experiments on text-to-text tasks demonstrate only a 5.5% performance drop compared to the original LLM.

### Strengths
1. The paper proposes the first MoE-based native speech interaction model, effectively addressing the catastrophic forgetting issue in native MLLMs, which is a critical research topic.
2. The adaptive modality-specific MoE design is innovative and is supported by ablation studies showing its advantages over MoExtend, PureMoE, LoRA, and Random Modality-Specific approaches.
3. The paper is written clearly and includes code in the supplementary material, enhancing reproducibility.

### Weaknesses
1. The paper uses batch parallel decoding, which increases computational costs and creates an unfair comparison with baselines that do not use this technique. Results without batch parallel decoding should be provided.
2. The research is based on a relatively weak backbone, making it unclear how the model would perform with stronger backbones like Qwen3-30B-A3B.
3. There is a lack of comparison with key baselines in Table 2&3 such as Kimi-Audio and Step-Audio2-Mini, which are also native MLLMs. Table 2 suggest that DeepOmni underperforms these models on Spoken QA. 

Minor:
1. GPU model details should be added for latency testing.

### Questions
Please address the issues described in the Weaknesses section. Resolving these concerns could improve the paper’s evaluation.

### Soundness
3

### Presentation
3

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
This paper proposes DeepOmni, a native multimodal large language model for speech interaction that leverages Mixture of Experts (MoE) architecture to mitigate catastrophic forgetting. The key contribution is an adaptive modality expert selection strategy that dynamically assigns experts to audio or text modalities based on their "modality load." The model outperform other native MLLMs like GLM-4-Voice on text-to-text, speech-to-text and speech-to-speech tasks.

### Strengths
1. Addresses Important Problem: Catastrophic forgetting in native multimodal speech models is a genuine and pressing challenge. The paper tackles a real bottleneck that limits the practical deployment of end-to-end speech interaction systems.
2. Novel Adaptive Selection Strategy: The adaptive modality expert partitioning based on modality load is creative and well-motivated. Unlike random assignment, this data-driven approach intelligently identifies which experts are suitable for audio vs. text.
3. Comprehensive Evaluation: The experimental evaluation is thorough, covering multiple dimensions: spoken QA, ASR, TTS, and LLM benchmarks.

### Weaknesses
1. Weak Baselines in Comparison:
The results section appears to compare against relatively weak baselines. Why do Tables 2–5 not include comparisons with Qwen-2.5-OMNI and Kimi-Audio? Notably, Kimi-Audio is itself a non-modular speech LLM, making it an important baseline for fair evaluation.

2. Questionable Claims About Modular SLM Limitations:
The paper’s claims regarding the limitations of Modular Speech Language Models are not fully substantiated. These models remain end-to-end differentiable, for instance, Qwen-OMNI can leverage its generated LLM hidden representations to encode paralinguistic cues. Did the authors perform any experiments showing that modular architectures are indeed worse at modeling such paralinguistic information?


3. Lack of Analysis:
While the method demonstrates improved performance, there is limited insight into why it mitigates catastrophic forgetting more effectively than other methods. What distinct knowledge patterns do the audio and text experts capture? How does modality isolation help preserve capabilities? A deeper analysis—e.g., via probing or visualization—would greatly strengthen the paper.


4. Clarity and Presentation Issues: The paper is difficult to follow in several sections. The term e(h_l)_i should be explicitly defined in Eq. (2). Algorithm 1, which seems central to the contribution, is hard to interpret and should be explained in greater detail. The Expert Selection Statistics remain unclear. The multiplicative selection criterion (lines 22–23) seems arbitrary—why this specific formulation and not other combinations? An ablation or stronger motivation is needed. Formatting in Sections 3.3 and 3.4, as well as reference styling (perhaps using \citep{}), should also be improved.


5. Section 3.3 (Audio–Text Alignment):
The paper mentions “downsampling the audio and using text padding tokens to align their lengths.” More details should be provided—specifically, how much padding is applied and whether it affects training stability or convergence.

### Questions
Please check weaknesses, particularly 1,2 and 4

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents DeepOmni, a multimodal spoken language model that integrates adaptive modality-specific experts within a Mixture-of-Experts (MoE) architecture. The goal is to alleviate catastrophic forgetting in native multimodal large language models (MLLMs). DeepOmni introduces an adaptive modality expert selection strategy based on modality token load and employs a three-stage training procedure of modality alignment, unimodal training, and cross-modal joint training. Experiments on spoken QA, ASR, TTS, and text benchmarks demonstrate that DeepOmni reduces language performance degradation to 5.5%, substantially lower than existing native MLLMs (typically over 20%), while maintaining real-time response latency (<0.5 s). Overall, the paper contributes a novel and well-engineered MoE-based framework for building end-to-end speech interaction models that effectively balance linguistic competence and acoustic generation.

### Strengths
1. The work claims to be the first native MLLM built upon an MoE-based LLM backbone with a 3-stage post-training and addresses the catastrophic forgetting in native MLLM. Solid and highly effective.

2. It proposes an effective and intuitive expert partition strategy that selects modality-specific experts based on modality load, and the proposed model achieves a low performance drop in language capacity.

### Weaknesses
1. The paper claims native MLLMs preserve richer paralinguistic features as part of its motivation, but the evaluation lacks essential quality-based metrics to substantiate this claim and compare the expressive quality of the proposed model against other native baselines.

### Questions
1. See weakness 1. Can we see results comparing DeepOmni's speech output against other native MLLMs on quality metrics like prosody and emotional expression?


2. The process for designating the 2 shared modality experts is missing from the adaptive partitioning mechanism (Algorithm 1). Can the authors clarify this step?

3. The Phase 3 pseudo-code in Algorithm 1 puts $\text{top-}k$ inside a loop iterating over $j$. This looks confusing, as the intent seems to be applying $\text{top-}k$ globally, like \text{Audio Experts}_{l} \leftarrow \text{top-}k\!\left(
    \left\{ \rho_{l,j}^{A} \cdot (1 - \rho_{l,j}^{T}) \right\}_{j=1}^{M},\, k
\right)

4. Some formatting issues for inline citations. Some should be using citep

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
5

### Summary
To mitigate catastrophic forgetting in native MLLMs, this paper proposes DeepOmni for adaptive modality expert learning in a MoE-based MLLM. DeepOmni goes through stages of adaptive modality expert selection based on modality load, specialized single-modality training with instruction data from different modalities, and then joint multimodal collaborative training using cross-modal instruction data. Experimental results show that DeepOmni achieves a 5.5% relative performance drop over the original LLM, substantially lower than some MLLMs such as GLM-4-voice. The E2E dialogue latency remains 0.5 secs for efficient voice interaction.

### Strengths
1.	Using a MoE architecture for developing MLLMs has been explored in earlier works, as well as dynamic modality expert selection such as in prior works of LLMoE etc. The single-modality expert training and then cross-modal expert training has also been explored in the prior Uni-MoE framework. The main contribution of this work seems to investigate the impact of these previously proposed approaches on mitigating catastrophic forgetting of text capabilities in LALMs and omni models, which is an important research question. The experimental results show that DeepOmni achieves a 5.5% relative performance drop over the original LLM, which is better than the 6.5% relative drop from Qwen2.5-omni (a dense model) over its backbone LLM.

2. The analysis of the number of audio experts, modality load of experts at different layers are useful.  The comparison between PureMoE, LoRA-PureMoE,  Random Modality-specific MoE, Adaptive Modality-specific MoE shows clear advantages of adaptive modality-specific MoE over the less principled approaches.

### Weaknesses
1.	Some important related, non-contemporaneous works are missing in theoretical and empirical comparisons, for example, strong MLLMs such as Kimi-audio, Ming-lite-omni (which is also a MoE-based omni model). Hence, the presentation of the experimental results is misleading. For example, in Table 2 performance on Spoken QA,  Kimi-audio and Qwen2.5-omni achieved much better performance than the proposed DeepOmni, yet their evaluation results are missing. In Table 3 evaluating the T2T performance and the relative drop of the LALMs/omni models, Kimi-audio and other dense or MoE-based MLLMs are missing, such as Uni-MoE, Ming-lite-omni. 

2. DeepOmni is built upon a weak backbone, DeepSeek-V2-Lite, which is further verified by its poor performance on text capabilities as shown in Table 3. As a 15.7B-A2.4B MoE model, its average score is 53.06, much worse than qwen2-7B's 70.52, qwen2.5-7B's 73.62, and GLM-4-9B's 64.08, with all these being dense models. With a low performing backbone, it is difficult to fully justify the effectiveness of the proposed  dynamic modality expert selection, uni-modality expert training and cross-modal expert training. It is important to evaluate these proposed approaches on a more competitive MoE backbone.

3. The batch parallel decoding used in DeepOmni, as also used in mini-omni and other works,  expands a single audio input into a batch size of two, with one audio+text sample, and one text-only sample, and embed the text-only output into the audio generation process. This is more a hybrid workaround rather than a principled solution for the speech and text interference in parallel speech-text modeling.

4. For S2S, the speech generation performance needs to be evaluated, for example, reporting WER and UTMOS.

5.	The ablation study in Appendix focuses on investigating the number of acoustical experts and analysis of modality load across different layers showing the benefit of dynamic modality expert selection, but the analysis of the multi-stage training is not presented.

### Questions
1.	There are some formatting issues. For example, the citations could be added using \citep, so that it would appear as, for example, (Radford et al., 2023). The current citation formatting right after text, e.g., (ASR) Radford et al. (2023), degrades readability.

### Soundness
3

### Presentation
2

### Contribution
2
