# Think Out Loud, Pause in Silence: Confidence-Guided Reflect–Pause–Abort for Robust  Audio Perceptual Understanding

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 0, 0

## Abstract
Large Audio Language Models (LALMs) mainly fail for two errors: perceptual errors misidentifying background sounds or speaker turns, and reasoning errors drifting rationales that decouple from acoustic evidence. To address these issues, we propose an adaptive framework that couples perceptual grounding with computation that expands only when needed. First, we introduce **PAQA**, a Perceptually grounded Audio QA dataset of 7,470 multiple-choice items that pairs multi-speaker, background-rich audio with stepwise reasoning and reflection annotations, enabling supervision of verifiable audio-grounded rationales. On the modeling side, we propose **ConfAudio**, which unifies explicit, reflective reasoning (fine-tuned on PAQA) with implicit, pause-driven latent computation trained via GRPO. A confidence-aware controller monitors lowest-group-confidence (LGC) during decoding to insert pauses when uncertainty rises and to abort unstable trajectories, thereby reallocating compute toward hard perceptual segments. To stabilize the training process, we design **a composite reward function** that balances answer correctness, reasoning–answer consistency with perceptual robustness, and output format. Across PAQA, MMAU-mini, and MMAR, ConfAudio consistently improves both accuracy and consistency, particularly in noisy, multi-speaker conditions. Our results demonstrate that confidence-guided, adaptive reasoning—grounded in verifiable acoustic evidence—mitigates the dominant perceptual and reasoning failure modes in audio question answering.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes ConfAudio, an adaptive framework that integrates explicit reflective reasoning with implicit pause-driven latent computation for audio-language models. A confidence-aware controller monitors the Lowest Group Confidence (LGC) to trigger PAUSE or ABORT during decoding, reallocating compute to perceptually challenging segments. The authors also introduce PAQA, a 7,470-sample Audio-QA dataset with perceptual grounding and reasoning annotations. Experiments on PAQA, MMAU-mini, and MMAR show consistent gains in accuracy and reasoning consistency.

### Strengths
(1) The combination of explicit reflection and pause-based latent computation is useful and effectively addresses perceptual errors in multi-speaker or noisy audio.

(2) The LGC mechanism provides an interpretable control signal for adaptive computation.

(3) PAQA is a well-motivated dataset focusing on perceptual and reasoning alignment.

(4) The method yields noticeable empirical gains on multiple audio QA benchmarks.

### Weaknesses
(1) The paper repeatedly claims to reduce perceptual errors but does not report WER, CER, or any other speech-level metric. Without these measures, it is unclear whether ConfAudio truly improves perceptual understanding or simply overfits to dataset bias.

(2) The paper mentions that increasing pause tokens “roughly doubles training time,” but provides no inference-time measurements of latency, throughput, or compute–accuracy trade-offs.

(3) The paper never visualizes when and how often PAUSE and ABORT are triggered, nor their effects on hidden-state trajectories or reasoning quality. It is unclear whether latent reasoning genuinely occurs or if pauses merely prolong decoding.

(4) The definition of LGC is incomplete. Parameters such as window size, stride, smoothing, and thresholds are not reported. No sensitivity, calibration, or robustness analysis is conducted to validate its reliability or to measure false triggers.

(5) The method combines reflective fine-tuning and GRPO reinforcement post-training, but omits key details such as learning rate, batch size, gradient clipping, and reward variance reduction. There is no evidence that GRPO converges reliably under confidence gating.

(6) The ABORT mechanism may prematurely terminate valid reasoning chains, as the authors note qualitatively, but no quantitative statistics or examples are given to assess its impact on correctness.

(7) Incomplete baselines. Several strong models are missing from comparison, including Audio Flamingo 3, Baichuan-Omni-1.5, Qwen2.5-Omni, GPT-4o Audio, and Gemini 2.5 Pro. Including these baselines would clarify whether ConfAudio’s gains are competitive at the current frontier.

(8) PAQA is described as containing both “7,470” and “8k” items in different sections. The paper does not specify train/dev/test splits, speaker overlap, augmentation procedure, or MUSAN license terms, which weakens reproducibility.

(9) The composite GRPO reward is described qualitatively as balancing correctness, consistency, and format, but lacks explicit coefficients or normalization. No ablation quantifies the contribution of each term or checks for saturation or instability.

### Questions
(1) Could the authors report WER or other perceptual metrics to substantiate the claim of improved perceptual robustness?

(2) Please provide quantitative measurements of inference latency, throughput, and accuracy trade-offs when varying pause or abort thresholds.

(3) How frequently are PAUSE and ABORT tokens triggered, and how do they affect reasoning quality or hidden-state trajectories?

(4) What are the specific parameters for LGC (window size, stride, smoothing, thresholds), and how sensitive is performance to these values?

(5) Could the authors include GRPO training details (learning rate, batch size, gradient clipping, reward variance control) and provide convergence or stability plots?

(6) How often does the ABORT mechanism terminate correct reasoning, and can the authors include examples of such cases?

(7) Will the authors extend baseline comparisons to include recent large-scale audio reasoning systems such as Audio Flamingo 3, Baichuan-Omni-1.5, Qwen2.5-Omni, GPT-4o Audio, and Gemini 2.5 Pro?

(8) Please clarify PAQA’s final dataset size, splits, and license terms, and describe how speaker overlap and augmentation are handled.

(9) Could the authors specify the exact GRPO reward formula, coefficients, and normalization, and add ablations to show each term’s contribution?

If the authors can address all the issues and questions raised above with thorough analyses, additional experiments, and clearer reporting, I will raise my overall score.

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
This paper tackles perceptual and reasoning errors in Large Audio Language Models (LALMs). It introduces PAQA, a new dataset of 7,470 items with noisy, multi-speaker audio and reflection annotations. It also proposes the ConfAudio framework, which unifies explicit "think out loud" reflective reasoning with implicit "pause in silence" latent computation. A confidence-guided controller adaptively inserts pauses or aborts generation based on uncertainty , improving performance on challenging audio benchmarks.

### Strengths
1. Introduced the PAQA dataset, featuring 7,470 items with multi-speaker, background-rich audio and reflection annotations, which facilitates future research.
2. Proposed the ConfAudio framework, which originally unifies explicit reflection ("Think Out Loud") and implicit, confidence-guided pause-driven computation ("Pause in Silence") to solve key perceptual and reasoning errors.
3. Demonstrated consistent improvements in accuracy and consistency across multiple challenging benchmarks (MMAU Test-mini, MMAR) against strong baselines.

### Weaknesses
1. The paper's overall presentation quality is low, which obstructs understanding. Key illustrations, such as Figure 2 (the framework overview) and Figure 4 (ablation results), suffer from low resolution, rendering text and labels blurry and difficult to read.
2. The paper proposes a sophisticated composite reward function with several novel components, including "BGS robustness" and "Speaker-ASR fidelity". However, it fails to provide any ablation studies that isolate the impact of these specific reward components. It is unclear how much the "Speaker-ASR fidelity" reward.
3. While the PAQA dataset is a primary contribution, the paper omits crucial statistical information. There is no description of the audio data's characteristics, such as the total duration (in minutes), or the distribution (average, min, max) of lengths.

### Questions
see Weakness 2 and Weakness 3.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper tackles a challenge in audio understanding with a focus on perceptual errors, especially when there is enironmental sound or complex speaker context. The paper curates a 7K dataset for this challenge with rich reasoning and reflection annotations. The paper also proposes a pause-and-reflect framework for computation allocation and more accurate reasoning. The paper uses a standard RL method, GRPO, with a custom reward function that considers several aspects of the output quality. Finally, the paper verifies the effectiveness of the proposed method on MMAU and MMAR compared to the base Qwen2-Audio model.

### Strengths
- The construction of the PAQA dataset requires heavy engineering.
- The proposed method with pausing and reflection has not been explored in the audio understanding field. 
- Experiments show gains on MMAU and MMAR.

### Weaknesses
- First of all, the authors are likely not honest in the use of LLMs. For example, section 5.1 is a typical output from LLM -- it makes no sense and contains factual errors. 
- The novelty of the paper is limited. The method is heavily based on the "Think before you speak" paper in ICLR 2024. Besides, the data curation is engineering-focused and less innovative in terms of methodology. 
- While the GRPO part of the paper seems novel (in terms of reward design), I disagree with several designs. First, the BGS robustness reward discourages reasoning based on background sound, but in certain cases the background sound can be useful and we should not add this inductive bias to the model. If background sound is really not desired, one can use a denoising model to pre-process. Second, the length reward is too empirical. The reasoning length should be decided by the model and can be different for very different tasks. All in all, these designs add too much inductive bias to the model and may result in benchmark hacking. 
- For results, the paper only reports poor results on MMAU and MMAR. There are numerous models with higher numbers not reported in Table 2, and many of them outperform the proposed method. 

There are also some minor issues
- The writing of the paper is not clear. The method is not understandable without going back-and-forth to the references. This harms the readability of the paper.
- The mathematical expressions are sometimes not understandable -- e.g. L202-203.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper addresses common failures in audio language models, specifically errors in perceiving sounds and in reasoning logically from audio evidence. The authors propose a system called ConfAudio, which is designed to reason more carefully. They also introduce a new dataset called PAQA to train and test their model on challenging audio with multiple speakers and background noise. The main idea behind ConfAudio is that the model can pause to "think" silently when it's not confident about an answer, and it can also reflect on its initial reasoning to correct mistakes. The authors present experiments showing that their method performs better than existing models on several audio question-answering tasks.

### Strengths
1. The paper correctly identifies that models often struggle with noisy, complex audio and can "hallucinate" answers that don't match the acoustic evidence.
2. The idea of a model that can pause, reflect, and correct itself when it lacks confidence is appealing. The distinction between explicit reflection and implicit "pausing" is an interesting concept.
3. The design of the PAQA dataset, which focuses on multi-speaker and background-rich audio, is well-motivated

### Weaknesses
The paper contains a significant number of citations to non-existent scientific papers. This is a serious breach of academic integrity. It prevents reviewers and readers from verifying the paper's claims, understanding its relationship to prior work, and trusting the authors' research. This issue alone is grounds for rejection.

### Questions
I was curious about the training collapse and recovery shown in Figure 8. It shows an interesting dynamic related to the length reward. While the model stabilized, did this "shock" during training have any lasting impact on the final model's capabilities or stability? Have you considered alternative reward shaping strategies, such as a smoother penalty function, to avoid such "shock"?

### Soundness
1

### Presentation
1

### Contribution
1
