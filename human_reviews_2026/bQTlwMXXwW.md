# DVC-SGRL: Adapting MLLMs for Temporally Precise Dense Video Captioning via Semantically Guided Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Dense Video Captioning (DVC) aims to localize and describe multiple events within untrimmed videos. While methods using Multimodal Large Language Models (MLLMs) show promise, their ability to precisely localize event boundaries remains a significant limitation. This weakness stems from a reliance on supervised fine-tuning with cross-entropy loss, which frames timestamp prediction as a classification task. In this formulation, the model learns only to match timestamps exactly, with no awareness of how close a prediction is to the ground truth. This limits its ability to interpret time as a continuous signal, hindering accurate event localization. To address this, we introduce DVC-SGRL, a reinforcement learning framework that provides semantically guided temporal supervision, enabling general-purpose MLLMs to be successfully adapted for dense video captioning. Our approach leverages the model's powerful captioning abilities to improve its weaker temporal localization through a novel matching mechanism and corresponding rewards mechanism. Our semantically-guided reward function uses strong matches in caption content to create robust learning signals for refining event boundaries. This ``soft alignment" approach, which decouples the evaluation of content and timing, offers far more informative supervision than standard classification losses. Experimental results demonstrate that DVC-SGRL achieves significant improvements in both localization and captioning performance, ultimately reaching state-of-the-art results on YouCook2 and ActivityNet Captions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DVC-SGRL, a reinforcement learning framework for dense video captioning that adapts multimodal large language models to achieve temporally precise event localization and fluent caption generation. It introduces semantically guided rewards that align predicted and reference events based on caption similarity, allowing strong linguistic understanding to guide boundary refinement.

### Strengths
1. **Strong experimental results:** DVC-SGRL achieves superior performance on both YouCook2 and ActivityNet, surpassing prior state-of-the-art methods in both localization and captioning quality.
2. **Sound method design:** The approach effectively addresses the temporal insensitivity of cross-entropy loss through GRPO-based reinforcement learning, providing a well-motivated and efficient solution that avoids architectural modifications or additional temporal tokens.

### Weaknesses
1. **Limited novelty:** While temporal sensitivity is indeed an important challenge, prior works [1, 2, 3] have already explored solutions to this issue. The main advancement of DVC-SGRL lies in applying GRPO to dense video captioning, which, although effective, represents an incremental rather than fundamentally novel contribution.
2. **General applicability**: The method is highly tailored toward dense video captioning. Therefore, general capability of the model beyond dense video captinoing might be limited.

### Questions
1. Are there specific reason or ablations on choice of $\alpha, \beta, \gamma, \delta$ for reward calculation? 
2. Does DVC-SGRL also capable of using speech input as Vid2Seq? Or does it already uses speech input?

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
5

### Summary
This paper identifies a fundamental weakness in existing multimodal Large Language Model (mLLM) approaches for Dense Video Captioning (DVC): imprecise temporal localization since mLLM is trained with token classification. To address this, the paper proposes DVC-SGRL, a two-stage training framework. Stage 1 performs standard SFT to align a general-purpose MLLM with the task format, notably representing time as natural language strings (e.g., "01:35 - 01:42") to avoid architectural changes. Stage 2, the core innovation, uses reinforcement learning (specifically, GRPO) with a semantically-guided reward function. This function first matches predicted events to ground-truth events based on caption similarity, not temporal overlap. This "soft alignment" allows the MLLM's strong semantic captioning ability to provide a learning signal for its weaker temporal localization, using a reward that combines caption quality, semantically-matched localization, and a separate localization-only score.

### Strengths
- The paper is well-written and easy to follow.
- To the best of my knowledge, this is the first paper, which adapts GRPO to a dense video captioning task.
- From the author’s experiments, the proposed DVC-SGRL achieves the best performance compared to other baselines on two benchmarks (ActivityNet and YouCook2). The performance gain seems to be meaningful. In particular, from Table 3, the reinforcement learning-based training strategy shows the performance improvement over other training strategies.

### Weaknesses
- More deeper analysis of the semantic matching strategy is required. Compared to existing RL-based mLLMs designed for temporal grounding, the core and original method of this paper is a semantically-guided reward formulation. But, there is a lack of deeper analysis of the semantic matching strategy.
    - First, I wonder why the author applies the Hungarian algorithm to find the optimal assignment. Since the Hungarian algorithm performs one-to-one matching, it may be problematic when the number of predicted captions is different from that of reference captions. It would be better if the paper discussed how to resolve this case.
    - Second, for the Hungarian matching, the paper only uses pairwise caption similarities as a measurement. But, in the DVC task, not only semantic matching but also temporal matching should be considered.
    - Third, I wonder how robust the Hungarian matching algorithm is. It would be better if the author included the analysis concerning the performance of the Hungarian matching algorithm.
    - Forth, I think that employing SODA [1] as a verifiable reward function can play a role in matching rationale. Could you include the experimental results of the model trained with SODA as a verifiable reward function?
- There are some missing baselines such as VTimeLLM, VideoLLaMA2, and VidChain.

[1] Fujita, Soichiro, et al. "Soda: Story oriented dense video captioning evaluation framework." ECCV, 2020.

### Questions
The paper seems to apply the same reward value to all the captions in the sequence generated by multimodal LLMs. But, I think that the better way is to apply different reward values to each caption since the caption’s quality is different.

### Soundness
2

### Presentation
3

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
The paper highlights a limitation of standard cross-entropy training in dense video captioning: timestamps are treated as discrete labels, so the model cannot distinguish between predictions that are slightly off versus significantly incorrect.
To address this, the authors introduce DVC-SGRL, a two-stage training pipeline.
In the first stage, supervised fine-tuning teaches the model the structural format and linguistic patterns required for DVC.
In the second stage, GRPO-based reinforcement learning incorporates the proposed semantically guided matching strategy.

### Strengths
- The authors propose DVC-SGRL, a two-stage training pipeline consisting of supervised fine-tuning followed by reinforcement learning. They show that this ordering (SFT → RL) yields the best performance compared to other training sequences.
- They design a composite reward with four components: caption reward, caption-matched localization reward, traditional IoU-based localization reward, and a format reward.
- They adopt human-readable timestamps to avoid relying on special time tokens during supervised training.
- The method achieves strong results on YouCook2 and ActivityNet, outperforming baseline models.
- The paper includes comprehensive ablation studies that demonstrate the effectiveness of the approach across multiple design choices.

### Weaknesses
- How sensitive is the method to the weighting coefficients in the reward function? 
- How effective is the use of human-readable timestamps? Although the authors list this as a contribution, there is no analysis demonstrating its impact on performance.
- How does the number of training epochs for the SFT and RL stages affect the final performance?

### Questions
- In Table 3, the Recall value is incorrectly bolded.

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
Dense Video Captioning (DVC) requires models to both localize events in untrimmed videos and describe them. While VLMs are promising for this task, they struggle with precise event localization. This stems from their reliance on supervised fine-tuning (SFT) which treats timestamp prediction as a classification task. In this formulation, all incorrect timestamps are treated equally regardless of their proximity to the ground truth, preventing the model from interpreting time as a continuous signal.

The paper introduce DVC-SGRL, a two-stage training framework designed to adapt general-purpose VLMs for precise DVC. The model is taught to express event boundaries using natural language strings, rather than specialized time tokens, maintaining compatibility with pretrained models. Then uses an IoU based reward function to further train it. The model is evaluated on YouCook2 and Acitivtynet.

### Strengths
The proposed approach is different from prior works and the experiments show it is beneficial. However, prior works treating timestamps as text are common, such as Vid2Seq. 

The approach maintains full compatibility with pre-trained MLLMs. It does not require architectural modifications, which is nice.

### Weaknesses
The novelty is a bit limited, as it is mostly just a new reward function for dense captioning tasks. It is further questionable how meaningful the ground truth timestamps are, as historically there has always been disagreement and ambiguity in the timestamps of actions from human annotators. 

The reward function is also a bit concerning. The implementation details reveal that all weighting coefficients (α,β,γ,δ) are set to 1. While simple, it is highly improbable that a uniform weighting is optimal across different datasets with varying densities of events (e.g., cooking steps in YouCook2 vs. sparse activities in ActivityNet). It is also unclear how important each of the components of the reward function are. The ablation in table 2 doesn't have huge differences between the settings, so it isn't clear if they are statistically significant.

The authors admit that their autoregressive design for predicting event boundaries "limits temporal precision on longer videos". Representing time purely as text tokens ("MM:SS") in a single sequence can lead to drifting errors or context window issues in very long untrimmed videos with high event density.

The paper claims to train a "single, unified model" that avoids dataset-specific tuning. However, this "generalist" model is only trained on a combination of two datasets: YouCook2 and a subset of ActivityNet . These are both relatively standard, activity-centric datasets, and the only datasets the paper is evaluated on. A true "generalist" DVC model should be robust to vastly different video domains

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
