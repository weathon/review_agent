# Zoom-Zero: Reinforced Coarse-to-Fine Video Understanding via Temporal Zoom-in

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Grounded video question answering (GVQA) aims to localize relevant temporal segments in videos and generate accurate answers to a given question; however, large video-language models (LVLMs) exhibit limited temporal awareness. Although existing approaches based on Group Relative Policy Optimization (GRPO) attempt to improve temporal grounding, they still struggle to faithfully ground their answers in the relevant video evidence, leading to temporal mislocalization and hallucinations. In this work, we present **Zoom-Zero**, a coarse-to-fine framework that first localizes query-relevant segments and then temporally zooms into the most salient frames for finer-grained visual verification. Our method addresses the limits of GRPO for the GVQA task with *two key innovations*: **(i)** a zoom-in accuracy reward that validates the fidelity of temporal grounding prediction and facilitates fine-grained visual verification on grounded frames; **(ii)** token-selective credit assignment, which attributes rewards to the tokens responsible for temporal localization or answer generation, mitigating GRPO’s issue in handling multi-faceted reward signals. Our proposed method advances grounded video question answering, improving temporal grounding by 5.2\% on NExT-GQA and 4.6\% on ReXTime, while also enhancing average answer accuracy by 2.4\%. Additionally, the coarse-to-fine zoom-in during inference further benefits long-form video understanding by preserving critical visual details without compromising global context, yielding an average improvement of 6.4\% on long-video benchmarks. Our code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Zoom-Zero, a coarse-to-fine framework designed to improve grounded video question answering (GVQA) by addressing the poor temporal awareness of large video-language models (LVLMs). The method first performs a coarse-grained pass over the entire video to localize query-relevant temporal segments, and then executes a fine-grained pass by "zooming in" on these segments with higher spatial resolution to verify visual evidence. To train this model, the authors enhance the Group Relative Policy Optimization (GRPO) reinforcement learning algorithm with two primary innovations: (1) a zoom-in accuracy reward, which ensures that the localized segments contain the necessary visual information to answer correctly, and (2) a token-selective credit assignment (TokenAdv) mechanism, which decouples multi-faceted rewards and assigns them to the specific tokens responsible for either temporal grounding or answer generation. Extensive experiments demonstrate that Zoom-Zero achieves state-of-the-art performance on several GVQA and long-video understanding benchmarks.

### Strengths
* The concept of a coarse-to-fine temporal zoom-in is highly intuitive, mirroring human visual cognition by first getting a general understanding and then focusing on important details. This provides an effective solution to the inherent trade-off between maintaining long-range temporal context and capturing fine-grained visual details.

* The paper clearly identifies a key weakness of the uniform credit assignment problem. The proposed TokenAdv solution, which selectively attributes credit to different parts of the generated text, is a logical and well-justified improvement. Additionally, the zoom-in accuracy reward is a clever mechanism to enforce a stronger link between the localized visual evidence and the final answer.

* The method demonstrates significant performance gains over strong SFT-based and RL-based baselines across multiple challenging benchmarks, including NEXT-GQA, ReXTime, CG-Bench, VideoMME.

### Weaknesses
* The two-pass nature of the coarse-to-fine framework inherently increases computational cost and latency at inference time compared to single-pass methods. The "Divide-and-Conquer" strategy, while effective, is even more costly, with a reported 2.3x increase in inference time. A more detailed discussion of this efficiency trade-off would be beneficial.

* The framework's performance is heavily reliant on the initial coarse-grained pass to successfully identify the correct temporal segment. If the initial localization is incorrect, the fine-grained pass has no mechanism for recovery and will analyze an irrelevant portion of the video, potentially leading to a confident but incorrect answer.

### Questions
Please see the Weaknesses

### Soundness
3

### Presentation
3

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
This paper proposes an improved GRPO algorithm tailored for grounded video question answering by reflecting a temporal zoom-in strategy and a decoupled credit assignment mechanism for multi-task reward optimization. The experiments on NExT-GQA and ReXTime have demonstrated remarkable performance improvements. The method also shows effectiveness for long-form video question answering. Interestingly, the zoom-in idea also works well on a naïve video partition approach.

### Strengths
1.	The two innovations -- temporal zoom-in and decoupled credit assignment, sound reasonable and are easy to understand.
2.	The presentation is well-structured and easy to read.
3.	The experiment results are good (yet to be justified, see weakness).

### Weaknesses
1.	The coarse-to-fine zoom-in strategy would severely increase the memory and computation burden during GRPO optimization. There is a lack of comparison and analysis on this limitation. 
2.	There are several serious problems in Table 1, making the comparison ineffective.   
-	First, NExT-GQA only provides temporal labels for validation and test sets. It seems that the authors use the validation set for training and compare with those methods for zero-shot testing in Table 1 (all non-RL methods). This should be explicitly specified for valid comparison. 
-	Second, the authors mix up evaluation metrics for QA accuracy (Acc) and Grounded QA Accuracy (Acc@GQA). All compared SFT methods report zero-shot GQA accuracy while Qwen2.5-VL and RL-based methods report QA accuracy without considering grounding. For fair comparison, Acc@GQA is required for all methods; reporting isolated QA and grounding results is less meaningful for grounded VideoQA task.
-	Third, the official Acc of VideoChat-R1 on NExT-GQA is 70.6, but the result given in Table 1 is 69.8. Any explanations for this discrepancy?
3.	The IoG metric is problematic and should be discouraged, as one can obtain perfect IoG value by simply returning the video length as temporal prediction. I recommend to use Intersection over Prediction (IoP) as suggested in NExT-GQA.

### Questions
I will increase my score if the weaknesses are addressed in the rebuttal.

### Soundness
3

### Presentation
4

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
Zoom-Zero proposes a reinforced coarse-to-fine framework for grounded video question answering (GVQA), where the model first localizes query-relevant temporal segments and then “zooms in” for fine-grained visual verification. By introducing a zoom-in accuracy reward and token-selective credit assignment, it enhances evidence faithful temporal grounding and answer accuracy, outperforming prior GRPO-based LVLMs across GVQA and long-video benchmarks.

### Strengths
1. **Strong Results.** The proposed model achieves state-of-the-art performance on grounded video QA and long video understanding benchmarks, clearly demonstrating its effectiveness.
2. **Clarity of Writing.** The paper is well-organized and clearly written, allowing readers to easily follow the technical details and rationale of the proposed approach.
3. **Motivation for Token-Level Advantage Estimation.** The paper insightfully identifies and addresses a key limitation in prior GRPO-based methods of collapsing multiple rewards into a single scalar, offering a well-motivated solution through token-level advantage estimation.

### Weaknesses
1. **Novelty compared to VideoChat-R1.** The method largely resembles those of VideoChat-R1.
    1. How is the temporal grounding reward different from the IoU reward used in VideoChat-R1?
    2. Similarly, how does the zoom accuracy reward differ from the accuracy or recall reward in VideoChat-R1?
2. **Novelty compared to Qwen2.5-VL + frame selection methods.** Since Qwen2.5-VL inherently supports dynamic frame sampling, the zoom-in capability seems intrinsic to the base model rather than a novel contribution of Zoom-Zero. Therefore, frame selection methods applied to Qwen2.5-VL could achieve similar spatial zoom-in behavior as Zoom-Zero.
3. **Comparison with related works.** Frame selection for long video understanding has been extensively studied [1, 2, 3, 4]. A more comprehensive comparison with these works should be provided.
    
    [1] Hu et al, M-LLM Based Video Frame Selection for Efficient Video Understanding, CVPR 2025
    
    [2] Zhang et al., Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs, ICCV 2025
    
    [3] Wu et al., AdaFrame: Adaptive Frame Selection for Fast Video Recognition, CVPR 2019
    
    [4] Tang et al., Adaptive Keyframe Sampling for Long Video Understanding, CVPR 2025
    
4. **Validation of ‘verifiability’.** The authors claim that the zoom-in accuracy reward “verifies that grounded segments contain requisite evidence to answer the query.” However, no concrete mechanism ensures that the grounded segments indeed contain sufficient information.
    1. How does the presence of answer in zoomed region verified?
    2. What are the actual statistics of zoomed regions that contain versus lack the required evidence?

### Questions
Please refer to weaknesses section.

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
4

### Summary
The paper tackles a key challenge in Grounded Video Question Answering (GVQA) — ensuring that large video-language models (LVLMs) not only generate correct answers but also ground their reasoning in the correct temporal segments of a video. Zoom-Zero extends the GRPO-based reinforcement fine-tuning framework with two key innovations, including coarse-to-fine temporal zoom-in and zoom-in reward design. Experiments show consistent but moderate improvements over strong baselines, validating the practical value of hierarchical grounding and refined RL training.

### Strengths
1. The paper addresses an important and well-defined limitation in current GRPO-based LVLM reinforcement fine-tuning for video understanding — namely, the gap between answer correctness and temporal grounding fidelity. The authors articulate this motivation clearly and convincingly, showing that existing methods often fail to verify whether grounded frames truly contain the visual evidence supporting the answer.
2. The paper does not simply introduce a two-stage pipeline; it integrates it into a reinforcement learning framework in a principled way. The introduction of the Zoom-in Accuracy Reward (RZoom) and Token-Selective Credit Assignment (TokenAdv) demonstrates a nuanced understanding of GRPO’s limitations.
3. The experiments are extensive and well-controlled. The authors benchmark across six datasets (NExT-GQA, ReXTime, CG-Bench, VideoMME, MLVU, LVBench), comparing against both SFT-based and RL-based baselines of similar model scales (7B–8B). The inclusion of both short- and long-video tasks ensures that improvements are consistent across temporal lengths. Results are reported with standard metrics (IoU, mIoU, R@K, accuracy) and are presented clearly.

### Weaknesses
1. The novelty of this paper is marginally below the acceptance borderline. While the proposed Zoom-Zero framework introduces a coarse-to-fine “temporal zoom-in” mechanism and two reinforcement learning enhancements (the zoom-in accuracy reward and token-selective credit assignment), these components represent incremental extensions rather than substantial methodological breakthroughs.
2. The overall design is heuristic mainly and system-oriented, building on existing GRPO-based RL fine-tuning frameworks such as VideoChat-R1 and TVG-R1. The “zoom-in” paradigm is conceptually intuitive. It has been explored in prior hierarchical video reasoning works, while the newly introduced rewards and credit assignment strategies refine but do not fundamentally advance the underlying learning algorithm.
3. The paper does not provide a detailed analysis of the computational overhead introduced by the coarse-to-fine zoom-in process. The two-stage inference (coarse localization + fine-grained zoom-in) likely increases latency and resource usage, especially for long videos. This is a practical concern for real-time applications. No comparison of inference time or FLOPs with baseline methods is provided.

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
