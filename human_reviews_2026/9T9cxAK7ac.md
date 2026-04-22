# VCoT: Visual Chain-of-Thought for Continual Learning in Day-Night Object Tracking

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Stable tracking in both daytime and nighttime is essential for applying single object tracking to real-world scenarios. Traditional daytime trackers mainly rely on clear appearance features, which leads to significant performance degradation under nighttime conditions. Conversely, nighttime trackers often incorporate low-light enhancement techniques to improve robustness but struggle to maintain comparable accuracy in daytime environments. To address this challenge, we propose a novel framework, termed Visual Chain-of-Thought (VCoT), which reformulates object tracking as a structured reasoning process. VCoT follows a three-stage cognitive path of Observe–Recall-Infer–Memorize: it first observes and extracts the appearance and motion features of the current frame; then retrieves and fuses relevant historical prompts from a memory pool via an attention mechanism to enable context-aware reasoning; and finally employs gradient-based importance evaluation to update the memory by selectively retaining the most valuable knowledge. This design allows the model to integrate real-time observations with historical experiences, while achieving continual learning and effective knowledge transfer across tasks. Extensive experiments on multiple challenging benchmarks demonstrate that VCoT consistently outperforms existing methods under diverse illumination conditions. Codes will be available at https://github.com/Gkk10/VCoT.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a novel object tracking framework termed the Visual Chain-of-Thought (VCoT), which is designed to address the challenge conventional trackers encounter in simultaneously adapting to the two disparate illumination conditions of daytime and nighttime.

The core concept of this framework is to emulate the human cognitive process, structuring it as a three-stage "Observe-Recall-Infer-Memorize" cycle:

Observation: Extracting the appearance and motion features of the current frame.

Recall-Inference: Fusing the current observation with historical experiences stored in a "memory pool" to perform inference.

Memorization: Employing a continual learning mechanism to evaluate the importance of new experiences (based on gradients) and selectively storing the most valuable information into the "memory pool".

### Strengths
The application scenarios and the inherent difficulties of the task are accurately demonstrated through illustrative examples.

An analysis of various challenging factors is conducted; furthermore, the qualitative comparison analyzes the distinct characteristics of VCoT and other trackers.

Interpreting the tracking problem from the perspective of Chain-of-Thought (CoT) presents a novel and insightful approach.

### Weaknesses
Lack of Novelty: While the paper interprets the utilization of temporal information from the perspective of Chain-of-Thought (CoT), it fundamentally amounts to conventional temporal information processing. Furthermore, the contributions exhibit significant overlap with existing works such as ARTrack, ARTrackv2, and MixViT, appearing to be merely a combination of established methods with minor modifications.

Restricted Application Scenarios: The method's applicability is limited; ultimately, it solely addresses the challenge of illumination variations.

Incomplete SOTA (State-of-the-Art) Comparison: The paper does not include comparisons against the latest trackers (MCITrack, etc). Additionally, evaluations on more general-purpose datasets, such as LaSOT, are absent.

ARTrack: Wei, Xing, et al. "Autoregressive visual tracking." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
ARTrackV2: Bai, Yifan, et al. "Artrackv2: Prompting autoregressive tracker where to look and how to describe." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.
MixViT: Cui, Yutao, et al. "Mixformer: End-to-end tracking with iterative mixed attention." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
MCITrack: Kang, Ben, et al. "Exploring enhanced contextual information for video-level object tracking." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 4. 2025.

### Questions
Does the proposed methodology incorporate any unique design elements specifically targeting the challenge of day-night transitions?

Furthermore, is its applicability restricted to this context, or can the approach be generalized to other challenges or scenarios?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes VCoT (Visual Chain-of-Thought) for single-object tracking that must work across day/night. The core idea is to reformulate tracking as a reasoning loop (Observe, Recall–Infer, Memorize), where the tracker (i) builds appearance and residual-motion “observation prompts” from the current frame, (ii) retrieves and fuses relevant historical prompts from a memory pool via attention, and (iii) updates that memory using a gradient-based importance score so only useful prompts are retained. Prompts are prepended to backbone tokens and used to guide the transformer. Trained with day/night as sequential tasks (continual learning framing), VCoT reports SOTA or competitive results on both night (UAVDark135, NAT2024-1, DarkTrack2021, LLOT) and day (DTB70, VisDrone2018, UAVDT, OTB100) benchmarks, plus large-scale GOT-10k and TrackingNet; ablations attribute gains to each stage and to the memory pool mitigating forgetting.

### Strengths
(1) A prompt-centric tracking framework that explicitly retrieves from historical context rather than relying only on the current frame, with a clear mathematical derivation of prompt creation and injection.

(2) A gradient-guided memory writing rule (top-M by averaged per-token gradient norms with K-step smoothing) that is simple and effective.

(3) A promising framing of day to night training and evidence that naive joint training causes forgetting

(4) Good experimental results

### Weaknesses
(1) The paper says prompts are formed from linear-projected appearance and residual-motion features, encoded via a transformer, then projected to a fixed-length tensor and concatenated to tokens. But values of L/D, which layers get prompts, and how many prompts per frame / per level are not specified very well. Similarly, the memory capacity and smoothing/window are not well-stated in the main text. This limits reproducibility and makes Figures 1–2 feel high-level rather than operational. 

(2) Although framed as Chain-of-Thought, the method is attention over a prompt memory plus gradient-based selection, which is kind of close to current prompt/memory trackers. The paper claims to be “first to introduce CoT in tracking,” but the novelty seems incremental relative to prompt-based retrieval + memory. What is the new reasoning capability here beyond (i) motion-residual prompts and (ii) memory-augmented attention? 

(3) The paper argues that recalling history & using motion prompts compensates for appearance loss at night, but it seems that there is no targeted analysis that directly shows illumination-invariance gains (e.g., day to night within the same scene, or controlled photometric corruptions). The “continual learning” ablation uses LaSOT (day), not a day to night paired scenario; it shows forgetting mitigation, not illumination robustness. 

(4) Baselines are “retrained on the same datasets,” but several are not designed for night or for CL; it’s unclear whether night-specialized or prompt-memory baselines are tuned equivalently

Other minor weaknesses:
(5) The abstract calls Observe–Recall–Infer–Memorize a “three-stage” path, but lists four verbs. Though the authors mentioned that Recall-Infer is a single stage, this is quite confusing to readers. 

(6) Tables label the method “UniTrack (Ours)” rather than VCoT, which is kind of confusing

### Questions
(1) What are L and D exactly (e.g., what do they look like)? Which backbone stages receive prompts? How many prompts per frame and how are positions encoded? 

(2) What are the shapes of M (capacity) and K (smoothing window) exactly? What is the write policy/frequency? Is there class/scene stratification?

(3) Is memory updated online at test? If so, how is the importance approximated without gradients? If not, maybe scope “continual learning” to training-time only.

(4) It is interesting to see experiments with controlled day to night transformations (brightness/contrast/noise) on the same sequences, reporting with/without memory and motion prompts

(5) It is also interesting to see FPS/FLOPs/params comparison.

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
This paper proposes a novel continuous learning framework for visual object tracking, VCoT, which achieves stable object tracking under both day and night lighting conditions.

The core idea is to introduce a human-like cognitive reasoning pipeline—Observe, Recall, Infer, and Memorize—to achieve structured, human-like adaptability. Experimental results on multiple benchmark datasets for day and night scenes demonstrate that VCoT maintains stable and superior performance under different lighting conditions, outperforming existing methods, including several strong baseline models.

### Strengths
1. The paper is good written

2. The concept in the field of tracking is very new.  Introducing a “Visual Chain-of-Thought” into object tracking sounds novel. 

3. The paper achieved sota performance.

### Weaknesses
1.The comparison methods is insufficient. There are losts of sota Method need to be compared in 2025. There are also sota methods need to be compared, such as ARTrackV2, LoraT.

2.There are no efficiency comparisons and analysis (FPS, FLOPs), which are important for visual object tracking.

### Questions
Can authors provide more sota comparison and comlexity analysis?

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
The authors tackle the weaknesses of conventional visual tracking algorithms, where they solely rely on the appearance feature representations. This paper proposes to employ visual chain-of-thought (VCoT) for visual tracking formulation, where the authors formulate this as observe-recall-infer-memorize steps. The authors claim that they are the first to employ CoT formulation for visual tracking task, and the experimental results show strong results on multiple benchmark datasets, especially for tracking under low-light night conditions.

### Strengths
- The authors propose an interesting approach for solving the visual tracking task, where they redefine the visual tracking process apart from conventional tracking algorithms.

- The objective of the proposed method is explained in a straightforward manner, with simple formulation which can be easily implemented and reproduced by other researchers.

- The authors performed extensive experimental validations and comparisons on multiple well-known visual tracking datasets and algorithms, and they also conducted some ablation experiments to facilitate further understanding for the proposed work.

### Weaknesses
- The chain-of-thought (CoT) concept used in this paper seems to be largely deviated from the original concept used in the natural language processing (NLP) field. Generally, CoT is thought to be a dynamic reasoning process that is variable in length, and its steps include sequential causal chains that are interpretable and flexible enough for reaching diverse conclusions at the end. However, the so-called CoT in this paper seems to be a fixed pipeline, with concrete heuristic steps of observe-recall-infer-memorize, with no room for logical reasoning and sequential process.

- The proposed method explicitly uses nighttime datasets (BDD100K-Night, SHIFT-Night) in a continual learning setup, whereas most comparison baselines are trained only on daytime datasets. Fair comparison with equivalent training datasets and backbone network should be conducted.

- Is the "continual learning" formulation necessary? The proposed gradient-based memory retention essentially acts as a simple replay buffer rather than a principled continual learning algorithm, and no comparison to well-known continual learning baselines such as EwC and LwF are not conducted.

### Questions
Please refer to the weaknesses section. Although the authors' take on the visual tracking task is appealing, the machine learning concepts used in the proposed method seem to be overstated and deviated from its original form.

### Soundness
2

### Presentation
2

### Contribution
2
