# Episodic Memory Representation for Long Video Understanding

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 8

## Abstract
Video Large Language Models (Video-LLMs) excel at general video understanding but struggle with long-form videos due to context-window limits. Consequently, recent approaches focus on keyframe retrieval, condensing lengthy videos into a small set of informative frames. Despite their practicality, these methods simplify the problem to static text-image matching, overlooking spatio-temporal relationships crucial for capturing scene transitions and contextual continuity, and may yield redundant keyframes with limited information, diluting salient cues essential for accurate video question answering. To address these limitations, we introduce Video-EM, a training-free framework inspired by the principles of human episodic memory, designed to facilitate robust and contextually grounded reasoning. Rather than treating keyframes as isolated visual entities, Video-EM explicitly models them as temporally ordered episodic events, capturing both spatial relationships and temporal dynamics necessary for accurately reconstructing the underlying narrative. Furthermore, the framework leverages chain-of-thought (CoT) thinking with LLMs to iteratively identify a minimal yet highly informative subset of episodic memories, enabling efficient and accurate question answering by Video-LLMs. Extensive evaluations on multiple mainstream long-video benchmarks demonstrate the superiority of Video-EM, which achieves highly competitive results while using fewer frames.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Video-EM, a training-free framework for long-form video understanding inspired by human episodic memory. Instead of treating keyframes as isolated tokens to VLLMS, Video-EM groups them into temporally ordered key events, expands events to recover missing context, and builds rich episodic memory representations that capture when, where, what, and which objects. It then uses Chain-of-Thought reasoning to iteratively select a minimal but informative subset of episodic memories before passing them to a VLLM

### Strengths
* Clear motivation.
* Training-free approach that can equip state-of-the-art VLLMs with improved performance.
* The paper appears to be aware of the related work.
* The key event selection is sound and they expand each event to recover query-relevant context that similarity-based approaches may miss. This sounds novel and important as pure semantic retrieval can yield a sparse set of disjoint frames. 
* Video-EM reduces frames while improving accuracy.
* The paper provides ablation studies.

### Weaknesses
* The adaptive event expansion module feels somewhat heavy for a training-free method; simpler alternatives (e.g., adjacent-frame motion thresholds) could be discussed or compared.
* Heavy reliance on object detectors and captioners where failure cases of these modules may propagate.
* A notable limitation is that the method introduces several hyperparameters across multiple stages (e.g., similarity thresholds, expansion limits, CoT confidence and depth, temporal gap $\Delta t$). While the authors provide ablations showing relative robustness, the number of hyperparameters is still large, and tuning them in practice may be non-trivial.
* As a video agent Video-EM requires too many different models which can lead to efficiency problems and lack of end-to-end practicality. 
* The baselines differ from dataset to dataset. While this is ok it is a bit difficult to assess Video-EM's capabilities.
* You should test Video-EM with LLMs other than the Qwen family (such as VideoLLaMA3, InterVL3...).
* Results are not state-of-the-art, however, improvements over backbone models are achieved.


Minor comments:
* Authors use \citet instead of \citep.
* Use Gemini 2.5 pro instead of 1.5 version.

### Questions
* How do you capture object-level semantics $q_0$ and scene-level context $q_s$?
* Why do you need an adaptive event expansion mechanism based on the spatio-temporal difference metric? What does such complex method bring to the table? Couldn't simpler methods yield similar results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Video-EM, a training-free framework designed to improve the performance of Video Large Language Models in understanding long-form videos by overcoming context window limitations and frame redundancy. Video-EM reformulates long-form video question answering by treating isolated keyframes as temporally ordered episodic events, capturing essential spatio-temporal relationships often missed by traditional static sampling methods. The framework involves three core components: Key Event Selection, Episodic Memory Representation (which encodes dynamic scene narratives and relationships), and a Chain-of-Thought reasoning module that iteratively selects a minimal yet highly informative subset of memories. Extensive experiments across four long-video benchmarks demonstrate that Video-EM enhances the accuracy and efficiency of some Video-LLM backbones using fewer frames on average.

### Strengths
- Instead of treating selected frames as disconnected images (a stated limitation of previous keyframe retrieval methods), Video-EM reformulates them as temporally ordered episodic events, avoiding the temporal discontinuities that often disrupt the semantic narrative of events in traditional methods.
- Video-EM leverages a Chain-of-Thought (CoT) thinking strategy to iteratively identify and retrieve a minimal yet informative subset of episodic memories.
- Video-EM is a training-free framework that can be integrated with off-the-shelf Video-LLM backbones without requiring retraining or architectural modification.

### Weaknesses
- Video-EM is a complex, multi-stage pipeline that relies heavily on the quality and coordination of several external, specialized foundation models.
- The concept of episodic memory has been explored by HERMES [1], also a plug-and-play model, with similar claims as Video-EM, yet the differences/similarities between the two are not specified, nor were the results of HERMES discussed in the manuscript.
- Several plug-and-play modules for Video-LLM accuracy/efficiency improvements have been published in recent years such as FastV [2], VisionZip [3], VFlowOpt [4] in addition to the aforementioned HERMES [1]. I am curious about the comparison results with theses other plug-and-play frameworks in terms of accuracy/efficiency tradeoffs, and also in terms of methodology.
- While Video-EM successfully reduces the number of frames processed by the final Video-LLM (from 41 frames down to an average of 9 on EgoSchema, for example), the preceding processing steps require extensive computation across multiple large models (CLIP, DINOv2, RAFT, Grounding-DINO, Tarsier2-7B, Qwen3-8B). I thus believe, the slight accuracy improvement does not justify the upstream cost of putting such a system together.
- I also think such a system is very fragile. A deficiency in the initial retrieval stage (Key Event Selection) or the intermediate processing stages directly impacts the quality of the final input provided to the Video-LLM. It follows that these results would be a headache to replicate.
- The author’s efficiency claims are not substantiated. Fewer frames do not equal more efficient.
- Ambiguous variable definition: In section 3.2, the "Adaptive Event Expansion" paragraph, the authors define alpha as a variable with a value between 0 and 1, yet immediately after that, the paper states that alpha is set to 2. I am quite confused by that. 
- I think Figure 2 has too much text, is quite convoluted, and the bright red color is not easy on the eye. 


[1] Faure, Gueter Josmy, et al. "Hermes: temporal-coherent long-form understanding with episodes and semantics." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

[2] Chen, Liang, et al. "An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[3] Yang, Senqiao, et al. "Visionzip: Longer is better but not necessary in vision language models." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[4] Yang, Sihan, et al. "Vflowopt: A token pruning framework for lmms with visual information flow-guided optimization." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Questions
See weaknesses plus
- The paper highlights the reduction in frames input to the final Video-LLM (e.g., 41 frames down to an average of 9 on EgoSchema). What is the complete end-to-end inference latency (or total computational cost) for the full Video-EM pipeline (including Key Event Selection, Episodic Memory Representation, and Chain-of-Thought steps using all five foundation models and Qwen3-8B)? How does this total cost compare to the baseline model running the maximum allowed frame input?
- The paper acknowledges that the method is "limited by the accuracy of captioners and object detectors". What testing or simulation was performed to quantify how a decrease in accuracy (e.g., failure rate) in a crucial upstream component (such as Grounding-DINO missing key objects or Tarsier2-7B generating an inaccurate Dynamic Scene Narrative) propagates and impacts the final Video-LLM performance?
- Given the strong claims of superiority over prior methods, why were empirical comparisons against other existing plug-and-play, training-free long-video understanding frameworks with similar goals, such as HERMES (which also uses episodes and semantics), omitted? Providing context for these comparisons is crucial for substantiating Video-EM's novelty and competitive edge in the crowded field of V-LLM accelerators.
- In the description of the multi-grained semantic retrieval (L193 onwards), is the summation of equation (1) over the set Q={q1,q2,q3} or Q={q,qo,qs}? In other words, what is qi and why do we have Wq1, Wq2 and Wq3 but no Wq, Wqo and Wqs?

### Soundness
1

### Presentation
2

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
The paper proposes a new pipeline for preparing video features for LLMs. Beyond simple keyframe retrieval, it introduces an agentic flow designed to capture temporally ordered events and reconstruct the underlying narrative. Based on the extracted components, the authors employ a CoT prompting strategy to enhance reasoning and improve understanding. Experiments are conducted across several benchmark datasets.

### Strengths
1. The paper attempts to construct scene graphs to decompose video content, which is an interesting idea.

2. The proposed pipeline is reasonable, and the implementation details are concrete and easy to understand.

3. The experiments are comprehensive, covering most mainstream long-video benchmarks currently available.

### Weaknesses
1. The performance does not reach state-of-the-art results. For example, it is notably inferior to Video-XL-2 [1]. Additionally, some training-free retrieval methods (e.g., BOLT [2]) are missing from the comparison table, which weakens the technical contribution.

2. The main contribution lies in pipeline design rather than technical innovation. The approach feels closer to a text-based agent framework, so the title’s emphasis on “representation” may be misleading—it seems more like an engineering effort.



[1] Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification

[2] BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding

### Questions
The numbers reported in Table 1 (for Qwen2.5-VL) show a large discrepancy compared to the original paper. For instance, LVBench should report 45.3 for Qwen2.5-VL-7B. This inconsistency raises concerns about the results’ reliability. Although the relative improvement over the baseline is significant, the absolute performance values are not aligned with prior reports.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel framework, Video-EM, to improve the performance of video QA tasks of long-form video understanding via generating clip-level descriptions and scene details of key events as episodic memory representations and answering on VLMs with them.
The key components to get episodic memory representations are to select key frames, build events via expanding adjacent frames, generate summaries of the form {when, where, what, which object} on events, and construct scene details of object counts and location relationship. The framework integrates Chain-of-Thought (CoT) reasoning on VLMs with them.
It seems that the effectiveness on the proposed methods are validated experimentally with the state-of-the-art results on 4 long-video understanding benchmarks.

### Strengths
- The paper identifies the bottlenecks in the previous methods for long-form video understanding, which focus on context window limitations and keyframe redundancy. The proposed idea seem  well-motivated and easy to analyze with readable representation for key event as episodic memories.

- Video-EM looks training-free and integrated with other Video-LLM backbones. It shows good modularity and extensibility. 

- The paper provides experimental results on 4 benchmarks (Video-MME, LVBench, HourVideo, Egoschema), consistently outperforming state-of-the-art methods with fewer frames.

### Weaknesses
- It seems that the overall performance of Video-EM highly depends on computer vision modules such as object detection, boundary decision and captioning components. In complex or atypical scenes, misdetections or poor captions can undermine the reliability.

- I think it would be helpful to provide failure cases to consider weakness and robustness for the audience.

### Questions
- While the modularity of Video-EM is emphasized, the dependencies between modules (e.g., how errors propagate from object detection to CoT reasoning) are not deeply analyzed. 
The robustness of the system under suboptimal conditions (e.g., noisy input, failed detection) is not empirically validated, which is crucial for assessing the reliability of the proposed approach. 
How does the framework handle errors in object detection or captioning? Are there any mechanisms within somewhere such as CoT reasoning to mitigate or correct such errors?

- I don’t think this manuscript provides enough information to reproduce all of the results. It can be helpful to open the code to resolve this issue. Code will be publicly available?

### Soundness
3

### Presentation
3

### Contribution
3
