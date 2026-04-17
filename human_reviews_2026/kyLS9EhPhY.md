# Threading Keyframe with Narratives: MLLMs as Strong Long Video Comprehenders

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Employing Multimodal Large Language Models (MLLMs) for long video understanding remains a challenging problem due to the dilemma between the substantial number of video frames (i.e., visual tokens) versus the limited context length of language models. Traditional uniform sampling often leads to selection of irrelevant content,  while post-training MLLMs on thousands of frames imposes a substantial computational burden. In this paper, we propose _Narrating KeyFrames Capturing_ (Nar-KFC), a plug-and-play module to facilitate effective and efficient long video perception. Nar-KFC generally involves two collaborative steps. First, we formulate the _keyframe_ selection process as an integer quadratic programming problem, jointly optimizing query-relevance and frame-diversity. To avoid its computational complexity, a customized greedy search strategy is designed as an efficient alternative. Second, to mitigate the temporal discontinuity caused by sparse keyframe sampling, we further introduce interleaved textual _narratives_ generated from non-keyframes using off-the-shelf captioners. These narratives are inserted between keyframes based on their true temporal order, forming a coherent and compact representation. Nar-KFC thus serves as a temporal- and content-aware compression strategy that complements visual and textual modalities. Experimental results on multiple long-video benchmarks demonstrate that Nar-KFC significantly improves the performance of popular MLLMs. Codes are publicly available at https://github.com/bofang98/Nar-KFC.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of using Multimodal Large Language Models (MLLMs) for long video understanding, where the high number of video frames conflicts with the limited context length of language models. The authors propose Nar-KFC, a plug-and-play module that enhances understanding efficiently. Their method involves two steps: first, selecting keyframes by optimizing for query-relevance and frame-diversity using an efficient greedy search; second, inserting textual narratives from non-key frames to ensure temporal coherence. Nar-KFC acts as a content-aware compression technique, and experiments on various benchmarks show it significantly boosts the performance of existing MLLMs.

### Strengths
1. This paper presents a well-written and systematically organized study on a highly practical challenge.
2. The proposed Nar-KFC method offers significant value for real-world deployment as it is a training-free, plug-and-play module that effectively tackles long-video understanding without imposing a substantial computational burden.
3. Nar-KFC's effectiveness is validated through comprehensive experiments on multiple benchmarks, where it demonstrates a notable performance improvement for various MLLMs on tasks like question-answering.

### Weaknesses
1. Limited Task Diversity in Evaluation: The proposed method is evaluated exclusively on question-answering (QA) benchmarks. While frame compression may suffice for QA, its efficacy remains unverified for tasks demanding higher visual granularity, such as video captioning or video OCR. Broader evaluation across diverse tasks is needed to fully ascertain the method's applicability and potential limitations.
2. Outdated Baseline Models: The experiments utilize MLLM baselines from the previous year (e.g., Qwen2-VL, InternVL-2). Given the rapid progress in the field, the plug-and-play value of Nar-KFC should be demonstrated on more recent models (e.g., Qwen2.5-VL, InternVL-3) to ensure its contribution remains relevant and complementary to enhanced base capabilities.

### Questions
I noticed that all the baselines used in the paper are based on 7B or 8B models. How would the proposed method perform on larger or smaller models, and what impact would model size have on the approach?

### Soundness
1

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
This paper introduces Narrating KeyFrames Capturing (Nar-KFC), a plug-and-play module designed to enhance long video understanding in multimodal large language models (MLLMs). Nar-KFC addresses the keyframe selection problem by formulating it as an integer linear programming task that jointly optimizes query relevance and frame diversity. To reduce computational cost, it also offers a greedy search alternative. Additionally, Nar-KFC leverages an off-the-shelf captioning MLLM to generate descriptive captions for the keyframes. A simplified variant, KFC, which omits the captioner, is also evaluated for ablation. Both Nar-KFC and KFC are tested across multiple backbone MLLMs and benchmark datasets, demonstrating their effectiveness and versatility.

### Strengths
* The paper is clearly written and well-structured.
* Nar-KFC demonstrates consistent performance gains across different MLLMs, effectively validating its utility.
* Exploring improved frame sampling strategies for long video understanding is a valuable research direction.

### Weaknesses
* Unlike token compression approaches, Nar-KFC relies primarily on frame selection for information compression. The optimization criteria, i.e. frame-level diversity and query-frame relevance, are computed at the global frame level. This raises concerns about scenarios involving subtle or localized changes (e.g., small objects evolving over time), where high overall frame similarity might cause truly informative frames to be inadvertently discarded, as such nuances may not be captured by coarse frame-level metrics.
* The most significant performance improvements appear to stem from the captioner module, which enriches keyframes with descriptive text. However, this component is computationally expensive, potentially offsetting the efficiency gains expected from reduced frame counts.
* Since the method does not incorporate any token-level reduction mechanism, the number of selected keyframes is inherently constrained by the MLLM’s effective context length and its capacity to handle multiple images, shaped by its pre-training and SFT. This limitation could become problematic for extremely long videos with frequent scene transitions, where even a moderate number of keyframes may exceed effective modeling capacity. A more comprehensive comparison with token-reduction-based methods (rather than only keyframe selection approaches) would better contextualize Nar-KFC’s trade-offs between performance, robustness, and scalability.

### Questions
* Using Qwen2-VL as the MLLM and captioner at the same time leads to performance degradation, I wonder if the authors can provide more analysis on that matter.

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
3

### Summary
This paper introduces Nar-KFC, a novel and practical training-free framework designed to enhance long-video comprehension for existing Multimodal Large Language Models (MLLMs) constrained by limited context windows.

### Strengths
1.This paper presents Nar-KFC, a hybrid representation method that interleaves visual keyframes with textual narratives, offering a novel perspective on video compression and long-video understanding. The authors formulate keyframe selection as an Integer Quadratic Programming (IQP) problem, systematically optimizing both query relevance and frame diversity, making it a more principled alternative to heuristic approaches.

2.The proposed method holds strong practical value: it is a plug-and-play, training-free module that can be directly applied to existing multimodal large language models (MLLMs) to significantly enhance long-video comprehension without retraining, opening a new direction for efficient hybrid video representation.

### Weaknesses
1. A significant potential risk of this method is that the lightweight captioner used for narrative generation may introduce errors or hallucinations. More importantly, this captioner operates in a "query-agnostic" manner; it merely describes the content of non-keyframe segments, and this content may be entirely irrelevant to the user's specific query. This results in the Nar-KFC method actively injecting a substantial volume of irrelevant noise into the MLLM's context . If a given narrative happens to contradict the visual content of the keyframes or the query itself, this could severely mislead the MLLM. 

2. The keyframe selection is entirely dependent on the quality of the upstream VLM (CLIP) used to compute the $S_{QR}$ and $S_{FD}$ scores. This reliance is potentially unreliable, as the model may fail to select rational keyframes for queries involving complex relations or fine-grained details. Furthermore, the MLLM models employed for the experiments (e.g., InternVL2, Qwen2-VL) appear to be outdated.

### Questions
1. The query-agnostic captioner may poses risks of noise and hallucination. Could the authors provide a failure analysis quantifying how often MLLM errors stem from misleading narratives?

2. The keyframe selection relies entirely on CLIP, which may be unreliable for complex or fine-grained queries, and the MLLM baselines used are outdated. Could the authors adopt a more advanced VLM for selection and validate Nar-KFC on a recent MLLM (e.g., Qwen2.5-VL)? It would also be valuable to show performance differences when using newer VLMs, highlighting how the framework scales with stronger vision-language backbones.

### Soundness
3

### Presentation
3

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
This paper introduces Nar-KFC, a plug-and-play module designed to improve long video understanding for Multimodal Large Language Models (MLLMs). Nar-KFC selects representative keyframes by jointly optimizing relevance and diversity through an efficient greedy search, then enriches temporal coherence by inserting textual narratives from non-keyframes. This dual strategy achieves both temporal consistency and content compression, enabling MLLMs to process long videos more effectively. Extensive experiments on multiple benchmarks show that Nar-KFC significantly boosts MLLM performance with minimal computational overhead.

### Strengths
1. Well-written and easy to follow.

2. Keyframe selection is formulated as an integer quadratic programming problem with a customized greedy search, providing clear theoretical grounding rather than relying on heuristic rules.

3. The plug-and-play design is flexible and compatible with various MLLMs, training-free, which reduces computation cost and overfitting risk.

4. Demonstrates consistent performance gains across different models (e.g., InternVL2, Qwen2-VL) and model sizes, showing its generalizability.

### Weaknesses
My main concerns are with the experiments:

1. The number of benchmarks is limited. As a training-free module, Nar-KFC needs more diverse benchmarks to convincingly demonstrate its generality and robustness.

2. The base models are outdated — the strongest one, LLaVA-Video, is already a year old. Including 1–2 more recent VLMs (e.g., Qwen2.5-VL, Qwen3-VL, Intern3-VL) would strengthen the claims.

3. Some baselines in Table 1 seem questionable; for instance, LLaVA-Video’s VideoMME performance should be higher than 55.9 / 56.7.

4. The paper should discuss and compare with more recent keyframe selection methods [1][2] to contextualize the contributions.

[1] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow. ICCV25

[2] From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment. ICCV25

### Questions
Please refer to the Weaknesses section. The authors should seriously address the concerns, especially the experimental aspects, as failing to do so may negatively impact the paper’s evaluation.

### Soundness
3

### Presentation
3

### Contribution
2
