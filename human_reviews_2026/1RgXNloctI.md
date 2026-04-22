# CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Reasoning Video Object Segmentation is a challenging task, aiming at  generating a mask sequence from an input video given a complex and implicit text query. While existing works finetune Multimodal Large Language Models (MLLM) for the task, they still fail in video inputs given complex temporally-sensitive queries, indicating their lack of temporal and spatial integration in complex scenarios. In this paper, we propose **CoT-RVS**, a novel framework employing the zero-shot Chain-of-Thought (CoT) capability of MLLM to address these complex challenges by **temporal-semantic reasoning**: CoT-RVS analyzes the visible objects within a given frame that possibly match the language query (semantic), and chooses a corresponding keyframe for each object that can be observed effortlessly among all frames (temporal). Notably, the CoT-RVS framework is training-free and compatible with closed-source MLLMs, which can be applied to Reasoning Video Instance Segmentation. Our framework's training-free feature further allows its extension to process online video streams, where the CoT  is used at test time to update the object of interest when  a better target starts to emerge and becomes visible. We conduct extensive experiments on video object segmentation with explicit and implicit queries. The results show that CoT-RVS significantly outperforms previous works in both cases, qualitatively and quantitatively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CoT-RVS, a zero-shot, and training-free framework for Reasoning Video Object Segmentation (Reasoning VOS). The primary goal is to segment objects in a video based on complex, implicit, or temporally-sensitive text queries.The core idea is to decouple high-level temporal-semantic reasoning from low-level segmentation and tracking. The method uses a modular, multi-agent pipeline:MLLM Keyframe Selector, Reasoning Image Segmenter, Video Processor.

### Strengths
1. A key aspect of the paper's design is its problem decomposition. Instead of using a single model to handle both temporal reasoning and pixel segmentation, it decouples the task into two stages: 1) First, an MLLM ($F_{key}$) performs high-level temporal-semantic reasoning to identify the most suitable keyframe and target description ; 2) Subsequently, specialized vision models ($F_{seg}$, $F_{vid}$) execute segmentation and tracking based on the clear instructions provided by the MLLM . This architecture shifts the challenge of complex temporal understanding from the pixel domain to the semantic reasoning domain.
2. The paper provides sufficient experiments to validate its method. It was tested not only on four standard benchmarks (MeViS, Refer-DAVIS-17, ReVOS, ReasonVOS) but also on a specially constructed, temporally-sensitive T-ReasonVOS dataset to validate its core hypothesis. The experimental results show that the method achieves performance exceeding the compared SOTA methods on these benchmarks, particularly on T-ReasonVOS 
3. The framework is designed to be training-free and modular. This design allows for the replacement of the MLLM component (e.g., GPT-4o, Gemma3) without retraining. The architecture also supports an extension from single-object (VOS) to multi-instance (VIS) segmentation (via the MLLM outputting a list) and is refactored for an online version (via periodic keyframe updates ).

### Weaknesses
1. the paper's most compelling conclusion—superiority on temporally-sensitive tasks—relies heavily on the newly created T-ReasonVOS dataset . Given that this dataset was manually filtered by the authors and is not yet publicly released, this raises concerns regarding reproducibility and potential selection bias.

2. the ablation studies are incomplete. While they validate the importance of the CoT process and sampling rate, the paper fails to fully explore its claimed modularity by, for instance, swapping the $F_{seg}$ (Seg-Zero) or $F_{vid}$ (SAM2) components to test the framework's generalizability

3. the system is highly dependent on the MLLM's adherence to specific prompt formats.

4. The framework's practical viability is questionable due to high computational latency and cost.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CoT-RVS, a training-free framework for Reasoning Video Object Segmentation. Instead of end-to-end finetuning, the framework is modular. It decomposes the R-VOS task into three distinct stages: an MLLM-based Keyframe Selector that uses zero-shot Chain-of-Thought prompting to perform temporal-semantic reasoning, an off-the-shelf Reasoning Image Segmentation Model to generate a key mask, and a Video Processor (i.e., SAM2) to track the object throughout the video. The paper demonstrates that this framework achieves state-of-the-art performance on several R-VOS benchmarks, with extensions to Reasoning Video Instance Segmentation and online streaming video.

### Strengths
1. The core idea of a training-free, modular framework is compelling. This approach cleverly composes the strengths of large pre-trained models to bypass the need for task-specific finetuning.
2. The quantitative results are strong, showing the framework outperforms prior SOTA methods on multiple benchmarks.
3. The framework is shown to be flexible. The authors demonstrate its applicability beyond standard R-VOS, with extensions for Instance Segmentation and Online Reasoning VOS.

### Weaknesses
1. The primary weakness is that the framework is an integration of existing, powerful components rather than a new technical method. It "stitches together" a large MLLM, an image segmenter, and a video tracker. The system-level design is a valid engineering contribution, but the paper would be stronger if it more clearly articulated this as a novel contribution beyond "stitching".
2. The paper's central claim, embedded in its title, is the power of the Chain-of-Thought reasoning process. However, this claim is supported almost exclusively by a qualitative ablation in Figure 8. The paper lacks a quantitative ablation study regarding CoT.

### Questions
1. What is the key difference between this work and ThinkFast?
2. According to Table A4 in Appendix G, LLaVA-1.5-7B and Qwen2.5-VL-3B are poor at key frames selection. How are the results achieved in the main paper?
3. Why only CoT-RVS-LLaVA supports online mode, while Gemma and GPT-4o do not support?

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
This paper proposed a novel training-free framework called CoT-RVS. This method leverages the zero-shot CoT capability of MLLM to select the optimal keyframes for segmentation. This framework outperforms existing methods both qualitatively and quantitatively.

### Strengths
1.	The proposed CoT-RVS architecture can be seamlessly extended to online video stream processing (Online Reasoning VOS), showing strong practical potential.
2.	The method primarily relies on the zero-shot capability of MLLMs, requiring no additional training and exhibiting good generalization ability.

### Weaknesses
1.	From a pipeline perspective, the proposed method shows limited distinction from prior training-free works such as AL-Ref-SAM. The explicit introduction of Chain-of-Thought reasoning has also been explored in previous reasoning segmentation studies, indicating limited methodological novelty.
2.	Although the introduction emphasizes the importance of reasoning over temporal-semantic correlation, this motivation is not clearly reflected in either the method design or the experiments.
3.	The MLLM keyframe selector and reasoning image segmentation model communicate through text, which may lead to inconsistencies between the described instance and the segmented target when multiple similar objects exist.
4.	The method requires long-chain reasoning on each sampled frame, limiting the length and complexity of the input video, and potentially causing issues such as forgetting or hallucination. This restricts scalability to complex or long-duration videos.
5.	Using GPT-4o as the base model makes the comparison with other methods unfair; parameter amount, inference cost, and latency introduced by the CoT mechanism should be reported.
6.	The performance on Reasoning VOS tasks using LLaVA-1.5 and Gemma-3 is not significant, as shown in Table 4.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a training-free referring video object segmentation by chain-of-thought capability of multimodal LLM in multi-agent framework, termed as CoT-RVS. CoT-RVS contains a multimodal LLM keyframe selector, a reasoning image segmentation model, and a video processor that track the masked selected object instances over the entire video. The CoT-RVS can be applied to offline video settings as well as online video settings. Experimental results show CoT-RVS outperforms prior methods on reasoning VOS benchmarks.

### Strengths
1) This paper proposes a framework to use the chain-of-thought capabilitity of pretrained MLLMs with prompting to perform reasoning first and segmentation then. Experiments show this design outperforms prior approaches and maintain training-free, being versatile with proprietary models.

2) The author also proposes a simple adaptation of this framework to online causal video settings, which is relatively underexplored by previous reasoning VOS methods but with great potential values.

### Weaknesses
There are some major concerns as listed follows:

1) For online video settings, could the authors please report the inference FPS and latency of this pipeline? It seems requiring many components' working to finish segmentation and might take too much time when handling online videos.

2) I was wondering whether this is pipeline agentic enough to be called an agent framework? The steps in this pipeline seem highly fixed and rule-based and lacking being autonomous enough. I would like to hear some author's discussion on this.

3) To what extent the performance gain is credited to the SAM2 as a video processor? As one knows, SAM2 excels at video object segementation given prompts. Is this pipeline still effective (and outperforming prior methods) when using SAM1 as a segmentation model?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
