# LOVE-R1: Advancing Long Video Understanding with Adaptive Zoom-in Mechanism via Multi-Step Reasoning

- Decision: Reject
- Scores: 6, 6, 2, 6

## Abstract
Long video understanding is still challenging for recent Large Video-Language Models (LVLMs) due to the conflict between long-form temporal understanding and detailed spatial perception. LVLMs with a uniform frame sampling mechanism, which samples frames with an equal frame size and fixed sampling rate, inevitably sacrifice either temporal clues or spatial details, resulting in suboptimal solutions. To mitigate this dilemma, we propose LOVE-R1, a model that can adaptively zoom in on a video clip. The model is first provided with densely sampled frames but in a small resolution. If some spatial details are needed, the model can zoom in on a clip of interest with a large frame resolution based on its reasoning until key visual information is obtained. The whole process is implemented as a multi-step reasoning process. To train the reasoning ability, we first finetune the model on our collected 38k high-quality CoT data and enhance it with decoupled reinforcement finetuning. As outcome rewards can not provide fine-grained process supervision, we decouple multi-step reasoning into multiple single-step reasoning and optimize the internal zoom-in ability explicitly. Experiments on long video understanding benchmarks show that our model with the slow-fast adaptive frame sampling mechanism achieves a great trade-off between sampling density and frame resolutions, and LOVE-R1 outperforms our baseline Qwen2.5-VL by an average of 3.1\% points across 4 common long video understanding benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes LOVE-R1, a large video-language model designed for efficient and accurate long video understanding. The approach combines dense, low-resolution frame sampling for global temporal context and selectively zooms in on high-resolution segments when finer spatial details are necessary, as determined by the model’s reasoning process. The method is trained with a three-stage post-training paradigm. Experiments on several long video benchmarks show consistent improvements over strong baselines such as Qwen2.5-VL, achieving state-of-the-art performance with a 3.1% average gain. Overall, the paper contributes a novel adaptive sampling framework and a reasoning-based training pipeline that effectively balance temporal coverage and spatial detail for long video understanding.

### Strengths
**1. Originality:**
The paper addresses a fundamental limitation of current video-language models, the trade-off between temporal coverage and spatial fidelity under constrained context length. By introducing an adaptive, query-driven zoom-in mechanism, it reframes long video understanding as a progressive multi-step reasoning process (decision, zoom-in, answer). This formulation is both intuitive and impactful, demonstrating clear conceptual originality. 

**2. Technical Soundness:**
The proposed three-stage post-training framework, comprising slow–fast template finetuning, chain-of-thought (CoT) initialization, and decoupled reinforcement optimization, is logically structured and empirically validated. Each stage contributes a distinct function.

**3. Empirical Validation:**
The experimental evaluation is extensive and convincing. The proposed LOVE-R1 model consistently outperforms strong baseline across multiple long video understanding benchmarks (Video-MME, LongVideoBench, LVBench, MLVU). The ablation studies (Tables 2–3) clearly attribute gains to specific model components, confirming the utility of adaptive sampling and multi-step reasoning.

### Weaknesses
**1. Marginal Gain from Decoupled RL:**
A core innovation presented for the training process is the "decoupled reinforcement finetuning" (Stage 3). The ablation study in Table 2b shows that this decoupled optimization (labeled "Single-Step Optimization") achieves an overall score of 66.2%, only a marginal 0.5% improvement over the standard multi-step GRPO baseline ("Multi-Step Optimization"), which scored 65.7%. This minimal gain does not strongly justify the added complexity of the decoupled approach, and the paper does not clearly demonstrate the specific conditions or failure cases under which this method becomes essential.

**2. Missing Error Analysis and Efficiency Metrics:**
The paper lacks a crucial discussion of the model's specific failure modes. While ablation studies compare against baselines like "random zoom-in", there is no qualitative or quantitative analysis of when the LOVE-R1 model itself fails. For instance, the analysis does not explore the frequency or nature of incorrect zoom-in localizations, errors in the decision-making step (i.e., deciding to zoom vs. to answer), or misinterpretations of the detailed "slow video" frames.

**3. Unknown Computation Cost:**
Furthermore, the paper omits any reporting on computational cost. A primary motivation for an adaptive "slow-fast" mechanism is to balance performance with efficiency. However, the paper provides no metrics on inference time, memory footprint, or a clear discussion of the practical efficiency trade-offs between sampling density and spatial resolution.

### Questions
1. Given the marginal 0.5% gain from decoupled reinforcement finetuning, under what conditions does this method significantly outperform standard GRPO, justifying its added complexity?
2. Can the authors provide error analysis on zoom-in failures, such as mislocalization rates or incorrect decision-to-zoom vs. answer choices?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes LOVE-R1, a VLM with an adaptive zoom-in mechanism based on multi-step reasoning. The mode processes long videos with a "slow-fast" frame sampling template for fine-grained spatial details in key clips. With a three-stage pipeline: (1) slow-fast template fine-tuning, (2) Chain-of-Thought (CoT) cold start with annotated data to build basic reasoning abilities , (3) decoupled reinforcement fine-tuning to address the sparsity of outcome-based rewards. Experiments on long video benchmarks show LOVE-R1 outperforms Qwen2.5-VL.

### Strengths
1. The explored problem is meaningful and the motivation is clear. The grounding guided slow frame zoom in makes sense in long video understanding.
2. The slow-fast template design and the three stage training pipeline makes sense. The single-step supervision is crucial in video reasoning.
3. The CoT data construction is meaningful and is a contribution to the community.

### Weaknesses
1. The context length adopted in the paper is limited and the scalability remains doubtful. More experiments on models with large context window is necessary.
2. The CoT data derived from grounding data is narrow in domain and fails to cover general long video scenarios, which limits the generalization ability.
3. The paper lacks the efficiency analysis on the computation overhead introduced by the zoom in operation.

### Questions
Except the grounding data, are there other types of data that can be used for single-step optimization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new reasoning pipeline for Video-LLMs that performs temporal retrieval from the original clip and then zooms in to extract fine-grained details in later passes. The training is divided into three stages, initialized from Qwen2.5-VL. Experiments demonstrate clear gains over the baseline.

### Strengths
1. The method is well-explained, and the implementation details are sufficient for the community to reproduce the results.

2. The concept of temporal retrieval followed by multi-pass reasoning is intuitive and well-founded. It makes sense that video reasoning would benefit from zooming into relevant segments after a coarse-level pass.

### Weaknesses
1. The concatenation design highlighted in Figure 2 appears straightforward and not technically novel. Its inclusion as a major methodological highlight seems unnecessary.

2. The paper does not evaluate temporal grounding, even though many benchmarks (e.g., Charades-STA) provide standardized evaluation protocols for video temporal localization. Only video-QA experiments are reported, and the results are not state-of-the-art compared to 7B-scale counterparts, despite the proposed model processing more frames and multiple reasoning iterations.

### Questions
1. For each retrieval step, is only one slow-motion clip retrieved? If so, how does the model handle cases where clues are distributed across multiple, temporally distant clips? Does the framework support multi-hop retrieval?

2. In Table 1, the Stage 2 results appear to degrade compared to Stage 1 on the first two benchmarks where slow videos are uniformly sampled. This is counterintuitive. Does it suggest that those benchmarks may not actually benefit from detailed subclip reasoning?

3. Does the paper analyze the average number of reasoning iterations performed across different benchmarks? Additionally, can the model adaptively terminate early when sufficient information has been gathered, i.e., does it learn an implicit “early stop” or “reasoning drop-out” behavior?

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
5

### Summary
This study proposes the LOVE-R1 model, aimed at addressing the common spatio-temporal trade-off problem in long video understanding. The model constructs long video understanding as a multi-step reasoning process, first obtaining a global temporal overview through a low-resolution, high-density "fast video" stream, and then adaptively "zooming in" on specific key segments as needed, using a high-resolution "slow video" stream to capture spatial details. The model's "zooming in" decision-making ability is optimized through a three-stage training strategy and providing fine-grained process rewards, not solely relying on the correctness of the final answer.

### Strengths
The main advantage of love-r1 lies in its novel "amplification" mechanism, which provides a reasonable and effective paradigm for addressing the conflict between temporal density and spatial details in long videos. Second, the proposed "decoupled reinforcement fine-tuning" method tackles the issue of sparse multi-step rewards in GRPO by introducing process rewards, thereby more accurately enhancing the model's temporal localization capability. Finally, the experiments in the paper are solid, achieving performance better than baseline models on multiple long video benchmarks, and demonstrating the effectiveness of its key components (such as the amplification mechanism, video templates, and training method) through comprehensive ablation studies (e.g., Table 2).

### Weaknesses
1. The success of the entire reasoning chain depends on the model's ability to correctly identify the area that needs "zooming in" in the first step, but the "fast video" resolution is only around 168x168. If key clues, such as distant small objects or blurred text, cannot be recognized at this resolution, the model may make an incorrect "zooming in" decision in the first step, leading to all subsequent high-resolution analyses being based on wrong segments.
2. The context length may be limited by the "additional video template" (Template c): this template is cumulative, requiring the model to always retain the complete low-resolution "fast video" in the context while appending new high-resolution "slow video" segments at each inference step. This strategy leads to increased token consumption, hindering the model's ability to perform longer and deeper reasoning within a fixed context length, such as 16k.

### Questions
1. Will the constructed 38k CoT dataset be open-sourced?

### Soundness
4

### Presentation
3

### Contribution
3
