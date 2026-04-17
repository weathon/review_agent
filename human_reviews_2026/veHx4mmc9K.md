# FrameMind: Frame-Interleaved Chain-of-Thought for Video Reasoning via Reinforcement Learning

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Current video understanding models rely on fixed frame sampling strategies, processing predetermined visual inputs regardless of the specific reasoning requirements of each question. This static approach limits their ability to adaptively gather visual evidence, leading to suboptimal performance on tasks requiring either broad temporal coverage or fine-grained spatial detail. In this paper, we introduce **FrameMind**, a novel end-to-end framework trained with reinforcement learning that enables models to dynamically request visual information during reasoning through Frame-Interleaved Chain-of-Thought (FiCOT). Unlike traditional approaches, FrameMind operates in multiple turns where the model alternates between textual reasoning and active visual perception, using tools to extract targeted frames or video clips based on identified knowledge gaps. To train effective dynamic sampling policies, we propose Dynamic Resolution Frame Sampling (DRFS), which exposes models to diverse temporal–spatial trade-offs during learning, and DRFS-GRPO, a group-relative policy optimization algorithm that learns from outcome-based rewards without requiring frame-level annotations. Extensive experiments on challenging benchmarks like MLVU and VideoMME demonstrate that our method significantly outperforms existing models, advancing the state of the art in flexible and efficient video understanding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes FrameMind, a novel reinforcement learning–based framework for video reasoning that introduces Frame-Interleaved Chain-of-Thought (FiCOT)—an iterative reasoning process where the model dynamically requests visual evidence (e.g., specific frames or clips) during inference based on its current reasoning state. To train this adaptive perception policy, the authors develop Dynamic Resolution Frame Sampling (DRFS) and a tailored RL algorithm, DRFS-GRPO, which learns from trajectory-level rewards without frame-level annotations.

### Strengths
1. Innovative Reasoning Paradigm: FiCOT fundamentally rethinks video understanding as an active, iterative perception-reasoning loop, enabling the model to adaptively gather visual evidence.

2. Effective and Annotation-Efficient Training: The DRFS-GRPO framework enables end-to-end policy learning using only video-question-answer triples, avoiding the need for costly frame-level supervision.

### Weaknesses
1. On line 207, the authors mention that a VideoClip can sample 8–20 frames from a specific time interval; however, the manuscript nowhere explains how the specific frame number is determined.

2. The tool-call reward in Equation 7 incentivizes the model to invoke tools merely for the sake of invoking them. If the model can already produce the correct answer in the first reasoning step, any subsequent tool call only increases computational overhead without providing additional benefit. Nevertheless, Equation 7 strongly encourages such redundant and unhelpful tool invocations.

3. Equations 7, 8, and 13 contain numerous magic numbers, yet the authors provide no justification, analysis, or ablation studies regarding their selection.

4. Despite substantially increasing inference latency, the performance gains achieved by introducing additional tool calls are marginal: on Video-MME, FrameMind improves by less than 2% over Video-R1, and shows virtually no improvement on MVBench.

5. The evaluation is conducted on only three benchmarks, which is insufficient. The performance of FrameMind on other established long-video understanding benchmarks, such as LongVideoBench and LVBench, remains unknown.

6. While the paper mentions the data sources, it offers no details about the data curation process. It is unclear how the authors selected the final 7.6K samples from the much larger initial dataset.

### Questions
1. On line 257, the authors state that “DRFS enables the policy to learn which sampling strategy is most effective.” However, in standard evaluation settings, the sampling rate and resolution are typically fixed in advance. Under such conditions, it is unclear how the model can autonomously determine or adapt its “sampling strategy” during inference.

2. The turn_sum tag is first introduced on line 856, but its content is never described in the main text. What information does this tag contain, and how—if at all—does it influence the final results?

3. In the examples provided in the paper, the model consistently follows a reasoning pattern of first invoking VideoClip and then calling FrameAt. Is this sequential pattern representative of the majority of cases? If so, the authors should explain why the model converges to this specific strategy. If not, they should provide quantitative statistics, e.g., the frequency of different tool-calling sequences of different video length.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents FrameMind, a reinforcement learning framework for video reasoning that enables adaptive perception during reasoning. It introduces Frame-Interleaved Chain of Thought (FiCOT), where the model alternates between textual reasoning and targeted visual queries to retrieve relevant frames or clips as needed. To train the model effectively, the authors design Dynamic Resolution Frame Sampling (DRFS), which allows flexible frame selection across different resolution levels, and propose DRFS-GRPO, a group-relative PPO algorithm that learns from trajectory-level feedback instead of frame-level supervision. Experiments on benchmarks such as MVBench, MLVU, and VideoMME demonstrate consistent improvements over open-source video MLLMs and competitive results compared to proprietary models, showing enhanced performance on both short and long video understanding tasks.

### Strengths
1. The paper introduces a well-designed framework that integrates reinforcement learning with adaptive visual perception, allowing the model to dynamically select and utilize visual evidence during reasoning rather than relying on fixed frame sampling. This design effectively bridges the gap between perception and reasoning in video understanding.
2. The proposed Dynamic Resolution Frame Sampling (DRFS) and DRFS-GRPO algorithm are technically sound and novel. They enable efficient training without frame-level supervision and demonstrate clear performance gains across multiple benchmarks, validating the generality and robustness of the approach.

### Weaknesses
1. The paper does not provide sufficient explanation or empirical analysis on how the hyperparameters in Equations 7, 8, and 13 are determined. It remains unclear how these choices affect the overall performance and stability of the proposed framework.
2. The evaluation is limited to only three benchmarks—MVBench, MLVU, and Video-MME—which may not be sufficient to comprehensively demonstrate the general effectiveness of FrameMind across diverse video reasoning scenarios.
3. The baseline models used for comparison are relatively outdated, such as LLaMA-VID and Video-LLaVA from 2023, which limits the relevance of the comparison. Moreover, the paper lacks evaluation against recent RL-based methods, for example VersaVid-R1, which would provide a more meaningful assessment of the proposed approach.

### Questions
1. Could the authors provide more details on the selection process and sensitivity of the hyperparameters in Equations 7, 8, and 13? For example, have the authors conducted ablation or sensitivity analyses to show how different values affect training stability and final performance? Clarifying this would help assess whether the proposed framework is robust to hyperparameter choices.
2. Have the authors considered evaluating FrameMind on additional benchmarks, such as LongVideoBench, VideoMME-Pro, or activity-centric datasets like Something-Something V2 or NExT-QA? Broader evaluation would strengthen the claim of general effectiveness across diverse video reasoning tasks.
3. Could the authors include comparisons with more recent and competitive models, especially RL-based approaches such as VersaVid-R1 or R1-VidGPT? Including these results would make the empirical evaluation more convincing and position FrameMind more clearly within the current research landscape.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
FrameMind proposes a solution to Video QA by training a VLM to use tools for querying frames and information. It effectively trains the model to use tools to query different frame-rates and resolutions, and achieves this by generating parallel rollouts with different samples from a `resolution-ladder’. It modifies GRPO to handle this setup, and trains the model end-to-end to be able to query videos for more information as needed. The results demonstrate that FrameMind outperforms existing models on standard video QA benchmarks.

### Strengths
1) I think this is a creative and thorough way to improve video and proposes a novel way to effectively train VLMs to query for more information. 

2) The experiments are generally comprehensive and the results are quite strong, demonstrating a clear improvement over the state of the art.

3) The paper is easy to follow and well written, allowing readers to clearly grasp the main contribution and how it works.

### Weaknesses
1) The novelty is slightly limited. An existing work VITAL[1] does a similar approach, augmenting a VLM with  tools and GRPO training to improve chain-of-thought for video QA.  While, there is  clear novelty in the multi-resolution training and modification to GRPO to handle this, it’s very specific to the problem of teaching a model to effectively query video frames and hard to see how the proposed method could generalize to other problems.

2) The tool set is slightly limited - one could image several other tools being helpful for video QA in general, such as detection, tracking, segmentation, cropping, etc. However, omitting these is acceptable in my view.

3) I would appreciate some visual / qualitative samples of what the model selects compared to other methods to better understand its performance in practice. Figure 2 is hard to parse because the frames are so small.

[1] Zhang, H., Gu, X., Li, J., Ma, C., Bai, S., Zhang, C., ... & Tang, Y. (2025). Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416.

### Questions
The only main concern I have is about the slightly limited novelty of the overall problem - frame selection is a well studied problem, and prior works have attempted to improve this with RL and through augmenting VLMs with tools + GRPO. If this were addressed I'd gladly increase my rating.

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
3

### Summary
FrameMind tackles adaptive video understanding by enabling models to dynamically select frames during reasoning through a Frame-Interleaved Chain-of-Thought (FiCOT) process. It combines textual reasoning with tool-assisted frame retrieval, guided by Dynamic Resolution Frame Sampling (DRFS) and trained via a custom RL algorithm, DRFS-GRPO. This setup allows the model to choose between broad scans and fine-grained focus at each step. Experiments on benchmarks like MVBench, MLVU, and VideoMME show that FrameMind outperforms prior open-source and even some proprietary models, achieving state-of-the-art accuracy with fewer frames. The results validate its effectiveness for efficient, frame-aware video reasoning.

### Strengths
- Methodology - By interleaving chain-of-thought with active frame selection (FiCOT), the model can dynamically adjust its perception strategy on-the-fly. This is interesting and different from the conventional one-pass video processing. The use of reinforcement learning (DRFS-GRPO) to train the perception-policy is also a strength, as it allows learning from outcome-based rewards without requiring any hard-coded frame annotations. The paper shows that this training strategy is effective through ablation study.
- Quantitative Results - FrameMind delivers state-of-the-art performance across benchmarks, outperforming open-source models and even surpassing GPT-4V on VideoMME. It achieves strong results on both short and long video tasks while using up to 4× fewer frames, highlighting its effectiveness.

### Weaknesses
- Evaluation Scope - The paper focuses solely on video question-answering tasks, leaving its generalizability to other video understanding tasks—like captioning or real-time analysis—unclear. While effective for QA, it’s unknown whether FiCOT and the learned policy extend to more open-ended or interactive scenarios. Broader evaluation would strengthen the work.
- Reward Design and Parameter Sensitivity - The RL setup uses a composite reward with multiple hyperparameters, but the paper doesn't discuss how sensitive performance is to these choices. For instance, the 2–3 turn efficiency bonus lacks justification, and poor tuning could lead to under- or overuse of tools. This sensitivity may limit generalization and complicate adoption in new domains. 
- Inference Efficiency - FrameMind uses fewer frames than baselines (e.g. 4x fewer on MLVU), but does this translate to faster inference time in practice? While fewer frames are processed overall, the agent performs multiple reasoning turns and tool calls. Measuring this would inform how the approach might scale in real-world deployments or if the dynamic strategy mainly benefits accuracy/efficiency in terms of frame budget rather than latency.

### Questions
- While FrameMind uses fewer frames, does the multi-turn strategy result in faster inference overall? Have you measured wall-clock runtime or compute cost compared to single-pass models to assess real-world scalability?
- How sensitive is performance to the 2–3 turn bonus (λ_turn = 0.5)? Could this discourage longer reasoning when needed? Clarifying the rationale and whether other turn limits or bonus values were tested would be helpful.
- Can FiCOT extend beyond structured QA to open-ended tasks like video captioning or long-form answers? If so, what adjustments—such as new reward signals or stopping criteria—would be needed to support those use cases?
- Did you observe any common failure modes for FrameMind? For example, are there instances where the model selects an irrelevant frame or misinterprets the question and thus retrieves suboptimal evidence?

### Soundness
3

### Presentation
3

### Contribution
3
