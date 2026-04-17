# Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Long-form video understanding, characterized by long-range temporal dependencies and multiple events, remains a challenge. Existing methods often rely on static reasoning or external visual-language models (VLMs), which face issues like complexity and sub-optimal performance due to the lack of end-to-end training. In this paper, we propose Video-MTR, a reinforced multi-turn reasoning framework designed to enable iterative key video segment selection and question comprehension. Unlike traditional video reasoning pipeline, which generate predictions in a single turn, Video-MTR performs reasoning in multiple turns, selecting video segments progressively based on the evolving understanding of previously processed segments and the current question. This iterative process allows for a more refined and contextually aware analysis of the video. To ensure intermediate reasoning process, we introduce a novel gated bi-level reward system, combining trajectory-level rewards based on answer correctness and turn-level rewards emphasizing frame-query relevance. This system optimizes both video segment selection and question comprehension, eliminating the need for external VLMs and allowing end-to-end training. Extensive experiments on benchmarks like VideoMME, MLVU, and EgoSchema demonstrate that Video-MTR outperforms existing methods in both accuracy and efficiency, advancing the state-of-the-art in long-form video understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Video-MTR is a reinforced multi-turn reasoning framework for long-form video understanding, addressing challenges like long-range temporal dependencies and multi-event complexity. Unlike existing methods (static single-turn reasoning or external VLM-reliant agentic paradigms), it enables iterative key segment selection and end-to-end training via reinforcement learning (RL). Built on Qwen2.5-VL-7B, it uses a gated bi-level reward system (trajectory-level: answer correctness; turn-level: frame-query relevance via IoU) to guide multi-turn reasoning. It starts with uniform frame sampling, then retrieves relevant segments iteratively until confident or reaching a 3-turn limit. The proposed model is trained on only 8K curated samples (from NExT-GQA and QVHighlights), far fewer than existing methods (256K–4.4M).
Experimental Results: Outperforms baselines on 4 benchmarks (VideoMME: 62.2%, MLVU: 49.8%, LVBench: 41.8%, EgoSchema: 63.4%) with 7B parameters and ≤64 frames, matching large proprietary models (e.g., Gemini-1.5-Pro) with fewer resources.

### Strengths
Strong Long-Video Reasoning: Multi-turn iteration avoids missing critical info in long videos, with larger accuracy gains for longer videos (+6.3% on long vs. +4.6% on short in VideoMME) and complex tasks (+8.1% on multi-detail tasks in MLVU).

High Efficiency:
Data-efficient (8K samples vs. hundreds of thousands/millions).
Compute-efficient (7B parameters, ≤64 frames) vs. large models or high-frame-budget methods.
Balanced latency (427.2 ms at 3 turns) vs. accuracy.

End-to-End & Tool-Independent: Eliminates external VLMs, avoiding heterogeneous component complexity and enabling unified optimization of segment selection and question comprehension.

Robust & Generalizable: Performs consistently across benchmarks; works for smaller models (Qwen2.5-VL-3B) with accuracy gains.

### Weaknesses
1. The idea of first grounding/localization and then answering questions is not novel.

2 .Limitations in Complex Tasks: Struggles with multi-event (e.g., action-order) tasks (early stopping due to training bias) and fine-grained perception (coarse frames blur micro-actions like "brush dipping vs. mixing").

3. Turn-level rewards need high-quality temporal annotations, making scaling to new domains costly.

### Questions
No

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
This paper proposes Video-MTR, a reinforced multi-turn reasoning framework designed to enable iterative key video segment selection and question comprehension. Video-MTR performs reasoning in multiple turns, selecting video segments progressively based on the evolving understanding of previously processed segments and the current question. Extensive experiments on benchmarks like VideoMME, MLVU, LVBench, and EgoSchema demonstrate that VideoMTR outperforms existing methods in both accuracy and efficiency, advancing the state-of-the-art in long video understanding.

### Strengths
1. As claimed by the authors, this could be the first attempt to incorporate multi-turn reasoning in the context of long video understanding.
2. The proposed method can adaptively select important frames for the question.
3. Experiments on multiple long-video benchmarks show that Video-MTR outperforms its backbone Qwen2.5-VL with the same number of frames.

### Weaknesses
1. Qwen2.5-VL can support up to 768 frames and outperforms the proposed Video-MTR with the input of 64 frames. More experiments should be conducted to investigate whether Video-MTR can outperform Qwen2.5-VL with more frames.
2. It's weired that the QA accuracy in Figure 4 exceed 1. Some explanations should be provided.
3. The information of compared baseline models is not given, especially their backbone models.
4. While introducing multi-turn reasoning, the efficiency compared to baselines should be analyzed.
5. The training framework is complicated with various tricks, which may be unstable in other scenarios.

### Questions
Please reply to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Video-MTR, a reinforcement-learning based framework for multi-turn reasoning in long video understanding. The main idea is to start with uniform frame sampling, and then use an MLLM to iteratively decide whether to retrieve additional frames or answer the question. The policy is optimized using PPO with a gated bi-level reward that combines final-answer correctness and turn-level frame-query relevance. Experiments on VideoMME, MLVU, and EgoSchema show improvements over existing open-source baselines on long videos while using relatively few training examples.

### Strengths
- The paper studies the under-explored idea of combining reinforcement learning with long-video understanding, and the proposed method is novel. The empirical results are quite strong and Video-MTR beats a number of competitive baselines.
- The paper is mostly clearly written and easy to read. The proposed method is well-motivated.
- The paper conducts a number of ablation experiments, and in addition to QA accuracy it also measures latency as an additional metric in some experiments in the appendix.

### Weaknesses
- The design of the reward function seems to be a bit ad-hoc. It would be useful to know how sensitive the method is to hyper‐parameters (thresholds and bonus amounts in each stage).
- The paper is lacking some error analysis and discussion on failure modes, e.g. when wrong segments are retrieved.

There are a few occasions in the paper where there might be ambiguities or inaccurate claims, and I would hope that they can be clarified in the paper:
- Around line 78 the authors claim that "this is the first attempt to incorporate multi-turn reasoning in the context of long video understanding." This claim is a bit inaccurate since one can argue that the large body of existing works in "agentic" video models (such as "VCA: Video Curious Agent for Long Video Understanding", Arxiv 2412.10471) are also doing multi-turn exploration and refinement of the selected video frames within a long video.
- Around line 335 the authors claim that "Most open-source long-video methods operate with ≤ 128 frames." It used to be the case but today more and more models support longer context. For example the Qwen2.5-VL-7B model that the paper refers to officially supports a context length of 131072 and 768 frames when processing videos.

### Questions
I would like the authors to discuss the following concerns I have on the methodology and the experiments. They are not necessarily weaknesses of the paper, but rather questions I would like to gather more information on from the authors:
- The dataset the authors curated has a size of only 8K, which might be a bit too small for long videos. Do the author agree that it is a valid concern that policy may rely on heuristics tied to those training datasets rather than truly understand how to retrieve, and might not work well on new video domains?
- For ultra long videos for questions that require complex reasoning and fine-grained understanding at multiple locations of the long video, retrieving only 32/64 frames and only using 3 turns might not be sufficient. Do you have a sense on why increasing these numbers did not lead to uniform increase in performance in your experiments?

### Soundness
3

### Presentation
3

### Contribution
3
