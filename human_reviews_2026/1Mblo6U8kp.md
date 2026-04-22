# VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Recent studies have shown that long chain-of-thought (CoT) reasoning can significantly enhance the performance of large language models (LLMs) on complex tasks. However, this benefit is yet to be demonstrated in the domain of video understanding, since most existing benchmarks lack the reasoning depth required to demonstrate the advantages of extended CoT chains. While recent efforts have proposed benchmarks aimed at video reasoning, the tasks are often knowledge-driven and do not rely heavily on visual content. To bridge this gap, we introduce **VideoReasonBench**, a benchmark designed to evaluate **vision-centric, complex video reasoning**. To ensure visual richness and high reasoning complexity, each video in VideoReasonBench depicts a sequence of fine-grained operations on a latent state that is only visible in part of the video. The questions evaluate three escalating levels of video reasoning skills: recalling observed visual information, inferring the content of latent states, and predicting information beyond the video. Under such task setting, models have to precisely recall multiple operations in the video, and perform step-by-step reasoning to get correct final answers for these questions. Using VideoReasonBench, we comprehensively evaluate 18 state-of-the-art multimodal LLMs (MLLMs), finding that most perform poorly on complex video reasoning—e.g., GPT-4o achieves only 6.9\% accuracy—while the thinking-enhanced Gemini-2.5-Pro significantly outperforms others with 56.0% accuracy.  Our investigations on "test-time scaling" further reveal that extended thinking budget, while offering none or minimal benefits on existing video benchmarks, is essential for improving the performance on VideoReasonBench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a video question answering benchmark focusing on the visual reasoning ability of the MLLMs. In specifice, the videos are evenly created for six tasks and the question requires the compherehensive understanding to answer. Experiemnts shows that existing MLLMs struggles in answering the questions, and enableing thinking ability significantly boosts the models' performance.

### Strengths
● The paper is well motivated. Evaluating the visual reasoning ability of MLLMs can be crutial. 
● The proposed benchmark is of reasonable size and distribution.
● Experiments verified the proposed benchmark is challanging for existing MLLMs, and performance gain of CoT supports the claim that the questions require visual reasoning to solve.

### Weaknesses
● The proopsed data construction pipeline is highly specialized for the proposed types of questions, and can hardly generalize to boarder question types.
● In table 2, the experiment shows that vid2txt setting significantly outperforms traditional VidQA setting (over 48% on gemini-2.5-flash). This somewhat violates the claim that the benchmark is challanging the visual centric reasoning ability of MLLMs as this implies that the perceptual ability seems playing a more important role  than reasoning for the low performance of current MLLMs.

### Questions
● The comparison of vid2txt setting in table 2 seems suggensting that failure of current MLLMs are mainly caused by the percetion failure rather than reasoning failure as the performance gap between vid2txt setting and traditional VidQA setting is significantly larger than that between models w./w.o. thinking. Therefore, the main challenge of this benchmark is perception rather than reasoning. How do author explain this performance gap and its contradictory with the main claim?

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
3

### Summary
This paper introduces a diagnostic benchmark for video reasoning that treats each clip as a sequence of visible operations acting on a partially hidden state. Tasks span perception, latent-state inference, and counterfactual prediction across both synthetic and real scenes with programmatically generated ground truth. Evaluations show current multimodal models struggle, especially when fine-grained temporal perception and long-horizon reasoning are required; converting videos to textual state–operation traces and allowing longer chains of thought markedly narrows the gap. The paper highlights perception bottlenecks, the value of explicit reasoning, and the need for methods that integrate stronger visual tracking, external memory, and executable reasoning.

### Strengths
1. The benchmark formalizes video as “latent state + a sequence of visible operations,” decomposed into three reasoning levels with six concrete skills, which makes failure modes inspectable and comparable.

2. The study evaluates a wide span of open-source MLLMs

### Weaknesses
1. Since most tasks are scored with a judge LLM, would you consider reporting a brief robustness study (e.g., prompt ablations, temperature sweeps, and an alternative judge model) and an agreement statistic?

2. To make cross-model comparisons easier, could you add some description about budget setting, such as fps/frame count and max tokens?

3. The results indicate accuracy drops as state size and operation count increase, and when the final state is only revealed at the end. What may happen if we extend scenarios to minutes-scale durations or larger state spaces?

### Questions
Please refer to the weakness

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
4

### Summary
The paper presents VideoReasonBench, a vision-focused benchmark that emphasizes step-wise reasoning over long sequences of visually observable operations on partially observable latent states such as sliding-tile board states. It defines three increasing skill levels: recall for fine-grained operation perception, infer for latent state inference, and predict for counterfactual planning. It includes six demonstration types: Number, Circle, Cup, File, Card, and Chip. The dataset contains 240 videos and 1,440 questions, with structured prompts and automatic answer generation based on state-transition scripts. Evaluation uses a text-only LLM judge; for Predict Operation, it simulates model-proposed operations to verify target states. Among 18 MLLMs, most models perform poorly with less than 10 percent accuracy overall. Gemini-2.5-Pro in thinking mode reaches 56 percent, and the benchmark shows strong sensitivity to thinking budget and heavy reliance on vision compared to earlier video benchmarks.

### Strengths
The benchmark covers both synthetic and real-world capture paths, including Matplotlib programmatic renderings, terminal screenshots, and manual videos, with balanced distributions across skills and demos and controlled settings for state size and operation length to adjust difficulty.

### Weaknesses
W1: For five of six skills, correctness is determined by a text-only LLM judge given GT and the model output. Even with careful prompts, LLM graders can show bias or instability and may prefer format-matched answers over truly correct content. The paper does not report inter-judge agreement, self-consistency, or adversarial sensitivity such as paraphrase, verbosity, or distractors. This is quite concerning.
For Predict Operation, answers are extracted using an LLM, then simulated. This introduces another potential failure point in extraction, but there is no error analysis distinguishing extraction failures from planning failures.
vid2txt is generated from the ground-truth state transitions. It works well for isolating perception versus reasoning, but it effectively provides a clean event log the models do not need to perceive. It is possible that the performance gap partly reflects a best-case text abstraction that few real video systems could realistically obtain.


W2: The authors stated that the videos and questions in these benchmarks often resemble those in earlier benchmarks and fall short in demanding deeper video reasoning. It is unclear how this dataset helps address that issue. It is also unclear in what way the previous ones resemble.



W3: According to the paper, all videos are synthetic or semi-synthetic demonstrations built from controlled state-transition scripts rather than natural videos.
It seems the dataset favors discrete and symbolic reasoning. All operations appear atomic and reversible, meaning the benchmark strongly favors models that can internally represent counting, position updates, or ordering. This raises the concern that it may overfit to symbolic reasoning rather than perceptual or intuitive physics reasoning.
It may also underestimate challenges such as motion blur, occlusion, object continuity, or ambiguous visual events found in open-world settings. But this is a minor concern.

### Questions
How consistent is the LLM-as-judge across paraphrased or differently formatted answers?

Can you analyze the potential bias introduced by the six procedural video types and clarify whether these discrete settings generalize to naturalistic visual reasoning?

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a vision-centric benchmark for complex video reasoning. By revealing latent states only at the beginning or end of the video and presenting a sequence of visible actions, the benchmark systematically evaluates three progressive levels of reasoning: (Level 1) exact recall of the action sequence and counts; (Level 2) inference of invisible states from observed actions and comparison of differences; (Level 3) counterfactual prediction of future states or required actions. The benchmark spans six scenario types (numeric sliders, board flipping, cup–lid swapping, file operations, card stacks, chip–cup), comprising 240 videos and 1440 QA pairs, with evaluation via an LLM-based textual judge and executable simulation.

### Strengths
1. The work effectively identifies a key deficiency in current video benchmarks: weak alignment with reasoning and low demand for exploiting video information. For instance, Table 2 shows that using raw video input underperforms a video-to-text pipeline, and Table 3 indicates that prior benchmarks can be partially solved with text-only inputs.
2. As a benchmark for assessing MLLM reasoning, it demonstrates a positive correlation between accuracy and scaled reasoning length, which aligns with expectations for reasoning-focused evaluation.

### Weaknesses
1. The task format is relatively narrow, covering only six highly customized scenarios. If models are exposed to similar data during training, they may quickly overfit or “hack” the benchmark.
2. The substantial gap between open-source and closed-source models in Table 2 warrants analysis. Is this driven by differences in training data coverage or by capability gaps? For example, open-source models may have been trained on fewer or less diverse data and thus not encountered similarly customized scenario tasks, leading to lower performance.
3. In Fig. 5(b), the largest thinking budget (8192) yields the shortest response length. Please explain this phenomenon, e.g., by providing illustrative cases.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
