# VideoZoomer: Reinforcement-Learned Temporal Focusing for Long Video Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable progress in vision-language tasks yet remain limited in long video understanding due to the limited context window. Consequently, prevailing approaches tend to rely on uniform frame sampling or static pre-selection, which might overlook critical evidence and unable to correct its initial selection error during its reasoning process. To overcome these limitations, we propose VideoZoomer, a novel agentic framework that enables MLLMs to dynamically control their visual focus during reasoning. Starting from a coarse low-frame-rate overview, VideoZoomer invokes a temporal zoom tool to obtain high-frame-rate clips at autonomously chosen moments, thereby progressively gathering fine-grained evidence in a multi-turn interactive manner. Accordingly, we adopt a two-stage training strategy: a cold-start supervised fine-tuning phase on a curated dataset of distilled exemplar and reflection trajectories, followed by reinforcement learning to further refine the agentic policy. Extensive experiments demonstrate that our 7B model demonstrates diverse and complex reasoning patterns, yielding strong results across a broad set of long video understanding and reasoning benchmarks. These emergent capabilities allow it to consistently surpass existing open-source models and even rival proprietary systems on challenging tasks, while achieving superior efficiency under reduced frame budgets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an agentic video processing framework: the model is first provided with several frames at a low frame rate and then it can use the temporal zoom tool to obtain high-frame-rate clips at autonomously chosen moments. The framework is worked in a multi-turn manner. The training undergoes a two-phase recipe: a cold-start supervised fine-tuning phase and a reinforcement learning phase. And the authors construct a cold start dataset with 11k samples and diverse reasoning patterns. The resulting model VideoZoomer achieves a remarkable performance improvement over the baseline Qwen2.5-VL 7B.

### Strengths
1. The paper constructs a cold start dataset with diverse reasoning patterns, especially the reflection data. Construction reflection data from self-generated failure CoTs is interesting and reasonable, which could be a better way to collect high-quality CoT data.
2. The model is evaluated on a broad range of benchmarks, including long video understanding benchmarks and long video reasoning benchmarks. The robust improvements across multiple benchmarks validate the effectiveness of the proposed methods.
3. The paper also demonstrates that VideoZoomer can further equip a frame selector for frame initialization to improve the performance, showing that the method is orthogonal to other dynamic frame sampling methods.

### Weaknesses
1. This work uses LongVideoReason as the training dataset and constructs a cold start training dataset. And the reflection data is constructed from incorrect rollouts. The rollouts are multi-step CoTs, and the errors may occur at any step, CoTs for zoom-in or CoTs for answers.  However, it seems that the LongVideoReason dataset does not have gt timespan annotations. How to determine whether the CoTs for zoom-in are correct or not?
2. The cold start dataset is relatively small.
3. The ablation study on reflection cold start data is not convincing, as more sft data is used.
4. The performance of Qwen2.5-VL 7B baseline on long video understanding benchmarks is relatively low with 128 frames. Can you provide more inference details?
5. The framework is a multi-step reasoning process. A common failure case is that the reasoning can not stop (i.e. the model zooms in on the video endlessly or the model does not provide a valid answer within the budget). How to handle this failure case during training and inference?
6. Compared with other dynamic frame sampling works, this work introduces another sampling variable, 'fps'. And in the provided cases, Figures 4 and 8, fps=8 is extremely high compared to other sampling strategies. Can authors show how the predicted fps affects the performance?
7. The 'fotmat' in Equation (1) is a typo.

### Questions
Please see the weaknesses above.

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
3

### Summary
VideoZoomer is an agentic long-video QA framework that starts with a coarse “glance” and then issues targeted zoom calls over selected time spans to gather fine-grained evidence before answering. It’s trained in two stages—cold-start supervised trajectories (including reflection to correct failures) followed by reinforcement learning with rewards that encourage accurate, well-formatted, and purposeful tool use. Experiments across diverse long-video understanding and reasoning benchmarks show consistently better accuracy–efficiency trade-offs than single-pass baselines, and the approach remains compatible with stronger initial frame selectors.

### Strengths
1. This paper introduces a clear agentic framework that couples coarse “glance” perception with targeted temporal zooming, yielding a principled separation between broad coverage and fine-grained evidence acquisition.

2. This paper evaluates across diverse long-video understanding and reasoning benchmarks, with the largest gains on tasks that require precise temporal detail, supporting the method’s intended use case.

### Weaknesses
1. The zoom tool is basically one-dimensional: A lot of long-video questions hinge on tiny textual clues (scoreboards, signs, on-screen text), and just “adding frames” won’t reliably capture those.

2. The cold-start data comes from external frontier models (e.g., GPT-style teachers). That brings possible style bias

3. Multi-round zooming can be expensive in practice.

### Questions
1. How does accuracy change with 0/1/2/3/4 zoom calls? Is there a point of diminishing returns?

2. When the model zooms the wrong time span, does it recover in later rounds, or does it lock in and fail?

3. How often does the model answer correctly without making any zoom calls, and what types of questions and accuracy are those?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces VideoZoomer, an agentic model enabling MLLMs to dynamically adjust their visual focus during reasoning. Beginning with a low-frame-rate overview, the model uses a temporal zoom tool to capture high-frame-rate clips at key moments, progressively gathering fine-grained evidence. The training process involves supervised fine-tuning on curated datasets, followed by optimization with DAPO. Experimental results demonstrate that the 7B model exhibits diverse and sophisticated reasoning capabilities, achieving strong performance in long video understanding tasks.

### Strengths
- The paper contributes a training dataset comprising 11,000 trajectories, which is used to enhance the tool-calling capabilities of models.  

- The case visualizations presented in the paper are good

### Weaknesses
- The technical contributions of the paper are limited, as its main novelty lies in providing a curated training dataset to enhance the tool-calling capabilities of models.

- The method adds a bonus to tool-call rewards when the final answer is correct, which increases the unnecessary frequency of tool use. For example, the model may continue calling tools unnecessarily, retrieving irrelevant clips even after it already has the correct answer, ultimately inflating the reward.

- While the paper mentions using 11,000 trajectories for the cold-start phase of training, it does not specify the data used during the RL phase, leaving gaps in transparency and reproducibility.

### Questions
- Will further scaling up the training dataset continue to improve the model’s performance?  

- The paper mentions starting with 64 frames as input, gradually increasing by 16 frames each time. However, in Figure 1 (the right part), the number of input frames for VideoZoomer seems inconsistent with this description.  

- Will the code and the training dataset used in the paper be open-sourced?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the video LLM with self-bootstrapped clip selection capability by SFT and RL to reduce the frame sampled and inference cost. This paper presented VideoZoomer, a agentic framework integrating <video_zoom> tool call and <think> thinking abilities, to select video clips for its processing. The proposed VideoZoomer method can reach outperforming accuracies compared to previous methods while using less input frames. The author also performed detailed ablation studies on cold-start, reflection, RL, and score design.

### Strengths
1) This paper is one of the first cohort to explore the frame selection method by agentic RL in video LLM domain, delivering substantial novelty. Also, the agentic RL framework is not simply adopted from the image/language domain to the video domain. Instead, it incorporates tool calling and methods designed specifically for videos (e.g., temporal zoom-in, on-policy reflection).

2) The resulted agentic model by GRPO training, outperforms the baseline model by a large margin, especially on the long video understanding task. The author also detail key components in proposed RL framework, showing that the cold-start and reflection finetuning is essential for good RL models in reasoning video frame selection model, which is of great valuable insights.

### Weaknesses
Overall this paper is of great technical value and soundness, but there are some minor concerns listed below:

1) There are several minor typos across texts, including line 151: "stragety", line 277: "fotmat" and more.  The author should perform grammar and word check throughout the paper;

2) Are other capabilities of video LLMs (Qwen 2.5-VL) well maintained? Like short video captioning?

3) Qwen 2.5-VL is known to lack of native <think></think> reasoning capabilities. The authors performed off-policy warm-start and on-policy reflection SFT to enable the reasoning capability but is it robust when the input data is out of distribution of SFT training data, like very simple CLEVRER data? Just curious and I want to hear the authors' insights into it.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3
