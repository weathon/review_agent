# FrameOracle: Learning What to See and How Much to See in Videos

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Vision-language models (VLMs) have advanced video understanding, but their performance is limited by the number of input frames they can process. Existing frame sampling strategies, such as uniform or fixed-budget selection, often fail to adapt to variations in information density or task complexity, resulting in inefficiency and information loss. To address this, we present **FrameOracle**, a lightweight and plug-and-play module that predicts both (1) which frames are most relevant to a given query and (2) how many frames are needed. FrameOracle is trained using a four-stage curriculum, with the first three stages relying on weak proxy signals such as cross-modal similarity. In the final stage, it leverages stronger supervision from a new dataset we introduce, **FrameOracle-41K**, the first large-scale VideoQA collection to provide keyframe annotations specifying the minimal set of frames required to answer each question. Extensive experiments across five VLMs and six benchmarks demonstrate that FrameOracle reduces 16-frame inputs to an average of 10.4 frames without any loss in accuracy. When starting from 64-frame candidates, it reduces the input to an average of 13.9 frames while improving accuracy by 1.4\%, achieving state-of-the-art efficiency-accuracy trade-offs for scalable video understanding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses a key challenge in video understanding with vision-language models (VLMs):
How to efficiently select the most relevant frames from a video to answer a given query, while also determining how many frames are actually needed. To answer these questions the authors propose a 'plug-and-play' frame selection (named FrameOracle) to select which and how many frames to use for a given query. In addition, the paper also contributes a dataset named FrameOracle-41K which contains information regarding the important frames for each query inside the dataset.

### Strengths
1 - Adaptive frame selection method that selects how many frames and which frames to use, an extension of what current SOTA models are doing.
2 - New dataset: the dataset is a significant contribution of this work, despite the little attention given to it.
3 - The authors have considered many different datasets which increase the reliability of the method in the considered settings.

### Weaknesses
After carefully reading the paper I have the following doubts which I categorize as weaknesses:

1 - The adaptive frame selection setting: While the authors claim adaptive frame selection, they are in fact doing adaptive frame selection from a fixed pool of frames. This is not aligned with the nature of videos which can be in variable sizes. This becomes more critical when you apply the method to long video understanding, depending on the quantity of information the video contains and the dynamic, using 16 or 64 frames (uniformly sampled before the frame selection mechanism) is not enough let alone when reduced to ~10 frames. While this is mentioned as a limitation from the authors, I fail to recognize an important contribution from the method if the frame selection cannot operate on variable sequence length (or even fixed but complete) since it is always bounded to the correctness of the uniform sampling.

2 - Plug and play: The method is not actually plug and play since the query is encoded from the vlm tokenizer, which means for every VLM with different tokenizer, you have to train a separate model. It would have been plug and play if you would have used the for example the SigLIP language encoder.

3 - Feature fusion: the cross-modal fusion is an integral part of your technical contributions, but it is unclear how you do it. Do you concatenate the tokens?

4 - Transformer encoder layer: Features are fused and then sent to the transformer encoder layer. How does the transformer process the tokens, is it a global attention (i.e. all tokens from all frames), is it spatial and then temporal, or is it only spatial? This is related to weakness 3 also. I guess the architecture part is a bit undermined in this work. The reason why I stress this point is because the frame selection is mainly a mechanism to reduce computation while keeping or improving the accuracy. In terms of memory, for an LLM/VLM the most expensive operation is the self-attention (let's consider a plain self-attention) due to quadratic memory scaling. Now, if the encoder layer is using self-attention among all the tokens, it means the transformer encoder has memory requirements similar to those of the LLM/VLM during computations (the difference would be the number of heads) with exception to the system prompt tokens. So, while reducing latency, the frame selector has big memory requirements. I suggest this point is clarified.

5 - Why not use the visual tokenizer of the VLM directly, the method is not plug-and-play anyway. The visual tokenizer of the VLM is already align with language (so no stage-1 training), and can possibly ease the 'which frames to use' problem. (Note, this will not affect my evaluation negatively, is more for my own curiosity.)

6 - The 4 stage training: What would happen if the training would consist only of stage-1 and 4? An experiment would be interesting. Additionally, stage-2 and -3 freeze and unfreeze the Rank head and K head. If you train the model (even with a low learning rate) in stage-3 while you freeze Rank head, the performance of Rank head will decrease. Now how big of the problem this is depends on the frames given in input and the video composition and it would be problematic for long videos and scenarios with high sensitivity on the frame choice. The approach is not validated rigorously to have a conclusion on this matter.

7 - Validation: While the paper validates across many datasets and compares with previous works in different datasets, I think it is evaluated in a very shallow way without depth. The ablations are not very comprehensive, just the 4 stages. No insights on the visual backbone choice or text tokenizer, and many more (see above).

8 - Dataset: The dataset, in my understanding would be the bigger contribution to this work but it is clearly overlooked and very little analysis and experiments are done with it. 

9 - While the work considers only frame selection mechanism, entering in the long video world, I guess is fair to consider and compare against works that apply to long video understanding (is not necessary for the proposed method to surpass those works, but to have an idea how it actually helps when compared to methods designed for long videos). 
You can have a look at: "Moviechat+: Question-aware sparse memory for long video question answering." IEEE Transactions on Pattern Analysis and Machine Intelligence (2025), which is training free but uses a memory layer (plug-and-play) to compress frames rather than select them, or "ReWind: Understanding Long Videos with Instructed Learnable Memory." Proceedings of the Computer Vision and Pattern Recognition Conference (2025). This is more a suggestion to see how selecting compares to compression. 


Given these concerns, I will suggest a weak reject. The work has no significant technical contributions, it is incremental. In addition, the theoretical contributions are not significant to compensate the technical ones. The dataset is the only significant contribution but is clearly not the main focus of the paper. I am open to improve my score if my concerns are addresses or my interpretations are deemed as not correct.

### Questions
Check the Weaknesses section.

### Soundness
2

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
5

### Summary
The paper proposes a lightweight and plug-and-play module capable of dynamically selecting a variable number of frames based on the difficulty of each question.

In addition, the authors introduce a curriculum-based training strategy to effectively train this frame selection module.

The paper also designs a data generation pipeline that provides the minimal set of frames required to answer each question, forming the first VideoQA dataset with such keyframe annotations.

### Strengths
1. The paper is detailed and clearly presented, with strong motivation and solid overall design.

2. A novel module is introduced to jointly predict the number of frames to select and frame-level importance scores, along with a carefully designed training paradigm. The method is validated across multiple backbones, showing good generalization.

3. The authors build a new data generation pipeline, providing the first VideoQA dataset with minimal keyframe annotations, which is a valuable contribution to the community.

### Weaknesses
1. The proposed multi-head design for predicting both frame count and frame importance is reasonable, but it depends on the backbone’s global reasoning capability. The paper should clarify whether different backbones lead to significantly different results.

2. It would be helpful to compare against a strategy that fixes or predicts a total information budget (instead of a frame count) as the selection target — would such a formulation be more reasonable?

3. The paper adopts a curriculum learning scheme with four progressive training stages, and the ablation study supports each stage’s usefulness. However, is such staged training truly necessary? Could joint training achieve similar results? A comparison would make the claim more convincing.

4. Since the selected frames are all highly important, do they sometimes concentrate around similar patterns, causing redundancy in visual information? A discussion or visualization could help clarify this.

### Questions
Please see the weakness.

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
The paper introduces FrameOracle, a lightweight and plug-and-play frame selection module that dynamically determines both the most relevant frames and the number of frames required for a given video understanding task. To support its training, the authors also present FrameOracle-41K, the first large-scale VideoQA dataset annotated with keyframes that specify the minimal frame subset needed to answer each question. The proposed approach is evaluated across six benchmarks and compared against five vision-language models (VLMs), demonstrating strong efficiency gains while preserving task accuracy.

### Strengths
- The paper introduces FrameOracle-41K, a large-scale dataset specifically created for keyframe selection in VideoQA, with annotations indicating the minimal set of frames required to answer each question.

- The proposed method improves computational efficiency by reducing the number of processed frames, while still maintaining comparable or better accuracy than full-frame baselines.

- FrameOracle outperforms existing keyframe selection methods, showing better frame relevance and stronger downstream task performance across multiple benchmarks.

### Weaknesses
- The data generation process heavily relies on another agent model for producing keyframe annotations, which raises concerns about potential bias, annotation noise, and the dependency of the dataset quality on the agent’s capabilities.

- The data generation pipeline appears relatively simple and lacks clear novelty. How does it differ from existing data generation approaches, and what unique contributions does it offer?

- The training process consists of four distinct stages, which adds considerable complexity to the pipeline and may hinder scalability and ease of adoption in practical settings.

### Questions
How does a simple baseline, such as uniform sampling, perform in comparison? For instance, when FrameOracle reduces the number of frames from 16 to 10.4 on average, how does uniform sampling of 10 frames from the original 16-frame sequence compare in terms of performance?

### Soundness
2

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
2

### Summary
Authors propose FrameOracle, a lightweight, plug-and-play selector for video-VLMs that predicts both which frames are relevant to a given query and how many frames are actually needed, tackling the inefficiency of uniform or fixed-budget sampling. It’s trained via a four-stage curriculum that begins with weak proxy signals and culminates in supervised fine-tuning on a new dataset, FrameOracle-41K, which supplies keyframe annotations specifying the minimal sufficient frames per question. Across five VLMs and six benchmarks, FrameOracle cuts 16-frame inputs to ~10.4 with no accuracy loss and trims 64 candidates to ~13.9 while improving accuracy by ~1.4%.

### Strengths
- Performance Gains

FrameOracle reduces frame usage while maintaining or improving accuracy across six diverse benchmarks and five different video-language models


- Plug-and-Play Generalization without Co-Training

Unlike most keyframe selection methods, FrameOracle operates independently of the base VLM, requiring no co-training or model-specific tuning — showing strong transferability and making it highly practical for real-world deployment

- A Novel Dataset (FrameOracle-41K)

The paper contributes a large, purpose-built dataset with keyframe supervision, enabling both training for adaptive frame selection

### Weaknesses
- Marginal Gains at Larger Compute Budgets

When starting from a large candidate pool (e.g., 64 frames), efficiency gains diminish, achieving only modest FLOP and latency reductions, which limits its benefit for already optimized pipelines

- Performance on Fine-Grained Temporal Tasks

FrameOracle underperforms heuristic methods like KFC on datasets such as MLVU, which require precise temporal reasoning and multi-event grounding.

### Questions
Can you show more examples from the dataset itself.

### Soundness
3

### Presentation
2

### Contribution
3
