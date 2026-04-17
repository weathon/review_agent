# ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning

- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Video understanding is inherently intention-driven—humans naturally focus on relevant frames based on their goals. Recent advancements in multimodal large language models (MLLMs) have enabled flexible query-driven reasoning; however, video-based frameworks like Video Chain-of-Thought lack direct training signals to effectively identify relevant frames. Current approaches often rely on heuristic methods or pseudo-label supervised annotations, which are both costly and limited in scalability across diverse scenarios. To overcome these challenges, we introduce ViaRL, the first framework to leverage rule-based reinforcement learning (RL) for optimizing frame selection in intention-driven video understanding. An iterated amplification strategy is adopted to perform alternating cyclic training in the video CoT system, where each component undergoes iterative cycles of refinement to improve its capabilities. ViaRL utilizes the answer accuracy of a downstream model as a reward signal to train a frame selector through trial-and-error, eliminating the need for expensive annotations while closely aligning with human-like learning processes. Comprehensive experiments across multiple benchmarks, including VideoMME, LVBench, and MLVU, demonstrate that ViaRL consistently delivers superior temporal grounding performance and robust generalization across diverse video understanding tasks, highlighting its effectiveness and scalability. Notably, ViaRL achieves a nearly 15% improvement on Needle QA, a subset of MLVU, which is required to search a specific needle within a long video and regarded as one of the most suitable benchmarks for evaluating temporal grounding.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses the challenge of intention-driven video understanding, focusing on selecting relevant frames in videos to answer queries effectively. The authors propose ViaRL, a novel framework that uses rule-based RL to optimize frame selection by leveraging the answer accuracy of a downstream MLLM as a reward signal. This trial-and-error learning approach eliminates the need for expensive annotations and mimics human-like learning.

### Strengths
The paper presents a novel framework, ViaRL, which leverages rule-based reinforcement learning to optimize frame selection in video understanding tasks. Central to the approach is the Visual Iterated Amplification training strategy, an innovative iterative refinement process that alternates between optimizing the frame selector and the answer model, providing strong motivation and technical soundness. The effectiveness of ViaRL is demonstrated through comprehensive experiments on several challenging benchmarks, including VideoMME, LVBench, and MLVU, where it consistently achieves improvements. Given the difficulty of the tasks and the good results, I believe that reinforcement learning policy and the cyclic training is really helping to reason on the temporal axis so that it can select the most important frames. Very interesting work.

### Weaknesses
The major weaknesses of this work are the following:

1 - It would be nice to see how the method affects other VLMs which are not flexible on the resolution/quality of input data.

2 - While the method performs really well in one of the most challenging problems of video understanding, it lacks comparison in more generic tasks like answer generation (for Q&A and captioning for example) to the the impact of this 'specialization' on other capabilities of the network.

3 - The ablation on cyclic training maybe needs a bit more deepening on the diminishing returns with additional cycles. The authors claim that it is due to the imperfect nature of MLLMs to provide correct answers for each visual scene and the limited info contained in 8 frames. I think these claims might need some more experimenting to see if the limit is on the method or on the MLLMs serving as supervisors.

### Questions
Check above.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces ViaRL (Visual Iterated Amplification Reinforcement Learning), a novel framework to address the challenge of efficient, query-driven frame selection in long videos. The authors argue that direct video reasoning in MLLMs is less effective than first mastering temporal grounding . The framework uses a cyclic, two-stage "Visual Iterated Amplification" training strategy: first, the Selector is improved via RL, and second, the Answer Model is instruction-tuned using the improved Selector's frame selections, creating a feedback loop where both models progressively enhance each other. Experiments show significant gains, especially a nearly 15% improvement on the Needle QA temporal grounding benchmark

### Strengths
- The paper is well-written and easy to follow.
- The training details are explained in detail. It improves the reproducibility of the paper.

### Weaknesses
- Unclear Inference Cost: The paper motivates its approach by citing the high cost of processing all frames. However, the proposed ViaRL framework requires two sequential MLLM forward passes at inference time
- From my perspective, the proposed paper lacks the technical novelty. Compared to existing GRPO-based works, the different part is to introduce frame selection before question answering. However, there have been multiple works that solve question-answering tasks with the frame selection. 
- I'd like to see the contribution of the frame selection trained by reinforcement learning with verified rewards. It would be better if the paper included the performance comparison of the frame selection with other frame selection methods and the impact of them to the downstream tasks.

### Questions
- The current reward system is fundamentally tied to MCQ benchmarks . I wonder if the proposed ViaRL still works well on the open-ended generative tasks such as open-ended QA.

### Soundness
2

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
This paper introduced ViaRL, a framework to use rule-based RL to optimize the frame selection process in video understanding tasks. It is an iteration process is used in the CoT process and ViaRL use the accuracy of downstraming model as the reward signal. In more detail, they first processed a video understanding dataset by using CLIP to sample frames based on visual-textual similarity and then filter out less informative tasks. They using reinforce++ to finetune Qwen model to select frames and using another answer model to provide training signal. Each model is tuned in turn to help each other. 
They demonstrated good performance on popular video understanding benchmark including VideoMME, LVBench, and MLVU.

### Strengths
1. This paper provided a RL framework for temporal grounding without human annotation. By iteratively refine the selector and the answer model on the training set, the performance of Qwen is improved.
2. The paper also provided useful tricks. For example, using idea from existing work to mark the frame index in the frame corner.
3. The author achieved great performance on Qwen model.

### Weaknesses
1. The RL is only tested on Qwen-2.5-VL, therefore it is hard to know if the method could generalize to other models.
2. In the data preparation process, CLIP is used to sample relevant frame to the question, which could be inaccurate. CLIP is measuring the semantic similarity between the frame and the answer, while a frame could be barely connected with the question when it is alone but important when in frames context. I also do not see any experiments supporting this sampling process.
3. Table 2 is never discussed in the paper. For example, why using all components yield the worse performance among the table?
4. The author mentioned several RL rewards but only length reward is analyzed. Then it is not clear why the author use those rewards. Although they are proven useful in other tasks in other paper, it is not clear whether they help in this case.

### Questions
1. In table 1, the bold number only represents higher number between Qwen and Qwen+ViaRL, but the authors did not mention this. The bold results are worse than many open-source MLLMs and Proprietary Models. This is a minor issue so I put it in questions section. Hope the authors could clarify this in the paper.

2. Why choosing Qwen-7B as the answer model? Here I have several questions. 1. After improving, will the answer model achieve sota performance? If not, why not using existing sota model (open sourced or not) to provide the reward?

### Soundness
3

### Presentation
2

### Contribution
3
