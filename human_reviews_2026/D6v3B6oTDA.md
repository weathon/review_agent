# Video-STR: Reinforcing MLLMs in Video Spatio-Temporal Reasoning with Relation Graph

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Recent progress in Multimodal Large Language Models (MLLMs) has demonstrated strong semantic understanding capabilities, but struggles to perform precise spatio-temporal understanding. Existing spatio-temporal methods primarily focus on the video itself, while overlooking the physical information within the video, such as multi-object layouts and motion. Such limitations restrict the use of MLLMs in downstream applications that demand high precision, including embodied intelligence and VR. To address this issue, we present Video-STR, a novel graph-based reinforcement method for precise Video Spatio-Temporal Reasoning. Building upon the capacity of Reinforcement Learning with Verifiable Reward (RLVR) to improve model abilities, we introduce a reasoning mechanism using graph representation based on Group Relative Policy Optimization (GRPO) to guide the model in inferring the underlying spatio-temporal topology of scenarios during the thinking process.To resolve the lack of spatio-temporal training data, we construct the STV-205k dataset with 205k question-answering pairs, covering dynamic multi-object scenes in both indoor and outdoor environments, to support the model training. Experiments show that Video-STR achieves state-of-the-art results on various benchmarks, outperforming the base model by 13% on STI-Bench, and demonstrating the effectiveness of our approach and dataset. Code, model, and data will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper describes a dataset consisting of Question-Answer pairs about object-based tasks (such as counting, discances, movements, etc.) that were automatically generated from existing video dataset (TAO, ScanNet and KITTI). It also describes a reward design for RL-based training, based on the positions of objects as well as their positions relative to one another. It then shows that together, these two ingredients make it possible to fine-tune Qwen2.5-VL-7B to achieve relatively high recognition performance on several video understanding benchmarks (STI-bench, V-STaR, etc.). Ablations furthermore show that all the ingredients contribute positively to performance improvements on these benchmarks.

### Strengths
The paper is quite easy to follow. It describes an (automatically generated) dataset, a training method and a model (based on fine-tuning Qwen2.5-VL-7B), all of which will be publicly released. The model yields fairly good performance on some video understanding benchmarks, specifically those that test for an understanding of object positions and their relations.

### Weaknesses
I have the concern that the object-centric data and reward designs are benefitting specifically tasks that measure the ability to understand object positions and relations, without enhancing the model’s visual understanding overall. The results suggest that this is indeed the case, as model performance on tasks like STI-bench and V-STaR is comparably high, while this is not nearly as much the case for tasks like Video-MME (where the model is in fact significantly behind the state of the art). 

Overall, this paper describes an engineering effort, based on automatically generated data and RL-training with object-centric rewards, to mildly improve performance on video-understanding benchmarks (and most significantly on object-centric benchmarks). I am not quite sure how this advances our understanding of visual capabilities, models and model limitations overall. I am also missing a view of the limitations of the object-centric design. For example, are there any tasks where performance degrades rather than benefits from this design?

### Questions
After fine-tuning the model on graph-based spatio-temporal reasoning, how is it applied to the downstream benchmark tasks? Is there additional fine-tuning involved? Any task-specific (few-shot?) prompting?  

The work is reminiscent of older works on graph-based grounding, such as “Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations”, Krishna et al 2016, and many related and follow-up papers). It would be good to see a discussion on how it is related to this line of research. 

(small comment:) The frames in the figures (for example, Figure 1, Figure 2) are impossible to see properly or understand in a print-out version of the paper.

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
This paper presents a graph-based reinforcement learning method for Video Spatio-Temporal Reasoning which models inter-object relationships using relation graphs. It introduces a graph-based reasoning mechanism in the Group Relative Policy Optimization (GRPO) [Shao et al. 2024] for inferring the underlying spatio-temporal topology of scenarios during the thinking process. The paper proposes STV-205k dataset with 205k question-answering pairs for model training and reports SOTA performance on various benchmarks including STIBench.

### Strengths
The motivation and problem definition are valid. 

The graph based reasoning approach is interesting and seems novel, although I am not aware of all the research in the area of video reasoning. Nevertheless, a graph-based approach appears to be a viable direction for video reasoning. 

The idea of generating a QA dataset (STV-205K) from existing datasets is innovative and useful. 

Reported experimental results are promising.

### Weaknesses
My biggest concerns include limited contribution, lack of clarity and the possible unfair comparison in experiments. Details below. 

Graph-based spatio-temporal reasoning is not a new approach and has been done previously. This work appears to be a combination of existing techniques with limited methodological novelty.  

The very first sentence of the abstract is grammatically incorrect. Overall, the writing and presentation of the paper can certainly be improved as I find it quite confusing. Which “model” is being referred to at each place? And how distances are calculated in images? How do you get Euclidean distances from images/videos? Are they in pixels and if so, how meaningful are pixel based distances in a graph?  

What is X in Eqn.(8)? It should be defined after the equation and not only in the Appendix. Going to the proof in the Appendix, X seems to be the object locations and not distances and locations should not be rotation invariant. On the other hand, graph edge weights being Euclidean distances will remain rotation invariant. Also, images/video frames are 2D images, why is the rotation R \in SO(3)? 

Also, I did not notice the proof of Theorem 2 in the Appendix (maybe I overlooked but there were no numbers with Theorem proof in the Appendix). I am also not sure if these two theorems are worth mentioning at all. 

At this point, I should also highlight the ambiguity in the proposed dataset description. It is not explicitly mentioned which data modality and labels are obtained from the three datasets (TAO, ScanNet and KITTI) e.g. ScanNet and KITTI also have depth images/point clouds. There is no mention of 3D, depth or point cloud in the paper so I assume the paper is about conventional video reasoning but then “Euclidean distance” is mentioned at many places which is confusing. Yes, the Euclidean distances can be in pixel units but that would be meaningless in a scene graph e.g. the chair is two pixels to the right of the table. 

Referring to “The model is prompted to generate a graph” on page 5, which model is prompted? The following sentence is also confusing where “model” is used twice. Are they both the same model or two different models? 

On page 2 it is mentioned that “we extend the Group Relative Policy Optimization (GPRO)” but in Section 4.3, it is not clearly mentioned how GPRO is extended in this paper. 

The results in Table 1 look very good but I am wondering if the improvement is mainly due to the additional training data from newly proposed STV-205k dataset. Since the other methods did not use this additional dataset, the comparison is unfair. 

Page 7: “SFT achieves localized improvements”. What does “localized improvement” mean? 

In Fig.4(b), the improvement on “Static” sub-task is quite substantial compared to “Temporal”. This makes me wonder how the graph encodes temporal information. 

In Sec. 5.3, “Video-260k”, “SR-91k” are mentioned, but no reference or any further information is given about these datasets. 

Overall, the proposed method may have merit but that is obscured by unclear presentation. Moreover, unfair comparison with existing works also undermines its credability.

### Questions
Please see weaknesses above.

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
The paper introduces Video-STR, using graph-based reasoning and reinforcement learning to improve spatio-temporal understanding in video MLLMs. The authors build a 205k QA dataset and extend GRPO with graph rewards. Results are promising (13% gain on STI-Bench), but the work has several technical gaps and unclear design choices that weaken the contribution.

### Strengths
- The graph representation for multi-object scenes makes sense and has nice rotation-invariance properties
- The benchmarking across 6 different datasets shows the approach generalizes reasonably well

### Weaknesses
- Missing critical implementation details. For example: How do you actually detect and track objects to build these graphs?; During inference, does the model predict object locations itself, or do you use an external detector?
- 205k automatically generated QA pairs with no human verification is risky. Template-based generation could introduce systematic biases
- The paper says the model is "prompted to generate such a graph" but doesn't show how. Is it predicting coordinates in structured format? 
- Which reward components actually matter? No ablation on R_format, R_n, R_e, Rl_individually
- GRPO generates 8 responses per sample - what's the actual training time and memory cost?

### Questions
Please answer the questions in the weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose Video-STR, a graph-based RL framework to model the fine-grained interactions in video, enabling advanced reasoning in scenarios with heavy physical information. To facilitate the model training, they collect a large-scale training dataset, STV-205K. The experimental results demonstrate the effectiveness of the training data, training strategy (graph-based GRPO). Video-STR outperforms existing video reasoning models on most benchmarks.

### Strengths
1. The paper is well-written and has a good organization.
2. The proposed Video-STR is reasonable and can truly process complicated video scenarios.
3. The STV-205K is carefully constructed and well aligned with the proposed framework.
4. The experiments are comprehensive, and the ablations are solid.

### Weaknesses
This paper is of good quality, and I just have a question about the setting of the dataset. It looks like there are still mostly recognition-type data in the dataset, not real reasoning data. From line 173-190. However, the design of the graph can be further explored, for example, some data like the intention of a motion.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
