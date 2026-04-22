# UniCoD: Enhancing Robot Policy via Unified Continuous and Discrete Representation Learning

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 2, 8, 8

## Abstract
Building generalist robot policies that can handle diverse tasks in open-ended environments is a central challenge in robotics. To leverage knowledge from large-scale pretraining, prior work has typically built generalist policies either on top of vision-language understanding models (VLMs) or generative models. However, both semantic understanding from vision-language pretraining and visual dynamics modeling from visual-generation pretraining are crucial for embodied robots.
Recent unified models of generation and understanding have demonstrated strong capabilities in both comprehension and generation through large-scale pretraining. We posit that robotic policy learning can likewise benefit from the combined strengths of understanding, planning and continuous future representation learning. Building on this insight, we introduce UniCoD, which acquires the ability to dynamically model high-dimensional visual features through pretraining on over 1M internet-scale instructional manipulation videos. Subsequently, UniCoD is fine-tuned on data collected from the robot embodiment, enabling the learning of mappings from predictive representations to action tokens. Extensive experiments show our approach consistently outperforms baseline methods in terms of 9\% and 12\% across both simulation environments and real-world out-of-distribution tasks. Demos and code can be found at \href{https://sites.google.com/view/uni-cod}{our anonymous website}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a VLA called UniCoD. The model is based on the mixture of transformers paradigm trained on both continuous and discrete losses. The model is trained in two stages: the first stage only contains vision-language data and trains the model on CE next-token prediction and  MSE loss. The second stage introduces action data and trains a flow-matching expert, as well as the MSE loss for the generation expert. The authors present good results on SimplerEnv-WindowsX [table 1], calvin [table 2] and on a real-world robot [figures 4,5].

### Strengths
- The authors provide gains on multiple benchmarks and real-world evaluations.
- The ablations in Table 4 clearly show that the MSE loss helps.

### Weaknesses
1. The novelty is limited.
2.  The used baselines are not consistent among tables (e.g, table 2 doesn't have octo.) and standard baselines like Groot models are missing.
3. The writing is not very clear about what the contribution is.

### Questions
1. Why are the baselines not consistent among tables? E.g., table 2 doesn't have octo. Can you add all baselines to all tables? 
2. Table 4 seems to show that pretraining barely helps, it only gives 2 points. Is that correct?
3. Can you clarify what the novelty is?
4. Could you add error bars to your results?
5. Could you ablate the use of discrete predictions too in table 4?

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
5

### Summary
This paper introduces UniCoD, a novel framework for training generalist robot policies by unifying discrete and continuous representation learning. The core challenge it addresses is that existing methods typically rely on either vision-language models (for understanding) or generative models (for dynamics), while both are crucial for robotics. UniCoD learns to simultaneously understand tasks via discrete language representations and model world dynamics by predicting continuous future visual features. Experiments conducted in simulation (SimplerEnv, Calvin) and on two real-world platforms (a 7-DOF arm and a 12-DOF dexterous hand) show that UniCoD achieves state-of-the-art performance. It significantly outperforms baseline methods, demonstrating superior generalization to novel objects and tasks.

### Strengths
The method involves a two-stage training process:
1. Pre-training: The model is first pre-trained on over 1 million internet-scale instructional videos and embodied VQA data to learn these joint representations. It's a heavy work. 
2. Fine-tuning: An action expert is then added, and the model is fine-tuned on robot-specific data, learning to map its predictive representations to action tokens.

### Weaknesses
1. This framework is followed the understanding&generation framework. Please discuss the difference between CoT-VLA, Up-VLA and HybridVLA. 
2. The author need to give more information about the efficiency of the model control, especially when the paper utilize such heavy framework to perform the robot control.
3. I wonder whether the ``pretraining on over 1M internet-scale instructional manipulation videos'' is beneficial for all downstream manipulation tasks. Particularly, please show that the generation quality of the images after the downstream manipulation finetuning.
4. I don't know the meaning of utilizing the such kind of heavy pipeline to perform VLA. The performance improvement is completely unable to offset the overall computational overhead, like compared to CogACT in SimplerEnv-Google Robot Benchmarks.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
UniCoD is a unified multimodal framework that uses a MoT architecture to integrate text understanding, visual prediction, and action execution for robotic manipulation.  

In the first stage, it learns joint vision–language embeddings by aligning textual instructions with visual observations and predicting future visual states in a continuous feature space using a frozen visual encoder. Then, UniCoD fine-tunes this model with embodiment data by introducing an action expert that learns continuous action distributions via flow matching, enabling coherent mapping from multimodal inputs to robot actions.

Results show that UniCod achieves state-of-the-art results across both simulated and real-world environments.

### Strengths
By jointly training on text, vision, and continuous future prediction, UniCoD builds deeply aligned multimodal embeddings that are resilient to noise or missing information in any single modality.

The modular MoT design enables selective fine-tuning (e.g., only the action expert), reducing computational cost and preventing catastrophic forgetting of general skills.

The dual-objective training (cross-entropy for language, MSE for vision, flow matching for actions) helps maintain stable convergence and robust multimodal coordination.

### Weaknesses
The model design with multiple expert modules (for language, vision, generation, and action) requires substantial computational resources and careful coordination. This can make training costly, difficult to reproduce, and potentially unstable without large-scale infrastructure. Are there analysis on the cost for fine-tuning on new tasks and new envs?

How about the interpretability of the model? For example, are there any safety or hazard-preventing modules or self-correcting modules of the model?

### Questions
See weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
UniCoD is a VLA policy that couples discrete language/understanding tokens with continuous predictions of future visual features in a Mixture-of-Transformers. The model is pretrained on ~1M instructional/manipulation videos to predict future visual embeddings and VQA/planning tokens, then fine-tuned with an action expert via flow matching while retaining the future-prediction head. The premise is that policy learning improves when semantic reasoning, planning, and explicit future representations are learned jointly. Reported experiments show state-of-the-art results on SimplerEnv and CALVIN, plus strong performance on two real-robot setups. Ablations (Table 4) suggest that future-feature prediction accounts for a substantial share of the gains.

### Strengths
1. The proposed procedure is intuitive and well-motivated, combining discrete reasoning with continuous visual forecasting.

2. The authors did conduct extensive experimental analyses, including on two real-world robots, with a clearly detailed protocol. The results show that UniCoD achieves the state-of-the-art performance in multiple benchmarks.

3. The paper is clearly written, and the ablation studies performed by the authors are thorough and convincing.

### Weaknesses
1. While the experimental results are detailed, they are only point-estimates with no std or confidence-intervals. The authors should add these, since they did multiple trials. This is a significant weakness.

### Questions
1. Please add statistical analyses of the results for multiple trials.
2. The dataset is very large. Is it guaranteed that unseen objects are not in the training dataset ?

### Soundness
3

### Presentation
4

### Contribution
3
