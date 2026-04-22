# RoboOmni: Actions Are Just Another Modality for Your Vision-Language Models

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 4, 8

## Abstract
Integrating Vision--Language-Models (VLMs) into robotics has enabled building generalizable Vision-Language Action (VLA) models for robotic manipulation. While decoupled designs with a separate action expert often outperform unified frameworks, the latter (e.g., OpenVLA) present an appealing, conceptually integrated architecture. Nevertheless, current unified approaches typically suffer from poor historical context integration and distribution shift given their incapability of predicting action chunking.

We introduce **RoboOmni**, a unified multi-modal next-token prediction framework for robotic manipulation designed to overcome these issues. Compared with decoupled approaches, **RoboOmni** unifies the multi-modal representations and minimizes the distribution gap between vision-language pretraining and action finetuning. Besides, in contrast to prior unified approaches, **RoboOmni** brings in the action chunking mechanism by *Multi-Token Action Prediction* (MTAP) that supports both FAST and Bin tokenizers, and crucially alleviates the action distribution shift issue when training with noisy real-world data. Specifically, by preserving the original VLM training pipeline, **RoboOmni** naturally supports co-training with multi-modal information and various VLM optimization techniques, *e.g.,* fast inference optimization, which significantly improves the generalization capabilities and extensibility of **RoboOmni**.

We conduct extensive experiments on both the CALVIN benchmark and a real-world robot, demonstrating state-of-the-art (SOTA) performance. Our MTAP implementation with the FAST tokenizer achieves a 94.4% average success rate on CALVIN. Furthermore, we show that our Bin tokenizer implementation, deployed with existing VLM serving frameworks like SGLang, achieves a 27x inference time speedup compared with OpenVLA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an alternative architecture called RoboOmni for building VLA models which aims to stay closer to the original training paradigm of vision-language models. This is achieved in multiple ways. Firstly, actions are treated as an additional modality and are interleaved with a history of task descriptions, camera frames and proprioception. In comparison to classical VLA architectures, this changes the input space to incorporate historical information and optimized VLM serving pipelines are used to remain efficient at inference time. Secondly, instead of training a separate action head that predicts action chunks, the authors utilize existing multi-token prediction methods to predict multiple tokens with minimal modifications to the VLM.

### Strengths
- The paper challenges some fundamental assumptions about how VLA models should be built and proposes a novel architecture which carefully handles problems such as inference speed with historical information, action tokenization and action chunking in an elegant way.
- The method utilizes various auxiliary tasks for pretraining such as VQA, visual grounding and trace prediction within a single multimodal autoregressive model, which demonstrates the versatility of RoboOmni.
- RoboOmni demonstrates strong results on CALVIN in both ABCD and ABC settings when compared to a wide variety of state of the art baselines.
- While not novel in itself, the multi-token action prediction mechanism effectively speeds up inference without the need of diffusion/flow matching or bidirectional attention.

### Weaknesses
- The presentation of the manuscript could be improved:
	- The bottom and top right subfigures of Figure 1 are hard to parse as the fonts are too small.
	- The notation in section 3.1 can be confusing. It's not clear if the hidden state $z_k$ corresponds to the last token of the final transformer layer for each replica or something else.
	- The real robot experiment should be part of the main paper in my opinion.
- On the real world experiment, the presented baselines are different from the CALVIN benchmark. For example, pi0-FAST, which is the most competitive baseline on CALVIN is not tested in this setting. Also, the baselines which do end up featured such as OpenVLA and Octo are outdated in terms of performance compared to e.g. OpenVLA-OFT.
- Similarly, the ABC experiment in CALVIN does not include pi0-FAST. This makes it difficult to compare as, overall, the most competitive baseline is then only tested in a single experiment.
- The overall inference speed of the model is slower than pi0-FAST while only yielding small improvements in terms of performance.

### Questions
- You mention that pi0-FAST is trained on a similar data mixture as RoboOmni. How did you incorporate auxiliary task pretraining into pi0-FAST?
- As a follow-up, I am uncertain about the comparison of RoboOmni with pi0-FAST. The ablation in Table 4 with regards to training strategies shows that each strategy can give a competitive advantage over pi0. However, could one not use similar strategies for pi0 and achieve similar results?
- Why are different MTAP strategies used for bin and FAST tokenizers? Are they chosen to achieve a good balance of speed and performance, as the latter strategy is more expressive at the cost of parallelism?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method that unifies data from multiple modalities and optimizes them within a shared language space.

### Strengths
1. In terms of results, it indeed outperforms some previous methods, with particularly impressive performance in the pure bin case.

2. Very solid real-world experiments, which are quite impressive.

### Weaknesses
1. The evaluation is conducted only on the CALVIN benchmark, which is not very convincing, as the tasks in this benchmark involve relatively simple action types. Therefore, even simple bin-based actions can perform well. More challenging cases are needed to further validate the approach.

2. The authors’ main motivation seems to lie in exploring the potential effects of the data composition method. However, many existing studies have already investigated similar approaches, so from a motivational perspective, the novelty is somewhat limited.

3. It is unclear why the authors compared the performance with Pi0 in simulation but did not include this comparison in real-world experiments — this is somewhat problematic.

### Questions
Please refer to the weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a training framework for VLA which treats actions as another modality in multi-modal language models called RoboOmni. RoboOmni tokenizes actions to leverage next-token prediction and large VLM training/fine tuning pipelines while enabling action chunking by leveraging multi-token prediction similar to (Gloeckle et al., 2024) while being compatible with different action tokenization strategies including binning and FAST tokenization. By treating actions as another modality, RoboOmni enables reusing VLM training including co-training with multi-modal data and inference pipelines for very efficient training and inference. The paper shows that RoboOmni outperforms other state-of-the-art VLAs which use decoupled policies for action prediction.

### Strengths
1. The paper presents a nice framework that enables action chunking in VLAs without resorting to decoupled policy heads as common in literature. 
2. Since actions are treated as just another modality in this work, RoboOmni is able to leverage the progress in VLMs in both training and inference pipelines.
3. The paper presents extensive results on CALVIN benchmark and real world tasks along with detailed ablations showing the relative performance gain for each component.

### Weaknesses
1. Why not train on OXE?\
The whole point of the VLA literature is generalization to new tasks and robots. It would have been great to see if leveraging the multi-modal co-training and treating actions as a first-class modality in the training regime instead of as an afterthought in the form of decoupled action policies leads to better generalization to new tasks.


2. Missing results on visual tasks.\
Since RoboOmni has the same “interface” for both text and action prediction, it would be nice to see how well it retains the visual knowledge learnt during the pre-training phase and how it affects the performance on downstream robotics tasks. 



3. Is there a good reason for not comparing against Pi0/Pi0.5 for the real world tasks?

### Questions
Please see weaknesses for questions / suggestions. The paper does a nice job of identifying a very specific problem in the literature and solving it. I expect it to be very relevant to the VLA community and therefore lean towards accepting the paper.

### Soundness
4

### Presentation
4

### Contribution
3
