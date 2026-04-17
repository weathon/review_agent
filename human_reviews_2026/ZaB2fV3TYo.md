# Grounding Bodily Awareness in Visual Representations for Efficient Policy Learning

- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Learning effective visual representations for robotic manipulation remains a fundamental challenge due to the complex body dynamics involved in action execution. In this paper, we study how visual representations that carry body-relevant cues can enable efficient policy learning for downstream robotic manipulation tasks. We present $\textbf{I}$nter-token $\textbf{Con}$trast ($\textbf{ICon}$), a contrastive learning method applied to the token-level representations of Vision Transformers (ViTs). ICon enforces a separation in the feature space between agent-specific and environment-specific tokens, resulting in agent-centric visual representations that embed body-specific inductive biases. This framework can be seamlessly integrated into end-to-end policy learning by incorporating the contrastive loss as an auxiliary objective. Our experiments show that ICon not only improves policy performance across various manipulation tasks but also facilitates policy transfer across different robots. The project website: https://anonymous.4open.science/w/ICon/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes an auxiliary objective for end-to-end learning of visual policies which employ a vision-transformer image encoder. The authors take a contrastive learning approach for encouraging agent-environment disentanglement in the learned ViT features. The objective leverages a segmentation mask of the agent (acquired with SAM) to classify ViT patches to either agent or environment, and employs an inter-token contrastive loss based on these classes. The method is evaluated on imitation learning by training visual diffusion policies on simulated robotic manipulation environments.

### Strengths
**Overview**
- Well written paper.
- Method seems novel.
- Potentially applicable to any policy that employs a ViT image encoder.
- Evaluated on numerous environments.

I am willing to raise my score if the points raised in the Weaknesses and Questions sections are addressed.

### Weaknesses
**Overview**
- The specific method is not well-motivated other than the agent-environment disentanglement.
- No comparison with other segmentation-based representation learning objectives.
- Information loss in produced token masks compared to the original pixel mask.
- Empirical performance gains are not significant.

**Method Motivation**

Why is your specific approach better than others for acquiring agent-environment disentanglement in feature space? This is neither discussed nor empirically evaluated in this work. You cited many papers that apply the same agent-centric principles, could you provide comparisons with one or more of these methods?

Additionally, I think it would be beneficial to perform the following ablations to motivate your contrastive approach:
1. Replacing the agent-mask-based contrastive objective with an agent-mask-based reconstruction objective. (Correct me if I am mistaken, but you are currently only comparing with a full-scene reconstruction baseline).
2. Replacing the contrastive objective with a simple inductive bias: applying a learned additive embedding to the patch features based on their association to environment vs. agent (total of 2 learned embeddings). This suggestion is based on the same principles as your contrastive approach---distinguishing between agent and environment features. If this is indeed beneficial for downstream policy performance, maybe it will be learned end-to-end, i.e., the embeddings will converge to be dissimilar?

**Token Masks**

The token mask threshold results in a coarser segmentation than the original pixel-based segmentation. Why is this preferred over, e.g., processing the masked image with two parallel ViT layers, once with the agent masked out and once with the environment masked out? The rest of the pipeline can stay the same after this initial layer by combining the tokens from both images such that they also attend to each other in the following layers.

**Experiments**

4/5 RLBench, 2/3 Robosuite and 1/2 transfer tasks are within a standard deviation from the base diffusion policy. It is standard to highlight in bold results that are within a standard deviation from the best performing method for clarity. Maybe evaluating with a larger number of seeds will help distinguish the performance gain. Currently, the performance gains look marginal at best and do not justify the additional method complexity.

### Questions
- Hyperparameters should be detailed in the Appendix. Specifically, what are the values of gamma and lambda? Do they have to be tuned per-task or is the algorithm robust to these hyperparameters?
- How robust is the method with respect to hyperparameters in terms of the training stability evaluated in Section 4.5?
- Figure 6: Why does the CLS token specifically attend only to the agent? What do the other tokens attend to?

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
4

### Summary
The paper presents a (class-supervised) contrastive representation learning approach of tokens from a Vision Transformer, termed Inter-Token Contrast (ICon), for the purpose of robotic manipulation. The core idea is to segment-out tokens relevant to the agent (e.g., the robotic gripper) and contrast them with tokens representing everything else. The method is implemented within a diffusion-based policy, and demonstrated on various tasks and robots from two simulated benchmarks.

### Strengths
* The utilization of FPS over 2D feature/token maps to ensure coverage is interesting.
* I appreciate the stability analysis.
* Nice video visualizations on the project website.
* Open-source data and code!

### Weaknesses
* The method relies on a pre-trained supervised segmentation model. While it is true that these models are performing very well, they are still prone to wrong detection or missing objects for unseen data.
* The masking procedure relies on heuristic (class thresholding) where it is unclear what would happen if part of the gripper and a small object (e.g., a small block) occupy the same tokens.
* I found the multi-level contrastive loss description in L246-251 confusing: I don’t understand how $\gamma$ is tuned per-layer and I also don’t understand the reasoning for setting $\gamma >0$ for deeper and shallower layers. Given the reason provided in the paragraph, I would expect something like “for shallower layers $\gamma$ is larger/smaller than deeper layers as the earlier layers produce more entangled representations”.
* The computational cost is unclear. For training, masks are pre-computed (I assume this is because the segmentation model, SAM, is large), FPS can also be time-consuming, and memory-wise, the ViT encoder needs to be probed at several layers, saving the features for the contrastive loss. What about inference time? Are observations masked on the fly?
* Overall, except for 3 tasks, it seems that the improvement is marginal compared to the baselines, and for some tasks all methods fall short. While I appreciate the authors chose hard benchmarks, given my assumed computational cost, it is hard to get convinced regarding the contribution (and given that masking has already been proven to be useful in decision making, also referenced in the related work section).

### Questions
* ViT: it is unclear from the text if the ViT encoder is pre-trained, fine-tuned or trained from scratch. Can you please clarify this?
* Regarding the masking procedure concern I raised under Weaknesses, can the authors please clarify what would happen in multi-object environments where objects and gripper share tokens?
* Can the authors please clarify my concern regarding the computational cost under Weaknesses?
* Are policies multi-task or trained per-task? Are they goal-conditioned?
* The authors mention they use action chunking (“Temporal Ensemble”), to my understanding, and that it was critical for the performance. Can you clarify this and what is the chunk size used in practice?
* I’m not sure I understand this sentence in the conclusion (L469-470) “restricts its applicability to other commonly used visual encoder architectures in visuomotor policy learning, such as ResNet”. Why is that? Basically, it is possible to extract 2D features maps (that represent the regions in the corresponding input image) and apply downsampled masks over them. What prohibits you from applying the contrastive loss on these features? Or do you mean the downstream processing of the post-contrasted features?
* Have the authors experimented with other types of self-supervised loss, perhaps ones that do not require negative samples (e.g., BYOL, [Grill, Jean-Bastien, et al. "Bootstrap your own latent-a new approach to self-supervised learning." Advances in neural information processing systems 33 (2020): 21271-21284.](https://arxiv.org/abs/2006.07733))

Overall, I raised several concerns regarding the method as detailed above. I’m willing to increase my score given convincing answers and clarifications to my review.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces ICon, a simple contrastive objective that encourages ViTs to disentangle representations of the agent’s body from the environment in visuomotor RL. Using pre-extracted masks of the agent's body, the authors use farthest-point sampling to sample tokens which correspond to the agent's body and to the environment and appliy an InfoNCE loss, as an auxiliary loss for Diffusion Policy. The paper is clearly written and the approach to solve the problem is technically sound, although the usage of hard positive/negatives of agent/environment in the contrastive learning part might introduce some problems. 
The experiments of the authors on RLBench and Robosuite do show that ICon leads to improvements on downstream tasks and some cross-robot transfer. However, the experimental section could be strengthened to better showcase the strenght of the learned representations.

### Strengths
1) The approach to disentangle agent-specific and environment specific features is technically sound. The FPS also helps obtain more useful representations w.r.t. random sampling. 
2) The improvement in success rates for a few tasks in Robosuite and RLBench seems to be consistent. 
3) With little finetuning the method seems to easily transfer across different robots although it is not clear how much tuning is needed and how well the method would perform zero-shot.

### Weaknesses
* The benchmarks used to validate the method are limited. Training concurrently ViT with the policy also seems to limit the ability to do some pretraining. I believe it would be useful if the objective was also used not only concurrently with the policy, but with a large pretraining setup that can later be used out of the box for variosu downsteram tasks without tuning.

* How much fine-tuning is needed ? Also would it work out of the box ? I believe that at its current state since we train on one robot at each time the results do not generalize directly.

* The method as the authors state is tied to ViTs and needs agent masks. This is however not a major weakness since ViT are currently the standard go-to for visual representations.

* The gains are consistent but also are small. It would be nice also if the authors included more baselines with popular pretrained encoders (such as CLIP). 

* The training stability and ablation studies are all done in different environment at each time. This is weird - please use a single unified framework for doing your ablation studies and stability evaluation. 

* The overhead of using FPS is not quantified.

### Questions
1) I believe that the current direction of the authors is useful and intuitive and it would make sense to help robotic agents disentangle features from their body and the environment. In my opining the object the authors propose would be also very useful in pretraining of ViTs to use in downstream tasks afterwards. This can allow to train on multiple robotic agents and generalize out of the box to new ones.

2) What is the overhead of using FPS ? How would it compare with dense sampling of tokens ? Also, how would it compare with some simple uniform grid sampling ?

3) Please use a common unified ablation framework.

4) How sensitive is your method in the mask extracted for the agent. Have you done any studies on this ? 

5) One major question and a problem with contrastive learning is the usage of hard negatives. At its current state the positive=agent, negative=environment pair is a bit blunt and can worsen some representations since some features from the environment might be similar with the agents (e.g. texture or color). Have the authors consider an alternative to contrastive learning ? Would it be better maybe to force some features and not all features in the feature vector of the token to be distant and not all at once ?

### Soundness
2

### Presentation
3

### Contribution
2
