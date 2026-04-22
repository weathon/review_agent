# Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling

- Avg Score: 7.33
- Decision: Accept (Oral)
- Scores: 6, 8, 8

## Abstract
We introduce Latent Particle World Model (LPWM), a self-supervised object-centric world model scaled to real-world multi-object datasets and applicable in decision-making. LPWM autonomously discovers keypoints, bounding boxes, and object masks directly from video data, enabling it to learn rich scene decompositions without supervision. Our architecture is trained end-to-end purely from videos and supports flexible conditioning on actions, language, and image goals. LPWM models stochastic particle dynamics via a novel latent action module and achieves state-of-the-art results on diverse real-world and synthetic datasets. Beyond stochastic video modeling, LPWM is readily applicable to decision-making, including goal-conditioned imitation learning, as we demonstrate in the paper. Code, data, pre-trained models and video rollouts are available: https://taldatech.github.io/lpwm-web

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel particle based architecture for learning world models, action-conditioned video generation models. In addition to previous Particle based video models, the authors propose to combine these with per-particle latent actions, adding additional flexibility to the model.

Reviewer positionality: while I have co-authored related work, I do not normally review vision-centric papers, but RL papers. This will bias my review and I invite the authors to correct me on field-specific standards.

### Strengths
The authors thoroughly motivate their work and propose a reasonable addition to particle-based generative dynamics models.

The authors thoroughly ablate the base model changes as well as the proposed per-latent particle, which makes the evaluation of the method admirably robust. I have some minor questions on some of the scores, which I believe are due to my unfamiliarity with the datasets.

While the writing could be improved (see below) the overall method is straightforward and clear, and the

### Weaknesses
The gains on the vision datasets seem very marginal, and not significant at a reasonable confidence interval. Provided confidence intervals seem to overlap, that makes the results very hard to judge.

No information is provided on how confidence intervals (the +/- numbers in the tables) were computed. This makes the previous issue more problematic to assess.

The robotics experiments do not seem to compare planning with other latent action methods with the proposed method. That limits the clarity and scientific impact of the comparison. Can the authors provide some insights into the performance of e.g. DDLP on this task?

The robotics experiments are limited in scope and not performed on benchmarks where object interaction is particularly crucial, or where objects have complex dynamics. As one of the main motivation for learning latent actions is decision making and imitation, I think expanding the experimental section here and adding more details on the setup would greatly increase the relevance of this paper to the community. For example, while the method performs well on tasks 1 and 3, it completely fails on task 2, but what this signifies is not discussed. Furthermore, results presented in the main paper are cherry picked without justification, which I consider very bad practice.

Please specify the exact tasks chosen in OG-Bench for ease of comparison with other works.

Table 7 compares the changes to DLP introduced in this paper. The proposed method (evaluated in Table 8 on the same dataset) seems to strongly underperform this baseline, yet this is not mentioned in the ablations. I assume this is because one is tested as an image model and one as a video model, but without this context the results are confusing.

Another issue with the paper is that it only formally fulfills the criteria of being 9 pages long. The method is de-facto impossible to fully comprehend and re-implement with the main body of the paper, and so the appendix becomes essentially a part of the main paper. I would strongly advise the authors to revise and tighten the writing in the main paper to move important architectural details to there. For example, the introduction is quite lengthy, with a slightly superfluous aside on the human visual system, and dominated by a massive image, and some details on the datasets could easily be moved to the appendix. This is not a reason for me to recommend rejection, but it would probably improve the paper to tighten the writing.

### Questions
How are causal actions handled in the framework. I am not quite able to tell whether the dynamics are conditioned on all particle actions or not. If they aren't how are causal relations (a robot moves a block) modelled, and if they are, how is causal confusion resolved? Is the assumption that enough data will solve this issue? 

Is the number of particles architecturally fixed? If yes, what is the principle advantage of particles over slots?

Can known action information (e.g. robotics datasets where action information is readily available) be used to improve and ground latent action learning. In the imitation experiments an action mapping is learned post-hoc, but for applied decision making researchers (robotics, RL, etc.) feeding available action information in at training time would greatly improve the applicability of the method.

Can pre-trained segmentation or feature learning models such as SAM, Dino, or V-JEPA be used to further improve the method or replace some components, to reduce the load of training on large scale video datasets.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper tackles the problem of conditional world modelling with latent action discovery. The main contribution of the work is the extension of the DDLP approach to handle per-particle action latents. This gives the model more flexibility in modelling complex scenes and shows SoTA video modelling results.

### Strengths
* Introducing per-particle action latent, rather than considering a single global action latent (as done in past works) is very sensible, especially for complex datasets, as it introduces an additional degree of freedom and increases the representational power of the model.
* The idea of the context module that combines external conditioning with implicit actions is sound and effective. This provides a universal way to implement action conditioning.
* I find the interplay between the inverse dynamics and the latent policy modules very elegant. The learned dynamics learns the latent action space, while the policy learns the prior, and at the same time regularizes the training of the former module.
* SoTA results on video modelling

### Weaknesses
* The novelty is somewhat limited - the authors extend the previous work by intruding per-particle action latent. Nevertheless, it has shown improved results.
* I feel like the choice to keep all M particles (and not to perform filtering to avoid tracking) sacrifices the ability to separate real objects from “empty” slots, and thus sacrificing interpretability and explicit object modelling - which in my opinion is a nice property of the original particle models.

### Questions
What would it take to scale this method to work on any dataset zero-shot?

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
This paper introduces Latent Particle World Model (LPWM), an extension of Deep Latent Particles (DLP) for self-supervised object-centric world modeling in videos. LPWM decomposes scenes into latent particles (keypoints with attributes like position, scale, depth, transparency, and features) and models stochastic dynamics using a novel CONTEXT module that infers per-particle latent actions via inverse dynamics and a learned policy prior. The model is trained end-to-end as a temporal VAE, supporting conditioning on actions, language, images, and multi-view inputs. Evaluations focus on video prediction/generation across synthetic and real-world datasets (e.g., OBJ3D, BAIR, Mario), where it outperforms baselines in metrics like LPIPS and FVD. Additionally, LPWM is applied to goal-conditioned imitation learning on PandaPush and OGBench-Scene, showing competitive results.

### Strengths
1. The idea of incorporating object-centric methods into world model is well-motivated. The proposed method of flexible and supports diverse conditioning (actions, language, goals, multi-view), which is practical. 

2. The authors provided extensive experimental results to show the effectiveness of the proposed LPWM. Results on real-world datasets (e.g., BAIR, Bridge) highlight its robustness beyond simulated environments.
3. LPWM achieves SOTA on stochastic video prediction (e.g., FVD of 85.45 on Sketchy vs. baselines) and competitive imitation learning performance (e.g., outperforming HIQL on OGBench task3). Ablations validate key design choices (per-particle vs. global actions, latent dimensions). The compact model size and efficiency (e.g., matching larger diffusion models on BAIR-64) underscore the benefits of inductive biases over pure scaling.

4. Particle decompositions (keypoints, masks, bounding boxes) provide inherent explainability, aligning with neuroscience inspirations (e.g., "what-where" pathways), which could appeal to applications in robotics or microscopy.

5. The supplementary material of this work is sufficient and comprehensive.

### Weaknesses
1. The contributions are somewhat incremental, as the authors extend an existing video prediction method (DDLP) by introducing a context module for additional conditioning. However, this should not warrant rejection, given the extensive experiments demonstrating the effectiveness of these improvements.

2. The experiments could be strengthened. While the evaluations focus primarily on video prediction, the world model is intended for policy training. Comparisons with other world models (e.g., the Dreamer series and variants) in terms of sample or learning efficiency would be valuable.

3. There are also concerns regarding the hyperparameters. The model uses diverse hyperparameters for different tasks, as detailed in Table 5. This may undermine the method's generality and raise questions about hyperparameter selection. Additional ablation studies on hyperparameter sensitivity would be beneficial.

### Questions
1. The paper extends DDLP via the CONTEXT module for additional conditioning (actions, language, images). Could you explain how this theoretically or practically surpasses limitations of prior object-centric video prediction methods? 

2. Experiments focus on video prediction, yet world models aim to support decision-making (e.g., policy training). Could you compare LPWM to the Dreamer series or other world models on sample or learning efficiency? Alternatively, justify why the current evaluation suffices for decision-making potential. 

3. Table 5 shows varied hyperparameters across tasks, potentially limiting generality. Could you explain their selection and provide ablation results on sensitivity (e.g., to latent action dimension or learning rate)?

### Soundness
3

### Presentation
4

### Contribution
3
