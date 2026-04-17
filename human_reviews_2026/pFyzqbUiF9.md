# Vid2World: Crafting Video Diffusion Models to Interactive World Models

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4, 6

## Abstract
World models, which predict future transitions from past observation and action sequences, have shown great promise for improving data efficiency in sequential decision-making. However, existing world models often require extensive domain-specific training and still produce low-fidelity, coarse predictions, limiting their usefulness in complex environments. In contrast, video diffusion models trained on large-scale internet data have demonstrated impressive capabilities in generating high-quality videos that capture diverse real-world dynamics. In this work, we present _Vid2World_, a general approach for leveraging and transferring pre-trained video diffusion models into interactive world models. To bridge the gap, Vid2World systematically explores _video diffusion causalization_, reshaping both the architecture and training objective of pre-trained models to enable autoregressive generation. Additionally, it incorporates a _causal action guidance_ mechanism to enhance action controllability in the resulting interactive world models. Extensive experiments across multiple domains, including robot manipulation, 3D game simulation, and open-world navigation, demonstrate that our method offers a scalable and effective pathway for repurposing highly capable video diffusion models into interactive world models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a straightforward yet effective method for adapting a pre-trained video diffusion model into an action-conditioned world model. The core contributions are two-fold: 1) "Video Diffusion Causalization," which enforces temporal causality through a causal masked attention layer and a temporal convolution layer with a novel "extrapolative weight transfer" technique, and 2) "Causal Action Injection," which integrates action-conditioning via causal guidance with dropouts. The authors conduct extensive experiments across diverse domains, including robotic manipulation (RT-1), game simulation (CS:GO), and real-world navigation (NWM), demonstrating that their proposed model achieves higher fidelity in video prediction compared to existing world models.

### Strengths
- Simplicity and Effectiveness: The primary strength of this work lies in its simple and elegant approach. The proposed modifications to transform a standard video diffusion model into a world model are minimal, yet they prove to be highly effective across multiple challenging domains.
- Broad Applicability: The method's effectiveness is convincingly demonstrated on a variety of environments, from simulated robotics and games to real-world navigation. This suggests the proposed techniques are general and widely applicable.
- High-Fidelity Generation: The paper provides strong quantitative and qualitative results, showing that the model can generate high-fidelity, fine-grained, and action-conditioned video sequences that are visually superior to prior work.

### Weaknesses
- Insufficient Motivation for Extrapolative Weight Transfer: While the paper introduces "extrapolative weight transfer" as a key component, its theoretical and empirical motivation remains underdeveloped. The appendix (A.1) presents a simple counter-example involving a linearly increasing kernel, but this theoretical argument is not supported by empirical evidence. To strengthen this claim, the authors should provide experiments that verify the existence and prevalence of such temporal patterns in the learned kernels of baseline models, thereby justifying the need for their proposed solution.

- Lack of Experiments Validating Utility as a World Model: The paper claims to build a "world model," but the experiments primarily focus on video prediction fidelity (e.g., FVD, FID, SSIM). A true world model should be useful for downstream tasks like planning and control. The evaluation is missing crucial experiments, such as: (1) Reinforcement Learning Tasks: Can an RL agent be successfully trained using the learned model as the environment simulator? Demonstrating successful policy learning would provide compelling evidence of the model's utility and temporal consistency. If this is not feasible, a discussion of the model's limitations for such tasks is warranted. (2) Quantitative Interactivity Metrics: The current evaluation lacks metrics that specifically measure the model's fine-grained controllability and causal soundness in response to actions.

- Ambiguity in the Role of Large-Scale Pre-training: The experimental setup creates a potential ambiguity. The proposed model leverages an internet-scale pre-trained video model, whereas the robotics and gaming experiments are conducted on domain-specific datasets. It is unclear which components of the model's success are attributable to the novel architectural changes versus the powerful, pre-trained backbone. An ablation study clarifying the benefits of internet-scale pre-training for domain-specific tasks like RT-1 or CS:GO would be highly valuable.

- Incomplete Reporting of Metrics: For the real-world navigation (NWM) experiments, key fidelity metrics such as FVD, FID, and SSIM appear to be missing from the results tables. Including these would allow for a more direct and fair comparison with other models and experiments within the paper.

### Questions
- Figure 6: This figure presents a crucial comparison of auto-regressive generation capabilities. However, the motivation for why this specific comparison is necessary and what insights it provides is detailed in Appendix C.6 but absent from the main text. Briefly explaining this context in the main paper would significantly improve the figure's clarity and impact.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes Vid2World, a framework that turns a single video into a 4D (3D + time) dynamic scene representation. Unlike typical video diffusion models that just generate future frames, this method tries to reconstruct a consistent, camera-aware world that can be rendered from novel viewpoints over time. Technically, the authors build on top of DynamiCrafter and modify it for causal, autoregressive video generation with action conditioning. They introduce weight-transfer tricks and causal attention masking so that the model can generate frames one by one while maintaining spatial-temporal coherence.

### Strengths
The idea of turning raw videos into a dynamic 4D world is compelling. Most video models only hallucinate future frames from a fixed camera, so explicitly reconstructing a consistent world is an interesting step forward.

I appreciate that they go beyond synthetic datasets and test on RT-1 real robot data and a large-scale human gameplay dataset (CS:GO). It shows they’re not only targeting clean, toy data.

The architectural changes to turn an offline video diffusion model into a causal and action-aware generator are reasonable. Especially, I think the “extrapolative weight transfer” idea for converting bidirectional attention layers into causal ones is practical.

The results (especially novel-view video rollout) are visually impressive and clearly better than standard video diffusion models that ignore camera motion.

### Weaknesses
No true 3D ground-truth evaluation.
The method claims to build a 4D world, but there’s no quantitative evaluation comparing it to 3D scene reconstruction methods (like NeRF, BANMo, DynamicNeRF, etc.). Most results are still evaluated only on video quality metrics or visuals. It's hard to tell if the “world” is actually geometrically consistent or just looks plausible.

Mostly relies on pre-trained models.
A large part of the pipeline inherits DynamiCrafter. The core novelty is in causalization and conditioning, but I sometimes felt like the paper is more of a clever adaptation rather than a fundamentally new representation of dynamic scenes.

Action conditioning is not strongly validated.
They claim the model can take actions and simulate their effects in the generated world, but I didn’t see strong evidence that the generated world actually follows the action semantics accurately, especially in RT-1 tasks. It feels closer to action-guided video synthesis than true action-driven world modeling.

No comparison to video-to-3D methods.
Works like Nerfies, Vid2NeRF, DynIBaR also reconstruct dynamic scenes from monocular videos. It would help to position Vid2World against those rather than only comparing to video diffusion baselines.

Computation is heavy and unclear.
Training takes 4×A100 for 100k steps with 3 FPS video clips. The paper doesn’t report inference speed or memory usage. If the goal is a usable world model, some efficiency discussion would help.

### Questions
see weaknesses.

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
3

### Summary
The paper proposes a method to build world models from video diffusion models. It argues that this requires two key steps: (1) ‘causalization’ of the model (i.e., the current state should only depend on the past), and (2) ‘action conditioning’ (i.e., the possibility for the future to be influenced by a present action). Especially point (2) is nontrivial since most training data does not contain actions.

### Strengths
The problem is relevant and interesting.
The paper is linguistically well written (in places, it sounds like LLM lingo).
The authors tackle the problem by applying a clever combination of methods, mostly from the literature. 
The execution seems competent. 
The empirical results look strong. 
I enjoyed reading the paper.

### Weaknesses
•	The method description does not provide clear intuition for why the action conditioning works; the paper could better convey the functioning of the model (beyond the mechanics).
	•	The use of the word “causal” is ambiguous—sometimes merely time-directional, while elsewhere invoking counterfactuals/interventions in a broader sense. This limits the potential audience of the paper.

### Questions
The description of the method did not really give me an intuition why the action conditioning works. If the authors have a good intuition on this, then it would be helpful to share this.

One thing that confused me was the use of the world ‘causal’. In some places, this is restricted to time direction; however, the model also talks about counterfactuals and (at least implicitly) interventions, i.e., notions from a community where ‘causal’ has a much broader meaning. I think this should be clarified; otherwise, statements such as "non-causal generative modeling could be inherently easier than its causal counterpart” are prone to misunderstanding. The best would be to add some explanation and discussing connections to “interventional/causal world models” in the causality community (e.g. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9363924, maybe also https://arxiv.org/pdf/2110.06562 ). 

Generalization stress test (video game): Having a video of the following kind would be compelling: 
For a video game, condition on a start frame, and show the time evolution for sequences of random actions (and mark the actions under the video). I’d like to see a long video, to understand where it starts breaking. Also, I would like to see this both (a) in the case where that game was included in the action-labelled part of the training set, or (b) it was not included (but potentially other similar games). I don’t expect it to work great - but I’d like to understand where it breaks.

Final question, Error accumulation (l1410): the authors argue that other models (DIAMOND, Fig 10) get blurry over time and Vid2World does not. I would like to understand this better. If there is inherent stochasticity in the dynamics (and I believe this is the case at least for some of the considered tasks), and you predict the mean, then things necessarily get blurry. If you don’t predict the mean (and I realize Vid2world does not), then things should sometimes go in the wrong direction. Of course the respawn example Fig 12 shows this, but it should also happen in less obvious cases?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper focuses on dealing with temporal causality in adapting pretrained video models to interactive world models. To enable cost-efficient adaptation from foundation video models, the paper explores video diffusion causalization by transferring the pretrained weights, and causal action guidance that injects frame-level action conditions independently. The experiments show some improvements in simulation quality compared to previous baselines on three difference benchmarks.

### Strengths
S1) The proposed method is simple and effective. The writing is easy to follow.

S2) I like the scope of the paper, which is trying to establish general world models for various domains through adapting from foundation video models.

### Weaknesses
W1) I believe the core contribution of this paper is how to causalizing temporal convolution layers. Unfortunately, the most recent video models (e.g., NVIDIA Cosmos and WAN, also mentioned by the authors) are typically using pure DiT architectures without temporal convolutions. Therefore, the main story of this paper is not applicable to those models, which somewhat diminishes the significance of the contribution.

W2) Frame-level action conditioning is not new, and the paper misrepresents its difference from previous works (Line 259). Specifically, existing approaches such as IRASim and Cosmos-Predict-ActionCond already employ frame-level injection, offering the same functionality as the proposed method.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Vid2World, an approach to convert pre-trained video diffusion models into iteractive, action-conditioned world models. The core idea is to utilize the key physical knowledge already encoded in video generation model, and then converting bidirectional layers into casual/temporal layers.

### Strengths
- The paper proposes an interesting solution to solve the problem of converting a non-causal video diffusion model into an autoregressive, interactive world model by manipulating the layers and further finetune the model.
- By transferring priors from a pre-trained video diffusion model, Vid2World produces high-fidelity predictions that significantly outperform other world models on metrics like FVD and FID.
- The approach is demonstrated across multiple, diverse domains.

### Weaknesses
- The base model is pre-trained on passive, action-free videos, which has strong prior that may be mismatching the goal of world models. This potential mismatch and its effect should have been discussed by the paper. 
- The converted autoregressive model may suffer from error accumulation. Converting a diffusion model to such an AR model may make this problem even more pronounced.

### Questions
This work is build on top of bidirectional video diffusion models. This underlines the core methods around causal mask adaptation. Would it be possible to start from strong autoregressive video models without the need to repurpose from bidirectional ones to casual ones?

### Soundness
3

### Presentation
3

### Contribution
3
