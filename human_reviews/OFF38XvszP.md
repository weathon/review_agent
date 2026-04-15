# Slot Structured World Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 5

## Abstract
The ability to perceive and reason about individual objects enables humans to build a robust understanding of the environment and its dynamics. Replicating such abilities in artificial systems would represent a significant milestone toward building intelligent agents. Contrastive Learning of Structured World Models (C-SWMs) took a step in this direction, proposing an unsupervised approach to embed images as compositions of individual object representations and model their pair-wise relationships. Yet the proposed architecture presents an encoder that cannot disambiguate different objects characterized by the same visual features, and the method has only been tested in settings where encoding just the object position and velocity was sufficient to learn the dynamics of the environment.
In this regard, we introduce Slot Structured World Models (SSWMs), a class of world models augmenting C-SWMs with a pretrained object-centric encoder. We further propose a version of the Spriteworld environment that includes simple physics to challenge these models. Quantitative and qualitative measures show that the proposed method outperforms the baseline on the given environment, although it presents severe limitations in multi-step prediction.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes to improve contrastive learning of structured world model with slot-attention mechanisms in its visual encoders. The argument lies in the original feed-forward networks can be challenged by scene objects of similar appearances and a variation of object numbers in inference time. The adopted slot-attention is expected to address this for its object-centric properties. The validation includes predicting GNN dynamics for an interactive spriteworld task, where geometry shapes with different colours can have some simple interactions. The suggested slot structured world model outperforms the baseline when multiple prediction steps are considered. It is also shown to yield more accurate masks to associate scene objects.

### Strengths
1. The motivation and idea are clear and straightforward to follow.
2. The writing is clean in general and does not have much readability issue.
3. The approach might be promising to address more complicated scenarios.

### Weaknesses
1. Limited novelty. Both GNN latent dynamics and slot attention for object-centric representation are not new.
2. Lacking relevant literature review. GNN dynamics learned from image data have been extensively researched in tasks with more physics realism, e.g. see (a) and (b).
3. The experiment results could be stronger. The original C-SSM paper includes multiple benchmarks including interactions beyond simple geometry shapes such as Atari environments. It would be more convincing to see the comparison on these benchmarks and even more as in (a) given physics simulation data is easily to acquire nowadays.

(a) Li et al, Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids, ICLR 2019
(b) Shi et al, RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Objects with Graph Networks, RSS 2022

### Questions
1. The core contribution seems more related to the awareness of the objectiveness of entities in the image. Given the paper also demonstrates the usage of a pre-trained slot attention model, what makes it preferred comparing to an obvious pipeline: using some object segmentation foundation models to obtain the mask and then apply the shared encoder and GNN? 

2. Is there any implicit assumption about the object appearance, such as always fully observable and with a cosntant appearance? How will it work on first-person-view scenarios where a robot gripper might be partially or fully viewed? Can it work well on more realistic data such as 3D objects whose appearance may vary according to perspective and more complex physical interaction as in (a)?

3. How the number of message propagation steps in GNN transition would scale when a long-distance effect is expected? Will the short-cut edge for rigid bodies in (a) be necessary?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors proposed a slot structured world model to learn object-centric representations.

### Strengths
1. SSWM learns distinct attention masks for each object.
2. SSWM outperform C-SWM on Interactive Spriteworld.

### Weaknesses
1. Baselines: There are other works including slotformer[1] and slotdiffusion[2] which also combined slot attention and other temporal modules to learn object-centric representations. The authors should compare SSWM with these baseline models.

2. Data: These two related works mentioned early also evaluated their methods on much more complicated data than Interactive Spriteworld. The authors should show the effectiveness of SSWM on more challenging benchmarks.

3. Lack of novelty: Slot attention plus GNN updating seems incremental.


[1]. Wu, Z., Dvornik, N., Greff, K., Kipf, T. and Garg, A., 2022. Slotformer: Unsupervised visual dynamics simulation with object-centric models. arXiv preprint arXiv:2210.05861.

[2]. Wu, Z., Hu, J., Lu, W., Gilitschenski, I. and Garg, A., 2023. SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models. arXiv preprint arXiv:2305.11281.

### Questions
I think this work is not ready for publication.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose Slot Structured World models (SSWM), a object centric world model to that uses pretrained slot attention to extract the object representations and a GNN based dynamics model to train the world model. The authors show experiments on an Interactive Spriteworld environment where the agent moves among a set of other objects colliding with them. Compared to a previous work, Contrastive Structured world models, SSWM outperforms significantly on Mean reciprocal rank and Hits at rank k metrics which intuitively measure the rollout capability of the world model

### Strengths
Overall, the paper is well written and is fairly easy to understand in a single read.

### Weaknesses
1. **Claims**: I find the core claims of this paper to be exaggerated. 


  (a) Towards the end of page 1, the authors say:

> This paper therefore proposes a new type of dynamics model that embeds an object-centric encoder
and a GNN-based world model.

This is not true. Works like [2] and Interaction networks [3] have the same core idea of using object-centric representations and a GNN based dynamcis model to predict next state of objects. However, these works don't get the object representations in an unsupervised fashion. However, the claim of the above sentence is the proposal of a new dynamics model that integrates GNNs and object centric encoder, which is incorrect.



(b) While enumerating the core contributions of the paper, the authors say that :

> this paper proposes the first learned dynamics model that can isolate individual objects and reason about their (action-conditional) interactions from raw pixel input and can disambiguate between multiple objects with similar appearance. 

To the best of my knowledge, both these statements are inaccurate as several works have shown to address these two things. 

For example: SlotFormer [4], SILOT [5], STOVE [6], SCALOR [7] and numerous other works in the field of unsupervised object-centric video tracking, can all disambiguate between multiple objects and have a learned dynamics model to separate individual objects and reason about them.

Because of the following reasons, I feel that the claims of the paper seem exaggerated to me. I am happy to discuss this with the authors actively during the rebuttal phase.


2. **Methodology**: I do not see this method as a significant change from what SlotFormer does where they use a Transformer as their dynamics interaction module. The primary difference is the choice of modeling the transition dynamics as opposed to a new framework. So, in order to claim that GNN based modeling is more suitable, a comparison should be shown. In the SlotFormer paper, the authors compare against a GNN-based model DCL where SlotFormer outperforms DCL on CLEVERER based reasoning tasks (VQA). I'm curious if the authors have performed any such experiments.

3. **Ablation Study**: The iterative GNN module is definitely a bottleneck as the message passing needs to be done $K-1$ times in the worst case. An ablation study of how much this matters for the environments considered would be an important experiment for showing the iterative mechanism's importance.

4. **Environment**: There are several already existing benchmarks that can be used such as BBS dataset from [1].
I'm curious so as to why the authors didn't use the BBS dataset or any other existing benchmarks such as MOVie, CLEVERER for validating their experiments.

C-SWM was introduced in ICLR 2020, and the field of unsupervised (generative) object-centric world modeling has progressed significantly where works typically show their performance on complex datasets and environmets. I do feel that this works lacks concrete evaluation on that end. I would encourage the authors to look into this for the next iteration of the manuscript.

5. **Metrics**: Given that the object centric encoding is obtained via pre-training SlotAttention and a decoder can be used to see the reconstruction of the the predicted latent states adding MSE of the rollouts would be beneficial as well to see how accuracte the reconstruction is.

6. **Experiments on Reasoning tasks**: The core claim of the paper suggests that SSWM are good at reasoning (Contribution 1), however there are no experiments to show this ability of the world model. Results of SSWM on benchmarks such as the Visual Question Answering in the CLEVERER dataset would validate these claims.


-----
## **References**:

[1] Learning Robust Dynamics through Variational Sparse Gating, NeurIPS 2022 (https://github.com/arnavkj1995/BBS)

[2] Compositional Video Prediction, ICCV 2019

[3] Learning Long-term Visual Dynamics with Region Proposal Interaction Networks, ICLR 2021

[4] Slotformer: Unsupervised visual dynamics simulation with object-centric models, ICLR 2023

[5] Exploiting Spatial Invariance for Scalable Unsupervised Object Tracking, AAAI 2020

[6] Structured object-aware physics prediction for video modeling and planning, ICLR 2020

[7] Scalor: Generative world models with scalable object representations, ICLR 2020

----

### **Rationale for current rating**
As mentioned above, I don't believe the core contribution of the work is any different from already proposed dynamics models and the environments on which SSWM has been evaluated is insufficient. Based on these two primary concerns, I would like to vote for the rejection of the paper. I will, however, make a final decision after (a) rebuttal by authors and (b) discussion with the other reviewers.

### Questions
7. World models are being used extensively in Visual model based RL -- so I am curious so as to if the authors had tried running any RL experiments?

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes to combine slot-attention encoder with Graph Neural Network (as a dynamic model) to model the dynamic of each slot based on a state and an action. The authors also enhance Spriteworld environment with physics to allow GNN to model physical interaction between slots-objects. The proposed model outperforms the baseline (C-SWM) on the Spriteworld benchmark.

### Strengths
1) SSWM clearly outperforms C-SWM as a baseline on the proposed Spriteworld benchmark.
2) The qualitative analysis confirms it and shows nice disentanglement of objects in the SSWM model. 
3) Simple design of the SSWM model (slot-attention + GNN).

### Weaknesses
1) I believe a failure in the slot-attention mechanism to effectively disentangle objects is likely to compromise the entire method. Given that slot-attention has shown limited success in parsing objects in real-world image datasets, this can limit robustness and applicability of the SSWM method. So it would be interesting to check how the SSWM performance degrades as object disentanglement produced by the encoder degrades.
2) The novelty is limited as the paper suggested a simple combination of two ideas. 
3) The authors consider only one baseline. It would be nice to have more baselines, such as a simple autoencoder, and a latent next state predictor utilizing only one slot. Plus, it would be beneficial if authors can consider different slot-encoders as well. 
4) The authors test their method only on Spriteworld environment. It would be beneficial for the paper to include additional environments, for instance, 2D shapes, some Atari games, or even more complex settings like Minecraft. It would be interesting to see failure cases of object disentanglement in these environments along with success cases.

### Questions
Will it help in terms of metrics if one propagates the dynamic loss from GNN to encoder?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
