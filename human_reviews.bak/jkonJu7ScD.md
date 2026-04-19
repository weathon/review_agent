# MIND: Masked and Inverse Dynamics Modeling for Data-Efficient Deep Reinforcement Learning

- Decision: Reject
- Scores: 3, 5, 6, 5

## Abstract
In pixel-based deep reinforcement learning (DRL), learning representations of states that change because of an agent’s action or interaction with the environment poses a critical challenge in improving data efficiency. Recent data-efficient DRL studies have integrated DRL with self-supervised learning (SSL) and data augmentation to learn state representations from given interactions. However, some methods have difficulties in explicitly capturing evolving state representations or in selecting data augmentations for appropriate reward signals. Our goal is to explicitly learn the inherent dynamics that change with an agent’s intervention and interaction with the environment. We propose masked and inverse dynamics modeling (MIND), which uses masking augmentation and fewer hyperparameters to learn agent-controllable representations in changing states. Our method is comprised of a self-supervised multi-task learning that leverages a transformer architecture, which captures the spatio-temporal information underlying in the highly correlated consecutive frames. MIND uses two tasks to perform self-supervised multi-task learning: masked modeling and inverse dynamics modeling. Masked modeling learns the static visual representation required for control in the state, and inverse dynamics modeling learns the rapidly evolving state representation with agent intervention. By integrating inverse dynamics modeling as a complementary component to masked modeling, our method effectively learns evolving state representations. We evaluate our method by using discrete and continuous control environments with limited interactions. MIND outperforms previous methods across benchmarks and significantly improves data efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work proposes to use self-supervised tasks including reconstruction and next-state prediction based on masked input states, to improve deep reinforcement learning algorithm. The proposed method also combines Transformer to jointly model the short state sequence input. The proposed method is compared to multiple baselines and exhibits better performance in Atari and DMC benchmarks.

### Strengths
- The proposed method is experimentally shown to be effective on two benchmarks.

- Analysis on masking strategy is extensive and informative.

### Weaknesses
1. **Confusing diagram**. Figure 1 is very misleading to explain "reconstruction" and "inverse dynamics modeling". The reconstruction usually refers to predicting the original image, while the actual method is about minimizing error between the latent representations from two networks. This diagram also does not help explain action prediction (inverse dynamics modeling). The current diagram is showing a comparison of image embeddings.

2. **Clarity**. Seciton 3 does not explain:

   * Why a Transformer module is introduced to help modeling
   * How action is predicted from the model, using current and next states
   * How the online network performs state representation regression and action prediction simultaneously. Are there seperate heads for each auxiliary task?

3. **Baselines**. Clearly there are missing baselines like RAD [1], DrQv2 [2] according to this study [3] on self-supervised learning with RL, and possibly [4] which uses Transformer for next frame prediciton, and vanilla MAE [5]. With the current baselines I can not evaluate the significance of proposed method, considering the novelty of the proposed method is limited. 

4. Due to extra architecture and parameter introduced in the Transformer module, are the baseline models using the same/similar architecture to ensure the fairness? Is the Transformer important in the modeling?

[1] Laskin, Michael et al. “Reinforcement Learning with Augmented Data.” ArXiv abs/2004.14990 (2020)

[2] Yarats, Denis, et al. "Mastering visual continuous control: Improved data-augmented reinforcement learning." arXiv preprint arXiv:2107.09645 (2021).

[3] Li, Xiang et al. “Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?” ArXiv abs/2206.05266 (2022)

[4] Micheli, Vincent, Eloi Alonso, and François Fleuret. "Transformers are sample efficient world models." arXiv preprint arXiv:2209.00588 (2022).

[5] He, Kaiming et al. “Masked Autoencoders Are Scalable Vision Learners.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021): 15979-15988.

### Questions
Please see weaknesses.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel representation learning objective MIND that integrates inverse dynamics modeling with the masked modeling of consecutive states. This dual approach enables the representation to encapsulate both the agent-controllable aspects as well as static features within the image. The effectiveness of this method is demonstrated through empirical results on the Atari 100K and DMControl-100K benchmarks.

### Strengths
The paper is very well-written. The empirical evaluation is comprehensive for tasks with discrete action spaces Atari-100K. Additionally, comprehensive ablation studies of the learning objective as well as hyperparameter choices are conducted.

### Weaknesses
The evaluation for continuous control tasks are very limited. DMControl 100K is originally proposed and evaluated in CURL, which consists of the simplest tasks in DMControl. Later works such as Dreamer-v2/v3, DrQ-v2, A-LIX[1], ATC[2], and TACO[3] mainly consider medium-difficulty tasks as well as the more challenging humanoid domain. Thus, more tasks and especially harder medium-difficulty tasks should be evaluated for the proposed method. I recommend the author to provide a comparison at 1M/2M of harder tasks instead of the six tasks presented in the paper. Also, A-LIX[1], ATC[2], and TACO[3] should be discussed and compared in the empirical evaluation.

### Questions
As shown by prior works such as [4, 5], the single-step inverse model is theoretically and empirically non-sufficient to capture the full agent-centric representation. For example, in an empty gridworld, a pair of positions separated by two or more spaces may be assigned an identical representation without incurring additional loss in a one-step inverse model.. In contrast, a multi-step inverse model is both theoretically sufficient and in practice achieves good performance. Would MIND benefit from multi-step inverse modeling instead of single-step inverse?

I am willing to raise the score if my above two questions/concerns (a more comprehensive evaluation of continuous control tasks and multi-step inverse model instead of single-step) are addressed. I understand that given the time constraint of rebuttal, the authors are not able to address them fully. But it would be great at least to see the additional experiments on a few medium DMControl tasks.

**Additional References that are not included in the paper**
- [1] Edoardo Cetin, Philip J. Ball, Steve Roberts, Oya Celiktutan, Stabilizing Off-Policy Deep Reinforcement Learning from Pixels, ICML 2022
- [2] Adam Stooke, Kimin Lee, Pieter Abbeel, Michael Laskin, Decoupling Representation Learning from Reinforcement Learning, ICML 2021
- [3] Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daumé III, Furong Huang. TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning, NeurIPS 2023
- [4] Riashat Islam, Manan Tomar, Alex Lamb, Yonathan Efroni, Hongyu Zang, Aniket Didolkar, Dipendra Misra, Xin Li, Harm van Seijen, Remi Tachet des Combes, John Langford, Principled Offline RL in the Presence of Rich Exogenous Information. ICML 2023
- [5] Alex Lamb, Riashat Islam, Yonathan Efroni, Aniket Didolkar, Dipendra Misra, Dylan Foster, Lekan Molu, Rajan Chari, Akshay Krishnamurthy, John Langford, Guaranteed Discovery of Control-Endogenous Latent States with Multi-Step Inverse Models.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a hybrid objective of something very similar to BYOL (predicting representations of a target network) while the online network is masked, while combining this with a one-step inverse model.  This approach has nice empirical results and solid empirical analysis on Atari 100k.  The big downside I see to this paper is that the method is not extremely novel and the justifications are ad-hoc, whereas the field of representations for RL has seen rapid progress in theoretical analysis, so it should be possible to provide more detailed justifications for the technique.  Nonetheless, I see this paper as a solid empirical contribution.

### Strengths
-The improvements over reasonable baselines like SPR and DrQ are considerable on the Atari 100k, which is a fairly difficult setting.  The improvements on DM control suite are also convincing.  
  -The analysis of what the model is learning is reasonably thorough.  
  -I appreciated the study of wall-clock time, showing that the method is practical to use on a single GPU and is cheaper than other methods.

### Weaknesses
-The justification in this paper for using inverse dynamics along with masked prediction is fairly ad-hoc and informal.  This isn't the end of the world, but the understanding of this area from theoretical RL is rapidly advancing (Efroni 2022, Lamb 2022, for example).  There have also been a few empirical papers with related ideas such as the InfoPower paper as well as the ACRO paper (Islam 2022).  One simple thing that might be worth trying is for the IT model, use the observation k steps in the future (with k sampled U(1,5) for example) and then predict the first action.  Some theoretical work has suggested the value of this approach.  
  -The basic approach seems similar to combining BYOL (with masking as the augmentation) with a one step inverse model.

### Questions
-Why is there an IT network, rather than reusing the MT network for the IT network, perhaps with a distinct head for predicting inverse dynamics?  
  -Why not also use masking of the inputs to the IT network?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to learn representations for reinforcement learning  using a combination of inverse dynamics training and masked modeling.

### Strengths
- The problem of learning representations from reinforcement learning makes sense
- The paper appears to improve performance over baselines
- The paper evaluates across a set of different environments, showing improved performance across them
- The paper runs a comprehensive set of ablations

### Weaknesses
- In equation one, p, c, q,  is not defined
- It would be good to report confidence intervals in results, for instance in Table 1
- The paper doesn't seem very novel -- it uses momentum contrast masked loss + a standard inverse dynamics representation learning objective

### Questions
Can the authors provide some intuition why masking + momentum contrast outperforms other baseline methods?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
