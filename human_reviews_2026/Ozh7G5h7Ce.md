# Shared Modular Recurrence for Universal Morphology Control

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
A universal controller for any robot morphology would greatly improve computational and data efficiency. By utilizing contextual information about the properties of individual robots and exploiting their modular structure in the architecture of deep reinforcement learning agents, steps have been made towards multi-robot control. When the robots have highly dissimilar morphologies, this becomes a challenging problem, especially when the agent must generalize to new, unseen robots. In this paper, we hypothesize that the relevant contextual information can be partially observable, but that it can be inferred through interactions for better multi-robot control and generalization to contexts that are not seen during training. To this extent, we implement a modular recurrent transformer-based architecture and evaluate its (generalization) performance on a large set of MuJoCo robots. The results show a substantial improved performance on robots with unseen dynamics, kinematics, and topologies, in four different environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper builds upon previous modular robotic control approaches, and proposes to infer unobservable but relevant contextual information from history interactions using recurrent networks to enhance cross-embodiment generalization. The resulting modular recurrent transformer-based architectures, R-MeMo and R-MoMo, are validated across four commonly adopted MuJoCo environments, yielding notable performance gains compared with original networks without recurrence.

### Strengths
1.This paper addresses an important problem in robotic control, i.e., learning universal controllers generalizable to morphologically different agents. The motivation of inferring contextual information from environmental interactions is interesting (though I respectfully believe this motivation is not fully supported by the experiments; Please see Weakness 1). 

2.The experimental results are promising, outperforming the latest baselines by a large margin. Four simulation environments with varying difficulty levels are examined, increasing the credibility. 

3.The authors provide detailed research background and related works, clearly delineating the relationships between their work and the literature.

### Weaknesses
1.One of the key claims of this paper is that some unobservable contextual features could be inferred from environmental interactions. However, the notable performance drops seen when some critical features are removed, as reported in Figure 8, indicate that much of the contextual features could not be successfully recovered, which seems contradictory. 

2.Since the proposed methods, R-MeMo and R-MoMo, largely build upon MetaMorph and ModuMorph, the authors are suggested to provide a more detailed introduction to their architectures, in order not to cause confusion in readers not familiar with these prior works.

### Questions
1.The use of RNN for dealing with POMDP seems a common practice. Following Weakness 1, how could one eliminate the possibility that the modular recurrence is merely learning to recover unobservable state transitions (as in a standard POMDP setting) rather than morphological contexts? I would be happy to raise my rating if the authors could by some means disentangle these two, for example, by showing correlation between RNN representations and morphological features. 

2.Could the authors explain why, in the R-MoMo architecture, AOH is fed into the base controller (i.e., Transformer) rather than into the context encoder and used for generating network parameters alongside the observable contexts?

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
3

### Summary
The paper proposes addressing the challenging problem of training a single Reinforcement Learning policy to control any robot morphology. The authors hypothesise that key robot characteristics (like friction or damping) are partially observable introduce a per-limb Recurrent Neural Network (RNN) into modular transformer-based architectures, demonstrating a substantial and consistent improvement in generalizing to robots.

### Strengths
* The use of RNN for the partially observable CMDP is well-motivated in Section 2.2. It would be even nicer to provide a few quantitative evidence to show the partial observability. 
* Empirically, the work successfully demonstrates a significant and consistent increase in generalisation performance to unseen robot morphologies, dynamics, and kinematics over strong baselines (MetaMorph and ModuMorph).

### Weaknesses
* While the paper hypothesises that recurrence allows the agent to infer specific unobservable context (like friction or damping), the paper does not include an explicit analysis or visualisation that confirms what the RNN is encoding or how well it correlates with the true (unobservable) physical properties. 
* The authors also comment on the slow training speed of the RNN. It would be helpful to provide the training speed of the experiments. And also some ablation studies on the key hyperparameters, e.g. RNN's latent state, shared network.

### Questions
1. For R-MoMo and R-MeMo, why are the positions of the RNN different? Can they inserted both before the embedding or before the transformer? 
2. Can you briefly comment on the training and testing robots in Section 5.4? Are they sampled from the same distribution? 
3. We know that RNN is an architecture to solve a Meta RL problem. Are there other advanced Meta RL methods helpful to this case?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper builds upon previous transformer-based universal control methods (MetaMorph and ModuMorph) by introducing a recurrent model to handle partially observable robot contexts. These robot contexts include properties like robot limb mass, shape, gear ratio for each joint, etc, as listed in the appendix. The authors hypothesize that the relevant contextual information can be partially observable, and in such cases, the added recurrent module allows the agent to infer hidden contextual information, achieving better multi-robot control and better generalization to unseen robot contexts. The experiments on MuJoCo show consistent improvements in zero-shot generalization across unseen robot morphologies, dynamics, and kinematics, demonstrating that integrating recurrence helps the controller adapt more effectively to diverse and unfamiliar robots.

### Strengths
- The performance gain is very consistent, showing promising benefits of the proposed shared modular recurrence.
- Though I find simple and incremental, this work presents a very clear motivation and well-defined problem setting.

### Weaknesses
- The scope of the experiments seems very limited to me considering the large diversity of possible robot morphologies. The paper does not analyze how the recurrent mechanism scales with larger or more diverse sets of robots, nor does it investigate the relationship between the amount of training data and the observed generalization gains. Without experiments on different dataset sizes or more complex morphological distributions, I'm not convinced that the proposed recurrent module truly improves generalization in a scalable and robust manner.
- The proposed approach mainly extends existing transformer-based frameworks MetaMorph and ModuMorph by introducing a recurrent layer to handle partial observability, which is also a hypothesis brought by this work. Although this modification leads to performance improvements, it represents a relatively minor architectural change without introducing new learning principles or theoretical insights. The paper positions the work as addressing partial observability, but the underlying idea of adding recurrence to capture temporal dependencies is conceptually straightforward and has been widely explored in prior reinforcement learning and robotics studies. Therefore, the novelty and contribution may not be sufficient.
- Several figures use inconsistent evaluation scales, which makes it difficult to visually compare performance across different settings, especially for training/testing comparison. For example, in Figure 4 the returns in Incline reach nearly 3000 during training, whereas in Figure 6 the corresponding test returns drop below 1000.

### Questions
- Are there any other methods beyond MetaMorph and ModuMorph that can be integrated with recurrence?
- Why in Fig. 8, train and test performance in Flat Terrain, the variance of ModuMorph when provided with body_mass is exceptionally large? Besides, the x-axis labels in Fig. 8 are slightly misaligned.
- Why do you think making the context of a robot partially observable is important? To my understanding, the context of a control task is usually available.

### Soundness
3

### Presentation
2

### Contribution
2
