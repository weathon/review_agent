# Improving and Accelerating Offline RL in Large Discrete Action Spaces with Structured Policy Initialization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Reinforcement learning in combinatorial action spaces requires searching over exponentially many joint actions to simultaneously select multiple sub-actions that form coherent combinations. Existing approaches either simplify policy learning by assuming independence across sub-actions, which often yields incoherent or invalid actions when coordination is required, or attempt to learn action structure and control jointly, which is slow and unstable. We introduce Structured Policy Initialization (SPIN), a two-stage framework that first pre-trains an Action Structure Model (ASM) to capture the manifold of valid actions, then freezes this representation and trains lightweight policy heads for control. On challenging DM Control benchmarks, SPIN improves average return by up to $39\%$ over the state of the art while reducing time to convergence by up to $12.8\times$.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Problem: 

The paper tackles offline reinforcement learning in large, discrete combinatorial action spaces, settings where the agent must select from exponentially many joint actions (composed of multiple sub-actions) and ensure these selected sub-actions form coherent combinations. This is relevant for domains like healthcare decision support, robotics, recommendation systems, and fleet management, where online exploration is costly, risky, or infeasible.

Approach:
- The authors propose Structured Policy Initialization (SPIN), a two-stage framework that decouples representation learning from control. :
(a) Action Structure Model (ASM) is trained to learn an action representation function, 
(b) ASM is frozen and lightweight policy heads are trained for downstream RL control on this learned action representation.

### Strengths
- SPIN offers a principled and empirically validated way to accelerate and improve offline RL in large discrete combinatorial action spaces, primarily by decoupling structure learning and control, thus making learning tractable and robust as complexity grows. The separation of structure and control is motivated and clearly shown to overcome the slowness/instability of joint learning 

- SPIN works with multiple offline RL algorithms (IQL, AWAC, BCQ). Overall, SPIN is a promising and elegant approach that reframes discrete combinatorial control as a representation problem.

### Weaknesses
- The current framework requires architectural compatibility between ASM and policy modules for effective weight transfer. This can limit its integration with arbitrary RL architectures and restrict broader applicability.

### Questions
- The paper notes that SPIN is compatible with IQL and AWAC but not CQL. Could you elaborate on stability issues that arise with value-regularization methods and whether hybrid objectives could reconcile them?

- Why was masked conditional modeling chosen over contrastive or next-sub-action prediction? Did you test alternative pretext tasks, and if so, how did they compare?

- Have you evaluated SPIN on higher-arity combinatorial domains (e.g., VRP or job-shop scheduling) where the sub-action semantics differ? Would the same ASM formulation apply without state–action token alignment?

### Soundness
3

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
3

### Summary
This paper introduces SPIN, a two-stage approach designed to improve efficiency in discrete combinatorial action spaces. Specifically, it separates representation learning from policy learning. In the first stage, an action structure model learns a representation function that captures the manifold of valid actions. In the second stage, this representation is frozen and reused, with lightweight policy heads built on top of the pre-trained action structure model. The experimental results demonstrate clear benefits in terms of both performance and efficiency.

### Strengths
The paper is clearly written and well motivated.

The proposed idea is straightforward, and the algorithm is compatible with actor–critic frameworks, which enhances its applicability across a wide range of settings. The experimental results demonstrate the superiority of the proposed approach compared with the three selected baselines.

### Weaknesses
While the focus on offline RL is relevant, it is not sufficiently justified in the paper. In particular, SAINT is originally an online approach, which has been used here in an offline setting for comparison. In my view, it is not entirely fair to claim that SAINT jointly learns the action structure and control, as it was designed for a different purpose. This raises questions about the validity of the comparison.

The evaluation is also somewhat limited. The implementation details for the selected baselines are not described clearly, making the fairness of the comparison uncertain. While it is understandable that the authors aimed to keep architectural choices consistent, comparing directly with the original implementations of the baseline methods would strengthen the credibility of the results.

There are a few relevant works in this area that the authors may wish to consider for experimental comparison, such asOHIO (https://openreview.net/forum?id=dTPz4rEDok), and MERLION (https://proceedings.mlr.press/v162/gu22b/gu22b.pdf). Particularly, the paper claims the decoupling the representation learning from control. However, MERLION also learns reusable action embeddings. The contribution over MERLION remains unclear in the paper.

### Questions
1. Please justify the experimental comparison. 
2. Please clarify the contributions with respect to the earlier works, especially, MERLION.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors proposed an algorithm for RL with combinatorial action spaces. The proposed method has two stages. We learn the action structure during the first stage, and then we learn  a policy in the second stage.

### Strengths
By separating the learning of action structure and policy, the proposed algorithm overcomes the computational cost issue that a previous work named SAINT has.

### Weaknesses
Determining when to finish pretraining and move on to policy training is crucial. Stopping pretraining too early could lead to poor action structure modeling (Sec 6.1 illustrates the importance of sufficient pretraining), and stopping pretraining too late could lead to the same computational cost issue that is with SAINT. The authors do not provide an approach to choose the stopping time of pretraining. 

The proposed method largely reuses the policy architecture in SAINT, and thus the novelty of this work is limited.

### Questions
Could the authors provide an approach to choose the stopping time of pretraining?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose an RL method to handle large discrete action spaces. The method pretrains a transformer with masked action inputs to reconstruct the action, then do RL on top of the learned transformer representation. They show this method matches other large action RL baselines in just a fraction of their training times. The method is evaluated in a modified version of DM Control where the action space is discretized.

### Strengths
- The method is very simple and intuitive. By doing self supervised learning on the large action space, one can learn a more meaningful action representation than the original one. This will lead to large downstream gains. 
- The paper is well written and easy to understand.
- The experiment section has interesting analysis results to pin down why SPIN is helpful.

### Weaknesses
### Empirical evaluation feels toy and contrived
- these methods are all evaluated in rather artificial RL tasks (hopper, quadruped, etc.), where they take a popular benchmark (DM Control) and then factorize the action space. While useful for fast iteration and initial scientific insight, it is insufficient for convincing me that this method, or even the problem of large discrete action space, is useful. The authors motivated the problem by citing natural problems with large action spaces like recommender systems, robot assembly, etc. Could the authors show results in a more realistic problem setting?


### Method novelty 
- In terms of methodological novelty, there's not too much at the high level. When you have noisy or high dimensional data, e.g. noisy sensors, high dim images, representation learning is the first thing we try to improve the signal to noise ratio of our data. So doing this for actions, using a standard masked reconstruction objective, seems very obvious, and not too "novel". On the other hand, this method is "novel" in the sense of applying the masked reconstruction objective to this particular problem where action spaces are large. 
- This can be seen as a special case of literature studying masked transformers for decision making problems [1], where the mask of the transformer is just set to the action modality. It would be interesting to compare SPIN against a masked transformer baseline that does masking over both state and action modalities. 

[1] Masked Trajectory Models for Prediction, Representation, and Control

[2] PASTA: Pretrained Action-State Transformer Agents

### Questions
See weaknesses, I would like to see more realistic experiments. For method novelty, it would be addressed by better framing, and also comparing against a standard representation learning approach like masked reconstruction over the entire input sequence.

### Soundness
3

### Presentation
3

### Contribution
2
