# Inversely Learning Transferable Rewards via Abstracted States

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
Inverse reinforcement learning (IRL) has made significant progress in recovering reward functions from expert demonstrations. However, a key challenge remains: how to extract reward functions that generalize across related but distinct task instances. In this paper, we address this by focusing on transferable IRL—learning intrinsic rewards that can drive effective behavior in unseen but structurally aligned environments. Our method leverages a variational autoencoder (VAE) to learn an abstract representation of the state space shared across multiple source task instances. This abstracted space captures high-level features that are invariant across tasks, enabling the learning of a unified abstract reward function. The learned reward is then used to train policies in a separate, previously unseen target instance without requiring new demonstrations in the target instance. We evaluate our approach on multiple environments from Gymnasium and AssistiveGym, demonstrating that the learned abstract rewards consistently support successful policy learning in novel task settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a method for transfer inverse RL where they can transfer rewards to unseen but similar environments.  They use a multi-task VAE, with shared encoder and task-specific decoders, to learn a shared, abstract state space between all training tasks, and a WGAN discriminator to distinguish between expert and policy trajectories on the abstract state space.  This discriminator can then be transferred directly to a new task without new demonstrations.  They demonstrate this on MuJoCo-Gym locomotion tasks with different disabled limbs and on an Assistive Gym task.  They also provide theoretical results that link reward transferability to the abstract state densities between tasks.

### Strengths
1. The authors present a strong idea of learning abstract state representations for a shared reward function.
2. They present promising empirical evidence.  The reward transfer from Ant to HalfCheetah is especially surprising.

### Weaknesses
1. It's unclear how the multi-task VAE actually aligns the state representations between different tasks.  Even with the discriminative objective, it's entirely possible that the encoder does not well align semantically equivalent states.  There's some evidence that the learned embeddings do align based on the t-SNE plots but this may not be the case in higher dimensions.

2. The definition of the problem setting and what tasks can transfer rewards between are imprecise.  The locomotion experiments all deal with transferring the same locomotion tasks between characters with different active limbs or embodiments.  The Human Assistive gym tasks do transfer rewards between feeding and itching tasks but seem to require an extra task reward so it's unclear what is being transferred here.

3. Overall the experiments lack breadth.  Does the method generalize to more harder locomotion or manipulation tasks when the reward functions are more complex?  Can it handle other types of unseen environments (environment dynamics differences, obstacles, different reward functions).

### Questions
1. How is your method able to generalize to HalfCheetah from training on Ant tasks?
2. Is the encoder and reward function frozen for the target task?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces TraIRL, a method for learning transferable reward functions in inverse RL. The core idea is to learn a reward function over an abstract state space that is invariant to the dynamics of different source tasks. The method uses a multi-head VAE to learn this shared abstract representation from expert demonstrations. A Wasserstein GAN objective is then used to structure this abstract space by discriminating between expert and learner trajectories, guiding the learning of a reward function that captures task-agnostic intent. The learned abstract reward is then transferred to a novel target task to train a policy without requiring target-domain demonstrations.

### Strengths
*   **Principled Approach to Disentanglement:** The paper proposes a well-motivated method to disentangle a task's core reward from its specific dynamics by learning a reward function in an abstract state space. This is a significant conceptual strength.
*   **Strong Empirical Performance:** The method demonstrates superior performance over strong baselines in transferring rewards across tasks with different dynamics within the same domain (e.g., MuJoCo Ant with different disabled legs).
*   **Theoretical Grounding:** The paper provides a formal analysis (Theorem 2) that delineates the conditions under which the learned reward is transferable, linking performance to the structural properties of the abstract space.

### Weaknesses
*   **Unverifiable Theoretical Assumptions:** The main theoretical result, Theorem 2, hinges on the "structural alignment" assumption that optimal policies in source and target tasks are close in the abstract space. The paper provides no mechanism to verify this assumption for a new target task, rendering the guarantee non-constructive. The theory explains when transfer works, but provides no guidance on how to ensure it.
*   **Incomplete Reward Transfer:** In the AssistiveGym experiments, the learned abstract reward is insufficient to solve the target task and requires supplementation with an explicit, goal-specific reward shaping term. This suggests the method learns a useful behavioral prior (e.g., how to move smoothly) rather than a complete, transferable reward function that specifies the goal.
*   **High Complexity and Sensitivity:** The framework combines a VAE, a WGAN-GP, and a reward network, resulting in a complex system with many interacting components and hyperparameters. This complexity could pose a significant barrier to reproducibility and practical use.
*   **Misleading Cross-Domain Results:** The impressive Ant-to-HalfCheetah result in Table 2 is from a "one-shot" setting that requires a target expert trajectory and a different training objective (cycle loss), as detailed only in Appendix D.6. Presenting this in the main paper without context overstates the method's zero-shot capability.

### Questions
1.  Why was the Ant-to-HalfCheetah result in Table 2 presented without clarifying its one-shot nature in the main text? This seems to overstate the method's zero-shot transfer capabilities. Would you consider adding the zero-shot result to the table for a more complete and transparent comparison?
2.  The need for additional reward shaping in AssistiveGym suggests the learned "reward" is more of a behavioral prior. How does your method's performance (with shaping) compare to a simpler baseline that uses the same goal-based shaping but replaces the learned reward with a simple, handcrafted shaping term (e.g., action magnitude penalty)? This would help isolate the value of the learned abstract component.
3.  Regarding the structural alignment assumption: Is there any metric that can be computed from the learned VAE and source data to estimate the potential for transfer to a new target domain *before* committing to a full RL training run? Without this, the applicability of the method seems to rely on trial and error.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of reward transfer between environments in inverse reinforcement learning (IRL). To learn transferable rewards, the paper introduces the TraIRL method, which first learns a state abstraction and then learns a reward using a standard IRL framework within this abstracted state space. Experiments across two MuJoCo domains demonstrate that TraIRL can learn rewards that transfer effectively to unseen environments.

### Strengths
1. The paper investigates the important problem of reward transfer between related tasks in IRL. It is crucial that rewards learned through IRL generalize to unseen settings.
2. The paper formally studies the problem of reward transfer in Section 4.5.
3. The comparisons in MuJoCo Gym and Assistive Gym show that TraIRL outperforms baselines in generalization to new target tasks. The analysis in Section 5.2.1 also confirms that TraIRL learns a meaningful state abstraction.

### Weaknesses
1. TraIRL lacks novelty compared to prior IRL algorithms. The method first learns a state encoding and then performs standard IRL on top of this learned representation. Simply learning a state encoding before applying IRL offers limited advancement over prior work.
2. The experiments are limited to domains that are already well suited for learning an easily transferable state encoding. Both domains use the ground-truth state. In MuJoCo Gym, the state encoder only needs to ignore the joint information and focus on the torso, as described in Section 1. Thus, the state encoder can merely filter out irrelevant observations. The paper does not compare to the simple baseline of manually removing these clearly non-transferable features in reward learning. A more meaningful evaluation would involve complex observation spaces where learning a transferable state representation for rewards is challenging, such as image-based observations.
3. The paper lacks sufficient empirical evaluation. Results are presented for only three settings. Additional evaluations are needed to fully assess TraIRL’s performance.
4. The paper does not report the performance of f-IRL, which TraIRL uses as its underlying IRL algorithm.
5. Reporting performance based on only ten episodes per task is insufficient. Since the experiments are conducted in fast simulated environments, many more episodes should be used for evaluation.
6. The paper does not examine how performance changes as the number of expert trajectories varies. Does TraIRL maintain strong performance relative to baselines as the number of trajectories increases? Do baselines also learn good state abstractions when provided with more expert demonstrations? Is TraIRL still able to learn state abstractions with fewer demonstrations?

### Questions
1. How does TraIRL compare to simply removing joint information from the state and retaining only the torso information for reward learning in MuJoCo Gym?
2. What does “ground state density” refer to in line 391?
3. Why are the baseline results omitted for Half Cheetah in Table 2?
4. What is the performance impact of updating the encoder with the reward, as mentioned in line 243?
5. How does TraIRL’s performance compare to regular f-IRL?
6. Do the baselines also jointly train the reward across multiple tasks as TraIRL does?

### Soundness
3

### Presentation
3

### Contribution
2
