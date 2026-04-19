# H-GAP: Humanoid Control with a Generalist Planner

- Decision: Accept (spotlight)
- Scores: 8, 8, 6

## Abstract
Humanoid control is an important research challenge offering avenues for integration into human-centric infrastructures and enabling physics-driven humanoid animations.
The daunting challenges in this field stem from the difficulty of optimizing in high-dimensional action spaces and the instability introduced by the bipedal morphology of humanoids. 
However, the extensive collection of human motion-captured data and the derived datasets of humanoid trajectories, such as MoCapAct, paves the way to tackle these challenges. In this context, we present Humanoid Generalist Autoencoding Planner (H-GAP), a state-action trajectory generative model trained on humanoid trajectories derived from human motion-captured data, capable of adeptly handling downstream control tasks with Model Predictive Control (MPC).
For 56 degrees of freedom humanoid, we empirically demonstrate that H-GAP learns to represent and generate a wide range of motor behaviors. Further, without any learning from online interactions, it can also flexibly transfer these behaviours to solve novel downstream control tasks via planning. Notably, H-GAP excels established MPC baselines with access to the ground truth model, and is superior or comparable to offline RL methods trained for individual tasks.
Finally, we do a series of empirical studies on the scaling properties of H-GAP, showing the potential for performance gains via additional data but not computing.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper incorporates VQ-VAE, a prior transformer, and MPC planning to construct H-GAP, a state-action generative model for humanoid trajectories. Experiments show that H-GAP learns to generate a wide range of human behaviors through imitation learning and flexibly transfers to unseen downstream tasks through planning without retraining.

### Strengths
The proposed method outperforms MLP-backed behavior cloning in imitation learning by a large margin. Despite being a generalist policy, it also outperforms MPPI in most tasks by large margins in downstream tasks and performs reasonably well against offline RL specialist policies. 

We also observe that the performance of imitation learning increases as the model and data scale up, paving the way to a possible future foundation model for humanoid control. However, we observe a drop in downstream task performance as the model size increases.

### Weaknesses
1. The provided link in the abstract results in 404 not found, and this submission comes without supplementary videos. This, unfortunately, makes it difficult to assess the naturalness of the generated human motion, which is critical to humanoid motion generation. 
2. The paper claims that H-GAP provides a basis for a foundation model for humanoid control. However, only the performance of imitation learning improves with the model scale. The performance of downstream tasks drops for larger models. In my view, this does not constitute a foundation model's basis because supporting downstream tasks is the most important feature of a foundation model.

### Questions
- What are the metrics used in table 1 and table 2?
- How long does it take to train H-GAP?
- I would assume that the optimal number of samples $N$ corresponds to the length of the intended trajectory $T$. What is the choice of $N$ for your experiments when $T=M=16$ and $L=4$?
- Following up from the above question, how long does it for H-GAP to produce the reported results in table 2? How does it compare to the other methods?
- Does the temperature change throughout algorithm 1? What are the choices of $\Upsilon$ and $\rho$?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper tackles the challenging problem of general humanoid control, which is difficult due to high-dimensional action spaces and instability. The authors propose a Humanoid Generalist Autoencoding Planner (H-GAP) that learns a forward dynamics latent space model from the MocapACT dataset and propose model predictive control for downstream tasks. The forward dynamics model is trained using a transformer over discrete latent codes learned using a VQVAE that employs multiple codes per transition. The effectiveness of H-GAP is demonstrated through experiments on a 56 DoF humanoid across imitation benchmarks as well as analyses of its scaling properties on different model sizes.

### Strengths
- Trains a single prior transformer for motion completion in a more principled way to jointly learn a motion prior and forward dynamics model that can be queried autoregressively and used for MPC planning. Introduces multi-code latent space modeling with VQVAE and transformer to accurately capture complex humanoid state-action sequences.
    
- Analyzes scaling properties on model sizes up to 300M parameters, revealing limitations in downstream task performance despite accuracy gains.
    
- Demonstrates strong imitation learning on a 56 DoF humanoid, validating the model's ability to represent diverse motor behaviors from the MocapAct dataset.

### Weaknesses
- The paper could analyze the coverage of different motion skills by randomly sampling initial states and generating rollouts from the model. This would help quantify the diversity of motions that can be produced [1].
    
- For analyzing imitation abilities, the model could be conditioned on a small segment of reference trajectories rather than just initial states. This would better evaluate how well it can leverage larger context sequences. The current approach in Table 1 mainly reveals modeling/memorization capabilities rather than imitation abilities.
    
- Additional experiments could be designed to isolate the planning contributions of the model beyond just trajectory modeling. For example, a conditional generation task could be constructed using out-of-distribution initial states in order to see if the model is able to “catch-up” to imitate a trajectory from a different initial polse from the original trajectory.
    
- More complex downstream tasks requiring long-term planning could be explored to go beyond trajectory imitation. The current tasks are limited to state distributions from the training data. New tasks like going to specified locations, or following a particular heading and and orientation[1] would require guiding the motion generation process. This could reveal how well the model can piece together behaviors.
    
- Overall, evaluating the model on tasks and scenarios that require leveraging the full context sequence in a generative way would provide better insight into the planning abilities. The current experiments focus primarily on trajectory modeling in different flavors. (imitation based Downstream tasks and imitation benchmark)
    

[1] Dou, Zhiyang et al. “C·ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters.” *ArXiv* abs/2309.11351 (2023): n. pag.

### Questions
NA

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a Humanoid Generalist Autoencoding Planner (H-GAP), consisting of a VQ-VAE that maps state-action trajectories into potentially discrete codes based on MoCapAct, a transformer capable of modeling future distributions based on past observations, and an MPC planning algorithm based on multiple trajectories. Extensive experiments show that H-GAP can benefit from larger and more diverse datasets, demonstrating the potential of this line of work.

### Strengths
The proposed H-GAP is even simpler than the existing offline RL methods while it is general for different downstream control tasks, meaning that the algorithm does not need access to the simulator and train different high-level policies.

Experimental results also show that the proposed H-GAP can outperform offline RL policy and other traditional MPC algorithms.

### Weaknesses
Looking at the visualization on the website, I'm wondering why there is severe jitter compared to the ground truth sequence. I'm not sure if this is a result of the transformer not modeling temporal smoothness well, or if VQ-VAE doesn't represent state-motion clips well.

I'm still not fully convinced of the ability of the proposed method to model a variety of humanoid motions (e.g., backflips) that have more complex patterns than locomotion. While the paper emphasizes data and model scaling experiments, which I appreciate, train data is a random sampling of the dataset, not of how performance changes when more action categories are added.

I guess a potential limitation is the need to generate multiple trajectories. I would like to know how efficient the proposed method is, what the number of trajectories that typically need to be sampled, and whether these hyperparameters need to be changed to accept more trajectories as the motion database becomes more complex.

### Questions
I'm not sure I understand correctly that the whole framework doesn't use a simulator at all. So what's the point of modeling action trajectories here, why not just state trajectories? And if so, why don't we just follow the existing kinematics-based human motion generation?

Overall, the author's response to the concerns is needed to make the final decision. I am happy to increase the rating if my concerns are addressed.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
