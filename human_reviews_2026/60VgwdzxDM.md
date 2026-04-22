# One-Step Flow Q-Learning: Addressing the Diffusion Policy Bottleneck in Offline Reinforcement Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Diffusion Q-Learning (DQL) has established diffusion policies as a high-performing paradigm for offline reinforcement learning, but its reliance on multi-step denoising for action generation renders both training and inference slow and fragile. Existing efforts to accelerate DQL toward one-step denoising typically rely on auxiliary modules or policy distillation, sacrificing either simplicity or performance. It remains unclear whether a one-step policy can be trained directly without such trade-offs. To this end, we introduce One-Step Flow Q-Learning (OFQL), a novel framework that enables effective one-step action generation during both training and inference, without auxiliary modules or distillation. OFQL reformulates the DQL policy within the Flow Matching (FM) paradigm but departs from conventional FM by learning an average velocity field that directly supports accurate one-step action generation. This design removes the need for multi-step denoising and backpropagation-through-time updates, resulting in substantially faster and more robust learning. Extensive experiments on the D4RL benchmark show that OFQL, despite generating actions in a single step, not only significantly reduces computation during both training and inference but also outperforms multi-step DQL by a large margin. Furthermore, OFQL surpasses all other baselines, achieving state-of-the-art performance in D4RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the slow multi-step denoising and unstable optimization inherent in diffusion-based policies, this paper introduces One-Step Flow Q-Learning (OFQL). OFQL reformulates the diffusion denoising process within the Flow Matching (FM) framework and learns an average velocity field that enables direct one-step action generation. Experiments across diverse D4RL tasks demonstrate that OFQL achieves the highest average normalized scores among all compared methods. Moreover, its one-step sampling design substantially improves both training and inference efficiency.

### Strengths
OFQL is an efficient reinforcement learning algorithm that introduces average velocity fields within the Flow Matching framework, enabling it to model complex policy distributions without relying on distillation procedures or auxiliary networks. The method offers a unified training–inference pipeline, using the same one-step model consistently in both phases. Empirically, OFQL outperforms DQL and other strong baselines in terms of both policy performance and computational efficiency.

### Weaknesses
1.	OFQL relies on the MeanFlow Identity to enable one-step sampling for the learned policy. However, the Jacobian–vector product computation in Eq. (11) may become computationally demanding for large-scale models.

2.	As acknowledged in the paper’s limitations, it remains unclear whether OFQL can scale to high-dimensional action spaces (e.g., humanoid control or vision-based RL). Moreover, the stability of the proposed one-step policy under non-stationary or online settings has not been investigated.

3.	The paper provides no formal analysis establishing the expressive equivalence between the average-velocity one-step formulation and traditional multi-step diffusion policies.

### Questions
1.	Does learning an average velocity field constrain the representational power compared to DDPM’s full reverse process?

2.	How sensitive is OFQL to flow ratio and time-sampling distribution?

3.	Does one-step flow matching better handle out-of-distribution states than diffusion-based DQL?

4.	How exactly does the Q-gradient interact with flow learning?

### Soundness
3

### Presentation
4

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
This paper proposes One-Step Flow-Q-Learning, an offline RL algorithm that enables one-step action generation during both training and inference. The method is closely related to DQL but leverages the average velocity parametrization following MeanFlow. The action sampling in this parametrization can be done in a single step, which reduces inference cost and avoids the huge computation and memory costs of backpropagation through the multi-step sampling chain. Experiments on the D4RL benchmark show that OFQL surpasses all other included baselines in overall performance.

### Strengths
1. The method shows empirical advantages in policy performance, training speed and inference time. 
2. The paper is easy to follow.

### Weaknesses
1. The proposed method lacks novelty. The only main difference between the proposed method and DQL is replacing the diffusion loss in actor training with a MeanFlow loss.
2. The experiments are not adequate. Only results on state-based D4RL tasks are included, and no visual observation task results are reported.
3. The argument in Lines 262-264 is not clear. Flow matching cannot "in principle, enable one-step generation", as the sampling trajectory is straight only when the target distribution is a delta distribution or when rectification or similar techniques have been used. The following sentences in this paragraph are accurate.

### Questions
1. How many diffusion steps are used for the multi-step diffusion policy baselines? Is the number aligned with the original papers?
2. Can the proposed method be extended to visual observation tasks? Are there any challenges for the method in high-dimensional input scenarios?

### Soundness
2

### Presentation
3

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
This paper proposes **One-Step Flow Q-Learning (OFQL)**, a novel framework for offline reinforcement learning that reformulates Diffusion Q-Learning (DQL) within the **Flow Matching (FM)** paradigm. By learning an **average velocity field** instead of a marginal one, OFQL enables accurate **one-step action generation** during both training and inference—eliminating the need for multi-step denoising and recursive backpropagation. This design substantially improves training and inference efficiency while maintaining, and even improving, performance. The authors demonstrate strong results across D4RL benchmarks.

### Strengths
* **Clear conceptual advancement:** Reformulating DQL under the flow-matching framework and introducing an average velocity field is a novel and elegant idea that directly addresses the core inefficiency of multi-step denoising.
* **Simplicity and effectiveness:** Unlike prior one-step approaches that depend on auxiliary modules or policy distillation, OFQL remains conceptually clean while achieving superior results.
* **Strong empirical results:** The method outperforms DQL and other diffusion-based baselines by a significant margin on D4RL, demonstrating both **efficiency** and **robustness**.
* **Illustrative toy example:** The toy experiment effectively clarifies the intuition behind the average velocity field and supports the main claim.
* **Readable and well-organized:** The paper is well-written, clearly structured, and easy to follow even for readers not deeply familiar with flow-matching methods.

### Weaknesses
* The theoretical justification for why learning an **average velocity field** leads to better one-step performance could be elaborated further. Currently, the paper provides an intuitive explanation but lacks a deeper analytical connection to diffusion dynamics.

### Questions
1. Could the authors provide a more formal justification for why **average velocity learning** preserves accuracy in one-step action generation?
2. Are there scenarios (e.g., highly multimodal action distributions) where the **average velocity** assumption might underperform?

Typo: citation in line 151

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To overcome the limitations of DQL, the paper proposes replacing the multi-step denoising policy used in training and inference with a one-step denoising policy. Unlike other one-step approaches that require an auxiliary teacher network for distillation, the paper adopts a mean-flow policy that directly approximates the denoising process. The proposed method demonstrates strong empirical performance and improved efficiency.

### Strengths
The method is simple, clear, and effective. By replacing only the diffusion policy component with the mean-flow policy, the approach achieves both higher sampling efficiency and competitive performance. The toy example nicely illustrates the advantage of reparameterizing from $v$ to $u$, providing a clearer intuition for the underlying mechanism.

### Weaknesses
Given that mean-flow generative modeling has already shown strong one-step FID results on image generation tasks, it would be valuable to see this approach applied to more complex environments beyond D4RL, such as robotic control or high-dimensional decision-making settings.

### Questions
The model performs worse on the Kitchen and AntMaze-Large-Diverse tasks, which are relatively more challenging within the D4RL benchmark. Do the authors have any insights into these results?
Could it be that the mean-flow policy limits exploration during training, leading to reduced performance on tasks requiring greater stochasticity?

### Soundness
3

### Presentation
3

### Contribution
3
