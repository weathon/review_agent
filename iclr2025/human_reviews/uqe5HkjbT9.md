## Human Reviewer 1

### Summary
This paper studies multi-task multi-agent reinforcement learning. It proposes a trajectory clustering-based approach, where trajectories from different environments or tasks are clustered based on similarities in the latent space. The clustering label is then used as an additional input for the policy。

### Strengths
Multi-task multi-agent reinforcement learning is an important challenge for MARL. The proposed trajectory clustering approach is innovative and offers a novel solution.

### Weaknesses
[1] The primary concern is whether the clustering approach is truly necessary. From my understanding, the purpose of clustering is merely to provide a label as additional information for the policy. It seems that the proposed approach only evaluates tasks seen during training, without considering unseen tasks during test time. Does this approach generalize to unseen tasks? If so, could the authors provide experimental results to support this? If not, why is the clustering necessary to provide a label, since we could easily have labels for the tasks during training and use it to train the trajectory class predictor? Perhaps simply using the task ID as a one-hot vector could replace the entire clustering module, as this has been effective in some single-agent RL work, see 1'.

[2] The notations in Section 3 are unclear and could lead to confusion. Please consider improving them.

[3] I think some discussions and comparisons with Updet ,see 2', is necessary, since Updet could be thought of a multi-task marl algorithom



1' Hansen, N., Su, H., & Wang, X., 2023. TD-MPC2: Scalable, robust world models for continuous control. arXiv preprint arXiv:2310.16828

2' Hu, S., Zhu, F., Chang, X., & Liang, X. (2021). Updet: Universal multi-agent reinforcement learning via policy decoupling with transformers. arXiv preprint arXiv:2101.08001.

### Questions
[1] It seems that the policy module does not propagate gradients back to the clustering module. I wonder why these two modules are not coupled, allowing the clustering module to have some trainable components that could help extract useful information for the policy?

[2] The paper emphasizes the coverage loss significantly. However, why is distributing throughout the embedding space considered beneficial?

[3] In equation (7), why do you directly sum the embedding vectors? Does this make sense, or could there be better ways to handle this?

[4] How do you manage different observation and input dimensions at the same time for different tasks? 

[5] "To evaluate performance, we consider the overall return value instead of the win-rate, as the learned policy may be specialized for specific goal tasks while being less effective for others in multi-goal tasks." Could you elaborate further on this point?

[6] For comparison with QMIX, do you use PyMARL2 (see 1') or PyMARL? PyMARL2 fine-tunes the parameters for QMIX and shows better performance. For example, in Figure 10, in the 6h_vs_8z scenario, based on my experience, QMIX could achieve a better return at 4M environment steps.


1', Hu, J., Jiang, S., Harding, S. A., Wu, H., & Liao, S. W. (2021). Rethinking the implementation tricks and monotonicity constraint in cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2102.03479

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a novel multi-agent reinforcement learning framework called TRAMA (TRajectory-class-Aware Multi-Agent reinforcement learning) to address the issue of policy generalization in multi-goal tasks. TRAMA introduces the idea of trajectory clustering, where trajectories are embedded and clustered to enable agents to recognize the type of task they are experiencing based on their trajectories, and use this prediction as additional information for decision-making, thereby learning policies that can adapt to different goal tasks. The main components of TRAMA include: 1) constructing a quantized latent space using a modified VQ-VAE to generate trajectory embeddings; 2) performing trajectory clustering based on the embeddings; 3) training a trajectory-class predictor and class representation model to generate trajectory-class-aware policies. TRAMA achieves better performance than multiple baseline methods on various multi-goal tasks in StarCraft II.

### Strengths
1. TRAMA effectively addresses the policy generalization problem in multi-goal tasks. Traditional MARL methods often learn policies that are only effective for specific tasks and lack generalizability. By enabling agents to predict trajectory classes and learn class-related policy representations, TRAMA significantly improves the adaptability of policies, which is validated by the experimental results showing that TRAMA achieves higher average returns than other methods on multiple multi-goal tasks (Figure 6).
2. The modified VQ-VAE quantization method better learns trajectory embeddings. By considering both time steps and trajectory classes in the coverage loss, the quantized vectors are more evenly distributed in the embedding space of different trajectory classes (Figure 3), providing a good foundation for subsequent clustering.
3. Trajectory clustering and the class predictor allow agents to accurately infer the trajectory class from local observations. Experiments show that after training, the agents' prediction accuracy for trajectory classes can stabilize at a high level (Figures 7, 8), enabling the policy to make class-related decisions based on the class representation.
4. Ablation studies and parameter analysis investigate the impact of main modules and hyperparameters (Figures 11, 12, 18), enhancing the interpretability and reproducibility of the method.

### Weaknesses
1. The paper does not theoretically analyze the superiority of TRAMA. The authors only demonstrate the advantages of TRAMA over other methods from experimental results, lacking rigorous theoretical derivation and complexity analysis. For example, does the introduction of trajectory clustering and class prediction modules significantly increase the computational overhead? No quantitative analysis is provided.
2. The experimental evaluation is not comprehensive enough. The paper only constructs and tests 4 multi-goal tasks in the StarCraft II environment. However, there are many ways to compose multi-goal tasks. Can these 4 tasks represent the main scenarios in the real world? Moreover, the authors only compare TRAMA with a few representative MARL algorithms. Are there other multi-goal generalization methods that are not included in the comparison? More benchmarks and baselines would make the experimental results more convincing.
3. The number of trajectory classes is predetermined, and the paper does not provide a principled method for setting the classes. It only discusses the impact of different class numbers on performance in the ablation study. So how to adaptively determine the optimal number of classes based on task characteristics? What are the criteria for class division? These questions require further exploration.
4. This paper only focuses on tasks with discrete action spaces. Is it equally applicable to continuous action spaces? Additionally, in other types of multi-agent tasks, such as mixed cooperative-competitive tasks, is trajectory clustering still effective? These issues need to be analyzed and validated in future work.

### Questions
1. The paper does not theoretically analyze the superiority of TRAMA. The authors only demonstrate the advantages of TRAMA over other methods from experimental results, lacking rigorous theoretical derivation and complexity analysis. For example, does the introduction of trajectory clustering and class prediction modules significantly increase the computational overhead? No quantitative analysis is provided.
2. The experimental evaluation is not comprehensive enough. The paper only constructs and tests 4 multi-goal tasks in the StarCraft II environment. However, there are many ways to compose multi-goal tasks. Can these 4 tasks represent the main scenarios in the real world? Moreover, the authors only compare TRAMA with a few representative MARL algorithms. Are there other multi-goal generalization methods that are not included in the comparison? More benchmarks and baselines would make the experimental results more convincing.
3. The number of trajectory classes is predetermined, and the paper does not provide a principled method for setting the classes. It only discusses the impact of different class numbers on performance in the ablation study. So how to adaptively determine the optimal number of classes based on task characteristics? What are the criteria for class division? These questions require further exploration.
4. This paper only focuses on tasks with discrete action spaces. Is it equally applicable to continuous action spaces? Additionally, in other types of multi-agent tasks, such as mixed cooperative-competitive tasks, is trajectory clustering still effective? These issues need to be analyzed and validated in future work.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces a new multi-agent reinforcement learning framework called TRAMA which aims to address the challenge of multi-goal tasks. The multi-goal tasks are especially challenging since they require agents to adapt to goals without relying on task-specific policies. TRAMA incorporates the vector quantize variational autoencoder (VQ-VAE) approach to create a discrete space for trajectory embeddings. With clustering, the embeddings provide information on which trajectory class agents are experiencing and then can be used to improve the decision making process. Experiments are conducted over a suite of SMACv2 tasks and the proposed method achieves superior performance.

### Strengths
The paper is well written and easy to follow. The approach to incorporate VQ-VAE for identifying goals is a novel branch of study in MARL. The authors conducted extensive experiments demonstrating the capability of the model. Notably the paper provides insightful visualizations, such as the VQ codebook visualization, which makes the result more convincing. Overall the proposed method achieves a significant improvement in performance in many different settings in SMACv2.

### Weaknesses
1. One major concern I have is regarding the novel contribution of TRAMA as compared to the LAGMA framework. Since LAGMA proposed adapting the VQ-VAE framework to multi-agent RL, the paper may benefit from a more detailed discussion on the unique contributions brought by TRAMA. 
2. The motivation behind applying k-means clustering to the VQ-VAE embeddings requires further clarification. Since the embeddings are already quantized, it would be helpful to further explain why the k-means clustering is still necessary 
3. Following the previous point, the authors may consider providing additional insights into the stability of the k-means clustering process. For me while the result in figure 13 is indeed encouraging, it is in the meantime counterintuitive to see why VQ-VAE embeddings consistently form clear clusters. Additional explanation would strengthen the arguments.

### Questions
1. Since the clustering process is trained on a dataset, would the model be able to detect or adapt to out-of-distribution trajectories which better simulate real-life applications? 
2. Given the unsupervised nature of VQ-VAE learning and trajectory clustering, how can the model ensure that the resulting clusters are distinct and meaningful with respect to specific tasks?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper focuses on utilizing multi-agent reinforcement learning (RL) for solving the multi-goal task. An approach of TRajectory-class-Aware Multi-Agent reinforcement learning (TRAMA) is proposed by generating the trajectory embeddings, making trajectory clustering, learning trajectory-class-aware policy with trajectory-class representation. Finally, experiments were conducted on SMACv2 domain, including four modified tasks (multi-goal tasks) and two conventional SMACv2 tasks, and compared TRAMA with QMIX, RODE, LDSA,  MASER, QPLEX, EMC, and LAGMA.

### Strengths
- The multi-agent RL has the potential to solve more complex tasks. This paper aims to use multi-agent RL to solve the multi-goal task. The main contribution of this paper is by proposing a framework with doing clustering and prediction on the latent space and learning the policies based on the generated trajectory-class representation. Although the main idea is straightforward, there is some novelty in the algorithm design.

### Weaknesses
- The presentation should be improved. What's the formal definition of the multi-goal task? Why multi-agent RL can solve this better than the single-agent RL? What's the relationship between the multi-goal and trajectories? In Figure 2, what's the difference between "$s_t$" and "$o_t^i$"? what are the goals? How different are they? How to formulate them into the objective? How agents interact with each other and the environment, and learn various goals? Is the whole training process end-to-end? or are there some pre-training process? I do believe there needs some clarifications on the problem background, assumptions, and details of the methodology, currently it's not clear to me.
- Empirical evaluation: what's the difference between the modified tasks and conventional tasks on SMACv2? Are they all multi-goal settings? What are these goals? In the results shown in Figure 6, it looks the improvement of TRAMA over others is marginal. For Figures 7 and 8, it's better to report the numbers instead of showing the curves. In conventional tasks on SMACv2, it looks TRAMA barely didn't have the advantage over others. If this is the case, how could TRAMA work better on multi-goal tasks than other methods? It feels like the methodology is quite complex, but the benefit is small. Overall, the current conclusion is still vague to me.

### Questions
- see my comments in the weaknesses.
- Did the replay buffer store the trajectories only, instead of transitions?
- How many agents were used? what are the number of trajectory classes and centroid? How did you select them?
- How did you collect the trajectories? What's size of the replay buffer?

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
5

### Confidence
3