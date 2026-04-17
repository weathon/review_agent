# Trajectory Generation with Conservative Value Guidance for Offline Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Recent advances in offline reinforcement learning (RL) have led to the development of high-performing algorithms that achieve impressive results across standard benchmarks. However, many of these methods depend on increasingly complex planning architectures, which hinder their deployment in real-world settings due to high inference costs. To overcome this limitation, recent research has explored data augmentation techniques that offload computation from online decision-making to offline data preparation. Among these, diffusion-based generative models have shown potential in synthesizing diverse trajectories but incur significant overhead in training and data generation.
In this work, we propose Trajectory Generation with Conservative Value Guidance (TGCVG), a novel trajectory-level data augmentation framework that integrates a high-performing offline policy with a learned dynamics model. To ensure that the synthesized trajectories are both high-quality and close to the original dataset distribution, we introduce a value-guided regularization during the training of the offline policy. This regularization encourages conservative action selection, effectively mitigating distributional shift during trajectory synthesis.
Empirical results on standard benchmarks demonstrate that TGCVG not only improves the performance of state-of-the-art offline RL algorithms but also significantly reduces training and trajectory synthesis time. These findings highlight the effectiveness of value-aware data generation in improving both efficiency and policy performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes TGCVG, a novel generative data augmentation method designed for offline RL algorithms. TGCVG tackles two limitations of prior works: computational overhead and distribution shift. TGCVG trains Transformer instead of Diffusion models and applies conservative value guidance to mitigate those issues. Experiment results show TGCVG outperforms prior baselines.

### Strengths
- The idea is easy to follow
- Limitation of prior works and How to mitigate the limitations are clearly stated
- Strong empirical results against several baselines across diverse tasks

### Weaknesses
- It seems that $\lambda$ and $K$ should be heavily tuned across different environments and datasets. I'm not sure the method can be generalized to unseen tasks without extensive hyperparameter tuning, which is crucial in offline RL.

- It would be better to add ablation studies on different sizes of the dataset, which is crucial for evaluating the performance of generative data augmentation.

- It would be better to add ablation studies on different sizes of the generated dataset, which is also crucial for evaluating the performance of generative data augmentation. I'm also struggling to find how many transitions are augmented during the training.

### Questions
- It would be better to specify the number of generated transitions for each round, the ratio of transitions from the original offline dataset and the generated dataset during policy training, and the update-to-data (UTD) ratio for a clearer understanding.

- It would be better to visualize t-SNE visualization of generated data distributions on diverse tasks for more comprehensive understanding of the behavior.

- At first, I understood the conservative value guidance as using conservative RTG values for generating trajectories with the Transformer policy. However, it seems that we train Transformer-CQL and directly use the network for generating an augmented dataset. Is there any reason why the current method is preferred over the aforementioned method? I'm not asking about the experiment, just curious. I think we can achieve similar results when we carefully tune the hyperparameter for choosing a conservative RTG value.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a novel data augmentation method for offline RL, TGCVG, using the transformer-based augmentation policy with the dynamics model. In contrast to diffusion-based augmentation methods such as GTA [1], TGCVG incurs lower overhead during both training and dataset augmentation. Using the TGCVG-augmented dataset, offline RL policies achieved better performance than with any other augmentation baseline.

[1] Lee, Jaewoo, et al. "Gta: Generative trajectory augmentation with guidance for offline reinforcement learning." Advances in Neural Information Processing Systems 37 (2024): 56766-56801.

### Strengths
1. Well motivated

The paper is well motivated and clearly constructed. Without increasing the complexity of the policy algorithm, TGCVG improves the performance of offline RL policies by generating high-quality data guided by the conservative Q-function.

2. Improved efficiency for training and augmentation

Because the augmentation process is fully offline, its impact on decision-time efficiency is limited; nevertheless, Figure 4 indicates that TGCVG is 4× more efficient for training and augmenting the offline dataset.

3. Fresh insight about the conservatism quality of TD3BC and CQL

The authors show that Transformer-CQL generates in-distribution samples, whereas Transformer-TD3BC does not; consequently, the CQL policy cannot learn a conservative policy from Transformer-TD3BC’s augmented data.

### Weaknesses
1. Concerns about the hyperparameter sensitivity

While the authors conduct an ablation study on $\lambda$, they vary its value from 0.1 to 1, suggesting that TGCVC requires extensive hyperparameter tuning. In addition, Table 6 lists another hyperparameter, $K$, but lacks an ablation, further increasing the tuning burden for practitioners.


2. Results on larger-scale benchmarks would be beneficial

Recently, the scalability of offline RL has emerged as an important theme [2]. OGBench [3] provides a diverse suite of offline tasks with higher-dimensional state spaces and larger datasets. I understand the rebuttal window limits time and compute, but if TCGVC were corroborated by strong results on OGBench, its impact and practical value would be substantially enhanced.

[2] Park, Seohong, et al. "Horizon Reduction Makes RL Scalable." arXiv preprint arXiv:2506.04168 (2025).

[3] Park, Seohong, et al. "OGBench: Benchmarking Offline Goal-Conditioned RL." The Thirteenth International Conference on Learning Representations.

### Questions
Because the hyperparameters are the pair $(\lambda, K)$, the search space is combinatorial. To make offline RL algorithms viable in the real world, where online interaction is unavailable, we must be able to select good hyperparameter combinations purely offline; hence, simplicity in offline RL methods is crucial [4]. I therefore argue that a thorough practical guideline for joint tuning of $(\lambda, K)$ is essential for this paper.

[4] Fujimoto, Scott, and Shixiang Shane Gu. "A minimalist approach to offline reinforcement learning." Advances in neural information processing systems 34 (2021): 20132-20145.

### Soundness
3

### Presentation
2

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
This paper proposes TGCVG (Trajectory Generation with Conservative Value Guidance), a novel data augmentation framework for offline reinforcement learning (Offline RL). TGCVG consists of transformer based policy network trained using Conservative Q-learning(CQL), learned dynamics model that predicts next states and rewards to generate synthetic transitions. TGCVG guides generation process with conservative value guidance, keeping them close to data distribution while favoring high-value regions. The augmented trajectories are then mixed with the original dataset and used to train standard offline RL. Extensive experiments on D4RL benchmarks (MuJoCo, Maze2D, AntMaze) show that TGCVG improves baseline performance across domains. And produces high-quality, dynamically consistent synthetic data according to novelty, optimality, and dynamic MSE metrics.
- An LLM was used to improve writing.

### Strengths
1. Low computational cost: The proposed method achieves significantly lower training and data-generation overhead compared to diffusion-based approaches such as GTA — up to 10× faster in both stages.

2. Strong empirical performance: Comprehensive evaluation on D4RL benchmark tasks demonstrates consistent and robust performance gains against a wide range of baselines.

3. Clarity and presentation: The paper is well-written and well-organized, with clear motivation, sound theoretical reasoning, and a coherent algorithmic presentation that makes the method easy to follow.

### Weaknesses
1. Lack of evaluation on high-dimensional or robotics domains: The paper does not include results on more complex D4RL tasks such as Adroit or Kitchen, which are higher-dimensional and closer to real-world robotics scenarios.

2. Ablation on value guidance and policy choice: It would be valuable to see how performance changes with or without the conservative value guidance, or when using Decision Transformer (DT) as the policy for augmentation.

3. Missing experimental details: In Table 3, results for HalfCheetah are not presented, and experiments where Transformer-CQL itself serves as the policy appear insufficiently explored.

### Questions
Could the authors explicitly define the metrics used for novelty, optimality, and dynamic MSE?
For example, novelty seems to be measured via L2 distance, but is this metric appropriate given the domain characteristics of the datasets? A short justification or alternative metric discussion would strengthen the paper.

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
3

### Summary
The paper proposes TGCVG, a framework that combines CQL transformer with a dynamics model to generate high-quality, in-distribution trajectories for offline reinforcement learning.
By guiding trajectory synthesis with conservative value estimates, TGCVG improves both stability and sample efficiency.

### Strengths
1. The paper addresses an important problem in offline reinforcement learning—how to generate high-quality trajectories that remain in-distribution to improve policy learning stability.

2. The proposed TGCVG framework is both novel and solid, effectively leveraging conservative Q-learning to guide trajectory generation and ensure distributional consistency.

3. The experimental results are strong and impressive, showing clear and consistent improvements across multiple D4RL benchmarks.

4. The paper is well-written and clearly presented, making the motivation, methodology, and findings easy to follow.

### Weaknesses
1. The experimental section could be more comprehensive. In particular, an ablation study comparing conservative trajectory generation with standard generation would help clarify the effectiveness of the proposed mechanism.

See quetions below.

### Questions
1. How is the dynamics model trained in this work? Please clarify the training objective, data source, and other details.

2. How does the model handle terminal indicators when generating or evaluating trajectories? This part is not clearly presented in the main paper.

3. In Table 2, why does SynthER perform worse than the version without augmentation? 

4. Can the authors provide more explanation on why restricting trajectories to be in-distribution leads to better performance, especially when the dataset contains medium-expert level data because in-distribution seems to bring similar information with the original dataset.

5. Have the authors tested TGCVG on visual RL environments to examine whether the proposed framework generalizes beyond state-based tasks?

### Soundness
3

### Presentation
3

### Contribution
3
