# Learning Diverse Skills for Behavior Models with Mixture of Experts

- Decision: Reject
- Scores: 4, 6, 4, 4, 4

## Abstract
Imitation learning has demonstrated strong performance in robotic manipulation by learning from large-scale human demonstrations. While existing models excel at single-task learning, it is observed in practical applications that their performance degrades in the multi-task setting, where interference across tasks leads to an averaging effect. To address this issue, we propose to learn diverse skills for behavior models with Mixture of Experts, referred  to as Di-BM. Di-BM associates each expert with a distinct observation distribution, enabling experts to specialize in sub-regions of the observation space. Specifically, we employ energy-based models to represent expert-specific observation distributions and jointly train them alongside the corresponding action models. Our approach is plug-and-play and can be seamlessly integrated into standard imitation learning methods. Extensive experiments on multiple real-world robotic manipulation tasks demonstrate that Di-BM significantly outperforms state-of-the-art baselines. Moreover, fine-tuning the pretrained Di-BM on novel tasks exhibits superior data efficiency and the reusable of expert-learned knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Di-BM, a method integrating a Mixture of Experts (MoE) into the robotic learning framework, using an Energy-Based Model (EBM) to route tasks to suitable experts. The primary goal is to address the performance degradation and "averaging effect" of existing imitation learning models in multi-task settings. The authors propose an MoE structure that enables experts to specialize in sub-regions of the observation space, and which can be easily integrated into existing imitation learning methods as a "plug-and-play" module. The paper conducts comprehensive real-world experiments and analyses to show the effectiveness of the approach, demonstrating strong multi-task performance and superior data efficiency when fine-tuning.

### Strengths
- The authors performed a thorough empirical investigation, including key ablation studies on MoE formulation and the number of experts, to show how their proposed approach compares against baselines.
- Di-BM performs well across multiple real-world manipulation tasks, indicating that the proposed diverse skill learning approach can effectively address the "averaging effect" in a challenging multi-task environment.

### Weaknesses
- The discussion of related work on MoE in robotics appears to be missing several relevant, recent methods [1, 2, 3]. It would be difficult to determine the precise contribution and novelty of Di-BM without discussing how it compares to or differs from these existing approaches.
- The experiment settings could be clearer. For instance, the paper's central claim is about 'multi-task training', but the exact training procedure isn't explicitly defined. The authors should clarify if "multi-task training" means a single policy was trained on an aggregated dataset containing all 9 real-world manipulation tasks, as this is critical to evaluating the claims about mitigating the "averaging effect".
- The real-world experiment results are based on "10 trials per task". This sample size may be too small to robustly justify the effectiveness of the approach, as the difference between, for example, a 0.80 and 0.60 success rate (as seen in Table 1) might not be statistically significant. The authors should consider running more trials or, at a minimum, acknowledging this as a limitation.

[1] Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning

[2] Mixture-of-experts network with task-oriented perturbation for visual reinforcement learning

[3] Sparse diffusion policy: A sparse, reusable, and flexible policy for robot learning

### Questions
Regarding the discussion in Section 4.1:
1. The paper states that the action entropy bonus $H(\pi(a|o))$ is omitted from Equation (2). However, this term was not present in Equation (2) as defined in Section 3.2. Could the authors please clarify this discrepancy in the presentation?
2. Can the author elaborate on how the diffusion model's architecture specifically replaces the function of this entropy bonus, as it is a key design choice?

### Soundness
2

### Presentation
3

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
This paper proposes **Di-BM**, a novel Mixture of Experts (MoE) framework designed to improve multi-task performance in robotic imitation learning. The key problem it addresses is the "averaging effect" that plagues policies trained on diverse datasets. 
It utilizes Energy-Based Models (EBMs) as the gating mechanism, which models each expert's favored observation distribution.
This allows experts to autonomously specialize in sub-regions of the observation space, which correspond to primitive skills. 
The method is implemented as a plug-in for Diffusion Policy (DP) and evaluated on 9 real-world manipulation tasks. The results show significant improvements in success rates over the valinla DP baseline and MoE variants. 
Furthermore, the paper demonstrates that the pre-trained Di-BM model exhibits superior data efficiency when fine-tuned on novel tasks.

### Strengths
1. **Novel Direction**: The core idea of applying MoE to imitation learning is novel.

2. **Strong Empirical Results**: The paper provides strong quantitative evidence for its claims. The performance leap over the baselines is substantial. The real-world experiments demonstrate the potential of practical applications.

3. **Analysis of Learned Experts**: The visualization of expert activation $\pi(e|o)$ in Fig. 3 supports the hypothesis that the experts are specializing in distinct phases or primitive skills that are shared across different tasks.

4. **Data Efficiency for Fine-Tuning**: The fine-tuning experiment in Sec. 5.6 shows that the pre-trained Di-BM model adapts to new tasks more efficiently than baselines. This suggests the learned primitive skills are reusable and indicates potential for future scalability.

5. **Clarity**: The paper is well-written and easy to follow. The methodology and motivation are clearly explained, supported by effective figures and visualizations.

### Weaknesses
1. **Limited Methodological Novelty**: Although applying MoE to imitation learning is novel, the core methodology follows the prior work of MoE in RL [1].

2. **Sensitivity to Hyperparameter**: The authors mention the model's sensitivity to the KL regularization coefficient $\beta$ (Sec. 5.5, Fig. 4). A $\beta$ that is too small causes all experts to "slack off" and avoid difficult parts of the observation space. This sensitivity could present challenges when scaling Di-BM to more complex tasks or datasets.

3. **Missing Analysis of Computational Overhead**: The paper claims "minimal computational overhead" without explicit analysis. At inference, the model must first compute the expert probabilities via the gating network $g_{\phi}$ before executing the selected expert $f_{\theta}$. Especially in robotics applications, where real-time inference and reaction speed are crucial. An analysis of inference-time overhead is important.

### Questions
1. Figure 3 shows that the router utilizes different experts as a task progresses. How can we be sure the router is learning to select the *best* expert for each stage? The paper shows performance for individual experts on *full* tasks (Table 3), but it would be more convincing to see an analysis of individual expert performance on manually-partitioned task stages to verify the router's choices.

2. How does the routing strategy evolve during training? One would expect experts to have similar weights initially and then specialize. Is this specialization and the resulting routing strategy consistent across different training runs with different random seeds? In other words, do the experts learn the same set of "primitives" each time?

3. In Sec. A.6, the authors state that when using 8 experts, some are underutilized or "pruned". If these experts are simply pruned, why does the model's overall performance drop substantially (Table 8)  instead of matching the 5-expert model? Furthermore, Figure 3 also shows that some experts are underutilized. Given this, would it be possible to develop a method that dynamically adjusts or prunes the number of experts during training?

4. The paper's main novelty is adapting the Di-SkilL [1] framework from RL to IL. Could you elaborate on the specific challenges encountered and design choices in this adaptation?

I am willing to raise my score if my concerns are addressed.

[1] Celik, Onur, Aleksandar Taranovic, and Gerhard Neumann. "Acquiring diverse skills using curriculum reinforcement learning with mixture of experts." ICML, 2024.

### Soundness
3

### Presentation
4

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
This paper introduces Di-BM, a Mixture of Experts (MoE) framework designed to improve multi-task imitation learning for robotic manipulation. Traditional behavior models trained on diverse demonstrations suffer from task interference and “averaging effects.” Di-BM addresses this by associating each expert with a distinct observation distribution, modeled via an energy-based model. The gating network automatically allocates data to the most suitable expert, allowing each to specialize in a subset of primitive skills. They represent each expert using a diffusion model and evaluate their model on real-world robotic manipulation tasks.

### Strengths
- Gating visualisations nicely show that the model utilises different experts
- The methods shows strong empirical results, showing improvement on several real-world robotic tasks, verified through ablations and visualizations
- The method can be incorporated seamlessly into existing imitation learning architectures

### Weaknesses
- The paper does not mention related work that uses very similar methodology and goals, namely [1] and [2]. 

- In [1] they show that the optimal gating can be computed in closed form, making it unnecessary to learn a model in every iteration but it is sufficient to only learn a gating at the end of training. What benefit do the authors see when learning the gating?  

- Additionally, in [1] the authors establish convergence guarantees from an expectation-maximisation perspective. Do the authors think similar results could be applied here? As an additional comment, in e.g. [3] they show that the diffusion noise matching loss is a lower bound on the marginal likelihood. In that sense, the expert objective could be seen as a lower bound on a weighted maximum likelihood objective. 

- The authors only consider real-world experiments. It would be good to also include comparisons in established simulation-based benchmarks. On that note, there exists a benchmark that is designed for diverse behaviour learning, see [4]. It would be interesting to see if the proposed method improves behaviour diversity over existing methods.

- It would be good if the authors provide an ablation study showing how sensitive the performance is with respect to the KL regularisation parameter. From Figure 4, it seems like even minor changes can have a huge impact on the learned model. How difficult is it to choose the parameter? Did the authors tune the parameter per task or did they find a setting that worked for all?

[1] Information maximizing curriculum: A curriculum-based approach for learning versatile skills
[2] Curriculum-based imitation of versatile skills
[3] Towards Diverse Behaviors: A Benchmark for Imitation Learning with Human Demonstrations
[4] Understanding Diffusion Models: A Unified Perspective

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

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
This paper introduces Diverse Skills for behavior models (Di-BM), an imitation learning method that can learn policies for multi-task settings. More concretely, Di-BM employs Mixture of Experts policies that can specialize on a subset of the available training data by employing an energy-based gating network. The gating network represents an observation distribution per expert, thereby allowing each expert to specialize in observation-action regions it favors. The paper shows, on real-robot experimental tasks, that the proposed method works nicely and is able to learn multi-task policies.

### Strengths
- The paper is well-written. The reader can easily follow the story of the paper and understand the motivation behind the proposed methods
- The benefits of choosing an MoE policy representation are well-grounded by intuitive figures (e.g., Fig.3) on the training data
- The method is validated on real-robot experiments, emphasizing its strengths

### Weaknesses
- The work lacks important related works that have employed similar ideas on learning parameterized distributions over the input (observation) space [1,2]. How does the proposed method algorithmically differ from these methods? 
- Better description of the data set; i.e., which tasks are included? Are different robot types used? .... It's hard to infer the "difficulty" of a task. Commenting on the task difficulty could help the reader. 
- Although I appreciate the real-robot experiments, I believe the paper would be strengthened if Di-BM could be benchmarked on common benchmark suites, for example, such as LIBERO [3] or Robocasa [4].

[1] M. X. Li, et al. Curriculum-based imitation of versatile skills. ICRA 2023.
[2] D. Blessing, et al. Information Maximizing Curriculum: A Curriculum-Based Approach for Imitating Diverse Skills. NeurIPS 2023.
[3] O. Mees, et al. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. RA-L 2022. 
[4] B. Liu, et al. Libero: Benchmarking knowledge transfer for lifelong robot learning. NeurIPS 2023.

### Questions
- In Section 4.2 it is said that the observations $o \sim p(o)$ are sampled to cover a sufficiently large batch of observations. Does this mean the observations are sampled from the offline data set, or do the observations come from the environment by sampling from it?
If the observations come from the data set, how does the method behave if we can not sample a representative batch of observations for approximating the normalization constant? Or in general, was this a problem observed during the experiments?

- It seems that single experts are even better than the baselines from Table 2. For example, all experts from Table 3 are better on the Rearrange cup task than the Task-wise MoE model from Table 2. Is there an intuitive explanation behind it? Intuitively, I would have expected that the Task-wise MoE performs better consistently compared to the single experts. 

- Does the MoE integration also work for goal-conditioned, score-based diffusion policies[5]? 


[5] M. Reuss, et al. Goal-conditioned Imitation Learning using Score-based Diffusion Policies. RSS 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an MoE architecture and training method for robotic multi-task imitation learning. To mitigate multi-task gradient interference, a gating network is used to map the observation to the distribution of experts, and each expert policy is trained to handle a specific subset of observations. The training framework can adaptively learn the gating network using energy-based models, rather than predefine task assignment. Experiments on some real-world manipulation tasks demonstrate its effectiveness compared with previous MoE strategies.

### Strengths
- The MoE framework introduced in this paper can learn task assignment autonomously, enhancing the potential to learn from large-scale, unstructured multi-task datasets.
- Experimental results on real-robot tasks demonstrate its effectiveness. Qualitative visualizations show that the models do learn some meaningful task assignments.
- The paper writing is well organized and easy to follow.

### Weaknesses
- Limited technical contributions. After reading the method sections, it seems that most of techniques are adopted from the two prior works (Celik et al., 2022; 2024). The main difference is changing their reinforcement learning setting to the imitation learning setting.
- Lack of reproducible simulation experiments. There are a lot of multi-task imitation learning benchmarks in simulation that are widely used in prior robotic imitation learning research, like Meta-World, Libero, RoboCasa, and RoboTwin. However, the paper only uses real-world experiments, making the results less reproducible.
- The effectiveness on large-scale datasets, like Open-X-Embodiments, has not been studied.

### Questions
- What are the main technical differences between the proposed method and Celik et al., 2022; 2024?

### Soundness
2

### Presentation
3

### Contribution
2
