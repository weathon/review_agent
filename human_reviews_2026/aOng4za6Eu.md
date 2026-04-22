# RILe: Reinforced Imitation Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Acquiring complex behaviors is essential for artificially intelligent agents, yet learning these behaviors in high-dimensional settings, like robotics, poses a significant challenge due to the vast search space. There are three main approaches that address this challenge: 1. Reinforcement learning (RL) defines a reward function, which requires extensive manual effort, 2. Inverse reinforcement learning (IRL) uncovers reward functions from expert demonstrations but relies on a computationally expensive iterative process, and 3. Imitation learning (IL) directly compares an agent's actions with expert demonstrations; however, in high-dimensional environments, such binary comparisons often offer insufficient feedback for effective learning. To address the limitations of existing methods, we introduce RILe (Reinforced Imitation Learning), a framework that learns a dense reward function efficiently and achieves strong performance in high-dimensional tasks.
Building on prior methods, RILe combines the granular reward function learning of IRL and computational efficiency of IL.
Specifically, RILe introduces a novel trainer–student framework: the trainer distills an adaptive reward function, and the student uses this reward signal to imitate expert behaviors. Uniquely, the trainer is a reinforcement learning agent that learns a policy for generating rewards. The trainer is trained to select optimal reward signals by distilling signals from a discriminator that judges the student's proximity to expert behavior. We evaluate RILe on general reinforcement learning benchmarks and robotic locomotion tasks, where RILe achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel trainer–student framework that unifies reinforcement learning, inverse reinforcement learning, and adversarial imitation learning. Instead of relying on a static discriminator-based reward, RILe employs a trainer agent that learns a reward policy through reinforcement learning, guided by a discriminator that evaluates expert-likeness. This enables the reward function to be learned on-the-fly, in parallel with the student’s policy, producing adaptive, temporally coherent rewards that evolve as the student improves. The approach achieves state-of-the-art performance across complex continuous control benchmarks like MuJoCo and LocoMujoco while maintaining IRL-level reward quality with AIL-level efficiency. RILe’s main contributions are: (1) a dynamic reward-learning mechanism via a trainer policy, (2) a unified and efficient formulation of imitation learning, and (3) empirical and theoretical evidence that adaptive reward shaping enhances robustness and sample efficiency.

### Strengths
The paper is original, introducing a novel trainer–student paradigm that transforms reward learning in imitation learning from a static, adversarial process into a dynamic, cooperative one. The approach creatively integrates ideas from RL, IRL, and AIL into a unified framework that learns both policy and reward jointly and efficiently. The technical quality is strong, with clear theoretical motivation, sound algorithmic formulation, and comprehensive experiments demonstrating consistent improvements in performance, robustness, and efficiency. The paper is also well-written and clear, effectively conveying complex multi-agent learning dynamics through intuitive explanations and visualizations. In terms of significance, RILe meaningfully advances the field by addressing long-standing issues of myopic reward learning and instability in imitation learning, offering a scalable, general framework applicable to high-dimensional control and potentially to broader meta-reward learning problems.

### Weaknesses
While the paper presents a compelling and well-executed framework, its core innovation—training the reward function via reinforcement learning—may be seen as an incremental extension rather than a fundamentally novel concept. Empirically, while results are strong, the analysis of reward dynamics remains primarily qualitative; quantitative metrics of reward adaptivity or temporal consistency would better substantiate the claimed non-myopic behavior. Finally, the computational overhead and stability characteristics of jointly training three networks (student, trainer, discriminator) are not fully explored, and an ablation on trainer capacity or update frequency could clarify the method’s efficiency–stability trade-offs.

### Questions
1.Quantify Non-Myopic Reward Behavior:
A central claim is that RILe’s trainer produces temporally coherent, non-myopic rewards. Could the authors provide a quantitative metric or visualization (e.g., temporal correlation of rewards with long-term returns, or trajectory-level variance analysis) to empirically demonstrate this adaptivity beyond qualitative plots?

2.Analyze Computational Cost and Stability:
Since RILe trains a student, trainer, and discriminator simultaneously, what is the computational overhead compared to GAIL/AIRL? How sensitive is training to hyperparameters like the trainer’s update frequency, learning rate, or capacity? An ablation study would strengthen claims of efficiency and scalability.

3.Theoretical Clarifications:
The paper argues that the trainer’s reward cannot be represented as a static transformation of the discriminator. Could the authors include a brief proof sketch or intuition in the main text (not just the appendix) to make this key point more accessible?

### Soundness
4

### Presentation
4

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
This paper proposes RILE, an algorithm for adversarial imitation learning along the lines of GAIL. The main novelty is the proposal of a trainer agent, in addition to the typical discriminator and student (imitation)  policy. The trainer’s role is to provide an adaptive reward for the student, to allow the student to learn more efficiently from the discriminator’s classification-based binary reward signal. RILE is evaluated on classic Mujoco and Mujoco locomotion benchmark tasks.

### Strengths
- RILE leads to small but broad improvement across the board in Mujoco+
- In addition to the positive core result, there are a good number of experiments exploring the properties of various components of RILE (e.g. the reward function comparison (Fig 4), the impact of the function transform for the trainer agent's reward (Sec 5.1), the comparison of RILE to diffusion-based discriminators (Sec 5.5), and the study of the impact of noise). 
- RILE leads to significantly improved robustness to noise & covariate shift compared to GAIL (App A.1) - I think this is a pretty strong and interesting result, much more so than the supplemental analysis results presented in the main paper (e.g. the comparison to DRAIL). I suggest presenting this in the main paper and expanding the analysis in this direction.

### Weaknesses
- **The existing results in the paper do not fully convince me that the method is sound.** The main result (Fig 4) presents relatively small empirical improvements in Mujoco, that in my view, do not justify the significant increase in algorithm complexity that RILE presents. Unstable multi-agent dynamics are already a challenge that adversarial imitation learning methods must cope with without introducing a third agent. The fact that various empirical tricks (e.g. freezing the trainer, tuning the exploration of the learner) are necessary to make RILE work, only further strengthens this opinion. 
    - For this paper, theoretical analysis would go a long ways towards convincing me of the soundness of the method. One questions I have is: (1) Can we describe the behaviors of the trainer, student, and discriminator at the Nash equilibrium (i.e. if all three agents have converged to their respective optimal policies)
        
- **Missing experiment: train GAIL/AIRL with the transformed discriminator reward directly.** The core idea is introducing a trainer agent that optimizes for a reward function consisting of a transformation of the discriminator-based reward. The paper argues that the trainer agent's behavior is fundamentally different from directly providing the discriminator reward because it optimizes the *long term* discriminator reward. While this is transparently true from the objectives alone, it's not immediately clear that optimizing the long term discriminator reward is good (or why it would be good). Thus, there should be an empirical comparison between RILE and training the closest baseline (so, either GAIL or AIRL) with the transformed discriminator reward directly. 

- **Insufficient contextualization with related work.** Currently, only the related work in IL and IRL is discussed. These works form the foundation of the paper, so indeed, they are related. However, these topics are already covered in the introduction and background. The paper would be better positioned with respect to the literature by including a discussion of the following topics:     
    - Reward shaping literature: for example, [https://arxiv.org/abs/2103.09159](https://arxiv.org/abs/2103.09159), [https://arxiv.org/pdf/2104.06687](https://arxiv.org/pdf/2104.06687))
     
    - Automated experience replay type algorithms, which also introduce another agent whose job is to improve the learning of the main agent (e.g. [https://www.ijcai.org/proceedings/2019/0589.pdf](https://www.ijcai.org/proceedings/2019/0589.pdf))
        
    - Automated Curriculum learning with RL (e.g. [https://arxiv.org/abs/1812.00285](https://arxiv.org/abs/1812.00285))

### Questions
There are some questions embedded in the text above. Here are some additional questions. 

- Why do you compare against BCO and GAIfO? This is not an imitation learning from observation setting.
    
- Why does DRAIL-GAIL do slightly worse than GAIL (Fig 6d)? What is the number of seeds used for this comparison?

Here are also some typos / small writing issues I noticed. 

- The end of paragraph 2 in the intro (introducing ‘AIRL’) is redundant with the part of the 3rd paragraph introducing AIL
    
    - By the way, using AIRL as an acronym for the general problem of adversarial inverse RL collides with the algorithm AIRL (Fu et al. 2018).  
        
- Comment on Table 1: there should be a  pointer to the tables in the Appendix with the standard errors for Table 1 in the main text of the paper.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces ​​Reinforced Imitation Learning (RILe)​​, a novel framework that addresses the challenge of learning from expert demonstrations in high-dimensional environments. The core problem is that existing methods face a trade-off: ​​Inverse Reinforcement Learning (IRL)​​ provides dense, nuanced reward signals but is computationally expensive due to its iterative nature, while ​​Adversarial Imitation Learning (AIL)​​ methods like GAIL are efficient but often provide sparse, binary rewards that offer poor guidance.
RILe's key innovation is a ​​trainer-student framework​​ where both the reward function and the policy are learned simultaneously via reinforcement learning.

This setup allows the trainer to learn a dynamic, context-sensitive reward function that adapts to the student's current proficiency, combining the dense reward learning of IRL with the single-loop efficiency of AIL. Extensive experiments on MuJoCo and LocoMujoco benchmarks show that RILe achieves state-of-the-art performance, particularly in high-dimensional tasks.

### Strengths
​​1. The core idea of using an RL agent to learn a reward-generating policy is highly original and represents a paradigm shift from existing methods.

​​2. The paper is backed by an extensive set of experiments covering ablation studies, computational analysis, fairness comparisons, and performance evaluations on diverse benchmarks. The results are consistently strong.

​​3. By achieving high performance with significantly better computational efficiency than IRL methods, RILe offers a more practical and scalable solution for imitation learning in complex domains.

4. ​​The paper provides deep analysis beyond task performance, including visualizations of reward landscapes and dynamics, which offers compelling evidence for the proposed mechanism.

### Weaknesses
1. The paper acknowledges that training the three-component system (student, trainer, discriminator) introduces stability challenges. While strategies like freezing the trainer are mentioned (and detailed in Appendix B), the main text lacks a discussion on how sensitive the method is to hyperparameters related to this stability (e.g., the frequency of freezing). A more quantitative analysis of this sensitivity would be helpful.

2. ​​The comparison is excellent but could be even more compelling by including a "naive" baseline of training multiple independent RL agents with different seeds. This would help isolate the improvement due to the coordinated trainer-student learning versus simply the capacity to learn multiple policies.

3. ​​While the method is intuitively well-motivated and empirically validated, a more formal theoretical analysis of the convergence properties of the two-level RL system would further strengthen the contribution.

4. While the trainer-student architecture is intuitively reasonable and novel, there are works that also adopt similar teacher-student architecture to address the challenges in learning from demonstrations. For example, [1] uses the teacher-student architecture to facilitate the reward learning in Inverse RL. If the settings are the same, then baseline comparison might be necessary. Otherwise, discussion of the difference of the settings or ideas would be helpful to make this paper more comprehensive.

[1] Wan, Z., Wu, J., Yu, X., Zhang, C., Lei, M., An, B., & Tsang, I. (2025). FM-IRL: Flow-Matching for Reward Modeling and Policy Regularization in Reinforcement Learning. arXiv preprint arXiv:2510.09222.

### Questions
1. The strategies in Appendix B (freezing the trainer, different buffer sizes) are crucial for stable training. How sensitive is RILe's final performance to the specific timing of the trainer freeze or the ratio of buffer sizes? Did you observe a wide range of workable settings, or does it require precise tuning?

2. The learned reward policy \pi is a central artifact. Beyond the visualization in Fig. 4, did you analyze the behaviorof the trained trainer agent? For instance, does it learn to provide higher rewards for states that are "closer" to the expert's state distribution, effectively learning a potential function or a curriculum?

3. ​​The trainer's reward is currently based on the discriminator's output. Did you explore alternative reward signals for the trainer that might be less adversarial, such as a metric based on state/distribution matching (e.g., MMD, Wasserstein distance), and how that might affect the overall performance and stability?

4. There are similar works that utilize trainer-student architecture for learning from demonstrations, like FM-IRL ([1]). What is the key difference of the settings and the idea? If your settings are the same, these works should be incorporated as baselines for comparison. Otherwise, discussion or acknowledgement would be necessary.

[1] Wan, Z., Wu, J., Yu, X., Zhang, C., Lei, M., An, B., & Tsang, I. (2025). FM-IRL: Flow-Matching for Reward Modeling and Policy Regularization in Reinforcement Learning. arXiv preprint arXiv:2510.09222.


Glad to raise the scores if these concerns are addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
RILe frames imitation as a teacher–student game: a trainer RL policy shapes rewards from a discriminator while a student learns the control policy. The aim is IRL-like, adaptive feedback without IRL’s outer loop and to alleviate AIL’s near-binary/sparse signals. Strong numbers are reported on MuJoCo/LocoMujoco, with qualitative evidence that rewards adapt over training.

### Strengths
Clear, appealing reframing: optimizing a teacher policy instead of using a static discriminator reward.

Practical promise: dynamic reward shaping within a single loop; results are competitive/strong on high-dimensional control.

Modular: can plug in stronger discriminators and, in principle, other evaluators

### Weaknesses
Baseline oddity: In several tasks GAIL is the strongest/near-strongest baseline. For a 2016 method, this is atypical and raises concerns about coverage (missing stronger recent IL/IRL/offline-IL variants) and tuning fairness (architectures, budgets, entropy, normalization, replay).

Sparsity not fundamentally addressed: The trainer still derives its teaching signal from the discriminator; when the student is far from expert support, feedback can remain sparse/off-manifold. The method looks like reshaping/time-spreading discriminator feedback rather than densifying it (e.g., via expert-proximity surrogate rewards).

Complexity/stability: Joint training of student–trainer–discriminator increases tuning burden; sensitivity to freezes, entropy, buffer design is under-analyzed.

### Questions
Why is GAIL so strong here? Provide compute-normalized, architecture-matched tuning logs; if GAIL wins, diagnose (reward normalization, discriminator SNR, replay horizon, entropy, etc.).

Does RILe increase reward density when far from expert states? Report reward/advantage density conditioned on distance-to-expert (e.g., occupancy KL or NN-distance bins).

Ablate the trainer discount/entropy (incl. γ_T=0): when does the trainer collapse to a static transform of the discriminator?

Compare to densification baselines (e.g., expert-proximity surrogate rewards) and to stronger recent IL/IRL/offline-IL families (DICE/ValueDICE, DAC, IQL/CQL-fD), under matched budgets.

### Soundness
2

### Presentation
3

### Contribution
2
