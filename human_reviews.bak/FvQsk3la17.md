# Langevin Soft Actor-Critic: Efficient Exploration through Uncertainty-Driven Critic Learning

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 3

## Abstract
Existing actor-critic algorithms, which are popular for continuous control reinforcement learning (RL) tasks, suffer from poor sample efficiency due to lack of principled exploration mechanism within them. Motivated by the success of Thompson sampling for efficient exploration in RL, we propose a novel model-free RL algorithm, \emph{Langevin Soft Actor Critic} (LSAC), which prioritizes enhancing critic learning through uncertainty estimation over policy optimization. LSAC employs three key innovations: approximate Thompson sampling through distributional Langevin Monte Carlo (LMC) based $Q$ updates, parallel tempering for exploring multiple modes of the posterior of the $Q$ function, and diffusion synthesized state-action samples regularized with $Q$ action gradients. Our extensive experiments demonstrate that LSAC outperforms or matches the performance of mainstream model-free RL algorithms for continuous control tasks.
Notably, LSAC marks the first successful application of an LMC based Thompson sampling in continuous control tasks with continuous action spaces.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a novel model-free RL algorithm, Langevin Soft Actor Critic (LSAC), which enhances critic learning through uncertainty estimation rather than just optimizing the policy. LSAC features three innovations: approximate Thompson sampling via distributional Langevin Q updates, parallel tempering for exploring multiple posterior modes, and diffusion-synthesized state-action samples regularized with Q action gradients. Experiments show that LSAC outperforms or matches mainstream model-free RL algorithms in continuous control tasks and is the first successful application of Langevin Monte Carlo-based Thompson sampling in continuous action spaces.

### Strengths
- This paper uses LMC combined with distributed Q to learn critic, and also uses the action gradient method to improve the diversity of samples in the buffer. It is an interesting work;
- The experiments in this paper are very comprehensive, and it has achieved very good performance in many environments, and the curve rises very quickly;
- Compared with diffusion as a policy, this paper does not have the risk of too long sampling time in actual use.

### Weaknesses
- In the related work, I strongly suggest you add the work on combining diffusion with online reinforcement learning. Although they focus on the policy, they are also directly related to your paper. Eg. Policy Representation via Diffusion Probability Model for Reinforcement Learning (Yang et al.), Learning a Diffusion Model Policy from Rewards via Q-Score Matching (Psenka et al.), Diffusion Actor-Critic with Entropy Regulator (Wang et al.), etc. Please write a paragraph in the paper to summarize.

- I don't agree with the paragraph around line 68. Methods like SAC and DSAC-T are soft policies. They balance exploration and utilization by introducing policy entropy. How can this be classified as a trick? I think their core is also heuristic, which is essentially the same category as your method.

- In Algorithm 1, there are 4 for loops for each policy update, and the time-consuming action gradient is used, but the time efficiency is not analyzed, and the performance difference brought by different action gradient steps is not analyzed. Please add time efficiency analysis: for example, when training for 8 hours at a time, compare the performance of the algorithms (draw a curve with time as the horizontal axis and TAR as the vertical axis).

- You don’t seem to have experimentally proved some of the problems caused by high-dimensional action space. Compare your current method with the previous method to see why this problem disappears (analyze the experimental results).

- Based on the citations of DSAC-T, I found a work Diffusion Actor-Critic with Entropy Regulator (Wang et al.). Please compare their performance numerically.

- I think the core work in Q-update is to combine Provable and practical: Efficient exploration in reinforcement learning via Langevin Monte Carlo with DSAC-T: Distributional Soft Actor-Critic with Three Refinements. This way of updating Q can indeed alleviate overestimation while bringing better performance, but the novelty may be less.

I'll consider score changes based on your response.

### Questions
- In line 9 of Algorithm1, you directly replaced $a_M$ with $a_{M'}$, which destroyed the Markov chain. What problems exist in fusing such data with $B_D$? Please analyze it theoretically.

- Algorithm1 Why doesn't the strategy also choose to update on $B_i$?

- Line341 Is the frequency of updating the diffusion model too low? Does the data generated in this way really have many benefits for sample distribution? Please analyze the sample distribution, as well as the performance and sample distribution results corresponding to different update frequencies.

- Why do other algorithms in Table1 have no variance? It seems that you didn't say how you calculated the data in the table.

- Do you also sample 20 samples like DSAC-T for each interaction with the environment?

- In the high-dimensional action space task Humanoid-v3, why is your performance much lower than DSAC-T?

### Soundness
3

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
3

### Summary
This paper introduces a new model-free RL algorithm Langevin Soft Actor Critic (LSAC). It uses Langevin Monte Carlo (LMC) for updating the Q function to help exploration and improve performance, the authors pointed out several key challenges in taking this approach and propose solutions that show benefit to performance in the ablations. When compared to a number of other baseline RL algorithms on MuJoCo and DMC benchmarks, LSAC shows stronger overall performance. This is the the first successful application of a Langevin Monte Carlo (LMC) based Thompson sampling in RL continuous control setting.

### Strengths
**originality**

- authors study and show how to do Langevin Monte Carlo (LMC) based Thompson sampling in RL continuous control setting properly and achieve strong performance, this is a novel contribution. 

**quality**

- overall good quality of presentation and writing. 
- fairness of comparison: I appreciate that the authors discuss comparison fairness and provide full details of the hyperparameter setting in the appendix and give source code. 

**clarity**
- overall clear 

**significance**
- strong results: Figure 1, table 1 show that proposed method has fairly consistent stronger results compared to other baselines. 
- results show how to properly use Langevin Monte Carlo (LMC) based Thompson sampling in DMC and MuJoCo. 
- ablations: it is good to have ablations to understand the effect of different design choices.

### Weaknesses
Some of the arguments made in the paper seems not adequately supported, some more explanations or experiments will help:

- Line 346: the authors argue LSAC has good performance while being simple in implementation. Why is it simple in implementation? It seems to me it has a number of additional components/hyperparameters compared to a baseline such as SAC, and these extra components seem not trivial to implement correctly. The complexity of the method seems a weakness to me. 
- Line 460: The authors argue that Figure 5 shows LSAC as more stable. I don't feel convinced. The fluctuation in its performance seems similar to DSAC-T, and even when compared to DIPO, LSAC has some wild changes in Figure 5 (c). 
- Is there a comparison on computation efficiency? With the additionl components in the proposed method, how much computation/wall clock time does it take to finish a run compared to other algorithms? 
- Although a lot of details and discussion are given to show an effort to compare to alternative algorithms in a fair manner, it is a bit unclear to me how 2 things are made fair: (1) network capacity, the proposed approach seems to have more critics and additional networks (such as in the diffusion part), how does it compare to other baselines in terms of e.g. number of parameters? And (2) UTD ratio, if the proposed approach is doing more updates, is it still fair to compare it to baselines that take limited updates? I understand that some of them might not work well with more updates due to instability, but in that case, one might argue it would be good to compare to algorithms that do benefit from ensemble of networks and higher UTD such as REDQ?

### Questions
- It seems to me a main part of the strong performance comes from a better exploration technique. In the baselines you compared to, are some of them focused on exploration techniques? If we replace the exploration in proposed method with some other recent exploration technique, will we obtain a similar strong performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Authors of this paper introduces a model-free online RL algorithm named Langevin Soft Actor-Critic (LSAC). The algorithm incorporates  Langevin Monte Carlo (LMC) updates for critic learning with parallel tempering and action refinement on diffusion-synthesized trajectories. it also utilizes distributional Q objective, allowing diverse sampling from multimodal Q posteriors. Through experiments on MuJoCo benchmarks, the algorithm is shown to outperform/match the performance of SOTA baselines.

### Strengths
Novel ideas of combining LMC updates, parallel tempering, and diffusion-synthesized action refinement in the LSAC algorithm.
By efficiently exploring state-action spaces through LMC based Q updates and multimodal Q posteriors with parallel tempering, LSAC has the potential to learn a better policy in complex environments.
LSAC demonstrates competitive or superior performance against several established baselines on standard continuous control benchmarks.
The methodological choices in LSAC are grounded in theoretical analysis presented in the paper.

### Weaknesses
The implementation seems very complex even by reading through the pseudo code. 
The computational cost of LSAC lacks discussion in the paper.
Additional parameters and hyper-parameters are introduced, including 10 parallel critics, additional diffusion buffer, inverse temperature, etc, and ablation study shows that LSAC is sensitive to some of the parameters, which may limit its scalability to different applications.

### Questions
As all the techniques introduced and incorporated into LSAC and complexity of the implementation, any analysis why LSAC is not working better than DSAC in the humanoid environment (which is also regarded as the one of the most complex environments among the benchmarks) ? 
What is the impact of each key component (LMC updates, parallel tempering, diffusion models) on the performance of LSAC? Have you tried to remove one of these components and see the affect the overall effectiveness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper aims to address inefficient exploration and bad sample usage in actor-critic algorithms for continuous control problems. They propose Langevin Soft Actor-Critic (LSAC) which uses adaptive Langevin Monte Carlo (aSGLD) to sample from a Q-value distribution, improving exploration by capturing uncertainty. A diffusion model is also applied to generate diverse synthetic state-action pairs, refined by Q-action gradients to focus on high-reward areas, thereby enhancing sample efficiency. The algorithm demonstrates improved or comparable empirical performance on MuJoCo and DeepMind Control Suite benchmarks compared to standard RL algorithms. An ablation study is also provided.

### Strengths
1. They propose a new method that models each Q(s, a) value as a distribution. Then, an LMC-based sampling algorithm is used to sample the posterior of the Q value, which encourages exploration. The accordingly distributional Bellman operator is used.
2. They propose using a diffusion model to generate more state-action pairs to update the critic function, where each pair's action is replaced with its Q action gradient updated ones. The use of generative models helps increase the sample efficiency.
3. Thorough experiments are conducted, with the algorithm implemented on two benchmarks, demonstrating improved or comparable empirical performance.

### Weaknesses
My primary concerns are with novelty and clarity.

**Contribution:**
Based on my reading of this submission, this work extends Langevin Monte Carlo (LMC) [2] from discrete to continuous action spaces (e.g., MuJoCo tasks). It builds on [2] with the same critic modeling as [1] and integrates a diffusion model for generating synthetic replay data [4].

The main novelties introduced in this work seem to include: 
1. extending the use of multiple Q-values from Langevin-gradient-based Bayesian neural learning [3] to RL.
2. using gradient ascent-refined actions in diffusion-based policy optimization [5] to improve data diversity in experience replay [4].
3. reducing the number of aSGLD updates for the critic in [2] from $O(K)$ to a single iteration. However, I am uncertain whether this improvement arises from the diffusion model generating more data, which increases the size of batch in Algorithm 2.

Despite the outlined contributions, I have several **questions** regarding the arguments and claims made in the submission:
1. the paper claims multiple critics act as parallel tempering to address slow mixing in LMC. However, LMC is described as an optimization rule, not a distribution approximation. Does the issue exist in the LMC-inspired critic update? And does the gain in Figure 6 come from parallel tempering or simply reduced TD target variance, as in SAC, since all critics share the same temperature?
2. from the ablation study, the distributional critic modeling seems most impactful. Is this due to exploration from the variance in the Gaussian Q-values distribution? My previous question regarding the distributional TD loss is that: for the distributional TD loss $L(\psi)$ and normal deterministic TD loss $TD_c(\psi)$, $L(\psi) = \frac{TD_c(\psi)}{\sigma_{\psi}} + \log(\sigma_{\psi})$ as shown in Eq(15), higher $\sigma_{\psi}$ increases exploration (second term) but reduces TD loss (first term). Therefore, I am curious whether this $\sigma_{\psi}$-$TD_c(\psi)$ trade-off inside the training of $L(\psi) $ is related to the improvement in Figure 3,4. 


**Presentation:**
 many techniques or claims are applied without sufficient interpretation or explanation. For example, in section 3, the paragraph **Distributional Critic Learning with Adaptive Langevin Monte Carlo** mostly explains [1] and [2], and Equation (9) is the same as lines 9-11 of Algorithm 2 in [2]. It is better to move them to the preliminary section. As a reader, it is hard to understand the paper without reading other references. When presented in the Algorithm Design section, I would expect the work to provide a self-contained explanation of its logic and methodology.

I truly appreciate that the author included comparisons of running time and parameter numbers. But when a paper adapts old algorithms to a new task, I would expect this work can provide deeper analysis and insights specific to the task at hand. This would enhance the impact.


[1] Jingliang Duan, Wenxuan Wang, Liming Xiao, Jiaxin Gao, and Shengbo Eben Li. DSAC-T: Distributional soft actor-critic with three refinements.

[2] Haque Ishfaq, Qingfeng Lan, Pan Xu, A Rupam Mahmood, Doina Precup, Anima Anandkumar, and Kamyar Azizzadenesheli. Provable and practical: Efficient exploration in reinforcement learning via Langevin Monte Carlo. In The Twelfth International Conference on Learning Representations

[3] Rohitash Chandra, Konark Jain, Ratneel V Deo, and Sally Cripps. Langevin-gradient parallel tempering for bayesian neural learning

[4] Cong Lu, Philip Ball, Yee Whye Teh, and Jack Parker-Holder. Synthetic experience replay. Advances in Neural Information Processing Systems, 36, 2024.
 
[5] Long Yang, Zhixiong Huang, Fenghao Lei, Yucun Zhong, Yiming Yang, Cong Fang, Shiting Wen, Binbin Zhou, and Zhouchen Lin. Policy representation via diffusion probability model for reinforcement learning.

### Questions
see weakness

### Soundness
2

### Presentation
1

### Contribution
2
