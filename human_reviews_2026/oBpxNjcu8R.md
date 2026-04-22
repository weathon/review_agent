# CSO: Refining Robotic Policies via Skill Distribution Alignment and Skill-Grained Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 0, 6, 4

## Abstract
Discretizing continuous actions into skills using methods like VQ-VAE has emerged as a powerful paradigm for robotic manipulation.
However, the quantization errors in discretizing continuous actions yield a suboptimal training distribution for the prior, degrading its performance.
While reinforcement learning offers a path for refinement, its direct application is challenging, suffering from unstable encoder updates and a granularity dilemma in importance sampling.
To address these challenges, we introduce Cascaded Skills Optimization (CSO), a two-stage post-training framework.
First, to rectify the initial policy's suboptimal distribution, CSO employs Rejection-Sampling Supervised Fine-tuning to align the model's observation-to-skill mapping with the distribution of successful online trajectories via supervised fine-tuning.
Second, to resolve the granularity dilemma, CSO introduces Skills Policy Optimization, which computes an independent, clipped importance ratio for each skill, enabling more stable and efficient updates.
Our post-training strategy delivers highly competitive performance on challenging benchmarks like LIBERO and MetaWorld, with its effectiveness further validated on a physical robot.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Cascaded Skills Optimization (CSO), a two-stage, critic-free framework to refine skill-based robotic policies trained via imitation learning. CSO first uses Rejection-Sampling Supervised Fine-tuning (R-SFT) on successful online trajectories to stabilize the policy's observation encoder, which is then frozen. Second, it introduces Skills Policy Optimization (SPO), a novel algorithm that resolves the "granularity dilemma" by computing clipped importance ratios at the skill level rather than the token or trajectory level. The method is shown to significantly improve performance on LIBERO, MetaWorld, and real-world tasks.

### Strengths
- The method focuses on a key practical problem: refining powerful, discretized imitation learning policies.
- The two-stage design, which cleanly separates representation stabilization (R-SFT + encoder freeze) from policy optimization (SPO), is an elegant solution to the identified problem of unstable encoder updates.
- The paper is supported by a comprehensive set of experiments, including LIBERO, MetaWorld, and real-world tasks.
- The paper is well-organized.

### Weaknesses
- Clarity:
    - Sections E.2 and E.4 in the appendix appear to be identical duplicates.
    - In Figure 2, standardizing the color scheme for the same method across figures would improve readability.
    - The token-level optimization baseline is referred to as "PPO" in Table 3 but "GRPO" in Figure 3. This should be standardized.
- The R-SFT stage introduces a strong, unstated assumption. The R-SFT stage requires collecting a large buffer of successful trajectories using the initial imitation-learned policy. While the authors state their experiments achieved "nearly 100%" data collection success, this assumption is not guaranteed to hold in a general property. The paper should explicitly state this limitation. If this assumption fails and sufficient successful trajectories cannot be collected, it may exacerbate the "unstable update"? In addition, the paper does not provide an ablation study on the number of successful samples required for R-SFT.

### Questions
1. Did the authors verify if DAPO's 'clip-higher' strategy is applicable to VLAs? Are the optimal clip hyperparameters for the GRPO/GSPO baselines (taken from LLMs) necessarily optimal for robotics? I suggest a "Clipping Fraction" analysis (similar to GSPO's) for all three methods (GRPO/GSPO/SPO) to provide a more conclusive ablation.
2. SPO is motivated by aligning the "optimization unit" with the "decision unit," while GSPO aligns with the "reward unit." Could the authors elaborate on why the former is more suitable for VLA, while the latter was effective for LLMs?
3. Was an iterative approach considered (e.g., looping Stage 1 and Stage 2) to progressively refine the policy and encoder? What are the reasons for choosing a fixed two-stage design over such an iterative loop?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces a method for reinforcement learning finetuning of imitation learning policies that use VQ-VAE for discretization. The idea is to first align the observation-to-skill mapping with successful online trajectories via supervised fine-tuning, and then reinforcement learning to do skill level updates.

### Strengths
Imitation learning baselines:
The paper has a lot of imitation learning baselines.

Presentation:
The method is presented and explained clearly. Figure 1 and Algorithm 1 clearly demonstrate the proposed method.

### Weaknesses
Claims are not supported:
The paper claims the VQ-VAE quantization error results in poor performance. "We argue that this initial quantization error is exacerbated by the imitation learning policy and ultimately amplified during online execution." But do not provide evidence for this. 
The paper claims "A primary issue (of RL finetuning on pretrained policy) is the unstable update of the observation encoder." There is also no experimental evidence to support this. 

Results are weak:
On the main LIBERO benchmarks, the method achieves 97.5% whereas the second best method achieves 94.2%, for a total of 3.3% improvement over 50 episodes and 3 random seeds. On MetaWorld there is a 5% improvement, and on real robot experiments there is 6% improvement. These are very marginal improvements and may be a result of statistical error. 

Ablation on RL method:
Figure 3 shows ablation for the proposed RL method compared to baselines. There are no error bars to indicate confidence and does not provide how many seeds are used and the improvement is around 2%. There is not enough evidence to claim this is better than baseline.

### Questions
Does this method work better than RL methods that directly run discrete RL on the learned discrete codes for skills such as SAQ (https://arxiv.org/abs/2310.11731) and Aquadem (https://arxiv.org/abs/2110.10149)? 

How many seeds is the ablation on policy optimization methods in Figure 3 over?

### Soundness
1

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
4

### Summary
This paper presents Cascaded Skills Optimization (CSO), a two-stage post-training framework for improving imitation-learning robotic policies based on action discretization (e.g., VQ-VAE). The authors identify two main limitations: quantization-induced distribution shift and the challenge of RL fine-tuning due to encoder instability and the granularity dilemma between token- and trajectory-level updates.
CSO addresses these issues via:
(1) Rejection-Sampling Supervised Fine-tuning (R-SFT): collects successful trajectories for IL-based fine-tuning and freezes the encoder for stability;
(2) Skills Policy Optimization (SPO): a critic-free, skill-level policy optimization method that mitigates the granularity dilemma through skill-level importance sampling.
Experiments on LIBERO, MetaWorld, and real-robot tasks show that CSO achieves state-of-the-art performance with strong robustness and generalization.

### Strengths
1. The paper very clearly articulates why pure IL policies fail (quantization error & distribution shift) and why simple RL fine-tuning also fails (encoder instability & granularity dilemma) . The two stages of the CSO framework are designed to systematically address these two core challenges.

2. Decoupling the optimization process into "representation stabilization" (R-SFT) and "policy optimization" (SPO) is a clever and effective design. Freezing the encoder after R-SFT is a key insight, providing a stable foundation for the subsequent RL stage and avoiding representation collapse from inconsistent gradients.

3. The SPO algorithm itself is a valuable contribution. By performing updates at the "skill" level (rather than token or trajectory), it successfully finds a balance point in the granularity dilemma. This avoids the high variance of token-level updates and the low efficiency and sensitivity of trajectory-level updates.

4. The method achieves SOTA or highly competitive results on LIBERO (including its challenging Long suite) and MetaWorld ML45. The ablation study strongly demonstrates the synergistic effect of CSO's two components; "w/o R-SFT", "w/o SPO", and "w/o Encoder Freeze" all lead to significant performance drops, justifying the authors' design choices. The granularity comparison clearly shows SPO (Skill-level) is superior in both performance and stability compared to Token-level (PPO) and Trajectory-level (GSPO).

5. The method's successful deployment on a physical robot and its ability to correct fine-grained errors in the IL policy greatly enhance the paper's credibility and practical value.

### Weaknesses
1. **Dependence on R-SFT initial success rate:** The CSO framework's startup relies heavily on the R-SFT stage's ability to collect successful trajectories. If the pre-trained IL policy has a very low (e.g., <1% or 0%) initial success rate for certain tasks, the R-SFT stage will become extremely sample-inefficient or even impossible to start. Although Appendix G discusses mitigating this with "persistent sampling", this may still be a significant bottleneck in practice.

2. **Coarse credit assignment in SPO:** A core design of the SPO algorithm is that it is "critic-free" and uses a trajectory-level advantage (based on relative reward) . This same advantage value is then assigned to every skill within that trajectory. This is a very coarse credit assignment mechanism that ignores temporal information. Intuitively, the critical skill that led to the task success might only occur at the end of the trajectory, yet SPO gives equal credit to the initial skills.

3. **Framework complexity:** Compared to end-to-end RL fine-tuning, CSO is a more complex, multi-stage online optimization process. It requires two separate online data collection and optimization loops: first, collection and fine-tuning for R-SFT , followed by collection and optimization for SPO. This increased complexity could be a barrier to adoption.

### Questions
1. As mentioned in W1, the R-SFT stage depends on collecting successful trajectories. In your experiments (e.g., LIBERO-Long), the IL baseline (QueST) had a success rate of 69.1%. This means ~30% of rollouts in the R-SFT stage were "failures." For tasks with much lower success rates (e.g., 5% or 1%), how many samples would R-SFT need to collect $\mathcal{D}_{succ}$ (e.g., 200 trajectories)? Does the framework fail completely on tasks where the initial policy has a near-0% success rate?

2.  As noted in W2, SPO assigns a uniform trajectory advantage $A(\tau^{(i)})$ to all skills $Z_t$ in that trajectory. This seems to contradict the fine-grained goal of "skill-level" optimization.
    (a) Did you experiment with more sophisticated credit assignment mechanisms, such as introducing a learned value function (Critic) to estimate per-skill advantages, instead of the "critic-free" approach?
    (b) Is this simplified credit assignment a potential reason why the LIBERO-Long (96.0%) performance is slightly lower than LIBERO-90 (97.7%) (i.e., long-horizon tasks suffer more from this problem)?

3.  The ablation study (Table 2) shows that "w/o Encoder Freeze" after R-SFT causes performance to drop from 96.0% to 89.6%. This is strong evidence that freezing is effective. Does this imply that the policy gradients from the SPO stage (even at the skill level) are still unstable enough to destroy the good representation learned by R-SFT? In other words, is freezing the encoder "masking" a fundamental instability that still exists in the SPO gradient?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Cascaded Skills Optimization (CSO), a two-stage framework for refining imitation-learned robotic policies represented with discrete skill sequences. Stage 1 performs rejection-sampling supervised fine-tuning on successful online trajectories to stabilize the encoder and align the policy with successful trajectories. Stage 2 introduces a skill-level variant of GSPO that computes importance ratios and clipped updates per skill sequence rather than per token or entire trajectory, aiming to balance stability and credit assignment. The method achieves strong empirical performance on both simulation benchmarks and real-robot experiments.

### Strengths
1. CSO achieves good performance in manipulation simulation benchmarks and hardware experiments compared to baselines. 
2. The ablation studies are thorough and empirically validate the effectiveness of the proposed modules. In particular, their skill-level approach indeed achieves better performance than token-level PPO and trajectory-level GSPO.

### Weaknesses
1. The proposed Skill Policy Optimization (SPO) is a minor adaptation of GSPO, where each skill is treated as an independent one-step trajectory under sparse reward. The algorithmic change is somehow straightforward, and the novelty primarily lies in framing rather than algorithmic substance.
1. The Rejection-Sampling Supervised Fine-Tuning (R-SFT) phase trains only on successful online trajectories, which may stabilize learning but also narrows the encoder’s representation to the manifold of success states. This design could harm generalization and exploration during RL fine-tuning, as the encoder is later frozen and may not encode failure or off-distribution states well. The author should provide further justification on why it is reasonable to confine the data to successful trajectories for representation learning.

### Questions
1. How does the length of each skill sequence (i.e., number of tokens per skill) affect CSO’s performance?

### Soundness
3

### Presentation
3

### Contribution
2
