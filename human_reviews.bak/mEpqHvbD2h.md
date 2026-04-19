# Diffusion Policy Policy Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
We introduce Diffusion Policy Policy Optimization, DPPO, an algorithmic framework including best practices for fine-tuning diffusion-based policies (e.g. Diffusion Policy) in continuous control and robot learning tasks using the policy gradient (PG) method from reinforcement learning (RL). PG methods are ubiquitous in training RL policies with other policy parameterizations; nevertheless, they had been conjectured to be less efficient for diffusion-based policies. Surprisingly, we show that DPPO achieves the strongest overall performance and efficiency for fine-tuning in common benchmarks compared to other RL methods for diffusion-based policies and also compared to PG fine-tuning of other policy parameterizations. Through experimental investigation, we find that DPPO takes advantage of unique synergies between RL fine-tuning and the diffusion parameterization, leading to structured and on-manifold exploration, stable training, and strong policy robustness. We further demonstrate the strengths of DPPO in a range of realistic settings, including simulated robotic tasks with pixel observations, and via zero-shot deployment of simulation-trained policies on robot hardware in a long-horizon, multi-stage manipulation task.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to use PPO to finetune the diffusion-model based policy in an offline-to-online setting. By integrating denoising steps of diffusion model into task MDP, authors could utilize PPO to finetune the diffusion model for improved performance. Extensive efforts are put to find the best practice in fine-tuning diffusion policy, in terms of hyperparameters, network architecture and algorithmic variants. Finally,  the efficacy and the performance of the proposed framework  is demonstrated on various experiments in both simulated and real-world scenarios.

### Strengths
This paper contains tons of discoveries related to fine-tuning diffusion policy via PPO. Specifically, numerous aspects like denoising steps to be fine-tuned, network architecture, GAE variants  and effects of expert data are all studied with experimental results reported. As fine-tuning diffusion policy for real-world robot is a very important field, these experiments are welcome and contain significant value  for the community. Last but not least, the authors also provide an empirical explanation for why their method obtains better results than others (i.e. structural exploration).

$~$

## Originality 
Utilizing RL method to finetune diffusion model is in great need. And the fruitful experimental results presented in this work  are of great value and originality.

## Quality

Many experiments and ablation studies are conducted and the best practices are summarized from them.

## Clarity

Unfortunately, this paper is not clearly written. In the main papers (1-10 pages), authors’ writing lacks a central logic  and the results are presented without  a focus. I have great trouble understanding the contribution and experimental results during reading. 

## Significance
The large amount of experiments conducted is of great value to the community, especially considering there are also real-world experiments.

### Weaknesses
Unfortunately, I personally think the presentation of this paper is very poor. With the extensive and fruitful experiments results, the authors fail to present them in a concise and focused manner in the main pages and make me quite confused during reading and comprehending this paper.

Firstly, there are three components that are actively mentioned and studied: the diffusion policy, the PPO method and the best practice for combining them together. 

But based on my comprehension, the diffusion policy and PPO used in this work are of no novel changes & contributions.

Combining these 2 together and successfully applying on real-world robotic system, is with great value and potential hidden evils. But  the authors fail to present this to me  in a clear and concise manner: for example, there are  experiments comparing diffusion policy (DP) to gaussian parameterized policy, but the advantage of DP to vanilla parameterization is already well-established and not the contribution of this paper, in my opinion. Since this paper is already over-sized, I suggest the authors to relocate some contents into appendix and focus on your main contribution in the main pages



To summarize, I hope the organization and presentation of this paper could be improved such that the problem formulation, method design and experimental results could be tightly chained and focused on  your main contribution.

### Questions
1. Could you change the title "Diffusion Policy Policy Optimization" to provide more contexts to readers? Current title is too short and abstract and your work is not as general as it indicates. 
2. I would  suggest deleting/relocating some contents in early chapters as your paper is already over-sized and some of these contents are already known.
4. in line 249, the definition of $\gamma_{\text{ENV}}^t$ seems to be incorrect? I thinks it should be $\gamma_{\text{ENV}}^{t^\prime-t}$? or are there any further treatments of this discount factor?
4. there lacks a pseudo-code for the whole algorithm, please provide one. For example, how is your V function trained?
5. for fine-tuning control policy in RL, most existing algorithms finish in less than 1e6 steps. But the results presented throughout this paper is conducted in 1e6~1e7 steps. Could you truncate the figures to 5e5 such that we could better compare the performance of various methods?

6. Could you provide more comparisons of DPPO against established offline-to-online methods like [1] and [2]?

>[1] Wang, Shenzhi, et al. "Train once, get a family: State-adaptive balances for offline-to-online reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).
>
>[2] Zhang, Haichao, Wei Xu, and Haonan Yu. "Policy Expansion for Bridging Offline-to-Online Reinforcement Learning." The Eleventh International Conference on Learning Representations.

7. Could you provide more explanations about the worse performance of DAWR, DRWR and Cal-QL? In appendix, the result of the performance of Cal-QL on  Can and Square are almost zero, which is surprising to me. Could you elaborate more on this? BTW, what is the performance of Cal-QL when only offline training is performed?

8. for DRWR, in the weight function, could you minus a V function from the estimated return such that its’ more aligned toward DPPO?
9. in line 988-994, you mentioned speedup and less GPU memory usage is obtained by fine-tuning only the last few K denoising steps. But the method is to  use a trainable $\theta$ for the last K steps and a frozen $\theta_{\text{FT}}$ for previous steps. How does this method improve the training speed and reduce GPU memory?

10. Sect. 6 of structural exploration is of much importance and novelty. Could you try ablating on this phenomenon for DP? for example, by tuning the denoising process such that DP itself performs less or more structural exploration and see if the performance is affected?

There are many interesting, fruitful discoveries and experiments in this work. But I find none of them is studied exhaustively or presented clearly.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The papers introduces Diffusion Policy Policy Optimization (DPPO), an algorithmic framework designed to fine-tune diffusion-based policies using policy gradient optimization (PPO) from reinforcement learning (RL). DPPO enables structured exploration, enhances training stability, and demonstrates the robustness and generalization during deployment. It improves policy performance across various benchmarks, including tasks with pixel-based observations and long-horizon rollouts, which have traditionally been challenging for previous RL methods.

### Strengths
(1) The paper introduces a novel dual-layer MDP framework and the use of PPO to fine-tune Diffusion-Policy, enhancing its performance in various tasks.
(2) The paper proposes several best practices for DPPO, such as fine-tuning only the last few denoising steps and using DDIM sampling to replace some denoising steps, improving efficiency and performance.
(3) DPPO shows a strong performance in all benchmark tests, particularly in tasks involving pixel-based observations and long-horizon rollouts.
(4)  DPPO achieves highly efficient zero-shot transfer from simulation to real-world tasks in robotics, demonstrating a minimal sim-to-real performance gap.

### Weaknesses
(1) Figure 10 shows that the pre-trained Diffusion Policy can model multimodality in the behavior policy. However, after fine-tuning, the Diffusion Policy gradually converges into a deterministic policy. In this case, does it lose the advantage of expressing multimodal action distributions that Diffusion Policy is known for?

(2) In the experiments, DPPO is only compared against other diffusion-based methods, without any comparison to existing state-of-the-art Offline2Online or Sim2Real algorithms, such as Uni-O4 [1] and O3F [2]. Clearly, there is a significant optimal gap in environments like HalfCheetah.

(3) The motivation for fine-tuning models is typically to reduce the number of online interactions and improve sample efficiency. However, DPPO does not seem to significantly enhance sample efficiency, as its convergence speed is similar to other online reinforcement learning methods.

(4) The v-D4RL [3] dataset could have been considered in the experiments for broader evaluation.

(5) Compared to QSM [4], DPPO does not introduce particularly novel concepts but rather provides engineering or empirical adjustments, such as fine-tuning certain denoising steps and replacing value estimation with advantage estimation.

[1] Kun Lei, et al. Uni-O4: Unifying Online and Offline Deep Reinforcement Learning with Multi-Step On-Policy Optimization
[2] Seunghyun Lee, et al. Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble
[3] Cong Lu, et al. Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations
[4] Michael Psenka, et al. Learning a Diffusion Model Policy from Rewards via Q-Score Matching

### Questions
Please see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework (DPPO) to enable fine-tuning pre-trained diffusion-based policies through policy gradient methods. They compare DPPO with rich baselines in common benchmarks. They conclude best practices for DPPO and conduct ablation studies to understand the effectiveness of DPPO.

### Strengths
* This paper conducts rich experiments and analysis. They compare DPPO with other fine-tuning RL methods for diffusion policies and other fine-tuning RL policies. They also conduct several ablation studies and visualization to analyze the effectiveness of DPPO.
* This paper is well-written and well-structured. They conclude the best practice for DPPO and the benefits of such framework.

### Weaknesses
* The novelty of the paper is limited. Previous works have proposed the idea that considers a denoising process as multi-step MDP. This paper simply integrate this MDP into the environmental MDP. The tricks to adapt PPO for better fine-tuning are also seen in the previous works like GAE except fine-tuning the last few steps.

### Questions
* In the original diffusion policy paper, the performance in the transport is nearly 100% success rate, which conflicts with the results in the paper.
* How about the results compared with the baselines that introduce a residual policy?
* How about the results compared with the baselines that use RLPD or Cal-QL for the pre-trained diffusion policies?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a framework for finetuning a pretrained diffusion policy using on-policy RL, specifically PPO, and show that their method improves performance on long-horizon sparse-reward tasks and works on real hardware.

### Strengths
- The authors explore the possibility of fine-tuning diffusion-based policies with on-policy RL and demonstrate some surprising results. Specifically, the authors find improved performance over baselines on long-horizon and sparse reward tasks. The authors also demonstrate how their method can leverage environment parallelization to train faster than off-policy RL methods. Given the abundance of GPU-accelerated simulators these days, I believe this to be a major strength as off-policy RL finetuning methods cannot leverage parallelization in the same way, leaving these high-throughput simulators as a relatively untapped resource until now. 

- The authors provide extensive comparisons to a variety of other RL finetuning methods, including two of their own novel baselines. The ablations are also thorough. 

- The paper is well written and the quality of the presentation is high.

### Weaknesses
- Much of section 6, including the first few paragraphs up to and including Figure 8 are redundant and unnecessary. At this point there are a number of diffusion policy papers applied to various tasks as well as the original diffusion policy paper itself that show that diffusion models consistently outperform GMM and Gaussian policies at capturing the modes of a multi-modal distribution, and so this result is no longer surprising or novel. I'd have rather seen this space dedicated to showing more experimental results that were instead put in the appendix, such as the comparisons in Figure 12 to methods that finetune on additional expert data, or perhaps a more in-depth analysis on why DPPO does better on long-horizon sparse-reward tasks. "How does this method compare to just collecting and training (via BC or whatever else) on additional expert data?" was a question that came to mind while I read the results, which is an important one to discuss and one I imagine other readers will have. 

- While the technical contribution is fairly good, the novelty of this approach is relatively low. The proposed method follows the more general and somewhat repetitive trend of "take diffusion policy and try it on task X and potentially finetune with method Y" style papers in the robot learning community. 

Despite the somewhat low novelty, I believe that the impressive results and the fact that this paper is one of the first, if not the first, diffusion policy finetuning methods capable of leveraging high-throughput simulators is an important contribution to the robotics and ML communities.

### Questions
- What intuition, if any, do the authors have on why DPPO does better on long-horizon sparse-reward tasks? Is there something special about finetuning with PPO as opposed to other finetuning strategies that helps? 

- In Figure 6, why does DPPO-UNet get outperformed by DPPO-MLP on state-based tasks? And why is DPPO-UNet not present on the pixel-based versions of the task? 

- On lines 264-265 the authors state *"We propose clipping the diffusion noise schedule at a larger-than-
standard noise level to encourage exploration and training stability."*. If more noise encourages exploration, would it not be more beneficial to use the variance-exploding line of diffusion models such as EDM instead of the variance-preserving ones? 

- In Figure 7 we see positive sim2real transfer from pretraining only, but negative transfer after fine-tuning. Do the authors have some intuition as to why this is the case?

### Soundness
4

### Presentation
4

### Contribution
2
