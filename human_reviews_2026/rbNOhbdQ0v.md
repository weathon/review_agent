# Making Offline Model-Based Reinforcement Learning Work on Real Robots

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Reinforcement Learning (RL) has achieved impressive results in robotics, yet high-performing pipelines remain highly task-specific, with little reuse of prior data. Offline Model-based RL (MBRL) offers greater data efficiency by training policies entirely from existing datasets, but suffers from compounding errors and distribution shift in long-horizon rollouts. Although existing methods have shown success in controlled simulation benchmarks, robustly applying them to the noisy, biased, and partially observed datasets typical of real-world robotics remains challenging. We present a principled pipeline for making offline MBRL effective on physical robots. Our RWM-O extends autoregressive world models with epistemic uncertainty estimation, enabling temporally consistent multi-step rollouts with uncertainty effectively propagated over long horizons. We combine RWM-O with MOPO-PPO, which adapts uncertainty-penalized policy optimization to the stable, on-policy PPO framework for real-world control. We evaluate our approach on diverse manipulation and locomotion tasks in simulation and on a real quadruped, training policies entirely from offline datasets. The resulting policies consistently outperform model-free and uncertainty-unaware model-based baselines, and fusing real-world data in model learning further yields robust policies that surpass online model-free baselines trained solely in simulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an offline model-based RL pipeline that can be successfully deployed on real robots. The pipeline consists of two stages. First, they train an autoregressive dynamics model on the offline dataset to predict long-horizon rollouts. The dynamics model captures both aleatoric uncertainty (via Gaussian variance) and epistemic uncertainty (via bootstrap ensembling). Second, they train a policy using an uncertainty-aware model-based PPO algorithm (MOPO-PPO). Specifically, they train the policy on model-imagined rollouts, with the epistemic uncertainty (ensemble variance) subtracted from the reward to encourage conservative behavior. This pipeline demonstrates strong performance gain over uncertainty-unaware baselines across three simulated environments (Reach-Franka, Velocity-G1, Velocity-ANYmal-D), and successfully transfers to a real-world Velocity-ANYmal-D task. Through careful ablation, the paper analyzes the effect of the uncertainty penalty and establishes a correlation between the model's prediction error and uncertainty estimate. Overall, the paper makes a solid empirical contribution towards deploying offline model-based RL in the real world.

### Strengths
1. The paper makes several empirical contributions to improve upon prior offline model-based RL methods and to eventually deploy offline model-based RL on a real robot.  
2. The paper conducts thorough ablation studies to justify each design choice. They establish a correlation between epistemic uncertainty and model prediction error, justifying the need for the uncertainty penalty. They further analyze the effect of different lambda coefficients for the uncertainty penalty. When lambda is too large, the policy becomes too conservative and stays still. When lambda is too small, the policy starts exploiting the model. Last but not least, they compare different data mixtures, providing valuable insights into what data mixture should be used for offline model-based RL.

### Weaknesses
1. Despite the empirical contributions, the paper lacks novelty since every component has been proposed by / studied in prior work (e.g., uncertainty-aware dynamics models, uncertainty-penalized policy optimization). 
2. The real-world experiments lack comparison to immediately relevant baselines beyond a hard-coded data collection policy and an online model-free policy. 
3. All experiments run from low-dimensional observations. It is unclear how scalable the method is to high-dimensional observations. For one thing, it is hard to quantify epistemic uncertainty using pixel reconstruction.

### Questions
**Major**

1. Can you compare to an offline model-based RL baseline in the real-world experiments? This is important since you claimed that your method is "the first demonstration of offline MBRL operating reliably on real robotic hardware." The implication is that other offline MBRL methods fail to do so.
2. A close follow-up to MOPO, COMBO [1], is omitted from all the discussions. Their main argument is to directly learn a conservative Q function instead of using an uncertainty reward penalty. And it seems to work better than MOPO. Can you either justify the omission or provide a comparison, at least in the simulated benchmark?
3. In Table 1, what exactly is the online model-free policy? It seems that on most data mixtures, your method is worse than the online model-free policy. Is this to be expected?

**Minor**

4. Despite the justifications in Appendix A.5, it's still unclear to me why PPO is better than SAC. Can you explain a bit more why this is the case?
5. When rolling out the model, you predict the next observation conditioned on the ensemble mean. Would this potentially lead to mode averaging when the dynamics is multimodal (high aleatoric uncertainty)? 
6. Can you add a detailed description of each simulated environment (Reach-Franka, Velocity-G1, Velocity-ANYmal-D)? Currently it is unclear what each task involves.
8. Line 216, "By incorporating uncertainty-aware modeling into the dynamics learning process, RWM-O enables robust trajectory forecasting in offline settings with uncertainty effectively propagated over long horizons." This sentence is confusing.

References:

[1] Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, Chelsea Finn. COMBO: Conservative Offline Model-Based Policy Optimization. NeurIPS 2021.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The contribution proposes an offline MBRL algorithm and evaluates it in simulation and on real hardware. The MBRL algorithm relies on an autoregressive robot world model. Ensembles are used to capture epistemic and aleatoric model uncertainty. The policy optimization penalizes exploration in regions with high epistemic uncertainty. This way overfitting to dynamics model errors is discouraged. The method is evaluated in simulation and on a real-world hardware experiment.

### Strengths
The paper is very well written and results are presented nicely. The proposed approach seems reasonable and is well explained. The hardware application is impressive and comparisons in simulations are exhaustive.

### Weaknesses
- The central and strong claim of this work represented in the abstract and title is that this is the '[...] first demonstration of offline MBRL operating reliably on real robotic hardware.' If I understand correctly, at least [1] (See Sec. IV - algorithms) and maybe to some extend [2] evaluated offline model-based reinforcement learning on real robotic hardware. Therefore, one cannot claim that this work is the first application of offline MBRL on real hardware. What "reliable" means in the claim is not further defined. With this very strong claim I would have at least expected an exhaustive review on related hardware applications of offline MBRL and why the application presented in this paper is more significant and indeed the first reliable one. The claim of the title and the abstract is somewhat softened in the introduction: "this is the first demonstration of uncertainty-penalized offline MBRL operating reliably on a physical robot". If this is the real claim then the abstract and title should be changed accordingly in my opinion.

- The authors mention in Sec. 2.2 that there are many methods to incorporate uncertainty estimation into MBRL ([3] may be an additional relevant related work here as they achieve a rollout length of around 30 steps on average). The paper differentiates itself from those methods solely by mentioning that those methods have not been applied on hardware "While these methods achieve impressive performance in controlled simulation benchmarks, applying them to real-world robotics remains a significant hurdle, where reliability and robustness demand both accurate long-horizon modeling and stable policy learning." This makes it hard to evaluate what the methodological contribution of the paper is. Additionally, it seems that none of the baselines chosen in Sec. 5 were uncertainty aware. Therefore, results support that uncertainty awareness is important in general but do not support the effectiveness of the method proposed in this paper compared to others in the uncertainty aware space.

Additionally, I am unsure if ICLR is the right venue for the paper. Since the main contribution is to "make ... learning work on real robots" I'd suspect a robotics venue might be more fitting. 


[1] G. Zhou, L. Ke, S. Srinivasa, A. Gupta, A. Rajeswaran and V. Kumar, "Real World Offline Reinforcement Learning with Realistic Data Source," 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 7176-7183, doi: 10.1109/ICRA48891.2023.10161474.

[2] X. Li, W. Shang and S. Cong, "Offline Reinforcement Learning of Robotic Control Using Deep Kinematics and Dynamics," in IEEE/ASME Transactions on Mechatronics, vol. 29, no. 4, pp. 2428-2439, Aug. 2024, doi: 10.1109/TMECH.2023.3336316.

### Questions
- I would suggest making the central claim of the paper '[...] first demonstration of offline MBRL operating reliably on real robotic hardware.' more specific and adding a literature review to support the more specific claim.

### Soundness
1

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes the framework Offline Robotics World Model  (RWM-O), which extends the previous contribution Robotics World Model (RWM) to the offline setting by adding uncertainty regularizations to the predictions of the dynamics model. The paper also introduces MOPO-PPO, which extends MOPO to use PPO as the base policy optimization algorithm. Finally, the framework uses uncertainty-regularized rewards to prevent the learned policy from drifting from the data distribution. The paper compares to standard baselines from the RL literature on simulated benchmarks and also demonstrates the algorithm on a real quadruped robot.

### Strengths
$\textbf{Clarity}$: The paper is well written, and it is easy to understand how the many components fit together. 

$\textbf{Thorough Evaluations}$: The paper carefully ablates key design decisions and demonstrates how they must be tuned to achieve strong results.

$\textbf{Real world results:}$ The paper demonstrates that the algorithm can be applied directly to a high-dimensional real-world quadruped.

### Weaknesses
$\textbf{Missing Related Work}$: The paper largely connects to related work presented at mainline machine learning venues, but misses many developments in applying MBRL (and RL more generally) presented at robotics venues. Given the heavy emphasis the paper places on practical real-world deployment, omitting these works is a major weakness. For example, the following papers apply real-world RL to quadrupeds: 

- “A Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning” (Smith et al., RSS 2023)

- “Learning to Walk from Three Minutes of Real-World Data with Semi-structured Dynamics Models” (Levy et al., CoRL 2024)

- “Date-Efficient Reinforcement Learning for Legged Robots” (Yang et al., CoRL 2020)

Specifically, the final two papers use uncertainty mitigation techniques to get MBRL to work for real world quadrupeds in a batch offline RL setting (i.e. multiple rounds of collecting data, then updating the policy with offline MBRL). Doing fully offline RL vs. batch offline RL is a minor distinction from prior work, in the opinion of this reviewer.



$\textbf{Real world results are not Surprising}$:  As a concrete examples (Yang et al., CoRL 2020) and (Levy et al., CoRL 2024) learn locomotion policies with only ~40k and ~20k real world environment steps (if my math is correct). This is around an order of magnitude less real world data than the experiments presented in this paper. This is not meant as a direct comparison, as the experimental set ups are obviously different. However, I note this to highlight that I do not find the presented results surprising or impressive, given what has been accomplished previously. 

$\textbf{Benchmarks:}$ I question whether the benchmark experiments are informative about what will happen in the real world. Specifically, the current results only demonstrate significant gains when optimal expert data is available. However, I do not think this is a realistic assumption for systems such as quadrupeds — if we had optimal data, wouldn’t we already have an optimal real world policy? What is the real world benefit of the method if it doesn’t show gains when only sub-optimal data is available? }


$\textbf{Minimal Technical Contribution}$: I’m impressed that the authors brought together many different techniques and got them to actually work on a real robot. However, this style of paper only makes a strong publication if the results are surprisingly strong. Given the previous works mentioned above, I do not believe the paper passes this bar in its current form, making the limited technical novelty an additional weak point.

### Questions
- Why is offline MBRL the correct approach for real word learning for the tasks in the paper? Given that the current results do not “move the needle” in terms of capabilities, I believe this point needs to be thoroughly defended.  

- Can the given approach succeed when optimal data is not available?

### Soundness
2

### Presentation
3

### Contribution
2
