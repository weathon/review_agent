

{0}------------------------------------------------

# GUIDED REINFORCEMENT LEARNING WITH ROLL-BACK

Anonymous authors

Paper under double-blind review

## ABSTRACT

Reinforcement learning-based solutions are increasingly being considered as strong alternatives to classical system controllers, despite their significant sample inefficiency when learning controller tasks from scratch. Many methods that address this issue use prior task knowledge to guide the agent’s learning, with several recent algorithms providing a guide policy that is sometimes chosen to execute actions instead of the learner policy. While this approach lends excellent flexibility as it allows the guide knowledge to be provided in any format, it can be challenging to decide when and for how long to use the guide agent. Current guide policy-based approaches typically choose a static guide sampling rate empirically and do not vary it. Approaches that transfer control use simple methods like linear decay or require hyperparameter choices that strongly impact the performance. We show that under certain assumptions, the sampling rate of the guide policy can be calculated to guarantee that the mean return of the learning policy will surpass a user-defined performance degradation threshold. To the best of our knowledge, this is the first time a performance guarantee has been established for a guided RL method. We then implement a guided RL (GRL) algorithm that can make use of this sample rate and additionally introduce a roll-back feature in guided RL with roll-back (GRL-RB) to adaptively balance the trade-off between performance degradation and rapid transfer of control to the learner. Our approach is simple to implement on top of existing algorithms, robust to hyperparameter choices, and effective in warm-starting online learning.

# 1 INTRODUCTION

The sample inefficiency of model-free reinforcement learning (RL) remains one of the greatest barriers to the wider adoption of RL-trained models in everyday life (Yu, 2018). Real-world applications of RL are increasing, such as in refining the large language model ChatGPT (Ouyang et al., 2022), or for efficient cooling of commercial buildings (Luo et al., 2022). However, real-world environments often translate to complex environments for RL agents (Dulac-Arnold et al., 2021). This can be due to sparse rewards, high stochasticity and very large observation and action spaces. As such, RL agents may require extensive training to be effectively deployed in such environments.

Many systems currently run by controllers or human operators demonstrate potential performance gains from automation. Often in such systems, there is an available policy that governs the behaviour of the original controller (e.g., a set of fuzzy or crisp rules). Alternatively, in a system controlled by a human operator, data that describes the behaviour running the system could be collected. Rather than forcing an RL-based controller to commence learning from scratch, such prior policies or expert data could be exploited to warm-start an RL agent before deployment in an online environment (Luo et al., 2022). Additional challenges of real-world environments make a warm-starting solution more attractive. Firstly, while the sample inefficiency of model-free RL is often handled by training across many parallel environments (Dulac-Arnold et al., 2021), real-world environments cannot be similarly replicated. Further, for many applications, there is a need to ensure a certain level of performance when a model is deployed to the real world, e.g. for robotics (Zhao et al., 2020), (Lobbezoo et al., 2021) and self-driving cars (Maramotti et al., 2022)(Isele et al., 2018). These types of agents would ideally commence at a reasonable performance level, such that they require only fine-tuning in the online environment. It is important that a warm-starting solution can not only accelerate the learning of the agent but minimise performance degradation after transfer to the online environment.

{1}------------------------------------------------

In this paper, we focus on a ‘roll-in’ method, where control is gradually shifted from a guide policy to a learning policy in a stepped manner. The focus of our Guided Reinforcement Learning (GRL) method is to allow online learning while reducing performance degradation. We show that given a user-defined acceptable degradation threshold, the online agent can be theoretically guaranteed to maintain an evaluation return above this threshold under certain assumptions. The most challenging of these assumptions is that the agent is able to converge fully between roll-in steps. To ease the challenges in meeting this assumption, guided RL with roll-back (GRL-RB) is introduced to enable tuning of the guide sampling rate, to respect the given threshold. Further, GRL-RB is flexible in the structure of the prior policy (e.g. heuristics, decision trees, a policy learned through imitation learning/offline RL etc.) and can be implemented on top of existing algorithms. Finally, this method gradually transfers full control to the learning agent, to ensure exploration is not hindered by the prior knowledge and to minimise distribution mismatch between the replay buffer trajectories and the trajectories induced by the learning agent’s policy (Fujimoto et al., 2019).

Our contributions are threefold: a) A guided reinforcement learning approach (GRL) that enables the gradual transfer of control to a learning policy, with a guaranteed online performance above a user-defined threshold. (b) The theoretical derivation and experimental confirmation of guide sampling rates for GRL under defined assumptions. (c) A GRL with a roll-back algorithm (GRL-RB) that helps to retain the performance guarantee of GRL while relaxing its assumptions.

# 2 PRELIMINARIES AND RELATED WORK

**Markov Decision Process (MDP).** An RL problem must be representable as a MDP (Bellman, 1957). Such a problem can be described by a 5-tuple  $\langle S, A, P, R, \gamma \rangle$ , where  $S$  is the state space,  $A$  is the action space,  $P : S \times A \times S \rightarrow [0, 1]$  describes the next state transition probability  $P(s_{t+1}|s_t, a_t)$  given the current state and action,  $R$  describes the reward function  $S \times A \rightarrow \mathbb{R}$  and  $\gamma \in [0, 1]$  is a discount factor. A trajectory  $\tau$  in an MDP describes a finite sequence of states and actions:  $(s_0, a_0, s_1, a_1, \dots, s_{T-1}) \in S \times A$  taken in the environment.

**Policy.** A policy  $\pi$  governs the next action  $a_{t+1}$  an agent will take in an MDP - depending on the current state  $s_t$  - resulting in a step reward  $r_t$ . The policy can be stochastic  $a_{t+1} \sim \pi(\cdot|s_t)$  or deterministic  $a_{t+1} = \mu(s_t)$ .

**Guide, Learning and Online Sampling Policies.** The guide and (online) learner policies will be denoted  $\pi_g$  and  $\pi_l$  respectively. The online sampling policy  $\pi$  is a combination of  $\pi_g$  and  $\pi_l$ , depending on which policy is chosen to be sampled at  $t$ .

**Undiscounted Episodic Return.** For a discrete action and observation space, the undiscounted episodic return  $\tilde{R}_\pi$  for an agent in an MDP following  $\pi$  is:

$$\tilde{R}_\pi = \sum_{\tau} \left( \left[ \rho_0(s_0) \prod_{t=0}^{T-1} p(s_{t+1}|s_t, a_t) \pi(a_t|s_t) \right] \sum_{t=0}^{T-1} r_t \right) \quad (1)$$

where  $\rho_0$  is the initial state distribution (Achiam, 2018).

One way to categorize warm-start algorithms is by focusing on the area of the RL pipeline where knowledge is inserted, namely, replay buffer, parameter updates, initial function parameters and action selection. We review action selection-based methods in this section, as they are the most relevant to our work, and include a review of the other methods in Appendix A. We also review the roll-back concept integral to our GRL-RB algorithm, and its previous use in the literature.

## 2.1 ACTION SELECTION

Action selection-based methods use either an online learning agent initialised with randomised parameters or a guide policy to choose actions. This method has become increasingly popular due to its inherent flexibility - the guide policy can be of any format - e.g. a pre-trained RL policy, a decision tree, a heuristic or a list of rules (Chang et al., 2015; 2023; Liu et al.; Uchendu et al., 2023). An obvious challenge in any guide policy-based approach is the question of how and when to reduce or remove the influence of the guide agent during the learner’s training. This is especially important when the underlying algorithm is on-policy, such that the data distribution eventually reflects the distribution induced by the learner’s policy. We find that most current approaches simply choose the

{2}------------------------------------------------

sampling rate empirically, and often do not vary it throughout training. Such a method for supervised learning was used in Daumé et al. (2009), where subsequent ‘guide’ classifiers are created through linear interpolation  $\beta h' + (1 - \beta)h$  of the previous guide  $h'$  and current policy  $h$ . While the choice of the constant  $\beta$  is discussed in terms of providing a bound on the supervised loss, it is not varied during training. Another linear interpolation uses a constant  $\beta$  to interpolate between a continuous action  $u^{MPC}(t)$  from a Model Predictive Control method, and a DDPG policy action  $a(t)$ .

A number of methods used horizon-based sampling (Chang et al., 2015; 2023; Liu et al.; Uchendu et al., 2023; Daoudi et al., 2024; Yang et al., 2022; Zhang et al., 2024) where a guide policy is sampled until some horizon  $h$  is reached, after which the learner probabilistically chooses the actions (or vice versa). Alternatively, each time step may have a probability of sampling the learner. These probabilities may not vary throughout training (Chang et al., 2015; 2023), or may be gradually reduced. In Liu et al., prior to  $h$  (sampled randomly), one of many oracles or the learner itself is sampled, depending on a prediction of the oracle’s expertise in that state by their learned value functions. As the learner improves, it will tend to be chosen over the oracles. In Jump-Start Reinforcement Learning (JSRL) proposed by Uchendu et al. (2023), sampling the guide first allows the learner agent to start closer to its goal near the episode end, giving the agent an easier task to reduce performance degradation. While this means that the learner should initially be sampled for short periods that gradually increase, in practice this depends highly on having constant  $N$ . This is because if the episode runs for longer than  $N$ , the learner will be sampled. In some environments, this may leave the agent unable to reach the goal within the environment’s time limit  $T$ . If  $T \gg N$ , the learner may end up being sampled for most of the episode, as shown in Appendix B.

Daoudi et al. (2024) instead used multiple local guides. The learner policy is used for action selection until a local guide satisfies a confidence function indicating their knowledge of the current state. The guide then produces an action which is then modified by a bounded, parameterised perturbation. This was an effective method for continuous action spaces, though the guide is again used throughout learning. In Yang et al. (2022), control in a safety-focused constrained MDP is handed back to the guide when a safety violation occurs. Again, no procedure is implemented to ensure control is fully transferred to the learner. While a linear decay method is implemented, it was not effective compared to the safety-violation based method. Finally, an adaptive sampling approach in (Zhang et al., 2024) produced samples for an imitation-learning based policy by framing sampling as a MDP, with the action space involving either sampling the online learner or expert. However, given the sampler begins as a randomised policy, it would be difficult to make the performance guarantees we are interested in using this method. This is similarly true for Liu et al., where performance would be very difficult to guarantee while the oracles’ value functions are being learned.

## 2.2 ROLL-BACK

The concept of reverting to safe or previous operational states enhances the safety and stability of reinforcement learning systems. (Hans et al., 2008) introduced an approach where safe actions are identified based on a modified Bellman equation that prioritizes minimal future rewards rather than maximal returns. This strategy involves maintaining a backup policy to revert to when the current policy encounters unsafe or suboptimal performance. Actions leading to “fatal” transitions—where rewards fall below a threshold—are avoided, and the agent reverts to previously validated safe actions using a level-based exploration strategy. (Ma et al., 2019) expanded the “roll back” principle to a novel regret-based mechanism within a navigation agent. This mechanism enhances the navigation strategy by allowing the agent to revert to better alternatives based on past decisions. This retrospective analysis is supported by heuristic aids and progress estimation, enabling the agent to continuously refine its decision-making process. The recovery mechanism in Dasagi et al. (2019) allows the RL agent to revert to previous, stable versions of its policy upon detecting performance drops using statistical methods like the Mann-Whitney U-test. This recovery strategy helps mitigate the risks associated with aggressive policy updates that can degrade learning progress, ensuring more reliable and continuous improvement over time. GRL-RB relies on a similar principle to the above method, where instead the previous safe policy is the guide policy.

# 3 GUIDED REINFORCEMENT LEARNING (GRL)

The first focus of this work is to enable a  $\pi_l$  to explore an environment while sampling a  $\pi_g$  with sampling rate  $\alpha$ , which is carefully chosen to guarantee that performance doesn’t fall below a user-

{3}------------------------------------------------

162 defined performance threshold. We also wish to fully transfer control to  $\pi_l$  by the end of training,  
163 while avoiding the issues with over-sampling  $\pi_l$  that were discussed in Section 2. We chose to  
164 employ a similar sampling approach to Chang et al. (2015) and Chang et al. (2023), but where  $\pi_g$   
165 is sampled with probability  $\alpha$  at any time during the episode. Additionally, we track the fractional  
166 use of  $\pi_l$  during an episode, and only enable sampling of  $\pi_l$  if  $\frac{n_{\pi_l}}{t} < \alpha$ , where  $\frac{n_{\pi_l}}{t}$  is the fraction of  
167 time steps during which  $\pi_l$  has been sampled by time step  $t$ . Additionally, we combine this with a  
168 curriculum approach, by gradually reducing  $\alpha$  to 0 throughout training. We use a similar method to  
169 JSRL (Uchendu et al., 2023) in deciding when to move to the next curriculum stage, which occurs  
170 when the evaluation return improves upon  $\hat{R}_{\pi_g}$ . Our approach, Guided Reinforcement Learning  
171 (GRL) is formally described in Algorithm 1 of Appendix D. To enable the implementation of GRL,  
172 we derive its  $\pi_g$  learning rate  $\alpha$  below.

###### 174 3.1 DERIVING GUIDE SAMPLING RATES FOR GRL

175 To ensure that the agent evaluation score remains above a user-defined performance degradation  
176 threshold, we derive a sampling rate  $\alpha$  for  $\pi_g$  to be used in GRL. The *evaluation score* refers to  
177 the mean undiscounted return for an episode in the environment. Initially, the guide policy  $\pi_g$  is  
178 evaluated for  $N_i$  episodes (prior to commencing online training), to determine its mean evaluation  
179 score  $\hat{R}_{\pi_g} = \frac{1}{N_i} \sum_{n=1}^{N_i} R_n$ . The user-defined performance degradation threshold is a value between  
180 the minimum possible score and  $\hat{R}_{\pi_g}$ , representing the minimum mean score that the user can accept  
181 for the policy  $\pi$ . The user can define this by choosing  $\mu \in [0, 1]$ , as per Equation 2. The score for this  
182 policy is written as  $\hat{R}_\pi = \frac{1}{N} \sum_{n=1}^N R_n$ . The degradation threshold is defined as:

$$184 \quad \hat{R}_\pi \geq r_{min} + \mu(\hat{R}_{\pi_g} - r_{min}) \quad (2)$$

185 We provide the below results for the sampling rate derivations and their experimental validation.

###### 188 3.1.1 METHOD: DERIVATION

189 To simplify the derivation of  $\alpha$ , we modify the toy ‘Combination Lock’ MDP (Uchendu et al., 2023).  
190 This is an episodic MDP with fixed time horizon  $H$  (corresponding to the length of the combination  
191 code). The agent must correctly choose the next digit in the combination sequence (the optimal action,  
192  $a^*$ ), otherwise if the agent chooses the wrong digit (the non-optimal action,  $\bar{a}$ ), the episode ends  
193 immediately. All rewards are 0, unless the agent reaches the end of the combination, where it receives  
194 a sparse reward of  $r$  (see Appendix C for a full description of the environment). Sampling rate  $\alpha$   
195 will be derived for three variations of  $\pi_l$ ,  $\pi_g$  and the reward: variation 1 - optimal  $\pi_g$ , terminating  
196  $\pi_l$ , sparse reward; variation 2 - non-optimal  $\pi_g$ , non-terminating  $\pi_l$ , sparse reward; and variation  
197 3 - non-optimal  $\pi_g$ , non-terminating  $\pi_l$ , dense reward. The derived  $\alpha$  is then used as input to GRL  
198 (Algorithm 1), and online training can commence. If the online algorithm is sufficient, it will allow  
199 the learner policy to converge (such that  $\hat{R}_\pi = \hat{R}_{\pi_g}$ ). At this point, rather than always choosing  
200 non-optimal action, the learning policy  $\pi_l$  now has (at least) a  $(1 - \alpha)$  probability of choosing the  
201 optimal action. As such, we can reduce the guide sampling policy by  $(1 - \alpha)$ , such that the new guide  
202 sampling rate is  $\alpha - (1 - \alpha)$ . We can continue to apply the procedure until the guide sampling rate  
203  $\alpha = 0$ , as per Algorithm 1. The assumption of convergence between steps of  $\alpha$  is key to the success  
204 of GRL. In complex environments, it can be very difficult to guarantee this convergence, as shown  
205 in Figure 4. This challenge motivates the introduction of a roll-back mechanism, in our proposed  
206 GRL-RB described below.

###### 207 3.1.2 METHOD: IMPLEMENTATION AND SETTINGS

208 We implemented our proposed GRL (Algorithm 1) on top of the Implicit Q-Learning algorithm  
209 (Kostrikov et al., 2022). We used the Clean Offline Reinforcement Learning (CORL) (Tarasov et al.,  
210 2023) implementation of IQL<sup>1</sup>. IQL was chosen for all of our experiments, to ease the transition  
211 from offline-to-online training for the AntMaze experiments in Section 4. The Combination Lock  
212 environment does not have offline data, so instead the guide was implemented as an oracle, able to  
213 determine the next number in the sequence, given the current observation. As such, IQL was only

214 <sup>1</sup><https://github.com/tinkoff-ai/CORL/blob/main/algorithms/finetune/iql.py>

{4}------------------------------------------------

used in online training mode for these experiments. For variation 3, which uses a negative dense reward, we used the challenging AntMaze tasks from the D4RL benchmarks (Fu et al., 2020), which involve an 8-DoF quadruped robot attempting to reach a goal square in a maze environment. We ran experiments for AntMaze-Umaze-v2 with a reward of  $r = -1$  for each time step until the Ant reaches the goal. The episodes automatically terminate at 700 time steps if the Ant fails to find the goal. Hyperparameter specifications can be found in Appendix F.

**Comparison Algorithms** We designed three comparison algorithms based on the methods discussed in Section 2. Two static guide algorithms are generated (based on Chang et al. (2015), Chang et al. (2023), Daumé et al. (2009)), which sample the guide for 25% and 75% of the time throughout the entirety of training ('S25%' and 'S75%' in plots respectively). Following one of the approaches from (Yang et al., 2022), a *dynamic* linear decay method ('LD' in plots) is also introduced, which decreases the use of the guide agent from 100% to 0% by  $\frac{1}{n}$  every evaluation, where  $n$  is the number of evaluations during training. Our experiments found that this problem was too challenging for standard RL. Despite the conceptual simplicity of the Combination Lock environment, the agent is required to correctly choose a digit between 0-9 ten times in a row (a  $10^{-10}$  probability of doing it by chance). We tested an implementation of PPO (Raffin et al., 2021), however it (unsurprisingly) made no progress in  $1e5$  time steps. Similarly, we used an implementation of vanilla SAC Raffin et al. (2021) for the AntMaze environments, and despite trying 100 different hyperparameter combinations per environment (using hyperparameter tuning framework Optuna (Akiba et al., 2019)), we found it could not improve without the guidance from the offline training component.

## 3.2 RESULTS AND EXPERIMENTAL VALIDATION

We present results for the sampling rate for three variations and their experimental validation. For completeness, Appendix E presents the full derivations.

### 3.2.1 VARIATION 1: OPTIMAL GUIDE POLICY

For an optimal guide policy,  $a_g = a^*$  every time step that  $a \sim \pi_g$ . We conservatively assume that every learner action  $a_l$  is non-optimal  $a_l = \bar{a}^*$ , thus instantly ending the episode. While  $\pi_l$  (which would usually be randomly initialised) is unlikely to be this poor, this assumption represents the worst case scenario. Thus, the sampling rate  $\alpha$  represents an upper bound on the rate required to guarantee Equation 2. Every trajectory  $\tau$  is a sequence of optimal guide actions (unless the first action is  $a \sim \pi_l$ ) taken with probability  $\alpha$ , followed either by a learner action  $a_l$  that then ends the episode prematurely (at  $t = h$ ), or a guide step that completes the episode successfully (at  $t = H$ ). While the result is a  $\geq$  relation (as per Equation 2), choosing an  $\alpha$  that is larger will simply result in a slower transfer to the  $\pi_l$ , so we substitute it for  $=$  here.

**Result.** The derived sampling rate for an optimal  $\pi_g$  and a terminating  $\pi_l$  is:

$$\alpha = \mu^{\frac{1}{H}} \quad (3)$$

Equation 3 is used in the GRL algorithm, which is then applied to the Combination Lock environment, with a return degradation threshold  $\mu = 0.75$  (Figure 1). As expected, the mean score of the agent was successfully maintained above the threshold  $\mu \hat{R}_{\pi_g}$ , and quickly reached the original guide score. In this case,  $\pi_g$  was already optimal, however this procedure allows the learning agent to explore for a better solution without resulting in excessive performance degradation.  $(1 - \alpha)$  increases smoothly throughout training until control is transitioned fully to  $\pi_g$ . While the LD is somewhat similar to GRL, it does not use the performance degradation-based  $\alpha$ , and so runs the risk of being overly-conservative,

![Figure 1: Optimal guide policy results. (a) Return: A line graph showing Evaluation Return (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 800). The GRL algorithm (red line) quickly reaches a return of approximately 0.8, while the LD algorithm (blue line) reaches approximately 0.6. The S25% (green dashed line) and S75% (purple dashed line) algorithms maintain a constant return of approximately 0.4. The threshold mu (red dashed line) is at 0.75. The shading around the GRL and LD lines represents 1-sigma across 50 random seeds. (b) Mean evaluation sampling rate: A line graph showing Evaluation Sampling Rate (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 800). The GRL algorithm (red line) starts at 1.0 and decreases to approximately 0.4. The LD algorithm (blue line) starts at 1.0 and decreases to approximately 0.2. The S25% (green dashed line) and S75% (purple dashed line) algorithms maintain a constant sampling rate of approximately 0.4.](1a6d75c94d3fd49936527eadd25e9278_img.jpg)

Figure 1: Optimal guide policy results. (a) Return: A line graph showing Evaluation Return (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 800). The GRL algorithm (red line) quickly reaches a return of approximately 0.8, while the LD algorithm (blue line) reaches approximately 0.6. The S25% (green dashed line) and S75% (purple dashed line) algorithms maintain a constant return of approximately 0.4. The threshold mu (red dashed line) is at 0.75. The shading around the GRL and LD lines represents 1-sigma across 50 random seeds. (b) Mean evaluation sampling rate: A line graph showing Evaluation Sampling Rate (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 800). The GRL algorithm (red line) starts at 1.0 and decreases to approximately 0.4. The LD algorithm (blue line) starts at 1.0 and decreases to approximately 0.2. The S25% (green dashed line) and S75% (purple dashed line) algorithms maintain a constant sampling rate of approximately 0.4.

Figure 1: Optimal guide policy results: (a) **Return.** The chosen return degradation threshold  $\mu = 0.75 * \hat{R}_{\pi_g}$  is respected.  $\hat{R}_{\pi_g}$  indicates the original score of the guide. Shading shows  $1 - \sigma$  across 50 random seeds. (b) **Mean evaluation sampling rate** of the learning policy  $\pi_l$  throughout training.

{5}------------------------------------------------

resulting in a slow transfer from  $\pi_g$  to  $\pi_l$ . LD also decreases  $\alpha$  every evaluation no matter the  $\hat{R}_{\pi_g}$  result, meaning it could also transfer too quickly. In this case, however, with an optimal  $\pi_g$ , LD is able to perform well. The static guidance clearly breaches the performance degradation threshold, as the sampling rate is far lower than  $\alpha$ . In Figure 1b, S25% and S75% sometimes do not adhere strictly to their respective 25% and 75% sampling rates for two reasons. Firstly, given there are 10 steps in this particular problem, the exact sampling rates cannot be attained. Secondly, since the episode ends following a non-optimal action,  $1 - \alpha$  can exceed the given rate if the first action is chosen to be a learner action.

### 3.2.2 VARIATION 2: NON-OPTIMAL GUIDE POLICY AND NON-TERMINATING LEARNING POLICY

We will now assume that the guide policy  $\pi_g$  only takes the optimal action  $a^*$  with probability  $\beta_g < 1$ . We will also assume that the learning policy does not immediately terminate the episode, but only takes the non-optimal action  $\bar{a}^*$  with probability  $\beta_l < 1$ .

**Result.** The derived sampling rate for a non-optimal  $\pi_g$  and a non-terminating  $\pi_l$  is:

$$\alpha = \frac{\mu^{\frac{1}{H}} \beta_g - (1 - \beta_l)}{\beta_g - (1 - \beta_l)} \quad (4)$$

It is easy to see that if the learning policy is always terminating, i.e.  $\beta_l = 1$ , then we immediately recover Equation 3. Moreover, if a practitioner does not know  $\beta_g$  and knows only the empirical mean evaluation return  $\hat{R}_{\pi_g}$ , they can easily use the relationship in Equation 16 to substitute in for  $\beta_g$ . Writing  $R_{\pi^*} = r$ :

$$\beta_g = \left( \frac{\hat{R}_{\pi_g}}{R_{\pi^*}} \right)^{1/H} \quad (5)$$

Depending on the environment,  $\beta_l$  might be difficult to determine or the criticality of the application might force the practitioner to assume  $\beta_l = 1$ , representing the worst case scenario. If the application has a little more flexibility, then a lower  $\beta_l$  would reduce the time needed to transfer from  $\pi_g$  to  $\pi_l$ .

The experimental validation of the result for a non-optimal guide and non-terminating learning policy was run by forcing  $\pi_g$  to take a random action with  $1 - \beta_g$  probability, and fixing the  $\pi_l$  action to the correct choice with  $1 - \beta_l$  probability. The parameters for these experiments are given in Appendix F.2 and the results are shown in Figure 2. Again, the degradation threshold is respected by GRL, and online exploration of the space allows the learning agent to quickly surpass the mean evaluation score of the non-optimal guide agent, which starts at a low level of performance ( $\hat{R}_{\pi_g} \approx 0.35$ ). While the degradation threshold is also respected by LD, the non-optimal guide demonstrates the tendency for LD to be over-conservative, whereas GRL makes use of the derived  $\alpha$  to transfer as fast as possible while respecting the degradation threshold.

### 3.2.3 VARIATION 3: DENSE REWARD

We now consider two cases of a dense reward. The first is a positive step-wise reward  $r_t = r$  ( $r > 0$ ) for each non-terminal action taken at time step  $t$ . Environments that may use such a reward are those where an agent must choose actions to continue a task as long as possible, such as the aforementioned Combination Lock, or the classic control problem, CartPole. The latter is a negative step-wise reward  $r_t = r$  ( $r < 0$ ) for each non-terminal action taken at time step  $t$ , which could be applicable in environments where an agent must complete as task as quickly as possible, such as navigating a maze.

![Figure 2: Non-optimal guide and non-terminating learner results. (a) Return: A line graph showing Evaluation Return (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 600). The legend includes LD (red solid), S75% (orange dashed), S25% (green dashed), GRL (purple solid), H (blue dashed), and R_pi_g (light blue dashed). LD and GRL show high return, while S75% and S25% show lower return. (b) Mean evaluation sampling rate: A line graph showing Evaluation 1-alpha (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 600). The legend is the same as in (a). LD and GRL show high sampling rate, while S75% and S25% show lower sampling rate.](096d7a8a21933900dad68d82ae8a97fb_img.jpg)

Figure 2: Non-optimal guide and non-terminating learner results. (a) Return: A line graph showing Evaluation Return (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 600). The legend includes LD (red solid), S75% (orange dashed), S25% (green dashed), GRL (purple solid), H (blue dashed), and R\_pi\_g (light blue dashed). LD and GRL show high return, while S75% and S25% show lower return. (b) Mean evaluation sampling rate: A line graph showing Evaluation 1-alpha (y-axis, 0.0 to 1.0) versus Step (x-axis, 0 to 600). The legend is the same as in (a). LD and GRL show high sampling rate, while S75% and S25% show lower sampling rate.

Figure 2: Non-optimal guide and non-terminating learner results: (a) **Return**. The chosen return degradation threshold  $\mu = 0.75 * \hat{R}_{\pi_g}$  is respected. The  $\hat{R}_{\pi_g}$  line indicates the original score of the guide. (b) **Mean evaluation sampling rate** of the learning policy  $\pi_l$  throughout training.

{6}------------------------------------------------

**Result: Negative Dense Reward.** The derived sampling rate for a non-optimal  $\pi_g$  and a non-terminating  $\pi_l$  with a negative step-wise (dense) reward is:

$$\alpha = \frac{\mu^{\frac{1}{H}} - (1 - \beta_l)}{\beta_g - (1 - \beta_l)} \quad (6)$$

**Result: Positive Dense Reward.** The derived sampling rate for a non-optimal  $\pi_g$  and a non-terminating  $\pi_l$  with a positive step-wise (dense) reward is:

$$\sum_{h=1}^H (\beta')^h (1 - \beta')^{1-\delta(H-h)} h r_t - \mu \hat{R}_{\pi_g} \geq 0 \quad (7)$$

where  $\beta' = \alpha \beta_g + (1 - \alpha)(1 - \beta_l)$ .

Given that the AntMaze environment definitely aligns to the non-terminating learner variation, it can be difficult to choose an appropriate  $\beta_l$  as per Section 3.2.2. We chose  $\beta_l = 0.1$  to be conservative, however given the far longer time between evaluations in AntMaze compared to Combination Lock, and the excellent warm-start form the guide, the agent is able to improve significantly even after just one update. Because of this, the online learner in Figure 3a degrades to nowhere near the threshold, we could have chosen an even riskier  $\beta_l$ . The parameters for these experiments are shown in Appendix F.3 and the results in Figure 3. Again, we see an overly-conservative LD and S25%, while S75% shows how overuse of  $\pi_l$  during initial training can affect performance.

![Figure 3: Negative Dense Reward Results. (a) Return: A line plot showing Return (y-axis, -100 to 100) vs Step (x-axis, 1.0 to 2.0). Four lines represent LD (orange), S25% (green), S75% (red), and GRL (blue). LD and S25% show a sharp drop to -100. S75% and GRL show a more gradual decline. A dashed green line at Return ≈ -50 indicates the original score. (b) Mean evaluation sampling rate: A line plot showing Sampling Rate (y-axis, 0.0 to 1.0) vs Step (x-axis, 1.0 to 2.0). LD (orange) increases linearly to 1.0. S25% (green) is constant at 0.25. S75% (red) is constant at 0.75. GRL (blue) starts at 0.25 and increases to 1.0, with a shaded area representing variance.](84e2ac543ffc4145dc85b05a48ec62e3_img.jpg)

Figure 3: Negative Dense Reward Results. (a) Return: A line plot showing Return (y-axis, -100 to 100) vs Step (x-axis, 1.0 to 2.0). Four lines represent LD (orange), S25% (green), S75% (red), and GRL (blue). LD and S25% show a sharp drop to -100. S75% and GRL show a more gradual decline. A dashed green line at Return ≈ -50 indicates the original score. (b) Mean evaluation sampling rate: A line plot showing Sampling Rate (y-axis, 0.0 to 1.0) vs Step (x-axis, 1.0 to 2.0). LD (orange) increases linearly to 1.0. S25% (green) is constant at 0.25. S75% (red) is constant at 0.75. GRL (blue) starts at 0.25 and increases to 1.0, with a shaded area representing variance.

Figure 3: Negative Dense Reward Results: (a) **Return.** The chosen return degradation threshold with  $\mu = 0.9$  is respected. The  $\hat{R}_{\pi_g}$  line indicates the original score of the guide. (b) **Mean evaluation sampling rate** of the learning policy  $\pi_l$  throughout training.

# 4 GUIDED REINFORCEMENT LEARNING WITH ROLL-BACK (GRL-RB)

As discussed above, the assumption of convergence between updates of  $\alpha$  ensures the success of GRL. However, if  $\pi_l$  has not yet converged such that we can be confident it is sampling the optimal action with  $1 - \alpha$  probability, then decreasing  $\alpha$  for the next step of training may cause GRL to over-sample the non-optimal action and cause the return to fall below Equation 2. To help the GRL algorithm recover from non-convergence, and minimise the number of evaluations that fall below the defined threshold, we introduce the Guided RL with Roll-Back (GRL-RB) (Algorithm 2, described fully in Appendix G). In GRL-RB, if the evaluation return falls below Equation 2, then  $\alpha$  is ‘rolled back’ to the value that attained the previous best return. Beyond maintaining good online performance, another benefit of the GRL-RB approach is to speed up the transfer of learning from  $\pi_g$  to  $\pi_l$ . GRL-RB allows the algorithm to take larger steps of  $\alpha$  (and more frequently), while minimising degradation through the roll-back mechanism.

As a straightforward demonstration of the value of GRL-RB, we make intentionally poor hyperparameter choices for GRL in the Combination Lock environment. This makes convergence unlikely, and we can immediately see how GRL struggles to respect the chosen threshold (Figure 4). By comparison, GRL-RB gives the learner the opportunity to return to the previous best-performing  $\alpha$ , and recover. Additional experimental results in this environment are provided in Appendix F. Experiments with the algorithms in a more challenging environment, AntMaze, are shown in Figure 5. We compare GRL-RB with JSRL (Uchendu et al., 2023) and another offline-to-online method, IQL (Kostrikov et al., 2022). See Appendix G.1.2 for details. To demonstrate GRL-RB’s flexibility to transfer hyperparameters, we include a robustness experiment in Figure 9 in Appendix G.1.3.

These results show the roll-back mechanisms effectiveness in helping GRL to maintain the performance threshold with non-ideal hyperparameter choices, and in challenging environments. In Figure 4, we show the maximum and minimum score attained across 50 runs to show all threshold violations. While LD is more effective in the beginning with small  $1 - \alpha$  values, it can also be pushed

{7}------------------------------------------------

![Figure 4: GRL, GRL-RB and LD in the Combination Lock environment. (a) Return: Evaluation Return vs Step. (b) Mean evaluation sampling rate: Evaluation Return vs Step.](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

Figure 4 consists of two line plots. Plot (a) shows 'Evaluation Return' on the y-axis (ranging from 0.0 to 1.2) against 'Step' on the x-axis (ranging from 0 to 250). It compares LD (blue dashed line), GRL (green solid line), and GRL-RB (red solid line). A horizontal red dashed line at approximately 0.8 represents the performance threshold. Plot (b) shows 'Evaluation Return' on the y-axis (ranging from 0.0 to 1.2) against 'Step' on the x-axis (ranging from 0 to 250). It compares LD (blue dashed line) and GRL (green solid line). Both plots include shaded regions representing the maximum and minimum scores across all runs.

Figure 4: GRL, GRL-RB and LD in the Combination Lock environment. (a) Return: Evaluation Return vs Step. (b) Mean evaluation sampling rate: Evaluation Return vs Step.

Figure 4: GRL, GRL-RB and LD in the Combination Lock environment, under variation 1, with an intentionally poor choice of hyperparameters. Shading shows the maximum and minimum score attained across all runs to highlight the performance threshold violations. (a) **Return**. The online training return. (b) **Mean evaluation sampling rate** of the learning policy  $\pi_l$  throughout training.

![Figure 5: Results for the Antmaze environments. (a) AntMaze Umaze v2. (b) AntMaze Medium Play v2. (c) AntMaze Large Play v2.](d864789b0d8384da1d22fd6a5d76bbdf_img.jpg)

Figure 5 displays three line plots labeled (a), (b), and (c), representing results for AntMaze environments. Plot (a) is for 'AntMaze Umaze v2', plot (b) for 'AntMaze Medium Play v2', and plot (c) for 'AntMaze Large Play v2'. Each plot shows 'Evaluation Return' on the y-axis against 'Step' on the x-axis. The x-axis is split into two segments: the first from 0.00 to 0.75 (or 1.00) and the second from 1.00 to 1.44 (or 1.75). The plots compare JSRL (blue dashed line), IQL (green dashed line), LD (red dashed line), and GRL-RB (Ours) (orange solid line). In all environments, GRL-RB (Ours) achieves higher evaluation returns faster and reaches higher final returns compared to the other algorithms.

Figure 5: Results for the Antmaze environments. (a) AntMaze Umaze v2. (b) AntMaze Medium Play v2. (c) AntMaze Large Play v2.

Figure 5: Results for the Antmaze environments. Offline pre-training phase at step  $< 1e6$ . The same pre-trained policy is then used for all algorithms as the online guide agent (step  $> 1e6$ ). While the stepwise penalty  $r = -1$  (from Section 3.2.3) was used as the guide’s evaluation score and to calculate the performance degradation threshold, we report the standard normalized AntMaze scores for better comparison with the literature.

onto the next stage before convergence (as  $1 - \alpha$  is increased no matter the evaluation score), so its performance can deteriorate at higher  $1 - \alpha$  values. GRL achieves a reasonably good average results, but clearly starts being challenged around 50 time steps, and cannot recover effectively. Around time step 125 by comparison, we see GRB-RL start to struggle, however its  $\alpha$  is immediately rolled back and its return quickly climbs above the threshold. We see further evidence of GRL-RB’s ability to effectively handle the trade-off between transfer speed and performance in Figures 8e and 8f, where even though GRL achieves a faster transfer, the return achieved by GRL-RB is better throughout training, as it self-adjusts to adhere to the threshold as required. Figures 5 and 9 show that guided policy methods that do not consider performance degradation struggle in the initial stages of transfer (even if they recover eventually in simpler environments, e.g. Figure 5a) so mechanisms like the roll-back are needed for applications where it is critical to maintain good performance.

# 5 DISCUSSION

Transferring from  $\pi_g$  to  $\pi_l$  is a careful balance between speed and managing performance degradation. For complex tasks,  $\pi_l$  may take time to learn the optimal behaviours in the environment, improving upon the behaviour of  $\pi_g$ . However, an extremely slow transfer may negate some of the benefits of warm-starting learning using a guide, and so ideally we would like  $\pi_g$  to transfer as fast as possible to  $\pi_l$ , while assuring some level of performance. In our first contribution, GRL, we introduced an algorithm that enabled a successful transfer from  $\pi_g$  to  $\pi_l$ . This algorithm is simple to implement on top of existing algorithms (e.g., IQL) and does not require any additional training compared to vanilla RL, as  $\pi_g$  stays static throughout learning. As  $\pi_g$  is sampled any time throughout an episode, it avoids using the learner for many consecutive time steps, which can take the agent so far from an optimal trajectory that it cannot recover (e.g. Figure 6). The user is free to set the initial sampling rate  $\alpha$  of GRL, and so in our second contribution, we show how this can be chosen most advantageously.

{8}------------------------------------------------

$\pi_I$  does need to explore to some extent, which will inevitably result in some level of degradation, however for many applications it is possible to specify an acceptable level of exploration degradation. In our second contribution, we have sought to take advantage of this specified performance degradation threshold. Our theoretical results in Section 3.2 show  $\alpha$  can be chosen in certain environments to enable the fastest transfer possible while respecting performance requirements. LD is reasonably effective but can be over-conservative and slow to transfer. The static  $\pi_g$  sampling algorithms (S25% and S75%) are clearly unable to converge unless  $\pi_g$  is already optimal, so it is surprising that this method continues to be used in the guided policy literature (Section 2). Our analysis has shown that GRL is the only method that respects the performance degradation threshold in all experiments.

In an ideal situation, GRL alone would be sufficient to enable knowledge transfer from  $\pi_g$  to  $\pi_I$ . The result in Figure 8a shows clearly that when the policy is able to fully converge between updates, the roll-back mechanism introduces no additional advantage to the algorithm. When convergence is not assured however, our third contribution, GRL-RB, is able to adaptively adjust its sample rate to better respect the performance degradation threshold. The results in Figure 8 (Appendix G.1.1) show the strong utility of the roll-back mechanism of GRL-RB in alleviating these issues. Specifically, Figure 8c-8h show that rolling back the sample rate allows the performance to recover. In Figure 8d, it is clear to see the learner sampling rate is dropped automatically at time steps above 250, due to the agent’s drop in evaluation score at the corresponding time steps as in Figure 8c. The agent’s score then gradually increases until it returns to its previous optimal value. On average, GRL-RB is able to maintain performance above the threshold even with obviously poor choices of hyperparameters.

The AntMaze results in Section 4 further demonstrate the utility of both GRL and GRL-RB, in a more complex environment. While IQL without any modification does very well when it has a variety of data available in its replay buffer (Table 6), it struggles when the replay buffer is cleared for online learning. This makes it unsuitable to applications without a dataset available (e.g., when  $\pi_g$  is a set of rules). Furthermore, the same policy is shifted from offline to online learning, meaning it must be of the same format (i.e., a neural network). JSRL also gets some reasonable results, however it suffers from the horizon-based action-selection, as discussed in Section 2. It can struggle to recover from the initial overuse of the learner agent in the more challenging environments (medium and large). Conversely, performance degradation does not occur in these environments for GRL-RB or LD, as our percentage-based sampling method ensures the restricted use of the learner. Even in the extreme case presented for the additional robustness experiment in Appendix G.1.3, the GRL-RB method is effective in preventing degradation by constantly rolling back the sampling rate when required.

A limitation of GRL-RB is that the roll-back is only triggered once the score has fallen below the threshold. A valuable addition would be a mechanism to predict or detect whether this is about to occur, to prevent the violation altogether. Further, future work could make use of a statistical method to compare the evaluation score distributions (as in Dasagi et al. (2019)), to provide an enhanced measure of evaluation score improvement. Finally, the derivations of  $\alpha$  provided above have a fairly limited scope compared to the wide variety of environments and reward schemes used in RL. In future work, we will aim to derive more results for other kinds of environments.

# 6 CONCLUSION

We introduce two algorithms, GRL and GRL-RB, to enable the transfer of prior knowledge from a guide policy  $\pi_g$  to learning policy  $\pi_I$ . Our theoretical analysis and experimental confirmation show that for environments with certain reward schemes, an initial sample rate  $\alpha$  for the  $\pi_g$  can be derived to ensure the mean evaluation score does not fall below a user-defined threshold. This is important, as many systems that currently use classical controllers may require to maintain a certain evaluation score, due to reasons including safety, cost or user experience. It is thus critical that algorithms designed to replace such systems have some guarantee of performance. To the best of our knowledge, this is the first time a performance guarantee has been established for a guided RL method.

The sample rates we derive do have a requirement that the agent is able to fully converge to the original evaluation score between evaluations, before  $\alpha$  is progressed. Choosing the correct hyperparameters to assure this can be challenging, so we introduced GRL-RB to allow the agent the opportunity to return to a previous best  $\alpha$ . Results in the Combination Lock and AntMaze environments show that GRL-RB enables effective self-correction, providing some flexibility for use in more complex environments.

{9}------------------------------------------------

## REFERENCES

- 486  
487  
488 Joshua Achiam. Spinning Up in Deep Reinforcement Learning. [https://spinningup.](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)  
489 [openai.com/en/latest/spinningup/rl\\_intro.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), 2018.
- 490 Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna:  
491 A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM*  
492 *SIGKDD international conference on knowledge discovery & data mining*, pp. 2623–2631, 2019.
- 493  
494 Richard Bellman. A markovian decision process. *Indiana Univ. Math. J.*, 6:679–684, 1957. ISSN  
495 0022-2518.
- 496  
497 Jonathan D Chang, Kianté Brantley, Rajkumar Ramamurthy, Dipendra Misra, and Wen Sun. Learning  
498 to generate better than your llm. *arXiv preprint arXiv:2306.11816*, 2023.
- 499  
500 Kai-Wei Chang, Akshay Krishnamurthy, Alekh Agarwal, Hal Daumé III, and John Langford. Learning  
501 to Search Better than Your Teacher. In *Proc. of the 32nd International Conference on Machine*  
502 *Learning (ICML 2015)*, volume 37, pp. 2058–2066, 2015.
- 503  
504 Lihan Chen, Lihong Xu, and Ruihua Wei. Energy-saving control algorithm of venlo greenhouse  
505 skylight and wet curtain fan based on reinforcement learning with soft action mask. *Agriculture*,  
506 13(1):141, 2023.
- 507  
508 Paul Daoudi, Bogdan Robu, Christophe Prieur, Ludovic Dos Santos, and Mervan Barlier. Enhancing  
509 reinforcement learning agents with local guides. *arXiv preprint arXiv:2402.13930*, 2024.
- 510  
511 Vibhavari Dasagi, Jake Bruce, Thierry Peynot, and Jürgen Leitner. Ctrl-Z: Recovering from Instability  
512 in Reinforcement Learning. *CoRR*, abs/1910.03732, 2019.
- 513  
514 Hal Daumé, John Langford, and Daniel Marcu. Search-based structured prediction. *Machine learning*,  
515 75:297–325, 2009.
- 516  
517 Gabriel Dulac-Arnold, Nir Levine, Daniel J Mankowitz, Jerry Li, Cosmin Paduraru, Sven Gowal, and  
518 Todd Hester. Challenges of real-world reinforcement learning: definitions, benchmarks and  
519 analysis. *Machine Learning*, 110(9):2419–2468, 2021.
- 520  
521 Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, and Jeff Clune. First return, then  
522 explore. *Nature*, 590(7847):580–586, February 2021.
- 523  
524 Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, and Pieter Abbeel. Reverse  
525 curriculum generation for reinforcement learning. In *Conference on robot learning*, pp. 482–495,  
526 2017.
- 527  
528 Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4RL: Datasets for Deep  
529 Data-Driven Reinforcement Learning. *CoRR*, abs/2004.07219, 2020.
- 530  
531 Scott Fujimoto and Shixiang (Shane) Gu. A minimalist approach to offline reinforcement learning.  
532 In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), *Advances*  
533 *in Neural Information Processing Systems*, volume 34, pp. 20132–20145, 2021.
- 534  
535 Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without  
536 exploration. In *International conference on machine learning*, pp. 2052–2062, 2019.
- 537  
538 Vinicius G Goecks, Gregory M Gremillion, Vernon J Lawhern, John Valasek, and Nicholas R  
539 Waytowich. Integrating behavior cloning and reinforcement learning for improved performance in  
540 dense and sparse reward environments. *arXiv preprint arXiv:1910.04281*, 2019.
- 541  
542 Alexander Hans, Daniel Schneegaß, Anton Maximilian Schäfer, and Steffen Udluft. Safe exploration  
543 for reinforcement learning. In *Proc. of the 16th European Symposium on Artificial Neural Networks*  
544 *(ESANN 2008)*, pp. 143–148, 2008.
- 545  
546 Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan,  
547 John Quan, Andrew Sendonaris, Ian Osband, et al. Deep q-learning from demonstrations. In  
548 *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018.

 Rest of paper (reference and Appendix) is removed.