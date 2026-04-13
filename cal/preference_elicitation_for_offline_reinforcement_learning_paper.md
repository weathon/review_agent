

{0}------------------------------------------------

# PREFERENCE ELICITATION FOR OFFLINE REINFORCEMENT LEARNING

**Alizée Pace**

ETH AI Center, ETH Zürich  
MPI for Intelligent Systems, Tübingen  
alizee.pace@ai.ethz.ch

**Bernhard Schölkopf**

MPI for Intelligent Systems &  
ELLIS Institute Tübingen

**Gunnar Rätsch**

ETH Zürich

**Giorgia Ramponi**

University of Zürich

## ABSTRACT

Applying reinforcement learning (RL) to real-world problems is often made challenging by the inability to interact with the environment and the difficulty of designing reward functions. Offline RL addresses the first challenge by considering access to an offline dataset of environment interactions labeled by the reward function. In contrast, Preference-based RL does not assume access to the reward function and learns it from preferences, but typically requires an online interaction with the environment. We bridge the gap between these frameworks by exploring efficient methods for acquiring preference feedback in a fully offline setup. We propose Sim-OPRL, an offline preference-based reinforcement learning algorithm, which leverages a learned environment model to elicit preference feedback on simulated rollouts. Drawing on insights from both the offline RL and the preference-based RL literature, our algorithm employs a pessimistic approach for out-of-distribution data, and an optimistic approach for acquiring informative preferences about the optimal policy. We provide theoretical guarantees regarding the sample complexity of our approach, dependent on how well the offline data covers the optimal policy. Finally, we demonstrate the empirical performance of Sim-OPRL in various environments.

## 1 INTRODUCTION

While reinforcement learning (RL) (Sutton and Barto, 2018) achieves excellent performance in various decision-making tasks (Kendall et al., 2019; Mirhoseini et al., 2020; Degrave et al., 2022), its practical deployment remains limited by the requirement of direct interaction with the environment. This can be impractical or unsafe in real-world scenarios. For example, patient management and treatment in intensive care units involve complex decision-making that has often been framed as a reinforcement learning problem (Raghu et al., 2017; Komorowski et al., 2018). However, the timing, dosage, and combination of treatments required are critical to patient safety, and incorrect decisions can lead to severe complications or death, making the use of traditional RL algorithms unfeasible (Gottesman et al., 2019; Tang and Wiens, 2021). Offline RL emerges as a promising solution, allowing policy learning from entirely observational data (Levine et al., 2020).

Still, a challenge with Offline RL is its requirement for an explicit reward function. Quantifying the numerical value of taking a certain action in a given environment state is challenging in many applications (Yu et al., 2021). Preference-based RL offers a compelling alternative, relying on comparisons between different actions or trajectories (Wirth et al., 2017) and being often easier for humans to provide (Christiano et al., 2017). In medical settings, for instance, clinicians may be queried for feedback on which trajectories lead to favorable outcomes. Unfortunately, most algorithms for preference acquisition require environment interaction (Saha et al., 2023; Chen et al., 2022; Lindner et al., 2021) and are therefore not applicable to the offline setting.

Lack of environment interaction and reward learning are thus two critical challenges for real-world RL deployment that are rarely tackled jointly. In this work, we address the problem of prefer-

{1}------------------------------------------------

ence elicitation for offline reinforcement learning by asking: *What trajectories should we sample to minimize the number of human queries required to learn the best offline policy?* This presents a challenging problem as it combines learning from offline data and active feedback acquisition, two frameworks that require opposing inductive biases for conservatism and exploration, respectively.

To the best of our knowledge, the only strategy proposed in prior work is to acquire feedback directly over samples within an offline dataset of trajectories (Shin et al., 2022, Offline Preference-based Reward Learning (OPRL)). We propose an alternative solution that queries feedback on *simulated rollouts* by leveraging a learned environment model. Our offline preference-based reinforcement learning algorithm, Sim-OPRL, strikes a balance between conservatism and exploration by combining pessimism when handling states out-of-distribution from the observational data (Jin et al., 2021; Zhan et al., 2023a), and optimism in acquiring informative preferences about the optimal policy (Saha et al., 2023; Chen et al., 2022). We study the efficiency of our approach through both theoretical and empirical analysis, demonstrating the superior performance of Sim-OPRL across various environments.

Our contributions are the following: (1) In Section 3, we first formalize the new problem setting of preference elicitation for offline reinforcement learning, which allows for **complementing offline data with preference feedback**. This framework is crucial for real-world RL applications where direct environment interaction is unsafe or impractical and reward functions are challenging to design manually, yet experts can be queried for their knowledge. (2) In Section 5, we provide theoretical guarantees on eliciting preference feedback over samples from an offline dataset, complementing work of Shin et al. (2022). (3) Next, in Section 6, we propose our own **efficient preference elicitation algorithm** based on simulated rollouts in a learned environment model, and establish its improved theoretical guarantees. (4) Finally, we develop a practical implementation of our algorithm and demonstrate its **empirical efficiency** and scalability across various decision-making environments.

## 2 RELATED WORK

Our problem setting shares similarities with Offline RL and Preference-based RL, which we summarize below. We position ourselves relative to our closest related works in Table 1 and extend our discussion in Appendix B.

**Offline RL.** Offline Reinforcement Learning has gained significant traction in recent years, as the practicality of training RL agents without environment interaction makes it relevant to real-world applications (Levine et al., 2020). However, learning from observational data only is a source of bias in the model, as the data may not cover the entire state-action space. Offline RL algorithms therefore output pessimistic policies, which has been shown to minimize suboptimality (Jin et al., 2021). Model-based approaches show particular promise for their sample efficiency (Yu et al., 2020; Kidambi et al., 2020; Rigter et al., 2022; Zhai et al., 2024; Uehara and Sun, 2021). In this work, we study the setting where reward signals are unavailable and must be estimated by actively querying preference feedback.

**Preference-based RL.** Rather than accessing numerical reward values for each state-action pair as in traditional online RL, preference-based RL learns the reward model through collecting pairwise preferences over trajectories (Wirth et al., 2017). Different preference elicitation strategies have been proposed for this framework, generally based on knowing the transition model exactly or on having access to the environment for rollouts (Christiano et al., 2017; Saha et al., 2023; Chen et al., 2022; Lindner et al., 2021; Zhan et al., 2023b; Sadigh et al., 2018; Brown et al., 2020). Prior theoretical and empirical work (Lindner et al., 2021; Chen et al., 2022) show that, in this setting, the most efficient preference elicitation strategy is to actively reduce the set of candidate optimal

Table 1: **Comparison of related work on preference elicitation.**

| Framework                     | Offline | Policy-Based Sampling | Robustness Guarantees | Practical Implementation |
|-------------------------------|---------|-----------------------|-----------------------|--------------------------|
| PbOP (Chen et al., 2022)      | ✗       | ✓                     | ✓                     | ✗                        |
| MoP-RL (Liu et al., 2023)     | ✗       | ✓                     | ✓                     | ✓                        |
| REGIME (Zhan et al., 2023b)   | ✗       | ✓                     | ✓                     | ✗                        |
| FREEHAND (Zhan et al., 2023a) | ✓       | ✗                     | ✓                     | ✓                        |
| OPRL (Shin et al., 2022)      | ✓       | ✗                     | ✗                     | ✓                        |
| <b>Sim-OPRL (Ours)</b>        | ✓       | ✓                     | ✓                     | ✓                        |

{2}------------------------------------------------

policies, rather maximize information gain on the reward function – our theoretical and empirical results reach the same conclusion for the offline setting.

**Offline Preference-based RL.** The development of preference-based RL algorithms based on offline data only is critical to settings where environment interaction is not feasible for safety and efficiency reasons. Still, this framework remains largely unexplored in the literature. While Zhu et al. (2023); Zhan et al. (2023a) demonstrate the value of pessimism in offline preference-based reinforcement learning, they do not consider how to query feedback actively. On the other hand, Shin et al. (2022) propose an empirical comparison of different preference sampling trajectories from an offline trajectories buffer. In Section 5, we provide a theoretical analysis of their approach, then propose an alternative sampling strategy based on simulated trajectory rollouts in Section 6, which benefits from both theoretical motivation and superior empirical performance.

## 3 PROBLEM FORMULATION

### 3.1 PRELIMINARIES

**Markov Decision Process.** We consider the episodic Markov Decision Process (MDP), defined by the tuple  $\mathcal{M} = (\mathcal{S}, \mathcal{A}, H, T, R)$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $H$  is the episode length,  $T : \mathcal{S} \times \mathcal{A} \rightarrow \Delta_{\mathcal{S}}$  is the transition function,  $R : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$  is the reward function. We assume an initial state  $s_0$ , but our analysis generalizes to a fixed initial state distribution. At time  $t$ , the environment is at state  $s_t \in \mathcal{S}$  and an agent selects an action  $a_t \in \mathcal{A}$ . The agent then receives a reward  $R(s_t, a_t)$  and the environment transitions to state  $s_{t+1} \sim T(\cdot | s_t, a_t)$ . We describe an agent's behavior through a policy function  $\pi : \mathcal{S} \rightarrow \Delta_{\mathcal{A}}$ , such that  $\pi(a|s)$  is the probability of taking action  $a$  in state  $s$ . Let  $\tau = (s_0, a_0, \dots, s_H, a_H)$  denote the trajectory of state-action pairs of an interaction episode with the environment. With an abuse of notation, we also write  $R(\tau) = \sum_t R(s_t, a_t)$ . Let  $d_{\pi}^{\tau}$  denote the distribution of trajectories (or state-action pairs, overloading notation) induced by rolling out policy  $\pi$  in transition model  $T$ . We denote the expected return of policy  $\pi$  as  $V_{T,R}^{\pi} = \mathbb{E}_{\tau \sim d_{\pi}^{\tau}}[R(\tau)]$ , and  $\pi^* = \operatorname{argmax}_{\pi} V_{T,R}^{\pi}$  denotes the optimal policy in  $\mathcal{M}$ .

**Preference-based Reinforcement Learning.** Rather than observing rewards for each state and action, we receive preference feedback over trajectories. For a pair of trajectories  $(\tau_1, \tau_2)$ , we obtain binary feedback  $o \in \{0, 1\}$  about whether  $\tau_1$  is preferred to  $\tau_2$ . We define the preference function  $P_R$  and assume that preference labels follow the Bradley-Terry model (Bradley and Terry, 1952):

$$P_R(\tau_1 \succ \tau_2) := P(o = 1 | \tau_1, \tau_2) = \frac{\exp(R(\tau_1))}{\exp(R(\tau_1)) + \exp(R(\tau_2))} = \sigma(R(\tau_1) - R(\tau_2)), \quad (1)$$

where  $\succ$  denotes a preference relationship and  $\sigma$  is a monotonous increasing link function. Within this framework, *preference elicitation* refers to the process of sampling preferences to obtain information about both the preference function and the system dynamics (Wirth et al., 2017).

### 3.2 OFFLINE PREFERENCE ELICITATION

We assume access to an observational dataset of trajectories  $\mathcal{D}_{\text{offline}} = \{\tau : \tau \sim d_{\pi_{\beta}}^{\tau}\}$ , where  $\pi_{\beta}$  is an unknown behavioural policy in  $\mathcal{M}$ . As in Offline RL, we do *not* have access to the environment to observe transition dynamics or rewards under alternative action choices. We assume *not* to have access to the reward function, but we can query preference feedback from a human to generate a dataset of preferences  $\mathcal{D}_{\text{pref}} = \{(\tau_1, \tau_2, o)\}$ .

**Optimality Criterion.** Based only on our offline dataset  $\mathcal{D}_{\text{offline}}$ , our goal is to recover a policy  $\hat{\pi}$  that minimizes suboptimality in the true environment with as few human preference queries as possible. Let  $\pi_{\text{offline}}^*$  denote the *optimal offline policy* estimated based on the offline data, with access to the true reward function  $R$ , and let  $\epsilon_T$  denote its suboptimality. Since preference elicitation only allows us to estimate the reward function, we do not aim to achieve a suboptimality less than  $\epsilon_T$  – although this is not formally a lower bound for our problem, as shown in Appendix A.3. Our objective is then formalized as follows.

**Definition 3.1** (Optimality Criterion of Offline Preference Elicitation). Let  $\pi^*$  be the optimal policy in  $\mathcal{M}$  and  $\hat{\pi}$  be the estimated optimal policy based on an offline dataset  $\mathcal{D}_{\text{offline}}$  and  $N_p > 0$  preference queries. Let  $\epsilon_T$  be the inherent suboptimality assuming access to the true reward function.

{3}------------------------------------------------

We say that a sampling strategy is  $(\epsilon, \delta, N_p)$ -correct if for every  $\epsilon \geq \epsilon_T$ , with probability at least  $(1 - \delta)$ , it holds that  $V_{\pi, R}^{\pi} - V_{\pi, R}^{\pi^*} \leq \epsilon$ .

Our work is the first to formalize this important problem, which faces the challenge of balancing **exploration** when actively acquiring feedback and **bias mitigation** in learning from offline data.

**Function classes.** We estimate the reward function and transition kernel with general function approximation; let  $\mathcal{F}_R$  and  $\mathcal{F}_T$  denote the classes of functions considered respectively. We also assume a policy class  $\Pi$ . Our theoretical analysis also requires the following assumptions and definitions, which are standard in preference-based RL (Chen et al., 2022; Zhan et al., 2023a).

**Assumption 3.1** (Realizability). *The true reward function belongs to the reward class:  $R \in \mathcal{F}_R$ . The true transition function belongs to the transition class:  $T \in \mathcal{F}_T$ . The optimal policy belongs to the policy class:  $\pi^* \in \Pi$ .*

**Assumption 3.2** (Boundedness). *The reward function is bounded:  $0 \leq \tilde{R}(\tau) \leq R_{\max}$  for all  $\tilde{R} \in \mathcal{F}_R$  and all trajectories  $\tau$ .*

**Definition 3.2** ( $\epsilon$ -bracketing number). *Let  $\mathcal{F}$  be a class of real functions  $f : \mathcal{X} \rightarrow \mathbb{R}$ . We say  $(l, u)$  is an  $\epsilon$ -bracket if  $l(x) \leq u(x)$  and  $|u(x) - l(x)|_1 \leq \epsilon$  for all  $x \in \mathcal{X}$ . The  $\epsilon$ -bracketing number of  $\mathcal{F}$ , denoted  $\mathcal{N}_{\mathcal{F}}(\epsilon)$ , is the minimal number of  $\epsilon$ -brackets  $(l^n, u^n)_{n=1}^N$  needed so that for any  $f \in \mathcal{F}$ , there is a bracket  $i \in [N]$  containing it, meaning  $l^i(x) \leq f(x) \leq u^i(x)$  for all  $x \in \mathcal{X}$ .*

Let  $\mathcal{N}_{\mathcal{F}_R}(\epsilon)$  and  $\mathcal{N}_{\mathcal{F}_T}(\epsilon)$  denote the  $\epsilon$ -bracketing numbers of  $\mathcal{F}_R$  and  $\mathcal{F}_T$  respectively. This measures the complexity of the function classes (Geer, 2000). For instance, with linear rewards  $\mathcal{F}_R := \{R : \tau \rightarrow \theta^T \phi(\tau)\}$ , the  $\epsilon$ -bracket number is bounded by  $\log \mathcal{N}_{\mathcal{F}_R}(\epsilon) \leq \mathcal{O}(d \log \frac{B G}{\epsilon})$ , where  $\|\phi\|_2 \leq B$  and  $\|\phi(\tau)\|_2 \leq G \forall \tau \in \mathcal{T}$ , and  $d$  is the dimension of the feature space (Zhan et al., 2023a).

**Definition 3.3** (Transition concentrability coefficient, Zhan et al. (2023a)). *The concentrability coefficient w.r.t. transition classes  $\mathcal{F}_T$  and the optimal policy  $\pi^*$  is defined as:*

$$C_T(\mathcal{F}_T, \pi^*) = \sup_{\tilde{T} \in \mathcal{F}_T} \left[ \frac{\mathbb{E}_{(s,a) \sim d_T^{\pi^*}} [\|T(\cdot|s, a) - \tilde{T}(\cdot|s, a)\|_1]}{\sqrt{\mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{offline}}} [\|T(\cdot|s, a) - \tilde{T}(\cdot|s, a)\|_1^2]}} \right],$$

The concentrability coefficient measures the coverage of the optimal policy in the offline dataset. Note that  $C_T$  is upper-bounded by the density-ratio coefficient:  $C_T(\mathcal{F}_T, \pi^*) \leq \sup_{(s,a) \in \mathcal{S} \times \mathcal{A}} d_T^{\pi^*}(s, a) / d_T^{\pi_\beta}(s, a)$ , where  $\pi_\beta$  is the behavioural policy underlying  $\mathcal{D}_{\text{offline}}$ .

## 4 OFFLINE PREFERENCE-BASED RL WITH PREFERENCE ELICITATION

In this section, we propose a general framework for offline preference-based reinforcement learning. The next two sections propose two different preference elicitation strategies. As learning must be carried out in two distinct stages, with environment dynamics based on  $\mathcal{D}_{\text{offline}}$  and reward learning on  $\mathcal{D}_{\text{pref}}$ , we adopt a model-based approach which we summarize in Algorithm 1.

**Model Learning.** We first leverage the offline data to learn a model of the environment dynamics, fitting a transition model  $\tilde{T}$  and an uncertainty function  $u_T$  through maximum likelihood:

$$\begin{aligned} \tilde{T} &= \operatorname{argmax}_{\tilde{T} \in \mathcal{F}_T} \mathbb{E}_{(s,a,s') \sim \mathcal{D}_{\text{offline}}} [\log \tilde{T}(s'|s, a)], \\ u_T(s, a) &= \max_{\tilde{T}_1, \tilde{T}_2 \in \mathcal{T}} |\tilde{T}_1(\cdot|s, a) - \tilde{T}_2(\cdot|s, a)|_1 \cdot R_{\max}, \end{aligned}$$

where  $\mathcal{T} = \{\tilde{T} \in \mathcal{F}_T \mid \mathbb{E}_{(s,a,s') \sim \mathcal{D}_{\text{offline}}} [\log (\tilde{T}(s'|s, a) / \hat{T}(s'|s, a))] \leq \beta_T\}$ , defining a confidence set over the maximum likelihood estimator (MLE), and  $\beta_T$  is a margin hyperparameter.

**Iterative Preference Elicitation and Reward Learning.** As with the transition model, our algorithm estimates the reward function  $\tilde{R}$  and its uncertainty function through maximum likelihood over iteratively collected preference data  $\mathcal{D}_{\text{pref}}$ :

$$\begin{aligned} \tilde{R} &= \operatorname{argmax}_{\tilde{R} \in \mathcal{F}_R} \mathbb{E}_{(\tau_1, \tau_2, o) \sim \mathcal{D}_{\text{pref}}} [o \log P_{\tilde{R}}(\tau_1 \succ \tau_2) + (1 - o) \log P_{\tilde{R}}(\tau_2 \succ \tau_1)], \\ u_R(\tau) &= \max_{\tilde{R}_1, \tilde{R}_2 \in \mathcal{R}} |\tilde{R}_1(\tau) - \tilde{R}_2(\tau)|_1, \end{aligned}$$

{4}------------------------------------------------

#### **Algorithm 1** Offline Preference-based Reinforcement Learning with Preference Elicitation

**Input:** Observational trajectories dataset  $\mathcal{D}_{\text{offline}}$ . Significance  $\delta \in (0, 1)$ , preference budget  $N_p$ .**Output:**  $\hat{\pi}^*$ 

- 1: Estimate  $\hat{T}$  and  $u_T$  via maximum likelihood over the observational data  $\mathcal{D}_{\text{offline}}$ .
- 2:  $\mathcal{D}_{\text{pref}} \leftarrow \emptyset$ .
- 3: **for**  $k = 1, \dots, N_p$  **do**
- 4:   Generate trajectory pairs  $(\tau_1, \tau_2)$ . **► Preference Elicitation:** Sections 5 and 6
- 5:   Collect preference label  $o$  for  $(\tau_1, \tau_2)$ .
- 6:    $\mathcal{D}_{\text{pref}} \leftarrow \mathcal{D}_{\text{pref}} \cup \{(\tau_1, \tau_2, o)\}$ .
- 7:   Estimate  $\hat{R}$  and  $u_R$  via maximum likelihood over the preference data  $\mathcal{D}_{\text{pref}}$ .
- 8: **end for**
- 9:  $\hat{\pi}^* \leftarrow \operatorname{argmax}_{\pi \in \Pi} \mathbb{E}_{\tau \sim d_{\pi}^{\pi}} [\hat{R}(\tau) - u_R(\tau) - u_T(\tau)]$

where  $\mathcal{R} = \{\hat{R} \in \mathcal{F}_R \mid \mathbb{E}_{(\tau_1, \tau_2, o) \sim \mathcal{D}_{\text{pref}}} [\log(P_{\hat{R}}(\tau_1 \succ \tau_2) / P_R(\tau_1 \succ \tau_2))] \leq \beta_R\}$  defines the confidence set and  $\beta_R$  is a hyperparameter. We also define preference uncertainty between two trajectories  $\tau_1, \tau_2$ :

$$u_{P_R}(\tau_1, \tau_2) = \max_{\hat{R}_1, \hat{R}_2 \in \mathcal{R}} |P_{\hat{R}_1}(\tau_1 \succ \tau_2) - P_{\hat{R}_2}(\tau_1 \succ \tau_2)|_1. \quad (2)$$

The choice of trajectory sampling strategy for preference elicitation in line 4 (Algorithm 1) is critical to efficiently obtaining an  $\epsilon$ -optimal policy. We present two possible strategies in Sections 5 and 6.

**Pessimistic Policy Optimization.** Finally, our algorithm outputs a policy  $\hat{\pi}^*$  that is optimal while ensuring robustness to modeling error. This means optimizing for the worst-case value function over the remaining transition and reward uncertainties (Levine et al., 2020):

$$\hat{\pi}^* = \operatorname{argmax}_{\pi \in \Pi} \min_{T \in \mathcal{T}, \hat{R} \in \mathcal{R}} V_{\hat{T}, \hat{R}}^{\pi}. \quad (3)$$

Based on this objective, we define the pessimistic transition and reward models as follows:  $\hat{T}_{\text{inf}}, \hat{R}_{\text{inf}} = \operatorname{argmin}_{\hat{T} \in \mathcal{T}, \hat{R} \in \mathcal{R}} \max_{\pi \in \Pi} V_{\hat{T}, \hat{R}}^{\pi}$ . Our analysis provides a worst-case robustness guarantee when considering well-calibrated confidence intervals, as detailed in Sections 5.1 and 6.1. In other words, following prior work (Chen et al., 2022; Zhan et al., 2023a), our theoretical analysis assumes that modeling elements can be identified with no optimization error. We then complement this algorithmic framework with a flexible practical implementation in Section 6.3.

## 5 PREFERENCE ELICITATION FROM OFFLINE TRAJECTORIES

A first strategy for preference elicitation without environment interaction is to sample trajectories directly from the offline dataset. Shin et al. (2022) propose this approach as Offline Preference-based Reward Learning (OPRL), and design a uniform and uncertainty-sampling variant:

- OPRL Uniform Sampling:**  $\tau_1, \tau_2 \sim \mathcal{D}_{\text{offline}}$
- OPRL Uncertainty Sampling:**  $\tau_1, \tau_2 = \operatorname{argmax}_{\tau_1, \tau_2 \in \mathcal{D}_{\text{offline}}} u_{P_R}(\tau_1, \tau_2)$

We provide a theoretical analysis of the performance of OPRL.

### 5.1 THEORETICAL GUARANTEES.

We obtain the following result, proven in Appendix A.4. The suboptimality of the estimated policy  $\hat{\pi}^*$  is bounded by the policy evaluation error for the optimal policy  $\pi^*$ . This error decomposes into a term depending on transition model estimation, and one on reward model estimation.

**Theorem 5.1.** For any  $\delta \in (0, 1]$ , let  $\beta_T = c_T^{\text{MLE}} \log(HN_{\mathcal{T}}(1/N_o)/\delta)/N_o$  and  $\beta_R = c_R^{\text{MLE}} \log(N_{\mathcal{T}}(1/N_p)/\delta)/N_p$ , where  $N_o = H|\mathcal{D}_{\text{offline}}|$  is the number of observed transitions in the observational dataset and  $c_T^{\text{MLE}}, c_R^{\text{MLE}}$  are universal constants. The policy  $\hat{\pi}^*$  estimated by Algorithm 1, with preference elicitation based on offline trajectories, achieves the following suboptimal-

{5}------------------------------------------------

ity with probability  $1 - \delta$ :

$$V^{\pi^*} - V^{\hat{\pi}^*} \leq \underbrace{H R_{\max} C_T(\mathcal{F}_T, \pi^*) \sqrt{\frac{C_T}{N_o} \log\left(\frac{H}{\delta} \mathcal{N}_{\mathcal{F}_T}\left(\frac{1}{N_o}\right)\right)}}_{\text{transition term } \epsilon_T} + \underbrace{2\alpha\kappa C_R(\mathcal{F}_R, \pi^*) \sqrt{\frac{C_R}{N_p} \log\left(\frac{1}{\delta} \mathcal{N}_{\mathcal{F}_R}\left(\frac{1}{N_p}\right)\right)}}_{\text{reward term}}$$

where  $\alpha = 1$  for uniform sampling or  $\alpha \leq 1$  for uncertainty sampling,  $C_R$  is a concentrability measure for the reward function,  $\kappa = \sup_{r \in [-R_{\max}, R_{\max}]} \frac{1}{\sigma^2(r)}$  measures the degree of non-linearity of the link function, and  $C_T, C_R$  are universal constants.

In the special case where both the transition and reward functions are learned on a fixed initial preference dataset (no preference elicitation;  $|\mathcal{D}_{\text{offline}}| = 2N_p$ ), we recover Theorem 1 from Zhan et al. (2023a). Importantly, the coefficient  $\alpha$  allows us to motivate the superior efficiency of uncertainty sampling over uniform sampling, observed empirically in Shin et al. (2022) and in our own experiments (Section 7). Uncertainty sampling learns accurate reward models with fewer preference queries when  $\alpha < 1$ , but can perform like uniform sampling in harder problems ( $\alpha = 1$ ).

## 6 PREFERENCE ELICITATION FROM SIMULATED TRAJECTORIES

We now propose our alternative strategy for generating trajectories for offline preference elicitation: **Simulated Offline Preference-based Reward Learning (Sim-OPRL)**. This method simulates trajectories  $(\tau_1, \tau_2)$  by leveraging the *learned environment model*. This overcomes a limitation of OPRL, which is only designed to reduce uncertainty about the reward functions in  $\mathcal{R}$ , by instead reducing uncertainty about which policies are plausibly optimal. Our approach is inspired by efficient online preference elicitation algorithms (Saha et al., 2023; Chen et al., 2022), which we modify for practical implementation. We account for the offline nature of our problem by avoiding regions that are out of the distribution of the data: the sampling strategy is *optimistic* with respect to uncertainty in rewards, but *pessimistic* with respect to uncertainty in transitions.

We summarize our approach to generating simulated trajectories for preference elicitation in Algorithm 2. First, we construct a set of candidate optimal policies  $\Pi_{\text{offline}}$ , containing policy  $\pi_{\text{offline}}^*$  (optimal policy under the pessimistic model and the true reward function) with high probability – as demonstrated in Appendix A.5.2. Next, within this set of candidate policies, we identify the two most exploratory policies  $\pi_1, \pi_2$ , chosen to maximize preference uncertainty  $u_{P_R}$ . Finally, we roll out these policies within our learned transition model to generate a trajectory pair  $(\tau_1, \tau_2)$  for preference feedback.

#### --- Algorithm 2 Preference Elicitation through Simulated Trajectory Sampling. ---

**Input:** Pessimistic transition model  $\hat{T}_{\text{inf}}$ . Reward confidence set  $\mathcal{R}$  and preference uncertainty function  $u_{P_R}$ .

**Output:**  $(\tau_1, \tau_2)$

- 1: Estimate optimal offline policy set:  $\Pi_{\text{offline}} = \{\pi \mid \exists \hat{R} \in \mathcal{R} : \pi = \text{argmax}_{\pi \in \Pi} \mathbb{E}_{\tau \sim d_{\hat{T}_{\text{inf}}}^{\pi}} [\hat{R}(\tau)]\}$
  - 2: Identify exploratory policies:  $\pi_1, \pi_2 = \text{argmax}_{\pi_1, \pi_2 \in \Pi_{\text{offline}}} \mathbb{E}_{\tau_1 \sim d_{\hat{T}_{\text{inf}}}^{\pi_1}, \tau_2 \sim d_{\hat{T}_{\text{inf}}}^{\pi_2}} [u_{P_R}(\tau_1, \tau_2)]$
  - 3: Rollouts in model:  $\tau_1 \sim d_{\hat{T}_{\text{inf}}}^{\pi_1}, \tau_2 \sim d_{\hat{T}_{\text{inf}}}^{\pi_2}$ .
- 

We first provide a theoretical analysis of the performance of Sim-OPRL, before proposing a practical implementation of our entire preference elicitation and policy optimization algorithm.

### 6.1 THEORETICAL GUARANTEES

We decompose suboptimality in a similar way to Section 5.1, but obtain a reward suboptimality term that depends on the learned dynamics model instead of the true one, and on  $\pi_{\text{offline}}^*$  instead of  $\pi^*$ :

$$V^{\pi^*} - V^{\hat{\pi}^*} \leq \underbrace{(V_{T, R}^{\pi^*} - V_{\hat{T}_{\text{inf}}, R}^{\pi^*})}_{\text{transition term } \epsilon_T} + \underbrace{(V_{\hat{T}_{\text{inf}}, R}^{\pi_{\text{offline}}^*} - V_{\hat{T}_{\text{inf}}, \hat{R}_{\text{inf}}}^{\pi_{\text{offline}}^*})}_{\text{reward term}}. \quad (4)$$

Analysis of the suboptimality due to transition error is identical to above, but the reward term is thus significantly different. By design, our sampling strategy ensures good coverage of preferences over  $\pi_{\text{offline}}^*$  within the learned environment model, which **eliminates the concentrability term for the reward  $C_R$** . We refer the reader to Appendix A.5 for the proof of Theorem 6.1.

{6}------------------------------------------------

**Theorem 6.1.** For any  $\delta \in (0, 1]$ , let  $\beta_T = c_T^{\text{MLE}} \log(HN_{\mathcal{F}_T}(1/N_o)/\delta)/N_o$  and  $\beta_R = c_R^{\text{MLE}} \log(N_{\mathcal{F}_R}(1/N_p)/\delta)/N_p$ , where  $N_o = H|D_{\text{offline}}|$  is the number of observed transitions in the observational dataset and  $c_T^{\text{MLE}}, c_R^{\text{MLE}}$  are universal constants. The policy  $\hat{\pi}^*$  estimated by Algorithm 1, with a preference sampling strategy based on rollouts in the learned transition model, achieves the following suboptimality with probability  $1 - \delta$ :

$$V^{\pi^*} - V^{\hat{\pi}^*} \leq \underbrace{HR_{\max} C_T(\mathcal{F}_T, \pi^*) \sqrt{\frac{c_T}{N_o} \log\left(\frac{H}{\delta} N_{\mathcal{F}_T}\left(\frac{1}{N_o}\right)\right)}}_{\text{transition term } \epsilon_T} + 2\kappa \underbrace{\sqrt{\frac{c_R}{N_p} \log\left(\frac{1}{\delta} N_{\mathcal{F}_R}\left(\frac{1}{N_p}\right)\right)}}_{\text{reward term}}$$

where  $\kappa = \sup_{r \in [-R_{\max}, R_{\max}]} \frac{1}{\sigma(r)}$  measures the degree of non-linearity of the link function, and  $c_T, c_R$  are universal constants.

### 6.2 DISCUSSION

Our theoretical results demonstrate that the learned policy can achieve performance comparable to the optimal policy, and thus satisfy our optimality criterion in Definition 3.1, provided it is covered by the offline data ( $C_T(\mathcal{F}_T, \pi^*), C_R(\mathcal{F}_R, \pi^*) < \infty$ ). Following our analysis, a suboptimal dataset requires more preferences to achieve a certain policy performance, as the concentrability terms  $C_T$  or  $C_R$  are large. Empirical results in Section 7 confirm that sample efficiency is worse when the behavioral policy is more suboptimal.

**Offline Trajectories vs. Simulated Rollouts.** While both OPRL and Sim-OPRL depend on the offline dataset for estimating environment dynamics, they induce different suboptimality in modeling preference feedback. Simulated rollouts are designed to achieve good coverage of the optimal offline policy  $\pi_{\text{offline}}^*$ , which avoids wasting preference budget on trajectories with low rewards or high transition uncertainty. In contrast, as shown in Zhan et al. (2023a), due to the dependence of preferences on full trajectories, the reward concentrability term  $C_R$  in Theorem 5.1 can be very large. While sampling from the offline buffer is not sensitive to the quality of the transition model, good coverage of the optimal policy is needed from both transition and preference data to achieve low suboptimality.

**Transition vs. Preference Model Quality.** Our theoretical analysis also suggests an interesting trade-off in the sample efficiency of our approach, depending on the accuracy of the transition model. The width of the confidence interval reduces as significance parameter  $\delta$  or dataset size increase, or as function class complexity  $N_{\mathcal{F}_T}$  decreases. For a target suboptimality gap  $\epsilon$ , provided the optimal offline policy  $\pi_{\text{offline}}^*$  has a gap  $\epsilon_T < \epsilon$ , then the number of preferences required is of the order of  $\mathcal{O}(\log(1/\delta)/(\epsilon - \epsilon_T)^2)$ . A more accurate transition model should therefore require fewer preference samples to achieve a given suboptimality, which we again confirm empirically.

### 6.3 PRACTICAL IMPLEMENTATION

We now complete the general algorithmic framework discussed above with a possible implementation strategy, allowing for empirical validation. In fact, with minor changes to the following framework, our paper also proposes a feasible implementation of related theoretical algorithms (Chen et al., 2022; Zhan et al., 2023a). We refer the reader to Appendix C for further detail.

**Model Learning and Policy Optimization.** Following prior work in offline reinforcement learning (Yu et al., 2020), we train ensembles of  $N_T$  and  $N_R$  neural network models for the transition and reward functions on different bootstraps of the data (Lakshminarayanan et al., 2017), denoted  $\{\hat{T}_1, \dots, \hat{T}_{N_T}\}$  and  $\{\hat{R}_1, \dots, \hat{R}_{N_R}\}$ . We estimate MLE and uncertainty functions as follows:

$$\hat{T}(\cdot | s, a) = \frac{1}{N_T} \sum_{i=1}^{N_T} \hat{T}_i(\cdot | s, a); \quad u_T(s, a) = \max_{i, j \in [1, N_T]} |\hat{T}_i(\cdot | s, a) - \hat{T}_j(\cdot | s, a)|_1 \cdot R_{\max}$$

$$\hat{R}(s, a) = \frac{1}{N_R} \sum_{i=1}^{N_R} \hat{R}_i(s, a); \quad u_R(s, a) = \max_{i, j \in [1, N_R]} |\hat{R}_i(s, a) - \hat{R}_j(s, a)|_1$$

{7}------------------------------------------------

Each  $\hat{R}_i$  in the ensemble has an associated preference function defined by the Bradley-Terry model, with  $\sigma$  as the sigmoid function. We obtain preference uncertainty through variation over the ensemble as in Equation (2). Recall that transition and reward models are trained on  $\mathcal{D}_{\text{offline}}$  and  $\mathcal{D}_{\text{pref}}$  respectively; for computational efficiency, we sample preferences in batches of  $\sigma$  to reduce the number of reward model updates needed.

We approximate the pessimistic objective in Equation (3) by penalizing the reward function with the uncertainty, as in Lagrangian formulations of model-based offline RL (Yu et al., 2020; Rigter et al., 2022). We solve for the following objective with a traditional reinforcement learning algorithm:

$$\hat{\pi}^* = \text{argmax}_{\pi \in \Pi} \mathbb{E}_{(s,a) \sim d_{\pi}^{\pi}} [\hat{R}(s, a) - \lambda_R u_R(s, a) - \lambda_T u_T(s, a)], \quad (5)$$

where hyperparameters  $\lambda_T, \lambda_R$  control the degree of conservatism. Note that in our theoretical analysis, this was achieved through parameters  $\beta_T, \beta_R$  which affect the width of the confidence intervals  $u_R$  and  $u_T$ , but their exact value cannot be estimated. We show in Appendix A.2 that Equation (5) indeed lower bounds the true value function, under well-calibrated uncertainty estimates.

**Near-Optimal Policy Set and Exploratory Policies.** Sim-OPRL requires constructing  $\Pi_{\text{offline}}$ , a set of near-optimal policies within a pessimistic model of the environment. Following Lindner et al. (2021), we obtain a policy model for each element  $\hat{R}_i$  of the reward ensemble. Policy models are optimized to maximize returns under the transition model  $\hat{T}$  and the reward function  $\hat{R}_i - \lambda_T u_T$ , ensuring pessimism w.r.t transitions. Next, the most exploratory policies are identified by generating rollouts of each candidate policy within the learned model  $\hat{T}$ . The trajectories  $(\tau_1, \tau_2)$  induced by different policies and maximizing the preference uncertainty function  $u_{P_R}(\tau_1, \tau_2)$  are used for preference feedback. We refer the reader to Appendix C for further detail.

## 7 EXPERIMENTAL RESULTS

In this section, we demonstrate the effectiveness of our preference elicitation strategy, Sim-OPRL, across a range of offline reinforcement learning environments and datasets. We demonstrate its **superior performance over OPRL, as expected from our theoretical analysis**.

Since our closest related works do not propose any experimental validation (Chen et al., 2022; Zhan et al., 2023a), we propose a practical implementation of Preference-based Optimistic Planning (PbOP) in Appendix C; this elicitation method queries feedback over trajectory rollouts in the *true environment* (Chen et al., 2022). We also compare against OPRL (Shin et al., 2022) with uniform and uncertainty-sampling. Finally, we report the performance of  $\pi_{\text{offline}}^*$  and  $\pi^*$  as upper bounds for the performance of our algorithm: the former is trained in the learned transition model with access to the true reward, and the latter has full knowledge of both transition and reward function.

We compare different preference elicitation strategies on a range of environments detailed in Appendix D. Among others, we explore environments from the D4RL benchmark (Fu et al., 2020) identified as particularly challenging offline preference-based reinforcement learning tasks (Shin et al., 2022), as well as a medical simulation designed to model the evolution of patients with sepsis (Oberst and Sontag, 2019). As detailed in Appendix D, these environments consist of high-dimensional state spaces with continuous or discrete action spaces, follow complex transition dynamics, and have sparse or non-linear rewards and termination conditions. This makes them representative of the challenge of learning a reward function and learning offline in a real-world application. In particular, the sepsis simulation environment is commonly used in medically-motivated offline RL work (Tang and Wiens, 2021; Pace et al., 2023), and highlights another advantage of Sim-OPRL over OPRL: it does not require feedback on real trajectories from the observational dataset. In a sensitive setting such as healthcare where data access is carefully controlled, it may be attractive to query experts about *synthetic* trajectories rather than real samples.

**Performance against State-of-the-Art.** Performance and sample complexity results with different preference elicitation methods are given in Figure 1 and Table 2. Within the offline approaches, Sim-OPRL consistently achieves better environment returns than OPRL with much fewer preference queries. In line with our theoretical analysis, our empirical results therefore demonstrate that policy-based sampling in Sim-OPRL is more efficient than maximizing information gain on the reward function (uncertainty-based OPRL), which echoes similar conclusions reached in prior work on online preference elicitation (Lindner et al., 2021; Chen et al., 2022).

{8}------------------------------------------------

Table 2: **Sample complexity**  $N_\epsilon$  **under different preference elicitation strategies**, to reach a suboptimality gap of  $\epsilon = 20$  over normalized returns. Mean and 95% confidence interval over 6 experiments. The best-performing offline method is highlighted in bold. ✗ marks when the target suboptimality could not be achieved. Note that PbOP has an advantage by having access to direct interaction with the environment.

| Environment        | OPRL Uniform | OPRL Uncertainty | <b>Sim-OPRL (Ours)</b>         | PbOP (Online) |
|--------------------|--------------|------------------|--------------------------------|---------------|
| Star MDP           | $32 \pm 4$   | $30 \pm 4$       | <b><math>4 \pm 2</math></b>    | $4 \pm 2$     |
| Gridworld          | $105 \pm 11$ | $66 \pm 7$       | <b><math>49 \pm 7</math></b>   | $32 \pm 4$    |
| MiniGrid-FourRooms | $92 \pm 7$   | $53 \pm 5$       | <b><math>41 \pm 5</math></b>   | $25 \pm 3$    |
| HalfCheetah-Random | $108 \pm 9$  | $71 \pm 8$       | <b><math>50 \pm 10</math></b>  | $36 \pm 3$    |
| Sepsis Simulation  | ✗            | $642 \pm 72$     | <b><math>225 \pm 46</math></b> | $75 \pm 11$   |

![Figure 1: Environment returns under different preference elicitation strategies. The figure contains five subplots (a-e) showing normalized environment returns (0-100) against the number of preferences. (a) StarMDP: Sim-OPRL (Ours) reaches near-optimal returns quickly. (b) Gridworld: Sim-OPRL (Ours) and PbOP (Online) perform best. (c) MiniGrid-FourRooms: Sim-OPRL (Ours) and PbOP (Online) perform best. (d) HalfCheetah-Random: Sim-OPRL (Ours) and PbOP (Online) perform best. (e) Sepsis Simulation: Sim-OPRL (Ours) and PbOP (Online) perform best. OPRL methods (Uniform and Uncertainty) consistently underperform.](bedcca5cdf168e3508ef511d94ec514c_img.jpg)

Figure 1: Environment returns under different preference elicitation strategies. The figure contains five subplots (a-e) showing normalized environment returns (0-100) against the number of preferences. (a) StarMDP: Sim-OPRL (Ours) reaches near-optimal returns quickly. (b) Gridworld: Sim-OPRL (Ours) and PbOP (Online) perform best. (c) MiniGrid-FourRooms: Sim-OPRL (Ours) and PbOP (Online) perform best. (d) HalfCheetah-Random: Sim-OPRL (Ours) and PbOP (Online) perform best. (e) Sepsis Simulation: Sim-OPRL (Ours) and PbOP (Online) perform best. OPRL methods (Uniform and Uncertainty) consistently underperform.

Figure 1: **Environment returns under different preference elicitation strategies**. Mean and 95% confidence interval over 6 experiments. Environment returns are normalized between 0 and 100. Only OPRL and Sim-OPRL are fully offline.

As an upper bound for the performance of our algorithm, we include baselines that have access to the environment: we report the performance of the optimal policy  $\pi^*$ , as well as that of an algorithm querying feedback over optimistic rollouts in the *real* environment (Chen et al., 2022, PbOP). In Figure 1, the PbOP method naturally reaches a superior policy with fewer samples as it allows environment interaction and can thus improve its estimate of the transition model in parallel to learning the preference function. As supported by our theoretical analysis, this result stresses the importance of having a high-quality transition model to make our method effective. We explore this in more detail in our following ablations.

**Algorithm Ablations.** We conduct ablations for our algorithm on a simple tabular MDP, with results in Figure 2. This example (transition and reward details deferred to Appendix D) illustrates the importance of pessimism with respect to the transition model. Even with access to true rewards,  $\pi^*_{\text{offline}}$  is pessimistic to avoid the out-of-distribution state, as it is unclear how to reach it. Thus, in Figure 2, we see a drop in performance if pessimism is not applied to the output policy (purple lines). This confirms the theoretical insights from Zhu et al. (2023); Zhan et al. (2023a), who demonstrate the importance of pessimism in offline preference-based RL problems. Pessimism is also crucial in simulated rollouts, to avoid wasting preference budget on regions of low confidence — as value estimates are inaccurate in any case. This is reflected in lower performance without pessimism w.r.t the transition model

![Figure 2: Algorithm ablations (StarMDP). The plot shows normalized environment returns (0-100) against the number of preferences. Sim-OPRL (Ours) reaches near-optimal returns. Sim-OPRL without pessimism in output policy or rollouts shows significantly lower performance, indicating the importance of pessimism.](e8ff6e66c77a8e96203c9f8db8f0986f_img.jpg)

Figure 2: Algorithm ablations (StarMDP). The plot shows normalized environment returns (0-100) against the number of preferences. Sim-OPRL (Ours) reaches near-optimal returns. Sim-OPRL without pessimism in output policy or rollouts shows significantly lower performance, indicating the importance of pessimism.

Figure 2: **Algorithm ablations (StarMDP)**.

{9}------------------------------------------------

![Figure 3: Preference sample complexity N_p as a function of the properties of the observational data. (a) As a function of offline dataset size. (b) As a function of dataset optimality.](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 3 consists of two line plots. Plot (a) shows the 'Number of Preferences' (y-axis, 0 to 50) versus the 'Number of Offline Trajectories' (x-axis, 0 to 100). Plot (b) shows the 'Number of Preferences' (y-axis, 0 to 50) versus the optimality ratio  $\|d^*(s, a) / d^{\pi_\theta}(s, a)\|_\infty$  (x-axis, 1 to 100). Both plots compare 'OPRL Uncertainty' (blue line with circles) and 'Sim-OPRL (Ours)' (green line with circles). Shaded regions represent 95% confidence intervals, and 'x' marks indicate points where the target suboptimality could not be achieved.

Data for Plot (a):

| Number of Offline Trajectories | OPRL Uncertainty (N <sub>p</sub> ) | Sim-OPRL (Ours) (N <sub>p</sub> ) |
|--------------------------------|------------------------------------|-----------------------------------|
| 10                             | 48                                 | 48                                |
| 20                             | 38                                 | 32                                |
| 30                             | 35                                 | 18                                |
| 40                             | 32                                 | 8                                 |
| 50                             | 22                                 | 5                                 |
| 60                             | 20                                 | 5                                 |
| 70                             | 18                                 | 4                                 |
| 80                             | 15                                 | 3                                 |
| 90                             | 12                                 | 2                                 |
| 100                            | 10                                 | 1                                 |

Data for Plot (b):

| $\ d^*(s, a) / d^{\pi_\theta}(s, a)\ _\infty$ | OPRL Uncertainty (N <sub>p</sub> ) | Sim-OPRL (Ours) (N <sub>p</sub> ) |
|-----------------------------------------------|------------------------------------|-----------------------------------|
| 1                                             | 50                                 | 22                                |
| 2                                             | 30                                 | 5                                 |
| 5                                             | 32                                 | 5                                 |
| 10                                            | 35                                 | 15                                |
| 20                                            | 38                                 | 35                                |
| 50                                            | 48                                 | 50                                |
| 100                                           | 50                                 | 50                                |

Figure 3: Preference sample complexity N\_p as a function of the properties of the observational data. (a) As a function of offline dataset size. (b) As a function of dataset optimality.

Figure 3: **Preference sample complexity  $N_p$  as function of the properties of the observational data**, to reach a suboptimality gap of  $\epsilon = 20$  over normalized environment returns (Star MDP). Mean and 95% confidence intervals over 6 experiments.  $\times$  marks when the target suboptimality could not be achieved.

in Figure 2 (brown line), and which **could be seen as the naive adaptation of online preference elicitation methods to our setting** (Chen et al., 2022; Lindner et al., 2022). We also note the importance of optimism with respect to the reward uncertainty, both in OPRL in Figure 1 and in our own model-based rollouts in Figure 2.

**Transition vs. Preference Model Quality.** Next, we empirically study the trade-off between transition and preference model performance in our problem setting. Still in the Star MDP, in the low-data regime, the error  $\epsilon_T$  incurred in estimating the value function due to the misspecification of the transition model is large. As dictated by our theoretical analysis and as visualized in Figure 3a, this significantly increases the number of preference samples  $N_p$  required to achieve good final performance. At the other end of the spectrum, if the offline dataset is large and allows modeling the transition model accurately, then  $\epsilon_T$  is small and the number of preference samples  $N_p$  needed shrinks. We observe a similar trend for both Sim-OPRL and our OPRL uncertainty-sampling baseline.

We also measure how the coverage of the optimality of the dataset affects performance in our setting. In Figure 3b, we vary the behavioral policy  $\pi_\theta$  underlying the offline data, ranging from optimal (density ratio coefficient = 1) to highly suboptimal (large density ratio coefficient). The concentrability terms  $C_T$  and  $C_R$  are challenging to measure as they require considering entire function classes, but we report the accuracy of the maximum likelihood estimate for both models in Appendix E. We observe that preference elicitation methods perform best when the data is close to optimal (with the exception of a fully optimal, non-diverse dataset making reward learning from preferences challenging). More preference samples are required if the observational dataset has poor coverage of the optimal policy (large  $C_T(\mathcal{F}_T, \pi^*)$ ), as the transition and reward models become less accurate for the trajectory distribution of interest. We also validate this conclusion on HalfCheetah datasets of varying optimality in Appendix E.

## 8 CONCLUSION

Our work shows the potential of integrating human feedback within the framework of offline RL. We address the challenges of preference elicitation in a fully offline setup by exploring two key methods: sampling from the offline dataset (Shin et al., 2022, OPRL) and generating model rollouts (Sim-OPRL). By employing a pessimistic approach to handle out-of-distribution data and an optimistic strategy to acquire informative preferences, Sim-OPRL balances the need for robustness and informativeness in learning an optimal policy.

We provide theoretical guarantees on the sample complexity of both approaches, demonstrating that performance depends on how well the offline data covers the optimal policy. Empirical evaluations on various environments confirm the practical effectiveness of our algorithm, as Sim-OPRL consistently outperforms OPRL baselines in all settings.

Overall, our approach not only advances the state-of-the-art in offline preference-based RL but also takes a significant step toward improving the practical utility of offline RL. This opens up new avenues for real-world applications of RL in healthcare, robotics, and manufacturing, where interaction with the environment is challenging but domain experts can be queried for feedback. Looking forward, a natural extension will be to explore alternative sources of information from experts, still without direct environment interaction.

 Rest of paper (reference and Appendix) is removed.