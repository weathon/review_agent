

{0}------------------------------------------------

# BRIDGING THE GAP BETWEEN SL AND TD LEARNING VIA Q-CONDITIONED MAXIMIZATION

Anonymous authors

Paper under double-blind review

## ABSTRACT

Recent research highlights the efficacy of supervised learning (SL) as a methodology within reinforcement learning (RL), yielding commendable results. Nonetheless, investigations reveal that SL-based methods lack the stitching capability typically associated with RL approaches such as TD learning, which facilitate the resolution of tasks by stitching diverse trajectory segments. This prompts the question: *How can SL methods be endowed with stitching property and bridge the gap with TD learning?* This paper addresses this challenge by exploring the maximization of the objective in the goal-conditioned RL. We introduce the concept of Q-conditioned maximization supervised learning, grounded in the assertion that the goal-conditioned RL objective is equivalent to [maximizing the expected Q-function under given goal distribution](#), thus embedding Q-function maximization into traditional SL-based methodologies. Building upon this premise, we propose **Goal-Conditioned Reinforced Supervised Learning (GCREinSL)**, which enhances SL-based approaches by incorporating [maximizing Q-function](#). **GCREinSL** emphasizes the maximization of the Q-function during the training phase to predict the maximum [Q-function within the distribution](#). This [optimized in-distribution Q-function is then employed during the inference phase to guide the selection of optimal actions](#). We demonstrate that **GCREinSL** enables SL methods to exhibit stitching property, effectively equivalent to applying goal data augmentation to SL methods. Experimental results on offline datasets designed to evaluate stitching capability show that our approach not only effectively selects appropriate goals across diverse trajectories but also outperforms previous works that applied goal data augmentation to SL methods.

## 1 INTRODUCTION

Recently, numerous methods that frame reinforcement learning RL as a purely SL problem ([Schmidhuber, 2020](#); [Chen et al., 2021](#); [Emmons et al., 2021](#); [Chane-Sane et al., 2021a](#)) function by correlating input states and desired goals with optimal actions. These techniques assign labels to state-action pairs based on future outcomes (e.g., achieving a goal) derived from offline datasets, subsequently maximizing the likelihood of these actions as optimal for producing the intended results. Collectively termed outcome-conditioned behavioral cloning algorithms (OCBC), these approaches have exhibited commendable performance on standard offline benchmarks ([Emmons et al., 2021](#)). Nevertheless, recent investigations ([Yang et al., 2023](#); [Ghugare et al., 2024](#)) have highlighted a critical shortcoming of these SL methodologies: the lack of trajectory stitching capability. This property, commonly found in temporal-difference (TD)-based RL algorithms employing dynamic programming (e.g., CQL([Kumar et al., 2020](#)), and IQL([Kostrikov et al., 2021a](#))), is vital for addressing tasks that require the integration of multiple trajectory segments. Thus, enhancing OCBC methods to incorporate this characteristic and bridging the gap with TD approaches has emerged as a significant area of research.

In this paper, we examine this issue within goal-conditioned RL, focusing on navigating between certain state-goal pairs that, while not co-occurring during training, are present in isolation. In [sparse-reward goal-conditioned RL](#), [TD-based RL methods](#) often face challenges such as [instability during training](#) due to difficulties in accurately estimating the value function, [inefficiencies in optimization](#) ([Van Hasselt et al., 2018](#); [Kumar et al., 2019a](#)), and [high sensitivity to hyperparameters](#) ([Henderson et al., 2018](#)). In contrast, OCBC methods are simpler, more efficient, and free from these issues,

{1}------------------------------------------------

making the development of novel OCBC approaches highly valuable. However, OCBC lacks the critical trajectory stitching property inherent to TD-based RL methods. Addressing this limitation to enable stitching and bridge performance gaps in challenging environments is a key focus of current research. We have observed that certain sequence modeling (Yamagata et al., 2023a; Wu et al., 2023; Zhuang et al., 2024) techniques are enabling Decision Transformer (DT) (Chen et al., 2021) within OCBC methods to acquire stitching property. However, these methods are primarily effective within goal-conditioned scenarios. Drawing motivation and inspiration from state-of-the-art max-return sequence modeling method (Zhuang et al., 2024), we propose the concept of Q-conditioned maximization supervised learning within the context of goal-conditioned RL. Specifically, since the objective in goal-conditioned RL is equivalent to maximizing the expected Q-function across all possible goals under the given goal distribution, we commence in Section 4.1 by examining a maze example to illustrate the detrimental impact of naively setting the Q-function to highest possible value on trajectory stitching. An illustrative example, shown in Fig. 1, highlights the relationship between a failing trajectory (with  $Q = 0$ , where the agent starts from the initial state but fails to reach the final goal) and a successful trajectory (with  $Q = 1$ , where the agent reaches the final goal but does not originate from the initial state). Ideally, the Q-function should start at 0 and shift to 1 when transitioning to the successful trajectory. This requirement contrasts with the oversimplified approach of artificially assigning a Q-function of 1.

And then we propose the concept of Q-conditioned maximization supervised learning, a framework that embeds the maximization of Q-function into supervised learning. This approach aims not only to maximize the probability of selecting appropriate actions but also to predict the highest attainable in-distribution Q-function. To achieve this, we utilize expectile regression (Aigner et al., 1976; Sobotka & Kneib, 2012), which seeks to ensure that the predicted Q-function closely approximates the maximum Q-function that can be realized from the available historical trajectory. In the inference pipeline, the model first predicts the current maximum Q-function and then identifies the best action based on the offline dataset distribution, guided by this predicted maximum. Our findings indicate that Q-conditioned maximization supervised learning acts as a form of goal data augmentation for OCBC methods, leading to substantial improvements in their stitching capability. Additionally, we present Goal-Conditioned Reinforced Supervised Learning (**GCReinSL**), which implements Q-conditioned maximization supervised learning for OCBC methods, including DT (Chen et al., 2021) and Reinforcement Learning via Supervised Learning (RvS) (Emmons et al., 2021). This framework reinforces supervised learning through the maximization of the Q-function. In scenarios involving trajectory stitching, as demonstrated in Fig. 1, GCReinSL typically predicts a value of 0 at the starting point and transitions to a prediction of 1 upon switching to a successful trajectory, reflecting the predicted in-distribution maximum Q-function.

We briefly summarize our main contributions as follows: (1) Inspired by max-return sequence modeling (Zhuang et al., 2024), we propose a novel supervised learning framework in goal-conditioned RL based on our concept of Q-conditioned maximization, which endows OCBC methods with stitching ability. (2) We demonstrate that **GCReinSL** is equivalent to goal data augmentation for OCBC methods. (3) Experimental results in Ghugare et al. (2024) offline datasets, designed to test stitching ability, show that **GCReinSL** not only significantly enhances the stitching capability of OCBC methods but also outperforms relevant goal data augmentation works. Additionally, in the goal-conditioned D4RL (Fu et al., 2020) offline datasets, our method continues to outperform related sequence modeling methods which also perform trajectory stitching.

## 2 RELATED WORK

**Goal-conditioned RL.** This paper focus on goal-conditioned RL, a topic explored extensively in prior research through various methodologies. Approaches such as conditional supervised learning (Ding et al., 2019; Gupta et al., 2020; Lynch et al., 2020; Ghosh et al., 2021; Emmons et al., 2021), actor-critic frameworks (Andrychowicz et al., 2017; Nachum et al., 2018; Zhu et al., 2021; Chane-Sane et al., 2021b), model-based strategies (Schmeckpeper et al., 2020; Charlesworth & Montana, 2020; Mendonca et al., 2021), and distance metric learning (Tian et al., 2020; Nair et al., 2020; Durugkar et al., 2021; Liu et al., 2023a; Wang et al., 2023; Reichlin et al., 2024) have been employed to learn goal-conditioned policies. These methods have demonstrated success across diverse tasks, including real-world robotic systems (Ma et al., 2022; Shah et al., 2022; Zheng et al., 2023a). Unlike techniques that depend on manually defined reward or distance functions, our approach builds on a

{2}------------------------------------------------

self-supervised formulation of goal-conditioned RL, treating the task as one of predicting future state visitation (Eysenbach et al., 2020; 2022b; Zheng et al., 2023b; Ghugare et al., 2024).

**The Stitching Property** The concept of stitching, as discussed by Ziebart et al. (2008), is a characteristic property of TD-learning algorithms such as those described by Kumar et al. (2020); Kostrikov et al. (2021a), which employ dynamic programming techniques. This property enables these algorithms to integrate data from diverse trajectories, thereby improving their ability to handle complex tasks by effectively utilizing available data (Cheikhhi & Russo, 2023). On the other hand, most SL-based RL methods lack this property. Brandfonbrener et al. (2022); Yang et al. (2023) provide examples where SL algorithms do not perform stitching and Ghugare et al. (2024) also indicates this from the perspective of combinatorial generalisation. In contrast, we use a simple maze example to illustrate this viewpoint from the perspective of maximizing the RL objective.

**Data Augmentation in RL** Data augmentation, as an efficient method for improving generalization ability, has been applied in RL (Lu et al., 2020; Stone et al., 2021; Kalashnikov et al., 2021; Hansen & Wang, 2021; Kostrikov et al., 2021b; Yarats et al., 2021) and SL (Shorten & Khoshgoftaar, 2019). We have noticed that some methods (Char et al., 2022; Yamagata et al., 2023b; Paster et al., 2023) use dynamic programming to enhance existing trajectories to improve the performance of SL algorithms. However, they still require dynamic programming. Another methods which are very similar to ours is to only perform data augmentation for SL (Yang et al., 2023; Ghugare et al., 2024). However, they may have the problem of not being able to correctly provide the augmented goal data such as unreachabele goals. Unlike these two methods, we approach from the perspective of maximizing the goal-conditioned RL objective and endow the SL method with the ability to stitch trajectories, providing agents with a more reasonable selection of augmented goals.

## 3 PRELIMINARIES

### 3.1 GOAL-CONDITIONED RL IN CONTROLLED MARKOV PROCESS

We will study the problem of goal-conditioned RL in a controlled Markov process with states  $s \in \mathcal{S}$ , actions  $a \in \mathcal{A}$ . The dynamics are  $p(s' | s, a)$ , the initial state distribution is  $p_0(s_0)$ , the discount factor is  $\gamma$ , and a reward function  $r(s, a, g)$  for each goal. The goal-conditioned policy  $\pi(a | s, g)$  is conditioned on a pair of state and goal  $s, g \in \mathcal{S}$ .

We denote the  $t$ -step action-conditioned policy distribution  $p_{\pi}^t(s_t | s_0, a_0)$  as the distribution of states  $t$  steps in the future given the initial state  $s_0$  and action  $a_0$  under  $\pi$ . For a policy  $\pi$ , define as the distribution over states visited after exactly  $t$  steps. We define the discounted state occupancy distribution as:

$$p_{\pi}^+(s_t | s, a) \triangleq (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t p_{\pi}^t(s_t | s, a), \quad (1)$$

where  $s_{t+}$  is the variable that specifies a future state corresponding to the discounted state occupancy distribution. For a given distribution over goals  $g \sim p_G$ , the objective of the policy  $\pi$  is to maximize the probability of reaching the goal  $g$  in the future:

$$\max_{\pi(\cdot | \cdot, \cdot)} \mathbb{E}_{p_0(s_0)p_G(g)\pi(a_0 | s_0, g)} [p_{\pi}^+(g | s_0, a_0)]. \quad (2)$$

Following prior work (Eysenbach et al., 2020; Chane-Sane et al., 2021b; Blier et al., 2021; Rudner et al., 2021; Eysenbach et al., 2022b; Bortkiewicz et al., 2024), we define the reward function  $r(s, a, g)$  for each goal as the probability of reaching the goal at the next time step:

$$r(s_t, a_t, g) \triangleq (1 - \gamma)p(s_{t+1} = g | s_t, a_t). \quad (3)$$

And the Q-function can be defined for a policy  $\pi(\cdot | \cdot, g)$ :

$$Q^{\pi}(s, a, g) \triangleq \mathbb{E}_{\pi(\cdot | g)} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t, g) \mid \begin{array}{l} s_0 = s, \\ a_0 = a \end{array} \right]. \quad (4)$$

**Theorem 3.1** (Rephrased from Proposition 1 of Eysenbach et al. (2022b)). *The Q-function for the goal-conditioned reward function in Eq. (4) is equivalent to the probability of goal  $g$  under the discounted state occupancy distribution:*

$$Q^{\pi}(s, a, g) = p_{\pi}^+(s_{t+} = g | s, a). \quad (5)$$

{3}------------------------------------------------

The proof is in Appendix A.1. This proposition indicates that Q-function is equivalent to the discounted state occupancy distribution. Thus, from Eq. (2) and Eq. (5), we can conclude that the objective of the policy  $\pi$  in goal-conditioned RL is equivalent to maximizing the expected Q-function over all possible goals under the given goal distribution  $p_G(g)$ .

**Remark 1.** Translating rewards to probabilities simplifies the analysis of goal-conditioned RL problem and allows probabilistic estimation methods (e.g., VAE (Kingma & Welling, 2014)) to be repurposed for Q-function estimation.

Our work focuses on the offline goal-conditioned RL setting (Levine et al., 2020), the agent can only access a static offline dataset  $\mathcal{D}$  and cannot interact with the environment. The offline dataset  $\mathcal{D}$  can be collected by some unknown policies (Levine et al., 2020; Prudencio et al., 2023). We can express the offline dataset as  $\mathcal{D} := \{\tau_i\}_{i=1}^N$  (Ghugare et al., 2024), where  $\tau_i := \{\langle s_0^i, a_0^i, r_0^i \rangle, \langle s_1^i, a_1^i, r_1^i \rangle, \dots, \langle s_T^i, a_T^i, r_T^i \rangle\}$  is the goal-conditioned trajectory and  $N$  is the number of stored trajectories. In each  $\tau_i$  for  $i \in 1, \dots, N$ ,  $s_0^i \sim p_0(s_0)$ .

### 3.2 OUTCOME CONDITIONAL BEHAVIORAL CLONING (OCBC) METHODS

We present empirical results using a simple and popular class of goal-conditioned RL methods: Outcome conditional behavioral cloning (Eysenbach et al., 2022a) (DT (Chen et al., 2021), URL (Schmidhuber, 2020), RvS (Emmons et al., 2021), GCSL (Chane-Sane et al., 2021a) and many others (Sun et al., 2019; Kumar et al., 2019b)). These SL methods take as input the offline dataset  $\mathcal{D}$  and learn a goal-conditioned policy  $\pi(a | s, g)$  using a maximum likelihood objective:

$$\max_{\pi(\cdot|\cdot)} \mathbb{E}_{(s,a,g) \sim \mathcal{D}} [\log \pi(a | s, g)]. \quad (6)$$

## 4 METHODOLOGY

In this section, we start with a simple maze example to illustrate why classical OCBC methods and the naive Q-conditioned maximization approach are unlikely to solve the trajectory stitching problem. And then we employ a VAE as a neural probability estimation model to approximate the Q-function. Further, we introduce the concept of Q-conditioned maximization supervised learning and theoretically demonstrate that this paradigm can achieve maximum Q-function without encountering out-of-distribution (OOD) issues. We also demonstrate that Q-conditioned maximization supervised learning is equivalent to goal data augmentation for OCBC methods. Finally, we outline the implementation details of our Q-conditioned maximization supervised learning, **GCReinSL**, focusing on three key aspects: the model architecture, the loss function utilized during training, and the inference pipeline.

### 4.1 TRAJECTORY STITCHING EXAMPLE

In the offline RL literature, trajectory stitching has garnered significant attention. Ideally, an offline agent should be able to combine overlapping suboptimal trajectories into optimal ones (Kostrikov et al., 2021a; Liu et al., 2023b). Both theoretical (Ghugare et al., 2024) and empirical studies (Yang et al., 2023) have demonstrated that SL methods lack the ability to perform effective stitching. The following example provides a detailed explanation of this limitation.

**Example** The Fig. 1 depicts a toy maze, where  $s_0^1$  is the starting state,  $g$  is the final goal with reward  $r = 1$ ,  $g'$  is a boom goal with  $r = -1$  and other states are all  $r = 0$ . The offline dataset contains two trajectories one trajectory  $\tau_1$  starts from the initial state  $s_0^1$  and reach the goal  $g_1$  but doesn't reach the final goal while another  $\tau_2$  reaches the final goal  $g$  but doesn't start from  $s_0^1$ .  $s_t$  is the intersection of two trajectories and  $g'$  is the boom goal that we aim to avoid reaching. Trajectory stitching expects the agent can follow the first half of  $\tau_1$  (from starting state  $s_0^1$  to  $s_t$ ) and then take the second half of  $\tau_2$  (from  $s_t$  to the goal  $g$ ) to reach the goal. We first explain why the typical OCBC methods might fail.

![Figure 1: A maze example for trajectory stitching analysis. The diagram shows a grid world maze. A starting state s_0^1 is at the bottom left. A goal g is at the top right, marked with a star and labeled (r=1). A boom goal g' is at the bottom left, marked with a red dot and labeled (r=-1). Two trajectories, tau_1 and tau_2, are shown. tau_1 starts at s_0^1 and ends at g_1. tau_2 starts at s_0^2 and ends at g. The trajectories intersect at state s_t. The maze contains several other states, some with rewards (r=0).](f5e131a3fffe09aa98db055df84e4378_img.jpg)

Figure 1: A maze example for trajectory stitching analysis. The diagram shows a grid world maze. A starting state s\_0^1 is at the bottom left. A goal g is at the top right, marked with a star and labeled (r=1). A boom goal g' is at the bottom left, marked with a red dot and labeled (r=-1). Two trajectories, tau\_1 and tau\_2, are shown. tau\_1 starts at s\_0^1 and ends at g\_1. tau\_2 starts at s\_0^2 and ends at g. The trajectories intersect at state s\_t. The maze contains several other states, some with rewards (r=0).

Figure 1: A maze example for trajectory stitching analysis.

{4}------------------------------------------------

If we set initial Q-function as  $\hat{Q}_0 = 0$  at the starting state, the agent will smoothly reach the intersection state  $s_t$ . However, since Q-function is still zero  $\hat{Q}_t = 0$  at the state  $s_t$ , OCBC methods will reach the state  $g_1$  rather than  $g$ . Only when  $\hat{Q}_t = 1$ , OCBC methods is possible to follow  $\tau_2$ . But  $\hat{Q}_t = 1$  is impossible to obtain given  $\hat{Q}_0 = 0$ . If we apply the naive max approach and set the initial  $\hat{Q}_0 = 1$ , the agent might directly walk towards the boom goal  $g'$  ( $r = -1$ ) because  $\hat{Q}_0 = 1$  is the OOD Q-function for the starting state.

If the OCBC methods are endowed with capability to maximize the Q-function like goal-conditioned RL, Let's see what might happen. At the starting state  $s_0^1$ , only  $\tau_1$  is contained in dataset so the model will predict  $\hat{Q}_0 = 0$ . When offline agent comes to the intersection  $s_t$ , the latter segments of both trajectories are available. If the OCBC methods are able to maximize Q-function, then  $\tau_2$  is more likely to be selected since the Q-function  $Q = 1$  is larger. This inspires us to bring the capability of maximizing Q-function back into supervised learning.

### 4.2 Q-FUNCTION ESTIMATION WITH VAE

The central aim of goal-conditioned RL is to identify the best action for a given state and goal by maximizing the Q-function. To achieve this, the first task is to accurately estimate the Q-function. Drawing on previous research (Wu et al., 2022) and Theorem 3.1, we implement a Variational Autoencoder (VAE) architecture as a probabilistic modeling tool. More specifically, we apply a Conditional Variational Autoencoder (CVAE) (Sohn et al., 2015) for probability estimation. In our framework, the probability  $p_{\pi}^{\pi}(g | s_0 = s, a)$  is modeled by a Deep Latent Variable Model, expressed as  $p_{\psi}(g | s, a) = \int p_{\psi}(g | z, s, a) p(z | s, a) dz$ , with a prior distribution  $p(z | s, a) = \mathcal{N}(0, I)$ . Although directly calculating the marginal likelihood  $p_{\psi}(g | s, a)$  is computationally infeasible, VAE utilizes an approximate posterior  $q_{\varphi}(z | s, a, g) \approx p_{\psi}(z | s, a, g)$ , enabling joint optimization of  $\psi$  and  $\varphi$  parameters via the evidence lower bound (ELBO):

$$\begin{aligned} \log p_{\psi}(g | s, a) &\geq \mathbb{E}_{q_{\varphi}(z | s, a, g)} \left[ \log \frac{p_{\psi}(g, z | s, a)}{q_{\varphi}(z | s, a, g)} \right] \\ &= \mathbb{E}_{q_{\varphi}(z | s, a, g)} [\log p_{\psi}(g | z, s, a)] - \text{KL} [q_{\varphi}(z | s, a, g) || p(z | s, a)] \\ &\stackrel{\text{def}}{=} -\mathcal{L}_{\text{ELBO}}(s, a; \varphi, \psi). \end{aligned} \quad (7)$$

After training this VAE, we can approximate the probability  $p_{\pi}^{\pi}(g | s, a)$  in Eq. (5) by  $-\mathcal{L}_{\text{ELBO}}$ . To obtain an estimation with lower bias between  $\log p_{\psi}(g | s, a)$  and  $p_{\pi}^{\pi}(g | s, a)$  in Eq. (5), we use the importance sampling technique following Rezende et al. (2014); Kingma & Welling (2019); Wu et al. (2022):

$$\begin{aligned} \log p_{\psi}(g | s, a) &= \log \mathbb{E}_{q_{\varphi}(z | s, a, g)} \left[ \frac{p_{\psi}(g, z | s, a)}{q_{\varphi}(z | s, a, g)} \right] \\ &\approx \mathbb{E}_{z^{(l)} \sim q_{\varphi}(z | s, a, g)} \left[ \log \frac{1}{L} \sum_{l=1}^L \frac{p_{\psi}(a, g, z^{(l)} | s)}{q_{\varphi}(z^{(l)} | s, a, g)} \right] \\ &\stackrel{\text{def}}{=} \widehat{\log p_{\pi}^{\pi}}(g | s, a; \varphi, \psi, L). \end{aligned} \quad (8)$$

From the reward and probability transformation in Theorem 3.1, the value of the Q-function can be derived.

### 4.3 Q-CONDITIONED MAXIMIZATION SUPERVISED LEARNING

After estimating the Q-function, we aim to equip supervised learning with additional maximizing Q-function objective, analogous to the methods employed in RL. And during inference, the supervised learning can select optimal action conditioned on the in-distribution maximized Q-function. We introduce the expectile regression as Q-function loss to achieve this.

Expectile regression (Newey & Powell, 1987) is well studied in applied statistics and econometrics and has been introduced into offline RL recently (Kostrikov et al., 2021a; Wu et al., 2023; Zhuang et al., 2024). Specifically, the Q-function loss based on the expectile regression is as follows:

$$\mathcal{L}_Q^m = \mathbb{E}_{(s, a, g) \in \mathcal{D}} [m - \mathbb{1}(\Delta Q < 0)] \Delta Q^2, \quad (9)$$

{5}------------------------------------------------

![Figure 2: Illustration of weight. A plot titled 'Expectile Regression' showing the loss function |m - 1/(Q < 0)|ΔQ^2 on the y-axis against ΔQ on the x-axis. Three curves are shown for m = 0.5 (blue), m = 0.7 (orange), and m = 0.9 (green). The curves are U-shaped, with the minimum at ΔQ = 0. The green curve (m = 0.9) is the steepest for positive ΔQ, while the blue curve (m = 0.5) is the steepest for negative ΔQ. A red arrow points to the right side of the plot, indicating that as m increases, the weight increases for positive ΔQ.](c54b3ca7603d65d4589151bc3a49d054_img.jpg)

Figure 2: Illustration of weight. A plot titled 'Expectile Regression' showing the loss function |m - 1/(Q < 0)|ΔQ^2 on the y-axis against ΔQ on the x-axis. Three curves are shown for m = 0.5 (blue), m = 0.7 (orange), and m = 0.9 (green). The curves are U-shaped, with the minimum at ΔQ = 0. The green curve (m = 0.9) is the steepest for positive ΔQ, while the blue curve (m = 0.5) is the steepest for negative ΔQ. A red arrow points to the right side of the plot, indicating that as m increases, the weight increases for positive ΔQ.

Figure 2: Illustration of weight.

here  $Q = Q^\pi(s, a, g)$ ,  $\Delta Q = Q - \hat{Q}$  and  $\hat{Q}$  can come from the supervised learning model (e.g. DT model can independently predict both the Q-function and the corresponding actions). Here  $m \in (0, 1)$  is the hyperparameter of expectile regression. When  $m = 0.5$ , expectile regression degenerates into standard regression, also MSE loss.  $\hat{Q}$ , which aligns with the asymmetric curves in Fig. 2. But when  $m > 0.5$ , this asymmetric loss will give more weights to the  $Q$  larger than  $\hat{Q}$ . Besides, the red arrow shows the weight increases as the  $m$  becomes larger. In other words, the predicted Q-function  $\hat{Q}$  will approach larger  $Q$ .

To unveil what the Q-function loss function has learned and offer a formal elucidation of its role, we introduce the following theorem:

**Theorem 4.1.** Suppose Q-function is predict by the model itself, we first define  $\text{SG} \doteq (s, g, a, Q)$ . For  $m \in (0, 1)$ , denote  $Q^m(\text{SG}) = \arg \min \mathcal{L}_Q^m(\text{SG})$ , then we have

$$\lim_{m \rightarrow 1} Q^m(\text{SG}) = Q_{\max},$$

where  $Q_{\max} = \max_{s, a, g} Q(s, a, g)$  denotes the maximum Q-function with actions from offline dataset.

The proof is in Appendix A.2. In other words, Theorem 4.1 indicates the loss  $\mathcal{L}_Q^m$  will make the model predict the maximum Q-function when  $m \rightarrow 1$ , which is similar to the maximizing objective in goal-conditioned RL.

**Corollary 1.** The concept of Q-conditioned maximization supervised learning is equivalent to applying goal data augmentation for supervised learning (SL) methods, enabling it to exhibit stitching property.

The proof is in Appendix A.3. Corollary 1 indicates that Q-conditioned maximization supervised learning can select state-goal pairs formed by trajectory stitching, which is consistent with the discussion presented in Section 4.1.

### 4.4 IMPLEMENTATION OF GCREinSL

Now, we will focus on the specific implementation of GCREinSL, describing the architecture input and output, training, and inference procedures. Specifically, this section describes the training and inference pipeline using two typical OCBC algorithms: DT and RvS. Other supervised learning algorithms can be implemented in a similar manner. The overall structure of GCREinSL for DT is depicted in Fig. 3, with RvS being similar, differing only in terms of its architecture.

#### 4.4.1 GCREinSL FOR DT

**Model Architecture** To accommodate the Q-conditioned maximization for DT (Chen et al., 2021), which predicts the maximum Q-function and utilizes it as a condition to guide the generation of optimal actions, we have positioned Q-function between state and goal. In detail, the input token sequence of GCREinSL for DT and corresponding output tokens are summarized as follows:

$$\begin{aligned} \text{Input:} & \left\langle \dots, s_{t-1}^{(n)}, Q_t^{(n)}, a_t^{(n)} \right\rangle \\ \text{Output:} & \left\langle \hat{Q}_t^{(n)}, \hat{a}_t^{(n)}, \square \right\rangle \end{aligned}$$

$s_{t-1}^{(n)}$  represents a token formed by concatenating  $s_t^{(n)}$  and  $g_t^{(n)}$  (Schaul et al., 2015). When predicting the  $Q_t^{(n)}$ , the model takes the current state  $s_t^{(n)}$  and previous  $K$  timesteps tokens  $\langle s_g, Q, a \rangle_{t-K}^{(n)} = (s_{t-K+1}^{(n)}, Q_{t-K+1}^{(n)}, a_{t-K+1}^{(n)}, \dots, s_{t-1}^{(n)}, Q_{t-1}^{(n)}, a_{t-1}^{(n)})$  into consideration. For the sake of simplicity,  $\text{SG}_{t-K}^{(n)}$  denotes the input  $[(s_g, Q, a)_{t-K}^{(n)}; s_{t-1}^{(n)}]$ . While the action prediction  $\hat{a}_t$  is based on  $(\text{SG}_{t-K}^{(n)}, Q_{t-K}^{(n)}) = [\langle s_g, Q, a \rangle_{t-K}^{(n)}; s_{t-1}^{(n)}, Q_t^{(n)}]$ . The  $\square$  represents this predicted token neither participates in training nor inference so we ignore it. At the timestep  $t$ , different tokens are embedded by different linear layers and fed into the transformers (Vaswani et al., 2017) together. The output Q-function  $\hat{Q}_t^{(n)}$  is processed by a linear layer.

{6}------------------------------------------------

![Figure 3: Overview of GCREinSL for DT. (a) Model Architecture: A sequence of tokens [sg_{t-1}, Q_{t-1}, a_{t-1}, sg_t, Q_t, a_t] is processed by a GCREinSL for DT block to produce predicted tokens [\hat{Q}_{t-1}, \hat{a}_{t-1}, \hat{Q}_t, \hat{a}_t]. (b) Training Loss: A VAE Probability Estimator takes state s_t, action a_t, and goal g_t to produce a probability distribution. Expectile Regression takes Q_t and \hat{Q}_t to produce a loss. The total loss is the sum of the VAE loss and the Expectile Regression loss plus MSE loss for actions. (c) Inference Pipeline: The environment provides state s_t and goal g_t to a GCREinSL for DT block, which predicts the next state Q_t and action a_t. The environment then executes the action a_t to produce the next state s_{t+1} and goal g_{t+1}, which are concatenated to form the next state-goal pair sg_{t+1}. The process repeats until the goal is reached.](1956f44611abd5c3c41049836aa78ad8_img.jpg)

Figure 3: Overview of GCREinSL for DT. (a) Model Architecture: A sequence of tokens [sg\_{t-1}, Q\_{t-1}, a\_{t-1}, sg\_t, Q\_t, a\_t] is processed by a GCREinSL for DT block to produce predicted tokens [\hat{Q}\_{t-1}, \hat{a}\_{t-1}, \hat{Q}\_t, \hat{a}\_t]. (b) Training Loss: A VAE Probability Estimator takes state s\_t, action a\_t, and goal g\_t to produce a probability distribution. Expectile Regression takes Q\_t and \hat{Q}\_t to produce a loss. The total loss is the sum of the VAE loss and the Expectile Regression loss plus MSE loss for actions. (c) Inference Pipeline: The environment provides state s\_t and goal g\_t to a GCREinSL for DT block, which predicts the next state Q\_t and action a\_t. The environment then executes the action a\_t to produce the next state s\_{t+1} and goal g\_{t+1}, which are concatenated to form the next state-goal pair sg\_{t+1}. The process repeats until the goal is reached.

Figure 3: The overview of **GCREinSL** for DT: (a) Model Architecture: The Q-function is the third inputs of **GCREinSL** for DT and the outputs contain Q-value and actions. (b) Train Loss: As a Q-conditioned maximization sequence model, **GCREinSL** for DT not only maximizes the action likelihood but also maximizes Q-function by expectile regression. (c) Inference Pipeline: When inference, **GCREinSL** for DT first 1) gets state and goal from the environment to predict the in-distribution maximum Q-function. Then 2) predicted in-distribution max Q-function is concatenated with state and goal to predict the optimal action. Finally, 3) the environment executes the predicted action to Q-function the next state.

**Training Loss** Since the model predicts two parts,  $\hat{Q}_t$   $a_t$  and  $\hat{a}_t$ , the loss function is composed of Q-function loss and action loss. For the action loss, we adopt the MSE loss function of DT and simultaneously adjust the order of tokens:

$$\mathcal{L}_a = \mathbb{E}_{t,n} \left[ a_t^{(n)} - \pi_\theta \left( \mathbf{SG}_{t-K}^{(n)}, Q_{t-K}^{(n)} \right) \right]^2. \quad (10)$$

The Q-function loss is the expectile regression with the parameter  $m$ :

$$\begin{aligned} \mathcal{L}_Q^m &= \mathbb{E}_{t,n} [m - \mathbb{1}(\Delta Q < 0)] |\Delta Q^2|, \\ \text{with } \Delta Q &= Q_t^{(n)} - \pi_\theta \left( \mathbf{SG}_{t-K}^{(n)} \right). \end{aligned} \quad (11)$$

Two loss functions have the same weight so the total loss is  $\mathcal{L}_a + \mathcal{L}_Q$ .

**Inference Pipeline** For each timestep  $t$ , the action is the last token, which means the predicted action is affected by state from the environment and the Q-function. The Q-function of the trajectories output by the sequence modeling exhibit a positive correlation with the initial conditioned Q-function (Chen et al., 2021; Zheng et al., 2022). That is, within a certain range, higher initial Q-function typically lead to better actions. In classical Q-learning (Mnih et al., 2015), the optimal value function  $Q^*$  can derive the optimal action  $a^*$  given the current state. In the context of sequence modeling, we also assume that the maximum Q-function are required to output the optimal actions. The inference pipeline of the **GCREinSL** is summarized as follows:

$$\xrightarrow{\text{Env}} (sg_0) \xrightarrow{\pi_\theta} Q_0 \xrightarrow{\pi_\theta} a_0 \xrightarrow{\text{Env}} (sg_1) \xrightarrow{\pi_\theta} Q_1 \xrightarrow{\pi_\theta} a_1 \rightarrow \dots \quad (12)$$

Specially, the environment initializes the state-goal pair  $(sg_0)$  (i.e.,  $s_0$  and  $g_0$  are concatenated to form  $sg_0$ ) and then the sequence modeling  $\pi_\theta$  predicts the maximum Q-function  $Q_0$  given current state-goal pair  $(sg_0)$ . Concatenating  $Q_0$  with  $(sg_0)$ ,  $\pi_\theta$  can output the optimal action  $a_0$ . Then the environment transitions to the next state  $s_1$  and the reward  $r_1$ . It should be noted that this reward  $r_1$  will **not** participate in the inference. Repeat the above steps until the trajectory comes to an end. The overall algorithm of **GCREinSL** for DT is shown in Appendix B.1.

#### 4.4.2 GCREinSL FOR RVs

**Architecture** To accommodate the Q-conditioned maximization for RvS (Emmons et al., 2021), which also predicts the maximum Q-function and utilizes it as a condition to guide the generation of optimal actions. Unlike **GCREinSL** for DT, we construct an actor model for predicting actions and a

{7}------------------------------------------------

value model for predicting Q-function. In detail, the input of **GCReinSL** for RvS and corresponding output are summarized as follows:

**Input:**  $s_t, g_t, Q_t(s_t, a_t, g_t)$

**Value Model Output:**  $\hat{Q}_t(s_t, g_t)$

**Actor Model Output:**  $\hat{a}_t \left( s_t, g_t, \hat{Q}_t(s_t, g_t) \right)$

When predicting the  $\hat{Q}_t$ , the value model takes the current state  $s_t$  and desired goal  $g_t$ . For action  $\hat{a}_t^{(n)}$ , We adopt an actor model that incorporates Q-values for inference.

**Training Procedure and Inference Pipeline** Like **GCReinSL** for DT, the total loss function is also composed of Q-function loss and action loss, and the form is the same. At each step of the inference pipeline, the value model outputs the maximum Q-function value for the input state-goal pair, and then the actor model outputs the corresponding action. Note that in this state-goal pair, the state and the goal are treated as distinct elements. In the context of RvS, we also assume that the maximum Q-function are required to output the optimal actions. The training procedure is similar to that of **GCReinSL** for DT, with the key distinction that the prediction of Q-values is generated by a value model. The inference pipeline of the **GCReinSL** is summarized as follows:

$$\xrightarrow{\text{Env}} (s_0, g_0) \xrightarrow{v_\phi} Q_0 \xrightarrow{\pi_\phi} a_0 \xrightarrow{\text{Env}} (s_1, g_1) \xrightarrow{v_\phi} Q_1 \xrightarrow{\pi_\phi} a_1 \rightarrow \dots \quad (13)$$

Specially, the environment initializes the state-goal pair  $(s_0, g_0)$  and then the value model  $v_\phi$  predicts the maximum Q-function  $Q_0$  given current state-goal pair  $(s_0, g_0)$ . Concatenating  $Q_0$  with  $(s_0, g_0)$ ,  $\pi_\phi$  can output the optimal action  $a_0$ . The overall algorithm of **GCReinSL** for RvS is shown in Appendix B.2.

## 5 EXPERIMENTS

To rigorously evaluate the stitching capability of **GCReinSL**, we employ the offline goal-conditioned datasets configuration as outlined in Ghugare et al. (2024). For the evaluation, we follow the methodology outlined by Ghugare et al. (2024), modifying the the **GCReinSL** policy to navigate between previously unseen combinatorial (state, goal) pairs and subsequently measure the success rate. We then add the corresponding goal data augmentation techniques into the OCBC methods for a comparative analysis with our proposed approach. We additionally compared **GCReinSL** with the previous sequence modeling methods on D4RL (Fu et al., 2020) complex offline Antmaze-v2 datasets. Both offline goal-conditioned datasets are characterized by sparse rewards (i.e, reaching the goal results in a reward of 1, otherwise 0) and are designed to test stitching capabilities.

### 5.1 EXPERIMENTAL SETUP

We conducted a series of comparative experiments by implementing the OCBC methods within the same framework, as well as related goal data augmentation approaches. Specifically, we select RvS (Emmons et al., 2021) and DT (Chen et al., 2021), two competitive methods in OCBC, as baseline models for comparison. For goal data augmentation methods, we select Swapped Goal Data Augmentation (SGDA) (Yang et al., 2023) and Temporal Goal Data Augmentation (TGDA) (Ghugare et al., 2024) as advanced methodologies to serve as comparative baselines for our goal data augmentation study. SGDA (Yang et al., 2023) proposes a method that randomly choose augmented goals from different trajectories. TGDA (Ghugare et al., 2024) proposed a another goal data augmentation approach from the perspective of combinatorial optimization. It employs k-means (Lloyd, 1982) to cluster the goal and certain states into a group, and samples goals from later stages of these state trajectories as augmented goals. For related sequence modeling approaches, we select state-of-the-art methods, including Elastic Decision Transformer (EDT) (Wu et al., 2023) and Max-Return Sequence Modeling (Reinformer) (Zhuang et al., 2024), as baselines. Both of these methods, like ours, exhibit stitching property without requiring dynamic programming. Additionally, we compare these sequence modeling approaches to traditional reinforcement learning methods such as CQL and IQL. All experiments are conducted using five random seeds. Detailed implementations and hyperparameter settings are outlined in Appendix C and Appendix D, respectively.

{8}------------------------------------------------

![Figure 4: Performance of the original OCBC, as well as OCBC with corresponding goal data augmentation, compared to our SL method on the Pointmaze datasets. The figure consists of three bar charts for Pointmaze-Umaze, Pointmaze-Medium, and Pointmaze-Large. Each chart compares the Success Rate of four methods: OCBC (blue), SGDA (orange), TGDA (green), and GCReinSL (Ours) (red) across two tasks: DT and RvS. In all cases, GCReinSL (Ours) achieves the highest success rate, followed by TGDA, then OCBC, and finally SGDA.](b93cbfb52e37619e688175a6aad9edd9_img.jpg)

| Dataset          | Method          | DT   | RvS  |
|------------------|-----------------|------|------|
| Pointmaze-Umaze  | OCBC            | 0.00 | 0.00 |
|                  | SGDA            | 0.15 | 0.25 |
|                  | TGDA            | 0.20 | 0.85 |
|                  | GCReinSL (Ours) | 0.50 | 1.00 |
| Pointmaze-Medium | OCBC            | 0.45 | 0.45 |
|                  | SGDA            | 0.20 | 0.15 |
|                  | TGDA            | 0.60 | 0.60 |
|                  | GCReinSL (Ours) | 0.70 | 0.50 |
| Pointmaze-Large  | OCBC            | 0.25 | 0.05 |
|                  | SGDA            | 0.20 | 0.15 |
|                  | TGDA            | 0.25 | 0.15 |
|                  | GCReinSL (Ours) | 0.35 | 0.35 |

Figure 4: Performance of the original OCBC, as well as OCBC with corresponding goal data augmentation, compared to our SL method on the Pointmaze datasets. The figure consists of three bar charts for Pointmaze-Umaze, Pointmaze-Medium, and Pointmaze-Large. Each chart compares the Success Rate of four methods: OCBC (blue), SGDA (orange), TGDA (green), and GCReinSL (Ours) (red) across two tasks: DT and RvS. In all cases, GCReinSL (Ours) achieves the highest success rate, followed by TGDA, then OCBC, and finally SGDA.

Figure 4: Performance of the original OCBC, as well as OCBC with corresponding goal data augmentation, compared to our SL method on the Pointmaze datasets from Ghugare et al. (2024). We use the final score as the report. **GCReinSL** not only improves the performance of DT and RvS in all tasks, but also outperforms exist goal data augmentation methods.

### 5.2 TESTING THE ABILITY OF **GCReinSL** AND COMPARED WITH PREVIOUS GOAL DATA AUGMENTATION METHODS

As shown in Fig. 4, it is evident that DT and RvS are struggle to demonstrate stitching property, particularly in the Pointmaze-Umaze and Pointmaze-Large datasets, where their performance is notably poor. However, when Q-conditioned maximization is incorporated into the OCBC methods, performance improvements were observed across all tasks, albeit to varying degrees. This enhancement is attributed to the fact that **GCReinSL** allows for the sampling of unseen (state, goal) combinations during the training phase, thereby improving the generalization and stitching capability of the models. Our **GCReinSL** consistently outperforms the other data augmentation approaches across all Pointmaze datasets, particularly in the more complex Pointmaze-Medium and Pointmaze-Large datasets. This suggests that our approach enables the selection of more suitable goals, facilitating more effective trajectory stitching.

### 5.3 SCALING TO HIGHER-DIMENSIONAL DATASETS

To evaluate the applicability of our **GCReinSL** to tasks with higher-dimensional input spaces, we implemented it on a robotic control dataset with 111-dimensions (Antmaze (Ghugare et al., 2024)). In Fig. 5, we observe that **GCReinSL** improves the performance of DT and RvS across all Antmaze datasets, with particularly notable improvements on the medium and large datasets.

![Figure 5: Performance on high-dimensional Antmaze datasets. The figure consists of three bar charts for Antmaze-Umaze, Antmaze-Medium, and Antmaze-Large. Each chart compares the Success Rate of four methods: OCBC (blue), SGDA (orange), TGDA (green), and GCReinSL (Ours) (red) across two tasks: DT and RvS. In all cases, GCReinSL (Ours) achieves the highest success rate, followed by TGDA, then OCBC, and finally SGDA.](e8ff6e66c77a8e96203c9f8db8f0986f_img.jpg)

| Dataset        | Method          | DT   | RvS  |
|----------------|-----------------|------|------|
| Antmaze-Umaze  | OCBC            | 0.00 | 0.00 |
|                | SGDA            | 0.00 | 0.05 |
|                | TGDA            | 0.05 | 0.15 |
|                | GCReinSL (Ours) | 0.10 | 0.15 |
| Antmaze-Medium | OCBC            | 0.12 | 0.18 |
|                | SGDA            | 0.05 | 0.12 |
|                | TGDA            | 0.15 | 0.25 |
|                | GCReinSL (Ours) | 0.28 | 0.28 |
| Antmaze-Large  | OCBC            | 0.10 | 0.00 |
|                | SGDA            | 0.05 | 0.00 |
|                | TGDA            | 0.00 | 0.00 |
|                | GCReinSL (Ours) | 0.12 | 0.02 |

Figure 5: Performance on high-dimensional Antmaze datasets. The figure consists of three bar charts for Antmaze-Umaze, Antmaze-Medium, and Antmaze-Large. Each chart compares the Success Rate of four methods: OCBC (blue), SGDA (orange), TGDA (green), and GCReinSL (Ours) (red) across two tasks: DT and RvS. In all cases, GCReinSL (Ours) achieves the highest success rate, followed by TGDA, then OCBC, and finally SGDA.

Figure 5: Performance on high-dimensional Antmaze datasets: **GCReinSL** can still improve the performance of DT and RvS on high-dimensional Antmaze datasets. We also use the final score as the report. However, in some datasets such as Antmaze-Medium, **GCReinSL** is inferior to advanced TGDA method.

### 5.4 COMPARED GCREINSL WITH THE PREVIOUS MAX-RETURN SEQUENCE MODELING METHOD

We also compared our method with relevant sequence modeling approaches that perform stitching property on the standard offline dataset D4RL (Fu et al., 2020), specifically on the Antmaze-v2 datasets, as shown in Table 1. From Table 1, it is evident that in the majority of the AntMaze datasets,

{9}------------------------------------------------

particularly in the complex medium and large AntMaze tasks, the **GCReinSL** approach demonstrates superior performance, significantly closing the gap with TD learning methods such as CQL.

| Antmaze-v2     | RL                |                    | Sequence Modeling |              |                   |                        |
|----------------|-------------------|--------------------|-------------------|--------------|-------------------|------------------------|
|                | CQL               | IQL                | DT                | EDT          | Reinformer        | <b>GCReinSL (ours)</b> |
| umaze          | <b>94.8 ± 0.8</b> | 84.00 ± 4.1        | 64.5 ± 2.1        | 67.8 ± 3.2   | <b>84.4 ± 2.7</b> | 80.1 ± 5.3             |
| umaze-diverse  | 53.8 ± 2.1        | <b>79.5 ± 3.4</b>  | 60.5 ± 2.3        | 58.3 ± 1.9   | 65.8 ± 4.1        | <b>67.2 ± 5.3</b>      |
| medium-play    | <b>80.5 ± 3.4</b> | 78.5 ± 3.8         | 0.8 ± 0.4         | 0.0 ± 0.0    | 13.2 ± 6.1        | <b>49.0 ± 3.5</b>      |
| medium-diverse | 71.0 ± 4.5        | <b>83.5 ± 1.8</b>  | 0.5 ± 0.5         | 0.0 ± 0.0    | 10.6 ± 6.9        | <b>51.7 ± 4.4</b>      |
| large-play     | 34.8 ± 5.9        | <b>53.5 ± 2.5</b>  | 0.0 ± 0.0         | 0.6 ± 0.5    | 0.4 ± 0.5         | <b>28.2 ± 1.8</b>      |
| large-diverse  | 36.3 ± 3.3        | <b>53.0 ± 3.00</b> | 0.0 ± 0.0         | 0.0 ± 0.0    | 0.4 ± 0.5         | <b>30.2 ± 2.4</b>      |
| <i>Total</i>   | <i>371.2</i>      | <i>432.0</i>       | <i>126.3</i>      | <i>126.7</i> | <i>174.8</i>      | <i>306.4</i>           |

**Table 1:** The normalized best score on D4RL (Fu et al., 2020) Antmaze-v2 datasets. The results come from its original Reinformer (Zhuang et al., 2024) paper except **GCReinSL**. The best result is **bold** and the blue result means the best result among sequence modeling.

### 5.5 ABLATION STUDY

In this section, we analyze the impact of the hyperparameter  $L$  in the probability estimator and  $m$  in the Q-function loss. As illustrated in the left panel of Fig. 6, the performance does not exhibit a linear relationship with increasing values of  $L$ . Therefore, we set  $L = 500$  as the default value for the datasets employed in Ghugare et al. (2024). For the D4RL Antmaze-v2 dataset (Fu et al., 2020), we select  $L = 5$ , in line with the methodology outlined by Wu et al. (2022).

As stated in Theorem 4.1, as  $m \rightarrow 1$ , the learned Q-function asymptotically converges to the maximum Q-function within the offline distribution.

Given that a higher in-distribution Q-function corresponds to improved action selection, we can infer that performance will improve as  $m$  approaches 1. The experimental results presented in the right panel of Fig. 6 are consistent with this theoretical prediction.

However, larger values of  $m$  do not consistently lead to more effective training or higher performance; in some cases, they may result in a performance decline. This could be attributed to overfitting to excessively large Q-function values present in the offline dataset.

![Figure 6: Ablation study of different hyperparameter L and m. The left panel shows Success Rate for Pointmaze-Large L Ablation with methods RvS and DT for L=100, 500, 1000. The right panel shows Success Rate for Pointmaze-Medium m Ablation for methods DT and RvS as m varies from 0.5 to 0.999.](64bacf564ff025df294b6d30341c76df_img.jpg)

Figure 6 consists of two plots. The left plot, titled 'Pointmaze-Large L Ablation', shows the Success Rate (x-axis, 0.00 to 0.60) for two methods, RvS and DT, across three values of L (100, 500, 1000). For RvS, the success rate increases with L, while for DT, it decreases. The right plot, titled 'Pointmaze-Medium m Ablation', shows the Success Rate (y-axis, 0.00 to 0.75) as a function of m (x-axis, 0.5 to 0.999) for DT (orange dashed line with circles) and RvS (green solid line with squares). Both methods show an overall increasing trend in success rate as m increases, with RvS generally outperforming DT, especially at higher m values.

Figure 6: Ablation study of different hyperparameter L and m. The left panel shows Success Rate for Pointmaze-Large L Ablation with methods RvS and DT for L=100, 500, 1000. The right panel shows Success Rate for Pointmaze-Medium m Ablation for methods DT and RvS as m varies from 0.5 to 0.999.

**Figure 6:** Ablation study of different hyperparameter  $L$  and  $m$  in Ghugare et al. (2024) datasets. (left): The performance on the Pointmaze-Large dataset when applying different values of  $L$  to the importance sampling estimator. (right): The trend of last results as  $m$  varies on Pointmaze-Medium dataset.

## 6 CONCLUSION

In this work, we propose the paradigm of Q-conditioned maximization supervised learning which considers the RL objective that maximizes Q-function for SL-based methods (OCBC methods). Both theoretical analysis and experiments indicate that our proposed model **GCReinSL** reduces the performance gap between itself and classical RL approaches. However, our approach still exhibits a gap compared to classical RL methods and is sensitive to certain hyperparameters. Future work could focus on developing more robust SL architectures that are better suited for scenarios where classical RL excels, particularly in trajectory stitching. This would provide a more nuanced understanding of the respective strengths and applications of each approach.

 Rest of paper (reference and Appendix) is removed.