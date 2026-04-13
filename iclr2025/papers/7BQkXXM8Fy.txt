

{0}------------------------------------------------

# WHAT MAKES A GOOD DIFFUSION PLANNER FOR DECISION MAKING?

Haofei Lu<sup>1</sup> Dongqi Han<sup>2†</sup> Yifei Shen<sup>2</sup> Dongsheng Li<sup>2</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Microsoft Research Asia

luhf23@mails.tsinghua.edu.cn

{dongqihan,yifeishen,Dongsheng.Li}@microsoft.com

## ABSTRACT

Diffusion models have recently shown significant potential in solving decision-making problems, particularly in generating behavior plans – also known as diffusion planning. While numerous studies have demonstrated the impressive performance of diffusion planning, the mechanisms behind the key components of a good diffusion planner remain unclear and the design choices are highly inconsistent in existing studies. In this work, we address this issue through systematic empirical experiments on diffusion planning in an offline reinforcement learning (RL) setting, providing practical insights into the essential components of diffusion planning. We trained and evaluated over 6,000 diffusion models, identifying the critical components such as guided sampling, network architecture, action generation and planning strategy. We revealed that some design choices opposite to the common practice in previous work in diffusion planning actually lead to better performance, e.g., unconditional sampling with selection can be better than guided sampling and Transformer outperforms U-Net as denoising network. Based on these insights, we suggest a simple yet strong diffusion planning baseline that achieves state-of-the-art results on standard offline RL benchmarks.

## 1 INTRODUCTION

Decision making by learning from offline data has been a fundamental approach in robotics and artificial intelligence (Bellman, 1957). It enables agents to acquire complex behaviors by observing and mimicking expert demonstrations, circumventing the need for explicit programming or exhaustive exploration. However, this paradigm faces significant challenges, particularly when dealing with long-horizon planning and high-dimensional action spaces. The complexity of modeling sequential dependencies and capturing the intricacies of action distributions makes it difficult to scale traditional methods (Deisenroth & Rasmussen, 2011) to more complex tasks (Parmas et al., 2018).

Recently, diffusion models have achieved remarkable success in image and video generation, demonstrating their ability to handle complex distribution and long-range dependencies (Ho et al., 2020; Dhariwal & Nichol, 2021). Inspired by these works, several recent studies have applied diffusion models to planning sequential decisions, especially with continuous state and action spaces such as robotic manipulation tasks (Janner et al., 2022; Ajay et al., 2022; Lu et al., 2023; Li et al., 2023). The diffusion models are used to approximate the sequence of states and actions from current time step to future – and by exploiting the diffusion models’ conditional generation capacity such as diffusion guidance (Ho et al., 2020; Ho & Salimans, 2021), the model can make plans (i.e. state trajectory) with desired properties such as reward maximization (i.e. offline reinforcement learning (Levine et al., 2020)).

Despite achieving impressive performance across a diverse array of tasks, there has been limited exploration into the fundamental components and mechanisms that constitute an effective diffusion planning model for decision making. Previous research exhibits a lack of consistency and coherence in design choices. It remains uncertain whether sub-optimal design choices might hinder the full

†This work was done during the internship of Haofei Lu (luhf23@mails.tsinghua.edu.cn) at Microsoft Research Asia. Correspondence to: Dongqi Han <dongqihan@microsoft.com>

{1}------------------------------------------------

potential of diffusion models within decision-making domains. Specifically, existing approaches have not adequately addressed essential facets such as the choice of diffusion guidance algorithm, network architecture, and whether the plan should contain states or state-action pairs. This naturally raises the following fundamental question:

*What makes a good diffusion planner for decision making, especially offline RL?*

We seek to answer the question by conducting a comprehensive empirical investigation into key design choices in diffusion models for decision-making, in particular for state-based robotics tasks. Our work contributes to the field of decision making and diffusion models in several aspects.

- **Comprehensive experiments:** We conducted an extensive empirical study to explore what constitutes an effective diffusion planner. By training and evaluating over 6,000 models, we analyzed key components critical to decision making in diffusion planning, including guided sampling algorithms, network architectures, action generation methods, and planning strategies.
- **Insights and tips:** We ran detailed experiments and data analysis to understand the role of each key component in constituting a good diffusion planner. In particular, we discovered that certain design choices, contrary to common practice in diffusion planning actually lead to better performance. Our work offers intuitive explanations and practical tips about the choices and provides insights about the strengths and limitations of diffusion planning.
- **A simple yet strong baseline:** Building on the insights from our study, we suggest a simple yet highly competitive baseline, named *Diffusion Veteran* (DV). This model achieves state-of-the-art performance in planning tasks in standard offline RL benchmarks.

## 2 BACKGROUND AND RELATED WORK

**Offline Reinforcement Learning** (Fujimoto et al., 2019; Levine et al., 2020; Fu et al., 2020) is a subfield of reinforcement learning (RL) where the agent learns from a fixed dataset of past experiences. This dataset typically consists of state-action-reward-next-state tuples, which encapsulate the agent’s interactions with the environment. The challenge in offline RL is for the agent to derive an effective policy from this static dataset without further exploration or interaction with the environment. Two major challenges arise in this context. First, the state and action spaces may be high-dimensional and involve long-range dependencies, making it difficult to model effectively (Levine et al., 2020). Second, the learned policy must be optimal, even though the behavior policy that generated the offline data may be sub-optimal or different from the desired policy (Fujimoto et al., 2019).

Recently, diffusion models have emerged as a powerful framework for tasks such as image and video generation due to their ability to model complex distributions (Croitoru et al., 2023), which could mitigate the first problem. Moreover, diffusion guidance techniques (Ho et al., 2020; Ho & Salimans, 2021) allow the model to generate samples that adhere to the desired properties. The second challenge in offline RL, learning an optimal policy, can be addressed by diffusion guidance techniques to produce behavior that maximizes rewards. Building on this insight, a growing body of research has explored the use of diffusion models to generate behavior trajectories, denoted as  $\tau$ .

**Diffusion planning** (Ajay et al., 2022; Janner et al., 2022; Liang et al., 2023; Dai et al., 2023; Yang et al., 2023; Li et al., 2023; Yang et al., 2023; Chen et al., 2024; Dong et al., 2024c) considers that at the time step  $t$ , a trajectory  $\tau$  consists of the current and subsequent  $H$  steps of state-action pairs or states:

$$\tau = \begin{bmatrix} s_t & s_{t+1} & \dots & s_{t+H-1} \\ a_t & a_{t+1} & \dots & a_{t+H-1} \end{bmatrix}, \text{ or } \tau = [s_t \ s_{t+1} \ \dots \ s_{t+H-1}]. \quad (2.1)$$

There is a guidance function to model the reward, such as the immediate reward  $r_t$  or the state value function  $v(s_t) = \mathbb{E} \left[ \sum_{h=0}^{\text{end}} \gamma^h r_{t+h} \right]$ , where  $\gamma$  is the discount factor (Sutton & Barto, 1998). In classifier guidance (CG) (Ho et al., 2020), a guidance network is learned simultaneously with the diffusion model, whose input is the generated trajectory and the output is accumulated rewards or value function. The gradient of the guidance network is used in the generation process of diffusion model to maximize the rewards. Examples of diffusion planning with CG are (Janner et al., 2022; Liang et al., 2023; Zhang et al., 2022). In classifier-free guidance (CFG) (Ho & Salimans, 2021), it

{2}------------------------------------------------

![Figure 1: Diffusion planning framework for decision making. (a) shows the generation of a sequence plan using the denoising process of a diffusion model. (b) shows the important components and candidates in the framework.](1b7d539e02a202c2cf2d97698b911447_img.jpg)

Figure 1 illustrates the Diffusion planning framework for decision making. Part (a) shows the generation of a sequence plan using the denoising process of a diffusion model. It includes a Gaussian Noise input with shape  $[H, N^s]$ , where  $H$  is the number of planning steps and  $N^s$  is the state dimension. A Robot trajectory dataset (demonstrations) is used for training a Denoising Network. The network generates a sequence of states  $s_1, s_2, s_3$  over time steps  $t, t+1, \dots, t+H-1$ . A 3-joints robot arm is used as an illustrative example. The output is a Trajectory Plan with shape  $[H, N^s]$ . Part (b) shows the important components and candidates in the framework. The framework consists of a Planning Strategy, a Denoising Network Backbone, and Action Generation. The Planning Strategy includes Dense-step planning and Jump-step planning. The Denoising Network Backbone includes a Transformer and a U-Net. The Action Generation module includes two diffusion models: one that learns the joint distribution of state and action, and another that learns and uses inverse dynamics. The framework also includes Guided Sampling Algorithms: Classifier guidance, Classifier-free guidance, Monte Carlo Sampling with Selection, and Unconditional. A star indicates the preferred choice in experiments.

Figure 1: Diffusion planning framework for decision making. (a) shows the generation of a sequence plan using the denoising process of a diffusion model. (b) shows the important components and candidates in the framework.

Figure 1: **Diffusion planning framework for decision making.** (a) The generation of a sequence plan using the denoising process of a diffusion model. A 3-joints robot arm is used as an illustrative example. (b) Important components and candidates in the framework. Each color corresponds to one component in the framework. A star indicates the preferred choice in experiments.

takes the desired reward or value function as an additional argument feed into the diffusion process. Instances are (Ajay et al., 2022; Li et al., 2023; Yang et al., 2023). However, despite some literature reviews such as Zhu et al. (2023), the field lack a systematical study to elucidate the design space of diffusion planning in offline RL with substantial experimental results.

**Diffusion policy** (Pearce et al., 2023; Wang et al., 2023b; Hansen-Estruch et al., 2023; Chen et al., 2023) is another kind of popular usage of diffusion model in decision making. The trajectory only includes  $\tau = a_t$ , without lookahead planning. The model is trained by combining the loss of imitation learning and model-free RL as in classic offline RL methods (Kumar et al., 2020; Fujimoto & Gu, 2021). Diffusion policy methods hope to improve the performance of by leveraging the capacity of diffusion model to model complex distribution of actions (policy function). A recent study (Dong et al., 2024b) investigated the design space of diffusion policy, proposed that diffusion policies such as DQL (Wang et al., 2023b) can be a computationally efficient and powerful candidate for decision-making tasks.

## 3 STUDY DESIGN

### 3.1 KEY COMPONENTS AND MECHANISMS OF DIFFUSION PLANNER

Recent pioneering work in diffusion planning (Janner et al., 2022; Ajay et al., 2022; Chen et al., 2024) has demonstrated the potential of this approach in offline RL. However, the design choices in these studies vary significantly, and it remains unclear whether there is an optimal configuration for different domains. Our aim is to conduct a systematic analysis supported by comprehensive experimental results. To achieve this, we begin by listing key design components (excluding common deep learning hyperparameters such as learning rates) that have varied in previous studies. See Fig. 1(b) for an overview.

**Guided sampling algorithms:** Classifier guidance (CG) (Ho et al., 2020), Classifier-free guidance (CFG) (Ho & Salimans, 2021), Monte Carlo sampling with selection (sample  $N$  unconditional trajectories and select the best, the criteria of which is given by a critic function learned simultaneously with diffusion model). Most previous diffusion planners used CG (Janner et al., 2022; Wang et al., 2023a; Chen et al., 2024) or CFG (Ajay et al., 2022; Li et al., 2023; Yang et al., 2023) for offline RL.

**Denoising network backbone:** U-Net (Ronneberger et al., 2015); Transformer (Vaswani et al., 2017). U-Net was used in most previous diffusion planners for state-based offline RL (Janner et al., 2022; Ajay et al., 2022; Wang et al., 2023a; Li et al., 2023; Chen et al., 2024)).

{3}------------------------------------------------

**Action Generation:** Learn joint distribution of state and action and directly execute the generated action at the current step (used in, e.g. Janner et al. (2022); Liang et al. (2023)); Learn and use inverse dynamics to compute action from state plan (used in e.g., Ajay et al. (2022); Wang et al. (2023a)).

**Planning strategy:** Dense-step planning means the planned trajectory  $\tau$  (Eq. 2.1) corresponds to contiguous  $H$  steps in the environment (this is a conventional setting in diffusion planning (Janner et al., 2022; Ajay et al., 2022; Lu et al., 2023)); Jump-step planning models  $H \times m$  environment steps, where  $m \in \mathbb{N}^+$  is the planning stride; Hierarchical planning (studied by Li et al. (2023); Chen et al. (2024)).

Details of the implementation are deferred to Appendix A and B.

### 3.2 EXPERIMENT PROCEDURE

Given the multitude of components involved, it is challenging to draw scientific conclusions directly from the collective results. Therefore, we structured our study using the following procedure:

- (1) Conduct a comprehensive search on the key components (Sect. 3.1) by combining grid search and manual tuning to obtain the best results.
- (2) Evaluate the effect of each component using the control variable method; that is, modify only one component of the best model at a time and compare it with the original.
- (3) After identifying which components are important and understanding how they affect performance, perform a deeper analysis to derive useful insights.

### 3.3 BENCHMARK

We conducted experiments on the D4RL dataset (Fu et al., 2020), one of the most widely used benchmarks for offline RL and imitation learning. The dataset covers a variety of task domains, including maze navigation, robot locomotion, robot arm manipulation, and vehicle driving, among others. For our experiments, we selected three sets of behavior planning tasks that were most commonly studied in prior works in offline RL and diffusion planning (Janner et al., 2021; Ajay et al., 2022; Janner et al., 2022; Liang et al., 2023; Li et al., 2023; Lu et al., 2023; Chen et al., 2024). These tasks (Fig. 2) encompass both planning and control challenges, providing a comprehensive evaluation in various problem settings. The performance metric considered in this work is the standard RL objective: the average total rewards in an online testing episode.

![Figure 2: Rendering of the benchmarking tasks. The image shows three rows of task examples. The first row shows 'Frank Kitchen' (a 3D simulation of a kitchen with a robot arm) and 'Maze2D' (three 2D grid worlds with different maze patterns). The second row shows 'AntMaze' (four 2D grid worlds with different maze patterns). Above each row are labels for the dimension of the action space (dim(A)) and the state space (dim(S)). For Frank Kitchen, dim(A)=9 and dim(S)=60. For Maze2D, dim(A)=2 and dim(S)=4. For AntMaze, dim(A)=8 and dim(S)=29.](4b9457ad2400572dbf0c0c9f7c825643_img.jpg)

Figure 2: Rendering of the benchmarking tasks. The image shows three rows of task examples. The first row shows 'Frank Kitchen' (a 3D simulation of a kitchen with a robot arm) and 'Maze2D' (three 2D grid worlds with different maze patterns). The second row shows 'AntMaze' (four 2D grid worlds with different maze patterns). Above each row are labels for the dimension of the action space (dim(A)) and the state space (dim(S)). For Frank Kitchen, dim(A)=9 and dim(S)=60. For Maze2D, dim(A)=2 and dim(S)=4. For AntMaze, dim(A)=8 and dim(S)=29.

Figure 2: **Rendering of the benchmarking tasks considered in this study, where  $\dim(\mathcal{S})$  and  $\dim(\mathcal{A})$  denote the dimension of the state and action spaces.**

**Maze2D** involve navigating a 2D maze, requiring the agent to find an optimal path to a goal. These tasks are used to test planning capabilities in environments where spatial reasoning is critical.

**AntMaze** presents a navigation challenge with a simulated ant robot. The agent controls a multi-legged robot to navigate through a 2D maze, combining both locomotion and planning.

{4}------------------------------------------------

| Category           | Env<br>Dataset | Kitchen |         |             |        |        |                   |        | Maze2D      |       |       |       |              |
|--------------------|----------------|---------|---------|-------------|--------|--------|-------------------|--------|-------------|-------|-------|-------|--------------|
|                    |                | Mixed   | Partial | avg.        | L-div. | L-play | Antmaze<br>M-div. | M-play | avg.        | L     | M     | Umaze | avg.         |
| Non-diffusion      | BC             | 47.5    | 33.8    | 40.7        | 0.0    | 0.0    | 0.0               | 0.0    | 0.0         | 5     | 30.3  | 3.8   | 13.0         |
|                    | BCQ            | 8.1     | 18.9    | 13.5        | 2.2    | 6.7    | 0.0               | 0.0    | 2.2         | 6.2   | 8.3   | 12.8  | 9.1          |
|                    | CQL            | 51.0    | 49.8    | 50.4        | 61.2   | 53.7   | 15.8              | 14.9   | 36.4        | 12.5  | 5.0   | 5.7   | 7.7          |
| Diffusion Policies | IQL            | 51.0    | 46.3    | 48.7        | 47.5   | 39.6   | 70.0              | 71.2   | 57.1        | 58.6  | 34.9  | 47.4  | 47.0         |
|                    | SfBC           | 45.4    | 47.9    | 46.7        | 45.5   | 59.3   | 82.0              | 81.3   | 67.0        | 74.4  | 73.8  | 73.9  | 74.0         |
|                    | DQL            | 62.6    | 60.5    | 61.6        | 56.6   | 46.4   | 78.6              | 76.6   | 64.6        | –     | –     | –     | –            |
|                    | DQL*           | 55.1    | 65.5    | 60.3        | 70.6   | 81.3   | 82.6              | 87.3   | 80.5        | 186.8 | 152.0 | 140.6 | 159.8        |
|                    | IDQL           | 66.5    | 66.7    | 66.6        | 67.9   | 63.5   | 84.8              | 84.5   | 75.2        | 90.1  | 89.5  | 57.9  | 79.2         |
|                    | IDQL*          | 66.5    | 66.7    | 66.6        | 40.0   | 48.7   | 83.3              | 67.3   | 59.8        | –     | –     | –     | –            |
| Diffusion Planners | CEP            |         |         | 64.8        | 66.6   | 66.6   | 83.8              | 83.6   | 74.7        | –     | –     | –     | –            |
|                    | Diffuser       | 52.5    | 55.7    | 54.1        | 27.3   | 17.3   | 2.0               | 6.7    | 13.3        | 123   | 121.5 | 113.9 | 119.5        |
|                    | AdpDfSr        | 51.8    | 55.5    | 53.7        | 8.7    | 5.3    | 6.0               | 12.0   | 8.0         | 167.9 | 129.9 | 135.1 | 144.3        |
|                    | DD             | 75.0    | 56.5    | 65.8        | 0.0    | 0.0    | 4.0               | 8.0    | 3.0         | –     | –     | –     | –            |
|                    | HD             | 71.7    | 73.3    | 72.5        | 83.6   | –      | 88.7              | –      | –           | 128.4 | 135.6 | 155.8 | 139.9        |
|                    | DV (Ours)      | 73.6    | 94.0    | <b>83.8</b> | 80.0   | 76.4   | 87.4              | 89.0   | <b>83.2</b> | 203.6 | 150.7 | 136.6 | <b>163.6</b> |

Table 1: **Normalized performance of various offline-RL methods.** Our results (DV) are averaged over 500 episode seeds. The results of other methods are obtained from literature. We omit the variance over seeds for simplicity; however, it can be found in the detailed tables in Appendix D. The best average performance on each task set are marked in bold fonts. BC: vanilla imitation learning, BCQ: Fujimoto et al. (2019), CQL: Kumar et al. (2020), IQL: Kostrikov et al. (2021), SfBC: Chen et al. (2023), DQL: Wang et al. (2023b), IDQL: Hansen-Estruch et al. (2023), DQL\* and IDQL\*: replicated by Dong et al. (2024b), CEP: Lu et al. (2023), Diffuser: Jenner et al. (2022), AdpDfSr: Liang et al. (2023), DD: Ajay et al. (2022), HD: Chen et al. (2024).

**Franka Kitchen** simulates a robot arm performing a variety of manipulation tasks in a kitchen environment to achieve task goals across multiple stages.

## 4 EXPERIMENTAL RESULTS

We trained and evaluated over 6,000 diffusion models by sweeping the key components discussed in Sect. 3.1 and other hyper-parameters (See Appendix B for details).

By summarizing the results from the experiments, we identified one kind of diffusion planning framework, called the Diffusion Veteran (DV). The pseudocode of DV can be found in Algorithm 1. As shown in Table 1, DV outperforms all previous diffusion planning and diffusion policy methods. We hope DV will serve as a simple yet strong baseline for future research in diffusion planning.

#### Algorithm 1: Diffusion Veteran (DV) Simplified Pseudocode

**Input:** Planning horizon  $H$ , Dataset  $\mathcal{D}$ , Discount factor  $\gamma$ , Candidate num  $N$ , Planning stride  $M$ .

**Initialize:** Diffusion Transformer Planner  $\epsilon_\theta$ , Diffusion Inverse dynamics  $\epsilon_\omega$ , Critic  $V_\phi$

```

1 Calculate accumulated discounted returns  $R_t = \sum_{h=0}^{\text{end}} \gamma^h r_{t+h}$  for every step  $t$ .
2 Function TRAINING:
3   Sample  $s_{t,t+M}, \dots, s_{t+(H-1)M}, a_{t,t+M}, \dots, a_{t+(H-1)M}, R_t$  from  $\mathcal{D}$ 
4   Train planner  $\epsilon_\theta$  using  $s_t$  as condition and  $s_{t,t+M}, \dots, s_{t+(H-1)M}$  as target output
5   Train Inverse dynamics  $\epsilon_\omega$  using  $s_t, s_{t+M}$  as input,  $a_t$  as target output
6   Train critic  $V_\phi$  using  $s_{t,t+M}, \dots, s_{t+(H-1)M}$  as input,  $R_t$  as target output
7 end
8 Function EXECUTION( $s$ ):
9   Randomly generate  $N$  plans using  $\epsilon_\theta$ , while fixing the first state as  $s$  during sampling
10  Select the best plan using critic  $V_\phi$ 
11  Use the inverse dynamics  $\epsilon_\omega$  to generate action using  $s$  and the next state in the best plan
12 end

```

With DV in place, we can then analyze the impact of each component in diffusion planning by looking into how each component influences its performance. Each of the following sub-sections will focus on one component that we have found to be crucial. In the end of this section, we will conclude our findings into practical tips.

{5}------------------------------------------------

### 4.1 ACTION GENERATION

![Figure 3: Comparison of performance between two action generation strategies. The figure consists of two bar charts. The left chart shows performance for Maze2D (L, M, U) and Kitchen (M, P) environments. The right chart shows performance for Antmaze (L-D, L-P, M-D, M-P) environments. Both charts compare 'Separate' (blue) and 'Joint' (orange) strategies. A vertical dashed line at 50% indicates on-par performance. The dimension of the action space, dim(A), is 2 for Maze2D and 9 for Kitchen, and 8 for Antmaze.](55d2bfe1c3d04e86df8d7a104d802172_img.jpg)

Figure 3 data (approximate values):

| Environment | $\dim(\mathcal{A})$ | Separate | Joint |
|-------------|---------------------|----------|-------|
| Maze2D-L    | 2                   | 203.6    | 202.9 |
| Maze2D-M    | 2                   | 150.7    | 156.0 |
| Maze2D-U    | 2                   | 136.6    | 146.3 |
| Kitchen-M   | 9                   | 73.6     | 62.9  |
| Kitchen-P   | 9                   | 94.0     | 53.7  |
| Antmaze-L-D | 8                   | 80.0     | 49.9  |
| Antmaze-L-P | 8                   | 76.4     | 56.7  |
| Antmaze-M-D | 8                   | 87.4     | 36.3  |
| Antmaze-M-P | 8                   | 89.0     | 34.0  |

Figure 3: Comparison of performance between two action generation strategies. The figure consists of two bar charts. The left chart shows performance for Maze2D (L, M, U) and Kitchen (M, P) environments. The right chart shows performance for Antmaze (L-D, L-P, M-D, M-P) environments. Both charts compare 'Separate' (blue) and 'Joint' (orange) strategies. A vertical dashed line at 50% indicates on-par performance. The dimension of the action space, dim(A), is 2 for Maze2D and 9 for Kitchen, and 8 for Antmaze.

Figure 3: **Comparison of performance between two action generation strategies.** "Separate" learns and uses inverse dynamics to compute action from state plan. "Joint" means learning joint distribution of state and action and directly executing the generated action at the current step (see "action generation" in Fig. 1(b)). A straightforward conclusion drawn from the results is that "Separate" is better than "Joint" when tackling higher-dimensional action spaces. The vertical dashed line indicates on-par performance.

The choice of action generation design (Sect. 3.1) remains a subject of ongoing debate within the field. On one side, the pioneering diffusion planner, Diffuser, along with subsequent studies (Janner et al., 2022; Liang et al., 2023; Chen et al., 2024), employs a diffusion model to generate the joint distribution of action and state trajectories ("joint"). In contrast, studies by Ajay et al. (2022); Wang et al. (2023a); Du et al. (2024) have adopted inverse dynamics to generate actions based on planned states ("separate").

Our experimental findings favor the latter approach: Although both strategies perform comparably in simpler environments such as Maze2D, which lacks robotic control elements, the "separate" approach significantly outperforms the "joint" strategy in more complex settings like Kitchen and AntMaze, which feature robotic control and higher-dimensional action spaces.

This observed disparity may be attributed to the additional complexity introduced when modeling the joint distribution of sequential states and actions, compared to modeling only the states. This complexity becomes particularly pronounced in environments where state transitions involve more complex actions due to higher-dimensional action spaces.

We tested both diffusion models and vanilla MLP as the inverse dynamics, and found similar performance between them. We adhered to diffusion inverse dynamics (Appendix B.1).

### 4.2 PLANNING STRATEGY

![Figure 4: Performance change of DV over planning stride. The figure contains three line graphs for Kitchen, Antmaze, and Maze2D environments. Each graph plots 'Value' (y-axis) against 'Planner Stride' (x-axis). Four lines are shown: Antmaze-L-D (blue), Antmaze-L-P (orange), Antmaze-M-D (green), and Antmaze-M-P (red). A dashed line with circles represents the 'Average'. A red star on the x-axis indicates the choice of DV (Stride=1). In all environments, the performance generally decreases as the planner stride increases, with the star indicating the optimal stride.](92c6488dfdf125bc1b7bb6a235394652_img.jpg)

Figure 4: Performance change of DV over planning stride. The figure contains three line graphs for Kitchen, Antmaze, and Maze2D environments. Each graph plots 'Value' (y-axis) against 'Planner Stride' (x-axis). Four lines are shown: Antmaze-L-D (blue), Antmaze-L-P (orange), Antmaze-M-D (green), and Antmaze-M-P (red). A dashed line with circles represents the 'Average'. A red star on the x-axis indicates the choice of DV (Stride=1). In all environments, the performance generally decreases as the planner stride increases, with the star indicating the optimal stride.

Figure 4: **Performance change of DV over planning stride.** It reduces to dense-step planning when Stride=1. The star indicates the choice of DV.

One crucial result we found is that jump-step planning (Sect. 3.1) is beneficial in almost all cases, despite the fact that most previous work used dense-step planning. This is observed in DV (Fig. 4) and generally in diffusion planners (see Appendix D for extensive results).

An obvious benefit from jump-step planning is that with the same planning steps, the model can look ahead farther. This may be crucial for planning tasks that require long-term credit assignment. The choice of stride should be related to the actual clock-time interval between two environment

{6}------------------------------------------------

steps. Nonetheless, we suggest to try jump-steps and sweep the stride. This observed phenomenon also implies that the diffusion planner should play the role of planning at a more abstract level or with a longer timescale. Interestingly, this is consistent with the neuroscientific fact that the intrinsic timescale of the prefrontal cortex (higher-level planning) is longer than that of the motor cortex (low-level control) (Murray et al., 2014; Runyan et al., 2017; Wang et al., 2018). A recent study (Chen et al., 2024) demonstrated impressive planning performance (Table 1, HD) using multi-timescale diffusion planning. Exploring the hierarchical paradigm of diffusion planning could be an interesting future direction.

### 4.3 DENOISING NETWORK BACKBONE

![Figure 5: Using Transformer as the backbone of denoising network. (a) Performance comparison between Transformer and U-Net across Antmaze, Kitchen, and Maze2d environments. (b) Visualization of attention weights for the first layer in the Transformer network during the denoising process.](a6a8016b231533e7f34b550f4676afc6_img.jpg)

Figure 5(a) shows bar charts of Expected Return (Norm) for Transformer (blue) and U-Net (orange) across three environments: Antmaze, Kitchen, and Maze2d. In Antmaze, Transformer outperforms U-Net in all sub-tasks (Antmaze-L, Antmaze-M, Antmaze-H) and the Average. In Kitchen, Transformer outperforms U-Net in Kitchen-M and Kitchen-P, but U-Net performs better in Kitchen-L. In Maze2d, Transformer outperforms U-Net in Maze2d-L and Maze2d-M, but U-Net performs better in Maze2d-U and the Average.

| Environment | Sub-task  | Transformer | U-Net |
|-------------|-----------|-------------|-------|
| Antmaze     | Antmaze-L | 80.6        | 76.7  |
|             | Antmaze-M | 76.4        | 80.8  |
|             | Antmaze-H | 81.4        | 85.8  |
|             | Average   | 89.0        | 81.0  |
| Kitchen     | Kitchen-M | 73.6        | 35.2  |
|             | Kitchen-L | 94.0        | 28.8  |
|             | Kitchen-P | 83.8        | 32.0  |
|             | Average   | 83.8        | 32.0  |
| Maze2d      | Maze2d-L  | 203.6       | 172.1 |
|             | Maze2d-M  | 150.3       | 145.9 |
|             | Maze2d-U  | 136.7       | 126.5 |
|             | Average   | 163.9       | 155.2 |

Figure 5(b) shows heatmaps of attention weights for the first layer in the Transformer network. It compares attention weights for 'Kitchen-mixed-V0' and 'Kitchen-partial-V0' environments at 'Planning stride = 1' and 'Planning stride = 4'. The heatmaps show that attention is focused on specific steps in the trajectory, with a characteristic attention length of approximately 25 steps.

Figure 5: Using Transformer as the backbone of denoising network. (a) Performance comparison between Transformer and U-Net across Antmaze, Kitchen, and Maze2d environments. (b) Visualization of attention weights for the first layer in the Transformer network during the denoising process.

Figure 5: Using Transformer as the backbone of denoising network. (a) Performance comparison between Transformer and U-Net. The Transformer outperforms U-Net in 8 out of 9 sub-tasks and in all 3 main tasks. The amount of parameters in U-Net is comparable to that in Transformers. Note that the error bars in Kitchen are too small to visualize (See Table 10 for numerical results). (b) Visualization of attention weights of the first layer in the Transformer network during the denoising process. More plots can be found in Appendix D.

Most diffusion planners on the D4RL dataset use 1-D U-Net for the denoising network. It is natural to question whether attention is all you need (Vaswani et al., 2017) for diffusion planning. Thus, we examined the benefit of replacing U-Net with the Transformer architecture as the backbone of the denoising model (Sect. 3.1) (see Appendix B for details about network structures). The experimental results clearly support the utilization of Transformer (Fig. 5(a)) in diffusion planning, consistent with the latest trend in image and video generation (Peebles & Xie, 2023; OpenAI, 2024).

We conducted a case study by looking into the attention weights of the trained Transformer in the Kitchen environment (Fig. 5(b)), which reflect the temporal credit assignment (i.e., to how many steps later should be paid attention in the planning sequence). First, we see that the model pays more attention to the long-range element in the trajectory compared to the short-range ones. It suggests that the long-term dependency is crucial in this task, which breaks the local inductive bias of convolutional neural networks such as U-Net. Second, an interesting finding is that the characteristic attention length is consistent even with different planning stride (Sect. 4.2):  $6 \text{ (attention step)} \times 4 \text{ (stride)} \approx 25 \text{ (attention step)} \times 1 \text{ (stride)}$ , as depicted in Fig. 5(b). It suggests that the Transformer finds the invariant correlations across the stride, contributing to the generalization performance.

More generally, we found long-term attention existing in the Transformer in the other tasks as well, although the attention patterns vary across different tasks. The attention patterns typically feature slashes, which attend to a fixed number of steps prior, and vertical lines, which attend to key steps. We have included the attention weights visualization in Appendix D. In-depth study will be needed to fully understand the role of long-term dependency and why Transformer is observed to outperform UNet in the future.

### 4.4 IMPACT OF NETWORK SIZE

Since the experimental results are in favor of Transformer, one may wonder whether a "scaling law" (Kaplan et al., 2020) holds, in particular, whether performance scales up with model depth (Ye et al.,

{7}------------------------------------------------

![Figure 6: Performance change over depth of the Transformer network as diffusion planner. The figure contains three line plots for Kitchen, Antmaze, and Maze2D environments. Each plot shows Expected Return (Norm) on the y-axis versus Model Depth (2, 4, 6, 8) on the x-axis. A red star at depth 2 indicates the choice of DV. In Kitchen, performance peaks at depth 4 and then slightly declines. In Antmaze, performance generally increases with depth, with Antmaze-M-P reaching the highest return. In Maze2D, performance is relatively stable across depths, with Maze2D-L reaching the highest return.](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

Figure 6: Performance change over depth of the Transformer network as diffusion planner. The figure contains three line plots for Kitchen, Antmaze, and Maze2D environments. Each plot shows Expected Return (Norm) on the y-axis versus Model Depth (2, 4, 6, 8) on the x-axis. A red star at depth 2 indicates the choice of DV. In Kitchen, performance peaks at depth 4 and then slightly declines. In Antmaze, performance generally increases with depth, with Antmaze-M-P reaching the highest return. In Maze2D, performance is relatively stable across depths, with Maze2D-L reaching the highest return.

Figure 6: **Performance change over depth of the Transformer network as diffusion planner.** The star indicates the choice of DV.

2024). The results presented in Fig. 6 pass two clear messages: First, 1-layer Transformer is not enough, except for the simplest sub-task (Maze2D-U). Second, a deeper model is not always better. This may be due to an intrinsic difference between decision making and natural language processing and limitations of dataset size and quality, which requires further study to systematically address.

### 4.5 GUIDED SAMPLING ALGORITHMS

![Figure 7: Analysis of guided sampling algorithm. (a) Radar chart comparing performance of CFG, CG, None, and MCSS algorithms across seven environments: antmaze-L-d, kitchen-P, kitchen-M, maze2d-U, maze2d-M, maze2d-L, and antmaze-L-p. MCSS (red) generally shows the highest performance. (b) Histograms of value distribution for kitchen-mixed-v0, antmaze-large-diverse-v2, and maze2d-large-v1. The x-axis represents normalized return from -1.00 to 1.00, and the y-axis represents proportion. The distributions show varying degrees of concentration at high values, with maze2d-large-v1 showing a strong peak near 1.00.](7ff005f9556dc6518981bb92091d36ab_img.jpg)

Figure 7: Analysis of guided sampling algorithm. (a) Radar chart comparing performance of CFG, CG, None, and MCSS algorithms across seven environments: antmaze-L-d, kitchen-P, kitchen-M, maze2d-U, maze2d-M, maze2d-L, and antmaze-L-p. MCSS (red) generally shows the highest performance. (b) Histograms of value distribution for kitchen-mixed-v0, antmaze-large-diverse-v2, and maze2d-large-v1. The x-axis represents normalized return from -1.00 to 1.00, and the y-axis represents proportion. The distributions show varying degrees of concentration at high values, with maze2d-large-v1 showing a strong peak near 1.00.

Figure 7: **Analysis of guided sampling algorithm.** (a) Performance comparison among different guided sampling algorithms for reward maximization. (b) Histogram of the value (accumulated discounted return in the future  $\sum_{t=0}^{\text{end}} \gamma^h r_{t+h}$ , normalized to  $[-1, 1]$ ) of the data points in each environment. For AntMaze, the failed trajectories are omitted since their values are all 0.

Another inconsistent design in previous work lies in the choice of guided sampling algorithm (Sect. 3.1), which enables the diffusion planner to generate plans that perform better than the average level of the dataset. Fig. 7(a) visualizes the corresponding empirical results (normalized) in our model. We can draw several conclusions from the results.

First, classifier guidance (CG) is comparable with classifier-free guidance (CFG), despite the fact that CFG is generally considered better than CG in image synthesis (Ho & Salimans, 2021). A potential reason is that the target value of CFG may need to be adjusted over time since the total rewards an agent can obtain in the future may vary depending on the task stage, but we can only use a fixed target value for CFG since there is no trivial solution.

Also, we observed that non-guidance can be better than guidance – Monte Carlo sampling with selection (MCSS) performs overall the best, except for Franka Kitchen where MCSS lags slightly behind CFG. This is an important finding since existing diffusion planners usually used CG or CFG (Chen et al., 2023; Wang et al., 2023b). To understand the potential underlying reasons, we plotted the value distribution of data in each environment (Fig. 7(b)). It can be seen that in Maze2D and AntMaze, there is a substantial amount of optimal and near-optimal experiences, whereas in Kitchen most samples are sub-optimal (note that here the optimality is with respect to condition of diffusion model). This may explain why CFG performs better than MCSS in Kitchen. Thus we can propose a hypothesis: No guidance (MCSS) can be better than guided generation (CG, CFG) if the dataset contains a substantial portion of expert demonstration.

{8}------------------------------------------------

### 4.6 COMPARISON TO DIFFUSION POLICY

![Figure 8: Average performance of methods on different tasks. Four bar charts show expected return (normalized) for Maze2D, Kitchen, AntMaze, and MuJoCo locomotion tasks. Methods compared include BCO, CoL, IOL, SIBC, DQL*, IDQL*, Diffuser, AdptDiff, DD, and DV (Ours). Horizontal dashed lines indicate the best performance across methods.](91be14371a97fb5ce9eeb29ae18d07c3_img.jpg)

Figure 8 consists of four bar charts showing the expected return (normalized) for different tasks. The tasks are Maze2D, Kitchen, AntMaze, and MuJoCo locomotion. The methods compared are BCO, CoL, IOL, SIBC, DQL\*, IDQL\*, Diffuser, AdptDiff, DD, and DV (Ours). Horizontal dashed lines indicate the best performance across methods.

| Task              | BCO | CoL | IOL | SIBC | DQL* | IDQL* | Diffuser | AdptDiff | DD   | DV (Ours) |
|-------------------|-----|-----|-----|------|------|-------|----------|----------|------|-----------|
| Maze2D            | ~10 | ~10 | ~40 | ~70  | ~150 | ~80   | ~120     | ~140     | ~140 | ~150      |
| Kitchen           | ~15 | ~50 | ~50 | ~45  | ~60  | ~60   | ~65      | ~55      | ~65  | ~80       |
| AntMaze           | ~5  | ~40 | ~60 | ~65  | ~75  | ~70   | ~60      | ~15      | ~10  | ~80       |
| MuJoCo locomotion | ~50 | ~65 | ~75 | ~85  | ~90  | ~85   | ~80      | ~75      | ~85  | ~85       |

Figure 8: Average performance of methods on different tasks. Four bar charts show expected return (normalized) for Maze2D, Kitchen, AntMaze, and MuJoCo locomotion tasks. Methods compared include BCO, CoL, IOL, SIBC, DQL\*, IDQL\*, Diffuser, AdptDiff, DD, and DV (Ours). Horizontal dashed lines indicate the best performance across methods.

Figure 8: **Average performance of methods on different tasks.** The horizontal dashed line indicates the best performance over all methods. DV (Diffusion planning) stands out in Kitchen, Maze2D, and AntMaze; while DQL (Diffusion policy) (Wang et al., 2023b) outperforms all diffusion planning methods in MuJoCo locomotion tasks. Refer to the caption of Table 1 for method details.

Diffusion planning and diffusion policy represent two key approaches within diffusion-based decision-making. After examining the core components of diffusion planners, we turn to a comparison of diffusion planning and diffusion policy across different environments. The experimental results are illustrated in Fig. 8. We observed that diffusion planning outperforms diffusion policy in AntMaze, Kitchen, and Maze2D, whereas diffusion policy excels in MuJoCo locomotion tasks. The first three environments require precise goal achievement, such as positioning an object exactly, necessitating long-term planning. This makes them well-suited to diffusion planning, which generates entire trajectories in one step. Furthermore, these environments feature sparse reward structures, posing challenges for model-free RL algorithms typically used in diffusion policies (Wang et al., 2023b). In contrast, the objective in MuJoCo is simply to control agents to run faster, a task that is less related to lookahead planning and does not require intricate planning. RL loss functions can help diffusion policy (Wang et al., 2023b) achieve better results in such scenarios.

### 4.7 VALIDATIONS ON ADROIT DATASET

To examine whether the conclusions drawn from our experiments can generalize to other tasks, we conducted experiments on the Adroit Hand dataset (Rajeswaran et al., 2018; Fu et al., 2020), which features motion-captured human data applied to a realistic, high-degree-of-freedom robotic hand, including both challenges from planning and control. It encompasses 8 challenging tasks highlighted in the original paper, including as pen twirling, door opening, hammer use, and object relocation. We found that the results are consistent with our findings, supporting the generalizability across tasks. The detailed results are deferred to Appendix C.

### 4.8 PRACTICAL TIPS TO TAKE HOME

**Takeaway 1:** Diffusion planning is most effective for tasks requiring long-term credit assignment, while diffusion policies better fit locomotion tasks that demand less long-term planning (Sect. 4.6)

**Takeaway 2:** It is recommended to generate state plans with diffusion planners and use an inverse dynamics model to compute the corresponding actions (Sect. 4.1).

**Takeaway 3:** Implementing jump-step planning can be highly beneficial; experimenting with different planning strides is encouraged (Sect. 4.2).

**Takeaway 4:** It is worth trying to use Transformer as the backbone of diffusion planner, especially in

{9}------------------------------------------------

the tasks that require long-term lookahead planning (Sect. 4.3).

**Takeaway 5:** A single-layer Transformer is insufficient for effective planning (Sect. 4.4).

**Takeaway 6:** Larger models do not necessarily lead to better performance in diffusion planner for offline RL (Sect. 4.4).

**Takeaway 7:** Non-guidance approaches, such as Monte Carlo unconditional sampling with selection, can outperform classifier or classifier-free guidance when the dataset contains enough near-optimal trajectories (Sect. 4.5).

## 5 DISCUSSIONS

**Synergy between diffusion planning and diffusion policy.** A significant avenue for future research involves a deeper exploration of the distinctions between diffusion planning and diffusion policy. Drawing on Daniel Kahneman’s seminal work *Thinking, Fast, and Slow* (Kahneman, 2011), human cognitive processes are categorized into System 1 and System 2. Diffusion policies are analogous to System 1 processes, as they operate rapidly and efficiently (Wang et al., 2023b), making them well-suited for tasks such as locomotion (Fig. 8) that do not require extensive deliberation or long-term planning. These policies manage routine decision making with the same efficiency as intuitive responses in human cognition. Conversely, diffusion planning mirrors System 2 thinking, characterized by its slower, more deliberate, and effortful nature. This approach is particularly effective for tasks that demand long-term credit assignment (Fig. 8), involving more computations to develop effective plans. In RL terminology, diffusion planning can be broadly classified as model-based, while diffusion policy aligns with model-free methodologies. Investigating the interplay between these two systems presents a compelling intersection for both machine learning and cognitive neuroscience (Gläscher et al., 2010; Duan et al., 2016; Botvinick et al., 2019). Studies from cognitive science indicate that the brain may use a synergistic approach which arbitrates and selects the better system according to the current situation, and the preference may change over time (Lee et al., 2014; Han et al., 2024). We anticipate extensive future research focused on integrating the strengths of diffusion planning and diffusion policies to enable both efficient and effective decision-making AI.

**Computational efficiency.** Despite the effectiveness of diffusion planners, their computational cost is substantial. Our work is orthogonal to the optimization of computational cost (Dong et al., 2024a). Nonetheless, future work may consider new schemes such as the consistency model (Song et al., 2023) to improve computational efficiency.

**Interpretability and safety.** Our study focuses on a single performance metric (total return), potentially overlooking qualitative aspects such as the interpretability and reliability of the diffusion planner. Future work may consider issues such as explainability (Puiutta & Veith, 2020) and safety (Xiao et al., 2023) of diffusion planning. Leveraging the experiences from computer vision domain will be worth investigating.

**Sustainability.** Our work required significant computational resources, particularly in terms of GPU energy consumption, as we trained and evaluated thousands of models across diverse tasks. However, this investment in energy is not without purpose. We aim to provide a solid foundation for future research. Subsequent work can build upon our findings, reducing the need for extensive trial-and-error experimentation. In this way, our research contributes to energy efficiency in the long term, as researchers can reference our results and apply proven methods rather than duplicating resource-intensive exploratory efforts.

**Open problems and future directions.** In the current study, we have focused on standard Markov decision process problems (Bellman, 1957) using a popular offline RL benchmark (Fu et al., 2020). The planning and control are based on joint states and coordinates. Numerous untouched problems exist, such as vision-based decision making (Du et al., 2024; Yang et al., 2024), goal-conditioned reinforcement learning (Liu et al., 2022; Wang et al., 2023a), partially observable environments (Schmidhuber, 1991), offline-to-online deployment (Matsushima et al., 2021), and the scalability of diffusion planning models (Kaplan et al., 2020). Future efforts are anticipated to fully address these limitations. However, even within the scope of the current work, we have found several interesting phenomena and tips that are counter to common practices. Our work should be considered as a new but solid starting point for behavior planning using decision models.

 Rest of paper (reference and Appendix) is removed.