# R2Ps: Worst-Case Robust Real-Time Pursuit Strategies Under Partial Observability

Runyu Lu1,2, Ruochuan Shi2,1, Yuanheng Zhu2,1,†**, Dongbin Zhao**2,1 1 School of Artificial Intelligence, University of Chinese Academy of Sciences∗
2 State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences lurunyu17@mails.ucas.ac.cn {ruochuan.shi,yuanheng.zhu,dongbin.zhao}@ia.ac.cn

## Abstract

Computing worst-case robust strategies in pursuit-evasion games (PEGs) is timeconsuming, especially when real-world factors like partial observability are considered. While important for general security purposes, real-time applicable pursuit strategies for graph-based PEGs are currently missing when the pursuers only have imperfect information about the evader's position. Although state-of-the-art reinforcement learning (RL) methods like Equilibrium Policy Generalization (EPG) and Grasper provide guidelines for learning graph neural network (GNN) policies robust to different game dynamics, they are restricted to the scenario of perfect information and do not take into account the possible case where the evader can predict the pursuers' actions. This paper introduces the first approach to worst-case robust real-time pursuit strategies (R2PS) under partial observability. We first prove that a traditional dynamic programming (DP) algorithm for solving Markov PEGs maintains optimality under the asynchronous moves by the evader. Then, we propose a belief preservation mechanism about the evader's possible positions, extending the DP pursuit strategies to a partially observable setting. Finally, we embed the belief preservation into the state-of-the-art EPG framework to finish our R2PS learning scheme, which leads to a real-time pursuer policy through crossgraph reinforcement learning against the asynchronous-move DP evasion strategies. After reinforcement learning, our policy achieves robust zero-shot generalization to unseen real-world graph structures and consistently outperforms the policy directly trained on the test graphs by the existing game RL approach.

## 1 Introduction

Pursuit-evasion game (PEG) is an important topic long examined in the fields of robotics and security (Vidal et al., 2001; 2002; Chung et al., 2011). Many real-world tasks can benefit from the solution to an abstracted PEG, e.g., guiding a team of cops to capture a robber and aligning a team of guards to defend against an intruder. In comparison with traditional differential games (Margellos & Lygeros, 2011; Zhou et al., 2012), graph-based PEGs are convenient for describing complicated scenarios, possibly with a large scale. When we use graphs as a common structural representation, the actions of the pursuers and the evader can be abstracted as moving from a vertex to an adjacent one at each discrete timestep. The edges between the vertices can possibly represent urban streets in reality.

However, exactly solving graph-based PEGs is computationally expensive (see Goldstein & Reingold
(1995)). Even under a slight structural change, the worst-case robust pursuit strategies can be different and thus require a large amount of time to be recomputed. For example, when a traffic jam happens in the city, the related edges in the PEG graph can be frequently removed and added. This severely limits the real-time applicability of the existing methods featuring mathematical programming (Vieira et al., 2008; Horak & Bo ´ sansk ˇ y, 2017). Besides, real-world factors like partial observability, which `
1 leads to PSPACE-hardness even under a fixed opponent (see Papadimitriou & Tsitsiklis (1987)), further increase the difficulty of deriving a well-performing pursuit strategy within a time limit. Reinforcement learning (RL), which has demonstrated strong generalization capabilities in domains like large language models (see Chu et al. (2025)), provides an alternative solution to this problem. We may train a parameterized policy represented by a suitable neural network, e.g., a graph neural network (GNN) (Wu et al., 2020), on a diverse set of graphs and then generalize it to the unseen graph structures. Unfortunately, while RL has been applied to solving large-scale PEGs (Xue et al., 2022; 2021), existing research focuses more on its scalability rather than generalization capability. The methods like MT-PSRO (Li et al., 2023) and Grasper (Li et al., 2024) are limited to few-shot generalization to unseen opponent strategies and initial conditions. As is pointed out by Zhuang et al. (2025), they still have difficulty adapting to rapid changes of graph structures. The state-of-theart method, Equilibrium Policy Generalization (EPG) (Lu et al., 2025a), first examines zero-shot generalization at the level of graphs. However, whether the paradigm of EPG works under partial observability remains underexplored. Besides, all of the mentioned works do not consider the possible case that the evader may have stronger observation capabilities than the pursuers. This makes the strength of the learned pursuit strategies less convincing for real-world security purposes.

In this paper, we present an approach to finding pursuit strategies that are both worst-case robust and real-time applicable under partial observability. We start by analyzing a dynamic programming
(DP) algorithm for efficiently solving Markov PEGs and proving that it also finds optimal strategies when the evader can predict the pursuer's action and move asynchronously. With a belief update mechanism, we further extend the DP policies to a partially observable setting. The belief preservation serves to avoid the complexity of recording all observation histories through abstracting opponent information for effective decision-making. Finally, we embed the belief preservation mechanism into the reinforcement learning framework of EPG and train a generalized GNN pursuer policy under partial observability. We then evaluate the worst-case robustness of our real-time RL pursuer policy. Specifically, the contributions of this paper are threefold:
- We theoretically analyze a dynamic programming (DP) algorithm and extend the optimal strategies induced by this algorithm to asynchronous-move and partially observable scenarios.

We prove that the DP algorithm induces strictly optimal pursuit and evasion strategies when the evader moves asynchronously and design a belief preservation mechanism against the possibly unobserved evaders. Under belief preservation, we verify that the extended pursuer policy remains strong against the provably optimal perfect-information evader.

- We practically train an observation-based pursuer policy across different graph structures, deriving the first worst-case robust real-time pursuit strategies (R2PS) applicable to dynamically changing PEGs with partial observations. We combine our belief preservation mechanism with the state-of-the-art robust policy generalization paradigm, EPG, and provide an inference time complexity bound for our GNN-represented RL pursuer policy.

- Through extensive experiments, we verify that under partial observability, our RL training against the asynchronous-move DP evaders under a diverse set of graphs leads to robust zeroshot performance in unseen real-world graphs. Comparative results reveal the superiority of our RL approach over the standard game RL approach, PSRO (Lanctot et al., 2017).

## 2 Preliminaries 2.1 Problem Formulation

Adversarial games with partial observability can be generally represented by partially observable stochastic games (POSGs), where equilibrium learning has been rigorously examined in existing game-theoretic research (e.g., Lu et al. (2025b;c)). However, this formulation considers all possible observation histories and leads to a large set of decision points whose size is possibly exponential in the time horizon of the game. For the worst scenario of pursuit-evasion, while the pursuers have limited observation capabilities, the evader could still obtain the global information of the game. Since at least one side of the players possesses perfect information, we consider first expressing PEGs as two-player zero-sum Markov games and then extending the definitions to incorporate practical adversarial factors like partial observability and asynchronous moves of the evader. Two-player zero-sum Markov game. An infinite-horizon two-player zero-sum Markov game is represented by a tuple (S, A, B,P*, r, γ*), where S is the state space, A is the action space of the max-player (who aims to maximize the cumulative reward), B is the action space of the min-player
(who aims to minimize the cumulative reward), P ∈ [0, 1]|S*||A||B|×|*S|is the transition probability matrix, r ∈ [0, 1]|S*||A||B|* is the reward vector, and γ ∈ (0, 1) is the discount factor. In PEGs, the max-player is the team of m pursuers, and the min-player is the evader. We use a termination function f : S → {0, 1} to mark the states where the pursuit is successful. When f(s) = 1, the game is terminated, and a reward of +1 is received. Otherwise, a reward of 0 is received. The discount factor γ < 1 encourages the pursuers to capture the evader as soon as possible. Graph-based pursuit-evasion game. Considering the requirements of formulating large-scale real-world scenarios, we describe states and actions on a graph structure G = ⟨V, E⟩: V is the set of vertices v. The global state s = (sp, se) in a game is an element of V
m × V, where sp =
(v 1 p, v2p, · · · , vm p) ∈ Vm, and se = ve ∈ V. An edge e = (*v, v*′) ∈ E defines the adjacency between two vertices v, v′ ∈ V. For example, when we represent an urban scenario by a graph G, an edge e can be used to describe a unit length of streets. The valid actions of the m + 1 agents in a graph-based PEG are either moving to an adjacent vertex via an edge or staying at the current node. Policy and value function. Following common notations, we denote by (*µ, ν*) the joint policy of the two players, where µ is the policy of the max-player (pursuers) and ν is the policy of the min-player (evader): µ(s) ∈ ∆(A) (resp., ν(s) ∈ ∆(B)) is the max-player's (resp., min-player's) action distribution at state s ∈ S. Since ∆(A) is the probability simplex over A, µ(*s, a*) corresponds to the probability of selecting action a ∈ A at state s. Given the joint policy, we further define the value function V
µ,ν(s) = E [P∞
t=0 γ tr(st, at, bt)|s0 = s; *µ, ν* ] as in Markov decision processes.

Solution concept. A Nash equilibrium (NE) in a game is a joint policy where each individual player cannot benefit from unilaterally deviating from his/her own policy (Roughgarden, 2016). Specifically, in a two-player zero-sum MG, an NE (µ
∗, ν∗) satisfies V
µ,ν∗≤ V
µ
∗,ν∗≤ V
µ
∗,ν for any µ and ν at all states. As is well known, every MG with finite states and actions has at least one NE, and all NEs in a two-player zero-sum MG share the same value V
∗(s) = V
µ
∗,ν∗(s) = maxµminνV
µ,ν(s) =
minνmaxµV
µ,ν(s) (Shapley, 1953). In two-player zero-sum Markov games, Nash equilibrium can be viewed as a globally optimal joint policy since both players cannot be exploited by their worst-case opponents when the players move synchronously (simultaneously). Game extension. Since Markov games only take into account synchronous moves and full observations, we further allow for two variations concerning asynchronous moves and partial observability. In reality, the worst evader (from the pursuers' perspective) may have good predictions of the pursuit actions. Therefore, we allow it to decide after the pursuers' move a at each timestep. In this case, the evader policy ν(s) is transformed into an asynchronous one ν(*s, a*), and we say that a strategy is optimal for the pursuer/evader side at state s if the worst-case termination timesteps of all possible trajectories starting from s are maximized/minimized. Besides, the availability of sensors may not allow the pursuers to observe an agent that is far away (while the worst evader can). In this case, the pursuer policy µ(s) is transformed into µ(o), where o is the history of the pursuers' local observations.

## 2.2 Dynamic Programming For Markov Pegs

The traditional marking algorithm (Chung et al., 2011) provides a general idea of recursively finding optimal strategies in perfect-information PEGs. If all possible evading actions lead to the states that have been marked, then we can also mark the current state, which means the pursuers can capture the evader starting from this state. However, a direct implementation of the marking algorithm incurs a time complexity much higher than the theoretical lower bound Ω(|S|). In view of this gap, Lu et al. (2025a) introduce a dynamic programming (DP) algorithm (see Algorithm 1) that guarantees near-optimal time complexity for solving Markov PEGs. Algorithm 1 computes a distance table D through preserving a queue Q. Intuitively, the distance value D(s) indicates the worst-case timestep for the pursuer side to capture the evader starting from the global state s = (sp, se), which is guaranteed through the use of a minimax policy

µ
$$\mathrm{\Phi}^{*}(s_{p},s_{e})=\operatorname*{arg\,min}_{\mathrm{\scriptsize~neighbor~}n_{p}\mathrm{\scriptsize~of~}s_{p}}$$
. (1)
$$\left\{\operatorname*{max}_{\mathrm{neighbor~}n_{e}{\mathrm{~of~}}s_{e}}D(n_{p},n_{e})\right\}$$
Algorithm 1: Dynamic Programming for Markov PEGs Input: Graph G = ⟨V, E⟩, Pursuer Number m, and Termination Function f : V
m *× V → {*0, 1}
1 Initialize an empty queue Q and the distance table D = ∞
2 for pursuer state (positions) sp ∈ Vm do 3 for evader state se ∈ V do 4 if f(sp, se) = 1 **then** 5 D(sp, se) ← 0 6 Push (sp, se) into Q
7 end 8 end 9 end 10 **while** Q *is not empty* do 11 Pop the first element (sp, se) from Q
12 for *evader neighbor* ne ∈ Neighbor(se), ∄n
′e ∈ V,(ne, n′e) ∈ E, D(sp, n′e) > D(sp, se) do 13 for *pursuer neighbor* np ∈ Neighbor(sp) ⊂ Vm, D(np, ne) = ∞ do 14 D(np, ne) ← D(sp, se) + 1 15 Push (np, ne) into Q
16 end 17 end 18 end Output: Distance Table D
Under synchronous moves, the evader's policy is symmetrically defined as

$$\nu^{*}(s_{p},s_{e})=\operatorname*{arg\,max}_{\mathrm{\scriptsize~neighbor~}n_{e}\mathrm{\scriptsize~of~}s_{e}}\,\left\{\operatorname*{min}_{\mathrm{\scriptsize~neighbor~}n_{p}\mathrm{\scriptsize~of~}s_{p}}\,D(n_{p},n_{e})\right\}.$$
$$(2)$$
. (2)
Using mathematical induction, Lu et al. (2025a) prove that the joint policy (µ
∗, ν∗) is a near-optimal pure strategy (the proof can be found in Appendix A.1):
Theorem 1. *If there exists a pure-strategy Nash equilibrium in the Markov PEG, then the joint policy*
(µ
∗, ν∗) *defined by (1) and (2) is a Nash equilibrium.*

## 3 Extending Dynamic Programming Policies To Asynchronous Moves And Partial Observability

In this section, we further show that the distance table D generated by the DP algorithm (Algorithm 1) can also be used to construct the optimal evader policy under asynchronous moves, as well as the observation-based pursuer policies under partial observability. 3.1 ASYNCHRONOUS-MOVE SETTING When the evader moves asynchronously, we define the DP policy for the evader as ν
∗(sp, se, np) = arg max neighbor ne of se
{D(np, ne)} , (3)
where np is the neighbor of sp that the pursuers choose to move to in the current decision step, which is perceived or predicted by the evader in advance. With this information as an additional input, the evader can decide based on the pursuers' positions after their decision rather than before. As a result, the policy (3) no longer requires the inner enumeration in (2). In this case, we can show that the pursuer policy (1) and evader policy (3) induced by the distance table D are strictly optimal at all states. We start our analysis by proving Lemma 1, which reveals the minimax essence of the distance table D. The detailed proof can be found in Appendix A.2.

Lemma 1. When D(np, ne) > 0*, Algorithm 1 guarantees that*

$$D(n_{p},n_{e})=\operatorname*{min}_{\mathrm{\scriptsize~neighbor~}s_{p}\mathrm{\scriptsize~of~}n_{p}}\left\{\operatorname*{max}_{\mathrm{\scriptsize~neighbor~}s_{e}\mathrm{\scriptsize~of~}n_{e}}D(s_{p},s_{e})\right\}+1.$$

Using Lemma 1, we can further prove that D(s) implies the best possible worst-case timesteps starting from state s for both pursuer and evader sides under the asynchronous-move setting. The main results are shown as follows, and the omitted proofs can be found in Appendix A.3-A.5.

Theorem 2. Starting from any state s = (sp, se) *satisfying* D(s) = d < ∞, µ
∗ guarantees pursuit within d *steps against any evasion strategy, and* ν
∗ avoids being captured in less than d steps by any pursuit strategy. Based on the definition of optimal strategies in the asynchronous-move setting (see Section 2.1), Theorem 2 directly implies the following corollary:
Corollary 1. For any state s = (sp, se) with D(s) < ∞*, both* µ
∗ and ν
∗ *are optimal strategies.*
Furthermore, we use Theorem 3 to show that whether m perfect-information pursuers are sufficient to capture the evader starting from state s can be determined by whether D(s) < ∞:
Theorem 3. Starting from any state s = (sp, se) *with* D(s) = ∞, ν
∗can never be captured by any pursuit strategy.

$$3.2$$

## 3.2 Partially Observable Setting

Since the DP algorithm provably generates optimal strategies when both pursuer and evader sides have full observations, it is appealing to reuse the distance table D to construct a pursuit strategy under partial observability for real-world security purposes. We expect that the observation-based pursuer policy, which is extended from the DP policy under perfect information, should effectively extract history information and align with the original policy when the observation range is infinity. We consider the following partially observable setting for the pursuers, who may serve as guards in a large area. The PEG begins because an intruder is observed, whose initial position is revealed to the pursuers. Once the game starts, the position of the evader (intruder) can no longer be detected unless it is in the observation range of at least one pursuer. For example, setting the observation range to be 2 means that the evader can be detected only when its distance to one pursuer is less than 3. Under the partially observable setting, the observation history o induces the possible positions of the evader, which we denote by a set Pos. This set is initialized as {se}, where se is the initial position of the evader. As the game proceeds, it is updated based on the pursuers' observations at each timestep:

the game proceeds, it is updated based on the pursuers observations at each time:  $\text{Pos}_\text{new}=\begin{cases}\begin{matrix}\{s_e\}&\text{evader is observed at}s_e,\\ \text{Remove}(\text{Neighbor}(\text{Pos}_\text{old}))&\text{evader is not observed.}\end{matrix}\end{cases}$
where the operator Remove(·) excludes all currently observed positions (since the evader is currently unobserved) from the possible evader positions represented by Neighbor(Posold), which corresponds to the set of one-step neighbors of the nodes in Posold. Given Pos, we can express µ(o) as µ(sp,Pos) and construct a minimax policy that bounds the worst-case pursuit timesteps if we assume that the pursuers resume full observability after this step:

$$\mu(s_{p},\mathrm{Pos})=\operatorname*{arg\,min}_{\begin{subarray}{c}\operatorname*{min}\\ \operatorname*{neighbor}n_{p}\text{of}s_{p}\end{subarray}}\left\{\max_{s_{e}\in\mathrm{Pos\,neighbor}n_{e}\text{of}s_{e}}D(n_{p},n_{e})\right\},$$ $$=\operatorname*{arg\,min}_{\begin{subarray}{c}\operatorname*{min}\\ \operatorname*{neighbor}n_{p}\text{of}s_{p}\end{subarray}}\left\{\max_{n_{e}\in\mathrm{Negibor}(\mathrm{Pos})}D(n_{p},n_{e})\right\}.$$
$$({\mathfrak{H}})$$

While this policy is applicable to the case of partial observability, it is based on an assumption that the observation limitation is not continual. Under continual partial observability, we find that averaging the timesteps through preserving a **belief** about the evader's position can further encourage effective pursuit, especially when the set Pos is large. The belief-averaged pursuer policy is expressed as

cially when the set $\mathsf{Pos}$ is large. The belief-averaged pursuer policy is expressed as  $$\mu(s_{p},\text{belief})=\operatorname*{arg\,min}_{\text{neighbor n}_{p}\text{of}s_{p}}\,\left\{\frac{\sum_{\text{s}_{e}}\text{belief}(\text{s}_{e})\max_{\text{neighbor n}_{e}\text{of}s_{e}}\text{D}(\text{n}_{p},\text{n}_{e})}{\sum_{\text{s}_{e}}\text{belief}(\text{s}_{e})}\right\},$$
$$(4)$$

$$(6)$$
$$\mathbf{\Omega}(T)$$

, (6)
where the belief function is initialized to be 0 except for the initial evader position and updated by 
$$\mathrm{\begin{array}{c}{{\mathrm{belief}_{\mathrm{new}}(s_{e})\leftarrow\left\{\begin{array}{c c c}{{}}&{{}}&{{0}}&{{s_{e}\not\in\mathrm{Pos},}}\\ {{}}&{{}}&{{}}&{{}}\\ {{}}&{{}}&{{\sum}}&{{}}\end{array}\right.}}\nu(v,s_{e})\mathrm{\begin{array}{c}{{}}\\ {{}}\end{array}}\nu(v,s_{e})\mathrm{\begin{array}{c}{{}}\\ {{}}\end{array}}s_{e}\in\mathrm{Pos}.}$$
(
0 se ∈/ Pos,
neighbor v of se
ν(*v, s*e)beliefold(v) se ∈ Pos. (7)
Since the pursuer side cannot obtain the evader's policy ν when no prior knowledge is available, ν(v) is set to be a uniform distribution over Neighbor(v) by default. As the original DP policy µ
∗(s) is provably optimal, Lemma 2 guarantees that both the positionextended policy µ(sp,Pos) and the belief-averaged policy µ(sp, belief) maintain the pursuit optimality when there is unlimited observation capability. The proof can be found in Appendix A.6. Lemma 2. *When* Pos *is always a singleton, both pursuer policies (5) and (6) will be reduced to their* perfect-information counterpart (1).

Note that the time complexity of preserving Pos and belief is only O˜(|V|) at each timestep, where O˜
hides the additional factor of enumerating the neighbors. Since the average degree in the real-world graphs can be small (see Table 1 in Section 5), the computation is practically efficient. In Appendix B, we provide the illustrations of the belief preservation process for a more intuitive understanding.

## 4 Finding Robust Real-Time Pursuit Strategies (R2Ps) Via Adversarial Reinforcement Learning Across Graphs 4.1 Adversarial Reinforcement Learning

Since the DP algorithm has a lower-bound time complexity exponential in the agent number, it can be impractical to directly apply the DP policies in real time when the graph structure of the game dynamically changes. In view of this problem, we further combine our belief preservation mechanism with the idea of Equilibrium Policy Generalization (EPG) (Lu et al., 2025a) to construct a reinforcement learning method, which makes use of some preprocessed D tables and the induced policies to train a generalized pursuer policy across a diverse set of graphs. We use the cross-graph RL policy for zero-shot generalization under unseen graph structures, aiming to derive worst-case robust real-time pursuit strategies (R2PS) under partial observability.

![5_image_0.png](5_image_0.png)

Figure 1 illustrates the cross-graph reinforcement learning pipeline, which features unexploitable evader policies as adversaries. The training set contains graphs with various topologies Gi and the DP policies (µ
∗ i
, ν∗
i
) induced by the preprocessed D tables. In each iteration, a graph Gi along with the policy (µ
∗ i
, ν∗
i
) is sampled. Under graph G = Gi, we use µ
∗ = µ
∗
ias the reference policy to guide policy training and use ν
∗ = ν
∗
ias the adversarial policy. Following the principle of EPG, we train a cross-graph pursuer policy through reinforcement learning against ν
∗ with the guidance of µ
∗.

Specifically, for a transition (*s, a, b, r, s*′) in the replay buffer: s is a randomly generated global state in the sampled graph; a is the pursuers' joint action sampled from the current policy model πθ, which is ideally a graph neural network (Wu et al., 2020) with parameter θ that enables real-time inference; b is the evader's action generated from the asynchronous-move opponent policy ν
∗(3); the instant reward r and the next state s
′are generated by the PEG dynamics under graph structure G = ⟨V, E⟩.

Given state s, the reference policy µ
∗ generates a deterministic reference action a
∗ = µ
∗(s) and serves to construct the policy loss L(θ |s ) = Jπ(θ |s ) + βDKL (µ
∗(s), π(s)) = Jπ(θ |s ) − β log πθ(*s, a*∗), (8)
where Jπ(θ |s ) is the original policy loss of any backbone (multi-agent) reinforcement learning algorithm (e.g., MAPPO (Yu et al., 2022)), and β is a hyperparameter that balances policy guidance
(for efficient exploration) and reinforcement learning loss (for policy optimization).

When training pursuers under partial observability, we transform the input of the policy model πθ by s ← (sp,Pos, belief)
and use the observation-based policy µ(sp,Pos) (5) or µ(sp, belief) (6) to replace µ
∗(s) (1), where Pos and belief are the preserved evader information under partial observability. For dynamic games like PEGs, the policy space has certain transitivity structures. Czarnecki et al. (2020) show that the strategies in real-world games have different levels of transitive strength, with Nash equilibrium being the strongest. In a single-graph PEG, reinforcement learning against the optimal evader policy ν
∗ helps to exclude the pursuer policies that are transitively weaker. Crossgraph training is similar to finding the joint part of the remaining strategies and abstracting them to a worst-case robust policy under a diverse set of graph structures, where the divisions on the policy space through adversarial RL can be different. Imagine that a half space is excluded after each single-graph division and that the division criteria of different graphs are independent due to structural distinctions. In this ideal case, the cross-graph policy will be improved at an exponential level across a diverse training corpus, leading to robust pursuit strategies even under partial observability.

## 4.2 Implementation And Complexity Analysis

Technically, we use soft-actor critic (SAC) (Haarnoja et al., 2018; Christodoulou, 2019) as the backbone RL algorithm and employ a decentralized architecture with a parameter-sharing graph neural network (GNN) (Cao et al., 2023; Lu et al., 2025a) to represent the graph-based policy of the homogeneous pursuers. The SAC algorithm features a self-adaptive entropy regularization that balances exploration and exploitation, with double Q-learning (Hasselt, 2010) employed to avoid overestimation. The GNN architecture combines multi-head self-attention (Vaswani et al., 2017) with adjacent-matrix masks to encode graph-based states. The state embedding is then sent into a decoder followed by a pointer network (Vinyals et al., 2015) for graph-based policy output.

The implementation details and hyperparameter setting are reserved in Appendix C to save space1.

According to the corresponding analysis, the overall time complexity of computing the graph-based state feature is O(n 2m), where n = |V| is the number of vertices in the graph, and m is the number of pursuers. Since the complexity of GNN queries is also O(n 2m), and the complexity of preserving Pos and belief is O˜(n), the overall inference time complexity of the RL pursuer policy at each timestep is only O(n 2m) + O(n 2m) + O˜(n) = O(n 2m). In comparison, the time complexity of recomputing DP policies is O˜(n m+1) under dynamically changing graph structures (see Lu et al.

(2025a)), as Algorithm 1 needs to be repeatedly executed. Here we briefly show the inference time gap arising from this complexity distinction. When n = 1000 and m = 2, it takes over 2 minutes to run Algorithm 1 at each timestep using an Intel Core i9-13900HX CPU. The inference time of our GNN-represented RL policy, however, is less than 1 second under the same condition. Our subsequent tests further show this inference can be reduced to below 0.01 seconds under GPU accelerations.

## 5 Evaluations

Here we provide our experimental evaluations of single-graph DP pursuers and cross-graph RL pursuers under partial observability. We assume that there are two pursuers (m = 2) against the single evader. This is a reasonable setting in view of the graph-theoretic result that 3 pursuers with full observations can always capture the evader in any planar graph (Fromme & Aigner, 1984). The initial position is randomly generated under the restriction that the distance between the evader and the pursuers is larger than the observation range of 2. Besides, no observation sensors except for the 1Code can be found at https://github.com/Cahemgco/EPG code.

pursuers themselves are allowed. The test graphs include Grid Map (a 10 × 10 grid), Scotland-Yard Map (from the board game Scotland-Yard), Downtown Map (a real-world location from Google Maps), and 7 famous real-world spots (from Times Square to Sydney Opera House). The graph details are shown in Appendix D.1, and the statistics of these graphs are shown in Table 1 (left).

Node Degree Diameter Shortest Path DPPos DPbelief

Grid Map 100 3.60 18 0.00 0.59 **0.78**

Scotland-Yard Map 200 3.91 19 0.00 0.44 **0.63**

Downtown Map 206 2.98 19 0.02 0.73 **0.90**

Times Square 171 2.58 22 0.01 0.41 **0.69**

Hollywood Walk of Fame 201 2.42 31 0.01 0.25 **0.48**

Sagrada Familia 231 2.60 25 0.00 0.24 **0.36**

The Bund 200 2.53 29 0.03 0.30 **0.57**

Eiffel Tower 202 2.34 38 0.29 0.69 **0.94**

Big Ben 192 2.48 34 0.08 0.54 **0.74**

Sydney Opera House 183 2.33 37 0.05 0.47 **0.87**

## 5.1 Evaluations Of Extended Dp Pursuers

We first evaluate the strength of the extended DP pursuers under partial observability (Section 3.2).

We denote by DPPos the position-extended pursuer (5) and by DPbelief the belief-averaged pursuer
(6). The pursuers succeed (f(s) = 1) when at least one of them is adjacent to the evader on the graph within 128 timesteps, and the success rates are averaged over 500 tests. To simulate the difficult case for security purposes, the evader is set to be the provably optimal DP evader (3) with global observations and asynchronous moves. For an intuitive comparison, we also include the result of directly following the shortest path to the evader under full observability. As is shown in Table 1 (right), the shortest-path strategy can hardly capture the optimal DP evader. In comparison, though under a limited observation range of 2, the extended DP pursuers demonstrate significantly higher success rates. Besides, DPbelief consistently outperforms DPPos. This result verifies that the direct minimax policy (5) can be improved through belief averaging. Actually, since equation (5) treats all possible positions as equal, the result of the inner max can be very large when the size of Pos is large, leading to pessimistic pursuit behaviors like staying at certain "rest points." We further take a look at how observation capabilities could affect success rates. We increase the observation range and evaluate the performance of DPbelief (6). As is shown in Table 6 (Appendix D.2), the success rates monotonically increase with the observation range and reach 100% when the range exceeds 5. While D(·) is an accurate estimator of the worst-case pursuit distance in Markov PEGs, it becomes an optimistic one under partial observability. Nevertheless, the experimental results show that combining this optimistic estimator with belief information can maintain the strength of the DP-based pursuit strategies, even under very limited observation capabilities.

## 5.2 Evaluations Of Generalized Rl Pursuers

Now, we implement and evaluate our cross-graph reinforcement learning method aimed at R2PS (Section 4). We discretize the maps from the Dungeon environment (Chen et al., 2019) to construct a synthetic training set containing 150 graphs and further include 150 random urban locations from Google Maps to create a large training set with a total of 300 graphs, where the maximum node number is no more than 500. We apply the R2PS learning scheme to the synthetic training set and the large training set. Appendix C.4 provides the learning curves of the pursuer policies under partial observability. As is shown in Figure 4, using the extended DP pursuers as guidance (β = 0.1) helps to improve the training efficiency over pure reinforcement learning (β = 0) under either training set. Policy-Space Response Oracles (PSRO) (Lanctot et al., 2017) is a general reinforcement learning method extended from the game-theoretic approach of double oracle (DO) (McMahan et al., 2003) for equilibrium finding. Here we compare the zero-shot performance of our generalized pursuer policy with a PSRO policy that is directly trained on the 10 test graphs using 10 iterations (10000

Evader Policy Stay DPsync DPasync BRasync

Pursuer Policy Ours PSRO Ours PSRO Ours PSRO Ours

Grid Map 1.00 1.00 **1.00** 0.94 **1.00** 0.88 1.00

Scotland-Yard Map 1.00 1.00 **1.00** 0.47 **0.76** 0.00 0.73

Downtown Map **1.00** 0.99 **1.00** 0.88 **0.99** 0.03 0.92

Times Square **1.00** 0.93 **1.00** 0.16 **0.95** 0.04 0.27

Hollywood Walk of Fame **1.00** 0.95 **0.90** 0.00 **0.38** 0.00 0.10

Sagrada Familia **0.99** 0.93 **0.96** 0.07 **0.20** 0.00 0.20

The Bund **1.00** 0.95 **0.92** 0.31 **0.25** 0.04 0.23

Eiffel Tower **1.00** 0.99 **1.00** 0.97 **1.00** 0.52 0.55

Big Ben **1.00** 0.99 **1.00** 0.29 **0.82** 0.24 0.65

Sydney Opera House **1.00** 0.98 **1.00** 0.07 **0.95** 0.11 0.31

episodes per iteration). Our RL policy aimed at R2PS, however, is pretrained under the synthetic training set with 150 graphs for 30000 episodes (β = 0.1) and then trained under the 150 random urban graphs for 70000 episodes. Since our training process never comes across the test graphs, our RL policy has to zero-shot generalize to these unseen graph structures during evaluations. As is shown in Table 2, our pursuer policy consistently outperforms the PSRO pursuer policy in the real-world graphs against a variety of opponents, where:
- Stay corresponds to an evader that stays at the initial position. Since the initial distance between the pursuers and the evader is larger than the observation range, and the pursuers have no prior knowledge about the evader's policy, staying still is a reasonable strategy and leads to the occasional failure of these RL pursuers.

- DPsync corresponds to the DP evader policy (2) under synchronous moves, and DPasync corresponds to the strictly optimal policy (3) under asynchronous moves. It is clear that the asynchronous-move evaders are much stronger than the synchronous-move ones due to the advantage of forecasting the pursuers' decisions. Against DPasync, the PSRO pursuers struggle under most of the test graphs in comparison with ours.

- BRasync corresponds to the best-responding asynchronous-move evader directly trained against our RL pursuers in the test graphs for 30000 episodes (converged). Even under this worst case, the success rates of our generalized pursuers are over 50% in half of the graphs.

Since our worst-case zero-shot performance is clearly better than the PSRO policy directly trained on the test graphs, we can say that our real-time strategies are worst-case robust even under varying graph structures, which implies that our approach achieves R2PS under partial observability.

## 5.3 Scalability Tests And Ablation Studies

Node Number Sucess Rate RL Time (s) DP Time (s)

Times Square 1805 0.56 0.009837 101

Hollywood Walk of Fame 1251 0.46 0.007917 33

Sagrada Familia 2065 0.33 0.009895 139

The Bund 1723 0.46 0.008117 83

Eiffel Tower 1825 0.41 0.009616 96

Big Ben 1681 0.49 0.007752 79

Sydney Opera House 744 0.76 0.007648 6

Now we further verify the real-time pursuit capability under the graphs with higher complexity. We create another set of test graphs based on the seven famous locations in Table 1 (from Times Square to Sydney Opera House). Compared to the original graphs, the new graphs double both the map range and the discretization accuracy, leading to significantly larger node numbers. The success rates of our RL pursuer policy against the optimal evader DPasync and the inference time comparisons under an NVIDIA GeForce RTX 2080 Ti GPU are shown in Table 3. Clearly, our RL policy requires significantly smaller inference time in comparison with DP and maintains desirable overall performance under large graphs in comparison with the results in Table 2. Figure 6 (Appendix D.2) provides the scaling plots of our GNN-based RL policy inference and DP computation time. We are also curious about whether our RL policy trained under the limited observation range of 2 can demonstrate better performance when the observation range is larger during inference time. As is shown in Table 7 (Appendix D.2), the success rates of our RL pursuers monotonically increase with the observation range. This additional result implies that our RL policy trained with the minimum observability can be directly applied to the cases with better sensing capabilities. Finally, we examine how the belief updates affect pursuit performance. As we have mentioned, our belief preservation (7) always employs a uniform evader policy ν since we could not access prior information about the true opponent. However, if we manage to obtain such information in reality, we can instantly improve the pursuit performance by replacing ν with the actual evader policy. As is shown in Table 4, utilizing known opponent information improves success rates against the best-responding evader BRasync. On the other hand, if we reduce the belief update frequency from every single step (original) to every 2 or 3 steps, then the pursuit success rates will instantly decline. This result further demonstrates the benefits of our belief update mechanism.

| Belief Update Condition   | Known Opponent   | Original   | Every 2 Steps   | Every 3 Steps   |
|---------------------------|------------------|------------|-----------------|-----------------|
| Grid Map                  | 1.00             | 1.00       | 0.60            | 0.42            |
| Scotland-Yard Map         | 0.99             | 0.73       | 0.34            | 0.28            |
| Downtown Map              | 1.00             | 0.92       | 0.61            | 0.39            |
| Times Square              | 0.42             | 0.27       | 0.18            | 0.17            |
| Hollywood Walk of Fame    | 0.13             | 0.10       | 0.04            | 0.03            |
| Sagrada Familia           | 0.28             | 0.20       | 0.12            | 0.05            |
| The Bund                  | 0.54             | 0.23       | 0.13            | 0.12            |
| Eiffel Tower              | 0.81             | 0.55       | 0.32            | 0.29            |
| Big Ben                   | 0.82             | 0.65       | 0.40            | 0.25            |
| Sydney Opera House        | 0.54             | 0.31       | 0.22            | 0.15            |

## 6 Conclusion

This paper presents a general approach to worst-case robust real-time pursuit strategies under partial observability and varying graph structures. We first theoretically examine a dynamic programming (DP) algorithm and prove that it can unify the solutions to Markov PEGs with either synchronous moves or asynchronous moves. Then, we propose a belief preservation mechanism to efficiently abstract evader information from the observation histories of the pursuers and thus construct the observation-based pursuer policies. Finally, we embed the belief preservation mechanism into the framework of EPG (Lu et al., 2025a) to find robust real-time pursuit strategies, fulfilling crossgraph reinforcement learning against the asynchronous-move DP evader under partial observability. Experiments show that our observation-based DP pursuers can be used as guidance to facilitate efficient policy exploration during RL training. Under unseen real-world graph structures, our crossgraph policy manages to generate real-time pursuit strategies with worst-case robustness, consistently outperforming the PSRO policy directly trained under the test graphs. Comparative results also reveal that the pursuers can benefit from belief updates, while the evader benefits from asynchronous moves.

In this work, the belief preservation mechanism provides an affordable way to handle partial observability in real time. We show that this mechanism can be effectively combined with the existing PEG methods like DP and EPG. After adversarial reinforcement learning across graphs, a generalized pursuer policy under belief preservation is eventually derived, leading to the first worst-case robust real-time pursuit strategies under partial observability. Hopefully, the current research on PEGs could encourage subsequent works on the broader research topics concerning real-world security.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grants 62293541 and 62136008 and in part by the Beijing Nova Program under Grant 20240484514. We also thank all the reviewers for their helpful suggestions during the ICLR review process.

## References

Yuhong Cao, Tianxiang Hou, Yizhuo Wang, Xian Yi, and Guillaume Sartoretti. Ariadne: A reinforcement learning approach using attention-based deep networks for exploration. In *2023 IEEE* International Conference on Robotics and Automation (ICRA), pp. 10219–10225. IEEE, 2023.

Jiajun Chai, Zijie Zhao, Yuanheng Zhu, and Dongbin Zhao. A survey of cooperative multi-agent reinforcement learning for multi-task scenarios. *Artificial Intelligence Science and Engineering*, 1 (2):98–121, 2025. doi:10.23919/AISE.2025.000008.

Fanfei Chen, Shi Bai, Tixiao Shan, and Brendan Englot. Self-learning exploration and mapping for mobile robots via deep reinforcement learning. In *Aiaa scitech 2019 forum*, pp. 0396, 2019.

Petros Christodoulou. Soft actor-critic for discrete action settings. *arXiv preprint arXiv:1910.07207*,
2019.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V
Le, Sergey Levine, and Yi Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. In *Forty-second International Conference on Machine Learning*, 2025.

Timothy H Chung, Geoffrey A Hollinger, and Volkan Isler. Search and pursuit-evasion in mobile robotics: A survey. *Autonomous robots*, 31:299–316, 2011.

Wojciech M Czarnecki, Gauthier Gidel, Brendan Tracey, Karl Tuyls, Shayegan Omidshafiei, David Balduzzi, and Max Jaderberg. Real world games look like spinning tops. Advances in Neural Information Processing Systems, 33:17443–17454, 2020.

M Fromme and M Aigner. A game of cops and robbers. *Discrete Appl. Math*, 8:1–12, 1984.

Arthur S Goldstein and Edward M Reingold. The complexity of pursuit on a graph. Theoretical computer science, 143(1):93–112, 1995.

Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algorithms and applications. *arXiv preprint arXiv:1812.05905*, 2018.

Hado Hasselt. Double Q-learning. *Advances in Neural Information Processing Systems*, 23, 2010. Karel Horak and Branislav Bo ´ sansk ˇ y. Dynamic programming for one-sided partially observable `
pursuit-evasion games. In *International Conference on Agents and Artificial Intelligence*, volume 2, pp. 503–510. SCITEPRESS, 2017.

Marc Lanctot, Vinicius Zambaldi, Audrunas Gruslys, Angeliki Lazaridou, Karl Tuyls, Julien Perolat, ´
David Silver, and Thore Graepel. A unified game-theoretic approach to multiagent reinforcement learning. *Advances in Neural Information Processing Systems*, 30, 2017.

Pengdeng Li, Shuxin Li, Xinrun Wang, Jakub Cerny, Youzhi Zhang, Stephen McAleer, Hau Chan, `
and Bo An. Grasper: A generalist pursuer for pursuit-evasion problems. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems, pp. 1147–1155, 2024.

Shuxin Li, Xinrun Wang, Youzhi Zhang, Wanqi Xue, Jakub Cern ˇ y, and Bo An. Solving large-scale `
pursuit-evasion games using pre-trained strategies. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 11586–11594, 2023.

Runyu Lu, Peng Zhang, Ruochuan Shi, Yuanheng Zhu, Dongbin Zhao, Yang Liu, Dong Wang, and Cesare Alippi. Equilibrium policy generalization: A reinforcement learning framework for crossgraph zero-shot generalization in pursuit-evasion games. In *The Thirty-ninth Annual Conference* on Neural Information Processing Systems, 2025a.

Runyu Lu, Yuanheng Zhu, and Dongbin Zhao. Divergence-regularized discounted aggregation:
Equilibrium finding in multiplayer partially observable stochastic games. In *The Thirteenth* International Conference on Learning Representations, 2025b.

Runyu Lu, Yuanheng Zhu, Dongbin Zhao, Yu Liu, and You He. Last-iterate convergence to approximate Nash equilibria in multiplayer imperfect information games. IEEE Transactions on Neural Networks and Learning Systems, 36(8):13859–13873, 2025c. doi:10.1109/TNNLS.2024.3516693.

Kostas Margellos and John Lygeros. Hamilton-Jacobi formulation for reach-avoid differential games.

IEEE Transactions on Automatic Control, 56(8):1849–1861, 2011.

H Brendan McMahan, Geoffrey J Gordon, and Avrim Blum. Planning in the presence of cost functions controlled by an adversary. In Proceedings of the 20th International Conference on Machine Learning (ICML-03), pp. 536–543, 2003.

Christos H Papadimitriou and John N Tsitsiklis. The complexity of Markov decision processes.

Mathematics of operations research, 12(3):441–450, 1987.

Tim Roughgarden. *Twenty lectures on algorithmic game theory*. Cambridge University Press, 2016. Lloyd S Shapley. Stochastic games. *Proceedings of the National Academy of Sciences*, 39(10):
1095–1100, 1953.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

Rene Vidal, Shahid Rashid, Cory Sharp, Omid Shakernia, Jin Kim, and Shankar Sastry. Pursuitevasion games with unmanned ground and aerial vehicles. In *Proceedings 2001 ICRA. IEEE* International Conference on Robotics and Automation (Cat. No. 01CH37164), volume 3, pp. 2948–2955. IEEE, 2001.

Rene Vidal, Omid Shakernia, H Jin Kim, David Hyunchul Shim, and Shankar Sastry. Probabilistic pursuit-evasion games: Theory, implementation, and experimental evaluation. IEEE Transactions on Robotics and Automation, 18(5):662–669, 2002.

Marcos AM Vieira, Ramesh Govindan, and Gaurav S Sukhatme. Optimal policy in discrete pursuitevasion games. *Department of Computer Science, University of Southern California, Tech. Rep*,
2008.

Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. Advances in Neural Information Processing Systems, 28, 2015.

Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S Yu. A
comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems, 32(1):4–24, 2020.

Wanqi Xue, Youzhi Zhang, Shuxin Li, Xinrun Wang, Bo An, and Chai Kiat Yeo. Solving large-scale extensive-form network security games via neural fictitious self-play. In Proceedings of the 28th International Joint Conference on Artificial Intelligence, 2021.

Wanqi Xue, Bo An, and Chai Kiat Yeo. NSGZero: Efficiently learning non-exploitable policy in large-scale network security games with neural Monte Carlo tree search. In *Proceedings of the* AAAI Conference on Artificial Intelligence, volume 36, pp. 4646–4653, 2022.

Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of PPO in cooperative multi-agent games. Advances in Neural Information Processing Systems, 35:24611–24624, 2022.

Zijie Zhao, Honglei Guo, Shengqian Chen, Kaixuan Xu, Bo Jiang, Yuanheng Zhu, and Dongbin Zhao.

Empowering multi-robot cooperation via sequential world models. In *The Thirteenth International* Conference on Learning Representations, 2026.