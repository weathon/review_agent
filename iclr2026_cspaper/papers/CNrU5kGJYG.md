# Trojanto: Action-Level Backdoor Attacks Against Trajectory Optimization Models

Yang Dai1 Oubo Ma2 Longfei Zhang1 **Xingxing Liang**1 ∗
Xiaochun Cao3 Shouling Ji2 Jiaheng Zhang4 Jincai Huang1 ∗ **Li Shen**3 ∗
1Laboratory for Big Data and Decision, National University of Defense Technology 2Zhejiang University 3Shenzhen Campus of Sun Yat-sen University 4National University of Singapore
{daiyang2000, zhanglongfei, liangxingxing, huangjincai}@nudt.edu.cn, {mob, sji}@zju.edu.cn,caoxiaochun@mail.sysu.edu.cn, jhzhang@nus.edu.sg, mathshenli@gmail.com

## Abstract

Trajectory Optimization (TO) models have achieved remarkable success in offline reinforcement learning (offline RL). However, their vulnerability to backdoor attacks remains largely unexplored. We find that existing backdoor attacks in RL, which typically rely on reward manipulation throughout training, are largely ineffective against TO models due to their inherent sequence modeling nature and large network size. Moreover, the complexities introduced by high-dimensional continuous action further compound the challenge of injecting effective backdoors. To address these gaps, we propose TrojanTO, the first action-level backdoor attack against TO models. TrojanTO is a post-training attack and employs alternating training to forge a strong connection between triggers and target actions, ensuring high attack effectiveness. To maintain attack stealthiness, it utilizes trajectory filtering to preserve the benign performance and batch poisoning for trigger consistency. Extensive evaluations demonstrate that TrojanTO effectively implants backdoors across diverse tasks and attack objectives with a low attack budget (0.3% of trajectories). Furthermore, TrojanTO exhibits broad applicability to DT, GDT, and DC, underscoring its scalability across diverse TO model architectures.

## 1 Introduction

Offline reinforcement learning (offline RL) has emerged as a prominent research area, distinguished by its capability to derive policies using existing datasets without requiring interaction during training. In this field, trajectory optimization (TO) models, such as Decision Transformer (DT) (Chen et al., 2021b) and Decision ConvFormer (DC) (Kim et al., 2023), have gained popularity due to the powerful modeling capabilities (Vaswani et al., 2017). These capabilities unlock the potential of TO models for embodied intelligence (Wei et al., 2023; Brohan et al., 2023), robotic control (Chen et al., 2021b; Kim et al., 2023; Hu et al., 2023), and other domains involving tasks with continuous action spaces. Despite their success, the potential security risks of TO models remain largely underexplored, as their unique architecture and training paradigm are distinct from those threatening traditional RL agents. Backdoor attacks are one of the security threats to RL agents. An adversary typically embeds backdoors by manipulating transitions during the agent's training process (Cui et al., 2024; Rathbun et al., 2024b; Ma et al., 2025; Zhang et al., 2025). Once trained, the victim agent behaves normally under benign conditions, but it executes a predetermined malicious action when the trigger is activated. These attacks are effective against agents that operate on principles related to Bellman equations, as they refine their policies based on reward signals to maximize long-term returns. For these agents, reward manipulation is the crucial attack vector. However, this attack paradigm is challenging to implement against TO models. TO models directly fit target actions and minimize reconstruction loss rather than relying on reward maximization. Moreover, as TO models continue to scale in size and training cost, attacks coupled with the training phase become increasingly impractical and infeasible.

∗Corresponding authors.

1 Additionally, achieving precise manipulation in high-dimensional continuous action spaces presents a more significant challenge than in finite discrete action spaces. This heightened difficulty stems from the infinite nature of continuous action spaces, where actions are represented by real-valued vectors rather than a finite set of distinct choices. Consequently, developing effective action-level backdoor attacks for TO models under low attack budgets remains a significant challenge. This paper proposes TrojanTO, the first action-level backdoor attack against TO models. We first conduct empirical studies to investigate the influence of the three fundamental elements, i.e., (action, state, reward) on the efficacy of backdoor against TO models. We find that both the target action and the trigger design are crucial, whereas reward manipulation is unnecessary. As a post-training attack, TrojanTO decouples the attack from the training process and operates by efficiently modifying the pretrained TO model. To achieve this, an alternating training module is used to forge a strong coupling between the trigger and the target action. Moreover, to simultaneously achieve high effectiveness and stealthiness, TrojanTO employs trajectory filtering and batch poisoning. These modules preserve stealthiness by minimizing the impact on the benign performance and ensuring consistent trigger association during training and evaluation. The contributions of this paper are as follows:
- To the best of our knowledge, this work presents the first systematic study of action-level backdoors in offline RL and introduces a novel post-training attack paradigm. Our findings underscore an underexplored threat vector for the TO models.

- Our comprehensive investigation into the factors influencing the TO model security reveals that both action and state are essential elements. Consequently, the evaluation of action-level backdoors should encompass diverse target actions.

- Based on the principle of consistent poisoning, we propose TrojanTO. Extensive experiments demonstrate the effectiveness of TrojanTO across a variety of RL tasks and TO models, evaluated in scenarios involving diverse target actions.

## 2 Related Works

Backdoor attacks pose a significant threat to RL. TrojDRL (Kiourti et al., 2019) is a seminal work, inspiring subsequent investigations into backdoor in RL (Yang et al., 2019; Ashcraft & Karra, 2021; Wang et al., 2021; Cui et al., 2024; Rathbun et al., 2024a;b; Ma et al., 2025; Rathbun et al., 2026), including extensions to multi-agent RL (Chen et al., 2022; Zheng et al., 2023; Yu et al., 2024; 2025). In the offline RL, Ma et al. (2019) first investigated backdoors within tabular environments and linear quadratic regulator control systems. Recently, Gong et al. (2024b) proposed a method for generating poisoned trajectories derived from a pre-trained evil policy to implement the policy-level backdoor. However, the applicability of these existing methods is largely confined to traditional RL algorithms or simplistic offline settings, and they fail to address the emerging paradigm of TO models.

In summary, the vast majority of existing RL backdoors are implemented as training-time attacks that bind the backdoor to the agent's training loop, typically via reward manipulation. However, this paradigm is incompatible with TO models for two key reasons: first, the prohibitive computational cost of retraining such large-scale models makes it impractical. Second, their training objective is not directly influenced by reward signals. The most relevant prior work is Baffle (Gong et al., 2024b). It is a data poisoning backdoor in offline RL, modifying the training dataset to implant a policy-level backdoor by injecting malicious trajectories generated from a pre-trained adversarial policy. However, a high poisoning rate (10%) limits its practical applicability and stealth. These limitations motivate the need for a more practical and stealthy backdoor attack against TO models. Post-training attacks represent a more realistic threat model, yet they remain largely unexplored in the context of RL. Detailed introduction and potential scenarios in RL are provided in Appendix A.

## 3 Preliminary And Problem Setup 3.1 Offline Reinforcement Learning

Given a dataset consisting of N trajectories {τi}
N
i=1, where τi = (s0, a0, r0, · · · , sT −1, aT −1, rT −1, sT ). The action at is generated by the behavior policy πβ from the state st in the time step t, i.e. at ∼ πβ(st). The next state st+1 and reward rt are determined by the dynamics p(s
′, r|*s, a*). T
is the trajectory's length. Traditional offline RL algorithms Mao et al. (2024a;b) aim to maximize the expected return, typically by utilizing Bellman equations within the Markov Decision Process. In contrast, TO models reframe this problem as a sequence modeling problem and intrinsically process the input sequence to generate the output sequence (Chen et al., 2021b; Kim et al., 2023; Dai et al., 2024). Specifically, the TO model, denoted as π(·), takes a sequence of actions at−K:t−1, states st−K+1:t, and corresponding returns-to-go (RTGs) Rˆt−K+1:t as input sequence, where the RTG at a timestep t, denoted as Rˆt, represents the sum of future rewards PT
t
′=t rt
′ . During evaluation, this is typically initialized with a target return Rˆ0 and updated via Rˆt+1 = Rˆt − rt.

From the output sequence of π(·), the action aˆt is extracted from the element corresponding to the state st. This is formally expressed as aˆt = π(at−K:t−1, st−K+1:t, Rˆt−K+1:t)t, where (·)t signifies the extraction of the output element pertinent to st. The primary objective of TO models is to minimize a reconstruction loss, L = E(*R,s,a* ˆ )∼τ h1T
PT −1 t=0 LMSE/CE(ˆat; at)
i, where LMSE/CE
represents the MSE for continuous action spaces and the Cross-Entropy for discrete action spaces.

## 3.2 Backdoor Attacks In Rl

Recent studies have established the vulnerability of RL agents to backdoor attacks. These attacks are generally classified into two primary types: policy-level and action-level. Both types of attacks can significantly impact sequential decision-making, as detailed in Appendix E. Policy-Level Backdoor. The adversary's objective is to manipulate the victim agent's long-term objectives, e.g., minimizing the returns the agent receives whenever the trigger is activated (Yang et al., 2019; Wang et al., 2021; Gong et al., 2024b; Yu et al., 2025). It focuses solely on whether the adversary's objective can be achieved and does not consider the model's specific actions. Action-Level Backdoor. The adversary's objective is to compel the victim agent to output a specific target action whenever the trigger is activated (Kiourti et al., 2019; Ashcraft & Karra, 2021; Chen et al., 2022; Cui et al., 2024; Rathbun et al., 2024b; Ma et al., 2025). Such fine-grained control could (1) manipulate a single action at a critical moment to cause irreversible and catastrophic outcomes; (2) orchestrate complex, long-term manipulations through sequential trigger activation; (3) flexibly pursue diverse objectives simply by altering the trigger activation patterns.

## 3.3 Threat Model

We consider a potent supply-chain attack scenario where the adversary aims to implant a backdoor into the pretrained TO model without access to the original training dataset. Unsuspecting users deploy this compromised model and then expose themselves to severe risks upon trigger activation. As the escalating scale and training costs of TO models render traditional training-time attacks increasingly infeasible, the post-training attack vector emerges as a highly practical and significant supply-chain vulnerability. To better position TrojanTO and clarify these critical distinctions, we categorize RL backdoor attacks by their stage of intervention: (1) **Pre-training** (Gong et al., 2024b): The adversary poisons the dataset *prior to* the model training. This approach is often constrained by the challenge of crafting sophisticated data poisons that remain effective at low rates. (2) Duringtraining (Rathbun et al., 2024b): The adversary manipulates the training loop directly (e.g., by altering reward signals). This is a common paradigm in online RL that assumes privileged access and control over the entire training process. (3) **Post-training** (TrojanTO): The adversary modifies a pretrained model. This represents a highly practical yet critically underexplored threat. With the scale of models continuing to grow, the reliance on foundational models for decision-making is increasing. Adversary's Objective. The adversary aims to craft the backdoored model π˜ such that its output approximates the target action a
† whenever the trigger δ is activated. Simultaneously, its sequential decision-making must remain indistinguishable from that of the original policy π on benign inputs. This dual objective can be formally expressed as the minimization of the following loss function:

$$\operatorname*{min}_{\hat{\pi}}\sum_{s}\left\|\hat{\pi}([a],[s]+\delta,[\hat{R}])_{t}-a^{\dagger}\right\|+\lambda\left\|\hat{\pi}([a],[s],[\hat{R}])_{t}-\pi([a],[s],[\hat{R}])_{t}\right\|,$$
s
 , (1)
(1) $\frac{1}{2}$
where *∥ · ∥* denotes the L2 norm, λ ∈ [0, 1] is a hyperparameter balancing the two objectives. The term [s] represents the state sequence over the past K time steps, i.e., st−K+1:t. Similarly, [Rˆ] and
[a] denote the sequence of RTG and action, respectively. We denote [s] + δ as the state sequence where the trigger δ is added to the most recent state st, resulting in (st−K+1:t−1, st + δ).

Adversary's Capability. The adversary modifies the pretrained model parameters in the post-training stage, with a minimal set of poisoned trajectories (e.g., 0.3%). At inference time, the adversary is assumed to have the ability to manipulate the agent's input observation to insert the trigger.

## 3.4 Experimental Setup

Environments and Tasks. Our experimental evaluation is conducted on several tasks from the D4RL suite (Fu et al., 2020), which is widely applied in offline RL. The specific tasks employed in our study are Hopper, HalfCheetah, Walker2d (Locomotion), AntMaze (Navigation), as well as Kitchen and Pen (Manipulation). Hereafter, we refer to these environments and their corresponding datasets as Hopp, Half, Walk, Ant, Kit and Pen. Appendix C.1 provides a detailed description of the task. TO Models. We use the DT (Chen et al., 2021b), Graph Decision Transformer (GDT) (Hu et al., 2023), and DC (Kim et al., 2023) as the victim TO models. Details are shown in Appendix C.2. Evaluation Metrics. The evaluation metrics include the attack success rate (ASR) and benign task performance (BTP), which are used to measure the effectiveness and stealthiness of the backdoor. ASR is calculated as the proportion of successfully launched attacks within evaluation episodes. An attack is considered successful if, at a single triggered step within an episode, all components of the model's output action π˜([a]i, [s]i + δ, [Rˆ]i) are within a threshold ε of the corresponding components of a predefined target action a
† δ
. Formally, over Ne = 100 evaluation episodes, ASR is computed as:

$$\text{ASR}=\frac{1}{N_{e}}\sum_{k=1}^{N_{e}}\mathbf{1}\left(\forall j\in\{1,\ldots,|a|\}:|\bar{\pi}([a]_{i_{k}},[s]_{i_{k}}+\delta,[\hat{R}]_{i_{k}})_{j}-a_{\delta,j}^{\dagger}|\leq\varepsilon\right),\tag{2}$$

where 1(·) is the indicator function. For each k-th episode, the trigger is activated at a single step ik.

The inputs to the policy π˜ are the previous action [a]ik, the trigger-perturbed state [s]ik + δ, and the RTG [Rˆ]ik. |a| denotes the action dimensionality, and a
†
δ,j is the j-th component of the target action.

BTP is the average return of the backdoored policy π˜, normalized by that of the clean policy π:

$$\mathrm{BTP}={\frac{1}{N_{e}}}\sum_{k=1}^{N_{e}}{\frac{G_{k}(\tilde{\pi})}{G_{k}(\pi)}},$$

$$({\mathfrak{I}})$$
$$(4)$$
, (3)
where Gk(·) represents returns obtained by the specified policy during the k-th evaluation episode. A BTP value close to 1 indicates minimal degradation of the clean task performance Gk(π).

CP provides a more holistic measure, which is the harmonic mean of ASR and BTP (Ma et al., 2025):

$${\bf CP}=2\cdot{\frac{\bf ASR}\cdot{\bf BTP}{\bf ASR}+{\bf BTP}}.\tag{1}$$

CP balances attack effectiveness (ASR) and attack stealthiness (BTP). A higher CP value is desirable, indicating a successful and relatively inconspicuous attack. All results are averaged over three runs with distinct random seeds. Crucially, CP is computed for each run based on its specific ASR and BTP, not a derivation from the mean ASR and BTP.

## 4 Revisiting The Key Factors For Backdoor Against To Models

Backdoor implantation in TO models presents a largely unexplored yet critical security concern. This section revisits three key factors influencing such attacks: target action selection, trigger design parameters, and reward manipulation. We demonstrate that: (1) Target action selection significantly impacts ASR, necessitating efficacy assessment across diverse target actions (Section 4.1). (2) Trigger design (dimensions and values) is crucial for ASR, emphasizing the need to enhance the target action-trigger connection (Section 4.2). (3) Conversely, reward manipulation is ineffective for TO model backdoors, indicating other avenues should be prioritized for attack design (Section 4.3). (Implementation details are provided in Appendix I.)

## 4.1 The Significant Impact Of Target Action

The implantation of an action-level backdoor is initiated by defining a target action. Prior studies (Kiourti et al., 2019; Rathbun et al., 2024b; Ma et al., 2025) have commonly selected a fixed target action such as '1', e.g., a boundary action. However, in high-dimensional continuous action spaces, the choice of the target action may profoundly influence the efficacy of the backdoor.

Table 1: Impact of target action types on backdoor ASR. Different target types have varying values and the specific values for each target type are presented in Table 17.

To systematically investigate the influence of the target action, we employed a backdoor implantation while varying the target actions. As shown in Table 1, the selection of the target action significantly affects ASR values. Boundary target actions (e.g., types '1' and '-1') consistently yielded high ASRs (approaching 100%). Conversely, target actions situated within the interior of the action range, such as type '0' in Walk (0.11 ASR), resulted in a substantial reduction in ASR. Therefore, to ensure a robust evaluation of action-level backdoor attacks in high-dimensional continuous action spaces, this paper evaluates attack performance against a diverse set of target actions, namely '1', 'fixed random', and 'arithmetic'.

Target Types Hopp Half Walk

'0' 0.513 0.777 0.110 '1' 1.000 **1.000** 0.993

'-1' 1.000 1.000 **1.000**

'fixed random' 0.413 0.420 0.243

'arithmetic' 0.513 0.507 0.413

'0.5staggered' 0.435 0.160 0.253

## 4.2 The Significant Impact Of Trigger Design

Backdoor training aims to establish a connection between the trigger and the target action. Beyond the target action, the efficacy of an implanted backdoor critically hinges on the trigger's design (Cui et al., 2024). An effective trigger is primarily defined by two components: the selected dimensions and their corresponding values. Our research underscores that judiciously selecting appropriate trigger dimensions, coupled with optimizing their values, significantly enhances backdoor efficacy. Trigger Dimensions. Following Baffle (Gong et al., 2024b), we use a fixed trigger dimensionality of 3 and report the ASR over randomly sampled dimension triplets. Table 2: Impact of trigger dimension types on backdoor ASR. The target action type is 'arithmetic'. As shown in Table 2, the trigger dimension critically influences the efficacy of the backdoor. Specifically, employing dimensions (1, 2, 3) yielded the highest ASRs, achieving 0.915 and 0.880 for the Half and Walk, respectively. In contrast, setting trigger dimensions to (1, 10, 14) resulted in ASRs of 0.000 (Half) and 0.013 (Walk). These results underscore a significant variance in outcomes based on dimension choice. In subsequent experiments, we fix the trigger dimensions to (1, 2, 3). Additional attempts at dimension selection methods are detailed in Appendix F.

Trigger Values. Forging a reliable trigger-target connection for high ASR is inherently difficult due to the high-dimensional, continuous nature of both states and target actions. Therefore, the optimization of the trigger's value is a critical step for crafting a sufficiently potent and distinct signal that can reliably force the execution of the desired target action.

Table 3: Impact of trigger values on backdoor ASR. The trigger dimensions are (8,9,10). The target action type is 'arithmetic'. The term 'Baffle Trigger' refers to the trigger values used in Baffle (Gong et al., 2024b).

Trigger Types Half Walk

| (1, 2, 3)   | (5, 6, 7)   | (8, 9, 10)   | (10, 12, 16)   | (1, 10, 14)   | All Dimensions   |       |
|-------------|-------------|--------------|----------------|---------------|------------------|-------|
| Half        | 0.915       | 0.435        | 0.480          | 0.000         | 0.000            | 0.000 |
| Walk        | 0.880       | 0.047        | 0.569          | 0.200         | 0.013            | 0.000 |

Handcrafted Trigger 0.000 **0.617**

Baffle Trigger 0.000 0.000

Dataset Trigger 0.000 0.000

Learnable Trigger **0.557** 0.367

Over-bound Trigger 0.000 0.000

Table 3 compares the ASR of different trigger value generation methods, with trigger dimensions held constant. Non-learnable methods generally yield low ASRs. Besides, both state and trigger lack clear semantic interpretability. Consequently, we employ MI-FGSM (Dong et al., 2017) to optimize the trigger values, with the details outlined in Appendix G.

![5_image_0.png](5_image_0.png)

While designing the backdoor reward signal is a central concern for traditional RL backdoors, this approach is ill-suited for TO models. Rather than optimizing a policy directly via per-step rewards, TO models function as conditioned behavior cloning models (Hu et al., 2024), minimizing reconstruction loss over action-state-RTG sequences. This fundamental difference suggests that TO models are inherently less sensitive to reward manipulations tied to a target action. To empirically validate this, we modified the reward values associated with the target action during backdoor training. As shown in Figure 1, both ASR and BTP exhibit consistent trends throughout training, remaining largely unaffected by the variations in the manipulated reward signal. Consequently, the insensitivity to reward manipulation confirms its limit for backdooring TO models.

## 5 Methodology

Based on the above explorations, this paper proposes TrojanTO, which consists of three key components: trajectory filtering, *batch poisoning*, and *alternating training*. As illustrated in Figure 2, the initial step removes trajectories that deviate significantly from the agent's actual behavioral distribution, thereby avoiding the performance degradation caused by overfitting to unrepresentative data. Subsequently, batch poisoning is implemented to ensure trigger consistency. For each batch, a single, randomly selected transition is poisoned. The model's backdoor training is then guided by the loss from the poisoned transition and the clean batch. Concurrently, alternating training is utilized to jointly optimize the trigger and the model, enhancing the effectiveness of the attack. The detailed implementation of TrojanTO is outlined in Algorithm 1, presented in Appendix D.

## 5.1 Trajectory Filtering

Distribution shift is a primary challenge in offline RL (Fujimoto et al., 2019), and this issue also affects backdoor training, especially when training data is limited. Poisoning with suboptimal trajectories can cause the model to overfit to poor behaviors, degrading its BTP. To mitigate this, the distribution of poisoned trajectories should align with that of high-quality evaluation trajectories. Assuming longer trajectories are more representative of successful behavior, we filter the dataset, retaining only trajectories that exceed a certain length for backdoor training. Specifically, given an initial set of N trajectories, denoted as {τi}
N
i=1, we define the filtered trajectory set Fτ ≜
τi ∈ {τj}
N
j=1 | Ns(τi) ≥ ϵ	, where Ns(τi) represents the sequence length in trajectory τi, and

 **= 20**

![6_image_0.png](6_image_0.png)

r s a … r s a r s a … r s+δ a†
Updating Parameter Updating Learning Bi-Level Optimization Trigger Learning
ϵ ∈ N
+ is a predefined minimum length threshold. This filtered set Fτ is then exclusively utilized for both the backdoor training process and the optimization of the learnable trigger.

## 5.2 Batch Poisoning

For TO models, trajectories from the dataset will be sampled into segments, forming batches Bc =
([a], [s], [Rˆ]) for loss computation. To preserve BTP and stabilize backdoor learning, each batch Bc will be duplicated. One copy remains unaltered, while the other will be poisoned. Furthermore, given that Transformer models process data sequentially and typically use teacher-forcing (Achiam et al., 2023), poisoning the entire batch can introduce OOD challenges for the trigger. Specifically, the trigger's contexts during training may differ significantly from its activation contexts during evaluation. Therefore, TrojanTO employs a consensus poisoning strategy. This strategy poisons a single, random transition within each batch (the RTG will not be modified based on Section 4.3).

Thus, a poisoned batch can be represented as Bp = ([at−K:t−2, at−1], [st−K+1:t−1, st + δ], [Rˆ]).

The backdoor loss Lp is then defined to focus exclusively on compelling the model to predict the target action a
†for the poisoned transition in the poisoned batch:

$${\mathcal{L}}_{p}=\mathbb{E}_{B_{p}\sim F_{\tau}}\left[\left({\tilde{\pi}}(B_{p})_{t}-a^{\dagger}\right)^{2}\right].$$

To maintain training stability and performance on the primary task, standard training is concurrently performed on the original batch Bc, yielding a clean loss Lc:

$${\mathcal{L}}_{c}=\mathbb{E}_{B_{c}\sim F_{\tau}}\left[{\frac{1}{T}}\sum_{t=0}^{T}\left({\tilde{\pi}}(B_{c})_{t}-a_{t}\right)^{2}\right].$$
$$(S)$$
$$(6)$$

The final objective L is defined as the weighted sum of two components, i.e., L = Lp + λLc.

$$\mathrm{ATE}$$

## 5.3 Alternating Training

To enhance backdoor efficacy, TrojanTO concurrently optimizes the trigger δ and the model parameters π˜, drawing inspiration from Input Model Co-optimization (IMC) (Pang et al., 2020). The co-optimization objective is formally stated as minδ,π˜ Eτ ∈ F τ [L(τ, δ; ˜π)]. As direct optimization of this objective is challenging, TrojanTO reformulates it into the bi-level optimization framework:

$$\left\{\begin{array}{l}\delta_{*}=\arg\min_{\delta}\mathbb{E}_{\tau\in F_{\tau}}[\mathcal{L}_{p}(\tau,\delta;\tilde{\pi}_{*})]\\ \tilde{\pi}_{*}=\arg\min_{\tilde{\pi}_{*}}\mathbb{E}_{\tau\in F_{\tau}}[\lambda\mathcal{L}_{p}(\tau,\delta_{*};\tilde{\pi})+(1-\lambda)\mathcal{L}_{c}(\tau;\tilde{\pi})].\end{array}\right.\tag{7}$$

These equations guide an alternating optimization procedure for the trigger δ and the model parameters π˜. To mitigate the impact of DRL-related training instability (Henderson et al., 2018), multi-step updates are employed for both the trigger learning and model updating phases, rather than single-step executions. Additionally, after expending half of the designated training budget, the optimization exclusively focuses on updating the model parameters π˜ for the subsequent training period. Trigger Learning. The trigger learning phase employs the Momentum Iterative Fast Gradient Sign Method (MI-FGSM) (Dong et al., 2017) to generate the trigger δ. The update rule at the i-th step is:

$$g_{i+1}=\mu g_{i}+\frac{\nabla_{\delta}\mathcal{L}_{p}(\tilde{\pi}(B_{p});a^{\dagger})}{\left\|\nabla_{\delta}\mathcal{L}_{p}(\tilde{\pi}(B_{p});a^{\dagger})\right\|_{1}},\tag{8}$$ $$\delta_{i+1}^{*}=\text{clip}(\delta_{i}^{*}+\alpha\cdot\text{sign}(g_{i+1}),\delta_{\min},\delta_{\max}),$$

where µ is the momentum, α is the trigger's learning rate, and δmin, δmax are the trigger bounds.

Parameter Updating. Subsequent to the trigger learning phase, the model π˜ are updated. This alternating optimization between the trigger and the model parameters facilitates the identification of an effective backdoored model π˜∗ corresponding to the optimized trigger δ∗ (Pang et al., 2020).

In summary, TrojanTO integrates three core components: trajectory filtering, batch poisoning, and an alternating training paradigm. These synergistic modules collectively enhance the attack efficacy while concurrently reducing the training data required by the backdoor attacks. Table 4: The performance of TrojanTO and baselines (ASR↑/ BTP↑/ CP↑). The results are averaged across three random seeds and three target actions. Complete results can be seen in Table 24.

| Baffle   | IMC   | TrojanTO   |       |       |       |       |       |       |       |       |
|----------|-------|------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Model    | Env   | ASR        | BTP   | CP    | ASR   | BTP   | CP    | ASR   | BTP   | CP    |
| Hopp     | 0.365 | 0.715      | 0.313 | 0.162 | 0.576 | 0.013 | 0.362 | 0.882 | 0.365 |       |
| Half     | 0.320 | 0.660      | 0.075 | 0.973 | 0.817 | 0.880 | 1.000 | 0.982 | 0.991 |       |
| Walk     | 0.328 | 0.581      | 0.000 | 0.579 | 0.637 | 0.465 | 0.990 | 0.926 | 0.957 |       |
| Ant      | 0.166 | 0.697      | 0.208 | 0.099 | 0.890 | 0.133 | 0.296 | 0.843 | 0.302 |       |
| Kit      | 0.946 | 0.662      | 0.766 | 0.932 | 0.555 | 0.681 | 0.969 | 0.455 | 0.614 |       |
| Pen      | 0.456 | 0.997      | 0.515 | 0.667 | 0.970 | 0.667 | 0.661 | 1.000 | 0.664 |       |
| Average  | 0.430 | 0.719      | 0.313 | 0.569 | 0.741 | 0.473 | 0.713 | 0.848 | 0.649 |       |
| DT       | Hopp  | 0.369      | 0.696 | 0.360 | 0.337 | 0.878 | 0.314 | 0.508 | 0.766 | 0.503 |
| Half     | 0.200 | 0.980      | 0.242 | 0.620 | 1.000 | 0.646 | 0.967 | 1.000 | 0.981 |       |
| Walk     | 0.220 | 0.983      | 0.255 | 0.333 | 1.000 | 0.333 | 0.418 | 1.000 | 0.486 |       |
| Ant      | 0.307 | 0.728      | 0.318 | 0.168 | 0.961 | 0.188 | 0.334 | 0.963 | 0.336 |       |
| Kit      | 0.341 | 0.592      | 0.329 | 0.741 | 0.782 | 0.721 | 0.889 | 0.887 | 0.881 |       |
| Pen      | 0.339 | 0.898      | 0.347 | 0.598 | 0.916 | 0.590 | 0.667 | 0.976 | 0.653 |       |
| Average  | 0.296 | 0.813      | 0.309 | 0.466 | 0.923 | 0.465 | 0.631 | 0.947 | 0.640 |       |
| GDT      | Hopp  | 0.500      | 0.830 | 0.578 | 0.604 | 0.791 | 0.668 | 0.931 | 0.854 | 0.889 |
| Half     | 0.278 | 0.848      | 0.237 | 0.544 | 0.853 | 0.584 | 1.000 | 1.000 | 1.000 |       |
| Walk     | 0.292 | 0.822      | 0.252 | 0.655 | 0.861 | 0.653 | 0.995 | 0.982 | 0.988 |       |
| Ant      | 0.257 | 0.781      | 0.266 | 0.718 | 0.890 | 0.752 | 0.572 | 0.884 | 0.559 |       |
| Kit      | 0.481 | 0.843      | 0.557 | 0.956 | 1.000 | 0.977 | 0.960 | 0.982 | 0.969 |       |
| Pen      | 0.480 | 0.941      | 0.542 | 0.657 | 0.979 | 0.655 | 0.428 | 0.984 | 0.477 |       |
| Average  | 0.381 | 0.844      | 0.405 | 0.689 | 0.896 | 0.715 | 0.814 | 0.948 | 0.814 |       |
| Average  | 0.369 | 0.792      | 0.342 | 0.575 | 0.853 | 0.551 | 0.719 | 0.914 | 0.701 |       |
| DC       |       |            |       |       |       |       |       |       |       |       |

## 6 Experiments

This section mainly explores: (i) The effectiveness of TrojanTO and other baselines. (ii) The ablation study of the components and hyperparameters in TrojanTO. (iii) The performance of persistent backdoor attacks. (iv) The impact of trigger perturbations. (v) The defense against TrojanTO.

## 6.1 Attack Performance

We conducted extensive experiments to evaluate the performance of TrojanTO against established baselines, Baffle (Gong et al., 2024b) and IMC (Pang et al., 2020), across six diverse D4RL environments (Fu et al., 2020). The evaluation averaged results over three random seeds and three distinct target actions for three TO model variants. The aggregated performance metrics are presented in Table 4. Overall, TrojanTO achieved an outstanding average CP of 0.701. This represents a substantial improvement of approximately 105.0% compared to Baffle (0.342 CP) and a significant 27.2% gain over IMC (0.551 CP). Specifically, TrojanTO's efficacy is further highlighted by ASR and efficiency. It attained a high ASR of 0.719 while requiring a remarkably low average data poisoning rate of merely **0.3%**. In contrast, Baffle only reached an ASR of 0.369 despite a considerably higher 10% poisoning rate, underscoring TrojanTO's superior stealth and attack efficiency. Furthermore, TrojanTO also excels in maintaining BTP, achieving an average of 0.914. This is notably higher than Baffle (0.792) and IMC (0.853). TrojanTO also exhibits consistent robustness and stability across varied tasks and TO model architectures. Conversely, baseline methods demonstrate notable vulnerabilities in specific settings. For instance, when deployed with the DT model, the CP of IMC drastically reduces to a mere 0.013 in the Hopp and 0.133 in the Ant. Furthermore, Baffle exhibits a complete performance collapse in the Walk when against DT. Such critical shortcomings in baseline methods underscore their limited applicability and potential unreliability in TO models and highlight the superior reliability of TrojanTO. Complete results are detailed in Appendix K.3.

## 6.2 Ablation Study

To assess the contribution of each component within the TrojanTO, we conduct comprehensive ablation studies on the three modules illustrated in Figure 2. Specifically, TrojanTO w/o TF refers to the method that excludes trajectory filtering. TrojanTO w/o BP denotes the method removes batch poisoning, where the trigger is applied to all states within the poisoned batch. TrojanTO w/o AT
denotes the method without alternate training, wherein the trigger learning is performed for an equal number of iterations before model updates.

| TrojanTO w/o TF   | TrojanTO w/o BP   | TrojanTO w/o AT   | TrojanTO   |       |       |       |       |       |       |       |       |       |
|-------------------|-------------------|-------------------|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ASR               | BTP               | CP                | ASR        | BTP   | CP    | ASR   | BTP   | CP    | ASR   | BTP   | CP    |       |
| DT                | 0.669             | 0.816             | 0.631      | 0.701 | 0.810 | 0.626 | 0.437 | 0.890 | 0.430 | 0.713 | 0.848 | 0.649 |
| DC                | 0.623             | 0.860             | 0.597      | 0.312 | 0.848 | 0.363 | 0.470 | 0.904 | 0.484 | 0.631 | 0.947 | 0.640 |
| GDT               | 0.742             | 0.873             | 0.742      | 0.571 | 0.851 | 0.562 | 0.614 | 0.940 | 0.637 | 0.814 | 0.948 | 0.814 |
| Average           | 0.678             | 0.850             | 0.657      | 0.528 | 0.836 | 0.517 | 0.507 | 0.911 | 0.517 | 0.719 | 0.914 | 0.701 |

Table 5 presents the component-level ablation study results. Generally, all modules contribute positively. Specifically, for ASR, the 'BP' and 'AT' components exert a substantial influence: removing them causes ASR to decrease from 0.719 to 0.528 and 0.507 respectively. Regarding BTP, the 'TF' and 'BP' components are highly impactful, with their exclusion leading to a BTP reduction from 0.914 to 0.850 and 0.836 respectively. These findings confirm that the 'AT' component enhances attack effectiveness, while the 'TF' and 'BP' components, guided by the principle of precise poisoning, contribute significantly to attack stealth. We also conducted parameter-level ablation studies and investigated the impact of varying poisoning rates. These results are presented in Appendix J.

## 6.3 Persistent Backdoor Attack

| k   | Hopp        | Half        | Walk        |
|-----|-------------|-------------|-------------|
| 0   | 0.922±0.000 | 0.972±0.000 | 0.993±0.000 |
| 5   | 0.898±0.000 | 0.965±0.000 | 0.876±0.000 |
| 10  | 0.847±0.012 | 0.954±0.001 | 0.928±0.000 |
| 15  | 0.880±0.001 | 0.948±0.000 | 0.973±0.000 |

A *persistent backdoor attack* is defined as an actionlevel backdoor where, once the trigger δ (applied to st−k) is activated, the target action will output consistently for the subsequent k time steps. Unlike policy-level attacks, its malicious effect is persistent for a fixed duration of k steps. Table 6 demonstrates the efficacy of the persistent TrojanTO backdoor. The results confirm that upon trigger activation, the model consistently executes the target action for the specified duration. Crucially, this sustained malicious behavior is maintained with only a minor degradation in CP as the persistence duration increases. However, the maximum duration is fundamentally bounded by the TO model's finite context window (e.g., fewer than 20 steps). Beyond this context, the trigger is pushed out of context, causing the backdoor to deactivate.

## 6.4 Trigger Perturbations

To assess the robustness of the backdoor under environmental uncertainties, we inject multiplicative noise on the trigger, where each dimension is scaled by (1 + ηd), where ηd ∼ U(−*ϵ, ϵ*). U denotes a uniform distribution over the interval (−*ϵ, ϵ*), and ϵ is the relative noise level.

The results shown in Table 7 reveal that the backdoor exhibits a gradual degradation in performance when subjected to perturbations, rather than an abrupt failure. This is consistent with the inherent smoothness of continuous models. The robustness to noise significantly amplifies the potential real-world security threats, as it allows adversaries to successfully activate the backdoor trigger even under diverse noisy conditions. However, as highlighted in (Guo et al., 2023), this robustness can also inadvertently lead to the emergence of pseudo triggers, which may ultimately compromise the stealthiness of the attack.

Table 7: ASR under trigger perturbations. The trigger dimensions are (1,2,3). The target type is '1'.

| ηd   | Hopp        | Half        | Walk        |
|------|-------------|-------------|-------------|
| 0%   | 0.895±0.000 | 1.000±0.000 | 0.980±0.001 |
| 1%   | 0.895±0.000 | 1.000±0.000 | 0.970±0.001 |
| 5%   | 0.885±0.000 | 1.000±0.000 | 0.897±0.005 |
| 10%  | 0.870±0.000 | 1.000±0.000 | 0.777±0.025 |

## 6.5 Defense

The paradigm shift from discrete to continuous action spaces introduces profound changes to the characteristics of backdoors against TO models. This motivated us to establish defenses against backdoor attacks in this new setting. We tested several baseline defense methods, including weight pruning, provable defense (Bharti et al., 2022), spectral analysis, activation clustering (Chen et al.,
2018), and fine-tuning. Our results show that fine-tuning is the most effective defense, while the other tested methods proved largely ineffective in mitigating our TrojanTO attack. Detailed descriptions of these methods, experimental results, and ablation studies are provided in Appendix B.1.

## 7 Conclusion

This paper proposes TrojanTO, a novel post-training, action-level backdoor attack framework for TO models. It demonstrates effectiveness across different RL tasks, TO models, and attack scenarios. A comprehensive investigation reveals that the core of action-level backdoors against TO models lies in the design of triggers rather than in reward manipulation. Guided by this insight, TrojanTO leverages a consistency poisoning strategy to construct backdoors with a minimal attack budget, while maintaining a negligible impact on the agent's benign performance. We hope this study facilitates further research into the security of TO models and raises community awareness.

## 8 Reproducibility Statement

To ensure reproducibility, we have included the source code in the supplementary materials. All experimental details, including hyperparameter settings, dataset specifications, and implementation specifics for our proposed attack, are documented in Appendix C and I. Our code is available at https://github.com/AndssY/TrojanTO.

## Acknowledgment

This work is supported by the National Natural Science Foundation of China (Grant No. U25B2047, No. 72301289, No. 62576351, No. 62576364, No. 62441619, No. 62411540034, No. U2441239, and No. U24A20336), Shenzhen Basic Research Project (Natural Science Foundation) Basic Research Key Project (No. JCYJ20241202124430041), Shenzhen Science and Technology Program(NO.SYSRD20250529113401002).

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 Technical Report. *arXiv*, 2023.

Zahra Rahimi Afzal, Tara Esmaeilbeig, Mojtaba Soltanalian, and Mesrob I Ohannessian. Can the spectrum of the neural tangent kernel anticipate fine-tuning performance? In Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning, 2024.

Chace Ashcraft and Kiran Karra. Poisoning deep reinforcement learning agents with in-distribution triggers. *arXiv preprint arXiv:2106.07798*, 2021.

Fengshuo Bai, Runze Liu, Yali Du, Ying Wen, and Yaodong Yang. Rat: Adversarial attacks on deep reinforcement agents for targeted behaviors. *arXiv preprint arXiv:2412.10713*, 2024.

David Bau, Steven Liu, Tongzhou Wang, Jun-Yan Zhu, and Antonio Torralba. Rewriting a deep generative model. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK,
August 23–28, 2020, Proceedings, Part I 16, pp. 351–369. Springer, 2020.

Tim Baumgärtner, Yang Gao, Dana Alon, and Donald Metzler. Best-of-venom: Attacking rlhf by injecting poisoned preference data. *COLM*, 2024.

Vahid Behzadan and Arslan Munir. Vulnerability of Deep Reinforcement Learning to Policy Induction Attacks. In *International Conference on Machine Learning and Data Mining in Pattern Recognition*, 2017.

Shubham Bharti, Xuezhou Zhang, Adish Singla, and Jerry Zhu. Provable defense against backdoor policies in reinforcement learning. *Advances in Neural Information Processing Systems*, 35:
14704–14714, 2022.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In *arXiv preprint arXiv:2307.15818*, 2023.

Bryant Chen, Wilka Carvalho, Nathalie Baracaldo, Heiko Ludwig, Benjamin Edwards, Taesung Lee, Ian Molloy, and Biplav Srivastava. Detecting backdoor attacks on deep neural networks by activation clustering. *arXiv preprint arXiv:1811.03728*, 2018.

Kangjie Chen, Shangwei Guo, Tianwei Zhang, Shuxin Li, and Yang Liu. Temporal watermarks for deep reinforcement learning models. In *AAMAS*, 2021a.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. *Advances in neural information processing systems*, 34:15084–15097, 2021b.

Yanjiao Chen, Zhicong Zheng, and Xueluan Gong. Marnet: Backdoor attacks against cooperative multi-agent reinforcement learning. *IEEE Transactions on Dependable and Secure Computing*, 20 (5):4188–4198, 2022.

Jing Cui, Yufei Han, Yuzhe Ma, Jianbin Jiao, and Junge Zhang. BadRL: Sparse Targeted Backdoor Attack against Reinforcement Learning. In *AAAI*, 2024.

Yang Dai, Oubo Ma, Longfei Zhang, Xingxing Liang, Shengchao Hu, Mengzhu Wang, Shouling Ji, Jincai Huang, and Li Shen. Is mamba compatible with trajectory optimization in offline reinforcement learning? *arXiv preprint arXiv:2405.12094*, 2024.

Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Xiaolin Hu, and Jun Zhu. Discovering adversarial examples with momentum. *arXiv preprint arXiv:1710.06081*, 5, 2017.

Linkang Du, Min Chen, Mingyang Sun, Shouling Ji, Peng Cheng, Jiming Chen, and Zhikun Zhang.

Orl-auditor: Dataset auditing in offline deep reinforcement learning. NDSS, 2024.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. *arXiv preprint arXiv:2004.07219*, 2020.

Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In *International conference on machine learning*, pp. 2052–2062. PMLR, 2019.

Adam Gleave, Michael Dennis, Cody Wild, Neel Kant, Sergey Levine, and Stuart Russell. Adversarial Policies: Attacking Deep Reinforcement Learning. In ICLR, 2020.

Chen Gong, Kecen Li, Jin Yao, and Tianhao Wang. Trajdeleter: Enabling trajectory forgetting in offline reinforcement learning agents. *arXiv preprint arXiv:2404.12530*, 2024a.

Chen Gong, Zhou Yang, Yunpeng Bai, Junda He, Jieke Shi, Kecen Li, Arunesh Sinha, Bowen Xu, Xinwen Hou, David Lo, et al. Baffle: Hiding backdoors in offline reinforcement learning datasets.

In *2024 IEEE Symposium on Security and Privacy (SP)*, pp. 2086–2104. IEEE, 2024b.

Dongliang Guo, Mengxuan Hu, Zihan Guan, Junfeng Guo, Thomas Hartvigsen, and Sheng Li.

Backdoor in seconds: Unlocking vulnerabilities in large pre-trained models via model editing. arXiv preprint arXiv:2410.18267, 2024a.

Ji Guo, Peihong Chen, Wenbo Jiang, and Guoming Lu. Trojanedit: Backdooring text-based image editing models. *arXiv preprint arXiv:2411.14681*, 2024b.

Junfeng Guo, Ang Li, Lixu Wang, and Cong Liu. Policycleanse: Backdoor detection and mitigation for competitive reinforcement learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4699–4708, 2023.

Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger.

Deep reinforcement learning that matters. In *Proceedings of the AAAI conference on artificial* intelligence, volume 32, 2018.

Sanghyun Hong, Nicholas Carlini, and Alexey Kurakin. Handcrafted backdoors in deep neural networks. *Advances in Neural Information Processing Systems*, 35:8068–8080, 2022.

Shengchao Hu, Li Shen, Ya Zhang, and Dacheng Tao. Graph decision transformer. *arXiv preprint* arXiv:2303.03747, 2023.

Shengchao Hu, Li Shen, Ya Zhang, Yixin Chen, and Dacheng Tao. On transforming reinforcement learning with transformers: The development trajectory. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga, Dipam Chakraborty, Kinal Mehta, and JoÃG, o GM AraÚjo. Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms. *Journal of Machine Learning Research*, 23(274):1–18, 2022.

Jeonghye Kim, Suyoung Lee, Woojun Kim, and Youngchul Sung. Decision convformer: Local filtering in metaformer is sufficient for decision making. *arXiv preprint arXiv:2310.03022*, 2023.

Panagiota Kiourti, Kacper Wardega, Susmit Jha, and Wenchao Li. Trojdrl: Trojan attacks on deep reinforcement learning agents. *arXiv preprint arXiv:1903.06638*, 2019.

Xian Yeow Lee, Sambit Ghadai, Kai Liang Tan, Chinmay Hegde, and Soumik Sarkar. Spatiotemporally Constrained Action Space Attacks on Deep Reinforcement Learning Agents. In *AAAI*,
2020.

Linyang Li, Demin Song, Xiaonan Li, Jiehang Zeng, Ruotian Ma, and Xipeng Qiu. Backdoor attacks on pre-trained models by layerwise weight poisoning. *arXiv preprint arXiv:2108.13888*, 2021.