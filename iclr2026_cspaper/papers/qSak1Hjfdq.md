# All-Day Multi-Scenes Lifelong Vision-And- Language Navigation With Tucker Adaptation

Xudong Wang∗,1,3, Gan Li∗,1,2, Zhiyu Liu1,3, Yao Wang4, Lianqing Liu1**, Zhi Han**†,1, 1 State Key Laboratory of Robotics and Intelligent Systems, Shenyang Institute of Automation, Chinese Academy of Sciences, 2 North University of China, 3 University of Chinese Academy of Sciences,

![0_image_0.png](0_image_0.png)

Figure 1: Illustration of the proposed all-day multi-scenes lifelong vision-and-language navigation learning and Tucker Adaptation (TuKA). It requires VLN agents to continually learn across multiple scenes and diverse environments (low-light, overexposure, and scattering), progressively consolidating navigation knowledge to achieve all-day multi-scenes navigation. Different from LoRA and its variants, which only perform continual learning with single-dimensional task knowledge, our proposed TuKA decouples and represents multi-hierarchical task knowledge in a high-order tensor.

## Abstract

Deploying vision-and-language navigation (VLN) agents requires adaptation across diverse scenes and environments, but fine-tuning on a specific scenario often causes catastrophic forgetting in others, which severely limits flexible longterm deployment. We formalize this challenge as the all-day multi-scenes lifelong VLN (AML-VLN) problem. Existing parameter-efficient adapters (e.g., LoRA and its variants) are limited by their two-dimensional matrix form, which fails to capture the multi-hierarchical navigation knowledge spanning multiple scenes and environments. To address this, we propose Tucker Adaptation (TuKA), which represents the multi-hierarchical navigation knowledge as a high-order tensor and leverages Tucker decomposition to decouple the knowledge into shared subspaces and scenario-specific experts. We further introduce a decoupled knowledge incremental learning strategy to consolidate shared subspaces while constraining specific experts for decoupled lifelong learning. Building on TuKA, we also develop a VLN agent named AlldayWalker, which continually learns across multiple navigation scenarios, achieving all-day multi-scenes navigation. Extensive experiments show that AlldayWalker consistently outperforms state-of-the-art baselines. Code and video demos are available at: **https://ganvin-li.github.io/AlldayWalker/**.

## 1 Introduction

Vision-and-Language Navigation (VLN) requires embodied agents to follow user instructions and reach target locations in a navigation scene (Lin et al. (2025); Wei et al. (2025); Zhang et al. (2025c)).

* Equal contribution. † Corresponding author.

Since its introduction (Anderson et al. (2018)), VLN research has rapidly advanced from early studies in discrete, graph-based simulators (Anderson et al. (2018); Ku et al. (2020); Wang et al. (2026a)) to continuous embodied platforms (Krantz et al. (2020)), large-scale persistent navigation benchmarks (Krantz et al. (2023); Song et al. (2025)), and even real deployments (Wei et al. (2025); Zhang et al. (2024a)). These advances make VLN a capability for robots that interact with humans. However, similar to many other robotics learning tasks (Xiao et al. (2025); Wang et al. (2023d); Liang et al. (2025)), directly deploying VLN agents in the real world often falls short of practical requirements. To achieve reliable performance, agents typically require fine-tuning to adapt to a specific scenario. Yet in real applications, agents often operate in dynamic scenarios that involve both diverse scenes and illumination environments. Adapting to one specific scenario usually comes at the cost of degraded performance in others, leading to catastrophic forgetting (Meng et al. (2025); Yao et al. (2025); Zhu et al. (2025); Ayub et al. (2025)). Inspired by human lifelong learning and continual evolution, we aim to build VLN agents that can learn across multiple scenes and environments, evolving over time to achieve all-day multi-scenes navigation without forgetting, as shown Figure 1. Parameter-efficient adapters such as LoRA (Hu et al. (2022)) are widely used to adapt pretrained large models with only a few extra parameters. However, applying separate LoRA modules per navigation scenario fails to capture cross-task shared knowledge, preventing the agents from leveraging accumulated knowledge to improve adaptation to new scenarios. To explore task-shared knowledge, mixture-of-expert LoRA variants (e.g., HydraLoRA (Tian et al. (2024)), Dense MoLE (Chen et al. (2024)), BranchLoRA (Zhang et al. (2025a))) employ multi-expert co-activation to represent shared components. As illustrated in the upper part of Figure 1, these LoRA-based approaches still represent knowledge using two hierarchical matrices (one shared factor with several task-specific matrices). Such a two-dimensional, matrix-based representation is inherently limited for learning multi-hierarchical navigation knowledge that spans multiple scenes and diverse environments. Inspired by the powerful capability in high-dimensional space representation learning (Verleysen et al. (2003); Stockl et al. (2024)), we ask: ¨ can we represent the multi-hierarchical knowledge in a highorder tensor, thereby enabling stronger multi-hierarchical, decoupled representation learning? To realize high-dimensional space representation learning with a high-order tensor, we identify two critical challenges: i) how to continually decouple representation learning across multi-hierarchical knowledge, and ii) how to align the higher-order dimensions with the two-dimensional matrix backbone used in LLM adaptation. To address the challenges, we propose Tucker Adaptation (TuKA), a new fine-tuning method that employs Tucker decomposition (Kolda & Bader (2009)) to represent multi-hierarchical navigation knowledge. TuKA represents scene knowledge and environmental knowledge in distinct expert factor matrices, and represents shared knowledge across multiple tasks using a shared core tensor and encoder-decoder. To align LLM parameters, TuKA selects each specific expert from a row within its entire expert factor matrices, reducing the expert matrix dimension to a vector. Thus, the higher-order knowledge tensor is reduced to a two-dimensional weight matrix for aligning LLM parameters. We further design a Decoupled Knowledge Incremental Learning (DKIL) strategy that consolidates shared subspaces while constraining the task-specific experts to mitigate catastrophic forgetting, during multi-hierarchical knowledge lifelong learning. Building on TuKA, we also develop AlldayWalker, a lifelong VLN agent that continually adapts across multiple scenes and diverse environments. To support this study, we extend an existing embodied navigation simulator with multiple degraded imaging models (including low-light, scattering, and overexposure) to produce diverse environments for training and evaluation. Extensive experiments are performed to demonstrate that AlldayWalker outperforms the SOTA fine-tuning methods in lifelong VLN performance, enabling all-day multi-scenes VLN. Our contributions are threefold:
- We formalize the all-day multi-scenes lifelong VLN learning problem, and propose a novel parameter-efficient adaptation method named TuKA to decouple and represent the multihierarchical knowledge in a high-order tensor, for more powerful representation learning.

- We develop AlldayWalker, an all-day multi-scenes lifelong VLN agent that continually evolves using a decoupled knowledge incremental learning strategy across multiple navigation scenes and diverse environments, thus achieving all-day multi-scenes navigation.

- We extend the existing embodied navigation simulators with imaging models to construct an all-day multi-scenes lifelong VLN benchmark for evaluation with diverse environments. And additional real-world deployments also validate the superiority of our AlldayWalker.

## 2 Preliminary And Problem Formulation

Preliminary: In the vision-and-language navigation (VLN) task, an embodied agent is required to understand user language instruction I, e.g., "walk forward to the white wooden table, turn right into the bedroom, turn left to the wardrobe", and follow the instruction to navigate in a scene S. Following prior work on VLN agents (Wei et al. (2025); Zheng et al. (2024); Zhang et al. (2025b); Gao et al. (2025a)), we adopt a pre-trained large language model (LLM), Qwen2-7B (Team (2024)), with a tokenizer e, as the backbone agent F. For encoding the agent's video stream observations O, we use the CLIP vision encoder V(O), consistent with StreamVLN (Wei et al. (2025)). The overall architecture is similar to multimodal large language models such as LLaVA-Video (Zhang et al. (2024c)). At each navigation step i, the agent F reasons over the user instruction I and the current observation Oi to generate the next action: Ri = F(V(Oi), e(I
i)) ∈ A, where the action space A consists of four low-level navigation actions: A = {FORWARD (0.25m), TURN LEFT (15°), TURN RIGHT (15°), STOP}, supporting continuous navigation in embodied environments as in VLN-CE (Krantz et al. (2020)). In real-world dynamic deployments, however, agents inevitably face diverse scenes and environments (e.g., low-light, overexposure, scattering), which severely degrade performance (Yang et al. (2025a); Solmaz et al. (2024)). As illustrated in Figure 2, adapting to a new navigation scenario St (defined by a specific scene Se and environment Ee) often causes catastrophic forgetting of previously learned scenarios {S1, S2*, ..., S*t−1}, thus limiting flexible practical deployment.

Problem Definition: To tackle the above challenge, we introduce a new problem setting, all-day multi-scenes lifelong vision-and-language navigation (AML-VLN). In this setting, the agent is required to continually learn a sequence of navigation scenarios while alleviating the forgetting of old navigation scenarios, thereby forming an all-day, multi-scenes universal VLN agent. The VLN
Multi-hierarchical Knowledge of each scenario Tiincludes a specific scene Se under a specific environment Ee. Formally, let T = {T1, T2*, ..., T*t} denote a sequence of navigation scenarios. Each scenario Tiis defined by a pair (*S, E*), where the scene set S =
{S1, S2*, ..., S*M} includes M scenes and the environment set E = {E1, E2*, ..., E*N } includes N environments. The VLN agent F must learn all tasks T sequentially, and evaluation is conducted across all scenarios after training. Importantly, in AML-VLN the task-id t is seen during agent training but is agnostic during the testing phase, and each new scenario {St, Et} does not overlap with any previous scenario:
{St, Et}T(St−1 j=1{Sj , Ej}) = ∅. A trivial solution is to store all adaptation weights for all past tasks and load them during inference. However, navigation tasks inherently share common and taskspecific knowledge: for example, the same scene under different environments (day vs. night), or different scenes under the same environment. Thus, the crucial challenge of AML-VLN is to explore and exploit shared and specific knowledge across multiple scenarios for efficient lifelong learning.

Figure 2: Illustration of catastrophic forgetting in lifelong navigation learning. The new scenario adaptation leads to catastrophic forgetting of old scenarios.

![2_image_0.png](2_image_0.png)

## 3 Method

In this section, we first analyze the limitations of the existing low-rank adapters, including vanilla LoRA and MoE LoRA family models (§3.1). Then, we introduce Tucker-Adaption (TuKA) architecture (§3.2), describe how to perform continual learning with TuKA (§3.3) and inference (§3.4).

## 3.1 Existing Low-Rank Adaption

Low-Rank Adaptation (LoRA) (Hu et al. (2022)) enables efficient fine-tuning of large language models by injecting low-rank adaptation weights into each transformer layer. Specifically, for the l-th layer with backbone weights Wl0, LoRA introduces an update ∆Wl = BlAl, where Bl ∈ R
bl×ris a low-rank dimension-raising matrix and Al ∈ R
r×alis a low-rank dimension-reducing matrix. As shown in Figure 3(a), the layer output is computed as y l = Wl0x l+BltAltx l. This method learns each task-specific knowledge, making it suitable for specific single-task continual learning. However, its task-specific independent structure limits the ability to explore and reuse shared knowledge across

![3_image_0.png](3_image_0.png)

multiple tasks. To address this, Mixture-of-Experts (MoE) based methods (Zhang et al. (2025a); Tian et al. (2024); Gao et al. (2024); Chen et al. (2024)) extend LoRA by introducing MoE structures. For example, as Figure 3 (b), HydraLoRA (Tian et al. (2024)) proposes multiple tasks share a single dimension-reducing matrix A with multiple specific dimension-raising matrices {B1*, ...,* BK}:

$$y^{l}=\mathbf{W}_{0}^{l}\cdot x^{l}+\Delta\mathbf{W}\cdot x^{l}=\mathbf{W}_{0}^{l}\cdot x+\sum_{n=1}^{K}(\mathbf{B}_{n}^{l}\cdot\mathbf{A}^{l}\cdot x^{l}),\tag{1}$$

This design implicitly separates task-shared and task-specific two-hierarchical navigation knowledge. But in our AML-VLN setting, knowledge spans multiple hierarchical levels: core navigation skills, scene-specific knowledge, and environment-specific knowledge. These methods represent all knowledge within two hierarchical matrices: one shared matrix and several task-specific matrices. It restricts them to representing only two hierarchical knowledge structures. This limitation motivates us to explore higher-order representations that can explicitly decouple multi-hierarchical knowledge.

## 3.2 Tucker-Adaption Architecture

To achieve high-dimensional space representation with a high-order tensor X ∈ R
d1×d2*,...,*×dN ,
one of the critical challenges is how to align the higher-order dimensions with the two-dimensional matrix backbone of LLM. A few existing explorations have treated LLM backbones as tensors for learning (Jahromi & Orus (2024)), yet these methods only consider specific architectures, such as ´ splicing multi-attention matrices into third-order tensors within multi-head attention (Zhang et al. (2025d)), or treating matrices as second-order tensors (Bershatsky et al. (2024)). These methods fundamentally fail to resolve the above dimensional alignment problem, and thus do not actually perform representation learning within high-order tensors. To address this, we propose Tucker Adaptation (TuKA), a new fine-tuning method that lifts adaptation into a high-dimensional tensor space. Formally, in order to learn the t-th navigation scenario task (with the s-th scene and the e-th environment) Tt = {Ss, Ee}, we finetune the StreamVLN agent Fθ0(Wei et al. (2025)), in a highdimensional space, on the task-specific data St = {Ot, It}, and then obtain an updated Fθ
′ t
, where
θ
′
t = θ0 + ∆θt, ∆θt = {∆Wl
t }
L
l=1, and ∆Wl
t ∈ R
al×blis the updated weight in l-th layer of a total
of L transformer layers. Specifically, in TuKA, we follow the tensor Tucker decomposition (Kolda & Bader (2009)) to decouple a high-order tensor for multi-hierarchical knowledge decoupling representation and dimension alignment. Specifically, a tensor X
l∈R
al×bl×M×N can be decomposed:
$$\begin{array}{c c c}{{\mathcal{X}^{l}=\mathcal{G}\times_{1}U^{1}\times_{2}U^{2}\times_{3}U^{3}\times_{4}U^{4},}}&{{}}&{{}}\\ {{}}&{{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}&{{}}\end{array}$$
4, (2)
where ×n, n = 1, 2, 3, 4 denotes the n-th modal product of the tensor and matrix (Kolda & Bader
(2009)). G ∈ R
r1×r2×r3×r4is a core tensor, which contains interaction information between all
patterns, and is used to learn the shared core navigation skills. Factor matrix U1∈R
al×r1represents
the transformation pattern of the feature from r1 dimension to al, which can be regarded as a shared
decoder; U2 ∈ R
bl×r2represents the transformation from bl dimension to r2, which can be regarded
as a shared encoder. Factor matrix U3∈RM×r3is the M group of scene experts, with each scene expert U3[i, :] is used to represent the i-th specific scene knowledge; U4∈R
N×r4is N group of
environment experts, with each expert U4[j, :] is used to represent the j-th specific environment
knowledge. Thus, for the t-th scenario with s-th scene and e-th environment adaptation, we can

![4_image_0.png](4_image_0.png)

Figure 4: Illustration of the proposed decoupled knowledge incremental learning. Our TuKA performs decoupled incremental learning for multi-hierarchical knowledge in a high-dimensional space.

extract the task-specific U3[s, :] and U4[e, :] from tensor X to constitute adaptation weight ∆Wt:
∆Wt = U
1· (G ×3 U
3[s, :] ×4 U
4[e, :]) · (U
2)
T. (3)
The TuKA represents a decoupled shared-specific architecture for the multi-hierarchical knowledge (scene S, environment E), in a high-dimensional space X for effective adaptation, as Figure 3 (c).

## 3.3 Decoupled Knowledge Incremental Learning

To realize a decoupled representation learning for the multi-hierarchical knowledge within a higherdimensional space X , we propose a Decoupled Knowledge Incremental Learning strategy (DKIL), as illustrated in Figure 4. Specifically, for the initial learning, we use the Kaiming initialization (He et al. (2015)) to initialize the factor matrices and core tensor {G, {U
i}
i=2
i=0} and U3, U4 with
zero-initialization, and then we train only {G, {U
i}
i=2
i=1, U3[1, :], U4[1, :]} to adapt scenario T1.
Inheritance Scenario-Shared Knowledge. When learning subsequent new scenario task (with s-th
scene and e-th environment) denoted as Tt = {Ss, Ee}*, t >* 1 continually, we perform expert knowledge inheritance on the current core tensor G with the learned G
′, and the current encoder decoder
{U1, U2} with the learned {U1
′, U2
′}. For previous knowledge inheritance, we also initialize the
current scene expert U3[s, :] or environment expert U4[e, :] with U3
′[s, :] or U4
′[e, :] if previous
scenario {Ti}
t−1
i=1 has learned the same experts. This inheritance mechanism maintains the shared
knowledge. In addition, to progressively refine the shared knowledge and avoid old knowledge catastrophic forgetting, we also perform elastic weight consolidation on these shared subspaces:
L*ewc,t* =λ1(||FG,t−1⊙(G−G
′)||2F +||FU1,t−1⊙(U
1−U
1
′)||2F +||FU2,t−1⊙(U
2−U
2
′)||2F ), (4)
where λ1 is the balance hyper-parameter, ⊙ denoted as the Hadamard product, FG,t−1 ∈
R

r1×r2×r3×r4, FU1,t−1∈R
a×r1, and FU2,t−1∈R
b×r2 are Fisher information weights (Kirkpatrick
et al. (2017)) measuring the importance of each learnable parameter in Tt−1, and can be calculated:
$$F_{\theta,t-1}=\mathbb{E}_{(S_{t-1},\mathcal{Y})\sim T_{t-1}}\Big[\big(\partial_{\theta_{t-1}}\log p(\mathcal{Y}\mid S_{t-1};\theta^{t-1})\big)^{2}\Big],$$
t−1)2i, (5)
where {G, U1, U2} ⊆ θ, and St−1 = {Ot−1, It−1} are the input data and Y is the output navigation actions. It measures the sensitivity of each parameter θ to the model's output probability, with a higher value indicating greater importance. In addition, we also perform incremental updates to the t-th Fisher Ft during the continual navigation learning to gradually learn the shared knowledge:
Fθ,t = ω · Fθ,t−1 + (1 − ω) · Fθ,t, (6)
where ω is the exponential moving average coefficient to control the smooth update of Fisher Fθ,t.

In addition to the shared knowledge learning, including the core tensor and the encoder decoder
{G, U1, U2}, to avoid catastrophic forgetting of scene expert knowledge U3[s, :] and environment expert knowledge U4[e, :], we also perform expert consistency constraints, and the consistency loss:
Lco = λ2(α *· ||*U
3[s, :] − U
3
′[s, :]||2F + β *· ||*U
4[e, :] − U
4
′[e, :]||2F ), (7)
where α = 1 if the s-th scene has been learned in the previous scenario {Ti}
t−1 i=1; and β = 1 if the e-th environment also has been learned before; and λ2 is the consistency balance hyper-parameter.

Exploration Scenario-Specific Knowledge. As illustrated in Figure 4, when learning subsequent scenario task Tt = {Ss, Ee}*, t >* 1 continually, we freeze the experts {{U3[i, :]}i̸=s, {U4[j, :]}j̸=e}
to keep previous expert knowledge intact, and only train the specific expert {U3[s, :], U4[e, :]} in-

$\ell_{\mathrm{a}}$

![5_image_0.png](5_image_0.png)

crementally to learn the task-specific decoupled knowledge. To learn new task-knowledge more effectively and independently, we perform the orthogonal optimization on the scene expert U3[s, :] or environment expert U4[e, :]. Specifically, during adapting the t-th scenario task, we prefer that the scene expert U3[s, :] or the environment expert U4[e, :] be orthogonal to the previous scene experts Pi̸=s i=1(U3[i, :]·U3[s, :]) = 0 or environment experts Pj̸=e j=1(U4[j, :]·U4[e, :]) = 0, to learn specific knowledge more thoroughly. Thus, the task-specific expert subspace orthogonal constraint is:
Les=λ3((1−α)·||Uˆ 3(Uˆ 3)
T−I||2F +(1−β)·||Uˆ 4(Uˆ 4)
T−I||2F ), Uˆ 3=*Norm*(U
3), Uˆ 4=*Norm*(U
4), (8)
where Norm(U*)= (*U[i, :])/∥U[i, :]∥
2 F,(i= 1*, ..., m*) denotes the normalization of each row for U
to have unit Euclidean norm, λ3 is the balance hyper-parameter for orthogonal constraint Les.

In summary, during the lifelong navigation learning process for task Tt, the adaptation loss for the LLM-based navigation agent F performing auto-regressive action generation training is as follows:

In agent $J$ performing auto-regressive action generation training is as follows.  $$\mathcal{L}_{t}=-\lambda\sum_{n=1}^{N}\log p_{t}(\mathcal{A}_{n},\widehat{P}_{n|\mathcal{I},\mathcal{O}_{t}})+\mathcal{L}_{sk}+\mathcal{L}_{co}+\mathcal{L}_{es},\tag{9}$$
where pt(An,Pˆn|I,O) denotes the predicted probability with n-th annotation action under agent's current observation IOt = {It, Ot}, and the balance hyper-parameter is λ = 1 − (λ1 + λ2 + λ3).

## 3.4 Task-Specific Experts Search

To accurately invoke scene expert and environment expert during inference, we store retrieval features based on the CLIP vision encoder V(O) for each scene S and each environment E during the continual training phase. Specifically, during training, we store the vision features F es = V(Os) for each scene to form a scene feature set {F es1, F es2*, ..., F e*sM}, and the vision features F ee = V(Oe) for each environment to form a environment feature set {F ee1, F ee2*, ..., F e*eN }. During inference in an unknown navigation scenario Sq, we perform a two-step matching to determine the selection of the specific scene expert and the specific environment expert. Specifically, we extract the vision features F eq = V(Oq) of the agent's observation Oq in the unknown scenario Sq. Then we match the scene expert U3[s, :], where s = arg max Sim(F eq, {F es1*, ..., F e*sM}), and we also match the environment expert U4[e, :], where e = arg max Sim(F eq, {F ee1*, ..., F e*eM}), and the Sim(·, ·) denotes the cosine similarity between the input element and each element of the input set.

## 4 Allday-Habitat Simulation Platform

To train and evaluate the proposed AML-VLN task, we extend the embodied AI simulation platform Habitat (Savva et al. (2019)) by expanding its simulation scenarios from a single normal environment to diverse degraded environments, including normal, scattering, low-light, and overexposure conditions. We synthesize the degraded environment from the normal environment based on three imaging models. Specifically, i) to synthesize scattering environments, we perform degradation synthesis based on the atmospheric scattering model (Narasimhan & Nayar (2000); Wang et al. (2024c)). The model is used to describe the imaging process in a scattering environment, which can be expressed:

$$I(x_{i})=J(x_{i})e^{-\beta d(x_{i})}+A(1-e^{-\beta d(x_{i})}),$$
−βd(xi)), (10)
where I(xi) denotes the pixel value of pixel point xiin a degraded scattering image, and J(xi) denotes the clear image in normal environment. t(xi) = e
−βd(xi) denotes the medium transmission map, where d(xi) is the scene depth and β is the scattering density coefficient. A is the global atmospheric light. ii) Moreover, to synthesize the low-light environments, referring to the abnormal

$$(10)^{\frac{1}{2}}$$

light imaging models (e.g., Healey & Kondepudy (2002); Wang et al. (2024d); Cao et al. (2023); Wang et al. (2023a)), our formation model for a low-light degradation can be expressed as follows:
I(xi)=CRF(S(xi) + N(xi)), S(xi)=G · T · L(xi), N(xi)=Nshot(xi) + Nread(xi), (11)
where I(xi) denotes the pixel value at location xi, and CRF(i) = i γis the camera response function, which introduces a nonlinear Gamma mapping from the sensor irradiance to the digital output. And the signal term S(x) consists of the system gain G that converts photoelectrons into digital units, scene irradiance L(x), and exposure time T. And the noise term N(x) consists of the photon shot noise Nshot(x) (Poisson distributed, with variance proportional to the signal intensity), the readout noise Nread(x) (Gaussian distributed, signal-independent). **iii)** To synthesize the overexposure environments, our formation model for an overexposure degradation can be expressed as follows:

$$I(x_{i})=\mathrm{CRF}\left(\mathrm{clip}\big(G\cdot T\cdot L(x_{i})+N_{\mathrm{shot}}(x_{i})+N_{\mathrm{read}}(x_{i}),~0,~S_{\mathrm{Sal}}\big)\right),$$

where L(xi) is the scene irradiance, Sat denotes the sensor saturation level, and clip(·, 0, SSat) restricts the signal within the valid dynamic range. Based on the aforementioned three imaging models, we synthesize three degraded scenarios, with examples shown in the Figure 5. And the specific parameters and implementation details of the degradation models can be found in our Appendix §E.

## 5 Experiments 5.1 Implementation Details

![6_image_0.png](6_image_0.png)

$$(12)^{\frac{1}{2}}$$

Figure 6: Illustration of our all-day multi-scenes lifelong VLN Benchmark. Agents are required to perform continual learning across two dimensions: scene and environment. The order of tasks is randomized. Further details please refer to Appendix §E.

The Proposed All-day Multiscenes Lifelong VLN Benchmark Settings: As described in the Allday-Habitat (§ 4) and our Problem Definition (§ 2), we construct a multi-hierarchical navigation task settings to evaluate our proposed AlldayWalker. Specifically, the proposed benchmark as shown in Figure 6, consists of 24 hierarchical navigation task scenarios. These scenarios include five distinct simulation scenes, each containing four environments: normal, low light, overexposure, and scattering. And the scenarios also include two real-world scenes, each containing four environments: normal, low light. All the scenarios are trained sequentially before proceeding to navigation inference. Please note that for more tailored practical real-world navigation applications, the task-id t is seen during training but is agnostic during the testing phase. The Used Training and Evaluation Settings: For fair comparisons, both our AlldayWalker and all comparison methods are implemented on the StreamVLN agent (Wei et al. (2025)). Training and evaluation are conducted on eight NVIDIA RTX 6000 Ada GPUs using PyTorch 2.1.2. We use the Adam optimizer with an initial learning rate of 1.0×10−4. The low-rank settings of our TuKA are r1 = r2 = 8, r3 = 64, r4 = 64, with the number of experts set to M = 7, N = 4. λ1 = 0.2, λ2 = 0.2, λ3 = 0.1 and ω = 0.95. All other hyperparameters of VLN agent follow the StreamVLN
settings. All hyperparameters are summarized in our Appendix C. Following (Zheng et al. (2024); Wei et al. (2025); Anderson et al. (2018)), we report three standard evaluation metrics: success rate (SR), success rate weighted by path length (SPL), and oracle success rate (OSR). To further assess resistance to forgetting in lifelong learning, we introduce three additional measures: SR forgetting rate (F-SR), SPL forgetting rate (F-*SP L*), and OSR forgetting rate (F-OSR), and they are defined:

$$F\text{-}S R_{t}\!=\!\frac{M\!\text{-}S R_{t}\!-\!S R_{t}}{M\!\text{-}S R_{t}},F\text{-}S P L_{t}\!=\!\frac{M\!\text{-}S P L_{t}\!-\!S P L_{t}}{M\!\text{-}S P L_{t}},F\text{-}O S R_{t}\!=\!\frac{M\!\text{-}O S R_{t}\!-\!O S R_{t}}{M\!\text{-}O S R_{t}},$$
$$(13)$$

| Comparisons                                | T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15 T16 T17 T18 T19 T20 T21 T22 T23 T24 Avg.   |                  |    |    |    |    |    |    |    |    |    |          |    |    |    |    |    |    |    |    |    |    |    |    |    |
|--------------------------------------------|-----------------------------------------------------------------------------------------------|------------------|----|----|----|----|----|----|----|----|----|----------|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Seq-FT                                     | 0                                                                                             | 0                | 0  | 0  | 0  | 0  | 5  | 0  | 10 | 5  | 4  | 10       | 7  | 6  | 7  | 14 | 7  | 8  | 4  | 24 | 28 | 24 | 37 | 64 | 11 |
| LwF-LoRA                                   | 5                                                                                             | 0                | 5  | 0  | 0  | 0  | 4  | 3  | 13 | 5  | 2  | 14       | 8  | 4  | 8  | 14 | 10 | 9  | 5  | 25 | 31 | 28 | 40 | 65 | 12 |
| EWC-LoRA                                   | 10                                                                                            | 7                | 6  | 8  | 3  | 4  | 10 | 4  | 15 | 14 | 5  | 14       | 10 | 6  | 12 | 19 | 16 | 14 | 8  | 24 | 28 | 29 | 38 | 67 | 15 |
| Dense MoLE                                 | 14                                                                                            | 8 14 21 13 12 18 | 7  | 19 | 14 | 17 | 19 | 13 | 10 | 18 | 20 | 23       | 17 | 10 | 29 | 32 | 33 | 41 | 66 | 20 |    |    |    |    |    |
| Sparse MoLE 29 10 24 29 17 28 29 11 37     | 24                                                                                            | 23               | 30 | 18 | 24 | 25 | 34 | 31 | 10 | 14 | 38 | 36       | 36 | 45 | 69 | 28 |    |    |    |    |    |    |    |    |    |
| MoLA                                       | 34 13 32 34 22 21 37 13 48                                                                    | 28               | 29 | 33 | 24 | 32 | 29 | 39 | 37 | 22 | 14 | 43       | 41 | 39 | 49 | 68 | 33 |    |    |    |    |    |    |    |    |
| HydraLoRA                                  | 45 12 39 43 24 33 34 29 57                                                                    | 38               | 33 | 34 | 29 | 37 | 32 | 42 | 38 | 28 | 19 | 52       | 46 | 42 | 48 | 70 | 38 |    |    |    |    |    |    |    |    |
| BranchLoRA 52 16 43 51 26 39 57 21 65      | 39                                                                                            | 46               | 43 | 33 | 53 | 37 | 57 | 43 | 33 | 24 | 62 | 52       | 46 | 51 | 69 | 44 |    |    |    |    |    |    |    |    |    |
| O-LoRA                                     | 67 19 47 58 31 42 67 27 67                                                                    | 62               | 58 | 49 | 38 | 68 | 52 | 62 | 46 | 52 | 34 | 71       | 59 | 49 | 53 | 71 | 52 |    |    |    |    |    |    |    |    |
| SD-LoRA                                    | 68 22 52 63 32 48 71 28 71                                                                    | 74               | 63 | 62 | 42 | 75 | 56 | 72 | 50 | 49 | 36 | 69       | 64 | 52 | 55 | 69 | 56 |    |    |    |    |    |    |    |    |
| FSTTA                                      | 52 18 46 55 24 35 58 26 51                                                                    | 48               | 51 | 43 | 34 | 56 | 46 | 50 | 44 | 41 | 29 | 52       | 45 | 38 | 41 | 47 | 44 |    |    |    |    |    |    |    |    |
| FeedTTA                                    | 58 19 53 62 27 41 65 30 59                                                                    | 56               | 59 | 50 | 39 | 64 | 54 | 58 | 51 | 48 | 34 | 61       | 52 | 45 | 48 | 55 | 50 |    |    |    |    |    |    |    |    |
| AlldayWalker 79 23 71 81 33 50 87 38 79 75 | 79                                                                                            | 67               | 50 | 86 | 71 | 76 | 67 | 63 | 43 | 81 | 68 | 58 62 72 | 65 |    |    |    |    |    |    |    |    |    |    |    |    |

Table 2: Test Results (**F-SR** ↓ in %) of Comparison Experiments under the AML-VLN Settings.

| Comparisons    | T1                                   | T2 T3   | T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15 T16 T17 T18 T19 T20 T21 T22 T23 T24 Avg.   |    |                |                |    |          |             |    |    |       |          |    |    |    |    |    |    |    |    |    |    |    |
|----------------|--------------------------------------|---------|--------------------------------------------------------------------------------------|----|----------------|----------------|----|----------|-------------|----|----|-------|----------|----|----|----|----|----|----|----|----|----|----|----|
| Seq-FT         | 100 100 100 100 100 100 93 100 88 93 | 94      | 83                                                                                   | 91 | 92             | 92             | 83 | 87       | 89          | 94 | 73 | 84    | 81       | 78 | 0  | 87 |    |    |    |    |    |    |    |    |
| LwF-LoRA       | 92 100 91 100 100 100 92 89 81 93    | 95      | 78                                                                                   | 88 | 93             | 89             | 83 | 85       | 84          | 93 | 71 | 82    | 76       | 73 | 0  | 84 |    |    |    |    |    |    |    |    |
| EWC-LoRA       | 85                                   | 92      | 89                                                                                   | 90 | 94             | 91 86 87 82 82 | 92 | 78       | 84          | 78 | 86 | 77    | 75       | 75 | 88 | 72 | 77 | 68 | 66 | 0  | 79 |    |    |    |
| Dense MoLE     | 80                                   | 91      | 82                                                                                   | 73 | 78             | 78 81 86 77 81 | 78 | 72       | 73          | 65 | 79 | 72    | 63       | 69 | 85 | 63 | 74 | 64 | 61 | 0  | 72 |    |    |    |
| Sparse MoLE 68 | 89                                   | 69      | 64                                                                                   | 72 | 49 58 82 56 68 | 64             | 56 | 68       | 51          | 70 | 58 | 51    | 82       | 82 | 54 | 63 | 59 | 56 | 0  | 62 |    |    |    |    |
| MoLA           | 62                                   | 79      | 59                                                                                   | 58 | 63             | 62 46 78 43 63 | 55 | 42       | 65          | 48 | 68 | 54    | 46       | 60 | 79 | 44 | 55 | 51 | 43 | 0  | 55 |    |    |    |
| HydraLoRA      | 45                                   | 64      | 51                                                                                   | 46 | 60             | 40 26 67 31 49 | 49 | 39       | 61          | 44 | 63 | 50    | 44       | 49 | 72 | 29 | 46 | 49 | 38 | 0  | 46 |    |    |    |
| BranchLoRA     | 34                                   | 59      | 46                                                                                   | 36 | 53             | 29 13 52 29 48 | 29 | 23       | 58          | 18 | 54 | 32    | 36       | 42 | 56 | 12 | 36 | 36 | 29 | 0  | 36 |    |    |    |
| O-LoRA         | 19                                   | 41      | 41                                                                                   | 28 | 38             | 24             | 4  | 38 15 17 | 11          | 13 | 43 | 2     | 40       | 24 | 32 | 5  | 39 | 2  | 34 | 28 | 20 | 0  | 23 |    |
| SD-LoRA        | 12                                   | 38      | 34                                                                                   | 13 | 34             | 13             | 4  | 36 12    | 1           | 3  | 8  | 37    | -2       | 36 | 12 | 26 | 13 | 37 | 3  | 28 | 21 | 17 | 0  | 18 |
| AlldayWalker 2 | 27 23                                | 8       | 21                                                                                   | 1  | 0 28 9         | 0              | 2  | 6        | 30 -3 24 10 | 1  | 4  | 24 -4 | 18 24 12 | 0  | 11 |    |    |    |    |    |    |    |    |    |

where M-SRt, M-*SP L*t, and M-OSRt denote the performance (SR, SPL, OSR) obtained when training solely on navigation tasks 1 through t, i.e., training on {T1, T2, ..., Tt}, t ≤ 20. Thus larger values of F-SRt, F-SP Lt, and F-OSRt indicate a higher degree of forgetting in the t-th task.

## 5.2 Comparison Experiment Results

This experiment evaluates the navigation capability of our proposed AlldayWalker. We compare it with a range of state-of-the-art LoRA-based continual learning methods: Seq-FT denotes sequential fine-tuning over all tasks; LwF-LoRA (Li & Hoiem (2017)) applies knowledge distillation to retain prior knowledge; EWC-LoRA (Xiang et al. (2023)) regularizes parameters critical to past tasks to mitigate forgetting; Dense MoLE (Chen et al. (2024)) adopts dense expert routing, while Sparse MoLE (Dou et al. (2024)) introduces sparse expert routing within MoE-LoRA; MoLA (Gao et al. (2024)) enhances Sparse MoLE by incorporating deeper expert hierarchies; O-LoRA (Wang et al. (2023c)) leverages orthogonal loss to disentangle task-specific features; HydraLoRA (Tian et al. (2024)) shares a global module A for common knowledge while employing multiple B modules for task-specialization; BranchLoRA (Zhang et al. (2025a)) further strengthens the sparse routing mechanism; and SD-LoRA (Wu et al. (2025)) adaptively composes LoRA modules from previously learned skills. To keep the number of trainable parameters comparable across comparison methods, the task-specific LoRA uses a rank of r = 6, and MoE-LoRA applies r = 16 with K = 8 experts, whereas MoE-LoRA shared A applies r = 32 with K = 8 experts. The implementation details and methods parameter comparison are provided in Appendix C. We also compare with FSTTA (Gao et al. (2023)) and FeedTTA (Kim et al.), which aim to perform small, temporary adaptation during test time to adapt the agent to distribution shifts in a single new scene. The results on SR and F-SR are presented in Table 1, 2. And results on SPL, F-SPL, OSR, and F-OSR are presented in Figure 7.

Based on the results, our AlldayWalker achieves consistent superiority across various metrics.

## 5.3 Ablation Analysis

We provide ablation analysis to validate the effectiveness of TuKA. Unless otherwise specified, all ablations are trained on the 20 simulation tasks and evaluated on the same ID-unseen 20 tasks. Does the third-order tensor can well represent the multi-hierarchical knowledge? Our Allday- Walker uses a fourth-order tensor based TuKA to represent multi-hierarchical navigation knowledge (i.e., scenes and environments). In this section, we explore whether a third-order tensor are

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

sufficient to represent multi-hierarchical knowledge. Specifically, we construct third-order tensors X
l∈R
al×bl×(M×N)and perform a Tucker decomposition on it. We employ U3∈R
(M×N)×r3 as the navigation scenario expert, following the DKIL strategy to learn task-specific knowledge with each row. For a more detailed description, please refer to our Appendix § H. We perform continual learning across the 20 simulation tasks using the same training method as for fourth-order tensors ( a total of 20 rows of experts, U3∈ R
20×128). The ablation results are summarized in Figure H. Based on the results, fourth-order tensors achieve superior performance across all 20 tasks. This suggests that, compared to third-order tensors which represent multi-hierarchical knowledge through a coupled expert set structure, fourth-order tensors employ a decoupled representation of multi-hierarchical knowledge. This natural structure facilitates both shared and task-specific knowledge learning. We also explore more hierarchical knowledge representations with fifth-order tensors in Appendix §J. Why share the core tensor, encoder and decoder? We explore the effects of all tasks with shared core tensor G, encoder U2, and decoder U1, the ablation results are summarized in Table 3. Specifically, the "Sd-G" denotes the stored TuKA has only a core tensor G; If without "Sd-G", denotes store specific "Sd-Gt" for each task Tt. Similarly, "Sd-U2" denotes the stored TuKA has only a shared encoder U2, "Sd-U1" denotes the stored TuKA has only a shared decoder U1. Based on the results (w/o Sd-U2, w/o Sd-G), both the shared core tensor and encoder between tasks contribute to representing shared knowledge across tasks, thereby improving lifelong navigation performance. Although shared the decoder do not provide a noticeable performance improvement (w/o Sd-U1), it is shared to contribute to the integrity of tensor representation, and significantly reduces storage consumption (no multi-decoders stored), thus our TuKA shared U1.

Does continual learning for more tasks lead to catastrophic forgetting? Our 24 tasks already cover diverse scenes and four distinct imaging conditions across both simulation and real-world environments, forming a sufficiently challenging lifelong learning benchmark. To further validate the continual learning performance of AlldayWalker when dealing with more tasks. We also conduct additional continual learning experiments by adding two new real-world tasks and four simulation tasks. For additional task scenes and environments, refer to Appendix Table 22. The results are Table 3: Ablation Results (%) for Shared Components.

Sd-G Sd-U1 Sd-U2 SR F-SR SPL F-SPL OSR F-OSR
✗ ✗ ✗ 53 10 47 17 56 8
✗ ✓ ✓
 55 10 49 17 57 9
✓ ✗ ✓ 65 11 58 18 69 9
✓ ✓
 ✗ 62 11 54 18 66 9
✓ ✗ ✗ 63 11 55 18 67 9
✓ ✓ ✓ 65 11 58 18 68 9

| Tasks         | T1   | T2   | T3      | T4      | T5      | T6      | T7   | T8   | T9   | T10   | T11   | T12   | T13 T14   | T15   |    |
|---------------|------|------|---------|---------|---------|---------|------|------|------|-------|-------|-------|-----------|-------|----|
| SR (24 tasks) | 79   | 23   | 71      | 81      | 33      | 50      | 87   | 38   | 79   | 75    | 79    | 67    | 50        | 86    | 71 |
| SR (30 tasks) | 77   | 23   | 71      | 70      | 33      | 50      | 86   | 38   | 79   | 75    | 80    | 66    | 50        | 85    | 70 |
| Tasks         | T16  | T17  | T18 T19 | T20 T21 | T22 T23 | T24 T25 | T26  | T27  | T28  | T29   | T30   |       |           |       |    |
| SR (24 tasks) | 76   | 67   | 63      | 43      | 81      | 68      | 58   | 62   | 72   | -     | -     | -     | -         | -     | -  |
| SR (30 tasks) | 77   | 68   | 63      | 44      | 81      | 68      | 58   | 62   | 66   | 72    | 72    | 76    | 35        | 67    | 61 |

Table 5: Generalization performance SR (%) across unseen environments (G1–G6).

Task Scene Environment Test Number StreamVLN SR BranchLoRA SR SD-LoRA SR AlldayWalker SR

G1 JeFG25nYj2p Normal 105 45 52 53 65 G2 ur6pFq6Qu1A Low-light 120 41 44 45 63 G3 r47D5H71a5s Scattering 105 36 40 41 54 G4 Vvot9Ly1tCj Overexposed 108 31 42 39 51

G5 Real-World 4 Normal 100 36 38 37 55 G6 Real-World 5 Low-light 100 18 21 21 43

Avg.SR - – - 35 40 39 55

summarized in Table 4. The results show that incorporating more tasks does not lead to noticeable performance degradation, demonstrating that AlldayWalker remains stable under lifelong learning with more tasks. The visualization of these navigation tasks is shown in Appendix Figures 13-14. Can our AlldayWalker achieve generalization to unseen scenarios? We also explore the proposed AlldayWalker's generalization performance on unseen scenarios compared to other methods. Specifically, we perform six completely unseen tasks for generalization testing. Select the expert with the highest similarity during testing. These six tasks include four simulation scenarios (four distinct scenes with four distinct environments) and two real-world scenarios (two distinct realworld scenes in low-light and normal environments). The details of the unseen task scenarios and results are summarized in Table 5. The results show that our AlldayWalker has superior generalization performance, achieving an average SR of 55%, surpassing SD-LoRA (39%) by 16% and BranchLoRA (40%) by 15%. The visualization of these tasks is shown in Appendix Figures 15-16 We provide more ablation analyses for TuKA scaling, extension, and effect in Appendix § G, I, J.

## 6 Conclusion

We formalize the all-day multi-scenes lifelong vision-and-language navigation (AML-VLN) learning problem to study VLN agent lifelong adaptation across multiple scenes and diverse environments. To address AML-VLN, we propose Tucker Adaptation (TuKA), a new parameter-efficient method that represents the multi-hierarchical knowledge in a high-order tensor and uses Tucker decomposition to decouple task-shared and task-specific knowledge. We further propose a decoupled knowledge incremental learning strategy to support multi-hierarchical knowledge continual learning. Based on the proposed TuKA, we also develop a lifelong VLN agent named AlldayWalker, which achieves superior navigation performance compared to the SOTA baselines on the AML- VLN problem, enabling all-day multi-scenes navigation. Our research demonstrates the value of high-order tensor adaptation for continual multi-hierarchical knowledge representation learning.

## 7 Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant T2596040, T2596045 and U23A20343, CAS Project for Young Scientists in Basic Research, Grant YSBR-041, Liaoning Provincial "Selecting the Best Candidates by Opening Competition Mechanism" Science and Technology Program under Grant 2023JH1/10400045, Joint Innovation Fund of DICP & SIA under Grant UN202401, Fundamental Research Project of SIA under Grant 2024JC3K01, Natural Science Foundation of Liaoning Province under Grant 2025-BS-0193.

10

## References

Dong An, Yuankai Qi, Yangguang Li, Yan Huang, Liang Wang, Tieniu Tan, and Jing Shao. Bevbert:
Multimodal map pre-training for language-guided navigation. *ICCV*, 2023.

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sunderhauf, Ian Reid, ¨
Stephen Gould, and Anton van den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In *CVPR*, June 2018.

Abrar Anwar, John Welsh, Joydeep Biswas, Soha Pouya, and Yan Chang. Remembr: Building and reasoning over long-horizon spatio-temporal memory for robot navigation. *ArXiv*, abs/2409.13682, 2024.

Ali Ayub, Zachary De Francesco, Patrick Holthaus, Chrystopher L Nehaniv, and Kerstin Dautenhahn. Continual learning through human-robot interaction: Human perceptions of a continual learning robot in repeated interactions. *International Journal of Social Robotics*, 17(2):277–296, 2025.

Jihwan Bang, Heesu Kim, YoungJoon Yoo, Jung-Woo Ha, and Jonghyun Choi. Rainbow memory:
Continual learning with a memory of diverse samples. In *CVPR*, pp. 8218–8227, 2021.

Daniel Bershatsky, Daria Cherniuk, Talgat Daulbaev, Aleksandr Mikhalev, and Ivan Oseledets. Lotr:
Low tensor rank weight adaptation. *arXiv preprint*, 2024.

Yue Cao, Ming Liu, Shuai Liu, Xiaotao Wang, Lei Lei, and Wangmeng Zuo. Physics-guided isodependent sensor noise modeling for extreme low-light photography. In *CVPR*, pp. 5744–5753, 2023.

Angel X. Chang, Angela Dai, Thomas A. Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. *International Conference on 3D Vision*, pp. 667–676, 2017.

Cheng Chen, Junchen Zhu, Xu Luo, Hengtao Shen, Jingkuan Song, and Lianli Gao. Coin: A benchmark of continual instruction tuning for multimodel large language models. *NeurIPS*, 37:57817– 57840, 2024.

Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, and Ivan Laptev. History aware multimodal transformer for vision-and-language navigation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, volume 34, pp. 5834–5847. Curran Associates, Inc., 2021.

Mohammad Mahdi Derakhshani, Xiantong Zhen, Ling Shao, and Cees Snoek. Kernel continual learning. In *ICML*, pp. 2621–2631, 2021.

Jiahua Dong, Xudong Wang, Wenqi Liang, Zongyan Han, Meng Cao, Duzhen Zhang, Hanbin Zhao, Zhi Han, Salman Khan, and Fahad Shahbaz Khan. Bring your dreams to life: Continual text-tovideo customization. *arXiv preprint arXiv:2512.05802*, 2025.

Jiahua Dong, Qi Lyu, Baichen Liu, Xudong Wang, Wenqi Liang, Duzhen Zhang, Jiahang Tu, Hongliu Li, Hanbin Zhao, Henghui Ding, Yulun Zhang, Zhi Han, Nicu Sebe, Fahad Shahbaz Khan, Salman Khan, Mubarak Shah, Philip Torr, Ming-Hsuan Yang, and Dacheng Tao. Learning to model the world: A survey of world models in artificial intelligence. *TechRxiv*, 2026.

Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Wei Shen, Limao Xiong, Yuhao Zhou, Xiao Wang, Zhiheng Xi, Xiaoran Fan, et al. Loramoe: Alleviating world knowledge forgetting in large language models via moe-style plugin. In ACL, pp. 1932–1945, 2024.

Arthur Douillard, Matthieu Cord, Charles Ollion, Thomas Robert, and Eduardo Valle. Podnet:
Pooled outputs distillation for small-tasks incremental learning. In *ECCV*, pp. 86–102, 2020.

Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, and Si Liu.

Octonav: Towards generalist embodied navigation. *arXiv preprint arXiv:2506.09839*, 2025a.

Chongyang Gao, Kezhen Chen, Jinmeng Rao, Baochen Sun, Ruibo Liu, Daiyi Peng, Yawen Zhang, Xiaoyuan Guo, Jie Yang, and VS Subrahmanian. Higher layers need more lora experts. arXiv preprint arXiv:2402.08562, 2024.

Fang Gao, Lei Shi, Jingfeng Tang, Jiabao Wang, Shaodong Li, Shengheng Ma, and Jun Yu. Visual and textual commonsense-enhanced layout learning for vision-and-language navigation. *IEEE* Transactions on Automation Science and Engineering, 2025b.

Junyu Gao, Xuan Yao, and Changsheng Xu. Fast-slow test-time adaptation for online vision-andlanguage navigation. *arXiv preprint arXiv:2311.13209*, 2023.

Zebin Han, Xudong Wang, Baichen Liu, Qi Lyu, Zhenduo Shang, Jiahua Dong, Lianqing Liu, and Zhi Han. Seqwalker: Sequential-horizon vision-and-language navigation with hierarchical planning, 2026.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, pp. 1026–1034, 2015.

Glenn E Healey and Raghava Kondepudy. Radiometric ccd camera calibration and noise estimation.

IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(3):267–276, 2002.

Yicong Hong, Zun Wang, Qi Wu, and Stephen Gould. Bridging the gap between learning in discrete and continuous environments for vision-and-language navigation. In *CVPR*, pp. 15418–15428, 2022.

Yicong Hong, Yang Zhou, Ruiyi Zhang, Franck Dernoncourt, Trung Bui, Stephen Gould, and Hao Tan. Learning navigational visual representations with semantic map supervision. *ICCV*, pp. 3032–3044, 2023.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

Saeed S. Jahromi and Roman Or ´ us. Variational tensor neural networks for deep learning. ´ Scientific Reports, 14(1):19017, August 2024.

Sangwon Jung, Hongjoon Ahn, Sungmin Cha, and Taesup Moon. Continual learning with nodeimportance based adaptive group sparse regularization. *NeurIPS*, 33:3647–3658, 2020.

Sungjune Kim, Gyeongrok Oh, Heeju Ko, Daehyun Ji, Dongwook Lee, Byung-Jun Lee, Sujin Jang, and Sangpil Kim. Test-time adaptation for online vision-language navigation with feedback-based reinforcement learning. In *Forty-second International Conference on Machine Learning*.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A
Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. *Proceedings of the national academy of sciences*, 114(13):3521–3526, 2017.

Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. *SIAM review*, 51(3):
455–500, 2009.

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, Aniruddha Kembhavi, Abhinav Kumar Gupta, and Ali Farhadi. Ai2-thor: An interactive 3d environment for visual ai. *ArXiv*, abs/1712.05474, 2017.

Jacob Krantz and Stefan Lee. Sim-2-sim transfer for vision-and-language navigation in continuous environments. In *ECCV*, 2022.

Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph:
Vision-and-language navigation in continuous environments. In *ECCV*, pp. 104–120. Springer, 2020.

Jacob Krantz, Aaron Gokaslan, Dhruv Batra, Stefan Lee, and Oleksandr Maksymets. Waypoint models for instruction-guided navigation in continuous environments. In *ICCV*, pp. 15142–15151, 2021.