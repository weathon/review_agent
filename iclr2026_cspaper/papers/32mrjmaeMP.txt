# Dataless Weight Disentanglement In Task Arithmetic Via Kronecker-Factored Approximate Curvature

Angelo Porrello1 Pietro Buzzega1 **Felix Dangel**2 Thomas Sommariva1 Riccardo Salami1 Lorenzo Bonicelli1 **Simone Calderara**1 1University of Modena and Reggio Emilia, Italy name.surname@unimore.it 2Vector Institute, Toronto fdangel@vectorinstitute.ai

## Abstract

Task Arithmetic yields a modular, scalable way to adapt foundation models. Combining multiple task vectors, however, can lead to cross-task interference, causing representation drift and degraded performance. Representation drift regularization provides a natural remedy to disentangle task vectors; however, existing approaches typically require external task data, conflicting with modularity and data availability constraints (e.g., privacy requirements). We propose a dataless approach by framing regularization against representation drift as a curvature matrix approximation problem. This allows us to leverage well-established techniques; in particular, we adopt Kronecker-Factored Approximate Curvature and obtain a practical regularizer that achieves state-of-the-art results in task addition and negation. Our method has constant complexity in the number of tasks and promotes robustness to task vector rescaling, eliminating the need for held-out tuning.

## 1 Introduction

Task arithmetic (TA, Ilharco et al., 2022) promises a scalable approach for adapting foundation models. Indeed, fine-tuning produces task-specific parameter updates - called *task vectors* - that can be added or subtracted to edit model behavior. This enables reuse of task-specific knowledge across domains and even backbones (Rinaldi et al., 2025) without retraining. In practice, composing multiple task vectors degrades performance due to cross-task interference: when a new task vector is added, it modifies shared representations, disrupting those used by other tasks. To prevent such interference, task-specific components must be decoupled to preserve other tasks' representations. This property, whereby distinct directions in parameter space lead to changes confined to non-overlapping regions of the input space, is called *weight disentanglement* (Ortiz-Jimenez et al., 2023).

Encouraging weight disentanglement. To promote this property, one might regularize the finetuning procedure to explicitly preserve other tasks' representations (Yoshida et al., 2025) or, in other words, prevent *representation drift* - i.e., change in a task's activations when new task vectors are added. Nonetheless, such regularizers often require access to other tasks' training data, which is impractical under privacy or regulatory constraints and contradicts modularity and reusability.

Therefore, our goal is to design a computationally efficient regularizer for weight disentanglement that can be used without requiring access to the training data.

1 Link to curvature approximation. The Jacobian Gram matrix is an instance of the generalized Gauss-Newton (GGN) matrix (Schraudolph, 2003), an extensively studied object in the context of second-order optimization (Martens, 2010; 2020). This link allows us to leverage prior research on EuroSAT - SVHN
EuroSAT - SVHN
DTD - SUN397
−3 −1 1 3 𝛼2
−3 −1 1 3
−3 −1 1 3 𝛼2
−3 −1 1 3
−3 −1 1 3 𝛼2
−3 −1 1 3 𝛼1𝜏1 𝜏2 𝛼1𝜏1 𝜏2 𝛼1𝜏1 𝜏2

![1_image_0.png](1_image_0.png)

Linear F
T

OURS
𝜃0 𝜃0 𝜃0 Figure 1: Weight disentanglement *(left)* without and *(right)* with Jacobian Gram regularization.

efficient curvature approximations. Specifically, we adopt Kronecker-factored approximate curvature (KFAC, Martens & Grosse, 2015), a block-diagonal approximation of the GGN, where blocks correspond to layers and each block is a *Kronecker product* of two small matrices. KFAC drastically reduces storage and computation while still capturing most intra-layer correlations, bridging the gap between oversimplified diagonal approximations and the intractable full GGN of interest. Adapting KFAC for TA. KFAC–based regularization faces a key limitation when applied to multitask arithmetic: its associated regularizer cannot be accumulated exactly across tasks. The per-task regularizers induce memory and computational costs that grow linearly in the number of tasks. Going beyond the existing approximation, we propose an aggregation scheme that merges per-task curvature factors into a single surrogate, yielding *constant* complexity in the number of tasks. We show that linking the weight disentanglement objective to curvature-aware optimization yields state-of-the-art performance in *task addition* and *negation* (Ilharco et al., 2022). Furthermore, our method exhibits desirable properties, such as *task localization* - i.e., distinct task vectors govern separate, localized regions in function space associated with different tasks - and robustness to task vector rescaling, which renders performance insensitive to scaling coefficients and thus eliminates the need for held-out tuning. In summary, our contributions are the following:
- We derive a regularizer for task arithmetic - called TAK (Task Arithmetic with KFAC regularization) - that improves weight disentanglement without using external data.

- We scale representation drift regularization by aggregating per-task regularizers into a single surrogate, ensuring *constant* complexity and storage regardless of the number of tasks.

## 2 Background: Task Arithmetic And Linearized Fine-Tuning

Setup. Let f : R
D × R
P → R
C denote a neural network that processes a datum x ∈ R
D via parameters θ ∈ R
P into a prediction f(x, θ) ∈ R
C . During training, these predictions are compared to a target y ∈ R
Y via a criterion function c : R
C × R
Y → R with the goal to minimize the empirical risk over a training data set D = {(xn, yn)}n. We start from a model pre-trained on a large source dataset D0, yielding pre-trained weights θ0. Our goal is to fine-tune this model on a specific downstream task t with data set Dt, to obtain the task-specific fine-tuned weights θ
⋆
t.

Task Arithmetic. The above fine-tuning procedure is typically repeated for multiple (T) tasks, yielding *task vectors* {τt := θ
⋆
t − θ0}
T
t=1. Such vectors form the core of TA, which posits that simple linear operations in weight space can induce targeted transformations in function space. This enables combining the capabilities of multiple task vectors to build a multi-task model without additional training, through simple linear combination (*task addition*): given the individual task vectors
{τt}
T
t=1, the composed model has parameters θ0 +PT
t=1 αtτt with αt ∈ R (in the simplest case, αt = 1). TA also addresses the removal of task-specific knowledge (*task negation*) by subtracting, rather than adding, a task vector. However, na¨ıve linear composition is prone to interference, as overlapping task-vector updates often conflict and degrade the composed model's performance. Linearized fine-tuning. Ortiz-Jimenez et al. (2023) empirically show that TA benefits from model linearization, particularly when applied during both training and inference. This approach replaces the network with its linear approximation around the pre-trained weights, (f, θ0) ↔ flin as flin(x, θ) = f(x, θ0) + Jθf(x, θ0)(θ − θ0), (1)
with Jθf(x, θ0) ∈ R
C×P the Jacobian of the model's prediction on datum x with respect to its parameters, evaluated at θ0. This encourages weight disentanglement in TA, a property whereby task vectors influence the model only on their own tasks, leaving its behavior unchanged elsewhere. Our goal is to construct a regularizer to encourage this property during linearized fine-tuning.

| Algorithm 1 Idealized and practical representation drift regularizer for task t ′ Algorithm 2 Linearized FT on task vector τt ′ Require: Initial weights θ0, dataset Dt ′ , task vector τt ′ merged curvature matrix G−t ′ 1: Linearize the net: (f, θ0) → flin(•, τt ′ − θ0) 2: while not converged do 3: Draw a mini-batch B ∼ Dt ′ 4: Minimize objective Eq. (7) on B w.r.t. τt ′ 5: end while 6: return Task vector τt ′ T Require: Network f(·, θ0), tasks {Dt} t=1,t̸=t ′ 1: Compute per-task GGNs {Gt̸=t ′} (Eq. (3)) (approximate via KFAC, Sec. 3.3) 2: Merge over tasks: G−t ′ = P t̸=t ′ λtGt (optional: merge KFACs, Eq. (8)) 3: return Quadratic form: τ 7→ τ ⊤G−t ′τ   |
|---|

## 3 Making Representation Drift Regularization Data-Free

Simplified setup with two tasks. Model linearization simplifies the learning dynamics, allowing us to analyze how editing affects the model. We conduct this analysis in feature space through the lens of *representation drift*, the change in the last-layer activations of a task t when adding a new task t
′:

Pre-edit
$$\begin{array}{ll}\mbox{dil}&\mathbf{z}_{t}(\mathbf{x})=f_{\mbox{lin}}(\mathbf{x},\mathbf{\theta}_{0}+\alpha_{t}\mathbf{\tau}_{t})\stackrel{{\mbox{\scriptsize cdf}}}{{\rightarrow}}\mathbf{z}_{t,t^{\prime}}(\mathbf{x})=f_{\mbox{lin}}(\mathbf{x},\mathbf{\theta}_{0}+\alpha_{t}\mathbf{\tau}_{t}+\alpha_{t^{\prime}}\mathbf{\tau}_{t^{\prime}})\ \left(\mbox{rep}\right)\\ &\Longrightarrow\ \left(\mbox{\scriptsize{\sc{Repression}}}\right)\ \ \Delta_{t\to t,t^{\prime}}(\mathbf{x}):=\|\mathbf{z}_{t,t^{\prime}}(\mathbf{x})-\mathbf{z}_{t}(\mathbf{x})\|_{2}^{2}\end{array}$$

$\mathbf{r}$ 4. 
$\eqref{eq:walpha}$

′ )Post-edit
representation If the drift ∆t→t,t′ (x) vanishes for all x ∈ Dt, the newly added task vector τt
′ will not interfere as it does not change the model's behavior for inputs from task t. Interference between the two tasks can be reduced by penalizing representation drift (Yoshida et al., 2025) via the neural network function space distance (Dhawan et al., 2023) L
drift t→t,t′ (τt
′ ) := 1/|Dt|Px∈Dt ∆t→t,t′ (x). However, the regularizer for τt
′ requires accessing data of the external task t. This may violate segregation policies, impose significant storage demands, and prevent independent training, ultimately reducing flexibility for decentralized training. These issues make direct optimization of this objective impractical in many real-world settings, such as decentralized (McMahan et al., 2017; Kairouz et al., 2021) or privacy-preserving learning scenarios (Abadi et al., 2016; Bonawitz et al., 2017).

## 3.1 Connecting Representation Drift Regularization To Curvature Matrices

Now, we reformulate the regularization objective to eliminate its dependence on external task data.

Thanks to the linearization, the representation drift from Eq. (2) simplifies into ∆t→t,t′ (x) =
∥Jθflin(x, θ0)(αtτt − (αtτt + αt
′τt
′ ))∥
2 2 = α 2 t
′∥Jθflin(x, θ0) τt
′∥
2 2. The associated regularizer is1 L
drift t→t,t′ (τt
′ ) = α 2 t
′τ
⊤
t
′ Gt(θ0)τt
′ with Gt(θ0) = 1 |Dt| Px∈Dt Jθf(x, θ0)
⊤Jθf(x, θ0) (3)
Note that the network Jacobian's Gramian Gt(θ0) ∈ R
P ×P - after initial pre-computation - does not require further data access. This idealized training loop is shown in Alg. 1 (black font). In exchange for eliminating the data dependency, however, we now face the challenge of computing the P × P Gramian. This is infeasible even for small neural networks. Thankfully, we can interpret Gt as a curvature matrix that is well-known in the optimization literature: the generalized Gauss-
Newton (GGN) matrix (Schraudolph, 2003; Martens, 2020). This connection allows us to build on well-established approaches from the optimization literature to efficiently compute structural parametric approximations of Gt, ultimately allowing us to make Alg. 1 practical (red font).

## 3.2 The Generalized Gauss-Newton (Ggn) Matrix

The GGN is a curvature matrix related to the Hessian and arises from partial linearization: The Hessian of a function composition ℓ = c ◦ f is ∇2ℓ = ∇2(c ◦ f), while the GGN is ∇2(c ◦ flin).

The standard setting in the second-order optimization literature sets f to be the neural network, and c the criterion function used for training. We now introduce the GGN in this context, showing that the Jacobian Gram matrix from Eq. (3) is an instance of the GGN that results from replacing the training criterion with the squared loss. We can then easily transfer existing GGN approximations.

1In the following, we suppress lin since the Jacobians of f and flin coincide at θ0.

GGN in the training setting. Consider the neural network f with criterion function c (e.g. crossentropy) and training data D from Sec. 2. For sample n, define fn := f(•, xn) and cn := c(•, yn). The example-wise loss is then given by ℓn = cn ◦ fn, and training minimizes the empirical risk L(θ) = 1 |D| Pnc(f(xn, θ), yn) := 1 |D| Pnℓn(θ) := 1 |D| Pn(cn ◦ fn)(θ). (4)
For brevity, we use cn to denote the value cn(fn(θ)), and [•]i for slicing (e.g. [a]iis the i th entry of a). Differentiating the empirical risk twice and applying the chain rule yields the Hessian and its Gauss–Newton decomposition (Schraudolph, 2003; Martens, 2020), containing the GGN G(θ):
∇2L(θ) = G(θ) + R(θ) := 1 |D| Pn
(Jθfn)
⊤∇2cn(Jθfn) +1 |D| Pn PCm=1[∇cn]m∇2[fn]m . (5)
For models that are linear in the parameters, the residual R(θ) vanishes, as it depends on second derivatives, (zero in the linear case). The GGN then coincides with the Hessian of the risk under linearization and, for likelihood-based losses, with the Fisher information matrix (Amari, 2000).

The Jacobian's Gram matrix as GGN. The GGN in Eq. (5) generalizes the Jacobian Gram matrix from Eq. (3), used for representation drift regularization, by additionally weighting the Jacobians with the criterion function's Hessian ∇2c. If we choose squared error cn(f) = 1/2∥f − yn∥
22rather than the training criterion, the GGN becomes the Jacobian Gram matrix exactly, since ∇2cn = IC .

Hence, the matrix Gt(θ0) of the quadratic form in Eq. (3) corresponds to a curvature matrix: the GGN of the loss L(θ) *(Eq.* (4)*) when the training criterion is the squared loss.*
While the GGN is impractically large to compute or store for neural networks, the literature has developed scalable structured approximations for it. In the following, we build on these approximations (specifically, KFAC) and study how to adapt and extend them in the context of task arithmetic. 3.3 KRONECKER-FACTORED APPROXIMATION OF THE GENERALIZED GAUSS-NEWTON We rely on a structured GGN approximation called *Kronecker-Factored Approximate Curvature* (KFAC) introduced by Martens & Grosse (2015) for fully-connected, then generalized to convolutional (Grosse & Martens, 2016), recurrent (Martens et al., 2018), and transformer architectures (Eschenhagen et al., 2023). KFAC has been successfully applied to optimization (Osawa et al., 2019), pruning (Wang et al., 2019), Laplace approximations (Daxberger et al., 2021; Ritter et al., 2018) and influence functions (Grosse et al., 2023). For an in-depth tutorial, see Dangel et al. (2025). Parametric form. For a net with L layers and parameters θ 1*, . . . ,* θ L, KFAC approximates the GGN
as block-diagonal. Each block corresponds to a layer, G(θ) = blockdiag(G(θ 1)*, . . . ,* G(θ L)), and is further approximated as a Kronecker product, G(θ l) ≈ Bl ⊗ Al. To evaluate the approximation's quadratic form for representation drift regularization, we simply store the Kronecker factors
{(Blt, Alt)}l from task t, then evaluate (without instantiating the Kronecker product (Loan, 2000))

L
$$\operatorname*{drift}_{\tau\to t,t^{\prime}}(\tau_{t^{\prime}})=\alpha_{t^{\prime}}^{2}\tau_{t^{\prime}}^{\top}G_{t}(\theta_{0})\tau_{t^{\prime}}$$
′ , (6)
KFAC ≈ α 2 t ′PL l=1 τ l⊤ t ′ (Blt ⊗ Alt)τ l t
with τ l denoting the part of τ corresponding to the parameters in layer l.

KFAC for a single layer. To illustrate the approximation, consider a single fully-connected layer l in a neural network, with associated weights Wl ∈ R
D1×D2(we omit biases for simplicity). The layer processes an intermediate input representation a l n ∈ R
D2for datum xn into an intermediate output representation z l n = W aln ∈ R
D1. Further, let θ l:= vecWl ∈ R
D1D2 denote the row-flattened weights. The layer's GGN block is G(vec θ l) = 1/|D|Pn(Jθlfn)
⊤∇2cn(Jθlfn) and simplifies into a sum of Kronecker products by using the chain rule JvecWlfn = (Jzln fn)(JvecWlz ln) where JvecWlz ln = ID1 ⊗ a l⊤
n(e.g. Dangel et al., 2020) to obtain G*(vec*Wl) = 1 |D| Pn
(Jzln fn)
⊤∇2cn(Jzln fn) ⊗ a lna l⊤
n:= En[Bln ⊗ Aln
].

For the last equality, we use En[•] = 1/|D|Pn
•n for averaging over the data set. KFAC assumes En[•n ⊗ ⋆n] ≈ En[•n] ⊗ En[⋆n], yielding a single Kronecker product involving the small factors Al ∈ R
D2×D2, Bl ∈ R
D1×D1to approximate the intractable block G(vecWl) ∈ R
D1D2×D1D2:

G(vecWl)
KFAC
$\frac{\mathrm{AC}}{l}\left(\frac{1}{|\mathcal{D}|}\sum_{n}(\mathrm{J}_{\mathbf{z}_{n}^{l}}f_{n})^{\top}\nabla^{2}c_{n}(\mathrm{J}_{\mathbf{z}_{n}^{l}}f_{n})\right)\otimes\left(\frac{1}{|\mathcal{D}|}\sum_{n}\mathbf{a}_{n}^{l}\mathbf{a}_{n}^{l}\right)$
l⊤ n
$$\top\,\}:=\mathbf{B}^{l}\otimes\mathbf{A}^{l}\,.$$

![4_image_0.png](4_image_0.png)

Variations. KFAC computes two covariances per layer: (i) the input covariance Al = En[a lna l⊤ n
],
and (ii) the output gradient covariance Bl = En,m[g ln,mg l⊤
n,m] of pseudo-gradients g ln,m :=
(Jzln fn)
⊤sn,m obtained by backpropagating vectors sn,m ∈ R
C related to the Hessian ∇2cn. There exist different variations to compute Bland - since it is a priori unclear which approach works best in the context of TA - we consider two variants that differ in cost (details in (Dangel et al., 2025)):
(i) **Exact** (Botev et al., 2017) uses C backpropagations per datum and exactly computes Bl; *(ii)*
Monte-Carlo (MC, Martens & Grosse, 2015) randomizes the exact approach and computes an unbiased MC estimate of Bl using *M < C* backpropagations per datum (typically, M = 1).

## 3.4 Multi-Task Training Procedure & Regularization Merging

Na¨ıve multi-task regularization. While we focused on two tasks, extending to multiple tasks introduces new challenges. To promote disentanglement when training the task vector τt
′ , we penalize representation drift with respect to other tasks t ̸= t
′. Starting with the standard training loss LDt
′ (τt
′ ) = 1/|Dt′ |P(x,y)∈Dt′
c(flin(x, τt
′ + θ0), y), the overall fine-tuning objective becomes

$$\mathcal{L}_{\mathcal{D}_{s}}(\mathbf{\tau}_{t^{\prime}})+\beta\sum_{t\neq t^{\prime}}\lambda_{t}\mathcal{L}_{t\to t,t^{\prime}}^{\text{dist}}(\mathbf{\tau}_{t^{\prime}})\stackrel{{\text{KEVC}}}{{\approx}}\mathcal{L}_{\mathcal{D}_{s^{\prime}}}(\mathbf{\tau}_{t^{\prime}})+\beta\sum_{t\neq t^{\prime}}\lambda_{t}\sum_{l=1}^{L}\mathbf{\tau}_{t^{\prime}}^{l^{\top}}(\mathbf{B}_{t}^{l}\otimes\mathbf{A}_{t}^{l})\mathbf{\tau}_{t^{\prime}}^{l},\tag{7}$$

where β and λt control the overall and task-specific regularization strengths, respectively. We weight tasks by data set size, λt = |Dt|/Pt̸=t′ |Dt|. Given a pre-computed KFAC of each task t ̸= t
′, this formulation enables regularization without requiring direct access to data sets of external tasks. Accumulated regularizer. A key limitation of the objective in Eq. (7) is that we must store the Kronecker factors individually for each task, incurring O(T) memory and run time cost. To address this, we build upon the accumulated regularizer G−t
′ (θ l0) = Pt̸=t
′λtGt(θ l0) for layer l and approximate it with a single Kronecker product that captures the contribution of all other tasks:

$$-\iota^{\prime}(\theta_{0}^{l})\stackrel{\mathrm{\scriptsize{\textsc{KFAC}}}}{\approx}\sum_{t\neq t^{\prime}}\lambda_{t}B$$

$\frac{1}{2}$  . 

′ λtBlt ⊗ Alt
merge
≈Pt̸=t
. (8)
$\sum_{t\neq t'}B^l_t\Big)\otimes\Big(\sum_{t\neq t'}\lambda_t A^l_t\Big)$. 
Empirically, this heuristic (Eq. (8)) matches the un-merged formulation's performance (Eq. (7)).

## 4 Experiments

Task addition. We evaluate performance on the 8 Vision benchmark (Ilharco et al., 2022), which covers eight classification datasets. Using CLIP (Radford et al., 2021) as the foundational vision backbone, we collect eight checkpoints during training for each method and subsequently merge them into a single unified model. Additional details on training and datasets are provided in App. E. Following the original setup (Ortiz-Jimenez et al., 2023), we report both absolute and normalized accuracy. We further analyze the role of the rescaling coefficient α: (i) setting αt = α = 1 for all tasks, corresponding to plain task-vector addition, and *(ii)* tuning α on a cross-task validation set.

| Method                               | Dataless   | α    | ViT-B/32   | ViT-B/16   | ViT-L/14   |      |      |      |
|--------------------------------------|------------|------|------------|------------|------------|------|------|------|
| Pre-trained                          | -          | -    | 48.4       | -          | 55.4       | -    | 65.0 | -    |
| Individual                           | -          | -    | 90.9       | -          | 92.4       | -    | 93.8 | -    |
| Linear Fine-Tuning Linear FT         | -          | 1.0  | 76.7       | 87.2       | 80.2       | 88.9 | 88.0 | 94.8 |
| -                                    | Best       | 78.8 | 89.9       | 82.0       | 90.9       | 88.0 | 94.8 |      |
| τ Jp (Yoshida et al., 2025)          | ×          | 1.0  | 85.0       | 97.4       | 88.2       | 98.3 | 90.9 | 98.3 |
| Best                                 | 85.6       | 98.2 | 88.6       | 98.7       | 91.1       | 98.5 |      |      |
| Diag. GGN (Porrello et al., 2025)    | ✓          | 1.0  | 80.1       | 92.3       | 82.9       | 93.2 | 87.9 | 96.3 |
| Best                                 | 80.2       | 92.5 | 83.0       | 93.3       | 88.0       | 96.4 |      |      |
| TAK, Ours                            | ✓          | 1.0  | 85.8       | 97.6       | 88.3       | 97.9 | 91.6 | 99.3 |
| Best                                 | 86.0       | 97.8 | 88.3       | 98.1       | 91.6       | 99.3 |      |      |
| Non-Linear Fine-Tuning Non-linear FT | -          | 1.0  | 32.0       | 32.9       | 27.4       | 28.2 | 45.3 | 47.5 |
| -                                    | Best       | 73.5 | 80.4       | 77.0       | 82.9       | 84.5 | 89.7 |      |
| Attn. Only FT (Jin et al., 2025)     | -          | 1.0  | 22.5       | 23.3       | 22.8       | 23.4 | 66.2 | 69.7 |
| -                                    | Best       | 78.2 | 86.3       | 80.4       | 87.1       | 88.2 | 93.8 |      |
| TaLoS† (Iurada et al., 2025)         | ✓          | Best | 79.7       | 90.8       | 82.6       | 92.4 | 88.3 | 95.2 |
| Attn. Only FT                        | ✓          | 1.0  | 60.3       | 64.5       | 59.0       | 62.3 | 82.1 | 87.2 |
| + TAK, Ours                          | Best       | 83.1 | 91.3       | 84.3       | 91.0       | 89.9 | 95.9 |      |

Table 1: **Task addition** results on 8 Vision. The "α" column specifies how task vector coefficients are chosen. "1.0" denotes that all coefficients are fixed to 1.0, with no tuning. Numbers marked with
† for TaLoS (Iurada et al., 2025) are taken from the original paper. See Fig. 2 for a task-wise plot.

Comparison with related works. We present a comparative analysis of our regularizer TAK in two distinct regimes. On one hand, we evaluate it in the *linearized regime*, for which it was originally designed; on the other, we examine whether its benefits also extend to the *non-linear regime*. If so, this would broaden the applicability of our approach to most state-of-the-art learning frameworks. Linearized fine-tuning regime. We refer to Fig. 2 (left) for a depiction of the per-task absolute accuracy of the merged model in the linearized regime, while Tab. 1 reports the quantitative results on the 8 Vision benchmark. The results indicate that our KFAC-regularized approach yields substantial improvements against the baseline, achieving performance on par with τ Jp (Yoshida et al., 2025) while avoiding any reliance on external data from other tasks. This makes our method not only more flexible but also inherently privacy-preserving, without sacrificing accuracy. Furthermore, whereas competing methods often require coefficient grid search, TAK proves highly robust: even a simple addition of task vectors (α = 1) performs competitively, suggesting that post-hoc tuning can be safely omitted. As a side note, the evidence on ViT-B/32 suggests that the smaller the model scale, the more crucial curvature regularization becomes for achieving strong final performance. In this setup, we also compare against an approach inspired by Porrello et al. (2025) and apply curvature regularization using a coarse diagonal approximation of the GGN. While both methods exploit curvature information from the pre-trained model, ours relies on KFAC, providing a more accurate estimate that captures intra-layer dependencies. Results show that improved curvature approximations yield larger gains in Task Arithmetic; notably, even diagonal regularization outperforms na¨ıve linear fine-tuning, underscoring the role of regularization in enabling weight disentanglement. Non-linear fine-tuning regime. We now consider the non-linear fine-tuning regime (Tab. 1 and Fig. 2, right). In this setting, alternative approaches attempt to approximate linear behavior without fully linearizing the model. For example, TaLoS (Iurada et al., 2025) follows a different route and identifies a subset of parameters that consistently exhibit low gradient sensitivity across tasks and updates only these sparse components. This promotes weight disentanglement during fine-tuning while avoiding the computational bottlenecks of full linearization, enabling efficient task addition and negation. Instead, the authors of Attention-Only Fine-Tuning (Jin et al., 2025) fine-tune only the attention layers of Transformers, showing that this strategy implicitly induces *kernel-like* behavior.

| Method                       | Dataless   | ViT-B/32   | ViT-B/16   | ViT-L/14   |         |      |      |
|------------------------------|------------|------------|------------|------------|---------|------|------|
| Targ. ↓                      | Cont. ↑    | Targ. ↓    | Cont. ↑    | Targ. ↓    | Cont. ↑ |      |      |
| Pre-trained                  | -          | 48.4       | 63.3       | 55.4       | 68.3    | 65.0 | 75.5 |
| Non-linear FT                | -          | 20.4       | 60.5       | 20.4       | 65.3    | 18.1 | 72.4 |
| Linear FT                    | -          | 9.3        | 60.5       | 8.3        | 65.5    | 7.5  | 72.1 |
| TaLoS† (Iurada et al., 2025) | ✓          | 11.0       | 60.7       | 10.6       | 66.1    | 10.7 | 73.6 |
| τ Jp (Yoshida et al., 2025)  | ×          | 6.7        | 60.8       | 4.7        | 66.0    | 3.7  | 73.0 |
| TAK, Ours                    | ✓          | 3.4        | 62.4       | 3.4        | 66.4    | 3.5  | 72.6 |

Table 2: **Task negation** on 8 Vision. We report the minimum accuracy on target tasks while preserving at least 95% of the pretrained model's accuracy on control tasks.

| Method        | Dataless Abs. Norm.   |      |      |
|---------------|-----------------------|------|------|
| Individual    | -                     | 85.9 | -    |
| MTL           | -                     | 83.6 | -    |
| Non-lin. FT   | -                     | 75.7 | 87.7 |
| Linear FT     | -                     | 76.9 | 92.8 |
| Attn. Only FT | -                     | 72.9 | 85.2 |
| TaLoS         | ✓                     | 76.3 | 93.4 |
| τ Jp          | ×                     | 81.3 | 100  |
| TAK, Ours     | ✓                     | 78.7 | 98.9 |

![6_image_0.png](6_image_0.png)

(a) Task addition results for **T5-base**. All reported scores correspond to the bestperforming α values; the results obtained with α = 1 are provided in the appendix.

Figure 3: Results for language tasks. *Left*: impact of different training strategies and sensitivity to α hyperparameter. *Right*: effects of different regularizations on linear and non-linear fine-tuning. In this regard, although our regularization is not theoretically exact in the non-linear regime, its applicability can still be justified whenever linearized behavior is implicitly enforced. For this reason, in the non-linear setting we pair our regularizer with Attention-Only Fine-Tuning, which has been shown to induce approximately linear fine-tuning dynamics in Transformers, thereby providing a practical way to extend our method beyond the strictly linearized regime. The results in Fig. 2 (right) show that this is the case: when fine-tuning only attention layers, our approach proves beneficial even in the non-linear regime. Moreover, in this setting, the choice of the α coefficient has a stronger impact on the final accuracy. However, TAK remains the most robust on average, a trend further confirmed by an experiment reported in one of the subsequent paragraphs. Unlearning. We herein investigate a setting where each task vector is subtracted from the pre-trained model. In doing so, we use ImageNet as a control task to verify whether subtraction selectively removes the corresponding task without erasing general knowledge. As shown in Tab. 2, our model achieves stronger forgetting of target tasks while better preserving the control task, surpassing that of the main competitor, τ Jp (Yoshida et al., 2025). Notably, since our regularizer is dataless, it avoids the challenges associated with transferring and storing a "large" data set such as ImageNet to perform regularization. This property is especially promising in the context of the massive data sets used today to train conversational models, where the cost of data access and management is critical.

Task addition (*language tasks*) Following Stoica et al. (2025), we test across six natural language tasks: SNLI (Bowman et al., 2015), MultiNLI (Williams et al., 2018), SICK (Marelli et al., 2014),
SciTail (Khot et al., 2018), RTE (Wang et al., 2018), and QNLI (Wang et al., 2018), fine-tuning the T5-base model (Raffel et al., 2020). As shown in Fig. 3, TAK consistently outperforms the baselines, particularly under non-linear fine-tuning, thus corroborating the findings observed in vision. However, leveraging data from other tasks (τ Jp) yields additional gains, suggesting that textual domains may still benefit from even more accurate curvature estimation.

![7_image_0.png](7_image_0.png)

Figure 4: For ViT-B/32 (8 Vision), we analyze the sensitivity of different merging strategies to the scaling coefficient α; a similar analysis for ViT-B/16 is reported in the Appendix. Left: α-sweep accuracy of post-hoc merging strategies in the non-linear regime, compared with our linearized and regularized models. Right: performance of merging methods on linearized checkpoints.

![7_image_1.png](7_image_1.png) 
Comparison of model merging strategies. Fig. 4 compares existing post-hoc approaches for merging task vectors, including TIES (Yadav et al., 2023), TSV (Gargiulo et al., 2025), and ISO (Marczak et al., 2025). We remark that these methods operate after training and are therefore complementary to our approach, which instead acts during training and produces explicitly weight-disentangled task vectors. To assess the benefits of in-training regularization, in Fig. 4a we perform an α-sweep over the range [0, 2], focusing on *performance stability* - here, α scales the merged parameters θ0 + αM({τt}
T
t=1), where M(·) denotes the merging strategy. Under KFAC regularization (green curve), simple task-vector summation (Task Arithmetic, TA) achieves the best peak performance and exhibits strong robustness, with accuracy remaining stable over a wide interval of α values. This property makes our approach particularly suitable when α cannot be tuned, e.g., in the absence of a validation set. In practice, this robustness removes the need to access validation data from other tasks, which may be unavailable or undesirable to share. Moreover, as our method TAK relies on simple Task Arithmetic, it avoids expensive operations such as the SVD required by ISO and TSV. As a result, it can be applied in on-the-fly and adaptive model-merging settings (Crisostomi et al., 2026), enabling efficient personalization for specific user requests. In Fig. 4b, we analyze merging techniques applied to checkpoints obtained in the linearized regime. TA and TIES benefit the most from curvature regularization, whereas ISO and TSV already perform competitively without it. Nevertheless, their performance remains consistently below that of TAK, i.e., Task Arithmetic with curvature regularization. Additional results are reported in App. F. Curvature regularization enables Task Localization. We show that our approach enables a clear separation between training and out-of-distribution examples. Indeed, given an input x and a task vector τt, we measure ∥Jθf(x, θ0)τt∥
2 2, which we interpret as a *normalcy score* for task t. With our regularization (Eq. (3)), these scores are indeed forced to remain low for examples outside the t-th training distribution. As illustrated in Fig. 5, this is exactly what we observe in practice: the distribution of ∥Jθf(x, θ0)τt∥
2 2 is pushed toward zero whenever the input does not belong to task t.

With the na¨ıve linear fine-tuning, this behavior is instead much less clear.

| Method                 | Complexity   | α    | ViT-B/32   | ViT-B/16   | T5-base   |      |      |      |
|------------------------|--------------|------|------------|------------|-----------|------|------|------|
| Abs.                   | Norm.        | Abs. | Norm.      | Abs.       | Norm.     |      |      |      |
| Na¨ıve Multi-Task FT   | O(T)         | 1.0  | 86.5       | 98.4       | 88.0      | 97.5 | 78.5 | 97.0 |
| Best                   | 86.6         | 98.5 | 88.1       | 97.6       | 78.5      | 97.0 |      |      |
| Accumulated reg. (TAK) | O(1)         | 1.0  | 85.8       | 97.6       | 88.3      | 97.9 | 78.6 | 98.7 |
| Best                   | 86.0         | 97.8 | 88.3       | 98.1       | 78.7      | 98.9 |      |      |

![8_image_0.png](8_image_0.png)

| Exact       | MC=1 (ours)   |     |
|-------------|---------------|-----|
| A [s]       | 1.4           | 1.4 |
| B [s]       | 91.5          | 0.2 |
| Total [min] | 198.7         | 3.9 |

(b) Computation time for the KFAC approximation. Reported times for A and G correspond to the *average* over a batch of 8 examples, while the last row shows the total time (in minutes) required to compute the KFAC approximation for all tasks of 8 Vision.

This indicates that, under TAK's curvature regularization, each task vector influences the network output only for inputs drawn from its own training distribution. Moreover, this property suggests a natural use of our method for out-of-distribution detection, as it provides a principled mechanism to assess whether an input lies within the model training distribution. A complementary analysis in the non-linear fine-tuning regime is provided in App. F.5, where we compare our method against TaLoS and attention-only fine-tuning and observe that the same task-localization behavior persists. Na¨ıve multi-task training vs. **accumulated regularizer.** We herein investigate the impact of the heuristic used in our approach, which accumulates the Kronecker matrices (see Eq. (8)) and thereby avoids a linear cost in the number of tasks. To this end, we run experiments using the idealized na¨ıve multi-task training described in Eq. (7). Our findings, reported in Tab. 3, show that the gap between the idealized and the actual approach is marginal for medium-sized architectures such as ViT-B/16 in vision and T5-base in text. For ViT-B/32, we instead observe a small but consistent gap in favor of the idealized training objective, which aligns with our experience that smaller architectures tend to be more sensitive to curvature regularization and hence to the quality of the approximation. Training costs. Fig. 6 analyzes the overhead introduced by our approach, which is twofold: estimating the KFAC matrices (before training) and computing the regularizer (during training). No overhead is introduced at inference time. With a single Monte Carlo sample, estimating all KFAC matrices for the 8 Vision tasks (128 examples per task) takes **only 4 minutes**, a very limited amount of time compared to the exact approach from Botev et al. (2017). During training, the overhead mainly depends on the chosen regime, with linearized fine-tuning having the largest computational footprint. Nonetheless, KFAC regularization requires only a negligible amount of additional resources, i.e., roughly one third of the training time of τ Jp (Yoshida et al., 2025). This efficiency arises because the τ Jp penalty requires a second forward–backward pass through the (slower) linearized model. Moreover, since TAK does not rely on data for regularization, it avoids the repeated cost of loading new batches into GPU memory, another factor that slows down τ Jp. Memory footprint. Fig. 6 (right) reports the peak VRAM usage across training regimes. KFAC
introduces a small increase relative to unregularized baselines: in the linearized regime, it shows a
+12% overhead (11.5 → 12.9 GB) w.r.t. linear fine-tuning, while in the non-linear attention-only training it shows a +22% increase (6.8 → 8.3 GB). For reference, τ Jp peaks at 12.3 GB (+7% vs. linear FT), and standard non-linear fine-tuning reaches 8.5 GB. No memory overhead incurs at inference since regularization is inactive. Notably, aggregating all per-task KFAC factors into a single surrogate keeps the training footprint of our method at O(1) w.r.t. the number of tasks.

![9_image_0.png](9_image_0.png)

KFAC estimation. In Fig. 7a, we analyze the effect of varying the number of examples and MC samples used for curvature estimation. Our findings (Fig. 7a, Left) indicate that using 128–256 examples is already sufficient to saturate performance, yielding results comparable to those obtained with 30% of each training set. Moreover, final performance is generally on par with that obtained with the exact approximation of Botev et al. (2017). With respect to Monte Carlo sampling, only a few samples per example (1–2) are sufficient. Surprisingly, performance deteriorates beyond this point, with variance across seeds increasing as the number of MC samples grows. Overall, increasing the number of MC samples is less effective than using more data with fewer MC samples. KFAC compression. Unfortunately, the memory cost of storing KFAC matrices scales quadratically with the layer width, which may become challenging for very large models. To mitigate this cost, we evaluate how aggressively KFAC matrices can be compressed - via dynamic 8-bit quantization, structured pruning, block-diagonalization, and truncated SVD (see App. F.6) - without harming accuracy. On ViT-B/16 (8 Vision), these techniques yield substantial memory savings with only minor performance loss (Fig. 7b). The block-based strategy provides the best trade-off, decreasing memory from approximately 550 MB (full KFAC) to about 70 MB - 87% reduction - while incurring only ∼1-point drop in absolute accuracy (88.3 to 87.1). We additionally analyze whether the KFAC matrices can be moved off-GPU during training without introducing prohibitive overhead. To do so, we evaluate a regime where the penalty loss is computed and backpropagated **only once** every N training steps. As illustrated in Fig. 8, applying the loss every 16 steps leads to a modest degradation (∼1.4 points) relative to applying it at every iteration. This demonstrates that scheduling curvature updates can effectively amortize memory transfers and enable GPU–CPU factor shuffling without compromising the usefulness of the regularizer.

![9_image_1.png](9_image_1.png)

.

## 5 Conclusions

We investigate curvature-based regularization as a means to enhance weight disentanglement in Task Arithmetic and propose TAK (Task Arithmetic with KFAC regularization), a dataless, efficient, and effective approach that makes the simple summation of task vectors competitive with state-of-the-art merging strategies, without additional tuning. We demonstrate applicability in linearized and nonlinear regimes, and show that it enables a clear separation between in- and out-of-distribution examples. Our work calls for releasing additional assets together with the pre-trained weights without having to open-source the training data. Such information, e.g. gradient accumulators of the adaptive optimizer used for training (Li et al., 2025), or in our case KFAC, enable further downstream applications with foundation models. Finally, further extending these ideas to models adapted either via standard full or parameter-efficient fine-tuning remains an important direction.

## Acknowledgments

We acknowledge the CINECA award under the ISCRA initiative, for the availability of highperformance computing resources and support. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute. Simone Calderara is supported by the Horizon Europe Chips Joint Undertaking under the NexTArc project (HORIZON-JU-Chips-2024-2-RIA). NexTArc - Next Generation Open Innovations in Trustworthy Embedded AI Architectures for Smart Cities, Mobility and Logistics (Grant Agreement ID: 101194287, DOI: 10.3030/101194287). Additionally, the research activities of Angelo Porrello have been partially supported by the Department of Engineering "Enzo Ferrari" through the program FAR2025DIP (CUP E93C25000370005). We also gratefully acknowledge Symboolic s.r.l. for funding the PhD position of Thomas Sommariva and for the significant contribution of Lorenzo Bonicelli.

## Reproducibility Statement

To ensure the reproducibility of our results, the complete source code, including model implementations, hyperparameters, and evaluation scripts, is integrated into the Mammoth framework. The codebase will be made publicly available at https://github.com/aimagelab/mammoth to support further research and facilitate benchmarking.

## Disclosure On The Use Of Language Models

Large Language Models (LLMs) were used exclusively to improve the clarity and polish of the writing. All scientific ideas, methodological contributions, experimental designs, analyses, and conclusions presented in this paper originate entirely from the authors.

## References

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In *Proceedings of the 2016 ACM SIGSAC*
conference on computer and communications security, 2016.

Alessandro Achille, Aditya Golatkar, Avinash Ravichandran, Marzia Polito, and Stefano Soatto.

Lqf: Linear quadratic fine-tuning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2021.

Shun-Ichi Amari. Natural gradient works efficiently in learning. *Neural Computation*, 2000. Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. Advances in Neural Information Processing Systems, 2019.

Sanjeev Arora, Simon S Du, Zhiyuan Li, Ruslan Salakhutdinov, Ruosong Wang, and Dingli Yu.

Harnessing the power of infinitely wide deep nets on small-data tasks. International Conference on Learning Representations, 2020.

Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacypreserving machine learning. In *proceedings of the 2017 ACM SIGSAC Conference on Computer* and Communications Security, 2017.

Aleksandar Botev, Hippolyt Ritter, and David Barber. Practical gauss-newton optimisation for deep learning. In *International Conference on Machine Learning*, 2017.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2015.

Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. *Proceedings of the IEEE*, 2017.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2014.

Donato Crisostomi, Alessandro Zirilli, Antonio Andrea Gargiulo, Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri, Iacopo Masi, and Emanuele Rodola. MASS: Moerging through ` adaptive subspace selection. In *International Conference on Learning Representations*, 2026.

Felix Dangel, Stefan Harmeling, and Philipp Hennig. Modular block-diagonal curvature approximations for feedforward architectures. In International Conference on Artificial Intelligence and Statistics, 2020.

Felix Dangel, Runa Eschenhagen, Balint Mucs ´ anyi, and Tobias Weber. Kfac from scratch. ´ *arXiv*,
2025.

Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace redux - effortless bayesian deep learning. In Advances in Neural Information Processing Systems, 2021.

Nikita Dhawan, Sicong Huang, Juhan Bae, and Roger Baker Grosse. Efficient parametric approximations of neural network function space distance. In International Conference on Machine Learning, 2023.

Runa Eschenhagen, Alexander Immer, Richard E. Turner, Frank Schneider, and Philipp Hennig.

Kronecker-factored approximate curvature for modern neural network architectures. In Advances in Neural Information Processing Systems, 2023.

Antonio Andrea Gargiulo, Donato Crisostomi, Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri, and Emanuele Rodola. Task singular vectors: Reducing task interference in model merging. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, 2025.

Aditya Golatkar, Alessandro Achille, Avinash Ravichandran, Marzia Polito, and Stefano Soatto.

Mixed-privacy forgetting in deep networks. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2021.

Roger Grosse and James Martens. A kronecker-factored approximate Fisher matrix for convolution layers. In *International Conference on Machine Learning*, 2016.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamile Luko ˙ siˇ ut¯ e, Karina Nguyen, ˙ Nicholas Joseph, Sam McCandlish, Jared Kaplan, and Samuel R. Bowman. Studying large language model generalization with influence functions, 2023.

Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In IEEE International Conference on Computer Vision, 2021.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations, 2022.

Leonardo Iurada, Marco Ciccone, and Tatiana Tommasi. Efficient model editing with task-localized sparse fine-tuning. In *International Conference on Learning Representations*, 2025.