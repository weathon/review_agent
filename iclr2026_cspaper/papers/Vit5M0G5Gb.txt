# Saddle-To-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures

Yedi Zhang1,* Andrew Saxe1,2 **Peter E. Latham**1 1Gatsby Computational Neuroscience Unit, University College London 2Sainsbury Wellcome Centre, University College London
*Correspondence: yedi@gatsby.ucl.ac.uk

## Abstract

Neural networks trained with gradient descent often learn solutions of increasing complexity over time, a phenomenon known as simplicity bias. Despite being widely observed across architectures, existing theoretical treatments lack a unifying framework. We present a theoretical framework that explains a simplicity bias arising from saddle-to-saddle learning dynamics for a general class of neural networks, incorporating fully-connected, convolutional, and attention-based architectures. Here, *simple* means expressible with few hidden units, i.e., hidden neurons, convolutional kernels, or attention heads. Specifically, we show that linear networks learn solutions of increasing rank, ReLU networks learn solutions with an increasing number of kinks, convolutional networks learn solutions with an increasing number of convolutional kernels, and self-attention models learn solutions with an increasing number of attention heads. By analyzing fixed points, invariant manifolds, and dynamics of gradient descent learning, we show that saddle-to-saddle dynamics operates by iteratively evolving near an invariant manifold, approaching a saddle, and switching to another invariant manifold. Our analysis also disentangles data-induced and initialization-induced saddle-tosaddle dynamics. In particular, the former leads to low-rank weights while the latter to sparse weights. Equipped with the theory, we predict the effects of data distribution and weight initialization on the duration and number of plateaus in learning. Overall, our theory offers a framework for understanding when and why gradient descent progressively learns increasingly complex solutions.

## 1 Introduction

Deep neural networks trained with gradient descent often learn functions of increasing complexity over the course of training (Arpit et al., 2017; Kalimeris et al., 2019; Rahaman et al., 2019; Saxe et al., 2019; Refinetti et al., 2023; Bhattamishra et al., 2023; Abbe et al., 2023). This dynamical simplicity bias has been observed across architectures (Shah et al., 2020; Teney et al., 2022; Edelman et al., 2024), tasks (Karkada et al., 2025; Wurgaft et al., 2025; Wang & Pehlevan, 2025), and training paradigms ranging from supervised (Rahaman et al., 2019) to reinforcement (Schaul et al., 2019) and self-supervised learning (Simon et al., 2023). A particularly striking manifestation is stage-like dynamics: extended plateaus in loss alternating with bursts of rapid improvement as networks progress through increasingly complex input-output maps (Saxe et al., 2014; 2019). These dynamics, known as "saddle-to-saddle" dynamics because they can result from trajectories passing near a sequence of saddle points (Jacot et al., 2022; Berthier, 2023; Pesme & Flammarion, 2023), have been documented in deep linear networks (Saxe et al., 2014; 2019; Gissin et al., 2020; Jacot et al., 2022), two-layer and deep ReLU networks (Maennel et al., 2018; Boursier et al., 2022; Chistikov et al., 2023; Wang & Ma, 2023; Kumar & Haupt, 2024; Zhang et al., 2025a; Wu et al., 2025; Bantzis et al., 2026), and self-attention models (Boix-Adsera et al., 2023; Rende et al., 2024; Geshkovski et al., 2024; Zhang et al., 2025b; Varre et al., 2025), and have been hypothesized to be universal (Ziyin et al., 2025; Kunin et al., 2025). Yet the same architectures can also exhibit smooth, exponential training dynamics, simply by changing the initialization (Jacot et al., 2018; Tu et al.,

![1_image_0.png](1_image_0.png)

2024; Kunin et al., 2024); and more broadly, the emergence of stage-like dynamics can hinge on the data distribution (Yoshida & Okada, 2019; Goldt et al., 2020) and architectural choices (Orhan & Pitkow, 2018). These diverse findings raise foundational questions about the nature of dynamical simplicity bias in deep neural networks. Is there a universal mechanism driving stage-like dynamics, or a collection of architecture-specific mechanisms? Is there a principled link between stages and simplicity, such that earlier stages in learning are simpler? And if simplicity does underlie these dynamics, what is the operative notion of simplicity, and how does it reflect an architecture's inductive bias? Here we answer these questions. We show that for a range of architectures, including linear networks, ReLU networks, convolutional networks, quadratic networks, and linear self-attention (Figure 1B-G), there is a universal mechanism, saddle-to-saddle dynamics, driving stage-like learning.

The operative notion of simplicity is the number of effective units in the architecture, i.e., hidden neurons, convolutional kernels, or attention heads. In particular, first we show that fixed points in the loss landscape are recursively embedded: fixed points of smaller networks are embedded in saddle points of larger networks, yielding a nested hierarchy of saddles. Second, we show that saddle points are connected by invariant manifolds along which a larger network behaves like a smaller one, preserving simplicity along the connecting trajectories. Third, the link between saddleto-saddle dynamics and simplicity arises from the interplay of the saddle hierarchy and timescale separation. Specifically, timescale separation steers dynamics toward invariant manifolds associated with simple input-output maps, thereby controlling the complexity increment at each stage. We also disentangle data-induced and initialization-induced timescale separation, showing that the former leads to low-rank weights (Figure 1B,C) while the latter leads to sparse weights (Figure 1F,G). Together, this theory paints a unified picture of embedded saddles, invariant manifolds, and dynamics which give rise to a simplicity bias across architectures, and predicts when instead non-stage-like behavior will arise. Related work. We are inspired by a line of pioneering research that began with the seminal work of Fukumizu & Amari (2000) and continued in subsequent studies (Inoue et al., 2003; Amari et al., 2006; Wei et al., 2008; Amari et al., 2011; Fukumizu et al., 2019; Simsek et al., 2021; Zhang et al., 2021). In particular, Fukumizu & Amari (2000) first discovered a hierarchy of fixed points in twolayer fully-connected nonlinear neural networks. While their fixed points could, in principle, be extended to convolutional and attention-based architectures, they did not explore this, as convolutional architectures had not been popularized and attention-based architectures had not been invented. We study the fixed points across fully-connected, convolutional, and attention-based architectures. Further, we go beyond fixed points to study invariant manifolds and saddle-to-saddle dynamics, with implications for simplicity bias. A more detailed literature review is provided in Appendix A.

## 2 Network Setup

Let f(x) represent a neural network with input x ∈ R
D. We focus on one layer in the network with H units and trainable parameters θ1:H,

$$f(\mathbf{x};\mathbf{\theta}_{1:H})=g_{\text{out}}\left(\sum_{i=1}^{H}\phi(g_{\text{in}}(\mathbf{x});\mathbf{u}_{i})\mathbf{v}_{i}\right),\quad\text{where}\mathbf{\theta}_{i}=\begin{bmatrix}\mathbf{v}_{i}\\ \mathbf{u}_{i}\end{bmatrix}.$$  $\mathbf{\theta}_{i}$ ($\mathbf{\theta}_{i}$) represent the processing after and before this layer, which are 
(1)  $\frac{1}{2}$ ................................. (1)  ... 
. (1)
Here gout(·) and gin(·) represent the processing after and before this layer, which are usually deeper and shallower layers of the network. The weights are ui ∈ R
Nu , vi ∈ R
Nv , and thus θi ∈ R
Nu+Nv .

We place the second-layer weight vi on the right because ϕ(gin(x);ui) may be a scalar (as in a fully-connected layer) or matrix (as in a self-attention layer). The network output f(x; θ1:H) can be a scalar or vector. We will specify their dimensionality when we make them concrete. The definition of a layer in Equation (1) incorporates major architectures. For a fully-connected layer, a unit is a hidden neuron: ϕ(z; w, b) = σ(w⊤z + b) where σ(·) is the activation function and w, b are the weight and bias. For a convolutional layer, a unit is a convolutional kernel: ϕ(z;u) = σ(u ∗ z) where ∗ denotes convolution. For a self-attention layer, a unit is an attention head: ϕ(Z; K, Q) = I⊗smax(ZQK⊤Z⊤)Z where smax(·) denotes row-wise softmax and K, Q
are the key and query weights. A self-attention layer fits into our definition as follows, ATTN(Z) = smax(ZQK⊤Z
⊤)ZV = I ⊗ smax(ZQK⊤Z
⊤)Zvec(V ) = ϕ(Z; K, Q)v. (2)
We note that this is not a common notation for self-attention; we present it solely to show that Equation (1) incorporates self-attention. Hence, statements we will make about Equation (1) apply to fully-connected, convolutional, and self-attention architectures.

Let {xµ, yµ}
Pµ=1 be a supervised learning training set. The training loss is averaged over the training set L =
1 P
PPµ=1 ℓ(yµ, f(xµ)), where the loss function ℓ is second order differentiable with respect to f(x), including common choices like squared error loss. The parameters are trained with gradient flow on the training loss,

$\dot{\theta}=-\frac{\partial\mathcal{L}}{\partial\theta}=-\frac{\partial\mathcal{L}}{\partial f(\mathbf{x})}\frac{\partial f(\mathbf{x})}{\partial\theta}.$  ior of gradient descent in the limit of a small learning rate.  
Gradient flow captures the behavior of gradient descent in the limit of a small learning rate. Definition 1. A point θ
∗is a fixed point of the gradient flow dynamics in Equation (3) if ∂L
∂θ
θ∗ = 0.

## 3 Loss Landscape: Embedded Fixed Points

In this section, we establish that saddles generally exist in networks described by Equation (1). We show that a fixed point of a narrow network gives rise to a set of fixed points in a wider network.

$$({\mathfrak{I}})$$

These fixed points are constructed by embedding the narrow network into the wider network, as formalized in Theorem 1. Theorem 1 (Embedded fixed points). *If a network defined by Equation* (1) with (H − 1) units has a fixed point θ
∗
1:(H−1) *yielding an input-output map* f
∗(x), then there exists θ1:H ∈ S such that a network with H *units implements the same map* f
∗(x) and θ1:H is a fixed point.

We construct θ1:H by setting the first (H − 1) *units to* θ
∗
1:(H−1) 
and modifying them as follows.

(i) For any ϕ, the set S *includes*

$$\mathbf{u}_{H}=\mathbf{u}_{i}^{*},\,\mathbf{v}_{H}=\gamma_{v}\mathbf{v}_{i}^{*},\,\mathbf{v}_{i}=(1-\gamma_{v})\mathbf{v}_{i}^{*},\quad\gamma_{v}\in\mathbb{R},\,i\in\{1,\cdots,H-1\}.$$
$\mathbf{a}=\mathbf{a}$. 
(ii) If ∃uzero such that ∀z, ϕ(z;uzero) = 0, the set S *includes* uH = uzero, vH = 0. (5)
(iii) If ϕ(z;u) is degree-1 homogeneous in u, that is ∀α ∈ F, ϕ(z; αu) = αϕ(z;u)*, where* F = R
for general homogeneous functions, and F = R≥0 for positively homogeneous functions, e.g.,
the ReLU activation function, the set S *includes* uH = γuu
∗
i, vH = γvv
∗
i, vi = (1 − γuγv)v
∗
i, γv ∈ R, γu ∈ F, i ∈ {1, · · · , H − 1}. (6)
(iv) If ϕ(z;u) is linear in u, that is degree-1 homogeneous, ∀α ∈ R, ϕ(z; αu) = αϕ(z;u)*, and* additive, ϕ(z;ui) + ϕ(z;uj ) = ϕ(z;ui + uj ), the set S *includes*

$(\mathbf{u},\mathbf{u}_{1})+(\mathbf{u},\mathbf{u}_{j})=\gamma_{(\mathbf{u}_{1}}\mathbf{u}_{1}+\mathbf{u}_{j})$, $\mathbf{u}_{1}\in\mathbb{R}$, $\mathbf{u}_{H}=\sum_{i=1}^{H-1}\gamma_{u_{i}}\mathbf{u}_{i}^{*},\ \mathbf{v}_{H}=\sum_{i=1}^{H-1}\gamma_{v_{i}}\mathbf{v}_{i}^{*},\quad\gamma_{v_{i}},\gamma_{u_{i}}\in\mathbb{R}$, $\mathbf{v}_{i}=\mathbf{v}_{i}^{*}-\gamma_{u_{i}}\sum_{j=1}^{H-1}\gamma_{v_{j}}\mathbf{v}_{j}^{*},\quad i=1,\cdots,H-1$.  
$$\mathbf{u}_{H}=\mathbf{u}_{\mathrm{zero}},\mathbf{v}_{H}=\mathbf{0}.$$
$$(4)$$
$$({\boldsymbol{S}})$$
$$\left(7\right)$$
The proof of Theorem 1, which is provided in the Appendix E, consists of two steps. First, verify that for the weight configurations given above, the width-H network implements the same input-output map as the width-(H − 1) network. Second, show that gradients of the weights in the width-H network are either equal or proportional to those in the width-(H − 1) network, which are zero. Remark 1. Equation (4) is valid for any activation function ϕ, while the rest are valid for ϕ with specific properties, implying that certain properties of ϕ give rise to a larger set of embedded fixed points in weight space. Equations (4) and (5) were first discovered by Fukumizu & Amari (2000). We extend these two constructions with Equations (6) and (7). This extension is crucial for studying learning dynamics, as the saddles visited during learning turn out to fall under Equations (5) to (7) but not Equation (4). By induction, we obtain Corollary 2 by repeatedly applying Theorem 1 to embed multiple units in one layer and embed units in multiple layers of a deep network, with each layer defined by Equation (1).

Corollary 2. If a depth-L network with hl *units in layer* l(l = 1, · · · , L) has a fixed point yielding an input-output map f
∗(x), then for a depth-L network with Hl ≥ hl units in each layer, there exist weight configurations such that the network implements the same map f
∗(x) and the weight configurations are fixed points. Theorem 1 and Corollary 2 indicate that the global minima of a narrow network, even if they incur nonzero training loss, remain fixed points of the gradient flow dynamics in any wider network with the same architecture. For example, the global minimum of a width-1 network typically lacks the expressivity to fit the training set and thus incurs nonzero loss. In a wide network capable of achieving zero loss, the fixed points corresponding to the width-1 network global minimum are either saddles or local minima. They are guaranteed to be saddles in deep linear networks with rank-r (r ≥ 1) target maps (Baldi & Hornik, 1989; Kawaguchi, 2016) and, under mild conditions, are saddles in general architectures (Fukumizu & Amari, 2000; Fukumizu et al., 2019). In Figure 1, we show six cases where the network first visits a saddle, corresponding to a solution expressible by the architecture with a single unit. The network then converges to a stable fixed point, corresponding to a solution expressible with two units. The fixed points visited during learning fit into three different categories in Theorem 1. In panels (B,C), the fixed points visited during learning are described by Equation (7), corresponding to rank-one and rank-two weights. In panels (D,E), the fixed points are described by Equation (6), corresponding to one and two rays of proportional weights. In panels (E,F), the fixed points are described by Equation (5), corresponding one or two units with large weights with the rest being near zero.

## 4 Invariant Manifold: Effectively Narrow Networks

An invariant manifold of a dynamical system is a manifold such that any point starting on it remains on the manifold under the system's evolution. In Theorem 3, we show that for gradient flow dynamics of the class of neural networks we consider, invariant manifolds always exist. Further, these invariant manifolds correspond to weight configurations that make the network effectively narrower than its actual width. Theorem 3 (Invariant manifolds). Let T *be any time such that one of the following conditions (i)-* (iv) holds in a network defined by Equation (1)*. Then, in each case, the stated relationship between* the weights is preserved for all t ≥ T under gradient flow dynamics:
(i) For any ϕ*, two units have equal weights:* θi = θj .

(ii) If ∃uzero such that ∀z, ϕ(z;uzero) = 0*, a unit has zero weights:* vi = 0,ui = uzero.

(iii) If ϕ(z;u) is homogeneous in u, two units have proportional weights: θi = γθj , γ ∈ F.

(iv) If ϕ(z;u) is linear in u*, any number of units have linear dependence:* θi =Pj̸=i γjθj .

The precise definitions of homogeneity and linearity are given in Theorem 1.

The proof of Theorem 3 is provided in the Appendix F and is relatively straightforward. For example, when θi = θj , the gradients of θi and θj are equal and thus they stay equal for all future time. The invariant manifolds are larger in weight space when ϕ has zero, homogeneity or linearity properties, similar to the enlarged set of embedded fixed points in Theorem 1. When the weights of a network lie on an invariant manifold, its input-output map is expressible with fewer units than its actual width: simply remove the i-th unit and appropriately modify the remaining weights (see Appendix F.3). Further, we can have more than one constraints; e.g., θ1 = θ2 and θ3 = θ4. Each added constraint reduces the effective width by 1. Hence, when weights evolve on an invariant manifold, the simplicity of the network's input-output map is constrained by the effective width associated with the invariant manifold, rather than the actual width. The invariant manifolds indicate that there exist gradient flow paths connecting pairs of embedded fixed points defined in Theorem 1 (see Appendix F.4). Following such a path corresponds to an iteration of saddle-to-saddle dynamics. To see this, starting from an embedded fixed point with effective width h, we may apply a carefully chosen small perturbation that moves the weights onto the invariant manifold with effective width (h + 1). This perturbation corresponds to breaking exactly one constraint. By Theorem 3, the dynamics then remains on the invariant manifold for all time, eventually converging to a fixed point on it, that is, an embedded fixed point with effective width (h + 1). This process is one saddle-to-saddle transition: from the saddle with effective width h to the saddle with (h + 1). We illustrate this process in Figure 1A. In the next section, we develop heuristic arguments showing that the gradient flow dynamics can, in some cases, naturally evolve near such saddle-to-saddle paths on the invariant manifolds.

## 5 Saddle-To-Saddle Dynamics

The embedded fixed points (Section 3) and invariant manifolds (Section 4) hold for general architectures defined by Equation (1). To analyze learning dynamics, however, we must work with concrete architectures. We focus on two-layer networks where ϕ(x;u) is a homogeneous polynomial in the weights u, studying the linear and quadratic cases in detail. The linear case includes fully-connected linear networks and convolutional linear networks. The quadratic case includes quadratic networks (defined by Equation (71)) and linear self-attention. Both types of architectures exhibit saddle-tosaddle dynamics, but their mechanisms differ. We show that the mechanism in the linear case is a timescale separation between directions across all units due to the distribution of the data, while the the mechanism in the quadratic case is a timescale separation between units due to initialization.

$$\mathrm{L}\mathrm{I}\mathrm{N}\mathrm{E}$$

5.1 LINEAR CASE: TIMESCALE SEPARATION BETWEEN DIRECTIONS
Consider a two-layer network in which ϕ(x;u) is linear in the weights u,

$$f(\mathbf{x};\mathbf{\theta}_{1:H})=\sum_{i=1}^{H}\mathbf{v}_{i}\mathbf{u}_{i}^{\top}\mathbf{z}(\mathbf{x})\equiv\mathbf{W}\mathbf{z},\quad{\mathrm{where~}}\mathbf{v}\in\mathbb{R}^{N_{v}},\,\mathbf{u},\mathbf{z}\in\mathbb{R}^{N_{u}}.$$
$$\mathbf{(8)}$$
Nu . (8)
Here z(x) denotes any function of the input x, as ϕ(x;u) is linear in u but not necessarily linear in x. The gradient flow dynamics of Equation (8) trained on squared loss is
$$\dot{\mathbf{v}}_{i}=\left(\mathbf{\Sigma}_{yz}-\mathbf{W}\mathbf{\Sigma}_{zz}\right)\mathbf{u}_{i},\quad\dot{\mathbf{u}}_{i}=\left(\mathbf{\Sigma}_{yz}-\mathbf{W}\mathbf{\Sigma}_{zz}\right)^{\top}\mathbf{v}_{i},$$  where the data statistics are $\mathbf{\Sigma}_{yz}=\frac{1}{P}\sum_{\mu=1}^{P}\mathbf{y}_{\mu}\mathbf{z}_{\mu}^{\top},\mathbf{\Sigma}_{zz}=\frac{1}{P}\sum_{\mu=1}^{P}\mathbf{v}_{i}$.  
⊤vi, i = 1, · · · *, H,* (9)
µ. When the weights
are initialized to be small, i.e., vi(0) = O(ϵ),ui(0) = O(ϵ), i = 1, · · · , H, the first terms in Equation (9) dominate: Σyz − WΣzz = Σyz + O(ϵ
2). The weights thus approximately evolve as
a linear dynamical system (Equation (10)), which we analyze in Theorem 4. Theorem 4 (Timescale separation between directions). *Consider the linear dynamical system*
Let the singular value decomposition of Σyz *be given by* Σyz =PD
$\boldsymbol{v}_i,\quad i=1,\top$  $\sum_{\mu=1}^P\boldsymbol{z}_\mu\boldsymbol{z}_\mu^\top$. 
k=1 skqkr
k, D = min(Nv, Nu)
with singular values s1 ≥ · · · ≥ sD, and let the largest singular value s1 *have multiplicity* r
(1 ≤ r < D*). Let the initial weights be sampled independently from a Gaussian distribution*
N (0, ϵ2) with a small ϵ. When the projection of the weights on the span of the top r *singular vectors*
reaches O(1)*, that is*
$$\dot{\mathbf{v}}_{i}=\mathbf{\Sigma}_{yz}\mathbf{u}_{i},\quad\dot{\mathbf{u}}_{i}=\mathbf{\Sigma}_{yz}^{\top}\mathbf{v}_{i},\quad i=1,\cdots,H.\tag{10}$$  _composition of $\mathbf{\Sigma}_{yz}$ be given by $\mathbf{\Sigma}_{yz}=\sum_{k=1}^{D}s_{k}\mathbf{q}_{k}\mathbf{r}_{i}^{\top}$, $D=\min(N_{y},N_{y})$._
$$\|\mathbf{P\theta}_{i}\|=O(1),\quad\text{where}\mathbf{P}=\frac{1}{2}\sum_{k=1}^{r}\begin{bmatrix}\mathbf{q}_{k}\\ \mathbf{r}_{k}\end{bmatrix}\begin{bmatrix}\mathbf{q}_{k}^{\top}&\mathbf{r}_{k}^{\top}\end{bmatrix},\,\mathbf{\theta}_{i}=\begin{bmatrix}\mathbf{v}_{i}\\ \mathbf{u}_{i}\end{bmatrix},\tag{11}$$  _the projection on the remaining subspace is $\|(\mathbf{I}-\mathbf{P})\mathbf{\theta}_{i}\|=O(e^{1-s_{r+1}/s_{1}})$ almost surely._
$$(9)$$
$${}^{*}\ ,\ \Pi,$$
We provide the proof in Appendix G.2 and the intuition here. The second and first-layer weights
vi,ui grow exponentially along the singular vectors qk, rk, respectively, at the rate e
skt. Relative
to the dominant growth rate e
s1talong the top singular vectors, the components along other singular
vectors decay as e
(sk−s1)t, k = r + 1, · · · , D. Consequently, during the early phase, the weights
become increasingly aligned with the top singular vectors and thus approximately rank-r. Taking r = 1 as an example, the weights become approximately rank-one; specifically, vi aligns with q1, and ui aligns with r1 for every i. Theorem 3 implies that rank-r weights constrain a linear network to an invariant manifold corresponding to effective width r. Since the early phase dynamics drives the weights to be approximately rank-r, the network evolves near the invariant manifold and approaches a fixed point on it.
This is the first iteration of saddle-to-saddle dynamics. In weight space, the weights move from the initial saddle at zero to the second saddle. In function space, the network learns a more complex
solution, changing from a constant zero function to a rank-r projection of the target linear map.
Subsequent iterations of saddle-to-saddle dynamics operate similarly. The dynamics near a rank-r saddle, corresponding to a plateau in the loss, is again approximately a linear dynamical system
a placed in the loss, is again approximately a linear dynamical system  $$\hat{\mathbf{v}}_{i}=\widetilde{\mathbf{\Sigma}}_{yz}\mathbf{u}_{i},\quad\hat{\mathbf{u}}_{i}=\widetilde{\mathbf{\Sigma}}_{yz}^{\top}\mathbf{v}_{i},\quad i=1,\cdots,H.\tag{12}$$

## Where Σe Yz Is Σyz Projected Onto A Rank-(D − R) Subspace; See Appendix G.3. Via The Same Reasoning As Theorem 4, The Weights Grow The Fastest Along The Top Singular Vectors Of Σe Yz. Low-Rank
Weight Growth Will Again Place A Linear Network Near An Invariant Manifold With Few More Effective
Units, Guiding The Dynamics Toward A Fix Point On That Manifold.
To Summarize, In The Linear Case, Distinct Singular Values Of The Input-Output Correlation Matrix Induce A Timescale Separation Between Weight Growth Along Different Directions. If All Singular Values Are Distinct, The Timescale Separation Leads To Approximately Rank-One Weight Growth During A Loss Plateau, Causing The Escape Path From A Saddle To Closely Follow An Invariant Manifold With One More Effective Unit. 5.2 Quadratic Case: Timescale Separation Between Units

We now consider a two-layer network in which ϕ(x;u) is quadratic in the weights u,

$$f(\mathbf{x};\mathbf{\theta}_{1:H})=\sum_{i=1}^{H}v_{i}\mathbf{u}_{i}^{\top}\mathbf{Z}(\mathbf{x})\mathbf{u}_{i},\quad{\mathrm{where~}}v_{i}\in\mathbb{R},\mathbf{u}_{i}\in\mathbb{R}^{D},\mathbf{Z}\in\mathbb{R}^{D\times D}.$$
$$(13)$$

Here Z(x) denotes any function of the input x. For example, linear self-attention fits into Equation (13) with Z(x) being a cubic function of the input x, and ϕ(x;u) a quadratic function of the key and query weights u = [vec(K), vec(Q)]. We consider the scalar output case because it already has saddle-to-saddle dynamics and involves non-closed-form solutions. The gradient flow dynamics of Equation (13) trained on squared loss is given by Equation (44). Near small initialization, the quadratic terms in Equation (44) dominate. In Proposition 5, we analyze the approximate dynamics and show that one unit with the largest initialization grows much faster than the rest. Proposition 5 (Timescale separation between units). *Consider the dynamical system*
$$\dot{v}_{i}=\mathbf{u}_{i}^{\top}\mathbf{\Sigma}_{yZ}\mathbf{u}_{i},\quad\dot{\mathbf{u}}_{i}=2v_{i}\mathbf{\Sigma}_{yZ}\mathbf{u}_{i},\quad i=1,\cdots,H.\tag{14}$$
Assume ΣyZ *is symmetric and has both positive and negative eigenvalues. Let the initial weights be* sampled independently from a Gaussian distribution N (0, ϵ2) with a small ϵ. When weights in one
of the units reaches O(1), the rest of the units is O(ϵ) *almost surely.* We provide derivations in Appendix H.2 and the intuition here. The intuition is that the quadratic dynamics in Equation (14) is a rich-get-richer process. We can get a flavor of such dynamics by considering the simplest quadratic dynamics, v˙i = v 2 i, which has the solution

$$v_{i}(t)=\left({\frac{1}{v_{i}(0)}}-t\right)^{-1},\quad i=1,\cdots,H.$$
$$(15)$$

By solving for $t$ with $i$ and $j$, we can write $v_i(t)$ in terms of $v_j(t)$ as . 
$$v_{i}(t)=\left[\frac{1}{v_{j}(0)}\left(\frac{v_{j}(0)}{v_{i}(0)}-1\right)+\frac{1}{v_{j}(t)}\right]^{-1}.$$
$$(16)$$

−1. (16)
Assuming initial conditions of order O(ϵ), for example vi(0) ∼ N (0, ϵ2), and letting vj be the unit with the largest initial value, we see that when vj (t) ∼ O(1), the other units are still small: vi(t) ∼ O(ϵ) for i ̸= j. Thus, under quadratic dynamics v˙i = v 2 i, distinct initial conditions of the units induce a timescale separation in their growth. Although the general case, analyzed in Appendix H.2, is more complicated, the timescale separation between units essentially comes from the same mechanism. In Theorem 3(ii), we showed that if ϕ(x; 0) = 0 ∀x, then nonzero weights in one unit and zero weights in the rest of the units constrain a network to an invariant manifold with effective width one. Since the early dynamics drives one unit to grow much faster than the rest, the network evolves near the invariant manifold with effective width one and approaches a fixed point on it. This is the first iteration of saddle-to-saddle dynamics. Subsequent iterations operate similarly. Starting near the first saddle, one unit has nonzero weights and (H − 1) units still have small weights. The dynamics near the first saddle drives one of the (H − 1) units to grow much faster than the rest. Hence, the escape path from the first saddle again approximately follow the invariant manifold with two effective units, steering the dynamics toward a fixed point on that manifold. This process repeats.

For ϕ(x;u) that is quadratic in u and has ϕ(x; 0) = 0 ∀x, the distinct initial weights in each unit induce a timescale separation between the weight growth in different units. One unit grows much faster than the rest, causing the escape path from a saddle to closely follow an invariant manifold with one more effective unit. Higher-order polynomial activation. If ϕ(x;u) is a homogeneous polynomial of degree p > 2 in the weights u, we conjecture that there is still a timescale separation between units, possibly even stronger than the quadratic case. Our intuition is that the dynamics near zero has a similar flavor to the scalar dynamics, v˙i = v p i
. By similar reasoning to Proposition 5, the unit with the largest initialization grows much faster than the rest, causing a timescale separation between units. The dynamics in the cubic (p = 3) case is consistent with our intuition, as shown in Figure 4G.

![7_image_0.png](7_image_0.png)

General nonlinear activation. If ϕ(x;u) is a general nonlinear activation function, we can Taylor expand ϕ(x;u) around u = 0. With small initialization, u ≈ 0, the early dynamics near initialization is dominated by the lowest-order non-vanishing term in the Taylor expansion, assuming the data statistic associated with that term is nonzero. For example, in a two-layer fully-connected tanh network, the lowest-order non-vanishing term is the linear term. The tanh network thus develops rank-one weights in the early phase near initialization, similar to Theorem 4. However, the subsequent dynamics is not necessarily saddle-to-saddle, since rank-one weights do not generally correspond to invariant manifolds for tanh networks; see Figure 4D. By comparison, in a two-layer fully-connected network with activation ϕ(x;u) = u
⊤x · tanh(u
⊤x), the lowest-order non-vanishing term is quadratic. The network thus has a timescale separation between units similar to Proposition 5, and exhibits saddle-to-saddle dynamics as shown in Figure 4F.

## 6 Implications

We now validate our theory and demonstrate its predictive power by examining how the network width, data distribution, and initialization affect learning dynamics. Effect of network width. Our analysis in Section 5.1 shows that in linear networks, the timescale separation occurs between directions across all units. Consequently, increasing the number of units in linear networks has little effect on the dynamics, provided there are enough units to learn all directions. In contrast, the analysis in Section 5.2 implies that increasing the number of units in networks where ϕ(x;u) that is quadratic in u can shorten the plateaus. That is because the timescale separation in the quadratic case occurs between learning different units due to their distinct initial values. When sampling initial weights from a fixed distribution, increasing the number of weights reduces the gaps between adjacent samples, thereby shortening the plateaus. Simulations in Figure 2A confirm our theoretical prediction. In this case, increasing the number of heads of linear self-attention, for which ϕ(x;u) is quadratic in u, speeds up learning, while increasing the width of fully-connected linear networks does not. This demonstrates an interesting, theoretically grounded advantage of scaling up linear self-attention over scaling up fully-connected linear networks. Effect of data distribution. In linear networks, the timescale separation in learning different directions arises from the distinct singular values of Σyz. In Figure 2B, we let the singular values of Σyz follow a power law. As expected, decreasing the power law exponent narrows the gaps between singular values, thereby shortening the plateaus. When the exponent is 0, all the singular values are equal, eliminating the plateaus except the initial one corresponding to the escape from the saddle at zero. In this case, the largest singular value has multiplicity r = D in Theorem 4, causing the solution to jump directly from effective width 0 to D, skipping the stages in between. By contrast, in networks for which ϕ(x;u) is quadratic in u, the timescale separation is due to the distinct initial values in the units. Therefore, setting the positive singular values of ΣyZ to be equal shortens but does not eliminate plateaus. Simulations with linear self-attention in Figure 2B confirm our prediction. Effect of initialization structure. According to our theory, to have saddle-to-saddle dynamics the initialization must be near an invariant manifold, and the escape path from saddles must follow an invariant manifold. Perhaps surprisingly, however, initializing near a saddle is not a necessary condition. In Figure 2C we initialize the weights near an invariant manifold but away from saddles; for linear networks, this corresponds to large low-rank weights with a small perturbation. As predicted, learning undergoes saddle-to-saddle dynamics. Because the initialization is away from saddles, there is not a plateau at the start; the loss first drops exponentially and then exhibits plateaus followed by sigmoid-shaped drops. To our knowledge, this regime has not previously been observed. If we initialize near the invariant manifold associated with exactly the required number of effective units, loss undergoes a rapid exponential drop, even though the network learns a solution with low-rank weights, which is the feature learning solution in linear networks (Dominé et al., 2025). This result adds nuance to the common view that exponential loss curves are often a hallmark of lazy learning (Jacot et al., 2018; Chizat et al., 2019). Effect of initialization scale. We examine the effect of initialization scale when using an isotropic Gaussian distribution, a common choice in practice. As shown in Figure 2D, increasing the initialization scale gradually shortens the plateaus. Saddle-to-saddle dynamics becomes weaker in the sense that the learning trajectory does not approach the saddles as closely as it does with small initialization. For intermediate initialization scales, plateaus are less pronounced, yet the network still approximately learns solution of increasing complexity, similar to the case with small initialization. In architectures that have saddle-to-saddle dynamics, we conjecture that the distance from the initial weights to invariant manifolds associated with low effective width determines the strength of feature learning. This criterion can be viewed as an extension of prior beliefs, in which the relative scale of initial weights across layers (Dominé et al., 2025) or the rank of initial weights (Liu et al., 2024) were thought to determine the strength of feature learning.

## 7 Discussion

We studied the gradient flow dynamics of a broad class of architectures, analyzing fixed points, invariant manifolds, and dynamics near fixed points. Our theoretical framework reveals a general mechanism for saddle-to-saddle dynamics and provides a definition of simplicity that reflects the inductive biases of different architectures. When a network exhibits saddle-to-saddle dynamics, it recruits one or a few new effective units during each transition and learns solutions of increasingly complexity, where complexity is measured by the minimal number of units required for the architecture to express the solution. On a high level, we identify a mechanism behind the intuition that a neural network can decompose a task into smaller pieces and learn piece by piece over time. The learning process sometimes reconstructs the network's own architecture, one unit at a time. Condition for saddle-to-saddle dynamics. Saddle-to-saddle dynamics depends on two conditions: (i) the escape path from saddles closely follows invariant manifolds with few additional effective units; and (ii) the initialization is close to an invariant manifold with fewer effective units than needed to attain zero loss. As an example violating the first condition, two-layer tanh networks with small initialization develop rank-one weights during the early phase. This is because the tanh function is approximately linear near zero, and thus the early dynamics is approximately a linear dynamical system, similar to Theorem 4. However, since tanh is not homogeneous, rank-one weights do not correspond to an invariant manifold with effective width one. Consequently, tanh networks are not guided to approach the saddle with one effective unit, and probably do not have saddle-tosaddle dynamics in general. As an example violating the second condition, large isotropic random initialization is almost surely away from invariant manifolds. Thus, neural networks with large random initialization generally do not exhibit saddle-to-saddle dynamics. A special case violating the second condition is when an architecture has full expressivity with a single unit, such as linear networks with scalar input or scalar output (Shamir, 2019), and linear self-attention with merged key and query weights (Zhang et al., 2025b). Deep networks. The fixed points and invariant manifolds in Sections 3 and 4 apply to general deep networks defined by Equation (1), whereas the analysis of dynamics in Section 5 only applies to two-layer networks. Nonetheless, many deep newtorks still exhibit saddle-to-saddle dynamics, with some showing a timescale separation between directions and some between units, as shown in Figure 5. Although a general treatment of deep network dynamics is beyond the scope of this paper, we propose a conjecture for predicting which type of timescale separation (between directions or units) arises within a layer of a deep network. We conjecture that the order of the activation function ϕ(gin(x);ui), whether it is linear or quadratic in ui, continues to predict learning behaviors, including the type of the timescale separation and the effects of width and data distribution. In deep networks, gin(x) in Equation (1) may involve weights that are not specific to any individual unit of the layer under consideration, i.e., weights in shallower layers not indexed by i. For example, let us consider the second hidden layer of a depth-3 linear fully-connected network:

$$f(\mathbf{x})=\sum_{i=1}^{H}\mathbf{v}_{i}\phi(g_{\rm in}(\mathbf{x});\mathbf{u}_{i})=\sum_{i=1}^{H}\mathbf{v}_{i}\mathbf{u}_{i}^{\top}g_{\rm in}(\mathbf{x}),\quad\mbox{where}g_{\rm in}(\mathbf{x})=\mathbf{W}\mathbf{x},\tag{17}$$

where W is the first-layer weight matrix.1 Since ϕ(gin(x);ui) = u
⊤
igin(x) is linear in ui, we predict a timescale separation between directions similar to Section 5.1, and that the weights acquire an additional rank during each saddle-to-saddle transition. This is consistent with the existing literature (Gidel et al., 2019; Gissin et al., 2020) and our simulations in Figure 5. We further note that deep networks introduce several new questions that do not arise in the twolayer setting. If deep networks visit a sequence of embedded fixed points and learn increasingly complex solutions by recruiting additional effective units, which layers recruit additional units at each increase in complexity? This question is particularly interesting for transformers, which have self-attention, fully-connected layers, and skip connections. With skip connections, a deep network may also learn increasingly complex solutions by recruiting additional layers. This possibility seems consistent with the literature on layer pruning showing that large-scale transformers maintain their performance when removing up to half of the deeper layers and performing a small amount of finetuning (Gromov et al., 2025). Another work modeled the increasingly complex solutions of a transformer by increasing the width of its fully-connected layers (Wurgaft et al., 2025). Exhaustiveness of fixed points and invariant manifolds. Although we have not identified any fixed points or invariant manifolds beyond Proposition 5 and Theorem 3, it remains an open question whether these are exhaustive. If not, under what conditions do they become so? If the fixed points are exhaustive under reasonable assumptions, they would provide a useful diagnostic: each plateau during training would indicate that the network is implementing a solution expressible by a narrower sub-network. Moreover, the fixed points and invariant manifolds we describe arise solely from the network architecture and thus hold for any training data set. A further question is whether particular data sets can induce more fixed points or invariant manifolds than the data-agnostic ones (Zhao et al., 2023; Misof et al., 2025). Other architectures and learning rules. At its core, our theory exploits the permutation symmetry of units in feed-forward neural networks defined by Equation (1). Permutation symmetry exists beyond feed-forward architectures and supervised learning rules. Indeed, stage-like learning curves have been observed in recurrent neural networks (Proca et al., 2025; Ger & Barak, 2025), and other learning rules, such as reinforcement learning (Schaul et al., 2019), self-supervised learning (Simon et al., 2023), and predictive coding (Innocenti et al., 2024). This suggests the possibility of an even broader theory that incorporates these architectures and learning rules, with progressive permutation symmetry breaking as a unifying explanation for progressive learning behaviors.

## Acknowledgments

We thank Samuel Liebana, Loek van Rossem, Erin Grant, Stefano Sarao Mannelli, Máté Lengyel, Valentina Njaradi, Aaditya K. Singh, Andrew Lampinen, and Jin Hwa Lee for helpful conversations, and anonymous reviewers for their constructive feedback. We thank the following funding sources: Gatsby Charitable Foundation (GAT3850 and GAT4058) to YZ, AS, and PEL; Sainsbury Wellcome Centre Core Grant from Wellcome (219627/Z/19/Z) to AS; Schmidt Science Polymath Award to AS. AS is a CIFAR Azrieli Global Scholar in the Learning in Machines & Brains program.

## References

Emmanuel Abbe, Enric Boix Adserà, and Theodor Misiakiewicz. Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics. In Gergely Neu and Lorenzo Rosasco (eds.), *Proceedings of Thirty Sixth Conference on Learning Theory*, volume 195 of Proceedings of Machine Learning Research, pp. 2552–2623. PMLR, 12–15 Jul 2023. URL https: //proceedings.mlr.press/v195/abbe23a.html.

El Mehdi Achour, François Malgouyres, and Sébastien Gerchinovitz. The loss landscape of deep linear neural networks: a second-order analysis. *Journal of Machine Learning Research*, 25(242): 1–76, 2024. URL http://jmlr.org/papers/v25/23-0493.html.

Madhu S. Advani, Andrew M. Saxe, and Haim Sompolinsky. High-dimensional dynamics of generalization error in neural networks. *Neural Networks*, 132:428–446, 2020. ISSN 0893-6080. doi: https://doi.org/10.1016/j.neunet.2020.08.022. URL https://www.sciencedirect.com/scienc e/article/pii/S0893608020303117.

Shun-ichi Amari, Hyeyoung Park, and Tomoko Ozeki. Singularities affect dynamics of learning in neuromanifolds. *Neural Computation*, 18(5):1007–1065, 05 2006. ISSN 0899-7667. doi: 10.1162/neco.2006.18.5.1007. URL https://doi.org/10.1162/neco.2006.18.5.1007.

Shun-ichi Amari, Tomoko Ozeki, Florent Cousseau, and Haikun Wei. Dynamics of learning in hierarchical models - singularity and milnor attractor. In Rubin Wang and Fanji Gu (eds.), Advances in Cognitive Neurodynamics (II), pp. 3–9, Dordrecht, 2011. Springer Netherlands. ISBN 978-90-481-9695-1.

Devansh Arpit, Stanisław Jastrze¸bski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In Doina Precup and Yee Whye Teh (eds.), *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pp. 233–242. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/arpit17a.html.

Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learners:
The silent alignment effect. In *International Conference on Learning Representations*, 2022. URL https://openreview.net/forum?id=1NvflqAdoom.

Yuri Bakhtin. Noisy heteroclinic networks. *Probability theory and related fields*, 150(1):1–42, 2011. Pierre Baldi and Kurt Hornik. Neural networks and principal component analysis: Learning from examples without local minima. *Neural Networks*, 2(1):53–58, 1989. ISSN 0893-6080. doi: https://doi.org/10.1016/0893-6080(89)90014-2. URL https://www.sciencedirect.com/scie nce/article/pii/0893608089900142.

Ioannis Bantzis, James B Simon, and Arthur Jacot. Saddle-to-saddle dynamics in deep relu networks: Low-rank bias in the first saddle escape. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=B4zcoLvjw0.

Raphaël Berthier. Incremental learning in diagonal linear networks. Journal of Machine Learning Research, 24(171):1–26, 2023. URL http://jmlr.org/papers/v24/22-1395.html.

Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. *Foundations of Computational Mathematics*, pp. 1–84, 2024.

Raphaël Berthier. Diagonal linear networks and the lasso regularization path, 2026. URL https:
//arxiv.org/abs/2509.18766.

Satwik Bhattamishra, Arkil Patel, Varun Kanade, and Phil Blunsom. Simplicity bias in transformers and their ability to learn sparse Boolean functions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5767–5791, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.317. URL https://aclanthology.org/2023.acl-long.317/.

Enric Boix-Adsera, Etai Littwin, Emmanuel Abbe, Samy Bengio, and Joshua Susskind. Transformers learn through gradual rank increase. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 24519–24551. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper _files/paper/2023/file/4d69c1c057a8bd570ba4a7b71aae8331-Paper-Conference.pdf.

Etienne Boursier and Nicolas Flammarion. Simplicity bias and optimization threshold in twolayer ReLU networks. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of *Proceedings of Machine Learning* Research, pp. 5241–5275. PMLR, 13–19 Jul 2025a. URL https://proceedings.mlr.press/v2 67/boursier25a.html.

Etienne Boursier and Nicolas Flammarion. Early alignment in two-layer networks training is a two-edged sword. *Journal of Machine Learning Research*, 26(183):1–75, 2025b. URL http: //jmlr.org/papers/v26/24-1523.html.

Etienne Boursier, Loucas PILLAUD-VIVIEN, and Nicolas Flammarion. Gradient flow dynamics of shallow relu networks for square loss and orthogonal inputs. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing* Systems, volume 35, pp. 20105–20118. Curran Associates, Inc., 2022. URL https://proceedi ngs.neurips.cc/paper_files/paper/2022/file/7eeb9af3eb1f48e29c05e8dd3342b286-P aper-Conference.pdf.

Yuan Cao, Zhiying Fang, Yue Wu, Ding-Xuan Zhou, and Quanquan Gu. Towards understanding the spectral bias of deep learning. In Zhi-Hua Zhou (ed.), Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 2205–2211. International Joint Conferences on Artificial Intelligence Organization, 8 2021. doi: 10.24963/ijcai.2021/304. URL https://doi.org/10.24963/ijcai.2021/304. Main Track.

Ping-yeh Chiang, Renkun Ni, David Yu Miller, Arpit Bansal, Jonas Geiping, Micah Goldblum, and Tom Goldstein. Loss landscapes are all you need: Neural network generalization can be explained without the implicit bias of gradient descent. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=QC10RmRb Zy9.

Dmitry Chistikov, Matthias Englert, and Ranko Lazic. Learning a neuron by a shallow relu network: Dynamics and implicit bias for correlated inputs. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in Neural Information Processing Systems*,
volume 36, pp. 23748–23760. Curran Associates, Inc., 2023. URL https://proceedings.neur ips.cc/paper_files/paper/2023/file/4af24e6ce753c181e703f3f0be3b5e20-Paper-Confe rence.pdf.

Lénaïc Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming.

In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

URL https://proceedings.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df 326cb29c57225459-Paper.pdf.