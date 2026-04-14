

{0}------------------------------------------------

# FMS PINN: FLOW-MATCHING SAMPLING FOR EFFICIENT SOLUTION OF PARTIAL DIFFERENTIAL EQUATIONS WITH SOURCE SINGULARITIES

**Anonymous authors**

Paper under double-blind review

## ABSTRACT

Singularities in the source functions of partial differential equations (PDEs) can pose significant challenges for physics-informed neural networks (PINNs), often leading to numerical instability and necessitating a large number of sampling points thereby increasing the computational time. In this paper, we introduce a novel sampling point selection method to address these challenges. Our approach is based on diffusion models capable of generative sampling from the distribution of PDE residuals. Specifically, we apply the optimal transport coupling flow-matching technique to generate more sampling points in regions where the PDE residuals are higher, enhancing the accuracy and efficiency of the solution. In contrast to existing approaches in the literature, our method avoids explicit modeling of the probability density proportional to residuals, instead using the benefits of flow matching to generate novel and probable samples from more complex distributions, thereby enhancing PINN solutions for problems with singularities. We demonstrate that this method, in certain scenarios, outperforms existing techniques such as normalizing flow-based sampling PINN. Especially, our approach demonstrates effectiveness in improving the solution quality for the linear elasticity equation in the case of material with complex geometry of inclusion. A detailed comparison of the flow matching sampling method with other approaches is also provided.

## 1 INTRODUCTION

Physics-Informed Neural Networks (PINNs) are used to solve Partial Differential Equations (PDEs) using neural networks. With the rapid development of computing resources and machine learning algorithms, PINNs have become popular for a wide range of realistic simulations Raissi et al. (2019). PINNs utilize automatic differentiation mechanisms to encode PDEs into loss functions, incorporating PDE residuals and boundary conditions. PINNs may be preferred over classical numerical solvers due to their easy coding algorithms for both forward and inverse problems, and their ability to handle high-dimensional problems. Despite the widespread success of PINNs in various PDE-related problems they often struggle with complex PDEs, leading to "failure modes" Wang et al. (2022). Specifically, PINN loss function is very non-convex making it challenging to find a global minimum using conventional optimization algorithms for neural network training, such as Adam. Moreover, according to the F-principle Xu (2020), in PINNs the low frequency features of the solution are captured emerge first, while it will take several training epochs to reproduce high frequency features. In this regard, PINNs may be not efficient for solving PDEs with high-frequency solutions, as shown in Chuprov et al. (2023). For simple PDEs (single-scale, single-mode), conventional PINNs can quickly achieve satisfactory solutions Buzaeu et al. (2023). However, for more complex PDEs, conventional PINNs often fall short as the low-frequency global solution deviates from the exact solution. To address these issues, recent years have seen the emergence of efficient implementations of the PINNs method. For instance, loss re-weighting methods McClenny & Braga-Neto (2023) and adaptive sampling strategies Gao et al. (2023) have been developed to find a balance between loss and probability distribution on weight or sampling, enhancing the performance of PINNs in complex scenarios.

{1}------------------------------------------------

The goal of this paper is to design a novel method of point sampling to address these challenges. This approach uses the idea of diffusion models which are capable of generative sampling from the distribution of loss residuals. The optimal transport flow-matching technique is applied to generate more sampling points in regions where the PDE residuals have large values, enhancing the accuracy and efficiency of the solution.

## 2 PINN OVERVIEW

Consider a domain  $\Omega$  on which we want to solve a partial differential equation  $\mathcal{D}u(\mathbf{x}) = s(\mathbf{x})$ , where  $s(\mathbf{x})$  is the source function,  $\mathcal{D}$  is a differential operator, while  $\mathbf{x}$  is a  $d$ -dimensional vector from  $\Omega$ . The domain is bounded by  $\partial\Omega$  on which solution is subject to the following boundary condition:  $\mathcal{B}u(\mathbf{x}) = g(\mathbf{x})$  for  $\mathbf{x} \in \partial\Omega$ , where  $\mathcal{B}$  is the boundary operator (e.g. Neumann or Dirichlet conditions). Thus, we have

$$\begin{aligned}\mathcal{D}u(\mathbf{x}) &= s(\mathbf{x}), & \forall \mathbf{x} \in \Omega \\ \mathcal{B}u(\mathbf{x}) &= g(\mathbf{x}), & \forall \mathbf{x} \in \partial\Omega.\end{aligned}\tag{1}$$

We consider a neural network  $u_\psi(\mathbf{x})$  that approximates the solution of Equation 1, where  $\psi$  represents the parameters of the neural network that will be optimized during the training process. The training of the neural network is based on the minimization of the following function:

$$\min_{\psi} L(\psi) = L_{PDE}(\psi) + L_{BC}(\psi) = \|r(\mathbf{x}_i, \psi)\|_{2, \Omega} + \|\mathcal{B}u(\mathbf{x}) - g(\mathbf{x})\|_{2, \partial\Omega},$$

where

$$L_{PDE, N}(\psi) = \sum_{\mathbf{x}_i \in \mathcal{S}_k} (\mathcal{D}u(\mathbf{x}_i, \psi) - s(\mathbf{x}_i))^2 \tag{2}$$

and

$$L_{BC, N}(\psi) = \sum_{\mathbf{x}_i \in \partial\mathcal{S}_k} (\mathcal{B}u(\mathbf{x}_i, \psi) - g(\mathbf{x}_i))^2. \tag{3}$$

The partial derivatives of PINN with respect to the vector  $\mathbf{x}$  can be computed on the basis of automatic differentiation libraries.

## 3 RELATED WORK

### 3.1 ADAPTIVE SAMPLING AND ITS RELATION TO GENERATIVE LEARNING

The easiest method for selecting points is a mesh grid (regular grid), that is often used in finite difference schemes. However, in Wu et al. (2023) it was shown that this approach can potentially yield trivial solutions and that the PINN solution derived on a uniform grid is more accurate than that obtained on a mesh grid. In addition, sampling based on pseudo-random series (e.g. Sobol sequences, Latin hypercube etc) can be utilized.

Adaptive sampling methods are based on the principle to select points based on their influence on the loss function. One of the algorithms that pioneered this approach is the so called Residual Aided Refinement (RAR) algorithm Lu et al. (2021). The RAR algorithm aims to improve the distribution of residual points during training by introducing additional points in areas where the PDE loss values are large after a certain number of iterations. An advanced version of RAR is called residual adaptive distribution algorithm: RAD Wu et al. (2023). The PINN training begins with uniformly distributed points. After a few iterations, residual values are evaluated, and new points are added in areas with high residuals. The PINN model is retrained with the updated set of points, and this process is repeated to improve accuracy. This algorithm is similar to the classical importance sampling method, an extension of Monte-Carlo methods.

The algorithm that apply importance sampling idea to estimation of loss and sampling points for PINN is Nabian et al. (2021), where they propose to a special proposal distribution that is based on calculation of residual error at nearest to a point specially selected seed points. For instance, the proposal is a distribution that is a PDE residual loss at nearest seed point divided by sum of all

{2}------------------------------------------------

residual values. It means that points with larger values of PDE residual are more likely to be added to a batch. A similar methods are presented in Nabian et al. (2021).

One of the first methods introduced for sampling for variance reduction purposes was normalizing flows. In image generation such methods as GANs Goodfellow et al. (2020), VAEs and diffusion models can be used to digest data distribution and get a new sample from the population. In Bond-Taylor et al. (2021) it was shown that GANs generation can achieve higher quality for high resolution images than VAEs. However, GANs can be unstable due to mode covering problem while VAEs are able to cover all modes of distribution.

Diffusion models Song et al. (2020) do not suffer from mode collapse and can beat GAN models in image quality. They are based on the idea of iterative refinement of an input noise signal until it converges to a specific data distribution, such as an image. Diffusion models are trained by the forward process of incremental noise injection into data image and sampling represents the inverse process of image generation from noise. However, they require a substantial time for training Xiao et al. (2021) as compared to GANs and VAEs.

To improve the inference time of diffusion models with minimal depreciation in quality flow matching for image generation, a method Lipman et al. (2022) was introduced that is based on the refinement of loss function of the continuous normalizing flows Mathieu & Nickel (2020). This model do not implicitly approximate the probability distribution but can produce high quality samples at reasonable time. Flow matching is more stable during training because of its loss function, making it a preferable choice for generation tasks compared to score-based diffusion methods, especially for low-dimensional data such as points of collocation for training PINNs. This is the reason why this model was used as a tool to add samples that move in the direction of large residual regions.

### 3.2 ADAPTIVE SAMPLING STRATEGIES FOR PINNs

In Tang et al. (2023a), the so called DAS PINN was proposed, in which a normalizing flow was applied for adaptive PDE residual sampling for solution of Poisson equation with singular source peaks, while in Wang et al. (2024a) a similar approach was used for a cavity flow problem. In Tang et al. (2023b) the Wasserstein GAN-like model (WGAN) was proposed to solve the Poisson equation with narrow peaks in the source function. It generalizes a sampling approach for any normalized residual distribution  $p_\alpha$  but also uses the KR-net architecture same, as in Tang et al. (2023a). The main difference of AAS-PINN Tang et al. (2023b) from DAS-PINN is that AAS PINN learns the distribution of residual with WGAN like a loss function that also applies regularization to the gradient of  $p_\alpha$ :  $\nabla p_\alpha$ , while Tang et al. (2023a) uses KL divergence loss.

Due to the fact that KR net model for probability density function (PDF) approximation is invertible, it implicitly models PDF. However let us note that it may face difficulties to approximate more complex probability density functions of points with large residual. Moreover, as it is articulated in Wang et al. (2024b) due to the fact that normalizing flows preserve the topology of the input space through continuous transformations, they face difficulties in representing certain simple classes of function Dupont et al. (2019).

This property leads to limited representation capabilities, high computational costs, and training problems in practical implementations Ho et al. (2019). That is why in this paper, we decided to use an architecture different from KR-net and other normalizing flow architectures that represent invertible transformations for our proposed method.

### 3.3 NORMALIZING FLOW PINN

In Tang et al. (2023a), a model was developed, which integrates the PINN training with sampling from a normalizing flow. Subsequently, the flow model is refined through the minimization of the cross entropy loss between the residual and the flow's output logarithmic density. The flow model, implemented as KR-net, is constructed with affine coupling layers and the Knothe-Rosenblatt rearrangement. This architecture is uniquely designed to calculate both the forward and inverse probability density.

The optimization problem for the flow model is articulated as minimizing the Kullback-Leibler (KL) divergence between the residual function and the probability density generated by the flow. This

{3}------------------------------------------------

approach was specifically applied to solve the Poisson equation with a single peak source function and two peaks.

In the following section, we compare the normalizing flow PINN with our approach, providing a comprehensive analysis of the performance and efficiency of sampling techniques.

### 3.4 FLOW MATCHING PINN

### 3.1 Flow matching

Flow matching Lipman et al. (2022) is the generative algorithm that can deal with high complexity of data. Unlike the normalizing flow method, it does not require the neural network transformation to be invertible. Instead of implicitly modeling the probability density function  $p_1(x)$ , the flow matching model enables sampling from this probability density function  $p_1(x)$  by modeling vector field flow dynamics restoration. As sampling is based on sampling prior of Gaussian distribution  $p_0(x)$  that though flow field dynamics  $f_\theta(x, t)$  is transformed into more complex distribution  $p_1(x)$ , where time  $t$  varies from 0 to 1. In this regard, the sample from unconditional probability density function can be generated with the solution of the ODE:

$$\begin{cases} dX_t = f(X_t, t)dt \\ X_0 \sim p_0 \end{cases} \quad (4)$$

This fact is proved in Lipman et al. (2022) (see Theorem 1 therein) that relates the conditional probability distribution law dynamics  $p_t(x|x_1)$  with vector field  $f_\theta(x, t)$  through the continuity equation :

$$\frac{d}{dt}p_t(x) = -\operatorname{div}(p_t(x)f_t(x)) \quad (5)$$

$$\int_0^1 E_{p_t(x)} \|f_\theta(x, t) - f(x, t)\| \quad (6)$$

Let us note that  $f(x, t)$  is unknown and this functional is not feasible to evaluate. It turns out that it can be reduced to the conditional flow dynamics  $f_\theta(x, t|z)$ , where  $z$  can be a latent variable that is sampled from a prior distribution. That is why according to Theorem 2 Lipman et al. (2022) the intractable integral in 6 can be reduced to the following optimization problem that is tractable to solve:

$$\int_0^1 \mathbb{E}_{q(z)p_t(x|z)} \|f_\theta(x, t) - f(x, t|z)\|^2 dt \rightarrow \min_\theta, \quad (7)$$

The minimum that is a solution to this optimization problem is attained on the real vector field  $f(x, t)$ .

In order to find the minimum of such optimization problem the gradient can be calculated as an expectation that is found using the Monte Carlo approximation, namely,

$$\nabla_\theta \int_0^1 \mathbb{E}_{q(z)p_t(x|z)} \|f_\theta(x, t) - f(x, t|z)\|^2 dt \quad (8)$$

where  $z \sim q(z)$ ,  $x \sim p_t(x|z)$ . The equivalence of this fact was proved in Appendix that is based Theorem 2 from Lipman et al. (2022). Here we consider special case of optimal transport conditional vector field:  $f(x, t|z) = (1 - (1 - \sigma_{\min})t)x + tz$  for this we can rewrite the flow matching loss as:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{q(x_1), p(x_0)} \|f_\theta(x, t) - (x_1 - (1 - \sigma_{\min})x_0)\|^2$$

where  $x_1$  belong to data sample. According Theorem 1 in Lipman et al. (2022) this marginal vector field produces the probability path that is justified through continuity equation 5.

{4}------------------------------------------------

#### 3.4.1 THE SAMPLING ALGORITHM FOR PINN WITH BOOTSTRAP REWEIGHING

In order to tackle complex singularities in the domain (e.g. narrow peaks in the source function), we propose the following adaptive sampling algorithm for PINN. We apply flow matching paradigm as a vector field approximation of residual distribution approximation to sample points according to this vector field to add new points to the collocation point set for PINN training. The idea of the algorithm is based on refinement of PINN though adding points from region where high residual is concentrated. Instead of approximating the probability density function proportional to residual distribution we use this residual distribution in our algorithm to construct a probable sample from this distribution. For this case the residual of  $r(x_i, \psi^k)$  is calculated from PINN:

$$r(x_i, \psi^k) = \mathcal{D}u(x_i, \psi^k) - s(x_i). \quad (9)$$

We construct a sub-sample  $\mathbb{A}_k$  of size  $M$  from previous sample  $\mathbb{S}_{k-1}$  proportional to residual values using weighted bootstrap procedure. We train the vector field neural  $f_\theta(x, t)$  net on this sub-sample  $\mathbb{A}_k$  by minimizing flow matching objective 7 reformulated for tractability as in 8.

This vector field  $f_\theta(x, t)$  governs the dynamics of point from prior distribution to the closest point in residual distribution. In this way by application of flow matching sampling we generate points according to this  $f_\theta(x, t)$  by ODE dynamics to construct a new sample  $\mathbb{V}_i$ :

$$\begin{cases} dX_t = f_\theta(X_t, t) dt \\ X_0 \sim p_0 \end{cases} \quad (10)$$

This ODE can be solved numerically by using the Euler-Maruyama discretization scheme.

Moreover, for efficient sampling of points we propose the following algorithm of solving PINN with generation point proposal based on flow matching form:

##### --- Algorithm 1: FMS PINN: PINN with matching flow ---

**Input** : number of points in initial sample  $N$ , number of points for training sample for vector field  $M$ , number of stages  $K$

Sample  $N$  points uniformly from the domain  $\Omega$  denote this set as  $\mathbb{S}_0 = \{x_i\}_{i=1}^N$  ;  
Train PINN model on sample  $\mathbb{S}_0$  by optimizing empirical loss

$$\min_{\psi} L(\psi, N) = L_{PDE, N}(\psi) + L_{BC, N}(\psi) \quad (11)$$

where  $L_{BC, N}(\psi)$  defined in 3 and  $L_{PDE, N}(\psi)$  defined in 2. ;

**for**  $k$  from 1 to  $K$  **do**

    Calculate  $r(x_i, \psi^k)$  at each point of  $\mathbb{A}_k$ , i.e., and get values  $\{r(x_i, \psi^{k-1})\}_{i=1}^N$ ;  
    Based on these weights, perform a weighted bootstrap resampling of points to form the root points  $A_k$  for the flow, denoted as  $\{x_i^s\}_{i=1}^N$ ;  
    Train the vector field  $f_\theta(x, t)$  on this root points sample  $A_k$  by optimizing flow matching objective as in 7;  
    Sample new points  $\mathbb{V}_k$  according to the vector field  $f_\theta(x, t)$  that corresponds to  $p_1(x)$  using the Euler method for solving ODE 10;  
    Construct a sample for PINN as  $\mathbb{S}_{k+1} = \mathbb{V}_k \cup \mathbb{S}_k$ ;  
    Train PINN model  $u(x, \psi^k)$  on  $\mathbb{S}_{k+1}$  by optimizing loss 11 ;

**return**  $u(x, \psi^k)$  ;

---

The sampling step of flow matching is performed following the steps listed below:

##### --- Algorithm 2: Flow-matching Sampling ---

**Input:** Trained network  $f_\theta$ , Sample-access to base distribution  $q$ , Step-size  $\Delta t$   
**Output:** Sample from target distribution  $p$   
 $x_1 \leftarrow \text{Sample}(q)$ ;  
**for**  $t = 1, (1 - \Delta t), (1 - 2\Delta t), \dots, \Delta t$  **do**  
     $x_{t-\Delta t} \leftarrow x_t + f_\theta(x_t, t)\Delta t$ ;  
**return**  $x_0$ ;

---

The idea of Algorithm 1 can be represented by this diagram shown in Figure.

{5}------------------------------------------------

![Figure 1: FMS PINN algorithm flowchart. The process starts with 'Add points to sample' and 'Sample points from set S_i'. These lead to a neural network diagram with 'Input' (z, y), 'Hidden' layers, and 'Output' (u_x, u_y). From the output, two paths emerge: 'Compute residual at each point of S_i' leading to a plot of r(x, y) = Du(x, y) - a(x), and 'Calculate PDE, BC losses' leading to formulas for Du(x, y), B(u(x, y) - g(x)), and loss functions L_PDE and L_BC. The residual plot leads to 'Construct subsample S_{i+1} using weighted bootstrap with weights proportional to residuals', which leads to 'Train flow matching generative model' (showing a table of f(x, t) and A), which finally leads to 'Construct sample S_{i+1}^f from flow matching generative model' (showing a plot of X_t ~ p(x)).](191a4a245a7d36d03be9a990d0f758f5_img.jpg)

Figure 1: FMS PINN algorithm flowchart. The process starts with 'Add points to sample' and 'Sample points from set S\_i'. These lead to a neural network diagram with 'Input' (z, y), 'Hidden' layers, and 'Output' (u\_x, u\_y). From the output, two paths emerge: 'Compute residual at each point of S\_i' leading to a plot of r(x, y) = Du(x, y) - a(x), and 'Calculate PDE, BC losses' leading to formulas for Du(x, y), B(u(x, y) - g(x)), and loss functions L\_PDE and L\_BC. The residual plot leads to 'Construct subsample S\_{i+1} using weighted bootstrap with weights proportional to residuals', which leads to 'Train flow matching generative model' (showing a table of f(x, t) and A), which finally leads to 'Construct sample S\_{i+1}^f from flow matching generative model' (showing a plot of X\_t ~ p(x)).

Figure 1: FMS PINN algorithm

## 4 NUMERICAL RESULTS

### 4.1 9 PEAKS PROBLEM

The issue of singular domains characterized by multi-modality in terms of peaks poses a significant challenge for simple generative models. These models, such as GANs, often struggle with the problem of mode collapse in terms of mode covering. This is why the Poisson equation with a source function consisting of 9 peaks is an important example to demonstrate the effectiveness of our method in enhancing PINN to address the problem of mode covering. By doing so, our method enables the production of accurate solutions that capture the complex structure of the domain.

The Poisson equation with 9 peaks looks as follows:

$$\begin{aligned} -\Delta u(\mathbf{x}) &= s(\mathbf{x}) \quad \text{in } \mathbb{D}, \\ u(\mathbf{x}) &= g(\mathbf{x}) \quad \text{on } \partial\mathbb{D}, \end{aligned} \quad (12)$$

where  $\mathbf{x} = [x_1, x_2]^T$  and  $\mathbb{D} = [-1, 1]^2$ . Here  $s(\mathbf{x})$  has centers in  $(x_0^i, y_0^i) = (-0.5, 0.5) + (\frac{\text{mod}(i, 3)}{2}, 0) + (0, \frac{\lfloor i/3 \rfloor}{2})$ ,  $i = 0, \dots, 8$  and is represented by  $s(\mathbf{x}) = \sum_{i=0}^8 s_i(\mathbf{x})$ , where

$$\begin{aligned} s_i(\mathbf{x}) &= -e^{-1000((x - c_{i,0})^2 + (y - c_{i,1})^2)} \left( (-2 \cdot 1000(x - c_{i,0}))^2 - 2 \cdot 1000 \right) - \\ &\quad -e^{-1000((x - c_{i,0})^2 + (y - c_{i,1})^2)} \left( (-2 \cdot 1000(y - c_{i,1}))^2 - 2 \cdot 1000 \right), i = 0, \dots, 8 \end{aligned} \quad (13)$$

We also evaluate the absolute difference profile at the validation dataset for a network approximating the PDE solution. For the network, we used a fully connected network (FCN) with 6 blocks and a layer width of 64. For the flow matching model, we used an optimal transport coupling based on a FCN network. We trained the flow vector field model for 2000 iterations, each time resampling points and repeating the resampling every 5000 iterations, adding 28000 points each time. For

![Figure 2: Comparison of solution for 9 peaks problem. (a) Reference solution: A heatmap showing 9 distinct peaks arranged in a 3x3 grid. (b) Solution of FMS PINN: A heatmap showing a smooth solution that covers the 9 peaks. (c) Solution of DAS PINN: A heatmap showing a smooth solution that also covers the 9 peaks. All three plots have a color bar on the right indicating the magnitude of the solution.](2c51e13558d27d3452dfec3fb73aaed1_img.jpg)

Figure 2: Comparison of solution for 9 peaks problem. (a) Reference solution: A heatmap showing 9 distinct peaks arranged in a 3x3 grid. (b) Solution of FMS PINN: A heatmap showing a smooth solution that covers the 9 peaks. (c) Solution of DAS PINN: A heatmap showing a smooth solution that also covers the 9 peaks. All three plots have a color bar on the right indicating the magnitude of the solution.

Figure 2: Comparison of solution for 9 peaks problem

the normalizing flow comparison, we used the KR-net implementation from Tang et al. (2023a) in

{6}------------------------------------------------

TensorFlow with the same number of points, epochs, and resampling stages. We added the 9 peaks loss functional and reference in TensorFlow to make the comparison. In Figure 12, we observe that the normalizing flow method fails to capture the main solution compared to the matching flow approach. The flow matching PINN solution accurately depicts all nine peaks, keeping the solution outside the peaks close to zero. Figure 3 show that, in most of the domain, the solution of the flow matching PINN is quite accurate. The mean square difference (MSE) comparison between normalizing flow PINN and matching flow PINN during training for the 9 peak Poisson equation is illustrated by the MSE metrics calculated for different epochs and depicted in Figure 4b. The MSE of the Flow matching PINN decreases and converges to an order of  $10^{-3}$ . Our method samples points from peaks center as depicted by Figure 4a.

![Figure 3(a): Residual profile of FMS PINN. A heatmap showing the residual distribution across a 2D domain from -1.0 to 1.0 on both axes. The color scale ranges from 1.0e-02 (blue) to 5.0e-02 (red). The plot shows a relatively smooth, low-residual surface.](c0843c6d138705289960d9f53a6e72a1_img.jpg)

Figure 3(a): Residual profile of FMS PINN. A heatmap showing the residual distribution across a 2D domain from -1.0 to 1.0 on both axes. The color scale ranges from 1.0e-02 (blue) to 5.0e-02 (red). The plot shows a relatively smooth, low-residual surface.

(a) Residual profile of FMS PINN

![Figure 3(b): Residual profile of DAS PINN. A heatmap showing the residual distribution across a 2D domain from -1.0 to 1.0 on both axes. The color scale ranges from 0.2 (blue) to 1.0 (red). The plot shows a dark blue background with nine distinct bright yellow/white spots, indicating high residual values at the peak locations.](c64e9e9f3b0b828a5f6ac70441877764_img.jpg)

Figure 3(b): Residual profile of DAS PINN. A heatmap showing the residual distribution across a 2D domain from -1.0 to 1.0 on both axes. The color scale ranges from 0.2 (blue) to 1.0 (red). The plot shows a dark blue background with nine distinct bright yellow/white spots, indicating high residual values at the peak locations.

(b) Residual profile of DAS PINN

Figure 3: Comparison of residual profiles for 9 peaks problem

![Figure 4(a): Resampled points of FMS PINN added at 1 stage. A scatter plot showing the distribution of resampled points across a 2D domain from -0.6 to 0.6 on both axes. The points are clustered around nine distinct centers, forming a 3x3 grid pattern.](84e2ac543ffc4145dc85b05a48ec62e3_img.jpg)

Figure 4(a): Resampled points of FMS PINN added at 1 stage. A scatter plot showing the distribution of resampled points across a 2D domain from -0.6 to 0.6 on both axes. The points are clustered around nine distinct centers, forming a 3x3 grid pattern.

(a) Resampled points of FMS PINN added at 1 stage

![Figure 4(b): MSE comparison of FMS PINN with DAS PINN for 9 peaks problem. A line plot showing the Mean Squared Error (MSE) on a logarithmic y-axis (from 10^-3 to 10^1) versus Epochs (from 0 to 20000). The FMS PINN curve (blue) starts at a high MSE, drops sharply, and then gradually decreases to approximately 10^-3. The DAS PINN curve (orange) starts at a high MSE, drops to about 10^-1, and then fluctuates around that level.](797231cfee084ca299de599340240401_img.jpg)

Figure 4(b): MSE comparison of FMS PINN with DAS PINN for 9 peaks problem. A line plot showing the Mean Squared Error (MSE) on a logarithmic y-axis (from 10^-3 to 10^1) versus Epochs (from 0 to 20000). The FMS PINN curve (blue) starts at a high MSE, drops sharply, and then gradually decreases to approximately 10^-3. The DAS PINN curve (orange) starts at a high MSE, drops to about 10^-1, and then fluctuates around that level.

(b) MSE comparison of FMS PINN with DAS PINN for 9 peaks problem

Figure 4: MSE plot and samples from FMS PINN

#### 4.1.1 FIVE-DIMENSIONAL TWO PEAKS PROBLEM

For five dimensional problem the two centers of peaks are placed in  $(x_1, x_2, x_3, x_4, x_5) = (0.5, 0.5, 0, 0, 0)$  and at  $(x_1, x_2, x_3, x_4, x_5) = (-0.5, -0.5, 0, 0, 0)$ . As in Zhang et al. (2024) the reference solution of this problem is:

$$u^*(x, y) = \sum_{i=1}^c \sum_{j=1}^d \exp \left[ -K \left( (x_j - x_j^i)^2 \right) \right], (x_1, x_2, \dots, x_5) \in \Omega \quad (14)$$

where  $K = 100$ . In order to make an inference of the model and compute numerical errors efficiently we follow the methodology as in Zhang et al. (2024) where the two-stage sampling strategy for inference where proposed, where firstly 100k points are sampled uniformly across the domain. Then these points are combined with 15k points drawn from Gaussian distributions, whose mean and covariance are determined by each part of the solution led by one of the centers. These points

{7}------------------------------------------------

Table 1: Comparison of linear elasticity equation PINN with Normalizing flow PINN in terms of MSE

| Method | 2 peaks problem 16 | 2 peaks in 5D | 9 peaks problem |
|-|-|-|-|
| FMS PINN | <b>7.7e-5</b> | <b>6.1e-3</b> | <b>4.2e-4</b> |
| DAS PINN | 5.2e-4 | 2.3 | 1e-1 |

with 10k points on the boundary are subsequently used to compute numerical errors for this five-dimensional two peaks problem.

For training purpose at initial step of training we draw 100k points from uniform distribution and 60k points from Gaussian centers. For optimizer algorithm we used Adam with learning rate 0.001.

We trained FMS PINN with sampling 40000 additional points at every resampling stage from the vector field trained via optimal transport flow matching objective on weighted bootstrap sub-sample from PINN training set  $S_{k-1}$  and its residual distribution as weights.

We compare our algorithm with DAS PINN approach on the same number of training points equal to 100k and 60k points from center and for comparison use normalizing flow architecture of KR-net. We see that our method successfully captures all features of the solution, while method based on normalizing flow DAS PINN fails to produce the solution for same number of points and resampling stages.

![Figure 5: 5D 2 peaks problem. (a) Projection on first two coordinates of reference solution of 5D 2 peaks problem. (b) Projection of solution of 5D 2 peaks problem by FMS PINN algorithm. (c) Projection of solution of 5D 2 peaks problem by DAS PINN algorithm.](aa14b9ec884bf40ce06c161be468cd84_img.jpg)

Figure 5 shows three heatmaps representing the projection of solutions for a 5D 2 peaks problem. (a) Reference solution: shows two distinct peaks. (b) FMS PINN: shows two distinct peaks, matching the reference. (c) DAS PINN: shows a single broad peak, failing to capture the two-peak structure.

Figure 5: 5D 2 peaks problem. (a) Projection on first two coordinates of reference solution of 5D 2 peaks problem. (b) Projection of solution of 5D 2 peaks problem by FMS PINN algorithm. (c) Projection of solution of 5D 2 peaks problem by DAS PINN algorithm.

Figure 5: 5D 2 peaks problem

![Figure 6: MSE comparison of FMS PINN algorithm with DAS PINN and comparison with reference. (a) MSE per epoch comparison of FMS PINN algorithm with DAS PINN. (b) Error profile of 5D 2 peaks problem by FMS PINN algorithm. (c) Error profile of 5D 2 peaks problem by DAS PINN algorithm.](f5e70cbe66e71e65b4ae4aa7816d266a_img.jpg)

Figure 6 compares FMS PINN and DAS PINN. (a) MSE per epoch: FMS PINN (blue) drops to ~10^-4, while DAS PINN (orange) plateaus at ~10^-2. (b) Error profile FMS PINN: shows two peaks with low error. (c) Error profile DAS PINN: shows a single broad peak with high error.

Figure 6: MSE comparison of FMS PINN algorithm with DAS PINN and comparison with reference. (a) MSE per epoch comparison of FMS PINN algorithm with DAS PINN. (b) Error profile of 5D 2 peaks problem by FMS PINN algorithm. (c) Error profile of 5D 2 peaks problem by DAS PINN algorithm.

Figure 6: MSE comparison of FMS PINN algorithm with DAS PINN and comparison with reference

Finally, Table 1 summarizes comparison results for the normalizing flow PINN and the flow matching PINN for Poisson problems with peaks in source function, revealing comparable efficiency of the proposed method comparing to the normalizing flow.

{8}------------------------------------------------

### 4.2 LINEAR ELASTICITY EQUATION

In this section we consider solving a special instance of the mechanical equilibrium equation for a rectangular plate with a unique geometric inclusion made of a second material that we call linear elasticity equation. The primary equation that governs the mechanism of stress under deformation is

$$\nabla \cdot \sigma = 0,$$

where  $\sigma$  is the stress tensor - the 2-nd order tensor describing the internal pressure state of the object. This equation is then can be represented:

$$\begin{aligned} C(1-\nu)\frac{\partial^2 u_x}{\partial x^2} + C\nu\frac{\partial^2 u_x}{\partial x\partial y} + \frac{1}{2}C(1-2\nu)\left(\frac{\partial^2 u_x}{\partial y^2} + \frac{\partial^2 u_y}{\partial x\partial y}\right) &= 0 & (\text{x-axis}) \\ \frac{1}{2}C(1-2\nu)\left(\frac{\partial^2 u_x}{\partial x\partial y} + \frac{\partial^2 u_y}{\partial x^2}\right) + C\nu\frac{\partial^2 u_x}{\partial x\partial y} + C(1-\nu)\frac{\partial^2 u_y}{\partial y^2} &= 0 & (\text{y-axis}), \end{aligned} \quad (15)$$

where  $E$  and  $\nu$  are the Young modulus and Poisson ratio-constants, describing the material properties, while  $u_x$  and  $u_y$  represent horizontal and vertical displacement respectively.

The detailed derivation of this equation can be found in the subsection A.2.

where  $C = \frac{E}{(1+2)(1-2\nu)}$  - constant.

We consider square plate with  $(x, y) \in [x_{min}, x_{max}] \times [y_{min}, y_{max}]$ . Dirichlet boundary conditions are enforced on horizontal displacement for the boundary of square.

$$\begin{cases} u_x(x, y) = -0.01, & x = x_{min}, & \forall y \in [y_{min}, y_{max}] \\ u_x(x, y) = 0.01, & x = x_{max}, & \forall y \in [y_{min}, y_{max}] \\ u_y(x, y) = 0.0, & \text{on the boundary} \end{cases}$$

We consider a specific kind of plate, that consists of one base material and second material in complex geometry inclusion, that is characterised with different Young modulus (= material property of stiffness)  $E$ . The geometric configurations are diamond and 2 circles.

Structure of the neural network is represented by 5 separate fully connected neural nets.

![Figure 7: Comparison of solution for 2 circles u_x problem. (a) Reference solution: A heatmap showing a smooth, linear gradient of u_x from -0.4 (blue) to 0.4 (red) across a square domain. (b) Solution of FMS PINN: A heatmap showing a smooth solution that closely matches the reference solution. (c) Solution of DAS PINN: A heatmap showing a noisy, less smooth solution compared to the reference and FMS PINN.](8f93234090c5bc224e4b9d035018a927_img.jpg)

Figure 7: Comparison of solution for 2 circles u\_x problem. (a) Reference solution: A heatmap showing a smooth, linear gradient of u\_x from -0.4 (blue) to 0.4 (red) across a square domain. (b) Solution of FMS PINN: A heatmap showing a smooth solution that closely matches the reference solution. (c) Solution of DAS PINN: A heatmap showing a noisy, less smooth solution compared to the reference and FMS PINN.

Figure 7: Comparison of solution for 2 circles  $u_x$  problem

For 30000 epochs of training we see that flow matching PINN outperforms DAS PINN. The results of MSE comparison for the flow matching PINN with the normalizing flow PINN is shown in Table 2. For two circles problem the flow matching method helps to improve quality, while for the diamond configuration it provides the solution of the same quality as the normalizing flow PINN.

Results of our method compared to the reference solution and DAS PINN for 2 circles case is illustrated in Figure 7 and Figure 8. PINN neural net architecture for our methods consists of 5 separate neural networks that have 5 fully connected layers with 40 neurons in each layer. As an optimizer, we use Adam with the scheduler ReduceLROnPlateau. As it is shown in Figure 9, our method captured all main patterns of the reference solution as we see our algorithm FMS PINN outperforms DAS PINN.

{9}------------------------------------------------

![Figure 8: Comparison of solution for 2 circles u_y problem. (a) Reference solution, (b) Solution of FMS PINN, (c) Solution of DAS PINN. Each plot shows a heatmap of the solution u_y on a 2D domain [0, 100] x [0, 100]. The color scale ranges from -1.00 (blue) to 1.00 (red). The reference solution (a) shows two distinct circular regions of high and low values. The FMS PINN solution (b) is smoother and less accurate. The DAS PINN solution (c) is more accurate and closer to the reference solution.](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 8: Comparison of solution for 2 circles u\_y problem. (a) Reference solution, (b) Solution of FMS PINN, (c) Solution of DAS PINN. Each plot shows a heatmap of the solution u\_y on a 2D domain [0, 100] x [0, 100]. The color scale ranges from -1.00 (blue) to 1.00 (red). The reference solution (a) shows two distinct circular regions of high and low values. The FMS PINN solution (b) is smoother and less accurate. The DAS PINN solution (c) is more accurate and closer to the reference solution.

Figure 8: Comparison of solution for 2 circles  $u_y$  problem![Figure 9: Comparison of error profiles for DAS PINN and FMS PINN for 2 circles problem. (a) Profile of 2 circles FMS PINN solution for u_x, (b) Profile of 2 circles DAS PINN solution for u_x, (c) Profile of 2 circles FMS PINN solution for u_y, (d) Profile of 2 circles DAS PINN solution for u_y. Each plot shows a heatmap of the error profile on a 2D domain [0, 100] x [0, 100]. The color scale ranges from -0.4 (blue) to 0.4 (red). The FMS PINN error profiles (a, c) show higher error values (more red) compared to the DAS PINN error profiles (b, d), which are more uniform and closer to zero (more green).](f519a5be118c846f631c992412353fb9_img.jpg)

Figure 9: Comparison of error profiles for DAS PINN and FMS PINN for 2 circles problem. (a) Profile of 2 circles FMS PINN solution for u\_x, (b) Profile of 2 circles DAS PINN solution for u\_x, (c) Profile of 2 circles FMS PINN solution for u\_y, (d) Profile of 2 circles DAS PINN solution for u\_y. Each plot shows a heatmap of the error profile on a 2D domain [0, 100] x [0, 100]. The color scale ranges from -0.4 (blue) to 0.4 (red). The FMS PINN error profiles (a, c) show higher error values (more red) compared to the DAS PINN error profiles (b, d), which are more uniform and closer to zero (more green).

Figure 9: Comparison of error profiles for DAS PINN and FMS PINN for 2 circles problem

Table 2: Comparison of Elasticity PINN with Normalizing flow PINN in terms of MSE

| Method | 2 circles $u_x$ | 2 circles $u_y$ | diamond $u_x$ | diamonds $u_y$ |
|-|-|-|-|-|
| FMS PINN | <b>1.5e-3</b> | <b>7.9e-3</b> | <b>4.6e-3</b> | 9.2e-3 |
| DAS PINN | 1.7e-2 | 1.2e-2 | 7.1e-3 | <b>8.6e-3</b> |

## 5 CONCLUSION

In this paper a novel approach referred to as flow-matching sampling is proposed. It allows to select points for PINNs training, at which the evaluation of the PDE residual is performed. The idea of the method is based on the generative matching flows and adaptive sampling.

The numerical experiments show that our approach helps to solve singular problems and enhance the solution. We have examined an efficiency of the proposed method for the Poisson equation and linear elasticity equation system. It has been shown that the proposed method in several cases allow to achieve more accurate solution than the normalization flow approach. The latter can be considered as the closest competitor of the flow-matching method. It has been shown that the flow-matching method is efficient in the case of singularities in the solution. In our future work we will examine this method on larger number of epochs.

## 6 REPRODUCIBILITY STATEMENT

All of our experimental results are fully reproducible, and we have documented all settings and parameters used in our experiments. Upon request from the reviewers, we are prepared to provide the code and detailed instructions to help to replicate our findings. For the comparison with the DAS PINN method, we utilized the publicly available repository at <https://github.com/MJfadeaway/DAS>. By employing this repository, we ensured that our comparative analysis was conducted under consistent conditions, thereby guaranteeing a fair and accurate assessment between our proposed approach and the DAS PINN algorithm.

 Rest of paper (reference and Appendix) is removed.