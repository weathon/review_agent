

{0}------------------------------------------------

# FLOW MATCHING ACHIEVES ALMOST MINIMAX OPTIMAL CONVERGENCE

**Kenji Fukumizu**

The Institute of Statistical Mathematics/Preferred Networks  
Tokyo, Japan  
fukumizu@ism.ac.jp

**Taiji Suzuki**

University of Tokyo/RIKEN AIP  
Tokyo, Japan  
taiji@mist.i.u-tokyo.ac.jp

**Noboru Isobe**

University of Tokyo  
Tokyo, Japan  
nobo0409@g.ecc.u-tokyo.ac.jp

**Kazusato Oko**

University of Tokyo/RIKEN AIP  
Tokyo, Japan  
oko-kazusato@g.ecc.u-tokyo.ac.jp

**Masanori Koyama**

Preferred Networks/University of Tokyo  
Tokyo, Japan  
masanori.koyama@weblab.t.u-tokyo.ac.jp

## ABSTRACT

Flow matching (FM) has gained significant attention as a simulation-free generative model. Unlike diffusion models, which are based on stochastic differential equations, FM employs a simpler approach by solving an ordinary differential equation with an initial condition from a normal distribution, thus streamlining the sample generation process. This paper discusses the convergence properties of FM for large sample size under the  $p$ -Wasserstein distance. We establish that FM can achieve an almost minimax optimal convergence rate for  $1 \leq p \leq 2$ , presenting the first theoretical evidence that FM can reach convergence rates comparable to those of diffusion models. Our analysis extends existing frameworks by examining a broader class of mean and variance functions for the vector fields and identifies specific conditions necessary to attain almost optimal rates.

## 1 INTRODUCTION

Flow matching (FM) (Lipman et al., 2023; Albergo and Vanden-Eijnden, 2023; Liu et al., 2023b) is a recent simulation-free generative model that produces samples of the target distribution by solving an ordinary differential equation (ODE) initialized with a source normal distribution. The vector field to define the ODE is trained by neural networks with the teaching data of random conditional vectors. This approach bypasses the computationally intensive Monte Carlo sampling required in the diffusion model, which is currently the standard in generative modeling. Various variations have been proposed to refine the learning of vector fields, such as OT-CFM (Tong et al., 2024), rectified flow (Liu et al., 2023b), consistent velocity field (Yang et al., 2024), equivariant flow (Klein et al., 2023), etc. A series of studies also emerge from the viewpoint of interpolating distributions (Albergo et al., 2023c;a).

FM has already been applied to various domains with promising performance. Among many others, the rectified flow method has been extended to high-resolution text image generation (Esser et al., 2024), and there are also many works on the application of FM to molecule generation (Hoogeboom et al., 2022; Guan et al., 2023; Bose et al., 2023; Dunn and Koes, 2024), text generation (Hu et al., 2024), speech generation (Le et al., 2023), motion synthesis (Hu et al., 2023), etc.

Although the methods have been developed on the solid theoretical basis of the flows and continuity equation, their statistical behaviors remain less understood. Recent works have established the convergence of the FM estimator to the true distribution under some distributional metrics (Albergo and Vanden-Eijnden, 2023; Benton et al., 2023b). Beyond the convergence, more detailed understandings,

{1}------------------------------------------------

such as convergence rates, are still an open question. In contrast, diffusion models have gained various theoretical understandings, including the convergence rate in terms of the number of steps (Chen et al., 2023; Benton et al., 2023a) and the sample size (Oko et al., 2023; Zhang et al., 2024). Among others, Oko et al. (2023) has shown that diffusion models achieve the minimax optimal convergence rate for a large sample size under the total variation metric and the almost minimax optimal rate under the 1-Wasserstein distance, where the max is taken over the true densities of the Besov space. This result theoretically supports the high generation ability of diffusion models.

This paper aims to bridge this gap by demonstrating that FM can achieve an almost minimax optimal convergence rate for a large sample size under the  $p$ -Wasserstein distance  $W_p$  for  $1 \leq p \leq 2$ , suggesting that FM has a theoretical ability comparable to diffusion models. This problem is significant for comparing the ability of FM methods and diffusion models, and revealing the difference between SDE and ODE in the generative models. Drawing on the methodologies of Oko et al. (2023), our analysis not only extends to a broader class of mean and variance parameters of Gaussian smoothing for conditional vector fields, but also specifies the conditions on these parameters under which the almost minimax optimal convergence rate can be achieved.

The contributions of this paper are as follows.

- We establish that a widely used class of conditional FM methods achieves an almost minimax optimal convergence rate under the  $p$ -Wasserstein distance ( $1 \leq p \leq 2$ ), marking the first theoretical demonstration of such optimal performance of FM.
- We provide an analytical derivation of the convergence rate under various settings of the parameters, mean and variance, to make a path that connects a source and target point.
- We reveal that the variance parameter, which specifies the contribution of the source, must be decreased around the target at a specific rate to attain an almost minimax optimal convergence rate.

## 2 FLOW MATCHING

Throughout the paper, data are in the  $d$ -dimensional space  $\mathbb{R}^d$ . The  $d$ -dimensional normal distribution with mean  $\boldsymbol{\mu}$  and covariance matrix  $V$  is denoted by  $N_d(\boldsymbol{\mu}, V)$ . For a probability  $P_a$  with index  $a$ , the lowercase  $p_a$  denotes its probability density function (p.d.f.).

### 2.1 REVIEW OF FLOW MATCHING

This subsection provides a general review of FM, following Lipman et al. (2023) and Tong et al. (2024). The aim of FM is to generate samples from the true probability  $P_{true}$ . FM methods realize it by a flow  $\varphi_{[\tau]}(\mathbf{x})$  ( $\tau \in [0, 1]$ )<sup>1</sup> that maps a sample from the standard normal distribution  $N_d(0, I_d)$  to that of  $P_{true}$ . The flow  $\varphi_{[\tau]}(\mathbf{x})$  is defined by a solution to the ODE

$$\frac{d}{d\tau} \mathbf{x}_{[\tau]} = \mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}) \quad (\tau \in [0, 1])$$

given by a desired vector field  $\mathbf{v}_{[\tau]}$ . FM generates a sample by solving the ODE with an initial point  $\mathbf{x}_{[0]}$  from  $P_{[0]} = N_d(0, I_d)$ ; in other words, the distribution at time  $\tau$  is the pushforward  $P_{[\tau]} = \varphi_{[\tau]\#} P_{[0]}$ . The pushforward  $P_{[1]}$  is expected to approximate  $P_{true}$ . In practice, we need to construct the vector field given training data  $\{\mathbf{x}^i\}_{i=1}^n$  of size  $n$ , which is i.i.d. samples from  $P_{true}$ .

The relation between the vector field  $\mathbf{v}_{[\tau]}(\mathbf{x})$  and the p.d.f.  $p_{[\tau]}(\mathbf{x})$  is given by the *continuity equation*:

$$\frac{\partial}{\partial \tau} p_{[\tau]}(\mathbf{x}) + \operatorname{div}(p_{[\tau]}(\mathbf{x}) \mathbf{v}_{[\tau]}(\mathbf{x})) = 0.$$

Typically, a neural network (NN) is used to construct  $\mathbf{v}_{[\tau]}(\mathbf{x})$ . However, it is not obvious how to prepare the desired  $\mathbf{v}_{[\tau]}(\mathbf{x})$  to teach NN. In FM methods, conditional random vectors  $\mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{z})$  given  $\mathbf{z}$ , which are to be easily prepared, are used to teach NN; a location  $\mathbf{x}_{[\tau]}$  is sampled by a conditional probability  $P_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{z})$  and the vector  $\mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{z})$  is assigned at  $\mathbf{x}_{[\tau]}$  as teaching data.

<sup>1</sup>We use  $[\tau]$  to denote the time  $\tau \in [0, 1]$  in this section and preserve  $\mathbf{x}_t$  for the reverse time indexing, which is adopted from Section 4 to align with the notation of diffusion models.

{2}------------------------------------------------

Throughout this paper, the condition is given by  $\mathbf{z} = (\mathbf{x}_{[0]}, \mathbf{x}_{[1]})$  with  $\mathbf{x}_{[0]} \sim P_{[0]}$  and  $\mathbf{x}_{[1]} \sim P_{true}$ . The vector  $\mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{z})$  is made so that it satisfies the conditional continuity equation:

$$\frac{\partial}{\partial \tau} p_{[\tau]}(\mathbf{x}|\mathbf{z}) + \operatorname{div} (p_{[\tau]}(\mathbf{x}|\mathbf{z}) \mathbf{v}_{[\tau]}(\mathbf{x}|\mathbf{z})) = 0. \quad (1)$$

A typical construction of  $\mathbf{x}_{[\tau]}$  is to use a path  $\mathbf{x}_{[\tau]}$  ( $\tau \in [0, 1]$ ) from  $\mathbf{x}_{[0]}$  to  $\mathbf{x}_{[1]}$  and define the conditional vector by its time derivative  $\mathbf{v}_{[\tau]}(\mathbf{x}|\mathbf{z}) := \frac{d}{d\tau} \mathbf{x}_{[\tau]}$  (see Sec. 2.2). For a deterministic path,  $P_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{z})$  is the delta function at a point in the path  $\mathbf{x}_{[\tau]}$ .

Note that, given  $(\mathbf{x}, \tau)$ , the vector  $\mathbf{v}_{[\tau]}(\mathbf{x}|\mathbf{z})$  is random by the choice of  $\mathbf{z} = (\mathbf{x}_{[0]}, \mathbf{x}_{[1]})$ ; different vectors may be assigned to the same location  $(\mathbf{x}, \tau)$ . Most importantly, by averaging over  $\mathbf{z} \sim Q$ , where  $Q$  is the joint distribution with marginals  $P_{[0]}$  and  $P_{[1]}$ , we can see that the p.d.f. of  $\mathbf{x}$  at time  $\tau$ ,

$$p_{[\tau]}(\mathbf{x}) = \int p_{[\tau]}(\mathbf{x}|\mathbf{z}) dQ(\mathbf{z}), \quad (2)$$

and the averaged vector field  $\mathbf{v}_{[\tau]}(\mathbf{x})$  given by

$$\mathbf{v}_{[\tau]}(\mathbf{x}) := \int \mathbf{v}_{[\tau]}(\mathbf{x}|\mathbf{z}) p_{[\tau]}(\mathbf{z}|\mathbf{x}) d\mathbf{z}, \quad p_{[\tau]}(\mathbf{z}|\mathbf{x}) := \frac{p_{[\tau]}(\mathbf{x}|\mathbf{z}) q(\mathbf{z})}{p_t(\mathbf{x})} \quad (3)$$

satisfy the continuity equation

$$\frac{\partial}{\partial \tau} p_{[\tau]}(\mathbf{x}) + \operatorname{div} (p_{[\tau]}(\mathbf{x}) \mathbf{v}_{[\tau]}(\mathbf{x})) = 0. \quad (4)$$

This provides the theoretical basis for FM methods; the averaged vector field  $\mathbf{v}_{[\tau]}$  transports  $N_d(0, I_d)$  to  $P_{true}$ .  $\mathbf{v}_{[\tau]}$  is learned by a NN  $\phi(\mathbf{x}, \tau)$  with noisy training data  $\{(\mathbf{x}_{[\tau]}, \tau), \mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{z})\}$ . Empirically, the conditional  $\mathbf{v}_{[\tau]}(\mathbf{x}|\mathbf{z})$  is given by the random sample  $\mathbf{z} = (\mathbf{x}_{[0]}, \mathbf{x}^i)$  and the location  $\mathbf{x}_{[\tau]} \sim P_{[\tau]}(\mathbf{x}|\mathbf{z})$  (or path) with uniform  $\tau$ . The NN is trained with the mean square error (MSE):

$$\min_{\phi} \mathbb{E} \|\phi(\mathbf{x}_{[\tau]}, \tau) - \mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{z})\|^2. \quad (5)$$

Note that  $\phi(\mathbf{x}_{[\tau]}, \tau)$  does not depend on  $\mathbf{z}$ . Since the MSE minimizer is the conditional expectation of the teaching data, the empirical minimizer  $\hat{\phi}$  is an estimator of  $\mathbf{v}_{[\tau]}(\mathbf{x})$ . Using the estimator  $\hat{\phi}$  and the corresponding flow  $\hat{\mathbf{v}}_{[\tau]}$  given by ODE, we obtain the estimator  $\hat{P}_{[1]}$  for  $P_{true}$  by sampling. In practice, to reduce the variance of  $(\mathbf{x}_{[0]}, \mathbf{x}_{[1]})$  and simplify the ODE solution, the optimal transport for pairing  $\mathbf{x}_{[0]}$  and  $\mathbf{x}_{[1]}$  is applied effectively (Tong et al., 2024; Pooladian et al., 2023).

### 2.2 PATH CONSTRUCTION

This paper focuses on the following class of paths to construct the conditional vector field. Let  $\mathbf{x}_{[0]} \sim P_{[0]} = N_d(0, I_d)$ , and  $\mathbf{x}_{[1]}$  be a sample of  $P_{[1]}$  (or the empirical distribution  $\hat{P}_{train} = (1/n) \sum_{i=1}^n \delta_{\mathbf{x}^i}$  in practice). A conditional path is defined by

$$\mathbf{x}_{[\tau]} := \sigma_{[\tau]} \mathbf{x}_{[0]} + m_{[\tau]} \mathbf{x}_{[1]} \quad (0 \leq \tau \leq 1, \sigma_{[\tau]} > 0, m_{[\tau]} > 0). \quad (6)$$

We assume that  $\sigma_t$  and  $m_t$  are monotonic,  $\sigma_{[\tau]} \rightarrow 1, m_{[\tau]} \rightarrow 0$  as  $\tau \rightarrow 0^+$ , and  $\sigma_{[\tau]} \rightarrow 0, m_{[\tau]} \rightarrow 1$  as  $\tau \rightarrow 1^-$ . Let  $\sigma'_{[\tau]}$  ( $m'_{[\tau]}$ , resp.) be the time derivative of  $\sigma_{[\tau]}$  ( $m_{[\tau]}$ , resp.). With sampling  $\tau \sim \text{Unif}[0, 1]$ , a random conditional vector is assigned at  $(\mathbf{x}_{[\tau]}, \tau) \in \mathbb{R}^d \times [0, 1]$  by

$$\mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{x}_{[0]}, \mathbf{x}_{[1]}) := \sigma'_{[\tau]} \mathbf{x}_{[0]} + m'_{[\tau]} \mathbf{x}_{[1]}. \quad (7)$$

Note that, due to  $\mathbf{x}_{[0]} \sim N_d(0, I_d)$ , the distribution of  $\mathbf{x}_{[\tau]}$  given  $\mathbf{x}_{[1]}$  equals  $P_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{x}_{[1]}) = N_d(m_{[\tau]} \mathbf{x}_{[1]}, \sigma_{[\tau]}^2 I_d)$ , and thus we call  $m_{[\tau]}$  and  $\sigma_{[\tau]}^2$  the mean and variance parameters, respectively. Since (6) leads  $\mathbf{x}_{[0]} = (\mathbf{x}_{[\tau]} - m_{[\tau]} \mathbf{x}_{[1]})/\sigma_{[\tau]}$ , the conditional vector (7) is written as

$$\mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{x}_{[1]}) = \sigma'_{[\tau]} \frac{\mathbf{x}_{[\tau]} - m_{[\tau]} \mathbf{x}_{[1]}}{\sigma_{[\tau]}} + m'_{[\tau]} \mathbf{x}_{[1]}. \quad (8)$$

This class covers some popular constructions of conditional vector fields in the literature.

{3}------------------------------------------------

- **Affine path:** one of the most popular constructions is the following,

$$\mathbf{x}_{[\tau]} := (1 - \tau)\mathbf{x}_{[0]} + \tau\mathbf{x}_{[1]}, \quad \mathbf{v}_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{x}_{[1]}) = \mathbf{x}_{[1]} - \mathbf{x}_{[0]}.$$

This corresponds to  $m_{[\tau]} = \tau$  and  $\sigma_{[\tau]} = 1 - \tau$ . In Lipman et al. (2023)  $\mathbf{x}_{[0]}$  and  $\mathbf{x}_{[1]}$  are generated independently, while in Tong et al. (2024) they are taken by the optimal transport in a minibatch. This case is covered by our result, which does not depend on the construction of joint distribution.

- **Diffusion:** Lipman et al. (2023) presents the diffusion path, which corresponds to the deterministic probability flow (Song et al., 2020). The conditional density is given by  $p_{[\tau]}(\mathbf{x}_{[\tau]}|\mathbf{x}_{[1]} = \mathbf{y}) = N_d(m_{[\tau]}\mathbf{y}, \sigma_{[\tau]}^2 I_d)$ . The setting  $\sigma_{[\tau]}^2 = 1 - m_{[\tau]}^2$  and  $\sigma_{[\tau]} \sim \sqrt{1 - \tau}$  is typically used.

## 3 CONVERGENCE RATE OF FLOW MATCHING

We assume that the true density  $p_{[1]}$  is included in the Besov space  $B_{p',q'}^s$  ( $s > 0, 0 < p', q' \leq \infty$ ) on the unit cube  $[-1, 1]^d$ , while the results can be extended to any size straightforwardly. The parameter  $s$  specifies the degree of smoothness and is most relevant in this paper. The definition of the Besov space is deferred to Appendix A.1. We use the  $r$ -Wasserstein distance  $W_r$  to measure the accuracy of the estimator. The distance  $W_r$  of the probabilities  $P_1$  and  $P_2$  on  $\mathbb{R}^d$  is defined by

$$W_r(P_1, P_2) := (\inf_{Q \in \Gamma(P_1, P_2)} \int \|\mathbf{x}_1 - \mathbf{x}_2\|^r dQ(\mathbf{x}_1, \mathbf{x}_2))^{1/r}, \quad (9)$$

where  $\Gamma(P_1, P_2)$  denotes the joint distribution of  $(\mathbf{x}_1, \mathbf{x}_2)$  with marginals  $P_1$  and  $P_2$ . It is well known that  $W_r(P_1, P_2) \leq W_{r'}(P_1, P_2)$  holds for  $r' \geq r \geq 1$ .

As discussed in Sec. 3.1, to obtain an accurate estimator, we need to adopt early stopping of ODE and use  $\hat{P}_{[1-T_0]}$  with small  $T_0$ . Our aim is to derive a bound of  $W_p(\hat{P}_{[1-T_0]}, P_{true})$  for a large sample  $n \rightarrow \infty$ . The informal version of our main result is summarized in the following theorem.

**Theorem 1** (Informal). *Suppose that the target probability  $P_{[1]}$  has p.d.f.  $p_{[1]}$  in the Besov space  $B_{p',q'}^s$  ( $[-1, 1]^d$ ) of smoothness degree  $s$ , and that  $n$  training data  $\{\mathbf{x}^{(i)}\}_{i=1}^n$  are i.i.d. samples from  $P_{[1]}$ . Assume that  $\sigma_{[\tau]} \sim (1 - \tau)^\kappa$  ( $\tau \rightarrow 1^-$ ) with  $\kappa \geq 1/2$ , the conditional vector field is given by (6) and (7), and that time-divided neural networks are used (see Sec. 4.3). Then, under several assumptions, the FM estimator  $\hat{P}_{[1-T_0]}$  with  $T_0 = n^{-R_0}$  with appropriate  $R_0$  satisfies, for any  $\delta > 0$ ,*

$$\mathbb{E}[W_2(\hat{P}_{[1-T_0]}, P_{true})] = O\left(n^{-\frac{s+(2\kappa)\kappa-1-\delta}{2s+d}}\right) \quad (n \rightarrow \infty), \quad (10)$$

where  $\mathbb{E}$  denotes the expectation over the training data.

It is known that a lower bound of the minimax convergence rate exists for the Wasserstein distance for probability estimation. We use the notation  $\gtrsim$  to mean the lower bound up to a constant factor.

**Proposition 2** (Niles-Weed and Berthet (2022)). *Let  $p', q' \geq 1$ ,  $s > 0$ ,  $r \geq 1$ , and  $d \geq 2$ . Then,*

$$\inf_{\hat{P}} \sup_{p \in B_{p',q'}^s([-1, 1]^d)} \mathbb{E}[W_r(\hat{P}, P)] \gtrsim n^{-\frac{s+1}{2s+d}} \quad (n \rightarrow \infty),$$

where  $\hat{P}$  runs over all estimators based on  $n$  i.i.d. samples from  $P$ .

For  $\kappa = 1/2$ , by Theorem 1 and Proposition 2, the upper bound  $n^{-\frac{s+1-\delta}{2s+d}}$  is almost the optimal convergence rate up to an arbitrarily small  $\delta > 0$ . In addition, this convergence rate coincides with that of the diffusion model given in Oko et al. (2023) for  $W_1$ . The above result indicates that the flow matching is as good as the diffusion model regarding the minimax convergence rate under  $W_1$ , where the max in minimax means sup over the Besov space.

### 3.1 KERNEL DENSITY ESTIMATION AND EARLY STOPPING OF ODE

In practice, with conditional density  $P_{[\tau]}(\mathbf{x}|\mathbf{x}_{[1]}) = N_d(m_{[\tau]}\mathbf{x}_{[1]}, \sigma_{[\tau]}^2 I_d)$ , the parameter  $\sigma_{[1]}$  is often set as a small positive value  $\sigma_{[1]} = \sigma_{\min} > 0$  so that (7) is well defined up to  $\tau = 1$  (e.g. Lipman et al., 2023). If  $\mathbf{x}_{[1]}$  is sampled from  $\hat{P}_{train} = \frac{1}{n} \sum_{j=1}^n \delta_{\mathbf{x}^j}$ , the obtained distribution equals to

$$\hat{p}_{[1]}(\mathbf{x}) = \int p_{[1]}(\mathbf{x}|\mathbf{x}_{[1]}) d\hat{P}_{train}(\mathbf{x}_{[1]}) = \frac{1}{n} \sum_{j=1}^n \frac{1}{(2\pi\sigma_{\min}^2)^{d/2}} \exp\left(-\frac{\|\mathbf{x}-\mathbf{x}^j\|^2}{2\sigma_{\min}^2}\right),$$

{4}------------------------------------------------

which is exactly the kernel density estimator (KDE) with the Gaussian kernel of bandwidth  $\sigma_{\min}$ . If the ODE is solved up to  $\tau = 1$  rigorously, the pushforward realizes this KDE. As is well known (Scott, 1992), the convergence rate of this KDE under MSE is  $O(n^{-4/(4+d)})$  at best by choosing the optimal  $\sigma_{\min}$  depending on  $n$ , which is much slower than the optimal rate  $n^{-2s/(2s+d)}$  under MSE for the true density in  $B_{p',q'}^s(I^d)$  (Liu et al., 2023a). Based on this consideration, we discuss the early stopping of the ODE, where we stop at  $\tau = 1 - T_0$  with small  $T_0 > 0$  and consider the convergence rate of the estimator  $\hat{p}_{[1-T_0]}$ . Notice that  $\hat{p}_{[1-T_0]}$  differs from KDE, since it is given by the trained vector field. For diffusion models, Oko et al. (2023) and Zhang et al. (2024) also discuss the estimator obtained by stopping the reverse SDE at  $T_0 > 0$  to derive the convergence rate.

### 3.2 RELATED WORKS

Among many literatures on the statistical convergence of diffusion models, the most relevant to this work is Oko et al. (2023). Although our analysis is based on Oko et al. (2023) and derives comparable results, there are significant differences. First, we analyze the more general settings for  $m_{[\tau]}$  and  $\sigma_{[\tau]}$  in the conditional distribution  $P_{[\tau]}(\mathbf{x}|\mathbf{y}) = N_d(m_{[\tau]}\mathbf{y}, \sigma_{[\tau]}^2 I)$ ; Oko et al. (2023) considers only the case of  $\sigma_t \sim \sqrt{t}$  and  $m_t \sim 1 - t$  (in reverse time  $t$ ), which is a typical choice for diffusion models. Consequently, we have shown that for  $\sigma_{[\tau]} \sim (1 - \tau)^\kappa$  with  $\kappa \geq 1/2$ , only  $\kappa = 1/2$  achieves the almost minimax optimal convergence rate. Second, due to the difference between the ODE and diffusion processes, the proof technique for relating the Wasserstein metric and the  $L_2$ -risk is very different. Our technique is based on Alekseev-Gröbner lemma to derive the bound for  $r$ -Wasserstein with  $1 \leq r \leq 2$ , while Oko et al. (2023) obtained the bound only for 1-Wasserstein. Third, this is the first theoretical result for FM showing a convergence rate that is almost optimal. Although FM has been recently used in many applications with competitive results to diffusion models, theoretical comparisons in terms of convergence rates have been lacking. The results of this paper show that both FM and DM can attain the same almost minimax optimal convergence rate for generalization error.

For FM, there are some recent works on convergence. Albergo and Vanden-Eijnden (2023) and Benton et al. (2023b) relate the Wasserstein distance to the  $L_2$ -risk of the vector fields and show convergence for a large sample size, but did not derive a convergence rate. Jiao et al. (2024) discusses convergence rates of FM applied in the latent space of the autoencoder and considers the discretization effect of the numerical ODE solution in their analysis. However, they did not include the degree of smoothness in developing the convergence rate. Albergo et al. (2023b) present a unifying view of the theory of diffusion models and FM with the upper bounds of discrepancy measures.

## 4 THEORETICAL DETAILS

This section rigorously presents the main result with the assumptions and shows the proof outline. In the sequel, we use *reverse time index*  $t = 1 - \tau$  ( $\tau \in [0, 1]$ );  $t = 0$  for  $P_{true}$  and  $t = 1$  for  $N_d(0, I_d)$ , which align with the notations of the diffusion models. We use  $\text{poly}(\log n)$  to indicate the term of  $O(\log^r n)$ -order with some natural number  $r$ , and  $\tilde{O}(n^\alpha)$  to mean the order up to  $\text{poly}(\log n)$  factor.

### 4.1 PROBLEM SETTING AND ASSUMPTIONS

With reverse time  $t$ , the definitions (7), (2), and (3) are modified by replacing  $[1 - t]$  with  $t$ :

$$P_t = P_{[1-t]}, \quad P_0 = P_{true}, \quad P_1 = N_d(0, I_d).$$

The flows  $\varphi_t$  and  $\hat{\varphi}_t$  are defined by solving the ODE from  $t = 1$  in the reverse time direction:

$$\frac{d}{dt} \varphi_t(\mathbf{x}) = \mathbf{v}_t(\varphi_t(\mathbf{x})), \quad \frac{d}{dt} \hat{\varphi}_t(\mathbf{x}) = \hat{\mathbf{v}}_t(\varphi_t(\mathbf{x})), \quad (11)$$

where  $\mathbf{v}_t(\mathbf{x})$  and  $\hat{\mathbf{v}}_t(\mathbf{x})$  are the vector field (3) and its neural estimate, respectively. The distributions at  $t \in [0, 1]$  are given by

$$P_t = (\varphi_t)_\# P_1, \quad \hat{P}_t = (\hat{\varphi}_t)_\# P_1, \quad (12)$$

where  $(\varphi_t)_\#$  and  $(\hat{\varphi}_t)_\#$  denote the pushforward by the respective flows  $\varphi_t$  and  $\hat{\varphi}_t$ .

In the remainder of this paper,  $\delta > 0$  is an arbitrarily small positive value. As in Oko et al. (2023), we introduce  $N$  to specify the number of basis functions of the  $B$ -spline for approximating  $p_t(\mathbf{x})$

{5}------------------------------------------------

and  $v_t(\mathbf{x})$ . This number  $N$  depends on the sample size  $n$  ( $N = n^{\frac{d}{2s+d}}$  is used), balancing the approximation error and complexity of the  $B$ -spline and NN. We set the stopping time  $T_0 = N^{-R_0}$  as discussed in Sec. 3.1 ( $R_0$  is specified later), and solve the ODE from 1 to  $T_0$ . For simplicity, the  $d$  dimensional cube  $[-1, 1]^d$  and the reduced cube  $[-1 + N^{-(1-\kappa\delta)}, 1 - N^{-(1-\kappa\delta)}]^d$  are denoted by  $I^d$  and  $I_N^d$ , respectively, where  $\kappa > 0$  is specified below in (A3). We make the following assumptions.

- (A1) The target probability  $P_0$  has support  $I^d$  and its p.d.f.  $p_0$  satisfies  $p_0 \in B_{p',q'}^s(I^d)$  and  $p_0 \in B_{p',q'}^s(I^d \setminus I_N^d)$  with  $\tilde{s} > \max\{6s - 1, 1\}$ .
- (A2) There exists  $C_0 > 0$  such that  $C_0^{-1} \leq p_0(\mathbf{x}) \leq C_0$  for all  $\mathbf{x} \in I^d$ .
- (A3) There is  $\kappa \geq 1/2$ ,  $b_0 > 0$ ,  $\tilde{\kappa} > 0$ , and  $\tilde{b}_0 > 0$  such that

$$\sigma_t = b_0 t^\kappa, \quad 1 - m_t = \tilde{b}_0 t^{\tilde{\kappa}}$$

for sufficiently small  $t \geq T_0$ . Also, there are  $D_0 > 0$  and  $K_0 > 0$  such that

$$D_0^{-1} \leq \sigma_t^2 + m_t^2 \leq D_0, \quad |\sigma'_t| + |m'_t| \leq N^{K_0} \quad (\forall t \in [T_0, 1]).$$

- (A4) If  $\kappa = 1/2$ , there is  $b_1 > 0$  and  $D_1 > 0$  such that for any  $0 \leq \gamma < R_0$

$$\int_{T_0}^{N^{-\gamma}} \{(\sigma'_t)^2 + (m'_t)^2\} dt \leq D_1 (\log N)^{b_1}.$$

- (A5) There is a constant  $C_L > 0$  such that  $\|\frac{\partial}{\partial \mathbf{x}} \int \mathbf{y} p_t(\mathbf{y}|\mathbf{x}) d\mathbf{y}\|_{op} \leq C_L$  for any  $t \in [T_0, 1]$ .

The higher degree of smoothness is assumed around the boundary of  $I^d$  in (A1) for a technical reason to compensate for the nondifferentiability of  $p_0(\mathbf{x})$  at the boundary by (A2). In (A3), it may be more natural to require  $\sigma_t^2 + m_t^2 = 1$  so that signal power can be maintained. However, in this paper, to pursue the flexibility of choosing  $\sigma_t$  and  $m_t$ , we allow bounded changes of  $\sigma_t^2 + m_t^2$  over  $t$ . (A4) is required to limit the complexity of the neural network model (see Lemma 5). In (A3),  $\kappa$  is assumed to be not less than  $1/2$ , because for  $\kappa < 1/2$ , the integral  $\int_{T_0} (\sigma'_t)^2 dt$  with  $T_0 = N^{-R_0}$  diverges to infinity as  $N \rightarrow +\infty$ , which causes the divergence of the complexity bound in Lemma 5. Note that the boundary case  $\kappa = 1/2$  is, in fact, popularly used for the diffusion model. In this case,  $(\sigma'_t)^2$  is the order  $1/t$  for  $t \rightarrow 0^+$  and the integral from  $T_0$  is of the order  $\log N$ , which still diverges to infinity as  $n \rightarrow \infty$ . As discussed in Section 4.3, we consider this integral only for a short time interval, and we will see that the  $W_2$  distance converges to zero as  $n \rightarrow \infty$ . (A5) is made to bound the Lipschitz factor in Theorem 3 under (A3) (see Lemma 10).

### 4.2 GENERALIZATION BOUND

It is known (Albergo and Vanden-Eijnden, 2023; Benton et al., 2023b) that, given two vector fields, the  $W_2$ -distance of the pushforwards of the same distribution by the corresponding flows admits an upper bound by the  $L_2$ -risk of the vector fields;

**Theorem 3.** *Let  $v_t(\mathbf{x})$  and  $\hat{v}_t(\mathbf{x})$  be vector fields such that  $\mathbf{x} \mapsto \hat{v}_t(\mathbf{x})$  is  $L_t$ -Lipschitz for each  $t$ , and  $P_t$  and  $\hat{P}_t$  be the pushforwards of distribution  $P_0$  by the corresponding flows at time  $t$  from  $t = 0$ . Then, for any  $t \in [0, 1]$ , we have*

$$W_2(\hat{P}_t, P_t) \leq \sqrt{t} \left( \int_0^t \int e^{2 \int_s^t e^{L_u} du} \|\hat{v}_s(\mathbf{x}) - v_s(\mathbf{x})\|^2 dP_s(\mathbf{x}) d\mathbf{x} ds \right)^{1/2}. \quad (13)$$

See Appendix B for the proof. From Theorem 3, we can consider the  $L_2$ -error  $\mathbb{E}[\int \int \|\hat{v}(\mathbf{x}, s) - v(\mathbf{x}, s)\|^2 dP_s(\mathbf{x}) d\mathbf{x} ds]$  of the vector field to obtain the bound of the  $W_2$  distance of the distributions. From the fact  $W_r \leq W_{r'}$  ( $1 \leq r \leq r'$ ), the same upper bound holds for  $W_r$  for  $1 \leq r \leq 2$ .

We first review a general method for bounding the generalization. We consider training within the general time interval  $[T_l, T_u]$  where  $T_0 \leq T_l < T_u \leq 1$ . For an estimator  $\phi(\mathbf{x}, t)$  of the true vector field  $v_t(\mathbf{x})$ , we define the loss function  $\ell_\phi^{T_l, T_u}(\mathbf{x})$  for  $\mathbf{x} \in I^d$  by

$$\ell_\phi^{T_l, T_u}(\mathbf{x}) := \int_{T_l}^{T_u} \int \|\phi(\mathbf{x}_t, t) - v_t(\mathbf{x}_t|\mathbf{x})\|^2 p_t(\mathbf{x}_t|\mathbf{x}) d\mathbf{x}_t dt, \quad (14)$$

{6}------------------------------------------------

where  $\mathbf{x}$  is the condition of  $\mathbf{v}_t(\mathbf{x}_t|\mathbf{x})$ . Although the definition depends on  $T_\ell$  and  $T_u$ , we omit them when there is no confusion. Given the training data  $\{\mathbf{x}_i^{1:n}\}_{i=1}^n$ , the vector field is trained with the teaching data  $\mathbf{v}_t(\mathbf{x}_t|\mathbf{x}^i)$  at the location  $(\mathbf{x}_t, t)$  ( $t \in [T_\ell, T_u]$ ), which is sampled from  $p_t(\mathbf{x}_t|\mathbf{x}^i)$  and the uniform distribution  $U([T_\ell, T_u])$ . Note that given  $\mathbf{x}^i$ , we can generate any number of  $(\mathbf{x}_t, t)$ . Thus, the sampling error in (14) is negligible and the training by a NN can be regarded as minimization of

$$\frac{1}{n} \sum_{i=1}^n \ell_\phi(\mathbf{x}^{(i)}). \quad (15)$$

See Oko et al. (2023, Section 4) for the discussion of the effect of sampling.

Let  $\hat{\phi}$  be the minimizer of (15) among the function class  $\mathcal{S}$ . The generalization error is then given by

$$\mathcal{E}_{gen} := \mathbb{E} \left[ \int \int_{T_\ell}^{T_u} \|\hat{\phi}(\mathbf{x}, t) - \mathbf{v}_t(\mathbf{x})\|^2 p_t(\mathbf{x}|\mathbf{y}) dt d\mathbf{x} p_0(\mathbf{y}) d\mathbf{y} \right] = \mathbb{E} \left[ \int \ell_{\hat{\phi}}(\mathbf{y}) p_0(\mathbf{y}) d\mathbf{y} \right]. \quad (16)$$

Let  $\mathcal{L} := \{\ell_\phi \mid \phi \in \mathcal{S}\}$  and  $\mathcal{N}(\mathcal{L}, \|\cdot\|_{L^\infty(I^d)}, \varepsilon)$  be the covering number of the function class  $\mathcal{L}$  with the  $\|\cdot\|_{L^\infty(I^d)}$ -norm. Then, a standard argument on the generalization error analysis derives the following upper bound (see Oko et al. (Theorem C.4, 2023) and also Hayakawa and Suzuki (2020)).

**Theorem 4.** *The generalization error of the minimizer of (15) among  $\phi \in \mathcal{S}$  is upper bounded by*

$$\mathcal{E}_{gen} \leq 2 \inf_{\phi \in \mathcal{S}} \int \int_{T_\ell}^{T_u} \|\phi(\mathbf{x}, t) - \mathbf{v}_t(\mathbf{x})\|^2 p_t(\mathbf{x}) dt d\mathbf{x} + \frac{\sup_{\phi \in \mathcal{S}} \|\ell_\phi\|_{L^\infty(I^d)}}{n} \left( \frac{37}{9} \log \mathcal{N}(\mathcal{L}, \|\cdot\|_{L^\infty(I^d)}, \varepsilon) + 32 \right) + 3\varepsilon. \quad (17)$$

From Theorems 3 and 4, it suffices to consider the approximation error (1st term) and complexity (2nd term) in (17) for deriving the  $W_2$  distributional bound.

#### 4.2.1 COMPLEXITY TERM IN GENERALIZATION BOUND

We first consider the complexity term, where the class  $\mathcal{S}$  is given by NN. A class of NN  $\mathcal{M}(L, W, S, B)$  with height  $L$ , width  $W$ , sparsity constraint  $S$ , and norm constraint  $B$  is defined as

$$\mathcal{M}(L, W, S, B) := \{\psi_{A^{(1)}, b^{(1)}} \circ \cdots \circ \psi_{A^{(2)}, b^{(2)}}(A^{(1)}\mathbf{x} + b^{(1)}) \mid A^{(i)} \in \mathbb{R}^{W_{i+1} \times W_i}, b^{(i)} \in \mathbb{R}^{W_{i+1}}, \sum_{i=1}^L (\|A^{(i)}\|_0 + \|b^{(i)}\|_0) \leq S, \max_i \|A^{(i)}\|_\infty \vee \|b^{(i)}\|_\infty \leq B\},$$

where  $\psi_{A, b}(\mathbf{z}) = A \operatorname{ReLU}(\mathbf{z}) + b$ . As shown in Theorems 7 and 8 later, it suffices to consider the NNs that satisfy

$$\|\phi(\mathbf{x}, t)\|_\infty \leq D(|\sigma'_t| \sqrt{\log n} + |m'_t|)$$

for some constant  $D$ . Also, we can see in Lemma A.2 that  $\mathbf{x} \mapsto \mathbf{v}_t(\mathbf{x})$  is Lipschitz continuous with Lipschitz constant proportional to  $1/t$  under (A3) and (A5). Reflecting these facts, we define the following NN class for training the vector field:

$$\mathcal{H}_n := \{\phi \in \mathcal{M}(L, W, S, B) \mid \|\phi(\cdot, t)\|_\infty \leq D(|\sigma'_t| \sqrt{\log n} + |m'_t|) \text{ for } \forall t \in [T_0, 1], \\ \mathbf{x} \mapsto \phi(\mathbf{x}, t) \text{ is } L_t\text{-Lipschitz for each } t \in [T_0, 1] \text{ where } L_t = \tilde{C}_L/t\}, \quad (18)$$

where  $D$  and  $\tilde{C}_L$  are some positive constants.

The supremum norm and the covering number in Theorem 4 are given in the following lemmas.

**Lemma 5.** *Let  $T_0 \leq T_\ell < T_u \leq 1$ . Under Assumption (A4), there is  $C_s > 0$  such that*

$$\sup_{\phi \in \mathcal{H}_n} \|\ell_\phi\|_{L^\infty(I^d)} \leq C_s (\log n)^{b+1}, \quad (19)$$

where  $b = b_1$  in (A4) for  $\kappa = 1/2$ , and  $b = 0$  for  $\kappa > 1/2$ .

See Appendix C.1 for the proof. To obtain this bound, we need to impose the upper bound of  $\phi$  as in (18). In practice, the vectors in the teaching data satisfy this upper bound, and thus  $\phi$  will naturally satisfy the same bound by the least square error solution. The following bound of the covering number for neural networks is given by Suzuki (Lemma 3, 2019).

**Lemma 6.** *For the function class  $\mathcal{H}_n$ , the covering number satisfies*

$$\log \mathcal{N}(\mathcal{L}, \|\cdot\|_{L^\infty(I^d)}, \varepsilon) \leq SL \log (\varepsilon^{-1} \|W\|_\infty Bn).$$

{7}------------------------------------------------

#### 4.2.2 APPROXIMATION ERROR FOR SMALL $t$

Recall that  $N$  specifies the number of basis functions of the  $B$ -spline for the approximation. We derive upper bounds of the approximation error of the NN model  $\mathcal{M}(L, W, S, B)$ , where  $L, W, S$ , and  $B$  are specified in terms of  $N$ . We will separate  $[T_0, 1]$  into two intervals,  $[T_0, 3T_*]$  and  $[T_*, 1]$ , where  $T_* := N^{-(\kappa^{-1}-\delta)/d}$ , and provide different upper bounds. The reason for this choice of division point  $T_*$  is sketched as follows and is detailed in C.2. In the approximation of the vector field, we use the  $B$ -spline approximation of densities as in Oko et al. (2023). To show a fast convergence rate, the first interval is more subtle because  $p_t(x)$  is rougher. In approximating the density on the smoother boundary region, we divide the region into small cubes, each of which uses  $N^\delta$  bases for  $B$ -spline approximation. To make the total number of  $B$ -spline bases comparable with  $N$ , the width  $a_0$  of the region should be  $a_0 = N^{(1-\kappa\delta)/d}$ . On the other hand, in Theorem 7, we need a concentration of an integral around the boundary region for a better approximation by the higher smoothness, and this limits the variance of the Gaussian kernel so that  $\sigma_t = t^\kappa \leq a_0$ . This derives  $t \leq N^{-(\kappa^{-1}-\delta)/d}$ . As a result, the division point is small enough as  $T_* := N^{-(\kappa^{-1}-\delta)/d}$ .

The approximation bound for  $t \in [T_0, 3T_*]$  with  $T_* := N^{-\frac{\kappa^{-1}-\delta}{d}}$  is given in the following Theorem (see Appendix C.4 for the proof).

**Theorem 7.** *Under assumptions (A1)-(A5), there is a neural network  $\phi_1 \in \mathcal{M}(L, W, S, B)$  and a constant  $C_6$ , which is independent of  $t$ , such that, for sufficiently large  $N$ ,*

$$\int \|\phi_1(\mathbf{x}, t) - \mathbf{v}_t(\mathbf{x})\|^2 p_t(\mathbf{x}) d\mathbf{x} \leq C_6 \{(\sigma_t')^2 \log N + (m_t')^2\} N^{-\frac{2\kappa}{d}}, \quad (20)$$

for any  $t \in [T_0, 3T_*]$ , where  $L = O(\log^4 N)$ ,  $\|W\|_\infty = O(N \log^6 N)$ ,  $S = O(N \log^8 N)$ ,  $B = \exp(O(\log N \log \log N))$ . Additionally, we can take  $\phi_1$  to satisfy

$$\|\phi_1(\mathbf{x}, t)\| \leq \tilde{C}_6 \{(\sigma_t') \sqrt{\log n} + |m_t'|\},$$

where  $\tilde{C}_6$  is a constant independent of  $t$ .

#### 4.2.3 APPROXIMATION ERROR FOR LARGE $t$

We derive a bound of the approximation error on any interval  $[2t_*, 1]$ , where  $t_* \geq T_* = N^{-\frac{\kappa^{-1}-\delta}{d}}$ . This is used to discuss the optimal convergence rate in Section 4.3.

**Theorem 8.** *Fix  $t_* \in [T_*, 1]$  and take an arbitrary  $\eta > 0$ . Under Assumptions (A1)-(A5), there is a neural network  $\phi_2 \in \mathcal{M}(L, W, S, B)$  and  $C_7 > 0$ , which does not depend on  $t$ , such that the bound*

$$\int \|\phi_2(\mathbf{x}, t) - \mathbf{v}_t(\mathbf{x})\|^2 p_t(\mathbf{x}) d\mathbf{x} \leq C_7 \{(\sigma_t')^2 \log N + (m_t')^2\} N^{-\eta} \quad (21)$$

holds for all  $t \in [2t_*, 1]$ , and the NN model satisfies  $L = O(\log^4 N)$ ,  $\|W\|_\infty = O(N)$ ,  $S = O(t_*^{-d\kappa} N^{d\kappa})$ , and  $B = \exp(O(\log N \log \log N))$ . Moreover,  $\phi_2$  can be taken so that

$$\|\phi_2(\cdot, t)\|_\infty \leq \tilde{C}_7 \{(\sigma_t') \log N + |m_t'|\}$$

with constant  $\tilde{C}_7 > 0$  independent of  $t$ .

See Appendix C.5 for the proof. The approximation error  $N^{-\eta}$  is arbitrarily small and is not dominant, while  $S$  may be dominant in the complexity term.

### 4.3 CONVERGENCE RATE UNDER WASSERSTEIN DISTANCE

We can consider a generalization bound based on Theorems 3, 4, 7, and 8 deriving the bounds for  $[T_0, 2T_*]$  and  $[2T_*, 1]$ . However, if we apply Theorem 8 for  $[2t_*, 1]$  with  $t_* = T_* = N^{-(\kappa^{-1}-\delta)/d}$ , the dominant factor of the log covering number in (17) is the sparsity  $S = O(t_*^{-d\kappa} N^{d\kappa}) = O(N)$ . From Theorems 3 and 4, the complexity part gives  $O((N/n)^{1/2})$  term in the  $W_2$  generalization error. If we plug  $N = n^{(2s+d)/d}$ , which is optimal for the MSE generalization, we have  $O(n^{-s/(2s+d)})$  for the upper bound of  $W_2$  generalization, which is slower than the lower bound in Proposition 2. To

{8}------------------------------------------------

achieve a better convergence rate, we will make use of the factor  $\sqrt{t}$  in front of (13) by dividing the interval  $[T_0, 1]$  into small pieces and using a NN for each small interval, as in Oko et al. (2023) for diffusion models.

Notice that, when time  $t$  is far from 0, the convolution of  $p_t(\mathbf{x}|\mathbf{y})$  with larger  $\sigma_t$  results in a smoother target vector field  $\mathbf{v}_t(\mathbf{x})$ , which is easier to approximate with a low-complexity model. On the other hand, when  $t$  approaches 0, with the fixed number of  $B$ -spline bases  $N$ , the approximation error bound  $\{(\sigma'_t)^2 \sqrt{\log N} + (m'_t)^2\} N^{-2s/d}$  can increase for large  $\sigma'_t$  or  $m'_t$  (e.g.  $\sigma_t \sim t^\kappa$  with  $\kappa < 1$ ). We therefore need a more complex model (that is, larger  $N$ ) than the one needed for larger  $t$ . Thus, it will be more efficient if the number of  $B$ -spline bases  $N$ , which controls the approximation error and complexity, is chosen adaptively to the time region  $t$ .

Specifically, we make a partition  $T_0 = t_0 < t_1 < t_2 < \dots < t_K = 1$  such that  $t_j = 2t_{j-1}$  for  $1 \leq j \leq K-1$  with  $2t_{K-1} \geq 1$ , and build a neural network for each  $[t_{j-1}, t_j]$  ( $j = 1, \dots, K$ ). Note that we train each network for interval  $[t_{j-1}, t_j]$  with  $n$  training data  $(\mathbf{x}_i)_{i=1}^n$ . We assume that  $t_j$  with  $T_* \leq t_j \leq 3T_*$  serves as the boundary to apply two different error bounds. The total number of intervals  $K$  is  $O(\log N) = O(\log n)$ , since  $2^K T_0 \geq 1$  with  $T_0 = N^{-R_0}$  can be achieved by  $K \geq R_0 \log_2 N$ . The constant  $R_0$  is fixed as  $R_0 \geq \frac{s+1}{\min(\kappa, \bar{\kappa})}$  so that  $W_2(P_{T_0}, P_0)$  is negligible (see the proof sketch of Theorem 9). In this setting, we have the following main result.

**Theorem 9** (Main result). *Assume (A1)-(A5) and  $d \geq 2$ . If the above time-partition is applied and a neural network is trained for each time division, for arbitrarily small  $\delta > 0$  and  $1 \leq r \leq 2$ , we have*

$$\mathbb{E}[W_r(\hat{P}_{T_0}, P_{\text{true}})] = O\left(n^{-\frac{s+(Q_0)^{-1}-1-\delta}{2s+d}}\right) \quad (n \rightarrow \infty). \quad (22)$$

*Proof Sketch.* Let  $J_j := [t_{j-1}, t_j]$  ( $j = 1, \dots, K$ ). We use a smaller neural network model for larger  $j$ . Specifically, the number of  $B$ -spline bases for  $J_j$  is  $N'_j := t_{j-1}^{-d\kappa} N^{\delta\kappa}$  for  $j > j_*$ , while  $N'_j = N$  for  $j \leq j_*$ , where  $j_*$  is defined as above. Note that  $N'_j \rightarrow \infty$  as  $N \rightarrow \infty$  due to  $\delta > 0$ , and that  $N'_j \leq N^{1-\delta\kappa} N^{\delta\kappa} = N$  for  $j \geq j_*$  due to  $t_j \geq N^{-\frac{\kappa-1-\delta}{d}}$ , which means a lower complexity. See also Figure E.1 in Appendix.

Next, we consider the bound of  $W_2$ -distance based on the partition. For each of  $j = 1, \dots, K$ , we introduce a vector field  $\tilde{\mathbf{v}}_t^{(j)}$  such that it coincides with the target  $\mathbf{v}_t$  for  $t \in [t_j, 1]$  and with the learned  $\hat{\mathbf{v}}_t$  for  $t \in [T_0, t_j]$ . Let  $Q^{(j)}$  be the pushforward from  $P_1 = N_d(0, I_d)$  to  $t = T_0$  by the flow of the vector field  $\tilde{\mathbf{v}}_t^{(j)}$ . Then,  $Q^{(0)} = P_{T_0}$ , the pushforward by the flow of the target  $\mathbf{v}_t$  from  $t = 1$  to  $T_0$ , and also  $Q^{(K)} = \hat{P}_{T_0}$ . Note also that  $\tilde{\mathbf{v}}^{(j)}$  and  $\tilde{\mathbf{v}}^{(j-1)}$  differ only in  $J_j$  by  $\mathbf{v}_t(\mathbf{x}) - \hat{\mathbf{v}}_t(\mathbf{x})$ . Therefore, from Theorem 3 and the Lipschitz assumption on  $\mathcal{H}_n$ , we have

$$\begin{aligned} W_2(P_0, \hat{P}_{T_0}) &\leq W_2(P_0, P_{T_0}) + \sum_{j=1}^K W_2(Q^{(j-1)}, Q^{(j)}) \\ &\leq \theta_n + C \sum_{j=1}^K \sqrt{t_j} \left\{ \int_{t_{j-1}}^{t_j} e^{2 \int_t^{t_j} (\tilde{C}/u) du} \int \|\hat{\phi}(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t)\|^2 p_t(\mathbf{x}) d\mathbf{x} dt \right\}^{1/2}, \end{aligned}$$

where  $\theta_n^2 = dt_0^2 n^{-\frac{2R_0\kappa}{s+d}}$  and  $\int \|\mathbf{y}\|^2 dP_0(\mathbf{y}) \tilde{b}_0^2 n^{-\frac{2R_0\kappa}{s+d}}$ , which is derived from Lemma 11 and (A3). We take a constant  $R_0 \geq (s+1)/\min(\kappa, \bar{\kappa})$  so that  $\theta_n$  is of  $O(n^{-\frac{s+1}{s+d}})$  and thus  $\theta_n$  is negligible. It is easy to see that the factor  $e^{\int_t^{t_j} (\tilde{C}/u) du}$  is bounded by a constant because of  $t_j = 2t_{j-1}$  by definition.

For simplicity, let  $t_* := t_{j_*}$ . From Theorems 7, 8, and 4, the generalization bound of  $\hat{P}_{T_0}$  is given by

$$\begin{aligned} &\mathbb{E} \left[ W_2(P_0, \hat{P}_{T_0}) \right] \\ &\leq \theta_n + \sum_{j=1}^{j_*} \sqrt{t_*} \left\{ C_6 \int_{t_{j-1}}^{t_j} \{(\sigma'_t)^2 \log N + (m'_t)^2\} N^{-2s/d} dt + \frac{N}{n} O(\text{poly}(\log n)) \right\}^{1/2} \\ &\quad + \sum_{j=k_*}^K \sqrt{t_j} \left\{ C_7 \int_{t_{j-1}}^{t_j} \{(\sigma'_t)^2 \log N + (m'_t)^2\} N^{-\eta} dt + \frac{t_j^{-d\kappa} N^{\delta\kappa}}{n} O(\text{poly}(\log n)) \right\}^{1/2} \\ &\leq \theta_n + C''' \sqrt{t_*} t_*^{-1/2} N^{-s/d} O(\text{poly}(\log n)) + C''' \sqrt{\frac{N}{n}} O(\text{poly}(\log n)) \\ &\quad + C''' \sum_{j=k_*}^K \left\{ \sqrt{t_j} N^{-\eta/2} O(\text{poly}(\log n)) + \sqrt{t_j} \frac{t_j^{-d\kappa/2} N^{\delta\kappa/2}}{\sqrt{n}} O(\text{poly}(\log n)) \right\} \end{aligned}$$

{9}------------------------------------------------

$$\leq \theta_n + C''' t_*^\kappa n^{-s/(2s+d)} O(\text{poly}(\log n)) + C''' \sqrt{t_*} n^{-s/(2s+d)} O(\text{poly}(\log n)) \\ + C''' \sum_{j=k_*}^K \left\{ \sqrt{t_j} N^{-n/2} O(\text{poly}(\log n)) + t_j^{\frac{1-d\kappa}{2}} n^{\frac{d\delta\kappa}{2(2s+d)} - \frac{1}{2}} O(\text{poly}(\log n)) \right\}.$$

In the third inequality, we use  $\int_{t_{j-1}}^{t_j} \{(\sigma_t')^2 + (m_t')^2\} dt = O(\text{poly}(\log n))$  for  $\kappa = 1/2$ , and the fact that it is bounded by a constant for  $\kappa > 1/2$ . Since  $\eta$  is arbitrarily large and  $\kappa \geq 1/2$ , neglecting the factors of  $\text{poly}(\log n)$ , the candidates of the dominant terms in the above expression are  $t_*^{1/2} n^{-\frac{1}{2s+d}}$  in the third term and  $t_*^{\frac{1-d\kappa}{2}} n^{\frac{d\delta\kappa}{2(2s+d)} - \frac{1}{2}}$  in the last summation. By balancing these two terms, the upper bound can be minimized by setting

$$t_* = C_* n^{-\frac{\kappa-1-\delta}{2s+d}}, \quad (23)$$

for some contact  $C_*$ , and the dominant term of the upper-bound is given by

$$\tilde{O}\left(n^{-\frac{s+(2\kappa)^{-1}-\delta/2}{2s+d}}\right). \quad (24)$$

This proves the claim by replacing  $\delta/2$  with  $\delta$  and absorbing the  $\text{poly}(\log)$  factor in  $\delta$ .  $\square$

From Proposition 2 and Theorem 9, if  $\kappa = 1/2$ , the FM method achieves an almost optimal rate up to the  $\text{poly}(\log n)$  factor and arbitrary small  $\delta > 0$ . On the other hand, for  $\kappa > 1/2$ , the obtained upper bound is not optimal. This suggests that the choice of  $\sigma_t \sim \sqrt{t}$  around  $t \rightarrow 0^+$  is theoretically reasonable. This is also a popular choice for the diffusion model.

### 4.4 DISCUSSION

In the derivation of the almost minimax optimal convergence rate, the use of neural networks for divided time intervals is a limitation of the current theoretical analysis. As discussed before Theorem 9, without this partition, the current analysis gives only  $\tilde{O}(n^{-\frac{1}{2s+d}})$ , which is not optimal for  $W_2$ . It is obviously an important question of how to avoid such a time division. In Oko et al. (2023), the optimal convergence rate for the diffusion model has been proved without time division for the total variation, which is  $\tilde{O}(n^{-\frac{1}{2s+d}})$ . The bound is based on Girsanov's theorem, which gives an upper bound of the KL divergence of SDE by the  $L_2$  losses of the drift estimation. To the best of our knowledge, no bounds are known for ODE with respect to the KL or total variation for the difference of vector fields. This is an important future direction for understanding the ability of FM theoretically.

## 5 CONCLUSION

This paper has rigorously analyzed the convergence rate of flow matching, demonstrating for the first time that FM can achieve the almost minimax optimal convergence rate under the 2-Wasserstein distance. This result positions FM as a competitive alternative to diffusion models in terms of asymptotic convergence rates, which concurs with empirical results in various applications. Our findings further reveal that the convergence rate is significantly influenced by the variance decay rate in the Gaussian conditional kernel, where  $\sigma_t \sim \sqrt{t}$  is shown to yield the optimal rate. Although there are several popular proposals for the mean and variance functions, theoretical justification or comparison has not been explored intensively. The current result on the upper bound (Theorem 9) provides theoretical insight on the influence of the choice of these functions.

Although this study offers substantial theoretical contributions, these insights are still grounded in specific modeling assumptions that limit broader applicability. In addition to the time-partition discussed in Sec. 4.4, this paper focuses primarily on assumptions utilizing Gaussian conditional kernels. However, other FM implementations might employ different path constructions, as suggested by recent proposals Kerrigan et al. (2023); Isobe et al. (2024). The theoretical implications of these alternative approaches remain an essential area for future research.

## ACKNOWLEDGMENTS

This work has been supported in part by JST CREST JPMJCR2015 and JSPS Grant-in-Aid for Transformative Research Areas (A) 22H05106.

 Rest of paper (reference and Appendix) is removed.