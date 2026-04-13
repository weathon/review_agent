

{0}------------------------------------------------

# --- MEDIAN CLIPPING FOR ZEROTH-ORDER NON-SMOOTH CONVEX OPTIMIZATION AND MULTI-ARMED BANDIT PROBLEM WITH HEAVY-TAILED SYMMETRIC NOISE

**Anonymous authors**

Paper under double-blind review

## ABSTRACT

In this paper, we consider non-smooth convex optimization with a zeroth-order oracle corrupted by symmetric stochastic noise. Unlike the existing high-probability results requiring the noise to have bounded  $\kappa$ -th moment with  $\kappa \in (1, 2]$ , our results allow even heavier noise with any  $\kappa > 0$ , e.g., the noise distribution can have unbounded expectation. Our convergence rates match the best-known ones for the case of the bounded variance, namely, to achieve function accuracy  $\varepsilon$  our methods with Lipschitz oracle require  $\tilde{O}(d^2\varepsilon^{-2})$  iterations for any  $\kappa > 0$ . We build the median gradient estimate with bounded second moment as the mini-batched median of the sampled gradient differences. We apply this technique to the stochastic multi-armed bandit problem with heavy-tailed distribution of rewards and achieve  $\tilde{O}(\sqrt{dT})$  regret. We demonstrate the performance of our zeroth-order and MAB algorithms for different  $\kappa$  on synthetic and real-world data. Our methods do not lose to SOTA approaches and dramatically outperform them for  $\kappa \leq 1$ .

## 1 INTRODUCTION

During the recent few years, stochastic optimization problems with heavy-tailed noise received a lot of attention from many researchers. In particular, heavy-tailed noise is observed in various problems, such as the training of large language models [3; 44], generative adversarial networks [13; 14], finance [35], and blockchain [43]. One of the most popular techniques for handling heavy-tailed noise in theory and practice is the gradient clipping [15; 6; 31; 34] which allows deriving high-probability bounds and considerably improves convergence even in case of light tails [37].

However, most of the mentioned works focus on the gradient-based (first-order) methods. For some problems, e.g., the multi-armed bandit [10; 1; 23; 4], only losses or function values are available, and thus, zeroth-order algorithms are required. Stochastic zeroth-order optimization is being actively studied. For a detailed overview, see the recent survey [11] and the references therein. The only existing works that handle heavy-tailed noise in convex zeroth-order optimization are [19; 20] which combine clipping and gradient smoothing [12] techniques. Under noise with bounded  $\kappa$ -th moment for  $\kappa \in (1, 2]$ , the authors obtain optimal high-probability convergence for  $d$ -dimensional non-smooth convex problems, i.e., function accuracy  $\varepsilon$  is achieved in  $\tilde{O}(\sqrt{d\varepsilon^{-1}})^{\frac{\kappa+1}{\kappa-1}}$  oracle calls. These rates match the optimal rates for first-order optimization [15], however, they degenerate as  $\kappa \rightarrow 1$ , and the convergence is not guaranteed for  $\kappa = 1$ .

For symmetric (and close to symmetric) heavy-tailed noise distributions, the degeneration issue can be handled via median estimators [46; 34], which are frequently used in robust mean estimation and robust machine learning [27]. In the case of first-order methods, the authors of [34] achieve better complexity guarantees and show that the narrowing of the distributions' class is essential for it. However, the possibility of application of the median estimators to the case of the zeroth-order optimization and multi-armed bandit remains open. In this paper, we address this question.

### 1.1 CONTRIBUTIONS

**Theory 1.** We propose our novel theoretical zeroth-order oracle (Assumption 4) that allows us to incorporate fine-grained features of the noise probability distributions. We use it to successfully utilize symmetry of the heavy-tailed noise and dramatically improve current convergence results.

{1}------------------------------------------------

054  
055  
056  
057  
058  
059  
060  
061  
062  
063  
064  
065  
066  
067  
068  
069  
070  
071  
072  
073  
074  
075  
076  
077  
078  
079  
080  
081  
082  
083  
084  
085  
086  
087  
088  
089  
090  
091  
092  
093  
094  
095  
096  
097  
098  
099  
100  
101  
102  
103  
104  
105  
106  
107

Table 1: Number of successive iterations to achieve a function’s accuracy  $\varepsilon$  with high probability; unconstrained optimization via Lipschitz oracle with bounded  $\kappa$ -th moment. Constants  $b, d, M_2'$  denote the batch size, dimensionality and the Lipschitz constant of the oracle, respectively.

|                   | ZO-clipped-SSTM [20]<br>$\kappa > 1$ , $b$ oracle calls per iter.                                                                                                                             | ZO-clipped-med-SSTM (this work)<br>$\kappa > 0$ , symmetric noise, $\frac{b}{\kappa}$ calls                                                                  |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Convex            | $\tilde{\mathcal{O}} \left( \max \left\{ \frac{d^{\frac{1}{2}} M_2'}{\varepsilon}, \frac{1}{b} \left( \frac{\sqrt{d} M_2'}{\varepsilon} \right)^{\frac{\kappa}{\kappa-1}}} \right\} \right)$  | $\tilde{\mathcal{O}} \left( \max \left\{ \frac{d^{\frac{1}{2}} M_2'}{\varepsilon}, \frac{1}{b} \left( \frac{d M_2'}{\varepsilon} \right)^2 \right\} \right)$ |
| $\mu$ -str. conv. | $\tilde{\mathcal{O}} \left( \max \left\{ \frac{d^{\frac{1}{2}} M_2'}{\varepsilon}, \frac{1}{b} \left( \frac{d(M_2')^2}{\mu\varepsilon} \right)^{\frac{\kappa}{2(\kappa-1)}} \right\} \right)$ | $\tilde{\mathcal{O}} \left( \max \left\{ \frac{d^{\frac{1}{2}} M_2'}{\varepsilon}, \frac{1}{b} \frac{d^2(M_2')^2}{\mu\varepsilon} \right\} \right)$          |

**Theory II.** We propose our novel ZO-clipped-med-SSTM (§3.2) for unconstrained optimization and ZO-clipped-med-SMD (§3.3) for optimization on convex compact which successfully incorporate median clipping technique. For any symmetric heavy-tailed noise with bounded  $\kappa$ -th moment  $\kappa > 0$ , our methods achieve non degenerating convergence rates with high-probability which match the optimal rates for ZO minimization under any noise with the bounded variance. In the Table 1, we provide convergence guarantees for the unconstrained case.

**Theory III.** We propose Clipped-INF-med-SMD (§4) for the stochastic multi-armed bandit (MAB) with symmetric heavy-tailed reward distribution. For MAB with  $d$  arms and time interval  $T$ , in Theorem 3, we obtain the  $\tilde{\mathcal{O}}(\sqrt{dT})$  bound on the regret, which is optimal and matches the lower bound  $\Omega(\sqrt{dT})$  for stochastic MAB with any reward distribution and bounded variance. Moreover, this bound holds not only in expectation but with controlled large deviations.

**Practice.** We demonstrate in the series of experiments (§5) on extremely noised real and synthetic data superior performance of our methods in comparison with previously known SOTA approaches.

We compare our algorithms with previous approaches and discuss its limitations in §6.

## 2 PRELIMINARIES

In this section, we introduce general notations and assumptions on optimized functions. We also recall popular gradient smoothing and clipping techniques.

**Notations.** For vector  $x \in \mathbb{R}^d$  and  $p \in [1, 2]$ , we define  $p$ -norm as  $\|x\|_p \stackrel{\text{def}}{=} \left( \sum_{i=1}^d |x_i|^p \right)^{\frac{1}{p}}$  and its dual norm as  $\|x\|_q$ , where  $\frac{1}{p} + \frac{1}{q} = 1$ . In the case  $q = \infty$ , we define  $\|x\|_\infty = \max_{i=1, \dots, d} |x_i|$ .

We denote the Euclidean unit ball  $B_2^d \stackrel{\text{def}}{=} \{x \in \mathbb{R}^d : \|x\|_2 \leq 1\}$ , the Euclidean unit sphere  $S_2^d \stackrel{\text{def}}{=} \{x \in \mathbb{R}^d : \|x\|_2 = 1\}$  and the probability simplex  $\Delta_+^d \stackrel{\text{def}}{=} \{x \in \mathbb{R}_+^d : \sum_{i=1}^d x_i = 1\}$ .

Median operator  $\text{Median}(\{a_i\}_{i=1}^{2m+1})$  applied to the elements sequence of the odd size  $2m+1$ ,  $m \in \mathbb{N}$  returns  $m$ -th order statistics. We also use short notation for max operator, i.e.  $a \vee b \stackrel{\text{def}}{=} \max(a, b)$ .

**Assumption 1** (Strong convexity). *The function  $f : \mathbb{R}^d \rightarrow \mathbb{R}$  is  $\mu$ -strongly convex, if there exists  $\mu \geq 0$  such that for all  $x_1, x_2 \in \mathbb{R}^d$  and  $\lambda \in [0, 1]$  :*

$$f(\lambda x_1 + (1 - \lambda)x_2) \leq \lambda f(x_1) + (1 - \lambda)f(x_2) - \frac{1}{2}\mu\lambda(1 - \lambda)\|x_1 - x_2\|_2^2,$$

If  $\mu = 0$  we say that the function is just “convex”.

**Assumption 2** (Lipschitz continuity). *The function  $f : \mathbb{R}^d \rightarrow \mathbb{R}$  is  $M_2$ -Lipschitz continuous w.r.t. the Euclidean norm, if there exists  $M_2 > 0$ , such that for all  $x_1, x_2 \in \mathbb{R}^d$  :*

$$|f(x_1) - f(x_2)| \leq M_2 \|x_1 - x_2\|_2.$$

If a differentiable function has  $L$ -Lipschitz gradient, we call it  $L$ -smooth.

**Randomized smoothing.** The main scheme that allows us to develop gradient-free methods for non-smooth convex problems is randomized smoothing [9; 12; 29; 30; 40]. For the fixed smoothing

{2}------------------------------------------------

parameter  $\tau > 0$ , we build a smooth approximation  $\hat{f}_\tau$  for a non-smooth  $f : \mathbb{R}^d \rightarrow \mathbb{R}$  as:

$$\hat{f}_\tau(x) \stackrel{\text{def}}{=} \mathbb{E}_u[f(x + \tau u)], \quad (1)$$

where  $u \sim U(B_2^d)$  is a random vector uniformly distributed on the Euclidean unit ball  $B_2^d$ .

If the function  $f$  is  $\mu$ -strongly convex (As 1) and  $M_2$ -Lipschitz (As 2), then the smoothed function  $\hat{f}_\tau$  is  $\mu$ -strongly convex and  $\sqrt{d}M_2/\tau$ -smooth. Moreover, it does not differ from the original  $f$  too much, namely, (See Lemma 2 from Appendix B.1)

$$\sup_{x \in \mathbb{R}^d} |\hat{f}_\tau(x) - f(x)| \leq \tau M_2. \quad (2)$$

**Clipping.** To handle heavy-tailed noise, we use a clipping technique which clips tails of gradient's distribution. For the clipping level  $\lambda > 0$  and  $q$ -norm, where  $q \in [2, +\infty]$ , we define the clipping operator  $\text{clip}$  for arbitrary non-zero gradient vector  $g \in \mathbb{R}^d$  as follows:

$$\text{clip}_q(g, \lambda) = \frac{g}{\|g\|_q} \min(\|g\|_q, \lambda).$$

## 3 ZEROTH-ORDER OPTIMIZATION WITH SYMMETRIC HEAVY-TAILED NOISE

In this section, we present our novel algorithms for zeroth-order optimization with independent and Lipschitz oracles. In subsection 3.1, we introduce the problem, symmetric heavy-tailed noise assumptions and median estimation with its properties. In subsection 3.2, we propose our accelerated batched ZO-clipped-med-SSTM for unconstrained problems. In subsection 3.3, we describe our ZO-clipped-med-SMD for problems on convex compacts. All proofs are located in Appendix B.

### 3.1 THEORY

We consider a non-smooth convex optimization problem on a convex set  $Q \subseteq \mathbb{R}^d$ :

$$\min_{x \in Q} f(x), \quad (3)$$

where  $f : \mathbb{R}^d \rightarrow \mathbb{R}$  is  $d$ -dimensional,  $\mu$ -strongly convex (As 1) and  $M_2$ -Lipschitz (As 2) function. A point  $x^*$  denotes one of the problem's solutions. In zeroth-order setup, the optimization is performed only by accessing the pairs of function evaluations rather than sub-gradients.

**Two-point oracle.** For any two points  $x, y \in \mathbb{R}^d$ , an oracle returns the pair of the scalar values  $f(x, \xi)$  and  $f(y, \xi)$ , which are noised evaluation of real values  $f(x)$  and  $f(y)$ . Moreover, noised values have the same realization of the stochastic variable  $\xi$  and can be written as

$$f(x, \xi) - f(y, \xi) = f(x) - f(y) + \phi(\xi|x, y),$$

where  $\phi(\xi|x, y)$  is the stochastic noise, whose distribution depends on points  $x, y$ .

#### 3.1.1 NOISE DISTRIBUTION.

We propose our novel assumption on distribution of  $\phi(\xi|x, y)$ , induced by a random variable  $\xi$ . It allows us to introduce symmetry and heavy-tailed noise with bounded up to  $\kappa$ -th moments,  $\kappa > 0$ .

**Assumption 3** (Symmetric noise distribution). *Symmetry.* For any two points  $x, y \in \mathbb{R}^d$ , noise  $\phi(\xi|x, y)$  has symmetric probability density  $p(u|x, y)$ , i.e.  $p(u|x, y) = p(-u|x, y), \forall u \in \mathbb{R}$ .

**Heavy tails.** We assume that there exist  $\kappa > 0, \gamma > 0$  and scale function  $B(x, y) : \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$ , such that  $\forall u \in \mathbb{R}$  holds

$$p(u|x, y) \leq \frac{\gamma^\kappa \cdot |B(x, y)|^\kappa}{|B(x, y)|^{1+\kappa} + |u|^{1+\kappa}}. \quad (4)$$

We consider two possible oracles:

**Independent oracle:**  $\phi(\xi|x, y)$  distribution doesn't depend on points  $x, y$ , i.e.,

$$\gamma \cdot B(x, y) \equiv \Delta. \quad (5)$$

**Lipschitz oracle:**  $\phi(\xi|x, y)$  distribution becomes more concentrated around 0 as  $x, y$  become closer:

$$|\gamma \cdot B(x, y)| \leq \Delta \cdot \|x - y\|_2, \quad (6)$$

where  $\Delta > 0$  is the noise Lipschitz constant.

{3}------------------------------------------------

This assumption covers a majority of symmetric absolutely continuous distributions with bounded up to  $\kappa$ -th moments. For example (Remark 5), if  $\xi$  has Cauchy distribution, then one can use

- Independent oracle:  $f(x, \xi) = f(x) + \xi_x, f(y, \xi) = f(y) + \xi_y$  with independent  $\xi_x, \xi_y$ .
- Lipschitz oracle:  $f(x, \xi) = f(x) + \langle \xi, x \rangle, f(y, \xi) = f(y) + \langle \xi, y \rangle$ , where  $\xi$  is  $d$ -dimensional random vector. Oracle gives the same realization of  $\xi$  for both  $x$  and  $y$ .

**Comparison with previous oracles.** Our Assumption 3 is quite different from the standard assumptions from [8; 20]. We make our assumption on variable  $\phi(\xi|x, y)$  with fixed  $x, y$ . It allows us to set and use fine-grained properties of the noise distribution, e.g., symmetry or heavy tails of particular type (4). In [20], the authors fix  $\xi$  and make assumption on  $x, y$ . Hence, they can not access the distribution of the noise and use only the fact of having bounded  $\kappa$ -th moment. Nevertheless, when  $\kappa \in (1, 2]$ , our Assumption 3 can be reduced to the standard one with the same constant, Remark 3.

We would like to highlight the fact that the common proof techniques from previous works can not be trivially generalized to apply symmetry without our novel assumption. For example, the proof of median estimator's properties Lemma 1 is based on completely different approach. We refer to Appendix A for more details and intuition behind Assumption 3.

#### 3.1.2 MEDIAN ESTIMATION.

In our pipeline, instead of minimizing the non-smooth function  $f$  directly, we propose to minimize the smooth approximation  $\hat{f}_\tau$  with the fixed smoothing parameter  $\tau$  via first-order methods. Following (2), the solution for  $\hat{f}_\tau$  is also a good approximate minimizer of  $f$  when  $\tau$  is sufficiently small.

Following [38], the gradient of  $\hat{f}_\tau$  at point  $x \in \mathbb{R}^d$  can be estimated by the vector:

$$\begin{aligned} g(x, \mathbf{e}, \xi) &= \frac{d}{2\tau} (f(x + \tau \mathbf{e}, \xi) - f(x - \tau \mathbf{e}, \xi)) \mathbf{e} \\ &= \frac{d}{2\tau} (f(x + \tau \mathbf{e}) - f(x - \tau \mathbf{e}) + \phi(\xi|x + \tau \mathbf{e}, x - \tau \mathbf{e})) \mathbf{e}, \end{aligned} \quad (7)$$

where  $\mathbf{e} \sim U(S_2^d)$  is a random vector uniformly distributed on the Euclidean unit sphere  $S_2^d$ . Moreover,  $\mathbf{e}, \xi$  are independent of each other conditionally on  $x$ . However, the noise  $\phi$  might have unbounded first and second moments. To fix this, we lighten tails of  $\phi$  to obtain an unbiased estimate of  $\nabla \hat{f}_\tau$ . For a point  $x \in \mathbb{R}^d$ , we apply the component-wise median operator to  $2m + 1$  samples  $\{g(x, \mathbf{e}, \xi^i)\}_{i=1}^{2m+1}$  with independent  $\xi^i$  and the same  $x$  and  $\mathbf{e}$ :

$$\text{Med}^m(\mathbf{x}, \mathbf{e}, \{\xi\}) \stackrel{\text{def}}{=} \text{Median}(\{g(x, \mathbf{e}, \xi^i)\}_{i=1}^{2m+1}). \quad (8)$$

The median operator can be applied to the batch of  $\{\mathbf{e}^j\}_{j=1}^b$  with batch size  $b$  and further averaging:

$$\text{BatchMed}_b^m(x, \{\mathbf{e}\}, \{\xi\}) \stackrel{\text{def}}{=} \frac{1}{b} \sum_{j=1}^b \text{Med}^m(x, \mathbf{e}^j, \{\xi\}^j). \quad (9)$$

For a large enough number of samples, median estimations have bounded second moment.

**Lemma 1** (Median estimation's properties). *Consider  $\mu$ -strongly convex (As. 1) and  $M_2$ -Lipschitz (As. 2) function  $f$  with oracle corrupted by noise under As. 3 with  $\Delta$  and  $\kappa > 0$ . If median size  $m > \frac{2}{\kappa}$  with norm  $q \in [2, +\infty]$ , then  $\forall x \in \mathbb{R}^d$  the median estimates (8) and (9) are unbiased, i.e.,*

$$\mathbb{E}_{\mathbf{e}, \xi}[\text{Med}^m(\mathbf{x}, \mathbf{e}, \{\xi\})] = \mathbb{E}_{\mathbf{e}, \xi}[\text{BatchMed}_b^m(x, \{\mathbf{e}\}, \{\xi\})] = \nabla \hat{f}_\tau(x),$$

and have bounded second moment, i.e.,

$$\mathbb{E}_{\mathbf{e}, \xi}[\|\text{BatchMed}_b^m(x, \{\mathbf{e}\}, \{\xi\}) - \nabla \hat{f}_\tau(x)\|_2^2] \leq \frac{\sigma^2}{b}, \quad (10)$$

$$\mathbb{E}_{\mathbf{e}, \xi}[\|\text{Med}^m(\mathbf{x}, \mathbf{e}, \{\xi\}) - \nabla \hat{f}_\tau(x)\|_q^2] \leq \sigma^2 a_q^2, \quad a_q = d^{\frac{1}{q} - \frac{1}{2}} \min\{\sqrt{32 \ln d - 8}, \sqrt{2q - 1}\}. \quad (11)$$

For independent oracle, we have  $\sigma^2 = 8dM_2^2 + 2\left(\frac{4\Delta}{\tau}\right)^2 (2m + 1) \left(\frac{4}{\kappa}\right)^{\frac{2}{\kappa}}$ , and, for Lipschitz oracle, we have  $\sigma^2 = 8dM_2^2 + (16m + 8)d^2 \Delta^2 \left(\frac{4}{\kappa}\right)^{\frac{2}{\kappa}}$ .

{4}------------------------------------------------

### 3.2 ZO-clipped-med-SSTM FOR UNCONSTRAINED PROBLEMS

We present our novel ZO-clipped-med-SSTM which works on the whole space  $Q = \mathbb{R}^d$  with the Euclidean norm. We base it on the first-order accelerated clipped Stochastic Similar Triangles Method (clipped-SSTM) with the optimal high-probability complexity bounds from [15]. Namely, we use its zeroth-order version ZO-clipped-SSTM from [20] with the batched median estimation (9).

#### Algorithm 1 ZO-clipped-med-SSTM

**Input:** Starting point  $x^0 \in \mathbb{R}^d$ , number of iterations  $K$ , median size  $m$ , batch size  $b$ , stepsize  $a > 0$ , smoothing parameter  $\tau$ , clipping levels  $\{\lambda_k\}_{k=0}^{K-1}$ .  
1: Set  $L = \sqrt{d}M_2/\tau$ ,  $A_0 = \alpha_0 = 0$ ,  $y^0 = z^0 = x^0$ .  
2: for  $k = 0, \dots, K-1$  do  
3:   Set  $\alpha_{k+1} = (k+2)/2aL$ ,  $A_{k+1} = A_k + \alpha_{k+1}$ ,  $x^{k+1} = \frac{A_k y^k + \alpha_{k+1} z^k}{A_{k+1}}$ .  
4:   Sample independently sequences  $\{\mathbf{e}_i\} \sim U(S_d^d)$  and  $\{\xi\}$ .  
5:    $g_{med}^{k+1} = \text{BatchMed}_b^m(x^{k+1}, \{\mathbf{e}_i\}, \{\xi\})$ .  
6:    $z^{k+1} = z^k - \alpha_{k+1} \cdot \text{clip}_2(g_{med}^{k+1}, \lambda_{k+1})$ ,  $y^{k+1} = \frac{A_k y^k + \alpha_{k+1} z^{k+1}}{A_{k+1}}$ .  
7: **end for**  
**Output:**  $y^K$

**Theorem 1** (Convergence of ZO-clipped-med-SSTM). *Consider convex (As. 1) and  $M_2$ -Lipschitz (As. 2) function  $f$  on  $\mathbb{R}^d$  with two-point oracle corrupted by noise under As. 3 with  $\Delta$  and  $\kappa > 0$ . We set batch size  $b$ , median size  $m = \frac{2}{\kappa} + 1$  and initial distance  $R = \|x_0 - x^*\|$ . To achieve function accuracy  $\varepsilon$ , i.e.,  $f(y^K) - f(x^*) \leq \varepsilon$  with probability at least  $1 - \beta$  via ZO-clipped-med-SSTM with parameters  $A = \ln \frac{4K}{\beta} \geq 1$ ,  $a = \Theta(\min\{A^2, \sigma K^2 \sqrt{A\tau}/\sqrt{d}M_2 R\})$ ,  $\lambda_k = \Theta(R/(\alpha_{k+1} A))$  and smoothing parameter  $\tau = \frac{\varepsilon}{4M_2}$ , the number of iterations  $K$  must be*

$$\tilde{O} \left( \frac{d^{\frac{3}{2}} M_2 R}{\varepsilon} \vee \frac{(\sqrt{d} M_2 R)^2}{b \cdot \varepsilon^2} \left( 1 \vee \left( \frac{4}{\kappa} \right)^{\frac{2}{\kappa}} \frac{d \Delta^2}{\varepsilon^2} \right) \right), \tilde{O} \left( \max \left\{ \frac{d^{\frac{3}{2}} M_2 R}{\varepsilon}, \frac{d(M_2^2 + d \Delta^2 / \kappa^{\frac{2}{\kappa}}) R^2}{b \cdot \varepsilon^2} \right\} \right),$$

for independent and Lipschitz oracle, respectively. Each iteration requires  $(2m+1) \cdot b$  oracle calls. Moreover, with probability at least  $1 - \beta$  the iterates of ZO-clipped-med-SSTM remain in the ball with center  $x^*$  and radius  $2R$ , i.e.,  $\{x^k\}_{k=0}^{K+1}, \{y^k\}_{k=0}^K, \{z^k\}_{k=0}^K \subseteq \{x \in \mathbb{R}^d : \|x - x^*\|_2 \leq 2R\}$ .

For Lipschitz oracle, the first term matches the optimal bound in terms of  $\varepsilon$  for the deterministic non-smooth problems [5], and the second term matches the optimal bound for zeroth-order problems with the finite variance [29]. Under "optimal bound" here, we mean the optimal bound for the problems with any noise. For the symmetric noise only, we are not aware of any proved bounds. In terms of  $d$ , we obtain the factor  $d M_2^2 + d^2 \Delta^2 / \kappa^{\frac{2}{\kappa}}$  instead of  $(\sqrt{d} M_2 + \sqrt{d} \Delta)^{\frac{2}{\kappa}}$  from [20].

In case of one-point oracle, while noise  $\phi$  is "small", i.e.,

$$\Delta \leq \left( \frac{\kappa}{4} \right)^{\frac{1}{\kappa}} \frac{\varepsilon}{\sqrt{d}} \quad (12)$$

convergence rate is preserved. This bound on  $\Delta$  is optimal in terms of  $\varepsilon$ , see [25; 33; 36].

For  $\mu$ -strongly-convex functions with Lipschitz oracle or independent oracle with small noise, we apply the restarted version of ZO-Clipped-med-SSTM. Algorithm's description, more details and results are located in Appendix C.1.

#### 3.2.1 EXTENDED CLASSES OF THE OPTIMIZED FUNCTIONS

**Remark 1** (Smooth objective). *The estimates presented in Theorem 1 can be improved by introducing a new assumption, namely the assumption that the objective function  $f$  is  $L$ -smooth with  $L > 0$ :  $\|\nabla f(y) - \nabla f(x)\|_2 \leq L\|y - x\|_2, \forall x, y \in \mathbb{R}^d$ . Using this assumption, we obtain the following value of the smoothing parameter  $\tau = \sqrt{\varepsilon/L}$  [see 11, the end of Section*

{5}------------------------------------------------

4.1]. Thus, assuming smoothness and convexity (As. 1) of the objective function and assuming symmetric noise (As. 3), we obtain the following estimates for the iteration complexity:  $\tilde{O}\left(\max\left\{\sqrt{\frac{LR^2}{\varepsilon}}, \frac{(\sqrt{d}R)^2}{b\varepsilon^2}\left(M_2^2 \vee \left(\frac{4}{\kappa}\right)^{\frac{2}{\kappa}} \frac{dL\Delta^2}{\varepsilon}\right)\right\}\right)$  and  $\tilde{O}\left(\max\left\{\sqrt{\frac{LR^2}{\varepsilon}}, \frac{d(M_2^2 + d\Delta^2/\kappa^{\frac{2}{\kappa}})R^2}{b\varepsilon^2}\right\}\right)$  for independent and Lipschitz oracle, respectively. These rates match the iteration's complexity for the full gradient coordinate-wise estimation.

**Remark 2** (Polyak–Łojasiewicz objective). The results of Theorem 1 can be extended to the case when the objective function satisfies the Polyak–Łojasiewicz condition via restarts: let a function  $f(x)$  is differentiable and there exists constant  $\mu > 0$  s.t.  $\forall x \in \mathbb{R}^d$  the following inequality holds  $\|\nabla f(x)\|_2^2 \geq 2\mu(f(x) - f(x^*))$ . Then, assuming smoothness (see Remark 1) and Polyak–Łojasiewicz condition for the objective function and assuming symmetric noise (As. 3), we obtain the following estimates for the iteration complexity:  $\tilde{O}\left(\max\left\{\frac{L}{\mu}, \frac{dL}{b\mu^2\varepsilon}\left(M_2^2 \vee \left(\frac{4}{\kappa}\right)^{\frac{2}{\kappa}} \frac{dL\Delta^2}{\varepsilon}\right)\right\}\right)$  and  $\tilde{O}\left(\max\left\{\frac{L}{\mu}, \frac{dL(M_2^2 + d\Delta^2/\kappa^{\frac{2}{\kappa}})}{b\mu^2\varepsilon}\right\}\right)$  for independent and Lipschitz oracle, respectively.

### 3.3 ZO-clipped-med-SMD FOR CONSTRAINED PROBLEMS

We propose our novel ZO-clipped-med-SMD to minimize functions on a convex compact  $Q \subset \mathbb{R}^d$ . We use unbatched median estimation (8) in the zeroth-order algorithm ZO-clipped-SMD from [19], which is based on Mirror Gradient Descent.

We define 1-strongly convex w.r.t.  $p$ -norm and differentiable prox-function  $\Psi_p$ . We denote its convex (Fenchel) conjugate and its Bregman divergence, respectively, as

$$\Psi_p^*(y) = \sup_{x \in \mathbb{R}^d} \{ \langle x, y \rangle - \Psi_p(x) \}, \quad V_{\Psi_p}(y, x) = \Psi_p(y) - \Psi_p(x) - \langle \nabla \Psi_p(x), y - x \rangle.$$

#### Algorithm 2 ZO-clipped-med-SMD

**Input:** Number of iterations  $K$ , median size  $m$ , stepsize  $\nu$ , prox-function  $\Psi_p$ , smoothing parameter  $\tau$ , clipping level  $\lambda$ .

1:  $x_0 = \arg \min_{x \in Q} \Psi_p(x)$ .

2: **for**  $k = 0, 1, \dots, K-1$  **do**

3: Sample  $\mathbf{e}$  from  $U(S_d^d)$  and sequence  $\{\xi\}$ .

4:  $g_{med}^{k+1} = \text{Med}^m(x^{k+1}, \mathbf{e}, \{\xi\})$ .

5:  $y_{k+1} = \nabla(\Psi_p^*)(\nabla \Psi_p(x_k) - \nu \cdot \text{clip}_q(g_{med}^{k+1}))$ ,  $x_{k+1} = \arg \min_{x \in Q} V_{\Psi_p}(x, y_{k+1})$ .

6: **end for**

**Output:**  $\bar{x}_K := \frac{1}{K} \sum_{k=0}^{K-1} x_k$

**Theorem 2.** Consider convex (As. 1) and  $M_2$ -Lipschitz (As. 2) function  $f$  with two-point oracle corrupted by noise under As. 3 with  $\kappa > 0$ . To achieve function accuracy  $\varepsilon$ , i.e.,  $f(\bar{x}_K) - f(x^*) \leq \varepsilon$  with probability at least  $1 - \beta$  via ZO-clipped-med-SMD with median size  $m = \frac{2}{\kappa} + 1$ , clipping level  $\lambda = \sigma a_q \sqrt{K}$ , stepsize  $\nu = \frac{D_{\Psi_p}}{\lambda}$ , diameter  $D_{\Psi_p}^2 \stackrel{\text{def}}{=} 2 \sup_{x, y \in Q} V_{\Psi_p}(x, y)$ , prox-function  $\Psi_p$  and  $\tau = \frac{\varepsilon}{4M_2}$ , the number of iterations  $K$  must be

$$\tilde{O}\left(\frac{(\sqrt{d}M_2 a_q D_{\Psi_p})^2}{\varepsilon^2} \left(1 \vee \left(\frac{4}{\kappa}\right)^{\frac{2}{\kappa}} \frac{d\Delta^2}{\varepsilon^2}\right)\right), \quad \tilde{O}\left(\frac{d(M_2^2 + d\Delta^2/\kappa^{\frac{2}{\kappa}})a_q^2 D_{\Psi_p}^2}{\varepsilon^2}\right) \quad (13)$$

for independent and Lipschitz oracle, respectively. Each iteration requires  $(2m + 1)$  oracle calls.

Bounds (13) match optimal in terms of  $\varepsilon$  bounds for stochastic non-smooth optimization on convex compact with the finite variance [42]. The upper bound for  $\Delta$  under which the convergence rate is preserved is the same as for unconstrained optimization (12).

For  $\mu$ -strongly-convex functions with Lipschitz oracle or independent oracle with small noise, we apply the restarted version of ZO-Clipped-SMD. Algorithm and results are located in Appendix C.2.

{6}------------------------------------------------

## 4 APPLICATION TO THE MULTI-ARMED BANDIT PROBLEM WITH HEAVY TAILS

In this section, we present our novel **Clipped-INF-med-SMD** algorithm for multi-armed bandit (MAB) problem with heavy-tailed rewards.

**Introduction.** The stochastic MAB problem [21] can be formulated as follows: an agent at each time step  $t = 1, \dots, T$  chooses an action  $A_t$  from a given action set  $\mathcal{A} = (a_1, \dots, a_n)$  and suffers stochastic loss. For each action  $a_i$ , there exists a probability density function for losses  $p(a_i)$ , and an agent doesn't know them in advance. An agent can observe losses only for one action at each step, namely, the one it chooses. At each round  $t$ , when action  $a_i$  is chosen (i.e.  $A_t = a_i$ ), stochastic loss  $\mu_{A_t} + \xi_{A_t}$  sampled from  $p(a_i)$  independently. Agent's goal is to minimize *average regret*:

$$\mathbb{E}[\mathcal{R}_T] = \sum_{t=1}^T [\mu_{A_t} - \mu^*], \quad \mu^* = \min_{a_i \in \mathcal{A}} \mu_i.$$

One of the main approaches for solving the MAB problem is to use reduction to the online convex optimization problem [17; 32]. Consider stochastic linear loss functions  $l_t(x_t) = (\mu + \xi_t, x_t)$ , with noise  $\xi_t$  and unknown fixed vector of expected losses  $\mu \in \mathbb{R}^d$ . The decision variable  $x_t \in \Delta_+^d$  can be viewed as the player's mixed strategy (probability distribution over arms), which they use to sample arms with the aim to minimize expected regret

$$\mathbb{E}[\mathcal{R}_T(u)] = \mathbb{E} \left[ \sum_{t=1}^T l_t(x_t) - \min_{u \in \Delta_+^d} \left( \sum_{t=1}^T l_t(u) \right) \right].$$

The player observes only sampled losses for the chosen arm, i.e., the (sub)gradient  $g(x) \in \partial l(x)$  is not observed in the MAB setting, and one must use an inexact oracle instead.

**Related works.** Bandits with heavy tails were introduced in [23; 4]. The heavy noise assumption usually requires the existence of  $\kappa \in (1, 2)$ , such that  $\mathbb{E}[\|\mu + \xi_t\|^\kappa] \leq \sigma^\kappa$  (in this work, we use different Assumption 3 with  $\kappa > 0$ ). In [4], the authors provide lower bounds on regret  $\Omega\left(\sigma d^{\frac{\kappa-1}{\kappa}} T^{\frac{1}{\kappa}}\right)$  and nearly optimal algorithmic scheme called Robust UCB. Recently, a few optimal algorithms were proposed [22; 47; 18; 7] with regret bound  $\tilde{O}\left(\sigma d^{\frac{\kappa-1}{\kappa}} T^{\frac{1}{\kappa}}\right)$ . HTINF [18] is an INF-type algorithm with a specific pruning procedure. Algorithm 1/2-Tsallis [47] is similar to HTINF. INF-clip [7] employs a clipping mechanism instead of pruning, it clips rewards at the initial stage of the algorithm construction process, prior to applying importance weighting. The main drawback of this procedure is that the importance weighting procedure can artificially produce a burst in the gradient estimator. Finally, APE [22] is a perturbation-based exploration strategy that uses a p-robust mean estimator. Its algorithmic scheme is UCB-type and is very different from our algorithm.

**Our approach.** We assume that noise  $\xi_t$  satisfy Assumption 3 for some  $\kappa > 0$ . We construct our **Clipped-INF-med-SMD** (Algorithm 3) based on Online Mirror Descent, but in case of symmetric noise we can improve regret upper bounds and make it  $\tilde{O}(\sqrt{dT})$  which is optimal compared to the lower bound  $\Omega(\sqrt{dT})$  for stochastic MAB with the bounded variance of losses. In our algorithm, we use an importance-weighted estimator:

$$\hat{g}_{t,i} = \begin{cases} \frac{g_{t,i}}{x_{t,i}} & \text{if } i = A_t \\ 0 & \text{otherwise} \end{cases},$$

where  $A_t$  is the index of the chosen (at round  $t$ ) arm. This estimator is unbiased, i.e.  $\mathbb{E}_{x_t}[\hat{g}_t] = g_t$ . The main drawback of this estimator is that, in the case of small  $x_{t,i}$ , the value of  $\hat{g}_{t,i}$  can be arbitrarily large. When the noise  $g_t - \mu$  has heavy tails (i.e.,  $\|g_t - \mu\|_\infty$  can be large with high probability), this drawback can be amplified. That is why we use robust median estimation with further clipping.

**Theorem 3.** Consider MAB problem where the conditional probability density function for each loss satisfies Assumption 3 with  $\Delta, \kappa > 0$ , and  $\|\mu\|_\infty \leq R$ . Then, for the period  $T$ , the sequence  $\{x_t\}_{t=1}^T$  generated by **Clipped-INF-med-SMD** with parameters  $m = \frac{2}{\kappa} + 1$ ,  $\tau = \sqrt{d}$ ,  $\nu = \frac{\sqrt{(2m+1)}}{\sqrt{T(36c^3+2R^2)}}$ ,

$\lambda = \sqrt{T}$  and prox-function  $\Psi_1(x) = \psi(x) \stackrel{def}{=} 2 \left(1 - \sum_{i=1}^d x_i^{1/2}\right)$  satisfies

$$\mathbb{E}[\mathcal{R}_T(u)] \leq \sqrt{dT} \cdot (8c^2/\sqrt{d} + 4\sqrt{(2m+1)(18c^2+R^2)}), \quad u \in \Delta_+^d, \quad (14)$$

{7}------------------------------------------------

#### --- **Algorithm 3** Clipped-INF-med-SMD ---

**Input:** Time period  $T$ , median size  $m$ , stepsize  $\nu$ , prox-function  $\Psi_p$ , clipping level  $\lambda$ .

- 1:  $x_0 = \arg \min_{x \in \Delta_d^+} \Psi_p(x)$ .
  - 2: Set number of iterations  $K = \left\lceil \frac{T-1}{2m+1} \right\rceil$ .
  - 3: **for**  $k = 0, 1, \dots, K-1$  **do**
  - 4:   Draw  $A_t$  for  $2m+1$  times ( $t = (2m+1) \cdot k + 1, \dots, (2m+1) \cdot (k+1)$ ) with  $P(A_t = i) = x_{k,i}$ ,  $i = 1, \dots, d$  and observe rewards  $g_{t,A_t}$ .
  - 5:   For each observation, construct estimation  $\hat{g}_{t,i} = \begin{cases} \frac{g_{t,i}}{x_{k,i}} & \text{if } i = A_t \\ 0 & \text{otherwise} \end{cases}$ ,  $i = 1, \dots, d$ .
  - 6:    $\sigma_{med}^{k+1} = \text{Median}(\{\hat{g}_t\}_{t=(2m+1) \cdot k + 1}^{(2m+1) \cdot (k+1)})$ .
  - 7:    $y_{k+1} = \nabla(\Psi_p^*)(\nabla \Psi_p(x_k) - \nu \cdot \text{clip}_q(\sigma_{med}^k, \lambda))$ ,  $x_{k+1} = \arg \min_{x \in \Delta_d^+} V_{\Psi_p}(x, y_{k+1})$ .
  - 8: **end for**
- 

where  $c^2 = (32 \ln d - 8) \cdot \left(8M_2^2 + 2\Delta^2(2m+1)\left(\frac{4}{\kappa}\right)^{\frac{2}{\kappa}}\right)$ . Moreover, high probability bounds from Theorem 2 also hold. Proof of Theorem 3 is located in Appendix B.3.

## 5 NUMERICAL EXPERIMENTS

In this section, we demonstrate the superior performance of our ZO-clipped-med-SSTM and Clipped-INF-med-SMD under heavy-tailed noise on experiments on syntactical and real-world data. Additional experiments and technical details are located in Appendix D.

### 5.1 MULTI-ARMED BANDIT

We compare our Clipped-INF-med-SMD with popular SOTA algorithms tailored to handle MAB problem with heavy tails, namely, HTINF and APE. We focus on an experiment involving only two available arms ( $d = 2$ ). Each arm  $i$  generates random losses  $g_{t,i} \sim \xi_t + \beta_i$ . Parameters  $\beta_0 = 3, \beta_1 = 3.5$  are fixed, and independent random variables  $\xi_t$  have the same probability density  $P_{\xi_t}(x) = \frac{1}{3 \cdot (1 + (\frac{x}{\pi})^2) \cdot \pi}$ .

For all methods, we evaluate the distribution of expected regret and probability of picking the best arm over 100 runs. The results are presented in Figure 1.

![Figure 1: Two line graphs showing performance metrics over 30,000 samples for 100 tracks and 2 arms. The left graph, 'Average expected regret', shows that Clipped-INF-med-SMD (blue line) maintains a lower regret (around 0.2) compared to APE (red line, around 0.25) and HTINF (green line, which decreases to around 0.1). The right graph, 'Probability of best arm choice', shows that HTINF (green line) quickly reaches a high probability (around 0.9), while APE (red line) and Clipped-INF-med-SMD (blue line) both stabilize around 0.6. Shaded regions represent 0.95 and 0.05 percentiles for regret and ± std bounds for probabilities.](f5e70cbe66e71e65b4ae4aa7816d266a_img.jpg)

Figure 1: Two line graphs showing performance metrics over 30,000 samples for 100 tracks and 2 arms. The left graph, 'Average expected regret', shows that Clipped-INF-med-SMD (blue line) maintains a lower regret (around 0.2) compared to APE (red line, around 0.25) and HTINF (green line, which decreases to around 0.1). The right graph, 'Probability of best arm choice', shows that HTINF (green line) quickly reaches a high probability (around 0.9), while APE (red line) and Clipped-INF-med-SMD (blue line) both stabilize around 0.6. Shaded regions represent 0.95 and 0.05 percentiles for regret and ± std bounds for probabilities.

Figure 1: Average expected regret and probability of optimal arm picking mean for 100 experiments and 30000 samples with 0.95 and 0.05 percentiles for regret and  $\pm$  std bounds for probabilities

As one can see from the graphs, HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does, which confirms the efficiency of the proposed method. In Appendix D.1, we provide technical details and additional experiments for different  $\kappa$ .

{8}------------------------------------------------

### 5.2 CRYPTOCURRENCY PORTFOLIO OPTIMIZATION

We choose cryptocurrency portfolio optimization problem for Clipped-INF-med-SMD real world application, since cryptocurrency pricing data is known by having heavy-tailed distribution. In our scenario, we have  $n = 9$  assets for investing. At step  $t$ , we choose assets' distribution  $x_{t,i} \in \Delta^n$  and then observe the whole income vector  $r_{t,i}$  for each asset  $i$ . The main goal is to maximize total income  $\max \mathbb{E} \sum_{t=1}^T \sum_{i=1}^n r_{t,i} x_{t,i}$  over a fixed time interval with length  $T$ .

Portfolio selection has the full feedback for all assets, while, in standard bandits, we observe only one asset per step. We adjust our Clipped-INF-med-SMD for the full feedback via calculating line 4 in Algorithm 3 for each asset  $i$ . As baselines, we use two strategies: hold ETH and the Efficient Frontier method [28] with maximal sharp ratio portfolio selected. For a dataset, we use open prices from Binance Spot for 2023.

The results are presented in Figure 2. As one can see, the Efficient Frontier strategy can't efficiently perform on cryptocurrency assets, and Clipped-INF-med-SMD achieved higher performance than just holding the ETH strategy, so it can be applied for detecting potentially promising assets.

![Figure 2: Strategies profit coefficient and Clipped-INF-med-SMD assets distribution over 2023 year. The figure contains two line graphs. The left graph shows the profit coefficient over time (2023-01 to 2024-01) for three strategies: Clipped-INF-med-SMD (blue line), Efficient Frontier (orange line), and Random (green line). The right graph shows the asset distribution (percentage) over time for the same three strategies, with individual lines for each asset (ETH, BTC, etc.).](891ff9b651838b7f59e9a1612a739e15_img.jpg)

Figure 2: Strategies profit coefficient and Clipped-INF-med-SMD assets distribution over 2023 year. The figure contains two line graphs. The left graph shows the profit coefficient over time (2023-01 to 2024-01) for three strategies: Clipped-INF-med-SMD (blue line), Efficient Frontier (orange line), and Random (green line). The right graph shows the asset distribution (percentage) over time for the same three strategies, with individual lines for each asset (ETH, BTC, etc.).

Figure 2: Strategies profit coefficient and Clipped-INF-med-SMD assets distribution over 2023 year

### 5.3 ZEROTH-ORDER OPTIMIZATION

To demonstrate the performance of ZO-clipped-med-SSTM, we follow [20] and conduct experiments on the following problem:

$$\min_{x \in \mathbb{R}^d} \|Ax - b\|_2 + \langle \xi, x \rangle,$$

where  $\xi$  is a random vector with independent components sampled from the symmetric Levy  $\alpha$ -stable distribution with different  $\alpha = 0.75, 1.0, 1.25, 1.5$ ,  $A \in \mathbb{R}^{l \times d}$ ,  $b \in \mathbb{R}^l$ . Note, that  $\alpha$  has the same meaning as  $\kappa$ , because this distribution asymptotic behavior is  $f(x) \sim \frac{1}{|x|^{1+\alpha}}$  for  $\alpha < 2$ .

For ZO-clipped-med-SSTM, the best median size is  $m = 2$ . We compare it with the median size  $m = 0$  which is basically ZO-clipped-SSTM. We additionally compare our algorithm with ZO-clipped-SGD from [20] and ZO-clipped-med-SGD — version of ZO-clipped-SGD with gradient estimation step replaced with median clipping version from our work.

The results over 3 launches are presented in Figure 3. The green lines on the graphs represent algorithms with median clipping. We can see that for extremely noised data  $\kappa \leq 1$ , our median clipping-based methods significantly outperform non-median versions. While, for standard heavy-tailed noise  $\kappa > 1$ , our methods do not lose to other competitors.

In Appendix D.2, we provide technical details about hyperparameters, additional experiments with enlarged number of launches and study asymmetric noise and its effect on our median methods.

**Tuning of  $m$ .** In experiments with both bandits and ZO methods, we grid search the median size  $m$  among the range  $[3, 5, 7]$ . We noticed that unlike the choice of continuous the clipping level, the choice of the discrete median size only slightly affects the convergence and does not require careful fine-tuning. This range is enough to find an optimal median size for optimal convergence.

## 6 DISCUSSION

### 6.1 LIMITATIONS

**Symmetric noise.** The assumption of the symmetric noise can be seen as a limitation from a practical point of view. It is indeed the case, but we argue that it is not as severe as it looks. A common

{9}------------------------------------------------

486  
487  
488  
489  
490  
491  
492  
493  
494  
495  
496  
497  
498  
499  
500  
501  
502  
503  
504  
505  
506  
507  
508  
509  
510  
511  
512  
513  
514  
515  
516  
517  
518  
519  
520  
521  
522  
523  
524  
525  
526  
527  
528  
529  
530  
531  
532  
533  
534  
535  
536  
537  
538  
539

![Figure 3: Four subplots showing the convergence of ZO-clipped-SSTM, ZO-clipped-med-SSTM, ZO-clipped-SGD, and ZO-clipped-med-SGD algorithms. Each subplot corresponds to a different alpha = kappa parameter (0.75, 1.0, 1.25, 1.5). The y-axis is f(x) - f(x^*) on a log scale from 10^-1 to 10^2. The x-axis is samples from 0.00 to 2.00e7. In all plots, ZO-clipped-med-SGD (green dashed line with circles) and ZO-clipped-SGD (red dashed line with circles) show faster convergence than ZO-clipped-SSTM (green solid line with circles) and ZO-clipped-med-SSTM (red solid line with circles).](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 3: Four subplots showing the convergence of ZO-clipped-SSTM, ZO-clipped-med-SSTM, ZO-clipped-SGD, and ZO-clipped-med-SGD algorithms. Each subplot corresponds to a different alpha = kappa parameter (0.75, 1.0, 1.25, 1.5). The y-axis is f(x) - f(x^\*) on a log scale from 10^-1 to 10^2. The x-axis is samples from 0.00 to 2.00e7. In all plots, ZO-clipped-med-SGD (green dashed line with circles) and ZO-clipped-SGD (red dashed line with circles) show faster convergence than ZO-clipped-SSTM (green solid line with circles) and ZO-clipped-med-SSTM (red solid line with circles).

Figure 3: Convergence of ZO-clipped-SSTM, ZO-clipped-med-SSTM, ZO-clipped-SGD and ZO-clipped-med-SGD in terms of a gap function w.r.t. the number of consumed samples from the dataset for different  $\alpha = \kappa$  parameters (left-to-right and top-to-bottom: 0.75, 1.0, 1.25, 1.5)

strategy to solve a general optimization problem is to run several algorithms in a competitive manner to see which performs better in practice. This approach is implemented in industrial solvers such as Gurobi. Thus, if we have different algorithms, each suited to its own conditions, we can simply test to see which one is faster for our particular case. In this scenario, we want a set of algorithms, each designed for its specific case. Our algorithm can serve as one of the options in such mix, since it provides considerable acceleration in a significant number of noise cases. Moreover, in experiments with non-symmetric noises (§D.2.1), our methods do not lose to the baselines. Hence, running our methods ends up with either typical convergence rates or faster rates for symmetric noises.

**Known  $\kappa$ .** In our Theorems 1, 2, 3, parameter  $\kappa$  is required to set optimal median size  $m = \frac{2}{\kappa} + 1$ . However, for the most common cases  $\kappa$  is at least 1 (i.e. expectation exists), hence we could take median size  $m = 3$ . In case when parameter  $\kappa \rightarrow 0$ , we leave the construction of an adaptive scheme [18] for future work. In practice, the choice of  $m$  can be limited to a small, discrete range.

### 6.2 COMPARISON WITH PREVIOUS WORKS

Unlike the baselines ZO-clipped-SSTM [20] and APE [22], HTINF [18] with simple clipping and general heavy-tailed noise assumption  $\kappa \in (1, 2]$ , our Algorithms 1, 2, 3 with median clipping can work with extremely heavy-tailed noises  $\kappa \leq 1$ . For any  $\kappa > 0$ , iterative complexity of our methods remains as if noise had bounded variance, namely,  $\tilde{O}(d^2\varepsilon^{-2})$  iterations to achieve function accuracy or average regret  $\varepsilon$ . In contrast, the best-known baselines' rates  $\tilde{O}((\sqrt{d\varepsilon^{-1}})^{\frac{\kappa}{\kappa-1}})$  deteriorate depending on  $\kappa$ . However, such breaking results can be guaranteed only for symmetric noises, which is not as serious limitation as it seems. Nevertheless, we show that, for asymmetric noises, our methods in practice are competitive as well and perform at the same level as the baselines (§D.2.1).

### 6.3 FUTURE DIRECTIONS

**Potential impact.** We believe that ideas and obtained results from our work can inspire the community to further develop both zeroth-order methods and clipping technique. Especially considering how effectively our algorithms can work in a wide range of noise cases. For example, Lipschitz [26] and linear [39] MAB and non-convex functions [24; 41; 45] remain out of the scope of our paper.

**Broader impact.** This paper presents work which goal is to advance the field of Optimization. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

 Rest of paper (reference and Appendix) is removed.