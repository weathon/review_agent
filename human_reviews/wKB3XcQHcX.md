# Speed Limits for Deep Learning

- Decision: Reject
- Scores: 6, 8, 6, 3

## Abstract
State-of-the-art neural networks require extreme computational power to train. It is therefore natural to wonder whether they are optimally trained. Here we apply a recent advancement in stochastic thermodynamics which allows bounding the speed at which one can go from the initial weight distribution to the final distribution of the fully trained network, based on the ratio of their Wasserstein-2 distance and the entropy production rate of the dynamical process connecting them. Considering both gradient-flow and Langevin training dynamics, we provide analytical expressions for these speed limits for linear and linearizable (e.g. NTK) neural networks. Remarkably, given some plausible scaling assumptions on the NTK spectra and spectral decomposition of the labels-- learning is optimal in a scaling sense. Our results are consistent with small-scale experiments with CNNs and FCNs on CIFAR-10, showing a short highly non-optimal regime followed by a longer optimal regime.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This problem considers the continuous dynamic of gradient-based learning problems. Studying the problem from a thermodynamic perspective, the paper derives a lower bound on the time for a process to go from an initial state to a target state using Wasserstein-2 distance. The paper further considers two realizations of the problem, namely the linear regression and the NTK, and presents the implication of the result under various limiting scenarios.

### Strengths
1. It seems an interesting idea to connect the machine learning optimization dynamic with notions in thermodynamics.

2. The paper provides interesting interpretations of the speed limit in linear regression and NTK learning.

### Weaknesses
1. I am not sure about the significance of this theoretical investigation. It seems that the paper only considers the lower bound of the time that goes from an initial parameter state to a target parameter state. It does not tell us information about e.g. the time lower bound to get to a near-stable parameter state, or the time lower bound to get to near-zero potential.

2. Characterizing the speed limit using the Wasserstein-2 distance is not easily interpretable since it is in general hard to compute the Wasserstein-2 distance. The paper seems only able to derive interpretable results under limiting conditions like no noise or infinite parameters.

3. The models considered in this paper are fairly simple. Both the linear regression and the learning under NTK assumption are linear models and are not representative of deep learning in general.

4. The writing of the paper can be improved. I hope to see formal theorems in the paper stating the main contribution, and notations need to be clearly stated. For instance, I am not sure what is $Z_T$ and $Z_0$ in Eq. (4). There is also no related works section.

### Questions
Given the conclusion in the paper that in the NTK regime the dynamic in the paper achieves a rate that is almost optimal, is this contradicting the fact that the continuous heavy-ball dynamic achieves a faster rate to converge to the stable point (see, e.g. https://link.springer.com/article/10.1007/s1022-01164-w)?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper applies recent advances in stochastic thermodynamics to analyze the efficiency of training neural networks. It derives analytical expressions relating the speed limit (minimum training time) to the Wasserstein 2-distance between initial and final weight distributions and the entropy production. For linear regression and neural networks in the NTK regime, exact formulas are provided for the quantities involved in the speed limit. Under plausible assumptions on the NTK spectrum (power law behavior) and residue (defined as the target minus the initial prediction), NTKs exhibit near-optimal training efficiency in the scaling sense. Small-scale experiments on CIFAR-10 qualitatively support the theoretical findings.

### Strengths
I think this is a technically sound paper that makes good contributions. The application of stochastic thermodynamics concepts to neural network training is novel. I have not seen it before in prior literature. The analysis done in the paper is insightful and to the best of my knowledge seems mathematically rigorous. The results on optimal scaling efficiency are intriguing. The writing is clear and relatively compact. It seems like the relevant prior works cited correctly. Moreover this paper provides a good literature review on various different topics across entropy production and the various bounds on that quantity.

### Weaknesses
It is unclear if the near optimal scaling efficiency result applies to large realistic models and datasets. The CIFAR-10 study used very small networks. It would be very nice to see an empirical example with a larger scope. 

Further, it would be nice to be more explicit of how much of the entropy production is due to the presence of nonzero initial weights. It would be nice to cover the case of either the perceptron or the NTK starting with $\theta_0 = 0$. 

It is rather unclear whether there are any takeaways from practitioners. Its not strictly necessary that their should be, but given the title one is left to wonder whether there are possible statements that can be made about the compute-optimal frontier.

### Questions
It's interesting that power law scalings in the target (ie in the residues) seem to imply an inefficiency factor that grows with dataset size. Can the authors comment on whether this is representative of realistic datasets? 

The initial transient period seems important. The current characterization is that the low modes are learned very quickly during this period, however many other things are also happening. For one, the kernel could be drastically realigning its eigenstructure (as in e.g. Atanasov Bordelon Pehlevan https://arxiv.org/abs/2111.00034). Relatedly, it would be interesting to see these empirical results for the NN as the feature learning parameter (as in $\alpha$ in Chizat and Bach https://arxiv.org/abs/1812.07956) is varied. 

It seems strange that in the high noise regime the formalism tells the perceptron learns in "zero time" when really its unable to learn at all. Am I understanding this correctly? 

To the best of my understanding, the only setting in which the W2 distance enters practically is when the marginals are delta functions. So it only is realized as the 2-norm in weight space. Is there any use to the optimal transport formalism beyond this?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a study of the "time" it takes for a neural network (NN) to travel from its initialization distribution to its final distribution (after training). The study is based on of the transport of the distribution of the parameters and the related evolution of the entropy of the system. To use this theoretical framework, it is necessary to do some assumptions: continuous-time training, full-batch optimization, simplified models (linear regression, Neural Tangent Kernel (NTK) setting), etc.

This theoretical study comes with a series of experiments, with a setting as close as possible as the theoretical assumptions (small learning rates, full-batch gradient descent). The experimental results are not entirely consistent with the theoretical predictions. The authors claim that the "training time" they have computed theoretically is close to the actual training time of NNs in the experiments.

### Strengths
## Originality

To my knowledge, this work is original. But I am not a specialist of statistical physics applied to NNs.

## Clarity

The authors made the effort to make their paper understandable to the reader who would not be a specialist in statistical physics applied to NNs. Overall, the paper is easy to read.

## Quality

The experimental section, despite being narrow (only one setting has been tested), provides enough results to evaluate the significance and the limitation of the theoretical section.

### Weaknesses
## Significance

### Narrowness of the theoretical setting

Only two setups have been studied: linear regression and NNs in the NTK regime. Moreover, the continuous-time SGD does not model faithfully the discrete SGD when training practical NNs on realistic data.

Moreover, the authors does not discuss how $T_{SL}$ (lower bound on the training time) obtained in the NTK regime compare to a hypothetical $T_{SL}$ obtained in finite-width NNs. Would it be larger of smaller? ...

### Motivation

Given the theoretical framework, I do not fully understand how this work is related to the usual challenges in deep learning. Can we use this work to improve optimization? to obtain theoretical guarantees? ...

### Questions
Main questions:
 * motivation: can we use this work to evaluate the quality of an optimizer?
 * stronger results: how does the $T_{SL}$ obtained in the NTK limit relate to some "$T_{SL}$" in the finite width setting?
 * experimental setup: the authors claim that a learning rate of $10^{-5}$ is small and apply the NTK setting to Myrtle-5; how to justify these choices? Is $10^{-5}$ really small enough? is Myrtle-5 wide enough to consider that we are in the NTK regime?

Other questions:
 * Eqn (10) is difficult to interpret: in the proof, the effective learning rate is $\eta/n$; how such a proof can be interpreted in the limit $n \rightarrow \infty$?
 * I may have misunderstood one point: Figure 1.b seems to indicate that the training time is about $10$-$30$ times larger than $T_{SL}$. We are far from the "$O(1)$" factor written in the claims... while it is true that the ratio of the trajectory lengths (Fig. 1.e) is of order 1. How to solve this contradiction?
 * it is clear to me that the result written in Eqn. (10) depends on the distribution of the data. To obtain such a result, the data are assumed to be Gaussian. It is then not surprising that Fig. 1.d contradicts Eqn. (10), since CIFAR-10 images are far from being Gaussian vectors.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The goal of the paper is to explore a notion of training time efficiency in deep learning through the lens of optimal transport and stochastic thermodynamics. The main technical tool is the Benamou-Brenier formula in optimal transport theory, which casts the $L^2$-Wasserstein distance $(W_2$) between initial and final distributions ($\mu_0$ and $\mu_T$) as the minimum of a specific cost functional over smooth paths $(\mu_t)_{0\leq t\leq T}$ induced by time-dependent potentials. The formula provides a lower bound on the transport time (T) in terms of $W_2$ and path-dependent cost.

The authors apply this bound to Langevin dynamics and gradient flow, focusing on linear and kernel regression. They propose to use (a tractable limit of) the bound as a metric for 'training speed inefficiency' in deep learning. Experiments with CNNs trained on small subsets of CIFAR10 with full batch gradient descent on mean squared loss reveal an initial 'inefficient' training phase, corresponding to an alignment of the NTK to the target, followed  by a phase where learning becomes 'efficient,' as evidenced by the training time saturating the proposed bound.

### Strengths
The paper introduces an innovative perspective by leveraging the speed limit concept from optimal transport theory to shed light on the training efficiency in deep learning. Given the importance and challenge of characterizing the training speed and efficiency of neural networks,  new theoretical insights in this topic are timely and have the potential for significant impact.

### Weaknesses
However, I have significant concerns about the scope and significance of the results, as well as the validity of certain claims. Additionally, I believe there is ample room for substantial improvement in the writing and the clarity of the presentation.  

**On Significance** 

1. The concept of 'optimal training speed' in the paper is highly specific to the introduced framework and appears to diverge from its conventional interpretation in the optimization literature. Unlike traditional contexts where it refers to achieving optimal convergence rates within a given class of algorithm applied to a specific problem, here, it pertains to conditions on the problem (choice of potential) under which a specific algorithm (Langevin dynamics) aligns with the derived lower bound on transport time (T) from the Benamou-Brenier formula. These two notions of optimality seem orthogonal, explaining potential confusion among reviewers. Reviewer 5tEv rightly points out that the framework doesn't provide suboptimality bounds or convergence rates, hence does not tell us anything about optimality  in the standard sense.  Aligning with Reviewer 8rXj, I believe the relevance to challenges in deep learning optimization remains unclear.


2. Section 2 exposes known results from the literature on the Benamou-Brenier formula (Prop 1 of Benamou-Brenier, 2000; Eq 11 of Vu & Saito, 2022).  The only novel contribution seems to be the simple analytical form (8) of the bound in the noiseless limit (gradient flow), which forms the basis of the NTK analysis in Section 3.2 and the experimental Section 4. However, it appears that the authors may not have explicitly noted that Equation (8) is a direct consequence of the Cauchy-Schwarz inequality:

$$ T \Delta L(T) = - T \int_0^T (\nabla_\theta L)^\top \dot{\theta} dt = T \int_0^T \||\dot{\theta}\||^2 dt \geq \||\theta_T - \theta_0\||^2$$ 

(where the equality holds for constant $\dot{\theta}$, i.e constant loss gradient along the trajectory).  This raises  questions about the relevance of the speed limit framework invoked by the authors, at least  in the noiseless setting (I will have more comments on the noisy setting of Section 3.1 below).  

3. In light of the above, the results in kernel regression (Section 3.2) appear to offer conditions on the target function that minimize the deviation of the flow from a constant gradient loss trajectory (the scale of the actual training time $T$ is fully determined by the kernel spectrum). While these results (along with known results about NTK time evolution and alignment behaviour in the early stage of training, see Fort et al 2020, Baratin et al, 2021,  Atanasov et al 2022) explain empirical observations in Section 4, their broader significance in the context of deep learning optimization remains unclear.

**On Soundness** 

4. Some claims and derivations raise validity concerns. For instance, as made explicit above Equ (21) in the appendix, (4) is obtained by choosing $p_T$ to be the stationnary (Boltzman) distribution, reached at infinite time $T \to \infty$. This raises doubts about the validity of the formula at finite times $T$. 
2. I have a similar concern for the linear regression analysis of Section 3.1. From Appendix C,  it appears the calculation of the lower bound (7) assumes that $p_T$ is the stationnary distribution (Gaussian centered on the mininum norm solution $\mu_T$)  reached only as $T \to \infty$, limiting the significance of the analysis.  I suspect the same calculation at intermediate finite time $T$ might be more challenging. 

5. Section 3.1 statements are puzzling. For example, Eq (12) considers the limit $d\to \infty$ and concludes "interestingly, in this regime, the parameters are not moving a lot and therefore the final distribution is very close to the initial one". But this seems to be merely due to the setting chosen by the authors, where  both initial and target distributions are zero mean Gaussians with O(1/d) variances,  so that $\theta_0 = \theta_\ast = 0$ as $d\to \infty$. Also  the statements following Eq (10), (11), (12) appear misleading, as they refer to a transport saturating the bound,  as opposed to the actual dynamics.

**On Presentation** 

The exposition of prior work on speed limits lacks self-containment and clarity.  For example, the formula (3) for entropy production is given without explanation nor intuition. A more effective presentation could start with the variational formulation (27) of the Wasserstein distance from Benamou & Bremier, 2000, build the link with langevin dynamics and gradient flow, and then discuss ways to compute the cost functional. The balance between the main body and the appendix also needs major improvement.  As it stands, the main body compiles (often scattered)  raw results  with (often vague) comments and interpretations, which make it challenging to grasp or appreciate the content without delving into details provided in the supplementary material (or even the prior literature on the topic).  

Overall, a (substantially) more integrated presentation might help clarify premises, contributions, and results' implications, making the paper more accessible to the intended audience.  

**References**

Fort et al, 2020. Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel. 

Baratin et al, 2021. Implicit Regularization via Neural Feature Alignment

Atanasov et al 2022. Neural Networks as Kernel Learners: The Silent Alignment Effect.

### Questions
**Miscalleneous**

* In Eq (7), (27), (28): the Wasserstein distance should be squared. 

* Deriving (6) from (5) may not be straightforward, as the term $\nabla_\theta \ln p$ in the integrand of (5) might depend on $\beta$. Additionally, while the derivation of (5) from (41) in Appendix A.6 is acknowledged, the derivation of (41) appears to lack normalization factors (log Z+/Z-) corresponding to the forward and backward distributions. It's not immediately clear to me if these factors cancel each other out.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
