# Discrete Compositional Generation via General Soft Operators and Robust Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
A major bottleneck in scientific discovery involves narrowing an exponentially large set of objects, such as proteins or molecules, to a small set of promising candidates with desirable properties. While this process can rely on expert knowledge, recent methods leverage reinforcement learning (RL) guided by a proxy reward function to enable this filtering. By employing various forms of entropy regularization, these methods aim to learn samplers that generate diverse candidates that are highly rated by the proxy function. In this work, we make two main contributions. First, we show that these methods are liable to generate overly diverse, suboptimal candidates in large search spaces. To address this issue, we introduce a novel unified operator that combines several regularized RL operators into a general framework that better targets peakier sampling distributions. Secondly, we offer a novel, robust RL perspective of this filtering process. The regularization can be interpreted as robustness to a compositional form of uncertainty in the proxy function (i.e., the true evaluation of a candidate differs from the proxy's evaluation). Our analysis leads us to a novel, easy-to-use algorithm we name trajectory general mellowmax (TGM): we show it identifies higher quality, diverse candidates than baselines in both synthetic and real-world tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses the problem of constructing/composing discrete objects that maximize an uncertain (learnt) proxy reward and are diverse. The starting point of the paper is Generative Flow Networks (GFNs) which learn to sample proportionally to some reward function. However the number of possible discrete objects is exponential to their length/number of their components. Thus sampling proportionally to the value of the reward function will tend to produce samples that are of low reward, simply because low reward samples will have collectively a larger probability of being selected than the few high reward samples. Thus the problem that the paper sets to address is given equation 1 of the paper:
$$ \max_{x_1, ..., x_k} \sum_i \exp{\Phi(x_i)} \text{ subject to } d(x_i, x_j) > \delta, \text{ eq. 1 } $$
where $\Phi(.)$ is the learnt reward function and $d(.,.)$ some distance measure meant to capture diversity in the set of selected structures. 

The paper casts this problem as learning a policy that is robust to the worst-case reward model, i.e. the value function that they seek to maximize is: 
$$v_{\mathcal R}^{\pi} (s) := \min_{ r \in \mathcal R} v^{\pi}(s)$$
Uncertain rewards create reward uncertainty sets and a robust policy should be optimal under the worst case reward; however at least as eq. 1 above is stated the initial optimization problem that the paper sets to address looks rather like a standard optimization problem and not a robust optimization problem.  

Given that one should learn a robust policy to address 1 then the paper proceeds with the following contributions:
* a regulariser that explicitly trades off between different regularisation functions associated with GFN, and ((soft)mellowmax), 
* an algorithm that uses the above regulariser and operates on the level of trajectories and samples from high reward regions
* a characterisation of the reward uncertainty sets (created by the uncertain reward) that are induced by convex regularises

The paper discusses the relation between GFNs, soft Bellman, and the (soft)-mellow max operators: 
* There is a direct equivalence between GFNs and soft Bellman (entropy regularised MDP/policy $H(\pi)$, both suffer from value accumulation: the value accumulation of a large number of suboptimal objects can overwhelm a single high reward object
* The mellowmax operator solves accumulation but suffers from dilution; a high reward object can get diluted by lower reward objects.
* The soft mellowmax addresses the dilution of mellowmax 

The paper argues for controlling the balance of accumulation and dilution and does so by addressing a regulariser that allows to seamslessly transition between the three operators.  GFNs and soft Bellman are entropy regularised MDP/policy $H(\pi)$); mellowmax/soft-mellowmax learn  KL regularised policies, $KL(\pi,d_s)$, where $d_s$ are the uniform distribution and the $softmax(Q)$ distributions respectively. The paper introduces the general mellow max regulariser that interpolates between the two as:
$$ q KL(\pi,d_s) + (1-q) (-H(\pi_s)) \text{ eq. 6 } $$
interpolating between policy entropy maximization and getting a policy that is close to the softmax of the $Q$ values, capturing GFNs, and (soft)-mellowmax for different values of $q$ and temperature. 

The paper then proceeds to introduce the trajectory general mellow max which builds upon eq 6 and operates over trajectories. The algorithm is evaluated over a set of synthetic and real world sequence generation benchmarks that show that TGM achieves high rewards against RL baselines, namely PPO and SAC, and it is on a par or better than GFN. The experiments show that TGM has a performance that is quite stable with respect to difference values of the interpolation coefficient .



In addition the paper discusses the equivalence of entropy regularised MDPs to robust MDPs with uncertain reward (robust RL). 

The paper proceeds with a discussion of the uncertainty sets introduced by the different regularisers, i.e. the entropy regulariser used in GFNs, the softmellowmax, and the general mellowmax. In particular the setting of discovering interesting structures one would like to maximize the proxy reward $\Phi$ while accounting for the difference, $\delta$, between $\Phi$ and the true uknown reward $r^*$, by learning a sampling distribution $p$ that satisfies:  
$$ \max_{p} \min_{\delta \in R} E_{x \sim p}[\Phi(x) + \delta(x)]  \text{ eq. 10 } $$
where $R$ is the reward uncertainty set. It attributes the failure of GFNs to maximise $\Phi$ to the fact that the uncertainty set produced by its regulariser does not contain the proxy reward.

### Strengths
* The introduces an interpolated regulariser which unifies three different operators and a respective algorithm that achieves the desiderata of oversampling high reward, demonstrated by the empirical evaluation
* The paper characterises the uncertainty reward sets associated with the different regularisers and provides an explanation on why GFlowNets do not oversample high reward regions by relying on their equivalence entropy regularised policies: their reward uncertainty set does not include the proxy reward.

### Weaknesses
As I have said in a previous review of the same paper: The paper addresses the property optimization problem for (robust) sequence generation in an RL setting. I understand that the core of the paper is the robust property optimization with RL [...] however there are different approaches to the same problem that are [still] not discussed in the related work. In particular there is quite some work in generative models that seek to do exactly the same, i.e. property optimization. 

See references provided in the previous review. I think it is still useful to discuss also such different modelling approaches to the same problem.

### Questions
* Robust optimization seeks to maximize an objective function in the worst case under uncertainties. The objective as given in equation (1) does not look like a robust optimization problem but rather takes $\Phi$ at face value and does not state anything about a worst-case setting. So why robust optimization is appropriate for eq. (1)? 
$$ \max_{x_1, ..., x_k} \sum_i \exp{\Phi(x_i)} \text{ subject to } d(x_i, x_j) > \delta, \text{ eq. 1 } $$

* how should I understand equation 10: 
$$ \max_{p} \min_{\delta \in R} E_{x \sim p}[\Phi(x) + \delta(x)], \text{ eq. 10 } $$
in the sense that $\delta$ is not something we have access to, is this more a theoretical concept to study the properties of the different regularisers? Reading it as is stated above there are two things that can be adjusted/learnt the sampling distribution $p$ and the $\delta$ function (or I guess equivalently the $\Phi$), however $\Phi$ is fixed. In addition the max/min problem reads as: find the distribution that maximizes $\Phi$ under the lowest error, this does not sound like optimization in the worst case setting.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces trajectory general mellowmax, a reinforcement learning method for discrete compositional generation tasks like molecular or protein design. The authors highlight how methods that learn a policy whose density/distribution is proportional to the reward (such as GFlowNets) tend to produce diverse but generally low-quality samples (since the high reward, low density samples are drowned out by the low reward, high density samples). To address this, the authors propose the general mellowmax operator, which unifies and interpolates between existing soft RL operators to better balance diversity and quality. Experiments on synthetic and biological benchmarks show that the proposed approach consistently generates higher reward and still diverse candidates than prior methods.

### Strengths
This paper is original in unifying multiple soft RL operators through the proposed general mellowmax framework. The proposed trajectory general mellowmax is significant in bridging GFlowNets and robust RL under a common theoretical perspective. The paper contains rigorous mathematical derivations, a well motivated algorithmic design, and thorough experiments across both synthetic and real wold biological design tasks. The problem motivation, operator formulation, and empirical findings are clearly articulated, and the work is significant in meaningfully addressing a key limitation of excess diversity (at the expense of high reward designs) in GFlowNets / similar constructions.

### Weaknesses
The mathematical notation can feel a bit dense at times and I found myself getting occasionally lost on what the various key parameters are meant to govern. It would be helpful to have a better illustration or explanation on the role of alpha, q, and omega in TGM - this is attempted in Figure 3, but requires the reader to go to other portions of the paper to understand what the axes are describing. So in general, I feel that the presentation and clarity of the paper could be improved. I also feel like the experimental section would benefit from including a baseline that is also designed to address the mode concentration issues of GFlowNets, such as temperature conditioned GFNs.

### Questions
Are there ways to experimentally demonstrate the connection to robust RL? E.g., by introducing noise into the proxy reward to evaluate how TGM performs relative to GFlowNets?

Have you compared TGM against established alternatives that address policy over smoothness / diversity such as temperature conditioned GFlowNets? I saw temperature conditioned GFNs mentioned in the paper, but not used as a baseline for comparison.

For the biological sequence design experiments in 6.2, it would be useful to understand how the diversity of the generated sequences varies across the baselines and TGM with different choices of q.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Trajectory General Mellowmax, a novel algorithm that outperforms GFN in finding high-reward targets on discrete state spaces. The experiments conducted in both the synthetic environment and the biological sequence search demonstrate that the newly proposed methods are effective and relevant.

### Strengths
- The proposed methods are novel and theoretically grounded, with a clear motivation.
- The numerical experiments are comprehensive, showing the relevance of TGM

### Weaknesses
- The writing is pretty challenging to follow.
- The motivating examples of "high reward path dominated by many low reward paths" seem to be the issue of the designed target distribution rather than the problem of the learning algorithm. For a 0-1 reward, if we reduce the temperature of the distribution to be proportional to exp(r/gamma) where gamma -> 0, the valid target examples are only going to be concentrated on positive reward ones, and the cases presented in the motivating examples no longer seem to be valid.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce the Trajectory General Mellowmax operator, motivated by the framework of regularized reinforcement learning operators, which tend to encourage more concentrated policy distributions, and seem to show improvements across datasets.

### Strengths
1. The paper addresses an important problem—developing a reinforcement learning method that can generate diverse responses while also producing high-scoring candidates useful for biological and chemical discovery.

2. The paper is well written and generally easy to follow.

### Weaknesses
1. It seems that several important baseline comparisons are missing (see questions for details).

2. I’m not sure it’s appropriate to compare GFN with TGM, since GFN involves training a Q-network and therefore requires significantly more computation. The compute budget should be made fair for a valid comparison (see questions for details).

3. I also find the motivating example somewhat unclear or unconvincing (see questions for details).

4. Some of the figures are hard to understand.

### Questions
1. It seems that several baselines ([A], [B], and [C]) are missing. Since these approaches also use a Q-function, comparing against them would make the evaluation fairer. In contrast, GFlowNets do not rely on a Q-function, which makes direct comparison more challenging, as their learning dynamics and compute requirements differ fundamentally. Moreover, in the middle panel of Figure 4, the GFlowNet performance appears to be increasing, does this suggest that GFlowNets could further improve with additional compute (e.g., equivalent to what is used for Q-function training)?

2. In the intuition for the motivating example, doesn’t the argument implicitly depend on the quality of the reward model? For instance, if the rewards were accurate, then each sequence wouldn’t necessarily need to have its reward lower bounded by a constant r, since poorly performing samples could simply receive large negative rewards. Wouldn’t this issue therefore be mitigated by using a more accurate reward model?

3. In general, GFlowNets aim to generate diverse and novel samples. With that in mind, could the paper report diversity and novelty metrics as discussed in [E] and [F]? Additionally, similar to Figure 5 in [D] and Figure 2 in [E], would it be possible to conduct synthetic experiments to evaluate whether TGM successfully recovers all modes?

4. Could you clarify how to interpret Figure 2? What exactly should the reader take away from it? (I don't think I understand the caption)

5. In lines 403–404, could you explain why $\delta = 28?$

---

**References**

[A] Mohammadpour, Sobhan, Emmanuel Bengio, Emma Frejinger, and Pierre-Luc Bacon. "Maximum entropy gflownets with soft q-learning." In International Conference on Artificial Intelligence and Statistics, pp. 2593-2601. PMLR, 2024.

[B] Lau, E., Lu, S., Pan, L., Precup, D., & Bengio, E. (2024). Qgfn: Controllable greediness with action values. Advances in neural information processing systems, 37, 81645-81676.

[C] Tiapkin, D., Morozov, N., Naumov, A., & Vetrov, D. P. (2024, April). Generative flow networks as entropy-regularized rl. In International Conference on Artificial Intelligence and Statistics (pp. 4213-4221). PMLR.

[D] Zhang, D., Malkin, N., Liu, Z., Volokhova, A., Courville, A., & Bengio, Y. (2022, June). Generative flow networks for discrete probabilistic modeling. In International Conference on Machine Learning (pp. 26412-26428). PMLR.

[E] Jain, M., Bengio, E., Hernandez-Garcia, A., Rector-Brooks, J., Dossou, B. F., Ekbote, C. A., ... & Bengio, Y. (2022, June). Biological sequence design with gflownets. In International Conference on Machine Learning (pp. 9786-9801). PMLR.

[F] Ekbote, C., Jain, M., Das, P., & Bengio, Y. (2022). Consistent training via energy-based gflownets for modeling discrete joint distributions. arXiv preprint arXiv:2211.00568.

### Soundness
4

### Presentation
4

### Contribution
2
