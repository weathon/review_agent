# Reflective Policy Optimization

- Decision: Reject
- Scores: 8, 5, 3, 3

## Abstract
On-policy reinforcement learning methods, such as Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO), often require significant data to be collected at each update, giving rise to issues of sample inefficiency. This paper introduces a novel extension to on-policy methods called Reflective Policy Optimization (RPO). RPO's fundamental objective is amalgamating prior and subsequent state and action information from trajectory data to optimize the current policy. This approach empowers the agent to engage in introspection and introduce modifications to its actions within the current state to a certain degree. Furthermore, theoretical analyses substantiate that our proposed method not only upholds the crucial property of monotonically improving policy performance but also adeptly contracts the solution space of the optimized policy, consequently expediting the training procedure. We empirically demonstrate the feasibility and efficacy of our approach in reinforcement learning benchmarks, culminating in superior performance in terms of sample efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a new policy optimization method named reflective policy optimization based on expanding surrogate function via condition state-action visitation distributions. This paper also provided a clipped surrogate function(following PPO) for efficient calculating.
Compared with current policy optimization methods, this reflective policy optimization performs better on many benchmarks.

### Strengths
This paper proposes a novel method with a tighter bound than TRPO. The clipped optimization objective provided by this method is simple and general enough and can be directly applied to many existing methods to replace the PPO objective. It shows improved or comparable performance in the experiments, however again considering the simplicity and generality of the method, this method is valuable.

Overall, the paper is well-written and easy to follow. Adding a detailed expression of the loss function for k=2 before Theorem 4.2 would have made it easier for me to understand.

### Weaknesses
It is better to add some experiments on the environments with discrete action space, e.g. MinAtar[1].

The gap between the original reflective policy optimization and the clipped version cannot be ignored.

The method is sensitive to some important hyperparameters, as shown in Figure 4, in Swimmer and Walker2d.


[1] Young, K. Tian, T. (2019). MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments. arXiv preprint arXiv:1903.03176.

### Questions
In the ablation experiment for k, when k increases the variance of the objective function will increase significantly, which puts forward a high requirement for fine adjustment of $\epsilon$ and $\beta$. So I want to know what is the setting of these hyperparameters in the ablation experiment. How to set such hyperparameters to get a reliable result, choosing the right k?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, authors propose a Reflective Policy Optimization(RPO) that considers state-action pairs for n-steps in the policy optimization algorithm. The paper theoretically shows that the relationship between the performance of two policies depends on the next state-action pair and performance’s lower bound can be similar to Trust Region Policy Optimization(TRPO). Authors prove that as n increases, the solution space shrinks and can be contained in the solution space of the (n-1) steps. The practical implementation is based on Proximal Policy Optimization(PPO), and experimental results show that it has better convergence speed and average return performance than the baseline algorithm in cliff-walking and mujoco environments.

### Strengths
1) For two policies, authors propose to maximize a generalized lower bound that directly takes into account that policy performance is related to the next state-action pair. In particular, they interestingly prove that the optimized policy is reflective through some theory and show that TRPO is a special case of this method, which allows the policy to be monotonically improved.
2) In this paper, authors consider multi-step RL directly from the policy optimization perspective, which is different from the previously proposed multi-step value estimation. In addition, for the practical implementation of the proposed algorithm, the objective function based on PPO was proposed and the correlation between hyperparameters was explained.
3) In the cliff walking environment, the proposed algorithm was experimentally shown to fall off the cliff less and reach the goal faster than the existing algorithms, which is consistent with the performance expected by the authors at the beginning of the paper. It also improved the average return and convergence speed compared to the existing algorithms in the mujoco environment, which is commonly tested in policy optimization.

### Weaknesses
1) The theorems and proofs proposed in the paper are interesting, but the performance shown in the experiments does not seem to be much different from the performance of the baselines. In particular, there are many mentions of convergence speed, but the results shown do not show a significant improvement over existing methods. It is necessary to show the performance improvement in an environment where the subsequent state-action can be better considered.
2) The clipping working environment that the authors consider seems a little less relevant to the general field of policy optimization. The proposed environment is more appropriate to be considered in a constrained reinforcement learning or safe reinforcement learning environment where dangerous situations should be avoided. Therefore, the comparison target and baseline algorithm should be the constrained reinforcement learning or safety reinforcement learning algorithm.
3) In the introduction section of the paper, it is argued that existing policy optimization algorithms do not directly consider the impact of subsequent state-actions in the trajectories. Also, in the example, if the agent repeatedly visits the cliff state, it is dangerous because agent is likely to act to fall off the cliff. However, in policy optimization, since there is a discounted sum of reward term, it is possible to provide feedback on the cliff state and the falling action. Therefore, the motivation of the proposed method is difficult to understand, as it allows us to consider the impact of subsequent state action under the influence of rewards.
4) The proposed algorithm increases the number of hyperparameters as the value of k increases. The authors claim that k=2,3 is suitable because of the stability of learning, but even with k=3, it requires 5 hyperparameters to adjust the clipping and learning rate. In addition, the experiments show the case of k=2, and the best performing hyperparameters are not consistent across experimental environments. Therefore, it can be said that the algorithm is sensitive to hyperparameters.

### Questions
1) Please answer the questions posed in the weaknesses
2) In Section 4, a new generalized lower bound is defined via Theorem 4.1. In this part, the authors introduce a new surrogate objective function, which has a slightly different semantics than the surrogate objective function in Theorem 3.1, just minus one. Also, they say that they directly optimize the information of current and future state-action pairs, can you explain what the difference is, or can you compare the difference experimentally?
3) The paper states that it applied multi-step RL directly from a policy optimization point of view, but it seems that Generalized Adversarial Estimation (GAE) was also used when training. could you tell me if there was any correlation between the GAE parameters and multi-step k in experiments?
4) The paper shows that the convergence speed increases with the value of K, but the performance is only shown in three environments. Can you explain whether the performance varies significantly depending on the difficulty of the environment or the dimensionality of the state and action?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper considers policy optimization from the trajectory perspective and proposes to optimize polices with the information from both the current state-action and the subsequent state-action pairs. To achieve this, the paper unfolds the performance improvement lemma over time steps to get the unrolled surrogate objective, i.e., the generalized surrogate, and a “residual” term. The paper then shows the “residual term” can be well bounded, yielding a policy improvement strategy: by iteratively pushing up the lower bound. The paper then use the same clipping scheme from the PPO and forms the Reflective Policy Optimization (RPO). This method has been evaluated on Mujoco control benchmarks and showed good performance.

### Strengths
### Originality
The idea of consider the subsequent state-action pairs in the policy optimization sounds interesting and novel. The theory (if correct) can add new insights into the policy gradient methods when considering unrolling it for a long horizon.

### Weaknesses
### Quality & significance
**There can be some technical misstatement & errors in the main contributions**.

First, the following statement can be erroneous: “If you recombine the above equation $(r_0-1)r_1 A^{\hat{\pi}}(s_1, a_1)]\cdot r_1 >0$, optimizing it will be found to increase the probability of $a_1$ . However, $A_{\hat{\pi}}(s_1 , a_1) < 0$, we should decrease the probability of $a_1$ . This would present a contradiction.” **Optimizing this objective can result in a bit more complicated consequence than what’s been stated here**: since the policy in both $r_0$ and $r_1$ is parameterized with a same set of parameters, optimizing the objective may increase the probability of $a_0$ and thus tip over the sign of $r_0-1$.  The problem is $a_0$ and $a_1$ are from the same parametrized policy. 


Second, **the monotonic improvement statement of Theorem 4.1 can be erroneous**. Consider a special case of $k=1$, which yields the TRPO bound as suggested by the paper (assume this statement is correct), then the lower bound will be $\alpha_0 \hat{L}_0(\pi, \hat{\pi}) - \hat{C}_1(\pi, \hat{\pi})$, 

which equals to $E_{s_0, a_0\sim\rho^{\hat{\pi}}} [r_0 A^{\hat{\pi}}(s_0, a_0)] - \frac{\gamma R_{max}}{(1-\gamma)^3}||\pi - \hat{\pi}||_1^2$. This is the surrogate objective implied by Theorem 4.1 and should be improved upon. However, **this surrogate objective is strictly negative for $\gamma=0.995$ used by this paper**: consider the first term

$E_{s_0, a_0\sim\rho^{\hat{\pi}}} [r_0 A^{\hat{\pi}}(s_0, a_0)] = E_{s_0, a_0\sim\rho^{\hat{\pi}}} [ (r_0-1) A^{\hat{\pi}}(s_0, a_0)] $ (this is because the advantage is defined on $\hat{\pi}$ and the integral over $\hat{\pi}$ will be zero. Hence, 

$E_{s_0, a_0\sim\rho^{\hat{\pi}}} [ (r_0-1) A^{\hat{\pi}}(s_0, a_0)] $ 

$\leq \frac{  ||\pi - \hat{\pi}||_1^2 Rmax }{1-\gamma} $ (this is a direct result from the proof of Corollary A.2 in appendix, i.e., $G_1$). 

Thus, 

$E_{s_0, a_0\sim\rho^{\hat{\pi}}} [r_0 A^{\hat{\pi}}(s_0, a_0)] - \frac{\gamma R_{max}}{(1-\gamma)^3}||\pi - \hat{\pi}||_1^2$

$\leq \frac{  ||\pi - \hat{\pi}||_1^2 Rmax }{1-\gamma} - \frac{\gamma Rmax}{(1-\gamma)^3}||\pi - \hat{\pi}||_1^2$

$\leq \frac{  ||\pi - \hat{\pi}||_1^2 Rmax }{1-\gamma} [ 1- \frac{\gamma}{(1-\gamma)^2}] < 0$ (for $\gamma =0.995)$

So, the maximum value of the lower bound is strictly negative. **It turns out to be impossible to get "a monotonically improving sequence of policies $\{\pi_i\}_{i=1}^{\infty}$ satisfying $\eta(\pi_0) \leq \eta(\pi_1) \leq ...$" for this case $k=1$**.

Please correct me if my understanding is wrong. 

### Clarity
There are some confusing points and unclear definitions. See in my questions.

### Questions
1. I don’t see the point in the examples of “cliff” and “treasure” states in the second paragraph of Introduction Section. Would the value function just suffice whether a state is favourable?

2. Most of the citations are using a wrong format. It seems that the author misused the citep & citet. 

3. there is a notation issue in the second line of $\eta(\pi) - \eta(\hat{\pi})$. The first term on RHS should be with (s_0, a_0).

4. The $||\pi - \hat{\pi}||_1$ has not been defined in the paper. 

5. The definition of the conditional occupancy measure is a bit confusing. From the proofs in the appendix, this conditional occupancy measure is a distribution with the conditions on the initial state-action distribution, i.e., the stationary state distribution induced from a predefined initial state distribution. Then I’m not sure if $(s_0, a_0)\sim\rho$ and $(s_i, a_i)\sim\rho(\cdot|s_{i-1}, a_{i-1})$ can be defined coherently. 

Some language issues (those do not affect my assessment)
* “between the performance of \pi and \hat{\pi} from a trajectory-based.”

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a method called Reflexive Policy Optimization, which the authors claim uses information “from trajectory data” more efficiently to optimize a policy. The authors claim that agents have “introspection” and “reflect on prior experience”. The method is claimed to have guaranteed monotonic progress improvement on the original policy performance objective and that it “contracts the solution space of the optimized policy”, thus “expediting” the training procedure. Moreover, the authors claim the method is superior to baselines on Mujoco tasks.

I didn’t really understand the motivation for the paper, and I found it very confusing and difficult to follow. I expand in the following sections.

### Strengths
I believe the authors want to say that their method performs a kind of hindsight credit assignment \citep{harutyunyan2019}, but this is just a hunch. I didn't really understand from the text.
The paper appears to be backed by theory and support their superior performance claims with empirical results. Unfortunately, I did not really understand the motivation and method to assess these  properly. Please see below my points of confusion. I am happy to revise my score if the authors the motivation and goal of the paper, and expand on the points below.


@article{harutyunyan2019,
  author       = {Anna Harutyunyan and
                  Will Dabney and
                  Thomas Mesnard and
                  Mohammad Gheshlaghi Azar and
                  Bilal Piot and
                  Nicolas Heess and
                  Hado van Hasselt and
                  Greg Wayne and
                  Satinder Singh and
                  Doina Precup and
                  R{\'{e}}mi Munos},
  title        = {Hindsight Credit Assignment},
  journal      = {CoRR},
  volume       = {abs/1912.02503},
  year         = {2019},
  url          = {http://arxiv.org/abs/1912.02503},
  eprinttype    = {arXiv},
  eprint       = {1912.02503},
  timestamp    = {Wed, 20 Apr 2022 07:47:18 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1912-02503.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

### Weaknesses
First, I do not understand why the authors believe proximal methods do not already account for the true gradient of the policy which also considers the contribution through the stationary distribution.  Indeed, in the original CPI paper \citep{kakade}, the authors first derive the form of the bound that contains the stationary distribution with the current policy, and not a prior policy, then use a mixture policy to ensure that the new policy is “close” to the prior one and justify replacing the current stationary distribution with the previous one. However, a lot of work has been done since, and it is known that proximal PG methods \citep{bhandari21, shani2019, vaswani2021} (and algorithmic implementation \citep{li2022analytical}, PPO, TRPO, MDPO \citep{tomar2020}, MPO \citet{abdolmaleki2018} etc.) are functional-gradient methods \citep{vaswani2021}, in the sense that the lower bound the algorithms optimize are linearizations of the policy performance objective in the direct policy representation. The functional gradient is $d_{\pi_t}^\top Q_{\pi_t}$. In that sense the stationary distribution that the algorithms use is w.r.t. (with respect to) the previous policy (see \citet{bandhari2019, vaswani2021, sutton2000, agarwal2019}). As they follow the true policy gradient, these methods are convergent \citep{bhandari21, vaswani2021, xiao2022, johnson2023}, at least their theoretical versions, under adaptive step sizes. The true policy gradient does take into account the gradient through the stationary distribution.
I did not understand at all the examples they give, which seem to say that “0” has some special meaning when it comes to the reward. 

The text is very unclear, vague, and grammatically incorrect, which makes it very difficult to follow the authors’ arguments.
In the motivating introduction, the authors discuss multi-step methods, but this is completely out of context, and use words which have not been defined and are not scientifically rigorous, like “we give a nice theory.”, “the policy can be reflective”, “ this empowers the agent to engage in introspection and introduce modifications to its actions within the current state to a certain degree”, “some admirable algorithms”, “direct optimization of this generalized surrogate objective function may have to be done very carefully”, “RPO efficiently utilizes ‘good’ experiences, makes adjustments based on ‘bad’ experiences’”

With respect to scientific correctness, at some point the authors completely remove a term from the bound saying that “this term “1” may adversely affect policy optimization. I did not understand that at all. The authors say “for $k = 1$, the $l_1$ norm constraints are replaced by KL constraints”, but this to me makes no sense at all. 

The notation is also confusing, as it seems the authors reuse “r” to represent the importance sampling ratio between the current policy and a prior one. It is unclear what “k” represents as it is not defined, just implied, and seems very important throughout the paper, with multiple reference and impacting the algorithms’ empirical performance.

Then, I also did not really understand the emphasis on the exploration-exploitation coupling on on-policy methods since the whole point of the paper is to improve off-policy methods like PPO and TRPO. These proximal methods are off-policy algorithms, which is the entire point of the efficiency of the methods, the fact that it can reuse prior experience (from the previous policy) multiple times to minimize the lower bound. Otherwise, on-policy policy gradient methods like \citet{sutton2000} (or algorithmic implementations with parallel streams of experiences like A3C \citep{mnih16}, Impala \citep{espeholt18}) directly use the PGT \citep{sutton2000} w.r.t the parameter vector and rebuild the linear lower bound at each time-step so they don’t need a lower bound surrogate objective that is quasi-concave and can be optimized for multiple updates. 


The authors claim the method maintains mototonic improvement, but it is known that proximal-gradient methods, with adaptive step size have this property \citep{alfano2023, chen2022, xiao2022, johnson2023}, since they are functional-gradient methods, at least for tabular direct or softmax parametrizations.

@misc{li2022analytical,
      title={An Analytical Update Rule for General Policy Optimization}, 
      author={Hepeng Li and Nicholas Clavette and Haibo He},
      year={2022},
      eprint={2112.02045},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}

@InProceedings{bhandari21,
  title = 	 { On the Linear Convergence of Policy Gradient Methods for Finite MDPs },
  author =       {Bhandari, Jalaj and Russo, Daniel},
  booktitle = 	 {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {2386--2394},
  year = 	 {2021},
  editor = 	 {Banerjee, Arindam and Fukumizu, Kenji},
  volume = 	 {130},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--15 Apr},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v130/bhandari21a/bhandari21a.pdf},
  url = 	 {https://proceedings.mlr.press/v130/bhandari21a.html},
  abstract = 	 { We revisit the finite time analysis of policy gradient methods in the one of the simplest settings: finite state and action MDPs with a policy class consisting of all stochastic policies and with exact gradient evaluations. There has been some recent work viewing this setting as an instance of smooth non-linear optimization problems, to show sub-linear convergence rates with small step-sizes. Here, we take a completely different perspective based on illuminating connections with policy iteration, to show how many variants of policy gradient algorithms succeed with large step-sizes and attain a linear rate of convergence. }
}
@article{mnih16,
  author       = {Volodymyr Mnih and
                  Adri{\`{a}} Puigdom{\`{e}}nech Badia and
                  Mehdi Mirza and
                  Alex Graves and
                  Timothy P. Lillicrap and
                  Tim Harley and
                  David Silver and
                  Koray Kavukcuoglu},
  title        = {Asynchronous Methods for Deep Reinforcement Learning},
  journal      = {CoRR},
  volume       = {abs/1602.01783},
  year         = {2016},
  url          = {http://arxiv.org/abs/1602.01783},
  eprinttype    = {arXiv},
  eprint       = {1602.01783},
  timestamp    = {Mon, 13 Aug 2018 16:47:40 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/MnihBMGLHSK16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@InProceedings{espeholt18,
  title = 	 {{IMPALA}: Scalable Distributed Deep-{RL} with Importance Weighted Actor-Learner Architectures},
  author =       {Espeholt, Lasse and Soyer, Hubert and Munos, Remi and Simonyan, Karen and Mnih, Vlad and Ward, Tom and Doron, Yotam and Firoiu, Vlad and Harley, Tim and Dunning, Iain and Legg, Shane and Kavukcuoglu, Koray},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {1407--1416},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--15 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf},
  url = 	 {https://proceedings.mlr.press/v80/espeholt18a.html},
  abstract = 	 {In this work we aim to solve a large collection of tasks using a single reinforcement learning agent with a single set of parameters. A key challenge is to handle the increased amount of data and extended training time. We have developed a new distributed agent IMPALA (Importance Weighted Actor-Learner Architecture) that not only uses resources more efficiently in single-machine training but also scales to thousands of machines without sacrificing data efficiency or resource utilisation. We achieve stable learning at high throughput by combining decoupled acting and learning with a novel off-policy correction method called V-trace. We demonstrate the effectiveness of IMPALA for multi-task reinforcement learning on DMLab-30 (a set of 30 tasks from the DeepMind Lab environment (Beattie et al., 2016)) and Atari57 (all available Atari games in Arcade Learning Environment (Bellemare et al., 2013a)). Our results show that IMPALA is able to achieve better performance than previous agents with less data, and crucially exhibits positive transfer between tasks as a result of its multi-task approach.}
}

@article{vaswani2021,
  author       = {Sharan Vaswani and
                  Olivier Bachem and
                  Simone Totaro and
                  Robert Mueller and
                  Matthieu Geist and
                  Marlos C. Machado and
                  Pablo Samuel Castro and
                  Nicolas Le Roux},
  title        = {A functional mirror ascent view of policy gradient methods with function
                  approximation},
  journal      = {CoRR},
  volume       = {abs/2108.05828},
  year         = {2021},
  url          = {https://arxiv.org/abs/2108.05828},
  eprinttype    = {arXiv},
  eprint       = {2108.05828},
  timestamp    = {Wed, 18 Aug 2021 19:45:42 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2108-05828.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@article{agarwal2019,
  author       = {Alekh Agarwal and
                  Sham M. Kakade and
                  Jason D. Lee and
                  Gaurav Mahajan},
  title        = {Optimality and Approximation with Policy Gradient Methods in Markov
                  Decision Processes},
  journal      = {CoRR},
  volume       = {abs/1908.00261},
  year         = {2019},
  url          = {http://arxiv.org/abs/1908.00261},
  eprinttype    = {arXiv},
  eprint       = {1908.00261},
  timestamp    = {Fri, 09 Aug 2019 12:15:56 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1908-00261.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@article{shani2019,
  author       = {Lior Shani and
                  Yonathan Efroni and
                  Shie Mannor},
  title        = {Adaptive Trust Region Policy Optimization: Global Convergence and
                  Faster Rates for Regularized MDPs},
  journal      = {CoRR},
  volume       = {abs/1909.02769},
  year         = {2019},
  url          = {http://arxiv.org/abs/1909.02769},
  eprinttype    = {arXiv},
  eprint       = {1909.02769},
  timestamp    = {Mon, 16 Sep 2019 17:27:14 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1909-02769.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@article{abdolmaleki2018,
  author       = {Abbas Abdolmaleki and
                  Jost Tobias Springenberg and
                  Yuval Tassa and
                  R{\'{e}}mi Munos and
                  Nicolas Heess and
                  Martin A. Riedmiller},
  title        = {Maximum a Posteriori Policy Optimisation},
  journal      = {CoRR},
  volume       = {abs/1806.06920},
  year         = {2018},
  url          = {http://arxiv.org/abs/1806.06920},
  eprinttype    = {arXiv},
  eprint       = {1806.06920},
  timestamp    = {Mon, 13 Aug 2018 16:48:15 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1806-06920.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@inproceedings{sutton2000,
 author = {Sutton, Richard S and McAllester, David and Singh, Satinder and Mansour, Yishay},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Solla and T. Leen and K. M\"{u}ller},
 pages = {},
 publisher = {MIT Press},
 title = {Policy Gradient Methods for Reinforcement Learning with Function Approximation},
 url = {https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf},
 volume = {12},
 year = {1999}
}

@misc{johnson2023,
      title={Optimal Convergence Rate for Exact Policy Mirror Descent in Discounted Markov Decision Processes}, 
      author={Emmeran Johnson and Ciara Pike-Burke and Patrick Rebeschini},
      year={2023},
      eprint={2302.11381},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
@misc{xiao2022,
      title={On the Convergence Rates of Policy Gradient Methods}, 
      author={Lin Xiao},
      year={2022},
      eprint={2201.07443},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
@article{tomar2020,
  author       = {Manan Tomar and
                  Lior Shani and
                  Yonathan Efroni and
                  Mohammad Ghavamzadeh},
  title        = {Mirror Descent Policy Optimization},
  journal      = {CoRR},
  volume       = {abs/2005.09814},
  year         = {2020},
  url          = {https://arxiv.org/abs/2005.09814},
  eprinttype    = {arXiv},
  eprint       = {2005.09814},
  timestamp    = {Fri, 22 May 2020 16:21:28 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2005-09814.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@misc{alfano2023,
	title        = {Linear Convergence for Natural Policy Gradient with Log-linear Policy Parametrization},
	author       = {Carlo Alfano and Patrick Rebeschini},
	year         = 2023,
	eprint       = {2209.15382},
	archiveprefix = {arXiv},
	primaryclass = {cs.LG}
}
@inproceedings{chen2022,
	title        = {Sample Complexity of Policy-Based Methods under Off-Policy Sampling and Linear Function Approximation},
	author       = {Chen, Zaiwei and Theja Maguluri, Siva},
	year         = 2022,
	month        = {28--30 Mar},
	booktitle    = {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
	publisher    = {PMLR},
	series       = {Proceedings of Machine Learning Research},
	volume       = 151,
	pages        = {11195--11214},
	url          = {https://proceedings.mlr.press/v151/chen22i.html},
	editor       = {Camps-Valls, Gustau and Ruiz, Francisco J. R. and Valera, Isabel},
	pdf          = {https://proceedings.mlr.press/v151/chen22i/chen22i.pdf},
	abstract     = {In this work, we study policy-based methods for solving the reinforcement learning problem, where off-policy sampling and linear function approximation are employed for policy evaluation, and various policy update rules (including natural policy gradient) are considered for policy improvement. To solve the policy evaluation sub-problem in the presence of the deadly triad, we propose a generic algorithm framework of multi-step TD-learning with generalized importance sampling ratios, which includes two specific algorithms: the $\lambda$-averaged $Q$-trace and the two-sided $Q$-trace. The generic algorithm is single time-scale, has provable finite-sample guarantees, and overcomes the high variance issue in off-policy learning. As for the policy improvement, we provide a universal analysis that establishes geometric convergence of various policy update rules, which leads to an overall $\Tilde{\mathcal{O}}(\epsilon^{-2})$ sample complexity.}
}

### Questions
See above.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
