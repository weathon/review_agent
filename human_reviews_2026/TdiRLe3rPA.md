# From Ticks to Flows: Dynamics of Neural Reinforcement Learning in Continuous Environments

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
We present a novel theoretical framework for deep reinforcement learning (RL) in continuous environments by modeling the problem as a continuous-time stochastic process, drawing on insights from stochastic control.
Building on previous work, we introduce a viable model of actor–critic algorithm that incorporates both exploration and stochastic transitions.
For single-hidden-layer neural networks, we show that the state of the environment can be formulated as a two time scale process: the environment time and the gradient time.
Within this formulation, we characterize how the time-dependent random variables that represent the environment's state and estimate of the cumulative discounted return evolve over gradient steps in the infinite width limit of two-layer networks.
Using the theory of stochastic differential equations, we derive, for the first time in continuous RL, an equation describing the infinitesimal change in the state distribution at each gradient step, under a vanishingly small learning rate.
Overall, our work provides a novel nonparametric formulation for studying overparametrized neural actor-critic algorithms.
We empirically corroborate our theoretical result using a toy continuous control task.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This article presents a continuous-time formulation of the dynamics of episodic actor-critic reinforcement learning. The goal of the paper is to provide a characterisation of the coupling between the environment and learning dynamics, going beyond the usual separation of these along their independent time scales. The authors build upon a range of existing works which bring stochastic analysis and control tools to bear in the analysis of RL (exploratory dynamics, Itô expansions, numerical schemes *à la* Kushner/Dupuis). Of course, this problem can’t be captured without further assumptions. The author’s assumptions are quite reasonable: smooth control-affine dynamics (which I would say is already above average complexity), linearised 2-layer hyperbolic tangent neural networks for the agent and critic. Combining infinite-width approximations of the neural nets with statistical theorems for the environment allows for the results, which are asymptotic normality characterisations of the changes in the value and the policy. This approach is representative of the ongoing convergence of stochastic analysis/control and Neural Network limits for RL, and is one of its first convincing and tangible applications to reinforcement learning.

### Strengths
_Soundness and Clarity_:
- I appreciated the down-to-earth tone of the paper: the results are presented first and foremost for what they rigourously are, without resorting to window dressing. The authors present clear assumptions and clearly discuss their results in the context thereof, which I highlight amongst the broader soundness and rigour of the article. 

_Presentation and Clarity_:
- Explanations of both RL methodology and control are quite extensive, and, I think, make the topics approachable to either community.

- Appendices are extensive and pedagogical. I would like to express my appreciation, in particular, for the efforts the authors put into having examples throughout the background, which bridge over to the calculations performed afterwards.

_Contribution, Quality, & Originality_: The approach taken by the authors is a difficult but promising one, and the results are noteworthy both for their inherent quality and the future avenues of research they seem to make plausible.

### Weaknesses
_Soundness_:
- The experimental section is quite underwhelming. Section 7.1 presents a clear visual argument and is fine (except the above point about graph legibility), but section 7.2 hardly contributes to the paper in my opinion. Note that I do not consider such an experimental validation necessary.

_Presentation_:
- Figures 2 and 3 are illegible due to the legend and labels being too small. 

- The appendices sorely need a table of contents. While the calculations themselves are detailed and well-explained, it is hard to keep track of the bigger picture due to their sheer size. Perhaps the authors would like to consider proof diagrams to better connect the different parts of the appendices. 

- There are quite a few typos in the body, the references, and the appendices. Most of them are inconsequential, but they distract the reader in an already dense and, at times, disorienting work.

### Questions
I have many *research* questions about the work, but none would affect my evaluation of it. Thus, I will refrain from bothering the authors and reviewers with them.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes a continuous-time theoretical framework for neural reinforcement learning (RL), bridging stochastic control and deep RL theory. The authors model actor-critic learning as a two-timescale stochastic process with an environment time and a gradient time and derive stochastic differential equations that describe the infinitesimal evolution of the state distribution under infinite-width linearized neural networks with vanishing learning rates. Theoretical results are complemented by small-scale empirical validation on a linear-quadratic regulator (LQR) environment.

### Strengths
- The combination of NTK methods from deep learning to study gradient updates and SDEs from control theory to study environment updates to provide a unified nonparametric framework across two timescales is neat
- Clear theoretical construction by leveraging Ito-Taylor expansions and martingale CLTs to formally connect gradient-time dynamics and environment evolution
- The LQR experiment confirms that the proposed dynamics can be simulated and produce expected theoretical behavior

### Weaknesses
- The entire analysis relies on single hidden layer networks, smooth dynamics, and small learning rates. These conditions, while analytically convenient, make the results difficult to generalize to realistic RL systems.
- The key theorems depend on the infinite-width and vanishing learning-rate limits. No discussion is provided on how these approximations break down for practical networks, nor whether finite-width corrections could meaningfully affect dynamics.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a novel theoretical framework for deep RL in continuous environments by modeling the problem as a continuous-time stochastic process, and it also provides a novel nonparametric formulation for studying overparametrized neural actor-critic algorithms.

### Strengths
The paper provides a strong theory for continuous time actor-critic model; providing a nonparametric formulation for studying overparameterized neural actor-critic algorithms.

### Weaknesses
The numerical experiments are very limitted, and only one-hidden-layer neural networks are considered.

### Questions
1. Extension to deeper architectures (multiple hidden layers): this would significantly strengthen the contribution. Also, please discuss the issue and add numerical experiments. 

2. Evaluation across diverse environments: please evaluate the method in multiple environments that vary in dynamics and difficulty.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
A framework for continuous-time actor-critic RL is introduced that accounts for noisy envrionments. Prior work is adapted by including exploratory stochastic dynamics that can be simulated faithfully in discrete time. Linearized single hidden layer neural networks are used for the actor and critic functions, allowing the formulation of a system of equations that captures the evolution of state over environment time and gradient steps up to an error term. Expressions for the gradients with respect to actor and critic parameters in continuous time are derived and used to create an episodic actor-critic algorithm. The algorithm is evaluated empirically on a one-dimensional toy example and superior exploration is shown when compared to an additive Wiener process.

### Strengths
The preliminaries are introduced in a succinct manner. The proposed framework appears to be very thorough incorporating all aspects of stochasticity in RL. The proof is lengthy and presumably rigorous. An algorithm listing sheds light on the presented approach.

### Weaknesses
The presented math is overwhelming and confusing at times with apparent inconsistencies of upper and lower case in symbols and of argument order.

It is stated early in the paper that "higher dimensional results follow" wich is later restated as "We believe high-dimensional results should follow.".

The empirical evaluation is very limited. Additional settings should be explored including higher-dimensional tasks.
The figures are very small and hard to read.

Minor errors:
- Line 82: "the RL agent can is"
- Line 108: $\Delta W$ suddenly upper case
- Lemma 4.2: First $v(x,t)$ then $v(T,x)$, are the arguments switched?
- Line 226: "The deterministic policy analog of Theorem 2 by Jia & Zhou (2022) gives an expression for **dicrete** time **the** policy gradient (Sutton et al., 1999) in continuous time setting"
- Theorem 4.3: the gradient is first lower case $g(t,x;\theta)$ then upper case $\mathcal G(t,x;\theta)$.
- Line 235: "... by sampling a single trajectory **to** and updating ..."
- Therome 6.1: extra word "estimate"?
- Line 376: "$d_{s}=d_{s}=1$", subscript is repeated

### Questions
- Why does the state-value function depend on time?
- You state that "Although Jia & Zhou (2022) provides empirical validation for a similar algorithm, they do not do so for the neural network actor and critic." Can reiterate the novelty in your work compared to theirs?
- Does the presented framework allow for an extension to multi-layer finit-width networks? Can you elaborate on how this might look like?
- A problem with infinite-width neural networks is that the gradients become neglibly small, effectively renouncing the capability for feature learning. Is this of relevance here?

### Soundness
4

### Presentation
2

### Contribution
3
