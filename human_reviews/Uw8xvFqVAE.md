# A representation-learning game for classes of prediction tasks

- Decision: Accept (poster)
- Scores: 8, 8, 6, 8

## Abstract
We propose a game-based formulation for learning dimensionality-reducing representations of feature vectors, when only a prior knowledge on future prediction tasks is available. In this game, the first player chooses a representation, and then the second player adversarially chooses a prediction task from a given class, representing the prior knowledge. The first player aims to minimize, and the second player to maximize, the regret: The minimal prediction loss using the representation, compared to the same loss using the original features. We consider the canonical setting in which the representation, the response to predict and the predictors are all linear functions, and the loss function is the mean squared error. We derive the theoretically optimal representation in pure strategies, which shows the effectiveness of the prior knowledge, and the optimal regret in mixed strategies, which shows the usefulness of randomizing the representation. For general representation, prediction and loss functions, we propose an efficient algorithm to optimize a randomized representation. The algorithm only requires the gradients of the loss function, and is based on incrementally adding a representation rule to a mixture of such rules.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The submission proposed a two-person game to solve the representation learning problem. It targets on a given class of tasks. A theoretical justification was made for the linear setting. A series of discussions were provided following the proposed algorithm for the general setting.

### Strengths
a) The submission proposed a game to solve the representation learning for a given class of tasks. The formulation is practical as one can control the relevant tasks.
b) The submission provided theoretical justifications for the linear setting to characterize the learned representations and the performance bounds (Theorems 2 and 3).

### Weaknesses
c) An algorithm is proposed to learn the representation for the general case. However, the evidence, from the theoretical or the practical perspective, is insufficient to determine the applicability of the proposed method.

### Questions
d) The analysis for the linear case seems to be adapted from the game theory. Could you please clarify the technical contribution, if any, in the analysis?
e) Is it possible to show the game will reach some situation, such as an equilibrium? Representations from the equilibrium may represent a negotiated outcome from both players.
f) Although Examples 6 -- 9 help us to understand the nature of Algorithm 1, one still cannot know Algorithm 1's applicability. Would it be better to compare Algorithm 1 with baselines to show the representations output from Algorithm 1 are better than the existing methods?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
They propose a game-based methodology to learn the classical machine learning problem of dimensionaity-reducing representations of feature vectors. On the machine learning they consider the linear setting and use the mean square error loss function. On the game theoretic side, they used mixed and pure strategies.

### Strengths
They propose an interesting way to study a very classical problem. The paper is well-written, and clear

### Weaknesses
The computation can be complex such as in solving Ap = b for the probability p in the mixed minmax strategy.

### Questions
The goal of RL is the minimize regret, while the goal of this seems to be to minimize MSE loss. There are some results relating to regret, but how does this relate to the MSE that results in applying this method? 

Also for Theorem 3, does this assume there are only finitely many possible feature vectors (or does it just assume that the dimension of the feature vectors is finite). Just to check my understanding, I assume the dimension of the covariance matrix matches the dimension of the feature vectors learned correct?

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
The authors propose a new game theoretic framework for learning low-dimensional representations $z= R(x) \in \mathbb{R}^r$ of unlabelled data $\\{\vec{x}_i\\} \subset \mathbb{R}^d$ (where $r \ll d$) in such a way that the learned representation would be useful for a variety of downstream learning tasks (specified by a class $\mathcal{F}$ of response functions).

The contributions are both theoretical (optimal linear representations for linear response etc) and practical/algorithmic (more general representations and response functions). This paper uses a different form of game than the one used by (Dubois et al 2020).
1. In (Dubois et al 2020), Player 1 chooses the learning task (input distribution, response function) and the score function. Then Player 2 (knowing the above) trains a representation for the input data. Finally, Player 1 evaluates the representation, using the chosen score function on an ERM classifier trained using the representation given by Player 2 (with IID data ~ (input distribution, response function)).

2. In this paper the game is a 2PZS game with: (i) Player 1 (representation player) chooses the representation mapping $R \in \mathcal{R}$ (ii) Player 2 chooses the response function $f \in \mathcal{F}$, where the data distribution ($P_{\mathbf{x}}$), loss function ($\mathsf{loss}$), and the class of prediction rules ($\mathcal{Q}$) are fixed. The payoff is given by
$$\mathsf{Payoff}(R,f) = \mathsf{Regret}(R, f | P_{\mathbf{x}}, \mathsf{loss}, \mathcal{Q}) = \min_{Q \in \mathcal{Q} \text{ on } \mathbb{R}^r} \mathbb{E}[\mathsf{loss}(f(\mathbf{x}),Q(R(\mathbf{x})) )] - \min_{Q \in \mathcal{Q} \text{ on } \mathbb{R}^d} \mathbb{E}[\mathsf{loss}(f(\mathbf{x}),Q(\mathbf{x}) )]$$

The game in this paper interchanges the order of the players (in a still useful way) and also abstracts the evaluation of the representation in order to make use of the 2PZS/saddle-point framework.

They then consider the minimax and maximin regret in terms of mixed strategies, which are equal due to the minimax theorem. The former is given by $$\min_{\mathcal{D}^{\mathrm{rep}} \in \text{distributions on } \mathcal{R}} \max_{f \in \mathcal{F}} \mathbb{E}_{R \sim \mathcal{D}^{\mathrm{rep}}} \mathsf{Regret}(R,f)$$ 

whereas the latter is given by
$$\max_{\mathcal{D}^{\mathrm{fn}} \in \text{distributions on } \mathcal{F}} \min_{R \in \mathcal{R}} \mathbb{E}_{f \sim \mathcal{D}^{\mathrm{fn}}} \mathsf{Regret}(R,f)$$ 

with the goal of characterizing the optimal minimax (= maximin) regret as well as the optimal $\mathcal{D}^{\mathrm{fn}}$ (parametrized) and as well as the optimal $\mathcal{D}^{\mathrm{rep}}$ (parametrized) that lead to the optimal regret. They are able to do this in the linear MSE case. In the general case, they propose an algorithm to find distributions (mixtures) over finitely many functions and representations, with no theoretical guarantees.

They also consider the minimax regret in pure strategies, given by
$$\min_{R \in \mathcal{R}} \max_{f \in \mathcal{F}} \mathsf{Regret}(R,f)$$ 
with the goal of finding an optimal saddle point representation $(R^\ast, f^\ast)$ in the linear MSE case.

The results in the paper are as follows:

**The linear MSE setting:** Where the data $\mathbf{x}$ is non-degenerate with zero-mean and covariance $\Sigma_{\mathbf{x}}$, the representations are linear ($R(\mathbf{x}) = R^\top \mathbf{x}$ for $R \in \mathbb{R}^{d \times r}$), the response functions are linear (response class is $\mathcal{F}_S$, consisting of functions of the form $f(\mathbf{x}) = f^\top \mathbf{x} + \varepsilon$, where $\varepsilon$ is heteroskedastic, mean-zero, noise, and the coefficient vector $f \in \mathbb{R}^d$ lies in the ellipsoid given by a positive definite matrix $S$), the loss function is mean squared error (MSE), and the prediction functions are also linear ($Q(\mathbf{x}) = q^\top \mathbf{x}$).

* Here they characterize the minimax pure regret, as well as the optimal saddle point pair $(R^\ast, f^\ast)$ giving the minimax regret w.r.t pure strategies (Theorem 2). Interestingly, the optimal representation involves _whitening_ the input vector and then projecting the result on the _top-$r$ eigenvectors_ of the $S$-adjusted version of the data covariance $\Sigma_{\mathbf{x}}$.
* They also characterize the minimax/maximin mixed regret, as well as the optimal mixed strategies $(\mathcal{D}^{\mathrm{fn}}, \mathcal{D}^{\mathrm{rep}})$ leading to this regret (Theorem 3).

The characterization results also generalize to infinite dimensional feature spaces $\mathcal{X}$ (rather than $\mathbb{R}^d$) with some more assumptions (independent noise). This is only done in Appendix F.

**General case**: Here they do implicitly assume some finite dimensional, differentiable, representation for the representations and response functions such that the saddle point problem giving minimax regret can be approximately-solved using an iterative procedure (proposed). The algorithm is motivated by the application of some theoretically-applicable concepts (for saddle point problems) --- iteratively adding to (and hopefully improving) the sets of representations and the response functions alternately,  using projected gradient descent on the regret w.r.t one (keeping the other set fixed), and then adjusting the weights assigned to the various representations and functions in the currently explored set using MWU (as in Freund and Schapire's adaptive game playing framework). However, there are understandably no theoretical guarantees, since the saddle-point problem involved is not convex-concave in general.

### Strengths
* The game-theoretic framework proposed by the paper is very interesting and is novel (in terms of application to representation learning) to the best of my knowledge.
* The results in the linear MSE case are very precise and complete. They give the intuitive expected results when the response class is $\mathcal{F}_S, S=\\{I_d\\}$ = unit-norm linear functions (Example 5) and when the response class is $\mathcal{F} = \\{I_d\\}$ = identity function (Appendix E). 
* The linear MSE results also seem to have some interesting consequences (depending on the structure and correlations of $S^{-1}$ and $\Sigma_{\mathbf{x}}^{-1}$) in general (interpreting the entries as some form of feature importance/correlation weights in the input data and in the prediction task respectively), which are very novel to the best of my knowledge, and could well be exploited in other works.
* The representation learning algorithm for the general case is a substantial contribution which is well-motivated using existing theoretical frameworks, even in the understandable absence of theoretical guarantees.
* The examples in the main paper are very useful in conveying the gist of the ideas.
* The experiments, while limited, are well-thought-out (validating the general algorithm in the linear MSE case, comparison with standard PCA etc).

### Weaknesses
* The experiments are very limited given the scope of the paper ("learning good representations for general prediction tasks", as it may said-to-be). The experiments do not compare to any state-of-the-art practical methods for learning representations at all, especially when applying the representations to different tasks (which I feel is important given the lack of theoretical guarantees for the general case algorithm). However, this paper may be viewed as introducing a _novel framework_ for representation learning/evaluation that links it to saddle-point-game theory (in a way that prior works do not) and hence make it possible for any advances in solving hard saddle-point-games to lead to better approaches for learning representations.
* The meat of the paper is mostly in what is effectively the supplementary material (after the references). The authors have albeit put substantial effort in trying to condense the ideas involved in the proofs as well as illustrate using good examples in the main section. However, it seems to be a losing battle, and any useful perusal of this paper must involve substantial parts of the supplementary.

### Questions
None

### Soundness
3 good

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers the online prediction problem of predicting features for learning functions. First, the paper focuses on the problem of online linear regression with reduced dimensionality. Then, it considers more general settings, including mixed representations or/and logistic regression. Finally some preliminary experimental results are shown.

### Strengths
The problem is well-motivated and suited for the conference. The problem formulations are reasonable. Although the first problem might look somewhat elementary, the theoretical results are solid.

### Weaknesses
So far the current results only focus on simple cases where the classifiers are linear. Ideally, some attempts to cope with nonlinear classifiers or nonlinear feature mapping would be appreciated.

### Questions
Is it possible to extend the first result (linear regression) to kernelized classifiers? If not, can you explain why?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
