# MatRL: Provably Generalizable Iterative Algorithm Discovery via Monte-Carlo Tree Search

- Decision: Reject
- Scores: 2, 8, 4, 6, 4

## Abstract
Iterative methods for computing matrix functions have been extensively studied and their convergence speed can be significantly improved with the right tuning of parameters and by mixing different iteration types. Hand-tuning the design options for optimal performance can be cumbersome, especially in modern computing environments: numerous different classical iterations and their variants exist, each with non-trivial per-step cost and tuning parameters. To this end, we propose MatRL – a reinforcement learning based framework that automatically discovers iterative algorithms for computing matrix functions. The key idea is to treat algorithm design as a sequential decision-making process. Monte-Carlo tree search is then used to plan a hybrid sequence of matrix iterations and step sizes, tailored to a specific input matrix distribution and computing environment. We also show that the learned algorithms provably generalize to sufficiently large matrices drawn from the same distribution. Finally, we corroborate our theoretical results with numerical experiments demonstrating that MatRL produces algorithms that outperform various baselines in the literature.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method for finding matrix algorithms using MC tree search. The MC tree search methodology is relatively standard, what is new is how to apply this framework to finding matrix algorithms, where some design choices were made, an experimental setup is given, and a limiting high probability result for matrix functionals is proven. The paper is clear in exposing its objective, the setup, and it is easy to read. For now though, it's not clear if this approach can be used out of sample.

### Strengths
The paper is easy enough to read, and relatively clear. There is some novelty in trying to discover matrix algorithms using MC tree search. I can also image an application. For instance, there is currently a concerted effort to find a matrix sign algorithm that, in 5 iterations (each consisting of one quintic polynomial), in bfloat16 precision, can approximate the matrix sign of the gradient (or momentum buffer).  This is needed to speed up the Muon method, and consequently beat the nanoGPT speed run competition:

https://github.com/KellerJordan/modded-nanogpt/tree/master

But no such clear cut application is given in the paper.

### Weaknesses
1. I see a significant issue with train/test distinction and out of distribution performance. Most of the experiments such as Figure 1 and Table 2 are evaluated on the exact same distribution you used to discover your algorithm. In that sense, it is not at all surprising it outperforms the other methods, but this gives little to no indication if the method will perform well out of distribution. If the users objective is to have a method working on one distribution, then you should include the time it takes to find this algorithm for this given distribution. Otherwise, the user could simply take a standard method, and just run it for more iterations, and still save time on having to find this algorithm. Your one example in Figure 2 of fitting an algorithm on $\mathcal{N}(0,1)$ on a Wishhart distribution with the same exact limiting spectrum is still a very limited out of distribution behavior.  Ultimately, for your approach to be useful, you need to show more of the behavior of the algorithms out of distribution.


2. Suppose again that I am user who needs a particular matrix function for a particular application. Your design space and number of hyper parameters is very large, with hyper-parameters
$C,\alpha, \epsilon_{tol}, E, T, D,$ and the choice of actions. How, as a user, can I be confident that it will be easy enough to choose these hyper parameters? Or was there some hyper-parameter tuning for each problem instance? Right now, I would not be confident in using Algorithm 1, and would be more comfortable working in the smaller design space of just one application.   


3.  There is a lot of work right now on computing $sign(A)$ because of the Muon method. I know your method is restricted to square symmetric, where as what Muon needs is a method for general rectangular. But you could have still compared your method on square symmetric to several of the competing new methods being developed, such as newtonschulz5/zeropower_via_newtonschulz5 used in Muon.

4. Limited action class, consider polynomials instead. The action class is very limited, with 2 possible updates for Inverse, and 4 for matrix sign. Furthermore, I suppose for the inverses required in the Newton or Halley steps, I suppose (this is not clear in the paper) that you use either a Newton-Schulz or Chebyshev step? Thus ultimately, you are always applying an odd monomial for computing the matrix sign. Why not short-cut this process, and simply restrict your actions to the class of odd monomials? You already consider the Quintic. You could expand upto the ninth degree, and just use that one type of update. 

5.   Limited to square symmetric matrices. You are open about this limitation, so I do not really hold it against you. But for now, you rely on the matrix being diagonalizable, and having congruence invariant operators. This excludes applications such as matrix sign applied to the non-square matrices (such as in the muon method you mention). I think you could generalize your setup to rectangular matrices, where you defined congruence using two orthonormal matrices, e.g. $f(UXV^\top) = Uf(X)V^\top$, and you use the singular values as state variables instead of the eigenvalues.

### Questions
1. How exactly do you compute the matrix inverse needed in your discovered Algorithm 3 and 4? 

2. Do you set the floating point precision in Algorithm 1?

3. For the CIFAR-10 experiment, you report "Log Relative Error". Are you using the same training data or some test/validation data?

** Minor questions***

1. This loss $\mathcal{L}$ in Eq (4) is not defined. At the least, you should say that is some loss function defined by your problem.

2/ On lines 197-233 you used $a\_j$ for the parameters of these step function $f_j$, but on line 245 you introduce this $(k\_1, \ldots, k\_{n\_{j}})$ as the parameters of $f_j$?  Also I assume $k:= (k\_1, \ldots, k\_{n\_{j}})$. Maybe what you meant to say was, that these are all the parameters chosen up to step $n\_{j}$?  I found this a bit confusing.

3. Section 4, you should clarify in the text that by "limiting distribution" you mean "the distribution as the dimension $m$ goes to infinity". 

4. Proposition 1: What exactly do you mean by " let .. $f_k^*$ the step $k$ transformation of eigenvalues of the algorithm found by Algorithm 1"? Do you mean, the resulting matrix function after running k steps of your algorithm? This is confusing, since you proposition places no condition on $k$ whatsoever (what about $k=0$?). This proposition is general statement about approximating matrix functions in high probability and high dimensions.  It is not really about your Algorithm 1, but instead a statement about how $P_m$ and $Q_m$ are asymptotically equivalent (or asymptotically indistinguishable) with respect to the functional $L_k.$

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces MatRL, a reinforcement-learning-based framework that automatically discovers iterative algorithms for computing matrix functions. The approach frames algorithm design as a sequential decision-making process, using Monte Carlo Tree Search to select and tune hybrid iterations. A key theoretical contribution is that the learned algorithms are shown to generalize to large matrices drawn from the same distributions, and with the same limiting eigenvalue spectrum. The authors demonstrate the method’s effectiveness on tasks such as matrix inversion, square roots, and sign functions in their experiments. There are several open directions and follow-up works, including extending these to rectangular and sparse matrices.

### Strengths
1. It is a novel framework and a new connection with reinforcement learning
2. The results generalize to large matrices and there are provable guarantees.
3. The experiments suggest that the method works better than non hybrid ones.
4. A key idea they exploit is that different methods work better on different parameter regimes such as condition number etc. The framework allows to use the ideal method in the required parameter regime without explicitly determining which regime you are in. I think this can help with the design of optimization algorithms in general.

### Weaknesses
1. Does not extend to rectangular matrices.
2. This seems like a new concept, so maybe some more basic explanations/background might help.

### Questions
1. Are there any works on this in the past?
2. Is there a high-level reason why these don't work for rectangular matrices?
3.    How large are the search and training times?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a reinforcement learning framework for learning matrix algorithms. This task is cast as a reinforcement learning problem where states represent intermediate steps of the computation, actions represent (parametrized) transformations, and the reward captures computational cost. The approach is demonstrated on several algorithmic tasks such as matrix sign and matrix square root, and appears to provide computational benefits compared to exist direct (eigenvalue decomposition) and iterative methods.

While I see some merit in this direction of research, I have a hard time understanding what the paper does and how experiments are conducted. Therefore, I believe the paper needs substantial improvements in the presentation clarity before being published.

### Strengths
* The paper considers the automated discovery of matrix algorithms, which has the potential to improve computational workloads by finding more efficient algorithms
* Numerical experiments suggest that computational improvements can be achieved compared to existing baselines / state-of-the-art algorithms

### Weaknesses
* The paper is, overall, hard to follow. It is easy to get lost in the various notations, and some terminologies are used without a clear definition
* The proof of several key theoretical results is deferred to the appendix, which makes it hard to validate the findings
* The experiments appear to be executed on a single matrix (see questions below), and the paper does not appear to discuss performance variability
* Proposition 1 appears to be an existential result based what seem to be relatively strong assumptions (identical spectrum distribution), which states (as far as I understand), that

### Questions
* In the proposed RL scheme, how is $\mathcal{L}$ captured? I did not see it mentioned in the reward, nor does it appear in Algorithm 1
* The presented RL setup considers a value function $V(s, a)$ that depends on both state and action. However, for the obtained algorithm to be applicable in practice, the policy should not depend on the state, only of the iteration number (this is because the state information comprises the matrix' spectrum, hence, one needs an eigenvalue decomposition to know the full state). Can the authors please explain how this is tackled in their approach?
* In proposition 4.1, I do not understand the relation between $m$ and $k$, can the authors please provide more context and explanations?
* Is the MCTS training phase executed on a _single_ matrix?
* Similarly, I have a hard time understanding whether the results reported in Section 5 are evaluated on a single matrix, or are averages over multiple matrices. If it's the former, results on a single sample may not be indicative of general performance. If the latter, some metric of performance variability would be useful
* In the SQRT experiments, I would expect that a cholesky factorization would be faster to compute that eigenvalue decomposition

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a reinforcement learning algorithm based on Monte Carlo Tree Search to discover faster algorithms for computing functions of large matrices. The authors demonstrate the approach on several example functions, including the matrix sign and square root. The proposed RL method appears effective at identifying efficient algorithms.

### Strengths
-	The RL algorithm managed to identify algorithms with better performance on the examples presented in the paper.

### Weaknesses
The training is applicable only when the state is reduced to the spectrum of the matrix and to functions within the Congruence Invariant Diagonal Preserving framework. This likely limits the general applicability of the resulting algorithms.

The results presented in Section 4 on generalization appear rather weak, as they essentially concern only matrices with asymptotically similar spectra. The title therefore seems somewhat optimistic. A deeper investigation of the algorithms’ generalization properties would have been valuable.

There is no apparent novelty in the reinforcement learning algorithm itself.

### Questions
Line 208: the loss function ${\cal L}$ is not specified below … it is done much later on examples.
Typo in eq (4): $n_i$ should be $n_{t_i}$.
The notation used 242-245 is not appropriate, before $k$ referred to the iteration number, here it refers to the vector $a_{j$ -- may be a better notation would be $a_j^{(k)}$ the vector of parameters used in the $k$-th iteration when using the function $f_j$. 
I find the discussion Line 250-259 unclear, while the message seems trivial – because the authors restrict themselves to a Congruence Invariant Diagonal Preserving setting. 
The same holds for discussion starting Line 260. The set of possible actions is indded huge. What does at stage $1,ldots,n_j$, only parameters $k1,\ldots,k_{n_j−1}$ are chosen: what do you mean? Don’t you select $k_{n_j}$ too?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to compute certain functions of matrices efficiently. The idea is to use reinforcement learning to tune the parameters within certain pre-existing iterative methods for computing these matrices, and to decide when to jump from one iterative method to another.

The paper seems to produce nice empirical results, showing that the tuned methods are indeed very fast ways of computing certain matrix functions, taking advantage of the actual runtime observed on your specific hardware to tune these runtimes. It's neat.

The method should apply to any spectral function $f(A)$ so long as it is efficient to compute some form of residual between the current iterative and $f(A)$. They use the square root and matrix sign functions in this paper.

The paper is fundamentally empirical, and most of the effort in the paper is put into characterizing how the space of iterative methods can be fit into a reinforcement learning framework. Theory is given, but doesn't exactly give any qualitative rates beyond the existence of an epsilon-delta guarantee.

### Strengths
The paper is interesting.

I'm pretty borderline on it because I'm not super clear how much "new science" there is in it; though if the paper is considered to be sufficiently novel then I don't really have any other concerns about its publication. I'm a bit of an outsider to RL as a community, so it's hard for me to judge how much this paper is really cool RL research (as opposed to like good engineering on top of standard RL tools).

The motivation is compelling. We want to compute spectral functions efficiently. Lots of existing methods exist. It's really annoying to figure out which iterative method you should be using right now. Machine learning seems like a good toolkit for answering this question.

The experiments are interesting and suggest that a large speedup is often possible. Would be nice to see this on a larger variety of functions that what's currently considered -- namely for a fancier function like log(A) or e^A or something like that.

The example algorithms generated, algos 2 3 and 4 on page 8 are pretty great payoffs.

Overall, I'm pretty cool with this paper. The intro is well written and the idea is simple but seems likely helpful.

### Weaknesses
The paper is confusing in places. The experiments are still weaker than I'd want them to be, in terms of validating a proposed mathematical algorithm on a wide variety of matrices.

I don't really understand all the twists and turns in the precise engineering of the titular MatRL algorithm [page 6]. The pseudocode is a bit too abstract, and the importance of all the elements feels a bit vague to me. I wouldn't be surprised this feels vague to me because I'm not as familiar with the RL side of thing though. _I don't really understand where the matrix A is even used in this algorithm_

The paper needs a bit reflowing as well. It's left very vague for a long time what kinds of degrees of freedom we have when designing iterative methods. For instance, Newton iteration shows that the procedure $X \leftarrow \frac12(X + X^{-1} A)$ converges to $X = A^{1/2}$. It is unclear though what parameters we might optimize here; can we replace $\frac12$ with something else? The nature of this question isn't really made clear enough early enough in the paper; a running example would be helpful here.

There's this paragraph on page 5, "Spectrum as state variables", which seems to suggest storing the eigenvalues of the matrices used in the iterative method as part of the state of the RL algorithm to reduce memory cost. This really confuses me because computing the eigendecomposition is slower than using an iterative method to compute f(A). Further, the RL algorithm needs to actually run primitives like matrix-matrix products on real matrices in memory to see 1) the actual runtime of these matrix-matrix products and 2) the numerical stability of the method examined. So I'm pretty thoroughly confused on this point.

Real-world evaluation of the proposed method is a bit lacking. There are huge datasets of matrices easily findable (eg suitesparse) where we can easily find many viable real-world input matrices. It'd be great to see a much larger number of matrices undergo the treatment of this proposed RL algorithm and the histogram of "ratio of MatRL runtime vs torch.linalg.eigh runtime". This would be needed for practitioners to seriously consider using the proposed method for finetuning their runtimes.

The precision achieved by all methods, including dense eigenvalue computation, is strangely low, capping out at 4 digits of precision. Perhaps this is an artifact of using 32 bit floats and using an error metric that squares our matrices (thereby square the condition numbers involved)?

In their experiments, they use a pretty darn large regularization of $10^{-3} I$ to ensure numerical stability. I'm not clear why this is so large, as this generally limits the accuracy of methods to not exceed about 3 digits of accuracy. I'm used to something much closer to machine epsilon, though maybe this is close to the square root of the machine epsilon? Either way, seems large to me.

The experiments have no confidence intervals, showing that they do or don't act consistently. Since their method involves some randomness in exploring their RL framework (to my understanding at least...), it seem like there should be some confidence intervals.

### Questions
Is definition 1 really meaningfully different than the definition of a spectral function?

## Typos & Recommended edits

Feel free to ignore anything in this list without further discussion

1. [throughout] use \citep where needed, so that citations are much easier to read around
2. [53] add units to the 2.9 and 1.7 numbers
3. [71] say "method" instead of "solution"
4. [72] "a desiderata of" to "desiderate from"
5. [76] "which iteration to use" isn't clear language here. Maybe "iterative method" is better?
6. [81] remove "algorithm"
7. [101] I disagree with this sentence. Other methods exists for accessing matrix functions. Krylov method. Eigendecompositions. Shrink the scope, perhaps.
8. [192] "where D is a distribution over random matrices defined on"
9. [216] Add "diagonal preserving"
10. [218] remove one of the two math expressions; I think the former one.
11. [222] comma after "is"
12. [222] "to find" instead of "finding"
13. [259] typo on spectrum
14. [Decoupled Actions on page 5] I don't get what's going on here. It's not clear to me how the continuous search space is searched, and it's not clear why decoupled actions resolves issues around continuous search spaces
15. [Decoupled Actions on page 5] I don't really follow the algorithmic idea here. The paragraph is just confusingly phrased to me.
16. [278] add "the" after "depending on"
17. [295] UCB should be UCT. Or the rest of the paper needs to say UCB.
18. [321] I don't understand why you need the probabilities around the convergence statements. Can't you just say that mu(X) and mu(Y) both weakly converge to mu*?
19. [326] Given that the set [a, b] is closed, do we actually needs the epsilons? Can't we actually just replace this with either [a, b] or (a, b)?
20. [351] Fix the quotes before "Hessian"
21. [Figs 1 and 2] what does the shaded region mean? How many trials was this run over?
22. [throughout] How did it take for MatRL to run to find methods like Algos 2, 3, and 4

### Soundness
3

### Presentation
2

### Contribution
2
