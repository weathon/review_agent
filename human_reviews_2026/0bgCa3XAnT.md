# Smoothing Binary Optimization: A Primal-Dual Perspective

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Binary optimization is a powerful tool for modeling combinatorial problems, yet scalable and theoretically sound solution methods remain elusive. Conventional solvers often rely on heuristic strategies with weak guarantees or struggle with large-scale instances. In this work, we introduce a novel primal-dual framework that reformulates unconstrained binary optimization as a continuous minimax problem, satisfying a strong max-min property. This reformulation effectively smooths the discrete problem, enabling the application of efficient gradient-based methods. We propose a simultaneous gradient descent-ascent algorithm that is highly parallelizable on GPUs and provably converges to a binary solution in sublinear time. Extensive experiments on large-scale problems—including Max-Cut, MaxSAT, and Maximum Independent Set with up to 50,000 variables—demonstrate that our method identifies high-quality solutions within seconds, significantly outperforming state-of-the-art alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a heuristic method to approximately solve a wide class of combinatorial optimization problems, which seek to minimize a real-valued function $F(x)$ over a binary vector $x\in\\{0,1\\}^n$. If such a function is replaced with its multilinear extension $f(x)$, the relaxation in which we minimize $f(x)$ over $x\in[0,1]^n$ is exact, i.e., equal to the original problem. Multilinear extensions are compact for many common combinatorial problems. This relaxation is of course non-convex.

To this relaxation, the authors add constraints to enforce integral solutions. These have the form $g(x_i)=0$ where $g$ is a convex function on $[0,1]$ that vanishes at and only at 0 and 1, such as $g(x_i)=x_i^2-x_i$. Using Lagrangian $L(x,y)=f(x)+\\sum_i y_i g(x_i)$, these constraints are then dualized, where the duality gap is shown to be zero.

Now gradient descent-ascent method is used to find a saddle point of the Lagrangian. This algorithm behaves much like a continuation method: it starts with a (large) positive dual variables $y$, in which case the function $x\\mapsto L(x,y)$ is convex, so global approximate solutions are preferred. During GDA, the components of $y$ are slowly decreasing, which is making the function non-convex and becoming more and more greedy. The algorithm is augmented with a heuristic that pushes $x_i$ away from $1/2$, to avoid possible trivial saddle points with $x_i=1/2$. The algorithm is very simple.

The convergence analysis shows that $y_i$ are bounded below, $x$ converge to integral solution, and convergence rate is linear.

In experiments, the authors compare the method to several SOTA heuristics for approximately solving large-scale combinatorial optimization problems. The problems used are max-cut, max-k-cut, max independent set, and max-sat. The instances are taken from public datasets, which are mostly random graphs. The reported results show that the method is always competitive and often the best in objective value, and one of the fastest.

### Strengths
The approach is very simple and elegant, resulting in a very simple (a few lines) but versatile algorithm. The theoretical analysis is elegant and also relatively simple. Each experiment is done carefully (but see below).

### Weaknesses
I have two major comments.

FIRST comment: The experimental comparison is too limited (the choice of problems, of instances, and also of competing methods). While one can always say this, in this case it is justified. Let me argue why. On line 102, the authors write that the method provably converges to a near-optimal solution. But this is not true: the presented theory says absolutely nothing about the quality of found suboptima. Thus, the method is only a heuristic (however elegant). Thus, justifying its practical utility fully relies on extensive experimental comparison.

The choice of tested problems is limited. For example, I am missing the MAP inference problem in MRF (aka discrete energy minimization), which has been subject to very intensive research in computer vision and machine learning, see e.g. experimental studies [1], [2] (with public datasets). Another possible problem is QAP, which is known to be very challenging; public QAP datasets are e.g. [4], [5].

The choice of instances (for tested problems) is limited to mostly random instances. But it is known that random instances tend to be much easier to optimize that real-world ones.

The choice of competing methods is also limited, biased to recent works. Of course, for MAP inference and QAP one should use SOTA dedicated methods, as in [1, 2, 3]. But even for the tested problems, off-the-shelf global ILP solvers (like Gurobi) might be worth trying. I am not entirely sure that they could handle the (large) tested instances, but it was observed (eg, in computer vision) that they can be surprisingly fast on many large ILP instances from practice. Perhaps, this would not be very competitive time-wise, but it would give  global optima to compare to.

SECOND comment: I am not sure about the level of novelty. Many heuristics have been shown to perform surprisingly well on certain classes of large-scale combinatorial problem instances. The presented method is similar to several such methods. One example is [3], which solves the multilinear formulation of the MAP inference problem approximately by ADMM (though on a different problem reformulation), with great results. This is probably subsumed by many later works on applying ADMM directly to nonconvex/integer problems.

The presented method can be easily reformulated by doing (block-)coordinate descent on primal variables, instead of gradient descent. By augmenting the Lagrangian, one gets ADMM.

Moreover, the method is very similar to continuation methods. The continuation method could minimize $f(x)+\\mu \\sum_i g_i(x_i)$, starting with a large positive $\\mu$ (which would make the function convex) and then slowly decreasing $\\mu$. In fact, this is almost what is going on in GDA, up to the fact that instead of a single variable $\\mu$ we have a vector $y$, whose components are decreased automatically as a result of gradient ascent. I tend to believe that such a continuation method would yielded similar experimental results. This would make the paper significantly less novel, from the theory side.

Note, instead of $n$ constraints $g(x_i)=0$ one could use a single constraint $\sum_i g(x_i)=0$, which would enforce iintegrality of $x$ in the same way. For $g(x_i)=x_i^2-x_i$, this would read $(x-1)^T x=0$. In this case, there would be only one $y$-variable, like in continuation method. When using GDA, would this give similar experimental results?

It would be useful to discuss the relation to other similar methods in more detail in the paper (or supplement). And try to experimentally identify the features of the method that make it perform well (a kind of ablation study).

Minor remarks:

- In Def 2, the term 'multilinear' collides with its standard meaning in linear algebra.

- In section 2, it is not clear if Lemma 1 and Prop 1 are novel or known before. Please make this clear.

- In Algorithm 1, the projection on a non-convex set (the union of intervals) does not exist, strictly formally. If $x=1/2$, there are two nearest points.

- A more detailed discussion around the example on line 222 might be useful. In fact, it means that GDA does not find a stationary point in this case. If GDA always found a stationary point of $L(x,y)$, the $\\delta$-heuristic would not be needed. In this respect, a nonconvex version of ADMM with convergence guarantees might be a better choice than GDA, cf. [6].

[1] Szeliski et al: A Comparative Study of Energy Minimization Methods for Markov Random Fields, ECCV 2006

[2] Kappes et al: A Comparative Study of Modern Inference Techniques for Structured Discrete Energy Minimization Problems. IJCV 2015

[3] Le-Huu, Paragios: Continuous Relaxation of MAP Inference: A Nonconvex Perspective. CVPR 2018

[4] Haller et al: A comparative study of graph matching algorithms in computer vision. ECCV 2022

[5] Burkard et al: QAPLIB -- a quadratic assignment problem library. J. Global Optim. 10(4), 391–403 (1997)

[6] Wang et al: Global Convergence of ADMM in Nonconvex Nonsmooth Optimization. 2018

### Questions
1) Is there an obstacle for testing on other combinatorial problems in experiments, such as MAP inference or QAP, and real-world instances, such as in [2]? Are there arguments to believe that the method would be competitive also on these problems?

2) Is there an argument why the method is qualitatively different from well-known continuation methods (as I described above), augmented with the $\\delta$-heuristic? Is there an argument why such a continuation method would perform much worse?

3) Please discuss novelty of your approach in more detail, comparing to similar methods and emphasizing the features that are essential for success.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
**Summary**

This paper proposes a first-order method-based primal heuristic for the binary optimization problem. The main idea is to adopt a continuous extension of the optimization objective as well as a minimax formulation of the binary constraint. This reformulation is then solved with first-order methods for minimax optimization. Some numerical experiments demonstrate the efficiency of the proposed approach.

### Strengths
**Strength**

The paper has a clear motivation and is easy to follow. The idea of using a first-order method as a primal heuristic seems interesting.

### Weaknesses
**Weaknesses**

Although the idea seems interesting, I have some concerns regarding the algorithm design and the experiment setup.

1. Heuristic nature of the approach

   Although the paper establishes some theoretical results regarding the problem reformulation and the convergence of the gradient descent-ascent (GDA) method, there is actually nothing to say about the quality of the generated solution. In particular, the convergence analysis algorithm (PDBO) largely deviates from traditional GDA, and it only guarantees convergence to a binary solution. In other words, the proposed mechanism is essentially a primal heuristic that can be implemented by a first-order method. Therefore, the convergence analysis in the paper can be misleading. 

2. Weak experiment evaluation

   In the experiments, the paper compares the proposed approach to several existing approaches from the literature. However, I find the following points unsatisfactory:

   - From the experiment, the proposed approach needs to be used jointly with other existing methods (e.g., ABS2) to give empirical speedup. 
   - The method claims to be able to exploit GPU architecture. However, there is no time comparison between the CPU/GPU versions of the PDBO algorithm.
   - The selected optimization problems can also be solved with mathematical optimization solvers. It would be nice to see a comparison with other optimization solvers' performance. 

Besides, the idea of using first-order methods to design MIP heuristics is not entirely new [1]. Solvers for mixed integer programming often have a solution polishing step that tries to sparsify a fractional solution using a sparsity-inducing loss function. 

Overall, I find the proposed approach interesting, and I'm willing to raise my score if the authors can

- clarify the heuristic nature of the approach
- add experiments that demonstrate the scalability of PDBO on GPU
- add experiments that compare PDBO with an optimization solver (commercial or open-source).

### Questions
**Questions**

1. The formulation in equation (1) is quite general and seems compatible with the indicator function of constraints. Can the approach deal with linear constraints on $x$?
2. The algorithm seems to promote a binary solution only when the dual multipliers $y$ are negative. However, as shown in line 231, the initialization requires $y^0$ to be strictly positive. To me, this initialization seems to be trying to encourage more exploration of the feasible region. Could you elaborate more on this?
3. Currently, PDBO needs a projection that pushes the solution away from the stationary value $0.5$. What if you instead try escaping the stationary points by adding random noise and rounding the solution at the end?
4. Can your approach provide a dual certificate on the quality of the solution?

**Minor issues**

1. Line 75

   Problem (1) and problem (1) are sometimes inconsistent.

2. Line 110

   It may be better to add an introductory paragraph to the beginning of **Section 2**

3. Line 275

   What does it mean by linear time?

**References**

[1] Cacciola, M., Forel, A., Frangioni, A., & Lodi, A. (2025, June). The Differentiable Feasibility Pump. In *International Conference on Integer Programming and Combinatorial Optimization* (pp. 157-171). Cham: Springer Nature Switzerland.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper proposed a new primal-dual method for the unconstrained binary optimization problem via some convex smoothing for the binary constraints. Additionally, this method is also compatible with GPU computing.  Then, the authors show the performance of the proposed method both theoretically and numerically.

### Strengths
1. The idea is very clear and intuitive.

2. The proposed method is also GPU compatible.

3. The performance of the method is illustrated via both theory and experiments.

### Weaknesses
1. It seems the literature review part does not thoroughly discuss papers about solving binary optimization via continuous relaxation. Also see Q1 for more details.

### Questions
1. I am not familiar with the binary optimization literature. Could the authors provide more literature review? In particular, I wonder whether this idea —i.e., transforming binary constraints into continuous constraints via g(x)=0 — appears in the literature of binary optimization, not limited to GNN-based methods. This idea seems very direct, intuitive, and useful.

2. Why a good $g$ should be symmetric? I feel the asymmetry could also express some prior beliefs of the value of $x_i$, especially if we have some warm start solution in hand.

3. In general, how to select lambda for constrained problems, and alpha, beta as well?

some minor things:
a. Is the binary constraint in (3) a typo?

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
2

### Summary
In this paper, the authors propose a primal-dual approach to solve quadratic unconstrained binary optimization (QUBO) problems. The idea is to consider a continuous relaxation of the original unconstrained version and introduce additional constraints to enforce integrality. Then a gradient descent-ascent algorithm is implemented to solve the primal-dual reformulation of the constrained version. Empirical evaluations on 4 classical QUBO problems show the proposed algorithm (PDBO) outperforming existing baselines.

### Strengths
1. The paper is well-written and easy to read. The primal-dual reformulation makes intuitive sense.
2. On large graph instances, PDBO finds better feasible solutions within a short time compared to baseline methods.

### Weaknesses
1. The empirical evaluations for Max-Cut and Max-k-Cut only presents 5 large instances in the main paper. The remaining instances present a less clear picture, especially for instances under 10000 nodes compared to the ABS2 baseline.

2. As the method is based on continuous relaxation, the comparison does not have any existing relaxation-based baselines. Another important missing baseline is exact solvers such as Gurobi. In terms of learning-based baselines, some recent works are missing, e.g., [1]. Adding those baselines would strengthen the submission.

[1]: Chen, Ming, et al. "Learning to solve quadratic unconstrained binary optimization in a classification way." Advances in Neural Information Processing Systems 37 (2024): 114478-114509.

### Questions
1. What are the reasonings behind the three conditions for the function $g$ on lines 173-175?
2. In the algorithm 1, line 3 has one condition for the partial derivative. Why is this condition necessary?
3. In Table 1, PDBO+ABS2 hybrid strategy exhibits vastly different runtime. What makes G67 and G72 take so much longer than the other three instances?

### Soundness
3

### Presentation
3

### Contribution
2
