# A Direct Second-Order Method for Solving Two-Player Zero-Sum Games

- Decision: Reject
- Scores: 8, 4, 2, 2

## Abstract
We introduce, to our knowledge, the first direct second-order method for computing Nash equilibria in two-player zero-sum games. 
To do so, we construct a Douglas-Rachford-style splitting formulation, which we then solve with a semi-smooth Newton (SSN) method.
We show that our algorithm enjoys local superlinear convergence. In order to augment the fast local behavior of our SSN method with global efficiency guarantees, we develop a hybrid method that combines our SSN method with the state-of-the-art first-order method for game solving, Predictive Regret Matching$^+$ (PRM$^+$). Our hybrid algorithm leverages the global progress provided by PRM$^+$, while achieving a local superlinear convergence rate once it switches to SSN near a Nash equilibrium. Numerical experiments on matrix games demonstrate order-of-magnitude speedups over PRM$^+$ for high-precision solutions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies iterative methods for solving two-player zero-sum games, where the (approximate) Nash equilibrium is the solution concept. More specifically, the authors design a second order method that achieves quadratic convergence when it is sufficiently close to a Nash equilibrium. Moreover, they develop a lifting framework that allows the use of a first-order method— with Predictive Regret Matching+ chosen in the paper—to warm-start their algorithm. Experimentally, they observe an improved performance in a collection of games, using a few different strategies to switch between first and second order information.

### Strengths
The paper introduces a second order method which appears to be quite novel. Although some of the preliminary results are relatively straightforward to derive, there is a degree of technicality needed to obtain the main theorems. Finally, the experimental results also look promising. Overall, I would say that it is a well written paper with a clearly presented idea that is both mathematically sound and experimentally supported.

### Weaknesses
I think that the only weakness of the work is connected with the switch from first to second order information. That is in Theorem 4, there is no bound for $k$ (unless I missed something?) and the HPSSN method, which I agree with the authors on being the most interesting conceptually, is hard to tune. But the second point is ameliorated to a degree by the performance of PSSNs and could also be a topic of further work.

### Questions
1. As noted in the weakness part, is there any bound on the time needed to enter the local region?

2. Could you describe in greater detail the difficulty of finetuning HPSSN? (lines 447 - 450). Also, did you consider any other schemes?

I will also note here some typos and suggestions. 

i. I did not immediately think of Bilinear Saddle point problem when I read the introduction title. You could consider using the full name, link to the problem statement or simply use "Introduction”.

ii. Given that the empty dot product can be used for a couple of operations in different products you may consider adding a sentence somewhere (we denote by $\circ$...)

iii. Similarly, you use $\hat Z ^\star$ in remark 3 and say “let $\hat Z ^*$ denote…” in the statement of Theorem 3. You could move it inside the remark.

iv. I think the statement of Theorem 4 should be altered a bit since you use both …$k$ sufficiently large… and $k\in N$. It should be $\{z_t\}, {t \ge k}$.

v. In line 744, there is a wrong sign.

vi. In line 769, there is a repetition of an expression.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a second-order method for computing Nash equilibria in two-player zero-sum games. The method is based on constructing an operator derived from the Douglas-Rachford splitting method. The fixed points of this operator in the lifted space correspond to Nash equilibria upon projection, and thus an equilibrium can be computed using a regularized semi-smooth Newton's method applied to the residual of the operator.  This second-order approach is used in conjunction with applying a first-order method (the paper particularly considers Predictive Regret Matching+) as a warm-start. The authors prove quadratic convergence of their second-order method in the local region near a Nash. Experimental results demonstrate the fast empirical convergence of this approach compared to using only a first-order method.

### Strengths
The (hybrid) second-order approach for computing NE in the two-player zero-sum game setting seems novel. The experimental results on random matrix games demonstrate the effectiveness of this method (compared to first-order methods), especially in the high-precision regime.

### Weaknesses
Overall, certain parts of the presentation are somewhat sloppy, which makes interpreting the paper's main theoretical contributions more challenging. In particular, the proposed second-order method relies on using a first-order method as a warm-start, but the main (local) convergence guarantee of the paper (Theorem 4) does not concretely connect the performance of the first-order method with the time to reach the local convergence regime. I believe the presentation would be strengthened if the authors were able to state a concrete end-to-end convergence rate for the Hybrid SSN method (Algorithm 1) using PRM+ for the warm start. It seems this could be achieved using the relationships between the residual norm and duality gap from Theorem 3. This would provide better theoretical insight into the empirical behavior of the algorithm observed in the experiments (where the main speedups occur in the high-precision regime). The authors also mainly suppose the first-order method is Predictive RM+, but it seems the framework could more generally use a different first-order method like Optimistic GDA. 

Writing suggestions:
+ The presentation of Definition 5, Remark 3, and Theorem 3 needs improvement. For example, the definitions of $dist(\cdot,\cdot)$ and $\hat Z^*$ should be given before Definition 5. 
+ The Hybrid SSN method framework (Algorithm 1) and the experimental results of the paper depend on the use of the PRM+ algorithm, however a description of PRM+ (other than one very high-level sentence in the introduction) is missing from the paper. Such a description should be included, at least in the appendix. See also the question below regarding whether PRM+ can be replaced with any other first-order method guaranteeing last-iterate convergence.

### Questions
Regarding the time to enter the local region in Theorem 4:
1. Using PRM+, can you provide a concrete quantitative bound on the number of iterates until the local quadratic convergence kicks in? 
2. If instead of PRM+ you use some other first-order method with a (non-asymptotic) last-iterate convergence guarantee (e.g., Optimistic GDA). Again what can be said about the time until the fast local convergence, parameterized by the last-iterate convergence rate of the first-order method?
3. Also, can you clarify the qualifiers on $\tau$ in Theorem 4 (also in Lemma 4)? This is unclear as written.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper develops a direct second-order method for solving bilinear saddle-point problems. It shows that a semi-smooth Newton (SSN) method achieves local quadratic convergence. It then develops a more practical hybrid approach by leveraging predictive RM+ to warm start the SSN method. Experiments indicate that the proposed method is promising in the high-precision regime.

### Strengths
On the positive side, solving bilinear saddle-point problems is a central problem in game theory and optimization. Designing practical, scalable algorithms for solving such problems is an important and active research topic. The use of second-order Newton-type methods is relatively unexplored in this area. The paper contributes to filling this gap. While second-order methods have a significantly higher per-iteration complexity than first-order methods, such as PRM+, the paper proposes a natural hybrid approach whereby one only switches to the more expensive SSN method once a desired accuracy has been reached. Such hybrid approaches are very much prevalent in state of the art LP solvers, but they have been relatively unexplored in the context of solving zero-sum games. So pushing in that research direction is, I believe, worthwhile. Furthermore, the paper is very well written and organized. The key ideas are nicely exposed and sufficiently explained in the main body.

### Weaknesses
On the negative side, there are very basic flaws in the experimental evaluation of the paper, and the results are quite underwhelming. Concerning the results shown in Tables 1 and 2, the first basic issue is that there is no comparison with LP solvers. A commercial solver such as Gurobi would instantly solve exactly such small games. So on the whole, while the method is supposed to be superior in the high-precision regime, the paper fails to benchmark against the best approach in that regime, which is linear programming. In particular, it's hard to believe that Gurobi would be unable to solve a 400x800 game in less than 5 seconds. Besides failing to benchmark against LP solvers, the paper should certainly expand the first-order benchmarks considered. I should say that it is particularly strange to evaluate against the last-iterate of PRM+ when that's not guaranteed to even converge in zero-sum games. In practice, alternating PRM+ is typically much better than simultaneous PRM+, so at the very least the table should also contain results concerning alternating PRM+, together with other first-order methods that have linear convergence, such as extra-gradient and optimistic GD. The second main flaw in the experimental setup is that the games tested are very small, and it's hard to draw meaningful conclusions from random matrix games. There is a suite of benchmark extensive-form games that are commonly used for such purposes, but the paper doesn't provide those experiments, besides a quick mention about Kuhn poker where PRM+ finds exact solutions very rapidly. I would expect to see experiments on large extensive-form games to be convinced about the practical promise of the method. There are concerns that the proposed method doesn't scale well in larger games because of some of the preprocessing steps required.

I can only recommend acceptance for such paper if the experiments show promise. There are not many new insights in the theory of the paper. In particular, the whole hybrid approach is predicated on the assumption that the region of superlinear convergence will be entered reasonably soon; from a theoretical standpoint, that's not guaranteed. If the paper provides more comprehensive experiments that show the superiority of the method compared to state of the art solvers, I would certainly be in favor of acceptance.

### Questions
- Can the authors explain what the timeout in Tables 1 and 2 is? I wasn't able to find that.
- It is strange to use the term "PCFR" in Tables 1 and 2 since it is a normal-form game. I would suggest switching to PRM+ instead.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work proposes a direct solver for zero-sum matrix games. It is based on a Douglas-Rachford splitting operator of the residual operator into the sum of the residual of the unconstrained operator and an operator that encodes the constraint of both players strategies to lie inside the probability simplex. The resulting update is then computed using a semi-smooth Newton method.

The authors prove that the resulting method converges superlinearly and corroborate this claim with numerical examples on a series of numerical experiments.

### Strengths
The authors achieve their goal of designing a provably superlinear converging solver by what appears to be expert use of advanced techniques of non-smooth and convex analysis.

### Weaknesses
The first thing that comes to mind when reading the authors' claim of designing "the first direct second-order method for computing Nash equilibria in two-player zero-sum games" is that two-player matrix games are well-known to reduce to linear programming, for which a wide range of methods are already in existence. 

The authors claim that "While this approach works in principle, it is impractical for large-scale games; even with state-of-the-art commercial solvers, the LP reformulation inflates the problem size and destroys exploitable structures in the payoff and constraint matrices, making exact solutions computationally prohibitive."

I have a hard time following this argument. After all, the reduction to linear programs yields problems of the form 

$$
\max_{x, v} \quad  v \quad 
\text{s.t.} \quad  A^\top x \ge v  \quad
\mathbf{1}_m^\top x = 1, 
x \ge 0.
$$

It seems hard to agree that increasing the number of decision variables by 1 amounts to meaningfully "inflating the problem size." When it comes to the exploitation of problem-specific structure, no detail is provided as to how the proposed method would be able to exploit such structure in a way conventional LP solvers can't.

### Questions
As described under "weaknesses," I suspect the comparison to LP-based approaches to be misleading. If the authors can convince me otherwise, I would be more than happy to reconsider my recommendation.

### Soundness
2

### Presentation
3

### Contribution
2
