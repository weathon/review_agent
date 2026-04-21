# Robustness of Quantum Algorithms for Nonconvex Optimization

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
In this paper, we systematically study quantum algorithms for finding an $\epsilon$-approximate second-order stationary point ($\epsilon$-SOSP) of a $d$-dimensional nonconvex function, a fundamental problem in nonconvex optimization, with noisy zeroth- or first-order oracles as inputs. We first prove that, up to noise of $O(\epsilon^{10}/d^5)$, perturbed accelerated gradient descent equipped with quantum gradient estimation takes $O(\log d/\epsilon^{1.75})$ quantum queries to find an $\epsilon$-SOSP. We then prove that standard perturbed gradient descent is robust to the noise of $O(\epsilon^6/d^4)$ and $O(\epsilon/d^{0.5+\zeta})$ for any $\zeta>0$ on the zeroth- and first-order oracles, respectively, which provides a quantum algorithm with poly-logarithmic query complexity. Furthermore, we propose a stochastic gradient descent algorithm using quantum mean estimation on the Gaussian smoothing of noisy oracles, which is robust to $O(\epsilon^{1.5}/d)$ and $O(\epsilon/\sqrt{d})$ noise on the zeroth- and first-order oracles, respectively. The quantum algorithm takes $O(d^{2.5}/\epsilon^{3.5})$ and $O(d^2/\epsilon^3)$ queries to the two oracles, giving a polynomial speedup over the classical counterparts. As a complement, we characterize the domains where quantum algorithms can find an $\epsilon$-SOSP with poly-logarithmic, polynomial, or exponential number of queries in $d$, or the problem is information-theoretically unsolvable even with an infinite number of queries. In addition, we prove an $\Omega(\epsilon^{-12/7})$ lower bound on $\epsilon$ for any randomized classical and quantum algorithm to find an $\epsilon$-SOSP using either noisy zeroth- or first-order oracles.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to apply quantum algorithms for finding the $\epsilon$-approximate second-order stationary point ($\epsilon$-SOSP) of a $d$-dimensional non-convex function, given noisy zeroth- (i.e., noisy evaluation query) or first-order(i.e., noisy gradient query) oracles as inputs. The study focuses on settings where only noisy zeroth-order (evaluation queries) or noisy first-order (gradient queries) oracles are accessible. Specifically, the authors analyze scenarios under various levels of noise, parameterized by poly $(\epsilon, 1/d)$,  and demonstrate that their approach significantly outperforms state-of-the-art (SoTA) methods regarding query complexity. Furthermore, they establish several lower bounds for alternative scenarios where efficient query complexity is not achievable with the current methods. Tables 1, 2 and 3 concisely summarize the main results, offering a comparative view of query complexity improvements relative to existing methods.

### Strengths
- Applying quantum oracles to speedup existing results is a timely and meaningful interesting problem.
- This work gives a dense collection of results.
- The summarization of the results in Tables 1, 2 and 3 is informative and clear.

### Weaknesses
- The presentation of technical results in this paper feels somewhat vague; for instance, there is no pseudocode or intuitive explanation of the underlying algorithms. This lack of detail impacts the paper's clarity and readability.

- The current presentation makes it challenging to assess the paper's novelty. For instance, when referencing Jordan's algorithm, the paper points to [2] for details on Jordan's gradient estimation (line 223) and states that it "replaces the gradient queries in PAGD with this oracle" (line 232). However, there is insufficient discussion on the complexity of replacing the oracle, which obscures an evaluation of the paper's originality.

- There are no related discussions on when the noise levels are practical, for example, what problem setting would generate a noise level of $\tilde{\Omega} (\epsilon^{1.5} / d)$? 

- The definition of $\epsilon$-SOSP doesn't align with the one in [1]. Instead of letting $\|F(x)\| \leq \epsilon$, the saddle point should satisfy $\|\nabla F(x)\| \leq \epsilon$?


- Several notations are used for the same purpose in this work, for example, [line 119-130], $\||$ , | |, and | are used to denote similar norms. The reviewer would recommend using one notation to denote all the norms.

Minor Comments:
- Some $O(\cdot)$ sign probably should be $\Theta(\cdot)$ in the column "noise strength" in Table 1?

[1] Jin, C., Ge, R., Netrapalli, P., Kakade, S. M., & Jordan, M. I. (2017, July). How to escape saddle points efficiently. In International conference on machine learning (pp. 1724-1732). PMLR.

[2]Chakrabarti, S., Childs, A. M., Li, T., & Wu, X. (2020). Quantum algorithms and lower bounds for convex optimization. Quantum, 4, 221.

### Questions
- It appears that $\nabla f(x+z) - \nabla f(x)$ is intended as an estimate of the Hessian rather than the gradient, as indicated in line 312?

-The most recent work that this paper compares against in Table 1 is from 2018. Are there been any more recent results in this direction?

### Soundness
2

### Presentation
1

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
The work studies the task of finding $\epsilon$-approximate second-order stationary points of $d$-dimensional non-convex functions with zeroth- or first-order noisy quantum oracles. Under various different noise regimes (as functions of $\epsilon$ and $d$), the work presents upper and lower bounds for the classical and the quantum query complexities of the task. 
With zeroth order oracle, it has been shown that the quantum algorithm proposed gives an exponential speedup (against classic lower bound) when the noise is at a small scale of $\tilde O( \epsilon^6 / d^4 )$, and a polynomial speedup (compared to the state-of-the-art classical algorithm) when the noise is of scale $[ \tilde \Omega(\epsilon^6 / d^4), O(\epsilon^{1.5} / d) ] $.
With first-order oracle, there is a polynomial quantum speedup (compared to the state-of-the-art classical algorithm) when the noise is of scale $\Theta(\epsilon / d^{0.5})$.

### Strengths
The paper provides a comprehensive set of results that cover a wide range of noise scale regimes, and oracle assumptions. The comparison with the classical results are meaningful and well presented.

### Weaknesses
The quantum query complexity bounds established are far from being tight.
Most of the algorithms are almost identical to the classical version with the gradient estimation replaced by Jordan’s algorithm. The analysis also looks like a rehash of the classical proofs by substituting the guarantees from Jordan’s algorithm. The lower bound constructions seem to be just borrowed from Jin et al., 2018a and Carmon et al., 2021 (up to a unitary rotation and a scaling factor).

### Questions
1. Can you give a more descriptive title to Section 4.3 like the surrounding sections?
2. Table 3 is for lower bounds in $\epsilon$. But the zeroth-order lower bound for deterministic classical algorithm entry is still an expression in $d$.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper investigates the robustness of quantum algorithms for non-convex optimization, in the context of finding an $\epsilon$-approximate second-order stationary point ($\epsilon$-SOSP) in a d-dimensional non-convex function. As is usually done in optimization literature, the algorithm has access to the function via noisy zeroth or first-order oracles that satisfy certain standard assumptions. 

The paper provides several foundational results for this setup in both upper and lower bounds. The upper bounds, under zeroth and first-order oracles, respectively, are nicely summarized in Table 1 and Table 2 under different noise regimes. Notably, under the zeroth order assumption and low noise regimes, the algorithms achieve exponential speed-ups. Additionally, they establish lower bounds for classical and quantum algorithms, summarized in Tables 1, 2, and 3, under various noise regimes and Zeroth and First order Assumptions.

### Strengths
The paper provides a comprehensive analysis of upper and lower bounds on query complexities of quantum algorithms for an important problem in non-convex optimization, and thus, the contributions are important to the community. The paper is well-written.

### Weaknesses
No major weaknesses. See questions.

### Questions
1. In the regime of Table 2 and Theorem 2.5 under the first-order assumption, are there known classical lower bounds?
2. What are the key algorithmic insights for results that achieve exponential speed-ups? 
3. In the final version, I will encourage the author to expand on open questions more systematically. In general, there are several open gaps to establish minimax optimal query complexities for the problem.

### Soundness
4

### Presentation
4

### Contribution
3
