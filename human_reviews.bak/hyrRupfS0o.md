# Revisiting Quantum Algorithms for Linear Regressions: Quadratic Speedups without Data-Dependent Parameters

- Decision: Reject
- Scores: 3, 8, 3

## Abstract
Linear regression is one of the most fundamental linear algebra problems. Given a dense matrix $A \in \mathbb{R}^{n \times d}$ and a vector $b$, the goal is to find $x'$ such that $\|| Ax' - b \||\_2^2 \leq (1+\epsilon) \min\_{x} \|| A x - b \||\_2^2$. The best classical algorithm takes $O(nd) + \mathrm{poly}(d/\epsilon)$ time [Clarkson and Woodruff STOC 2013, Nelson and Nguyen FOCS 2013]. On the other hand, quantum linear regression algorithms can achieve exponential quantum speedups, as shown in [Wang \emph{Phys. Rev. A 96}, 012335, Kerenidis and Prakash ITCS 2017, Chakraborty, Gily{\'e}n and Jeffery ICALP 2019]. However, the running times of these algorithms depend on some quantum linear algebra-related parameters, such as $\kappa(A)$, the condition number of $A$. In this work, we develop a quantum algorithm that runs in $\widetilde{O}(\epsilon^{-1}\sqrt{n}d^{1.5}) + \mathrm{poly}(d/\epsilon)$ time and outputs a classical solution. It provides a quadratic quantum speedup in $n$ over the classical lower bound without any dependence on data-dependent parameters. In addition, we also show our result can be generalized to multiple regression and ridge linear regression.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Linear regression is a fundamental problem in the field of optimization and linear algebra. In recent years, many quantum algorithms for linear regression problems have been proposed. However, the time complexity in some of the algorithms has a dependence on the condition number of the input matrix of size $n\times d$. In this work, the authors addressed this issue by proposing a new quantum algorithm with time complexity $O(\sqrt{n} d^{1.5}/\varepsilon + \text{poly}(d/\varepsilon))$ for obtaining an $\varepsilon$-approximate solution, which has no dependence on the condition number. The core technique to achieve this speedup is by using the quantum leverage score sampling procedure proposed in [AG23].

### Strengths
The authors applied the quantum leverage score sampling algorithm proposed in [AG23] to the linear regression problem, and obtained quantum algorithms for linear, multiple and ridge regression problems. The time complexity of their quantum algorithms has a square root dependence on the number of rows for the input matrix, a quadratic speedup in this parameter compared with their classical counterparts, at the cost of increasing extra dependence on the number of columns and the inverse of the precision parameters.

### Weaknesses
The work appears to lack novelty, as the primary techniques employed to develop the algorithm involve applying quantum leverage score sampling, as proposed in [AG23], and the classical leverage score sampling method for linear regression problems, as proposed in [CW13]. It appears that the paper directly replaces the leverage score sampling procedure from [CW13] with its quantum counterpart to achieve results. Furthermore, the concept of using a quantized leverage score sampling procedure for solving linear regression problems has previously been introduced in [Sha23]. 
Also, the result of the work seems a bit weak compared with the classical counterparts. For input an $n\times d$ matrix, a classical algorithm in [CW13] can run in time $O(nd \log (1/\varepsilon)) + \tilde{O} (d^3)$, which could have logarithmic dependence in the error parameter $\varepsilon$, but the quantum algorithm has a polynomial dependence on $\varepsilon$. For comparison, the quantum algorithm for linear programming proposed in [AG23] also achieves a speedup over their classical counterparts while keeping logarithmic dependence in the precision parameter $\varepsilon$.

### Questions
Same as the weaknesses part.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a quantum algorithm for linear regression, with the running time of O(\sqrt{n}d^{1.5}+d^w) where n is the number of data points, d is the number of variables and w is the matrix multiplication exponent. It also gives quantum algorithms for multiple regression and ridge regression based on the same ideas.

Classically, running time must be at least linear in n. Quantumly, algorithms with running time polylogarithmic in n are known but their running time depends on the condition number of the regression matrix and other extra parameters. The quantum algorithm in the current paper is the first one with a running time that is sublinear in n, polynomial in d and 1/\epsilon where \epsilon is the desired accuracy and no dependence on extra parameters. 

On the technical level, the results are based on combining the recent quantum algorithm by Apers and Gribling for computing leverage scores with classical algorithms for linear regression.

### Strengths
Natural research question: linear regression is an important computational task.
Clear improvement over previous work in the chosen parameter (n).
Writing generally good.

### Weaknesses
The paper is a combination of existing results from quantum algorithms (Apers-Gribling) and classical algorithms for linear regression but, given that the end result is interesting, I do not view this as a significant weakness.

### Questions
- Theorems 1.4 and 1.5 define that r is the row-sparsity of the matrix but the running time of the algorithms in them has no dependence on r. Consider removing r from them.
- "Classical linear algebra" section in the related work mentions a lot of references. It would be sufficient to give a smaller number of applications for classical linear algebra, as its significance is well understood.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper considers the problem of solving linear regression, multiple regression, and ridge linear regression using quantum algorithms. Previous quantum algorithms that leverage quantum linear algebra techniques have shown exponential speedups in the system size $n$, but they often depend on certain quantum linear algebra-related parameters, such as the condition number of the regression matrix. An open question is whether quantum speedups can be achieved without relying on data-dependent parameters. This paper answers this question in the affirmative by developing quantum algorithms for linear regression, multiple regression, and ridge regression with quadratic quantum speedups. Technically, the algorithms first generate a subspace embedding of the original regression matrix through sampling from its leverage score distribution, achievable in $O(\sqrt{n})$ time if given a quantum row query oracle. The algorithms then solve the regression problems within this reduced subspace, where the solutions also have small regrets in the original problems.

### Strengths
This paper develops algorithms that achieve quantum speedups without relying on data-dependent parameters, addressing a significant open problem in quantum linear systems research. The literature review is thorough and well-organized.

### Weaknesses
In my view, the techniques used in this paper are relatively straightforward. Specifically, the main result, Theorem 4.4 in Section 4, is proved by combining Lemma A.1, Lemma 4.1, Lemma 4.2, and Lemma 4.3. Among these, Lemmas 4.1 and 4.2 are from existing literature, Lemma 4.3 involves algebraic calculations on the runtime of matrix multiplication.

Minor comments:
1. There appears to be a grammatical issue in Lemma A.1.
2. Lemma 4.2 is missing the phrase "time" or "time complexity."

### Questions
I don't have further questions other than the existing ones above.

### Soundness
4

### Presentation
3

### Contribution
2
