# Quantum mechanical framework for quantization-based optimization: from Gradient flow to Schrödinger equation

- Avg Score: 2.00
- Decision: Reject
- Scores: 4, 2, 2, 0

## Abstract
This work presents a quantum mechanical framework for analyzing quantization-based optimization algorithms. 
The sampling process of the quantization-based search is modeled as a gradient-flow dissipative system, leading to a Hamilton–Jacobi–Bellman (HJB) representation. 
Through a suitable transformation of the objective function, this formulation yields the Schrödinger equation, which reveals that quantum tunneling enables escape from local minima and guarantees access to the global optimum. 
By establishing the connection to the Fokker–Planck equation, the framework provides a thermodynamic interpretation of global convergence. 
Such an analysis between the thermodynamic and quantum dynamic methodologies unifies combinatorial and continuous optimization, and extends naturally to machine learning tasks, such as image classification. 
Numerical experiments demonstrate that quantization-based optimization consistently outperforms conventional algorithms across both combinatorial problems and nonconvex continuous functions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a theoretical framework for analyzing optimization algorithms based on objective function "quantization." The authors present an ambitious mathematical narrative that aims to model this quantization-based search process as a gradient-flow dissipative system. This model is then linked to the Hamilton-Jacobi-Bellman (HJB) equation and, through a series of transformations, is claimed to yield the Schrödinger equation. The central thesis is that this connection reveals quantum tunneling as the fundamental mechanism for escaping local minima. The framework also draws parallels to the Fokker-Planck equation, suggesting a unified view of thermodynamic and quantum dynamics in optimization. The authors derive a stochastic gradient update rule from this theory and present experiments on both combinatorial (TSP) and continuous (machine learning) problems.

### Strengths
1. Extensive Theoretical Work: The authors have clearly invested a significant amount of effort in constructing an elaborate mathematical framework (Section 3). The attempt to connect concepts from optimal control (HJB), quantum mechanics (Schrödinger), and statistical physics (Fokker-Planck) is mathematically ambitious and, on the surface, intellectually stimulating.

2. Good Performance on Combinatorial Tasks: The gradient-free QTZ algorithm (Algorithm 1) demonstrates strong and robust performance on the TSP instances presented (Table 1), outperforming both SA and QIA, which is a non-trivial experimental result.

### Weaknesses
1. Leap of faith in assumption 4: The main problem of this paper's theoretical claim lies in assumption 4, where the authors posits the existence of a "virtual wave function" $\overline{\psi}$ and then defines a new density $\rho \triangleq \psi g$ (Eq 13) by arbitrarily multiplying this new "quantum" probability with the classical Gibbs distribution. This assumption is introduced ad hoc and is not derived from any preceding principles of optimization or quantization. The entire "quantum mechanical" part of the paper (Theorem 3.3) and the "thermodynamic" part (Theorem 3.2) are direct consequences of this single, unsubstantiated assumption. The paper fails to provide a rigorous justification for why signal quantization (a numerical rounding operation) should induce such a complex-valued wave function.
2. Lack of Novelty in the Final Algorithm: After this extensive theoretical journey, the final SDE (Eq 23) and its discrete update rule (Eq 24) are formally identical to Annealed Langevin Dynamics (SGLD). The paper's framework serves as an extremely complex re-derivation of this well-known algorithm, where the "quantization step size" $Q_p^{-1}(t)$ is simply a re-labeling of the temperature $T(t)$ in a cooling schedule. The connection between Langevin dynamics, Fokker-Planck, and escaping potential barriers (which can be interpreted as tunneling via path integrals) is standard, decades-old knowledge in statistical physics and is not a novel contribution of this work.
3. Unconvincing ML Experimental Support: The ML experiments (Sec 4.2) test the SGLD-like algorithm (Eq 25). The results in Table 3 are not compelling; the proposed QSLD/QSLGD methods are often outperformed by standard Adam or SGD (e.g., on CIFAR-100), which contradicts the paper's claims of superior optimization.

### Questions
1. How can the gradient-based theory (Section 3) be used to analyze the gradient-free Algorithm 1? If it cannot, why are these two parts presented together as a unified framework?
2. Can the authors provide a rigorous, first-principles derivation (not just an assertion) for how quantization (Definition 1) induces the "virtual wave function" $\overline{\psi}$?
3. For a given objective function $f(x)$, how would one find or write down the specific $\overline{\psi}$ (and thus $\rho$ via Eq 13) that this assumption claims must exist? Without a constructive proof, this assumption remains arbitrary.
4. Given that the final update rule (Eq 24) is formally identical to SGLD, what is the practical, algorithmic-level contribution of this framework beyond re-labeling temperature $T(t)$ as a quantization step $Q_p^{-1}(t)$?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This submission presents a framework for analyzing optimization algorithms from a quantum mechanical point of view. This framework starts with an optimization problem for the function $f$, and then proposes a general quantization depending on the step size. The authors showed that the level set search process can be modeled as thermodynamic evolution, which can further be modeled as a Schrödinger equation. The ground state of the Schrödinger dynamics can be found using adiabatic evolution, and because of the quantum mechanical tunneling effect, the quantum dynamics leads to global minima.

### Strengths
This submission examines the optimization problem from a physics perspective and provides numerical results to support its observations.

### Weaknesses
I found the manuscript very hard to read. The whole text is written in a physicsy language without a comprehensive computer science intuition. I am worrying that this submission might not find a suitable audience at ICLR. 

In particular, I was trying to grasp the main algorithmic results of this work, but I don't think Theorems 3.1, 3.2, or 3.3 can serve in this role. They just say how to map one type of dynamics to another, but they didn't say anything about why such maps work. For example, does the ground state of equation (17) solve the optimization problem? If so, in what quality? In many places, the authors make vague claims without quantitatively justifying them. For example, in Lines 334-336, "when the energy gap is sufficiently small, the quantum tunneling effect enables the system's state to transition to a lower energy..." This vague statement lacks precise characterization, such as in what probability, what the transition rate is, and how the rate depends on the energy gap. In addition, the general computer science audience might not know what the energy gap is.

This work might contain an important contribution; however, based on the current presentation, it is very hard to justify its value. For example, there's no convergence analysis, so it's hard to see whether this framework has any advantage compared with classical analysis. I think the ultimate goal, as pointed out by the authors, is to propose quantum algorithms based on this framework. Maybe this work will find its value once faster quantum algorithms are developed based on it.

Besides, there is existing work on using quantum mechanical frameworks to speed up optimization problems, for example, arXiv:2303.01471, arXiv:2503.15878, and arXiv:2505.14670. This submission cited the last one, but didn't capture their contribution. 

Minor comment: Line 364, redundant word "equation"

### Questions
Please address my concerns in the Weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors study the stochastic optimzation problem using the quantum-based framework.

### Strengths
The experimental results look good. If you are willing to further strengthen the experimental section, I would be open to reconsidering my score.

### Weaknesses
However, theoretical part seems to have already been established in the literature, with more comprehensive and in-depth analyses available.  Have the authors read the following papers？

* Bin Shi, Weijie Su and Michael I. Jordan; On Learning Rates and Schrödinger Operators, Journal of Machine Learning Research, 24(379):1-53, 2023
* B. Shi; On the Hyperparameters in Stochastic Gradient Descent with Momentum, Journal of Machine Learning Research, 25(236):1-40, 2024.

### Questions
Could you please explain Algorithm 1: Blind Random Search (BRS) with quantization-based optimization in detail?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper develops a theoretical framework for quantization-based optimization that links its sampling stage to gradient-flow dynamics, expressed through a Hamilton–Jacobi–Bellman (HJB) representation. The analysis leverages tools from Schrödinger operator theory and exploits a correspondence between Fokker–Planck dynamics and Schrödinger equations, aiming to provide a unified methodology across combinatorial and continuous optimization. The authors complement the theory with numerical studies on traveling-salesman problems, low-dimensional nonconvex tasks, and standard machine learning datasets (e.g., FashionMNIST, CIFAR), where the approach exhibits promising results.

### Strengths
- This work tries to propose a theoretical framework that unifies the analysis of quantization-based optimization for combinatorial and continuous problems.
- An algorithm named "Blind Random Search" (BRS) with quantization-based optimization is proposed. Despite its simple implementation, the authors propose a mechanism (under a set of assumptions) that this algorithm may work in practice. 
- Numerical experiments are provided to illustrate the practical performance of quantization-based optimization. The results appear to be promising, as the proposed method often outperforms existing quantum and classical optimization methods across a wide spectrum of application domains.

### Weaknesses
**The exposition lacks cohesion**: although the manuscript is dense with terminology and equations, central notions (quantization-based optimization, Schrödinger operators, adiabatic algorithms, HJB) are named but neither defined nor carried through, and sections do not build on one another **semantically** or **technically**.

1. The central concept is not adequately defined. The manuscript attempted to unify combinatorial and continuous optimization under a “quantization-based” umbrella, but it remains unclear what “quantization-based” precisely means in each setting and whether the same mathematical machinery applies across them. Please provide a formal definition (including assumptions), a minimal working example in each domain, and an explicit mapping that shows which results/tools transfer (and which do not) between the discrete and continuous cases.

2. The “Related Works” section omits several recent threads on nonconvex optimization grounded in quantum mechanics and quantum-inspired methods. For example: 
- Leng, Jiaqi, Ethan Hickman, Joseph Li, and Xiaodi Wu. "Quantum Hamiltonian Descent." arXiv preprint arXiv:2303.01471 (2023).
- Chen, Zherui, Yuchen Lu, Hao Wang, Yizhou Liu, and Tongyang Li. "Quantum Langevin dynamics for optimization." Communications in Mathematical Physics 406, no. 3 (2025): 52.
- Goto, Hayato, Kosuke Tatsumura, and Alexander R. Dixon. "Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems." Science advances 5, no. 4 (2019): eaav2372.

3. This paper would benefit from clearer notation and stronger narrative flow. Some symbols are non-standard or insufficiently defined, and transitions between paragraphs (even within a section) are difficult to follow. New concepts are repeatedly introduced without tying them back to earlier definitions or results; see some examples in the "Questions" section.

### Questions
- What does $f \in \mathbb{R}$ or $f \in \mathbb{Q}$ mean? $f$ represents a function, and it is ambiguous to claim that it belongs to the set of real/rational numbers. 
- Is the "quantization of f" (Definition 1) a new function? How should I interpret $f \in \mathbb{Q}$? What if the function takes irrational values?
- There should be a period at the end of Definition 4.
- Please justify the rationale behind Assumption 3: why "blind random search" (Algorithm 1) can be interpreted as a gradient flow? How should we interpret "gradient" in this process?
- Near Line 205: why "the set has a non-zero measure" implies "the spectrum of $f(x_{t+1})$ ....", and how does this "coincide with the eigenvalues of the 2-level Hamiltonian"? What is the spectrum of a function (standard definition of "spectrum" means the set of eigenvalues of a matrix, if finite-dimensional)? And which 2-level Hamiltonian?
- How is the "exponential kernel $\Phi$" related to any algorithms/procedures discussed in Section 2? Why the function $V$ (Eq. (8)) is defined over $[0,1]$, while $f(x)$ is apparently defined over $\mathbb{R}^d$?
- Thermodynamical evolutions (such as those in the Langevin dynamics) are fundamentally different from adiabatic quantum evolution. I do not follow the discussion on page 7. 
- In the numerical experiment (Figure 2), how is the "iterations" defined for quantum annealing? Isn't quantum annealing a continuous-time quantum evolution by default? And is it a fair comparison using "iterations"? Quantum clock rate (i.e., the frequency of implementing elementary quantum computation on quantum computers) can be much higher than classical clock rate, leading to very different wall-clock time scales.

### Soundness
1

### Presentation
1

### Contribution
1
