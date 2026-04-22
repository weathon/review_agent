# Efficient Submodular Maximization for Sums of Concave over Modular Functions

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Submodular maximization has broad applications in machine learning, network design, and data mining. However, classical algorithms often suffer from prohibitively high computational costs, which severely limit their scalability in practice. In this work, we focus on maximizing Sums of Concave over Modular functions (SCMs), an important subclass of submodular functions, under three fundamental constraints: cardinality, knapsack, and partition matroids. 
	Our method integrates three components: continuous relaxation, Accelerated Approximate Projected Gradient Ascent (AAPGA), and randomized rounding, to efficiently compute near-optimal solutions. We establish a $(1 - \varepsilon - \eta - e^{-\Omega(\eta^2)})$ approximation guarantee for both cardinality and partition matroid constraints, with query complexity $O\left(n^{1/2}\varepsilon^{-1/2} (T_1 + T_2)\right)$. For the knapsack constraint, the approximation ratio degrades by a factor of $1/2$, with query complexity $O\left(n T_1 + n^{1/2}\varepsilon^{-1/2} T_2\right)$, where $T_1$ denotes the computational cost of evaluating the concave extension, and $T_2$ denotes the computational cost of backpropagation. By leveraging efficient convex optimization techniques, our approach substantially accelerates convergence toward high-quality solutions. 
	In empirical evaluations, we demonstrate that AAPGA consistently outperforms standard PGA. On small-scale experiments, AAPGA achieves superior results in significantly less time, being up to $32.3\times$ faster than traditional methods. On large-scale experiments, our parallel multi-GPU implementation further enhances performance, demonstrating the scalability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors study the problem of submodular maximization where the objective function can be expressed as the sum of concave composed with modular functions (SCMs), subject to various types of constraints. The proposed algorithms utilize the continuous relaxation, Accelerated Approximate Projected Gradient Ascent (AAPGA), and randomized rounding.  The method achieves an approximation ratio of $1-\epsilon-\eta-e^{-\Omega(\eta^2)}$ for both cardinality and partition matroid constraints and $1/2(1-\epsilon-\eta-e^{-\Omega(\eta^2)})$ for general matroid constraints.

### Strengths
1. The paper addresses an interesting problem of submodular maximization where the objective function is formulated as a composition of concave and linear functions. The motivation is clearly presented, and the inclusion of representative examples such as the coverage and facility location functions helps illustrate the relevance of the studied setting.
2. The authors tackle an important challenge in continuous optimization methods for submodular maximization. While such methods often yield stronger approximation guarantees compared to discrete approaches, their high query complexity and computational overhead have limited their practical use. Hence, the effort to design continuous algorithms that are both query-efficient and computationally efficient is timely and significant.
2. The integration of deep neural networks into the study of submodular optimization is an innovative and inspiring direction. This approach has the potential to be generalized to a broader class of submodular functions and could open new avenues for combining learning-based techniques with classical optimization theory.

### Weaknesses
The key concern is the poor clarity and organization of the paper. Several important details are missing from the main text, and some parameters are introduced without proper definition or explanation (see the Questions section for specifics). The core ideas underlying the proposed algorithms are not clearly presented, making it difficult to follow the methodological contributions. Furthermore, the paper does not clearly articulate how its technical contributions differ from or improve upon existing work.

### Questions
1. The parameter $\eta$ in Algorithm 1 is introduced without explanation, which makes it difficult to interpret the theoretical guarantees.  
   - Is $\eta$ assumed to lie within the range \([0, 1]\)? If so, it seems that the approximation guarantee is valid only for certain values of \(\eta\), since when $\eta=0$ or $\eta=1$, the bound becomes trivial.  
   - In addition, the scaling factor $\beta$ does not appear in the theoretical guarantee—are $\eta$ and $\beta$ intended to represent the same parameter?  
   Clarification on this point is needed.  
2. The definition of the set $P'_L(\mathbf{x})$ (Line 206) takes a point $\mathbf{x}$ as input; however, in the actual expression, only $y^{(t)}$ appears. 
    It is unclear what $\mathbf{x}$ refers to in this context. Please clarify the definition and role of $\mathbf{x}$.
3. Can you clarify in detail why the proposed Accelerated Approximate Projected Gradient Ascent method is better than the existing PGA method?
4. The computational cost of \textbf{Algorithm 1} is not clearly analyzed. In particular, Line 5 in the pseudocode requires determining $i_t$, which involves evaluating Equation (4) for each possible $i_t$. How is the number of $i_t$ evaluations bounded? 
5. The paper mentions the use of \textbf{deep neural networks} to accelerate the computation of the supergradient. However, if the objective function is simply a composition of a concave and a linear function, it is unclear why a neural network is necessary for this task.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the maximization problem of the sum of concave composed with modular functions (SCMs), an important subclass of submodular functions, under cardinality, knapsack, and partition matroid constraints respectively. Paper proposes an optimization framework that integrates three key components: continuous relaxation, accelerated approximate projected gradient ascent, and randomized rounding. The experimental evaluation assesses the effectiveness and acceleration advantages of our method from three perspectives: convergence speed, small-scale submodular maximization, and large-scale submodular maximization.

### Strengths
The paper introduces a novel optimization framework that employs the concave extension during the optimization phase to enhance computational efficiency, while leveraging the multilinear extension in the rounding phase to ensure theoretical approximation guarantees. The relationship between these two extensions is bridged by a key lemma, thereby balancing computational performance with theoretical assurance. Furthermore, to address the non-differentiability of certain activation functions in SCMs, the paper presents a supergradient calculation formula based on the right-hand derivative and integrates it with the backpropagation mechanism of neural networks, effectively resolving gradient computation in non-differentiable cases.

### Weaknesses
1. It is unclear whether the results for the randomized algorithms in Section 4.2 represent averages from multiple runs or the outcome of a single run.

2. In the experimental section, several key parameters that affect the performance of the AAPGA, such as the approximation projection error $\delta$ and the step size $L_0$, are not provided.

3. Some related works on submodular maximization under knapsack constraints may require further discussion to contextualize the current paper, such as:

   * *Fast adaptive non-monotone submodular maximization subject to a knapsack constraint*
   * *Submodular maximization subject to a knapsack constraint: Combinatorial algorithms with near-optimal adaptive complexity*
   * *Streaming algorithms for constrained submodular maximization*
   * *Linear-Time Algorithms for Representative Subset Selection From Data Streams*

### Questions
The paper discusses how to handle the non-differentiability of the activation function in SCM, but Theorem 1 assumes that $F$ is continuously differentiable. There seems to be a discrepancy between these two statements. Could you please clarify the rationale behind this assumption?

### Soundness
2

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
4

### Summary
This paper proposes an efficient algorithm, Accelerated Approximate Projected Gradient Ascent (AAPGA), for maximizing Sums of Concave composed with Modular functions (SCMs). The method applies a Nesterov-accelerated projected gradient ascent optimization directly on the function's concave extension. Theoretically, the algorithm achieves a $(1-\epsilon-\eta-e^{-\Omega(\eta^{2})})$ approximation guarantee with a "square-root level" query complexity of $O(n^{1/2}\epsilon^{-1/2}(T_{1}+T_{2}))$, where $T_1$ is the cost of backpropagation and $T_2$ is the cost of function evaluation. Empirical evaluations confirm that AAPGA converges faster and achieves superior results compared to the standard Projected Gradient Ascent (PGA) method.

### Strengths
1.	The paper considers a subclass of submodular functions and provides the algorithm with sublinear query complexity.
2.	The paper considers several important constraints in the submodular maximization area. 
3.	Experimental results are provided to show the effectiveness of their algorithms.

### Weaknesses
1.	The meaning of the parameter $ \eta $ is barely explained in the abstract or the main text, which can be confusing to readers.
2.	The meanings of the parameter $T_1$ and $T_2$ are not consistent in the paper. It is confusing. 
3.	The knapsack results seem weak: the rounding procedure sacrifices a $1/2$ approximation ratio by enumerating the largest item in an outer loop.

### Questions
1.	Can the costs of $T_1$ and $T_2$ be made explicit as closed-form expressions? If we consider the traditional value oracle model, what is $T_1$ and $T_2$？
2.	The meanings of the parameter $T_1$ and $T_2$ are not consistent in the paper. For example, let us consider the time of evaluation of the concave extension. In the abstract, it is $T_2$, but in the corollary 1, it is $T_1$. The parameters in table 1 and in the description of table 1 (in page 2) are also not consistent. 
3.	What is the parameter $ \eta $? It seems this parameter appears in Lemma 1 which shows that $\eta$ is related to the SCM function itself. Thus if we cannot choose arbitrary small $\eta$, the approximation ratio of the proposed algorithm may be very bad.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work studies submodular function optimization where the submodular function is a concave function over the sum of modular functions (SCM).  The authors establish a continuous algorithm for cardinality constraint,  whose approximation ratio is (1-\epsilon) and query complexity is $\sqrt(n) (T_1+T_2)$, where $T_1$ is the time needed for evaluating the concave function and $T_2$ is time needed for back propagation.  Authors present algorithms for knapsack case with an approximation ratio 1/2-\epsilon. 

I would like to disclose that I have only superficial familiarity with continuous optimization techniques.

### Strengths
This work substantially improves the query complexity compared to previous works, especially for the cardinality constraint.  For example, for cardinality constraints, the best known query complexity is  $O(n/epsilon)$. Similarly, for knapsack, the best known quesry comlexity isO(nklog^2 n)$. 

The main technical contribution is the development of the Accelerated Approximate Projected Gradient Ascent algorithm, that converges faster than the known algorithms.

### Weaknesses
While the query complexity is smaller, when measured in-terms of $n$, it is not clear what is effect of T_1 and T_2. If they are large, then it is not clear if the proposed algorithm offers any advantage.

Typically, continuous optimization algorithms run slowly compared to the discrete versions. Given that it is not clear how much computational advantage the proposed algorithm offers compared to the LS+PGB algorithm that uses $O(n/epsilon)$ queries

### Questions
Please see the weakness
What is the effect of T_1 and T_2? How large could they be in the worst-case?
It is not clear to me that the rounding techniques proposed are any different from the known pipage rounding. Can you explain the differences?
Can you compare the performance with the algorithms listed in lines 59, 60 and 61?

### Soundness
3

### Presentation
3

### Contribution
4
