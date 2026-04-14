# Neural Superposition Networks

- Decision: Reject
- Scores: 3, 3, 6, 3

## Abstract
Machine learning models can be biased towards the solutions of given differential equations in two principal ways: through regularisation, or through architecture design. Recent research has successfully constrained neural network architectures to satisfy divergence-free fields and Laplace's equation in two dimensions. This work reinterprets these architectures as linear superpositions of general formulated solutions. The notion of superposition is then exploited to develop novel architectures which satisfy both these and novel differential equations. In addition to new architectures for Laplace's equation and divergence-free fields, we propose novel constraints apt for the heat equation, and even some nonlinear differential equations including Burgers' equation. Benchmarks of superposition-based approaches against previously published architectures and physics-informed regularisation approaches are presented. We find that embedding differential equation constraints directly into neural network architectures can lead to improved performance and hope our results motivate further development of neural networks architectures developed to adhere specifically to given differential constraints.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This work reinterprets the existing methods of using regularisation or architecture design as linear superpositions of general formulated solutions and uses this notion of superposition to develop architectures that satisfy various differential equations. Besides new architectures for Laplace’s equation and divergence-free fields, the work proposes constraints suitable for the heat equation and some nonlinear differential equations. Empirical results on solving some toy example PDEs show better performances than existing methods such as PINN, showing that embedding differential equation constraints directly into neural network architectures can improve performance.

### Strengths
1. The paper shows great understanding of the concept of physics-inspired/constrained machine learning, and proposed a novel approach to solve physical problem ground up. 
2. The proposed method is mathematically sound, and the experiments presented are intuitive.

### Weaknesses
1. The presentation of the paper is not clear. It would be much more clearer if the paper is structured in a way that the intention of solving physical problems (e.g. PDEs) is clear at the beginning.
2. The empirical results are weak, and only toy example level.

### Questions
Whereas existing methods like PINN does not scale to high dimensional PDEs, how is the scaling property of the proposed work?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents a unifying framework Machine learning-based solving of partial differential equations, where the differential constraints themselves are embedded into the architecture of the model. By introducing certain linear superpositions of such constrained mappings, this contribution presents new architectures capable of solving a number of PDEs with strong enforcement of the differential constraints

### Strengths
- The contribution introduces a novel architecture based on a superposition of ML-based solutions satisfying certain differential constraints
- A benchmark for a number of linear and nonlinear partial differential equations is establshed

### Weaknesses
- While the superposition NN seems to give better results than PINNs, which impose the constraint weakly through the loss, I fail to see why one would use a superposition NN against an architecture that directly imposes the constraint; in the case of the holomorphic NN for Laplace and the NCL for Navier-Stokes, those simple architectures seem to give better results than the superposition NN.
- For the Heat Equation and Burgers, no test is offered against an architecture solely enforcing the constraint, without the superposition
- It is unclear to me  whether superposition really provides any edge in the case of nonlinear PDEs. The Cole-Hopf transform works indeed for Burgers but this is a particular case, and for Navier-Stokes it appears none of the models manage to achieve good convergence.
- Regarding Navier-Stokes, could the authors identify why none of the methods succeed in achieving convergence ? ML methods (Unets, FNOs) were shown to work in the past for solving steady state incompressible NS, even on more complex domains than the one suggested
- Certain notations are a bit confusing: eq 9, the right hand side does not depend on i
- Certain typos: avoid repeating "Eq. equation", naybe just "Eq" or "equation"

### Questions
- Could the authors test a non-superposition NN which strongly enforces the heat equation constraint and compare against the superposition results for the heat equation (and Burgers ?)
- For Navier-Stokes, it would be relevant to study a case where at least some of the solvers proposed in the benchmark do converge, whether this means adding solvers to the benchmark, or reducing the difficulty of the problem at hand (maybe considering a simple square with periodic boundary conditions)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a framework for constructing neural networks that inherently satisfy linear differential equations through the principle of superposition. The authors show how their framework unifies previous architectures for divergence-free fields and Laplace's equation, while also enabling novel architectures for the heat equation and even some nonlinear equations like Burgers' equation. The work includes theoretical development and empirical validation across multiple differential equations.

### Strengths
1. The theoretical foundation is strong, with clear mathematical development based on superposition principles. The framework elegantly unifies existing architectures while providing rigorous justification for new ones.
2. The practical applicability is impressive, demonstrating success across multiple important differential equations. The method works well for Laplace's equation, heat equation, and even extends to some nonlinear cases like Burgers' equation through clever transformations.
3. The empirical evaluation is comprehensive, comparing against multiple baselines including physics-informed neural networks and specialized architectures. Results are presented with proper statistical analysis across multiple random seeds and clear performance metrics.
4. The method successfully generates novel architectures that outperform existing approaches. For example, their heat equation architecture shows better performance than PINNs while requiring fewer iterations, and their treatment of Burgers' equation is particularly innovative.

### Weaknesses
1. The theoretical guarantees are limited. While the paper proves that solutions satisfy the differential equations, there are no guarantees about approximation capabilities or convergence. The lack of universality proofs significantly limits our understanding of what the networks can and cannot represent.
2. The approach primarily targets linear differential equations, with extensions to nonlinear equations feeling somewhat ad hoc. The Cole-Hopf transformation for Burgers' equation, while clever, doesn't suggest a general strategy for handling nonlinear equations.
3. Implementation details crucial for reproduction are scattered throughout the paper and supplementary materials. The handling of boundary conditions and parameter initialization strategies needs more thorough treatment in the main text.
4. The experimental evaluation, while thorough for the chosen problems, focuses on relatively simple test cases. There's limited exploration of higher-dimensional problems or complex geometries that would be encountered in real applications.
5. The computational complexity analysis is insufficient. While the method shows good performance, there's no detailed analysis of memory requirements, training stability, or computation time compared to alternatives.

### Questions
1. What is the relationship between network size and approximation capability? Can you characterize the class of functions that can be represented as the number of superposition terms increases?
2. How does the method handle mixed boundary conditions in practice? The paper shows some examples, but what are the limitations when dealing with complex boundary geometries or discontinuous boundary conditions?
3. Can you develop a more systematic approach to handling nonlinear equations beyond the specific case of Burgers' equation? What properties must a nonlinear equation have for your approach to be applicable?
4. How does the performance scale with dimensionality? While 2D examples are shown, many practical applications require solving 3D or higher-dimensional problems.
5. The Lie group symmetries play a crucial role in your construction. How can one systematically identify and incorporate appropriate symmetries for new differential equations?
6. The initialization of network parameters seems important for performance. How sensitive is the method to different initialization strategies, and can you provide theoretical guidance for choosing good initializations?
7. For the heat equation results, how well does the method handle different diffusion coefficients? Is there a range of coefficients where the method performs particularly well or poorly?
8. The paper mentions potential extensions to stochastic differential equations. What modifications would be necessary to handle stochastic terms while maintaining the superposition property?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This submission proposes a method for neural networks to satisfy linear differential equations with boundary conditions of the form given in equation (1). The core idea is to linearly combine neural networks that satisfy the differential equation on an open set Omega, and optimize the parameters so that the boundary conditions are met. This method generalizes previous methods developed for particular differential equations. This technique is applied to six types of equations: four linear (Laplace 1, Laplace 2, Heat 1, and Heat 2), and two nonlinear (Navier-Stokes and Burgers'), and compared with other approaches such as PINNs, RAR, AA, and Holomorphic.

### Strengths
- The core idea of the method is simple to understand and well-illustrated in Figure 1.
- The submission does not over-claim the advantages of the method, and limitations are discussed (l. 437, l. 526).
- Experimental results are carefully reported with error margins in Table 1.
- Details are given on the different forms of the superpositions networks considered (2.2.(1-4), 3.(2-5))
- Code to reproduce the experiments is provided as supplementary material.

### Weaknesses
I would like to emphasize that I am not an expert in PINNs and differential equation solving using neural networks. Nevertheless, I find that the contributions of the submission are not strong enough to justify publication at ICLR, primarily due to the following reasons:

* While presented as a methodical contribution, the idea of a superposition network seems quite straightforward.
* Your method outperforms other benchmarks only for the Heat 2 equation. The results for the Burgers equation are not statistically significant, and for all other benchmarks, your method is outperformed by another. While I appreciate that there is no overclaim in the paper regarding the advantages of the method, I would like to understand its true practical advantage.
* There are no theoretical results (theorem/proposition/lemma) in the submission. While this could be acceptable with strong experimental support, the current results are not convincing (see above). 

In addition to these concerns, I find the paper to suffer from presentation issues: 
* The paper lacks figures to illustrate the method beyond a single schematic (Figure 1).
*  Several parts of the submission are in my opinion difficult to understand:
    * More details should be provided upfront about the holomorphic baseline method (l. 161) instead of later in the text (l. 356).
    * The Lie group discussion around l.197 is hard to follow and needs additional explanations.
    * The remark on l. 240 is hard to parse. For instance the second and third sentences sounds like the same.
    * The indices used for θ in equation 10 are not defined.
    * The explanations of the group action are insufficient (around l. 201 and l. 224).
    * Lines 330-331 are unclear and seem out of place.
    * The meaning of "extra weights" in lines 339-341 is not clear.
* Many details, such as the different forms of superposition networks (2.2.(1-4), 3.(2-5)), could be moved to the appendix to allow more space for explaining complex concepts like the group action and Lie group aspects.
* Minor Issues: There are typos like "Eq. Equation" throughout the paper (e.g., l. 87).

### Questions
* Why not apply your method to other differential equations than the ones presented here?
* Is there any computational advantage to your method? Since you only optimize on the boundary, do you observe faster convergence to the solution?
* Could you clarify the meaning of lines 330-331 ?
* What do you mean by "extra weights" in lines 339-341? Is it a scalar multiplicative factor in front of one of the two terms?

### Soundness
2

### Presentation
2

### Contribution
2
