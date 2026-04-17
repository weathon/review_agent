# Certified Neural Approximations of Nonlinear Dynamics

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Neural networks hold great potential to act as approximate models of nonlinear dynamical systems, with the resulting neural approximations enabling verification and control of such systems. However, in safety-critical contexts, the use of neural approximations requires formal bounds on their closeness to the underlying system. To address this fundamental challenge, we propose a novel, adaptive, and parallelizable verification method based on certified first-order models. Our approach provides formal error bounds on the neural approximations of dynamical systems, allowing them to be safely employed as surrogates by interpreting the error bound as bounded disturbances acting on the approximated dynamics. We demonstrate the effectiveness and scalability of our method on a range of established benchmarks from the literature, showing that it significantly outperforms the state-of-the-art. Furthermore, we show that our framework can successfully address additional scenarios previously intractable for existing methods— neural network compression and an autoencoder-based deep learning architecture for learning Koopman operators for the purpose of trajectory prediction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper provides a method for creating neural abstractions of nonlinear functions with certified error bounds. The method involves two key features. The first is a method for abstracting the nonlinear function being approximated using linear bounds to feed into a neural network verification tool. The second is an adaptive refinement strategy to efficiently check the property over the input space. The authors compare their method to previous work on neural abstractions and show that they are able to verify properties more quickly. They also scale their method to two new problems related to Koopman operators and neural network compression.

### Strengths
- I very much appreciate the writing quality, the simplicity of the notation, and the figures used throughout the paper. The paper was easy to follow.
- Neural abstractions are an interesting area of study that can help verify dynamical systems and, as the authors show, verify compressed representations.
- The two new problems introduced in the paper related to Koopman operators and neural network compression could be used as benchmarks for future techniques developed for neural abstractions.

### Weaknesses
- The proposed methodology seems to be a very specific instance of a more general idea. The general idea is to create a linear abstraction of a nonlinear function so that it can be used to verify the neural abstraction.  In the paper, the authors discuss creating this abstraction with a first-order approximation. However, there are multiple ways to obtain linear bounds on a nonlinear function (e.g. CROWN, which was used in the neural network compression example), and it is not justified why a first-order approximation is necessary.
- The method is only tested using the Marabou neural network verification tools, and the authors do not make clear whether other verifiers could also be used (and what effect this might have on performance).
- For the neural network compression example, the authors must use CROWN rather than a first-order approximation since the original network is not continuous. This modification to the methodology is only mentioned briefly at the end of the paper, and it is not clear how it can be incorporated into the partitioning strategy introduced in section 3.2.

### Questions
- Can the method be extended to other neural network verification tools? How will this affect performance?
- Could this method be subbed into the method presented by Abate et al., which checks properties using the final abstraction? If so, it would be useful to include this fact in the paper.
- How does the partitioning strategy work with CROWN?
- Line 465: what does “non-analytical dynamics” mean?
- Should proposition 1 be using the Jacobian or gradient?

Suggestions:
- Line 118: not -> nor
- “the procedure is relaxed to eventually split on all axes that enter Nj(x) nonlinearly, as this can impact the execution time of Marabou.” I am not sure what this means.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper certifies NN surrogates of nonlinear dynamics via first-order Taylor models with bounded remainders, adaptive partitioning, and a linear NN verifier (Marabou). It reports strong wall-clock gains over a dReal/SMT baseline and two extra demos (Koopman trajectories, compression).

### Strengths
- The technique is sound in theory.

### Weaknesses
The paper frames “avoiding SMT over nonlinear reals” as its contrastive novelty and compares to dReal (SMT) as “state of the art”. But modern NN-controlled-system (NNCS) verification frameworks built on Taylor/polynomial models (e.g., Verisig[1], POLAR[2]) already avoid SMT by using TM/polynomial propagation. The paper does not cite or discuss them, and the References section corroborates the omission. This leads to wrong baselining and an incomplete positioning of contributions. I believe the novelty of this paper is very unconvincing before the authors provide a direct comparison with Verisig and POLAR both theoretically and empirically.

[1] Ivanov, Radoslav, et al. "Verisig 2.0: Verification of neural network controllers using taylor model preconditioning." International Conference on Computer Aided Verification. Cham: Springer International Publishing, 2021.

[2] Huang, Chao, et al. "POLAR: A polynomial arithmetic framework for verifying neural-network controlled systems." International Symposium on Automated Technology for Verification and Analysis. Cham: Springer International Publishing, 2022.

### Questions
- Please clarify the novelty compared with Verisig and POLAR both theoretically and empirically.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes techniques for training and certifying neural network approximations of dynamical systems. The proposed method (i) constructs a loss function from the difference between the discrete- or continuous-time dynamics and the NN approximation, (ii) uses a Taylor-series approximation over hyperrectangle partitions and linear satisfiability checkers to check the approximation accuracy, and (iii) refines the hyperrectangle partition as needed to avoid false negatives in verification. The approach is evaluated on a collection of nonlinear ODE test systems to demonstrate scalability.

### Strengths
The approach is sound and improves on standard techniques such as dReal; to the best of my knowledge, this approach does not appear elsewhere in the literature and results in substantial speedup by converting nonlinear problems to (relaxed) linear ones. The presentation of the paper is easy to follow. The experiments include some nonlinear systems that scale up to seven dimensions, as well as a neural network compression example.

### Weaknesses
I have some questions regarding the results and contributions of the paper that are listed below.

### Questions
1. The paper claims to be about approximating dynamical systems, however, it appears to be about the more general problem of certifying the accuracy of a neural network approximation N to a known function f, with dynamics approximation as a motivating application. Can the authors comment on the generality of the results, and how they relate to the literature on function approximation via neural networks?

2. How does the paper compare to related works that use Taylor approximations and interval bounds for neural network reachability analysis (e.g., Verisig)?

3. In order to certify the results, upper and lower bounds on the Taylor approximation error are needed, which the authors compute by bounding the norm of the Hessian. In general, this requires solving a nonlinear optimization problem. How does the cost of computing these Taylor series bounds compare to the cost of directly solving the NN verification problem max{ ||f(x)-N(x)|| : x \in H_\delta(c)}?

4. I would be interested if the authors can comment on some of the simulation results. In particular, the fact that the 7-dimensional spacecraft model takes roughly the same order of magnitude to verify as the 3-dimensional steam governor is somewhat surprising.

### Soundness
4

### Presentation
4

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
This paper addresses the challenge of formally verifying neural networks that serve as approximations (abstractions) of nonlinear dynamical systems. The authors identify that existing state-of-the-art methods relying on Satisfiability Modulo Theories (SMT) solvers over nonlinear real arithmetic (e.g., dReal) face severe computational bottlenecks. To overcome this, the paper proposes a novel framework that replaces global nonlinear reasoning with local, certified linear approximations.

### Strengths
1. Significant Scalability Improvements: The shift from nonlinear SMT solving to utilizing established linear verification tools via local approximations is a pragmatic and impactful contribution.
2. Novel Applications for Verification: The paper moves beyond standard control benchmarks to demonstrate versatility.
    - Koopman Operators: Verifying trajectory-level predictions by treating the network as a discrete-time abstraction is a compelling use case.
    - Model Compression: Framing a large "teacher" network as the reference "dynamics" and formally verifying a highly compressed "student" network (98.4% smaller)  is an excellent demonstration of this framework's generality beyond analytical ODEs.
3. Effective Handling of Sparse Nonlinearity: The refinement strategy smartly decouples output dimensions and prioritizes splitting only along relevant input axes. This is highly effective for systems like the Jet Engine benchmark, where nonlinearity is confined to specific components, avoiding unnecessary partitioning of the entire state space.

### Weaknesses
1. Dependency on Local Linearity (The "Curse of Partitioning"): While avoiding the complexity of nonlinear SMT, the method trades it for potential combinatorial explosion in partitioning. For highly nonlinear functions where the second-order terms (Hessian) are large everywhere, the hyperrectangles must be extremely small to satisfy the error bounds. The current benchmarks may not fully stress-test this worst-case scenario where every dimension is highly coupled and nonlinear.
2. Koopman Verification Completeness: In the Koopman trajectory experiment, the method certified 99.03% of the domain, finding 29 counterexamples. While identifying safe regions is valuable, for safety-critical applications, the remaining 0.97% is the most important part. The paper does not deeply analyze why verification failed in those specific regions (e.g., genuine failure of the trained model vs. overly conservative bounds that could not be refined further).
3. Compression Benchmark Details: For the compression benchmark, CROWN is used to bound the large teacher network because it is a ReLU network (not twice differentiable). CROWN itself is a relaxation. The paper does not discuss how much additional conservatism (and thus extra partitioning) is introduced by using CROWN for the reference dynamics compared to an analytical function.

### Questions
1. The adaptive splitting strategy works well when nonlinearities are sparse or decoupled (e.g., Jet Engine ). Have you analyzed the behavior of your stack-based approach on a "pathological" synthetic system where every output depends highly nonlinearly on every input? At what point does the memory overhead of the stack  become a bottleneck.
2. In the Koopman experiment, 0.97% of the domain remained uncertified. Did you analyze these regions? Are they concentrated around specific state-space features (e.g., boundaries of basins of attraction), or are they scattered due to random approximation errors in the trained network?
3. How did the tightness of CROWN's bounds affect the verification time? Did you find that you needed significantly deeper partitioning for this task compared to benchmarks with analytical derivatives available for Lagrange bounds?

### Soundness
3

### Presentation
3

### Contribution
2
