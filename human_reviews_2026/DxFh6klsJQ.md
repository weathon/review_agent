# Electric Currents for Discrete Data Generation

- Avg Score: 2.80
- Decision: Reject
- Scores: 4, 2, 2, 4, 2

## Abstract
We propose **E**lectric **C**urrent **D**iscrete **D**ata **G**eneration (ECD$^{2}$G), a pioneering method for data generation in discrete settings that is grounded in electrical engineering theory. Our approach draws an analogy between electric current flow in a circuit and the transfer of probability mass between data distributions. We interpret samples from the source distribution as current input nodes of a circuit and samples from the target distribution as current output nodes. A neural network is then used to learn the electric currents to represent the probability flow in the circuit. To map the source distribution to the target, we sample from the source and transport these samples along the circuit pathways according to the learned currents. This process provably guarantees transfer between data distributions. We present proof-of-concept experiments to illustrate our ECD$^{2}$G method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper mainly aims to answer the research question: How can we design a principled, physics-inspired method to transfer probability mass between two discrete data distributions and thereby generate discrete data, with guarantees that the transfer exactly matches the target distribution?

### Strengths
1. **[Important] ECD2G framework (electric-current analogy for discrete generation).** The paper models a discrete generative task as current flow on an L-partite graph, whose input layer carries the source distribution and output layer carries the target distribution. Kirchhoff’s current law and Ohm’s law connect node potentials and edge currents.
2. **[Important] Conceptually appealing design.** The conceptual move of mapping discrete distribution transfer to unit electrical flow with potentials/currents is neat and yields closed-form expressions for a particular circuit family. The transport-plan independence is conceptually appealing.

### Weaknesses
1. **[Important] Empirical validation seems to be purely qualitative and very small-scale.** Experiments are 1D U→N and 2D Moons→Swiss Roll grids; there appear to be no quantitative metrics (e.g., KL divergence), no ablations on R/r/L, and no comparisons to DDM/DFM/GFlowNets.

2. **[Important] Scalability concerns in high-dimensional discrete spaces.** The movement probabilities require summing currents over all outgoing edges with a dense next layer; this can be prohibitive. The authors also acknowledge the need for factorisation.

3. **Limited guidance on hyperparameter selection and sensitivity** (e.g., L/R/r). Qualitative comments exist (γ controls change rate), but there seems to be no systematic study.

### Questions
Refer to "Weaknesses"

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces ECD2G, a novel generative modeling approach for discrete data that draws analogies between electric current flow in circuits and probability mass transport between distributions. It provides theoretical guarantees to prove that such transport 1) terminates in finite time and 2) correctly transports from the source distribution to the target distribution.While the theoretical framework is creative, the work suffers from significant practical limitations and insufficient experimental validation. It is only validated on toy datasets.

### Strengths
1. Novel theoretical framework: The perspective on combining electrical engineering principles into generative modeling is quite innovative and well-grounded in physics
2. Mathematical guarantees provided: Formal proofs for finite-step termination (Proposition 3.2) and correct transport (Corollary 3.3)
3. Conceptually clear framework: The L-partite graph structure and resistance parameterization are intuitive
4. Transport plan independence (Remark 3.4): Potential depends only on marginals, not on the coupling

### Weaknesses
1. Experimental validation is very limited: Model is only tested on trivial problems (1D: 50 states, 2D: 2,500 states) with no real-world applications despite claims about text, molecules, and images in the introduction
2. No quantitative metrics and no baseline comparisons: Only visual inspection of results, no distributional distance metrics, and zero comparison to discrete diffusion, flow matching, or other generative models
3. Fundamental scalability issue: Inference is bottlenecked by the denominator in Equation 9, which requires summing over S^D states—computationally infeasible for realistic problems
4. Factorization problem acknowledged but unsolved: Authors admit in Section 6 that scaling requires factorization, but provide no solution
5. Arbitrary design choices without justification: No guidance on selecting L, R, r parameters or circuit architecture

### Questions
1. Can the model be validated on real datasets such as ImageNet/CIFAR/MNIST or text generation benchmarks?
2. What is the computational cost of inference and training as a function of S, D, and L? How does it scale?
3. While innovative, what is the concrete superiority of this method compared to existing approaches like discrete diffusion models? What advantages justify the added complexity?
4. Why stop at D=2? Can you show results for D=3,4,5 to demonstrate where the method breaks down?
5. Can you provide even a preliminary factorization scheme to make the approach tractable beyond toy problems?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces ECD2G, a new generative model for discrete data inspired by electric current flow in circuits.

### Strengths
- The electric current-based generative models are novel and have the potential to be a good solution for discrete data generation.
- The overall methodology is explained clearly with a solid theoretical framework.

### Weaknesses
Generally I believe this work is promising but far from ready to publish yet, see the following comments:
- The motivation for using electric circuits to model the generative process seems somewhat arbitrary. It will be good if the authors could elaborate more on how they came to this idea.
- Some necessary comparison with other methods for discrete data generation is missing. The authors should either compare with the methods mentioned in lines 335-345, or explain why they are not comparable. 
- Study on the efficiency compared to other methods is missing.
- Currently the experiments are based on very simple tasks. A study on some more practical tasks may be necessary.

### Questions
- Could the authors elaborate more on how they came to this idea (i.e. connecting data generation with electric circuits)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Electric Current Discrete Data Generation (ECD$^2$G), a novel framework for generative modeling in discrete domains inspired by electrical engineering. The key idea is to model the transfer of probability mass between discrete data distributions as an electric current flow in a circuit. Input nodes represent samples from the source distribution, output nodes correspond to the target distribution, and the currents, learned via a neural network, encode the probability flow through the circuit. The authors develop a theoretical formulation grounded in Kirchhoff’s and Ohm’s laws, define a stochastic movement rule to realize the probability transfer, and propose a training algorithm that approximates the electric currents using regression.

### Strengths
1. The paper provides an extensive theoretical construction, linking discrete generation to well-established physical laws
2. The analogy with Kirchhoff’s current law ensures that probability mass is preserved during generation, offering a clear interpretability of the model’s constraints
3. Simple but consistent toy experiments show that the approach can approximately recover target distributions, supporting the conceptual feasibility

### Weaknesses
1. The mathematical notation is often ambiguous, making it difficult to follow key intuitions and derivations
2. While the physical analogy is interesting, the paper fails to convey why this analogy leads to practical or computational advantages over existing methods (e.g., discrete diffusion or flow matching)
3. Theoretical sections are disproportionately long and detailed, while experiments are extremely limited and do not convincingly demonstrate the model’s utility

### Questions
1. What are the computational or representational benefits of the electric circuit formulation compared to other discrete generative frameworks like Discrete Flow Matching or Discrete Diffusion Models?

2. A vertex $x_0 ∈X_{(0)}$ is defined as a (single) source vertex in $V^{(src)}$. Similarly, $x_L ∈X_{(L)}$ is a (single) sink in $V^{(snk)}$. What's the interpretation of $R_{\ell,\ell+1}(x_\ell,x_{\ell+1})$? I understood that $\ell$ is an index for layers, hence is $x_\ell$ a single node or all the nodes in a layer? In equation 11, what do you mean for $x_\ell = x_{\ell+1}$?

3. In equation 12, did you interpret the potential $\phi$ as a probability distribution? Are you exploiting the law of total probability? In that case, I would expected some integrals, since potentials are continuous. Which is the rationale of the equation?

4. The proposed electrical interpretation of optimal transport leaves me somewhat unconvinced for two main reasons:
- Defining, in probabilistic terms, the connection of a node $x$ with its nodes in $fan_{out}$ implies that its mass (or current intensity) would be transferred to only one of the output nodes. However, the physics of electrical circuits aligns more naturally with a fuzzy logic interpretation, since in reality the current would split and flow toward all output nodes.
- The statement \textit{``Remark 3.4 (Transport plan independence). Potential $\phi_\ell(x_\ell)$ is independent from $\pi(x_0, x_L)$''} does not correspond well to the physical behavior of electrical circuits. I admit, however, that since it's not clear to me the meaning of $x_0$ and $x_L$, it's difficult to discuss about it.
Could you better explain your intuition?

5. In equation 14 the ground truth is provided by a Monte Carlo method. May you better explain how to build a suitable network to address optimal transportation problems? May you also explain which is the benefit of your model since in any case there is the need of another estimation approach?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Electric Current Discrete Data Generation (ECD2G), a novel generative modeling framework for discrete data inspired by electrical circuit theory. The authors draw an analogy between the flow of electric current and the transfer of probability mass between data distributions. Using Kirchhoff’s and Ohm’s laws, they formalize a system where probability flows from a source distribution $p4 to a target distribution $q$ through a multi-layered circuit graph. Neural networks approximate the local currents between layers, enabling sampling and generation of discrete data via learned stochastic transitions.

### Strengths
* The paper introduces an idea that is well grounded and studied in physics (Kirchhoff's and Ohm's law)
* The paper is easy to follow

### Weaknesses
The research question is unclear to me: in the examples provided, it is unclear to me why the samples from the input distribution should follow the electrical flow to become samples of the target distribution. In contrast to the 2022 NeurIPS paper, the authors do not specify which partial differential equation has to be obeyed and why.

Also, what is the advantage of using a multi-layer architecture? Why is it difficult to learn this function directly?

### Questions
1. For what type of discrete datasets is the electric current model a good modelling assumptions?

2. Why is the method not compared to more standard approaches of learning latent embeddings for probability distribution transformations?

### Soundness
2

### Presentation
2

### Contribution
2
