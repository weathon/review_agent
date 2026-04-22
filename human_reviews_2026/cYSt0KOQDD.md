# Out of the Shadows: Exploring a Latent Space for Neural Network Verification

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Neural networks are ubiquitous. However, they are often sensitive to small input changes.
Hence, to prevent unexpected behavior in safety-critical applications, their formal verification -- a notoriously hard problem -- is necessary.
Many state-of-the-art verification algorithms use reachability analysis or abstract interpretation to enclose the set of possible outputs of a neural network.
Often, the verification is inconclusive due to the conservatism of the enclosure.
To address this problem, we propose a novel specification-driven input refinement procedure, i.e., we iteratively enclose the preimage of a neural network for all unsafe outputs to reduce the set of possible inputs to only enclose the unsafe ones. 
For that, we transfer output specifications to the input space by exploiting a latent space, which is an artifact of the propagation of a projection-based set representation through a neural network.
A projection-based set representation, e.g., a zonotope, is a "shadow" of a higher-dimensional set -- a latent space -- that does not change during a set propagation through a neural network.
Hence, the input set and the output enclosure are "shadows" of the same latent space that we can use to transfer constraints.
We present an efficient verification tool for neural networks that uses our iterative refinement to significantly reduce the number of subproblems in a branch-and-bound procedure.
Using zonotopes as a set representation, unlike many other state-of-the-art approaches, our approach can be realized by only using matrix operations, which enables a significant speed-up through efficient GPU acceleration.
We demonstrate that our tool achieves competitive performance compared to the top-ranking tools of the international neural network verification competition.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces a novel latent-space formulation for neural network verification based on projection-based set representations, such as zonotopes. The key idea is to interpret the sets propagated through a network as “shadows” of a shared higher-dimensional latent set, which enables transferring output constraints back to the input domain for iterative specification-driven input refinement. The proposed method integrates this refinement into a branch-and-bound verification framework, implemented purely with matrix operations for efficient GPU acceleration. Experimental results on VNN-COMP’24 benchmarks show competitive performance and substantial reductions in subproblems compared with state-of-the-art tools.

### Strengths
1.Conceptual originality -The latent-space interpretation of set propagation provides a fresh geometric perspective on verification problems.
2.Technical soundness-The integration of constrained zonotopes and iterative refinement is mathematically consistent and grounded in established reachability theory.
3.Practical scalability-The matrix-only implementation with GPU acceleration is elegant and demonstrates a real performance gain.
4.Comprehensive evaluation -The comparison with multiple VNN-COMP tools and detailed ablations add credibility to the empirical claims.
5.Potential for future extensions-The framework could generalize to other activation functions and verification domains (e.g., control systems).

### Weaknesses
Clarification of Competition Reference： The paper states that the proposed tool would rank among the top of VNN-COMP’24, which, while illustrative, might risk breaking double-blind policy if the authors were participants or submitted related work. A clearer phrasing could avoid potential self-identification. 
 
Limited Discussion on Generalization： While the latent-space formulation elegantly improves efficiency, its scalability to larger architectures beyond benchmarks (e.g., transformers or diffusion models) is not discussed, leaving some uncertainty about general applicability. 
 
Evaluation Scope： Experiments are strong on VNN-COMP tasks, yet cross-domain or real-world safety-critical benchmarks (e.g., perception-driven models) could further demonstrate practical robustness. Overall, these are minor concerns rather than major flaws.

### Questions
1.How does the proposed latent-space refinement behave when using non-piecewise-linear activations (e.g., tanh, sigmoid)? Can the method generalize without losing soundness guarantees?
2.The refinement procedure iteratively constrains inputs based on unsafe output sets. How is termination ensured in practice—by a fixed iteration limit or a convergence threshold?
3.Since the algorithm uses only matrix operations, could the authors discuss numerical stability issues (e.g., accumulation of floating-point errors) in deep networks?
4.Double-Blind Compliance: The manuscript notes that the proposed verifier “would place it among the top-ranking tools of the last VNN-COMP’24.” Could the authors clarify whether they personally participated in VNN-COMP’24 or reused their own competition submissions? If so, this phrasing might unintentionally compromise anonymity. A general comparative statement without potential self-identification could be safer under ICLR’s double-blind policy. 
5.It would be valuable to understand the trade-off between refinement accuracy and computational overhead—does aggressive refinement ever hurt verification completeness?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel refinement method for zonotope-based verification of the robustness properties of neural networks.

Specifically, it describes a latent space, meaning a hypercube, with the property that the input zonotope, as well as the intermediate and output zonotopes while and after propagating through the network, are all "shadows" of this hypercube, that is, they are projections. This method enables the creation of a relatively tight enclosure of the set of inputs.

The performance of the proposed procedure is demonstrated experimentally on SOTA benchmarks by incorporating the proposed refinement into a branch-and-bound algorithm. The experiments show that it is often competitive
with existing high-performing ones verifiers. Additionally, the number of subproblems that
must be solved is, in some cases, significantly fewer than without the proposed refinement.

### Strengths
Improving the performance of FNN verifiers, particularly those based on zonotopes, is a topic of significant relevance and a suitable contribution for ICLR.

Furthermore, the paper is excellently written and thoroughly polished,
making it a pleasure to read. This is particularly evident in the balance
between formal rigour and the intuitive explanations provided by the authors.

### Weaknesses
I identify one notable weakness:
- The experimentally demonstrated improvements, while present, are not dramatic. I perceive the incremental enhancement offered by this refinement as a valuable contribution. Nonetheless, as I am not a specialist in this precise domain, I wonder if the enhancement might be too modest or if further extensive experiments are required to substantiate the minor, yet existent, improvements due to this refinement approach.

Minor issues include:
- The explanation of "shadows" as a concept of refinement is appreciated. However, while it is clear that the latent hypercube remains the same across steps but is rotated (see, for instance, Figure 2), the connection between this idea and Proposition 2 is not entirely clear. A slightly clearer exposition could be beneficial.
- Although it is not central to the paper, the introduction's claim that "verification of FNN is in general undecidable" and "NP-complete for ReLU FNN" feels superficial. There is not a single "verification problem" for FNNs. Various notions exist, such as robustness and reachability, each contingent on parameters like input and output set specifications and FNN architecture types. Consequently, this has spawned extensive literature analysing the complexity of these problems in detail (see, e.g., [1],[2],[3] and others). Furthermore, the undecidability assertion appears misplaced. While referenced in another paper, so not a statement of the authors, it arises simply from the existence of undecidable sets of reals and is not an inherent property of "FNN verification" problems.
- Proposition 2: The definition of $d:=...$ is unclear to me. This may be due to a typo.

--- 
[1] Marco Sälzer, Martin Lange:
Reachability in Simple Neural Networks. Fundam. Informaticae 189(3-4): 241-259 (2022)

[2] Adrian Wurm:
Robustness Verification in Neural Networks. CPAIOR (2) 2024: 263-278 

[3] Moritz Stargalla, Christoph Hertrich, Daniel Reichman:
The Computational Complexity of Counting Linear Regions in ReLU Neural Networks. CoRR abs/2505.16716 (2025)

### Questions
- I am curious about the reasonable next steps for the approach to refinement you have proposed here. More specifically, I am attempting to understand whether this "novel view" you propose paves the way for follow-up works and, if so, in what manner?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel verification algorithm for neural networks that integrates an iterative, specification-driven input refinement into a branch-and-bound procedure. The core idea is to use the dependency information preserved by zonotope propagation to "transfer" output constraints back to the input space, thereby refining the input set to focus only on potentially unsafe inputs. The implementation is designed for full GPU acceleration via matrix operations. The evaluation on VNN-COMP'24 benchmarks demonstrates competitive, state-of-the-art performance, with extensive ablation studies validating the design choices.
However, the paper's central conceptual framing of a "novel latent space" is overstated, as it repackages the standard mathematical interpretation of zonotopes without providing a meaningful new insight for the verification task. Furthermore, the analysis lacks a thorough discussion of the computational overhead introduced by the iterative refinement procedure itself.

**Recommendation:**  
Despite the conceptual overreach regarding the "latent space," the paper presents a solid, practical, and effective verification algorithm with strong empirical results. The core technical contribution of the iterative input refinement is novel and valuable. Given the competitive performance and efficient implementation, I am inclined towards **acceptance**, provided the authors significantly revise the framing to more accurately reflect the nature of their contribution and address the points above.

### Strengths
1.  **Novel and Sound Method:** The iterative input refinement procedure (Proposition 2) is a novel and technically sound contribution. By iteratively constraining the input set based on the unsafe output specification, the method effectively reduces the number of subproblems in the branch-and-bound tree, leading to faster verification.
2.  **Efficient Implementation:** The decision to implement the algorithm using only matrix operations to leverage full GPU acceleration is a significant practical strength. The results show a substantial speed-up compared to CPU execution, making the approach scalable.
3.  **Convincing Evaluation:** The experimental evaluation is comprehensive and transparent. The tool achieves competitive performance against the top-5 tools from VNN-COMP'24. The ablation studies effectively demonstrate the individual contributions of the input refinement, GPU acceleration, and the proposed enclosure-gradient splitting heuristic.
4.  **Practical Impact:** The method leads to tangible improvements in verification performance, including a reduced number of subproblems, faster solve times, and a high success rate across diverse benchmarks.

### Weaknesses
1.  **Overstated Framing and Novelty of "Latent Space":** The paper's central conceptual framing—a "novel latent space" for verification—is overstated. The observation that a zonotope is a projection of a hypercube is a standard, well-known interpretation of the zonotope definition. Maintaining a single, constant hypercube by padding generator matrices with zeros is an algebraic convenience rather than a profound theoretical insight. The paper fails to provide a meaningful interpretation of this "latent space" for the verification task itself. Consequently, the "latent space" narrative feels more like a post-hoc metaphor than a foundational concept that the algorithm genuinely depends upon in a deep way. The core algorithmic contribution (the refinement procedure) could be clearly explained without this conceptual overhead.
2.  **Incomplete Algorithm Description:** The presentation of the core algorithm (Algorithm 1) relies heavily on appendices for critical components, including the adversarial input computation (Section 4.2) and constraint computation (Proposition 2 and Appendix B). This makes it difficult to understand the complete method from the main text alone.
3.  **Limited Analysis of Refinement Overhead:** While the paper demonstrates that refinement reduces the number of subproblems, it doesn't adequately analyze the computational cost of the refinement procedure itself. The trade-off between reduced branching and increased per-iteration complexity deserves more discussion.

### Minor Issues
- Typo: P3 L149: $\beta\in [-1,1]^{q}$
- Typo: P5 L262: constrain
- I had problems understanding the statement of Proposition 2 because of the way it is formulated. Clarify that "$\mathcal{X}|_{C\leq d}$ encloses the intersection of the input set with the preimage of the unsafe set" is the main statement.

### Questions
*   Could the authors clarify what is gained by the "latent space" interpretation beyond providing an intuitive name for the shared factor space? Does it lead to algorithmic insights that are not apparent from a standard zonotope dependency analysis?
*   The paper should temper its claims regarding the novelty of the latent space view and reframe the contribution to more accurately highlight the effective iterative refinement procedure and its efficient GPU implementation as the primary advances.
*   Could the authors provide more analysis of the computational overhead of the refinement procedure and its impact on overall verification performance?
*   The algorithm description would benefit from including more critical details in the main text, particularly for the adversarial input computation and constraint generation steps.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a latent-space framework for abstract-domain propagation in neural network verification. The main idea is that instead of treating each layer's abstract element separately, the authors view them as projections of a higher-dimensional hypercube. This allows the method to "pull back" unsafe output constraints into the latent space and iteratively refine the input set based on those constraints. The approach integrates with a branch-and-bound (BnB) verifier and is designed to run efficiently on GPUs. Experimental results on VNN-COMP'24 benchmarks show competitive performance with fewer subproblems and good results.

The paper proposes a creative latent-space formulation for zonotope-based verification that conceptually unifies input and output abstractions and enables input refinement. While the idea is novel and shows promise for improving scalability, clarification of the soundness of the refinement and further exploration of the empirical impact is needed. Overall, the contribution is original and technically interesting.

### Strengths
The paper introduces a clear and creative latent-space interpretation of zonotope propagation. Viewing all layer abstractions as projections of a shared higher-dimensional B-space provides an interesting way to reason about dependencies between input, hidden, and output abstractions.

Proposition 2 provides a sound theoretical basis for transferring output-space constraints back to input space, and the overall refinement process integrates cleanly within a branch-and-bound verification loop. Well written pseudocode.

The experimental evaluation covers multiple VNN-COMP'24 benchmarks and includes evaluation that isolate the effects of the proposed refinements, GPU batching, and splitting strategies. The results show tangible improvements in efficiency (fewer subproblems, faster verification) while maintaining comparable verification coverage.

Figures such as Figure 3 and the toy example make the geometric intuition accessible and help connect the abstract mathematics to practical verification behavior. Great figures overall.

### Weaknesses
While Proposition 2 and Corollary 1 justify the logical use of the refinement in verification, they don't prove that the latent-space refinement preserves enclosure of the reachable set. In particular, the paper does not formally establish that the refined zonotope Y_refined still over-approximates all reachable outputs of the corresponding refined input region X_refined.

Although the method is described as applicable to "projection-based" abstractions, the implementation and evaluation are restricted to zonotopes. Consider either extending the analysis to another domain (e.g., star sets) or narrowing the claim.

The paper's main conceptual innovation, the latent-space input refinement, is only briefly evaluated. Table 2(a) reports only a few benchmarks, where acasxu shows reduced subproblems but no timing improvement, and timing metrics are absent from Table 1. Expanding the evaluation to include additional benchmarks in Table 2(a) and timing data in Table 1 (even in the appendix) would clarify whether the reported efficiency gains primarily stem from the refinement itself or from implementation factors, and would enable readers to better judge the consistency and generality of the refinement's impact.

Minor Comments/Typos Found

"contrain" - > "constrain"
Missing % for some of the results of the tables.
"activiation" - > "activation"
Consistency in notation for vectors (bold vs. plain).
Table 1: define "Solved" in the caption and clarify what counts as verified/falsified.

### Questions
Discuss and clarify weaknesses above.

Additionally, the empirical results presented are mostly on smaller neural networks from VNN-COMP. Please clarify and discuss the capability to handle larger networks and other tasks, such as image classification. Can the approach work on adversarial robustness of MNIST, CIFAR, etc.? This would perhaps better highlight the capabilities of the approach to scale, as claimed in the contributions (and that the reviewer agrees is an interesting and potentially effective idea to exploit the latent space as an abstraction).

### Soundness
3

### Presentation
3

### Contribution
3
