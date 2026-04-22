# Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 6, 8, 8

## Abstract
*Automated circuit discovery* is a central tool in mechanistic interpretability for identifying the internal components of neural networks responsible for specific behaviors. While prior methods have made significant progress, they typically depend on heuristics or approximations and do not offer provable guarantees over continuous input domains for the resulting circuits. In this work, we leverage recent advances in neural network verification to propose a suite of automated algorithms that yield circuits with *provable guarantees*. We focus on three types of guarantees: (1) *input domain robustness*, ensuring the circuit agrees with the model across a continuous input region; (2) *robust patching*, certifying circuit alignment under continuous patching perturbations; and (3) *minimality*, formalizing and capturing a wide array of various notions of succinctness. Interestingly, we uncover a diverse set of novel theoretical connections among these three families of guarantees, with critical implications for the convergence of our algorithms. Finally, we conduct experiments with state-of-the-art verifiers on various vision models, showing that our algorithms yield circuits with substantially stronger robustness guarantees than standard circuit discovery methods, establishing a principled foundation for provable circuit discovery.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework for provable circuit discovery in mechanistic interpretability, offering formal guarantees of input robustness, patching robustness, and minimality. It introduces two algorithms—a greedy local-minimality search and a blocking-set duality method—for discovering circuits with verified faithfulness using α–β-CROWN. Experiments on MNIST, CIFAR-10, GTSRB, and TaxiNet confirm 100 % robustness within specified domains but show high computational cost.

### Strengths
1. Clear definitions and proofs for robustness and minimality in circuit discovery.

2. The greedy and hitting-set approaches are well motivated and practically implementable.

3. Connects mechanistic interpretability with formal verification through provable guarantees.

### Weaknesses
1. High computational cost: Verification with α–β-CROWN for each candidate circuit is extremely slow; scalability remains a bottleneck.

2. Small experimental scope: Only small convolutional and MLP networks are tested; no evidence on larger or modern architectures.

3. Limited interpretive analysis: The paper emphasizes correctness and robustness, but offers little discussion of what the discovered circuits mean semantically.

4. Evaluation imbalance: Most comparisons are against heuristic baselines without runtime or coverage trade-offs clearly analyzed.

### Questions
A recent paper, “Learning Minimal Neural Specifications” (Geng et al., NeuS 2025), finds that a small subset of neurons can often characterize a model’s robust behavior. Your minimal-circuit formulation appears conceptually related. Could you discuss the connections between these phenomena?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
In the context of mechanistic interpretability for neural networks, the authors use neural network verification methods to come up with circuits with provable guarantees. Authors claim to outperform conventional circuit discovery methods., i.e.e they provide stronger robustness guarantees for discovered circuits.

### Strengths
- Problem addressed is impactful and the authors identified clear blind spot in literature
- Thorough theoretical contribution
- Experiments rely on VNN-COMP community-benchmark. Results show authors method strongly outperforms the chosen baselines, across all experiments.
- Paper is clear and flows well. Authors contribution also clear.
- Literature review carried out with diligence. I am not a domain expert - fellow reviewers with greater expertise may have identified gaps.

### Weaknesses
- Some neural network verification concepts could have been introduced more in details (e.g. "Patching")
- Lack of running example make reading hard to instantiate into a real-world, impactful use case.

### Questions
- I could not find a discussion on the computational overhead of the method w.r.t. the baselines.  Could you briefly post a comment about it? (Apologies if I have missed)

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on automated circuit discovery in neural networks, one of the key challenges of mechanistic interpretability (MI). Current circuit discovery methods are heuristic and rely on sampling and approximations, therefore failing to provide theoretical guarantees about the obtained circuits (faithfulness and robustness to perturbations). The authors introduce a novel framework that uses neural network verification testing to discover circuits with three types of provable guarantees:
- Input-domain robustness, ensuring that the circuit's behavior remains faithful across a continuous input region.
- Patching-domain robustness, guaranteeing faithfulness under a continuous range of perturbations (patching) to the activations of non-circuit components.
- Minimality: The authors introduce and formalize four types of circuit minimality, hierarchically ordered (quasi-, local, subset- and cardinal minimality) and provide algorithms to either obtain or approximate those.

Some of the key technical contributions of this work include:
- A siamese network encoding, which allows standard verifiers to certify the aforementioned properties
- Identifying a "circuit monotonicity" property that enables stronger minimality guarantees
- Using a circuit-blocking-set duality based on Minimum Hitting Sets (MHS) to find cardinally-minimal circuits.

The authors validate their approach on several vision models and benchmarks, obtaining circuits with substantially stronger robustness guarantees than sampling-based baselines.

### Strengths
- The core contribution (applying formal verification to automated circuit discovery) is novel and significant. It directly addresses a fundamental and critical limitation in the field of MI (moving from heuristic approximations to provable guarantees), which has been acknowledged in many recent works in the literature. This work makes a significant step towards increasing the reliability of MI methods and is much-needed.
- The paper is technically excellent. The formalizations are clear, and the introduced hierarchy of minimality guarantees (Definitions 3 through 6) clarifies a concept often used loosely in the literature. All of the stated theorems are proved in depth in the Appendix, which is rare to find in this type of work.
- The experiments clearly support the paper's claims. The authors use the SOTA α,β-CROWN verifier on standard benchmarks, and their proposed methods achieve 100% certified robustness (Tables 1 and 2), whereas sampling-based methods largely fail. Finally, the experiments in section 5.3 are a good illustration of the trade-off between minimality strength (as defined in the proposed hierarchy) and runtime.
- The paper is very well-written, given the complexity of the topic.

### Weaknesses
- The most significant weakness (which the authors do acknowledge) is the scalability of the neural network verification methods. The experiments are conducted on relatively small vision models (e.g. ResNet2b) yet already require computation time in the minutes to hour range. Current MI research largely focuses on large LLMs or other models using the Transformers architecture, and as proposed, the authors' framework would likely be computationally intractable for such models. This is an inherent challenge of the verification field and by no means a flaw in the paper's methodology itself and doesn't detract from the foundational contribution of the work, but it does limit the immediate practical applicability to SOTA models.
- The paper could be strengthened by adding a more detailed qualitative analysis of the discovered circuits. For example, the provably robust circuit depicted in Figure 1 contains an extra component. A natural question is: What functional role does it play? An illustrative example would make the benefits of the approach even more tangible and would also be interesting from a research perspective.
- The paper uses a logit-difference metric for faithfulness, which works well for verification and is a standard choice in MI, but may not always accurately reflect the semantic meaning of a circuit that one may be interested in. It is not entirely clear if a small logit difference is always the most meaningful proxy for a circuit "doing the same thing" as the model (a circuit could potentially maintain a small logit difference but modify its internal representations or attributions in a way which may be significant). However, this is, again, the de facto standard used in most MI research and this is a fairly minor point.

### Questions
- Have the authors considered adapting the framework to certify different properties? For instance, instead of bounding the logit difference (which can be problematic, as stated above), could it be adapted to certify that the predicted class remains invariant across the input domain? This seems like a natural guarantee for classification.
- The complexity of MHS depends on enumerating blocking sets up to size t_max. What was the practical size of the blocking sets found in the experiments, and how large did t_max need to be to provide good lower bounds or find the minimal circuit? (My apologies if this is answered in the attached codebase, which I did not consult)
- For the example in Figure 1, do the authors have any functional hypothesis for why the components highlighted in green are essential for robustness (handling specific edge cases)? What kinds of input perturbations would cause the sampling-based circuit to fail, which the provably-robust circuit correctly handles?
- Do the authors have any insight into potential ways to apply this framework to larger models/architectures, such as Transformers? Are there specific verification techniques that offer a better trade-off for this application?

### Soundness
4

### Presentation
4

### Contribution
4
