# Verifying GNNs with Readout is Intractable

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
We introduce a logical language for reasoning about quantized aggregate-combine graph neural networks with global readout (ACR-GNNs). We provide a logical characterization and use it to prove that verification tasks for quantized GNNs with readout are (co)NEXPTIME-complete. This result implies that the verification of quantized GNNs is computationally intractable, prompting substantial research efforts toward ensuring the safety of GNN-based systems. We also experimentally demonstrate that quantized ACR-GNN models are lightweight while maintaining good accuracy and generalization capabilities with respect to non-quantized models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a logic called $q\mathcal{L}$ that characterizes the expressivity of *quantized* Aggregate–Combine–Readout Graph Neural Networks (ACR GNNs), a widely used extension of vanilla message-passing GNNs for graph classification.

The authors use $q\mathcal{L}$ to show (co)NEXPTIME completeness of verification tasks on quantized ACR-GNNs via reduction to SAT/VALIDITY on $q\mathcal{L}$.

In their experiments, they illustrate on synthetic data and the protein–protein interaction (PPI) dataset that quantization does not significantly degrade the performance of models.

### Strengths
- **(S1) Problem Impact**: The paper addresses and seems to close an existing gap in the characterization of GNNs posed by [Sälzer et al., 2025](https://arxiv.org/abs/2502.16244), and provides a formalism that describes an important and widely used class of GNNs/graph properties.

- **(S2) Thorough analysis of $q\mathcal L$**:The investigation of the proposed logic is thorough, with examples highlighting its “reinterpretability” and compatibility with counting modal logic/description logic. While the main paper introduces $q\mathcal{L}$ as an extension of $\mathcal{L}\_{\text{quantGNN}}$ from [Sälzer et al., 2025](https://arxiv.org/abs/2502.16244) to add global aggregation, I personally like the additional connection to $K_♯$ [Nunn et al., 2024](https://arxiv.org/abs/2405.00205).

- **(S3) Self-Contained**: The paper provides ample explanation and background to follow the dense theoretical setting.

### Weaknesses
## W1: Presentation

The write-up of the paper is poor in grammar and clarity of language, with a few questionable slips, the most apparent being: “**NEXPTIME** is the class of problems decidable by a non-deterministic algorithm running in **polynomial time** in the size of its input,” where the authors clearly meant **exponential**.

The notation quietly changes after Section 3, where $\mathrm{agg}\_{g}$ suddenly becomes $\mathrm{agg}\_{\forall}$ henceforth.

Typesetting is strange, e.g., in lines 210–214, where the definition of the box operators is centered and the diamond operators are defined in line. After these definitions, the next sentence starts with a lower-case “and.”

In general, Section 3: *Simulating modal logic in $q\mathcal{L}$* is written in a confusing manner. The authors aim to show that they can re-express any formula with modal operators and connectives as an atomic formula by simulating these operators arithmetically. It is initially unclear why this is done (e.g., to simplify further proofs?).

The manuscript also uses up a lot of space on examples, but the central aspects---i.e., the ACR-GNN verification task definitions vt1, vt2, vt3, and the reason why these tasks are important in particular---are mentioned only in passing. As they are the central objects of investigation in the manuscript, this priority in levels of detail seems strange.

While the examples are helpful to follow the manuscript, they are used a lot, take up a lot of space, are partially a bit more complex than necessary (e.g., the example in lines 190–200), and sometimes replace proper definitions or explanations of concepts---for example, the definition of subexpressions $E(\phi)$ in Section 4.1.

## W2: Motivation and Experiments

I understand the motivation of the theoretical contribution, as global aggregation is commonly used in GNNs, and the authors do state that all ML models are quantized. 

The experimental evaluation then goes on to demonstrate that quantization does not impact the performance of GNNs a lot. This setup seems largely irrelevant to the theoretical results and seems to undermine the motivation of the authors rather than support it. The experiments act as proof of concept of using quantized ACR GNNs in practice. 
If the practicality of these models needs to be established, then it is questionable why they should be investigated from a theoretical point of view.

I think the authors could make a much stronger experimental statement by taking (pretrained) state-of-the-art models from existing literature on tasks that the community cares about, and stressing that **because** of similar performance, their quantized surrogates can be verified as a proxy. A study of how much speedup in terms of *verification time* versus the *loss in performance* would then provide more support for the theoretical setup, and could act as a guideline for future research by pointing out the value of verifying strongly compressed surrogate models.


## W3: Novelty

While the contribution seems to address a gap in existing research, the manuscript (including the structure) seems heavily inspired by [Sälzer et al., 2025](https://arxiv.org/abs/2502.16244), with adaptations to support global readout. As such, a lot of the submitted manuscript feels like it repeats information and does not focus enough on the novel contribution and motivation of the investigated verification settings.

## W4: Gap in Theoretical Results

The authors introduce $q\mathcal{L}$ to formalize the computations of ACR-GNNs, and show by example that any computation of the GNN can be expressed in the logic. However, it seems that a formal statement is missing that proves that the logic and the class of GNNs are, in fact, **equally** expressive.

### Questions
Of my listed weaknesses, I would appreciate a brief discussion of the authors to address my concerns in W2 and W4.

In addition, I have the following question to possibly expand on the contribution in the paper.

- **Q1 Simulation of global readout**: One unexplored aspect of ACR verification is a result of [Jogl et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebf95a6f3c575322da15d4fd0fc2b3c8-Paper-Conference.pdf), which proposes simulating, e.g., an ACR-GNN with an AC-GNN without global readout by using a graph transformation on input graphs. As ACR-GNNs can then apparently be simulated from simpler-to-verify GNNs (“only” PSPACE-complete), the question arises why the difference arises, and whether a theoretical connection can be made. Does the translation of the formulas lead to a blowup of size?


---

## References

- **Sälzer, M., Schwarzentruber, F., & Troquard, N. (2025).** *Verifying Quantized Graph Neural Networks is PSPACE-complete.* IJCAI 2025. arXiv:2502.16244. <https://arxiv.org/abs/2502.16244>

- **Nunn, P., Sälzer, M., Schwarzentruber, F., & Troquard, N. (2024).** *A Logic for Reasoning About Aggregate–Combine Graph Neural Networks (K♯).* IJCAI 2024. arXiv:2405.00205. <https://arxiv.org/abs/2405.00205>

- **Jogl, F., Thiessen, M., & Gärtner, T. (2023).** *Expressivity-Preserving GNN Simulation.* NeurIPS 2023. <https://proceedings.neurips.cc/paper_files/paper/2023/file/ebf95a6f3c575322da15d4fd0fc2b3c8-Paper-Conference.pdf>

### Soundness
2

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
In this article, the authors present new results about the expressivity of GNNs.
The authors consider a quantized variant of ReLU GNNs, meaning that the numbers used in computations all have a finite number of bits. The main contribution of the authors is to design a certain logic called $q\mathcal{L}$, to capture global readout. 

The main contributions are:

- The logic $q\mathcal{L}$ is NEXPTIME, meaning that given, the language associated to $q\mathcal{L}$, then recognizing if a formula based on this language can be satisfied, can be performed in with a non-deterministic Turing machine, for some polynomial (n is a parameter measuring the size of the formula).

- Restricting to GNNs without global readout, by adapting their language $q\mathcal{L}$ to that case yields a PSPACE-complete version, meaning that (i) it is in PSPACE: one can solve the same problem as above in polynomial space, and (ii) any problem in PSPACE can be reduced polynomially to a problem of satisfiability of a formula of the qL version without global readout.

- Preliminary experiments on the impact of quantization on the performance and model size in practice

### Strengths
- The nature of the contribution is interesting and timely, in the line of research about the expressivity of GNNs.

- This article can be very interesting for logicians, while still be relevant for a more general audience.

- The illustrating experiments going along with the main theoretical contribution are interesting.

### Weaknesses
- While the paper claims potential implications for the safety of GNNs, the connection between the theoretical expressivity results and concrete safety aspects is not made explicit. Clarifying this link would significantly strengthen the paper’s broader impact.

- Given the technical nature of the paper, it would be helpful if the authors included a more accessible, high-level explanation of their main contributions for readers outside the logic community. 

- Some imprecision, right from the beginning of the paper: please see first Question below.

### Questions
- It is said in first page (contributions): ``NEXPTIME is the class of problems decidable by a non-deterministic algorithm running in polynomial time in the size of its input''. Except if the size refers to the size of a compressed representation of the input string, this is incorrect to me. Rather, NEXPTIME should be the class of decision problems solvable by a non-deterministic Turing machine in exponential time, i.e., time $2^{p(n)}$ for some polynomial $p$. Please clarify.

- typo: l. 392: ``we introduction'' ->  we introduce

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper establishes that verifying **quantized Aggregate–Combine Graph Neural Networks with global Readout (ACR-GNNs)** is **(co)NEXPTIME-complete**, revealing that the verification of quantized GNNs is inherently intractable. To prove this, the authors introduce a logical language **qL**, which extends previous logics for GNN verification to handle **global readout** and **quantized arithmetic**. They provide reductions from qL to a quantized variant of **Quantifier-Free Boolean Algebra with Presburger Arithmetic (QFBAPA𝕂)**, showing decidability and tight complexity bounds. They also consider a bounded-graph setting where verification becomes **(co)NP-complete** and provide a proof-of-concept verifier. Finally, they present experimental evidence that **quantized ACR-GNNs maintain accuracy** while reducing model size and inference cost, supporting the practical utility of quantized models.

### Strengths
**Originality**  
- The paper provides the **first complete logical characterization** and **tight complexity bounds** for verifying quantized GNNs with global readout.  
- The use of **qL logic** to capture global readout within the quantized framework is an elegant and novel extension of prior formalisms like K♯ and FOC2.

**Quality**  
- The proofs are rigorous, adapting known techniques (e.g., Hintikka sets and reductions to QFBAPA𝕂) to the quantized and global readout setting.  
- The authors’ reasoning links theoretical intractability with practical implications, motivating further research into scalable verification strategies.  
- The bounded-vertex relaxation and prototype implementation show a **constructive direction** for future research.

**Clarity**  
- The structure is clear: the paper walks from formal definitions, through complexity proofs, to bounded relaxations and experiments.  
- The motivating examples (e.g., verifying properties of “dog” graphs) effectively illustrate the semantics of qL.  
- The appendix and code references enhance reproducibility.

**Significance**  
- The (co)NEXPTIME-completeness result fills a major theoretical gap in GNN verification.  
- The bounded-graph relaxation connects deep theory to **practical model checking**, potentially influencing future verification frameworks.

### Weaknesses
1. **Practical implications remain limited**  
   - Although the theoretical contribution is strong, the **practical relevance** of (co)NEXPTIME results could be elaborated—how does this shape actual verification tool design?  
   - The experiments, while informative, are **tangential** to the main verification focus.

2. **Experimental evaluation is lightweight**  
   - The experiments only test quantization effects on model accuracy and size, not the **actual verification performance**.  
   - The absence of benchmarks comparing **verification time** or **SMT encoding scalability** leaves open questions about the applicability of qL in realistic settings.

3. **Readout semantics assumption**  
   - The fixed summation order assumption in global aggregation (noted in the limitations) is **non-standard in practice**, which might restrict the generality of the theoretical claims.

4. **Notation density and accessibility**  
   - Some sections (e.g., Hintikka sets and reduction construction) are mathematically dense and could benefit from illustrative diagrams or examples of intermediate steps.

### Questions
1. **Verification tractability**  
   - Given the (co)NEXPTIME-completeness, what verification techniques could still be practically feasible for small or structured graphs?  
   - Can symbolic abstractions or over-approximations make qL-based verification tractable in practice?

2. **Extension to other architectures**  
   - How would the complexity results change for other GNN architectures, such as recurrent or attention-based models?  
   - Could qL be extended to handle continuous activations or message-passing schemes beyond summation?

3. **Bounded verification**  
   - The bounded-vertex setting leads to (co)NP-completeness. Are there heuristics or practical solvers that can efficiently address this fragment?  
   - How scalable is the provided proof-of-concept verifier when N grows beyond small graphs?

4. **Quantization modeling**  
   - The assumption of saturating arithmetic in 𝕂 simplifies reasoning, but how would modular or IEEE-style rounding affect decidability or complexity?  
   - Are there concrete examples where quantization introduces or removes counterexamples compared to unquantized GNNs?

5. **Experimental depth**  
   - Can the authors provide **verification runtimes or success rates** for the bounded verifier on benchmark GNNs?  
   - How does quantization influence the **verifiability** (not just accuracy) of ACR-GNNs in the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper narrows its scope to a critical yet underaddressed gap in GNN verification: the theoretical complexity of quantized Aggregate-Combine Graph Neural Networks with global readout (ACR-GNNs). While "neural network verification is intractable" is a broad community consensus, this work focuses explicitly on GNN-specific structures—namely, global readout (a core component for graph-level tasks like molecule classification or protein interaction prediction)—and quantized arithmetic (standard in real-world deployments). It introduces the logical language qL to formalize ACR-GNN computations and graph properties, proves that ACR-GNN verification tasks (sufficiency, necessity, consistency) are (co)NEXPTIME-complete, and contrasts this with the PSPACE-completeness of readout-free quantized GNNs. Complementing theory, the paper validates that quantized ACR-GNNs retain high accuracy ($\pm$1% drop) with 60-74% size reduction and proposes a bounded-vertex relaxation (NP/coNP-complete) for practical verification.

### Strengths
- Before this paper, the community knew readout-free quantized GNN verification was PSPACE-complete, but readout’s impact was speculative. By proving (co)NEXPTIME-completeness, the work quantifies this impact: global readout pushes complexity into a higher class, meaning verification becomes exponentially harder with increasing input size. This clarity prevents wasted effort.

- The paper’s (co)NP-complete bounded-vertex relaxation is not a random heuristic but a principled response to the (co)NEXPTIME result: by limiting counterexamples to graphs with N vertices, it leverages the boundary between "unbounded intractability" and "bounded tractability." This may provide a roadmap for future work—e.g., optimizing N selection, or combining bounded verification with domain-specific constraints (e.g., molecule graphs have ≤100 atoms)—that would be impossible without knowing the exact point at which intractability sets in.

### Weaknesses
- The theoretical analysis and experiments focus exclusively on summation for local/global aggregation. However, other modern GNNs use max, mean, or attention-based aggregation—readout for these variants may introduce different complexity patterns (e.g., max aggregation reduces dependency between distant vertices). The paper’s failure to extend bounds to non-summation aggregation limits its relevance to a narrow subset of industrial GNNs.

- Extensive experiments validate quantized ACR-GNNs' accuracy/lightweight design but do not link to core verification challenges, feeling tangential to the paper’s central claim.

- While technically rigorous, the (co)NEXPTIME-completeness result reinforces a known trend (readout exacerbates complexity) rather than offering a transformative insight, limiting broader field impact.

### Questions
- For practical applications (e.g., molecular graph verification), how do you recommend choosing the maximum number of vertices N? Is there a way to determine N such that if no counterexample exists for N, it is unlikely to exist for larger graphs?

### Soundness
3

### Presentation
3

### Contribution
2
