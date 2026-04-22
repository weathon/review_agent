# Sound Verification of Deployed Neural Networks

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 6, 4, 6

## Abstract
Verification methods aim at mathematically proving desirable properties of neural networks, such as robustness to adversarial perturbations.  A verifier is sound if and only if it never claims that a neural network has the desired property when it does not.  It was shown recently that none of the currently known verifiers that are claimed to be sound are guaranteed to be sound when considering the deployed version of the verified network. Due to this, all the known verifiers are vulnerable to certain backdoor attacks, where an adversarial network passes verification but, in reality, it exhibits adversarial behavior in specific deployment environments. So far, it has been suspected that sound verification is prohibitively expensive if we wish to verify all possible executions&mdash;including parallel and stochastic ones&mdash;in deployment. *We are the first to propose an efficient error bounding technique that most known verifiers can apply to become practically sound.* The technique enables both interval bound propagation and symbolic propagation methods to remain sound even if the deployment environment randomly selects a valid ordering and parenthesizing of the arithmetic operations to compute the network. We present a theoretical foundation for our approach and demonstrate empirically that our technique indeed discovers all known deployment-specific attacks, introducing only a limited performance overhead.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Many neural network verifiers operate on a theoretical model, ignoring many practical aspects such as floating point inaccuracies.
This paper addresses this by designing verification algorithms that are also sound under these practical aspects.

### Strengths
- Coming from a formal verification perspective that usually ignores these practical aspects,
it's good to see that the field also explicitly expands in this area.
- The paper is easy to read, making a good trade-off between technical description and providing intuitive understanding.
- All technical details are rigorously developed.

### Weaknesses
- The contribution could be seen as marginal, given that verification under floating-point precision has been researched before.
- A running example could help with the intermediate understanding
- Only two simple verification algorithms are shown, which trivially also suffer from such constraints (e.g., huge output intervals).
- Explicitly stating the notation instead of just referencing other papers makes the paper self-contained.
- The evaluation could be expanded, e.g., on different data sets.

### Questions
- I believe VNN-COMP'25 had some issues with tolerances regarding floating points. Do you think these would have been resolved if all tools used your approach?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Existing neural network verifiers assume real arithmetic. Some account for floating-point error (FP error), but none of the existing verifiers account for the non-deterministic execution order of floating-point operations in modern architectures. This paper describes how existing bound propagation techniques for neural network verification can be extended to be sound under FP error and non-deterministic execution orders. The approach is based on bounding the maximal FP error that can accumulate from different execution orders for the primitive operations of a linear operation. Growing the bounds computed by interval arithmetic and zonotope propagation for neural networks by this FP error bound and using outward rounding makes these bound propagation approaches sound under FP error and non-deterministic execution orders. The experimental evaluation demonstrates that this additional rounding can detect several attacks on verifiers from the literature. The runtime overhead of the extension is up to 26% compared to an implementation without FP soundness.

### Strengths
This paper is a step towards extending the certificates provided by neural network verifiers to the real execution environments in which neural networks are deployed. I am not aware of comparable approaches in the literature. The proofs are correct, and the experimental evaluation is convincing.

### Weaknesses
The only significant weakness of this paper is its sloppy presentation. Please refer to the list of presentation issues below. I will raise my rating to "accept" if the authors address these issues during the rebuttal window. 

Beyond this, a weakness of this paper is that it can not address additional peculiarities of the hardware on which neural networks are deployed, so that the guarantees provided by this work still remain unsound in practice, as mentioned in the limitations section. Another weakness is that the paper only provides explicit methods for intervals and zonotopes, not for the most widely used polytope relaxations. In my opinion, both weaknesses are insignificant, given that research on extending the guarantees of neural network verifiers to real deployment environments is extremely sparse. Lastly, a hypothetical limitation of this paper is that its analysis breaks down if the product of the hidden layer width and the machine precision exceeds one, which might happen for very large networks in extremely quantised execution environments.

#### **Presentation Issues**
1. Reading the abstract, I did not know what "expression tree" referred to. Talking about the order of primitive operations, such as addition and scalar multiplication, would be easier to understand.
2. Instead of talking about "reasonable assumptions" in line 52, state that you provide soundness for non-deterministic execution orders, but not, for example, special GPU algorithms for matrix multiplication.
3. Quantify the "reasonably low overhead" in line 66. Also report the absolute numbers alongside the percentages.
4. Put brackets around your citations when they are not part of the sentence. For example, "is known to be NP-complete (Katz et al, 2017)" in line 75.
5. The sentence "sound (but not *necessarily* complete) verifiers aim to ... at the expense of completeness" is contradictory. You are talking about sound incomplete verifiers.
6. Please settle on one term for linear bound propagation instead of referring to it as "symbolic", "linear" (line 80), and "Polyhedra" (Table 1).
7. I appreciate that you cite early references for many approaches, such as Miné (2004) for linear interval expressions. In this spirit, there are clearly earlier references for IBP than Xu et al. (2020).
8. Stating that the outcome of an associative operation can strongly depend on the execution order in line 107 is contradictory.
9. The norm is not $p$ but ${\| \cdot \|}_p$ in line 127.
10. Your definition of $P(x^\ast)$ states that the *value* of the largest output of the network for $x$ needs to match the *value* of the true-class output for the reference input. That does not make sense. Since you have already introduced the notation, it is much easier to state $P(x^\ast) = \{x : y(x^\ast) = y(x)\}$. There are similar slip-ups in the remainder of the text.
11. omit the "very" in "very different" in line 161 or write "markedly different". 
12. The $D_{p, \epsilon, \ldots}$ in line 177 misses a $(x^\ast)$. 
13. "binary operation" is an ambiguous expression in line 186. I understood it to mean "expression with two arguments". However, IBP also uses intervals for unary operations. I think "binary" can safely be omitted here and in the remainder of the paper.
14. The sentence in line 190 is unclear. Besides, it could probably use a citation to "On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models" by Gowal et al. 
15. Introduce the $\delta_i$ variables more clearly. In particular, $\delta_i \in \mathbb{R}$. At first, I thought $\delta_i$ was a positive constant.
16. Similarly, introduce $\delta(n)$ in Lemma 1.
17. I think $l, u$ should lie in $X$, not $\mathbb{R}^n$ in Proposition 2.
18. Avoid formulations like "it is obvious" in line 344. Instead, give a brief reasoning. 
19. Your IBP ReLU relaxation is faulty. It should be $\max(l, 0)$, not $\min(l, 0)$ in line 248.
20. By "affine arithmetic" in line 362, are you referring to real arithmetic as opposed to floating-point arithmetic?
21. Line 368, "Of course, more sophisticated approximations can also be used". If you have an FP-sound formulation of CROWN, please provide it. Otherwise, strike this sentence.
22. What is $h$ in line 413?=
23. Also provide absolute runtime overheads in sections in line 445.
24. The references contain duplicated URLs and DOIs.
25. Some DOIs, such as 10.1007/978-3-319-77935-5\_9 are unresolvable.
26. Some abbreviations and tool names are not capitalized correctly in the references, such as "Intervalarithmetic.jl".

I did not read the entire appendix, but please also correct issues similar to the above in the appendix.

### Questions
- Equation (9) requires a `np.where` operation, or similar, to select among the three cases in a vectorised computation. Can that lead to any numerical issues?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Formal verification of neural networks does not guarantee safety in deployed models due to technical factors affecting floating-point computations, such as precision and computation order. The paper follows a method [Higham, (2002)] to bound the backward error of inner products and introduces a technique to formally bound the output value of a computation regardless of its execution order. In addition, the authors propose two improved versions of existing bounding methods (FPSoundIBP and FPSoundSymbolic). These methods are compared to the original versions (SoundIBP and SoundSymbolic) in terms of soundness, runtime, and accuracy (output-range similarity). The results demonstrate that the new versions guarantee soundness while preserving similar runtime.

### Strengths
1. The paper addresses relevant challenges in the formal verification of neural networks.
2. It takes a first step toward the formal verification of deployed neural networks.
3. The method is integrated into two common bounding mechanisms used in formal verification.

### Weaknesses
1. Limited evaluation: rows 1-10 in Table 1 appear in previous work [Szász et al. (2025)], so the new experimental results include only rows 11-12 in Table 1, which only confirm the theoretical results but do not supply additional information. Figure 1 and Table 2 are based on experiments with 100 input samples and one model trained on MNIST.

2. Limited soundness: 
- The claim that the method preserves runtime is not supported by the results for IBP, where the runtime increased by ~25%.
- The claim (Line 27, Line 65) that the method preserves accuracy (output-range) is not supported in the only check with respect to Order3 attack, where the output range is much larger for Order3. The method should be compared in other environments (Pr., Order1, Order2, Zombori et al. (2021)) as well.

3. Scalability: The assumption that $n\cdot\mu<1$ limits the scalability of the proposed method (if $n>1/\mu$). Moreover, multiplying $(2n − 1)$ times by $\Delta$ seems to significantly increase the bounds on the result.

4. Missing related literature: No prior work on formal verification of quantized networks [1, 2, 3] is mentioned.

5. Readability issues:

- The term “expression tree” appears five times (two of them in the abstract) before being explained at Line 215.

- The authors claim they are “closely following [Higham (2002)]” to bound inner products, but the technique of [Higham (2002)] is not explained at all, although it appears to be the core of the proposed bounding method.

- Line 361: “similar to DeepZ but without using affine arithmetic.” DeepZ's details are not explained in the paper.

- Line 413: What is h?

A. Towards Efficient Verification of Quantized Neural Networks (Huang et al., AAAI 2024).

B. QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks (Zhang et al., ASE 2022).

C. Scalable verification of quantized neural networks (Henzinger et al., AAAI 2021).
QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks (Zhang et al., ASE 2022)
Scalable verification of quantized neural networks (Henzinger et al., AAAI 2021).

### Questions
1. The work does not support complete verification. It is recommended to change the title to a more modest one (e.g., “Towards…”).

2. It is stated (Lines 240-243) that the error of each calculation can be expressed as a multiplication by 
$(1+\delta_i)$, but there is an example (Lines 153–155) where the result changes from 0 to 1. How does multiplication correct the error in this case?

3. Can the authors share the (average) values of $n$, $\delta_i$ and $\Delta$​ in the experiments? What is the average ratio $\delta_i/\Delta$? It can help the reader to approximate the effect on the output range.

4. It seems that all rows in Table 1, except the last two, are taken from [Szász et al. (2025)]. Is that correct? Why didn’t the authors state this explicitly?

### Soundness
1

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
3

### Summary
Existing verifiers typically validate only the ideal mathematical model of a neural network, whereas real-world deployments can introduce deviations due to floating-point precision, operation ordering, and parallel execution. Attackers may exploit these discrepancies to embed deployment-specific backdoors: a model appears safe during verification but behaves maliciously once deployed. This paper aims to incorporate all possible numerical execution behaviors that may occur during deployment into the verification process to ensure that robustness guarantees hold in realistic execution environments. For example, after a model is trained, an attacker may target a specific deployment platform with certain numerical properties, identify neurons whose behavior is highly sensitive to these properties, and craft trigger inputs that activate them only on that platform, thereby manipulating the model’s output while bypassing verification. To defend against such threats, the authors derive a order-independent relative error bound ∆ based on backward floating-point error analysis and use it to widen the inner-product computations so that the resulting intervals provably cover all possible floating-point outcomes across deployment environments. This ensures that verification conclusions remain valid under any feasible deployment execution.

### Strengths
1. The paper is among the first to explicitly address minute numerical discrepancies across deployment scenarios and demonstrate that these can be exploited to create stealthy backdoors affecting model trustworthiness.
2. The mathematical formulation and soundness proof are rigorous and professionally presented.
3. The structure and argumentation are generally well organized.

### Weaknesses
1. The motivation and threat model remain abstract. The description of deployment-specific backdoors in the Introduction is theoretical and may be difficult to follow for non-experts. A visual attack-flow illustration would significantly enhance clarity (e.g., attacker characterizing deployment → constructing environment-sensitive detector neuron → crafting trigger inputs → activation upon deployment).
2. Lack of flexibility and ablation for the widening parameter ∆. Although ∆ is theoretically derived, the paper does not evaluate the sensitivity of the verifier to its scaling (strictness vs. conservativeness trade-off). Ablation using a scaling factor (e.g., αΔ) or analysis across different depths / fan-ins would strengthen empirical understanding.
3. Insufficient explanation of runtime overhead causes. While empirical runtime curves are provided, the paper does not clearly articulate where the additional cost comes from (e.g., more unstable ReLUs → more linear relaxations → increased concretization). A short explanation—possibly with breakdown—would help guide future optimization.
4. Assumptions and applicability need clearer visibility. Some limitations (e.g., not addressing overflow/underflow, numerically approximated operations) appear only near the conclusion. These constraints are important in practice and should be highlighted earlier, including discussion of potential extensions.

### Questions
1. Could you include one or two illustrative diagrams in the Introduction showing:
(a) how an attacker detects or infers a target deployment setup,
(b) how environment-sensitive detector neurons are constructed or identified, and
(c) how trigger inputs activate malicious behavior only in deployment?
Using Order3 or the precision-based attack as an example would be highly instructive.

2. While ∆ is theoretically determined, could you comment on or experiment with its tunability?
For example, sweeping a scaling factor αΔ and reporting how robustness success rate, interval width, and runtime change? Also, how sensitive is ∆ to layer-wise fan-in? Are some layers more influential than others?

3. Could you provide a brief explanation or breakdown of runtime overhead sources?
Even a coarse analysis—such as contributions from interval scaling & outward rounding, unstable-ReLU relaxations, symbolic expression growth—would help clarify operational trade-offs and guide future improvements.

### Soundness
3

### Presentation
1

### Contribution
3
