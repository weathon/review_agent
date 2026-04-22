# Guaranteeing Privacy in Hybrid Quantum Learning through Theoretical Mechanisms

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Quantum Machine Learning (QML) is becoming increasingly prevalent due to its potential to enhance classical machine learning (ML) tasks, such as classification. Although quantum noise is often viewed as a major challenge in quantum computing, it also offers a unique opportunity to enhance privacy. In particular, intrinsic quantum noise provides a natural stochastic resource that, when rigorously analyzed within the differential privacy (DP) framework and composed with classical mechanisms, can satisfy formal $(\varepsilon, \delta)$-DP guarantees. This enables a reduction in the required classical perturbation without compromising the privacy budget, potentially improving model utility. However, the integration of classical and quantum noise for privacy preservation remains unexplored. In this work, we propose a hybrid noise-added mechanism, HYPER-Q, that combines classical and quantum noise to protect the privacy of QML models. We provide a comprehensive analysis of its privacy guarantees and establish theoretical bounds on its utility. Empirically, we demonstrate that HYPER-Q outperforms existing classical noise-based mechanisms in terms of adversarial robustness across multiple real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The manuscript proposes a hybrid privacy mechanism that combines classical differentially private input noise with depolarizing noise (quantum) to amplify differential privacy in quantum machine learning. The core idea is to consider a classical privacy mechanism to provide an initial DP and then use the depolarizing noise of the quantum circuit as a formal privacy amplifier.

### Strengths
The key strength is that this framework is the first of its kind: a novel theoretical framework for analyzing privacy (DP) in hybrid quantum-classical models. 
Second, it provides rigorous mathematical proofs (Theorems 1 and 2) showing that the quantum depolarizing noise acts as a privacy amplifier. 
The authors identify POVM trace balancing as a design rule to further tighten delta. This provided a concrete measurement layer condition for optimal amplification.
The paper is well written (except for the abstract) and is easy to follow. The work is commendable, lighting a torch towards a new direction of quantum noise-based differential privacy.

### Weaknesses
The abstract is misleading, although the manuscript follows by introducing DP and QDP eventually, the abstract says “noise means privacy” which is incomplete and misleading to a non-DP expert. 

The idea is very good. Leveraging the quantum noise as a DP mechanism is like realizing a foe can be a friend. But the formulation in the manuscript seems simpler than the actual noise scenario that is in quantum world.

It starts with the scope of the paper which considers only the depolarization noise. Before moving into the discussion of the theoretical formulation aspect, this umbrella assumption itself is flawed/incomplete.  The mathematical properties are not universal, although acknowledged, one of the most critical aspects is considered as future work.

For example, let’s just take some unavoidable temporal ones, like, amplitude damping noise (T1) or phase damping (T2). T1 is not a uniform contraction of the Bloch sphere. It asymmetrically pulls the state towards the ground state. It does not have the simple convex combination form. Another dangerous one is coherent noises. They contradict the whole concept of randomness; they are systematic. 

It is highly plausible that these noises might create a new vulnerability by systematically shifting outputs in a predictable way.

If the authors aim to show a proof a concept that the quantum noise has the potential to show DP properties, then, as per my judgment, it has already been done rigorously in the literature. The adaptation of quantum noise as an amplifier to classical form is a novelty but not enough. 

Experiments:
Datasets selected are although suitable for a first proof of concept, they are considered as the toy-datsets by the community; Mnist, FashionMNIST (just an extension with same dimension 28x28, and features as grayscale pixel value), and USPS. More complex, at least something like Cifar-10 is expected. 
This might steer in a new direction, say more qubits requirements. Adding qubits to a circuit generally reduces fidelity and forces η upward, quickly negating the dimensional advantage. Will privacy–utility curve changes shape? My intuition says it will.  

Clarification required regarding the narrative “HYPER-Q improves robustness in QML models.” 
The experiment compares HyperQ (0.1, 0.3) against classical-DP applied to the same QML model. While the HYPER-Q-protected QML model wins under very strict privacy budgets, it seems outperformed by a standard classical ResNet-9 when the privacy budget is relaxed. This suggests that the quantum advantage shown is not inherent to the QML model itself, but rather an artifact of the classical model being "over-penalized" by extreme noise. Is this happening? I would love a clarification.

### Questions
Is it quite impractical to assume that the quantum computers in today’s generation are unequipped with QEM techniques?
What kind of roles the famous QEMs will play in this formulation? For eg, Lets take ZNE.   What will be the effect on the variance? Will this deterministic extrapolation destroy the stochastic needed? Take PEC, for instance, it’s much more complicated.

### Soundness
2

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
2

### Summary
This paper proposes HYPER-Q, a new theoretical framework for ensuring differential privacy of hybrid quantum-classical machine learning (QML) models. The authors present the availability of quantum noise, previously considered only an error, as a useful resource to amplify privacy, and demonstrate the robustness of the proposed technique through experiments.

### Strengths
It addresses a very important and timely problem of ensuring privacy in practical hybrid quantum machine learning (QML) models.

It is considered a novel attempt to address how the classical DP mechanism and quantum noise can be theoretically combined to be exploited.

By mathematically proving the proposed technology using Theorem, theoretical completion is high, and it has been demonstrated through experiments.

### Weaknesses
Although the paper assumes that quantum noise is the perfect depolarizing channel, the noise in real NISQ devices takes a much more complex and asymmetric form. There is no analysis of whether the privacy "amplification" effect remains the same in non-defolarizing noise environments.

### Questions
In Figure 3, in the $\epsilon'=1.0$ interval, the classical model ResNet-9 performs better than HYPER-Q (QML). The benefits of HYPER-Q appear only under very strict privacy constraints ($\epsilon' \le 0.5$), which needs to be clarified why.

Theorem 1 (privacy amplification), the core of this paper, is proved to be the perfect depolarization channel for noise. Is privacy amplification always guaranteed even for general and asymmetric quantum channels?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates privacy guarantees for hybrid quantum-classical machine learning models. The authors propose HYPER-Q, a mechanism that composes a classical (ϵ,δ)-DP mechanism (e.g., input perturbation) with the intrinsic depolarizing noise (η) of a quantum circuit, which acts as a post-processing step. The paper provides a theoretical analysis of this composition, arguing that the quantum noise can act as a privacy amplifier. Theorem 1 analyzes the reduction in the failure probability δ, while Theorem 2 explores conditions for amplifying both ϵ and δ. A utility bound is also derived (Theorem 3). Empirical results on 5-qubit simulations suggest that for a fixed privacy budget, this hybrid approach can improve adversarial robustness over classical-only noise.

### Strengths
1.	The paper addresses an interesting and relevant problem: how to formally account for privacy in hybrid quantum-classical models. This is a valid direction for the QML community.
2.	The formal analysis of composing classical DP with quantum post-processing (Theorems 1 & 2) is a good theoretical starting point for this line of inquiry.
3.	The core idea of leveraging intrinsic quantum noise as a privacy-enhancing feature, rather than just a bug, is novel and worth exploring.

### Weaknesses
1.	The central, significant weakness is that the entire theoretical framework (Theorems 1 & 2, proofs) is valid only for the depolarizing channel. This is a highly simplified noise model. Real quantum hardware is dominated by other, more complex noise channels (e.g., amplitude damping, phase-flip, crosstalk) for which these proofs do not hold. Therefore, the paper's claims about providing privacy guarantees for "intrinsic quantum noise" are not general and may not apply to any practical quantum device.
2.	All experiments are conducted on a 5-qubit simulator. This is a "toy model" scale. It is well-known that QML model performance and trainability are highly dependent on scale. There is no evidence provided that the utility improvements (e.g., in Figure 2) will hold for QML models with a practical number of qubits (e.g., 50-100+). The claims of "practical viability" are severely undermined by this lack of scaling analysis.
3.	The proposed mechanism requires setting the quantum noise level η. The experiments show this choice is critical: η=0.1 works well, but η=0.3 leads to poor performance. The paper provides no principled method, algorithm, or even a strong heuristic for selecting an optimal η. This is a major methodological gap. A user cannot apply HYPER-Q without an expensive, brute-force grid search, making the method impractical.
4.	The theoretical utility bound (Theorem 3) is difficult to reconcile with the empirical results. The bound includes terms that explicitly increase the error as η increases, suggesting quantum noise is always detrimental to utility. This is in direct contradiction to the main empirical claim, which is that adding η=0.1 improves utility (accuracy) over the η=0 case (pure classical noise). This discrepancy is not addressed, raising questions about the tightness or practical relevance of the utility proof.

### Questions
1.	Given that all results are on 5-qubit simulations, what justification do you have that these findings, particularly the utility gains over classical models like ResNet-9, will scale to QML models of practical size (e.g., >50 qubits)?
2.	How do you reconcile the utility bound in Theorem 3, which implies utility worsens with η, with your empirical claim that utility improves when setting η =0.1 (compared to the η=0 baseline)? Does this not suggest the bound is too loose to be practically informative?
3.	Since your entire analysis is specific to the depolarizing channel, can you comment on the validity of your framework for more realistic noise models like amplitude and phase damping? Is it not the case that the claims of "guaranteeing privacy" are limited to a theoretical, idealized noise model?
4.	The paper shows η=0.1 is good and η=0.3 is bad. How should a practitioner choose η for a new dataset or model without resorting to a full hyperparameter sweep, which would be computationally expensive? Is there a theoretical principle to guide this trade-off?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes HYPER-Q, a hybrid privacy mechanism that composes classical input noise with intrinsic quantum noise in a hybrid QNN pipeline. and proves privacy amplification results (two theorems) and a utility bound (one theorem). It claims and demonstrates empirical improvements in certified and empirical adversarial robustness on MNIST / FashionMNIST / USPS under fixed end-to-end privacy budgets.

### Strengths
1. The original idea comes from the idea of using quantum post-processing to amplify the privacy.

2. The author proposed the detailed theoretical results. Theorems 1-3 show rigorous theoretical guarantees.

3. The experimental pipeline is clear. Uses standard datasets (MNIST/FashionMNIST/USPS) and compares to sensible baselines (Analytic Gaussian). Implementation details and compute resources are partially documented in the appendix.

### Weaknesses
While the proofs are provided in Appendix B, I want to flag spots where hidden assumptions could weaken results:

1. Corollaries 3 and 4 depend on POVM trace uniformity. Real measurements (projective or noisy POVMs) may violate these assumptions. This paper partially addresses an “optimal measurement” case, but it should discuss robustness to measurement mismatch.  Otherwise, the advantage the paper claimed is not so practical.

2. Many bounds in the theorems (such as Thm 1) seem to depend on the dimension $d$ of the encoded Hilbert space. In high-dimensional encodings, $d$ grows $2^n$. In Thm 1, the $\eta(1−e^\varepsilon)/d$ term shrinks. And obviously, to achieve the claimed result, there is a requirement for dimension $d$. It is worth discussing scaling and whether the amplification remains meaningful for small $\eta$ in practice. 

3. The classical noise is an input perturbation only. In classical DP for ML, many strong results come from gradient- or parameter-level DP (e.g., DPSGD). The authors should justify (or experimentally compare) input perturbation vs. other DP placements and explain how HYPER-Q composes with gradient DP if used in training.

### Questions
As I mentioned before, in quantum machine learning, the dimension of the encoded Hilbert space is quite crucial. 
1. Can authors discuss the influence of $d$ on the desired result?
2. It will be helpful to indicate how the parameters $d$ are configured in the experiment. I don't find the setting up of dimension $d$.

And there are some typos.

1. Comma missing. line 122 - 123. Line 339 - 342 and et.al.
2. Line 132: One column is missing the divergence expression.
3. Line 761-765: Where is the reference to the inequality?
4. Line 267-268: What is the statement of the post-processing theorem?

### Soundness
3

### Presentation
3

### Contribution
2
