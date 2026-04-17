# Exact Certification of Neural Networks and Partition Aggregation Ensembles against Label Poisoning

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Label-flipping attacks, which corrupt training labels to induce misclassifications at inference, remain a major threat to supervised learning models. This drives the need for robustness certificates that provide formal guarantees about a model's robustness under adversarially corrupted labels. Existing certification frameworks rely on ensemble techniques such as smoothing or partition aggregation, but treat the corresponding base classifiers as black boxes—yielding overly conservative guarantees. We introduce EnsembleCert, the first certification framework for partition aggregation ensembles that utilizes white-box knowledge of the base classifiers. Concretely, EnsembleCert yields tighter guarantees than black-box approaches by aggregating per-partition white-box certificates to compute ensemble-level guarantees in polynomial time.  To extract white-box knowledge from the base classifiers efficiently, we develop ScaLabelCert, a method that leverages the equivalence between sufficiently wide neural networks and kernel methods using the Neural Tangent Kernel. ScaLabelCert yields the first exact, polynomial-time calculable certificate for neural networks against label-flipping attacks. EnsembleCert is either on par, or significantly outperforms the existing partition-based black-box certificate. Exemplary, on CIFAR-10, our method can certify upto $\mathbf{+26.5\\%}$ more label flips in median over the test set compared to the existing black-box approach while requiring $\mathbf{100 \times}$  fewer partitions, thus challenging the prevailing notion that heavy partitioning is a necessity for strong certified robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EnsembleCert and ScaLabelCert, the methods to provide white-box certificates for partition-aggregation ensembles against label-flipping attacks. The methods utilize the white-box information about the base classifiers the target ensembles consists of. The protocols provide polynomial-time computable solutions for IP formulation for certification problem. Please use sparingly!

### Strengths
The work provides a novel theoretically grounded approach to certify the classification ensembles to label flipping attacks. The authors claim that a white-box knowledge of base classifiers can be used to significantly tighter ensemble-level certificates. The polynomial time solutions for relaxed IP formulations are provided, making the certificates computable at least for simple base classifiers.

### Weaknesses
No information about the computation overhead needed for the certification of an ensemble is provided. The authors indicate that the complexity of the relaxed IP problem for ensemble-wise certification scales quadratically with the number of dataset partitions, potentially making approach infeasible for large datasets and non-trivial base models. 

The improvement of an existing certification protocol (LabelCert) to provide sound certificates for infinitely-wide neural networks does not seem to be applicable in the experimental setup: the conditions up to which theoretical grounds of ScaLabelCert hold in terms of the base models' width are not studied. 

The effect of small constant C for base SVMs in ScaLabelCert remains understudied: at least empirical effect of the "smallness" of C on the performance of the classifiers as the function of dimensionality of the input samples and the cardinality of subdatasets used for training has to be studied.

### Questions
Please comment on the weaknesses above. I am willing to increase my score if the weaknesses are addressed.

### Soundness
3

### Presentation
2

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
The authors attempt to distinguish between poisoning defences that hold white box knowledge, versus those that don't, and attempt to demonstrate improved defensive utility.

### Strengths
On the surface, this paper is well written, is built upon interesting ideas, and works within an important space. The devil, however, very much is in the details (which I'll discuss in the following section). But I find the idea of trying to articulate and interrogate the intrinsic limitations of other approaches to be an interesting. 

The authors have also contextualised their work in the context of both deep proofs, and through references to more classical problems (including MCKP).

### Weaknesses
Okay, so, the aforementioned devil. To me, the primary problem is a lack of specificity (somewhat ironic, given the length of the appendices). For example, consider the white-box nature of the system. This is a crucial part of the overall conceptual landscape of this paper. Yet there are 25 different references to white-box information and 3 pages before the white-box information involved in the paper is defined in any way, which is on line 168. 

The white box nature of the paper also gives me pause on a logical level as well. For I'm not certain as to how realistic it is to be able to construct this for any problem of interest. There are black-box points of comparison, yet the important questions (to me) do not receive the level of attention that they deserve. And I don't mean my personal research interests - I mean questions on how this would scale, how this would be used, and what the real drawbacks of this sort of approach would be. Yes, the authors provide information on the P/NP complexity, and discuss polynomial time scaling, but this doesn't, to me, cover the actual practical realities of how this would actually behaved computationally, for systems of interest. 

Fundamentally, if this is not something that would be able to be realistically applied to problems of interest (due to the scaling of cost), I don't think the authors have appropriately contextualised what someone would get out of it. 

(Also L28 is not the correct use of the word exemplary)

### Questions
I'd appreciate answers to the following questions

1. Can you contextualise how "sufficiently small C" would behave across different problems of interest?
2. The focus upon looking at the absoute number of flips, rather than the proportion of flips within the dataset seems odd to me. Or, more ot the point, the authors do not make a clear case as to why the absolute number of flips is more important than how many flips three are relative to the size of the overall dataset. Could you comment upon this?
3. How would this scale? Yes, it may be polynomial-time calculable, but is it reasonable?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an approach for verifying piecewise-linear neural networks using SMT-based reasoning. The core idea is to encode networks using a combination of Linear Arithmetic (LA) and Equality with Uninterpreted Functions (EUF), treating non-linear activations (e.g., ReLU, MaxPool) as uninterpreted functions, while constraining their behavior through layered refinement. The authors propose a modular, layered abstraction-refinement encoding that separates neuron computations and enables partial constraint solving.

### Strengths
- Good theoretical contribution. The use of EUF to abstract activations while preserving soundness and completeness is well-motivated and cleanly integrated with LA solving.
- The slicing and auxiliary variable scheme is modular, allows early pruning, and avoids early enumeration of activation states.
- The tool can emit formal proofs (inveriT or veriT-compatible), aligning with the needs of safety-critical domains.
- Unlike existing verifiers which rely on relaxations or over-approximations, this method produces exact results (if it terminates).

### Weaknesses
- While the paper claims to be the first to apply LA+EUF to neural network verification, similar ideas have appeared in prior work. Ehlers (2017) and Reluplex (Katz et al., 2017) encoded ReLU and Max using symbolic logic and combined it with linear arithmetic.
Tools like Marabou and Planet also support exact ReLU/MaxPool verification with layered constraint refinement and symbolic splitting.
The proposed encoding (e.g., slicing, layered refinement) is well-structured and clean, but the core strategy is not entirely new. The true contribution lies in formalizing these ideas within a modular SMT framework—not in algorithmic novelty. As such, the paper slightly overstates its originality and should more precisely position its contributions relative to prior SMT-based verifiers.

- No comparison with standard baselines and benchmarks are basic (MNIST etc). The paper omits any empirical comparison with known verifiers (e.g., Marabou, Neurify, ReluVal, α-β-CROWN). Also lack of ablation or efficiency profiling.


- There is no discussion of potential scalability to modern deep learning models.

### Questions
1. Can your encoding support activations beyond ReLU/MaxPool (e.g., tanh or GELU)? If not, please clarify this limitation explicitly.
2. Why did you not compare against Marabou or α-β-CROWN on ACAS Xu or MNIST? These tools are standard and open-source.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates robustness certifications for partition-aggregation ensembles under label-flipping poisoning attacks for infinite-width neural networks.
The proposed method, EnsembleCert, provides ensemble-level certificates by aggregating white-box robustness certifications of the base classifiers.
The white-box knowledge to build certifications is extracted from base classifiers by means of ScaLabelCert, a method that relies on neural tangent kernels to obtain exact, polynomial-time certificates against label-flipping attacks.
The approach circumvents the overly conservative guarantees of other black-box techniques, and performs on-par or better than them, while being more efficient than existing Mixed Integer Linear Program approaches.

### Strengths
* **Clear theoretical contribution.**
The paper presents a clean and rigorous formulation of ensemble-level certification under label-flip attacks under the NTK assumption.
The reduction of the certification problem to a multiple-choice knapsack formulation, solvable via dynamic programming, is elegant and allows for tractable, provably tight ensemble certificates under the white-box setting.
The theoretical development is, as far as I can judge, internally consistent and sound.

* **Novel use of NTK-based exact certification.**
ScaLabelCert provides an exact, closed-form certification method for models under the NTK assumption.
Leveraging the NTK equivalence to derive analytical bounds on robustness is an interesting and, as far as I can judge, original direction compared to existing heuristic or black-box defences.

* **Tighter bounds and interpretability.**
Within its stated assumptions, EnsembleCert yields significantly tighter certified radii than prior black-box ensemble approaches, and the paper empirically demonstrates this with well-documented experiments.
The decomposition of ensemble robustness into per-partition contributions provides interpretability and insight into how partitioning affects robustness.

### Weaknesses
* **Finite width and NTK.**
For ScaLabelCert you rely on the assumption that the Neural Tangent Kernel (NTK) limit holds.
While I appreciate the clean mathematical treatment that NTK's dynamic allow for, I feel like a discussion of the applicability of your approach is not clearly presented.
In the abstract as well as throughout the text, for instance, you mention "sufficiently wide neural networks", alluding at the fact that, for these networks, you can provide exact certifications.
If I am not misunderstanding, though, your certification holds in the (exact) NTK limit, as you do not provide any description of a large but finite network width.
This gap between your theoretical advancements and standard neural network architectures should be more thoroughly discussed in my opinion.
In particular, for finite architectures, can the certificates you obtain be, in principle, arbitrarily wrong?

* **White-box inputs.**
Building on my previous comments, EnsembleCert relies on obtaining the smallest number of flips needed to change the prediction for one of the models in a partition.
This step too, relies on the NTK assumption, limiting the practical applicability of your approach.
While I agree that black-box estimates are often overly conservative, and agree that relying on white-box information can improve upon this, it seems to me that in a realistic setting what you can provide is heuristic statements, rather than guarantees.
Therefore, I think it may not be fair to compare against existing approaches that, while maybe overly conservative or prohibitively expensive, provide quantifiable _guarantees_ for realistic settings.
This to me is a crucial gap in the paper, and I look forward to any clarification on this point you may have.

* **Experiments with neural networks.**
In your experimental section you do not experiment with any neural network, nor validate the NTK approximation.
As you rely on soft-margin SVMs, which can be framed as convex optimization problems.
It remains unclear whether the performance boost of your approach stems from this fact mainly.
This setting seems to be far from the more realistic non-convex, finite-width neural network setting.
Therefore, I would say that your argument for scalability to larger neural networks is unsubstantiated.
If I missed some key parts of your argument, I am happy to discuss this point further.

### Questions
* Is it possible that the certificates produced under your NTK assumption are arbitrarily inaccurate for a finite network?
* How can you argue for your approach in a realistic setting where the NTK assumption may not be satisfied?
* How can you quantify how good your guarantees are if the NTK assumption does not hold exactly?
* Can you provide an heuristic/empirical or theoretical criterion for when a network is "sufficiently wide" for your guarantees to meaningfully apply?

### Soundness
3

### Presentation
2

### Contribution
2
