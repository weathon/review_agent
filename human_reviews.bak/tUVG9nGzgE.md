# Learning Conditional Invariances through Non-Commutativity

- Decision: Accept (poster)
- Scores: 5, 8, 6

## Abstract
Invariance learning algorithms that conditionally filter out domain-specific random variables as distractors, do so based only on the data semantics, and not the target domain under evaluation. We show that a provably optimal and sample-efficient way of learning conditional invariances is by relaxing the invariance criterion to be non-commutatively directed towards the target domain. Under domain asymmetry, i.e., when the target domain contains semantically relevant information absent in the source, the risk of the encoder $\varphi^*$ that is optimal on average across domains is strictly lower-bounded by the risk of the target-specific optimal encoder $\Phi^*_\tau$. We prove that non-commutativity steers the optimization towards $\Phi^*_\tau$ instead of $\varphi^*$, bringing the $\mathcal{H}$-divergence between domains down to zero, leading to a stricter bound on the target risk. Both our theory and experiments demonstrate that non-commutative invariance (NCI) can leverage source domain samples to meet the sample complexity needs of learning $\Phi^*_\tau$, surpassing SOTA invariance learning algorithms for domain adaptation, at times by over 2\%, approaching the performance of an oracle. Implementation is available at https://github.com/abhrac/nci.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The author present an idea of invariant learning, that is learning non-commutative invariances (NCI), which is conditioned on the target domain and therefore preserves target-specific information. When the target domain contains more semantically relevant information than the source domain, the authors theoretically show that the NCI approach is beneficial for domain adaptation. Empirical results are also provided and show the benefit of NCI approach.

### Strengths
The empirical results do show the benefit of the proposed NCI approach in domain adaptation tasks. The assumptions of asymmetry and non-commutative seems to be novel in the context of invariant learning. Also the authors aim to provide a general theoretical framework that captures the properties of invariant learning algorithms (through the lens of commutativity) and can explain the benefit of NCI.

### Weaknesses
The theory part of this work is very difficult to understand. Lack of necessary explanations of notations and definitions. See the questions for details.

### Questions
1. What is an optimal encoder $\Phi^{\star}$ (in what sense it is optimal, i.e., what is the objective function that it minimizes)?

2. In definition 1, what does an operator mean in the context of invariant learning (or domain adaptation)? Can you give an example of an operator in an invariant learning algorithm (e.g., DANN)?

3. You only give the definition of NCI for operators (in definition 1). What is a NCI encoder $\phi^{\star}$? Again, what does optimal mean in   Result 1?

4. In Theorem 1,  you write $s=\tau + \delta$. But how to do calculations between domain $s$ and $\tau$? You do not give any definition.

5. All the risks are not defined.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper shows that a provably optimal and sampleefficient way of learning conditional invariances is by relaxing the invariance criterion to be non-commutatively directed towards the target domain. Both their theory and experiments show that non-commutative invariance can leverage source domain samples to meet the sample complexity of learning the optimal target-specific encoder, surpassing SOTA invariance learning algorithms.

### Strengths
1. The theoretical analysis is comprehensive.
2. The experimental results are solid.

### Weaknesses
none

### Questions
none

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies Non-Commutative Invariance (NCI) as an efficient way of learning conditional invariances. The authors propose a method that leverages domain-specific information and focuses on the target domain while learning. They argue that relaxing the invariance criterion to be non-commutatively directed towards the target domain results in a more optimal and sample-efficient learning of conditional invariances. The paper highlights the superiority of NCI over SOTA invariance learning algorithms for domain adaptation.

### Strengths
**S1** The paper is well-written, with math notations clearly explained.

**S2** The empirical observations are supported by theoretical guarantees.

**S3** The discussion surrounding each theorem makes the theoretical results more understandable.

**S4** The paper presents extensive numerical experiments and demonstrates that their NCI-based approach surpasses existing SOTA algorithms in invariance learning.

### Weaknesses
Please see below.

### Questions
Section 3.1:

I suggest presenting the following statement as a formal assumption within a mathematical environment for better clarity.

"For the remainder of our analysis, we assume that the target domain contains more semantically relevant information than the source domain, i.e., ..."

Section 3.2:

Please ensure consistency in the use of symbols across theorems. If the symbol $\delta$ in Theorems 1 and 2 refers to different variables, this should be corrected to avoid confusion.

Clarification in Proof: The proof of Theorem 2 requires more clarity, particularly in the concluding part i.e. “ … samples from the target domain, which is the … This completes the proof of the theorem”. It's currently difficult to understand how the final result is derived from the given explanation. 

Related Work:

The technical comparison with the paper "Learning Conditional Invariance through Cycle Consistency" requires more detail.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
