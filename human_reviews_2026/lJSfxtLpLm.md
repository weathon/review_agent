# Convergence of Muon with Newton-Schulz

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 6

## Abstract
We analyze Muon as originally proposed, using the momentum orthogonalization with a few Newton-Schulz steps. The prior theoretical results replace this key step in Muon with an exact SVD-based polar factor. We prove that Muon with Newton-Schulz converges to a stationary point with the same rate as the SVD-polar idealization, up to a constant factor for given the number of Newton-Schulz steps $q$. We further analyze this constant factor, and prove that it converges to 1 doubly exponentially in $q$ and improves with the degree of a polynomial used in Newton-Schulz for approximating the orthogonalization direction.
We also prove that Muon improves the rank dependence compared to its vector-based counterpart, SGD with momentum. 
Our results explain why Muon with a few low-degree Newton-Schulz steps matches exact-polar (SVD) behavior at much faster wall-clock time, and explain how much momentum matrix orthogonalization via Newton-Schulz benefits over the vector-based optimizer. Overall, our theory justifies the practical Newton-Schulz design of Muon, narrowing its practice–theory gap.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the MUON optimizer, which is designed for matrix-structured parameters. In practice, MUON replaces expensive SVD computations with a few Newton-Schulz (NS) steps to orthogonalize its momentum. Previous analyses only covered the idealized SVD version. This paper gives the first convergence proof for the practical version that actually uses NS.

The main result shows that MUON with NS converges to a stationary point at the same rate as the SVD version, up to a small constant that converges doubly exponentially fast to 1 as the number of NS steps increases. The paper also shows that MUON improves rank dependence compared to SGD with momentum. Experiments on CIFAR-10 confirm the theoretical findings.

### Strengths
The paper is well written and well organized. The intuition behind the proofs is explained in a way that makes the technical results easy to follow.

This work provides the first convergence proof for MUON when the orthogonalization step is done with Newton–Schulz iterations. This directly corresponds to how MUON is implemented and used in practice, rather than analyzing an idealized or simplified variant. This is an important contribution because most previous works relied on the unrealistic assumption of an exact SVD-based orthogonalization, which is not used in real training.

The theoretical assumptions are reasonable and consistent with standard conditions in stochastic optimization. They are stated clearly and make sense for modern deep learning models. It is also appreciated that the authors do not assume extra smoothness or artificial conditions beyond what is standard for first-order methods.

The main theorems are precise, and the proofs are well-structured and clear. The paper shows in detail how the Newton-Schulz approximation error decreases and how it influences the convergence rate. The results are intuitive and easy to interpret.

Finally, the comparisons between MUON with NS, MUON with SVD, and SGD with momentum are well designed. They are based on the same assumptions, metrics, and convergence definitions, which makes the differences in behavior easy to understand. This unified view helps highlight the benefit of MUON’s matrix-aware design.

### Weaknesses
The paper is already strong in both quality and presentation, but there are still a few limitations.

The numerical experiments are pretty limited (which is fine given how popular Muon is), but it would be better to have larger-scale experiments when comparing methods. The results are shown only on CIFAR-10 using a small CNN. While this setup is enough to confirm the theory, it does not demonstrate the behavior of MUON in more realistic large-scale scenarios. Testing the optimizer on larger architectures, such as Transformers or vision models with high-rank layers, would make the empirical section much more convincing and highlight the practical impact of the theoretical results.

Although the theoretical assumptions are standard and well-justified, their validity in large deep networks can be questioned. In practice, smoothness and bounded variance often fail to hold exactly. For example, Transformer gradients can have heavy-tailed distributions, which makes the variance assumption only approximately true.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The goal of this work is to provide a formal analysis and proof of the convergence of the recently introduced and potentially useful MUON optimization algorithm using momentum orthogonalization with a few newton-steps. This work fills the gap between the existing theory on MUON with an exact SVD-based polar factor. My main concern is that MUON is not widely used, with mostly a few arXiv papers referenced for its use. 

Using standard analytic assumptions, tools and techniques, the authors provide complete convergence rates (dependence on the standard terms) for SGD with momentum, MUON with SVD, and MUON. Diverse numerical experiments on CIFAR-10 with modest sized CNN predominantly support their theory. Their results explain several contrasts between MUON w/SVD,MUON, and SGD w/momentum, and closes the theory-practice gap.

### Strengths
This paper is very clear and generally very well written.
The numerical results are extensive, but primarily in the appendix. I suggest moving several of the key plots from the appendix to the main text.

### Weaknesses
I did find many redundant statements of the main results throughout the paper (often with exact wording…); while I generally agree that repetition like this can enhance clarity of interpretation and impact, I found this manuscript to be excessive in that regard. The numerical results are extensive, but primarily in the appendix. I suggest moving several of the key plots from the appendix to the main text. Also, from what I saw, there is little additional information provided by the presentation of both Loss and Accuracy, so I suggest the authors pick one for the main text and move the other quantification to the Appendix. A bit more explanation of the details of the results presented in Figure 1 w.r.t. Learning dynamics across values of ‘q’ would be beneficial. A few sentences on mathematical definition and interpretation of the different norms used (Section 3.1) would be helpful for a broader audience. The definition of ‘r’on line 151 is very far removed from its first use on line 280; suggest moving. Remove variables from the abstract.

My

### Questions
Typos:
L53: ‘much’ -> ‘large’
L80: remove comma after ‘Shultz’
L84: ‘closer’-> ‘close’
L116: include ‘in “the” MUON…”
L293: ‘Decaying’ -> Decay

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates the convergence properties of the MUON algorithm, which demonstrates its superior properties against SGD or ADAM in terms of convergence speed. Other than existing works with an exact SVD orthogonalization of the momentum in its native matrix form, the paper studies the MUON with a finite steps of NEWTON–SCHULZ. The main results show that the MUON algorithm with a fixed finite steps of NEWTON–SCHULZ converges a stationary point with the same rate as the exact SVD-orthogonalization, up to a constant factor. The work further shows that this constant factor decays doubly exponentially in the fixed finite step of NEWTON–SCHULZ $q$ and improves with the degree of a polynomial used in NEWTON–SCHULZ $κ$.

### Strengths
The paper addresses a very demanding question about the convergence speed of the recently proposed MUON algorithm with a finite step of NEWTON–SCHULZ. As an approximate version of the MUON with the exact SVD orthogonalization of the momentum update, the presented analysis provides a solid theoretical understanding of the convergence behavior of the MUON in connection to its exact SVD counterpart, and ultimately guarantees the properties of deploying the MUON algorithm with a finite step of NEWTON–SCHULZ to large-scale training of DNNs, such as LLMs.

### Weaknesses
The core message conveyed in the paper was to show that the MUON algorithm with a finite step of NEWTON–SCHULZ converges similarly to some stationary point as its exact SVD counterpart in terms of convergence rate. However, one crucial aspect of training DNNs is the quality of stationary points. In other words, the discrepancy between stationary points generated by the MUONs with finite NEWTON–SCHULZ and the exact SVD can be large, thus is worth investigating.

### Questions
For a given setting of training a DNN, how close are the stationary points generated by the MUONs with finite NEWTON–SCHULZ and the exact SVD? If they are observed to be only satisfying the $\epsilon$ stationary point, are the qualities of these stationary points similar, in terms of generalizability to unseen data?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the convergence of Muon is analysed, for both the original and derivative versions. It is proved that Muon with Newton-Schulz converges with the same rate as SVD-polar idealization, and it can remove the square-root-of-rank loss compared to vector-based optimizers. The results explain the performance gain of Muon compared to vector-based optimizers, and are validated with experiments on the CIFAR-10 dataset and a CNN model.

### Strengths
This is one of the first convergence analysis result for Muon with finite steps of Newton-Schulz, which is a big step forward to narrow the gap between theory and practice.

### Weaknesses
The experimental part is relatively too simple. Experiments with larger and various datasets, larger number of parameters in the model, various model types such as MLP and transformer, more epochs, and more specific analysis how the results validate the theoretical analysis, will be nice.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
