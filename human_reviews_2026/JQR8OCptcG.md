# Improving Set Function Approximation with Quasi-Arithmetic Neural Networks

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Sets represent a fundamental abstraction across many types of data. To handle the unordered nature of set-structured data, models such as DeepSets and PointNet rely on fixed, non-learnable pooling operations (e.g., sum or max) -- a design choice that can hinder the transferability of learned embeddings and limits model expressivity. More recently, learnable aggregation functions have been proposed as more expressive alternatives. In this work, we advance this line of research by introducing the Neuralized Kolmogorov Mean (NKM) -- a novel, trainable framework for learning a generalized measure of central tendency through an invertible neural function. We further propose quasi-arithmetic neural networks (QUANNs), which incorporate the NKM as a learnable aggregation function. We provide a theoretical analysis showing that, QUANNs are universal approximators for a broad class of common set-function decompositions and, thanks to their invertible neural components, learn more structured latent representations. Empirically, QUANNs outperform state-of-the-art baselines across diverse benchmarks, while learning embeddings that transfer effectively even to tasks that do not involve sets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposed a novel trainable aggregation function for sets, called Neuralized Kolmogorov Mean, and a quasi-arithmetic neural networks which includes the Neuralized Kolmogorov Mean in it.  The authors explore both unary and binary (similar to set attention) QANNs. The main contributions of the paper are:
1. A novel aggregation function for sets - the Neuralized Kolmogorov Mean.
2. Theoretical justification of its soundness, including proofs of permutation invariance and universal approximation properties.
3. The results show some consistent although sometimes slight benefit over other methods, with the results about transfer learning to and from classification tasks helping to support the author’s claims of better structured latent spaces.

### Strengths
The main strengths of the paper include:
- the idea is quite simple and easily explained, and the paper is clear with its intentions, methods and outcomes.
- a novel learnable aggregations function
- theoretical justifications

### Weaknesses
The mean weakness of the paper are:
- poor comparison against other learnable aggregation functions
- the majority of the experiments relies on semi-synthetic datasets and very simple datasets such as Omniglot and MNIST, so there is no evidence that the benefits observed would translate into a more meaningful task.
- the accuracy on ModelNet40 for DeepSets is quite low compared to the original paper (82% for 100 points, and 90% for 5,000 points), but the authors report ~65% (see Table 5).

All the weakness are expanded and further explained in the question section below.

### Questions
The methodological and novelty part of the paper is convincing, but unfortunately the experimental sections is somewhat weak. I would like to mention that there are many methods in the literature, and it would be unrealistic to expect any paper to include all possible competitors. However, it is needed to compare against the most relevant and closely related methods, such as [1] and [2].

**Methodology**.
1. As it stands, the paper seems to claim (see for e.g. the abstract) that the learnable function is entirely novel. This is inaccurate, as other methods such as PNA[1] and LAF[2] (among others) have already introduced learnable aggregation functions to address the problem of having fixed aggregation functions. Why did the authors not position their proposed aggregation function in relation to [1] and [2]?

**Experiments**. 
1. Comparisons with closely related works (e.g. [1], and [2]) are missing. Is there a justification as of why those methods were not included in the experimental evaluation?
2. The reported accuracies for DeepSets[3] are much lower compared to the original work. The paper reports ~65% (see Table 5), while the original paper achieved 82% for 100 points and 90% for 5,000 points. Why is there such a discrepancy? Also comparisons agains other methods are missing as previously mentioned. What is the rationale for their exclusion?
3. The authors mention a potential application for aggregations is in graph learning tasks. There is quite some literature on neural aggregation for graph level tasks (in the context of combing node representations after message passing) and use of e.g. set transformers in that context can give performance uplift. I see no reason why this method would not work in that context, and I would encourage the authors to investigate this aspect, at least briefly. The large number of practical tasks that have graph representations would help increase the potential impact of this work.

**Results**.
1. The MNIST-Sets experiments are quite related to those presented in [2]. Why was this comparison not included?
2. The results for the MNIST-Sets experiments (Table 5, "sum" task) are difficult to interpret relative to DeepSets, which reported an accuracy (not MSE) of over 60% for sets containing 50 images. It is unclear how MSE is as a meaningful metric for this task, since the aggregation output (a sum) is an integer. What would the accuracy be for QUANN-1 and QUANN-2 on the sum task?
3. It is not clear why ablation 2 performs so much worse than Ablation 1, when it is the same as Ablation1 but includes more parameters? Is there any explanation?

**Minor issues**.
1. Table 5 is too small and it should be broken perhaps in two parts. Experiments from the different subsections are summarized in the same Table which makes the reading a bit confusing.
2. Research questions section: does the proposed approach [learn] a.
3. Missing closing parenthesis at the end of the Sum decomposition section

[1] Corso, Gabriele, et al. "Principal neighbourhood aggregation for graph nets." Advances in neural information processing systems 33 (2020): 13260-13271.

[2] Pellegrini, Giovanni, et al. "Learning Aggregation Functions." IJCAI. International Joint Conferences on Artificial Intelligence Organization, 2021.

[3] Zaheer, Manzil, et al. "Deep sets." Advances in neural information processing systems 30 (2017).

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
4

### Summary
This paper presents Universal Set Transformer (UST), which combines a set transformer with mini-batch consistency (MBC), resulting in a methodology that is both expressive and has the ability to handle large sets. Set and elements can be represented, performance is often better than baselines, and explainability is possible by means of element-wise attention scores.

### Strengths
The formulation of element-wise representation into the multiset attention is theoretically interesting. The integration of this multiset attention into an MBC processing, is practically relevant. The experimental results are convincing, the performance is good across tasks, the stability w.r.t mini-batch size is good.

### Weaknesses
The results and discussion read as if there are only advantages of the proposed method. What are limitations, where does UST not perform well? There is a discussion section, but it does not reflect on the possible disadvantages of UST.

Multiset attention is here posed as a completely new thing. In "Unlocking Slot Attention by Changing Optimal Transport Costs" [1] there is also attention across the elements in the multiset. The reference is missing in the related work.

### Questions
What is the difference between the multiset attention as presented in this paper, compared to ref [1] (see above)? If this distinction is significant, and can be motivated in technical/math terms, then I am willing to increase my score, because it is a good paper but this remaining point needs clarification.

### Soundness
4

### Presentation
3

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
The paper presents a novel approach for learning the pooling operator in set function approximation. By leveraging a neuralized version of the Kolmogorov mean, it manages to approximate various measures of central tendency.

### Strengths
The paper is well written and clear. The choice of Kolmogorov mean and its neutralized version is well-motivated. There is a sound discussion of the theoretical advantage of the proposed solution over alternatives, complemented with promising experimental results.

### Weaknesses
There were few works that explicitly addressed the problem of learning pooling operators in the past:

- Euan Ong, Petar Veličković, Learnable Commutative Monoids for Graph Neural Networks, LOG 2022.
- P. Zuidberg Dos Martires, Neural Semirings, NeSy 2021.
- G Pellegrini, A Tibo, P Frasconi, A Passerini, M Jaeger, Learning aggregation functions, IJCAI 2021.

While none of them directly suggests using the Kolmogorov mean, they all attempt to go beyond predefined aggregators, and at least one (neural semiring, arguably a not very popular paper) explicitly mentions the use of invertible neural networks. Positioning the contribution with respect to these works would better clarify its novelty.

Countability assumption. This is not really a weakness (most existing results, starting from deep sets, assume countable sets), but given its relevance (sets of real vectors are uncountable) it would be worth discussing the assumption in the limitations.

Minor: 
- please check your citation reference fornatting and use citep when appropriate. 
- Pk(X) -> Sk(X) [or viceversa] in the description of Eq. 2

### Questions
Can you clarify the main differences wrt alternative approaches for learning pooling operators?

Can you discuss more clearly the implications of the countability assumption?

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
This paper introduces quasi-arithmetic neural networks (QUANNs) for learning set functions. By parameterizing the Kolmogorov mean function (also called the quasi-arithmetic mean, and hence the name of the proposed architecture), QUANNs can be seen as an extension of the Normalized DeepSets architecture, where the standard mean function in Normalized DeepSets is replaced with a learnable Kolmogorov mean. More generally, QUANNs further allow aggregation over subsets of set elements akin to the Janossy pooling. The authors provide explanation for the theoretical benefits QUANNs, as well as hypotheses for the practical advantage of QUANNs. The author verify their hypotheses by carrying out a set well designed experiments using both synthetic and real-world data. Empirically, QUANNs are shown to outperform state-of-the-art set neural network baselines.

### Strengths
The idea to parameterize the Kolmogorov mean function and replace the standard mean pooling in existing set neural network architectures is intuitive and interesting. Empirically, the resulting QUANN architecture outperforms all baselines. It is good to see that a simple and principled modification leads to effective performance gain.

The experiments are well designed to empirically verify the authors' hypotheses on the practical advantages of QUANNs.

### Weaknesses
Writing can be improved. 
- Section 5.2 has a lot of references to the theorems in the appendix. It is difficult to follow the discussion in this section without having to read the theorems in the appendix. I think it would be helpful to at least state some informal and short versions of the theorems here. 
- Please use citet{…} and citep{…} appropriately, e.g., use citep{…} for references that appear in lines 28-32. There are many misuses of citet{…} or cite{…} throughout the entire paper, please fix these. In Line 116, should define S_k(X) here, not P_k(X). Be consistent in the use of notations in Equation (2) and (4).


I am not sure how practical the new architecture is. The authors should discuss the practical relevancy of set representation learning in the current state of machine learning. Can't LLMs/vLLMs solve the tasks considered in the experiments?

### Questions
For the synthetic data experiment, Ablation 2 should in theory be strictly better than Ablation 1. Why do the results in Table 3 show the opposite? Was it because of bad training, bad tuning/regulations, or else?

Also for the synthetic data experiment, many of the functions involve computing some average in one way or another. This aligns with the normalization used in QUANN. What happens if you want to learn functions that are not averages, but sums (e.g. vector norms)?

### Soundness
3

### Presentation
3

### Contribution
3
