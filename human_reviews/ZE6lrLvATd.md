# Improving Equivariant Networks with Probabilistic Symmetry Breaking

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Equivariance encodes known symmetries into neural networks, often enhancing generalization. However, equivariant networks cannot *break* symmetries: the output of an equivariant network must, by definition, have at least the same self-symmetries as its input. This poses an important problem, both (1) for prediction tasks on domains where self-symmetries are common, and (2) for generative models, which must break symmetries in order to reconstruct from highly symmetric latent spaces. This fundamental limitation can in fact be addressed by considering *equivariant conditional distributions*, instead of equivariant functions. We therefore present novel theoretical results that establish necessary and sufficient conditions for representing such distributions. Concretely, this representation provides a practical framework for breaking symmetries in any equivariant network via randomized canonicalization. Our method, SymPE (Symmetry-breaking Positional Encodings), admits a simple interpretation in terms of positional encodings. This approach expands the representational power of equivariant networks while retaining the inductive bias of symmetry, which we justify through generalization bounds. Experimental results demonstrate that SymPE significantly improves performance of group-equivariant and graph neural networks across diffusion models for graphs, graph autoencoders, and lattice spin system modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a method for symmetry breaking in equivariant models, particularly focusing on equivariant GNNs. The motivation and relevance of this work are clear. However, while the idea and motivation are compelling, the paper itself is disorganized, unclear, and lacks important details. As it stands, I cannot recommend its acceptance, and significant revisions will be necessary for it to be considered.

### Strengths
The paper presents a solid theoretical foundation on the subject and offers a well-motivated experiments section. The proposed method is supported by both theoretical insights and detailed experimental results.

### Weaknesses
The presentation of the paper is very poor. From the background through to the experiments, the notations and mathematical explanations are unclear throughout.

### Questions
The overall structure of the paper is very poor, and I will highlight the most critical issues in each section.

**Contribution**
In the contribution section, you need to clearly state your achievements. However, you provided an outline of the paper’s structure instead, which is typically referred to as an arrangement.

**Background and main text**
1. Are your distributions discrete or continuous? This distinction needs to be clearly stated, as it drastically changes the mathematics. Since you condition on $X=x$, it suggests discreteness, yet many of your sets appear infinite.
2. What are the input and output random variables, X and Y? What are their input and output spaces? This is very unclear.
3. There are no meaningful references to figures in the paper.
4. The figure legends are unclear.
5. In Equation 2, line 137, Y is a random variable, but your function f maps elements into $Y$. How does that make sense? Also, it’s unclear whether the probability is over X or over epsilon.
6. In Theorem 3.4, equation 3, it’s not clear what the probability is being taken over.
7. In the paragraph beginning at line 235, how do you compute this argmin efficiently over infinite groups or over the set of permutations?
8. In my opinion, Section 5, which extends the discussion to noise, should be moved to the appendix, while expanding your existing work in the main body would be more valuable.
9. Why is Theorem 6.1 necessary?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the inability of equivariant neural networks to break symmetries in their outputs, which is a limitation for certain tasks. The authors present a theoretical framework for representing equivariant conditional distributions, allowing symmetry breaking via randomized canonicalization. They propose SymPE (Symmetry-breaking Positional Encodings), a method that expands the representational power of equivariant networks while retaining symmetry inductive bias. Experimental results demonstrate that SymPE significantly improves performance across various applications, including diffusion models for graphs, graph autoencoders, and lattice spin system modeling.

### Strengths
1. The paper considers an important but challenging task.
2. The authors provide SymPE, a method motivated by both theoretical and empirical studies.
3. Even though the paper has a theoretical contribution, it is not difficult to read.

### Weaknesses
Actually, the paper is solid and interesting. The key limitation admitted by the authors is that the proposed method requires access to a canonicalization method, and more general theoretical results are also desirable.

### Questions
I have no questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes to sample a group element probabilistically to break the self-symmetry of an input to an equivariant network. This allows more flexible model outputs with lower symmetry, which is crucial for prediction and generation tasks with self-symmetries. The group element is sampled from an inversion kernel, which can be thought of as shifting the uniform distribution on the stabilizer of a data point $x$ by a learned canonicalization element $g = \gamma(x)$. The network then takes the randomly sampled $g$ as input in addition to $x$ to break symmetry.

### Strengths
The main idea of this paper is to take a randomly sampled group element as an additional input to the equivariant network to break the self-symmetry of an input. The proposed method is straightforward yet effective, tested on various tasks including graph reconstruction and generation, and the ground-state prediction of Ising models. The authors provide theoretical justifications for the method, explaining the motivation of key design clearly. There are also very helpful discussions on alternative methods to break symmetry, e.g. injecting totally random noise instead of that from a lower entropy.

### Weaknesses
* The analog to the positional encoding in Transformers is a bit confusing to me. In the setup of this paper, we want to break the **self-symmetry** of the data $x$, i.e. $gx=x$. However, in terms of sequence modeling, there is no self-symmetry in the data, i.e. if we permute the words in a sentence it would no longer be the same sentence. It is the symmetry of the **model** that we want to break. While you could still argue that your method also breaks the symmetry of the model in the sense that $f(gx) = gf(x)$ is not always true, in essence your method is still equivariant if you consider the joint action on the sampled group element. Also, the word "position" actually has different meaning in these two contexts. The positions in the Transformer PE refer to the indices of words in a sequence, i.e. **in** a single data point. On the other hand, in your method, the positions refer to the group element sampled from the inversion kernel, i.e. the hypothetical position **of** a data point on the group action orbit. I feel this is quite different from its literal meaning in the Transformer PE, and I don't feel this analog provides a good intuition.
* Regarding the encoding of the group element, I wonder if it's better to define things in the language of group representations, since it's already based on the assumption that $G$ acts freely on $v$ by matrix multiplication. Also, I feel the explanation in the text is good enough and Prop 4.1 does not provide any new information.
* Consider changing some notations, e.g. in L275 $e \leftarrow \tilde g v$, $e$ may be confusing because it can also refer to the identity element of the group.
* The experiment section could benefit from more clarifications. See questions.

### Questions
* Eq (1): should it be this way or $P(Y | X = g^{-1}x) = P(gY | X = x)$?
* Why are the results in Table 1 different from those in Satorras et al (Figure 5)? From my understanding, the "No SB" row in your table refers to an equivariant GNN without symmetry breaking and should match the EGNN performance reported by Satorras et al. If there are differences in settings that cause the discrepancy, they should be highlighted.
* In Table 2, apart from NLL, other metrics are more similar across different models. How do you interpret this? Also, DiGress + SymPE (nodes only) has better validity vol. stability scores but is not highlighted. Is this intentional or a mistake?
* L500: can you specify the action of $g \in p4m$ on $(J, h, \sigma)$?
* L508: What does it mean by "Hamiltonian parameters are randomly rotated by 90 deg"?
* Does your method have an advantage over G-CNN w/o symmetry breaking in predicting ground-states of Ising models partly because the interaction $J$ and the transverse field $h$ are initialized with self-symmetry? Would the results be different (maybe better for the baseline) if $J$ and $h$ are not constant on the lattice?
* L1259: "symmetry breaking elements are computed through a deterministic canonicalization". If the group elements are deterministic, how do you break symmetry?

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
The paper introduces a new method for breaking symmetries in equivariant neural networks through the probabilistic symmetries concept and randomized canonicalization. The proposed approach is implemented as a sampling from a canonicalization network, with additional positional encoding (SymPE), along with the equivariant network.

### Strengths
The paper shows a theoretical motivation for the proposed method and extends previous works on equivariant distribution to tackle the symmetry breaking case. It also conducts evaluations on three tasks: graph autoencoder (using EGNN), graph generation (on QM9 using DiGress), and ground-states of the Ising model (using G-CNN), with improvement over the equivariant baselines.

### Weaknesses
The limitation of the method, as already mentioned, is that it requires sampling from a canonicalization function in addition to using equivariant networks, which might lead to increased complexity and additional computations.

### Questions
There are some works on positional encodings for molecules beyond Laplacian, for example (1). Can you comment on that and how it might perform compared to SymPE?  

1. Dwivedi, Vijay Prakash, Luu, Anh Tuan, Laurent, Thomas, Bengio, Yoshua, and Bresson, Xavier. "Graph Neural Networks with Learnable Structural and Positional Representations." 2022

### Soundness
3

### Presentation
3

### Contribution
3
