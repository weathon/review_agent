# Separation Power of Equivariant Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
The separation power of a machine learning model refers to its ability to distinguish between different inputs and is often used as a proxy for its expressivity. Indeed, knowing the separation power of a family of models is a necessary condition to obtain fine-grained universality results. In this paper, we analyze the separation power of equivariant neural networks, such as convolutional and permutation-invariant networks.
We first present a complete characterization of inputs indistinguishable by models derived by a given architecture. From this results, we derive how separability is influenced by hyperparameters and architectural choices—such as activation functions, depth, hidden layer width, and representation types. Notably, all non-polynomial activations, including ReLU and sigmoid, are equivalent in expressivity and reach maximum separation power. Depth improves separation power up to a threshold, after which further increases have no effect. Adding invariant features to hidden representations does not impact separation power. Finally, block decomposition of hidden representations affects separability, with minimal components forming a hierarchy in separation power that provides a straightforward method for comparing the separation power of models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proves a number of results about the separation power of equivariant networks. In particular, it focusses on networks equivariant to regular / permutation representations of finite groups. The authors give a recursive formula for the set of pairs of inputs that are indistinguishable by any function in the function class under consideration. Further theorems clarify the role of non-linearities (they don't matter, provided they are non-polynomial), depth (increases separation power, but not indefinitely), multiplicity and invariant features (don't affect separability). Finally, the paper discussed implications for IGNs and CNNs.

### Strengths
+ The paper provides a detailed characterization of input separability for regular equivariant networks.
+ The theorems shed light on the factors that do and do not affect separability, which is quite insightful.
+ The theorems seem non-trivial and novel

### Weaknesses
- The paper is rather long-winded, with results starting only on page 7.
- The "twin network trick" is presented as a key insight, but it seems like a rather trivial observation to me. Eta(a) = eta(b) iff eta(a) - eta(b) = 0. 
- Not much intuition is provided, e.g. for theorem 1, which I found hard to understand or see why it must be true. Proofs are all in the appendix. It would be much better to move some background material to the appendix and explain the intuition behind the theorems and perhaps a proof sketch in the main paper.

### Questions
Typo: Line 162 has a typo; "ING" should be "IGN."

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work analyzes the separability of deep learning models with equivariant architectures (that is, architectures that are equivariant to specific group actions on the input and output spaces). Separability in this case means identifying pairs of points that take identical values under all networks of a fixed architecture. The work only considers separability of pairs of points. The paper makes a few assumptions on the network, the most notable being that the network uses a permutation representation (and hence the group $G$ is assumed to be finite). After setting up notation related to equivariant architectures, the work describes the “separation trick”, which essentially shifts perspective from looking for points $\alpha, \beta$ where $f(\alpha) = f(\beta)$ for all $f$ in the function class and instead looks at pairs of points $\alpha, \beta$ where $\eta(\alpha,\beta) = f(\alpha) – f(\beta) = 0$. That is, the zero locus of $\eta$. The paper proves a recursive formula about separability within this framework and then draws some conclusions about how various architecture choices impact separability (e.g., depth, width, etc.).

### Strengths
**Clarity:** The work is clearly written and relatively crisp. The Preliminaries section does a decent job introducing the basic tools needed in the paper (e.g., the notion of equivariance, etc.). An explanation of how the work fits into the current research landscape is provided. Despite being a theory paper, this work should be accessible to a wide range of readers.

**Problem importance:** While there is now a substantial body of work on equivariant architectures, both in terms of architectures for specific symmetry groups and applications, and theory, significant questions remain. Gaining a better mathematical understanding of what we sacrifice by imposing equivariance in our models is an interesting and important question that deserves more research.

### Weaknesses
**Theorem 1 is hard to read:** As this is one of the critical contributions of the work, it would be worth polishing it to make it more readily understandable, especially since the work takes pains in Section 4 to layout the framework for this result. It feels like in the process of trying to make the theorem more “informal” all the explanations were taken out but the mathematical notation was retained. For instance, what is the symbol “\leq” at line 377? The reviewer can make a guess, but it would be good to state this explicitly. Similarly, there a 6 different indices in (5). Is it possible to either break up this expression or build up to it? It is also a little hard to read since 4 of these indices ($h$, $k$, $\mathcal{Q}$, and $P$) don’t appear in the argument. This reviewer’s advise would be to cut down on some of the lengthy preliminaries and devote a bit more time to setting up and describing Theorem 1. It was hard for this reviewer to draw much insight from this Theorem.

**CNNs are equivariant to integer translation, $\mathbb{Z}^2$, not cyclic translation:** As far as this reviewer is aware, it is relatively uncommon to use circular padding in computer vision. For this reason, most CNNs are not equivariant to a product of finite cyclic groups, but rather to $\mathbb{Z}^2$. It would be worth stating that Example 4 does not represent the majority of CNNs.

**Hints at proof strategy:** While the reviewer appreciates that space constraints mean that the proofs need to be moved to the supplementary material, it would make the work more valuable if a high-level description of how the results were proven were included. After all, it is sometimes the case that the insights in the proof are more valuable than the theorems themselves. Again, the reviewer would advise shortening the preliminary material to provide more space. For instance, the preliminary material discusses group representations but (to this reviewer’s memory) the only place this later appears is a reference to the regular representation.

### Nitpicks:

-	Line 161: “From a practical viewpoint is fundamental to understand which are the hyperparameters …” $\mapsto$ “From a practical viewpoint it is fundamental to understand the hyperparameters…”

### Questions
- The reviewer was surprised that despite this being a result about equivariant architectures, group and representation-theoretic machinery featured very little (acknowledging that the reviewer did not go through the supplementary material closely). Is this the case or did the reviewer just miss something?
- Have the authors thought about separability of more than pairs of points? Can anything be said about this?

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
3

### Summary
This paper develops a theoretical framework for analyzing the separation power of equivariant neural networks, such as convolutional and permutation-invariant networks. The authors present a complete characterization of inputs that are indistinguishable by models derived by a given architecture, and derive how separability is influenced by hyperparameters and architectural choices (activation functions, depth, hidden layer width, and representation types). Key findings include: all non-polynomial activations achieve maximum separation power, depth improves separation up to a threshold, adding invariant features to hidden representations does not impact separation power, and and block decomposition of hidden representations affects separability.

### Strengths
The strengths of the paper are numerous. The authors provide a novel theoretical framework for analyzing separation power in equivariant networks, creatively combining group theory and functional analysis. Particularly interesting is the twin network trick which transforms a network separation problem into a zero locus problem for neural networks, allowing the application of recursive techniques for solving zero locus problems. The authors also provide new insights into how architectural choices affect the expressivity of a model – this is extremely valuable for the community.


The authors have provided clear motivations for the paper, rigorous mathematical proofs for all results, and clearly stated assumptions as well as limitations. Authors not only provide theorem statements but also give the reader intuitive understanding. The paper also does a good job of building from first principles which makes the read even more enjoyable!

### Weaknesses
The authors clearly state the limitations of the work. An obvious weakness is the computational complexity, but the goal of this paper is to build theoretical frameworks which is a very challenging task. It is understandable that the computational complexity is not practical. Though efficient computational approximates could be explored.

The paper could also benefit from experiments showing the effect (or lack thereof) of depth on separation power, as well as the impact of different activation functions. But again, the paper is a theoretical one and does not need to show experiments to be able to stand on its own legs.

It could be interesting to see the effect of initialization in separation power.

### Questions
Given the superpolynomial complexity of the recursive formula, are there special cases where computation becomes tractable?
What are the trade-offs between separation power and other desirable properties?
Is there a connection between degree of polynomial and separation power?
Have you explored connections between separation power and generalization bounds?
Are there cases where maximum separation power might be undesirable?

### Soundness
4

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
3

### Summary
This paper considers the separation power of equivariant networks. In the setting of permutation representations and element-wise nonlinearities, this paper charactierizes the set of identified points and then prove (1) all non-polynomial activations equivalently share the same and maximum separation power; (2) larger depth indicates better separation power; and (3) separation power is independent
of invariant hidden features.

### Strengths
* This paper is well-written and easy to follow.

* This paper considers an interesting and important problem in machine learning area and extends our understanding of separation power of specific equivariant architectures.

* This paper's definitions and statments are clear,  and its theoretical analysis is also solid.

### Weaknesses
* The theoretical results in paper are interesting but not so supervising, thus it seems unable to improve equivariant network design.

* The setting in this paper is somehow too specific (finite group, and feedforward architectures with element-wise activation if I understand correctly), which may limit its contribution.

More detail can be found in Questions.

### Questions
It is an interesting and solid paper to me. However, it may not have a significant contribution due to its limited setting. If I understand correctly, this paper considers only feedforward architectures with element-wise activations under finite groups. Moreover, many results seem not superising, which may weaken your contribution to help equivariant architecture design.

Feedforward networks with ReLU activations may only allow generalized permutation representations since equivariance properties require that the activation and group elements are commutative. If it is right and can be extened to general element-wise activations, it will enhance the soundness of your setting. What's more, the network with biases and input $x$ may be equivalent with that without baises and input $(x,1)$, which may further extend your results.

It may be not surprising to solve the identified set since this paper adopts element-wise activations and permutation representations. While all common activations seem to be monotonously increasing and bijective, the choice of activation would not be assumed to affect the separation power. Besides, the set $\rho(\mathcal{N})$ is assumed to have a lower bound. As a result, when we increase the depth to infinity, it is obvious that the identified set $\rho(\mathcal{N})$ would not change after a threshold of the depth. So a more explicit result would be more interesting. 

Therefore, I think this paper is interesting that may help improve our understanding to equivariant networks but may have relatively limited contributions.

### Soundness
3

### Presentation
3

### Contribution
2
