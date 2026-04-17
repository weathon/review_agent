# Some Neural Networks Inherently Preserve Subspace Clustering Structure

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
It has long been conjectured and empirically observed that neural networks tend to preserve clustering structure. This paper formalizes this conjecture. Specifically, we establish precise conditions for cluster structure preservation and derive bounds to quantify its extent. Through this analysis we are able to show that certain neural networks are learning parameters that preserve the clustering structure of the original data in their embeddings, without the need to impose mechanisms to promote this behavior. Extensive numerical analysis and experiments validate our results. Our findings offer deeper insight into neural network behavior, explaining why certain data types (such as images, audio, and text) benefit more from deep learning. Beyond theory, our findings guide better initialization, feature encoding, and regularization strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper theoretically investigates the ability of certain neural networks to preserve subspace clustering structures and formalizes this conjecture using empirical research.

### Strengths
1. The paper offers a novel perspective, connecting the empirical success of neural networks to the classical theory of subspace clustering.
2. The writing is generally clear. The proofs are well-structured and convey the core ideas.
3. The experimental design is comprehensive, progressing from synthetic data to diverse real-world datasets, supporting the arguments.
4. It provides a viable explanation for the effectiveness of deep learning on data with inherent clustering structure.

### Weaknesses
1. The theoretical extension from single-layer to multi-layer networks is overly simplified. Section 6 claims to extend the results to multi-layer networks, but this is not fully elaborated.
2. The paper states that CNNs/Transformers achieve high accuracy without maintaining the original structure. It is worth exploring in depth the circumstances under which maintaining the original structure is preferable to reconstructing a new one.
3. The paper does not clearly specify the specific loss function and architectural details used for training the neural network.

### Questions
1. What specific training objectives did you use in the deep network, LSTM, and real-data experiments in Sections 6, 7, and 8? Was it autoencoders and mean squared error loss, or something else?
2. For the multi-layer extension in Section 6, can it provide a rigorous mathematical formulation or derivation of how Theorem 3.1 applies to $L$ layers via the joint bound? What is the resulting total error bound, and what requirements does it impose on the error $\epsilon_\ell$ for each layer?
3. The theory assumes additive noise Z. Has the conclusion been explored for more complex, non-additive, or data-dependent noise?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper attempts to develop theoretical understanding on the observation of some neural networks preserving clustering structure, in which precise conditions for clustering structure preserving is provided. Some theoretical results are developed to bound the gap beween the projected differences between the clean data, transformed data and the output data after one-layer transform. Moreover, numerical analysis and empirical evaluation on real world data are shown, with some explaination and discussions.

### Strengths
+ It is appealing to investigate the observation of clustering structure preserving with some theoretical justification.

### Weaknesses
1. The mathematical analysis in the paper is confusing. Symbols used in proofs are not clearly defined or without any interpretation. For example, the norm of the matrix, $\| Z\|$ and $\| Z \|_\infty$ are not clearly defined, $\sin (\Theta)$ is not clearly defined. Even worse, the projection $P$ and $W$ are also not clear.   

2. The derivations in the proofs are incorrect. 
- For example, in the proof of Lemma 4.1, the third equality seems incorrect. How to build the equality from a trace of the two projected matrix to the squared Frobenius norm of them? 
- In the proof of Lemma 4.2, the reviewer cannot see how "plugging this and (5) in (6) we obtain the lemma." The reviewer cannot obtain the result in Lemma 4.2. The proof for Lemma 4.2 are wrong.  Moreover, for the matrix $W$, it is weight matrix, without any implicit assumption on the rank (if not specially claimed). 

3. Regarding to Theorem: It is not clear how the condition of $\delta_1$ which is defined as the singular value gap of $X$ and $X^\ast$ is actually used. 

4. The empirical evaluations are insufficient and incomplete. How the neural networks are trained actually? Which loss function is used? Are the empirical results obtained from the synthetic data generalizable in a broad sense, or it is just because of the special structure of the synthetic data? The way to generate the synthetic data is to form a set of nearly orthogonal subspaces. 

5. Is the initialization of $W$ reasonable? What happens if $W$ is initialized with an iid Gaussian of zero mean and variance of $1/m$? From the viewpoint of the reviewer, it is more natural and making sense, or even with nice property.  Why initialize $W$ with i.i.d. uniform entries in the range of $(- \sqrt(m), \sqrt(m))$? 

6. The worse results in Fig.2 (right panel) are not clearly interpreted.  In Fig. 5, why CNN and Transformers are failed to preserve the clustering structure? How the clustering accuracy is obtained in Fig.4?  

7. Minor issues: 
- The format for the citation in some places seems not properly used.
- The first three paragraphs of the introduction read like generated automatically via a LLM. Some contents are not factual.

### Questions
Please read the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a theoretical analysis of when neural networks preserve clustering structure under a union of subspaces (UoS) model. It derives conditions ensuring that a network’s learned representations retain the subspace structure of clean data, even when inputs are noisy. Through a perturbation analysis, the authors show that the row space of the learned representation closely matches that of the original data. Experiments support the theory, revealing that certain networks naturally maintain clustering structure, offering insights into initialization, feature encoding, and regularization.

### Strengths
- The paper fills a theoretical gap by analysing the clustering behaviour of neural networks using subspace clustering and perturbation theory, offering a proof for a long-standing empirical intuition.
- Synthetic experiments demonstrate that gradient descent naturally enforces the theoretical condition without explicit regularisation.
- Provides insights for neural network design decisions, like initialisation

### Weaknesses
- The UoS model and large spectral-gap condition may not hold in realistic settings
- The paper relies only on projection-distance measures; evaluating standard clustering metrics (e.g., NMI, ARI) could show how clustering performance is linked to the network architecture and behaviour of projection-distances. 
- The paper only briefly discusses that breaking subspace clustering structure, as seen in CNNs and Transformers, may actually contribute to their strong empirical performance. I found this insight actually very interesting, as these architectures are more commonly used in practice than MLPs.

### Questions
- How is the final clustering performance measured w.r.t. to the ground truth (e.g., with ACC, NMI or ARI) of a network linked to its preservation of subspace clustering structure? I imagine that clustering performance of CNN's or Transformers would generally be higher than the performance of MLPs
- In the paper you mention: "The large projection distances exhibited by Transformers and CNNs show that these networks are clustering through some mechanism other than the closed-form solution, and such mechanism does not preserve the original clustering structure." Could you elaborate on this? This is a very interesting insight, and I would be curious to see a more in depth discussion in the paper on this point.

### Soundness
2

### Presentation
3

### Contribution
3
