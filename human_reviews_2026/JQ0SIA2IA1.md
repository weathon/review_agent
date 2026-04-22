# Rigid invariant sliced Wasserstein via independent embeddings

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 8, 4

## Abstract
Comparing probability measures when their supports are related by an unknown rigid transformation is an important challenge in geometric data analysis, arising in shape matching and machine learning. Classical optimal transport (OT) distances, including Wasserstein and sliced Wasserstein, are sensitive to rotations and reflections, while Gromov-Wasserstein (GW) is invariant to isometries but computationally prohibitive for large datasets. We introduce Rigid-Invariant Sliced Wasserstein via Independent Embeddings (RISWIE), a scalable pseudometric that combines the invariance of NP-hard approaches with the efficiency of projection-based OT. RISWIE utilizes data-adaptive bases and matches optimal signed permutations along axes according to distributional similarity to achieve rigid invariance with near-linear complexity in the sample size. We prove bounds relating RISWIE to GW in the Gaussian case and empirically demonstrate dimension-independent statistical stability. Our experiments on cellular imaging and 3D human meshes demonstrate that RISWIE outperforms GW in clustering tasks and discriminative capability while significantly reducing runtime.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The classic optimal transport (OT) problem and its computationally efficient variant, the sliced OT version, are known to be non-invariant with respect to isometries in feature space. This article introduces a version of the sliced Wasserstein distance that is invariant with respect to rigid transformations, called RISWIE. RISWIE projects shapes onto intrinsic embedding axes (such as PCA or diffusion maps), computes pairwise 1D Wasserstein distances between projections, and finds the optimal signed permutation aligning the axes via the Hungarian algorithm. For embeddings that transform equivariantly under rigid motions, RISWIE recovers the true orthogonal transformation $R\in\mathcal{O}(d)$. The authors demonstrate the robustness and computational efficiency of RISWIE on 2D/3D point clouds datasets.

### Strengths
The RISWIE approach represents an interesting contribution to the OT and shape analysis fields. Its originality lies primarily in reformulating rigid invariance as a combinatorial axis-matching problem. This contrasts with previous isometric invariant OT problems, such as Gromov-Wasserstein and Wasserstein variants which directly minimize over all orthogonal transformations. The sliced nature of the problem ensures high computational efficiency, as demonstrated by the experimental results.

### Weaknesses
The article is not generally well written. Given the importance of the theoretical section, the writing and notation used are not always clear or completely rigorous. Especially in Section 3.1, which introduces the RISWIE approach and the algorithmic framework, some notations are missing, for example $P_\pi$ has never been defined. Overall, the presentation would benefit from careful completeness of notation.

A major conceptual and bibliographic gap concerns the absence of any reference, discussion and comparison with previous isometric invariant OT methods, the Procrustes-Wasserstein (PW) approaches, despite Gromov-Wasserstein (GW). By explicitly positioning RISWIE as an isometric invariant OT approach, the article demands a direct comparison with state-of-the-art techniques addressing the same problem. By writing _"While there has been work done to search over all point permutations and orthogonal transformations to make Wasserstein rigid-invariant, this formulation is NP-Hard."_, the authors jump over PW far too quickly. References like [1], [2], [3] should be considered as related works. Should also be included at least one experimental comparison against a PW implementation, which also represents a more computationally competitive alternative to GW.


[1] Alvarez-Melis, D., Jegelka, S., & Jaakkola, T. S. (2019, April). Towards optimal transport with global invariances. In The 22nd International Conference on Artificial Intelligence and Statistics(pp. 1870-1879). PMLR. \
[2] Even, M., Ganassali, L., Maier, J., & Massoulié, L. (2024). Aligning embeddings and geometric random graphs: Informational results and computational approaches for the Procrustes-Wasserstein problem. Advances in Neural Information Processing Systems, 37, 70730-70764. \
[3] Adamo, D., Corneli, M., Vuillien, M., & Vila, E. (2025). An in depth look at the Procrustes-Wasserstein distance: properties and barycenters. arXiv preprint arXiv:2507.00894.

### Questions
1. As stated above, the paper currently do not compare with PW approaches, even though the geometric problem addressed is the same.
A direct experimental comparison with at least one PW baseline would significantly increase the completeness. 
2. How sensitive is RISWIE to the choice of embedding (PCA vs diffusion maps) and to small perturbations of data (noise, near-degenerate eigenvalues)?
3. Theorem 1 states that RISWIE is invariant to rigid transformations $T(x)=Rx+t$ with $R\in\mathcal{O}(d)$. However, from the algorithmic formulation, RISWIE only searches over signed permutations of embedding axes, which represent a discrete subgroup of $\mathcal{O}(d)$. 
Could the authors clarify this point explicitly? Is the claim of invariance meant to hold for all orthogonal transformations $R\in\mathcal{O}(d)$, or only for those that act as signed permutations within the embedding space?
4. The output of Algorithm 1 should be just $D(X,Y)$ (or $D(X,Y)=D(Y,X)$)

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
4

### Summary
This work proposes Rigid-Invariant Sliced Wasserstein via Independent Embeddings (RISWIE) as an efficient dissimilarity measure, which, like the Gromov-Wasserstein (GW) distance, is invariant under affine transformations. The effectiveness of the methodology is demonstrated through several experiments, evaluating the obtained alignment, as well as its discriminative, clustering, and classification performance, in comparison to existing distances. In addition, its computational efficiency is assessed. On the theoretical side, statistical properties of the proposed method are provided, along with an analysis in the particular case of Gaussian distributions.

### Strengths
- The method is simple while clever. The general dissimilary between probability measures (Definition 2) is stated in great generality, and then two known prototypes of the embeddings, such as PCA and diffusion maps, as prototypes are utilize for demostrating the core properties of the tool. 
- The paper is well organized, and its contributions are sound.
- Different types of experiments are performed.

### Weaknesses
- Comparisons with Sliced Gromov-Wasserstein (SGW) and Rotation-Invariant SGW (RISGW) by Vayer et al. are not presented.

- In the experiments, only RISWIE with PCA embeddings is utilized, while other embedding methods such as diffusion maps are not considered.

- In the Statistical Properties section, the paper would benefit from briefly outlining the main steps of the derivations, particularly, in the second paragraph, where the authors discuss the advantage of the proposed distance with respect to the curse of dimensionality. Moreover, in the first paragraph, it would be helpful to emphasize that, in Theorem 7, the only assumption on the embeddings is that $\phi_i,\psi_j$  are bounded and measurable (if understood correctly).

- The paper would also benefit from including a brief sketch of the proof of Theorem 1 in the main text, similar to what is done for Theorem 2.

### Questions
- Could the authors elaborate on the usefulness of considering a sign in the permutations? Is this mainly relevant when PCA is used, since in that case the methodology also recovers a signed axis permutation for the alignments, interpretable at the level of eigenspaces?

- For my understanding: the terminology “slice” is used because pushforwards by real-valued functions are considered, but not in the standard sense of the Sliced Wasserstein distance, where projections onto different directions are used. Please confirm or clarify this interpretation. If the embeddings  $\phi$ and $\psi$ are chosen as projections onto directions, then the formulation would effectively consider only two directions.

- What happens in the case where $d_1=d_2=k$, and the embeddings $\phi$ and $\psi$ are taken to be the identity? To which dissimilarity does the definition of $D$ then reduce?

- Theorem 7: does the PCA embedding satisfy the required boundedness condition?

- In Theorem 4, upper bounds for the new distance are provided in terms of comparisons with GW. Could the authors comment on the existence or derivation of possible lower bounds?

Typos and style:
- Line 241: “transformatio” → “transformation.”
- Appendix, Section A.3, line 700: appears as “Table ??” — please correct the reference.
- Appendix, line 1748: consider renaming Remark 2 as a Proposition or Lemma, since a proof is included. It is unusual to separate proofs from remarks in this way.

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
3

### Summary
The authors propose a method to compare probability measures, $\mu$ and $\nu$, while being invariant to rigid transformations (rotations, reflections) and remaining computationally scalable. This problem is central to shape matching, where Gromov–Wasserstein (GW) is invariant but NP-hard, and Sliced Wasserstein (SW) is scalable but not invariant. RISWIE operates in 3 stages:
  
First, it computes a $k$-dimensional, data-dependent coordinate system for each measure *independently* (e.g., via PCA or Diffusion Maps), creating $k$ one-dimensional marginal distributions for $\mu$ and $k$ for $\nu$.  

Second, it solves an optimal assignment problem over the signed permutation group $\mathcal{O}^{\pm}_k$ to find the best way to pair and/or reflect the $k$ marginals of $\mu$ against the $k$ marginals of $\nu$.  

Third, the final distance is the aggregated $W_2$ cost of these $k$ optimal one-dimensional pairings.

The authors claim this decouples the hard invariance problem (a discrete $O(k^3)$ assignment) from the transport cost (fast 1D $W_2$), resulting in a rigid-invariant pseudometric that is significantly faster than GW and has superior statistical properties to $W_2$.

### Strengths
- The method cleverly reframes the invariant comparison problem not as a single, complex optimization (like GW), but as two independent, simpler steps:  find a "canonical" basis for each shape and then find the cheapest way to align these bases.  

- The method is more scalable than GW. The empirical results in Figure 2 and Table 2 clearly show that GW is computationally intractable for $n > 10^4$ samples, whereas RISWIE scales manageably. For applications where GW is the required invariant baseline, this method offers a viable, faster alternative.

### Weaknesses
- The central claim of rigid invariance (theorem 1) is not robust. It depends on the chosen embedding (e.g., PCA) returning a unique, ordered set of eigenvectors (i.e., distinct eigenvalues) that are perfectly equivariant to rotation. For any object with symmetries (e.g., a sphere, a cube) or in the presence of noise, eigenvalues can be degenerate ($\lambda_i \approx \lambda_j$). In this common scenario, the eigenvector basis is unstable and not unique, meaning the invariance property breaks. Although the paper acknowledges this in the discussion, this is not a minor limitation.

- The method is only a pseudometric.

- The paper claims to “match [...] the efficiency of Sliced Wasserstein.” However, Figure 2 shows that RISWIE–PCA is consistently slower in wall-clock time than SW, though the scaling trend is similar. The added $O(nd^2)$ cost from PCA dominates, making the method asymptotically worse than SW’s $O(Ln \log n)$ when $d$ is large and $L$ is moderate. Thus, the claim of “near-linear cost” holds only with respect to $n$, assuming $d$ is fixed.

### Questions
- Under what concrete spectral or distributional conditions does $D(x,y)=0$ guarantee rigid equivalence?

### Soundness
2

### Presentation
2

### Contribution
2
