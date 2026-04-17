# Identifiability Challenges in Sparse Linear Ordinary Differential Equations

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Dynamical systems modeling is a core pillar of scientific inquiry across natural and life sciences. Increasingly, dynamical system models are learned from data, rendering identifiability a paramount concept. For systems that are not identifiable from data, no guarantees can be given about their behavior under new conditions and inputs, or about possible control mechanisms to steer the system. It is known in the community that "linear ordinary differential equations (ODE) are almost surely identifiable from a single trajectory." However, this only holds for dense matrices. The sparse regime remains underexplored, despite its practical relevance with sparsity arising naturally in many biological, social, and physical systems.
 In this work, we address this gap by characterizing the identifiability of sparse linear ODEs. Contrary to the dense case, we show that sparse systems are unidentifiable with a positive probability in practically relevant sparsity regimes and provide lower bounds for this probability. We further study empirically how this theoretical unidentifiability manifests in state-of-the-art methods to estimate linear ODEs from data. Our results corroborate that sparse systems are also practically unidentifiable. Theoretical limitations are not resolved through inductive biases or optimization dynamics. Our findings call for rethinking what can be expected from data-driven dynamical system modeling and allows for quantitative assessments of how much to trust a learned linear ODE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits the identifiability of linear ODE systems $dx(t)/dt = Ax(t)$ when the system matrix $A$ is sparse. The authors show that, unlike the dense case where almost all systems are identifiable from a single trajectory, sparsity introduces a positive probability of unidentifiability. They define a sparse–continuous ensemble and prove a sharp phase transition: when sparsity exceeds roughly $1-ln n /n$, systems become globally unidentifiable with high probability. The paper also introduces a trajectory-level metric to quantify how close a trajectory is to being unidentifiable. Simulations confirm that identifiability deteriorates with higher sparsity, both theoretically and in practice, using SINDy and Neural ODE estimators

### Strengths
1. Provides a clear theoretical characterization of when sparse linear ODEs lose identifiability.
2. Decomposing failure probability into global vs. trajectory unidentifiability (Eq. 2) is conceptually clarifying, and the distance $d_A(x_0)$ is a helpful practical proxy.
3. The sharp threshold result is elegant and connects to known results in random matrix theory.
4. Writing is well structured, and assumptions are transparently stated.

### Weaknesses
1. The strong assumptions (noise-free, single continuous trajectory, full observability) limit real-world applicability.
2. For readers less familiar with random matrix theory, it would be helpful to include a brief intuition box or paragraph below Lemma 3 explaining why the identifiability transition occurs precisely at $1-ln n/n$. A short, high-level explanation would make this elegant result more accessible and highlight its connection to classic random graph thresholds.
3. In the main text, the normalized Hamming distance is described as divided by $n$, but Appendix C.2 defines it as normalized by $n^2$. Please clarify which version was used, and make it consistent.

### Questions
1. In Fig. 4, for small dimensions (e.g., $n=3,5$) SINDy sometimes achieves low normalized Hamming distance even at very high sparsity. Could you comment on this regime?
2. In Section 2 (“Discussion of assumptions and limitations”), you might consider citing a related work on identifiability under hidden confounders: Wang et al. (2024) Identifiability analysis of linear ODE systems with hidden confounders.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a theoretical study of the identifiability of autonomous, linear and noise-free ODEs from a single trajectory, focusing on systems where the drift matrix is *sparse*. The authors distinguish between two notions of unidentifiability: (i) system-level (or global) unidentifiability, where the system is unidentifiable regardless of the initial condition, and (ii) trajectory-level unidentifiability, which occurs only for initial states lying in invariant subspaces of the system. 

*For system-level unidentifiability*, they show that in sparse systems it is implied by rank deficiency (i.e., zero-eigenvalue degeneracy, Lemma 1), derive a lower bound on its probability via the occurrence of system matrices with two zero columns (Lemma 2), and extend the analysis (Lemma 3) using random graph theory to identify a dimensionality-dependent sparsity threshold governing unidentifiability. 

*For trajectory-level unidentifiability*, they prove that the probability of “unlucky” initial conditions lying in invariant subspaces is zero, so identifiability depends solely on the system-level property. They further analyze near-invariant initial conditions, introducing a notion of distance to invariant subspaces, and showing that the indistinguishability time between trajectories of two systems agreeing on such a subspace increases inversely with this distance (Lemma 4). 

Theoretical findings are finally supported by numerical experiments, which rely on neural network (Neural ODE) and symbolic regression (SINDy) methods.

### Strengths
1. The paper provides strong context through a clear discussion of related work, assumptions, and limitations. This situates the contribution well within the system identification and machine learning communities and makes its relevance easy to grasp.
2. The work extends classical results on the unidentifiability of linear systems to the practically important case of sparse systems, filling a notable gap in the existing theory.
3. The proofs are well structured and technically interesting. 
4. The experimental section is well designed and directly supports the theoretical claims. The authors also apply two widely used system-identification methods (Neural ODE and SINDy) to test the practical implications of their results. The exposition of these empirical findings is clear and coherent (albeit limited, see Weekness 2 below).
5. The appendix is comprehensive, providing detailed proofs, additional empirical results, and clear information for reproducibility.

Overall, the paper is well written, technically sound, and contributes meaningful theoretical insight into the identifiability of sparse linear systems.

### Weaknesses
1. The most immediate limitation lies in the restriction to linear and noise-free systems. While the authors are transparent about this assumption, it considerably narrows the scope of applicability to real-world or data-driven settings.

2. The results on empirical unidentifiability (Section 5.3) are interesting but somewhat limited. The authors do not explicitly connect their theoretical findings on trajectory-level unidentifiability with their empirical unidentifiability results. For example, it would be interesting to examine how the two system-identification methods considered behave for initial conditions close to invariant subspaces, especially since the trajectory length should influence distinguishability in that regime (as suggested by Lemma 4).

3. While the use of a Bernoulli mask to model sparsity is theoretically convenient, the authors neither discuss nor acknowledge (or at least I don't find such a discussion/acknowledgement) that real-world sparse networks often exhibit "structured sparsity" (e.g., hub nodes or community structure). The Bernoulli mask (i.e. Erdős–Rényi) model does not capture such degree statistics, which may limit the generality of the conclusions.

### Questions
1. Why did you title Section 3 “Global Unidentifiability” instead of “System-level Unidentifiability”?
Put differently, why introduce the term system-level unidentifiability later in Section 3 rather than consistently using global unidentifiability throughout? This shift in terminology is somewhat confusing.

2. In Appendix B, you discuss the sparsity of real-world gene regulatory networks.
What are the node degree statistics of these networks, and can they be reproduced by your sparse–continuous random matrix model? If not, how might structured or heavy-tailed degree distributions affect your theoretical results?

3. Within your setup, is there a way to study how the closeness of an initial condition to an invariant subspace influences the performance of Neural ODE and SINDy? In particular, does the length of the observed trajectory or the inter-observation interval play a significant role in this regime, as suggested by Lemma 4?

4. In Figure 4, SINDy appears to match the true sparsity pattern almost perfectly for the 3-dimensional case across all sparsity levels.
How should these empirical findings be interpreted in light of your sharp sparsity threshold for global unidentifiability, and the corresponding empirical verification in Fig. 2? In other words, why does SINDy succeed in this low-dimensional setting even in regimes predicted to be unidentifiable?

### Soundness
3

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
3

### Summary
The authors prove an identifiability result for a class of linear dynamics governed by a sparse-continuous random matrix. I went through the proofs in detail and they are reasonable, although this is outside my personal area of research so I can just say that the techniques and arguments seem reasonable. My only feedback is whether the authors can better motivate this class of sparse-continuous matrices and in what way they are relevant to machine learning. Of course identifying dynamics from time series is of interest, but a bit more discussion about why one should care about this is necessary.

### Strengths
Seems like a correct theoretical paper.

### Weaknesses
See  my response to the summary - the relevance of this class of matrices needs to be better motivated. I couldn't tell whether this is an impactful result and the authors chose this problem because its important, or if the authors just chose a class of matrices for which they knew how to prove something. if they can help me understand that better i will raise the score.

### Questions
no further questions

### Soundness
3

### Presentation
3

### Contribution
3
