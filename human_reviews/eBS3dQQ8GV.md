# Emergence of meta-stable clustering in mean-field transformer models

- Decision: Accept (Oral)
- Scores: 8, 5, 10, 8, 8

## Abstract
We model the evolution of tokens within a deep stack of Transformer layers as a continuous-time flow on the unit sphere, governed by a mean-field interacting particle system, building on the framework introduced in Geshkovski et al. (2023). Studying the corresponding mean-field Partial Differential Equation (PDE), which can be interpreted as a Wasserstein gradient flow, in this paper we provide a mathematical investigation of the long-term behavior of this system, with a particular focus on the emergence and persistence of meta-stable phases and clustering phenomena, key elements in applications like next-token prediction. More specifically, we perform a perturbative analysis of the mean-field PDE around the iid uniform initialization and prove that, in the limit of large number of tokens, the model remains close to a meta-stable manifold of solutions with a given structure (e.g., periodicity). Further, the structure characterizing the meta-stable manifold is explicitly identified, as a function of the inverse temperature parameter of the model, by the index maximizing a certain rescaling of Gegenbauer polynomials.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors consider the evolution of the token distribution in a limit of an infinitely deep transformer which yields a differential equation for the evolution of the tokens. This equation is further simplified by the assumption that the attention parameters are constant over layers, and all equal to the identity, and that there are no MLP subunits. An initial result is that they show that the limit of infinite particles commutes with the time evolution in the sense that there is a PDE that describes the evolution of an empirical distribution with $N$ large enough. From this starting point, they describe a departure from these solutions (initialized so that the limiting measure is the uniform measure)

1) an initial phase where a periodic fluctuation of a certain frequency, $k$, grows 
2) a quasi-linear phase where the fluctuations are well-described by the behavior of this mode around the uniform background
3) and finally a collapse phase where the evolution reaches close to a finite set of delta measures.

### Strengths
The strengths of this paper are its rigorous analysis of the problem, where they first extract and then rigorously prove that certain clustering dynamics arise directly from attention. This work makes it very clear what structures arise and how they evolve. This is a strong extension of prior work, which relies on less intuitive arguments, and so this paper finds a way to explain a complicated phenomenon more clearly than prior work.

### Weaknesses
The main weakness is the lack of connection to a practical setting. The problem considered is highly simplified, and there is little discussion concerning even qualitative arguments explaining what differs in the setting with the greatest interest -- deep transformers with MLP blocks and non-identity weight matrices. It is difficult to understand how to take these results and understand what they say about that setting. 

For example:
1. Prior work [1] suggests that the structure of the value, key, and query matrices is important and derives results in various (though simplified) cases
2. The work cited as the foundation for this work [2] discusses the BBGKY hierarchy -- a way to understand the joint law of all the particles by breaking them down in terms of the $n^{\text{th}}$ degree marginals.
3. Other work [3] considers the behavior, including the MLP, and finds (though non-rigorously) that the MLP significantly affects dynamics.

It is of significant interest if connections to ideas like these could be explained in the paper. Of course, the limited scope of one paper may hinder detailed analysis in all possible settings, but it would make it clear how to interpret the results in context.

If the authors could discuss this issue satisfactorily, I would be happy to increase my score for this paper.


**References:**

1) Geshkovski, Borjan, et al. "The emergence of clusters in self-attention dynamics." Advances in Neural Information Processing Systems 36 (2024).
2) Geshkovski, Borjan, et al. "A mathematical perspective on transformers." arXiv preprint arXiv:2312.10794 (2023).
3) Cowsik, Aditya, et al. "Geometric Dynamics of Signal Propagation Predict Trainability of Transformers." arXiv preprint arXiv:2403.02579 (2024).

### Questions
1. What is the relationship between this paper and the recent Geshkovski et al. paper "Dynamic metastability in the self-attention model" [1]?
2. Are there simple ways to extend this analysis as $N \to \infty$ in the discrete case? Is there a qualitative difference in this setting?
3. In practice, should we think that transformers operate in phase 1, 2, or 3?
4. How do we know that neglecting the remainder $R$ from equation 5 to 6 is acceptable?
5. What is the interpretation of the $\alpha \to 0$ limit (i.e. should it be interpreted as $N\to\infty$ again)?

**References:**

1. Geshkovski, Borjan, et al. "Dynamic metastability in the self-attention model." arXiv preprint arXiv:2410.06833 (2024).

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
1

### Summary
The authors build on existing work which models the evolution of tokens within a Transformer model. The authors prove the existent of a meta-stable phrase in learning.

Note: I fear I am not the right audience for this paper. I am not at all familiar with the literature, or its relevance in the ML community. I additionally understood little of the content of this paper.

### Strengths
- The paper appears to have a high degree of rigor.
- The goal of developing a rigorous theory of how the transformer works is important.

### Weaknesses
- It is not clear (to me) what the relevance of the conclusions made by the paper is.

### Questions
- Why does the  existence of a "meta-stable phase" imply that then the network learns a richer and more complex representation of the data? What is meant by richer and more complex?

- Are there any clear takeaways from this paper that a practitioner can use to build intuition and how and why a transformer works?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper presents a rigorous investigation of meta-stable clustering phenomena of mean-field Transformer models. By modeling transformer layers as continuous-time flows and employing a mean-field PDE framework, the authors present a very interesting approach to understanding intermediate clustering phases through linearization and higher-order approximation. Numerical experiments support the theoretical findings. This work opens a venue for understanding the meta-stable dynamics of Transformer models.

### Strengths
1. I appreciate the authors provide rigorous proofs that illustrate the convergence of the tokens to structured meta-stable manifolds, and identifies the influence of temperature, which is a key factor in understanding the mechanism of transformers. In particular, understanding the mean-field transformer model is beneficial for tasks involving long-context dependencies. 

2. Numerical experiments are presented to demonstrate that the theoretical predictions of the clustering dynamics align well with observed practical results. It is nice to see the number of clusters is influenced by the temperature, which I found surprising.

3. Overall, the paper is well-written, and the results are sound. The decomposition of the transformer dynamics into linear, quasi-linear, and collapsing phases allows for a systematic analysis of the clustering phenomena. I'm also excited to see new PDE techniques (the Grenier's scheme) applied to the analysis of transformer dynamics. The findings are both highly relevant to the field and of significant interest to the community.

### Weaknesses
1. (Minor) In the contribution bullet point 3, the authors state that "the periodicity developed in the first phase is maintained over exponentially long time intervals." The authors may want to clarify in what sense the time interval is "exponential long" and where in the paper this claim is made.

2. (Minor) Page 10, Line 504, Should "Figure 4" be "Figure 3"?

3. (Minor) Page 5, Line 233, should the coefficient $\hat{W}_k$ be $\tfrac{1}{\beta}I_k(\beta)$?

### Questions
I only have a minor question. The authors make a key Assumption 3 regarding convergence to the clustering phase. While they discuss this assumption based on (Markdahl et al., 2017), relying on periodicity, I am curious whether it would still hold in the generic case--specifically, in the case where the initial condition in equation (6) contains multiple modes. Could the authors provide some intuition and discuss potential implications of their result in this case?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Transformers are performant but not theoretically well understood. In particular, it is not clear how to think about the dynamics associated with a large stack of transformer layers. Geshkovski et al. (2023) made a formal analogy between these dynamics and the dynamics of a mean-field interacting particle system, and were able to characterize various aspects of these dynamics, at least in a simplified theoretical setting. The authors of this work build upon this earlier work and characterize not just the long-time behavior, but associated metastable dynamics, which may be important in practice.

### Strengths
The authors present a detailed mathematical analysis of a simplified model of transformers, and rigorously prove a variety of results. They are good about presenting their results at a high level in the main text, and leaving the fine technical details in SI. They closely connect their approach to a variety of recent work, principally the aforementioned work by Geshkovski et al.

### Weaknesses
The authors present their results clearly and do not appear to overstate them. My major concern is mostly that the authors' results concern kind of a toy setting somewhat far from the kinds of transformers people use in practice. One assumes Q = K = V = Id (line 123), one takes a limit where the number of layers go to infinity, the number of tokens $N$ is assumed to go to infinity, etc. This makes sense given that theoretical analysis of any kind is quite difficult, but makes it unclear to what extent the authors' results shed light on transformers 'in the wild'. It would be extremely helpful if the authors could provide some numerical experiments for slightly more realistic transformer models that show key predictions from their theory hold, at least approximately. 

Also, the authors could provide additional commentary on what happens if one tries to generalize their results. What is expected if Q, K, and V are not all the identity? How does one expect finite $N$ corrections to behave, and what do they change about dynamics?

### Questions
1. Do slightly more realistic transformers behave in a way consistent with the predictions of this theory? Relatedly, if the theory assumes Q = K = V = Id, what is the right point of comparison for a language model?
2. How does the theory change if certain assumptions (e.g., Q = K = V = Id, finite $N$) are modified or generalized?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the evolution of tokens in simplified transformer models using a mean-field formulation. By analyzing the long-term behavior of interacting particle systems, the authors prove results on metastable clustering, demonstrating that the model remains close to a metastable manifold of solutions. The study identifies the existence of linear, quasi-linear, and clustering phases in the token evolution.

### Strengths
The authors provide a comprehensive analysis of metastability that precisely describes the evolution of tokens. They successfully identify the existence of linear, quasi-linear, and clustering phases, contributing valuable insights into the behavior of simplified transformer models.

### Weaknesses
- The model considers all attention parameters $Q, K, V$ as fixed identity matrices. This significant simplification may limit the applicability of the theoretical results, as it does not capture the complexity of learned attention mechanisms.

- Some notations in the paper are not properly defined, which can hinder readers' understanding of the results. For instance, the notation $\hat{\rho}\_0$ related to Theorems 4.2 and 4.3 is not introduced. Additionally, $\mu\_{cluster}$ in Equation (8) seems to be defined on a one-dimensional space since $2\pi j / k_{max}$ is a scalar, making it unclear how to measure the W1 distance from $f_*(t)$. Providing clearer definitions and explanations would enhance the paper's readability.

- For sufficiently large $N$, the parameter $T_1$ becomes negative. This could restrict the applicability of the theorem to certain token counts. Clarification on how this affects the results and whether it imposes practical limitations would be beneficial.

- Minor: At the first glance, I thought the propagation of chaos result (Theorem 3.1) is also one of the main contributions but is actually weak due to the exponential growth of the bound. Now I see that this result serves to emphasize the importance of the metastability analysis. Clarifying this role could prevent potential confusion for readers.

### Questions
- In the integral representation of (USA-MF) (lines 148-149), the coefficient $1/N$ seems redundant since $\mu$ is an empirical distribution which contains the normalization term. Also, I’m not sure if (SA-MF) representation is also appropriate because of the same reason.

- Taking sufficiently large $N$, $T_1$ becomes negative. Does this mean the non-existence of linear phase?

- The convergence rate of W1 distance in Theorem 4.3 is not given in the main text. Can you clarify it?

### Soundness
3

### Presentation
3

### Contribution
3
