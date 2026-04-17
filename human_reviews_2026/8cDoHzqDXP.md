# On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via Hashing

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Associative Recall (AR) is the cognitive ability to learn and retrieve links between items in memory. In NLP, AR is used as a benchmark for evaluating the in-context memory capacity of architectures such as Mamba, and has been found to strongly correlate with language modeling performance. This paper explores AR from the perspective of mechanistic interpretability, aiming to reverse-engineer the exact internal algorithm used by Mamba to perform recall. Our key insight is that Mamba performs recall by implicitly learning linear hash functions, and we identify the low-level circuit that enables this behavior. Building on these findings and inspired by theoretical tools in similarity-preserving hashing, such as the Johnson–Lindenstrauss (JL) lemma, we develop a theoretical framework for analyzing AR, which we term recall scaling laws. For example, given a model’s state, context length, and embedding dimensions, our theory predicts a lower bound on the largest vocabulary size that can be perfectly recalled in the AR task. Empirical results show that this bound is tight and predictive, offering insights into how AR capacity scales with vocabulary size, state size, embedding size, and model architecture.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the associative recall mechanism in Mamba architecture. The first part proves that Mamba models can solve the multi-query associative recall task and demonstrates that the trained Mamba architecture closely follows the construction in the proof (up to equivalent orthogonal transformations). The second part builds on the construction to derive the scaling law for associative recall and empirically validates the law.

### Strengths
1. This paper addresses an important problem of how state space models retrieve information from their context.

2. The paper validates its theoretical constructions through empirical results.

### Weaknesses
1. The theoretical construction of Mamba architecture, solving multi-query associative recall, has already been discovered in the previous work (Huang et al. 2025). Empirical observations and scaling laws are novel, but these results lack connections to practical settings (since they are restricted to single-layer models) and quite straightforward corollaries from the construction.

2. The presentation of theoretical results can be improved. There are occasional notations without proper definitions (e.g., Dtilde in L167, N in 174), inconsistent variable names (e.g., different names between Figure 1 and Algorithm 1), inconsistent usage of notations (e.g., P_in and P_out use the same concatenation but should be along different dimensions), typos in equations (e.g., Equation 7 has unclear commas, wrong ordering of E and P). In general, these problems challenge understanding the ideas. The paper appears to have significant room for improvement by checking for consistency and refining the writing.

### Questions
1. How is the construction in this paper different from that in Huang et al. 2025?

2. Would the empirical observations (mechanistic and scaling laws) still be valid in multi-layer settings?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper performs a theoretical and mechanistic study of how Mamba solves associative recall, in particular the MQAR task. Focused on a simplified model (one-layer linear Mamba), the authors identity a solution for solving MQAR and verify such solutions present in the learned model weights. The authors then use JL lemma to theoretically show the required model size for solving associative recall for given task parameters. Finally, the theoretical claims are supported via empirical validations on both the simplified model and the original model.

### Strengths
1. The mechanistic interpretability analysis on validating the simplified Mamba recall circuit is interesting and nicely done (taking in account suitable invariances).

2. The quantitative analysis of how Mamba solves associative recall is thorough and empirically validated.

### Weaknesses
1. The theoretical construction of simplified one-layer linear Mamba largely follows from Huang et al. (2025), without providing enough remarks or discussions. Specifically, Theorem 1 uses the construction of Lemma 3 in Huang et al., and Theorem 2 largely reuses the idea of Theorem 2 in Huang et al. (i.e. using JL lemma to reduce dimensionality). Moreover, Theorem 4 in Huang et al. shows that using convolution and gating (together with a S4D-SSM) can solve MQAR, which takes a step towards understanding the role of gating, a direction that the authors remark for future work. The authors only highlight the difference with the study of Huang et al. in Sec 2, without properly attributing the similarities.

2. The mechanistic and theoretical study focus on a particular task (MQAR) to identify a recall circuit. However, it is unclear how generalizable this circuit is applicable to other related recall tasks (e.g., MQAR with repeating keys, induction head where keys and values are from the same vocabulary).

### Questions
1. In most of the theoretical constructions, the hidden state is treated as a matrix with size (N, d) (e.g., eqn 3). However, in equation 2, the hidden state $h_t$ is treated as a vector of size N. Can the authors clarify the purpose of eqn 2 and discuss its relations to the rest of the paper?

2. Fig.3 provides a very interesting inspection of the hidden state. I want to better understand the computation of $H''$: in lines 286-296, the authors comment $H''$ in eqn.13 does not exist as-is, but has to use $H'' = \Pi_{v, out} H_t' \Pi_{q, in}$. But it seems possible to compute eqn.13 from the learned embedding. Can the authors clarify?

3. How do the mechanistic and theoretical findings generalize to Mamba2?

4. The proof of Lemma 1: while the proof steps look correct to me, some wording and analysis look confusing. In line 1028-1030, the non-matching wrong fact pairs and non-matching empty pairs look the same from their mathematical definitions ($x_{\tau-1} \neq k_m, x_{\tau} \neq v_i$). Do the authors mean non-matching wrong fact as $x_{\tau-1} = k \neq k_m$ being a non-matching key, and non-matching empty pair as $x_{\tau-1} = v$ being an arbitrary value?

5. The proof of Thm 3, eqn (98): shouldn't it be $p_{\text{success}} = p_w^{N_f - 1} p_n^{V_v - N_f}$? can the authors double check the rest of the equations are correct?

Minor:

a. The notation $N_f$ in proof sketch of Lemma 1 and Theorem 3 means the same as $N_{\text{facts}}$?

b. What does the pixel color in Fig.6 mean (accuracy)? Also, the color bar is missing.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the Associative Recall (AR) capabilities of Mamba models (a Selective State Space Model, SSM) from a mechanistic interpretability perspective. The main contributions of this paper includes:

1. Reverse-engineer an underlying structure that enables Mamba to solve MQAR (Multi-Query Associative Recall). The peper first identifies a perfect non-compressive recall circuit with large state and model dimensions (Theorem 1), then relax the requirement for state and model dimensions through claiming a structure that solve MQAR with high probability (Theorem 2). The construction is verified through experiments. The paper further interprets Mamba’s recall as a linear similarity-preserving hash function, linking it to the Johnson–Lindenstrauss (JL) lemma.

2. Develop analytic AR scaling laws characterizing the success rate of a single-layer linear Mamba on solving the MQAR task, as a function of the state and model dimensions (N, D), vocabulary size V and number of facts $N_{\text{facts}}$ (Theorem 3). Further it derives the dimensions (N, D) required for solving an MQAR task with high probability, which are in the logarithm order of V. These theoretical results are validated empirically through experiments demonstrating matching scaling behavior.

### Strengths
This paper theoretically elucidates how the Mamba model performs associative recall, combining mechanism analysis, probabilistic modeling, and empirical verification.

$\textbf{Originality}$: The work introduces a novel mechanistic reconstruction of Mamba’s recall circuitry and interprets recall as a linear similarity-preserving hash, grounded in the Johnson–Lindenstrauss lemma. This creates an elegant conceptual link between state-space modeling and random projection theory.

$\textbf{Quality}$: The theoretical derivations are supported by empirical results. The accompanying simulations closely align with the analytical predictions, reinforcing the validity of the proposed framework.

$\textbf{Significance}$: This study offers the first theoretical scaling law for associative recall in Mamba models, offering quantitative insight into memory capacity and architectural trade-offs. These contributions advance theoretical understandings on the capability of selective SSMs.

### Weaknesses
As SSMs are not my primary area of expertise, my comments focus on readability and clarity. My main concern is that the paper is difficult to follow, even for readers with some familiarity with state-space models. To make the work more accessible to a broader audience, I encourage the authors to provide sufficient details when introducing new concepts or notations. Specifically, the following points could be clarified:

1. It would be beneficial to elaborate the example of MQAR (lines 094-100). What are keys, values, and queries in this context exactly?

2. What are the dimensions of each Mamba components ($\bar{A}_t,\bar{B}_t,C_t, \hat{x}_t, h_t, \hat{y}_t)$? Generally, when a matrix is first introduced, please specify its dimension. 

3. In Figure 1, what is $\hat{0}$? Also what is the circle product notation? If it is element-wise product, please be consistent with the $\odot$ notation. Please include sufficient details in the caption to explain how the circuit works.

4. In Theorem 1 and Theorem 2, please be specific what does $\textsf{expand}$ refer to.

5. If there are any typos in equations (5) and (7), please correct it.

6. In the definition of $\hat{E}_{in}$ (line 231), what are $W_0, W_1$?

7. In Figure 2, please explain what the axis labels t and $\tau$ refer to.

8. There appear to be inconsistencies in notation, e.g., $G_{kq}$ and $G_{qk}$ (in page 5), $N_{facts}$ and $N_f$ (in page 7).

9. In Figure 6,  it would be helpful to add a colorbar to indicate the numerical values represented by each color.

Overall, defining all variables clearly, avoiding notation inconsistency, and providing brief explanations when introducing new symbols or concepts would greatly enhance the paper’s readability and accessibility.

### Questions
1. Please refer to the weakness section.

2. In equation (2) (line 133), I assume $h_t, \hat{x}_t$ are vectors and $\bar{A_t}, \bar{B}_t$ are discretized matrices. How did element-wise multiplication applied here?

3. In the scaling law experiments, the authors report the maximum accuracy across trials. Could the authors clarify this choice? Intuitively, averaging results over multiple runs (or mean ± standard deviation) would provide a more robust and statistically meaningful comparison.

4. In this paper, all theoretical results are based on single-layer linear model. It would be valuable to discuss how these results provide insight into multi-layer SSM architectures, or to clarify what key challenges prevent extending the analysis to deeper Mamba models.

### Soundness
3

### Presentation
2

### Contribution
2
