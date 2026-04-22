# Generalization and Scaling Laws for Mixture-of-Experts Transformers

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 4, 8

## Abstract
We develop a theory of generalization and scaling for Mixture-of-Experts (MoE) Transformers that cleanly separates \emph{active} per-input capacity from \emph{routing} combinatorics. Conditioning on fixed routing patterns and union-bounding across them, we obtain a sup-norm covering-number bound whose metric entropy scales with the active parameter budget and incurs a MoE-specific overhead. Combining this with a standard ERM argument for squared loss we provided a generalization bound under a $d$-dimensional manifold model and $C^\beta$ targets, showing that approximation and estimation trade off in the same way as in dense networks once active parameters are counted appropriately. We further prove a constructive approximation theorem for MoE architectures, demonstrating that accuracy can be improved either by scaling active capacity or by increasing the number of available experts, with the better of the two mechanisms prevailing. From these results we derive neural scaling laws, covering model scaling, data scaling and compute–optimal tradeoffs.
The theory highlights that enlarging the expert pool at fixed sparsity influences performance only through a mild logarithmic routing term, whereas increasing active capacity per input drives the main gains in generalization and approximation. These insights provide principled guidance for the design of efficient sparse Transformer systems and clarify the fundamental tradeoffs underlying their empirical scaling behavior.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper develops the generalization error bound and scaling laws for Mixture-of-Experts (MoE) Transformers. 
It establishes both approximation and  generalization bounds that depend on the number of active expert rather than total number of parameters. The authors further derive scaling laws and show that performance improves mainly through increasing the number of active experts, while the number of experts contributes only logarithmically.

### Strengths
The paper provides novel and rigorous approximation and generalization error bounds for MoE Transformers.
It also derives scaling laws depending on the number of experts and active experts, offering theoretical design guidance for practical architecture construction. So, I think this paper make meaningful contribution.

### Weaknesses
The definition of the MoE block $\mathrm{MoE}^{(j)}$ in Definition 3.1 is not clear to me, making it difficult to understand what functions or learnable parameters are used inside each expert and router. Moreover, several details in the approximation theorem remain unclear to me. I summarize below in Question.

### Questions
1. How is $\mathrm{MoE}^{(j)}$ defined precisely? In particular, what functions are used for experts and routers, and are they learnable ?

2. In the approximation theorem (Theorem 3.2), the upper bound does not explicitly include sequence length $\ell$. Does the bound depend on $\ell$ ? If so, how does this dependence appear?

3. Related to the above, while the sentences following Eq. (2) explain that MLPs approximate local polynomials, the role of the Attention layer is unclear to me ? How is the attention used in the proof ? Or, Could we get the same result without it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper develops a theoretical framework for understanding generalization and scaling laws in Mixture-of-Experts (MoE) Transformers. The authors decompose the generalization bound into three components: approximation, estimation, and routing overhead. They derive rates showing that, when routing costs are negligible, MoE models follow the same power-law scaling as dense networks, but in terms of the *active* parameter count per token rather than total parameters. The analysis further identifies a routing-dominated regime where combinatorial routing complexity slows convergence and the system does not operate in the favorable power-law regime, giving design guidance to avoid this suboptimal regime.  Finally, the authors compare their theoretical exponents with empirical scaling results from recent large-scale MoE studies, reporting qualitative agreement.

### Strengths
* The paper tackles an important and interesting question: providing a theoretical framework for generalization and scaling in Mixture-of-Experts (MoE) Transformers. The topic is highly relevant given the growing use of conditional computation in large-scale models.
* The decomposition of the generalization error into approximation, estimation, and routing terms is conceptually clear and translates into a clear interpretation of scaling laws.
* The empirical discussion connecting theoretical exponents to results from EL (Tian et al., 2025) and JMSL (Ludziejewski et al., 2025) shows awareness of current empirical literature and helps situate the theory with observed trends.

### Weaknesses
* **Incorrect or inconsistent definition of a Transformer block.**
  Definition 3.1 describes the block as summing the outputs of the MHA and FFN, both applied to the same input, rather than applying the FFN to the *output* of the MHA as in standard Transformer architectures. The same incorrect structure appears in the appendix, despite citing Vaswani et al. (2017). For a paper focused on Transformer theory, this is a serious technical/conceptual issue that must be corrected or clearly justified.
* **Acknowledgment of closely related prior work.**
  Theorems 3.2 and 3.3 appear to closely parallel Theorems 2 and 1 in Havrilla & Liao (2024), respectively, in both result form and proof technique. Although that paper is cited elsewhere, the authors should explicitly state the relation, clarify what is novel in their constructive proof of Theorem 3.2, cited as one of the main contributions, and acknowledge direct inspiration where appropriate.
* **Incomplete proofs for central lemmas.**
  The proof of Theorem 3.2 depends on Lemma A.7, but only a sketch is given. This prevents confident verification of the proof.

* **Ambiguity in the role of ε in Theorem 3.3.**
  The statement claims that the generalization bound holds for “any ε > 0” (line 193), but the proof in Appendix D seems to fix ε to the value given by the upper bound in Theorem 3.2. It is unclear whether the bound is uniform over ε or only valid for this specific ε.
* **Notation and variable inconsistencies (N, N_eff, N_active).**
  The notation changes across sections without precise definitions. It is unclear what `N`, `N_eff`, and `N_active` exactly represent. See questions for more details.
* **Introduction with heavy notation.**
  The introduction lists main contributions using specialized notation that is only defined later, reducing readability. A more accessible intro, deferring technical symbols or providing brief intuitions, would help readers and reviewers grasp the contributions faster.

### Questions
1. **Transformer block definition:** Is the formulation in Definition 3.1 (and Appendix B.1) a typographical error or an intentional architectural variant? If intentional, please provide motivation and discuss implications for the approximation and generalization results.

2. **Relation to Havrilla & Liao (2024):** Can the authors explicitly state how the proofs of Theorems 3.2 and 3.3 differ from Theorems 2 and 1 in Havrilla & Liao (2024)? In particular, what assumptions, constructive steps, or technical lemmas are new here versus borrowed?

3. **Lemma A.7 details:** Given its central role, can the authors provide a full proof of Lemma A.7, or substantially expand the sketch?

4. **Quantification over ε in Theorem 3.3:** Does the generalization bound hold uniformly for all ε > 0, or only for the ε produced by the constructive approximation in Theorem 3.2? If the latter, please restate Theorem 3.3 to reflect this.

5. **Clarify N, N_eff, N_active:**   In Theorem 3.3, it is not clear to me what `N` represents, should it be `N_eff` instead? On line 229, `N_active` is mentioned for the first time, and then we find it again in the shorthand form of Theorem 3.3 given by equation (6). Here, it is not clear what the difference between `N_eff` and `N_active` is, are they supposed to be the same thing? That seems the case when reading equation 4 in theorem 3.3 (where `N` is used). However, the sentence on line 285 (Proposition 4.2) kind of implies that `N_active` and `N_eff` are two different things. Moreover, `N_eff` is described in line 125 as "the active parameter budget", alimenting the confusion.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work investigates the theoretical properties of Mixture-of-Experts (MoE) Transformers by deriving bounds on their approximation and generalization error. The analysis first establishes a manifold-based approximation bound and then constructs a generalization error bound that disentangles the distinct effects of approximation, estimation, and routing. Finally, the proposed framework is leveraged to study the scaling laws of MoE models and to determine the optimal relationship between sample size and model capacity.

### Strengths
- The paper provides a conceptually clear and theoretically grounded separation between the effects of active capacity and routing in Mixture-of-Experts models.
- It provides a systematic study of MoE generalization, including both approximation and generalization error bounds.
- The approximation bound depends on the intrinsic dimension of the data, rather than its ambient dimension, aligning theory with practical observations in high-dimensional settings.

### Weaknesses
My primary concern lies in the potential looseness of the proposed generalization bound and its implications for the paper’s conclusions. The manuscript does not appear to include experiments or numerical estimations to validate the tightness of the bound (please correct me if I am wrong).

I believe this is a critical point that should be addressed for the following reasons:

1. Relevance to Modern Architectures:
It is well-established that traditional generalization bounds—such as those based on Rademacher complexity—tend to be too loose to meaningfully explain the performance of deep neural networks like Transformers. The paper should discuss whether and how the proposed bound mitigates this limitation.

2. Justification for the Optimization Strategy:
The optimization in Section 4 is predicated on minimizing the generalization bound under a compute budget. This reasoning is only justified if the bound serves as a reasonably tight and reliable proxy for empirical performance. If the bound is too loose, the optimization framework becomes less substantiated.

To strengthen the paper, I strongly recommend that the authors either (i) provide empirical evidence of the bound’s tightness or (ii) include a detailed discussion of this potential limitation.

### Questions
1. The variable $N$ is used in Equation (4), but its definition does not appear to be provided in the statement of the theorem or the surrounding text. Could the authors please clarify what $N$ represents?

2. The bound in Equation (2) includes an exponent of $\frac{-2\beta}{d}$, which suggests that the bound may become looser as the dimension $d$ increases.
   *   Could the authors comment on the typical magnitude of $d$ for large language models and discuss the implications for the bound's tightness?
   *   Additionally, could the authors specify the dependence of the constant $C$ on $d$ (e.g., is it polynomial or exponential)?
3. In Theorem 3.3, what is the meaning of $\tilde{O}(\cdot)$ hides logarithmic factors?

4. Regarding Remark 4.3, the authors state that the approximation error and estimation error are balanced. This is a significant claim, as the trade-off between these two errors is central to many learning-theoretic arguments. Could the authors provide some empirical evidence to substantiate this statement?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper develops a theoretical framework for Mixture-of-Experts (MoE) Transformers, aiming to explain their generalization and scaling behavior through classical approximation and statistical-learning techniques.
In section 3 The authors derive three key results:

1. Theorem 3.2 (approximation): shows that an MoE Transformer with top- $k$ gating can uniformly approximate any smooth target $f\in C^\beta$ on a $d$ -dimensional data manifold at rate `$\\|T-f\\|_\infty^2 < \min\{N_{\text{eff}}^{-2\beta/d}, \\, M^{-2\beta/d}\}$`, where the key point is that the active capacity `$N_{\text{eff}}\!=\!L_T\Pi_{\text{attn}}+L_Tk\Pi_{\exp}$​` counts only the parameters actually used per input.
2. Theorem 3.4 (covering-number bound): quantifies model capacity by showing that the metric entropy scales with active parameters plus a logarithmic term in total experts, $\log \mathcal N(\delta) (\Pi_{\text{attn}}+L_Tk\Pi_{\exp})\log(1/\delta) + L_T\ell k\log(eM/k)$.
3. Theorem 3.3 (generalization): establishes that empirical-risk minimization over this class achieves bounded expected error `$\mathbb{E}\| \hat T_n - f \|_{L^2(Q)}^2 < N_{\text{eff}}^{-2\beta/d} + N_{\text{active}}/n + (L_T\ell k\log(eM/k))/n$`, decomposing approximation, estimation, and routing-overhead terms.

Sections 4 and 5 use this bound for further analysis: by optimizing it over $n,N_{\text{eff}},k,M$ , they recover power-law learning curves and compute-optimal trade-offs, and finally check qualitative consistency with two prior empirical MoE scaling studies (EL 2025, JMSL 2025).

--(I couldn't get my mathjax codes that include norm `\|` to be compiled here on markdown, so I have put them in `code`. I am sorry for this.)

### Strengths
A rigorous and well-structured theoretical treatment that brings classical function-approximation and generalization tools to the modern MoE setting. Some points that stood out to me when reading the work were the following:

- The partition-of-unity argument in Theorem 3.2 gives a clean geometric picture: local experts approximate smooth functions on manifold charts, and top- $k$ gating realizes the gluing through a $k$ -sparse, non-negative mixture. The distinction between active and total parameters is formalized neatly and tied to the idea that MoE performance scales with *active compute* rather than total parameter count.
- Theorems 3.3 and 3.4 adapt classical covering-number analysis to mixture-of-experts, isolating a new *routing term* $L_T \ell k \log(eM/k)$ that quantifies the overhead from top- $k$ combinatorics.
- Section 5 positions the theory as a conceptual explanation and sanity check for empirical MoE scaling trends and exponent identities observed in recent large-scale studies (EL 2025, JMSL 2025).

### Weaknesses
My main concern is that the results remain mostly conceptual: tightness, empirical validation, and real-world calibration are missing. Section 3.5 and the associated design heuristics extrapolate beyond what the mathematics justifies; without even simple toy-data verification, these read as aspirational.

- The paper contains no new experiments or diagnostics. Consequently, the practical claims ("operate near $k^\star$", "keep $M \approx k$") remain speculative. The paper’s impact would be greatly improved by adding even minimal diagnostic or toy experiments showing that the training heuristics derived from their upper bounds are empirically validated.
- In the same vein, the procedure for estimating the intrinsic dimension $d$ and smoothness exponent $\beta$ in practice is informal and lacks validation—essentially a list of suggestions rather than a concrete algorithm. More generally, statements such as "efficient MoE training at fixed compute" read as more prescriptive than the data support; the theory offers asymptotic guidance, not empirical evidence.
- Perhaps I missed it, but while reading Section 3 I kept wondering about how loose or tight the union-bound over routing patterns is. Similarly for the other upper bounds—some discussion or diagnostic measure of looseness (even qualitative) would be helpful, especially given the importance of these terms in the final scaling laws.

### Questions
- In practical settings where routers specialize during training, does the bounded-overlap assumption $s_0(d)$ still make sense? In what scenarios might this assumption break down?
- Have the authors considered validating the heuristic $d,\beta$ estimation on toy manifolds with known ground-truth smoothness, to check whether the fitted exponents recover the theoretical ones?
- Since Section 5 relies entirely on external datasets, could a minimal in-house experiment (e.g., synthetic regression) strengthen the empirical grounding of the theory?

### Soundness
4

### Presentation
3

### Contribution
3
