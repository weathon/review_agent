## Human Reviewer 1

### Summary
The paper addresses learning a Markov Equivalence class amongst causal graphs what admit latent variables in the Semi-Markovian model, interventional distributions based on soft mechanism changes and unknown post treatment selection variables. It is assumed that the identity of the interventions are known. 

Paper makes a case that equivalence classes under interventions and under selection for semi markovian models has been characterized. But the Markov equivalence classes become very different when both are present - interventions and selection variables on post interventional distributions. Authors define a generalized notion of Markov equivalence based on marginal invariance tests, conditional invariance tests and also conditional independence under selection. 

Authors characterize the Markov Equivalence class under this new setting, providing also a sound and complete constraint based discovery algorithm.

### Strengths
1) Paper tackles a novel problem in equivalence class characterization when interventions and selection variables are involved. Authors demonstrate how post interventional selection could give different invariance signatures of p(effect), p(cause) and p (effect|cause) across interventions with and without selection.

2) The characterization and sound and complete algorithms are very novel and elegant contributions as well.

3) Reasonable experimental demonstration that this works both with synthetic data and real world datasets.

### Weaknesses
I have only one principal concern - it is with respect to the definition of the Markov equivalence. It is defined only with respect to marginal invariance, conditional invariance between observational distribution and interventional distributions.

 However, it is possible to incorporate invariances between two interventional distributions (not involving the observational one) as in Kocaoglu et.al. 2019 by including augmented variables, one for each pair of interventional target sets. This is possible when a soft intervention on a node could makes the same causal mechanism change irrespective of the interventional collection it is in. For some reason this has not been incorporated in the current set up. This means even with respect to invariance tests, there is a bit more leeway in terms of constraining the equivalence class.  
rct64e3

### Questions
1) Can authors explain how one would extend to the setting where invariances across two interventional sets can be incorporated ?

2)  Why is the performance curves in the synthetic data very close to FCI-Interven (FIgure 6) ? Is it that latent variable presence dominates the selection variables effect ? Is it possible to introduce more selection variables than latents to see experimentally if the separation is bigger ?

3) Is there a typo in Line 357 where two squares case needs to be denoted $\\square - \\square$ with an edge in between ?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors present a new formulation for interventional causal discovery which allows for explicit modeling of so-called post-treatment selection and for considering the extent to which differential responses to interventions may distinguish causal relationships from selection patterns. They propose a new definition of Markov equivalence, called FI-Markov equivalence, to characterize equivalence classes for the new setting, and provide a new graphical representation for an FI-Markov equivalent class, called F-PAG. They then modify the canonical FCI learning algorithm to develop the F-FCI algorithm for learning F-PAGs and prove its correctness and completeness.

### Strengths
“Post-treatment selection bias” is a well-known phenomenon in empirical research and is an important issue in causal inference, with implications for research in medicine, the social sciences, and other fields. To my knowledge, this work presents the first systematic study of the interventional causal discovery, taking into account post-treatment selection. The introduced new concepts of Markov equivalence (FI-Markov equivalence) and graphical models for equivalence classes (F-PAGs) constitute an interesting contribution to the field. An important achievement is the development of a provably sound and complete algorithm, F-FCI, that enables the identification of cause-effect relationships, latent confounders, and post-treatment selection.

### Weaknesses
The relationship of this article to the paper introducing pre-treatemt selection framework presented at the ICLR by Dai et al. (2025) is not clear and should be further discussed.

### Questions
L. 106: Conditioning on S = 1 encodes selection for both pre-treatment selection 
and post-treatment selection. Q: Why don't you differentiate between these types of selection? Later, e.g., L.187 you write: S = 1 indicates the presence of post-treatment selection.

Eq.(1): what does the subscript s in p_s^{(k)} mean? Moreover, in the first product the probability p^{(i)} does not make sense. 

L. 134: Do you make any assumptions about the collection of intervention targets, other than that I^{(0)} is empty?

Def. 1 is not clear: explain I, {\mathcal I}, I^{(k)}.

L.213: improve the punctation.

L. 234: what do you mean by Aug_{I_K} ? Should be Aug_{I^{k}}?

L. 235: the corresponding augmented DAG --> you mean DAGs?

Definition 2 is not clear: what does X_i ∈ X_{\cup I} mean?

L. 332:   R-PAG --> F-PAG

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors investigate the issue of post-treatment selection in the context of causal discovery. They develop a thorough framework for characterizing post-treatment selection and introduce a new class of observationally equivalent causal structures in this context.  Furthermore, a new algorithm based on their framework is discovered and  investigated theoretically (soundness and completeness) and experimentally on synthetic and real-world data.

### Strengths
The problem that the paper addresses is well motivated and I agree that it is an overlooked yet critical issue. All claims are well supported by evidence. 

I am also not aware that this problem has been addressed (in this way) before. 

The conceptual and theoretical analysis of post-treatment selection is rigorous and extensive. The paper is clearly written, well motivated and supported by effective figures. 

The proposed algorithm is reasonable and the provided experimental results seem very sound and support the contribution by the paper.

### Weaknesses
The experiments are somewhat limited: 
(a) the synthetic data is rather simple, it would be interesting to see how the method performs with different non-linearities, forms of noise and larger graphs; including and ablation over those factors. It would be especially interesting to see how well the proposed method can perform on realistic simulators introduced by Robertson et al. (https://arxiv.org/abs/2506.06039). 


The discussion of limitations and future work for the method is fairly short and could benefit from being extended. 


NIT: There is an extra space at the end of line 166

### Questions
Please feel free to reply to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper addresses an important problem: interventional causal discovery under post-treatment selection. The authors systematically analyze how post-treatment selection affects causal identifiability and summarize invariant and variant distributional patterns that arise in such settings. Building upon existing interventional frameworks, they propose a novel causal formulation that explicitly models post-treatment selection, i.e., FI-Markov, F-PAG and  a new algorithm F-FCI. Experiments on both synthetic and real-world datasets demonstrate the effectiveness and robustness of the proposed approach.

### Strengths
1.	The characterization of invariant/variant rules under post-treatment selection and the introduction of the causal formulation are novel. 
2.	The paper provides a solid theoretical foundation for F-FCI.

### Weaknesses
1.	I did not observe any discussion in the paper regarding how F-FCI handles Type II inducing nodes.
2.	Figure 10 shows that GIES outperforms F-FCI under hard interventions in terms of F1 and recall, but the paper does not analyze why this occurs. A brief discussion would help clarify the underlying cause of this performance gap.
3.	There is a typo in Figures 10 and 11, R-FCI should be corrected to F-FCI.
4.	It would strengthen the paper if the authors could include a case study showing which edges existing causal discovery algorithms falsely infer due to post-treatment effects, and how F-FCI correctly identifies and removes these spurious edges.

### Questions
1.	How does F-FCI handle Type II inducing nodes?
2.	Why does GIES outperform F-FCI under hard interventions in terms of F1 and recall?
3.	Should “R-FCI” in Figures 10 and 11 be corrected to “F-FCI”?
4.	Can the authors provide a case study showing spurious edges caused by post-treatment selection that existing algorithms falsely discover, and that F-FCI correctly identifies?

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4