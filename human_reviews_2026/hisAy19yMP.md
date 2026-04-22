# AVERAGE CONTROLLED AND AVERAGE NATURAL MI-CRO DIRECT EFFECTS IN SUMMARY CAUSAL GRAPHS

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 4, 8

## Abstract
In this paper, we investigate the identifiability of average controlled direct effects and average natural direct effects in causal systems represented by summary causal graphs, which are abstractions of full causal graphs, often used in dynamic
systems where cycles and omitted temporal information complicate causal inference. Unlike in the traditional linear setting, where direct effects are typically easier to identify and estimate, non-parametric direct effects, which are crucial for handling real-world complexities, particularly in epidemiological contexts where relationships between variables (e.g., genetic, environmental, and behavioral factors) are often non-linear, are much harder to define and identify. In particular, we give sufficient conditions for identifying average controlled micro direct effect and average natural micro direct effect from summary causal graphs in the presence of hidden confounding. Furthermore, we show that the conditions given for the average controlled micro direct effect become also necessary in the setting where there is no hidden confounding and where we are only interested in identifiability by adjustment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper extends the theory of causal effect decomposition to dynamic and potentially cyclic systems. Building on classical notions of controlled and natural direct effects, the authors introduce average controlled micro direct effects and average natural micro direct effects that capture short-lag causal influences within summary causal graphs. The central theoretical contribution is a set of graphical identification conditions establishing when these micro-level effects can be identified nonparametrically, even in the presence of feedback and latent confounders. The framework thereby generalizes existing identification results for nested counterfactuals to a broader class of temporal or cyclic models. The work is mathematically rigorous and conceptually ambitious, with particular relevance for fields such as epidemiology, where quantifying direct causal effects under complex dependencies is essential.

### Strengths
1. Ambitious theoretical extension. The paper generalizes classical definitions of controlled and natural direct effects to dynamic and cyclic systems through the “micro” formulation, representing a substantial conceptual step beyond standard acyclic mediation models.
2. Clear linkage to existing theory. The proposed definitions of average controlled and average natural micro direct effects are faithful extensions of established causal inference concepts (Robins & Greenland, Pearl, Avin et al.) into the time-indexed domain.
3. Potential impact on causal mediation in feedback systems. By providing graphical identification conditions for micro direct effects in summary causal graphs, the paper opens new directions for mediation analysis in recurrent and continuous-time causal models.
4. Rigor and completeness. The theoretical development is mathematically precise, and the proofs are carefully structured, offering a thorough and internally consistent foundation for subsequent applied or methodological work.
5. Generality and scope. The results are formulated abstractly enough to encompass a broad class of dynamic systems, including those with feedback and latent confounding, making the framework widely applicable.

### Weaknesses
1. Accessibility and clarity.
The paper is technically dense, and the intuition behind the key definitions—especially the natural micro direct effect—is difficult to follow without deep familiarity with classical causal mediation theory. More verbal explanation or illustrative examples would help readers grasp the motivation and implications of the results.
2. Empirical demonstration.
Section 6 offers a well-chosen conceptual example in epidemiology that clarifies the kinds of settings where the theorems could apply, but it stops short of a genuine data analysis or simulation. The example illustrates applicability but does not substantively validate the framework empirically.
3.	Connection to prior identifiability results.
While the paper references Pearl and Avin et al., it could more clearly articulate how its graphical conditions differ from or extend earlier nonparametric identification results for nested counterfactuals. A concise comparative discussion would help situate the contribution relative to existing theory.
4.	Scope and assumptions.
The identifiability results depend on relatively strong structural assumptions about the summary causal graph, and it remains unclear how often those conditions hold in realistic dynamic data. Some discussion of practical plausibility or examples of compliant systems would be useful.
5.	Positioning within broader causal literature.
The paper would benefit from a clearer discussion of how “summary causal graphs” relate to other frameworks for representing feedback and temporal structure—such as dynamic SCMs or equilibrium SEMs—to help readers integrate this work into the larger causal-modeling landscape.
6.	Conclusion and synthesis.
The conclusion appropriately reiterates the main contribution—extending identification of micro-level direct effects to cyclic and confounded systems—and connects it to applied domains such as epidemiology. However, after a dense theoretical development, it functions more as a summary than a synthesis. The paper would be stronger if the authors distilled the intuitive meaning of the main theorems and clarified concretely what these new identification results enable beyond prior frameworks (e.g., Avin et al., Pearl). The “future work” paragraph, while interesting, somewhat dilutes the central message by listing multiple directions rather than emphasizing the principal conceptual advance.

### Questions
1. Clarification of intuition. The definitions of the average controlled and average natural micro direct effects are mathematically precise but conceptually demanding. Could the authors expand the intuitive explanation—perhaps by adding a small time-indexed causal diagram or a simple dynamic example—to illustrate what these effects represent in a concrete feedback system?
2. Relationship to classical identification results. The natural micro direct effect appears to generalize nested counterfactual identification results from Pearl (2001) and Avin et al. (2005). Could the authors clarify in what sense their graphical conditions differ from, or strictly extend, those earlier frameworks? For example, do their results reduce to the known back-door or mediation-formula cases when the system is acyclic?
3. Interpretation of summary causal graphs. How should readers interpret the “summary causal graph” formally relative to dynamic structural causal models (DSCMs) or equilibrium SEMs? Is an SCG a representation of lag-specific causal relationships, or does it correspond to an equilibrium approximation over micro time scales?
4. Practical identifiability. The paper’s identifiability conditions seem strong. Could the authors comment on their practical plausibility—for instance, in what classes of dynamic systems (linear Gaussian, continuous-time diffusion, etc.) do these conditions typically hold?
5. Potential empirical validation. Even a small simulation illustrating how the proposed identifiability results could guide estimation would greatly help readers. Do the authors plan to test these ideas empirically, or are there conceptual obstacles that currently prevent such a demonstration?
6. Notation and constants. Theorem 3 introduces constants such as λ and r(M_{\text{micro}}). Could the authors briefly explain their interpretation or role in identifiability, and whether they have any empirical meaning?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors investigate identifiability of direct effects from so-called Summary Causal Graphs (SCGs), which encode in a compact way full causal graphs often used in dynamical or temporal systems. More specifically, they investigate the identifiability of two types of direct effects: average controlled and average natural direct effects. The main contribution of the paper are new sufficient conditions (expressed in a graphical language) for identifying average controlled *micro* direct effects and average *natural micro* direct effect in SCGs in general non-parametric setting and in the presence of hidden confounding.

### Strengths
This article falls within the research area of ​​identifying causal effects using partially defined graph models as, e.g., CPDAGs and PAGs. Studies in this direction are very well motivated and important from both a theoretical and practical perspective. The authors consider summary causal graphs (SCGs), which are well known partially defined models that ignore temporal information and allow cycles. A key contribution of the work are new graphical identification criteria which extend the criteria proposed by Ferreira & Assaad (2024a) for SCGs but without hidden confounders and assuming only linear mechanisms. The proofs are based directly on Pearl's do calculus.

### Weaknesses
The main weakness is that the graphical identifiability criteria are not complete. Theorem 1, which is the main result of the paper, only gives sufficient conditions for identifiability. The question arises as to how large the difference is between identifiable cases and those that do not meet the conditions specified in the theorem. The authors note that these conditions are generally not necessary, but do not analyze this in more detail. 

This issue is addressed in some way in Proposition 1 that says that the conditions do become necessary but only considering identifiability by adjustment and, what is important, under Assumption 1 which means that there are no hidden confounding in the model. But allowing hidden confounders is a key property considered in the paper. 

Similar questions apply to the conditions in Theorem 2.

Another problem is that the authors do not discuss the algorithmic (and computational complexity) aspects of the graphical conditions (in Theorem 1 and 2). In particular, how can one compute the set of possible parents of PP(Y_t)?

### Questions
Please refer to my comments above. In particular, do you know instances violating the graphical conditions in Theorem 1 that are identifiable (with hidden confounders and not necessarily by the adjustment).

L. 151: ∀Y_t, ∀X_{t−γ} ∈ V \ {X_{t−γ}} should be: ∀Y_t, ∀X_{t−γ} ∈ V \ {Y_t}

Please give labels (a), (b), etc. in Figures, as e.g. in Fig.1.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates the identifiability of micro direct effects within summary causal graphs (SCGs), which are abstractions used to represent complex dynamic causal systems. This work extends previous research on the identification of micro direct effects in summary causal graphs, which primarily relied on linear assumptions, by instead considering the more realistic non-linear case to identify non-parametric direct effects. The paper's main contributions are providing sufficient conditions to identify the controlled micro direct effect (CDE) and the natural micro direct effect (NDE) from SCGs in the presence of hidden confounding. Furthermore, it shows that these conditions for CDE become necessary in the absence of hidden confounding when identification is restricted to adjustment.

### Strengths
- The paper tackles the identification of micro direct effects in non-parametric settings, an open challenge in summary causal graphs for causal inference. 
- The paper establishes sufficient graphical conditions for identifying both the average controlled micro direct effect and the average natural micro direct effect. 
- The paper effectively uses graphical examples (e.g., Figs 1-4) to demonstrate why identification is difficult, showing how different underlying time-series graphs can be compatible with the same SCG.

### Weaknesses
- The paper does not establish necessary and sufficient conditions for CDE in the general case with hidden confounding or for identification methods beyond simple adjustment.
- The paper relies on the assumption of a known maximal lag between causes and effects without adequately justifying why this is a reasonable or practical assumption in real-world applications.
- The examples provided can create a pessimistic impression, as they frequently illustrate scenarios where the CDE and NDE are not identifiable, suggesting such cases may be common, even with the sufficient conditions provided in this paper.
- The paper's contributions are purely theoretical and lack empirical validation through numerical experiments or applications to real-world data.

### Questions
Compared to the technical difficulty of identifying micro total effects (for which conditions in a non-parametric setting for SCGs have been established in previous work ), what are the specific technical challenges involved in identifying micro direct effects, as addressed in this paper?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies the identification of controlled direct effects (CDEs) and natural direct effects (NDEs) [Pearl, 2001] in summary causal graphs, which are abstractions of time-series causal graphs and hence may contain cycles and hidden confounders. In particular, the paper proposes sufficient conditions under which the CDEs and NDEs are identifiable, and necessary conditions for identifying CDEs when there is no latent confounder.

### Strengths
- In general, the paper is well structured and clearly written. The definitions and theorems are formulated clearly, and detailed proofs are provided in the appendix. 
- The subject of identifying direct effects is a good starting point, which can potentially lead to more general results on causal effects.
- The paper includes sufficient background on the definitions of two different direct effects, the time-series causal graphs, and identification methods such as do-calculus, which facilitates the understanding of the main results. I also found the illustrative examples and the real-world example (Section 6) very helpful.

### Weaknesses
1. While the paper provides sufficient conditions for identifying direct causes, it currently does not present necessary conditions in more general settings beyond identification by adjustment. As the authors also mentioned in the conclusion section, this is a limitation and could be a subject of future work. 
2. Line 158: "a stationarity assumption becomes necessary to satisfy the positivity assumption." Could you elaborate on this? These two concepts seem quite distinct to me.
3. I believe it would be helpful for general readers if the authors could provide a concrete example illustrating the difference between controlled and natural directed effects near their definitions. 
4. Line 283-286 (Property 1): The notation $z|_{Parents(Y_t,G)}$ is undefined.
5. Line 305 and below: The independence signs look tiny.
6. Line 460-461: Theorem 2 only shows a sufficient condition for identifiability, so it cannot be used to justify unidentifiability. Please provide more detailed explanations on why the NDE is unidentifiable.

### Questions
1. Is Assumption 1 identical to the causal sufficiency assumption commonly used in causal literature? If so, it may be worth explicitly mentioning this.
2. Definition 6. Is there a reason why the self-loop is stated explicitly? Why not treat $Y$ as a parent of itself when there is a self-loop?
3. Line 418 (Theorem 2): Does the identifying formula still qualify as an adjustment formula? Please clarify if it does not.

### Soundness
3

### Presentation
3

### Contribution
3
