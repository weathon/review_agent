# Gradient Descent Converges Linearly to Flatter Minima than Gradient Flow in Shallow Linear Networks

- Decision: Reject
- Scores: 3, 6, 3, 5

## Abstract
We study the gradient descent (GD) dynamics of a depth-2 linear neural network with a single input and output. We show that GD converges at an explicit linear rate to a global minimum of the training loss, even with a large stepsize--about $2/\textrm{sharpness}$. It still converges for even larger stepsizes, but may do so very slowly. We also characterize the solution to which GD converges, which has lower norm and sharpness than the gradient flow solution.
Our analysis reveals a trade off between the speed of convergence and the magnitude of implicit regularization.
This sheds light on the benefits of training at the ``Edge of Stability'', which induces additional regularization by delaying convergence and may have implications for training more complex models.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper analyzes the GD dynamic of a 2-layer linear network with scalar input and scalar output. The authors study the rate of convergence of GD and provide some properties of the reached solution by using conserved quantities during a gradient flow dynamic.

### Strengths
The topic is relevant and the approach of studying simplified models is worthwhile. 
The paper includes clear and well-designed figures.

### Weaknesses
General impression: The paper appears to have been written hastily, lacking structure, with numerous typographical errors, missing hypotheses, and unsatisfactory proofs that are difficult to follow. Major revisions requiring another round of review are necessary in my opinion.

Regarding the results: The contributions seem overall quite weak and insufficient for acceptance at ICLR.

Some comments/suggestions:

1. The authors state 
> Despite its simplicity, the objective (2), which has also been studied by prior work (Lewkowycz et al., 2020; Wang et al., 2022; Chen & Bruna, 2023; Ahn et al., 2024), is a useful object of study because...

By quickly checking some of these references, it appears they do not restrict their analysis to the case $x_i, y_i \in \mathbb{R}$ only.

2. The authors claim
> In addition to showing how fast gradient descent converges to some global minimizer, we can also describe which of the many possible solutions, a ⊤b = Φ, gradient descent will converge to (l. 083) 

The corresponding result in section 3 does not specify which solution is reached, but only describes certain properties of that minimizer (which could be satisfied by multiple solutions).

3. Furthermore, the analysis for Theorem 1 requires $Q(0) \neq 0$ at initialization. This should be stated in the hypotheses of Theorem 1 and in the paragraph following equation (7). Additionally, $Q(0) = 0$ is also a common hypothesis, this should be discussed.

4. The paragraph following equation (8) motivates Takeaway 1 but falls short of rigorously proving it.

5. Sections 4, 5, and 6 severely lack structure (Section 5, subsection 5.1??).

6. Appendix C is unclear:
- First, the conserved quantity is preserved for gradient flow (not necessarily gradient descent). Therefore, the conclusion
> These two lemmas imply that the norm of the solution found by gradient descent will always be smaller than λ∞ and since ε∞ = 0 we can compute it using the formula above

(note: two → one lemma?) about a solution found by *gradient descent* requires proof.
- There appears to be confusion between $\lambda$ before Lemma 7 (continuous) and after (discrete); the link is not examined.
- The computations for the "proof" of Lemma 7 are incorrect. While the result may be true, a proper proof is needed.
- not clear at what point is defined the maximal sharpness (the formulation of Lemma 7 suggests it is at initialization).
- For Lemma 8, $\eta$ needs to be less than 2/maximal sharpness; this should likely be also the case in Lemma 7 (the first inequality of Lemma 7 appears to lack any attempt of proof)?

### Questions
See above

and l. 068

> More importantly, the dynamics when optimizing (2) with gradient descent are qualitatively similar to the dynamics of training more complex models

Could the authors elaborate on this point?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper aims to study the GD dynamics of two-layer linear networks with scalar input and output. The authors present a detailed analysis of the model they introduce, being able to identify analytically all the relevant aspects of the setting. In particular, they show the ability of GD to converge for unexpectedly large learning rates (edge of stability), and a full characterization of the implicit regularization that GD brings. Finally, they merge these two aspects and show a trade-off between speed and regularization.

### Strengths
- All results presented are precise and analytical, presented with mathematical rigor and well written. The model is extensively analyzed and all possible aspects are discussed in detail.
- I think the most valuable takeaway from this paper is the proof that, even in a very simple model, gradient flow dynamics are inherently different from gradient descent. In particular, the paper shows that GD regularizes better in this setting, but more generally can be used as a proof that the common use of GF as a theoretical tool for understanding GD is not always well founded.

### Weaknesses
- I think the claim of studying flat linear networks is an overstatement. The setting is too simple to claim that it is a faithful model for networks. It seems to me that related work analyzing similar settings is either not as simplified as this, or is motivated by something else like a matrix factorization problem (or both).
- Alongside the previous point, I am not sure that the results presented are relevant enough to pass the bar of the conference.

### Questions
- Do the authors have any idea how to extend some of the exposed results to non-scalar inputs? For example, instead of $\vec x\in \mathbb{R}$, at least $\vec x\in \mathbb{R^2}$ ?
- Section 5 is less than a quarter page long. Ether there is a problem/ something missing, or it should be merged together with Section 6 and better contextualized.
Minor:
- I think displaying $\lambda$ in bold can be misleading, since in the rest of the paper the bold is used for vectors.
- Typo line 430: $\eta<\eta$.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper analyzes the effect of taking a large learning rate in training two-layer linear neural networks with single input and single output, formalized by $\mathbf{a}^\top\mathbf{b}$. First, the authors evaluate the imbalance between $\mathbf{a}$ and $\mathbf{b}$ and ensure that by taking a larger learning rate, GD converges to a more imbalanced solution.
The authors also analyze the convergence rate and show that a larger learning rate leads to a slower convergence speed.

### Strengths
The analysis of the EoS phenomenon is one of the significant topics in the deep learning theory literature. The convergence results exhibited in Theorem 1 and 2 provide informative insights into this problem.

### Weaknesses
While the topic treated in this paper is interesting, the latter section, specifically from section 5, needs to be completed. For example, I could not find the proof of Propositions 2 and 3 in the Appendix. Moreover, some results are demonstrated without detailed explanation. Regarding these drawbacks, I vote for rejection.

### Questions
The model treated in this paper is quite simple, while the same model has been treated in the existing literature. Is there any implication for more complex models, such as matrix factorization with multi-dimensional input?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper builds on recent studies of gradient descent on a two-layer linear network with scalar inputs and outputs.  The main results are on the location and speed of convergence of gradient descent.  By a careful analysis of the training dynamics which includes the edge of stability phenomenon, the main results elucidate the interplays among the learning rate, the implicit regularization, and the speed of convergence.  The theoretical results are supplemented by plots provided by numerical experiments.

### Strengths
This is a simple yet challenging setting, chosen well for this in-depth theoretical analysis.

The results are non-trivial and interesting, and proofs are provided in the appendix.

The plots from the experiments are detailed and complement the theoretical results well.

### Weaknesses
The grammar and spelling are poor throughout the paper, and especially in Sections 5-7, sometimes causing near unreadability.

In Sections 5-7, the presentation is confusing.  It is not even clear what the purpose of those sections is, I suppose they are meant to show some components of the proofs of Theorems 1 and 2, since they consist of only Definitions, Lemmas and Propositions, however that is not explained.  If that is true, then it is not clear how and to what extent Sections 5-7 provide outlines of the proofs of Theorems 1 and 2.

There are no suggestions for future work.

There are no details about the experiments, and no code as supplementary materials.

More minor issues:
- In the last paragraph on page 2, I think the multiplier of Q(t) in parentheses should be in absolute value.
- In equation (8), I am not sure where the $\sqrt{3}$ above $\approx$ comes from, and also what the meaning of writing a condition like this above $\approx$ is.
- In Theorem 2, after "If", what is $\epsilon$, is it meant to be $\epsilon(0)$?
- It seems Section 6 should have been Section 5.2.
- "Fig 3" versus "Figure 2", please be consistent.
- The citation of Wang et al. at the end of Section 6 should not be in parentheses.
- Proposition 3 contains $\eta < \eta$.
- Only one item in the References has a clickable link.

### Questions
In the second-level bullet points before Proposition 2, there are various claims about convegence, why are they true?

How is Proposition 3 "the formal and clean version of Theorem 2", e.g. it contains nothing like the last part of Theorem 2?  And in what sense is Theorem 2 informal and dirty?

The remarks after Proposition 3 seem to contradict the last part of Theorem 2, i.e. "we can prove that no matter the step size and the initialization we have linear convergence" is at odds with "we have convergence but it could be logarithmically slow"?

### Soundness
2

### Presentation
1

### Contribution
2
