# Can Transformers Do Enumerative Geometry?

- Decision: Accept (Poster)
- Scores: 3, 3, 5, 8

## Abstract
We introduce a Transformer-based approach to computational enumerative geometry, specifically targeting the computation of $\psi$-class intersection numbers on the moduli space of curves. Traditional methods for calculating these numbers suffer from factorial computational complexity, making them impractical to use. By reformulating the problem as a continuous optimization task, we compute intersection numbers across a wide value range from $10^{-45}$ to $10^{45}$. To capture the recursive nature inherent in these intersection numbers, we propose the Dynamic Range Activator (DRA), a new activation function that enhances the Transformer's ability to model recursive patterns and handle severe heteroscedasticity. Given precision requirements for computing the intersections, we quantify the uncertainty of the predictions using Conformal Prediction with a dynamic sliding window adaptive to the partitions of equivalent number of marked points. To the best of our knowledge, there has been no prior work on modeling recursive functions with such a high-variance and factorial growth. Beyond simply computing intersection numbers, we explore the enumerative "world-model" of Transformers. Our interpretability analysis reveals that the network is implicitly modeling the Virasoro constraints in a purely data-driven manner. Moreover, through abductive hypothesis testing, probing, and causal inference, we uncover evidence of an emergent internal representation of the the large-genus asymptotic of $\psi$-class intersection numbers. These findings suggest that the network internalizes the parameters of the asymptotic closed-form and the polynomiality phenomenon of $\psi$-class intersection numbers in a non-linear manner.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
Unfortunately, I have no expertise at all in computational enumerative geometry. My review will thus be quite superficial.

*Summary*

This paper proposes to use transfomer models to tackle what I understood is a central problem in enumerative geometry: computing the phi-class intersection numbers on the moduli space of curves. From my pretty crude rudimentary and pragmatical ML perspective, the authors reduce this problem to learning a multi modal function mapping input tuples of the form (quantum Airy structure datum [a tensor / sequence of tensors], genus [integer], number of marked points [integer], partitions [permutation-invariant set]) to output intersection numbers [sequence of integers (?)]. The model is trained on solutions computed using brute-force methods up to some genus, and evaluated on its ability to extrapolate to find solutions for higher genus (geni?). 

The main technical contribution of the papers are methodological and consist in 

(i) designing a specific multi-modal transformer architecture suited to the problem at hand (combining mostly existing models / techniques)
(ii) introducing a novel activation function specifically suited to model recursive functions, which are crucial to solve the problem. 

Experiments on synthetic data are provided demonstrating that the model seems to be able to extrapolate to higher geni than the ones seen in the training data. The authors also provide some more qualitative analysis to investigate to which extent the internal representations learned by the model encode mathematical structures that are known to be relevant to solve the problem.

### Strengths
(S1) Investigating to which extent the recent successes of transformer models can transfer to other tasks, such as the one of solving fundamental problems in mathematics, is worthwhile and relevant.

### Weaknesses
(W1) The relevance and technical aspects cannot be well understood / evaluated unless the reader has some non-trivial background knowledge of enumerative geometry.

(W2) The writing and exposition of the material can be improved.

### Questions
*Recommendation*

I recommend to reject the paper mainly because I believe ICLR is not a suited venue, both for referring this paper (this paper needs to be reviewed by at least one expert in enumerative geometry, I don't know if there are such reviewers at ICLR) and for disseminating it (a journal in the field of computational enumerative geometry may be more suited). Furthermore, in my opinion the presentation of the material can be improved in several aspects before publication.

 
*Comments and questions*

- To which extent incorporating the conformal prediction framework in your analysis necessary? I am afraid this adds an additional layer of complexity that further hinders the communication of your findings. Maybe this discussion should be deferred to the appendix, keeping only what is strictly necessary to understand the main conclusion of your experiments in the main paper. 

- I don't understand the paragraph on top of p. 7, and I don't think this is due to my lack of expertise in enumerative geometry. In particular, what does "the neural network embedding p_g,n ... is a vector space"  means ? How can a function be a vector space? What does "go to the inner product space" means? These are (to me) very loose nonsensical mathematical statements.

*Minor comments & typos*

- p.3 the acronym COO has not been introduced

- Figure 5 should be included in the main part of the paper. In general, avoid forward references to far away, especially in the appendix without mentioning that it is in the appendix. 

- Use capitalization when reference tables, figures, sections, equations, etc. in the text (no capitalization needed when referring to figures or tables in general). E.g. Figure lines 168, 197, 334, Section lines 204, Table lines 274,298, Equation lines 319, 389 ... ... 

- line 196 we -> We

- line 421: the sentence "The interesting thing is that this is the performance of the non-linear probe." could be rephrased to better suit a formal publication.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors investigate the ability of transformers to compute the psi-intersection numbers in geometry and found that they perform (unsurprisingly) very well in distribution and quite well outside of distribution, and performed a series of analyses to understand which structures are being learned this way. They find in particular that the model learns the Dilaton equation and some information about the exponential growth of these psi-intersection numbers.

Overall, this paper is written very carefully, with excellent explanations of what is being done, although the importance of some details is unclear (like: what do we need to know about psi-intersection numbers? most of the sophisticated formulae are not really used in a meaningful way). However I am not sure the work is very interesting from a geometric point of view (the interesting thing is to gain theoretical insight into what psi-intersection numbers, not get somehow numerically accurate estimates of them) or from a machine learning point of view (it is not clear what is more interesting about these numbers than about any sequence in the OEIS, say). The experimental results are not particularly surprising given what is known about transformers, or at least I don't see it. 

While this work can be viewed as a first step towards making progress in applying machine learning to enumerative geometry, and the carefulness of the writing and experiments should be commended, I don't think it brings a lot of interesting new informations about machine learning or enumerative geometry.

### Strengths
The care and clarity of the writing, the fact that some extensive research has been done, the general trust in the results that this paper inspires.

### Weaknesses
What do we learn about machine learning or enumerative geometry? We seem to learn something that could be expected, a particular case of a general phenomenon.

### Questions
It would be good if the authors could at least mention questions that would bring something interesting to the enumerative geometry (something feels interesting when they perform an analysis of the internal representation of the network, but it stops just before it gets interesting...).

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper introduced DynamicFormer to learn and predict the $\psi$-class intersection numbers. Experiments include both in-distribution results and out-of-distribution results. The author also presented some experiments to illustrate how transformers perform enumerative geometry. Meanwhile, the authors also investigated whether the proposed method could perform abductive reasoning and hypothesis testing to estimate the parameters of asymptotic form for intersection numbers.

### Strengths
+ The idea of using transformers to do enumerative geometry is new. 
     + Meanwhile, the authors proposed a new activation function, DRA, which found to be useful to improve the prediction performance.
     + The authors compared DRA with other popular activations functions in Figure 1 and Table 2.

+ Experiments show some evidence of transformers can learn to predict the $\psi$-class intersection numbers.
     + Meanwhile, the authors also presented a discussion on how transformers being able to achieve that by inspecting internal vector space of the model.

+ The author also investigated how inputs affect the model’s understanding of $\psi$-class intersection numbers and the parameters for large genus.

### Weaknesses
I am not an expert in "enumerative geometry". However, I think the paper lacks many important clarifications and discussions.
 
+  The paper lacked discussion of the reasons/motivations of using transformers.  At the moment, the paper seemed only a combination of  a popular neural network architecture and a new mathematical problem.
+ The "Related Work" section is quite weak at the moment: the authors spent only one paragraph to discuss related works and then summarized their contributions.
+  From my perspective, the proposed DRA is not the only way to capture the periodic behavior in data. This lacks sufficient discussion in the paper.
+ From experiments in Figure 1, the authors did not apply DRA to other neural network architectures (eg MLP), and provided readers with more discussions on that.
+ Lack of theoretical discussion on the proposed method.
+ Code is not available.

### Questions
+ I wonder why the authors choose transformers as the regression function? 
+ In Figure 1, have you tried to apply DRA to MLP or other potential neural networks?
+ Will the code and the datasets be available?

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
The paper proposes and tests the usage of transformers in the field of enumerative geometry, specifically regarding topological recursions and $\psi$-class intersection numbers. To accomplish this, the paper proposes a new class of activation functions called Dynamic Range Activators (DRAs), and presents evidence of their performance in predicting a simple recursive function as part of a fully connected neural network, and then their ability to predict $\psi$-class intersection numbers as part of their DynamicFormer architecture. The paper then attempts to investigate the trained DynamicFormer to see if it can predict other concepts in enumerative geometry, including the Dilation equation that stems from Virasoro constraints, as well as the asymptotic behavior of $\psi$-class intersection numbers using abductive reasoning, verified using counter-factual intervention.

### Strengths
* The new DRA functions, motivated by the evidence presented in the paper, are a significant contribution that may interest machine learning scientists.
* Training a DynamicFormer to predict $\psi$-class intersection numbers, which then allows one to investigate a system's deeper geometry, is a significant, novel contribution that will interest mathematicians investigating enumerative geometry.
* The use of Conformal Prediction to estimate uncertainty provides a concrete measure of confidence in the experimental results, contributing to the paper's soundness.
* The figures are clear and high-quality, with informative captions.
* The writing is clear and mostly organized, including the mathematical background, methodology, and results.

### Weaknesses
**Notice: These weaknesses have been adressed during the discussion phase, and apply only to the initial version of the manuscript. However, these will remain unedited for posterity.**

### Section 2
* The equations in this section use $\hbar$ without defining it in the text. It may be worth explicitly calling it the reduced Planck constant in the text.
* The last paragraph mentions excluding the tensor $C$ due to a decreased impact on the computed $\psi$-class intersection numbers, observed during experimentation. Appendix C justifies this exclusion, yet is not referenced in the text, making for seemingly unsound reasoning for excluding $C$. The authors should consider referencing appendix C here to further justify the exclusion of $C$, and expanding on this point within appendix C proper.

### Section 3
* In the first two paragraphs, the paper presents the DynamicFormer for the first time and references a figure placed within an unrelated appendix, resulting in a disjointed reading experience. The authors may consider moving some parts of section 3 (such as its first two paragraphs) into a new appendix showcasing the DynamicFormer in detail and including the figure close by.
* In the same paragraphs, the authors use the initials COO without previously defining them. These initials seem to appear nowhere else in the main text, and only in appendix B are they defined as Coordinate List. Besides hurting the paper's readability, this seems to be an implementation detail that does not need to appear in the main text.
* The last paragraph mentions the [DYN] registry tokens, but fails to reference appendix B1. It may be appropriate to reference it here.

### Section 5
* Equation 5.4 is presented without proof, with the authors claiming they used an approach described in Eynard et al. (2023). A sketch of the proof (perhaps in an appendix) will contribute to the work's soundness.
* Figure 3, and the relevant experiment, are based on the assumption that $A$ is rational. The authors should consider justifying the choice of testing only rational values of $A$, perhaps by connecting it back to equation 5.3, as proven by Aggarwal (2021).
* Figure 3 presents a significantly higher value of $R^2$ for $A=2/3$ compared to the values for $A=4/6$ and $A=6/9$, despite being identical numbers. This issue does not appear for other such sets of identical rational numbers, such as $A=3/4$ and $A=6/8$. Since the rest of the subsection on Abductive Reasoning relies on $A=2/3$ being the correct answer, **this error calls the entire subsection into question and significantly hurts the paper's soundness and overall rating**. The authors must justify how the $R^2$ of $A=2/3$ is different from the other two values, or replace the figure (and perhaps rewrite some of the supporting text). Based on the other values of the figure, it should be expected to see a maximal $R^2$ around $A=2/3$, but without such a significant jump.

### Typos
* Section 5.1 line 319: "The topological recursion formula equation 2.4 [...]". Consider removing either "formula" or "equation", or placing all of "equation 2.4" in parentheses.
* Section 5.1.1. has multiple citations included in sentences with their parentheses. The ICLR 2025 formatting instructions (section 4.1) require such references to not have parentheses except around the year. The references in question appear in lines 371, 378, and 381.
* Section 5.1.1 line 417: "As a result, We find an evidence [...]". "We" does not need to be capitalized, and "an" should be removed.
* Appendix C line 950: "Figure 6 shows (s) numerical [...]".
* The title of appendix D and the caption of figure 7 both mistakenly write Princip**le** Component Analysis instead of Princip**al** Component Analysis.

### Questions
**Notice: These questions have been answered during the discussion phase, and remain unedited for posterity.**

* What is the significance of $\hbar$ in the quantum Airy structure? How is it relevant specifically to training the DynamicFormer?
* Figure 3 may be a discrete sampling of an underlying (continuous?) map that gives an $R^2$ for each $A$, with a maximum at $A=2/3$. Can the authors characterize this map?
* Figure 4 shows a significantly weaker causal impact of $B$ on the number of intersection points, compared to $n$ and $d$. Though the authors call this unexpected in section 5's last paragraph, is there any explanation regarding the weak causal impact of $B$?

### Soundness
4

### Presentation
3

### Contribution
3
