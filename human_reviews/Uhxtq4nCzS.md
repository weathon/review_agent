# Divergence at the Interpolation Threshold: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 1

## Abstract
Machine learning models misbehave, often in unexpected ways. One prominent misbehavior is when the test loss diverges at the interpolation threshold, perhaps best known from its distinctive appearance in double descent. While considerable theoretical effort has gone into understanding generalization of overparameterized models, less effort has been made at understanding why the test loss misbehaves at the interpolation threshold. Moreover, analytically solvable models in this area employ a range of assumptions and use complex techniques from random matrix theory, statistical mechanics, and kernel methods, making it difficult to assess when and why test error might diverge; for instance, recent work found a divergence in noise-free toy nonlinear autoencoders, surprising the authors and raising questions about whether such an outcome should have been anticipated. In this work, we analytically study the simplest supervised model - ordinary linear regression - and show intuitively and rigorously when and why a divergence occurs at the interpolation threshold using basic linear algebra. We identify three interpretable factors that, when simultaneously all present, cause double descent. We demonstrate on real data that both models' test losses diverge at the interpolation threshold and that the divergence disappears when we ablate any one of the three identified factors. We conclude by using our fresh perspective to shed light on recent observations in nonlinear models concerning superposition and double descent.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work investigates the nature of the divergence of test performance and the interpolation threshold of double descent for ordinary least squares. The authors use simple linear algebra to decompose the loss into three terms (and a bias in the overparameterized case) that can be easily interpreted as factors that contribute to this phenomenon. Experiments are performed that ablate these three factors verifying that they provide a useful perspective. Next, the authors illustrate how the bias term of overparameterized regression can be interpreted as capturing the interaction between the size of the projection of the test point into the training subspace and the ideal parameters which introduces a representation learning perspective. Finally, they discuss how these intuitions might extend beyond the linear setting.

### Strengths
* I found the decomposition of the error in eqns (1) and (2) to provide a very clear perspective on the cause of divergence which, to the best of my knowledge, has not been previously derived.  
* I appreciated the minimalist approach to this paper. I think the core message is sufficient to be worthy of publication and the paper benefits from not being distracted by additional points.
* I appreciated the intuition of why it is increasingly likely that the Nth vector is unlikely to vary in the new dimension it has introduced. This was a nice perspective. It might even be nice to drive this home by providing a concrete example with, for example, Gaussian vectors where you could prove that each new singular value introduced is increasingly likely to be small (but this is at the discretion of the authors, not something I think is required). 
* I enjoyed the representation learning perspective of the bias term on p8 and in Fig 8. I found this perspective to be helpful in the context of this work. 
* I thought the experiments were sufficiently well constructed to illustrate the points made in the text (although I do have some concerns about the presentation of results included in the weaknesses section).

### Weaknesses
Overall, I am very positive about this work and think its core points (i.e. the decomposition and the projection of a test point) are instructive and a useful contribution. However, especially given that the goal of this paper was to be simple and accessible, I think that improving the exposition would greatly improve this work. I include some suggested improvements below and would be open to increasing my score if a revised version addressed these points.

* I think the statement on page 1 that "our goal is to explain intuitively and quantitatively why the test loss diverges at the interpolation threshold, without assumptions and without resorting to intricate mathematical tool" might be a very slight misrepresentation without mentioning that this is restricted to linear models. 
* I thought Fig 1 to be a strange choice and would prefer to see something that illustrates the message of the paper. Why not provide an illustrative plot of double descent in test loss with the decomposition of the three factors plotted in parallel?
* The other figures could also be improved in terms of clarity. The text size is far too small. The legends are sometimes incomplete (e.g. Fig 3 + 4 - what are the blue lines?). The titles are unclear (what does "condition" mean?)
* I thought Fig 7 in particular could do with some work. It is currently very difficult to understand exactly what is happening in each panel. 
* Given that the 3 bullet points on pages 5 & 6 are arguably the main point of the paper, I think their explanations could be sharpened somewhat. Particularly for the second point, I didn't find the text description to be intuitive and resorted to interpreting the expression myself. This is of course just a readability suggestion. It would also be nice to explicitly link each point to its corresponding plot. 
* There seems to be some contradiction in the ablation discussion on p7+8. It suggests "switching from ordinary linear regression to ridge regression", however, the actual ablation implemented is to "sweep different singular value cutoffs" and remove corresponding values. These are different things so it's not clear why ridge regression, which would apply shrinkage instead, is implied. 
* Similarly to previous figures, I thought the clarity of Fig 8 could be improved. In particular, the final two sentences of the caption could be spelled out better in the figure (and maybe also the caption). I don't think the current illustration does this final step justice (which is much more clear in the main text of the paper). I would suggest looking at e.g. Fig 3.4 of [1] for inspiration on how to improve the clarity of this style of figure. 
* It would be nice to include the code used to produce the results in this paper such that readers can easily interact with the experiments. 
* I think this work could have done a better job of contextualizing itself relative to other similar works (e.g. [2]). It would be nice to explain what other similar analyses of double descent (at least in the linear setting) have discovered about this phenomenon. 


Minor:
* "How much the training features X vary in each direction" - It might be clearer to specify that you refer to a direction r in the SVD rather than in the original X space. 
* page 5, point 2 references the wrong figure. 


[1] Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.

[2] Hastie, T., Montanari, A., Rosset, S., & Tibshirani, R. J. (2022). Surprises in high-dimensional ridgeless least squares interpolation. Annals of statistics, 50(2), 949.

### Questions
* "recall that our goal is to correlate fluctuations in the covariates x with fluctuations in the targets y" - this line was a little unclear to me. Could the authors expand on what this means exactly? I would interpret this term as capturing the information/signal of the first N of the D total features and therefore misses whatever information/signal is included in those final D - N features. Are you saying something more than this here? 

* I thought ablation 3 (obtaining $\beta^*$ via fitting to the full dataset) was a clever solution. However, I found Fig 5 difficult to follow. I think it would make more sense to illustrate the value of the $u_r \cdot E$ term (capturing the impact of the misspecification) so we can understand its contribution relative to the others. It's not totally clear to me how this would be combined with the current plot but at least the clarity of the current version should be improved.

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work investigates the least squares regression problem and identifies necessary conditions based on the SVD of the features for the test error to diverge at the interpolation threshold.

### Strengths
The paper is well written and easy to read. The intuitive discussion on the origin of the interpolation peak for least squares is interesting. The numerical experiments with real data sets illustrating the phenomenology and the visual schemes (e.g. Fig. 7 & 8) are helpful.

### Weaknesses
1. The result is very specific to least squares regression. For instance, the connection between the interpolation threshold and the rank of the features is a specificity of linear regression. Even in slightly more general settings, such as generalised linear models, this is not generally case. For example, in logistic regression the interpolation threshold corresponds the linear separability of the data [Rosset et al. 2003], and even for features that display a divergence at interpolation for regression display a kink rather than a peak for classification, see [Gerace et al. 2020] for an example in the context of random features. Although the authors argue that their criteria apply to other models (e.g. shallow auto-encoders), it is really not clear how general this is. The authors should consider adding at least a honest discussion of the limitations of this result.

2. The main results are contained in previous literature. For example: For instance, the relationship between the rank of the features and the interpolation threshold in linear regression has been discussed in [Hastie et al. 2022, Loureiro et al 2021], and the interplay between the features singular vectors and the target weights was studied extensively in [Wu & Xu 2020]. The fact that least-squares learns a representation which is the projection in the row-space of the features is a classical discussion in ML books [Bishop 2007]. The fact that double descent can occur with zero label noise due to model misspecification appeared in [Mei & Montanari 2022; Gerace et al. 2020; Loureiro et al 2021]. I appreciate the authors intention summarise and provide their own point of view to these results. However, I also feel that their justification that these previous works are "complex", "difficult" and "muddies the water" is subjective and comes across borderline disrespectful.

### Questions
- Although "under/overparameterised" is often used in the context of least-squares to refer to $N>D$ or $N<D$, this terminology is misleading, since in least-squares increasing the dimension $D$ both increases the number of parameters and decreases the sample complexity "N/D".

**Minor comments**
- The figures are small and hard to read. Different from Figs. 2-9, Fig. 1 is not in a vector format, so it gets pixelised when zoomed.

- To my best knowledge, the first works discussing the interpolation peak and how to mitigate them were [Opper et al. 1990; Krogh & Hertz (1991, 1992); Geman et al. 1992].

**References:**

[Rosset et al. 2003] Saharon Rosset, Ji Zhu, and Trevor J Hastie. Margin Maximizing Loss Functions.  Part of Advances in Neural Information Processing Systems 16 (NIPS 2003).

[Gerace et al. 2020] Federica Gerace, Bruno Loureiro, Florent Krzakala, Marc Mezard, Lenka Zdeborova. Generalisation error in learning with random features and the hidden manifold model. Proceedings of the 37th International Conference on Machine Learning, PMLR 119:3452-3462, 2020.

[Hastie et al. 2022]. Trevor Hastie, Andrea Montanari, Saharon Rosset, Ryan J. Tibshirani. Surprises in high-dimensional ridgeless least squares interpolation.  Ann. Statist. 50(2): 949-986 (April 2022). DOI: 10.1214/21-AOS2133

[Loureiro et al 2021]. Bruno Loureiro, Cedric Gerbelot, Hugo Cui, Sebastian Goldt, Florent Krzakala, Marc Mezard, Lenka Zdeborová.
Learning curves of generic features maps for realistic datasets with a teacher-student model. Part of Advances in Neural Information Processing Systems 34 (NeurIPS 2021).

[Wu & Xu 2020]. Denny Wu, Ji Xu. On the Optimal Weighted $\ell_2$ Regularization in Overparameterized Linear Regression. Part of Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

[Mei & Montanari 2022] Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics, 75(4):667–766, 2022.

[Bishop 2007]. Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2007.

[Opper et al. 1990] M Opper, W Kinzel, J Kleinz and R Nehl, *"On the ability of the optimal perceptron to generalise"*, 1990 J. Phys. A: Math. Gen. 23 L581.

[Krogh & Hertz 1991] A Krogh, J Hertz, *"A Simple Weight Decay Can Improve Generalization"*, NeurIPS 1991

[Krogh & Hertz 1992] A Krogh and J Hertz, *"Generalization in a linear perceptron in the presence of noise"*, 1992 J. Phys. A: Math. Gen. 25 1135.

[Geman et al. 1992] Geman, S., Bienenstock, E., and Doursat, R. Neural net-works and the bias/variance dilemma. Neural computation, 4(1):1–58, 1992.

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigates the double descent phenomenon within linear regression, particularly focusing on overparameterized least squares scenarios. It presents heuristic arguments and simple mathematical formulations to explain the occurrence of double descent. The authors dissect the prediction error into bias and variance components, further breaking down variance into three factors: the inverse of singular values, alignment with training sample directions, and model class limitations. While exploring how these factors contribute to double descent, the paper highlights the need for all three to be significant for the phenomenon to arise, noting the presence of bias in overparameterized settings and its potential impact on training loss.

### Strengths
1. The paper is well written and easy to follow.
2. The math is simple and results are all intuitive.

### Weaknesses
1. The paper's main contribution appears to rearticulate established "double descent" phenomena using simpler language, but it's unclear how this advances understanding beyond existing literature. Notably, there is an extensive body of work on double descent in the context of Kernel Ridge Regression (you can think of linear regression as an special case), such as the detailed discussion in Montanari et al. (https://arxiv.org/pdf/2308.13431.pdf), and references there.  
2.  Decomposing the risk to examine the influence of individual components is a natural thing to do. The practicality of quantifying these terms in real-world applications remains questionable in your work. The paper does not seem to provide a more insightful analysis of risk behavior compared to prior studies that make assumptions on data to yield meaningful interpretations. It is essential for the paper to clarify how its approach contributes to the existing knowledge base in a way that is both significant and applicable.

I think this analysis is a nice way of thinking about double descent, like for a blog post or a lecture note, but I do not see any significant contribution.

### Questions
See weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the double descent phenomenon in linear regression at the interpolation threshold. The authors decompose the test error into 3 components: 1. modelling (or irreducible) error, 2. bias and 3. variance.

The analysis provided is based on elementary linear algebraic arguments.

### Strengths
The problem considered is of broad interest in machine learning theory.

### Weaknesses
The main question is not formed well enough, as a result the main message is described only at a high level. This is unsatisfactory from a technical point of view.

The authors set out with a goal to understand why test loss diverges at the interpolation threshold, but the analysis does not provide a crisp enough answer to warrant looking at the simplified model of OLS.

The arguments are not statistically precise, which makes them hard to trust.

The points where the discussion is high level, e.g. the geometric intuition, the writing is hard to follow and the authors do not get their point across satisfactorily.

The discussion around prior work is very limited and does not fully and clearly explain what was known previously and what gap is this filling in the existing literature.

### Questions
The plots contain number of samples on the x-axis which leads to an inverted double descent curve (overparameterized models on the left of the threshold, underparameterized on the right). It might be better to plot the same with 1/n on the x axis, as is common in prior works, or state this different choice very clearly and justify it. The x-labels, y-labels, and plot titles, are too tiny and unreadable, please increase them.

The models considered are regularized (singular value thresholding). This is somewhat equivalent to ridge regression, which we know does not have the double descent if tuned optimally (Nakkiran et al. 2020, Optimal Regularization Can Mitigate Double Descent). As a result, many of the models considered in the overparameterized regime do not interpolate.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
