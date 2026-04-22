# Anchor–MoE: A Mean-Anchored Mixture of Experts for Probabilistic Regression

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 0, 4

## Abstract
We present Anchor-MoE, an anchored mixture-of-experts for probabilistic and point regression. A base anchor prediction is concatenated with the inputs and mapped to a compact latent space. A learnable metric window with a soft top-$k$ router induces sparse weights over lightweight MDN experts, which output residual corrections and heteroscedastic scales. Training uses negative log-likelihood with an optional held-out linear calibration to refine point accuracy. Theoretically, under Hölder-smooth targets and fixed partition-of-unity weights with bounded overlap, Anchor-MoE attains the minimax-optimal $L^2$ rate $N^{-2\alpha/(2\alpha+d)}$. The CRPS generalization gap is $\tilde{\mathcal{O}}\big(\sqrt{(\log(Mh)+P+k)/N}\big)$ under bounded overlap routing, and an analogous scaling holds for test NLL under bounded moments. Empirically, on standard UCI benchmarks, Anchor-MoE matches or surpasses strong baselines in RMSE and NLL, achieving state-of-the-art probabilistic results on several datasets. Anonymized code and scripts will be provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper describes a mixture of experts regression model (Anchor-MoE) for making probabilistic and point predictions. The method incorporates an optional anchor, a (point-estimate) regressor that can be freely chosen;   a gradient boosted decision trees model is used as anchor for the purpose of evaluations.  The experts are lightweight mixture density networks and a soft top-k router  operates on a latent space representation of covariate  inputs concatenated with the anchor prediction to produce k non-zero weights for sparsely combining experts. Theoretical analysis shows (under specified assumptions) that  minimax-optimal convergence rate for MSE can be achieved also covering the cases in which data is on a manifold or has sparse dependency on covariate components.  Bounds relating to CRPS and NLL generalization are also derived.  These theoretical results are used to guide some of the parameter choices in implementation. Ablation studies are provided for useful insights , for example to help understand the benefit of using an anchor.  Anchor-MoE is shown to perform competitively with other probabilistic regression methods, in particular NGboost, across standard UCI-benchmark datasets.

### Strengths
The paper presents significant new material: Demonstrating competitive performance of an MoE approach against other respected  mainstream probabilistic regression methodologies, in particular NGBoost, seems like an important result of interest to the ML community;  I have been unable to find other MoE performance comparisons of this nature.  The paper is further strengthened by derivations of bounds for MSE rates of convergence  and  bounds for CRPS and NLL generalization as well as capturing the high dimensional cases of data on manifolds and sparse dependency on covariate coordinates. I have again been unable to find such derivations/result in literature relating to similar MoE approaches.

### Weaknesses
The paper seems more inaccessible than it needs to be for non-domain experts who would benefit from fuller explanations and definitions in places. The reasoning underlying the construction of some of the  Anchor-MoE components could be better motivated for the non-MoE-specialist reader (through use of references and/or commentary) . Improvements in these areas would open up the paper to a wider readership. Convergence and related proofs in the appendix are very concise, taking for granted a significant amount of prior knowledge on the part of the reader and making it challenging and time consuming to gain confidence in their validity. Addressing these points would further strengthen the paper.

### Questions
1. How is the synthetic 1-d dataset constructed - for example are the oscillations in the point predictions recovered by anchor+MoE   in figure 4 also present in the true synthetic data and if not can you comment on why they might arise. It would be helpful to include the true expected values of the synthetic data on the plots

2. It’s unclear in the ablation analysis of table 2 precisely what has been added, changed or removed in each of the columns. Could this be expanded upon (both in the rebuttal response and  in the paper).  What is the message I should come away with from figure 3 ?  An expanded legend-text and/or main-text commentary of what the reader is meant to deduce from this figure would be helpful.

3. Do the covariate dimensions or other characteristics of the baseline datasets reveal anything interesting about where Anchor-MoE performs well or not so well in comparison to NGBoost and/or the other baseline methods?

4. Can you comment on how reasonable the assumptions A1-A3 and G1-G2 are in relation to  Anchor-MoE.

(Some  minor editorial points I detected: Several cross references are incorrectly made to section 3 instead of section 2. No bolding is provided in table 4.  The sentence at the end of the second paragraph of the introduction is missing an ending. In Lemma A1 shouldn’t the inequality be $\le$ rather than $\ge$?)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This work presents a new model for probabilistic and point-wise prediction, coined Anchor-MoE. Intuitively, the model is a two-layer hierarchical Gaussian mixture model, where the Gaussians are parametrized to predict the residual with respect to a shared "anchor" point, rather than freely predicting the mean, and which is calibrated afterwards. The authors meticulously present every small detail that defines their model, as well as involved theory proving the convergence rate and generalization bound of the proposed model. Finally, the authors empirically show the effectiveness of their model in a series of UCI datasets, comparing the proposed model with NGBoost, and ablating each part of the proposed model.

### Strengths
- **S1.** The combination of a hierarchical GMM with a common anchor is interesting and can be of interest to the community.
- **S2.** While I have not checked the correctness of the proofs, the provided results are interesting.
- **S3.** The provided empirical results are promising.

### Weaknesses
- **W1.** The writing needs significant changes. All the manuscript is confusingly written, with lots of keywords and terms that are never explained (e.g. CRPS), design decisions that are never justified (the paper is more declarative than anything else), and adjectives thrown here and there without further explanation (e.g. why is the latent space "compact"?).
 - **W2.** The structure of the manuscript is unconventional. The intro is in reality the related work section, the notation paragraph explains pretty much the entire approach, and the theory section seem completely independent of the rest of the work when it comes to the technical knowledge needed and employed.
- **W3.** There are many formatting issues: Citations are not properly formatted, there are no equation numbers, section numbers are wrong (all references to section 3 are rather section 2).
- **W4.** There is a significant lack of citations for all the keywords and terms unexplained in the paper, with a total of 19 references at the end of the main content. Two of these references are repeated, the reference of Stasinopoulos points to the R package rather than the actual paper, and that of Aad W. van der Vaart has a wrong publisher (Springer instead of Cambridge) and the doi link does not work.
- **W5.** The experimental section lacks a deep analysis of the results and, e.g., I cannot find where the conclusions drawn in the fourth paragraph of section 5 come from.
- **W6.** Regarding the methodology, it is rather overwhelming the number of options presented, without a clear choice of what approach to use (anchor/no anchor, cosine normalized or not, condition the expert or no, etc).
- **W7.** The experimental section lacks baselines, and the only one presented is not reproduced but taken from another work. The ablation study references to non-existent terms (No-Anchor, No-Router, No-Cal).
- **W8.** Results in the appendix do not have a proof at all (e.g. Lemma A.4 or Theorem A.5).

### Questions
Other feedback:
- On the point of the second paragraph of the intro, [another work](http://arxiv.org/abs/2203.09168) showed that explicitly modelling the distribution can also have the exactly opposite behavior, where better-learned points are prioritized over points yet-to-be-learned.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose Anchor-MoE for probabilistic and point regression tasks. Anchor-MoE is a combination of anchoring (with gradient boosted trees) and mixture of experts (Mixture Density Networks), although all modules have been previously proposed, the combination is novel and well thought-out. The authors further provide theoretical analysis on minimax risk rates and a bound on CRPS generalization gap. On UCI benchmarks, Anchor-MoE achieved comparable results to NGBoost on point regression RMSE with improved probabilistic regression NLL.

### Strengths
- The method is well-motivated and has solid theoretical backing.
- The combination of existing methods is novel, and the proposed framework is flexible with customizable modules.
- The empirical results are favorable for Anchor-MoE.

### Weaknesses
**[W1]** Figure 1 might be a misrepresentation of the proposed method.  
Currently, Figure 1 appears to omit several key components detailed in the text, which is vital for reader comprehension. Specifically, the figure is missing:
- The concatenation of the anchor prediction to the original input
- The projection from inputs to compact latent space
- The linear calibrator in the end (which is essential for the strong RMSE results)
- An arrow from x to Anchor
- This figure can benefit from a more detailed caption, to serve as a quick overview of the proposed method.

**[W2]** Computational and Capacity Analysis  
The paper currently lacks a thorough discussion on the computational overhead and parameter count of Anchor-MoE. Specifically, since there are quite a few pieces on top of the anchor model, it is a concern that the model is only bringing improvements because it has more capacity than the baseline used for comparison (NGBoost, Deep Ensembles, MC Dropout, etc.). It would be highly beneficial to see parameter counts, FLOPs, inference throughput, and training wall-clock time. 

**[W3]** Hyper-parameter Ablation  
This method has several tunable hyper-parameters, including number of MDN experts K, number of activated experts k, dimension of latent D. It would make the method more convincing if the authors can show how they chose these hyper parameters (and how expensive the hyper-parameter tuning phase is), alternatively it would be helpful if the authors can show empirical performance of Anchor-MoE with different choices of hyper-parameters. Without this analysis, the reported results may be limited to a narrowly tuned configuration.

**[W4]** The writing is confusing at times, examples:  
- There are several unexplained notations in Section 2 onder Notation paragraph, including $s_j$ and $\tau$, both were unspecified until later subsections. Similarly, Details were mentioned but never specified, including for example "mild entropy regularization" (line 81).

### Questions
See W1, W2, W3. I am willing to increase my rating if these can be addressed.

### Soundness
3

### Presentation
2

### Contribution
2
