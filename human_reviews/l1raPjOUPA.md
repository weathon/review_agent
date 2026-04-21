# Grokking at the Edge of Linear Separability

- Avg Score: 6.00
- Decision: Reject
- Scores: 5, 5, 6, 8

## Abstract
We study the generalization properties of binary logistic classification in a simplified setting, for which a "memorizing" and "generalizing" solution can always be strictly defined, and elucidate empirically and analytically the mechanism underlying Grokking in its dynamics.
Analyzing the final stages of training of logistic classification on Gaussian data with a constant label, we show that it may exhibit Grokking, in the sense of delayed generalization and non-monotonic test loss, when the parameters of the problem are close to a critical point. 
Specifically, we find that Grokking is amplified when the training set is on the verge of linear separability from the origin. Even though a perfect generalizing solution always exists, the implicit bias of the logisitc loss will cause the model to overfit if the training data is linearly separable from the origin. 
For training sets that are not separable from the origin, the model will always generalize perfectly in infinite time, but overfitting may occur at early stages of training. 
Importantly, in the vicinity of the transition, that is, for training sets that are almost separable from the origin, the model may overfit for an arbitrarily long time before generalizing.
We gain more insights by examining a tractable one-dimensional toy model that quantitatively captures the key features of the full model. 
Finally, we highlight intriguing common properties of our findings with recent literature, suggesting that grokking generally occurs in proximity to the interpolation threshold, reminiscent of critical phenomena often observed in physical systems.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work examines grokking, a phenomenon recently observed in neural network training. The author focused on a simple model for data generation and demonstrated that the occurrence of grokking depends on the ratio between data dimensionality and sample size. This analysis relates to research on implicit bias.

### Strengths
This work addresses an interesting and important problem. The analysis relates to key topics in understanding neural networks, such as implicit bias and overfitting.

### Weaknesses
1. The paper lacks clear presentation and organization. The connections between the analyses in each section are unclear. Without theorems or propositions in the main body, it is difficult to identify the main conclusions. It is recommended to add a roadmap or flowchart at the end of the introduction to outline the paper's structure, or to include summary statements at the end of each major section to better connect the analyses. Additionally, it is suggested that the authors present key findings as numbered theorems or propositions in the main text to make the primary conclusions more prominent.
2. The logistic loss classification with a single-label model studied in this work is overly simplistic, making it difficult to intuit how the analysis could be extended to other models. It is recommended to discuss potential extensions or limitations of the approach, such as elaborating on its limitations and future work. This could include addressing how the analysis might (or might not) generalize to more complex models.

### Questions
1. On page 2, the main contributions are summarized in five points, but only the first point mentions grokking. It’s unclear how the remaining four points relate to grokking. Please consider explicitly connecting each contribution to the central theme of grokking.
2. The single-label model is uncommon; it’s unclear how the analysis of this model extends to others, such as models with binary labels.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper considers a binary logistic regression with isotropic Gaussian inputs and constant (same) labels to study Grokking phenomenon (Power et al., 2022). For the analysis, the authors utilize prior results (Soundry et al., 2018; Nacson et al., 2019; Ji and Telgrasky, 2019) about the learning dynamics of logistic regression. The simplified nature of the setting allows the authors to illustrate Grokking and how it appears in detail. Specifically, they show that Grokking is related to linear separability and whether the data is linearly separable from the origin. Furthermore, they find an "interpolation threshold" for the ratio of input dimension to the number of samples and demonstrate that Grokking occurs near this threshold.

### Strengths
- **Studies interesting but simple setting for Grokking**, which is good for understanding the Grokking phenomenon. 

- **Provides insights about Grokking:** Specifically, the authors show the relationship between linear separability and Grokking for their setting

- **Easy-to-follow paper:** Although there are some issues (mentioned below) about the writing of the paper, it is mostly easy to read.

- **Detailed examination of the considered setting:** While their setting is oversimplified, the authors investigate Grokking in this setting in a detailed way.

### Weaknesses
- **Oversimplified setting:** Due to the oversimplified nature of the setting, it seems like most of the results in this paper are trivial extensions of prior results (Soundry et al., 2018; Nacson et al., 2019; Ji and Telgrasky, 2019) about the learning dynamics of logistic regression. The authors also mention their usage of (or relationship with) these prior results in several places. Therefore, the only additional benefit lies in the particular focus on Grokking. This limits the contribution of this paper. (please see Question 2 below as well)

- **Missing related work about Grokking:** There seems to be missing prior work studying Grokking under more realistic settings (Humayun et al., 2024; Golechha, 2024). Comparison with these works would be beneficial. (please see Question 3 below as well)

- **Wrong jargon usage in the writing**: 
  + In the abstract (Line 12), the authors write, "We analyze the asymptotic long-time dynamics of logistic classification on a **random feature model**." Similarly, they mention the "random feature model" in TL;DR as well. However, they study logistic classification using a linear model. By the "random feature model," they meant to say that the input distribution has a Gaussian distribution with isotropic covariance. However, the "random feature model" corresponds to a different model in literature (Rahimi and Rect, 2007). 
  + The authors say "the network" when referring to the linear model throughout the paper, which is confusing since there is no network in their setting. 
  + The authors write "long times" or "late times" to refer to the "final stages of the training", which is also not common practice in the literature.

- **The introduction does not mention the model and loss function of the setting.** These should be clarified before the contributions are mentioned.

Overall, I think this work should be extended to a more general setting, or at least the authors may show how the insights from the current work translate to examples of more complicated settings, such as more complex data (non-isotropic Gaussian inputs). Furthermore, related work should be extended to include the prior works that are missing and a comparison with them. Finally, the mentioned issues about the writing should be fixed. 

(Rahimi and Recht, 2007): Random Features for Large-Scale Kernel Machines (NeurIPS 2007).

(Humayun et. al., 2024): Deep networks always grok and here is why. (ICML 2024)

(Golechha, 2024): Progress Measures for Grokking on Real-world Tasks. (HiLD 2024)

### Questions
1. What do the authors mean with $\lesssim$ in Line 215? 

2. How do the insights from this work extend to more realistic settings? 

3. What is the additional insight of the current work compared to the missing related work (Humayun et al., 2024; Golechha, 2024)?

**Minor Issues**
- The cross-entropy loss in Eq. (1) does not include labels of data, which looks odd at first glance. The authors drop the labels from the equation since all inputs are assigned the same label. However, this should be mentioned as a note near the equation to improve clarity.
- To avoid confusion with empirical accuracy, mentioning "generalization accuracy in Line 137 might be helpful.
- Typo in line 163: "we the present"

**Update:** After the rebuttal, the rating was increased from 3 to 5 since the authors addressed the concerns during the rebuttal.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies a toy model for grokking in binary classification with a linear model trained using gradient flow on gaussian data with constant labels. They argue that grokking occurs when the ratio $d/n$ of parameters to datapoints approaches the critical threshold of ½ — this is the threshold where the data undergoes a sharp transition from being linearly separable to linearly inseparable. Using prior work characterizing gradient flow dynamics for logistic regression, they argue that close when $d/n$ is just barely below ½, it takes a long time to converge to the max-margin separating hyperplane, which explains the sudden shift in test accuracy even though train loss is low.

Edit: After considering the improved writing and new theoretical results in the revision, I have updated my score to a 6.

### Strengths
1. It is important to have toy models that we can actually analyze. Even though the model setup is admittedly toy, it still demonstrates interesting train and test time phenomenon.
2. The paper identifies that the margin to linear separability of the data from the origin determines whether the dynamics will grok or not.

### Weaknesses
1. I found that the paper was hard to read, and at times imprecise. One suggestion I have is to write the paper in a way where it is easy to identify what concrete properties or statements are being proved. 
2. For example, in Line 251, I was originally confused about why can we apply Eq (7), since that result only makes sense for the linearly separable case (especially since it’s talking about hard SVM). It might help if you formally stated the result of Ji and Telgarsky was stated and then also your variant where you include the bias, commenting explicitly on the difference between what “linearly separable” means in these two settings. 
3. It seems that the problem setup is (WLOG) assuming that all the labels are -1, so that to classify correctly we should output a negative scalar. Can you explicitly state this (for example, by just defining what the labels are)

### Questions
1. Line 197, you should specify that the first 0 is actually the zero vector in $d$ dimensions
2. I know that the Heaviside function $\Theta$ is relatively standard, but could the authors just define it when it is first introduced for readers who might not be familiar. (Minor point, but I feel like it would be more clear anyway if this was just defined as an indicator)
3. Could you change this $\lambda \lesssim \frac{1}{2}$ notation? This might be interpreted as $\lambda$ is bounded within a constant factor of $\frac{1}{2}$. Two candidate suggestions: $\lambda \uparrow \frac{1}{2}$ ot $\lambda \to \frac{1}{2}^-$. 
4. The paper cites Wendel’s theorem, which gives an asymptotic 0-1 law for the separability of the training data. Is there a non-asymptotic version of this? 
5. Appendix A, line 617. I believe you are missing the word “linearly separable”
6. In Appendix B, it is argued that the parameter $\alpha$ in Power et al. is analogous to that of $\lambda$. However, besides the fact that there is a “phase transition” with respect to these scalar parameters, I don’t see how one can connect them. Could the authors comment on this connection more,

### Soundness
2

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
4

### Summary
The paper identifies a simple linear classification setting and studies when "grokking" occurs, under Gaussian random data in the linear asymptotic regime. The setup is simplistic, but possibly useful to isolate the emergence of grokking as a result of implicit bias of gradient descent (flow). Relying on analytical tools including simple Gaussian error calculations and advanced implicit bias results from prior works, the authors demonstrate the regime of occurrence of the phenomenon. The results could be of interest to the theoretical statistical learning community, and the insights may be interesting also to the deep learning community.

Update after author response: raising score from 6 to 8

### Strengths
S1. The phenomenon of grokking has been of interest as one of the intriguing behaviors of neural network training. As remarked by the authors, this study does not need sophisticated assumptions, unlike some prior work, for demonstrating the phenomenon. The setup is short and the results are clear. 

S2. The analytical discussions show a useful application of implicit bias results from literature. While I did not check the derivation in detail, the calculations are sound.

S3. The paper is very well written. The contributions and limitations are clearly discussed.

### Weaknesses
To be of greater interest to the current machine learning community, the study may need to address the following weaknesses. That said, in the considered setting, the results are self-contained and interesting on their own.

W1. The impact of the study would be much higher if the analysis can be shown to generalize to more complex models, like discriminative labeling, and non-linear models. 

W2. From a practical point of view, having a theoretical model explain an empirical observation is a valuable contribution. In this regard, demonstrating direct counterparts and specific usefulness of the insights obtained from the model to large scale machine learning experiments may be considered. For example, does the "edge of separability" correspond to a certain regime of model size/training in deep models? Can a specific trend in delayed generalization be predicted or explained by the insights obtained?

### Questions
Questions:

Q1. The train/test loss and accuracy behaviors are still not fully understood. This paper takes a step in the right direction. While I appreciate simple models yielding interesting insights, from a practical perspective, the study would be much more impactful if the results can be extended to more realistic settings. Some thoughts and questions below:
	
	a. What are the additional results needed, if one were to try and extend the study to random example classification, but with multiple classes? Perhaps the smallest such extension could be iid normal features and random binary label assignments. From my understanding, there are similar linear separability threshold computations possible (e.g. Kini et al, 2021). Any discussion on this question could be beneficial.
	b. The authors note the study of non-linear regression, motivated by neural network training, to be a feasible extension, with fixed learned last-layer features and a trainable classifier head. My concern here is that it is unclear how the distributions of the last-layer features could be characterized. The features will likely not have nice Gaussian-like properties. Additional modeling of the learned feature extractor may be needed and may complicate the calculations.

Q2. I am trying to reconcile some of the results, and more generally, grokking itself with double descent. For example, one of the statements (line 56) is that in the separable regime, the generalization is worse. On the face, this seems to be different from what double	descent appears to predict. Is the apparent discrepancy stemming from not having any "feature selection" assumption in the model? Does double descent not occur in this linear model? More generally, both the phenomena seem to be connected, in the sense of being a result of the implicit bias of the loss optimization. A discussion elaborating the connection would be helpful to get a better picture of the learning dynamics. Specifically, I am curious about the simplest model that would exhibit both double descent and grokking, which might take the model closer to complex DNNs. A recent work (mentioned by the current paper) Davies et al, 2023 discusses the link between the two.

Q3. The discussion in Section 4 is not fully clear to me. It seems that the authors are constructing a representative deterministic setting in 1D, with the deterministic points constructed to match the macro statistical properties of the random feature model. Such a description could be interesting, with a bit more clarity. For instance, the overparameterization ratio $\lambda$ becomes a parameter of the value of a datapoint. With $d$ being set to 1, $\lambda$ appears to lose its original meaning. Could the authors help explain? The paragraph in App E.1 was somewhat helpful.

### Soundness
3

### Presentation
3

### Contribution
2
