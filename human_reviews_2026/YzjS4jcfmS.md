# How Muon’s Spectral Design Benefits Generalization: A Study on Imbalanced Data

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 8

## Abstract
The growing adoption of spectrum-aware matrix-valued optimizers such as Muon and Shampoo in deep learning  motivates a systematic study of their generalization properties and, in particular, when they might outperform competitive algorithms. We approach this  question by introducing appropriate simplifying abstractions as follows: First, we use imbalanced data as a testbed. Second, we study the canonical form of such optimizers, which is Spectral Gradient Descent (SpecGD)—each update step is $\mathbf{U}\mathbf{V}^T$ where $\mathbf{U}\mathbf{\Sigma}\mathbf{ V}^T$ is the truncated SVD of the gradient. Third, within this framework we identify a canonical setting for which we precisely quantify when SpecGD outperforms vanilla Euclidean GD. For a Gaussian mixture data model and both linear and bilinear models, we show that unlike GD, which prioritizes learning dominant principal components of the data first, SpecGD learns all principal components of the data at equal rates. We demonstrate how this translates to a growing gap in class balanced loss favoring SpecGD early in training and further show that the gap remains consistent even when the GD counterpart uses adaptive step-sizes via normalization. By extending the analysis to deep linear models, we show that depth amplifies these effects. We empirically verify our theoretical findings on a variety of imbalanced datasets. Our experiments compare practical variants of spectral methods, like Muon and Shampoo, against their Euclidean counterparts and Adam. The results validate our findings that these spectral optimizers achieve superior generalization by promoting a more balanced learning of the data's underlying components.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a simplified algorithm spectral GD, as a canonical algorithm to understand algorithms such as Muon, Shampoo, and SOAP. They adopt a linear model under squared-loss, using imbalanced classes as their data, where these classes have orthogonal means, and employ continuous dynamics or discrete dynamics that mimic the continuous case. The results show that for worst-class and class-balanced risks, when considering early phase loss decay, specGD is faster than GD with or without adaptive step-size (normalized gradient). They also prove that depth can accelerate the learning of minor components. Experiments are also provided under practical settings to validate the theoretical findings.

### Strengths
The paper provides a relatively thorough analysis of Spectral GD under imbalanced datasets, characterizing both the loss decay and the impact of network depth. The problem they studied is important towards understanding real applications, although under simplied settings. There is both theoretical and empirical evidence to support the claim.

### Weaknesses
The overall concern is that the setting of this paper is too strong, More precisely, the authors adopt the specGD, which is a perfectly preconditioned algorithm, while in practice, the main difficulty of adaptive algorithms is to design the preconditioning, and different preconditionings lead to different behavior. 

Regarding the theory, the assumptions seem pretty strong. The authors use linear model, orthogonal data, joint diagonalizability (another strong “orghogonality” assumption on data), zero (or near-zero) initialization, and gradient flow (or near-gradient flow).

### Questions
1. Please clarify the theoretical relationship between Spectral GD and practical adaptive methods such as Muon, Shampoo, and SOAP. In particular, does Spectral GD explicitly approximate their behavior along the dominant eigen-directions of the preconditioning matrix?

2. Equation (1): The definition of the inner product is unclear. Since \nabla L should have the same dimensions as W_t, the resulting inner product seems dimensionally inconsistent. Please provide a precise definition.

3. It would be helpful to explicitly present the update rules (for NGD, specGD, GD, and their gradient-flow versions) used in Theorems 1 and 2 for clarity.

4. The early phase and asymptotic behavior of specGD and (N)GD are different. It would be better if both can be included in the theorem to give more complete picture of these algorithms.

5. Section 3.3: Proposition 2 and its counterpart for the deep linear model do not seem clear. 

5.1 The requirements on \eta are not clearly stated. Theorems 1 and 2 analyze gradient flow, whereas Proposition 2 uses gradient descent. Please clarify the rationale and distinction between these settings.

5.2 The results hold when \delta\to\infty, which makes the current statement a bit wierd since you can then just assume zero initialization. The statement should be rewritten. 

5.3 The counterpart result for deep linear models is not formulated as a proposition. It may be clearer to combine these results into a single theorem and then discuss the role of depth.

5.4 The variable t appears in the results, but as t \to \infty, \bar{W} diverges.

5.5 There is a \Delta T that directly characterizes the time of learning different classes but this is not included in any theorems or propositions. 

5.6 Overall, section 3.3 needs to be reorganized.

6. Experiments compare toy algorithms (canonical algorithms) for theory, and the practical algorithms for real applications. However, it is unclear whether specGD can adequately represent Muon and Shampoo in practice, given their distinct preconditioning mechanisms. It would be better to include additional experiments that directly compare specGD with Muon and Shampoo for validation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper verified that spectral gradient descent learns all principal components of the data at equal rates, especially for imbalanced datasets. The conclusion extends to deep linear models and spectral optimizers including Muon and Shampoo, where the study concluded that these optimizers achieve better generalization through the balanced learning rates.

### Strengths
1. The study promotes the understanding of the success of spectral optimizers from a fundamental perspective, especially when the datasets are imbalanced in class or group.
2. The studies in this paper convered a wide range of neural architectures. The experimental designs are trustworthy.
3. Good story-telling from multiple perspectives.

### Weaknesses
1. This paper frames SpecGD as the canonical form for Muon and Shampoo. However, the practical implementations involve crucial components like preconditioner history accumulation  and various approximations/stabilization techniques. The paper could benefit from a more detailed discussion of how these practical elements might alter the "equal-rate" learning dynamics of the idealized SpecGD. 
2. The extended analytics to language modeling (next token prediction) is limited to rare and frequent tokens learning, while data imbalance on linguistics or knowledge and skills are investigated less in the study. 
3. Adam and other second-order preconditioning approximators should also be studied in simple linear models and image classification tasks, as its relation to second-order curvatures and simplicity in implementation and computation are still critical perspectives at present.

### Questions
1. The argument in Section 3.4 relies on assumptions of similar scaling between Hessian blocks (Theorem 6).  Could you comment on how violations of this assumption might affect the results? For instance, in real networks, do earlier layers and later layers exhibit vastly different spectral norms, and how would that impact the perturbation analysis?
2. Is it possible to construct an artifical dataset with a small proportion of domain-specific text corpus, e.g. mathematics or coding, and investigate if Muon exhibits a stronger learning progression in the domain, and if the learning dynamics generalizes to a knowledge-level?
3. While spectral optimizers may promote equal learning on imbalanced datasets, does the conclusion leads to a hypothesis that an ideal data preprocessing algorithm is equivalent to adopting Muon to Adam?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates when spectrum-aware optimizers like Muon and Shampoo generalize better than standard gradient descent methods by analyzing their canonical form, Spectral Gradient Descent (SpecGD). Using imbalanced data as a testbed under a mixture-of-gaussians data model, the authors prove for a linear model that SpecGD learns all principal components of the data at equal rates, unlike gradient descent which prioritizes dominant components. Paper shows that this leads to superior balanced accuracy early in training for linear and deep linear models under a Gaussian mixture data model. Authors also present experiments across image classification, natural language inference, and language modeling tasks demonstrate that Muon achieves faster generalization on minority classes and rare tokens compared to SGD, though Adam sometimes matches or exceeds Muon's performance.

### Strengths
1. The paper presents a clear empirical observation (Fig. 1) to motivate imbalanced data as a testbed, which is an effective approach.
2. The writing is clear, and the experiments cleanly motivate the theoretical arguments. The paper provides valuable insight in Theorem 1 on the gap between gradient descent and spectral GD (specifically between their continuous flow counterparts), showing that spectral GD outperforms GD.
3. The analysis extends beyond a single-layer linear network to a sequence of matrices, creating a deep linear network framework.
4. Experiments span several tasks (image classification, sentence relationship classification, next-token prediction), datasets (Dominoes, MultiNLI, TinyStories), and architectures, demonstrating that worst-group accuracy is generally learned faster with Muon. This is an interesting observation across various settings.

### Weaknesses
1. In the data model (DM), requiring orthogonal $\mu_i$ seems like a strong assumption. It is unclear whether orthogonal means for Gaussian mixtures is a standard assumption. What breaks down if this assumption is removed? In fact, in line 173, experiments are done with $\mu_i$ sampled uniformly from a Gaussian, so only $\mathbb{E}[\mu_i, \mu_j] = 0$ holds.
2. Why is joint diagonalizability a useful condition for the analysis? More importantly, are there other data models, theoretically or empirically, that would satisfy this condition? When does it break down? The paper does not discuss this condition extensively, but it would be briefly useful to discuss it in the text.
3. I am slightly confused about the takeaways for neural network training with non-linear activations. The analysis would be significantly harder, but the paper has a disconnect between presenting closed-form training dynamics for a linear model on a special data model (DM) and the motivation and experiments for deep neural networks on general image classification datasets. SVD analysis works for MSE minimization in a linear model, but would this be useful even in two-layer networks with a nonlinear activation?
4. In Proposition 2, the initialization for matrices is extremely specific, and it is unclear what all the values mean. This is unlike Proposition 1 and Theorem 1, where the initialization was zero. This makes me question the effect of initialization on the results.
5. Like the spectral dynamics visualized and derived for linear networks, the paper would benefit from a similar analysis in the experiment section that goes beyond just worst-group accuracy with different optimizers.

### Questions
1. Since the authors are arguing that Muon/Shampoo perform differently than Euclidean methods, can they add Adam to Fig. 1 (left) as well? Also add it to Fig. 3 and others. I am not sure what to read from the eigenvalue and eigenvector figures.
2. Is there any insight into what about the class imbalance/spectra dictates the speedup in generalizing on minority classes with Muon?
3. In Fig. 4, what is the structure of the linear model—just $Wx$?
4. In Proposition 1, there is a dependence on the iterate $t$ and step size $\eta$. What are the implications of the step size value?
5. A stretch question: why is momentum in Adam insufficient to leverage spectral information in the dataset? Experiments do show that Muon is not always better than Adam, e.g., in Fig. 8 at the end.

### Typos and Editorial Suggestions
1. The paper notes that Adam reduces to SignGD without momentum and preconditioning histories, which has been helpful for understanding Adam. Can the authors briefly mention here how Shampoo/Muon reduce to SpecGD and under what assumptions? This would make the writing parallel because in line 73, the authors argue for studying SpecGD as the canonical form for Shampoo and Muon.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tries to dive deep into the effectiveness of the optimizers which consider Spectral properties of the parameters like specifically MuOn. The authors study SpecGD which can be seen as an approximation of Muon optimizer. Authors show that the SpecGD method is effective on the imbalanced datasets, as it tends to learn the all the data moments at almost equal rate in comparison to NGD which tends to learn more aggressively from the data moments of the majority classes. They show that this effect is also prominent for deep networks theoretically, and show experimental validations for the same.

### Strengths
1.	The paper is well-written and quite comprehensive.
2.	Nice intuitive experimentation is done for each claim in theory which makes understanding easy and also shows real world applicability.
3.	The results of the paper uncover the reasons for effectiveness of the spectral optimizers like MuOn, with faster convergence on imbalanced datasets.

### Weaknesses
There is a line of work that shows that the generalization on imbalanced data could be improved by escaping saddle points. Hence, considering algorithms like Perturbed Gradient Descent (PGD) or Sharpness Aware Minimization could be used for optimization and show some connection with.

[R1] Jin, Chi, et al. "How to escape saddle points efficiently." International conference on machine learning. PMLR, 2017.
[R2] Rangwani, Harsh, et al. "Escaping saddle points for effective generalization on class-imbalanced data." Advances in Neural Information Processing Systems 35 (2022): 22791-22805.

### Questions
Could the authors explain more about the applicability of the data model they have considered with orthogonal means and specific variance’s generalizability in the real-world datasets?

### Soundness
3

### Presentation
4

### Contribution
3
