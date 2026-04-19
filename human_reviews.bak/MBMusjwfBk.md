# Prediction-Consistent Koopman Autoencoders

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 1, 1

## Abstract
Data-driven modeling of high-dimensional spatio-temporal dynamical systems, which are often governed by nonlinear partial differential equations (PDEs), poses a serious challenge in the absence of sufficient or high-quality training data. Recently developed Koopman autoencoders (KAEs) leverage the expressivity of deep neural networks (DNNs) and the spectral structure of Koopman operator to learn a reduced-order feature space exhibiting simpler linear dynamics. However, limited and noisy training datasets present a significant roadblock and results in a lack of generalizability due to inconsistency in training data. In this paper we propose the prediction-consistent Koopman autoencoder (pcKAE) which is capable of generating accurate long-term predictions even with limited and noisy training data. We introduce a consistency regularization term that enforces consistency among predictions at different time-steps, making pcKAE more robust and generalizable compared to its counterparts. An analytical justification is presented for such consistency regularization using the Koopman spectral theory. Experimentally, we demonstrate that with limited training data, pcKAE outperforms existing state-of-the-art KAE models for several test-cases, ranging from simple pendulum to kinetic plasmas, fluid flows and sea surface temperature data.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a method for predicting nonlinear dynamic using Koopman autoencoder neural networks. The authors suggest to augment existing techniques with a prediction constraint that promotes latent states to be linearly related. The method is evaluated on several datasets in comparison to several baseline approaches.

### Strengths
The paper is easy to follow; The method is described concisely and effectively; the evaluation section seems to be detailed.

### Weaknesses
The main weakness of this paper is the proposal of a loss term that was introduced and used before. Specifically, the paper by Lusch et al. "Deep learning for universal linear embeddings of nonlinear dynamics" discusses the same loss term (see page 4, 'linear dynamics'). Since the paper by Lusch et al., several other works have used a similar term, and it is generally known as one of the loss terms to be used in Koopman-based autoencoder frameworks. Thus, unfortunately, this work is not new from an algorithmic viewpoint.

Further, while the evaluation section is decent, the results are not promising. Specifically, the results in Tables 3, 4, 5 show that there is no statistical significance between cKAE and pcKAE (as measured by the standard deviation). This may explain why cKAE did not consider the additional loss term in their work.

### Questions
See above.

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a physics constrained autoencoder for time-series data forecasting. It builds on existing work done on Koopman Autoencoder and it's subsequent improvement consistent Koopman Autoencoder to propose prediction consistent KAE. The main idea of the paper is to use the time invariance property of the Koopman operator to enforce prediction consistency in the latent space. The final result is an added regularization term on top of the loss used in its predecessor cKAE.

### Strengths
Even though the ultimate contribution of the paper boils down to proposing a regularization term, the regularization itself is well motivated from a theoretical standpoint and is very clearly backed up by the improvements shown in the experiments. So, while the model may not be completely original it is definitely a significant improvement over its predecessors. The paper is also clearly presented and the experiments section is very thoroughly and fairly done. It is nice to see the confidence intervals and not just mean results being presented, and also full details of hyperparameter optimization being presented and fairly held the same across competing methods. The improvements in the results seem drastic and very significant!

### Weaknesses
I am not an expert in this field and had a hard time finding any weaknesses in the paper. But when I see a loss function of the form presented in Eq 12 it does make me wonder about both the data and time cost of grid search on those hyperparameters.

### Questions
I'd appreciate author's comment on the question raised in weakness section. 

From a visual perspective, it might also be worth adding a figure showing the predictive performance.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This manuscript introduced a prediction-consistent Koopman autoencoder (pcKAE) for predicting the behavior of dynamical systems. The authors state that by introducing the prediction consistency loss which satisfies the mathematical constraint, their pcKAE model leads to higher expressivity and generalizability. The authors provided some interesting findings on the dynamical system learning with the help of Koopman theory. The results seem to support the authors’ conclusion.

### Strengths
++ The incorporation of the prediction consistency loss improves the long-term predictability of cKAE. 

++ The paper is easy to read and understand.

### Weaknesses
-- The novelty of the paper is very limited. Adding the prediction consistency loss to the training process as a regularizer is not new, which has been used in many other similar models.

-- The experiments considered to demonstrate the capability of the model are rather simple. The authors should test the method on other complex systems, e.g., 2D/3D GS reaction-diffusion equations, 2D homogeneous isotropic turbulence at Re > 1000, etc.

### Questions
1. The authors should provide more creative thinking on the network structure or put forward some further theoretical analyses of the method instead of just doing some obvious mathematical derivation. For example, the authors may find it helpful to improve the latent space learning by adding Fourier transformation, e.g., introduced in [1], making the operator learning to focus on the frequency quantities of the dynamics. Such an improvement may make their work distinguish from the simple incremental one.

2. The author should design the experiments more carefully to prove the advantages of the mode. In particular, the experiments considered to demonstrate the capability of the model are rather simple. The authors should test the method on other complex systems, e.g., 2D/3D GS reaction-diffusion equations, 2D homogeneous isotropic turbulence at Re > 1000, etc.

3. The author should pay more attention on the writing. On page 4, the formulation of the forward loss, I guess, represents the $k$-step prediction of $\hat{\mathbf{x}}_{n+k}$. For different $n$, the symbol should have different meaning. Such a notation may lead to ambiguity. A similar issue exists on the formulation the backward loss of the same page. On page 7, the beginning of subsection 3.5, the authors used the same symbol to represent the train and test data by mistake. It's necessary to use more accurate notations in the manuscript for the sake of preciseness of scientific writing.

[1] Xiong, Wei et al. “Koopman neural operator as a mesh-free solver of non-linear partial differential equations.” ArXiv abs/2301.10022 (2023)

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes the predication-consistent Koopman autoencoder (pcKAE), which introduces a consistency regularization term that enforces consistency among predictions at different time-steps. It is capable of accurate long-term predictions with limited and noisy training data. The paper also presents an analytical justrification for consistency regularization using the Koopman spectral theory. The paper performs comparative experiments on 4 classic nonlinear systems or datasets to show its performance.

### Strengths
The paper proposes a new type of regularisation term, which contributes to better Koopman autoencoder for long-term prediction.

### Weaknesses
1. The statements or descriptions of the conclusions or technical points sometimes are fairly casual or inaccurate. We strongly recommend the authors to dive into the Koopman theory and address them carefully and rigorously. Like, "learn a reduced-order feature space exhibiting simpler linear dynamics", "Koopman operator maps *between* infinite-dimensional function spaces", "the dynamics can be linearly approximated", "by the finite-dimensional Koopman operator", etc. It seems the authors use the LTI system perspective to understand the linear property of Koopman operator, the dynamics of functionals (ie. observables) is linear, etc. 
2. Theorem 1, as the main result in theory, is not satisfactorily rigorous in math, which even involves mistakes (e.g., what is exactly the G? It is not matched with what claimed in its proof). If the latent space is just G, according to your math description, it is just a set of N_l functionals/elements?? And, the theorem does not carefully addresses the conditions of the flow or trajectory, x_t. The proof seems using the coordinates of the operator to show something, please address rigorous in math.  
3. The contribution of pcKAE may not be promising. As far as we can see, the paper contributes by introducing the so-called "prediction-consistency" regularisation term, which however is straightforward for enhancing the k-step predictiability.
4. The authors claimed the performance of pcKAE for long-term and high-dimensional prediction from noisy data. For long-term prediction, we assume pcKAE achieves it by increase the time-span in your proposed regularisation term, neglecting its practicability. For high-dimensional point, the authors seem misunderstanding this concept. Our nonlinear system eq.(1) is the state-space model without output equation, where the state is usually multivariate. High-dimensional (statistical) modeling or learning usually refers to such a task that the dimension of problem is so high that the data is deficit. Moreover, as indicated by the autoencoder (AE) word in pcKAE, is the dimension of latent layer N_l even smaller than the state dimension N_d? If so, yours actually deal with a very special case, where the dimension of the Koopman invariant space (that is large enough to model the given flow) is finite and rather small (smaller than the state dim). Actually the Koopman approach implicitly acqures the finite-dim lifted space is high enough, since it is used to approximate an infinite-dim space of functionals. Well, this is not argument for pcKAE only, it is for all KAE structures. The last point for noisy data: as we know for the Koopman setup, we are building the Koopman-based identification framework for state-space equation without process noise, as eq.(1); you have to be careful when addressing any properties or performance for noisy-data performance.

5. The comparative study is not enough. As the literature review has addressed, the paper has to show that pcKAE can really help to improve the performance of nonlinear system identification in the Koopman perspective.  There are many Koopman-based neural network models for time-series prediction. The only improvement over KAE showed in experiments may not convince readers of the values of pcKAE  for nonlinear system modeling.

### Questions
1. There seems mistakes in eq.(7), where there are matrix dimension-matching issues.
2. The propose loss function eq.(12) for pcKAE consists of so many regularisation terms, where how these regularisation parameters can be tuned in practice. Are the performance sensitive to the choice of these parameters?
3. What do you mean by "consistency" in your proposed regularisation term? It seems it is nothing related to the "consistency" in statistics or any well-known concept.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor
