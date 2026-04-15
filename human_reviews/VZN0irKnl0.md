# Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos

- Decision: Accept (Poster)
- Scores: 8, 6, 5, 6

## Abstract
In gradient descent dynamics of neural networks, the top eigenvalue of the Hessian of the loss (sharpness) displays a variety of robust phenomena throughout training. This includes early time regimes where the sharpness may decrease during early periods of training (sharpness reduction), and later time behavior such as progressive sharpening and edge of stability. We demonstrate that a simple 
-layer linear network (UV model) trained on a single training example exhibits all of the essential sharpness phenomenology observed in real-world scenarios. By analyzing the structure of dynamical fixed points in function space and the vector field of function updates, we uncover the underlying mechanisms behind these sharpness trends. Our analysis reveals (i) the mechanism behind early sharpness reduction and progressive sharpening, (ii) the required conditions for edge of stability, and (iii) a period-doubling route to chaos on the edge of stability manifold as learning rate is increased. Finally, we demonstrate that various predictions from this simplified model generalize to real-world scenarios and discuss its limitations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper analyzes four phases of training NNs (early time transient, intermediate saturation, progressive sharpening, edge of stability) from the perspective of the top eigenvalue of the Hessian of the loss. The authors show that these dynamics depend strongly on the parameterization of the network (width; standard vs. maximum parameterization) and the optimizer settings (large vs. small learning rate). This is not only done experimentally, but also in a linear toy model, the UV model. For this model, a fixed point analysis of the gradient dynamics can be performed, which shows that even this simple model can exhibit all four stages.

### Strengths
The paper is mathematically dense and, to the best of my knowledge, rigorous. The analysis of the edge of stability manifold as a chaotic attractor is indeed interesting, and so is the observation that EoS is correlated with improved performance. The analytical model for the UV system is valid and accessible. To the best of my (very) limited knowledge in this regard, this is the first analysis of this kind. Being not familiar with the related work, however, I am not very confident in my assessment.

The paper contributes to a better understanding of training neural networks by carefully studying a simple toy system and by showing that fully-connected NNs also exhibit similar properties. Via this connection, one can argue that the theoretical analysis indeed sheds light on the internal workings of NN training.

### Weaknesses
Among the weaknesses I would count (and please excuse my ignorance in this field):
- The analysis in Sec. 5.2 is quite simple and appears obvious or at least little surprising. While the observation that the EoS manifold is an attractor is quite relevant, the remaining statements (Result 1, Corollary 5.1, bifurcation diagrams) appear straightforward consequences of this observation. This is not a weakness of the paper in itself, but this observation could be used to shorten this section.
- It is not immediately clear how the observations made in this paper will affect deep learning engineering practices. In other words, some actionable insights would be appreciated, linking the insights from the paper to recommendations regarding learning rate, parameterization, and architectural choices.

### Questions
- What actionable insights can be derived from the analysis?
- Does Section 6 rely on the trace of the Hessian, or does it analyze its top eigenvalue? If it relies on the trace, can it be ensured that the qualitative picture is the same (i.e., do the results from Appendix C.5 for the UV model carry over to more complex architectures)?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work investigates training dynamics of a simple UV model and shows that it replicates several phenomenological properties that have been observed in larger networks. In particular, the same 4 "phases" of training are observed. The authors use the tractability of the UV model to identify why this occurs and when/why some of these phases can be "missed". The authors then show that some predictions from the UV model are found in larger models.

### Strengths
1. The paper is well written and motivated. 

2. The numerical experiments are well designed and insightful. 

3. The main conclusions of the analytical work are well explained and relatively easy to grasp. 

4. The goal of understanding training dynamics is an important one that is well aligned with the NeurIPS community.

### Weaknesses
1. The use of simplified models is important and I appreciate that the authors demonstrated that the UV model, despite being very simple, has some of the same qualitative properties as what has been observed in other models. While the authors provide evidence that the features of their analysis can be used to understand training dynamics in larger models, I think there are a few weaknesses in these results: 
    a. First, I think it is worth noting that (at least as far as I could see), comparing Fig. 1 and Fig. 7 (FCN vs. UV model), there seem to be some differences in behavior of the loss and $\lambda$. In particular, there are oscillations in the training loss that do not appear in the FCN, and there is less of a spike in the training loss and $\lambda$ for say the model with $\mu$P (Fig. 7C, F). This makes it unclear how "similar" the phenomenology are between the UV model and the training phases that have previously been studied. One way to try to quantify the "similarity" would be to use metrics for estimating topological conjugacy between training dynamics (arXiv:2302.09160). 
    b. The authors show that $c$, $\sigma_W$, and $s$, which are predicted in the UV model analysis to play an important role in determining EoS are predictive of EoS in larger networks (e.g., Fig. 4). However, it seems that some parameters are much more important than others, e.g. for large enough value of $s$ any value of $c$ suffices to see EoS. This was not discussed in the main text and it was unclear to me why this would be. Similarly, the authors say there is a correlation between EoS and test accuracy, which is in my opinion weakly shown in Fig. 4. Maybe this is due to the fact that the color is saturated at 1? Or perhaps the way of estimating EoS ($\eta \lambda^H / 2$ ) is not the best way of estimating EoS? 
     c. The lack of period doubling in large networks is a very interesting point (and the idea that the data could be suppressing it somehow is really cool!), but to me this suggests again the possibility that there are fundamental differences in the dynamics between the UV model (trained with one example) and modern architectures that makes it unclear how generally valid the analysis performed is. Providing more discussion on this would be helpful. 

2. I did not follow how the authors got that the loss was $\mathcal{L}(\theta) = 1/2 \Delta f^2 = y^2/2 (\eta_c \lambda / 2 - 1)^2$ in Sec. 5 ("Connections to sub-quadratic loss"). Providing more detail would be useful. 

3. The title of the paper "Universal sharpness" is perhaps too strong. I did not feel like the authors ever really made a claim of universality, nor did they provide much analysis of it, in the main text. 

MINOR POINTS: 
1. I was under the impression that the role of loss landscape sharpness and generalization was called into question by Dinh et al. (2017). Is there a reason this was not cited in the Introduction when discussing loss landscape and generalization? 

2. In Appendix E, the examples used to validate the hypothesis about the weight norms is a 3-layer FCN, while Fig 5 has a 2-layer FCN. Is there a reason for these differences? Were the weight norms consistent with the hypothesis in 2-layer FCNs?

3. The authors mention that fixed point analysis is a generally powerful approach (Sec. 7). I think it would be good to discuss it's limitations more, particularly for nonlinear DNNs. 

4. Very minor, but I think putting references in parantheses would make the paper a little easier to read. I.e., "the early training dynamics [Kalra and Barkeshli (2023)]" (page 1).

### Questions
Q1. How truly "similar" are the dynamics in the UV model and the FCNs used to motivate this paper? 

Q2. Does the analysis performed by the authors demonstrate "Universal" properties?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper covers a tremendous amount of theoretical work, presenting a simple model which the authors analyze to study the role of initialization and parameterization on initial movements in sharpness during training; the role of initialization, parameterization, and learning rate on the existence of the Edge of Stability; and finally drawing analogy between the EoS and period-doubling bifurcations.

### Strengths
The Edge of Stability (EoS) is an important phenomenon observed in recent research, spurring questions about the understanding of dynamics for non-convex optimisation with deep neural networks. This paper studies the early dynamics of these networks, providing insights on the early movements in sharpness and the existence of EoS later on in the training trajectory.

The theoretical analysis is clear, and the resulting phase diagram in Section 4 appears novel, and is insightful and easy to understand. 

The applicability of the analysis appears general to a number of common parameterizations and can be used to draw insights to deep neural networks in general.

### Weaknesses
While this work presents impressive analysis, revealing insights toward important phenomena in optimization, the empirical support for individual claims are less apparent, possibly as a combination of limited empirical evidence and poor presentation. Additionally, the theoretical approach is hugely similar to existing work (Lewkowyz, 2020), though many additional insights revealed, especially concerning early dynamics. Finally, the connection of this work to common motivations for optimization analysis (e.g. generalization, feature learning) is indirect - only the suggestion that the existence of EoS can improve generalization error is presented in Fig 4 and discussed sparingly. 

The areas for improvement in empirical support are detailed as follows:
1. Section 6.1 claims that $\mu P$ networks begin training in a flat region of the landscape where gradients point to increased sharpness, while NTP/SP networks experience a reduction, the evidence in Fig. 1 suggests that this distinction is not so clear. 
2. Section 6.2 suggests that real-world deep neural networks (sometimes) undergo an interim decrease in $\lambda$ and the weight norm, but the real-world models presented in Fig. 1 and App. E exhibit a decrease in weight-norm. As real-world initializations typically use 0-mean weight initializations, one might expect that at least some of the models can be initialized in the R1 regime and immediately enter EoS. Therefore, the evidence that initialization in R1 can exist for real-world networks is limited, questioning the extent to which the theoretical model can apply to real-world models.
3. In general, the oscillation of EoS is normally observed around $2/\eta$, which is also stated in the paper in Sections e.g. 1 and 3. However, the point of oscillation predicted by the UV model (as seen in the phase diagrams) do not necessary have to be near 2/\eta. This distinction does not appear to be sufficiently addressed. 
4. Some of the empirical evidence discusses the effects of SGD, but it is an open question how and whether EoS phenomena translates to SGD. The methodology at which EoS is computed for SGD experiments in the Appendix are extremely unclear. An important scaling for the learning rate as $\eta = c/\lambda_0$ is stated, but not justified or explained.

### Questions
1) Under light investigation, the mathematical derivations for Eqs. 1 and 2 are sound. However, one might expect that the dynamics of the theoretical model might be independent to the sign of $\delta f_t$. Can you provide an intuition to why this might not be the case?
2) What is the justification for setting $\eta = c/\lambda_0$?
3) What is a 'low/minimal sharpness bias', as in Section 6.1?
4) What can I do to load this PDF more quickly?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies nonlinear gradient descent dynamics in a two-layer linear model trained on a single example. The model in question (a "UV" model) extends the model analyzed in Lewkowycz et al. (2020) by including a nonzero target scalar and incorporating different parameterizations (standard / neural tangent / $\mu P$) via an effective width parameter. The resulting gradient descent dynamics has a rich phase portrait with several stable/unstable fixed points/lines and different evolution patterns depending on initial conditions and hyperparameters. In particular, the authors describe the absence or presence of a catapult regime, sharpness reduction, progressive sharpening, edge of stability oscillations. For the edge of stability (EoS) regime, it is shown to occur on a particular line in the parameter space at learning rates above a particular critical value, and its origin is related to the well-known period-doubling fractal mechanism. After that, the paper discusses a number of predictions one can make for realistic models based on the obtained phase picture of the toy model: a correlation of the parameterization with sharpness, temporary decrease of the weight norm, conditions of EoS and sharpness fluctuations.

### Strengths
The paper is easily readable and clear. A substantial background and literature review are provided. A large number of appendices provide substantial additional details of the methodology, computations, and experiments. 

The toy model proposed in the paper is simple, but sufficiently rich to describe multiple complex effects observed in realistic models (catapult regime, sharpness reduction, progressive sharpening, edge of stability oscillations). The paper gives a systematic discussion of how these effects depend on the initialization, parameterization, learning rate.

The paper uses the toy model to conjecture several qualitative predictions on the learning behavior in realistic models/data. These are illustrated by multiple experiments with different networks and data.

### Weaknesses
I generally like the paper, but I'm not convinced in the sufficient novelty and significance of its results. The UV model studied in the paper is a toy model and is only a slight generalization of the well-known model from Lewkowycz et al. (2020). The study of this model is a fairly straightforward analysis of the 2D phase portrait of a simple $(\Delta f, \lambda)$ dynamics with respect to several hyperparameters. The paper does not develop any new analytic methods or prove any new general theorems. All the regimes discussed in the paper - the catapult mechanism, edge of stability, etc. - have already been demonstrated repeatedly in very similar models in multiple previous works (and I should give the credit to the authors for carefully listing them in two related work sections). The added value of the present paper - detailed analysis of the extended toy model and some predictions for realistic models - can certainly be of interest to a part of ML community, but just looks relatively marginal to me on the ICLR scale.

### Questions
Line 813: "constant width networks in SP with learning rate $\eta=\Theta(1/n)$ are equivalent to those in NTP Yang & Hu (2021)." - Where exactly is this shown in Yang & Hu (2021)? Table 1 and Figure 2 there actually show SP and NTP as different parameterizations.

### Soundness
3

### Presentation
3

### Contribution
3
