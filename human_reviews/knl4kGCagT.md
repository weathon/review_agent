# ON TRAINING DERIVATIVE-CONSTRAINED NEURAL NETWORKS

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3

## Abstract
We refer to the setting where the (partial) derivatives of a neural network’s (NN’s)
predictions with respect to its inputs are used as additional training signal as a
derivative-constrained (DC) NN. This situation is common in physics-informed
settings in the natural sciences. We propose an integrated RELU (IReLU) acti-
vation function to improve training of DC NNs. We also investigate denormal-
ization and label rescaling to help stabilize DC training. We evaluate our meth-
ods on physics-informed settings including quantum chemistry and Scientific Ma-
chine Learning (SciML) tasks. We demonstrate that existing architectures with
activations replaced with IReLU activations combined with denormalization/label
rescaling better incorporate training signal provided by derivative constraints.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a series of technical improvements to better train neural networks with derivative constraints that are common in physics-informed settings. A new activation function, the integrated ReLU (IReLU) is proposed, along with other fixes such as denormalization and label rescaling.

### Strengths
- **The paper addresses an important open problem and presents concrete improvements over a comprehensive set of experiments.** Training derivative constrained networks is central to solving a lot of the physics-related applications, and it can be difficult in practice since it is hard to balance the derivative constraint in the total loss function. This paper proposes methods that address this difficulty by respecting the different numerical scales of the system. Improvements across many different network models as well as applications (quantum chemistry, fluid dynamics, diffusion-reaction, etc.) are shown in the experiment section.
- **Related work is discussed properly.** The authors adequately address the existing literature on derivative constrained neural networks and points out that their work differs from the literature in that they try to find general drop-in replacements that are suitable for a variety of different tasks and architectures.

### Weaknesses
- **The organization of the paper has room for improvement.** There is a motivation section in which the authors write about the experiments in detail, including the dataset size, numerical scales and units of the quantities of interest, but these details do not directly contribute to motivating the proposed methods. 
- **Some observations are left unexplained/unexplored.** In section 3 results, the authors observed that the "energy loss divided by 1000 is typically much lower than the force loss," and that "it is not easy to improve the relative difference, even for large values of $\beta$." Why should this be the case? Why does the energy loss in figure 1(b) show high variance with high $\beta$? Despite this being mostly a methods paper, I wish there was more discussion providing insight and analysis on these empirical observations.
- **Some intuitions can be made more mathematically precise.** It is a good thing that the paper provides lots of intuition and insights on the problems and solutions presented. However, some of these intuitions can be difficult to understand for readers not familiar to the subject. For example in section 4.2 "...DC NNs are more sensitive to *units* compared to typical training without derivative constraints because of the linearity of derivatives", and "we can interpret the constant $c$ as determining the *units* of $x$". It is still unclear to me why some of the internal normalization (e.g. batch normalization) will not respect the units. It would be greatly helpful if the authors would include a short example of a concrete model and toy data to further elucidate this point.

### Questions
- On the results in section 3: why does it makes sense to divide the energy loss by 1000 in the first place? Would it be more reasonable to somehow plot the "variance explained by the model" instead of the unscaled losses?
- How is label rescaling different from the usual dataset-preprocessing? In your opinion, why didn't previous works consider this simple procedure?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper tackles the problem of unstable training of derivative-constrained (DC) neural networks. Specifically, the authors proposed IReLU activation function to replace ReLU, which may lose training signals in DC settings. Furthermore, they also proposed eliminating normalization layers in current networks and rescaling the labels to reduce the sensitivity of the network with respect to units. Experiments were conducted on on a variety of architectures, datasets, and tasks including quantum chemistry and physics-informed neural networks.

### Strengths
* The paper tackles an important problem of stabilizing training of derivative-constrained neural networks. 

* The writing of the paper is clear. 

* Experiments are conducted on a wide range of tasks and architectures.

### Weaknesses
* The authors should introduce a name for their proposed method. 

* The ideas of the papers are quite incremental and mostly based on intuition. For example, the authors hypothesize that derivative-constrained NNs are sensitive to units without any theoretical/empirical evidence.

### Questions
1. The experimental results are questionable, especially ones with quantum chemistry neural networks where we can observe huge improvements. Can the authors provide more detailed clarifications/explanations for these improvements?

2. Why label rescaling is not necessary for PINN experiments? Does this mean the label rescaling is only designed for quantum chemistry experiments? Again, the results on quantum chemistry datasets are questionable.

3. Have the authors tried alternatives of ReLU, such as SiLU, ELU, etc.?

4. I would appreciate if the authors included ablation studies to show the effectiveness of each component: iReLU, denormalization, label rescaling. 

5. Normalization techniques were proposed to improve stability when training neural networks. Why removing them in the paper's setting can improve training stability? I would expect a more detailed answer other than the intuition of sensitivity to units mentioned in the paper.

6. Can IReLU be applied in derivative constraints involving higher-order derivatives, such as second-order? If not, can we have a more generic activation function for such cases?

I look forward to the authors' response and will be happy to increase my score if the authors can address my concerns.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Motivated by the convergence discrepancy of the loss terms in physics-informed training of neural networks, the paper proposes a new activation function, IRELU, and a label rescaling method, while abandoning common normalization techniques.

### Strengths
- The paper tries to address an important challenge with PINNs, concerning the discrepancy between loss terms in physics-informed training. 
- The direction taken by authors in focusing on the units of derivatives and labels is interesting.

### Weaknesses
- As also pointed out by the authors, the proposed IRELU activation has limited usability in physics-informed models, where one might need derivatives of an arbitrary order w.r.t. inputs, while derivatives for IRELU are $0$ for third and higher order derivatives. Even for second order PDEs, the effects of a constant second derivative of $1$ in IRELU (for $x>0$) need more attention and study. 
- Preventing vanishing and exploding gradients is one major characteristic of RELU. The gradient propagation of IRELU is not studied, though. As in your experiments, physics-informed training usually involves training with the solution data as well (IC, BC, etc.), where the first order derivative of IRELU ($=x$ for $x>0$) appears in the optimization. This is concerning for exploding gradients, especially since the paper also abandons normalization.
- The notion of the 'difficulty of learning derivative-constrained info based on the loss scaling term' is inaccurate. While convergence discrepancy of different loss terms is known to happen in physics-informed training, the convergence rate is shown to be in favor of the residual loss (derivative-constrained term) in some cases [1].
- Other works have studied activation functions in the physics-informed setting before [2, 3]. Lack of review and comparison with such works is surprising. Moreover, the authors do mention the adaptive loss scaling methods, but, there is again no comparison with those methods.


[1] Wang, S., Yu, X., & Perdikaris, P. (2020). When and why PINNs fail to train: A neural tangent kernel perspective. ArXiv. /abs/2007.14527

[2] Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33, 7462-7473.

[3] Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020). Adaptive activation functions accelerate convergence in deep and physics-informed neural networks. Journal of Computational Physics, 404, 109136.

### Minor Comments
- There are a few grammatical errors, and the readability can also be improved.
- In Sec 3.2, Results, the references to Fig. 2 seem to be meant for Fig. 1.

### Questions
1. A more in-depth study of the proposed activation function would be really helpful. Authors may want to explain what characteristics IRELU shares with RELU and how it addresses the issues with polynomial activation functions.
2. The presentation and readability can be greatly improved by adding more plots instead of tables; Especially, in the experiments and ablation study to show how the proposed methods contribute to addressing the loss discrepancy. Also, the plotting style in Figures 1a and 1b is not informative and rather confusing. 
3. Section 4.2 is very limited in justifying the proposed rescaling method and how it improves the learning of the derivative-constrained info. I would appreciate more details and insights regarding the choice of $C$ and why normalization methods are discouraged.
4. As mentioned in the Weaknesses, comparison with other activation functions that are designed for or tested with PINNs is crucial.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
