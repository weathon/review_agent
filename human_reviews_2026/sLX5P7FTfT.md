# Explaining Grokking and Information Bottleneck through Neural Collapse Emergence

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
The training dynamics of deep neural networks often defy expectations, even as these models form the foundation of modern machine learning.
Two prominent examples are grokking, where test performance improves abruptly long after the training loss has plateaued, and the information bottleneck principle, where models progressively discard input information irrelevant to the prediction task as training proceeds.
However, the mechanisms underlying these phenomena and their relations remain poorly understood.
In this work, we present a unified explanation of such late-phase phenomena through the lens of neural collapse, which characterizes the geometry of learned representations.
We show that the contraction of population within-class variance is a key factor underlying both grokking and information bottleneck, and relate this measure to the neural collapse measure defined on the training set.
By analyzing the dynamics of neural collapse, we show that distinct time scales between fitting the training set and the progression of neural collapse account for the behavior of the late-phase phenomena.
Finally, we validate our theoretical findings on multiple datasets and architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates learning phaes of neural nets focusing on the later training stages using wihtin-class variance as a key investigative lense.

### Strengths
* Important problem: Theory is generally lacking in this area
* Good alignment of theory and practice.
* Well written.

### Weaknesses
* There are some concerns regarding the suitability of in-class variance, which appear critical   (see questions).
* As for any theory, there are a number of assumptions.  and lack of common architectural elements
* There are concerns regarding novelty: Some of the findings appear known, some approaches seem to overlap with prior work, some key statements appear to be known, though not in this context and under a different analytical lense -  but the paper still of interest despite potential overlaps, but clarification is needed in this regard (see questions).
* Lack of practical implications (though not a must for a theory paper)

(Note, the initial judgement is more on the negative side due to a number of concerns - in particular the in-class variance as metric, but given questions are clarifyable an increase in score is appropriate as this is overall a strong paper)

### Questions
* How is this behavior if you include different kinds of normalization?
* Could you intuitively state the impact of assumption A.5 (in appendix)? In what sense, does it restrict initialization (e.g. compared to He or other initialization methods)?

* A variance decrease is also possible by simple scaling, e.g., say X are inputs to a layer then we have var[WX] for some weight matrix W, now if we scale X, i.e., we have cX with c<1, we have c^2*var[WX]. 
If you use regularization on weights you naturally scale weights and push them towards 0. So from this angle the metric appears not very suitable.


* You state "An intriguing observation in the existing information plane work is that, in the late stage of training, DNNs tend to compress I(Z; X) while preserving I(Z; Y ), thereby moving toward a more optimal solution with respect to the IB objective in Equation (1). 

This observation has already been made outside the IB framework, by looking at the reconstruction loss L(X,Z) over time (see [1] below). That is the paper argues that reconstructing the input X given intermediate representation Z gets better (L(X,Z) decreases) before increasing again (L(X,Z) increases), at the same time cross entropy CL(X,Y) loss only reduces, which is qualitatively identical to your Figure 1 showing I(Z,Y), I(Z,X), except that they also show a third phase, i.e., initially L(Z,X) remains constant.  Do you agree?

[1] Schneider, J., & Prabhushankar, M. (2024). Understanding and leveraging the learning phases of neural networks. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence (pp. 14886-14893).


* Why should in-class variance be a good measure? In particular, you show that it decreases, but it a better measure might something that changes across the phases ? Won't it? 

*  [2] (see below) argues that first the mean of the samples are fit, followed by those not well-aligned which might actually strongly impact representations though only minimally impacting the loss, which is fairly intuitive. Does not the initial convergence to the mean (per class) also apply to your formulation?   Essentially you minimize the population within class variance (Definition 3.1) if g(X) ~ Mean_(X,Y=c) g(X|c) ~ E_{X|Y=c}[g(X)].  In particular, the RNC1 score seems to measure just that and directly corresponds to the L2-norm used in [2]. Could you discuss the differences and similarities? 
The paper [2] argues that after the majority has been fit, learning slows down as but there can still be substanital changes -- the paper says a few (strongly different samples from the mean) can be repsonsible for this using the dot product. This seems to be very different from your approach as you are only concerned with the majority, are you? Or is it also relevant in your case ?

[2] Learning in NN: from fitting most to fitting a few. Neural Computing and Applications, 37(28), 23423-23446.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper attempts to provide geometric formulation for grokking’s delayed generalization. They do so by showing that grokking is the combination of a deep neural network following IB dynamics and achieving within-class variance collapse. It derives bounds linking this variance collapse to both generalization error and redundant mutual information, and shows empirically that variance collapse (RNC1) lags training-loss convergence under weight decay. The result is a heuristic unification of IB, grokking, and NC1.

### Strengths
This paper reframes grokking and IB as natural outcomes of a slowly emerging regularity principle (neural collapse). The paper is well-written, offers clear theorems that formalize variance–generalization connections, and provides empirical breadth.
The time-scale separation between fitting and collapse is a genuine insight for grokking dynamics, and the presentation is careful and reproducible.

### Weaknesses
However, the analysis treats neural collapse only through NC1, uses an one-sided IB inequality (Thm 3.4), and replaces Papyan 2020’s ETF-based NC2 metric with a condition-number surrogate that cannot ensure ETF geometry. The paper’s main claim is a unification of grokking and information bottleneck (IB) dynamics through neural collapse (NC). This is conceptually interesting but not rigorously established.

This unification is heuristic not proven: The entire IB connection rests on a one-sided variance-based upper bound (Thm 3.4) that does not prevent degenerate or trivial solutions (when compression is more important than capturing any relevant information). 

Theoretical results rely on strong simplifications (smooth activations, squared loss, pyramidal architecture, Assumption A.3, A.4, A.5). The theoretical guarantees in Theorem 4.3 depend on highly restrictive conditions (pyramidal architecture, smooth activations, squared-loss training) that differ from the experimental setups (ReLU + cross-entropy) . Theorem 4.3 and its proof explicitly assume optimization of the squared loss with weight decay, as given in Section 4.2.  All the convergence results refer to that MSE-based objective, not to the cross-entropy loss used in the experiments.

I also think replacing the NC2 metric with a condition-number surrogate without justification is sloppy. The co-occurrence of NC1 and NC2 is critical for the normative statement of neural collapse to hold (theorem 3 in Papyan 2020). NC2 is strictly defined as equal angles and equal norms both satisfied. While when condition number = 1, the perfectly isotropic mean configuration coincides with ETF. Many non-ETF configurations can also yield small condition numbers.

### Questions
1) Could flatness or implicit regularization, rather than NC1 dynamics, explain the delayed generalization you observe?

2) Is thm 3.4 a tight bound or a loose ceiling?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper demonstrates theoretically that the two training late-phase phenomena of grokking and information bottleneck (IB) dynamics are both due mainly to reduced population within-class variance.  It relates the measure of such variance reduction to the neural collapse degree present in the training dataset.  The paper shows that the model training duration ends sooner than the onset of the neural collapse progress, with the former dependent on the time taken to fit the training set and hence resulting in late-phase phenomena because neural collapse is responsible for such phenomena.  Experiments are conducted to validate its theoretical results.

### Strengths
This paper broadens the understanding of training dynamics by revealing the connection between late-phase phenomena and neural collapse.  It analyzes the variance of class-wise representations in the training data and shows that the decrease of empirical within-class variance leads to improved test accuracy, which takes place in the second compression phase to exhibit model grokking behavior after the first fitting phase. 

By considering the grokking and IB dynamics as two representative late-phase phenomena of DNNs, the paper is the very first to demonstrate that both phenomena can be explained in terms of the population within-class variance, offering new insights into their underlying mechanisms. 

The paper conducts a quantitative analysis on the discrepancy between the population within-class variance, allowing to evaluate the progression of neural collapse and implying a corresponding reduction in the population within-class variance to relate
the behaviors of grokking and IB dynamics to the development of neural collapse.

### Weaknesses
The experimental study for validating theoretical results is limited to DNN models with two classes, like the assumption of theoretical treatment.  It should be evaluated for the models targeting datasets with more classes, for generalization.

### Questions
Experiments on more general scenarios for models with more classes are desirable for result validation, since their results are not clear to hold.

### Soundness
3

### Presentation
3

### Contribution
3
