# DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Dataset bias often leads deep learning models to exploit spurious correlations instead of task-relevant signals. We introduce the Standard Anti-Causal Model (SAM), a unifying causal framework that characterizes bias mechanisms and yields a conditional independence criterion for causal stability. Building on this theory, we propose DISCO$_m$ and sDISCO, efficient and scalable estimators of conditional distance correlation that enable independence regularization in black-box models. Across six diverse datasets, our methods consistently outperform or are competitive in existing bias mitigation approaches, while requiring fewer hyperparameters and scaling seamlessly to multi-bias scenarios. This work bridges causal theory and practical deep learning, providing both a principled foundation and effective tools for robust prediction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors consider a 'Standard Anti-Causal Model' which is defined as a causal graph where predictions $\hat{Y}$ are derived from observed features X, which in turn can be influenced by confounding, collider variables, and the true label Y. (The label Y causes X, thus a predictor for $\hat{Y}$ makes predictions in the anti-causal direction). For predicting the true label Y, the authors suggest to only consider direct causal effects, without the influence of indirect or biasing factors. This is expressed as a causal stability condition. Eventually, a training goal is formulated as a standard loss minimization problem under the constraint that a to-be-trained model prediction becomes independent of the effects mediated through confounding or collider paths given the true label.

The authors propose a conditional distance correlation measure to approximate the independence of two variables from each other, and approximate the via a kernel regression from data. In the following two separate "DISCO" methods are defined to approximate the degree of conditional independence between the true and predicted target, given the biasing features.

Experiments are conducted on 5 (semi-)synthetic datasets, featuring different types of confounding and collider bias and generally seem to show the well working of the method and against several existing baselines.

### Strengths
The paper presents a novel approach for regularizing arbitrary ML model trainings towards unbiased predictions under the influence of biasing factors. While prior works tackled similar problem settings, the paper seems to generalize beyond to a broader class of causal structures that all can be handled with the proposed method. The integration of a causal stability criterion via an addition independence constraint might integrate well within existing setups and makes the method agnostic to the underlying ml model.

The paper is generally well motivated and structured. The presented derivations and proofs are mostly self-contained. The provided figures strongly support understanding of the required graphical criteria and the appendix contains detailed derivation of the individual propositions and theorems.

While the authors consider a particular type of semi-synthetic datasets, their method seems to excel at those in comparison to existing approaches.

### Weaknesses
**1) Related Work and Problem Setup.** While the authors cite and compare to several works throughout their paper and in experiments, no explicit related work section is provided. It is therefore difficult to pinpoint qualitative differences to existing approaches.

Given that the authors consider high-dimensional image data during their experiments, no reference to causal representation learning (CRL) literature is made. The presented task, however, seems to slightly differ to common image classification and CRL setups in that the biasing factors B seem to be assumed to already have been annotated (or are otherwise known) for all individual samples. The authors should be more explicit about this requirement, as such additional information is generally not provided for standard image datasets, but arguably simplifies the problem.


**2) Counterfactual Loss and Sufficient Variance. **The authors motivate the concept of causal stability via the optimization of counterfactual effects. On one hand, counterfactual data can commonly not be observed for real-world data. The authors could improve the paper by stating more explicitly which assumptions on the availability of data is made in the experiments to evaluate CF effects, or how to otherwise interpret the meaning of a counterfactual analysis in this work.

On the other hand, the optimization problem of eq. 8 and its realization in eq. 24, purely optimize the model by minimizing a loss via the model output. Common confounding, (e.g., camel<->desert, cow<->meadow), can usually not be broken for associational learners using a loss regularization, as this is caused by a lack of variation in the input. Works in the field of CRL therefore commonly assume sufficient variation of the data, assuming support over the whole distribution (e.g. occasional camel<->meadow samples...). From reading the paper, it is unclear to me whether the authors initially assume sufficient variability or whether the DISCO regularization terms exposes/requires the model to such additional data.


**3) Clarity.** Sec. 3.1 is rather dense in theory, but lacks in explanations on the particular choices and derivation made to arrive at the specific equations given in definitions 3 and 4. In particular, definition 3 is given without any further comment. Here, the the semantics of the additional "a" and "D" terms are unclear to me. The paper might be improved by providing further explanations on how to interpret the given terms in eq. 10.

Similarly, the authors use their distance correlation measures to construct their DISCO losses. While the overall derivation seems to be technically sound, the authors might improve this part by briefly discussing the qualitative differences between $DISCO_m$ and sDISCO estimation.


**Minor:**

* The authors could be more explicit on the role of the "standard assumptions of kernel regression", as mentioned in proposition 2, towards the well working or non-working of their method. I assume this relates to common smoothness conditions of the approximated function?
* Given the strong implications of the assumed causal graph towards identifiability of causal effects, the paper might be improved by moving a variant of figure B.1 directly into the main text.
* Some citations in the proof sketch of proposition 6 are not linked properly. Székely–Rizzo (2014) is therefore missing in the references.

### Questions
My questions primarily regard the above weaknesses:

1) Could be authors elaborate on the relation of their problem setup to that of common CRL tasks? How is the method contrasted to existing causal representation learning methods and how realistic is the assumption of having access to annotations on the biasing features, e.g., in common image datasets?
2) Which assumptions on the availability and sufficient variability of the data are made in the paper? Does the DISCO regularization impose additional requirements on the availability of the data?
3) Could the authors elaborate on how to interpret the role of the individual a and D terms in eq. 10? Similarly, the authors construct a new conditional distance measure in eq. 10. Why is this new measure required? E.g., how does it compare to existing ones like HSIC that equally construct kernels?
4) Could the authors elaborate on the assumptions of kernel regression required for their method? Often times kernel methods struggle to approximate non-smooth functions. Is this the case here and would it impact performance on discrete feature spaces?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a causal framework for mitigating bias in deep learning by ensuring that model predictions rely only on causal, stable information rather than spurious shortcuts. Using the Standard Anti-Causal Model (SAM), the authors show that achieving causal stability amounts to enforcing $\hat{Y}\perp B | Y$, i.e., models' predictions  $\hat{Y}$ should be independent of the spurious bias variable $B$ when conditioned on the true label $Y$, which is treated as the cause in this paper.

To implement the idea, they introduce two estimators of conditional distance correlation. $DISCO_m$ and $sDISCO$, which quantitatively measure this conditional dependence. The conditional distance correlation equals zero when the variables are conditionally independent, minimizing it is equivalent to removing the bias effect. Thus, during training, they add a DISCO-based penalty to the loss to train the model to make predictions that are conditionally independent of the spurious bias.

Experiments on five datasets show that both methods outperform or match state-of-the-art bias mitigation baselines, while using fewer hyperparameters. SAM also enables pathway-specific counterfactual analysis, offering causal interpretability of model behavior.

### Strengths
1. The paper formulates the debiasing goal through a clear causal formulation $\hat{Y}\perp B | Y$, and tranforms it into an optimization problem of minimizing conditional distance correlation, which is novel and well-motivated formulation.

2. The paper introduces two novel, and practical estimators $DISCO_m$ and $sDISCO$, that make conditional independence regularization computationally feasible.

3. The method is principled and theoretically grounded. 

4. The paper presents a moderately novel idea and experimental setting for evaluating causal stability, using pathway-specific counterfactual analysis in controlled scenarios to examine whether model predictions depend on stable causal effects.

### Weaknesses
1. Minor inconsistencies and potential overstatements:

(a) According to the definition of $\text{ctf-SE}$, and the derivation in Appendix C.1, should it be "$+\text{ctf-SE}$" in Equation 6 rather than "$-\text{ctf-SE}$"? The current sign seems inconsistent to me.

(b) Line 133- 134 claim that classical maximum likelihood estimation aims to maximize TV. Is there a theoretical justification, proof, or citation to support this claim?

2. The paper wants to enforce independence of the model's predictions from both confounders $Z$ and mediators $W$. This raises concerns about discarding potentially useful and stable mediating information, especially when parts of $W$ carry causal signals that are robust to distributional shifts. Enforcing full independence might inadvertently remove useful information, potentially harming model performance.

3. The baselines appear somewhat outdated. There are many recent approaches on mitigating spurious correlations after 2022. And the common metric Worst-group Accuracy is also not used in this paper, which is widely used in evaluating robustness for classification tasks. Including it would strengthen the evaluation and enable fairer comparison to prior work in this domain.

4. All experiments are conducted on static vision tasks. To validate the generalizability and scalability of the method, it would be valuable to include evaluations in broader settings such as text/NLP tasks, cross-modal learning, and large-scale foundation models. Such extensions would better demonstrate the method's practical impact and adaptability across modern machine learning paradigms.

### Questions
See in "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a causal-theoretic approach to bias mitigation in deep learning. The authors introduce a **Standard Anti-Causal Model (SAM)**, which unifies different bias mechanisms — such as confounders, colliders, and mediators — under a single framework.

From this model, they derive a conditional independence criterion:

```
Ŷ ⟂ B | Y
```
as sufficient for **causal stability**, ensuring that predictions depend only on direct causal effects from `Y` to `Ŷ`.

To operationalize this principle, the authors introduce two new estimators of conditional distance correlation, **DISCOm** and **sDISCO**. These are more computationally efficient than previous approaches, with sDISCO being the most efficient one. The estimators can be incorporated as differentiable regularizers in black-box neural networks.

### Strengths
- **Principled independence criterion:**  
  The independence constraint ``Ŷ ⟂ B | Y`` is derived clearly from causal reasoning. Intuitively, once the true label is known, the prediction should not depend on bias variables.

- **Practical regularization approach:**  
  Using conditional distance correlation as a differentiable regularizer is elegant and theoretically grounded. It avoids adversarial training or explicit causal graphs. The proposed estimators (DISCOm and sDISCO) are computationally more tractable than prior conditional-independence estimators.

- **Extends applicability:**  
  The method can be used with black-box neural networks, adding a new dimension to bias mitigation research.

### Weaknesses
- **Limited practical significance:**  
  The method assumes that bias variables are **known** during training. In realistic scenarios, biases are often unknown, making this assumption impractical.  
  While previous methods like GDRO and Last Layer Retraining (LLR) (or Deep Feature Reweighting) use group annotations, the field needs methods robust to unknown biases.  
  Furthermore, the empirical advantage over existing methods is unclear: for example, LLR achieves higher worst-group accuracy on Waterbirds (92.9%) vs. proposed method (89.4%).

- **Conditional distance correlation is not new:**  
  Conditional distance correlation (Wang et al., 2015) is a well-established statistical tool. The contribution here lies in making it differentiable for deep learning pipelines, which is an engineering advancement rather than a fundamentally new estimator.

- **Empirical evaluation is limited:**  
  Several widely used spurious datasets, such as CelebA, MultiNLI, and CivilComments, are not included. Key baseline methods, including Last Layer Retraining (LLR) and SELF, as well as invariant-based approaches like IRM, and Fishr, are absent. Furthermore, the claim that the method operates as a “black-box” is not fully substantiated—it's unclear whether the logits are available or if intermediate representations, such as the pre-softmax layer, are accessible.

- **Overstated generality:**  
  Claims that DISCO “mitigates all types of dataset bias uniformly” are not supported. More exhaustive evaluations on well-known biased datasets would be needed to support such a claim, including:
  -- MetaShift
  -- ImageNetBG
  -- NICO++
  -- Living17
  -- MIMICNotes
  -- MIMIC-CXR
  -- CheXpert
  -- CXRMultisite
  -- Waterbirds (common)
  -- CelebA (common)
  -- MultiNLI (common)
  -- CivilComments (common)

### Questions
Please address the mentioned concerns raised above. 

- How is the multi-bias scenario experiment constructed, and which model is used?  
- Why is the original Waterbirds dataset modified, and how does this affect comparability to prior work?  

Suggestions
- **Naming:**
SAM, commonly known as Sharpness-Aware Minimization, is a well-known optimization algorithm in machine learning that improves model generalization and is also used for spurious correlation and bias mitigation. To avoid confusion and distinguish the proposed method, it would be helpful to introduce a unique name or a distinguishing modifier for this variant.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper defines a condition to ensure prediction relies on the causal direct effect in an anti-causal setting. After proving theorems about causal stability and performativeness, the paper develops a distance-based estimator for conditional independence and uses it to build robust models.

### Strengths
- A cleanly motivated debiasing technique that focuses on direct effects.
- The distance-covariance measure of independent was new to me and is interesting given it doesn't need require estimation.
- Good experimental results, showing improvements compared to standard baselines.

### Weaknesses
See questions.

### Questions
- Authors say "Classical maximum likelihood estimation aims to maximize this difference." Is this true because of Pinsker? Provide a citation please.
- In the proof you use a conditional independence statement about observed variables to make a conditional independence statement about counterfactual variable $\hat{y}_{y^\prime, w^\prime}$. That needs to be justified.
- theorem 2 makes a claim about maximizing the direct effect. What does that say about actual performance for metrics like log-likelihood or accuracy? There is no statement of optimality here like the ones in https://arxiv.org/abs/2106.00545, https://arxiv.org/abs/2107.00520. It's not clear that your guarantees are better than the ones listed in the papers above.
- How is the distance-covariance related to MMD? The form bears similarities.
- Why are GDRO and c-MMD results not present for 3 datasets? Also what about the marginal independence / balancing estimators from Makar et al.?

### Soundness
3

### Presentation
3

### Contribution
2
