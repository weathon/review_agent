# A Phase Transition Induces Catastrophic Overfitting in Adversarial Training

- Decision: Reject
- Scores: 6, 6, 3, 3

## Abstract
We derive the implicit bias of Projected Gradient Descent (PGD) Adversarial Training (AT). We show that a phase transition in the loss structure of as a function of the adversarial budget $\epsilon$ manifests as Catastrophic Overfitting (CO). Below a critical threshold $\epsilon_c$, single step methods efficiently provide an increase in robustness, while above this critical point, additional PGD steps and/or regularization are needed. We show that high curvature solutions arise in the implicit bias of PGD AT. 
We provide analytical and empirical evidence for our arguments by appealing to a simple model with one-dimensional inputs and a single trainable parameter, where the CO phenomenon can be replicated. In this model, we show that such high curvature solutions exist for arbitrarily small $\epsilon$. Additionally, we can compute the critical value $\epsilon_c$ in single-step AT for bounded parameter norms. We believe our work provides a deeper understanding of CO that aligns with the intuition the community has built around it.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the Catastrophic Overfitting (CO) phenomenon in Projected Gradient Descent (PGD) Adversarial Training (AT). This paper shows that a phase transition in the loss structure of as a function of the adversarial budget $\epsilon$ manifests and provides analytical and empirical evidence for the arguments by appealing to a simple model with one-dimensional inputs and a single trainable parameter. Experiments are conducted to validate the findings.

### Strengths
- This paper investigates the CO phenomenon and carefully analyzes the properties of CO through a toy example, which is interesting and instructive.

### Weaknesses
- Although interesting, the toy example used in this paper is too simplified, which is far away from what we meet in deep learning.
- Some notations are not consistent, for example, in section 2.1, the paper uses $c$ to represent the number of classes, while in section 3 $o$ is used.
- For **Proposition 3.1**, there are some issues that should be fixed.
  - In line 161, the dataset is $\\{ (x_ i, y_ i) \\}_ {i=1}^n$ but not $\\{ (x_ i, y_ i) \\}_ {i=1}^i$
  - The condition that $\alpha_ s = \frac{1}{S}$ is used in the proof, so it should be stated in the conditions in Proposition 3.1. Furthermore, such an assumption is not usually used in practice, we usually use $\alpha_ s > \frac{1}{S}$ in practice.
- **[important]** The writing is very bad, especially about the formulation of the studied problem.
  - Since the paper studies CO, the authors should define CO formally. CO is not defined although it appears many times in this paper, including the theorems. For example, in Corollary 3.3, the statements involve CO, which makes the theorem informal since CO is not formally defined. Additionally, CO is also used in Corollary 4.2 and Corollary 4.3.
  - In lines 194-195, the paper writes: "we define the perturbation threshold at which the effective loss is minimized at a high negative curvature solution as the critical $\epsilon_ c$". The expression is ambiguous, what do you mean by "at a high negative curvature solution"? How do you quantify "high"? So the definition of $\epsilon_ c$ is unclear.
  - In lines 187-189, the second question, the paper writes: "Can we have solutions where CO appears for any $S$ and $\epsilon$?" What do you mean by solutions? Does it mean PGD AT solutions in the first question? Or the FGSM solutions? This should be clarified.
  - I can not find the proof for Corollary 3.3, perhaps the reason that the authors can not provide proof for Corollary 3.3 is that the definition of CO is not clear. I think the proof of Corollary 3.3 should be included after you show a clear definition of CO. Similar problem occurs in Corollaries 4.2 and 4.3.
- I can not find the significance of Proposition 3.2. In practice, perturbating $(x, y)$ with $\delta$ where $\Vert \delta \Vert \le \epsilon_c$ is equivlent to perturbating $(\alpha\cdot x,y)$ with $\delta$ where $\Vert \delta \Vert \le \alpha \cdot \epsilon_c$. Moreover, in line 215: "With Proposition 3.2 and Corollary 3.3, we have a mechanism to re-scale the dataset and produce smaller εc that applies to modern deep architectures like ResNet and any training dataset". Yes, we can do this, but it is meaningless to simultaneously rescale $\epsilon_ c$ and $x$.

In conclusion, the paper is somewhat interesting. However, the problem is not properly formulated and some of the results lack proof. I think this paper is not prepared to be published.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to uncover various mysterious properties of the Catastrophic Overfitting (CO) through investigating the implicit bias of Adversarial Training (AT). In particular, the authors design a toy example where the model has only one trainable parameter and the dataset is composed of only two data points. In this example, the authors reveal the existence of a cutoff $\epsilon_c$ such that the adversarially trained model is biased towards solution with high curvature when $\epsilon > \epsilon_c$, leading to the CO phenomenon. In addition, the authors also design several numerical experiments to support their theoretical claims.

### Strengths
In general, this paper is well written and organized, making understanding the main idea and logic of this paper fairly easy. The study of CO from the perspective of phase transition by showing the existence of a cutoff $\epsilon_c$ is indeed novel. Additionally, the study of the proposed toy model is comprehensive and aligns with the goal and questions raised in the introduction.

### Weaknesses
Despite the aforementioned strengths and advantages, I have several concerns regarding the overstatement of contribution that makes me unwilling to give a higher score and I will discuss them as follows.
 
- My first concern is about the insufficient discussion of the implicit bias of AT. The authors attempt to build the implicit bias of AT to link with the appearance of CO. However, Proposition 3.1 is only a Taylor expansion of the adversarial training loss: there is no explicit characterization of the “implicit bias” of AT as it is unclear what type of solution that AT will converge to. Besides, the use of Taylor expansion to the second-order actually assumes that $L(\theta)$ is at least $C^2$-smooth, which is also neglected by the authors. It is also rather odd to say that the higher order terms of a Taylor expansion could be more significant than the lower order ones (line 192-193). As this property is important for deriving the existence of a cutoff $\epsilon_c$, it is crucial to explain this point in detail, e.g., does it contradict the essence of Taylor expansion?

   On the other hand, as this paper discusses implicit bias of AT, the connection and difference between this work and related works for implicit bias of AT, e.g., Li et al, 2019; Lyu & Zhu, 2022, should be discussed. In particular, Lyu & Zhu, 2022 established the implicit bias of AT for homogeneous deep neural networks by showing that the solution converges to a KKT point of adversarial margin maximization problem. Therefore, I think the authors overstate their contribution regarding the implicit bias of AT. As Lyu & Zhu, 2022 also unified FGSM and PGD perturbations as scale invariant perturbations, I think it would be better to discuss how this property can be connected with the CO phenomenon since the implicit bias of AT is discussed more precisely there. 
- My second concern is regarding the lack of connection between the proposed toy model and practical deep neural networks. Though the characterization of the proposed toy model is somewhat comprehensive, the model is far from being realistic, as it has only one trainable parameter and the dataset has only two points. Almost all the theoretical claims are made for this toy model, which, however, are not connected to any type of realistic deep neural networks. It is unclear to me what properties are special to this toy model and what conclusions can be generalized to other models and why such generalization can be made.

----
Reference

Li et al, 2019. Implicit Bias of Gradient Descent based Adversarial Training on Separable Data.

Lyu & Zhu, 2022. Implicit bias of adversarial training for deep neural networks.

### Questions
1. How does Proposition 3.1 characterize the properties of the converged solution? Are there any additional conditions or assumptions for making the Taylor expansion eligible? 
2. When and how will the term promotional to $\epsilon^2$ become more significant than the the term promotional to $\epsilon$ as discussed in line 192-193?

   To me it is rather odd to say that the higher order terms of a Taylor expansion are more significant than the lower order ones. Please explain this point carefully.
3. Can previous results for the implicit bias of AT be connected with results in the current paper?
4. Which theoretical claims derived from the proposed toy model can be generalized to other models and why?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work investigates the phenomenon of catastrophic overfitting arised in the fast adversarial training (AT), where the multi-step PGD attack is replaced by a single-step PGD (also known as FGSM) to reduce AT’s training time. To explore the causes of catastrophic overfitting, the paper constructs a toy model in which the adversarial loss has a closed-form solution when adversarial examples are generated by FGSM. Through analysis of this toy model, the paper identifies a phase transition with respect to the perturbation radius $\epsilon$: when $\epsilon$ exceeds a certain threshold, the local minima of the FGSM-induced adversarial loss (referred to as the "effective loss" in this work) exhibits higher curvature and this local minima has a clearly mismatch with the minima of the adversarial loss induced by multi-step PGD. Consequently, a model that minimizes the FGSM-induced adversarial loss tends to have a high loss under multi-step PGD attacks, which the paper suggests as an explanation for catastrophic overfitting.

### Strengths
The paper constructs a simple one-dimensional toy model to examine catastrophic overfitting. This model enables clear visualization of the loss landscape, illustrating the effects of varying the perturbation radius on this landscape. By plotting and comparing the landscapes of two types of adversarial losses, the model provides a clear demonstration of the factors contributing to catastrophic overfitting.

### Weaknesses
The paper constructs a toy model to demonstrate a scenario where catastrophic overfitting provably occurs. This approach may have limited practical utility for addressing catastrophic overfitting in fast AT. A more valuable direction would be identifying the conditions under which catastrophic overfitting does not occur, allowing for improvements to fast AT by regularizing the model to meet these conditions.

Additionally, the toy model settings differ significantly from those in deep learning models, suggesting that the theoretical insights derived from the toy model may have limited applicability to real-world deep learning scenarios.



**Potential techincal issues**

The derivation of Proposition 3.1 appears to contain inaccuracies. At line 812 in the appendix, the entire derivation is based on the recurrence relation that $\delta_{s} = \delta_{s-1} + \frac{\epsilon}{S}{\rm sign}(g_{\theta}(x+\delta_{s-1}))$ where $g_{\theta}(x)=\nabla_{x}{\cal L}(f_{\theta}(x), y)$ as defined in Proposition 3.1. However, there are issues with this recursion:

1. it omits the projection operation used in AT (see Algorithm 1, step 7.)   
2. when updating $\delta_{s-1}$ in PDG-AT,  the gradient should be taken w.r.t $\delta$ rather than w.r.t $x$ . Specifically, the update should use the gradient  $\nabla_{\delta}{\cal L}(f_{\theta}(x+\delta), y)$ rather than  $g_{\theta}(x+\delta_{s-1})$. The same error also appears in (Algorithm 1, step 6).



Regarding Corollary 3.3, it seems to be derived from Proposition 3.2, but the derivation is unclear.  Proposition 3.2 simply establish that  $\max\limits_{\|\delta\|\le \epsilon}{\cal L}(f_{\theta}(W(\alpha x+\delta)), y)=\max\limits_{\|\delta\|\le \hat{\epsilon}}{\cal L}(f_{\theta}(\hat{W}( x+\delta)), y)$ with $\hat{W}= \alpha W$ and $\hat{\epsilon}  = \epsilon/\alpha$ (based on the derivations at line 893 in Appendix ).  How this result leads to the claims in Corollary 3.3 is not immediately clear.



 **Writing**

The writing in the paper is not particularly reader-friendly. For instance, the insights provided by Theorem 4.1 and Corollaries 4.2 and 4.3 are not clearly explained. Additionally, the connection between the results from the toy model analysis and their implications for understanding deep learning models is not effectively conveyed.

### Questions
- In the loss landscape shown in the top panel of Figure 1, why is catastrophic overfitting attributed to the increased curvature of the local minima?
- In Theorem 4.1, what does $\theta^{*} _ {k}$   represent?  Why do we choose $\theta_{k}$ as $b_{k}$? 
- In Corollary 4.2, why are the classification results for the points $x_{i}\pm \epsilon_{k/2S}$ considered?  How does this relate to catastrophic overfitting?
- What are the main takeaway messages of this paper? Is catastrophic overfitting attributed to that the loss landscape has local minima with high curvatures? If true, could you provide empirical evidence on deep learning models to validate this conclusion?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Little is known about why multi-step AT converges to locally linear solutions or which is the underlying phenomenon resulting in CO. This work fills this gap by connecting the empirical observations with a theoretical framework.

### Strengths
1. The authors show that a phase transition in the loss structure of as a function of the adversarial budget $\epsilon$ manifests as Catastrophic Overfitting (CO).

2. The authors show that high curvature solutions arise in the implicit bias of PGD AT.  The authors provide analytical and empirical evidence by appealing to a simple model with one-dimensional inputs and a single trainable parameter, where the CO phenomenon can be replicated.

3. The authors compute the critical value $\epsilon_c$ in single-step AT for bounded parameter norms.

### Weaknesses
1.Adversarial Training (AT) (Madry et al., 2018) and its variants have proven to be the most effective strategy towards achieving adversarially robust models. Where is this inference from. It is better to replace this descirbtion with “one of the most”

2.Despite the success of these methods and the efforts in understanding CO, little is known about why multi-step AT converges to locally linear solutions or which is the underlying phenomenon resulting in CO. According to this sentence, the motivation of studying underlying phenomenon resulting in CO is insufficient. Moreover, there is little logical connection before and after.

3.It is redundant to introduce the known PGD algorithm 1, if you don’t bring in additional important ideas. Besides, the initialization of perturbation is random, not just $0$.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
