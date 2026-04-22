# Does Flatness imply Generalization for Logistic Loss in Univariate Two-Layer ReLU Network?

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We consider the problem of generalization of arbitrarily overparameterized two-layer ReLU Neural Networks with univariate input. Recent work showed that under square loss, flat solutions (motivated by flat / stable minima and Edge of Stability phenomenon) provably cannot overfit, but it remains unclear whether the same phenomenon holds for logistic loss. This is a puzzling open problem because existing work on logistic loss shows that gradient descent with increasing step size converges to interpolating solutions (at infinity, for the margin-separable cases). In this paper, we prove that the \emph{flatness implied generalization} is more delicate under logistic loss. On the positive side, we show that flat solutions enjoy near-optimal generalization bounds within a region between the left-most and right-most \emph{uncertain} sets determined by each candidate solution. On the negative side, we show that there exist arbitrarily flat yet overfitting solutions at infinity that are (falsely) certain everywhere, thus certifying that flatness alone is insufficient for generalization in general. We demonstrate the effects predicted by our theory in a well-controlled simulation study.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies, on univariate data input, how the flatness translates into generalization for classification tasks. At first, it provides some inequality relation between a weighted TV norm and the sharpness (given by $\nabla^2 \mathcal{L}(\theta)$.

From there, it provides two results:
 - at first it shows that flatness does not necessarily imply generalisation in classification tasks
 - lastly, they provide (via Theorem 3.7) a bound on the excess risk for flat solutions, with some additional assumptions

### Strengths
Relating flatness to generalization is an important topic and tackling this in classification problems is not studied enough in the literature. As such, this type of problems is clearly worth investigating.

### Weaknesses
My main concern is that the provided results are very complex/hard to parse, while being somehow poor. In particular, I am not sure to understand what kind of message we can retain from this work, besides the "flatness is not enough to ensure generalization in classification" message. 

Besides stating complex results, some words are used in a confusing way in my opinion. In particular, the excess risk is defined with respect to the training data examples, which is totally unusual. In consequence, when the authors claim line 394 that their bound (given by Theorem 3.7) matches the minimax optimal rate of Zhang et al. (2024), I somehow disagree: the rate of Zhang et al. (2024) is for the true definition of excess risk (ie population loss), which is harder to bound. 

In Theorem 3.5, the authors suggest that their excess risk bound could be changed into a bound on the population loss (line 348), but I think this work would clearly benefit from a precise bound on this quantity (and the same for Theorem 3.7).

### Questions
How your "excess risk" bounds can be translated for bounds as the population loss? Of course it holds "asymptoticall" as mentioned in Theorem 3.5, but I guess precise rates would be needed here.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper analyzes whether flatness implies generalization in the logistic loss setting. The authors find that there exists solutions that are arbitrarily flat but overfit, thus in general flatness does not imply generalization with logistic loss. However, using a notion of "uncertain set", they show that a similar analysis to previous work can be performed to demonstrate that flatness implies generalization within a region bounded by the left-most and right-most uncertain sets.

### Strengths
The paper furthers work on the question of whether flatness implies generalization by investigating logistic loss. The authors provide a concrete example where flatness is not sufficient for generalization, which is in stark contrast to previous work with square loss in which flatness was sufficient. A new analysis technique involving the uncertain sets, as opposed to considering the entire domain, is developed to understand the conditions under which flatness implies generalization in this classification setting.

The experiments corroborate well the theoretical analysis.

### Weaknesses
The presentation is relatively dense, challenging to follow for a reader who is not a specialist in the area.  The new ideas in the paper should have been presented more clearly and discussed further.  Similarly, the significance of the main results could have been justified more and explained further.

Theorem 3.3 demonstrates that the generalisation gap of $f_\theta$ is small when $\theta$ is small, however this is rarely mentioned further in the paper.  It is finally compared with Theorem 3.7 which bounds excess risk, but the relationship between these is not elaborated.

### Questions
Around line 261, you say that a fair choice of $\gamma$ is not ensured. What would a fair choice of $\gamma$ be? Is this fair $\gamma$ when the term $\frac{1}{\Gamma(\gamma)}$ balances out the effects mentioned on line 256? Or is it when $\frac{1}{\Gamma(\gamma)}$ is small?

Are there any guarantees on the size of the uncertain sets? Would a larger $\mathcal{A}_\gamma$ mean that flatness is more likely to guarantee generalization?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the generalization behavior of overparameterized two-layer ReLU networks with univariate input, focusing on the logistic loss. While prior work showed that flat (stable) solutions under the square loss do not overfit, the authors find that this relationship is more subtle for logistic loss. They prove that flat solutions can achieve near-optimal generalization within certain uncertainty regions, but also show that some extremely flat solutions can still overfit. Thus, flatness alone does not guarantee generalization. The theoretical findings are supported by experiments.

### Strengths
This paper addresses an interesting topic—how dynamical stability, or the curvature of the loss landscape, influences the learned predictor in classification tasks. The authors present results from several perspectives, including bounds on the total variation of the predictor’s output, excess risk, and generalization performance. The results are clearly formulated, and the assumptions are well-specified. Additionally, the theoretical findings are supported by experiments.

### Weaknesses
I believe the main weakness of the paper lies in the lack of interpretation of its results. Unlike in the regression setting, where previous findings were relatively straightforward to interpret, several conclusions in this work remain unclear or insufficiently discussed (see questions below).

Moreover, the theory of dynamical stability referenced in the paper applies to local minima, which are equilibria of the gradient descent mapping. However, as the authors themselves note, in the overparameterized regime considered here, true global minima do not exist—only infima do. Consequently, the standard theory of dynamical systems does not directly apply in this context, which weakens the paper’s theoretical motivation. That said, there is empirical evidence showing that the top eigenvalue of the Hessian tends to be regularized during training [R3]. In this sense, the paper's results still have justification.

Additionally, a few of the paper’s contributions and insights are not entirely novel. For instance, the observation that sharpness (or flatness) alone does not guarantee generalization was already discussed in [R1] and [R2] as early as 2017. Another example is the well-known fact that once a network perfectly fits the training data, the weights diverge to infinity in order to minimize the loss, while the Hessian along this trajectory tends to zero. This phenomenon is precisely why prior work has often avoided studying this setting—under the sharpness metric, all interpolating solutions appear equally good. In other words, since the sharpness of any interpolating solution approaches zero, this measure cannot distinguish among them. Please see my questions at the next part in this regard.

**References:**\
[R1] - Exploring Generalization in Deep Learning\
[R2] - Sharp Minima Can Generalize For Deep Nets\
[R3] - Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability

### Questions
1) As discussed in the paper, "fully" trained models have vanishing Hessians, regardless of how their predictor function behaves. This implies that curvature cannot distinguish among different predictor functions. How should we interpret Thm. 3.1 and 3.5 in light of this fact? Specifically, for fully trained models, where the loss and the Hessian are effectively zero, the bounds are zero.

2) A similar question arises for Thm. 3.3 and 3.7. The weights for fully trained models are large, as the minimizers are only infima. In this case, the norm of the weights grows to infinity, and the predictor is not bounded, making the bounds trivial. How should we interpret these results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the relationship between flatness and generalizability for classification. The models considered are univariate (i.e., 1-dimensional input domain) ReLU networks. The authors prove bounds between loss Hessian max eigenvalue and weighted TV norms. Also the authors prove excess risk bounds.

### Strengths
In theorem 3.1 the authors prove an upper bound on the weighted TV norm of a function (from the input data to output predictions) in terms of the max eigenvalue of the loss hessian (which is a function from the parameters).

This result is an interesting link between two fundamentally different types of object.

In theorem 3.2, the authors construct an example function whose max eigenvalue of the loss hessian goes to zero, and therefore the function is flat (measured by TV norm). however, the function can have arbitrarily bad performance.

The authors also give an excess risk bound for bounded variation functions when a certain condition is satisfied (i'm not sure if I understand this condition however, see my question below).

### Weaknesses
The paper is very symbol heavy. For instance, the weight hγ in the TV norm in Theorem 3.1 is difficult to interpret to the point i can't really tell if the left hand side of inequality (5) is truly a measure of flatness.

### Questions
What is the  hγ,ζ (x,n)≥c inside Theorem 3.7? Again, the symbol heaviness makes this is quite difficult to parse.

Can the author give a clear cut definition of "minima stability"? Without knowing what the authors mean by it, it's hard to tell how to interpret theorem 3.7 under the lense of minima stability.

### Soundness
3

### Presentation
2

### Contribution
3
