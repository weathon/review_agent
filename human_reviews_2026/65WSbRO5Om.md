# DotMatch: Simplified Semi-Supervised Learning with the Log Dot Product Loss

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 6, 4, 4, 2

## Abstract
Semi-supervised learning (SSL) algorithms typically work by generating supervisory signals for unsupervised data using the model being trained, but such supervisory signals are generally imperfect, thus various techniques have been proposed to balance the signal-to-noise ratio, such as confidence-based pseudo-labeling, consistency regularization and entropy regularization. However, these methods often require careful tuning of hyperparameters, such as the confidence threshold in pseudo-labeling and the regularization strength in regularization methods, which is often a challenging task, particularly with limited labeled data available for validation. In this paper, we introduce DotMatch, an SSL algorithm that is capable of balancing the signal-to-noise ratio without any algorithm specific hyperparameters. Specifically, we introduce a novel consistency loss on unsupervised data to replace the cross-entropy loss, called the log dot product (LDP) loss, which is simply the negative log of the dot product between the predicted label distributions of weak and strong augmented views of an input. Compared to the cross-entropy loss with soft target, the LDP loss enjoys several benefits in the context of SSL: non confident examples have low impacts on model updates, as in confidence-based pseudo-labeling methods such as SoftMatch; predictions are encouraged to have a low entropy, as in entropy-regularized methods; and interestingly, its gradient is appropriately scaled relative to the gradient of the supervised loss, thus requiring no regularization constant. We additionally combine the LDP loss with distribution alignment to ensure the distribution of predictions on unlabeled data match that of the labeled data. We provide a theoretical analysis to explain the efficacy of DotMatch from the perspective of loss gradients. Extensive experiments show that DotMatch is competitive with state-of-the-art baselines without needing to tune any algorithm-specific hyperparameters for different datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a new method from Semi-Supervised Learning (SSL) called DotMatch.
The goal of this method is to simplify existing methods by removing user-specified hyper-parameters such a label confidence threshold or such as the weighting for the unlabeled (which in many method is set to 1 anyway).
This is achieved by replacing the cross-entropy (CE) loss $\text{CE}(x,y)=-\sum_i y_i\log(x_i)$ where $y$ is the expected label probability vector and $x$ the actual label probability vector, with a new loss called Log Dot Product (LDP) loss $\text{LDP}(x,y)=-\log(\sum_i x_i y_i)$.


Using this simple modification, the method adapts the classical unlabeled SSL loss for a neural network classifier $p(.;\theta)$ to $$L_{u}(\theta)=\text{LDP}(p(s(x);\theta),\text{nograd}(\text{DA}(p(w(x);\theta))))$$ where $w,s$ are respectively weak and strong augmentations.
Here $\text{DA}$ stands for distribution alignement and is a reweighting of the label probability vector obtained from $w(x)$ by an approximation of the ratio $\mathbb{E}_{y\sim\mathcal{L}}[y] / \hat{p}_y=\mathbb{E}_{x\sim\mathcal{U}}[p(w(x);\theta)]$ as introduced by ReMixMatch.

Results are evaluated with SotA baslines for MNIST, EMNIST, CIFAR10, CIFAR100, SVHN and STL confirming the method is competitive for various sizes of labeled subsets.

### Strengths
1. The method is simple and elegant.
2. The method uses all the unlabeled data in the batch rather than zero-ing low-confidence predicted labels unlike previous methods.
3. The paper is well written and easy to follow for the most part.

### Weaknesses
1. The definition for $\texttt{sumnorm}$ (line 364) appears to be missing, I assume it must be $\text{sumnorm}(x)=x/\sum_i x_i$.
2. The definition for $\epsilon$ also appears to be missing (line 364), I assume it must be a small constant to avoid division by 0.

### Questions
1. What value of $\epsilon$ did you use?
2. Is $\epsilon$ a function of the number of classes?

### Soundness
3

### Presentation
3

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
This paper introduces DotMatch, a novel semi-supervised learning (SSL) method that omits hyperparameters, such as confidence thresholds or regularization strengths. The key contribution is the Log Dot Product (LDP) loss, a new consistency loss that replaces the traditional cross-entropy loss for learning with unlabeled data. The LDP loss simply computes the log negative dot product between predictions between weak- and strong-augmentations. Moreover, when combined with distribution alignment, DotMatch achieves competitive or superior performance to other baseline methods. Furthermore, a theoretical gradient analysis is provided to support the empirical findings.

### Strengths
- This paper proposed a simple and intuitive method named DotMatch that omits hyperparameter tuning, which is easy to employ and simplifies the training process.
- This paper provides a rigorous theoretical framework to justify its method.
- The experiments are quite extensive and show promising performance improvement.

### Weaknesses
- While the LDP loss is formulated in a simple and elegant manner, it remains similar to standard consistency-based SSL approaches, where they simply use the prediction consistency between weak and strong augmentation to conduct SSL. As a result, consistency-based SSL either does not require hyperparameters to control the strength or threshold. Could you further justify why the DOT multiplication is different from the MSE or KLD-based consistency training loss?
- Semi-Supervised Learning under open world and distribution shift has been an important research topic that scales SSL from traditional IID scenario to more complicated real-world applications [1], [2], and [3]. How the proposed DotMatch can be successfully employed in open world SSL is worth further discussing and investigating because there is no justification nor experiments to demonstrate the robustness.  
[1] Saito et al., Openmatch: Open-set semi-supervised learning with open-set consistency regularization, in NeurIPS 2021.  
[2] Huang et al., Universal semi-supervised learning, in NeurIPS 2021.  
[3] Yu et al., Multi-task curriculum framework for open-set semi-supervised learning, in ECCV 2020.

### Questions
- How does LDP behave when the prediction distributions are nearly orthogonal (dot product close to zero)? Is there a risk of instability in the gradient magnitude?
- Can LDP be extended to multiview or multimodal SSL cases where prediction distributions may differ structurally? Does the proposed method remain effective if augmentations are weaker, i.e., the difference between weak and strong augmentations are small? Will LDP still be effective if the application is related to a different modality, such as Speech or Text SSL?
- Would introducing an adaptive temperature in LDP improve generalization further? How can such a temperature affect the learning performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DotMatch, an SSL method that replaces the usual unlabeled-data cross-entropy with a Log Dot Product (LDP) loss. Despite its simple form, LDP (i) automatically down-weights low-confidence unlabeled examples, (ii) implicitly encourages low-entropy predictions, and (iii) matches the gradient scale of supervised cross-entropy—without introducing additional algorithm-specific hyperparameters common in prior work. Experiments show that the proposed method works well on several benchmarks.

### Strengths
The paper is clearly written and easy to follow. 
The LDP loss naturally (i) down-weights low-confidence unlabeled samples, (ii) encourages low-entropy predictions, and (iii) matches supervised CE’s gradient scale. This hyperparameter-light design could be practically appealing as it minimizes algorithm-specific hyperparameter tuning that many SSL methods require.

### Weaknesses
The overall framework closely resembles FixMatch (adding DA), but with the consistency loss replaced by the proposed LDP objective. However, since similar dot-product–based losses have appeared in prior work and are here adapted to SSL, the contribution feels incrementally novel.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DotMatch, a semi-supervised learning (SSL) algorithm centered on a novel Log Dot Product (LDP) loss. The paper's primary and most significant contribution is its theoretical analysis of this loss function. Theorem 1 proves that the LDP loss's gradient norm is implicitly coupled with the target's confidence (entropy): high-entropy (uncertain) targets naturally produce near-zero gradients, while low-entropy (confident) targets produce large gradients. This presents an elegant, implicit mechanism for balancing signal-to-noise, contrasting with the explicit thresholding or re-weighting mechanisms of algorithms like FixMatch and SoftMatch. Empirically, DotMatch shows very strong performance in extremely low-label regimes.

### Strengths
## Reasons to Accept

---

* **Novel Theoretical Contribution:** The paper's core strength is its theoretical analysis of the LDP loss's gradient properties. Theorem 1 provides a "first-principles" explanation for an implicit, confidence-based filtering mechanism that emerges from the loss function's geometry alone. This is a novel and elegant contribution to the field.
* **Good Low-Label Performance:** DotMatch achieves state-of-the-art results in the most challenging, data-starved settings, such as EMNIST with 47 labels (~1 per class) and CIFAR-100 with 400 labels.
* **Elegant Loss Formulation:** The LDP loss itself is a clever, unified objective. As shown in the analysis and ablations, it successfully integrates three key SSL goals into one function: consistency regularization (via weak/strong augmentation), inherent entropy minimization (driving predictions toward one-hot targets), and implicit confidence-based re-weighting.
* **Strong Ablation Study (for LDP):** The ablation in Table 3 provides causal evidence for the LDP loss's effectiveness. It shows that LDP+DA (DotMatch) is dramatically better than using a standard Cross-Entropy loss with either hard or soft targets, confirming the LDP loss is the primary driver of the algorithm's success.

### Weaknesses
## Reasons to Reject

---

* **Hyperparameters:** The paper's main claim of having "no algorithm specific hyperparameters" is false. The DotMatch objective (Eq 5) explicitly includes a Distribution Alignment (DA) component. As defined in Section 4.3, this DA mechanism depends on $\hat{\pi}_{t}$, an "EMA of the prior," which is calculated using an "EMA momentum $m$". $m$ is algorithm-specific hyperparameter that is left un-ablated.
* **Missing Large-Scale Benchmarks:** The paper's empirical validation is only on small, "classic" datasets (e.g., CIFAR, EMNIST). It lacks experiments on standard large-scale benchmarks like ImageNet or WebVision. Prior work like SoftMatch/FixMatch/FreeMatch are evaluated on these.
* **Empirical Strength:** DotMatch is not universally better. It is significantly outperformed by FixMatch on SVHN (40 labels) and outperformed by FreeMatch and SoftMatch on CIFAR-10 (40 and 250 labels). This suggests its filtering may be an overly conservative liability on "easier" datasets where greedy, explicit methods (like FixMatch) are superior.

### Questions
## Questions for the Authors

---

* The paper’s core premise is "no algorithm specific hyperparameters". Why is $m$ not considered an algorithm-specific hyperparameter, and can you provide a full ablation study for it across datasets?
* Why were experiments on standard, large-scale benchmarks like ImageNet and WebVision omitted? SoftMatch and FreeMatch use these to prove scalability and noise robustness. Without them, it is difficult to assess the practical value of DotMatch. Could you add these results?
* Table 2 shows that DotMatch is outperformed by FixMatch on SVHN and by FreeMatch on CIFAR-10. Does this suggest that the LDP loss’s implicit, gradual filtering is actually a disadvantage on datasets where a model can quickly become confident, and that a "greedy" explicit threshold (like FixMatch's) is superior in those cases?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on semi-supervised learning (SSL) and proposes DotMatch, an algorithm that leverages multi-view consistency and distribution alignment (DA) to learn from unlabeled data without algorithm-specific hyperparameters: it first introduces the Log Dot Product (LDP) loss to measure consistency between weakly and strongly augmented views of unlabeled examples, then combines it with distribution alignment to match the predicted label distribution of unlabeled data with that of labeled data; with LDP loss down-weighting low-confidence examples and implicitly minimizing entropy, the framework achieves competitive performance on standard SSL benchmarks. While it has strengths in desirable qualities of SSL, hyperparameter-free design, theoretically grounded gradient analysis, it also faces issues like reliance on lack of novelty, insufficient experimental analysis and presentation issues.

### Strengths
1.Desirable qualities on SSL: The method proposes LDP loss with distribution alignment and achieves desirable qualities on SSL, namely consistency, entropy minimization and small gradient norms for low-confidence unlabeled examples.
2.Hyperparameter-free design: The method is hyperparameter-free without needing to tune any algorithm-specific hyperparameters for different datasets.
3.Theoretically grounded gradient analysis: The paper gives theoretically grounded gradient analysis and comparison of three losses (CE(hard), CE(soft) and LDP) through formula derivation and visualization.

### Weaknesses
1.Lack of novelty: The proposed method is largely built upon established SSL paradigms. Similar forms of Log Dot Product–based losses have been discussed in prior works, and the use of distribution alignment is also a well-known strategy. 

2.Insufficient experimental analysis: The experiments mainly report classification test errors without deeper quantitative or qualitative analyses to support the claimed advantages, such as confidence calibration, the contribution of unlabeled samples, or training dynamics visualization. More ablation or interpretability studies would strengthen the empirical validation.

3.Presentation issues: The paper suffers from inconsistent and potentially confusing notation—such as using both bold and non-bold versions of the same symbol to denote different quantities—and inconsistent symbol definitions.

### Questions
1.LDP loss Innovation: The proposed LDP loss shares a similar structural form with the Pairwise Objective introduced in the following ICLR 2022 paper. A clearer distinction between the two should be articulated.
OPEN-WORLD SEMI-SUPERVISED LEARNING. ICLR 2022 
2.DA originality: DA is also a commonly used correction algorithm. It is supposed to illustrate whether it has been improved or innovated, and also explain in detail the relevant details of the DA formula and its various symbols.
3.Limited comparison with existing methods: The comparison set in the experiments appears relatively narrow. Beyond the few mentioned baselines, the paper should consider including a broader range of semi-supervised learning methods, particularly recent state-of-the-art algorithms.
4.Dataset-specific performance concerns: The proposed method exhibits significant performance gains primarily on the EMNIST dataset, while the improvements on other benchmarks are marginal. This raises concerns about potential dataset-specific tuning or overfitting.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes the DotMatch method for semi-supervised learning (SSL) problem, with the proposed log dot product loss (LDP) loss applying to a classic form of SSL objective functions. Through optimum and gradient analysis of LDP, the authors claim that LDP has benefits that non-confident examples have a low contribution to the gradient and that the optima are encouraged to have a low entropy. These properties of LDP enable the proposed DotMatch method to be free of hyperparameter tuning. Experimental comparisons with other SSL methods on classic CV benchmarks are also conducted.

### Strengths
- The proposed method requires no algorithm-specific hyperparameters.
- The proposed method is easy to understand and implement.
- The proposed LDP loss is analyzed from both theoretical and experimental perspectives.

### Weaknesses
- The experimental results are weak. 1) In Table 2, the proposed DotMatch method achieves the best accuracy in only 5 out of 10 cases. 2) The experiments are conducted on relatively small scale CV benchmark datasets. Results on larger datasets such as Imagenet are lacking.
- As shown in Figure 2, when the target is close to uniform, LDP requires far more gradient steps than CE to reach the optimum. This raises my concerns about the computational efficiency of DotMatch. To address this, I think experimental comparisons on the training time of LDP with other methods should be reported.
- Figure 1 should include DotMatch to better support the claims.
- In Table 1, the notations $\boldsymbol{z}$ and $p \odot q$ are used but undefined.
- Line 930. Table 7 should be Table 2?

### Questions
- I cannot find which section supports the claim in the abstract that "its gradient is appropriately scaled relative to the gradient of the supervised loss, thus requiring no regularization constant." Could you specify this for me?

### Soundness
2

### Presentation
2

### Contribution
2
