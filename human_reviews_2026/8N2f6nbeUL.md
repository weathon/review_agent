# Noisy Scrubber: Unlearning Using Noisy Representations

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Machine Unlearning (MU) aims to remove the influence of specific data points from trained models, with applications ranging from privacy enforcement to debiasing and mitigating data poisoning. Although exact unlearning ensures complete data removal via retraining, this process is computationally intensive, motivating the development of efficient approximate unlearning methods. Existing approaches typically modify model parameters, which limits scalability, introduces instability, and requires extensive tuning. We propose Noisy Scrubber, a novel MU framework that learns to inject perturbations into the latent representations rather than modifying model parameters. To show Noisy Scrubber attains approximate unlearning we theoretically establish bounds on the parameter gap between original and exact unlearned model, as well as on the output discrepancy between Noisy Scrubber and exact unlearning. Empirical results on CIFAR-10, CIFAR-100, and AGNews demonstrate that Noisy Scrubber closely matches exact unlearning while being significantly more efficient, reducing unlearning gaps to 0.024, 0.129, and 0.006, respectively. Moreover, membership inference evaluations confirm that Noisy Scrubber removes information comparably to retraining. Our approach scales across model families in both vision and text, and introduces a flexible, attachable noise module that enables on-demand and reversible unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Proposes a Noisy Scrubber — a lightweight, pluggable module that injects noise into intermediate representations. The module $ p(\cdot, \phi)$ is trained for producing the noise that is then added to intermediate representations, so that the model’s output becomes uncertain on forgetting samples while remaining unchanged on retaining data. A distilled version (Distill-NS) merges the effect of both $\theta$ and $\phi$ into a single unified model via knowledge distillation.

### Strengths
1. Clear writing. Shifting unlearning focus from parameter manipulation to representation-space noise injection.

2. The proposed method is pluggable, data-efficient, and architecture-agnostic.

3. Experiments cover vision and NLP datasets, including both qualitative and quantitative metrics.

### Weaknesses
1. Around line 222, "In practice, strong convexity can be introduced during training through the addition of L2 regularisation." Does strong convexity refer to the global convexity of a deep neural network's loss function? Is there a reference that proves this statement?

2. The theory objective seems mismatched with the unlearning training objective. The bounds assume that the outputs approximate retraining, but the training uses the UEO loss. There is no justification that retraining produces UEO-like outputs.

3. Retrained models typically distribute probabilities unevenly across similar classes. Forcing uniformity could eliminate meaningful structure and hinder generalization. This also leads to a mismatch between the theory and the training objective.

4. This work introduces a learnable plug-and-play network for learning an extra objective other than the original model's, an idea that is quite similar to LoRA. However, the paper lacks a discussion and comparison with similar work based on LoRA [1].

[1] Cha S, Cho S, Hwang D, et al. Towards robust and parameter-efficient knowledge unlearning for LLMs. ICLR 2025

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Noisy Scrubber, which is a plug-in module that injects targeted noise into latent representations to unlearn a forget set F. The proposed model seeks to present an unlearning method that avoids changing the parameters of the model, but instead, for the samples in F, it adds noise to the latent representation to maximize the entropy of the prediction when the last layer is applied to them.  The paper provides some theoretical bounds under strong convexity and Lipschitz/smoothness assumptions. Experiments on CIFAR-10/100, AGNews, DBPedia with CNN/ResNet/SWIN/BERT report strong utility and privacy.

### Strengths
- The idea is straightforward and architecture-agnostic. 
- The method is easy to implement and enough details have been added by the authors.
- The paper is clearly written and easy to follow.

### Weaknesses
1. Although the theoretical results are claimed as the main contributions,  Proposition 1 is not new and has been presented by prior work on certified unlearning. For example lemma 8 in [1] derives the exact bound that you have; the only difference is that the bound you have is scaled by |F| because the bound in [1] is for the case in which the number of samples in two datasets differ by 1, but in your case they differ by |F| (the number of forget samples). The authors fail to reference prior works on machine unlearning that introduce the same bounds and instead introduce them as if they are new contributions. After the work in [1] there are newer works on certified machine unlearning that relax the assumptions about strong convexity, which do not apply to the models used in practice [2].
2. Authors in line 222 mention that “in practice strong convexity can be introduced during training through the addition of L2 regularization.” Addition of L2 regularization makes the objective strongly convex only when the objective is convex. That is not the case for deep neural networks that are non-convex. Strong convexity in general is a very restrictive assumption, which is why in works such as [2] on certified unlearning, the focus is on relaxing this condition.
3. The justification for avoiding a change in the model parameters for unlearning is not clear. In the worst-case scenario that an adversary gains access to the model they can easily recover information about the forget samples. As mentioned by the authors, the goal in machine unlearning is to derive model parameters that are indistinguishable from the parameters of a retrained model. If the goal is to just perform post-hoc unlearning with the maximum uncertainty principle used by authors, is there any advantage over the following pseudocode: 
- Compute $y = f(x,\theta^t)$
- If $||x-x'|| < \epsilon$ for some $x' \in F$:
    - Return a label other than $y$ chosen uniformly at random
- Else:
    - Return $y$

I claim this is what your method is ideally supposed to achieve. It tries to train a second neural network such that it generates near-0 noise for the samples not in $F$ to get the same prediction and return add noise to $x \in F$ such that the prediction becomes like a random predictor with uniform probability over the remaining classes. With the above pseudo-code we do not make any additional assumptions compared to what you make, and achieve the same goal.

4. The SVM-based MIA that is used for evaluations is very weak compared to the SOTA MIA methods in the literature. I encourage the authors to utilize SOTA MIA for their evaluations rather than relying on basic approaches. [4] introduces is one of the SOTA MIAs that is an adaptation of  [3], but more practical with a few shadow models. There are also MIAs designed specifically to evaluate unlearning methods [5,6].
5. There are more recent works on machine unlearning for classification models that have not been used as base-lines in the experiments. Please see [2,7,8,9,10].
6. The bound in proposition 2, which again relies on model smoothness and strong convexity assumptions, is exponential in the number of gradient descent steps (which is large in practice). This makes the bound vacuous and not useful in practice. The provided bound also doesn’t depend on the value of $\epsilon$ at all. It basically is a replica of proposition 1 because authors just assume $\epsilon=0$, which basically again gives us the difference of the original model and retrained model in equation $1$. So i think in general this proposition does not provide us with any useful and meaningful information about the unlearned model
7. In general recent works on unlearning [7] has shown that  the assumption in line 233 about the retrained model regarding their uncertainty on the forget samples is not accurate. In forgetting a random set of samples a retrained model would behave similarly on the forget samples as the test samples, which is why MIAs would achieve an AUC of 50% on detecting forget samples vs. test samples. 


[1] Neel, S., Roth, A., & Sharifi-Malvajerdi, S. (2021, March). Descent-to-delete: Gradient-based methods for machine unlearning. In Algorithmic Learning Theory (pp. 931-962). PMLR.

[2] Zhang, B., Dong, Y., Wang, T., & Li, J. (2024). Towards certified unlearning for deep neural networks. arXiv preprint arXiv:2408.00920.

[3] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis and F. Tramèr, "Membership Inference Attacks From First Principles," 2022 IEEE Symposium on Security and Privacy (SP), San Francisco, CA, USA, 2022, pp. 1897-1914, doi: 10.1109/SP46214.2022.9833649.

[4] Zarifzadeh, S., Liu, P., & Shokri, R. (2024, July). Low-cost high-power membership inference attacks. In Proceedings of the 41st International Conference on Machine Learning (pp. 58244-58282).

[5] Hayes, J., Shumailov, I., Triantafillou, E., Khalifa, A., & Papernot, N. (2025, April). Inexact unlearning needs more careful evaluations to avoid a false sense of privacy. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) (pp. 497-519). IEEE.

[6] Cadet, X. F., Borovykh, A., Malekzadeh, M., Ahmadi-Abhari, S., & Haddadi, H. (2025, June). Deep Unlearn: Benchmarking Machine Unlearning for Image Classification. In 2025 IEEE 10th European Symposium on Security and Privacy (EuroS&P) (pp. 939-962). IEEE.

[7] Ebrahimpour-Boroojeny, A., Sundaram, H., & Chandrasekaran, V. Not All Wrong is Bad: Using Adversarial Examples for Unlearning. In Forty-second International Conference on Machine Learning. 

[8] Cha, S., Cho, S., Hwang, D., Lee, H., Moon, T., & Lee, M. (2024, March). Learning to unlearn: Instance-wise unlearning for pre-trained classifiers. In Proceedings of the AAAI conference on artificial intelligence (Vol. 38, No. 10, pp. 11186-11194).

[9] Bonato, J., Cotogni, M., & Sabetta, L. (2024, September). Is retain set all you need in machine unlearning? restoring performance of unlearned models with out-of-distribution images. In European Conference on Computer Vision (pp. 1-19). Cham: Springer Nature Switzerland.


### Minor weaknesses:

1. I recommend authors to use bold font to show the best result in each column for their presented table and maybe an underlined one for the second-runner. It is difficult to read the tables.
2. line 284 (where a specified …) seems incomplete. I think some words are missing.

### Questions
1. What is the advantage of using a plug-in module that disrupts the correct prediction in the last layer by adding noise to the representation layer over a post-hoc module that checks for the vicinity to the samples in $F$ and then makes a wrong prediction if it is nearby to sample in $F$? Why the need for a separate neural network that is supposed to perform this task (see 3 in weaknesses for more details)? Could the authors perform some analysis to show how the nearby region of the forget samples transform under this modification to the representation space? That could provide some insights into its usefulness.

2. What is the use-case of proposition 2 that does not depend on $\epsilon$ and is exponential in the number of gradient decent steps? 

3. Please also see the weaknesses and see if they can be addresses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Noisy Scrubber, an approximate machine-unlearning (MU) method that doesn’t modify model weights. Instead, it adds a small learned “noise module” that perturbs intermediate representations so the model behaves like it had been retrained without the forget set. The paper also provides two bounds. 1. A parameter-gap bound between trained vs. retrained models and 2. An output-gap bound.

### Strengths
1.The paper proposed a formulation of approximate unlearning via an attachable, learnable noise module without retraining the model.
2.Providing the clear theoretical analysis that connects the data removal size to output and parameter deviation, providing interpretability uncommon in empirical MU works.
3.Conducting comprehensive experiments across several datasets with strong ToW and MIA metrics.

### Weaknesses
1.The theoretical analysis relies on strong convexity, which cannot directly be applied to deep neural networks, limiting the analysis’s value.
2.In proposition 1, the author proves the parameter-gap bound with several assumptions, like L-smooth, T steps, and step size /alpha, but does not talk about the correlation between those parameters. For example, the author can use Gronwall inequality here.
3.Regarding the UEO objective, the optimization primarily constrains the output probabilities in the simplex space, pushing the predicted distribution toward uniformity except for one entry.
4.In proposition 2, the output-gap bound assumes the existence of an optimal perturbation \epsilon such that the noisy model output approximates the retrained model’s output. This implicitly presumes a controlled Lipschitz behavior of the composed mapping g(h(x,\theta)), yet no quantitative bound on its Jacobian norm or layer-wise Lipschitz constants is given.

### Questions
1.Could the author explain how those parameters interact in proposition 1?
2.In proposition 2, the output-gap bound relies on the Lipschitz continuity of g(h(x,\theta)); has the author considered bounding the Jacobian norms to make the result quantitatively evaluable?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents an alternative strategy to unlearning where perturbation is used to change the latent representations using an optimization of knowledge distillation loss function with respect to the exact unlearned model.

### Strengths
+ good use of loss-based noisy generation
+ interesting design of the noise perturbation and model - independent parameters

### Weaknesses
- use of rather old models and datasets, with less diversity
- Parallels with differential privacy and MIA resistance not explored
- Limited evaluations against various unlearning methods

### Questions
The paper’s way for exploring unlearning using model-independent and distance-based parameter perturbation is interesting and I enjoyed reading on the approach. I saw some parallels with using DP to avoid MIA, though not much investigated in the paper or mentioned. 

I think the noise module being independent of the model is an interesting and useful feature, though I was not sure how it  “allows us to forget learned knowledge for a specific period of time”.

### Soundness
3

### Presentation
3

### Contribution
3
