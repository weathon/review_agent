# Revisiting the Variational Information Bottleneck

- Avg Score: 4.25
- Decision: Reject
- Scores: 6, 5, 3, 3

## Abstract
The Information Bottleneck (IB) framework offers a theoretically optimal approach to data modeling, though it is often intractable. Recent efforts have optimized supervised deep neural networks (DNNs) using a variational upper bound on the IB objective, leading to enhanced robustness to adversarial attacks. In these studies, supervision assumes a dual role: sometimes as a presumably constant and observed random variable, and at other times as its variational approximation. This work proposes an extension to the IB framework, and consequently to the derivation of its variational bound, that resolves this duality. Applying the resulting bound as an objective for supervised DNNs induces significant empirical improvements, and provides an information theoretic motivation for decoder regularization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper extends the variational information bottleneck by adding an entropy regularizer to the model that predicts the target y given the latent z. This is motivated by adding and variational bounding a second info bottleneck.

### Strengths
The paper boils down to a simple to implement and intuitive loss function.

### Weaknesses
There is a lot of justification that is somewhat verbose and subjective. The derivation is long and elaborate for what boils down to an extra regularizer with an extra tuning parameter.

### Questions
Here are some detailed comments and questions:

Use \log in latex and format the integral d.

The writing may be a little verbose. Examples: lines 260-264 restates things. (7) follows trivially from (4). The sentences preceding both (4) and (7) are also similar. lines 217-254 is repeat well known material.

Line 198 why carry p(x,y) around if everything is conditional on those later? I suspect dropping that saves some hassle later.

Lines 271 to 296 would be better expanded after moving lines 217-254 to an appendix. You could be explicit about the use of the chain factorization etc (although maybe the previous point about line 198 can avoid needing to deal with this?).

Line 249 who’s -> whose

Line 334 why is Z left as an r.v.? Please explain how to handle this with sampling. I feel like this is just the entropy of the y given the sampled z, so it should be written explicitly as such?

Line 452 Val column bolded wrongly, the vanilla method should be shown as the winner not the proposed method?

Table 1: it seems as though VIB best performance is at the boundary of your sweep, so we can’t tell if SVIB beats VIB?

Table 2: as previous comment.

A plot instead of tables 5 and 6 would be easier to absorb.

Tables 1 and 2: can you not fix lambda and show improvement generally? Varying this in-sample looks like overfitting to the untrained eye (but I think this is an illusion and the results are good). It just seems like sub optimal presentation given tables 5 and 6 show good robust performance over lambda. A plot >> tables of numbers.

Section 4: showing robustness is nice, and the methodology seems very good i.e. adversarial approaches.

Line 478 private -> special

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
- The paper picks up an issue identified in the Deep Variational Information Bottleneck paper, where the classifier can overfit to the learned representation, Z, of a VIB model, and proposes a new framework for supervised learning with IB, which they call Supervised Information Bottleneck (SIB), and a corresponding variational approach, SVIB.
- The core theoretical contribution is to add a constraint to the IB and VIB objectives that minimizes an upper bound on I(\hat{Y},Z), which is equivalent to maximizing a lower bound on H(\hat{Y}|Z). The paper shows that this new constraint is tractable in the SVIB setting.
- The paper provides experiments comparing SVIB to VIB and “vanilla” Maximum Likelihood models (trained with cross entropy) on ImageNet and natural language sentiment analysis.

### Strengths
- The paper is well-written and easy to read.
- Constrained maximization of H(\hat{Y}|Z) will clearly achieve the goal of preventing the classifier from overfitting to the representation.
- The theoretical approach is plausibly useful. A careful set of experiments could demonstrate its value beyond using VIB or CEB.

### Weaknesses
- In general, the experiments are of the correct form (comparisons between different IB approaches and Maximum Likelihood on clean and adversarial test sets), but they are unconvincing at supporting the main claim that SVIB substantially improves on other proposed tractable IB approaches, as pointed out in more detail below.
- One shortcoming of all of the experiments is that the VIB models are not given the same amount of hyperparameter tuning as the SVIB models – it appears that in all cases, the SVIB models get three times as many runs with different hyperparameters to find a setting that outperforms the VIB models.
- VIB on classification tasks often benefits from having a mixture distribution for r(z), whether learned or just distributed across part of the domain of Z, rather than having a single isotropic Gaussian distribution for r(z). It’s likely that your selected values of \beta would perform better in that setting, as it becomes easier for the model to learn to assign classes to different mixture elements as it sees fit, which makes the model more powerful (more powerful models can tolerate higher compression/higher values of \beta). This would likely benefit SVIB as well, so that it more reliably outperforms the Maximum Likelihood baseline on the test set.
- The paper is missing an important citation: CEB Improves Model Robustness, Entropy 2020. Overlooking this reference is a major shortcoming of the paper, since it studies the same question on one of the same datasets using the same Information Bottleneck framework, and it achieves substantially better results on that dataset than reported in this paper (its VIB results are also stronger than your VIB and SVIB results).
- The ImageNet table highlights SVIB results in settings where the VIB results appear to strongly overlap – it seems a stretch to claim that SVIB is doing better than VIB with a result of 53.4%+/-1.8% compared to 53.5%+/-0.2% for FGS with \epsilon=0.1, for example (and similarly but to a lesser extent for FGS with \epsilon=0.5).
- For all experiments, hyperparameter selection for VIB is questionable, as test set performance on the clean data appears to still be improving substantially at the smallest value of \beta. As \beta goes to 0, its performance should match the vanilla model on the clean data, but you stop exploring \beta when the test set performance is substantially worse than the vanilla model, indicating that probably neither the VIB nor the SVIB models are very close to optimally configured.
- The Conditional Entropy Bottleneck paper showed that CEB reliably outperforms VIB on both clean and adversarial examples on a variety of image datasets. The CEB Improves Model Robustness paper further explores that in detail on ImageNet. Since implementing CEB can be made parameter-equivalent to implementing VIB (and consequently SVIB), it seems like an important point of comparison. 
- In Figure 1, right-hand side, the H(\hat{Y}) circle is drawn in a way that does not respect the Markov chain constraint Y-X-Z-\hat{Y}. It is not possible to have H(\hat{Y}) overlap H(Y) in any area where H(Z) does not also overlap H(Y). Compare this to the Venn diagrams in the Conditional Entropy Bottleneck paper you cite, where similarly the Markov chain Z-X-Y prevents H(Z) from overlapping H(Y) anywhere that H(X) does not also overlap H(Y).
- Line 360: repeated word: “is uninformative about about Y”.

### Questions
- I think the theoretical contribution is solid and valuable to share with the community, but I think the empirical treatment is weak. I would be very happy to increase my rating if the experiments were improved, even if they did not show that SVIB is reliably better than VIB or CEB in all of the settings considered. Whatever the outcome for SVIB on more careful preliminary experiments would be a valuable scientific contribution.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper revisits the variational information bottleneck and extends it to a supervised variational information bottleneck.  The experiment on ImageNet and text classification shows that SIB achieves better results than VIB.

### Strengths
SIB performs better than VIB regarding classification accuracy and adversarial robustness.

### Weaknesses
Overall, the novelty is insufficient, and the motivation is unclear. Some related works are missing.

### Novelty:
1. Using variational lower or upper bonds to approximate mutual information is not novel.
2. SIB's application focuses on traditional image classification and adversarial attacks. It doesn't include other applications like time series or more challenging scenarios like out-of-distribution or few-shot learning. 
3. Compared to VIB, SIB adds $H( \hat{Y} \mid Z)$, involves new hyperparameters and new terms to approximate, and cannot guarantee to reach an accurate value for $H( \hat{Y} \mid Z)$.

### Motivation:
1. The title of the paper uses "revisiting," which refers to why the VIB is revisited and what the problem of VIB is.  These two parts are not clear in the paper. Figure 1 cannot demonstrate well since an overfitted decoder can also exist in SIB if the $\lambda$ is not set appropriately. 

### Missing Reference:

[1] Kolchinsky, Artemy, Brendan D. Tracey, and David H. Wolpert. "Nonlinear information bottleneck." Entropy 21.12 (2019): 1181.

[2] A. Zaidi and I. E. Aguerri, “Distributed deep variational information bottleneck,” in Proc. IEEE 21st Int. Workshop Signal Process. Adv. Wirel. Commun., 2020, pp. 1–5

[3] S. Sinha, H. Bharadhwaj, A. Goyal, H. Larochelle, A. Garg, and F. Shkurti, “DIBS: Diversity inducing information bottleneck in model ensembles,” in Proc. AAAI Conf. Artif. Intell., 2021, pp. 9666–9674

[4]  S. Mai, Y. Zeng, and H. Hu, “Multimodal information bottleneck: Learning minimal sufficient uni modal and multimodal representations,” IEEETrans. Multimedia, vol. 25, pp. 4121–4134, 2022

[5] K. W. Ma, J. P. Lewis, and W. B. Kleijn, “The HSIC bottleneck: Deep learning without back-propagation,” in Proc. AAAI Conf. Artif. Intell., 2020, pp. 5085–5092.

The paper should compare the above methods as well and also put them into a related work section.

### Questions
1. Could you please provide more experiments on PGD and AutoAttack?
2. Could you please visualize the latent representations of SIB and compare them with VIB?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper claims to provide the "theoretical optimal approach to data modeling", and to extend the framework, by deriving the a variational bound to resolve some problems of the previous framework. 

My understanding is that eq. 16 (derived by eq.3)  is the contribution of this work. The paper provides derivations to justify eq. 16. 

In the experimental session, the authors evaluate the performance in image and text classification of the new loss and the robustness to adversarial attack. 

The authors claim that the new loss outperforms the previous loss.

### Strengths
If the paper were clearer, the paper presents many contributions and analyses of the proposed terms. 

Two experiments (image and text classification) compare the "vanilla" and VIB with the new loss. 

Historical overview of IB.

### Weaknesses
The paper is largely unclear. 

It starts with the history of the IB, more than presenting the contribution of the work.

The presentation is not clear on the steps of the new loss.  

In the new loss, there seems to be a new contribution on the predictor, but the predictor (or classifier) is already included in the loss in the VIB. 

The impression is that the new loss introduces a new regularization term, but its justification is not clear. 

The abstract is unclear, what are the two points of the "dual role"? What is the "theoretically optimal approach to data modeling"?

### Questions
Would be nice to understand the difference of eq. 16 and the standard VIB.

### Soundness
1

### Presentation
1

### Contribution
2
