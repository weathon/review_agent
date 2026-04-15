# Identifying Drivers of Predictive Uncertainty using Variance Feature Attribution

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 6, 1, 8

## Abstract
Explainability and uncertainty quantification are two pillars of trustable artificial intelligence. However, the reasoning behind uncertainty estimates is generally left unexplained. Identifying the drivers of uncertainty complements explanations of point predictions in recognizing potential biases and model limitations. It additionally facilitates the detection of oversimplification in the uncertainty estimation process. Explanations of uncertainty enhance communication of and trust in decisions. They allow for verifying whether the main drivers of model uncertainty are relevant and may impact model usage in certain applications. So far, the subject of explaining uncertainties has been rarely studied. The few exceptions in existing literature are tailored to Bayesian neural networks or rely heavily on technically intricate approaches, such as auxiliary generative models, thereby hindering their broad adoption. We propose variance feature attribution, a simple and scalable solution to explain predictive aleatory uncertainties. First, we estimate uncertainty as predictive variance by adapting a neural network, for example, by equipping it with a Gaussian output distribution. We achieve this by adding a variance output neuron and can thereby rely on pre-trained point prediction models and fine-tune them for meaningful variance estimation. Second, we apply out-of-the-box explainers on the variance output of these models to explain the uncertainty estimation. This two-step method can be easily applied to any neural network with model-agnostic or model-specific explainers. We evaluate our approach in a synthetic setting where the data-generating process is known. We show that our method can explain uncertainty influences more reliably and faster than the established literature baseline CLUE, while the uncertainty estimation stage does not impede the accuracy of the model.
As an illustrative application, we fine-tune a state-of-the-art age regression model to estimate uncertainty and generate attributions for age prediction uncertainty. Our exemplary explanations highlight reasonable potential sources of uncertainty, such as laugh lines and frowning. Variance feature attribution provides accurate explanations for uncertainty estimates with little modifications to the model architecture and low computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper is about explanations for aleatoric uncertainty of the variance output in a regression model trained with the Gaussian NLL. The authors propose to use gradient-based saliency maps to explain the variance head of a regression model under the Gaussian assumption. There are experiments on a synthetic dataset and on age regression.

Contributions are:
- A scalable solution to explain aleatoric uncertainty in a regression model.
- A method to extend pre-trained regression models to also consider aleatoric uncertainty, by training the variance head.
- A synthetic toy regression problem with controllable factors for heteroscedastic aleatoric uncertainty prediction, allowing for efficient evaluation.
- Results on the toy regression problem and age regression.

### Strengths
- The paper's writing is good and mostly easy to understand.
- The synthetic benchmark for evaluating aleatoric uncertainty explanations seems to be novel and significant (Sec 2.4). It basically uses Gaussian with a variable variance that has heteroscedastic and homoscedastic noise terms.

(Unfortunately I do not find more strengths in this paper)

### Weaknesses
- I believe that explaining aleatoric uncertainty is not a very interesting problem, while the authors argue about explaining uncertainty, aleatoric uncertainty is just the uncertainty in the data, usually noise in the labels, etc, and this does not have the same impact as epistemic (model) uncertainty, which is usually the kind of uncertainty that is interesting as it provides feedback about the prediction being correct or not.

- There are variations of the Gaussian NLL that have much less problems in optimization, like the beta-Gaussian NLL, etc. These variations are not used in this paper, which decreases the value to the community. Below I provide a reference to the beta-Gaussian NLL paper:

Seitzer, Maximilian, et al. "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks." International Conference on Learning Representations. 2021.

- Section 2.2 presents a method to add variance outputs to pr-etrained regression models, while I believe that this is a kind of trivial extension and there are no new ideas here, additionally the authors do not evaluate this proposed method, for example, some basic questions might arise, is it better to pre-train on MSE and later train on Gaussian NLL, or directly train on Gaussian NLL and one of its variations? It is not clear what is the value of this method as there is no evaluation or comparison.

- There is no proper comparison to the state of the art or ablation studies. The synthetic experiments are compared against CLUE, and it seems there is a small qualitative improvement (the paper does not use any quantitative metrics), but there is no comparison for the more interesting age regression experiment, which lowers the value of such comparison. Overall I understand that there is not much state of the art in this sub-field of explaining uncertainty, but then I suggest to perform ablation studies, compare multiple explanation methods, and multiple methods to estimate the variance of the data, to obtain useful insights for future research.

- I am not sure how to interpret the results of the age regression experiment (Figure 4). It is difficult to evaluate and interpret explanations, and I believe it is more difficult to explain uncertainties, the authors make qualitative comparisons among the explanations, which is fine, but how do these explanations relate to the aleatoric uncertainty? In the age regression example, aleatoric uncertainty labels are not available, so it is very difficult to argue that the model is explaining its aleatoric uncertainty. I believe this experiment requires more thinking and a proper experimental design, opposite of the synthetic experiment.

Minor Comments
- The paper refers to "Aleatory" uncertainty in some parts, but the actual technical name is aleatoric uncertainty.
- In Figure 4, only the saliency maps are presented, I believe there is more information to be presented, like the ground truth age, and predicted age mean and standard deviations, so the user can see how these three values relate to the saliency maps. Just by looking at the saliency maps without the predicted standard deviation is meaningless, as the whole aim of the paper is to explain the aleatoric uncertainty output head.

### Questions
- What is the interest of explaining aleatoric uncertainty, as opposed to explain epistemic uncertainty? I believe explanations of epistemic uncertainty are much more useful to end users, so what is the value of a aleatoric uncertainty explanation?

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an XAI method based on the uncertainty of the prediction in the regression setting. The idea is to use a VAE-type of fitting with the black-box function $f(x)$ being the mean, assuming that all the training samples are available. Using the estimated predictive variance, the authors apply an existing attribution method of the Shapley value. In the computer vision domain, the authors report on interesting results.

### Strengths
- Uncertainty quantification in the context of XAI is a relatively new topic.
- The VAE-type variance estimation is a generic method and widely applicable as long as training data are available at hand.

### Weaknesses
- A few important contexts of the related work are missing.  A few recent works in the XAI community clearly point out the importance of uncertainty quantification. The following papers should be cited and discussed at least. 
	-  Xingyu Zhao, Wei Huang, Xiaowei Huang, Valentin Robu, and David Flynn. 2021. BayLIME: Bayesian local interpretable model-agnostic explanations. In Proceeding of the 37th Conference on Uncertainty in Artificial Intelligence (UAI 21). PMLR, 887–896
	- Tsuyoshi Idé, Naoki Abe: Generative Perturbation Analysis for Probabilistic Black-Box Anomaly Attribution. KDD 2023: 845-856
- The method is data-hungry. The assumption of the availability of training data may be unrealistic.

### Questions
Please comment on the problem setting, where training data are assumed to be available, in light of prior works in the XAI research. Also, please clarify the novelty in light of the existing work, as pointed out above. 

I will update my rating depending on your reply.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors address the task of identifying drivers of predictive uncertainty. To this end, they follow a 2-step approach. First, they adapt a neural network to a mixture density network with an additional neuron capturing variance. Next, they compute KernelSHAP values on the uncertainty.

### Strengths
The approach addresses an interesting problem explaining the drivers of uncertainty; their approach is very simple and combines 2 well-known paradigms in a straight-froward manner.

### Weaknesses
- The main contribution of the paper is to demonstrate that the straight-forward combination of 2 well-known concepts (MDNs and KernelSHAP); for this to be a valuable resource I miss a  comparison to baselines (e.g. Watson et al as cited by the authors) and a systematic _quantitative_ evaluation on a representative number of real-world datasets. The qualitative evaluation on IMDB-clean is promising but does not warrant the strong conclusions of the authors
- Important literature missing: While the authors mention some  recent work in their discussion of deep heteroscedastic regression, they miss the large body of literature following the introduction of this very model in 1994 as Mixture Density Networks in Chris Bishop's seminal paper (which is not even cited); the generalisation to introduce the variance neuron after training a vanilla network is trivial

### Questions
See above

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on the rarely studied subject of explaining uncertainties (versus explaining predictions). This area hasn't gotten a lot of attention because Bayesian approaches to neural networks hasn't been widely embraced (given the difficulties with training them). The authors propose "variance feature attribution" an approach to explain aleatoric uncertainty. They adapt a traditional neural network to be suitable by adding a variance output, and fine-tuning pre-trained point estimate models (the point becoming the mean of the distribution) to provide a useful variance estimation. They demonstrate their approach on a synthetic dataset such that the uncertainty can be controlled and compare their approach to explainability against CLUE. The goal is to identify which factors/features contribute to elevated (or reduced) levels of uncertainty. They also demonstrate their approach on a non-synthetic dataset (related to age regression) and show which areas of an image cause increase of uncertainty (marks around the eyes/mouth).

### Strengths
This is a well written paper that tackles an area that hasn't be given a lot of attention. I found the synthetic results very compelling, especially what was presented in Figure 3. I appreciate how they show that CLUE does explain the features causing uncertainty, but their approach makes the distinction between the features much more pronounced. The demonstration on a real-world dataset gives additional strength to their claims.

### Weaknesses
I had issues with Section 2.2, which I will discuss more in the questions. I believe some details were left out (or the authors thought they could be assumed) which would have made the section more explicit and clear. In that section the authors state "multi-layer regression head", which seems wrong to me. Is it suppose to be "multi-label regression head"? Also, once the additional output is added (to capture variance) and the Gaussian negative log-likelihood is used as a loss function, how many iterations should be performed?

### Questions
1. In section 2.2 do you mean "multi-label regression head" instead of "multi-layer regression head". If not, could you please elaboriate.
2. Can you explain why you used a batch size of 176 (it seems odd...well it is even, just uncommon).

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
