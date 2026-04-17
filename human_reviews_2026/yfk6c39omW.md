# Escaping Model Collapse via Synthetic Data Verification:  Near-term Improvements and Long-term Convergence

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6, 6

## Abstract
Synthetic data has been increasingly used to train frontier generative models. However, recent studies raise key concerns that iteratively retraining a generative model on its self-generated synthetic data may keep deteriorating model performance, a phenomenon often coined  model collapse. In this paper, we investigate ways to modify the synthetic retraining process to avoid model collapse, and even possibly help reverse the trend from collapse to improvement. Our key finding is that by injecting information through an external synthetic data verifier, whether a human or  a better model, synthetic retraining will not cause model collapse. Specifically, we situate our theoretical analysis in the fundamental linear regression setting, showing that verifier-guided retraining can yield near-term improvements, but ultimately drives the parameter estimate to the verifier's “knowledge center” in the long run. Our theory further predicts that, unless the verifier is perfectly reliable, these early gains will plateau and may even reverse. Indeed, our experiments across linear regression, Variational Autoencoders (VAEs) trained on MNIST, and fining-tuning SmolLM2-135M on the XSUM task confirm these theoretical insights.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies verifier-guided synthetic retraining: a loop that (i) generates synthetic data from the current model, (ii) filters it with an external verifier via a binary accept/reject rule, and (iii) retrains only on the verified subset. In a linear-regression setting, the authors give a one-step MSE bound that separates synthetic variance reduction from verification bias/variance. They then analyze the multi-round process and prove it behaves as a noisy contraction that converges to the verifier’s knowledge center. Experiments in linear regression and a CVAE on MNIST show early, clear gains with verification; long-run plateau depended on verifier quality.

### Strengths
- The work formalizes a widely used practice, i.e., filtering synthetic data with a verifier and shows how it changes collapse dynamics.
 
- A clear theoretical framework and detailed analysis followed by empirical evidence in simple setup.

### Weaknesses
- Core analysis is linear regression with a spherical verifier and a special synthetic design; extensions to non-linear models are discussed but not derived. This limits direct transfer of the rates/conditions to modern LMs/vision models. 

- MNIST/CVAE is a clean demo but dated; no language or large-scale vision results. Also, while FID trends are solid, downstream task metrics or human evals would strengthen claims about generative quality. 

- In the MNIST setup, the useful verifiers are trained with much more training data than the initial 500 data points for the VAE. In the reality, for images, we will normally use all data possible to train both the generative model and the verifier, therefore, another more practical setup is to fix the training set for both the original VAE and the verifier, and iterate from that.

### Questions
- How sensitive are the results to the shape of the verifier region? 

- A probabilistic perspective might also be interesting since these are generative models. The retraining process can be thought of as to move the original training data distribution towards the distribution defined by the verifier, which actually defines a generative model itself. When the retraining iteration goes to infinite, we are distilling the generative model defined by the verifier into a parametric generative model e.g., a VAE.

### Soundness
3

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
This paper investigates the effect of verifier quality on model collapse from training on self-generated synthetic data.  The authors structure their analysis from linear regression with a verifier that has controllable bias.  It is discovered that a mildly biased verifier may provide initial improvements, but will ultimately lead to performance plateaus or collapse over many iterations.  The authors evaluate their work on linear regression as well as preliminarily on a CVAE implementation on MNIST.

### Strengths
The paper is quite clearly presented, and the math is approachable to verify, being based on linear regression.

### Weaknesses
The paper is quite toy from the perspective of model (linear regression), data, and verifier.  This may be fine for theoretical proofs, but it does limit the scope of takeaway when considering the applicability to frontier generative models.  It would be interesting to see actual experiments on language modeling and large-scale pretrained models beyond such toy settings.  Particularly as there are a variety of reinforcement learning from verifiable rewards works on frontier models at the moment, it should not be difficult to apply it for the verifier-filtered experiments proposed here.

Even MNIST is a bit toy compared to available datasets of natural images.

The takeaways also make intuitive sense, rendering the findings a bit unsurprising - one does expect that a mildly biased verifier should help initially but performance should not be able to break free from the biases of the verifier when the scale of seen data is predominantly that filtered by the verifier.

### Questions
Why is verifier-based filtering in particular important to study thoroughly - even for frontier generative models?  Whereas naively retraining the generative model on self-generated synthetic data may indeed lead to model collapse, the field has seen success in avoiding this through verifiable rewards (Reinforcement Learning from Verifiable Rewards, as used in DeepSeek [1], Tulu3 [2]).  Why do filtered retraining from a verifier rather than use all data but labeled with a reward from the verifier (such as labelling unsuccessful generations with a 0)?  The assumption for the components is the same in both approaches, but RLVR seems to work better than verifier-based filtered training.

[1] DeepSeek-AI et al., DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025.

[2] Lambert et al., Tulu 3: Pushing Frontiers in Open Language Model Post-Training, 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper mainly studies the role of verification in preventing model collapse using a linear regression framework. Their analysis finds that, in the short term, the proper use of verification can enhance performance by reducing variance. However, in the long term, it may cause the model parameters to converge toward those used to construct the verifier. The results of this paper are supported by rigorous theoretical analysis and experimental evidence.

### Strengths
The paper is clearly written and easy to follow.

The claims in the paper are supported by rigorous theoretical analysis and comprehensive experiments.

The results are quite intuitive and make a lot of sense.

### Weaknesses
I think this paper presents a solid theoretical work. Below, I will list some drawbacks; however, they do not constitute reasons for rejection, as we all understand that in theoretical analysis, it is often challenging to address more complex but practical models.

1. During iterations, it is assumed that the synthetic covariates can only be the copies of a fixed orthonormal set. This assumption might be a little bit strong, can this be further relaxed?

2. It seems that the overparametrized regime has not been considered. Is it possible to generalize the results to the overparametrized regime?

3. In practice, the verifier may evolve over time. Could the authors consider whether a similar analysis can be conducted under this setting?

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the role of verifier-based filtering in retraining on generated data to avoid model collapse. It shows short-term variance reduction and long-term convergence toward the verifier’s knowledge center.

### Strengths
- The paper is clearly written and easy to follow.
- Verification plays an important role in preventing collapse in self-consuming models. It’s a simple and intuitive idea.

### Weaknesses
- I am mainly concerned about where the incremental theoretical contribution lies. The connection to prior verification in self-consuming works is not discussed in depth (see Question 1).
- The paper considers pure synthetic replacement. In practice, retraining typically mixes in real data or accumulates data.
- Experiments are small-scale (MNIST/CVAE only) and lack validation on larger datasets or modern models.

### Questions
1. The paper briefly cites [1] but does not clearly explain how it differs from that line of work. Could the authors clarify what the theoretical novelty is relative to [1][2], and how this work connects to recent studies [3][4]?
2. I have some concerns about experimental setup:
    - Does initializing on only 500 real MNIST images limit the baseline and exaggerate the apparent early gains?
    - The paper states that "the verifier is trained on varying amounts of real data together with an equal number of synthetic samples." Which round’s synthetic samples are used? Is the verifier retrained each iteration (i.e., changing over time), or fixed?
    - The paper mentions keeping the Top-$10\%$ of generated samples. If the verifier is binary, why is there a "top" ranking rather than simply using all passing samples (and reporting number)? If it is deterministic top-scoring selection, does this still match the theory's assumption?

[1] Damien Ferbach, Quentin Bertrand, Joey Bose, and Gauthier Gidel. Self-consuming generative models with curated data provably optimize human preferences. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[2] Xiukun Wei and Xueru Zhang. Self-consuming generative models with adversarially curated data. In Forty-second International Conference on Machine Learning, 2025.

[3] Kareem Amin, Sara Babakniya, Alex Bie, Weiwei Kong, Umar Syed, and Sergei Vassilvitskii. Escaping collapse: The strength of weak data for large language model training. In Scaling Self-Improving Foundation Models without Human Supervision, 2025.

[4] Xuekai Zhu, Daixuan Cheng, Hengli Li, Kaiyan Zhang, Ermo Hua, Xingtai Lv, Ning Ding, Zhouhan Lin, Zilong Zheng, and Bowen Zhou. How to synthesize text data without model collapse? In Forty-second International Conference on Machine Learning, 2025.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors analyze the learning dynamic of generative models iteratively retrained on their own data. The novelty comes from the fact that each step, (potentially bad) data are discarded from the retraining loop, using an external verifier.
Authors validate their theory on synthetic experiments and MNIST.

### Strengths
The topic is very timely: quantifying how much can be leverage from synthetic data is a central question for generative modelling.

### Weaknesses
- Disconnection between the analysis and motivation. As [1, 2, 3], authors analyze a linear regression setting setting, that does significantly differ from the generative model setting (as started by the authors in the conclusion). Did author try to make the analysis in the true generative model setting? Like [4]? Can authors provide justification of why to study this setting? and it is relevant to generative modelling?
- In particular, the proposed setting looks like well-studied semi-supervised learning results [4, 5]?
- Experiments: for Figure 3, authors mentioned they used the FID metric. I di not find any additional comment, so I assume authors use standard vanilla FID. If so, I think this is a mistake for multiple reasons:
    - FID rely on the Inception embedding, that is a standard for natural images: to the best of my knowledge, MNIST images are not considered natural images.
    - In addition, the point of using a embedding is to lower the dimensionality of the data, in order to have a better approximation of the empirical Wasserstein distance
To my knowledge, the best for MNSIT, is to train a classifier from scratch, and take the last layer for the embedding.
- Figure 3b, what is the "verifier training size" in the caption? Is this a way to vary the quality of the verifier?

[1] Elvis Dohmatob, Yunzhen Feng, and Julia Kempe. Model collapse demystified: The case of regression

[2] Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, and Julia Kempe. Strong model collapse

[3] Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, and Julia Kempe. A tale of tails: Model collapse as a change of scaling laws

[4] Damien Ferbach, Quentin Bertrand, Avishek Joey Bose, and Gauthier Gidel. Self-consuming generative models with curated data provably optimize human preferences

[4] Ben-David, Shai, Tyler Lu, and Dávid Pál. "Does Unlabeled Data Provably Help? Worst-case Analysis of the Sample Complexity of Semi-Supervised Learning." COLT. 2008.

[5] Zhou, Z.-H. (2018). A brief introduction to weakly supervised learn­
ing. National Science Review

### Questions
- "Suppose each eigenvalue of the design matrix X 0 is ω( n0 )", what is w? do you assume that the eigenvalues are strictly a function w of n0?
- I do not understand the takeaway of Theorem 3.1, especially I do not understand the discussion around Equation 9.
- "This contribution also clarifies a common misconception: even with a perfect verifier (θc = θ ⋆ ) and infinitely many synthetic samples in one iteration, convergence cannot occur in a single step. As
shown in Theorem 3.1, while infinite samples remove the synthetic variance term, the verification bias+variance term persists." I do not understand this comment: if the verifier is perfect, why would the verification bias persist?
- Could authors comment on the conclusion line 299, with the "3 phases" depending on the verifier, isn't it obvious that if the verifier is unbiased, it will help, and it is not a well-suited verifier, then the filtering procedure will not be helpful?

### Soundness
3

### Presentation
3

### Contribution
2
