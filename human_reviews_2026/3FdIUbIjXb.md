# A Conformalized Inference on Unobservable Variables

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Quantifying uncertainty in predicted unobservable variables is a critical area of research in statistics, artificial intelligence, and empirical science. Most scientific studies assume a specific structure involving unobservable variables for the data-generating process and draw inferences from a parameter of interest within that framework. Conformal prediction is a popular model-agnostic method for constructing prediction intervals for new observations. However, it typically requires observed true labels to build the prediction interval, making it unsuitable for unobserved latent variables. We propose a method to construct a prediction interval by leveraging sample-splitting of the training data and analyzing the discrepancy between two independently trained models. To ensure the identifiability of the distribution of this conformity score, we introduce a few assumptions regarding the distribution of the residuals of the predictions. Furthermore, we propose a residual orthogonalization to satisfy these assumptions with a coordinating regularization term. The performance of the proposed method was evaluated using both simulation and large language model experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Latent Conformal Prediction (LCP), a way to get coverage-valid prediction intervals for unobserved latent variables. The method splits data, trains two independent models, and uses their prediction discrepancy on a calibration split as a proxy for the residual distribution of an averaged predictor, enabling conformal calibration without ever observing the latent truth. To make this proxy identify the correct residual law, the authors posit mild assumptions on prediction residuals and propose residual orthogonalization. They demonstrate the approach on synthetic tasks and LLM-based preference learning, showing informative intervals and empirical coverage close to target levels, thereby extending CP to practical settings where the quantity of interest is never directly observed.

### Strengths
1. This work addresses a crucial but largely unexplored problem; the contribution is clear.
2. The proposed method is intuitive, lightweight, and creative.

### Weaknesses
1. The writing and presentation is unclear. Section 4 mixes problem setup with application examples; the general problem formulation should come first, with application-specific details moved to the experimental setup.
2. Framing the motivation through binary classification narrows the perceived scope. The paper’s true focus is uncertainty quantification for unobserved latent variables, and the narrative should emphasize this broader contribution.

### Questions
1. See weaknesses above.
2. The related work lacks discussion of CP methods aimed at improving prediction efficiency/informativeness in classification.
3. In Figure 1, what is α? If it denotes the miscoverage rate, it must lie in [0,1]; how can it be 3.0? Please clarify the axis/units or notation.
4. Please empirically validate the assumptions and provide a sensitivity analysis that characterizes performance as the degree of assumption violation increases.
5. Please provide intuition for the assumptions and for Theorem 2: What do the conditions mean operationally, and how do they translate into coverage?
6. How to adapt LCP to setting where data has potential distribution shifts?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new method called Latent Conformal Prediction (LCP) to quantify uncertainty and construct valid prediction intervals for unobservable latent variables. 
Standard conformal prediction methods are unsuitable for this task as they require observed true labels, which are unavailable for latent variables. The core idea of LCP is to use sample-splitting to train two independent models; by analyzing the discrepancy between these two models' predictions on a calibration set, the method can estimate the distribution of the conformity score without ever seeing the true latent values. 
To ensure the statistical identifiability of this residual distribution, the authors introduce a set of assumptions. The method's performance and ability to achieve guaranteed coverage rates are validated through experiments on synthetic binary classification data and a LLM preference learning task.

### Strengths
1. This paper provides a way to create valid prediction intervals for unobservable variables. This is important in the field of statistics. 
2. The paper includes a mathematical proof (Theorem 1) that, under its assumptions, the prediction intervals it creates will achieve the desired coverage level.
3. This paper also proposes "Localized LCP", an extension that makes the prediction intervals adaptive. 
4. The method is shown to work on both synthetic data and a real-world LLM preference learning task.

### Weaknesses
[Major Concern] I think the assumptions 1-3 are still too strong. It roughly means that one can construct independent (and symmetric) copies based on these assumptions. If so, with these assumptions, one could indeed construct the confidence interval. However, this hardly happens in reality. 

Another thought: if those assumptions hold, it seems that one can directly use the interval [-M, M] where $M = quantile(|\hat{f}(X_i)|)$. Since $\hat{f}(X_i)$ is also independent copies (on the probability space over x and y given the calibration set), it seems that the coverage still holds here? 

The authors try to solve this problem via the residual orthogonalization; however, it seems that this introduces another additional dependency on the trained parameter theta and the calibration fold. I think this is invalid in the conformal prediction literature. 

Minor concern: 
1. It would be better if the authors could conduct some more experiments on the hyperparameters gamma and K.
2. The data efficiency is a little bit loose here (not a big issue)
3. Seems that the coverage is not always guaranteed in the experiments? (fig1 and fig2)

### Questions
See above.

### Soundness
2

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
2

### Summary
This paper proposed a way of constructing prediction intervals for unobservable latent variables. Though I'm not a subject matter expert, I still think the paper provides valid theoretical and empirical contributions to the field of Conformal Prediction.

### Strengths
1. It's a nice new problem formulation. I haven't seen this before. Studying latent variables intervals are a nice thing to have and I'm quite surprised the existing literature on this problem is so sparse. 
2. The theoretical foundations are solid, and the coverage guarantees are nice. Specifically, I couldn't find any glaring issues with any of the proofs of the main claims. 
3. Good exploration of the application of the technique. Covering all of LLM preferences to binary classification is very good to see for conformal prediction papers.
4. Addressing heteroskedascity is good to see. It's not a central claim in this paper to my understanding but it is nevertheless a good additon. 
5. Nice addition with DPO. Again applying conformal prediction to help improve language modeling techniques is great!

### Weaknesses
None too serious.
1. Strong assumptions, but it's ok. Probably not possible to have results without them. Particularly, Assumption 2 is pretty strong, saying that the distribution of the residuals is equivalent. It would have been good to add an error analysis, if they are both guassians, based on the difference of the means.
2. The baseline mix could have been a bit stronger. A few more relevant baseliens would have been nice.  To my understanding, there are no real baselines of other methods in this paper? That would be really great to add. 
3. Verifying the assumptions empirically would have been a nice addition. Especially assumptions 1 to 3 that assume properties on the distributions of the residuals. Perhaps you could take some datasets and measure this difference and see how reasonable these are?
4. The notation was a little hard to read IMO. Especially between propositions 1 and 2.

### Questions
1. Can you see if you can add an analsysis based on the difference of the measn of the distributions for Assumption 2?
2. Can you add more baselines to the paper? There aren't any right now. 
3. Can you add an experiment to verify the assumptions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method called Latent conformal prediction to construct prediction intervals for latent variables instead of final outcomes. For example, the proposed approach offers a prediction interval of the predicted probability of classification instead of a confidence set of the class labels. The authors also use a regularization term to satisfy the orthogonality assumption required by their algorithm. Finally, experimental results on synthetic and real-world data are provided.

### Strengths
It is an easy-to-read paper. The paper is written in an intuitive approach. The authors showed an interesting use of conformal prediction for LLMs. The algorithm seemed very simple but effective.

### Weaknesses
Below I provide my comments.

### Major:
1. The authors mentioned the importance of obtaining correct latent variables in a system. However, it is not clear how a predictive interval of latent variables can help in decision-making. I would request the authors to make this precise.  
2. Even though we do not have ${Z_i}_{i=1}^n$  in the given dataset, 

can we extract 
${Z_i}_{i=1}^n$ 

from the model trained on 
${(X_i, Y_i)}_{i=1}^n$ ? 

Then, a prediction interval can be constructed from the extracted ${Z_i}_{i=1}^n$. I would request the authors to clarify this.  
3. It is not clear how the authors identify $\tilde{U}$ for $\tilde{f}$ by $\hat{V}/2$. The authors should provide some intuitive proof sketch in the main paper.  
4. No baselines. The paper does not compare the proposed approach with existing conformal prediction or uncertainty quantification methods. 
5. What’s the application of this approach? The authors should provide concrete examples or scenarios where constructing prediction intervals for latent variables is beneficial.   

### Minor:
1. The discussion in the remark from lines 303 to 313 should be discussed more intuitively. The main point is slight obscure. It is also not clear how  characteristic function not having a compact support affects  estimating the distribution of $\tilde{U}$.  I would request the authors to make this clear.
2. If with Corollary 1, the authors do not need the Fourier transform to calculate the distribution of the conformity score. Then, what is the utility of the remark at line 303 discussing the Fourier transform and inverse Fourier transform?  
3. The authors should provide more discussion on the LLM experiments. 
4. It is not clear how the metrics, such as length and MSE, are used to evaluate the algorithm.  
5. For calculating the MSE, how do the authors know the ground-truth latent variables in the LLM experiments?

### Questions
Here are my questions.

### Questions:
1. We have access to the outcomes $Y$ in our dataset. However, the latent variables should depend on the models we are training. How can that be model-agnostic? For example, the authors mentioned preference learning with the Bradley–Terry model. Since the reward depends on the model, the predictive interval will not be model-agnostic, right? I would request the authors to explain this.  
2. What did the authors mean by saying, “Instead of the logit of the probability of $Y = 1$, the problem is not normalized because there is one-dimensional freedom for translations of the reward $r(W, Y)$”?  
3. Why does Assumption 1 generally not hold?  
4. “We could expect the symmetric fluctuation of the predictions with respect to the random perturbation of the observation in the training set.” How do the authors randomly perturb the observations?

### Soundness
3

### Presentation
3

### Contribution
2
