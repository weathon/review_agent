# Continuous multinomial logistic regression for neural decoding

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Multinomial logistic regression (MLR) is a classic model for multi-class classification that has been widely used for neural decoding. However, MLR requires a finite set of discrete output classes, limiting its applicability to settings with continuous-valued outputs (e.g., time, orientation, velocity, or spatial position), which are common in neuroscience settings. To address this limitation, we propose Continuous Multinomial Logistic Regression (CMLR), a generalization of logistic regression to continuous output spaces. CMLR represents a novel exponential-family model for conditional density estimation (CDE), mapping neural population activity to a full probability density over external covariates. It captures the influence of each neuron’s activity on the decoded variable through a smooth, interpretable tuning function, regularized by a Gaussian process prior. The resulting nonparametric decoding model flexibly captures asymmetric and multimodal densities, and accommodates both linear and circular variables. To illustrate the performance of CMLR, we applied it to large-scale datasets from mouse and monkey visual cortex, mouse hippocampus, and monkey motor cortex, where it generally outperformed a wide variety of other decoding methods, including deep neural networks (DNNs), XGBoost, and FlexCode. It also outperformed a closely-related correlation-blind decoder, highlighting the importance of correlations for accurate neural decoding. The CMLR model provides a scalable, flexible, and interpretable method for decoding continuous variables from diverse brain regions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a combination of multinomial logistic regression with the gaussian processes defined on the weights.
The method is designed for neural decoding, where the goal is to estimate external variables from neural population activity, such as running speed or direction.
To handle computational complexity, the method assumed a univariate gaussian process per weight, uses standard radial basis function in the Fourier basis and trains the model as a stochastic variational inference.
Then they test the model on several neural datasets, comparing with XGBoost and DNNs.

### Strengths
1. **Clarity**. The paper is clearly written and suggests extensive (theoretical) comparison with prior work (*"Connections to Prior Work"* section)
2. **Various datasets**. The papers tests their method on various datasets from different species and data modalities (calcium imaging and electrophysiology)
3. **Provided implementation**. The code is attached in the supplementary material.
4. **Strong baselines**. The baselines are chosen to cover both bayesian and performance-drive methods (like DNN), covering the field.

### Weaknesses
1. **Unclear computational scalability**. The paper claims to provide a *"scalable framework"*, however, the practical aspects of scalability are under-explored. What are the computational restrictions? How many additional parameters the method provides and how longer does it take to train compared to modelling regression with uncertainty or the other baselines provided (such as FlexCode or XGBoost)?
2. **Baselines**.The comparison with standard multinomial logistic regression is missing.  For the other strong baselines, it is not clear how they were optimized per dataset if any optimisations were present.
3. **Lack of ablations for design choices justifications**.  While authors acknowledge in limitations that multivariate Gaussian processes could be used and fixed Fourier-domain bases and RBF kernels could be replaces by adaptive basis functions - all of these are not conceptual limitations of the methods but rather a list of sometimes very straightforward technical improvements (like multivariate Gaussian processes) and could be done within the current submission.
4. **Unclear interpretability gains**. See Q6 -  What are the additional interpretability benefits provided by CMLR? XGBoost can also give an interpretable weighted impact of each neuron on the target variable. Multinomial logistic regression with uncertainty can also be close in terms of interpretations.

### Questions
1. How is CMLR related to the following works [1-3]?
2. Why the XGBoost and DNNs lines are missing in Fig 2c and Fig 3A?
3. I might have missed it in the text but which loss function do you use to train the CMLR? Is it MSE loss everywhere?
4. Why there is no comparison with the standard multinomial logistic regression? You do not really analyse uncertainties in the main text, and even for uncertainties methods like Laplace Redux [3] could be used to derive uncertainties for a non-bayesian methods.
5. Have you tuned the hyperparameters of DNN and XGBoost per dataset? As your datasets had different sizes the ratio of data to parameters might be crucial for performance. How DNN parameters compared to the CMLR learn parameters?
6. What are the additional interpretability benefits provided by CMLR? XGBoost can also give an interpretable weighted impact of each neuron on the target variable.

References:  
[1] Chan, Antoni B. "Multivariate generalized gaussian process models." arXiv preprint arXiv:1311.0360 (2013).  
[2] Payne, Richard D., et al. "A conditional density estimation partition model using logistic Gaussian processes." Biometrika 107.1 (2020): 173-190.  
[3] Daxberger, Erik, et al. "Laplace redux-effortless bayesian deep learning." Advances in neural information processing systems 34 (2021): 20089-20103.  
[3] Murray, Iain, David MacKay, and Ryan P. Adams. "The Gaussian process density sampler." Advances in neural information processing systems 21 (2008).

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
The authors introduce continuous multinomial logistic regression (CMLR), a flexible nonparametric model that allows for both discrete- and continuous-valued outputs by mapping inputs to a full probability density using per-feature additive functions with Gaussian priors. The resulting model is applied to a wide variety of neural decoding tasks, both continuous and discrete, and shows impressive results compared to baselines.

### Strengths
The paper is well-written and clearly articulates both the gap in the current literature and the exposition of the method. As applied to the neural decoding problem, the weight functions become per-neuron tuning curves that provide more interpretability than tree-based or neural network approaches. The method is applied to a range of neural decoding problems (sensory and motor) and shows strong performance across all datasets. The limitations and future directions section makes clear that this is a rich model class with many exciting directions to explore, and the paper is a strong foundation from which to start this work.

### Weaknesses
My main concern is the thoroughness of the baseline comparisons. The authors chose a different number of Fourier components M and mini-batch size N' for each of their datasets, but did not clearly state how these values were chosen. Furthermore, it seems no attempt was made to search hyperparameter space for XGBoost or the DNN. Better hyperparameter tuning (especially for the baselines) would therefore give me more confidence in the stated results. The authors could, for example, choose 2 or 3 important hyperparameters per method and search over these by performing a train/val split of the 80% of training data and selecting the best hyperparameters on the validation data.

I am also concerned about the robustness of the results if they are indeed only reported on 20% of the data. My own experience in neural decoding has been that the train/test split can significantly affect model performance, and conclusions from one split may not hold with another split. Performing full k-fold cross validation (where every trial lands in the test set exactly once) would better control for noise introduced in the sampling process.

### Questions
What is the computational complexity of CMLR, i.e. how do training and inference time scale with D, T, M, J? The authors have included training times in the Appendix, but it might be useful to mention these numbers (or at least their order of magnitude) more explicitly in the main text.

How much data do I actually need to train CMLR? Fig S3 is very cool and helpful, and it would be interesting to see something similar with real data. Do the GP priors/additive structure of CMLR make it more amenable to fitting models with less data, as compared to XGBoost or DNNs?

Fig 2E/3C/4C/S4C: what is the value of J used for CMLR/FlexCode/NB? How was this value chosen?

I find the brief mention of uncertainty calibration very interesting; it is well-known that DNNs, for example, are very often poorly calibrated. It would be cool to see the PIT results (i.e. Fig S5) for some of the other methods. Is it possible to do this for DNNs? Better calibration could be as strong a selling point as better accuracy for scientific applications, and might be worth emphasizing this in the main text more.

More with uncertainty calibration: having never seen PIT histograms, I can believe they are a good approach for quantifying calibration, but they feel very disconnected from the actual datasets that are being analyzed. The mouse hippocampal data offers an interesting example: here CMLR tends to make mistakes when the true or decoded position is at one end of the track. What do uncertainty estimates look like for these mistakes? Does the uncertainty scale with the magnitude of the error? How do those compare to uncertainty estimates from the other methods?

Minor: not having a background in this literature, the following sentences in the first Intro paragraph were a bit confusing to me: "However, many neural decoding tasks involve continuous variables... . In such settings, researchers who wish to use MLR-like decoding methods are commonly forced to discretize the output variable into a finite number of classes." At this point I don't really know what "MLR-like decoding methods" are and the first thing I think of is "sure but doesn't linear regression work just fine?" Perhaps clarifying this point early will help convince readers unfamiliar with the literature.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces CMLR, a generalization of classical multinomial logistic regression to handle continuous output variables. Instead of discrete class weights, CMLR defines smooth, output-dependent weight functions with GP priors, allowing it to model conditional probability densities over continuous variables such as time, orientation, or spatial position. The authors derive a SVI algorithm in the Fourier domain for scalable learning on large datasets. They show CMLR’s performance on diverse neural decoding tasks across multiple brain regions (mouse and monkey V1, hippocampus CA1, motor cortex), and show that it outperforms Naive Bayes, FlexCode, XGBoost, and DNN in both accuracy and calibration.

### Strengths
1. Although it relies on strong modeling assumptions than nonlinear methods, the approach is flexible enough to handle circular and multidimensional outputs.

2. The method is interpretable, learning weight functions that correspond to tuning curves.

3. This method offers an attractive alternative for researchers who prefer to avoid the complexity and hyperparameter tuning required by nonlinear models while still getting strong decoding performance.

### Weaknesses
The DNN results in Figs 2 and 3 show relatively poor performance with large variance. While I understand that DNNs typically have higher variance than the proposed method, the degree of underperformance here suggests that the architecture or hyperparameter tuning might not have been well optimized for these decoding tasks. This raises some concern about the fairness of the comparison. It would be helpful for the authors to acknowledge this limitation in the paper. If, on the other hand, the proposed method achieves strong performance without requiring extensive hyperparameter tuning, that could be an additional advantage of the proposed method.

Relatedly, I would encourage the authors to clarify how they see the contribution of this work in the context of increasingly expressive modern models and large-scale neuroscience datasets. Although the proposed method is more interpretable, it may lag behind nonlinear models (e.g., transformers) in decoding performance. From a practical standpoint, the utility of the method might be limited. I would appreciate a clarification of how the authors envision their approach complementing or coexisting with more complex models.

### Questions
Despite using SVI for scalability, the method may still be computationally expensive for large-scale or high-dimensional datasets. A theoretical or empirical runtime comparison with baselines would strengthen the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper develops a novel approach, Continuous Multinomial Logistic Regression (CMLR), that is an extension of MLR to predict continuous-valued outputs, and showcase this approach for neural decoding (predicting behaviors/stimuli from neural activity). The model enables interpretable understanding of smooth 'tuning curves' of neurons within the decoding model via GP priors. They develop an approach to fit the model with stochastic variational inference. The authors demonstrate excellent neural decoding performance, in additional to interpretability of the model across multiple datasets using their method.

### Strengths
This is an original, creative, and novel technical development for neural decoding (and prediction problems more broadly). The method has a good blend of accuracy and interpretability (via the learned tuning curves within the model). The paper is clearly written and the authors are very thorough. It is a very strong paper overall.

### Weaknesses
My one fundamental concern is with the comparisons in results, specifically in terms of hyperparameter selection. I might have missed it, but it's not clear to me how/if hyperparameter tuning was done both for the author's model and for comparison models. It seems odd how poor many of the results are from what should be close to state-of-the-art approaches (XGBoost and DNNs), which makes me suspect poor hyperparameters (causing either overfitting or underfitting). Proper hyperparameter tuning on a held-out validation set (within the training set) should be done, if it wasn't already.

### Questions
1. Fig 4C - Euclidean error is hard to interpret - Coefficient of determination (the standard r2_score in sklearn) would be easier to interpret, and more standard for velocity decoding

2. Discussion: "First, unlike models such as XGBoost or DNNs, CMLR does not incorporate priors over outputs" - I don't understand this statement in terms of XGBoost and DNNs having priors on outputs.

3. Line 263 - closing bracket ] missing

4. I would put the methods in section 5.3 into an actual methods section. They feel very out of place when reading results

5. How long does your approach take to run compared to others?

### Soundness
3

### Presentation
3

### Contribution
4
