# Adaptive Priors from Learning Trajectories for Function-Space Bayesian Neural Networks

- Decision: Reject
- Scores: 5, 5, 5, 1, 3

## Abstract
Tractable Function-space Variational Inference (T-FVI) provides a way to estimate the function-space Kullback-Leibler (KL) divergence between a random prior function and its posterior. This allows the optimization of the function-space KL divergence via Stochastic Gradient Descent (SGD) and thus simplifies the training of function-space Bayesian Neural Networks (BNNs). However, function-space BNNs on high-dimensional datasets typically require deep neural networks (DNN) with numerous parameters, and thus defining suitable function-space priors remains challenging. For instance, the Gaussian Process (GP) prior suffers from scalability issues, and DNNs do not provide a clear way to set appropriate weight parameters to achieve meaningful function-space priors. To address this issue, we propose an explicit form of function-space priors that can be easily integrated into widely-used DNN architectures, while adaptively incorporating different levels of uncertainty based on the function's inputs. To achieve this, we consider DNNs as Bayesian last-layer models to
obtain the explicit mean and variance functions of our prior. The parameters of these explicit functions are determined using the weight statistics over the learning trajectory. Our empirical experiments show improved uncertainty estimation in image classification, transfer learning, and UCI regression tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In function-space VI there are two challenges: (1) how to specify meaning prior. (2) how to calculate the KL term in ELBO. This paper proposes to tackle the first challenge by first using SWAG to get a learning trajectory, and then constructing a tractable "prior" form based on it. Specifically, the authors only treat the last layer Bayesian and treat the feature extractor as fixed. The weighted sum of trajectory history is used to construct the feature extractor and the "prior". For the second challenge, the authors propose to use the adversarial context feature which removes the need of an external dataset for the context set. Experiment results show the proposed method has good uncertainty estimation in image classification and UCI regression.

### Strengths
- The adversarial context feature removes the need for an extra dataset as a context set.
- The constructed tractable "prior" is based on a posterior, and hence is capable of assign large variance for data differ from training set
- The paper is well-written and easy to follow.

### Weaknesses
- The proposed method is only for last-layer Bayesian neural network
- The construction of "prior" is computationally expensive and might be sensitive to the initialization

### Questions
- I put a quote sign on prior because this is more like a posterior. By definition prior is your belief in the problem before you see the data, but here the "prior" is constructed using SWAG and is loosely connected with the posterior obtained by SWAG. I think there needs to be at least a few sentences discussing this.

- The construction of the feature extractor seems overly complicated, is it really necessary to do so? What happens if you just use the MAP?

- The prior variance for T-FVI (line 472) seems pretty large (5, 10, 50), as I'm not very familiar with function space VI, I'm curious is this very common for FVI?

- As the adversarial context feature is combined with the tractable prior, it's hard to see the effect of it. How much does it affect the results when compared with using extra dataset as context set?

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
2

### Summary
The authors propose an explicit function-space prior for DNNs that can be applied to common model architectures and overcomes limitations of other function-space BNN approaches. They utilize DNNs as bayesian last-layer models and leverage the learning trajectory of the SGD from a set of checkpoints. They demonstrates the effectiveness across several (image) datasets and models, offering a practical method for function-space BNNs.

### Strengths
- **Clear structure**: From a macro perspective, the paper is well-organized, with a logical sequence of topics for good readability. Each section transitions smoothly, making it easy to follow the progression of ideas.
- **Contribution**: The authors seem to present an interesting and promising approach to function-space BNNs, addressing limitations that have previously hindered practical applications. The results demonstrate improvements over existing methods, showcasing the effectiveness of the proposed approach. However, since I am not very familiar with this field, I must trust the authors in this case. 
- **Comprehensive introduction, limitations and related work**: The introduction effectively outlines the challenges of BNNs, particularly in the function-space domain, and establishes the motivation for this work. Section 3 is particularly well-structured, providing clear explanations of the limitations of BNNs and framing them within specific perspectives. The related work section makes clear distinctions between previous approaches and the authors' work.
- **Technical explanations**: The technical explanations seem well-organized, making the complex concepts easier to follow. 
- **Diverse experimental perspectives**: The experiments cover three distinct perspectives, providing a quite comprehensive evaluation of the proposed approach and supporting its versatility.

### Weaknesses
**Grammar and Presentation**

Overall, the paper appears rushed and has numerous grammatical/wording issues. While they are minor individually, these errors accumulate and do not achieve the standard expected for an ICLR paper. Below is a list of specific issues I observed during my review (I eventually stopped listing them all):

- L.43, 46, 47-48, L.54: Improper usage or missing articles ("a," "the," pluralization issues).
- Abbreviations are repeatedly reintroduced (e.g., GP is introduced both in the intro and again at L.139).
- Abbreviation clarity: Some abbreviations have consistent capitalization (e.g., SGD), while ELBO only capitalizes "evidence".
- L.45: Should be "scalability issues" not "scalable issues".
- L.38: "NNs" is not introduced properly.
- L.39: "facilities" should be corrected to "facilitates".
- T-FVI is introduced for "function-space variational inference" but is inconsistently referenced (e.g., missing in L.54).
- L.409-410: "However, unlike theses work" is wrong. 
- Inconsistent capitalization (e.g., L.432 has "Averaging" capitalized).
- References: VIT transformer paper is referenced twice; some arXiv papers are cited instead of their final venues.
In related work: many citations are not properly bracketed (using \citep) (e.g., L.401, 406, 417).


**Clarity**

Another major issue is the overall clarity, which makes the paper difficult to follow in detail:

- The contributions are not clearly stated, particularly with respect to differences in “…, and adaptively introduce different levels of uncertainty based on the function’s inputs” and “adaptively incorporate higher uncertainties for each function’s input”, which are used frequently. To me, it is particular unclear what “higher uncertainties” means and why/how it is useful. This terminology is used throughout the paper without sufficient explanation.

- "Higher-dimensional dataset" from the introduction needs clarification. What does this mean exactly, and why is it relevant to the discussion? This term is introduced for motivation but not addressed in the results. Are the datasets used in the results actually high-dimensional?

- The introduction would benefit from a brief explanation of why BNNs are advantageous in the first place.

- The phrase "widely-used DNN architectures" could be more precise. Does the method apply to all CNNs, transformers, or specific models? Also, clarify whether it applies to image classification models exclusively or to other classification tasks as well (only mentioned shortly in the abstract). Are the methods also applicable to other domains? I do not suggest do to more experiments but rather be more precise about the domain and maybe mention limitations/possibilities in others. The data types should be more clearly stated in the introduction. 


**Result presentation**

The results section is short and difficult to follow due to the extensive usage of the appendix. It may be more effective to focus on two of the perspectives or to create more space for a comprehensive discussion.

- Table 1: The abbreviations for methods are not introduced. Although explained in the appendix, the table itself is hard to interpret without this.
- Tables and figures in general not well-placed throughout the text (e.g., Figure 5 is referenced before Figure 4, and Figure 4 appears two pages before it is discussed). This disrupts the flow and makes it challenging for the reader.
- While the design decisions regarding datasets and models may be understandable for image domain experts, a brief motivation for these choices would be beneficial.
- It’s unclear why only a subset of methods was selected for Table 2 (transfer approach); why? 
- Table 2: Metrics like auroc-c and auroc-s are not properly introduced in the text.
- The sections on transfer learning and regression are very brief, which is especially notable as the transfer approach is a highlighted contribution. Metrics and abbreviations for the regression task are not well-introduced, causing further confusion..

**Short Conclusion**
- The conclusion is very short and lacks a discussion of future work. 
- While it’s good that limitations are noted, the impact on practical adoption should be discussed. Additionally, suggestions on how this might be improved in the future would add value.

### Questions
I encourage the authors to address the weaknesses highlighted above, as well as to provide answers to the questions.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a novel method for specifying function-space priors in Bayesian neural networks that can be integrated to widely used neural network architectures. The parameters of these explicit functions are determined using the weight statistics over the learning trajectory by leveraging stochastic weight averaging Gaussian (SWAG) method. The work is motivated to address the scalability challenges of function-space priors such as Gaussian process prior in large neural network architectures with many parameters and high dimensional datasets. The proposed approach utilize context feature to compute the function-space KL divergence without relying on external datasets. Empirical evaluation is performed with experiments on image classification and regression, demonstrating the proposed method is effective in improving uncertainty estimation.

### Strengths
* The paper addresses an important problem of specifying meaningful function-space prior in Bayesian neural networks based on the learning trajectories through data.

* The use of adversarial context feature to compute the functional-space KL-divergence without relying on external dataset is an interesting approach, which could make the model to be more robust.

*  Empirical results supports the proposed method. However, the empirical evaluation could be strengthened by including experiments where all layers of the model are Bayesian. Please see the comments in the next section.

### Weaknesses
* While the paper is motivated to address the scalability issues of traditional Gaussian process priors with high-dimensional datasets and large neural network architectures with many parameters, the experimental evaluation is limited to a model where only the final layer is Bayesian. This raises concerns about the scalability of the proposed approach. Additionally, prior works [Ovadia et al. 2019, Xiao et al. 2022] has shown that the quality of uncertainty estimation in models where only the last layer is Bayesian is inferior to other comparable methods, such as Deep Ensembles and Monte Carlo Dropout. I encourage the authors to evaluate with a model (e.g. ResNet-18), where all layers are Bayesian in their experiments. This analysis would address the scalability concerns of the proposed function-space priors.

* The paper provides literature review of related works, but do not compare with any of them in their experiments. For example, how does the proposed method compare with other methods that define priors through Empirical Bayes [Krishnan et al. 2020, Shwartz-Ziv et al. 2022]

[Ovadia et al. 2019] Ovadia, Yaniv, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. "Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift." Advances in neural information processing systems 32 (2019).

[Xiao et al. 2022] Xiao, Yuxin, Paul Pu Liang, Umang Bhatt, Willie Neiswanger, Ruslan Salakhutdinov, and Louis-Philippe Morency. "Uncertainty Quantification with Pre-trained Language Models: A Large-Scale Empirical Analysis." In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 7273-7284. 2022.

[Krishnan et al. 2020] Krishnan, Ranganath, Mahesh Subedar, and Omesh Tickoo. "Specifying weight priors in bayesian deep neural networks with empirical bayes." In Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 04, pp. 4477-4484. 2020.

[Shwartz-Ziv et al. 2022] Shwartz-Ziv, Ravid, Micah Goldblum, Hossein Souri, Sanyam Kapoor, Chen Zhu, Yann LeCun, and Andrew G. Wilson. "Pre-train your loss: Easy bayesian transfer learning with informative priors." Advances in Neural Information Processing Systems 35 (2022): 27706-27715.

### Questions
* what is the motivation behind the adversarial context feature, specifically why FGSM attack method? Any other perturbations/corruptions to the features will serve the purpose of not relying on the external dataset?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
3

### Summary
The paper introduces a new prior designed for functional variational inference in Bayesian neural networks. To achieve this, the authors propose an explicit function-space prior by modeling deep neural networks (DNNs) as Bayesian last-layer models. Initially, the method constructs a prior in weight space similar to the approach used in SWAG (stochastic weight averaging Gaussian). From this weight-space prior, the corresponding functional prior is derived. The effectiveness of the proposed method is demonstrated through validation on tasks including image classification, transfer learning, and UCI regression.

### Strengths
- The paper aims to tackle an important problem which is choosing sensible priors for Bayesian neural networks.
- The paper is well-written and provide good illustrative figures that enhance the clarity of method.

### Weaknesses
-	The contribution is marginal, as functional variational inference is a well-established area in the literature. Additionally, the approach of constructing priors using SWAG has already been explored by Shwartz-Ziv et al. (2022). 
-	The proposed prior is not valid from a Bayesian perspective, as it resembles a posterior. It is constructed from the training data and multiple trained models, as illustrated in Figures 2 and 5.
-	The idea presented in this paper appears to be very similar to the work found at https://openreview.net/pdf?id=AZVmYg3LvS. If this paper is a revised version, a comparison with Table 2 in that paper raises a concern: why are all the results for the baseline methods in Table 1 significantly worse, particularly for the NLL, ECE, and AUROC metrics?
-	From lines 47 to 54, the paper discusses the work of Rudner et al. (2022), highlighting that it restricts the randomness to the last layer due to GPU memory limitations, which can reduce the flexibility of Bayesian neural networks (BNNs). However, the paper ultimately adopts a similar approach, effectively presenting a simple extension of that work. This could be misleading to readers, as the paper suggests a novel contribution while largely replicating the limitations discussed.

### Questions
Please see the third bullet in the Weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This manuscript presents a function-space prior that can be integrated into common deep neural network architectures, treating DNNs as Bayesian last-layer models. The mean and variance functions for the prior parameters are set using weight and feature statistics from the learning trajectory.

### Strengths
The manuscript focuses on the important aspect of using informative priors and applying them to efficient Bayesian inference.

### Weaknesses
The important weakness of the work is limited novelty. The method combines the two approaches SWAG and T-FSVI. The extensions proposed in this work are limited in novelty. There are already some works (eg. [1]) in the literature that capture the weight priors from the phase-I of training the neural network models which provide better accuracy and uncertainty quantification, for larger models.
From the results, the benefits of phase-I are limited. 

[1] Krishnan, Ranganath, et.al. "Specifying weight priors in Bayesian deep neural networks with empirical bayes." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 04. 2020.

### Questions
Please refer to comments in weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
