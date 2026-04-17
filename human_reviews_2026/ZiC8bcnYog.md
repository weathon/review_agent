# Cross Spline Net: A Simpler, More Interpretable and Unified Machine Learning Framework

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
We propose a new machine learning framework called cross spline net (XSN), along with a set of interpretability methods to understand the model. The framework is built on a combination of spline transformation and cross-network (Wang et al. 2017, 2021). Compared to the traditional black-box machine learning models like XGBoost and fully connected neural network (FCNN), the XSN framework has a few key advantages. First, it is simpler and less overfitted while being as performant or better in some cases. Second, it is flexible and unifies a variety of machine learning models under the same framework. With different choices of the spline layer, we can reproduce or approximate a set of non-neural network models (tree, MARS, SVM, etc.) under the same, unified neural network framework. By using scalable and powerful optimization algorithms available in neural network libraries, XSN avoids some pitfalls (such as being ad-hoc, greedy or non-scalable) in the original optimization methods used in these non-neural network models. Finally, we equip XSN with a set of interpretability tools that help users understand the model composition and feature effects, which is crucial to gain insights and confidence in the deployed model. We will use a special type of XSN, TreeNet, to illustrate our point. We believe XSN will provide a flexible and convenient framework for practitioners to build performant, robust and more interpretable models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces Cross Spline Net (XSN), a novel and interpretable machine learning framework. XSN features a simple yet effective architecture that unifies a range of existing ML techniques. Unlike traditional methods, it can leverage modern deep neural network (DNN) libraries for efficient optimization, while also offering built-in tools for interpretability.

### Strengths
- The explanations of the experimental settings are clear and sufficiently detailed.

### Weaknesses
- **Novelty:** The novelty of this work is questionable, as many similar approaches have been proposed in the past five years. For instance, [1, 2] also introduced unified frameworks bridging traditional ML methods and DNNs through polynomial formulations, while [3, 4] explored interpretable models based on polynomial neural networks and generalized additive models. Moreover, spline-based architectures for interpretability have already been utilized in KAN [5]. Although this work may differ in implementation details, the key distinctions are not clearly articulated in either the introduction or related work sections.
- **Claimed Advantages:** The authors argue that one advantage of XSN is the ability to apply DNN optimizers (e.g., Adam) to traditional ML techniques, whose original optimization procedures are often greedy, ad hoc, or not scalable to large datasets. However, many traditional ML algorithms involve convex optimization problems, for which stochastic optimizers may be unnecessary or even detrimental, potentially compromising desirable theoretical properties. Even if scalability is improved, such integration can typically be achieved without relying on XSN—for example, `scikit-learn` already supports stochastic gradient descent (SGD) for many estimators.
- **Framework Clarity:** Although the paper claims that XSN unifies a wide range of ML methods, the relationships between this framework and existing approaches are not clearly explained. Only a brief paragraph is dedicated to this topic, leaving the connections underdeveloped.
- **Experimental Results:** In the reported experiments, XSN underperforms several baseline methods in many cases. Moreover, training the TreeNet component appears significantly slower than both FCNNs and XGBoost. These results cast doubt on the practical utility and efficiency of the proposed framework.

[1] Zhang, Jiawei. "Rpn: Reconciled polynomial network towards unifying pgms, kernel svms, mlp and kan." _arXiv preprint arXiv:2407.04819_ (2024).

[2] Zhang, Jiawei. "RPN 2: On Interdependence Function Learning Towards Unifying and Advancing CNN, RNN, GNN, and Transformer." _arXiv preprint arXiv:2411.11162_ (2024).

[3] Duong, Viet, et al. "Cat: Interpretable concept-based taylor additive models." _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_. 2024.

[4] Dubey, Abhimanyu, Filip Radenovic, and Dhruv Mahajan. "Scalable interpretability via polynomials." _Advances in neural information processing systems_ 35 (2022): 36748-36761.

[5] Liu, Ziming, et al. "Kan: Kolmogorov-arnold networks." _arXiv preprint arXiv:2404.19756_ (2024).

### Questions
- **Related Work:** The paper should include a dedicated section on related work, discussing prior studies on polynomial neural networks, existing unified ML frameworks, and Generalized Additive Models (GAMs). This would help position XSN within the broader research landscape and clarify its contributions relative to established methods.
- **Terminology Clarification:** The authors should clearly define the term _“spline transformation.”_ As presented, it does not appear to correspond to the conventional concept of spline interpolation or regression, which typically uses low-order piecewise polynomials to approximate functions. A precise explanation is necessary, as this term does not seem to be widely used or recognized in the existing literature.
- **Figure Quality:** All figures in the paper should be presented in vector format to ensure clarity and readability.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Cross Spline Net (XSN), a novel machine learning framework that integrates spline transformations with cross-networks to create a unified, interpretable architecture for tabular data modeling. The authors demonstrate how XSN subsumes diverse model classes (including trees, MARS, and polynomial regression) under a single neural framework while leveraging modern optimizers like ADAM to overcome limitations of traditional greedy or ad-hoc training approaches.

### Strengths
The unification of spline transformations and cross-networks is an elegant contribution that bridges neural and non-neural approaches. By showing how XSN can approximate trees, MARS, and linear models, the authors provide a compelling framework for understanding these methods through a common lens. This conceptual clarity is rare and valuable.

The proposed interpretability tools (Algorithms 1–2) are methodologically sound and practically useful. The purification algorithm for enforcing hierarchical orthogonality is particularly noteworthy, as it addresses a key challenge in functional ANOVA decomposition. Visualizations of purified components (e.g., bike-sharing interactions) effectively showcase real-world insights.

### Weaknesses
The paper lacks theoretical guarantees regarding algorithmic convergence or the interplay among multiple splines, which is a significant concern for a novel framework. Without analysis of the optimization landscape—such as convexity properties or Lipschitz conditions—the reliability of XSN's training process remains uncertain. This omission is particularly noticeable given the hybrid nature of the architecture combining splines and cross-networks, where interactions between components could lead to complex loss surfaces. While empirical results are promising, theoretical grounding would substantially strengthen the methodological foundation.

The experimental evaluation, though extensive, has notable limitations in scope. High-dimensional scenarios (e.g., p > 100) are entirely absent, leaving scalability to modern large-scale datasets unverified. Given that spline transformations inherently expand feature space, performance degradation in high dimensions is a plausible concern that warrants investigation. Similarly, while runtime comparisons include 1M samples, the analysis focuses on computational speed rather than statistical performance at scale, leaving questions about stability and generalization in massive data regimes.

Baseline comparisons omit several critical competitors. The exclusion of modern interpretable methods like GAMI-Net and Explainable Boosting Machines is conspicuous, as these directly target the same niche of transparent yet performant tabular modeling. The comparison with XGBoost3 also appears methodologically skewed: constraining tree depth to 3 artificially limits its capacity to model higher-order interactions, while TreeNet2 explicitly permits up to three-way interactions. This asymmetry undermines the fairness of comparative claims.

Parameter sensitivity analysis is surprisingly absent for a framework advocating "off-the-shelf" usability. The robustness of TreeNet2 to variations in hyperparameters—particularly the number of spline bases (m), projection dimension (d), and learning rate—remains unexplored. This gap is problematic because real-world deployment often requires adaptation to diverse data characteristics, where sensitivity to default settings could limit practical utility. The paper's assertion about TreeNet2's generalizability would be more convincing with evidence of stability across hyperparameter perturbations.

Finally, the investigation of spline transformations is narrow, focusing exclusively on sigmoid-based functions. While suitable for approximating tree structures, this choice may not optimally capture other functional forms (e.g., periodic patterns or discontinuities). The framework's flexibility would be better demonstrated through a comparative analysis of alternative spline bases, which could reveal performance trade-offs and guide practical implementation.

I would be willing to raise my score if the above issues are well solved. Please correct me if I am wrong.

### Questions
Please refer to the weakness above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
They proposed a new machine learning framework called Cross Spline Net (XSN), which includes TreeNet - an interpretable model that estimates each feature’s main effect using sigmoid-based basis functions, applies a linear dimension-reduction layer, and captures feature interactions through a cross layer. The model was evaluated on real-world datasets.

### Strengths
**(S1).** The proposed model TreeNet seems to effectively capture higher-order feature interactions.

### Weaknesses
**(W1).** The literature review appears to be somewhat limited.
Since the authors propose TreeNet, a model that can be interpreted from the perspective of a functional ANOVA model, the literature review should also include relevant studies related to functional ANOVA model.
That is, I believe it would be necessary to discuss recent interpretable models based on the functional ANOVA model, such as NAM ([1]), NBM ([2]), and ANOVA-TPNN ([3]).
In particular, ANOVA-TPNN estimates functional ANOVA components using a tensor product of **sigmoid**-based basis functions, so a comparison with this model would be important.


**(W2).** The baseline models used in the experiments seem to be insufficient, and it would be beneficial to include an additional comparison with functional ANOVA–based models such as NAM ([1]), NBM ([2]), and ANOVA-TPNN ([3]).

**(W3).** The real datasets used in the experiments may not have sufficiently large sample sizes or input feature dimensions.
I think the proposed model requires additional evaluation on larger real-world datasets, such as those used in [2] or [3].

**(W4).** It is difficult to understand how the components of the functional ANOVA model are obtained from a trained TreeNet in Section 2.3. 
A more detailed explanation of this part seems necessary, particularly including Algorithm 1.
For example, in line 183 it says “l bases” what exactly does "l" refer to? Does it represent the dimension of the features after dimension reduction?

**(W5).**
In the paper, it is not clear what aspects of the proposed model are novel compared to existing models.
In my view, the proposed model and methodology seem to lack sufficient novelty to make a substantial contribution.
It would be helpful if the authors could better highlight which aspects of the proposed model and methodology are novel compared to existing approaches.



**References**

[1]. Agarwal, Rishabh, et al. "Neural additive models: Interpretable machine learning with neural nets." Advances in neural information processing systems 34 (2021): 4699-4711.

[2]. Radenovic, Filip, Abhimanyu Dubey, and Dhruv Mahajan. "Neural basis models for interpretability." Advances in Neural Information Processing Systems 35 (2022): 8414-8426.

[3]. Park, Seokhun, et al. "Tensor Product Neural Networks for Functional ANOVA Model." International conference on machine learning  (2025).

### Questions
**(Q1).** 
According to the experimental results in Section A, the run time for model training does not appear to increase significantly with respect to n and p.
However, in the case of Algorithm 2, the run time is expected to increase rapidly as p grows.
Therefore, I think it would be necessary to include additional experiments to verify whether the proposed TreeNet model can still provide interpretable results in a computationally feasible manner when p becomes large.
Specifically, would it be possible to provide experiments that analyze how the run time of Algorithm 2 changes as p increases?


**(Q2).**
If, as stated in **(W4)**, "l bases" indeed refers to the reduced feature dimension, then the components obtained through the purifying algorithm appear to be based on dimension-reduced latent features rather than the original input features.
Could the authors please clarify how TreeNet can still be regarded as an interpretable model under this setting?

**(Q4).**
Is it correct that the number of trees for the XGB model was set to 100 in the experiments? Fixing it at 100 seems rather small. Could you please explain the reason for this choice?

**[Minor]**

**(Q4).**
In Table 6, the input feature dimension of the Calhousing dataset is listed as 9, but isn’t it actually 8?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Cross Spline Net (XSN), a framework that combines models like trees, MARS, and SVM into one neural network structure. By using spline transformations and cross-network layers, XSN aims to achieve flexibility, scalability, and interpretability. It fits well with current research in explainable AI and hybrid modelling.

### Strengths
XSN provides a unified framework that can reproduce or approximate various non-neural network models.


The proposed framework is designed to be more interpretable than traditional black-box models like XGBoost and FCNN.

### Weaknesses
However, XSN also faces limits such as high computational cost, sensitivity to hyper-parameters, and weaker performance on jumpy data. These issues may make it less practical for large or time-sensitive tasks.

* Although XSN improves interpretability, it can still become complex when many polynomial terms are used. The required pruning and purification steps are costly and may not always work well. Interpreting high-order interactions is also difficult, even with the purification algorithm.

* XSN may still overfit on high-dimensional data if regularisation is not carefully applied.

* While XSN uses ADAM for optimisation, the large feature space created by spline transformations can slow down training on very large datasets.

* The model assumes that tabular data has mostly low-order interactions. This may not be true for complex real-world data, which limits its general use.

* XSN introduces many hyperparameters (e.g., number of cross layers, spline bases, and projection size). These require careful tuning, which can be slow and expensive.

* The experiments only compare XSN with XGBoost and FCNN. Including other interpretable models, such as GAMI-Net, would make the evaluation more complete.

* XSN handles jumpy data less effectively than XGBoost, which works better for such patterns. This could restrict its application in scenarios where the underlying data patterns are highly discontinuous or involve sharp changes

* While the results on public datasets are promising, the paper does not clearly show how XSN performs on large, real-world problems where simpler models might still work better.

### Questions
How does the computational cost of the interpretability tools, e.g., pruning and purification algorithms scale with the size and complexity of the dataset?


How does the interpretability of XSN compare to other interpretable models like GAMI-Net or SHAP-based methods in terms of ease of use and insights provided?


Are there specific types of data or problems where XSN is not suitable?

### Soundness
3

### Presentation
3

### Contribution
1
