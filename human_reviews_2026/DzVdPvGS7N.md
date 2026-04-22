# Sparse Deep Additive Model with Interactions: Enhancing Interpretability and Predictability

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 8, 4

## Abstract
Recent advances in deep learning highlight the need for personalized models that can learn from small or moderate samples, handle high-dimensional features, and remain interpretable. To address this challenge, we propose the Sparse Deep Additive Model with Interactions (SDAMI), a framework that combines sparsity-driven feature selection with deep subnetworks for flexible function approximation. Unlike conventional deep learning models, which often function as black boxes, SDAMI explicitly disentangles main effects and interaction effects to enhance interpretability. At the same time, its deep additive structure achieves higher predictive accuracy than classical additive models. Central to SDAMI is the concept of an Effect Footprint, which assumes that higher-order interactions project marginally onto main effects. Leveraging this principle, SDAMI employs a three-stage strategy to circumvent the  search complexity inherent in direct interaction screening: first, identify strong main effects that implicitly carry information about important interactions; second, exploit this information—through structured regularization such as group lasso—to distinguish genuine main effects from interaction effects; third, build subnetwork for identified main effect and interaction. For each selected main effect, SDAMI constructs a dedicated subnetwork, enabling nonlinear function approximation while preserving interpretability and providing a structured foundation for modeling interactions. Extensive simulations and applications with comparisons confirm SDAMI’s in reliability analysis, neuroscience, and medical diagnostics further demonstrate SDAMI's versatility in recovering effect structures across diverse scenarios and addressing real-world high-dimensional modeling challenges.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Sparse Deep Additive Model with Interactions (SDAMI), a framework that separates main effects (a sum of univariate functions) from interaction effects for greater interpretability. The paper focuses on the challenging case of small n and large p, where the risk of overfitting is significant for conventional deep learning models.  The authors introduce the concept of Effect Footprint: interaction effects leave a signal that can be detected via screening. They propose a mathematical formulation of the problem and a three-step approach to solving it:
- feature selection via a sparse additive screening procedure 
- group lasso to distinguish between main and interaction effects
- training deep learning models to model main and interaction effects.

The experimental results are on synthetic and real-world datasets. They also provide theoretical results:
- Conditions under which the effect footprint disappears, and screening may fail
- Asymptotic convergence results for SDAMI (feature selection and convergence of the probability estimator)

### Strengths
- The paper proposes the new concept of effect footprint and provides theoretical results to justify the use of screening and illustrate its limitations.
- Asymptotic convergence results provide a theoretical justification for the framework.
- The paper shows how SDAMI can be used to extract interpretable signals and understand the contribution of main and interaction effects.

### Weaknesses
1) The writing of the paper could be improved:
- The contribution section is somewhat repetitive (the list is a summary of the text just above).
- Line 157: Is “additive” the right word? I think “main effect” is more appropriate.
- Line 214: the word “partitioning” seems to indicate that there cannot be the same feature in the main and interaction effects.
- In the contribution, you mention that the simulations focus on p>>n. However, line 310 indicates that n is always greater than p (n >= p = 150).

2) As you mentioned in the limitations, the theoretical results are asymptotic. However, your motivation, small n and large p, corresponds to a very different setting.

3) The synthetic simulations correspond to very simple and somewhat too perfect cases (very similar to your formulation of the problem).

4) Experiments on real-world datasets are interesting. However, the number of datasets and baselines is very limited. You should include results on other Generalized Additive Models baselines such as NAM (Neural Additive Models) [1], GAMI-Net [2], and NODE-GAM [3]. 

5) Your formulation of the problem (equation 2) attempts to find a sum of main effects and an interaction model that takes all “interaction features” as input. The formulation is unclear because the model itself becomes a black box model (because of the interaction “model”). Are you then focusing on second-order interactions only?

6) Your framework corresponds to a case of Generalized Additive Models with interactions. You should indicate this in the paper and mention more literature on the subject.

[1]: Rishabh Agarwal, Levi Melnick, Nicholas Frosst, Xuezhou Zhang, Ben Lengerich, Rich Caruana, and
Geoffrey Hinton. Neural additive models: Interpretable machine learning with neural nets. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing
Systems, 2021.

[2]: Zebin Yang, Aijun Zhang, and A. Sudjianto. Gami-net: An explainable neural network based on generalized
additive models with structured interactions. Pattern Recognit., 120:108192, 2021

[3]: Chun-Hao Chang, Rich Caruana, and Anna Goldenberg. NODE-GAM: Neural generalized additive model
for interpretable deep learning. In International Conference on Learning Representations, 2022.

### Questions
- Could you test your method on other baselines (NAM, GAMI-Net, NODE-GAM) and other datasets? The datasets from these papers would be interesting to study.
- How much computing time does your method require compared to the other baselines?
- Are you focusing solely on second-order interaction effects? It would be good to clarify this further in the paper. 
- Do you think you can derive non-asymptotic guarantees for your framework? This would be more interesting for the case you are interested in (small n, large p).

### Soundness
3

### Presentation
2

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
The proposed work attempts to introduces a new neural additive model through equations (1) and (2), but after later implicit assumptions, reduces to training a neural network with feature selection.  The work then introduces a novel two-stage learning structure based on the idea of first learning the one-dimensional main effects before selecting the higher-order interactions using heredity.  The work then applies another training procedure to the specific neural network and applies to synthetic and real-world datasets. Recovery of additive models is demonstrated on several synthetic datasets and competitive performance on three real-world datasets is shown compared to LASSO, DNN, and fSpAM.

### Strengths
- The structural equation in Equation 1 has the potential to be novel if the authors do not implicitly assume that M⊂I
- The learning procedure has the potential to encourage soft pruning of irrelevant features

### Weaknesses
- Glosses over closely related NAM work as irrelevant due to lack of interaction modeling [1] and then ignores other NAM works which include interactions [2,3,4,5].
- The effect footprint does not seem to be novel and seems to be the same as existing works applying heredity.
- There are very few baselines compared against and relatively few datasets are used for comparison
- The impact of the important hyperparameters introduced by the paper are not explored 
- Figure 1 and Figure 4 are not given much explanation in the main paper


[1] Shiyun Xu, Zhiqi Bu, Pratik Chaudhari, and Ian J Barnett. Sparse neural additive model: Interpretable deep learning with feature selection via group sparsity
[2] Zebin Yang, Aijun Zhang, Agus Sudjianto.  GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions
[3] Chun-Hao Chang, Rich Caruana, Anna Goldenberg.  NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning
[4] James Enouen and Yan Liu. Sparse interaction additive networks via feature interaction detection and sparse selection.
[5] Minkyu Kim, Hyun-Soo Choi, and Jinho Kim. Higher-order Neural Additive Models: An Interpretable Machine Learning Model with Feature Interactions

### Questions
- What are the hyperparameter choices used for neural network training and other learning algorithms?
- What is the novel aspect of the effect footprint compared with existing works using heredity?
- Why is the implicit assumption M⊂I used?  What prevents this from reducing to the feature selection case?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces a new method for creating interpretable deep learning models when the number of training examples is much smaller than the number of dimensions.  The key idea is to characterize marginal effects of interactions and use those effects to guide the construction of a network with dedicated components for main effects and variable effects.  The paper provides theoretical support for the method, as well as extensive simulations to illustrate its utility in practice relative to baseline methods.

### Strengths
A key concept proposed here is the notion of the "Effect Footprint," in which "higher-order interactions leave detectable marginal signatures on main effects."  This concept seems like a useful abstraction.

I like how the related work section clearly lays out the two different categories of methods for addressing this problem, and then goes on to explain how the proposed method brings these two categories together.

The empirical analyes using simulations and real data are impressive, showing that SDAMI outperforms baseline methods both in terms of predictive accuracy and in accurately recovering the true underlying structure of the data.

### Weaknesses
I think the tension described in line 44 should be restated to include the obvious candidate class of models here, which are not "classical deep models" but classical linear models.  It's not obvious why we should bother with deep models when n << p.  These models are alluded to in line 77, though of course there are classical sparse models that do allow for higher-order associations.  These should be discussed.

Along those lines, the introduction doesn't do a great job of explaining how pre-training can address the problems outlined here; i.e., when n << p, but when you have some other collection of (potentially unlabeled) n' examples, where n' > p.

Figure 2 is confusing.  Based on the description in lines 197-209, it seems like there are two sets of inputs: main effect variables of size p and footprint variables of size q.  I think the input layer should have 1..p going into main effect blocks, and then 1...p+q going into the interaction block.

Line 258: It seems like you also have to address the other failure mode, when a variable leaves a misleading (i.e., false positive) footprint trace.

I thought Figure 5 was pretty difficult to interpret.  We are basically asked to take the authors word for it that these panels "reveal synergistic patterns consistent with cortical pooling."

Minor points:

line 120: Missing cite for SpAM.

Line 215: Give a cite for Mallow's C_p.

Line 432: diabete -> diabetes

### Questions
What would Figure 5 look like if the method did not work well?

Why does Figure 2 show q variables going into the main effect subnetworks?  Where are the footprint variables in this figure?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study proposes a deep learning model that is effective for datasets with a small number of samples and a large number of features. While conventional methods often suffer from overfitting and difficulties in feature selection, the proposed approach enhances model explainability and interpretability by exploiting feature sparsity and identifying interaction features. In particular, a novel measure called "Effect Footprint" is introduced to extract features that have no main effects but play an important role through interaction effects. This measure provides a framework for identifying indirect dependencies among features that were previously difficult to detect. Furthermore, the theoretical properties of the proposed model are analyzed, and its effectiveness is demonstrated through numerical experiments using both synthetic and real-world datasets.

### Strengths
This study addresses the important challenge of improving the explainability and interpretability of deep learning models. In particular, it introduces a unique framework that sparsely connects multiple subnetworks, enabling clear identification of feature contributions even in high-dimensional datasets.

The study proposes a novel measure called "Effect Footprint" to detect features that have no main effects but play important roles through interactions. This theoretical framework seems highly original, as it enables the identification of indirect feature dependencies that were difficult to capture with conventional methods.

The effectiveness of the proposed method is demonstrated not only through numerical experiments using synthetic and real-world datasets but also through theoretical analysis. The effort to validate the method from both theoretical and experimental perspectives enhances the credibility of this research.

### Weaknesses
The overall description in the paper is somewhat ambiguous, and the mathematical and algorithmic details of the proposed method are not sufficiently presented. In particular, the selection process of the feature set with interactions, $\mathcal I$, as well as the specific computation and utilization of the "Effect Footprint", are unclear, raising concerns about the reproducibility and comprehensibility of the method.

While the paper claims that identifying interaction features enhances the interpretability of the model, there appears to be no explicit mechanism for extracting such features from the actual model structure. I feel that there is a noticeable gap between the claims made in the Abstract and Introduction and what is actually demonstrated in the main text.

The necessity of using deep learning for small-sample, high-dimensional problems is not well justified. Numerous prior studies, particularly those based on linear models and traditional statistical approaches, have addressed interaction effects (e.g., the works listed below). The lack of discussion or comparison with these studies weakens the overall persuasiveness of the paper:

  Choi et al. Variable selection with the strong heredity constraint and its oracle property. JASA, vol.105, no.489, 2010.

  Bien et al. A lasso for hierarchical interactions. Annals of Statistics, vol.41, no.3, 2013.

  Nakagawa et al. Safe pattern pruning: An efficient approach for predictive pattern mining. KDD 2016.

### Questions
The paper's main contribution lies in the introduction of the Effect Footprint; however, it is unclear how this measure is specifically computed and utilized within the model. According to the Algorithm in the Appendix, it appears that SpAM is simply applied, but it is not well explained why this procedure enables the identification of Interaction-Only features. A more concrete theoretical and intuitive explanation would be helpful.

The proposed method constructs a deep learning model that handles interaction terms, but it remains unclear by what criteria or procedure the important interaction terms are finally extracted. The authors are requested to clarify how the network structure or learned weights are related to the selection of significant features.

The paper also presents a theoretical discussion on feature selection consistency, but it is questionable whether this property is independent of the learning performance of the deep neural network. It seems that the argument implicitly assumes that the model is correctly trained; therefore, such assumptions and dependencies should be explicitly described.

### Soundness
2

### Presentation
2

### Contribution
3
