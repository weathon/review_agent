# Your Consistency Model is Secretly a More Powerful Supervised Learning Paradigm for Learning Tasks with Complex Labels

- Decision: Reject
- Scores: 5, 5, 5, 8

## Abstract
Directly predicting labels from data inputs has been a long-standing supervised learning paradigm. Its trade-off between compression and prediction is studied under the information theory framework e.g. Information Bottleneck, especially in the context of deep learning. It typically assumes that the information content of labels is significantly less than that of data inputs, leading to model designs that prioritize compressing and extracting features from data inputs. In fact, recent supervised learning increasingly faces predicting complex labels, exacerbating the challenge of learning mappings from compressed latent features to high-fidelity label representations. Predictive bottlenecks emerge not only from compression limitations but also from the inherent complexity of feature-to-label transformations. This paper proposes incorporating scheduled label information into the model during training to better learn the prediction consistency mapping, which stems from the consistency mapping concept from generative consistency models. Unlike traditional approaches predicting labels directly from inputs, in this paper, the training of our designed conditional consistency involves predicting labels using inputs and noise-perturbed label hints, pursuing the predictive consistency across different noise steps. It simultaneously learns the relationship between latent features and a spectrum of label information from zero to complete, which enables progressive learning for complex predictions and allows multi-step inference analogous to gradual denoising, thereby enhancing the prediction quality. Experiments on vision, text, and graph tasks show the superiority of our consistency supervised training paradigm, over conventional supervised training in complex label prediction problems. Source code will be made publicly available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces a new supervised learning approach called Supervised Consistency Learning (SCL), which uses noise-perturbed label hints to improve predictions for complex labels. Unlike traditional methods, SCL progressively refines predictions through multi-step inference, enhancing performance in vision, text, and graph tasks by focusing on label complexity.

### Strengths
- This paper introduces a simple yet effective approach, showing strong performance in tasks like segmentation. 
- The idea of adding noise to labels to learn consistency is a unique concept, enabling models to refine predictions progressively. 
- The approach is easy to implement and can be broadly applicable across tasks involving complex labels, adding value to supervised learning methods.

### Weaknesses
1. The term “SCL” has already been used for supervised contrastive learning, so using a different abbreviation is recommended.

2.  While this method shows good performance on the segmentation task, it would be beneficial to explore its applicability on large-scale classification datasets with many classes, such as iNaturalist and Open Images.

3. How much longer does training take compared to standard supervised learning, in terms of both iterations and wall clock time? 

4. A key limitation of this method is the increased inference time, particularly in segmentation tasks. Details on the choice of $𝑁_\tau$ during inference, time taken per inference, and wall-clock time compared to a single standard inference on a supervised model would be helpful.

5. Most results in the experimental section were shown using single-step predictions, with limited results provided for multistep predictions. Specifically, Figure 3 shows qualitative results where multistep predictions perform better than single-step, but in Table 1, only results for single-step predictions are provided, and multistep results are absent. Presenting multistep results selectively raises doubts about the performance of multistep predictions.

6. The baseline models used in the experiments are quite outdated (segmentation models from 2017-2019, N-body simulation models from 2015-2017), so it is necessary to verify whether performance improvements still occur when applied to the latest methods.

7. There is no ablation study on the proposed method in the paper. Specifically, Equation 5 uses triadic distance as a loss, but an ablation study is needed to verify whether triadic distance is an essential component. Additionally, an ablation study on the step size in the training process seems also necessary, as it needs to be demonstrated that a large step size is essential, given that only single-step or few-step predictions are actually used during inference.

8. As it takes multiple forward passes for an inference, how does performance compare to ensemble methods, that require a similar number of inferences such as Bayesian model averaging, stochastic weight averaging, or SWAG?

9. Even though the method was proposed for supervised learning with a large number of classes, I wonder what would be the performances for tasks with a relatively small number of classes, e.g. classical classification problems such as imagenet or cifar-100.  

10. Definition 2.1 lacks the formal structure expected of a definition, so rephrasing it to clarify the characteristics of "complex labels" would enhance the readability.

11. In the introductory part and motivation, the authors mention information bottleneck (IB), but the derivation of the algorithm is fairly relevant to IB. More discussion or theoretical relationship between the proposed algorithm and IB would be beneficial. 

If the rebuttal is satisfactory, I am willing to raise the score.

### Questions
See weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper leverages the consistency model framework to solve a variety of problems across different domains: segmentation in the image domain, the N-body problem and combinatorial optimization in the graph domain, and next-token prediction in the language modeling domain. The authors demonstrate significant performance improvements over various baselines, emphasizing the benefits of adopting the consistency model framework over traditional supervised learning approaches. The proposed Supervised Consistency Learning (SCL) framework is a novel training paradigm that introduces a progressive label reconstruction approach. Instead of directly predicting labels from input data, SCL uses a combination of single data input and a time step t to produce a noisy target, and aims to reduce the distance on the output target to ensure consistency. This approach enhances the model's ability to learn complex labels, providing a structured learning process that facilitates multi-step inference akin to gradual denoising, thereby improving prediction quality in tasks with complex labels.

### Strengths
1. Novelty: This paper introduces a unique and previously unexplored model. The approach of producing different outputs for the same input x based on the time step t is quite novel. While it has some similarities to Bayesian concepts, it is particularly interesting as it allows for a kind of single-model ensemble, which is an uncommon characteristic.

2. Extensive Experiments: The experiments conducted in this paper are diverse and thorough, demonstrating a high level of completeness and providing convincing support for the proposed approach.

### Weaknesses
1. Inconsistencies with Consistency Model Definition: The paper Claims that SCL framework is a nevel training paradigm which embodies concistency model framework, however, I don't think whether SCL framework is Consistency model. Specifically, the model does not receive noisy inputs, which contradicts the typical definition of consistency models, where the output should adapt to noisy changes in the input. Rather, the model produces **different outputs based solely on varying time steps t, similar to using t as a prior in a Bayesian model**. The framework is much close to Baysian model. The SCL framework gets image and time t, which is similar to get image and prior importance. If the prior is important, then the model should sample the output in much uncertain distribution. In this viewpoint, the sampling algorithm is just baysian inferecne and it can explain, with marginal performance improvements.

2. Inefficiency in Multistep Prediction: The proposed multistep prediction approach raises questions about its efficiency. In the described algorithm, it appears that inference is performed for all time steps t and then averaged, as illustrated in Figure 3, where time step values range from 1000 to 0. This implies nearly 1000 inferences, which seems naively similar to multiple rounds of training and ensembling. Moreover, the performance gains compared to ensemble approaches appear marginal in most cases.

3. Low Reported Performance: Upon examining Table 1, the reported results seem unusually low compared to well-known baselines reported in other prominent papers. For example, MobileNetV2 + SCL shows a significant improvement over the baseline MobileNetV2, but the reported baseline mIoU value of 17.84 seems unusually low, raising concerns about its validity, especially considering that even the 2015 FCN architecture reported an mIoU of 30. This suggests that the baseline performance reported for MobileNetV2dilated is abnormally low. Similar discrepancies are observed with other baselines compared to values reported on platforms like Papers with Code.

### Questions
1. In table 3 GCN section, the inference time is reduced 25s to 24s. How can this happen? as shown in Algorithm 2, it uses multistep prediction. Meaning it inferences N times more than baseline.

2. Algorithm 1 does not fit to the paper's claim, as it wants to minimise the distance between noisy target and hard target. Why does the loss want to make every $y^t$ certain? In consistency model framework, it should outputs varying confidences with time$t$

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a learning and inference paradigm aimed at both classification and regressions tasks.
The method uses a time-conditional network like classically used in diffusion and consistency models.
The network takes two inputs, the input to classify and the noisified labels, and outputs the noise free labels.
So far this is a classical setup, already used in super-resolution literature, etc...
The claimed novelty lies in the loss which (1) maps the noisified labels to the noise-free labels and (2) that two independently noisified labels for the same input should map to the same labels. This departs from time-consistency proposed in consistency models. Instead, here the consistency applies to independently noisified labels.

### Strengths
- The main strength of the paper is a marginal improvement on several benchmarks.
- The algorithms help understand the paper

### Weaknesses
- The loss does not seem well grounded
  - The $\lambda_2$-term  $\lambda_2 d(f(x,y_t,t),f(x,y_{t'},t'))$ seems redundant with the $\lambda_1$-term $\lambda_1 (d(f(x,y_t,t),y) + d(f(x,y_{t'},t'),y))$ since the $\lambda_1$-term is already mapping $f(x,.,.)$ to $y$.
  - This loss also is disconnected from the standard consistency models definition since it has not self-teaching aspect (not recurrence), which is at odds with the title "Your Consistency Model is Secretly...". What is proposed here does not look like what is classically defined as a consistency model to me. To be more explicit the $\lambda_2$-term is not recurrent since it refers to two independently sampled noise (also called independent trajectories in the paper), and unlike in consistency models there's no 'stop_gradient' indicating that one side is acting as a teacher for the other.
  - I suspect the whole loss could be simplified to $d(f(x,y_t,t),y)$, which basically is, in expectation, the case when $\lambda_2=0$.
  
- Lack of ablation studies
  - Authors claim that more denoising steps lead to better results but I didn't see any ablation study confirming that claim. 
  - The $\lambda_1$ and $\lambda_2$ terms seem important in the loss and yet no in-depth numerical analysis of various combinations is shown, this would help me gain more insight into the necessity of the two terms for the proposed loss.

- Improvements are marginal overall: typical claimed gains are of 1 point (on various metrics and various domains) with one exception for MobileNetV2 which was originally performing a lot worse than other networks in the first place.

- The experiments cover 4 different domains leaving me with a feeling of breadth at the expense of depth. 

- The writing is hard to follow in general and there's a fair amount of repetition. In particular the method could be better presented although the inclusion of the algorithm helped a lot to comprehend what is actually being proposed.

### Questions
1. In the continuous label-setting, how does your method compare to diffusion-based regression like typically used for super-resolution (there are plenty of them that been published: ResShift, SR3, just to name a few)?
2. Quoting from the paper "... and the model trades the output diversity to better capture y. This calls for the requirement of consistency
extending across all trajectories ...". Why? What happens when you stick to the classical time-consistency definition from consistency models? I would be interested in seeing this claim actually substantiated by numerical results.
3. This is somewhat redundant with my points on weaknesses, namely the lack of ablations, but what happens when setting $\lambda_2=0$? I would really love to see an ablation on this.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explores an alternative method of doing supervised learning. The motivation is that, even though supervised learning has been largely successful in machine learning, when the label distribution is very complex (e.g. graphs and pixels),  learning the posterior distribution can be challenging. In this work, the author proposed learning a consistency model for the label distribution given the data distribution and noisy label samples. Through extensive experiments, including semantic segmentation, n-body simulation, combinatorial optimization and next-token prediction, the paper shows that their method consistently performs better without a large sacrifice in inference time compared to their discriminative modeling counterparts.

### Strengths
This work has a fair amount of novelty, in the sense that consistency is still a relatively new generative model, and there isn’t much work on modeling the target distribution. As for the writing, the proposed method is fairly straight-forward, as it is applying consistency model to the label distribution conditioned on the data distribution, rather than the typical setting of applying it to the data distribution itself. The introduction of consistency model, multi-step inference, and forward steps in categorical and continuous distributions are clearly defined, making the paper self-contained and providing sufficient information for the reader to follow through.  For experiments, the authors have done a fair job in testing their method in multiple settings, such as semantic segmentation, n-body simulation, combinatorial optimization and next-token prediction. Each setting is also provided a fair number of baselines and comparisons to clearly demonstrate that their method can perform better than current state-of-the-art. Overall, the method is sound and works reasonable well within expectations.

### Weaknesses
1.	One shortcoming is that the paper assumed the data provides sufficient information about the labels, but this often happens when there is an abundance of data. It is an assumption that is implicitly made by the method, but was not mentioned at all.
2.	The writing of the paper begins and motivates from the perspective of Information Bottleneck. However, from the writing, it’s not exactly clear how the high level intuition connects with the assumptions about the model. Intuitively, I believe the authors are trying to argue that consistency models also make the assumption that the distribution it’s trying to model (i.e. the target distribution here) is also trying to learn a low-dimensional representation, for instance. However, this is not explicitly said or mentioned. In other words, the discussion on IB seems to distract from the discussion rather than helping the discussion. The introduction would have read fine without any mentioning of IB.  Some additional comments regarding the connection in the related work section would make the motivation of the work more clear and prominent. 
3.	While one could argue that consistency models are a recent rising trend, the method is somewhat simply applying consistency models in place of other generative models for learning target distributions. In this sense, there is limited novelty. It can help if the authors can comment more on how consistency model achieves something that previous generative models cannot. This weakness is also listed as a question for the authors below also.

### Questions
1. Can the authors provide a justification for why this cannot be done with other generative models? In other words, what does consistency model offer as an advantage that other generative models (such as diffusion models or VAEs) cannot do? 
2. Can the authors provide some comments on how the method would perform in the case where there is class imbalance, or an extremely large number of classes (e.g. extreme classification settings)? Similarly, can the authors provide some comments also in the case there is domain shifts, or out-of-domain issues?

### Soundness
3

### Presentation
3

### Contribution
2
