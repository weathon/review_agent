# Start Smart: Leveraging Gradients For Enhancing Mask-based XAI Methods

- Avg Score: 5.75
- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
Mask-based explanation methods offer a powerful framework for interpreting deep learning model predictions across diverse data modalities, such as images and time series, in which the central idea is to identify an instance-dependent mask that minimizes the performance drop from the resulting masked input. Different objectives for learning such masks have been proposed, all of which, in our view, can be unified under an information-theoretic framework that balances performance degradation of the masked input with the complexity of the resulting masked representation. Typically, these methods initialize the masks either uniformly or as all-ones.
In this paper, we argue that an effective mask initialization strategy is as important as the development of novel learning objectives, particularly in light of the significant computational costs associated with existing mask-based explanation methods. To this end, we introduce a new gradient-based initialization technique called StartGrad, which is the first initialization method specifically designed for mask-based post-hoc explainability methods. Compared to commonly used strategies, StartGrad is provably superior at initialization in striking the aforementioned trade-off. Despite its simplicity, our experiments demonstrate that StartGrad enhances the optimization process of various state-of-the-art mask-explanation methods by reaching target metrics faster and, in some cases, boosting their overall performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce StartGrad, a efficient novel-gradient based mask initialize method designed for mask-based post-hoc attribution method, such as ShearletX or WaveletX. StartGrad utilizes gradients of the input features to initialize the masks more effectively, resulting fast convergence. The authors also theoretically proved that StartGrad is superior to commonly used initialized strategies like uniform or all-ones initialization. 

The authors conduct extensive experiments across both vision and time-series domains. The results showed that StartGrad consistantly accelerates the optimization process.

### Strengths
This paper is well written; especially when the authors introduce rate-distortion explanation(RDE) framework. The methodology, theoretical background is well organized and clearly explained.

Good novelty. The authors introduced a new perspective by focusing on mask initialization that has been overlooked. This approach seems to be radical rather than incremental.

The authors provide solid theoretical proofs under the RDE framework, showing that StartGrad offers a better trade-off between distortion and sparsity compared to traditional approaches.

The authors conducted extensive experiments across different domains(vision and time-series) with state-of-art RDE explanation methods, and demonstrated the effectiveness and practicality of StartGrad.

Extensive ablation studies. The authors provide extensive ablation studies, analyzing effect of various components such as quantile transformation.

### Weaknesses
Lack of Visualization: It would be beneficial to include a figure explaining the StartGrad method. Additionally, providing qualitative results for StartGrad would enhance the paper—for example, illustrating the differences in attribution maps after 50 iterations with different initializations of ExtremalMask.

### Questions
Can the RDE framework's attribution maps handle multi-label objects in a single image, such as an image containing both a cat and a dog? (This question is not related to the paper's rating; it's just out of the reviewer's curiosity.)

To quantify attribution methods on vision dataset, the fidelity metrics Most Relevant First (MoRF) deletion and Least Relevant First (LeRF) deletion are widely used. If these fidelity metrics were employed, comparison with other state-of-the-art attribution methods, such as the Layer-wise Relevance Propagation (LRP) family and the Class Activation Map (CAM) family, would have been possible. Why did the authors choose the conciseness-preciseness (CP) Pixel and L1 scores rather than MoRF and LeRF?

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
3

### Summary
The paper introduces a post-hoc explainability technique within the domain of XAI, focusing on a mask-based approach to identify which parts of an input are most crucial for generating an explanation. Existing approaches aim to optimize the mask such that the masked input achieves similar predictive performance as the original. However, the authors address an overlooked aspect: the initialization of these masks. Their method operates on the premise that standard initialization techniques are suboptimal. They propose and mathematically demonstrate that initializing the mask based on the most salient gradients is more effective. By using a quantile transformation function to identify the top gradients, they initialize the mask in a more targeted manner. This technique, named StartGrad, aims to improve the performance of mask-based methods from the start.

### Strengths
The writing is clear and easy to read.
The paper provides mathematical proofs to support most of its claims.
The topic is engaging and relevant.

### Weaknesses
**1. Typographical Error**

It appears that lines 217–218 contradict Equation 7. This discrepancy could lead to confusion and should be addressed for clarity.

**2. Overclaims**

The paper seems to overstate certain findings. For instance, lines 175–176 present an equation without constraints on the mask, and much of the analysis substitutes entropy with a norm p, which limits generalizability. Propositions 2 and 3 focus solely on initialization rather than the optimization process as a whole. This raises a question: can we be confident that StartGrad improves optimization across all optimizers, or might this only hold in specific cases?

**3. Experimental Section**

Results and Scope: Table 1 does not appear to provide results for StartGrad, which limits the insight we can gain into the method’s effectiveness. Additionally, the experimental section is underdeveloped. The current setup only uses ResNet-18 on a subset of 500 random ImageNet images, which restricts the findings’ generalizability in the vision domain. Including additional datasets, such as Pascal-Part [2] and Monumai[1], where ground truth is available, would strengthen the empirical validation. It would also be interesting to test the techniques on more DNN such as Swin Transformer and ViT.
*Convergence Speed:* It would be valuable to evaluate whether the proposed initialization method enhances the convergence speed of XAI algorithms.
*Comparative Analysis:* Comparing StartGrad to other post-hoc explainability [3,4,5,6,7,8] techniques would provide more context and relevance to the findings.

**4. Metrics Used**

The choice of metrics is unclear. A more detailed explanation of the metrics used in the evaluation would be beneficial. Additionally, considering established XAI metrics from frameworks like Quantus or Xplique could improve comparability and transparency.

[1] Lamas, Alberto, et al. "MonuMAI: Dataset, deep learning pipeline and citizen science based app for monumental heritage taxonomy and classification." Neurocomputing 420 (2021): 266-280.

[2] Chen, Xianjie, et al. "Detect what you can: Detecting and representing objects using holistic models and body parts." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.

[3] Selvaraju, R. R., Das, A., Vedantam, R., Cogswell, M., Parikh, D., & Batra, D. (2016). Grad-CAM: Why did you say that?. arXiv preprint arXiv:1611.07450.

Jamil, Md Shafayat, et al. "Advanced gradcam++: improved visual explanations of CNN decisions in diabetic retinopathy." Computer Vision and Image Analysis for Industry 4.0. Chapman and Hall/CRC, 2023. 64-75.

[4] Wang, Haofan, et al. "Score-CAM: Score-weighted visual explanations for convolutional neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. 2020.

[5] Muhammad, Mohammed Bany, and Mohammed Yeasin. "Eigen-cam: Class activation map using principal components." 2020 international joint conference on neural networks (IJCNN). IEEE, 2020.

[6] Srinivas, Suraj, and François Fleuret. "Full-gradient representation for neural network visualization." Advances in neural information processing systems 32 (2019).

[7] Sattarzadeh, Sam, et al. "Integrated grad-cam: Sensitivity-aware visual explanation of deep convolutional networks via integrated gradient-based scoring." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.

[8] Kapishnikov, Andrei, et al. "Guided integrated gradients: An adaptive path method for removing noise." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

### Questions
See most of the questions on the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduced a new gradient-based mask initialisation technique for mask-based explanation methods called StartGrad. This initialisation method trades-off mask performance against complexity of the resulting masked representation. The authors have completed a number of experiments that demonstrate StartGrad's ability to improve the performance of mask-based methods based on speeding up their optimisation process.

### Strengths
1. The theory and proofs in this paper are more than adequate, demonstrating the feasibility of balancing between complexity and information bottlenecks through information theory.

2. The paper has sufficient context, motivation. The authors have a solid understanding of the trade-off in mask-based XAI.

### Weaknesses
1.  The **experiments** in this paper are insufficient. the authors have only done experiments with $\lambda$ = 1,2 and 1,10. Since this hyperparameter directly affects the balance of trade-off, it is better to provide sufficient ablation study on lambda. 
   It would be useful for the authors to mention the impact of different orders of magnitude of $\lambda$ on network performance. For example, a graph with $log_\lambda$ on the x-axis and performance on the y-axis could be completed to evaluate the impact of lambda.

2. The contribution of this paper does not seem obvious enough. In some metrics, the method proposed by the authors decreases in comparison to baseline in the early iterations, while it is essentially flat in the later period (as shown in Table 3). It is quite doubtful that StartGrad is able to accelerate with improving performance as the authors claimed.
    For example,  in table 3,  StartGrad has less AUP than all ones strategy at 50 iteration steps and less AUR at 100 iteration steps. The 50 steps AUR has even better performence than 100 steps.

### Questions
This initialisation method seems to be sensitive to $\lambda$. Does the choice of $\lambda$ greatly affect the performance and speed of StartGrad? And how do the choice of $\lambda_1$ and $\lambda_2$ mentioned in the appendix affect those performances?
    The authors are advised to add sensitivity analysis on these hyperparameters. In addition, for $\lambda_1$ and $\lambda_2$, it is better for the experiment to show the impact of them separately. For example, analysis a trade-off or a similar effect on performance between them.

### Soundness
2

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
2

### Summary
This paper introduces StartGrad, a novel gradient-based initialization technique specifically designed for mask-based explainability methods in deep learning. While recent research has focused on developing new objective functions for mask-based explanations, the authors identify that initialization strategies have been overlooked despite their importance for optimization performance. The key contributions are (1) StartGrad: A new initialization algorithm that leverages gradient information to provide better starting points for mask optimization, transforming gradient values into initial masks using quantile transformation. (2) Theoretical Framework: A formal analysis showing that StartGrad is provably superior at initialization compared to standard strategies in balancing the fundamental tradeoff between distortion and sparsity. The authors unify existing mask-based methods under an information-theoretic framework and show how StartGrad can enhance their performance while maintaining simplicity of implementation.

### Strengths
1. The paper identifies and addresses an overlooked aspect of mask-based XAI methods - initialization strategy. While much work has focused on objective functions, this is the first paper to systematically study initialization.

2. The information-theoretic unification of existing mask-based methods provides a novel theoretical framework for analyzing these approaches.

3. The empirical evaluation is comprehensive (1) Covers different domains (vision, time-series), (2) applicable to multiple state-of-the-art methods (PixelMask, WaveletX, ShearletX, ExtremalMask), (3) The ablation studies thoroughly examine robustness to noisy gradients and alternative implementation choices.

4. The paper is well-structured and clearly written, with a logical flow from motivation to theory to experiments.

### Weaknesses
1. Reliance on Simplified Assumptions in Theoretical Analysis: The theoretical foundations of StartGrad depend on assumptions such as local linearity and neighborhood smoothness of the classifier's prediction function. While these assumptions are necessary for proving the benefits of StartGrad, they may not hold in highly non-linear or real-world scenarios, especially in models with complex architectures.

2. Domain Generalizability: The experiments are mainly centered on vision and time-series data, leaving an open question about how well StartGrad performs on other data modalities such as graphs or text. 

3. Narrow Experimental Scope in Terms of Model Variants: The study mainly evaluates StartGrad on specific XAI models like PixelMask, WaveletX, and ShearletX, using ResNet18 and VGG16 classifiers. The conclusions would be more compelling if tested across a broader set of models, such as transformers or graph neural networks

4. No analysis of explanation stability across different random seeds.

5.  Missing comparison with recent developments in efficient xai methods

### Questions
1. The effectiveness of StartGrad is heavily dependent on the accuracy of gradient signals. Could you clarify if there are specific scenarios (e.g., highly non-linear models or adversarial settings) where gradient inaccuracies significantly degrade performance? (e.g.  understanding if and how StartGrad could be improved or adapted in scenarios where gradients are less reliable could address concerns about robustness)

### Soundness
3

### Presentation
3

### Contribution
3
