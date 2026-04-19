# NAP2: Neural Networks Hyperparameter Optimization Using Weights and Gradients Analysis

- Decision: Reject
- Scores: 5, 3, 5, 3

## Abstract
Recent hyper-parameter tuning methods for deep neural networks (DNNs) generally rely on first using low-fidelity methods to identify promising configurations and then using high-fidelity methods for further evaluation. While effective, existing solutions treat DNNs as `black boxes', which limits their predictive abilities. In this work, we propose Neural Architectures Performance Prediction (NAP2), a `white box' hyperparameter optimization approach. NAP2 models the changes in the weights and gradients of the analyzed networks over time and can predict their final performance with high accuracy, even after a short training period. Our evaluation shows that NAP2 outperforms the current state-of-the-art both in its ability to identify top-performing architectures and in the amount of resources it utilizes. Moreover, we show that our approach is transferable, meaning it is possible to train NAP2 on one dataset and apply it to another.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents an approach to do hyperparameter optimization by building a predictive model based on the evolution of weights as features within the neural network. For each architecture decisions on which architecture is useful are based on the evolution of weights or features extracted from weights during the training process. Decision making based on these features is used using typical MAB methods such as HyperBand. Validation on this proposed approach is provided on CIFAR-10 and CIFAR-100 dataset.

### Strengths
The paper combines many different techniques in order to deliver empirical results.

The paper uses a somewhat new approach of inspecting the weights in order to make decisions on whether the architecture is useful or not.

The paper performs validation on CIFAR-10 and CIFAR-100 showing that their approach works in this setting.

### Weaknesses
The paper is quite incremental.

The paper does not cover many related works in this area which have used similar techniques. See [1], [2], [3] among others.

Even if the above related work did not exist, it is not clear to the reviewer whether the contribution of the paper is sufficiently novel or useful to give a score of accept.

The paper's validation on CIFAR-10 and CIFAR-100 may not be sufficient evidence for it to be shown to be useful in practice on larger neural networks than toy networks.

[1] Freeze-thaw Bayesian optimization. Kevin Swersky, Jasper Snoek, Ryan Prescott Adams. https://arxiv.org/abs/1406.3896.

[2] When to Prune? A Policy towards Early Structural Pruning. Maying Shen, Pavlo Molchanov, Hongxu Yin, Jose M. Alvarez. Proc. CVPR 2022.

[3] Unifying and Boosting Gradient-Based Training-Free Neural Architecture Search. Yao Shu, Zhongxiang Dai, Zhaoxuan Wu, Bryan Kian Hsiang Low. Proc. NeurIPS 2022.

### Questions
What do the authors believe is a reasonable path to move this paper towards some form of acceptance given the above concerns?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Different from the existing black-box HPO method that treats DNNs as black boxes, this paper proposes a method called NAP2 that treats DNNs as white boxes. NAP2 predicts final performance from weights and gradients of neural networks. The authors claim that the proposed method improves the performance over the baselines.

### Strengths
It is interesting that the final performance of a neural network training can be predicted by the information of its weights and gradients the learned predictors are transferable to other datasets.

### Weaknesses
The experiments are limited to neural architecture search only on CIFAR-10 and CIFAR-100, while the authors claimed that it is a hyperparameter optimization method. As they do not provide any theoretical perspective of this approach, they need more experiments in terms of tasks and datasets to support their claim.
    Gradient-based hyperparameter optimization, e.g., [1], can also be regarded as a white-box approach. Are there any comparisons with this line of works?

[1]: Maclaurin et al. "Gradient-based Hyperparameter Optimization through Reversible Learning," ICML 2015.
Questions:

### Questions
The paper only presents mini-batch steps, but how long does it take for architecture search in wall-clock time?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes Neural Architecture Performance Predictor (NAP2), which predicts final accuracy of neural network by analyzing dynamics of weights and gradients of network. NAP2 uses meta-features of neural network’s weights and gradients. NAP2 is utilized in neural architecture search by combining with Successive Halving. With sample architectures from NAS-Bench-101, NAP2-based NAS outperform than other hyperparameter optimization algorithms.

### Strengths
To the best of my knowledge, NAP2 is the first approach that trying to use meta-features of both weights and gradients. Also, the author’s hypothesis that “analyzing the ‘evolution’ of neural architectures (i.e., changes in weights and gradients) in their early training stages will enable a learning model to predict their final performance.” is reasonable.

### Weaknesses
The proposed method is more like a 'performance predictor' than a 'hyperparameter optimization method'[1]. NAP2 is trained with more than 1,000 evaluation results of fully trained neural network samples, but the comparison methods did not use them. It is more reasonable to compare it with performance predictors like BANANAS[2], NPENAS[3], or AG-Net[4].
(Since I do not have access to the dataset, I assume that the accuracy of the model is between 0 and 1 based on the paper's report that they use a sigmoid function for the output of the predictor.) Figure 6 shows that the mean square error between performance prediction and final accuracy is about 0.015, which is larger than the mean absolute error of other performance prediction results[5] in NAS-Bench-101 (mean absolute error < 0.01). Also, the result MSE > 0.86 in CIFAR-100 doesn't substantiate NAP2’s transferability if the scale of accuracy is 0 to 1.

[1] White, C., Zela, A., Ru, R., Liu, Y., & Hutter, F. (2021). How powerful are performance predictors in neural architecture search?. Advances in Neural Information Processing Systems, 34, 28454-28469.

[2] White, C., Neiswanger, W., & Savani, Y. (2021, May). Bananas: Bayesian optimization with neural architectures for neural architecture search. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 12, pp. 10293-10301).

[3] Wei, C., Niu, C., Tang, Y., Wang, Y., Hu, H., & Liang, J. (2022). Npenas: Neural predictor guided evolution for neural architecture search. IEEE Transactions on Neural Networks and Learning Systems.

[4] Lukasik, J., Jung, S., & Keuper, M. (2022, October). Learning where to look–generative nas is surprisingly efficient. In European Conference on Computer Vision (pp. 257-273). Cham: Springer Nature Switzerland.

[5] Siems, J. N., Zimmer, L., Zela, A., Lukasik, J., Keuper, M., & Hutter, F. (2020). Nas-bench-301 and the case for surrogate benchmarks for neural architecture search.

### Questions
I wonder if statistics of weights and gradients can represent the neural network without their connectivity. If those have meaningful information, similarly embedded networks will show similar performance. Is there experiment or evidence that supporting this hypothesis, like arch2vec[6] shows? (e.g. correlation between predicted performance and final accuracy, visualization of embedding vectors, etc.)
NAS-Bench-201[7] reports that the best architecture on CIFAR-10, CIFAR-100, ImageNet16-120 are slightly different. I also think the optimal architectures of different dataset should relate with complexity of dataset. Why NAP2 is “generic and transferable across datasets and architectures”?

[6] Yan, S., Zheng, Y., Ao, W., Zeng, X., & Zhang, M. (2020). Does unsupervised architecture representation learning help neural architecture search?. Advances in neural information processing systems, 33, 12486-12498.

[7] Dong, X., & Yang, Y. (2020). Nas-bench-201: Extending the scope of reproducible neural architecture search. arXiv preprint arXiv:2001.00326.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces NAP2, a hyperparameter optimization method for deep neural networks. Unlike existing approaches that treat DNNs as 'black boxes,' NAP2 is a 'white box' method that accurately predicts DNN performance by modeling changes in weights and gradients. NAP2 outperforms current state-of-the-art methods and is transferable across datasets, making it a valuable advancement in hyperparameter tuning for DNNs.

### Strengths
The paper introduces a method to predict model performance based on meta information and its feature map and gradients overtime. The idea is well-motivated. The method could potentially help improve hyperparameter optimization by reducing evaluation cost if the prediction of performance can be done with a small number of training budget.

The authors evaluate their methods on CIFAR-10 using the NAS-Bench 101, and show how it can be transfered to CIFAR-100.

### Weaknesses
While the idea is great, there are a few critical issues with the whole experiment desgin. 
1. If the focus of the method is for hyperparameter optimization, then the evaluation should follow a HPO pipeline, i.e., there should be no training dataset of models with ground-truth performance. The whole experiment should be in an online fashion. Otherwise, the proposed method is just a prediction model and do not show its usability in HPO.
2. The evaluation of resources used is unclear to me. Is the cost of training and infering NAP2 counted in calculation? Other baseline methods such as Successive Halving do not require extra computation, so you should probably compensate for those methods with extra computation to make it a fair comparison.
3. On CIFAR-100, the comparison methods seem to perform extremely bad, e.g., none of them get any precision > 0. I wonder if the configurations are reasonable according to the original work? It is unlikely that these methods should use the same set of configurations. 
4. For figure 6, is it saying that the prediction model trained on CIFAR-10 is able to predict on CIFAR-100 with MSE error < 0.1? This is a bit surprising because the scale of accuracy is quite different, i.e., on CIFAR-10 the models can achieve 0.9 accuracy but should be much lower like 0.6 on CIFAR-100. I wonder how is the proposed method handle this difference in scale? 
5. In Figure 2-5, it seems that all 3 methods do not benefit with more resources, which is not intuitive. I wonder if the authors could include a simple hill climbing or BO baseline to better illustrate the results?

Other weaknesses:
1. The figures are hard to read. Please increase the font sizes.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
