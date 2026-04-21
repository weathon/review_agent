# Deep Generalized Prediction Set Classifier and Its Theoretical Guarantees

- Avg Score: 3.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
A standard classification rule returns a single-valued prediction for any observation without a confidence guarantee, which may result in severe consequences in many critical applications when the uncertainty is high. In contrast, set-valued classification is a new paradigm to handle the uncertainty in classification by reporting a set of plausible labels to observations in highly ambiguous regions. In this article, we propose the Deep Generalized Prediction Set (DeepGPS) method, a network-based set-valued classifier induced by acceptance region learning. DeepGPS is capable of identifying ambiguous observations and detecting out-of-distribution (OOD) observations. It is the first set-valued classification of this kind with a theoretical guarantee and scalable to large datasets. Our nontrivial proof shows that the risk of DeepGPS, defined as the expected size of the prediction set, attains the optimality within a neural network hypothesis class while simultaneously achieving the user-prescribed class-specific accuracy. Additionally, by using a weighted loss, DeepGPS returns tighter acceptance regions, leading to informative predictions and improved OOD detection performance. Empirically, our method outperforms the baselines on several benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a new method that combines set-valued prediction with out-of-distribution detection in multi-class classification problems. The central idea is a risk minimization framework with a loss function that consists of three parts. The first two parts trade off set size and accuracy, while a weight parameter controls which of the two terms is more important. The third term allows to exclude atypical examples from acceptance regions. In addition, the penultimate layer of the neural uses random fourier features to approximate a Gaussian kernel. 

The authors present theoretical results that present (a) the quality of the random Fourier feature approximation (b) the convergence to Bayes risk when sample size increases. In the experiments the proposed methods is compared to three baselines on three datasets and three metrics. The metrics evaluate the set-valued prediction and OOD detection performance.

### Strengths
- The presented method is novel
- Overall the paper is well written (but some parts are unclear, see below)
- I agree with the authors that combined set-valued prediction and OOD detection is a key concept in satefy-critical applications of AI. 
- I liked that the authors describe the problem setting and the assumptions formally (Assumptions 1 and 2). This is often missing in OOD detection papers.

### Weaknesses
This is a quite technical paper, and I am afraid that I don't understand the method very well, despite having a background on the topic and spending quite some time to read the paper. The last part of objective function (1) is unclear to me. What are lambda_k and rho_k? Are these explained in the paper? The authors explain that rho_k is used to exclude atypical examples from acceptance regions, but I don't see yet how that's going on. Also, why is a Frobenius-penalty needed for the parameter matrices? This is quite atypical for deep learning methods, where regularization is typically done via early stopping in SGD.

Furthermore, the need for random Fourier features in the penultimate layer of the neural network is also unclear to me. What does this component add to the method, compared to just propagating the embedding to the output layer? 

I also find the connection to existing literature a bit weak. The literature on OOD detection is vast, so I understand that the authors cannot discuss every paper, but some essential papers are definitely missing. Assumption 1 clearly motivates why generative models / density-based models are a good approach to represent P(x|y) as a first step for combined set-valued prediction and OOD detection. The authors discuss a few methods that model P(x|y), such as the unpublished work of Hechtlinger et al. However, there are many other papers that also model P(x|y), such as:
Charpentier et al. Posterior network: Uncertainty estimation without ood samples via density-based pseudo-counts, ICLR 2021
Van Amersfoort et al. Uncertainty estimation using a single deep deterministic neural network, ICML 2020
Lee et al. A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks, Neurips 2018

Perhaps those papers don't evaluate set-valued prediction, but they can be immediately used for such purposes. From that perspective, I would argue that such methods are also better baselines than the current baselines. These methods are deep learning methods, and they are published, unlike two of the three papers that are currently used as baselines. If I would have to do simultaneous set-valued prediction and OOD detection for a specific application, I would be more tempted to try these methods first instead of the method proposed here, because for methods that model P(x|y) it is more clear what they are doing. For the proposed method I am not sure whether it is modelling a class-specific density P(x|y). This might be realized via the random Fourier features, but more explanation would be needed. 

Furthermore, in set-valued prediction there are also quite some methods that consider loss functions that consist of two parts: a part that minimizes accuracy, and another part that minimizes set size, see e.g. 
Mortier et al. Efficient set-valued prediction in multi-class classification, Data Mining and Knowledge Discovery, 2020. 
Titouan Lorieul, Uncertainty in predictions of deep learning models for fine-grained classification, PhD thesis, University of Montpellier, France. 

The proposed method has a lot of connections with such methods, but there are two differences: (1) the proposed method has an additional third part in the loss, (2) the other methods typically fit a probabilistic model first, and optimize set-based utility scores during the inference phase. Perhaps these two differences are enough to behave good for OOD detection as well, but that's still unclear to me.

### Questions
See above.

### Soundness
2 fair

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
This paper proposes a novel way to learn a set-valued classifier, called Deep Generalized Prediction Set (DeepGPS); the proposed method is capable of identifying ambiguous observations and detecting out-of-distribution observations. Also, it is the first set-valued classification with a theoretical guarantee and scalable to large datasets. In theory, this paper provides that DeepGPS attains the optimal expected prediction set size, while achieving the user-prescribed class-specific accuracy. The efficacy of DeepGPS is demonstrated by using MNIST/CIFAR10/Fashion-MNIST datasets and multiple baselines.

### Strengths
This paper proposes a learning approach for DeepGPS along with its theoretical properties in Thm1-3.

### Weaknesses
I appreciate the authors' careful analysis on the algorithm. I was initially surprised that the proposed approach can achieve a user-prescribed class-specific accuracy \gamma without training a base model and additional set predictor in a decoupled way. However, I found that it is simply due to the hyperparameter C tuning in a validation set. In this regard, I found that the stated guarantee in (1) of Thm3 is largely disconnected to the empirical results. 

Afterward, I was not convinced whether we actually need this complicated way in training the entire neural network. I’d rather simply use conformal prediction for the specified task (e.g., BCOPS). 

Also in representing the main results in Table 1, I think the class-specific accuracy results should be bolded if they are close to the desired level; however, the maximum values are bolded.

### Questions
1. Based on the appendix, the hyperparameter C is chosen to achieve the desired class-specific accuracy, i.e., “The tuning parameter C is determined such that the prediction set is smallest on the unlabeled part in the validation data when the misclassification rate is close to γ on the labeled part in the validation data.” Then, what’s the meaning of (1) of Thm3? I think without this theorem, we can heuristically achieve the desired class-specific accuracy via hyperparameter tuning over a validation set. 

2. Related to the above question, can you re-evaluate the benefit of DeepGPS compared to BCOPS? I think simple training in a decoupled way provides a stronger guarantee. 

3.  In Table 1, is there a specific reason that the accuracies are highlighted when it is the largest number? Otherwise, please use bold numbers if the class-specific accuracy results are close to the desired level.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors explore set prediction, or conformal prediction, within the context of out-of-distribution (OOD). In this prediction paradigm, rather than offering a singular classification result, a predictor provides a set of labels. This set is expected to encompass the true label with a high degree of certainty. When dealing with OOD, predicting an empty set becomes a significant indication, suggesting the assignment of an OOD label to the test data point. The authors introduce an algorithm that employs Random Fourier Features, ensuring scalability in relation to sample size. Furthermore, they present the adaptive weighted hinge loss and offset penalization techniques to boost classification efficiency. The paper theoretically investigates the expected prediction set size for their algorithm, showing that it approaches the optimal size as the sample size grows. Experimental outcomes underscore that their algorithm surpasses existing methods. Moreover, the components of the adaptive weighted hinge loss and offset penalization play pivotal roles in enhancing classification efficiency.

### Strengths
1. The paper is well-written and easy to follow.

2. Addressing set-valued classification issues in OOD scenarios is both demanding and imperative. Issues of trustworthiness and OOD can stymie the deployment of machine learning algorithms in real-world applications. Developing an algorithm for set-valued classification within OOD scenarios augments the applicability of machine learning techniques.

3. The proposed elements—adaptive weighted loss and offset penalization—are astutely crafted to evaluate the accuracy constraint more rigorously and to minimize the expected set size.

4. Theoretical insights guarantee that the classifier obtained by the proposed algorithm will attain the optimal expected set size achieved by the ideal classifier. This underscores the rationale behind the algorithm's design.

5. Experimental findings robustly attest to the proposed method's dominance over existing techniques in terms of the metrics evaluated.

### Weaknesses
1. The rationale behind incorporating Random Fourier Features is ambiguous. Attaining scalability can be realized by merely employing a fixed-width network as the penultimate layer. Resorting to infinite-dimensional kernel features as the penultimate layer seems unnecessary without a clear justification, making the algorithm's design seem somewhat ill-advised.

2. The theoretical findings seem to be direct derivations from the generalization bound established through the Rademacher complexity. Their technical significance remains dubious. Furthermore, given that the core contributions revolve around the introduction of adaptive weighted loss and offset penalization, the impact of these components on generalization error remains unexplored. Consequently, the results offer limited support for the algorithm's design.

3. There is likely an intrinsic trade-off between OOD recall and Efficiency as gauged in the experiments. Thus, assessing this trade-off's efficiency becomes crucial. A comprehensive superiority assertion for the proposed algorithm necessitates comparative analyses of such trade-off efficiencies.

4. It is also vital to assess the trade-off between OOD recall and Efficiency within the ablation studies.

5. The authors seem to incorporate the adaptive weighted loss with an aim to enhance the precision of class-wise error assessments. Therefore, to ascertain the efficacy of this component, evaluations of the precision of class-wise errors should be undertaken.

### Questions
1. Would the authors shed light on the imperative of integrating the Random Fourier Features into their methodology?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
