# AdCorDA: Classifier Refinement via Adversarial Correction and Domain Adaptation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 3, 6, 3

## Abstract
This paper describes a simple yet effective technique for refining a pretrained classifier network. The proposed AdCorDA method consists of two stages - adversarial correction followed by domain adaptation. Adversarial correction uses adversarial attacks to correct misclassified training-set classifications. The incorrectly classified samples of the training set are removed and replaced with the adversarially corrected samples to form a new training set, and then, in the second stage, domain adaptation is performed back to the original training set. Extensive experimental validations show significant accuracy boosts of over 5% on the CIFAR-100 dataset and 1% on the CINIC-10 dataset. The technique can be straightforwardly applied to the refinement of weight-quantized neural networks, where experiments show substantial enhancement in performance over the baseline. The adversarial correction technique also results in enhanced robustness to adversarial attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed AdCorDA, a classifier refinement framework consisting of two stages. In the first stage, those incorrectly predicted examples are modified by adversarial attacks to move to correct predictions. In the second stage, the model is adapted from the unmodified examples and modified examples back to the original domain. Through extensive experiments, AdCorDA is proven to improve the performance of the normally trained baseline classifier. Besides, it can also improve the robustness under adversarial attacks.

### Strengths
- The paper is well-written and easy to follow. 
- The idea of the two-stage correction-then-adaptation framework is intuitive and reasonable. 
- The proposed method is effective and evaluated by extensive experiments.

### Weaknesses
- I noticed that you set the attack budget $\epsilon=5e-4$, which is much smaller than the commonly used $\epsilon=8/255$. It is not convincing to claim that "our method provides significant robustness to adversarial attacks" under such a weak attack where the baseline model can still get over 15% accuracy. 
- The proposed method is only compared with a vanilla baseline. No other curriculum learning or domain adaptation methods are compared.

### Questions
- It seems that AdCorDA is only able to refine neural networks that don't get a very high training accuracy. Let's consider a highly overfitting scenario. Imagine that there is a highly over-parameterized neural network trained on a small-scale dataset, and this network is capable of overfitting the whole dataset, getting 100% training accuracy. This is likely to happen, as existing research like [1] already proved that over-parameterized neural networks can easily get 100% training accuracy even on randomly labeled CIFAR-10. In this case. no training examples are incorrectly predicted. How does AdCorDA work? 
- In Tab. 4 How are the baseline models trained? Via normal training or adversarial training? 
- Will stronger attacks help the first stage? I guess stronger attacks will increase the success rate of label correction.
- In Tab. 1, does BL-IST with Attack=None means just domain adaptation?

[1] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2021). Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3), 107-115.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this paper, the authors propose a method to refine a pre-trained model by incorporating adversarial correction and domain adaptation. The method is motivated by prior work which shows that applying domain adaptation from easy to hard in curriculum learning can speed up training. So they create an easy dataset using adversarial correction and then align the synthetic dataset with the original dataset by aligning the second-order covariance matrices. They conducted some experiments to validate the superiority of their method.

### Strengths
* The authors introduce a simple yet effective way to refine a pre-trained classifier.
* They conduct fair experiments to validate the performance.

### Weaknesses
* I suggest that the authors show a more intuitive figure to visualize the framework that includes the images and labels in the original dataset and also the corrected images. This will help the readers to gain more intuition for your method.
* The authors combine two existing techniques to get the framework without innovation. The adversarial attack or correction method and the domain adaptation method used by the authors are proposed by prior work. And the adopted domain adaptation method here is a very old and simple method which is proposed eight years ago. Considering there were so many effective domain adaptation methods proposed in the recent few years, why don't you use other domain adaptation methods to further improve the performance?
* In Section 3.3, the authors align the features of the weak classifier on the original dataset and the synthetic dataset. Considering that the difference between the original and the synthetic datasets is the corrected part, can we omit the correctly classified samples and only minimize the covariance difference for the adversarially corrected sample and the misclassified sample?
* How do you choose the hyper-parameters such as $\lambda,\epsilon$? Does your method work robustly for other choices of hyper-parameters? If not, how do you choose them?

### Questions
* Which attack (correction) method do you use? In the introduction part, the authors say that they use the adversarial attack method targeting the true label to get an easier dataset where the classifier has 100% accuracy. However, the BI method generates more difficult samples for the classifier. Do you use this method? And what's the role of auto-attack in 3.2?

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
3

### Summary
Summary: This paper introduces AdCorDA, a method for refining pretrained classifiers, particularly targeting smaller networks on edge devices. The proposed technique consists of two stages: (1) adversarial correction, where adversarial attacks adjust misclassified samples to be correctly labeled by the network, creating a synthetic training dataset, and (2) domain adaptation, where the network is adapted from this synthetic dataset back to the original dataset to bridge any distribution shift. Experiments on CIFAR-10, CIFAR-100, and CINIC-10 demonstrate that AdCorDA provides performance improvements, with gains of 5% on CIFAR-100 and 1% on CINIC-10, while also enhancing the robustness of both full-precision and quantized networks against adversarial attacks.

### Strengths
I really liked the idea of using adversarial example methods for fixing hard examples. I really liked the paper both in terms of novelty and contributions.

The paper was well written, and easy to follow.

### Weaknesses
1) I believe that the paper could really benefit from further experimentation. Even though CINIC 10 is kind of a bridge between ImageNet and CIFAR datasets, I still do not see the limiting factor from scaling up to imagenet experiments. I also wonder if the effectiveness  of  this method limited to CNNs. I would like to ask authors to try ImaeNet on Transformer architecture and report the results or simply explain why such experiments would not be practical/relevant.

### Questions
1) As I mentioned previously, I would really love to ask for extending the experiments to larger datasets like ImageNet and their structure to transformer based models. 

2) There are several studies that show that datasets like CIFAR and IMAGENET come with a noticeable number of incorrectly labeled samples. Intuitively in a perfect classifier, such noisy labeled samples should be predicted to have the correct label, which is different from the incorrect class specified in the labels from the dataset. I wonder how much of the robustness is gained from the removal of such images? Or maybe, the authors could visualize all the examples that start in Tw but do not end up in Ta. Do most of such examples have the wrong label assigned to them in the dataset? How much of such images end up in Ta? 

3) I cannot help but wonder regarding the effectiveness of your method and the noisy label settings. I was wondering if the authors have considered trying their method against noisy label settings for a basic setting like a resnet34 trained on 20% Symmetrical Noisy Labeled setting on CIFAR-10/100. 

If my questions/concerns are addressed, I will be happy to increase my score.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a method for refining pre-trained classifiers through adversarial correction and domain adaptation. Adversarial correction adds adversarial perturbations to misclassified samples to correct the network's predictions. Then an existing domain adaptation method Deep CORAL is applied to adapt from the corrected dataset back to the original data. Experiments on CIFAR-10, CIFAR-100, and CINIC-10 show consistent accuracy gains and improved adversarial robustness. The method is claimed to be friendly on edge devices.

### Strengths
1. The method for refining the classifier by adversarial correction and domain adaptation is novel to me.
2. Experiment results are pretty promising.

### Weaknesses
1. The insight behind the method is not clear. The authors did not make a strong effort to demonstrate why their method is effective. No theoretical proof is provided. And it is not clear what motivates this method. It is more like a heuristic without sufficient causal explanation.
2. It is questionable whether the adversarial correction is really necessary. From the experiments, the performance with no attack is very close to the methods with attacks, sometimes even slightly better. The authors did not demonstrate why they need adversarial correction. I am confusing the motivation for introducing this stage in your method. 
3. Only compare with baselines without adversarial attack and domain adaptation. I wonder if it's necessary to compare with other related methods, like methods for edge devices? It is ambiguous about the advantages of the proposed method, although the authors show some accuracy improvements. If you claim to achieve better accuracy, you need to compare with more classification methods. If you claim to achieve better adversarial robustness, you need to compare with more adversarial learning methods. If you claim efficiency and resource-friendly on the edge devices, you need to compare methods implemented on edge devices.
4. In Table 5 of the supplementary material, the running time and memory are reported. The results are somehow misleading. The time of adversarial correction is reported by batch while the other steps are reported by epoch. Thus adversarial correction should introduce much more computing time. I am concerned whether this method can really be implemented on edge devices, given such an extremely large additional computing time.

### Questions
1. Why do you use pretrained models, like ImageNet pretrained, Place365 pretrained? Does training from scratch for your method and baselines produce different results?
2. What do the choices of adversarial attack hyperparameters like epsilon, iteration step, step size, etc, influence in your method?

### Soundness
2

### Presentation
3

### Contribution
2
