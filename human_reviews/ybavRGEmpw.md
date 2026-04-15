# Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
We introduce the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These methods are based on a new class of optimal-transport-regularized divergences, constructed via an infimal convolution between an information divergence and an optimal-transport (OT) cost. We use these as tools to enhance  adversarial robustness  by maximizing the expected loss over a neighborhood of distributions, a technique known as distributionally robust optimization. Viewed as a tool for constructing adversarial samples,  our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence. We demonstrate the effectiveness of our method on malware detection and image recognition applications and find that, to our knowledge, it outperforms existing methods at enhancing the robustness against adversarial attacks. $ARMOR_D$ yields the robustified accuracy of 98.29\% against $FGSM$ and 98.18\% against $PGD^{40}$ on the MNIST dataset, reducing the error rate by more than $19.7\%$ and $37.2\%$ respectively compared to prior methods. Similarly, in  malware detection, a discrete (binary) data domain, $ARMOR_D$ improves the robustified accuracy under $rFGSM^{50}$ attack compared to the previous best-performing adversarial training methods by $37.0\%$  while lowering false negative and false positive rates by $51.1\%$ and $57.53\%$, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper defines a new divergence considering both the information divergence and optimal transport and then investigates adversarial robustness in terms of the defined divergence from a distributionally robust optimization perspective. Based on the analysis, the authors propose the ARMOR\_D algorithm and then obtain SOTA performance on MNIST and Malware compared with FGSM, PGD, TRADES, and MART.

### Strengths
- The paper generalizes the adversarial training framework and OT-DRO by defining a new divergence considering both the information divergence and optimal transport.
- The paper provides a theoretical analysis of the proposed framework.

### Weaknesses
- The empirical evaluation is not sufficient, which is not enough to show the effectiveness of the proposed method.
  - The baselines are weak. The latest baseline used in the paper is proposed in 2020. However, it is 2023 now, there are so many good methods proposed these years. So I think the authors should compare their methods with the latest SOTA method.
  - There are only **two small** datasets. Firstly, I think the authors should conduct experiments on more datasets to indicate consistent improvement. Secondly, the experiments should be conducted on larger datasets, MNIST is a very easy task, the authors should conduct experiments on CIFAR10, CIFAR100, and ImageNet.
  - The backbone is small, which is not consistent with the word "deep learning" in the topic of this paper. Larger neural networks should be used (together with larger datasets).
  - Stronger attack methods should be used. The paper only reports performance under FGSM and PGD attacks, which are not strong enough. Stronger attacks such as CW and AutoAttack should be used.
- There are some other works that investigate adversarial robustness in terms of distributionally robust optimization [1-2], the discussions of related works are mission. Furthermore, given the existence of these papers, I think this paper is not novel enough.
- The theoretical analyses only show that the proposed framework is more general, but do not show why the more general framework behaves better. In general, a more specific framework may contain additional knowledge about the task and lead to better performance, so the fact that a more general framework leads to better performance is strange to me. I think it needs to be carefully explained.


### references

[1] Certifying Some Distributional Robustness with Principled Adversarial Training.

[2] A Distributional Robustness Perspective on Adversarial Training with the $\infty$-Wasserstein Distance.

### Questions
- See the weaknesses.
- Why adding natural examples to the training can improve adversarial robustness? Does this still hold for larger datasets such as the CIFAR and the ImageNet?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper provides a novel method for adversarially robust training of deep learning models. Essentially, authors propose to use the extension of the common Adversarial Training where the internal maximization process should be conducted on top of both 1) divergence term between the "close" adversarial distribution and the original empirical one, and 2) optimal transport between these two divergences.

### Strengths
First of all, a lot of seeming correctly theoretical background and proofs are provided (especially in the Appendix A), which make the work standing out of a common adversarial robustness methods.
Additionally, authors provided a rigorous experimentation analysis of different variants of their method for two modalities (images and malware detection) and prove the superiority of their approach.
Finally, which is very nice, their method improves even no-attack setting, which is very important when we don't have adversaries - the system still should work reliable.

### Weaknesses
Although the paper is very well written with a lot of theoretical details, there are still some (I hope minor) weaknesses:
- the work used a super small and toy dataset for images - MNIST - the scale and reliability on results obtained working with it are unlikely used anywhere in reality. Moreover, the conclusion obtained can be misleading. The golden standard is to use ImageNet as a must for the image datasets (probably the same is for the Malware detection dataset, I have just not worked with it)
- when comparing the results shown in Tables 1 and 2, we can see that the best OT-regularized divergences methods in terms of accuracy are very different for image recognition and malware detection problem. What is the root cause behind it? No any written hypothesis or discussion. I guess (and see the item above) it is somehow related to MNIST dataset and that the results there are like 98-99% which makes all the improvement marginal and not very generalizable 
- starting the Page 2, the cost function $c(x,y)$ is introduced, but all the theorems and results are only valid if $c(x,y)$ is non-negative, but it is not mentioned in the main text of the paper (only in Appendix)

### Questions
Here are a couple of questions
- what was the original motivation to combine OT and DRO? E.g., some clear bad cases for OT are not in the common DRO and vice versa etc.
- what was the reasoning behind choosing a probability-related term for the cost function as a $g_{\delta}(z) = z/(1-z/\delta)$? To mimic sigmoid?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the novel approaches to enhancing the adversarial robustness of deep learning models. This paper enhances adversarial robustness by maximizing the expected loss over a neighborhood of distributions for distributionally robust optimization. For constructing adversarial samples, the proposed method allows samples to be both transported and re-weighted, according to the OT cost and the information divergence. The authors demonstrate the effectiveness of the proposed method on malware detection and image recognition applications and find that it outperforms existing methods at enhancing the robustness against adversarial attacks.

### Strengths
1. This work proposes a novel class of divergences for comparing probability distributions called optimal-transport-regularized divergences and uses them to construct distribution neighborhoods for use in distributionally robust optimization.
2. This paper proves a number of new properties of the OT-regularized divergences and demonstrates the effectiveness of the OT-regularized divergence method as a novel approach to enhancing adversarial robustness in deep learning leading to substantial performance gains.

### Weaknesses
1. For the classification problem on the continuous data, this paper tests the ARMORD adversarial robustness methods on MNIST digit classification, where the dimension of image in the MNIST is 28*28. Note that the computation of Optimal Transport (OT) is closely related to the dimension of data, thus it would be interesting to see the performance of the proposed approach and its counterparts on the image data with larger dimension such as CIFAR-10 with 3*32*32 or the tiny-ImageNet with 3*64*64. 
2. In the comparison of the performance for enhancing the robustness on the MNIST dataset, only the gradient-based attacks are used to evaluate the adversarial robustness. It would be more sufficient for the adversarial robustness of proposed Optimal-Transport-Regularized Divergences if there are evaluations of the optimization-based attack such as CW and the stronger comprehensive attack AutoAttack. 
3. Note that the proposed OT-regularized divergences may need more computational cost than the conventional divergence, thus it would be more meaningful for the practical use to provide the comparison of the computational costs between the proposed OT-regularized divergences based adversarial training and the counterparts.

### Questions
This paper proposes the optimal-transport-regularized divergences to enhance the adversarial robustness of deep learning models. Yet, it still needs more details and evidences to further explain the effectiveness of the proposed approach.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed adversarial training with optimal-transport-regularized divergence. It is a framework solving distributional robustness optimization problem where the distributional divergence is regularized by optimal transport. The authors comprehensively study the properties of this problem and demonstrate the effectiveness of the derived algorithms.

### Strengths
The paper proposes a framework that is generally applicable and mathematically elegant. The properties of the OT-regularized divergence are nice and the derived algorithm is nice. The paper is well-written and easy to read.

### Weaknesses
The theoretical part is nice, my major concern is the empirical part.

1. The datasets studied in the empirical study are very small and are generally considered as toy examples. Experiments on a larger dataset will make the empirical results more convincing.

2. The robustness evaluation is more comprehensive. For example, PGD and FGSM cannot comprehensively and reliably evaluate the robustness of a model as indicated in [A]. The authors should use auto-attack, which is an ensemble of four different attacks, to comprehensively evaluate the robustness and to make the results more convincing.

[A] "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks". Francesco Croce, Matthias Hein.
ICML 2020.

### Questions
The paper is generally well-written and interesting. My questions are the two points in the weakness part. The manuscript will be better with the empirical issues addressed.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
