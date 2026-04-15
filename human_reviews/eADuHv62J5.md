# HeroLT: Benchmarking Heterogeneous Long-Tailed Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 5

## Abstract
Long-tailed data distributions are prevalent in a variety of domains, including e-commerce, finance, biomedical science, and cyber security. In such scenarios, the performance of machine learning models is often dominated by the head categories, while the learning of tail categories is significantly inadequate. Given abundant studies conducted to alleviate the issue, this work aims to provide a systematic view of long-tailed learning with regard to three pivotal angles: (A1) the characterization of data long-tailedness, (A2) the data complexity of various domains, and (A3) the heterogeneity of emerging tasks. To achieve this, we develop the most comprehensive (to the best of our knowledge) long-tailed learning benchmark named HeroLT, which integrates 15 state-of-the-art algorithms and 6 evaluation metrics on 16 real-world benchmark datasets across 5 tasks from 3 domains. HeroLT with novel angles and extensive experiments (304 in total) enables researchers and practitioners to effectively and fairly evaluate newly proposed methods compared with existing baselines on varying types of datasets. Finally, we conclude by highlighting the significant applications of long-tailed learning and identifying several promising future directions. For accessibility and reproducibility, we open-source our benchmark HeroLT and corresponding results at https://anonymous.4open.science/r/HeroLT-9746/.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper develops long-tailed learning benchmark named HEROLT, which integrates 15 state-of-the-art algorithms and 6 evaluation metrics on 16 real-world benchmark datasets across 5 tasks from 3 domains. They provide the systematic view from three pivotal angles and do extensive experiments.

### Strengths
1. The paper is exceptionally well-written, with each part being introduced in a clear and concise manner, making it easy to understand.
2. The paper demonstrates substantial effort, with convincing experimental results.
3. The paper presents a refreshing classification approach, introduces new metrics, and provides reliable benchmarks.

### Weaknesses
1. Although a new metric is designed and experimental analysis is conducted, the originality of this work is minimal. It mostly summarizes existing methods with few new designs or practices. It would be more valuable if there were innovative ideas and contributions.
2. The selection of datasets seems extensive, but it is essentially a combination of different domains, without a clear purpose. The data is heterogeneous, while the methods are specialized for different problems. After the integration, specific domain methods are still used for specific domain testing, without transferability or comparison. What is the purpose of this integration?
3. At least in the field of image recognition, these methods cannot be considered as "state-of-the-art methods" as mentioned in the paper. They serve as baselines in recent papers. For example, [1] and [2] are methods with better performance.
4. The finding that "none of the algorithms statistically outperforms others across all tasks and domains, emphasizing the importance of algorithm selection in terms of scenarios" is not surprising but rather intuitive. The subsequent analysis is based on specific dataset experiments, and it remains a mystery how to better guide the design process.

[1] Long-Tailed Recognition via Weight Balancing. CVPR22

[2] Distribution alignment: A unified framework for long-tail visual recognition. CVPR21

### Questions
1. What are the specific purposes and implications of Assumption 1 and Assumption 2 in the subsequent context? The use of insufficiently  proof by contradiction is questionable and unconvincing. For example, in the examples of Assumption 1, the feature distribution can be linearly separable. It is confusing to make assumptions about the data without considering the specific training process of the model. If the data possessed such favorable properties, the long-tail distribution problem would not be so challenging. Personally, I believe the difficulties lie in the model's inability to effectively distinguish and generalize features.

2. See others in weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper provides a systematic view of long-tailed learning with regard to three pivotal angles: (A1) the characterization of data long-tailedness, (A2) the data complexity of various domains, and (A3) the heterogeneity of emerging tasks. It integrates 15 state-of-the-art algorithms and 6 evaluation metrics on 16 real-world benchmark datasets across 5 tasks from 3 domains and proposes an open-source toolbox HEROLT for the long-tailed tasks.

### Strengths
1. The paper involves a significant amount of work, and its structure is relatively clear.
2. This paper introduces Pareto-LT Ratio as a new metric for better measuring datasets and gives guidelines for the use of this metric.
3.  It provides a fair and accessible performance evaluation of 15 state-of-the-art methods on multiple benchmark datasets that are contributing to the community

### Weaknesses
1. Despite the considerable effort invested in this work, I think that, apart from introducing the Pareto-LT Ratio, the paper falls short of provoking deeper contemplation on the long-tail problem for the readers. Instead, it primarily serves as a means of testing existing methods as baselines across various tasks.
2. I have reservations about the two assumptions mentioned in Section 2.1. Modern deep models' feature space distributions depend on the outcomes of the model learning process, and as a result, these two assumptions may not hold in some cases. 
3. I find that the categorization of long-tail tasks in Angle 3 may not be entirely reasonable. For instance, should classification include both multi-label text classification and image classification? And is it appropriate to group single-image and video classification as image classification? Additionally, some other tasks like long-tail semantic segmentation have not been included. Furthermore, certain methods listed in Table 3 may be applicable to multiple tasks; for example, the concept of decoupling could be applied to both classification and detection (segmentation).
4. The citations of methods in the experiments are needed. Meanwhile, some recent methods such as Paco and FCC in the image classification have not been discussed.

[1] Cui, Jiequan, et al. "Parametric contrastive learning." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
[2] Li, Jian, et al. "FCC: Feature Clusters Compression for Long-Tailed Visual Recognition." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
1. What's the meaning of "support region" in the Assumption 1 in Section 2.1?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper builds a comprehensive benchmark for the long-tail problem, containing multiple datasets and algorithms.

### Strengths
1. The long tail problem is a practical and valuable research topic.
2. Benchmark contains different data types and tasks.

### Weaknesses
1. Most of the datasets and tasks seem to have been well constructed and the paper seems to merely concatenate them together without much additional integration.
2. The methods discussed in the paper appear to be somewhat dated. For instance, the image classification data that is prominently featured in Table 2 is based on methodologies that were largely developed before 2020. Considering that it is already near the end of 2023, this renders the insights provided by the paper less credible.
3. Referring to section C.1 of the paper, there is a misalignment in the model architectures and the number of epochs used when comparing different methods, which calls into question the article's claim of a 'fair evaluation'.
4. According to the four-point analysis in the upper half of the third page, the Pareto-LT Ratio still requires integration with the Gini coefficient and IF to describe the dataset. The former primarily represents the number of categories, while the latter two indicate data imbalance. So why not directly use the number of categories as a descriptor? Additionally, I do not find comprehensive experimental and theoretical evidences to support this four-point analysis.

### Questions
See above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper systematically investigates the long-tailed problem in various scenarios. It proposes a heterogeneous view from the aspect of 1) long-tailedness or extreme categories; 2) data complexity of various domains; and 3) heterogeneous tasks. Different from the previous metric (Imbalanced factor, gini coefficient), it proposes a new metric (Pareto-LT) to help measure the extreme number of categories. By conducting empirical studies on various long-tailed benchmarks, the authors provide some insights that different methods may work for different tasks with different long-tailed properties.

### Strengths
This paper is well-written, and I think it will contribute to the long-tailed learning community.

1. As far as I know, this is the first work that systematically benchmarks long-tailed learning problems. Moreover, it categorizes long-tailed learning problems from different angles, including the data imbalance/extremes, data types/domains, and applied tasks. It almost covers recent important long-tailed learning challenges.
2. This paper proposes a novel metric, the Pareto-LT Ratio, for measuring both the degree of imbalance and the extreme number of categories of long-tailed datasets.
3. This paper conducts empirical studies on multiple datasets and compares popular methods. It provides valuable insights regarding which types of methods may work on a certain type of task.
4. The authors open-source a toolbox for evaluating long-tailed datasets.

### Weaknesses
1. The problem definition in Sec. 2.1 is under an ideal situation. The authors present two assumptions, i.e., the Smoothness Assumption and the Compactness Assumption, which indicate that all data points are annotated and clustered well. However, in real-world scenarios, this might not be easily achieved. A category may contain multiple subcategories, which may obey Assumption 1. Also, the data might contain feature or label noise, making it hard to extract distinctive representations, thereby obeying Assumption 2. The authors should either consider these real-world problems, otherwise declare that they just define long-tailed learning under an ideal situation.
2. The definition of long-tailed learning (Problem 1) seems trivial. It is similar to the definition of supervised learning, except that you emphasize the importance of both head and tail categories. It is better to illustrate the long-tailed property, i.e., $\mathbb{P}(\mathcal{Y})$ obeys a long-tailed distribution, or $\mathcal{Y}$ has an extreme cardinality.
3. The imbalance factor (IF) and Gini coefficient seem to perform a similar effect. The authors mainly consider them equally when compared with the Pareto-LT Ratio. Since the IF metric is more widely used, it seems that the Gini coefficient is less important. I suggest the authors discuss more about the difference between IF and Gini coefficient.

### Questions
1. In Fig. 2(a), what is the meaning of the Pareto curve?
2. Why do you choose 20% as a threshold for the Pareto-LT Ratio metric?
3. I find the performance of OLTR on CIFAR-10-LT and CIFAR-100-LT too well. It almost surpasses all long-tailed learning methods using ResNet-32 (https://paperswithcode.com/task/long-tail-learning). However, I failed to find the reproduced results of OLTR on CIFAR-LT in previous works. I wonder if there are some mistakes in this paper.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
