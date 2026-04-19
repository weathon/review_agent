# BWS: Best Window Selection Based on Sample Scores for Data Pruning across Broad Ranges

- Decision: Reject
- Scores: 6, 5, 5

## Abstract
Data subset selection aims to find a smaller yet informative subset of a large dataset that can approximate the full-dataset training, addressing challenges associated with training neural networks on large-scale datasets. However, existing methods tend to specialize in either high or low selection ratio regimes, lacking a universal approach that consistently achieves competitive performance across a broad range of selection ratios. We introduce a universal and efficient data subset selection method, Best Window Selection (BWS), by proposing a method to choose the best window subset from samples ordered based on their difficulty scores. This approach offers flexibility by allowing the choice of window intervals that span from easy to difficult samples. Furthermore, we provide an efficient mechanism for selecting the best window subset by evaluating its quality using kernel ridge regression. Our experimental results demonstrate the superior performance of BWS compared to other baselines across a broad range of selection ratios over datasets, including CIFAR-10/100 and ImageNet, and the scenarios involving training from random initialization or fine-tuning of pre-trained models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method that aims to find informative subset of the original datasets which can be used to train the neural networks with small performance drop compared with model trained with whole dataset. The point of this paper is to propose a method that can do both universal and efficient selection of subset based on the difficulty score. To adaptively select the best subset, the authors propose a method based on kernel ridge regression. The proposed method can be used to select subset for both training from scratch and fine-tuning. Extensive experiments are conducted to verify the efficacy of the proposed method.

### Strengths
This paper gives a deep understanding of which kind of data can be useful for different size of subset and use kernel regression to analyze this problem theoretically. 
The situations  for hard sample and easy sample  to have benign effect is reasonable. 
The usage of kernel ridge regression for subset selection is interesting.  The details of each parts of the proposed method are illustrated clearly.
Extensive experiments validate the efficacy of the proposed method for training from scratch. 
The method can also be effective when used to select subset for fine-tuning.
Ablation studies also validate the robustness of the proposed method.

### Weaknesses
For the experiments on CIFAR-10 with noise, the proposed method is outperformed by Moderate DS for 3 ratios. Could the authors illustrate the noisy rate of the selected subset to check whether the proposed method is prone to choose noisy data under this setting?

The experiments on CIFAR-10 fine-tuning on VIT shows that CCS is consistently better than the proposed method, could the author give concrete analysis of this phenomenon?

### Questions
Please refer to weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a novel and universal coreset selection method called "Best Window Selection (BWS)" to strike a balance between sample diversity and model performance across a broad range of selection ratios. BWS first sorts all training examples w.r.t. the difficulty score and then prunes a specific number of the most difficult examples and easiest examples. By comparing BWS with other SOTA baselines, the evaluation results show that BWS outperforms other coreset selection methods.

### Strengths
1. This paper proposes a novel coreset selection method, BWS. Compared to previous work, BWS selects the best window more efficiently with kernel ridge regression, which is faster than training a model from scratch.

2. The evaluation results show that BWS achieves better or comparable results to other SOTA methods.

3. The overall writing is good and easy to follow.

### Weaknesses
1. Using kernel ridge regression to decide the best window is not quite intuitive. What is the motivation to use kernel ridge regression rather than training a small network to decide the best window?

2. The baseline evaluation results are inconsistent with data reported in the baseline method. For example, moderate are reported to have better performance than random on CIFAR10. CCS seems to have a better performance at 10% subset ratio than the numbers reported in the paper. It may be good to explain why the difference exists.

### Questions
I don’t fully understand why the performance of $w_s$ can represent the performance of models trained on the same subset. Could the authors further explain the connection between kernel regression and deep learning model training? What I currently feel is that it is more like an empirical transferability stuff studied in [1]: it is possible to use a small model to select coresets that transfer well to larger models.

[1] Coleman, C., et al. "Selection via Proxy: Efficient Data Selection for Deep Learning." International Conference on Learning Representations (ICLR). 2020.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents an approach, known as Best Window Selection (BWS), designed to tackle the challenges associated with data subset selection in machine learning. BWS allows for the adaptable selection of subsets based on sample difficulty scores and consistently delivers competitive performance over a broad range of selection ratios, spanning from 1% to 90%. It excels in comparison to existing score-based and optimization-based methods when applied to datasets like CIFAR-10/100 and ImageNet.

### Strengths
1) The problem studied is meaningful and significant: finding a versatile data selection approach capable of sustaining competitive performance across a diverse range of selection ratios.
2) Experiments show that the proposed BWS consistently outperforms other baselines, including both score-based and optimization-based approaches.
3) The authors provide code, which enhances the reproducibility.

### Weaknesses
1) The notion of a "window" refers to a fixed-length interval within a sorted dataset. The "Best Window Selection (BWS)" algorithm operates under the assumption that the most optimal subset should be contiguous regarding the level of difficulty. However, the paper lacks an in-depth analysis of this particular aspect.

2) It would be intriguing to explore the broader scenario where a "window" comprises several smaller intervals and varying starting points.

3) Figure 3's readability could be enhanced by employing more distinguishable colors and markers for clarity.

### Questions
Kindly refer to the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
