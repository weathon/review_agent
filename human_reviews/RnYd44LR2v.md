# OODRobustBench: benchmarking and analyzing adversarial robustness under distribution shift

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Existing works have made great progress in improving adversarial robustness, but typically test their method only on data from the same distribution as the training data, i.e. in-distribution (ID) testing. 
As a result, it is unclear how such robustness generalizes under input distribution shifts, i.e. out-of-distribution (OOD) testing. This is a concerning omission as such distribution shifts are unavoidable when methods are deployed in the wild. 
To address this issue we propose a benchmark named OODRobustBench to comprehensively assess OOD adversarial robustness using 23 dataset-wise shifts (i.e. naturalistic shifts in input distribution) and 6 threat-wise shifts (i.e., unforeseen adversarial threat models). 
OODRobustBench is used to assess 706 robust models using 60.7K adversarial evaluations. This large-scale analysis shows that: 1) adversarial robustness suffers from a severe OOD generalization issue; 2) ID robustness correlates strongly with OOD robustness, in a positive linear way, under many distribution shifts. The latter enables the prediction of OOD robustness from ID robustness. Based on this, we are able to predict the upper limit of OOD robustness for existing robust training schemes. The results suggest that achieving OOD robustness requires designing novel methods beyond the conventional ones. Last, we discover that extra data, data augmentation, advanced model architectures and particular regularization approaches can improve OOD robustness. Noticeably, the discovered training schemes, compared to the baseline, exhibit dramatically higher robustness under threat shift while keeping high ID robustness, demonstrating new promising solutions for robustness against both multi-attack and unforeseen attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides a large-scale benchmark for evaluating the adversarial robustness of models on datasets under distribution shift (OOD robustness). It supports 23 dataset-wise shifts (e.g., image corruptions) and 6 threat-wise shifts (i.e., different adversarial attacks) and is used to assess the OOD robustness of 706 pre-trained models. Based on the experimental analysis, this work has some insightful findings, e.g.,  1) robustness degrades significantly under distribution shift, 2) ID accuracy (robustness) strongly correlates with OOD accuracy (robustness) in a linear relationship. Based on the finding of the linear relationship, the authors propose to predict the OOD performance of models using ID performance. Finally, some explorative studies such as data augmentation have been conducted and demonstrated to be useful to improve the OOD robustness.

### Strengths
- Large scale evaluation with diverse distribution shift. OODRobustBench supports 29 types of distribution shifts and 706 types of models that can provide a good platform for researchers to further analyze the OOD robustness problem.

- Some interesting findings, e.g., adversarial training can boost the correlation between the ID accuracy (robustness) and OOD accuracy (robustness), and no evident correlation when ID and OOD metrics misalign.

- Useful guidance. I like the part of Section 5 that explores the usefulness of using multiple methods (even though they are from existing works) to enhance the OOD robustness.

- This benchmark systematizes the question of whether robustness acquired againt a specific threat model transfers to other threat models. Although this is not the first work that investigate this aspect, this is to my knowledge the first large-scale benchmark considering this question.

### Weaknesses
- Only one seen attack method (MM5) has been used for the evaluation, which is not practical for robustness analysis. 
Some findings are expected or have already been studied in existing works. 

- Some works [1, 2, 3] studied the linear correlation between ID performance and OOD performance, the authors need to add some related discussions. Instead of the finding of the correlation between ID accuracy (robustness) and OOD accuracy  (robustness) that is expected, the finding that there is a weak correlation between ID accuracy and OOD robustness for ImageNet is more attractive and needs a more detailed explanation.

- Section 4.3 analyzes the upper limit of OOD performance using the linear correlation. However, the authors did not consider factors that could further improve the limit such as OOD generalization methods which makes the conclusion not that convincing. Instead of analyzing the limits,  it is better to try more model accuracy prediction methods such as [1, 2, 3] to evaluate their effectiveness in assessing OOD robustness.  

[1] Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift, Neurips 2022.
[2] Leveraging Unlabeled Data to Predict Out-of-Distribution Performance, ICLR 2022. 
[3] Are labels always necessary for classifier accuracy evaluation? CVPR 2021

This is a borderline paper. Even though this paper provides the first benchmark for OOD robustness evaluation, there are some concerns that need to be addressed, the limited seen attack methods used, some findings already revealed by existing works, and lack of the study of OOD generalization methods.

### Questions
1. Compared to existing works [1, 2, 3], can you please summarize the new findings from OODRobustBench?
2. After improving the OOD robustness using the methods in Section 5, do you think the findings revealed by the previous sections will change?
3. Do you think other OOD generalization methods like unsupervised representation learning for OOD generalization [4] can help increase the limit of OOD robustness?

[1] Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift, Neurips 2022.
[2] Leveraging Unlabeled Data to Predict Out-of-Distribution Performance, ICLR 2022. 
[3] Are labels always necessary for classifier accuracy evaluation? CVPR 2021
[4] Towards Out-Of-Distribution Generalization: A Survey. Arxiv

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the out-of-distribution (OOD) generalization of adversarial robustness, when there is a shift on either the test data or threat model of the adversarial perturbation. The authors built a benchmark for such evaluation and conducted a comprehensive evaluation of many models. The paper shows a linear trend between the in-distribution (ID) performance and OOD performance on many models under adversarial attacks, but there are also models showing stronger robustness beyond the linear prediction in Section 5.

### Strengths
* The paper presents a benchmark (OODRobustBench) for evaluating the adversarial robustness under distribution shifts (either a shift on the test data or threat model of the adversarial perturbation), based on existing OOD test sets and variants of adversarial perturbations. 
* The paper conducted a comprehensive evaluation on many models. 
* The paper showed a linear trend between the ID adversarial robustness and the OOD adversarial robustness on most of the models, which is consistent with the linear trend observed in prior works (Taori et al., 2020, Miller et al., 2021) on dataset shifts without adversarial attack.

### Weaknesses
* This work is a straightforward combination of existing evaluations with little new contribution or understanding:
  * For the evaluation with data shifts, compared to existing works on evaluation with OOD datasets, this work simply adds existing adversarial attacks. Methods and conclusions are almost the same as previous works (Taori et al., 2020, Miller et al., 2021) on the linear trend.
  * For the evaluation on threat shifts, this work is almost the same as existing works mentioned in the "robustness against unforeseen adversarial threat models" paragraph in Section 2 but only adds more existing models. 

* Some discussions on the experiments are not very accurate:
  * "Surprisingly, VR also clearly boosts effective robustness under dataset shift
even though not designed for dealing with these shifts" and "Advanced model architecture significantly boosts robustness and effective robustness under both types of shift over the classical architecture": I don't agree these "significantly boost" the effective robustness. The gains are only around 1%~2%, which are not larger than the normal variations between different models in the linear fit (Figure 1).
  * "Training with extra data boosts both robustness and effective robustness for both dataset and threat shifts compared to training schemes without extra data (see Fig. 6a). The improved effective robustness suggests that this technique induces extra OOD generalizability." It is already known that altering the training data can interfere with the traditional effective robustness evaluation rather than truly improve effective robustness (Shi et al., 2023).
  * In Section 5, the authors rename the existing effective robustness from previous works (which have been widely adopted) into "effective accuracy" while redefine "effective robustness" to be the effective robustness under adversarial attacks. This is confusing. I would suggest the authors keep the original definition for effective robustness but give a new name for the particular effective robustness in this work (e.g, adversarial effective robustness). 

Shi, Z., Carlini, N., Balashankar, A., Schmidt, L., Hsieh, C. J., Beutel, A., & Qin, Y. (2023). Effective Robustness against Natural Distribution Shifts for Models with Different Training Data. arXiv preprint arXiv:2302.01381.

### Questions
* How are the results and conclusions in this paper fundamentally different from those in existing works? (See the weaknesses above.) What are the new implications of this work, beyond those already known in existing works (vulnerability against distribution shifts, linear trend, etc.)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper studies the adversarial robustness of image classifiers in presence of out-of-distribution (OOD) tasks. Given a model which has been (adversarially) trained on a specific dataset to be robust to a chosen attack (in-distribution task), the paper suggests to test how its robustness behaves when using either images from a different distribution (OOD dataset) or a different type of attack (OOD threat model): this provides an overview of generalization of robustness. Moreover, evaluating many existing and newly trained classifiers, the paper provides insights on which are the most relevant factors to achieve OOD robustness, which might be used to develop techniques for more robust models.

### Strengths
- Studying the generalization of adversarial robustness is a relevant topic (for example, as at test-time attackers are not limited to use the same attack seen during training), which has received only limited attention by prior works.

- The paper provides extensive evaluations on many classifiers spanning different datasets and (seen) threat models. This gives clear trends and insights about how to improve future robust models for better generalization.

### Weaknesses
- Similar analyses are already present in prior works, although on a (sometimes much) smaller scale, and then the results are not particularly surprising. For example, the robustness of CIFAR-10 models on distributions shifts (CIFAR-10.1, CINIC-10, CIFAR-10-C, which are also included in this work) was studied on the initial classifiers in RobustBench (see [Croce et al. (2021)](https://arxiv.org/abs/2010.09670)), showing a similar linear correlation with ID robustness. Moreover, [A, B] have also evaluated the robustness of adversarially trained models to unseen attacks.

- A central aspect of evaluating adversarial robustness is the attacks used to measure it. In the paper, this is described with sufficient details only in the appendix. In particular for the non $\ell_p$-threat models I think it would be important to discuss the strength (e.g. number of iterations) of the attacks used, since these are not widely explored in prior works.

[A] https://arxiv.org/abs/1908.08016  
[B] https://arxiv.org/abs/2105.12508

### Questions
- See the weaknesses mentioned above.

- Many non $\ell_p$ attacks have been proposed (see e.g. [A, B]). Is there a specific reason for the choice of those used in the paper?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The proposed approach combines the research direction of adversarial robustness and domain shift into a single benchmark: OODRobustBench. It measures how the adversarial robustness of networks trained on in-distribution data varies when evaluated on test data under distribution shift. It provides an evaluation over 706 robust models to draw insights into correlation of OOD and in-distribution (ID) robustness, upper limit on OOD robustness, and effect of training setup on OOD robustness.

### Strengths
This paper thoroughly explores the intersection of adversarial robustness and OOD generalization works in developing the OODRobustBench benchmark. It considers multiple ablations across ID and OOD robustness and performs 60.7K adversarial evaluations.The paper is also well written and very easy to parse.

### Weaknesses
While the proposed evaluation is rigorous, I believe that approach fall-short of being a standardized benchmark. It’s unclear what criterions are allowed for robust models. Would the benchmark include all adversarial robust approaches, such as preprocessing based defenses. If yes, how would the trend with robustness in such approaches correlate with OOD robustness? 

Is the specific trend in natural accuracy between IN and OOD data particular to robust models? Can authors provide some results/citation on how the correlation between IN-OOD accuracy correlated with IN-ODD accuracy on non-robust models?

### Questions
In figure 1.4, interestingly ID unseen robustness is higher than the seen robustness at lower robustness levels. This result is apparently counter-intuitive, as the trend quickly diminishes, at higher robustness levels. Can authors shed more light on this phenomenon.

Can authors also provide a concrete comparison on how the proposed benchmark is different from robustbench, in particular in-terms of benchmarking in-distribution robustness. 

To aggregate performance across different attacks in each group (OOD_d/OOD_t), shouldn’t harmonic mean be used to achieve better stability.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair
