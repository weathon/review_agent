# A margin-based replacement for cross-entropy loss

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Cross-entropy (CE) loss is the de-facto standard for training deep neural networks (DNNs) to perform classification. Here, we propose an alternative loss, high error margin (HEM), that is more effective than CE across a range of image-based tasks: unknown class rejection, adversarial robustness, learning with imbalanced data, continual learning, and semantic segmentation (a pixel-wise classification task). HEM loss is evaluated extensively using a wide range of DNN architectures and benchmark datasets. Despite all the experimental settings, such as the training hyper-parameters, being chosen for CE loss, HEM is inferior to CE only in terms of clean and corrupt image classification with balanced training data, and this difference is small. We also compare HEM to specialised losses that have previously been proposed to improve performance for specific vision tasks. LogitNorm, a loss achieving state-of-the-art performance on unknown class rejection, produces similar performance to HEM for this task, but is much poorer for continual learning and semantic segmentation. Logit-adjusted loss, designed for imbalanced data, has superior results to HEM for that task, but performs worse on unknown class rejection and semantic segmentation. DICE, a popular loss for semantic segmentation, is inferior to HEM loss on all tasks, including semantic segmentation. Thus, HEM often out-performs specialised losses, and in contrast to them, is a general-purpose replacement for CE loss.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a novel margin-based loss function, High Error Margin (HEM), which selectively averages over the examples in a batch that contribute the most to the loss.
This approach is intended to mitigate the issue of diminishing gradients commonly observed in margin-based losses toward the later stages of training. The paper presents empirical evidence demonstrating the advantages of the proposed method across several benchmarks.

### Strengths
The HEM loss is tested across multiple datasets, learning tasks, and architectures. Statistical significance is provided, showcasing the effectiveness.

Hyperparameters are selected in a clear and reproducible manner and optimized for the cross-entropy baseline rather than for HEM.
I value this highly, as overselling the observed effects with suitably chosen hyperparameters happens regularly.

### Weaknesses
The paper lacks a rigorous theoretical foundation for the claims that HEM mitigates catastrophic forgetting, yields more stable gradients than the Margin Maximization (MM) loss, or performs better on imbalanced datasets. The supporting arguments are largely heuristic, with no analytical results or relevant literature references provided to substantiate them.

### Questions
I have no questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Cross-entropy (CE) loss remains the standard for training deep neural networks in classification tasks. The authors introduce High Error Margin (HEM) loss, a general-purpose alternative that can outperform CE across diverse scenarios, including unknown class rejection, adversarial robustness, imbalanced data learning, continual learning, and semantic segmentation. Extensive evaluations on various architectures and benchmarks show that HEM performs on par with CE for balanced clean and corrupted image classification, while substantially improving performance in other settings. Compared to specialized task-specific losses - such as LogitNorm for open-set recognition, logit-adjusted loss for imbalance handling, and DICE for segmentation - HEM achieves more robust and general improvements, establishing itself as a versatile replacement for CE loss.

### Strengths
- The authors address an interesting topic.
- I like that, in addition to the standard classification, other aspects such as unbalanced data, adversarial robustness, continual learning, and semantic segmentation are also considered. 
- The small toy example (Table 1) is useful for illustrating the differences between or problems with losses.
- The paper is easy to read and understand. In particular, I think section 2 is appropriate as motivation. I find Appendix A useful for explaining the various losses.
- The authors' idea doesn't have a lot of novelty, but it is intuitive and also shows good results.
- The distinction between HEM and HEM- makes a comparison with other loss functions fair.
- In semantic segmentation, the results are really good, with improvements almost every time.
- The presentation of the experiments in the main paper is pleasant.
- All details for reproduction are provided and the code should be made available.
- There are numerous experiments. The choice of datasets and evaluation metrics is appropriate.

### Weaknesses
- In Figure 1, I would add information about how many different datasets and networks were averaged.
- The main paper states that 18 networks and 18 datasets are used. I find this somewhat misleading, as 5 datasets and 3 networks are initially used for the classification experiments (and another 4 for semantic segmentation), and the other datasets only represent extensions for the respective task.
- For the attack experiments, the MSP or MLS is used for the thresholding. Entropy would also be interesting at this point, as it often performs better.
- The models used for classification are all rather dated. I understand why Restnet is included for comparison purposes, as it is simply standard practice, but I would also test newer models.
- The comparison takes place in the normal classification setting as well as in semantic segmentation, where HEM is only successful in the latter. I would rather compare it to other tasks such as OOD detection and robustness against attacks, by replacing the CE loss with HEM in existing methods used for these tasks. It is well known that normally trained networks are not very robust against all kinds of threats, and even though HEM brings slight improvements here, the results are not comparable to those achieved with specially trained/created methods.

### Questions
- The direct comparison is the classification task, and HEM is always worse than CE at standard setting (even with common corruptions). For unknown classes (OOD), attacks, and imbalanced data, there are other methods that are specifically desgined/trained for this purpose (i.e., it is not expected that using only CE loss will obtain good results). So here, the comparison is only against the simplest/most logical baseline, but not against other methods that were actually created for this purpose. Shouldn't the comparison be made against other methods here?
- "As a result training with HEM is faster than training with CE (for example, it reduces training time by approximately 10% for a ResNet18 trained for 200 epochs on TinyImageNet)" Has this behavior also been observed in other experiments, particularly in semantic segmentation? If so, I would highlight that in the paper. 
- "It can be seen that HEM benefits most from the drop in learning rate near the end of training and that there are large fluctuations in the loss, and the other recorded metrics, before the learning rate drop at 100 epochs." That's an important point, so if you stop training too early, you might not see any improvement. Was this observed in all experiments? Any suggestions on how to address this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a loss function called “High Error Margin (HEM)” that should outperform cross-entropy (CE) in various classification tasks and metrics. It’s a margin-based loss, computed only over class errors higher than the mean, calculated for each data point. This makes the loss focus only on high-error logits. Moreover, the marginal loss seems beneficial because it assigns zero loss to already well-classified instances - addressing the main problem of CE, which continues penalizing samples that are already correctly classified, often leading to overfitting. The value proposition of this paper is clear but too broad - authors are trying to cover and improve too many classification applications (imbalanced datasets, continual learning, OOD detection, corrupted data accuracy, segmentation, and more). It’s impossible to really understand whether the proposed loss works without reading the appendices, which I see as a major flaw. The pitfalls of CE are well known, and while this HEM loss generally outperforms CE, in presented results, there’s always another loss that performs better than HEM in each application. Moreover, authors only tested a few losses considering the many tasks and metrics they aim to address.

### Strengths
The paper proposes the High Error Margin (HEM) loss, a margin-based alternative to cross- entropy (CE) for classification tasks. The motivation is clear: CE has well-documented limitations, including non-zero penalties for correctly classified samples and a tendency toward overconfident predictions on unseen data, leading to mis-calibration, poor robustness to out-of-distribution detection, and catastrophic forgetting.

HEM addresses these issues by adaptively averaging high-error logits (those whose errors exceed the mean across classes), thereby emphasizing misclassified or uncertain examples. The idea is in- tuitively appealing and well-motivated, as it seeks a general-purpose loss function that can perform more robustly than CE across many applications. The empirical evaluation is particularly comprehensive - covering diverse architectures, multiple datasets, and several application domains. The paper’s presentation is generally clear, and the problem statement is timely and relevant to ongoing research in loss function design and optimization.

### Weaknesses
The claim of general superiority is somewhat too broad. Figures 1 and 2 report a large number of experiments, but the main text provides insufficient explanation of these results and their configurations, with many key details relegated to the appendices.

Given the wide range of applications tested, comparing only three alternative losses (apart from CE and MM losses) may be insufficient.

For imbalanced data, in the literature CE is often used in combination with other techniques—such as random oversampling or sample reweighting—which are not considered here.

As also noted in the conclusion (lines 461–474), for each application there exists at least one specialized loss that outperforms HEM, except for semantic segmentation. However, for this specific task, the authors did not evaluate Focal Loss, another modification of CE that directly addresses the over-penalization of already well-classified samples.

Is it useful to propose a loss function that performs generally well, but falls short compared to specialized losses in each specific application?

### Questions
1. Please consider moving Appendix A.5 into the main body. It repeats Equation (1) as Equation (8), plus the MM formulation, which is only the average of the errors.
2. In the semantic segmentation experiments, Focal Loss should be included for comparison, since it is also a modification of CE that mitigates the issue of over-penalizing already well-classified samples.

### Soundness
3

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
3

### Summary
This paper proposes High Error Margin (HEM) loss as a general-purpose alternative to cross-entropy (CE) for training deep neural networks. Through extensive experiments across multiple architectures and datasets, the authors claim HEM outperforms CE on five diverse tasks: unknown class rejection, adversarial robustness, imbalanced learning, continual learning, and semantic segmentation. HEM only underperforms CE on clean/balanced classification by a small margin. Notably, HEM also outperforms task-specific specialized losses (LogitNorm, Logit-adjusted, DICE) on most tasks, positioning it as a universal replacement for CE loss.

### Strengths
- The paper is well written. 
- The proposed approach is technically sound. 
- The empirical experiment conducted span a wide range of datasets. 
- The proposed method seem to work on a wide range of problems.

### Weaknesses
- The proposed method seems to lack some theoretical justification. Some theoretical analysis on the proposed loss function, and on why the proposed HEM loss is better than regular margin loss can further strengthen the paper. 
- The claim to replace CCE loss is somewhat aggressive to me. From figure 2, the proposed loss function still underperforms CCE loss in the clean-data scenarios pretty significantly. 
- Following up on the previous point, in order to claim HEM as a "replacement" for CCE loss, additional experiments are needed in my opinion. 
- Unlike CCE loss, the proposed loss function has a hyper-parameter that needs to be tuned.

### Questions
- Does the proposed loss function work in other scenarios like NLP, and with other tasks like language model pre-training, which also uses CCE?

### Soundness
2

### Presentation
3

### Contribution
3
