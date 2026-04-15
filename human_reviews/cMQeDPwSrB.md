# Memorization Through the Lens of Curvature of Loss Function Around Samples

- Decision: Reject
- Scores: 5, 5, 5, 5, 6

## Abstract
Deep neural networks are over-parameterized and easily overfit the datasets they train on. In the extreme case, it has been shown that these networks can memorize a training set with fully randomized labels. We propose using the curvature of loss function around each training sample, averaged over training epochs, as a measure of memorization of the sample. We use this metric to study the generalization versus memorization properties of different samples in popular image datasets and show that it captures memorization statistics well, both qualitatively and quantitatively. We first show that the high curvature samples visually correspond to long-tailed, mislabeled, or conflicting samples, those that are most likely to be memorized. This analysis helps us find, to the best of our knowledge, a novel failure mode on the CIFAR100 and ImageNet datasets: that of duplicated images with differing labels. Quantitatively, we corroborate the validity of our scores via two methods. First, we validate our scores against an independent and comprehensively calculated baseline, by showing high cosine similarity with the memorization scores released by Feldman and Zhang (2020).  Second, we inject corrupted samples which are memorized by the network, and show that these are learned with high curvature. To this end, we synthetically mislabel a random subset of the dataset. We overfit a network to it and show that sorting by curvature yields high AUROC values for identifying the corrupted samples. An added advantage of our method is that it is scalable, as it requires training only a single network as opposed to the thousands trained by the baseline, while capturing the aforementioned failure mode that the baseline fails to identify.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
- The paper proposes using the curvature of loss function around each training sample, averaged over training epochs, as a scalable method, to measure of memorization of a given sample.  
- The metric is validated both in the setting of synthetic label noise, as well as against an independent and
comprehensively calculated baseline of memorization scores released by Feldman & Zhang (2020)

### Strengths
- Reduces the time to calculate memorization scores considerably as compared to the baseline of Feldman & Zhang(though a head to head comparison doesn't make sense as they can also calculate influence scores in their technique).
- Decently high cosine similarity with FZ baseline and high AUROC values for identifying the noisy label samples.

### Weaknesses
- Scalability: To capture the curvature of training samples in every epoch, additional backward passes are required(n=10 for these datasets), which requires an additional 10X the training time to capture the metric proposed. It can be partially circumvented by calculating the curvature per sample every few epochs, but can still continue to be computationally expensive.
- Limited to memorization: The baseline by Feldman & Zhang calculates both the memorization scores for the training examples, and influence scores for test examples. There is no natural extension of curvature as a metric to influence scores.
- Baseline: No comparison with any baseline is presented. Simple baselines like learning time, second split forgetting time etc. should be compared to.
- Hyperparameters : No discussion related to how the values of h and n affect the results obtained

### Questions
- Results of comparison with other simple baselines ?

- How does the value of h and n in the curvature calculation affect the results ?

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper propose to use a curvature inspired metric (trace of the squared Hessian of the per-example loss with respect to a particular input) to identify memorized examples. The paper show that this metric is consistent with a previous memorization metric (FZ scores) while being less computationally expensive. Moreover, the proposed metric, when trained without weight decay, identify a failure pattern of near identical images with different labels that is not extensively studied before. The metric is also shown to outperform several baselines in corrupted label detection and provide insights to learning dynamics at the presence of label noises from the perspective of curvatures.

### Strengths
This paper propose a simple curvature inspired metric with an approximation procedure that can be computed efficiently. This metric is shown to be effective at identifying examples memorized by neural networks. The idea is simple and the presentation is easy to follow. The experiment setup are clear and the results are sound.

### Weaknesses
1. There are a large body of work on data valuation for deep neural networks that studies similar questions. Those should be discussed in the related work and at least representative methods should be compared in the experiment.

2. This paper could benefit from a stronger demonstration of the utility of the proposed metric, beyond the simple corrupted label prediction experiment. A few hypothetical directions could be

    1. Could the curvature be extended to compute some kind of influence metrics from training to test examples like the FZ scores or other data valuation metrics?

    2. Theoretical analysis of how or why is the proposed metric connected to memorization and generalization.

    3. New algorithms designs, such as curriculum learning based on per-example curvatures, or maybe regularizers motivated by the curvature metric.

3. Unlike other metrics that measures in the output, logits, or weight space, the proposed metric probably depend more on the geometry of the inputs. Therefore, I think it is very valuable if the paper could present studies with different input domains.

### Questions
1. Do you observe the same behavior if the input-vs-logit space analysis in Fig. 7 is performed on a real dataset instead of a synthetic one?

2. It is described in the appendix how the two hyperparameters h and n are tuned. Can you show some hyperparameter tuning curves? In particular, I'm interested in not only what values are chosen, but also how robust the proposed metric is to hyperparameter changes.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies memorization in modern neural networks and aims to design a tractable approach to quantify the memorization of different training examples. Specifically, this paper considers loss curvature (averaged across different training epochs) with respect to a given training example as a memorization measure for that particular example. The paper leverages a finite step approximation of the per-example loss Hessian and explores the proposed curvature-based memorization score on MNIST, Fashion-MNIST, CIFAR-10/100, and ImageNet, with ResNet18 as the underlying model architecture. Visualizing the examples with high curvature-based memorization scores, the authors observed that such examples correspond to long-tailed, mislabeled, or conflicting examples (e.g., duplicate examples with different labels). The paper also compared the curvature-based memorization score with Feldman and Zhang’s stability-based memorization score and found a high correlation between the two scores. For a synthetic setup, where label noise is introduced to a fraction of training examples, the paper shows that noisy examples belong to the set of examples that receive the highest curvature-based memorization scores. Finally, the paper studies the average (over the entire dataset) curvature of per-example losses as training evolves. The paper shows that, during the overfitting phase of training, this average curvature shows a decreasing trend while the validation loss keeps increasing. The paper provides an explanation for this seemingly contradictory observation by arguing the decrease in the average curvature corresponds to the phase where training examples are already classified correctly and the model is increasing the margin of these examples. Thus, the paper highlights the importance of averaging the per-example loss curvature across entire training epochs to obtain an informative memorization score.

### Strengths
1) The paper studies an important problem of quantitatively characterizing the memorization behavior of various training examples. The paper proposes the per-example loss curvature (averaged across different training epochs) as such a quantitative measure.
2) The paper successfully shows that the proposed memorization score can consistently identify a range of overfitting (“memorization”)-prone examples across multiple image classification benchmarks. This suggests that the proposed score can be utilized for cleaning/denoising a given training dataset.
3) The computational cost of evaluating the proposed memorization score is smaller than some of the other widely accepted memorization scores in the literature, such as Feldman & Zhang’s score.
4) The paper carefully studies the behavior of per-example loss curvature as training progresses and highlights the importance of averaging the per-example loss curvature across multiple training epochs.

### Weaknesses
1) The main weakness of the paper is that it does not provide a comprehensive discussion on what it means to say that an example is being memorized. For example, the work of Feldman & Zhang calls those points to be memorized where the model only performs well when those points are present in the training set. What is the precise notion of memorization that this paper aims to capture?
2) It appears that the paper claims that their curvature-based memorization score aligns well with Feldman & Zhang’s memorization score as the two scores have a high correlation (cosine similarity). Such aggregate-level correlations can be misleading, especially while studying memorization behaviors which are very much tied to individual examples (e.g., see Section 3.5 in https://arxiv.org/pdf/2310.05337.pdf).
3) Most of the experiments in the paper are tied to a single model. It would be interesting to see how the proposed curvature-based memorization score behaves as one changes model architecture and/or size. Are key takeaways from this paper robust to such variations?
4) The paper claims that their proposed memorization score is significantly cheaper to compute compared to existing scores like Feldman & Zhang’s memorization score (see a question on this below). However, it appears that the *efficient* loss curvature calculation follows from the prior work, limiting the technical novelty of the contributions.
5) There is significant scope for improvement in the quality of the presentation. For instance

* In Section 3.1, both $d$ and $D$ are used to represent the input dimension. 
* In the line after Eq. (2) $v_i$ represents the $i$-th coordinate of the random Rademacher vector $v$, while in Eq (3), $v_i$ represents $i$ random *vector*. Also, the line after Eq. (3) refers to $v$ as a Rademacher *variable* instead of *vector* (a similar issue in Section 3.2). 
* In Eq. (4), please consider using $\approx$ instead of $=$. 
* In Eq. (6), please make the dependence of the square norm on indices $i$ (via $v_i$) and $t$ (via $W_t$) explicit.
* In Section 3.2, O(n) → O(nT) forward and backward passes?

Besides the issues mentioned above, multiple sentences in the paper require paraphrasing as their meaning is not entirely clear, and thorough proofreading is required to eliminate various typos.

### Questions
1) How does a memorizing network behave on duplicate examples with distinct labels? Which label is preferred by the model? Does the model encounter a decreasing loss curvature trend on such examples as training progresses? 
2) In Section 4.2, the authors state ``... These scores are likely to be independent of spurious correlations to curvature that other methods such as confidence of prediction might have, and hence serve as a good baseline.`` Could the authors elaborate on this statement on why Feldman & Zhang’s score is likely to be independent of spurious correlations (while other methods can potentially exhibit such correlations)?
3) In Section 4.2, the authors claim that computing Feldman & Zhang (FZ) score is ``~3 x more computationally expensive``. Why is it **only** 3x more expensive when computing FZ scores requires training a large number of models?
4) In Section 4.3, the authors mention ``...We recommend that curvature analysis should be used in conjunction with other checks, for a holistic view of dataset integrity. ``. Could the authors clarify which other checks they refer to?

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
This paper proposes to use the per-instance curvature metric to calculate the memorization of the sample by a neural network. The curvature is approximated by the trace of the hessian matrix, which is approximated by using Hutchinson’s method. The curvature scores obtained from the proposed method, are shown to correlate well with the scores obtained by Zhang and Feldman. Furthermore, the scores have been shown to be able to detect the samples which have presence of label noise in a synthetic setup.

### Strengths
The paper is clearly written and is easy to read.

The proposed curvature score is easier to compute in comparison with Zhang and Feldman, and can be practically used.

### Weaknesses
Novelty: The proposed method has a high degree of overlap with Garg and Roy 2023 [R2], where the similar per-instance score is used for determining the curvature. Further, the score in their paper is able to find long-tailed and rare samples, which is the application demonstrated in this paper. It would be great if authors can please clarify the novelty of this work in comparison with Garg and Roy 2023 [R2].

Missing Baselines: The authors don’t compare the proposed method to the Maini et al. 2022 [R1] for identification of the noisy labelled samples, however the setup followed is similar to that of the Maini et al. 2022 paper. I request the authors to please provide the comparison or the reason for the omission of the comparison.

Missing Concrete Application: The section on curvature dynamics for training provides interesting insights. However I couldn’t find any specific experiments which demonstrate its practical applicability.

[R1] Maini, Pratyush, et al. "Characterizing datapoints via second-split forgetting." Advances in Neural Information Processing Systems 35 (2022): 30044-30057.

[R2] Garg, Isha, and Kaushik Roy. "Samples With Low Loss Curvature Improve Data Efficiency." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
I am curious if can the above method be used to identify the samples which have inconsistent captions for the Vision Language Based Methods?

Further, it’s claimed in the paper that FZ scores can’t be used to find samples that are duplicate images with different samples. Can you please provide a concrete reason for this?

Is it possible to provide any theoretical results regarding the correctness of the curvature scores for finding the noisy labeled samples etc?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a metric to identify memorized points. The proposed method utilizes the average curvature of the loss function with respect to the input and argues that memorized points have a higher curvature score. They demonstrate this by adding label noise to the training dataset and training until overfitting, showing a higher curvature score for these points during training. They also illustrate quite a high cosine similarity between their score and the previous method by Feldman & Zhang (2020), which they call the FZ score in the paper. However, the advantage of their method is that they don't have to train as many models as required by the FZ method, demonstrating computational benefits.

### Strengths
The paper has an acceptable visualization of the experiments; however, more work can be done to make them more accurate and understandable. They have tried to justify their observations and conclusions by designing experiments. Different kinds of datasets are used in this paper, each of which helps to better understand the paper.

In general, I am leaning towards accepting the paper, and I am even open to reconsider my initial score upon improving the quality of the paper regarding the presentation and clarification of questions and points mentioned in the following.

### Weaknesses
1. Toy example in section 4.4: The number of training points is far less than in the test set, which is not a similar case to your image datasets. The training data does not have enough points to accurately represent the underlying distribution. Additionally, adding noise to it makes it more challenging for the model to learn the decision boundary correctly.
2. Section 4.4: It is a lengthy section and difficult to follow the main points of it; the presentation format can be improved.
3. Figure 4: It would be better to display mislabeled examples side by side.
4. Figure 5: The labels on the axes of the figure are not clear, and I can't interpret the results. Having a distribution over the corrupted samples would be more informative.
5. Tables: MEAN+STD missing in the tables. 
6. Appendix F: It does not explain any correlation between memorized sampel by FZ and your method.

### Questions
1. Introduction (Regarding “weakly labeled or have noisy annotations”): Can you apply your method in unsupervised settings?
3. Table 2: Did you use weight decay for other methods as well? Can you show the result of FZ method?
4. Section 4.3: Can you elaborate on why ROC curves are more reliable?
5. How is the threshold for the curvature of loss selected?
6. Does averaging the curvature over epochs remove the specific epoch signals? Isn't it better to study the difference between curvature evolution during time than taking an average?
7. Figure 8: Which type of sample triggers the peak in curvature, and do memorized points contribute more to that?
8. Figure 8: Why is the curvature of test points much higher than memorized points in CIFAR-100? And why don't we see that in ImageNet?
9. End of page 8: What does the sensitivity of test samples to perturbation mean?
10. Have any studies explored using the curvature of samples for inference attacks?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
