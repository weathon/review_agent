# UBERT: Unsupervised adaptive early exits in BERT

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 3

## Abstract
Inference latency is an issue in pre-trained networks like BERT due to their large size. To overcome this, side branches are attached at the intermediary layers with provision for early inference instead of inference only at the last layer. This facilitates the early exit of 'easy' samples and requires only 'hard' samples to pass through all layers, thus reducing inference latency.  However, the hardness of the samples is unknown a priori. This leads to the question of how to exit so that the accuracy and latency are well balanced. Also, the optimal choice of parameters involved in deciding exits can depend on the sample domain and hence need to be adapted. We develop an online learning algorithm named UBERT to decide if a sample can exit early. The decisions are based on confidence in inference exceeding a threshold at each exit point, and the algorithm simultaneously learns the optimal thresholds for all the exits. UBERT learns the optimal threshold for the sample domain using confidence observed at the intermediary layers without requiring any ground truth labels. We perform extensive experiments on five datasets with one and two early exits. We compare the performance against the case with no early exits, i.e., all samples exit at the last layer. UBERT achieves a 10\%-53\% reduction in time with a drop in accuracy in the range of 0.3\% - 5.7\% with one early exit. For the case with two exits, the time reduction increases to 32\%-70\% with only a marginal drop in accuracy of 0.1\%-3.9\%. The anonymized source code is available at https://anonymous.4open.science/r/UBERT-F2DF/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the settings for early-exiting thresholds, which determines which samples will be output at each early exit during inference. This is an interesting topic in both NLP and CV. An algorithm is proposed to decide the confidence thresholds without groud-truth labels. Experiments on NLP tasks show that the method outperforms existing early-exiting approaches. However, there are some overclaims, and important references are missing. Moreover, the experiment results are not convincing enough (see weakness).

### Strengths
1. The studied topic is interesting, and the motivation is clearly explained;
2. The method is technically sound.

### Weaknesses
1. **Overclaim**. The authors claimed that existing methods usually use labeled datasets to determine the early-exiting thresholds, and the proposed method removes the need for ground-truth labels. This is not correct. Maybe the cited baselines in NLP do need ground-truth labels. But if the authors pay attention to the dynamic models in the CV field (see the second part, **Missing references**), they can find that the decision of confidence thresholds can purely rely on the confidence distribution on the training/validation set. Specifically, one can decide the ratio of samples exiting at different exits, and solve the threshold based on the confidence scores of each exit without touching the ground-truth labels. In summary, the main contribution claimed by the authors, may not hold.

2. **Missing references**. It is recommended that the authors compare their method with the aforementioned strategies in the CV field [1,2,3]. 

3. **Inconvincing experiments**. In Tab. 2, the proposed method is compared with other baselines at a **fixed** computational cost. However, the main advantage of dynamic early exiting is one can adjust the thresholds for different computational budgets (see the smooth curves in [1,2,3]. It is kindly suggested that the proposed method is compared with the "ratio -> threshold" pipeline in the CV field.

[1] Huang et al, Multi-Scale Dense Networks for Resource Efficient Image Classification. 

[2] Yang et al, Resolution Adaptive Networks for Efficient Inference.

[3] Han et al, Dynamic Perceiver for Efficient Visual Recognition.

### Questions
See weaknesses.

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
This paper proposes an online algorithm based on multi-armed bandits for adjusting the confidence threshold of BERT model with early exit gates. The objective is to reduce the model latency while maintaining high accuracy. The authors build on the existing rich literature of adding early exit classifiers on top of intermediate layers. Here, they focus on the challenge of selecting a good confidence threshold for deciding for each exit gate if to "exit" or not. Specifically the authors assume a domain shift of the test data and propose an online algorithm for adjusting the threshold according to the observed data.

A multi-armed bandit online algorithm is proposed for updating the exit threshold. The reward is designed as the difference in confidence between the last layer and exit layer subtracted by the increased cost, for instances that didn't exit. To normalize the two measures to a similar range, the cost $o$ is some value in [0,1].

First, an algorithm for single exit is described, then extension to multiple exits is presented. The experimental setting is focusing on OOD evaluation and examines the number of transformer layers computed vs. accuracy: training on one classification task and evaluating on another related dataset with similar classification task (repeated for 5 pairs).

### Strengths
Focusing on online adjustment of the early exit threshold is novel and interesting. The proposed method is based on multi-armed bandit and provides  an upper bound on the regret. Detailed algorithms are provided and experiments on 5 classification NLP datasets.

### Weaknesses
I value the novelty of the method and find it interesting. However, I see several weaknesses in the current paper:

1. While the proposed method is presented as general, the applicability beyond a single exit layer significantly increases the complexity and the solution space, possibly leading to long exploration stage before converging (the regret bound is only in expectation).
2. Also, many of the hyper-parameters feel pretty specific to the examined setting and justified in the paper with hand-wavy statements (e.g. "strategically positioned", "due to overthinking similar to overfitting during training" etc.), or with references to the appendix that don't fully explain them. This limits the generalizability of the solution.
3. The value of the cost parameter $o$ that is given to the end-user as a handle for controlling the desired cost is a uninterpretable value between [0,1]. Therefore, at the end of the day it feels like the user will still need to have some further calibration for tuning the value of $o$ to match whatever practical cost they can afford in their own measure and units.
4. While I see novelty and value in online adjustments of the threshold. The unsupervised novelty is less clear: see for example [1, 2, 3]. [1] and [3] seem to work with unlabeled data, and [3] seem to focus on threshold calibration which might be good to compare against.
5.  The experiments feel a bit underwhelming and unclear:
* The evaluation metric only measures the number of transformers layers and doesn't take into account any potential overhead of the exits and the calibration (and the use of "Time" as the column heading) is confusing.
* Since the method focuses on online setup, it would be interesting to see the patterns over time.
* The baselines model are not described well (for example, unclear what is the difference between ElasticBERT and DeeBERT).
* It is unclear how come the UBERT models could be better than the baselines in both accuracy and cost? If the backbone model is identical and roughly monotonic (as assumed throughout the paper), then the threshold should only control the tradeoff between the two but cannot improve on both?...

[1] https://aclanthology.org/2020.acl-main.537/

[2] https://aclanthology.org/2020.acl-main.593/

[3] https://aclanthology.org/2021.emnlp-main.406/

### Questions
see points in weakness section above. Also:
1. In eq.1 : are $C_p$ and $C_l$ always computed by max over softmax? the argmax can be different between the layer $p$ and $l$, making the use of the delta as reward less convincing.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
Inference latency is a key issue in any pre-trained large language models like BERT. Typically, side branches are attached at the intermediate layers with provision of early exit to minimize the inference time. This paper proposes an online learning algorithm, dubbed as, "UBERT" to decide if a sample can exit early through an intermediate branch.

### Strengths
1. Paper is well-written and the problem setup is mostly clear.

### Weaknesses
I am not an expert in this domain. However, I have few concerns.
1. Is it necessary to formulate the problem as multi-armed bandit setup? As RL usually resource hungry algorithms and they can take huge time to optimize.

### Questions
Refer to weaknesses section.

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents an early exit method for BERT. The authors use MAB to adaptively find threshold value in an online manner. The evaluation is conducted on 5 classification tasks.

### Strengths
1. The ablation seems good.
2. The concept of finding adaptive threshold seems interesting.
3. The proposed method outperforms the compared baselines.

### Weaknesses
1. Evaluation is restricted.
2. The authors claim are often missing supporting literature or validating experiments. 
3. Authors exaggerates the efficacy of their method, even though they cannot also resolve the problem.  
4. The paper is missing the essential information, which prevents the paper to stand alone.
5. Missing details.

### Questions
1-1 Considering recent SOTA NLP models/LLMs, BERT variants are quite older and their size is much smaller, which can already run smoothly with restricted resources under the current HW. Therefore, the only evaluation with BERT-variant in this work makes me doubt about the motivation of this work. The authors should've considered larger language models and showed the generalizability of the proposed work. If the scope is only limited to BERT, its applicability/practicability is questionable as BERT is barely used in real applications.   

1-2 One of closely related work is F-PABEE that outperforms all existing methods in the literature. I highly recommend to add its result for comparison and analysis. 

1-3 Other previous papers like PABEE and F-PABEE, CoLA MNLI MRPC QNLI QQP RTE SST-2 (STS-B) are standardized for comparison. However, authors does not follow. 

2-1 "Even though it is anticipated that the final layer of the NN can have better accuracy than the intermediate layer": any support literature or experiments? This is the fundamental assumption, which is not validated throughout the paper. 

2-2 "The threshold is often determined using a labeled dataset during training and serves as a crucial reference point for decision-making during inference.": in what cases or papers?

2-3 "The optimal threshold value depends on the distribution of confidence levels at the attached exit, which can vary depending on the data distribution. ": any proofs?

2-4 " UEE-UCB Hanawal et al. (2022) leverage the MAB framework to learn the optimal exit in EENNs": How is their use of MAB different from this work? 

2-5 In Sec.2 in lines starting "LEE Ju et al. (2021b), DEE...", these works seem quite close to the proposed method. It would be recommended to have details comparison against the proposed work.

3-1 As authors noted, using a fixed threshold may yield suboptimal results. However, the proposed method finds the threshold based on the observations of previous samples, which cannot be also free from the same issue. So it seems "Consequently, UBERT sets itself apart" is not an appropriate claim.

3-2 The term online learning is quite confusing in this work. As in Sec. 6, the pretrained model is finetuned and this finetuned model is used to adaptively find threshold in an online manner. The adaptive finding is of course online, but this term (online learning/algorithm) is exaggerated and providing confusion. 

3-3 The authors keep using the term "optimal threshold" throughout the paper. However, it is optimal only if the given specific setting in Algorithm 1 and 2 is used. With naive changing the cost such as adding a value or scaling, it varies. It is hard to conclude that the proposed method optimally trade-off between latency and accuracy. Is there a curve, for example, UBERT-2 shows best accuracy with -59.5 time while the accuracy reduces with -58 or -60 time? If not, the use of this term seems not proper in this context. 

4-1 The detailed description of ElasticBERT and MAB is not provided. 

5-1 "Though confidence and latency are in different units, we add them after using a conversion factor.": I cannot find details of this process in the paper.

5-2 What should I do if I want to improve the latency while sacrificing the performance or vice versa? The new model should be trained again? If so, although the authors adaptively find the threshold with reward, it is hard for me think it as a benefit compared to other exiting methods. Other work simply change the number and run the model to adjust latency-performance.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
