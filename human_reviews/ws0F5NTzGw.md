# AdapTable: Test-Time Adaptation for Tabular Data via Shift-Aware Uncertainty Calibrator and Label Distribution Handler

- Avg Score: 4.50
- Decision: Reject
- Scores: 5, 5, 3, 5

## Abstract
In real-world applications, tabular data often suffer from distribution shifts due to their widespread and abundant nature, leading to a significant impact on the performance of machine learning models during testing. However, addressing these shifts in the tabular domain has been relatively underexplored due to unique challenges such as varying attributes and dataset sizes, as well as limitations representation learning capabilities of deep learning models for tabular data. Particularly, with the recent promising paradigm of test-time adaptation (TTA), where we adapt the off-the-shelf model to the unlabeled target domain during the inference phase without accessing the source domain, we observe that directly adopting commonly used TTA methods from other domains often leads to model collapse. We systematically explore challenges in tabular data test-time adaptation, including skewed entropy, complex latent space decision boundaries, confidence calibration issues with both overconfident and under-confident, and model bias towards source label distributions along with class imbalances. Based on these insights, we introduce AdapTable, a novel tabular test-time adaptation method that directly modifies output probabilities by estimating target label distributions and adjusting initial probabilities based on calibrated uncertainty.
Extensive experiments on both real-world distribution shifts and synthetic corruptions demonstrate the adaptation efficacy of the proposed method using unlabeled test data alone.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes AdapTable to address problems about tabular-specific test-time adaption. AdapTable uses a shift-aware uncertainty calibration module to correct the poor confidence calibration, and uses a label distribution handler to adjust the output distribution. Experimental results show the effectiveness of AdapTable.

### Strengths
This paper comprehensively investigate the problems about tabular-specific test-time adaption, and propose a new method called AdapTable to address these problems.

### Weaknesses
This paper does not investigate and justify (semantically and experimentally) the advantages of the proposed post-hoc output calibration method compared with traditional model calibration methods such as isotonic regression calibration and Platt calibration. Besides, it is not clear how the target label distribution output by the calibration model is guaranteed to be correct.

### Questions
1. What are the advantages of the calibration method proposed in this paper compared to the traditional calibration method such as isotonic regression calibration and Platt calibration?
2. How is the target label distribution ensured (or ensured with a certain probability) to be correct?

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Tabular data in real-world applications often face distribution shifts, impacting machine learning model performance during testing. Addressing these shifts in tabular data is challenging due to varying attributes and dataset sizes, and limitations of deep learning models for tabular data. To tackle these challenges, the AdapTable method is introduced, which estimates target label distributions and adjusts initial probabilities based on calibrated uncertainty, demonstrating its effectiveness in experiments with real-world and synthetic data shifts using unlabeled test data alone.

### Strengths
1. TTT on tabular data has its unique challenge, and authors well clarify this point in Sec.2. I really appreciate such clarification.
2. Experimental results are extensive and convincing.

### Weaknesses
1. Why SHIFT-AWARE UNCERTAINTY CALIBRATOR and  LABEL DISTRIBUTION HANDLER are combined into each other? I mainly  concern on whether your solution looks like a A+B combination. If you can clarify this point, I will be pleased to raise my score.

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies an interesting problem of test-time adaptation for tabular data. This problem is meaningful since tabular data suffers from distribution shift problems while lacking effectiveness in solving them. The authors propose a model-independent test-time adaptation method, which estimates the temperature for each sample during testing and modifies the label distribution of outputs. The experiments show a significant performance improvement on three datasets in the TableShift benchmarks.

### Strengths
1. This paper studies an interesting problem, namely test-time adaptation for tabular data.
2. The proposed method is suitable for addressing label distribution shifts in tabular datasets.

### Weaknesses
1. The experiments in this paper are insufficient. There are 15 datasets in the TableShift benchmark; however, only three of them are considered in the experiments. This makes the results of this paper unconvincing.
2. The performance improvement on ANES and Diabetes ReadMission is relatively weak in the Supervised setting (which is the setting with the overall best performance). This makes the proposed method weak.
3. The hyper-parameter of the proposed method lacks thoughtful discussion. For example, the alpha in the label distribution handler determines how quickly the model can adapt to the latest label distribution, which may seriously affect the performance.
4. This method calibrates the logits with sample-wise temperature and estimated label distribution, which can only handle the label distribution shift problems rather than covariate shift problems.
5. Minor issue: The left sub-figures in Figures 2 and 5 contain black borders.

### Questions
1. The authors should explain why only three datasets are adopted in the experiments and why these three datasets are selected.
2. The running time of the proposed method should be reported for both training and evaluation.
3. Please refer to the questions in the weakness.

### Soundness
2 fair

### Presentation
3 good

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
This paper focuses on the test-time adaptation problem for the tabular data. Specifically, the authors discuss the challenges related to test-time adaptation for tabular data and propose a new method including two modules: a shift-aware uncertainty calibration module and a label distribution handler. Experimental results show the proposal can improve learning performance on various datasets.

### Strengths
1) This paper studies the problem of adaptation for tabular data. This problem is important yet under-studied.

2) The authors conduct a large number of experiments and the results show that the proposal can improve performance.

### Weaknesses
1) Although the proposal can improve the performance, it is mainly a combination of some existing techniques. There is neither good theoretical analysis nor much inspiration, thus, the novelty and contribution are limited. From my personal point of view, I don’t really appreciate papers that improve performance by integrating multiple tricks and existing methods.
2) For tabular data, the shift may exist in various perspectives, such as the distribution shift, the feature dimension (new features occur or old features are lost), and class space varies. These problems should discussed separately.
3) Will the introduction of GCN in the method lead to a higher computational complexity of the algorithm? This should be discussed theoretically or empirically.

### Questions
As discussed above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
