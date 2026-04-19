# Un-Mixing Test-Time Normalization Statistics: Combatting Label Temporal Correlation

- Decision: Accept (poster)
- Scores: 6, 6, 8, 6

## Abstract
Recent test-time adaptation methods heavily rely on nuanced adjustments of batch normalization (BN) parameters. However, one critical assumption often goes overlooked: that of independently and identically distributed (i.i.d.) test batches with respect to unknown labels.  This oversight leads to skewed BN statistics and undermines the reliability of the model under non-i.i.d. scenarios. To tackle this challenge, this paper presents a novel method termed '$\textbf{Un-Mix}$ing $\textbf{T}$est-Time $\textbf{N}$ormalization $\textbf{S}$tatistics' (UnMix-TNS). Our method re-calibrates the statistics for each instance within a test batch by $\textit{mixing}$ it with multiple distinct statistics components, thus inherently simulating the i.i.d. scenario. The core of this method hinges on a distinctive online $\textit{unmixing}$ procedure that continuously updates these statistics components by incorporating the most similar instances from new test batches. Remarkably generic in its design, UnMix-TNS seamlessly integrates with a wide range of leading test-time adaptation methods and pre-trained architectures equipped with BN layers. Empirical evaluations corroborate the robustness of UnMix-TNS under varied scenarios—ranging from single to continual and mixed domain shifts, particularly excelling with temporally correlated test data and corrupted non-i.i.d. real-world streams. This adaptability is maintained even with very small batch sizes or single instances. Our results highlight UnMix-TNS's capacity to markedly enhance stability and performance across various benchmarks. Our code is publicly available at https://github.com/devavratTomar/unmixtns.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel test-time normalization layer to tackle the temporally correlated, distributionally shifted problems within the context of test-time adaptation, and boost the performance of test-time adaptation tasks involving various distribution shifts. The experiments demonstrate the method achieved SOTA.

### Strengths
1. The logic of this paper is clear and the performance is excellent.
The author argues that the assumption of test batches conforming to an i.i.d distribution can produce unstable and unreliable adaptations. This paper works on it and offers a reasonable approach.  
2. Extensive experiments on three benchmarks and solid ablation studies.
3. The paper is well organized and the idea is well presented.

### Weaknesses
After reading the paper, it is clear that the method delivers improvements on TTA tasks. However, there are some questions remain unanswered:
0. Some papers also explored the test-time non-iid setting, such as [a] and [b]. Compared with these papers, what's the contribution of the proposed setting and approach?
1. Why the hyper-parameter \alpha is set to be 0.5 in Equation (5) for implementation while initializing UNMIX-TNS components?
2. In Section 2.2.3, the hyper-parameter \lambda is not explained clearly.
3. While Tables 1 and 2 include 11 methods for comparison, the experiments on the video dataset (Table 3) only include seven methods.
4. Does the UNMIX-TNS update the same parameters with other TTA methods? And have the affine parameters in BN layers been changed within the optimization? Does the method require only single forward inference or multiple forward passes?
5. The authors mentioned that “only components that closely align with the instance-wise statistics undergo updates”, but equations (12) and (13) update all K BN statistics components with different assignment probabilities. 


[a] Robust Test-Time Adaptation in Dynamic Scenarios, ICCV2023
[b] Robust continual test-time adaptation: Instance-aware BN and prediction-balanced memory. NeurIPS2022

### Questions
Please refer to the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces UnMix-TNS, a new approach to address the challenges of test-time adaptation in neural networks under non-i.i.d. conditions, particularly focusing on data streams with temporal correlation. Traditional batch normalization (BN) techniques, fundamental in stabilizing neural network training, fall short under non-i.i.d. test scenarios, often encountered in real-world applications like autonomous driving. UnMix-TNS innovatively recalibrates instance-wise statistics within BN by mixing them with multiple unmixed components, simulating an i.i.d. environment. This method is adaptable to various state-of-the-art test-time adaptation strategies and seamlessly integrates with pre-trained architectures that include BN layers. The approach is empirically validated across diverse scenarios and shows promise in effectively handling data with temporal correlations. New datasets have also been introduced.

### Strengths
1. **Innovative Approach to Test-Time Adaptation:** The introduction of UnMix-TNS recalibrates instance-wise statistics by mixing with multiple unmixed statistics components. It targets the challenge of non-i.i.d. test batch distributions, a limitation in current test-time adaptation methods.
2. **Seamless Integration with Pre-Trained Models:** The ability of UnMix-TNS to integrate with existing pre-trained neural network architectures without significant modifications is also an advantage. This facilitates easier adoption in various applications.
3. **Empirical Validation:** The paper provides empirical evidence of performance improvements across multiple benchmarks, including those with common corruption shifts and natural shifts. UnMix-TNS demonstrates robustness in diverse scenarios. The introduction of new datasets like ImageNet-VID-C and LaSOT-C for realistic domain shifts in frame-wise video classification adds value.

### Weaknesses
1. **Lack of Theoretical Insight for Temporal Correlation Handling:** The abstract and introduction do not provide clear theoretical insights or explanations on how UnMix-TNS effectively deals with test data streams having temporal correlation. A deeper understanding of the underlying principles is crucial for assessing the robustness and reliability of the method.
2. **Potential for Bias in Instance Selection:** The effectiveness of UnMix-TNS might be influenced by the selection of instances from incoming test batches for updating statistics. If the selection is not well-designed, it could introduce bias, affecting the generalization of the model.

### Questions
- Could the authors provide more detailed theoretical insights or foundational principles on how UnMix-TNS adapts to test data streams with temporal correlation? Specifically, how does the method theoretically account for and mitigate issues that arise due to temporal dependencies in the data?
- What is the theoretical rationale behind the 'unmixing' approach in handling temporally correlated data? How does this strategy compare with traditional methods in terms of theoretical robustness against temporal variations in data streams?
- Could the authors elaborate on the criteria or algorithm used to select instances from incoming test batches for updating the UnMix-TNS statistics? How do the authors ensure that this selection process does not introduce bias, which might affect the model’s generalization capability?
- In situations where the test batches contain highly diverse or outlier instances, how does UnMix-TNS maintain the integrity of the normalization statistics? Is there a mechanism in place to detect and appropriately handle such anomalies?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes Un-Mixing Test-Time Normalization Statistics (UnMix-TNS), a novel BatchNorm adaptation strategy under the assumption that the test distribution is a shifting mixture. Empirical experiments on multiple CV tasks indicate that UnMix-TNS can adapt to the temporal correlation of unlabeled test streams.

### Strengths
Novel idea!

Intuitive toy example (i.e. Figure 1) illustrating the motivation of the proposed method.

Good empirical performance, improving over baseline methods most of the time. Mixes well (no pun intended) with other BN adaptation methods like Tent and LAME, making them even better. Synergy!

Negligible additional inference time cost (the authors should highlight it instead of burying it under a subsection titled "Ablation Studies").

### Weaknesses
UnMix-TNS seems tightly coupled to the convolutional layers, as it considers the channel dimension $C$ and all experiments are computer vision tasks. It is unclear whether it will work for other deep learning tasks, e.g. NLP.

The paper doesn't discuss how to choose $K$ and $\lambda$, and the additional experiments in Appendix C only study the sensitivity to $K$.

Algorithm 1 in Appendix B.2 is both helpful and confusing: helpful because it describes the algorithm concisely; but confusing because the shapes don't add up. For example, `instance_mean` should have shape `(B, 1, C)`, otherwise you cannot get `hat_mean` of size `(B, K, C)` through broadcasting. It would be better to include a runnable Python snippet in the Appendix, even if that's slightly more verbose.

### Questions
In Appendix B.2, it's unclear what `torch.var_mean(x, dim=[2, 3])` means. Does `x` have dimension `(batch, channel, height, weight)`?

From my understanding, UnMix-TNS will fail when test samples gradually shift from one cluster to another, in which case the active $\mu_k$ would follow the sample, causing two clusters to eventually overlap. How would you prevent that?

Not really a question, but an interesting experiment is to set $K$ to the number of corruption types, and see if $p_{b,k}^t$ can recover the corruption type for each sample correctly. Maybe each cluster will correspond to the label instead of corruption. Who knows?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies how to adapt the batch norm layers for test-time adaptation. Standard TBN method assumes that test samples are i.i.d. sampled from a single target distribution, while this assumption can be violated in real scenarios: samples can be drawn from multiple distributions and can be temporal correlated. The author propose a nuanced method called UnMix-TNS, which split the running statistics of each BN layer into K different components. For each testing sample, only the closest component will be refined, which makes the BN layer more stable and robust to changes in batch-wise label distribution. The author verify the proposed method over benchmarking datasets, several settings, a wide range of models, and compare to a variety of baselines.

### Strengths
1. Clear motivation: The author clearly summarizes key drawbacks of BN-based TTA methods. Figure 1 is intuitive and informative. 
2. Algorithm design: The algorithm is generally intuitive to me. Clearly if each component corresponds to a label, the algorithm is likely to be robust to the bias in label distribution. Also the author pay special attention on the initialization of UnMix-TNS components to preserve the statistical properties. 
3. Experiments: The proposed method is tested on a variety of datasets, models, evaluation protocols, and is compared to a wide range of related baselines.

### Weaknesses
Major (These weaknesses or concerns significantly affect my understanding and decision regarding this paper)

1. Multiple target domains: One of the claimed contribution is that UnMix-TNS has robustness when tested on continual domain and mixed domain. However, I am very confused which part of UniMix-TNS is designed for and beneficial to these multi-domain settings, especially the mixed domain setting. Considering that there is a new batch containing images from different domains, although different images may correspond to different UnMix-TNS components, they are finally normalized with the same mean and variance according to Eq (11). It seems like UnMix-TNS still treat multiple target domains as one single target domain. 
2. Unclear experimental setting and unsatisfactory performance: The author claimed that they follow the protocol outlined in (Lim et al., 2023). However, the performance for most of the baselines are significantly worse than the results in (Lim et al., 2023). Also, the proposed UnMix-TNS fails to outperform “Source” in mixed domains. I believe the comparison to baselines only makes sense if the test-time normalization is beneficial. (I presume that “Source” means no adaptation. Please correct me if I am wrong)
3. If the temporal correlation mainly 
4. Choice of K. The appendix discusses the influence of K. However, it is still unclear how to choose K in practice. How is it related to the number of classes and number of domains? (Although the number of domains might not be exposed.)

Minor (These minor weaknesses are not crucial but I believe fixing them will improve the quality of the paper) 
1. Figure 2 is pixelated. Please consider improving the dpi. 
2. The temporal correlation might be better explained in Section 2.1. Does it refer to correlation of feature, label, or domain? 
3. Page 5 after Equation (1), two exp have different font style. Also I recommend changing $\sim$ to $\approx$ since $\sim$ usually means “following the distribution of”.

### Questions
Besides the major weaknesses, there are several minor questions: 
1. In Equation (5), it seems like all $\mu_{k, c}^0$ for different $k$ distribute on the line of $\mu_c + t \sigma_c$. Is that intentional? Or it makes more sense if they do not have such low-rank structure? 
2. Equation (7). Are there any insight on using cosine similarity instead of L2 distance?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
