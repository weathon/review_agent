# Anti-Backdoor Coreset Selection via the Cumulative Entropy Criterion

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Recent training-time defenses against neural backdoors isolate a benign subset from poisoned training data, to learn a backdoor-free model from it. In this paper, we formulate this defense strategy as a coreset selection problem, giving rise to so-called “Anti-Backdoor Coreset Selection.” Since poisonous samples have a) lower prediction uncertainty and are b) less frequent than benign samples, coreset selection naturally focuses more on samples associated with benign functionality than the backdoor functionality. We use the Cumulative Entropy as selection criterion to further facilitate this effect. The metric tracks the learning dynamics of training samples and allowing us to select benign samples with high informativeness for the coreset. Additionally, we unlearn the chosen samples in each epoch to facilitate the separability between benign and poisonous samples. Together, this yields an exceptionally effective training-time defense that constructs a benign
coreset to train a backdoor-free model. Unlike prior defenses that compromise natural accuracy and fail against certain attacks, our method mitigates backdooring attacks consistently with a negligible impact on natural performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper presents a training-time backdoor defence framed as a coreset-selection recipe built on a rigid assumption: poisoned samples follow different learning dynamics; They become confidently classified much earlier than benign ones.
It uses cumulative entropy over epochs, with per-epoch normalisation, to rank and keep the “informative” subset, followed by warm-up + unlearning + retrain.
Because this assumption holds mainly for non-adaptive, simpler attacks, the method performs well on standard benchmarks but weakens on adaptive or stealthy attacks.

### Strengths
- The paper packages known ingredients, coreset selection, cumulative learning-dynamics signals, and light unlearning, into a tidy training-time pipeline for backdoor defence.
- The proposal is methodological, and the empirical results are positive across diverse attacks/datasets.
- It is a convincing engineering package of some established ideas.

### Weaknesses
- The novelty is in integration and execution rather than fundamentally new mechanisms. As mentioned in the paper, learning dynamics, coreset, entropy, and cross-epoch aggregation are priorly studied ingredients for the proposed pipeline. 
- The key assumption that poisons learn fast and become confident early may not hold in adaptive attacks. It would be helpful to include an experiment with an adaptive attack targeting this assumption.
- The assumption of the lack of reference/validation data is also overly strong. It would be helpful to include baselines that use reasonable amount of reference data.
- There are quite a few hyperparameters whose robustness should be tested.
- With the coreset retained being considerably smaller than the original data, it is unclear whether rare and complex cases (particularly in imbalanced problems) would be discarded.

### Questions
1. What is the theoretical motivation for the min-max scaling of entropy? How does the range of changes change as training progresses?
2. What is the difference between aggregated entropy and AUC (under the entropy curve)?
3. Since you train on a subset, can you report tail/edge-case accuracy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a training-time backdoor defense method. Unlike traditional defenses, such as the model pruning or poisoned samples detection, this method adopts a coreset selection strategy to extract a subset of representative and clean samples from a poisoned training dataset. After this, the selected coreset is used to retrain the model, effectively addressing the impact of the backdoor attack. The key innovation of this work is the coreset selection criterion: the authors introduce a Cumulative Entropy (CENT)-based sample selection method that, through the Warm-up, Unlearning, and Label Smoothing stages, enables more accurate identification of clean samples, resulting in a high-quality and clean training subset.

### Strengths
1.	The paper introduces an innovative coreset selection method by proposing cumulative entropy as a metric to quantify the information content and uncertainty of samples, demonstrating a certain level of originality. 
2.	The defense method does not rely on any additional clean datasets during implementation, and experiments conducted on multiple datasets under different attack scenarios show that ABCS significantly reduces the attack success rate. 
3.	A lot of ablation studies is designed to directly reveal the influence of each parameter on the model.
4.	The method does not rely on any additional clean reference dataset, and its training time is comparable to standard training, demonstrating strong practicality.

### Weaknesses
1.	the proposed method lacks theoretical justification and instead relies on empirical statistics showing that poisoned and clean samples exhibit significant differences in the Cumulative Entropy metric. 
2.	The method selects a coreset based on this metric, which reduces the overall information contained in the dataset; when the sample size is small, this reduction may affect the model’s primary accuracy.
3.	CENT prioritizes “high-information” samples, but this may introduce bias against representative samples from minority or rare distributions, thereby affecting fairness and generalization.

### Questions
1.	The proposed approach may reduce the training accuracy on datasets with limited sample sizes due to the smaller effective data volume. Please test the method on small datasets to confirm its stability. 
2.	The paper relies on differences in cumulative entropy to distinguish poisoned from clean samples, but lacks theoretical justification or clear visual evidence for this assumption; providing a formal analysis or illustrative results would clarify why cumulative entropy effectively captures poisoning characteristics and strengthen the method’s credibility.
3.	Prioritizing high-information samples may bias against minority or rare data, reducing fairness and generalization; incorporating distribution-aware sampling or fairness regularization could help balance information gain and diversity.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper reframes training-time backdoor defense as coreset selection. It proposes ABCS, which warms up a model, ranks samples by cumulative entropy across epochs, performs a light unlearning step to widen the uncertainty gap between clean and poisoned data, then retrains from scratch on the top informative subset. It’s evaluated on CIFAR-10 and Tiny-ImageNet against multiple attacks (BadNets, Blend, CLB, IAB, WaNet, ISSBA, Low-Frequency, Adaptive Blend) plus an all-to-all variant. ABCS selects ~55% of data with near-zero poison rate and preserves clean accuracy while matching or improving total training time versus naive training.

### Strengths
1.Recasts training-time backdoor defense as coreset selection with a principled cumulative uncertainty criterion, rather than explicit poisoned/benign splitting.

2.Provides clear motivation that poisoned samples tend to be less informative (low entropy) while clean, hard examples remain high-entropy.

3.The CENT formula, normalization, and rationale are precisely specified. Figures contrasting epoch-wise vs. cumulative selection help understanding.

### Weaknesses
1.Given that the paper fixes the target label and uses a global poisoning rate, does this imply that poisoned samples are drawn from the entire training set and can originate from any source class? However, real attackers often poison non-uniformly (e.g., only a single source classes) and at low rates. It’s unclear how sensitive ABCS is to class-selective or distribution-shifted poisoning, where high-entropy informative clean samples may be concentrated in specific regions.

2.The work lacks analysis in terms of the selection-ratio of the coresets. The size is driven by an automatic thresholding heuristic on cumulative entropy, and in different settings the required ratio varies widely, leaving practitioners without bounds that relate coreset size to poisoning rate/attack strength or guidance on how small a coreset remains safe across datasets and threats.

3.While the paper mentions adaptive scenarios, there is limited exploration of attackers who maximize entropy of poisoned samples (e.g., trigger randomization, training-time augmentations designed to keep poisoned examples uncertain) to subvert ABCS.

4.The method is strongly dependent on hyperparameter choices (e.g., warm-up length, number of selection epochs, and unlearning strength), which may limit its plug-and-play usability and robustness in real-world deployments.

### Questions
Please refer to “Weaknesses”.

### Soundness
2

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
This paper proposes a novel training-time defense against neural backdoor attacks called Anti-Backdoor Coreset Selection. The core idea formulates the defense as a coreset selection problem, aiming to extract a small, informative, and benign subset from a poisoned training set. The authors introduce the Cumulative Entropy criterion, which accumulates the prediction entropy of each sample over multiple training epochs to identify data valuable for the primary task and unlikely to be poisonous. ABCS consists of three phases: a warm-up phase, a coreset selection phase, and a final training phase on the selected clean coreset. Experiments on image and text benchmarks demonstrate that ABCS effectively mitigates various backdoor attacks, maintains natural accuracy comparable to training on a fully clean dataset, and incurs computational costs similar to standard training.

### Strengths
1. This paper reframes the backdoor defense problem during training as a coreset selection problem, breaking from the traditional dataset splitting approach. By leveraging the coreset concept from data-efficient learning for defense, it provides a novel and theoretically grounded perspective for training-time backdoor defense. This design is clearly demonstrated through the definition of ABCS and its comparison with traditional defenses.

2. The proposed Cumulative Entropy criterion effectively distinguishes benign from poisonous samples by accumulating entropy values over multiple epochs. In experiments, it performs best in reducing the poisoning rate of the coreset and the Attack Success Rate, forming the core support for ABCS's defensive effectiveness.

3. It does not rely on a clean reference dataset, avoiding manual vetting costs. The coreset size is smaller than the full dataset, leading to faster final training. The overall runtime is comparable to naive training, making it suitable for practical application scenarios.

### Weaknesses
1. The paper only empirically validates the effectiveness of ABCS without providing a theoretical explanation for why cumulative entropy separates benign and poisonous sample distributions better than single-epoch metrics or the mathematical relationship between coreset selection and backdoor defense effectiveness. The conclusion section of the document explicitly acknowledges this limitation.

2. Against adaptive attacks where the adversary knows the CENT criterion, ABCS's ASR increases significantly (ASR rises to 11.69% for A-Blend and 6.44% for WaNet), indicating a shortcoming in defending against highly knowledgeable attackers.

3. The selected coreset size is larger than the optimal coreset for a clean dataset, and the paper lacks a deep analysis of the trade-off between coreset size and defensive robustness/natural accuracy. There is potential for further data size reduction.

### Questions
1. Why is extracting a partial benign coreset superior to splitting out all benign samples, especially at low poisoning rates? Please support the necessity of this framework choice.

2. In Table 4, the coreset selection ratio $r_{se}$ for the WaNet attack (57.83%) is higher than for the Blend attack (53.85%). Is this difference directly related to the impact of these two attack types on sample learning difficulty? Furthermore, for stealthier attacks like A-Blend, is it necessary to adjust the currently fixed label smoothing factor $\epsilon$ (0.9) to enhance the unlearning effect, or is this $\epsilon$ value universally applicable across attack scenarios？

### Soundness
3

### Presentation
3

### Contribution
3
