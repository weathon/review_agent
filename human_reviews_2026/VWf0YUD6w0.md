# LLM-assisted Semantic Reasoning for Open-Set Active Learning

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Active learning (AL) methods primarily concentrate on closed-set annotations where irrelevant data is absent. However, real-world applications inevitably contain various forms of irrelevant data. This open-set annotation challenge has been explored in some studies, yet two key issues remain. The first is balancing between selecting maximally relevant data and querying uncertain samples, which often increases the proportion of irrelevant data. The second is the inability to distinguish between relevant and irrelevant samples before any labeling, commonly referred to as the cold-start problem. We tackle these challenges with our method named LaSeR (LLM-assisted Semantic Reasoning), which leverages LLM-generated image descriptions and VLM-based similarity scores to, introduce a metric capable of separating relevant from irrelevant data before labeling, and incorporates diversity in the selected samples to enhance model performance. Subsequently in later AL rounds, as more labeled data becomes available, we transfer this knowledge into a detector model to further improve the efficiency of our selection process. Extensive experimental results demonstrate that our method outperforms state-of-the-art AL approaches, as well as recent methods specifically designed for open-set active annotation on standard benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes LaSeR, an active learning framework for open-set image classification that operates without any labeled seed data, requiring only class names as input. The framework leverages LLMs to generate textual descriptions for each known class along with additional confuser classes representing likely irrelevant concepts, then scores unlabeled images using a VLM that measures similarity between images and relevant versus confuser descriptions. After the first round, LaSeR trains a detector to distinguish relevant from irrelevant samples and combines both the VLM and detector signals through linear interpolation. Experiments on CIFAR10/100 and Tiny-ImageNet show better early-round performance.

### Strengths
1. The proposed method removes labeled seeds and uses only class names and LLM-generated confusers.
2. The authors address the bottleneck of label scarcity in open-set active learning by leveraging available LLMs and VLMs.

### Weaknesses
1. The paper uses a linear combination of the VLM and detector scores with a coefficient \delta that is manually scheduled.
But, the text and equation are inconsistent: the paper states that “we gradually decrease \delta to rely more on the detector,” but the formula indicates the opposite (larger \delta gives more detector weight).
2. The method does not explicitly enforce diversity among the queried samples. Although the authors use multiple textual prompts, the selection strategy still picks the top-n samples by score, which may lead to redundancy.
3. The evaluation scope is limited to small low-resolution datasets. The method’s scalability and robustness remain unclear on larger datasets.

### Questions
1. The text says that \delta is decreased each round to rely more on the detector, but according to Eq. 5, increasing \delta gives the detector more weight. Could you clarify which version is implemented, and show how performance varies with different \delta schedules?
2. Do LLM-generated confusers sometimes include classes that never appear in the unlabeled pool? If so, does this harm the relevance score? Have you tried filtering out or down-weighting confusers with consistently low similarity to any image?
3. The ablation (Table 5) shows that multiple text prompts improve performance, suggesting some diversity benefit. Did you measure redundancy among the selected samples (e.g., feature overlap)? Would adding a diversity-based selection criterion further improve label efficiency?
4. Each round trains two networks for 300 epochs. How much additional training time does this require compared to standard OSAL baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces LaSeR (LLM-assisted Semantic Reasoning), a novel framework for open-set active learning (OSAL) that addresses the cold-start problem without requiring any initially labeled data. Unlike prior OSAL methods, LaSeR only uses textual labels of relevant classes and leverages large language models (LLMs) to generate diverse textual descriptions and semantically similar but irrelevant classes. These descriptions are then encoded via a vision-language model (VLM) to compute relevance scores for unlabeled images. In later active learning rounds, a CNN-based detector is trained on previously labeled data to further refine sample selection. The method dynamically balances VLM-based semantic reasoning and detector-based discrimination. Extensive experiments on benchmark datasets show that LaSeR outperforms state-of-the-art methods, especially in early rounds.

### Strengths
1. The paper introduces, for the first time, an OSAL method that operates without initial labeled data, thereby tackling the cold-start issue and bridging a gap in current research.
2. The method combines LLMs, VLMs, and CNN detectors, dynamically adapting its strategy at different stages to make full use of semantic reasoning and visual features.
3. The paper is clearly written and easy to follow. The proposed approach is simple yet effective, enhancing the interpretability and transparency of model selection by generating class descriptions and confusing classes.

### Weaknesses
1. The paper contains multiple language and expression errors, such as “where we DL is unavailable”, “We gradually decrease δ in later AL rounds”, and inconsistencies like “CIFAR10” vs. “CIFAR-10”, among others.
2. The paper lacks sufficient rigor, as the experimental results on CIFAR-100 and Tiny-ImageNet do not present all the compared methods comprehensively.
3. The introduction of LLMs or VLMs inherently brings substantial prior knowledge. It is therefore necessary for the authors to conduct additional validation on some non-standard datasets to better demonstrate the generalizability of the proposed method.
4. The training and score design of the detector are overly simple and are likely to underperform compared with existing methods such as EOAL or the more recent EAOA [1]. The advantage of the proposed approach mainly appears in the early stages, which also explains why its performance on CIFAR-10 becomes slightly inferior to the compared methods in later rounds.

[1] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach

### Questions
Please see the Weaknesses.

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
4

### Summary
This paper introduces LLM-assisted Semantic Reasoning (LaSeR), an active learning framework that addresses the challenge of being able to distinguish between, and effectively select the most relevant samples in a dataset that contains relevant and irrelevant classes. LaSeR integrates LLMs (to generated descriptions from the given labels of relevant classes) and VLMs (to measure the relevance of each image w.r.t. the label descriptions) to compute a relevance score which guides sample selection for the next active learning round. This process allows for the framework to mitigate the cold start problem, as it does not require an initial set of labeled data to guide the initial rounds of active learning. In later rounds, this framework also integrates a CNN-based detector to more effectively select the relevant and helpful samples. The authors also present experiments on popular AL datasets (such as CIFAR-10, CIFAR-100, Tiny-ImageNet) which showcase LaSeR outperforming other state-of-the-art methods in metrics like accuracy and precision, especially shinning in the early training rounds.

### Strengths
- The paper proposes a novel solution that addresses the cold start problem in open-set active learning which eliminates the need for initial labeled data, this ability to effectively start the AL process (and select relevant examples in early rounds) without initial labels make it especially useful in cases where labeled data is scarce and hard to find
- In the empirical experiments presented, LaSeR was showcased to outperform popular regular AL and OSAL-specific methods on a number of benchmarks (CIFAR-10, CIFAR-100, Tiny-ImageNet) in classification accuracy, especially in the early rounds (great for targeting the cold-start issue)
- Rigorous ablation studies were conducted to demonstrate the contribution and importance of each component within the proposed framework

### Weaknesses
- Limitations of the real-world applicability of the framework: The proposed framework seems to only be effective for image classification use cases (the framework setup rather specifically to that task, and the experiments are all conducted on image classification datasets), this limits its generalizability and utility for broader applications. The first step of the framework (using the LLM to generate descriptions / alternative class names) also assumes informative class names that LLMs can easily understand and operate on, which might further limit its effectiveness under situations where the class names are less informative / meaningful. Lastly, the evaluations were conducted on small scale image datasets (two of which, CIFAR-10 / CIFAR-100, are also closely related), which might not give a full representation of real-world settings.
- The proposed framework is heavily dependent on LLMs for generating class descriptions and class labels, however the computational costs and latency of these LLM calls are not reported or analyzed. It is unclear what the tradeoffs are and how they compare to the baseline methods, which is important for real-world use cases especially for applications with limited budget.
- There is minimal theoretical grounding to the formulation of some components of the proposed framework. Equations 1 and 4 are not rigorously derived (for example, why use subtraction between the components, instead of other potential measures such as taking the ratio / softmax / etc). The final relevance score is also a weighted average of the VLM-based and CNN-based score with an increasing weight on the CNN-based detector scores (linearly by 0.1 each round), however this weighting strategy also does not have a clear theoretical backing and it is unclear if the proposed rate of increase is optimal.

### Questions
- As mentioned above, there is a lack of discussion of the extra costs and latency incurred by the LLM / VLM calls. Can you provide some estimates and analysis for the extra computational costs + latency for these calls (and how they might scale with larger datasets with large number of classes)? It would also be helpful to provide some comparisons about the additional costs here vs cost savings from the reduced number of annotations required to achieve a targeted level of performance.
- The detector model is trained over n+1 classes, I am curious if treating all additional classes as a single “irrelevant” class is the best strategy here? I can see how lumping classes that might be pretty difference from one another can lead to poor decision boundaries. Has the authors experiments with alternative setups (eg. treating them as separate classes or group of classes)? I also wonder if the number of “irrelevant” classes increase, would this cause a massive imbalance in the label distribution?
- The selection precision for LaSeR is often outperformed by methods such as LfOSA or EOAL (especially beyond the first few AL rounds, and the accuracy performance of these baseline methods also sometimes become comparable with LaSeR in the later rounds). It seems like as we continue into the later AL rounds, LaSeR becomes less effective, do the authors think there might be some early stopping mechanism, or have suggestions on how to identify the point at which LaSeR’s performance gains are maximized before diminishing?
- [not a question but pointing out an error] 
    > “We gradually decrease δ in later AL rounds shifting the balance of the selection process toward the detector.” 

  In page 5, there is a typo in the paragraph below eq. 5, “decrease” here should be “increase”

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces LaSeR, a novel method for Open-Set Active Learning that distinctively tackles the "cold-start" problem by not requiring any initial labeled data. The core idea is to leverage the semantic reasoning of Large Language Models to generate textual descriptions for both relevant classes and, crucially, potentially confusing irrelevant classes. A Vision-Language Model is then used to compute a relevance score for unlabeled images based on these descriptions . In later rounds, the method adaptively integrates this score with a standard CNN-based detector trained on the newly acquired labels.

### Strengths
1. The paper addresses two key, practical challenges in active learning: the presence of irrelevant data (open-set) and the cold-start problem (no initial labels). Tackling OSAL without any initial labeled set is a valuable and challenging research direction.

2. The core idea of using an LLM to proactively generate descriptions of confusing irrelevant classes (e.g., "bird" for the relevant class "airplane") is a clever way to improve the VLM's zero-shot filtering capability.

3. The experimental results demonstrate that LaSeR outperforms baselines in the initial AL rounds, validating its effectiveness at solving the cold-start problem where other methods traditionally struggle.

### Weaknesses
1. The main criticism is that the overall construction is a fairly direct engineering integration of existing pre-trained models, the method's effectiveness hinges almost entirely on the powerful, pre-trained LLM and VLM, which, however, is expected to be better than the compared methods, since more information has been used. The contribution feels incremental.

2. The claim of no initial labeled data is qualified by requiring full knowledge of relevant class names; many OSAL setups do not assume this, so the setting is easier in some respects than standard OSAL.

3. In the experiments, the paper says "We report results of all OSAL methods from their original papers, if available," suggesting some curves are not re-run under this paper's exact setup (splits, temperature, budgets), which complicates fairness. Besides, the paper cites recent OSAL works like Mao et al. (2024) and Zong et al. (2024) in the introduction , but these are conspicuously absent from the experimental baselines. Finally, a simple CLIP-only acquisition rule using only class names (no LLM aug) would be a very relevant ablation/baseline.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
