# PACIA: Parameter-Efficient Adapter for Few-Shot Molecular Property Prediction

- Avg Score: 5.00
- Decision: Reject
- Scores: 3, 6, 6, 5

## Abstract
Molecular property prediction (MPP) plays a crucial role in biomedical applications, but it often encounters challenges due to a scarcity of labeled data. Existing works commonly adopt gradient-based strategy to update a large amount of parameter for property-level adaptation.  However, the increase of adaptive parameters can cause overfitting and lead to poor performance. Observing that graph neural network (GNN) performs well as both encoder and predictor, we propose PACIA, a parameter-efficient GNN adapter for few-shot MPP. We design a unified adapter to generate a few adaptive parameters to modulate the message passing process of GNN. We then adopt hierarchical adaptation mechanism to adapt the encoder on property-level and the predictor on molecule-level by the unified GNN adapter. Extensive results show that PACIA obtains the state-of-the-art performance in few-shot MPP problems, and our proposed hierarchical adaptation mechanism is rational and effective.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the few-shot molecular property prediction problem. Existing gradient-based few-shot methods generally need to update a large number of learnable parameters during the meta-test stage, which is prone to overfitting. To address this problem, this paper proposes the PACIA method the leverages a hypernet to generate adaptive parameters for each task and each molecule in a task.

### Strengths
1. This paper studies an important research problem.
2. The proposed method is clear in general and makes sense.

### Weaknesses
1. Lack of comparison to SOTA. In Tab. 1 and Tab. 2, the only previous works on few-shot MMP included are ADKF-IFT and PAR. The authors should compare the proposed methods with more existing SOTAs such as [a].
2. Lack of novelty. The core component is HyperNet. Based on HyperNet, the proposed PACIA makes no significant technical contribution.
3. Poor writing. Many sentences are grammatically wrong. Some examples are:
* [In Sec. 1] First is that ... difference.
* [In Sec. 1] The chemical space is enormous that ... range.
* [In Sec. 1] The molecule-level difference ... molecules.
* [In Sec. 1] While others ... accurately.
Note that four of the first 6 sentences of this paragraph are grammatically incorrect.
There are many errors in addition to the above examples. The authors are suggested to carefully proofread the paper to correct the errors.
4. The design in modulating propagation depth seems not fully reasonable. According to Eqn. (8), $[p]_l$ measures the probability of the event that "the $l$-th layer is in the model". However, from Eqn. (7),  the only constraint on $[p]_l $ is $\sum_l [p]_l = 1$. So it is likely that for some $1<i<j<L$, $[p]_i < [p]_j$. That is to say, a hidden layer (i.e., layer $j$) is more likely to be in the model than a layer before it (i.e. layer $i$), which is unreasonable.

### Questions
Please see "weakness".

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
PACIA is a novel approach aimed at addressing the challenges in Molecular Property Prediction (MPP) when labeled data is scarce. The authors identify that existing methods, which typically rely on a gradient-based strategy for property-level adaptation, are prone to overfitting due to the large number of adaptive parameters required. To overcome this, they introduce PACIA, a parameter-efficient Graph Neural Network (GNN) adapter specifically designed for few-shot MPP scenarios.

### Strengths
Innovative Solution: PACIA introduces a novel parameter-efficient adapter for Few-Shot Molecular Property Prediction (MPP), addressing the challenge of overfitting in scenarios with scarce labeled data. This approach stands out due to its unique application of a hierarchical adaptation mechanism, modulating both the encoder and predictor in a GNN framework.

Well-Defined Problem and Solution: The paper clearly defines the problem of few-shot MPP and presents PACIA as a well-justified solution. The hierarchical adaptation mechanism is meticulously designed, reflecting the high quality of the work.

Advancement in Methodology: The introduction of a parameter-efficient adapter and the application of hierarchical adaptation in GNNs for MPP represent a notable advancement in methodology, setting a precedent for future work in the domain.

### Weaknesses
Lack of Ablation Studies on Hypernetworks:
While the paper introduces the innovative use of hypernetworks for generating adaptive parameters, it lacks ablation studies or a deeper analysis of how different configurations of hypernetworks affect the performance of PACIA. Incorporating ablation studies or a detailed analysis focused on the hypernetworks component would provide valuable insights into its role and optimization, potentially leading to further improvements in PACIA’s performance.

Need for Broader Applicability and Generalization:
The paper validates PACIA’s performance in few-shot MPP problems, but it could strengthen its case by demonstrating the model’s applicability and generalization across a wider range of molecular property prediction tasks. Conducting experiments or providing examples of PACIA’s performance in diverse MPP tasks would showcase its versatility and generalization capabilities, further solidifying its contributions to the field.

### Questions
Comprehensive Comparison with Baselines: Could the authors provide additional comparisons with a broader range of existing methods, especially those that have shown promising results in related domains, to strengthen the validation of PACIA’s performance?

Analysis of Hypernetworks: How do different configurations or architectures of hypernetworks affect the performance of PACIA? Are there specific settings that are more optimal for this application?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a parameter-efficient approach for few-shot molecular property prediction (MPP) tasks by involving hypernetwork to modulate the GNN parameters. The proposed PACIA is built on top of a main encoder-decoder MPP network, and by learning from the training, a GNN adapter is trained to modulate the node embedding and GNN depth. Extensive experiments are conducted in two settings to evaluate the performance of PACIA. Several in-depth analysis is provided to further discuss the superiority of PACIA, such as running time.

### Strengths
+ The paper is interesting as it presents another direction for few-shot MPP. Unlike general gradient-based approaches, PACIA tends to learn certain key generalized parameters to minimize the training costs. 
+ The paper is well-written and easy to follow. 
+ The authors have conducted several in-depth analysis to comprehensively evaluate the performance of the proposed method.

### Weaknesses
- The approaches of modulating node embedding and GNN depth do not have sufficient theoretical support. It is more like experimental attempts. Can the authors provide more details about why the implementation is designed as such? How does such implementation ensure the adaptor learns sufficient information? 
- The main framework of PACIA is based on PAR, making the technical novelty incremental. 
- The figure font is small and hard to recognize. Fig.1 (b) is too abstract. The authors may consider plotting a more detailed overall framework to help understand their method.
- In Table 2, why the baseline methods are different? PAR is the most similar baseline model, and should be compared.

### Questions
See Weaknesses. 

The proposed method is interesting and has certain merit. I am also curious that will it works on general MPP problems? Did the authors try to see the performance not under the few-shot setting?

### Soundness
3 good

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
This paper delves into the challenges of few-shot molecular property prediction, highlighting two major limitations in current approaches: the neglect of molecule-level differences and a predisposition to overfitting. In response, the authors introduce a parameter-efficient adapter complemented by a molecule-adaptive predictor. The experimental results on various benchmark datasets have demonstrated the effectiveness of the proposed method.

### Strengths
1.	The paper focuses on an intriguing and pivotal issue. The scarcity of labeled datasets is a prevalent challenge in the realm of chemistry. 
2.	The proposed method is well-motivated. The introduction lucidly underscores the drawbacks of the existing works, and each module designed in this study directly addresses these shortcomings.
3.	The experimental results shown in Table 1 and Table 2 clearly demonstrate the effectiveness of the proposed method compared with the various baselines. Additionally, the authors have undertaken an exhaustive ablation study that accentuates the significance of each individual component.

### Weaknesses
1.	The exposition on the methodology appears somewhat nebulous, which hampers a clear comprehension of the distinct contributions of each module. Specifically, the average representation at the l-th GNN layer, as depicted in Equation (9), seems disconnected from subsequent steps. The property adaptation and molecule adaptation are not clear in the algorithm.
2.	There is a noticeable discrepancy in the baselines used for comparison in Tables 1 and 2. The rationale behind this difference remains unexplained. For instance, while PAR[1] is conspicuously absent from Table 2, it seems like a plausible candidate for few-shot molecular property prediction in FS-Mol.
3.	The presentation of results in Tables 1 and 2 would benefit from a consistent format, ensuring ease of interpretation for readers.


[1] Property-aware relation networks for few-shot molecular property prediction. NeurIPS 2021.

### Questions
Please refer to the weaknesses

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
