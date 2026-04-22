# Defending against Model Extraction for GNNs with Model Reprogramming

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
The goal of model extraction (ME) on Graph Neural Networks (GNNs) is to steal the functionality of GNN models. Defense against extracting GNN models faces several challenges: (1) existing defense primarily designed for defense against convolutional neural networks without considering the graph structure of GNNs; (2) watermarked-based defense is typically passive without preventing model extraction from happening and can only identify a model stealing after extraction has occurred; (3) they either require entirely defensive training from scratch or expensive computation during inference. To address these limitations, we propose an effective defense method that can reprogram the model with graph structure-based and layer-wise noise to prevent ME for GNNs while maintaining model utility. Specifically, we reprogram the target model to: (1) introduce graph structure-based disturbances that prevent the attacker from fully learning its functionality; (2) incorporate data-specific, layer-wise noise into the target model to enhance defense while maintaining utility. Therefore, we can prevent the attacker from extracting the reprogrammed target model and preserve the model's utility with improved inference efficiency. Extensive experiments and analysis on defending against both hard-label and soft-label ME for GNNs demonstrate that our strategy can lessen the effectiveness of existing attack strategies while maintaining the model utility of the target model for benign queries.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a defense mechanism against model extraction (ME) attacks on Graph Neural Networks by model reprogramming. The authors introduce layer-wise noise and graph structure-aware perturbations to the target model, enabling it to maintain high accuracy for in-distribution queries while misleading out-of-distribution queries. The method is designed to prevent ME attacks without requiring full model retraining or expensive inference-time computations.

### Strengths
1. The paper adapts model reprogramming to defend against model extraction for GNNs.
2. The method leverages graph structural features to tailor perturbations.

### Weaknesses
* While adaptive attacks are considered, the assumption that all attack queries are OOD, is simplified and depends on real-world applications since there exists various open-source data. If the attackers' queries are regarded as OOD, the problem could be degraded to OOD detection.
* The authors emphasize that they address a specific scientific challenge in the web domain. However, the experimental validation is conducted exclusively on biochemical datasets.
* In line 758, the assumptions of infinite clone model capacity is not be practical.
* The provided code link is empty.
* The reference is not up-to-date, only contains studies before early 2024.
* The citation format is wrong, please use the \citep comment.

### Questions
* Refine the work with more practical assumptions.
* Add recent citations.
* Improve presentations.
* Reproducible issues.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a defense method to prevent the model extraction attacks against graph neural networks. Different from previous works, the proposed method achieve the defense method from the perspective of graph structure to rerpgram model. Extensive experimentl results show validate the effectiveness of the proposed approach.

### Strengths
This submission has the following strengths:
- The proposed method is clealry motivated.
- The method achives the defense method against graph neural network extraction attacks from a new perspective.

### Weaknesses
This submission has the following weaknesses:
- The use of graph structures for defending against attacks has already been explored in other tasks, such as graph backdoor defense.
- Although the proposed method is effective in defending model extraction attacks, the utlity of graph nerual networks drop more compared with some baselines as shown in Table 4.

### Questions
I have the following questions/suggestions:
- What is the meaning of the two downward arrows shown for P-poison in Table 1?
- The references format seems not to be correct. It would be better to have citations follow the Name (Year) or (Name, Year) styles appropriately based on context rather than only Name (Year).
- Line 1159, 'in Sectoin ??', the specific section number is missing.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a model reprogramming method against model extraction attack. By injecting optimizable noise over input data, the defended model is designed to give error prediction for malicious prediction queries but maintain good performance when inputting normal queries. Extensive experiments have been done to validate the proposed method, and also theoretical guarantees are also given on lower bounded increasing loss on OOD queries,

### Strengths
**1.** The idea of applying reprogramming model to defend extraction attack is novel. It makes sense that after reprogramming the parameters will become different but still hold a good performance, and optimizable noise for once time is also efficient. This technique approach may potentially inspire more defending methods.

**2.** The paper is presented clean and well-formulated. The problems for attacker and defender are clean stated, and the methodology is also expressed in tidy formulations and algorithm. The theoretical proof in appendix is also well constructed.

**3.** Extensive experiments have been done to validate the effectiveness of the method. These not include general defense performance on different base models against various attack methods compared with baselines, but also ablation study on layer numbers and inner mechanism.

### Weaknesses
**1.** I'm not fully convinced by the motivation of the proposed defense that giving misleading classification for so called "OOD" data. The OOD data from attacker input is defined as data different from the training data in this paper, potentially sourcing from some nature input or synthetic data. However, if such a input from a common out-source query would be assumed as an OOD data and result in a wrong prediction, it seems also destroy the utility of the model for general usage; if such a common input would be assumed as ID data, it seems also no reason the attacker can only have OOD input since common input is always accessible. 

**minor**   C.6.2 contains fail section reference.

### Questions
**1.** (See weakness 1) Please make clarification on the OOD queries and ID queries and how the attacker's query could be divided from the normal user's (an application in a practical scenario as example is suggested). Considering this confusion casts my main doubt on the rationality of the problem definition and proposed method, I could **only give a 4 score** currently. While if well addressed, **I would increase my score to at least positive**.

**2.** What advantages such defense method occupies compared with direct OOD detection on input queries?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an active defense strategy to protect Graph Neural Networks (GNNs) from model extraction (ME) attacks. Model extraction attacks aim to replicate the functionality of a target GNN by querying it and using the responses to train a surrogate model. The authors highlight the limitations of existing defense methods, which are either reactive, computationally expensive, or not tailored to the unique structure of GNNs. To address these issues, the paper introduces a model reprogramming approach that incorporates graph structure-based disturbances and layer-wise noise. This method prevents attackers from extracting useful information from the model while maintaining its utility for legitimate queries. Extensive experiments demonstrate that the proposed defense reduces the effectiveness of both hard-label and soft-label ME attacks while preserving the model's performance on benign tasks. The paper presents a theoretical foundation for the method, and the results show that the defense is both efficient and effective in safeguarding GNNs without requiring full retraining or significant computational overhead.

### Strengths
1. This is the first work to introduce the concept of model reprogramming into GNN security defense, providing a new perspective for active protection against model extraction.

2. The method only injects learnable noise into intermediate layers without modifying the architecture or retraining the model, resulting in low computational cost.

3. The paper includes thorough experiments, demonstrating the defense's effectiveness on standard graph classification datasets.

### Weaknesses
- The paper assumes that attacker queries are entirely out-of-distribution in both data-based and data-free settings, yet this assumption is weakly supported.

- The experimental evaluation does not include defense tests against representative and state-of-the-art GNN model extraction attacks such as GNNStealing and STEALGNN, limiting empirical credibility.

- The justification for employing layer-wise noise as a defense mechanism is insufficiently discussed, and the proposed approach offers limited methodological innovation.

### Questions
- While the cited literature supports the assumption that synthetic data may be out-of-distribution in data-free settings, could the authors further clarify the rationale for extending this assumption to data-based scenarios?

- Could the authors conduct experiments on commonly used citation network datasets for GNN extraction attacks, and report standard fidelity metrics to better demonstrate the effectiveness of the proposed defense?

- Could the authors elaborate on the specific motivations or potential benefits of introducing trainable layer-wise noise, and how this approach differs from or improves upon conventional model fine-tuning strategies?

### Soundness
2

### Presentation
2

### Contribution
2
