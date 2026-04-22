# General Cross-Attack Backdoor Detector Based on Disturbance Immunity of Triggers

- Avg Score: 1.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 0, 2

## Abstract
Backdoor attacks aim to manipulate the behavior of DNNs under trigger-activated conditions. Data poisoning represents a standard approach for embedding triggers into victim models. Current backdoor detectors struggle to separate trigger-injected samples from the poisoned dataset, which severely suffer from two dilemmas. (1) Modern backdoor features are usually highly coupled with benign features. Existing detectors are almost pixel-based methods, which critically hinder the recognition performance of backdoor features. (2) Owing to the prior lack of poisoned sample distributions, most detectors are restricted to employing approaches akin to unsupervised clustering-based methods. Thus, they heavily rely on sufficient clean samples and deficient artificial priors to efficiently search for poisoned samples with poor generalization across various attacks. This paper introduces a brand-new perspective to reformulate the attackers' objective, ***i.e., backdoor attacks lead the victim models to classify the trigger disturbed by images into the target label***, to identify the community of attacks. Specifically, we propose the concept, ***Disturbance Immunity*** of triggers, and ***theoretically demonstrate that benign and backdoor features exhibit significant classification probability discrepancies across varying perturbations of clean image classes and intensities***. Subsequently, a few known conventional attack patterns are applied to label the poisoned dataset, and then the labeled dataset is perturbed in the above manner to drive the detector to learn the Disturbance Immunity of triggers. Thus, traditional unsupervised clustering-based detection can be transformed into a simple labeled binary classification task.  ***Currently, few method provides detection work based on direct commonality transfer, nor do they break the feature separation task with a labeled-conversion detection framework.*** Finally, we train and present an effective ***G***eneral ***C***ross-attack ***B***ackdoor ***D***etector (***GCBD***). With few clean images $(\leq 10)$, GCBD exhibits ***S***tate-***O***f-***T***he-***A***rt (***SOTA***) detection performance with satisfactory generalization on various SOTA attacks. Additionally, GCBD also supports direct toxicity detection in unseen samples during training, as proved by a more challenging test-time validation approach. Our code will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GCBD (General Cross-attack Backdoor Detector), a novel backdoor detection method based on the concept of "Disturbance Immunity of triggers."  The key insight is that backdoor triggers must maintain their effectiveness across varying image perturbations, while benign features show different probability behaviors under such perturbations. GCBD requires only ≤10 clean images and demonstrates effectiveness across various attacks including Narcissus, SIBA, and Grond.

### Strengths
The paper focuses on the objective of backdoor attacks that the trigger can stably cause misclassification regardless of perturbations in the benign content. The proposed method focuses on the 'attacker's objective' rather than the 'attacker's artifacts', which is interesting.

### Weaknesses
- The paper should include experiments on high-resolution datasets, such as ImageNet, to demonstrate the method's scalability.

- The author only applied their method to models with the ResNet architecture. I am interested in knowing its generalizability on a wider variety of model structures.

- When using ResNet34 as the victim model on the CIFAR-10 dataset, GCBD performs extremely poorly.

- I believe Figure 4 has a significant error. Metrics like ACC should not be stacked, as this is misleading and the stacked values are meaningless. A line chart would be more intuitive for showing performance trends over epochs.

### Questions
- Did the authors only try LSTM as the sequence processing model? Was an ablation study conducted comparing LSTM with other models capable of processing sequential data (e.g., GRU, RNN, or Transformer)?

- If a trigger does not possess "disturbance immunity," would GCBD classify it as clean?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GCBD (General Cross-Attack Backdoor Detector), a detection framework aimed at identifying poisoned samples across various backdoor attacks by exploiting what the authors term the Disturbance Immunity property of triggers. The key intuition is that backdoor features exhibit consistent classification behavior under image perturbations, while benign features vary more. The authors claim to theoretically demonstrate this property and then reformulate detection as a binary classification problem rather than unsupervised clustering. GCBD is trained on a few clean samples and has been demonstrated across multiple datasets and attacks.

### Strengths
The paper tackles an important problem, which is detecting poisoned data while reducing dependency on large clean datasets.

The proposed defense is evaluated across both static and dynamic triggers.

### Weaknesses
**1. Conceptual and methodological clarity**

The overall pipeline is unclear and conceptually confusing. The paper lacks a coherent and complete description of how GCBD operates end-to-end. Starting with Algorithm 1, the pseudocode does not clearly specify the algorithm's final output, how the so-called "sequence dataset" (D) is constructed and used during training, or how the resulting model generates the final detection scores. It is also unclear what model is being trained in Algorithm 1. Is it the LSTM-based classifier mentioned elsewhere in the text? Furthermore, the paper does not explain how the extracted "scores" or "sequences" from the victim model are transformed into binary detection decisions.
In addition, the defense pipeline itself, including whether it is applied once or iteratively during the sanitization process, is never explicitly defined. Even Figures 1 and 2, which are meant to illustrate the overall framework, are visually dense and fail to convey a clear flow of information. In particular, it remains unclear how the disturbance generation, labeling, and classification stages interact within the proposed mechanism.

**2. Missing source code and reproducibility**

To further complicate understanding of the proposed framework, the authors do not share their code, which could clarify the pipeline and the rationale for many key experimental details (e.g., hyperparameters, data generation specifics, trigger parameters) that are missing.

**3. Theoretical claims without proofs**
Although the paper repeatedly claims to “theoretically demonstrate” the Disturbance Immunity property, the provided “Observations 1 and 2” are actually assumptions, not proofs. In particular, no formal derivation or theorem is provided beyond high-level equations that restate assumptions. Hence, the theoretical contribution is misrepresented. These are empirical intuitions, not proven results. The theoretical contribution claim is thus false and overstated. 

**4. Weak experimental gain**

The paper shows limited or marginal improvements over strong baselines such as TeCo and AC when considering dynamic or unseen triggers (e.g., Input-Aware attack). The “cross-attack” claim is therefore overstated.
5. Experimental inconsistencies and weak justification
The poisoning rates used across attacks are inconsistent (ranging from 0.05% to 8%) without justification or tuning rationale. It is unclear whether these choices are fair or matched to prior baselines.
The proposed “test-time detection” setup (Section 4.2) is poorly explained: when does the defender have access to the test images? How does this differ from the prior evaluation in Section 4.1? The purpose of this stage is ambiguous, and it remains unclear whether it represents a realistic deployment scenario.

**5. Questionable novelty**

The claimed “new perspective” that backdoor effects rely on trigger presence rather than image content is not new. This is already well established in the backdoor literature (e.g., BadNets, SentiNet). Thus, the conceptual novelty of “Disturbance Immunity” is limited; it repackages a known observation with new terminology but without new theoretical or empirical depth.

**Minor**:  Figure 3 is missing the x and y-axis description. 

**Actionable Points**
Here is a list of actionable points for the authors to improve the paper from my perspective:

- Provide a precise, formal, and algorithmic description of the GCBD framework.
- Reframe Observations 1 and 2 as hypotheses or assumptions and remove any claim of theoretical proof unless formal derivations are added.
- Ensure consistent experimental settings across all baselines, and include more detailed hyperparameter choices.
- Explain when test-time detection is applicable, its difference from training-phase detection, and whether it represents a realistic use case.

### Questions
How is the defense classifier trained?

Is the proposed defense applied once to the dataset for sanitization, or iteratively?

How were poisoning percentages chosen across attacks?

How does GCBD handle cases where new triggers differ structurally from those used in training? The Input-Aware results suggest no competitive advantage over the other approaches at the state of the art. 

Since Observations 1–2 are assumptions, not formal proofs, can the authors provide any empirical ablation to support these claims?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposed a backdoor detector based on the disturbance immunity of triggers. Specifically, they give two assumptions: 1) disturbance immunity across classes: any original sample from different classes, the poisoned version’s feature close to a fixed vector; and 2) disturbance immunity across intensities: for any sample from a same class, the poisoned samples’ feature can be divided to a fixed vector and a distortion with tiny norm than the fixed one.

### Strengths
Strength:

1.	Systematical comparison: The authors evaluate their method with many existing defenses, by checking the TPR, FPR of the poison detection.

2.	Theoretical analysis: They formally argue the so-called disturbance immunity in a formal way.

### Weaknesses
Weakness:

1.	It is a little bit hard to follow this paper and the writing should be improved significantly.

2.	In Section 3 the proposed work, the authors first talk about the basic knowledge on the trigger injection, which is not originally proposed by the authors. I suggest moving this part to the preliminary. 

3.	For the disturbance immunity across intensities, the picture 2 show the intensity is the blending ratio of the trigger signal, but in observation 2 I don’t find any intensity definition. According to what the authors said, the observation 2 is very similar as observation 1 except limiting the x from one class.

4.	In Equ 10, the authors define the perturb_x:=(1-m)x and then in Equ 11 and 12, they said the (1-m) will not affect the prediction. It is possible right when x\approx 0 but once when it close to 1, this claim will definitely wrong.

5.	The authors claim the LSTM is trained as poison detector, but in section 3, I don’t see any description on training. Moreover, the definition on the ‘high dimensional sequences’ are missing. 

6.	Lack of the extension: the LSTM analyzes the features to determine whether a sample is poisoned or not. This means the trained LSTM only work for a fixed feature space. In real cases, they feature space are various with the change of the architecture.

7.	For the Naricissus attack, people have proved it more like an adversarial example attack instead of backdoor.

### Questions
See the previous weakness

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduced a General Cross-attack Backdoor Detector (GCBD), which provided a simple and efficient new direction for cross-attack backdoor detection through the unified perspective of disturbance immunity. The paper not only verifies on the traditional training set detection, but also designs a more practical test-time detection scenario. The experiments prove that although the detector is only trained on the most basic badNets/blend attacks, it can successfully detect multiple unseen attacks.

### Strengths
1. The paper transform unsupervised clustering detection into a supervised binary classification problem. Synthesize labeled samples using two conventional backdoors (BadNets, Blend) to train a lightweight LSTM detector to distinguish stable under perturbation. In this way, defenders do not need any attack prior or large-scale clean data, and can train a universal detector with only a small number of samples.
2. The accuracy and TPR of GCBD demonstrated high stability during both the training and testing phases

### Weaknesses
1. There is lack of experiment or theory to demenstarte the proposed two observations. Observation 1 is direct assumption, not derivation.  And this does not always hold true in fact. For example: 1) Input-aware attacks generate triggers of different styles (dynamic triggers) for different inputs, triggers may not be the same in the feature space 𝑣∗. 2)Multi-objective backdoors (jumping from different source classes to different target classes) clearly do not satisfy the same 𝑣∗. 
2. The paper claim that  they theoretically demonstrate that benign and backdoor features exhibits ignificant classification probability discrepancies across varying perturbations of clean image classes and intensities, but there is no lower bound of any margin is used to quantify this separability. 
3. there is a lack of discussion about the difference between GCBD and existing defense methods, such as STRIP.

### Questions
1. The paper only discussed all-to-one backdoor attacks. Can GCBD still work in all-to-all backdoor attacks? The proposed perturbation immunity hypothesis suggests that if each trigger corresponds to a different target label, this unified target class adsorption effect will be broken, so that GCBD may no longer be applicable?
2. How about the adaptive attacks? The paper is presented as a defense, but there is no discussion or analysis about adaptive attacks.
3. The method is only evaluated on low-resolution datasets and small models. This may limit the method's interest to a broader audience. How about the performance of GCBD on ImageNet and ViT?

### Soundness
2

### Presentation
2

### Contribution
2
