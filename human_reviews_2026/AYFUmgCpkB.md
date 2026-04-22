# Zero-Sacrifice Persistent-Robustness Adversarial Defense for Pre-Trained Encoders

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
The widespread use of publicly available pre-trained encoders from self-supervised learning (SSL) has exposed a critical vulnerability: their susceptibility to downstream-agnostic adversarial examples (DAEs), which are crafted without knowledge of the downstream tasks but capable of misleading downstream models. While several defense methods have been explored recently, they rely primarily on task-specific adversarial fine-tuning, which inevitably limits generalizability and causes catastrophic forgetting and deteriorates benign performance. Different with previous works, we propose a more rigorous defense goal that requires only a single tuning for diverse downstream tasks to defend against DAEs and preserve benign performance. To achieve this defense goal, we introduce **Ze**ro-Sacrifice **P**ersistent-Robustness **A**dversarial **D**efense (**ZePAD**), which is inspired by the inherent sensitivity of neural networks to data characteristics. Specifically, ZePAD is a dual-branch structure, which consists of a Multi-Pattern Adversarial Enhancement Branch (MPAE-Branch) that uses two adversarially fine-tuned encoders to strengthen adversarial resistance. The Benign Memory Preservation Branch (BMP-Branch) is trained on local data to ensure adversarial robustness does not compromise benign performance. Surprisingly, we find that ZePAD can directly detect DAEs by evaluating branch confidence, without introducing any adversarial exsample identification task during training. Notably, by enriching feature diversity, our method enables a single adversarial fine-tuning to defend against DAEs across downstream tasks, thereby achieving persistent robustness. Extensive experiments on 11 SSL methods and 6 datasets validate its effectiveness. In certain cases, it achieves a 29.20\% improvement in benign performance and a 73.86\% gain in adversarial robustness, highlighting its zero-sacrifice property.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ZeLAD, a “Zero-Sacrifice Lifelong Adversarial Defense” framework that integrates multiple adversarial and benign branches to achieve both clean accuracy and robustness without further tuning. The method designs a Robust Federal Decision Mechanism (RFDM) to adaptively weight branch predictions based on confidence, aiming to provide a single-tuning defense applicable across tasks. Experiments are conducted on multiple datasets and architectures.

### Strengths
1. The experiments are relatively comprehensive, covering several datasets, model scales, and attack types.
2. The paper is logically organized, and most sections follow a clear structure.
3. The proposed method is systematic and addresses the problem of balancing robustness and generalization in a targeted manner.

### Weaknesses
1. Core claim overstatement: The claim of “lifelong + zero-sacrifice with single tuning” is not well supported. Although the paper emphasizes a one-time tuning process, in practice each new downstream task still requires training local classifiers and possibly branch adaptation. The experiments only test cross-dataset transfer, not true lifelong learning without any retraining, making the central claim over-stated.
2. Over-packaged novelty: The technical novelty is limited. ZeLAD essentially combines multiple encoder branches with a handcrafted confidence-weighted ensemble (RFDM). The exponential weighting is heuristic, and there is no comparison with simpler ensemble or calibration baselines. The contribution is more about system integration than a fundamentally new defense principle.
3. Experimental and analysis issues:
    - It is suggested to include a comparison with a three-encoder average ensemble or PGD-AT baseline to validate RFDM’s effectiveness.
    - Hyperparameter choices (e.g., weighting coefficients, confidence scaling) and sensitivity analysis are not explained.
4. Writing and presentation problems:
    - Figure 2 should be moved to Section 3.2.2.
    - Eqs. (5) and (6) are unclear in mathematical form and explanation.
    - Unify terminology: use adversarial example consistently.
    - The introduction mentions RFDM, but later sections use inconsistent terms.
    - The PGD-AT citation is incorrect; it should be *“Towards Deep Learning Models Resistant to Adversarial Attacks”*, ICLR 2018, not the CW attack paper.
5. Mathematical writing issues:
    - Use calligraphic letters (e.g., $\mathcal{D}$) for datasets, not $D$; and use $D$ for distance.
    - Keep notations of encoder and classifier consistent ($E,F$ in Section 3.1 while $\mathcal{E},\mathcal{F}$ in Section 3.2).
    - Avoid blank lines after \begin{equation}.
    - Rewrite Eqs. (5) and (6) with explicit meaning of each symbol.
    - Annotate dimensionality for key notations if possible.
    - For clarity, it is suggested to include corresponding notation for each component in Figure 1.
6. Typos and minor language errors:
    - “generalibility” → “generalizability” (p.1 l.16)
    - “differ with” → “different from” (p.1 l.18)
    - “... uses two” → “that uses two” (p.1 l.23)
    - “task-specfic” → “task-specific” (p.2 l.75)
    - “taht” → “that”; “branche” → “branches” (p.9 l.446)
    - Algorithm 1 needs a thorough proofreading.
7. Related work limitations: The related work section is short and somewhat outdated. Please include more recent adversarial defense papers, such as:
    - *Probabilistic Margins for Instance Reweighting in Adversarial Training*, NeurIPS 2021.
    - *DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain*, NeurIPS 2024

### Questions
See Weaknesses. If the authors address my concerns, I am willing to raise the score.

### Soundness
3

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
4

### Summary
This paper addresses the vulnerability of self-supervised learning (SSL) models to downstream-agnostic adversarial examples (DAEs). The authors propose Zero-Sacrifice Lifelong Adversarial Defense (ZeLAD), a dual-branch framework designed to achieve adversarial robustness without sacrificing benign performance. ZeLAD integrates a Multi-Pattern Adversarial Enhancement branch and a Benign Memory Preservation branch to balance robustness and benign performance. It detects DAEs by evaluating branch confidence, eliminating the need for adversarial sample training. Extensive experiments on 11 SSL methods and 6 datasets show substantial improvements in both benign accuracy and adversarial resistance, validating its “zero-sacrifice” property.

### Strengths
1. The paper introduces ZeLAD, the first lifelong adversarial defense for pre-trained encoders that achieves robustness across multiple downstream tasks with a single tuning. Unlike prior task-specific adversarial training methods, ZeLAD generalizes effectively across SSL models and datasets, marking a substantial conceptual advancement in adversarial robustness research.
2. A major strength is ZeLAD’s dual-branch architecture: the Multi-Pattern Adversarial Enhancement (MPAE) branch for robustness and the Benign Memory Preservation (BMP) branch for maintaining clean-sample accuracy. This design enables the model to enhance adversarial defense without degrading benign performance, a key limitation of previous methods.

### Weaknesses
1. Although the paper compares ZeLAD to several classic defenses (e.g., TRADES, MART, Gen-AF), it does not include enough comparisons with the most recent or SSL-specific adversarial defense methods (only Table 7). This omission makes it harder to gauge ZeLAD’s relative progress within the latest research landscape.
2. The proposed approach requires multiple encoders and dual-branch inference, which could increase computational and memory overhead compared to single-encoder defenses. The paper provides limited discussion or quantitative evaluation of training/inference efficiency. I think it is very important for real world deployment. 
3. Although the paper claims “lifelong” robustness, the experiments only cover a limited number of sequential tasks. There is no long-term continual learning evaluation (e.g., over dozens of tasks or domain shifts), so the claim of lifelong adaptation remains somewhat speculative.

### Questions
1. The paper introduces a weighting parameter $\lambda$ in the loss function (Eq. 3), but its value or tuning procedure is not specified. 
2. Some hyperparameters (e.g., lambda, learning rate) are not explicitly stated.
   Will the authors release code or configuration files to ensure experimental reproducibility?
3. The paper uses a dual-branch architecture with independent encoders. Have the authors considered some techniques to reduce redundancy while preserving robustness?

### Soundness
2

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
5

### Summary
This paper introduces ZeLAD, a lifelong defense framework that protects self-supervised encoders from downstream-agnostic adversarial examples (DAEs) with a single tuning. ZeLAD employs dual branches to enhance adversarial robustness while preserving benign performance, and can even detect DAEs via branch confidence without explicit training.

### Strengths
1、The paper proposes a zero-sacrifice, lifelong adversarial defense method that not only maintainsbut also improves benign performance, while enhancing adversarial robustness.

2、Extensive experimental results demonstrate the effectiveness of the proposed method.

3、The paper is easy to follow.

### Weaknesses
1、The paper claims to build on the inherent sensitivity of neural networks to data characteristics, yet this idea is only briefly mentioned in the introduction (L54–57) without deeper investigation. No experimental validation, theoretical analysis, or concrete insight is provided to support this claim, which substantially undermines the rationale and validity of the proposed method. A more thorough analysis through exploratory experiments or theoretical justification is necessary.

2、The first and second claimed contributions both emphasize lifelong adversarial defense, which appear conceptually identical. I suggest merging them for conciseness and clarity.

3、The paper inconsistently uses adversarial sample and adversarial example. The terminology should be unified using the widely accepted adversarial example to maintain professional consistency.

4、The Related Work section remains too high-level and lacks discussion of key concepts (e.g., “Pre-trained encoders”， “pre-trained paradigm”) as well as core algorithmic ideas of representative methods—such as self-supervised learning approaches (SimCLR, MoCo) and DAEs on pre-trained encoders (PAP, AdvEncoder).

5、The Threat Model is essential for any security-related study and should appear in the main text rather than the appendix. Similarly, the discussion of challenges addressed by the proposed method would be better placed before the methodology section for improved logical flow.

6、The reported RA results (Tables 2 and 4) show large discrepancies compared to baseline values, raising concerns about correct reproduction of baseline methods. The causes of these gaps should be clarified.

7、The proposed method demonstrates higher time and storage costs than the baselines (Table S3, L899–916).

8、The overall writing quality requires improvement, particularly in table captions, many of which contain grammatical errors. For instance, “Table 1: BA(%) Baseline vs. ZeLAD in the semi-black-box scenario” is ungrammatical and should be revised for correctness and clarity.

### Questions
Given that the proposed method is claimed to build on a general property of deep neural networks — “neural networks inherently exhibit higher confidence in inputs that resemble the training data, a behavior attributed to the memorization of the data’s characteristics” — two questions arise.

1. Whether the method can be applied to multimodal pre-trained models such as CLIP or BLIP remains unclear. Experiments are needed to verify its scalability.

2. It is also unclear whether the method can generalize beyond image classification, for example to image retrieval, semantic segmentation, or object detection.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **ZeLAD (Zero-Sacrifice Lifelong Adversarial Defense)**, a framework aimed at defending pre-trained self-supervised encoders against **Downstream-Agnostic Adversarial Examples (DAEs)**. Unlike prior task-specific fine-tuning approaches, ZeLAD claims to provide a single, lifelong adversarial defense applicable across diverse downstream tasks while maintaining or improving benign accuracy.

ZeLAD employs a dual-branch architecture:
- MPAE-Branch (Multi-Pattern Adversarial Enhancement Branch): Combines multiple adversarially fine-tuned pre-trained encoders trained with diverse SSL methods to enhance robustness through representational diversity.
- BMP-Branch (Benign Memory Preservation Branch): Trained solely on clean data to preserve benign performance.

During inference, a Robust Federal Decision Mechanism (RFDM) fuses branch outputs by comparing confidence scores. The model also demonstrates the ability to detect adversarial samples based purely on branch confidence disparities, without explicit adversarial detection training.

### Strengths
1. **Novelty and Conceptual Contribution:**
- The paper introduces the idea of “zero-sacrifice lifelong adversarial defense”, reframing adversarial robustness as a feature combination problem rather than a tradeoff problem.
- The dual-branch design (MPAE + BMP) and the federated confidence fusion mechanism are novel and well-motivated.
2. **Comprehensive Empirical Evaluation:**
- Extensive experiments across multiple SSL encoders (e.g., SimCLR, BYOL, MoCo, DINO) and datasets (CIFAR10, ImageNet, STL10, etc.) demonstrate the generality of the method.
- Results consistently show significant improvements over baselines and other defenses (e.g., TRADES, MART, Gen-AF).
3. **Strong Practical Relevance:**
- Addresses a genuine and underexplored problem: the vulnerability of pre-trained encoders to DAEs in a downstream-agnostic setting.
- The claim of “single tuning for all downstream tasks” has significant implications for scalable and resource-efficient deployment.
4. **Adversarial Detection Without Supervision:**
- The method’s ability to detect adversarial samples using confidence asymmetry without explicit training is an elegant byproduct.
5. **Clarity and Organization:**
- The paper is generally well-written and logically structured, with helpful figures (e.g., the overall ZeLAD architecture diagram and confidence distributions).

### Weaknesses
1. **Methodological Clarity and Rigor:**
- While conceptually interesting, some mathematical formulations (e.g., hybrid loss and cosine distance adjustment) lack detailed derivations and theoretical justification.
- The Robust Federal Decision Mechanism (RFDM) is empirically defined, but its weighting function (Eq. 8) seems heuristic and not theoretically grounded.
- There is no formal analysis of why confidence alignment is a robust signal or how it generalizes across tasks.

2. **Evaluation Limitations:**
- Most experiments focus on classification tasks; it’s unclear whether ZeLAD extends to non-classification downstream tasks (e.g., segmentation, detection).
- The adversarial attack diversity could be further improved—results rely heavily on AdvEncoder; other recent black-box attacks are not deeply explored.

3. **Scope of “Lifelong” Claim:**
- The “lifelong” aspect mainly refers to single fine-tuning across multiple tasks rather than continuous adaptation. There is no evidence of incremental task adaptation or continual learning capability, making the “lifelong” terminology somewhat overstated.

4. **Computational Cost and Practicality:**
- Maintaining multiple encoders (two adversarially fine-tuned and one benign) increases inference-time complexity and memory usage, which could hinder scalability.

### Questions
**1. Clarification on the “Lifelong” Claim**
You define ZeLAD as a *“lifelong adversarial defense”* requiring only a single tuning. However, lifelong learning typically implies *continuous adaptation to new tasks* without full retraining.  
**Could you clarify how ZeLAD satisfies the lifelong learning property beyond single multi-task applicability?**  
Have you tested whether ZeLAD can handle incremental task addition or domain shifts without catastrophic forgetting?

 **2. Justification for the “Zero-Sacrifice” Property**
The paper claims that ZeLAD achieves robustness improvements *without sacrificing benign accuracy*.  
**What theoretical or empirical evidence supports this “zero-sacrifice” claim?**  
Does this property hold under stronger perturbation budgets or adaptive attacks that target both branches simultaneously?

 **3. Robustness of the Confidence-Based Fusion (RFDM)**
The Robust Federal Decision Mechanism (RFDM) fuses branch outputs using confidence-based weighting.  
**How reliable is this mechanism under confidence miscalibration, label noise, or adaptive attacks explicitly designed to manipulate confidence distributions?**  
Have you compared this heuristic to alternative fusion schemes (e.g., entropy weighting, temperature scaling)?

 **4. Computational and Practical Efficiency**
ZeLAD employs multiple encoders (two adversarially fine-tuned and one benign), which could increase inference cost.  
**What is the computational and memory overhead of ZeLAD compared to single-encoder fine-tuning methods?**  
Can you comment on its scalability to larger SSL encoders (e.g., ViT-L/16, CLIP) in real-world applications?

~PS: Any chance you want to name the paper ZeLDA: Zero sacrifice Lifelong Defence against Adversaries?~

### Soundness
3

### Presentation
3

### Contribution
3
