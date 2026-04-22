# Partition-Losses Fine-Tuning: Contamination-Robust Backdoor Unlearning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Large-scale training data and third-party checkpoints make training convenient but also leave room for poisoning-based backdoor attacks. These attacks embed a backdoor through data poisoning in the training set: the infected model behaves normally on clean inputs but predicts an attacker-chosen label whenever the trigger appears. The stealthiness poses risks for security-sensitive deployment and model reuse. Post-training fine-tuning has become a practical default defense as it is computationally efficient and does not require control over the original training pipeline. 
However, existing fine-tuning methods rely on a clean set to unlearn the backdoor indirectly. This assumption is fragile in reality: curation errors or undetected triggers can contaminate the "clean" set. As a result, state-of-the-art clean-only fine-tuning often fails to purify the backdoor behavior while maintaining the original functionality. We propose Partition-Losses Fine-Tuning (PL), a simple, architecture- and domain-agnostic loss modification that leverages both mostly clean and flagged mostly malicious samples. PL jointly minimizes benign loss and maximizes target-class loss, explicitly pushing the model away from the implanted trigger-to-target association. Comprehensive experiments show that PL matches or surpasses clean-only fine-tuning methods under the same computational budget while halving the required clean samples. Crucially, PL remains effective under realistic contamination of both fine-tuning sets and is stable across hyperparameter choices and data availability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose Partition-Losses Fine-Tuning (PL), a simple, architecture- and domain-agnostic loss modification that leverages both clean and flagged malicious samples in DNN to mitigate backdoor threats. PL jointly minimizes benign loss and maximizes target-class loss, explicitly pushing the model away from the implanted trigger-to-target association.

### Strengths
The paper is well-structured, clearly organized and presented.

### Weaknesses
1.Unrealistic Threat Model: The paper is partially motivated on the realism of the assumption that tuning on clean data to eliminate backdoors. However, while the PL method reduces the required number of clean samples by half and allows for contamination, it still necessitates backdoor-labeled malicious samples. How is this assumption ensured in practice?

2.Missing Technical Details -- Collection of $D_m$: The method mentioned for "flagging malicious inputs with the target class involves real-time monitoring and incident-response pipelines". What specifically does this entail? Additionally, I did not see theoretical validation or ablation studies regarding the scales of $D_m$ and $D_b$, which I believe are crucial for the implementation of PL and partly determine the sophistication of the objective function.

3.Missing Technical Details -- |$D_m$|=0: Although line 171 mentions, “Note that PL reduces exactly to clean-only fine-tuning when no flagged triggered samples are available,” this method is not discussed further. What does this method entail, and how's its performance? 

4.Theoretical Analysis: There is a lack of discussion regarding $D_m$ and $D_b$, along with upper and lower bound validations.

5.Evaluation of DNN Tasks: For text and tabular data, as well as classification and regression tasks, the methods of implanting triggers and training backdoors differ. I would like to see implementation modifications of PL across these DNN tasks and the corresponding results.

6.Choice of Evaluation Settings: The methods used in the attack settings lacks the reflection of SOTA approaches, and there is insufficient detailed description of the implementations of Adaptive Blend and SSBA methods.

7.Insufficient Discussion on the Impact of Different $\epsilon$ on PL Performance: The discussion is limited to cases where $\epsilon_b$ and $\epsilon_m$ are both 0.1. However, according to the proof section, “when $\epsilon_m$ < 1/2 , such an alpha always exists.” The boundary conditions and strategies for selecting alpha under different $\epsilon$ values are not discussed. Additionally, why is it claimed in line 431 that both $\epsilon$ = 0.1 is “realistic contamination”? What is the basis for this statement?

8.Outdated Related Work: In the related work section, the citations regarding fine-tuning range from 2018 to 2023 and do not include recent SOTA contributions.

### Questions
1.What do the real-time monitoring and incident-response pipelines for flagging malicious inputs with the target class specifically entail? What are their success and false positive rates?

2.Can the authors further discuss the scale or ratio of $D_m$ and $D_b$, including theoretical derivation and empirical evidence?

3.What is the methodology of the deployed attack methods in the evaluation setting? The baseline defense methods perform poorly in this context; under what settings were the experiments conducted in the papers they are proposed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Partition-Losses Fine-Tuning (PL), a simple and robust post-training defense against poisoning-based backdoor attacks. Unlike traditional fine-tuning methods that assume completely clean data, PL leverages both benign and flagged malicious samples by minimizing the loss on clean data while maximizing the loss on triggered samples. This dual-objective design explicitly unlearns the trigger-to-target association, making the model more resistant to contamination. Experiments across multiple datasets and attack types show that PL consistently achieves near-zero attack success rates with competitive clean accuracy, even when the fine-tuning data is partially contaminated. The method is lightweight, architecture-agnostic, and practical for real-world deployment in security-critical applications.

### Strengths
The main strength of Partition-Losses Fine-Tuning (PL) lies in its robustness, simplicity, and effectiveness. It performs well even with contaminated fine-tuning data, requires no access to the original training set, and is easy to implement. PL achieves near-zero attack success rates while maintaining high clean accuracy across different models and attacks, making it a practical and generalizable defense for real-world applications.

### Weaknesses
The main weakness of Partition-Losses Fine-Tuning (PL) is its dependence on prior knowledge of the backdoor trigger and target label, as well as its vulnerability to data partitioning errors.

PL assumes the defender can identify or flag a small set of triggered samples and their associated target class to form the malicious pool \( D_m \).  
However, in realistic scenarios, this assumption is often unrealistic — triggers and target labels are typically unknown or mislabeled.  
When the partition between the benign pool \( D_b \) and malicious pool \( D_m \) is inaccurate (e.g., due to high false-positive or false-negative rates), the joint optimization objective can become misleading, leading to:

1. Incomplete backdoor removal — residual trigger-target associations may persist.  
2. Clean accuracy degradation — benign knowledge may be unintentionally unlearned.  

Additionally, the trade-off parameter \( \alpha \), which balances clean accuracy and backdoor forgetting, is static and sensitive:
1. A large \( \alpha \) can severely harm benign accuracy.  
2. A small \( \alpha \) may fail to erase the backdoor.  

Compared with recent adaptive or self-supervised defenses (e.g., exposure-based or gradient inversion methods), PL lacks dynamic control and trigger-agnostic robustness.

### Questions
1. **Trigger Identification**  
   - How can PL be applied when the defender has *no access* to labeled triggered samples or the target class?  
   - Can the method be extended to *automatically detect* or *estimate* potential triggers?

2. **Partition Robustness**  
   - How sensitive is PL to mislabeling or contamination in the benign/malicious pools?  
   - What happens when the contamination rate exceeds the tested threshold (e.g., >10%)?

3. **Parameter Sensitivity**  
   - How should the trade-off coefficient \( \alpha \) be chosen or adapted during training?  
   - Can an *adaptive α-scheduler* improve stability between backdoor forgetting and clean accuracy?

4. **Generalization to Unknown Attacks**  
   - Does PL remain effective against *adaptive* or *unknown trigger* attacks that were not seen during fine-tuning?  
   - How does it perform on *semantic* or *dynamic* triggers instead of static patterns?

5. **Comparison with Modern Defenses**  
   - How does PL perform compared to *exposure-based*, *representation-level unlearning*, or *causal filtering* defenses introduced in 2024–2025?  
   - Can PL be combined with these methods to form a hybrid, more resilient defense?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a post-training backdoor defense method that leverages both benign and maliciously labeled samples. Instead of discarding suspicious data, the method formulates distinct loss objectives for benign and malicious samples, introducing a “forgetting signal” to remove trigger-related representations. The approach is theoretically grounded and experimentally validated on several benchmark datasets.

### Strengths
1.The idea of actively forgetting triggers via loss optimization, guided by labeled malicious samples, is original and contrasts well with existing “sample rejection” paradigms.

2.The paper provides formal proofs that justify the convergence and effectiveness of the proposed mechanism.

3.Clear motivation: The authors identify a real-world issue—data contamination in fine-tuning—and propose a method that attempts to address it in a principled way.

### Weaknesses
1.Semantic inconsistency: The paper first claims that a clean fine-tuning set is unrealistic due to possible contamination, but later assumes access to “clean samples.” The notion of “clean data” thus requires clarification—perhaps as “a mostly clean but partially contaminated sample pool.”

2.Limited deployment realism: The proposed defense assumes post-hoc identification of malicious inputs through monitoring or human response. This makes it a reactive mechanism, which may not prevent immediate harm in real-world attacks.

3.Narrow experimental scope: Experiments cover only single-target, explicit-trigger backdoor settings. The method’s performance under multi-target, multi-trigger, or stealthy (implicit) attacks remains unexplored.

### Questions
1.How robust is the proposed method when the “clean” pool is partially contaminated?

2.Could the defense mechanism be adapted to an online or proactive setting, rather than relying on post-attack detection?

### Soundness
3

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
While existing backdoor defense finetuning assumes access to an absolutely clean set, this is not realistic in practice. Thus, the authors propose an algorithm to take potentially contaminated data as inputs to achieve promising performance. 
The authors first design a partition loss that takes potentially contaminated benign and malicious datasets, and correspondingly designed an algorithm partition loss finetuning that optimizes the partition loss. PLFT is proved to be effective through comprehensive evaluations.

### Strengths
(1) The problem is well motivated.

(2) The partition loss is elegantly defined, and the partition loss finetuning algorithm is designed to effectively lowers the loss. It is also well proved through theory and supported by experiment results.

(3) The evaluation  is comprehensive.

### Weaknesses
My major concern on this paper is the correlation between \alpha and contamination rates in both benign and malicious set - intuitively I believe there's a correlation among them. Specifically, I think the paper is more complete if:

(1) The correlation of the parameters is proved theoretically (you can probably add a subsection in the method section)

(2) The correlation is studied in the evaluation.

### Questions
See Weakness section.

### Soundness
3

### Presentation
4

### Contribution
4
