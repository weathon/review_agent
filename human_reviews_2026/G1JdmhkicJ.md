# Towards Privacy-Guaranteed Label Unlearning in Vertical Federated Learning: Few-Shot Forgetting Without Disclosure

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
This paper addresses the critical challenge of unlearning in Vertical Federated Learning (VFL), a setting that has received far less attention than its horizontal counterpart. Specifically, we propose the first method tailored to *label unlearning* in VFL, where labels play a dual role as both essential inputs and sensitive information. To this end, we employ a representation-level manifold mixup mechanism to generate synthetic embeddings for both unlearned and retained samples. This is to provide richer signals for the subsequent gradient-based label forgetting and recovery steps. These augmented embeddings are then subjected to gradient-based label forgetting, effectively removing the associated label information from the model. To recover performance on the retained data, we introduce a recovery-phase optimization step that refines the remaining embeddings. This design achieves effective label unlearning while maintaining computational efficiency. We validate our method through extensive experiments on diverse datasets, including MNIST, CIFAR-10, CIFAR-100, ModelNet, Brain Tumor MRI, COVID-19 Radiography, and Yahoo Answers demonstrate strong efficacy and scalability. Overall, this work establishes a new direction for unlearning in VFL, showing that re-imagining mixup as an efficient mechanism can unlock practical and utility-preserving unlearning. The code is publicly available at https://github.com/bryanhx/Towards-Privacy-Guaranteed-Label-Unlearning-in-Vertical-Federated-Learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a pioneering investigation into the critical yet under-explored problem of label unlearning in Vertical Federated Learning (VFL). To address the severe privacy leakage risks of directly applying existing unlearning methods in VFL, the authors propose a novel few-shot, privacy-guaranteed unlearning framework. The core innovation lies in repurposing Manifold Mixup as a privacy mechanism to generate augmented embeddings that disguise the original label information. This is followed by a gradient-ascent-based forgetting step on these embeddings and a recovery phase to maintain model utility on the retained data. Extensive experiments across diverse image and text datasets demonstrate that the method achieves effective unlearning, strong utility preservation, and significant computational efficiency, establishing a new, practical direction for machine unlearning in VFL.

### Strengths
1. This is the first work to systematically define and address the challenge of label unlearning within the VFL paradigm.

2. The proposed method successfully removes the influence of target labels from both active and passive models without disclosing the identity of the labels to be forgotten, thereby mitigating privacy leakage risks inherent in unlearning approaches.

3. The method's efficacy is validated on a wide range of datasets, including images and text. This proves it is a modality-agnostic framework that operates effectively on embedding spaces, enhancing its practical applicability.

4. The proposed unlearning process is efficient, which demonstrates a 16x to 1200x speedup compared to the closest baseline and traditional retraining methods, respectively.

### Weaknesses
1. The theoretical guarantee of unlearning effectiveness relies on the strong assumption that the training loss is near zero, which may not hold perfectly in practice. Furthermore, the method's requirement for a small set of labeled public data might be difficult to fulfill in strict privacy scenarios where no such data can be disclosed.

2. The experimental validation is conducted in a synchronous VFL environment. Its effectiveness and stability in asynchronous VFL settings remain unexplored and pose a potential limitation for practical deployment.

3. The paper does not provide a comprehensive sensitivity analysis for key hyperparameters, such as the distribution of the Mixup coefficient (λ). The performance robustness under different hyperparameter choices is therefore not fully understood.

4. The security analysis is confined to the semi-honest adversary model. The proposed method lacks defensive mechanisms against malicious adversaries who could potentially sabotage the unlearning process by poisoning local embeddings or gradients.

### Questions
1. How would the method perform if the initial model is not well-converged (i.e., the training loss is not near zero)? What adaptations are needed to ensure robustness under such non-ideal conditions?

2. Can the proposed framework be effectively adapted to function in an asynchronous VFL environment? What modifications to the protocol would be required to maintain its efficacy and privacy guarantees under such conditions?

3. How sensitive is the unlearning performance to key hyperparameters, particularly the Mixup coefficient (λ)? Is there an optimal strategy for selecting or adapting λ during the unlearning process?

4. How can the framework be fortified against a malicious adversary who intentionally submits corrupted embeddings or gradients to prevent unlearning or degrade the global model's performance?

5. How does the method scale with an increasing number of labels to be unlearned sequentially? Does it suffer from catastrophic forgetting of retained knowledge or cumulative performance degradation over multiple unlearning requests?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the underexplored problem of label unlearning in Vertical Federated Learning (VFL), where the goal is to erase the influence of specific labels while maintaining privacy between active (label-holding) and passive (feature-holding) parties. The authors first identify a privacy leakage risk in existing retraining-based and boundary-based unlearning methods, which require explicit identification of samples tied to deleted labels. To address this, the paper introduces a few-shot label unlearning framework that re-purposes manifold mixup as a privacy-preserving embedding transformation, enabling both label removal and privacy protection without direct label disclosure. A recovery phase is added to restore performance on the retained data. Experiments on multiple datasets demonstrate that the proposed method achieves competitive unlearning and runtime efficiency while reducing label leakage compared to baselines.

### Strengths
- **Novel problem formulation.** The work clearly identifies a meaningful and underexplored setting, label unlearning in VFL. This is a distinct and practically relevant direction compared to the more common client or sample unlearning tasks.

- **Clear motivation and privacy analysis.** The authors systematically highlight the privacy risks of conventional unlearning in VFL, showing that retraining-based approaches lead to full label leakage. This analysis is well-motivated and establishes a strong need for privacy-aware unlearning mechanisms.

### Weaknesses
- **Limited algorithmic novelty and depth.** The method primarily combines two known ideas, manifold mixup and gradient ascent unlearning, without introducing fundamentally new mechanisms for privacy preservation or optimization. The theoretical component (Theorem 1) formalizes intuitive gradient alignment but does not establish strong guarantees of privacy or unlearning completeness.

- **Lack of formal privacy guarantees.** Although the paper claims “privacy-guaranteed” unlearning, there is no rigorous privacy analysis (e.g., differential privacy bounds or information-theoretic leakage quantification). The argument relies on heuristic intuition and empirical leakage reduction, which does not substantiate the “guarantee” claim in the title.

- **Reliance on public data undermines practicality.** The approach assumes access to a small public labeled dataset that contains the same label categories as the private data. This assumption is unrealistic in many sensitive domains (e.g., healthcare or finance), where such public analogs may not exist.

- **Overstated experimental superiority.** Although results show good accuracy retention and unlearning effectiveness, the differences from SSD or Amnesiac baselines are not statistically or conceptually large. For example, in several datasets (Brain MRI, COVID Radiography), the proposed method’s ASR improvements are marginal relative to simpler baselines.

- **Insufficient clarity on communication protocol and security.** The paper claims privacy preservation by only exchanging gradients on mixed embeddings, but lacks detailed analysis of what information could still be inferred from these gradients. Without a formal adversary model or leakage bound, the privacy benefit remains speculative.

- **Over-extended empirical scope with limited insight.** While the paper reports numerous datasets and metrics, it lacks deeper diagnostic analyses, such as ablation on mixup parameters (λ), effect of public data size, or visualization of embedding disentanglement that could substantiate the mechanism’s behavior.

### Questions
Please see the weakness section.

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
4

### Summary
This paper studies the problem of machine unlearning in the setting of vertical federated learning, where features and labels are owned by 2 different parties, and only a set of labels need to be unlearned. The proposed unlearning method consists of 3 steps: 1) use manifold mixup to generate sufficiently many samples for the unlearning algorithm; 2) run a vertical gradient ascent algorithm; 3)recover the accuracy on the remained data by re-training on these samples. Experiments are conducted to show the effectiveness of the proposed method.

### Strengths
- the studied problem is important
- experiment results look promising
- compared to a previous draft in Neurips 2025, a new SSD baseline is included in the experiments, which strengthens the experimental evaluation .

### Weaknesses
My primary and most significant concern with this work, which I also raised as a reviewer for a previous submission of this paper to NeurIPS 2025, is a fundamental ambiguity in the threat model. This core issue, which remains unaddressed, makes the paper's central motivation and claims inconsistent. 

In Section 3.2, the authors tried to argue that retraining leaks 100% membership information. But for standard machine unlearning, retraining always serves as a baseline, as the goal of unlearning is just to produce a model as if the unlearned data is never used. It appears the authors are implicitly redefining the unlearning goal: not only must the model be unlearned, but the passive party must also be kept ignorant of which data is being unlearned. This is a "process privacy" goal, which is a valid but very different objective from standard unlearning. This new goal is never rigorously defined.

Even if we accept the authors' new "process privacy" goal, they provide no evidence that their own method achieves it. In the propose method gradients are shared with the passive party. Gradients are well-known to leak information about the data they were computed on. The authors provide no proof or analysis that these gradients are any safer than an explicit unlearning request. They simply assume their method is private.

### Questions
The paper's narrative structure is disjointed. Section 3.2, which analyzes label leakage, is placed before the proposed method is introduced in Section 4. This is confusing, as it discusses the privacy properties of an algorithm the reader has not yet seen. This premature justification breaks the logical flow of the paper and makes the draft difficult to follow. The content requires polishing to present the problem, the proposed solution, and then the analysis in a more coherent order.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper looks into label unlearning in vertical federated learning (VFL). Standard retraining or boundary-unlearning methods leak which labels were deleted to passive parties. The proposed method uses a few-shot public dataset (≤40 samples per label), applies manifold mixup at the embedding level across passives, runs gradient ascent on mixed embeddings to push models away from forgotten labels, then performs small descent on retained labels for recovery.

### Strengths
1) VFL unlearning seems a rather underexplored area in machine learning so targets a relative problem. 
2)  Operates on concatenated embeddings with tiny public sets (≤40 samples), the extension of manifold mixing is very interesting.

### Weaknesses
- The claims about no deletion guarantees are fine and measured so while some questions can be raised about the applicability of the algorithm (especially how diverse the features are  with every client and how does that impact unlearning) I am hesitant to raise those questions given my unfamiliarity of the recent challenges in VFL.

- However,  my main concern remained here as to what extent the features of the few shot labels  are correlated with the features of the unlearned labels ( is there a sufficient bound of alignment of features of the few shot for effective unlearning). If there is weak correlation the algorithm would fail right? since the method relies that the "public" labels provide a valid direction. If the method would still succeed then that would strengthen then that would be very good. Would appreciate if the authors can provide some details

### Questions
Weaknesses and Questions merged

### Soundness
3

### Presentation
3

### Contribution
3
