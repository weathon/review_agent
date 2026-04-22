# BAPFL: Exploring Backdoor Attacks Against Prototype-based Federated Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Prototype-based federated learning (PFL) has emerged as a promising paradigm to address data heterogeneity problems in federated learning, as it leverages mean feature vectors as prototypes to enhance model generalization. However, its robustness against backdoor attacks remains largely unexplored. In this paper, we identify that PFL is inherently resistant to existing backdoor attacks due to its unique prototype learning mechanism and local data heterogeneity. To further explore the security of PFL, we propose BAPFL, the first backdoor attack method specifically designed for PFL frameworks. BAPFL integrates a prototype poisoning strategy with a trigger optimization mechanism. The prototype poisoning strategy manipulates the trajectories of global prototypes to mislead the prototype training of benign clients, pushing their local prototypes of clean samples away from the prototypes of trigger-embedded samples. Meanwhile, the trigger optimization mechanism learns a unique and stealthy trigger for each potential target label, and guides the prototypes of trigger-embedded samples to align closely with the global prototype of the target label. Experimental results across multiple datasets and PFL variants demonstrate that BAPFL achieves a 33%-75% improvement in attack success rate compared to traditional backdoor attacks, while preserving main task accuracy. These results highlight the effectiveness, stealthiness, and adaptability of BAPFL in PFL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the backdoor vulnerability of Prototypical Federated Learning (PFL). This paper first analyzes PFL's natural resistance to traditional attacks, then proposes BAPFL, the first backdoor attack specifically designed for PFL. BAPFL combines a Prototype Poisoning Strategy (PPS) to manipulate global prototypes via prototype flipping, and a Trigger Optimization Mechanism (TOM) to learn stealthy triggers aligned with target-class prototypes. Experiments show BAPFL achieves a high ASR while maintaining ACC.

### Strengths
1.This paper is the first to systematically study the backdoor security issues in PFL, an important variant of FL, filling a research gap. 

2.The paper fully validates the effectiveness of BAPFL's components and its robustness under various settings through extensive internal comparative experiments.

3.This paper is well written and easy to follow.

### Weaknesses
1. Fails to evaluate BAPFL against SOTA general-purpose backdoor defenses (e.g., FLAME [1]), leading to an insufficient assessment of its actual threat.

2.  Lacks evaluation against more complex, PFL-specific backdoor defense strategies, failing to demonstrate robustness against such targeted defenses.

3.  Lacks comparison against other SOTA attacks (e.g., Chameleon [2], A3FL [3]).   This is needed to benchmark BAPFL's relative strength, even if those attacks require adaptation to PFL.

4. The stealthiness claim relies solely on ACC, overlooking statistical anomaly analysis of the poisoned prototypes themselves.

Reference:

[1] Nguyen et al. FLAME: Taming backdoors in federated learning.USENIX Security’ 2022.

[2] Yanbo Dai, Songze Li. Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning. ICML2023.

[3] Zhang, Hangfan, et al. A3FL: adversarially adaptive backdoor attacks to federated learning. NeurIPS’ 2023.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper investigates security vulnerabilities in Prototype-based Federated Learning (PFL) under backdoor attacks. The authors identify that PFL’s prototype-based learning and data heterogeneity provide inherent resistance to conventional attacks. To address this limitation, the paper introduces BAPFL, a novel attack framework combining a Prototype Poisoning Strategy (PPS)and a Trigger Optimization Mechanism (TOM). PPS manipulates global prototypes to mislead benign clients, while TOM learns stealthy triggers aligned with target prototypes. Extensive experiments on MNIST, FEMNIST, and CIFAR-10 across multiple PFL variants demonstrate that BAPFL achieves a substantial improvement in attack success rate (33%–75%) over traditional backdoor attacks while maintaining comparable main-task accuracy. The results further highlight BAPFL’s robustness against advanced defenses and its adaptability to heterogeneous data settings.

### Strengths
1. Presents the first systematic investigation of backdoor attacks in PFL and identifies its intrinsic resistance to conventional attacks.
2. The integration of PPS and TOM for dual-direction prototype optimization is well-motivated and significantly enhances both attack effectiveness and stealthiness.
3. Appendix I demonstrates that BAPFL introduces minimal computational and communication overhead, suggesting its practical feasibility.

### Weaknesses
1. While ``stealthiness'' is discussed as a key objective, the paper lacks visual examples of trigger-embedded samples. Figure 3 visualizes prototypes but does not illustrate the actual modified inputs seen by benign clients. Including such examples would strengthen the claim of imperceptibility.
2. The statement that robust aggregation defenses exhibit ``limited effectiveness'' against BAPFL would benefit from deeper analysis. A theoretical discussion clarifying *why* prototype manipulation circumvents these defenses would provide stronger insight.
3. Appendix G.3 asserts that BAPFL is relatively insensitive to hyperparameters after normalization, yet this claim relies on limited perturbation testing. Presenting quantitative results (e.g., plots or tables) for small variations of λ₁, λ₂, and λ₃ would substantiate this conclusion.
4. The federated learning setup deviates from standard backdoor attack settings, which typically involve 100–200 clients with partial participation (e.g., 10% per round). The current configuration of 20 clients with full participation is non-standard; clarification and evaluation under typical settings are necessary.
5. The related work section omits discussion of several defense strategies proposed for FL backdoor attacks, particularly outlier-detection-based methods [1, 2]. Evaluation beyond robust aggregation would provide a more comprehensive comparison.
6. Experiments on more complex datasets (e.g., CIFAR-100 or Tiny-ImageNet) would strengthen claims of generalizability across diverse data distributions.
7. The large performance gap in ASR (33%–75%) compared to ``traditional backdoor attacks'' raises concerns about baseline adaptation. The paper should clarify whether baselines were properly adjusted for the PFL context or justify why they cannot be effectively adapted.
8. Minor: The meaning of ``attack rate'' (e.g., proportion of malicious clients) should be explicitly defined in Section 5.1 for clarity.

[1]. Nguyen, Thien Duc, et al. "{FLAME}: Taming backdoors in federated learning." *31st USENIX Security Symposium (USENIX Security 22)*. 2022.

[2]. Rieger, Phillip, et al. "Deepsight: Mitigating backdoor attacks in federated learning through deep model inspection." arXiv preprint arXiv:2201.00763 (2022).

### Questions
Please address the aforementioned weaknesses and clarifications, particularly regarding experimental setup consistency and baseline adaptation.

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
This paper focuses on the security of Prototype-Based Federated Learning (PFL), a paradigm that addresses data heterogeneity in federated learning (FL) by aggregating class prototypes (mean feature vectors) instead of full model parameters. PFL exhibits inherent robustness against existing backdoor attacks, attributed to the limited influence of poisoned prototypes (only affecting the embedding layer of benign models) and client data heterogeneity (some clients lack target label samples, breaking trigger-target mappings). To fill the gap of PFL-specific backdoor attacks, the authors propose BAPFL. Selects high-impact trigger samples via Euclidean distance and flips trigger prototypes to manipulate global prototypes away from trigger prototypes, pushing benign prototypes to diverge from trigger prototypes. Expands the target label space to cover benign clients’ local labels, learns label-specific stealthy triggers, and aligns trigger prototypes with the global prototype of the target label.

### Strengths
1-It addresses a critical, underexplored problem—PFL’s security against backdoor attacks. Prior work primarily focuses on vanilla FL’s backdoor vulnerabilities, while PFL’s unique prototype aggregation mechanism creates a distinct security landscape; the authors are the first to systematically analyze PFL’s resistance and design a dedicated attack, which fills an important research gap.

2-BAPFL’s dual-direction optimization (PPS + TOM) is well-motivated and tailored to PFL’s characteristics. PPS targets the prototype aggregation process (a core of PFL) to disrupt benign prototype learning, while TOM addresses client heterogeneity to ensure attack coverage—this design effectively overcomes PFL’s inherent resistance and is more innovative than adapting vanilla FL attacks to PFL.

3-The paper provides formal theoretical analysis, including two key assumptions (prototype-based classification rules) and two theorems (proving PPS’s ability to increase misclassification probability and TOM’s role in activating trigger-target mappings), which strengthens the methodological validity.

### Weaknesses
1-The paper claims PFL is applied in smart healthcare and autonomous driving, but all experiments use standard image datasets (MNIST, CIFAR-10) with synthetic data heterogeneity. There is no validation on domain-specific datasets (e.g., medical imaging datasets like ChestX-ray14 for healthcare) or under real-world constraints (e.g., limited client resources, intermittent connectivity), making it hard to assess BAPFL’s practical impact.

2-The paper assumes the adversary controls multiple compromised clients and has full access to global prototypes. In practice, adversaries may only control a single client or have incomplete knowledge of global prototypes (e.g., due to partial participation in aggregation rounds). The paper does not evaluate BAPFL’s performance under weaker adversary models (e.g., 1 compromised client, partial global prototype knowledge), limiting its generalizability.

3-The experiments only run for 200 training rounds. In real PFL systems, training may last for thousands of rounds, and benign clients’ continuous updates could dilute the backdoor effect. The paper does not test BAPFL’s persistence over extended training, leaving uncertainty about its long-term effectiveness.

### Questions
1-The paper claims PFL is applied in smart healthcare and autonomous driving, but all experiments use standard image datasets (MNIST, CIFAR-10) with synthetic data heterogeneity. There is no validation on domain-specific datasets (e.g., medical imaging datasets like ChestX-ray14 for healthcare) or under real-world constraints (e.g., limited client resources, intermittent connectivity), making it hard to assess BAPFL’s practical impact.

2-The paper assumes the adversary controls multiple compromised clients and has full access to global prototypes. In practice, adversaries may only control a single client or have incomplete knowledge of global prototypes (e.g., due to partial participation in aggregation rounds). The paper does not evaluate BAPFL’s performance under weaker adversary models (e.g., 1 compromised client, partial global prototype knowledge), limiting its generalizability.

3-The experiments only run for 200 training rounds. In real PFL systems, training may last for thousands of rounds, and benign clients’ continuous updates could dilute the backdoor effect. The paper does not test BAPFL’s persistence over extended training, leaving uncertainty about its long-term effectiveness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies that Prototype-based Federated Learning (PFL) is resistant to existing backdoor attacks due to its prototype mechanism and data heterogeneity. To overcome this, the authors propose BAPFL, which integrates a Prototype Poisoning Strategy (PPS) and a Trigger Optimization Mechanism (TOM). Experimental results show that BAPFL is effective.

### Strengths
* The paper is well-structured.

* This is the first backdoor attack method designed for Prototype-based Federated Learning (PFL).

* The experimental results validate the attack performance of the proposed method.

### Weaknesses
* The experimental settings are rather unusual. The clients use a very small local batch size of 4. What does the attack rate (AR) mean? Is it the proportion of malicious clients? Based on Table 1, the proposed attack method appears to reduce the model's main task accuracy (ACC). The datasets and models used are also simplistic. The authors should report the results on large-scale datasets and models for a more convincing evaluation. 

* A sensitivity analysis for $\lambda$ is missing.

* Some findings presented in Section 3.3 appear to overlap with findings in Bad-PFL.

* The paper lacks a discussion about potential defense mechanisms.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
