# RPP: A Certified Poisoned-Sample Detection Framework for Backdoor Attacks under Dataset Imbalance

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Deep neural networks are highly vulnerable to backdoor attacks, yet most existing defenses assume balanced data and overlook the pervasive class imbalance found in real-world settings—an omission that can amplify backdoor risk. This paper offers the first in-depth study of how dataset imbalance intensifies backdoor vulnerability, showing that (i) imbalance induces majority-class bias that increases susceptibility, and (ii) standard defenses degrade markedly as imbalance grows. To address this, we introduce Randomized Probability Perturbation (RPP), a certified poisoned-sample detection framework that works in a black-box setting using only model output probabilities. For any inspected input, RPP decides whether it has been backdoor-manipulated and provides provable within-domain detectability guarantees along with a probabilistic upper bound on the false positive rate. Extensive experiments on five benchmarks (MNIST, SVHN, CIFAR-10, TinyImageNet, and ImageNet-10), covering 10 backdoor attacks and 11 baseline defenses, demonstrate that RPP achieves substantially higher detection accuracy than state-of-the-art defenses, especially under class imbalance. RPP thus establishes a theoretical and practical foundation for defending against backdoor attacks in imbalanced real-world environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors consider the problem of bias due to class imbalance, particularly how it "amplifies" backdoor vulnerability. Experiments are conducted on image classifiers. The defense assumes an operational scenario (line 141,212) where backdoor triggers are to be detected among a pool (Z) of test samples. Based on the description of p. 5, I think the authors use a kind of training set "fuzzing" by random perturbations to try to discover backdoor triggers (poisoned samples) in a pool of data.

### Strengths
Addressing backdoors and bias is a challenging problem.

### Weaknesses
As the author ack on line 143, Observation 1 (line 161) has previously been noted.

The following reference, which adapts a backdoor defense to address model bias particularly to data imbalance, should have been cited particularly regarding Observation 2 (line 176):
[1] H. Wang et al.  Maximum Margin Based Activation Clipping for Post-Training Overfitting Mitigation in DNN Classifiers.  IEEE Trans. Artificial Intelligence (TAI), vol. 1, Aug. 2025.

How much data D is used for "Preliminary Model Training" (line 223) and where does it come from? I think it's the pool of test samples suggested in line 212. Under what practical scenarios does the pool necessarily contain backdoor triggers (poisoned samples) as assumed? 

The description on lines 219- is confusing.  

First, p(z|w) is poorly defined and it's inappropriate to cite another paper for a clear definition of such an important quantity to this paper.  In equation (3), p inherits a subscript which leads me to think it's a class posterior? How does equ (3) relate to equ (4) if at all? In any case, the notation in (3) and (4) is terrible. 

Also, the authors appear to be working on a before/during training scenario where the possibly poisoned _training_ data (D) of the "preliminary" model  is available to the defense, in addition to a dataset V that is known to be benign with respect to a possible backdoor.  The following (uncited) work is also in this context but does not assume a benign dataset V is available to the defense:
Z. Xiang et al.  Reverse Engineering Imperceptible Backdoor Attacks on Deep Neural Networks for Detection and Training Set Cleansing.  Elsevier Computers & Security, 2021.
Also, before considering a possible threshold for Delta P, explain where sigma comes from.

The fonts in the figures are too small.

Overall, the paper's presentation is sloppy and relevant prior work is not adequately considered.

### Questions
See above.

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
3

### Summary
The paper provides a well-motivated and well-supported certified defense framework addressing an important real-world issue in backdoor detection. Its contributions are technically solid and empirically convincing, though the conceptual novelty is somewhat limited.

### Strengths
The paper provides a thorough analysis of how dataset imbalance amplifies backdoor vulnerabilities and weakens existing defenses, which is a practical and underexplored aspect of backdoor research.

The work offers formal guarantees, including conditions for detectability, upper and lower trigger bounds, and provable control over false positive rate.

Evaluations on five benchmarks and ten attack types demonstrate strong performance, outperforming 11 state-of-the-art defenses in both detection accuracy and robustness under class imbalance.

### Weaknesses
The theoretical analysis assumes Gaussian perturbations and independent noise across samples, which may not hold for complex data distributions.

 Although adaptive scenarios are mentioned, the empirical defense-attack interplay is underexplored. More thorough experiments on adaptive trigger shaping or low-magnitude attacks would add credibility to claims of robustness.

### Questions
The theoretical analysis assumes Gaussian perturbations and independent noise across samples, which may not hold for complex data distributions.

 Although adaptive scenarios are mentioned, the empirical defense-attack interplay is underexplored. More thorough experiments on adaptive trigger shaping or low-magnitude attacks would add credibility to claims of robustness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the relationship between class imbalance and the vulnerability of deep neural networks to backdoor attacks. The authors empirically demonstrate that class imbalance in the training data can significantly increase the attack success rate (ASR), which is an important and insightful observation. To address this issue, the paper proposes a method for identifying poisoned samples caused by class imbalance. Unlike previous approaches that rely on global or class-level statistics (such as clustering or distribution estimation), the proposed method operates at the sample level, measuring the stability of output probabilities under random noise perturbations. Extensive experiments are conducted to validate its effectiveness.

### Strengths
1.The paper uncovers an important and previously underexplored phenomenon — that class imbalance can substantially amplify a model’s susceptibility to backdoor attacks. This observation provides a valuable perspective on how data distribution affects model security.

2.The proposed detection approach abandons conventional clustering- or distribution-based strategies, instead introducing a per-sample robustness criterion based on output stability under random perturbations. This idea is conceptually fresh, intuitive, and supported by convincing results.

3.The experiments cover various imbalance settings and provide strong evidence supporting both the motivation and the effectiveness of the proposed method.

### Weaknesses
1.Lack of validation on real-world long-tailed datasets:
All imbalance settings in the experiments are synthetically generated. The absence of evaluation on naturally imbalanced or long-tailed datasets  limits the practical generalizability of the results. Adding experiments on real datasets would significantly strengthen the paper’s claims.

2.Inconsistent notation:
Some symbols and notations are inconsistently used across sections, which slightly reduces readability. A careful revision to ensure consistency throughout the paper is recommended.

### Questions
1.How does the proposed approach perform on naturally imbalanced, long-tailed datasets instead of synthetic ones?

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
3

### Summary
The paper firstly presents investigation of how the dataset imbalance amplifies backdoor vulnerability. To address this, the paper proposes Randomized Probability Perturbation(RPP) that operates in a black-box setting using only model output probabilities. RPP leverages sample-level prediction stability under noise perturbations and integrates conformal prediction to provide provable detection guarantees and control over the false positive rate. Extensive experiments confirm that RPP achieves higher detection accuracy, establishing a theoretically grounded framework for defending against backdoor attacks with imbalanced data.

### Strengths
1. Originality:

   The paper presents the first work to systematically investigate and formalize the critical problem of how dataset imbalance amplifies backdoor vulnerability and cripples existing defenses. 

   The paper proposes Randomized Probability Perturbation(RPP), which adapts the concept of randomized smoothing to the distinct task of certified poisoned sample detection, proposing a paradigm shift from distribution-level signals used by prior defenses to a sample-level robustness metric.

2. Quality:

   The paper provides strong theoretical foundations, with formal guarantees for certified detectability (Theorem 5.1) and False Positive Rate control (Theorems 5.3, 5.4).

   The experimental section encompasses an extensive benchmark: 5 datasets, 2 imbalance types, 4 imbalance ratios, 10 diverse backdoor attacks, and 11 state-of-the-art baseline defenses. The results are consistent and support the paper's claims.

3. Clarity:

   The motivation is established in the introduction, which effectively critiques prior work and outlines the paper's contributions. The logical flow from problem identification to solution and evaluation is smooth and easy to follow.

4. Significance:

   RPP provides a black-box and certified tool for securing ML pipelines in critical domains like medical diagnosis and autonomous driving, where data is inherently imbalanced and security is paramount.

### Weaknesses
1. The paper's sample-level approach is commendable, but its global calibration of RPP thresholds remains vulnerable to distributional skew from class imbalance. A critical analysis of per-class detection performance and adaptive thresholding strategies is needed.
2. The practicality of requiring a pre-trained preliminary model on potentially poisoned data needs clearer justification regarding security and computational overhead. The exact deployment scenario and cost-benefit analysis compared to end-to-end defenses should be elaborated.

### Questions
1. Your method uses a global threshold derived from the calibration set. Given that the model's confidence and behavior can vary significantly between majority and minority classes, have you observed any systematic difference in the separation of clean vs. poisoned RPP values across different classes? Does the detection performance for poisoned samples degrade in the tail classes compared to the head classes?
2. Could you please elaborate on the precise real-world scenario for deploying this "prior-training" defense? Specifically, how does a defender obtain a "preliminary model" that has learned the backdoor without already being compromised? 
3. The core intuition is that poisoned samples have lower RPP. How does the full RPP method (with conformal prediction) compare against a much simpler baseline that directly thresholds the empirical  $\tilde{\Delta}P$ value, or against a baseline that flags a sample if its top-1 prediction is invariant over $J$ noise perturbations?

### Soundness
3

### Presentation
3

### Contribution
2
