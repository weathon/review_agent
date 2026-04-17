# Influence-Guided Active Search for Poisoned Data Forensics

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Data poisoning attacks that inject malicious samples into training data pose a serious threat to the reliability of machine learning. Existing defense approaches focus on the fully automated detection and removal of poisoned samples; the inherent limitation of automated detection is that effective cleaning also removes a significant portion of benign samples. In contrast, we consider the *forensic investigation of poisoned data*, which relies on the verification of each sample through manual inspection, comparison with alternative data source, or some other method. The key challenge of such a forensic investigation is that the verification of each sample is expensive, but there is a limited budget for the investigation. Therefore, the investigation must strategically select, one-by-one, which samples to verify—and possibly remove—to minimize the impact of the remaining poisons. We frame this as a non-myopic sequential search problem and introduce an *influence-guided active search approach*. Our approach integrates (i) a label-free influence score that identifies training samples with disproportionate impact on test-time predictions, and (ii) an adaptive query strategy that propagates information from verified samples to focus on regions of the dataset that are both influential and likely to be poisoned. We demonstrate the efficiency and efficacy of our approach on CIFAR-10 and Tiny ImageNet against state-of-the-art attack methods, Feature Collision, Bullseye Polytope, Gradient Matching, and Narcissus. We show that our approach removes poisoned samples more effectively than fully automated cleaning methods and baseline active-search methods. This establishes our approach as a practical tool for guiding forensic investigations of poisoned training data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper suggests that fully automated detection and removal of poisoned samples has an inherent limitation in also removing a significant portion of benign samples. In response, this paper proposes a novel framework that recasts the problem as a "forensic investigation," where a human expert can verify a limited number of samples. The core contribution is an influence-guided active search algorithm designed to maximize the impact of this limited verification budget. The method sequentially selects samples for verification by prioritizing those with the highest expected poison impact.

### Strengths
- Moving from fully-automated data cleaning to a budget-constrained forensic investigation is a practical and highly relevant shift.

- The proposed EPI metric is an intuitive solution. The ideal sample to check should be both likely to be poisoned and highly influential on the model behavior sounds reasonable.

- I think the label-free influence score is significant. It overcomes a major limitation of standard influence functions, which require a labeled test or validation set, making the proposed method applicable to more realistic deployment settings.

### Weaknesses
- The proposed method relies on a Hessian-based influence function. While this term is computed only once, it might still be computationally expensive for the large-scale models common today. A more detailed discussion of the practical computational cost would be beneficial.

- The poison probability is estimated using K-NN in the feature space of the poisoned model. This assumes that poisoned samples are somewhat clustered or near each other. If an attack is extremely stealthy and its samples are well-scattered, the K-NN estimate might be unreliable, weakening the EPI metric.

- The proposed method queries the top-K most influential samples until a poison is found. If the most influential samples are all benign, the search may be slow to start or even fail if the budget is exhausted before finding the first poison. The paper would benefit from a discussion on the sensitivity to this.

### Questions
- The poison probability estimate depends on the neighborhood size, K. How sensitive is the performance of the proposed method to this hyperparameter? How was K selected for the experiments?

- How does the proposed method perform if the initial top-K influential samples are all benign? Have you explored alternative strategies to find the first poison, and how do they compare?

- The paper focuses on clean-label, targeted attacks. How do you expect this framework to perform against other poisoning modalities, such as untargeted availability attacks (which just aim to degrade overall accuracy)? Would the influence score still be the correct heuristic?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces Influence-Guided Active Search (IGAS), a forensic approach to defend machine learning models against data poisoning attacks. Instead of relying on automated data cleaning, IGAS strategically selects samples for verification under a limited budget. It combines a label-free influence score, which measures each sample’s impact on model predictions without using labeled test data, with an adaptive active search strategy that focuses on samples most likely to be poisoned and influential. Experiments on CIFAR-10 and Tiny ImageNet show that IGAS outperforms existing defenses such as Meta-Sift and EPIC, effectively neutralizing attacks while preserving model accuracy using less than 0.5% of the training data.

### Strengths
The paper introduces a well-motivated shift from automated defenses to a forensic, human-in-the-loop approach for data poisoning mitigation. Its main strength lies in integrating influence analysis with active search, enabling efficient and targeted verification under limited resources. The proposed label-free influence formulation is practical, removing the need for labeled test data while effectively capturing each sample’s impact. Experiments are comprehensive across multiple datasets and attack types, showing strong empirical performance by reducing attack success rates with minimal queries. Overall, the method is conceptually novel, computationally efficient, and empirically convincing, offering a solid foundation for future forensic approaches in data security.

### Weaknesses
1.The method assumes access to a reliable oracle for sample verification, which may not be feasible or scalable in real-world applications. It also presumes that poisoned samples are sparse and locally consistent, an assumption that might fail under adaptive or distributed poisoning attacks.
2.The paper only compares with Meta-Sift and EPIC, missing stronger baselines such as Towards a Proactive ML Approach for Detecting Backdoor Poison Samples (Qi et al., 2023), and Multidomain active defense: Detecting multidomain backdoor poisoned samples via ALL-to-ALL decoupling training without clean datasets (MAD, 2024). These recent methods address sample poisoning more proactively and in multi-domain settings without relying on clean data.
3. The proposed method focuses primarily on clean-label poisoning attacks, such as Feature Collision, Bullseye Polytope, and Gradient Matching, and does not address trigger-based backdoor attacks.

### Questions
1. **Scope and Applicability**  
   The paper focuses on clean-label poisoning attacks. How might the proposed influence-guided framework perform under dirty-label or backdoor poisoning scenarios?

2. **Comparative Evaluation**  
   The study compares mainly against Meta-Sift and EPIC. How would IGAS perform relative to more recent defenses like Proactive ML Detection, or Multidomain Active Defense?

3. **Assumption Robustness**  
   The approach assumes that poisoned samples are rare and locally clustered. What are the potential weaknesses if this assumption fails, such as in adaptive or distributed attacks?

### Soundness
2

### Presentation
2

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
This paper reframes data-poisoning defense as budgeted forensic verification, where a human oracle inspects selected training samples rather than relying on fully automated detection. The central challenge is to maximize utility under a small verification budget. The authors propose an influence-guided active search that (i) computes a label-free influence score estimating how each training sample perturbs model predictions on an unlabeled test set, and (ii) applies an adaptive query policy that prioritizes samples with the highest expected impact. Across CIFAR-10 and Tiny ImageNet under Feature Collision, Bullseye Polytope, and Gradient Matching attacks, the method reduces attack success rates to <5% while maintaining test accuracy, demonstrating effective, budget-aware poisoning forensics.

### Strengths
1) **Novel problem framing.**
   Recasting poisoning defense as a **forensic, human-in-the-loop verification** problem—rather than a fully automated detection task—is both practical and distinctive, aligning with real workflows where limited expert review is budgeted for maximal impact.

2) **Label-free influence formulation.**
   The adaptation of influence functions (Eqs. 7–8) to operate **without labeled test data**—using the **L1 norm of model logits** as a proxy—addresses a common deployment constraint and makes the influence signal broadly usable.

3) **Strong empirical results.**
   The method delivers **large ASR reductions** (often from **20/20 to ≤1/20** cases) while **maintaining or improving test accuracy**, outperformin

### Weaknesses
1) **Limited technical novelty.** The method largely adapts existing components—an ENS-style active search (Jiang et al., 2017) and standard influence functions (Koh & Liang, 2017) with a straightforward unlabeled-data variant—and combines them via a simple expected-impact objective \(p_i \cdot I_i\). While the **label-free influence** is the most distinctive element, it currently lacks a theoretical justification for why the **L1 norm of logits** captures vulnerability, an empirical check against **labeled** influence as a sanity baseline, and a rationale for choosing **L1** over alternatives (e.g., **L2**, margin/entropy). Overall, the contribution reads as a pragmatic **reformulation and integration** rather than a fundamentally new algorithmic advance.

2) **Heuristic influence proxy.**
   The L1-logit “test-impact” proxy is plausible but ad-hoc; provide analysis of **when it correlates or fails to correlate** with true test loss to justify its use and guide practitioners.

3) **Limited evaluation breadth.**
   Current results are on **CIFAR-10/Tiny-ImageNet** with three attacks. To improve external validity, add **broader threat models** (e.g., backdoors, multi-target poisons) and **larger-scale datasets**

### Questions
Please refer to the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
