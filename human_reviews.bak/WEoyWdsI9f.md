# Quantifying and Defending against the Privacy Risk in Logit-based Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 5

## Abstract
Federated learning (FL) aims to protect data privacy by collaboratively learning a model without sharing private data among clients. Novel logit-based FL methods share model outputs (i.e., logits) on public data instead of model weights or gradients during training to enable model heterogeneity, reduce communication overhead and preserve clients’ privacy. However, the privacy risk of these logit-based methods is largely overlooked. To the best of our knowledge, this research is the first theoretical and empirical analysis of a hidden privacy risk in logit-based FL methods – the risk that the semi-honest server (adversary) may learn clients’ private models from logits. To quantify the impacts of the privacy risk, we develop an effective attack named Adaptive Model Stealing Attack (AdaMSA) by leveraging historical logits during training. Additionally, we provide a theoretical analysis on the bound of this privacy risk. We then propose a simple but effective defense strategy that perturbs the transmitted logits in the direction that minimizes the privacy risk while maximally preserving the training performance. The experimental results validate our analysis and demonstrate the effectiveness of the proposed attack and defense strategy.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores logit-based Federated Learning (FL) methods that aim to protect data privacy. It highlights a previously unnoticed privacy risk where a semi-honest server could potentially learn clients' private models from shared logits. The paper introduces an attack called Adaptive Model Stealing Attack (AdaMSA) and proposes a defense strategy to mitigate this risk. Experimental results confirm the effectiveness of the attack and defense strategy.

### Strengths
+ A novel Adaptive Model Stealing Attack (AdaMSA) is proposed to quantify the privacy risk of logit-based FL.
+ A simple yet effective defense strategy is proposed to achieve better privacy-utility trade-off.
+ A bounded analysis of privacy risks is provided for the proposed privacy attacks.
+ Extensive case studies.

### Weaknesses
- The value of the research question requires further justification.
 - The outcomes of the experiment need to be made more convincing. 
- Limited in-depth comparison with state-of-the-art solutions.

### Questions
Q1: The motivation of this article requires further justification by providing additional evidence. The authors mentioned that logit-based FL was developed to achieve communication-efficient FL. However, this is not a mainstream FL framework nor a mainstream communication-efficient FL framework. For example, asynchronous FL, gradient compression-based FL, gradient quantization-based FL, and generative learning-based one-shot FL are all widely adopted. Therefore, the reviewer's first concern is whether it is necessary and valuable to analyze the privacy risks of logit-based FL.

Q2: The objectives of the adversary's attack warrant further examination. While the paper articulates the adversary's aim as acquiring the private model θ, it is imperative to delve deeper into whether this private model θ can be subsequently leveraged for malicious purposes. It is worth noting that previous works in the field of privacy attacks primarily focus on the exfiltration of a client's confidential data. Consequently, a critical concern arises: can the adversary utilize the private model θ to reverse-engineer the original training data? This crucial aspect of the adversary's capabilities necessitates thorough investigation and discussion to assess the potential privacy risks associated with the acquired private model.

Q3: More advanced baselines need to be included to highlight the superiority of the proposed privacy attacks. Considering that FL was proposed in 2016 and the baseline scheme compared in this article was also proposed in 2016, whether this scheme is representative still needs to be discussed. It would be better if the authors could consider more baseline solutions (such as [1]).

[1] Takahashi H, Liu J, Liu Y. Breaching FedMD: Image Recovery via Paired-Logits Inversion Attack[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 12198-12207.

Q4: There is merit in exploring additional, perhaps more straightforward, security mechanisms to corroborate and strengthen the privacy assurances of logit-based Federated Learning (FL). It is essential to recognize that the fundamental premise underpinning the attacks in this article hinges on the server's ability to access the logits uploaded by the client. However, it is possible to mitigate this vulnerability through the implementation of secure aggregation techniques and the utilization of hardware-based Trusted Execution Environments (TEEs). These measures can effectively safeguard against the server's unauthorized access to logits. It is pertinent to acknowledge that these considerations do not diminish the innovative contributions of the article. Nevertheless, it would be advantageous for the authors to engage in a discourse on these potential defense mechanisms to provide a more comprehensive understanding of the robustness of logit-based FL with respect to privacy concerns.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work develops a model stealing attack (AdaMSA) in logit-based federated learning. Additionally, it provides a theoretical analysis of the bounds of privacy risks. It also proposes a simple but effective defense strategy that perturbs the transmitted logits in the direction that
minimizes the privacy risk while maximally preserving the training performance.

### Strengths
- The proposed attack is effective with a high attack success rate.
- Theoretical analysis is conducted to quantify the privacy risks in logit-based FL.
- Extensive empirical results support the theoretical analysis

### Weaknesses
- The proposed method is impractical to be executed or evaluated in real-world scenarios.
- The privacy risk metric can not express the true privacy risk of the setting.
- The notations are vague which makes it very hard to follow the analysis of the work

### Questions
1. Why does the accuracy on the private dataset express the success rate of the model stealing attack? What if the attack mode is very generalized which achieves high accuracy in the same data distribution?

2.  Since $D_{pub}$ is unlabeled, how to quantify Eq. 1 for $D_{mix}$ ?

3. The construction of $D_{mix}$ is based on the empirical sets of $D_{priv}$ and $D_{pub}$. How does it reflect the true distribution of $D_{priv}$ and $D_{pub}$?

4. Since the adversary cannot touch $D_{priv}$, how to construct $D_{mix}$?

5. What is the difference between the theoretical analysis of the work compared to Blitzer et al., 2007?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigates the hidden privacy risks in logit-based Federated Learning (FL) methods through a blend of theoretical and empirical approaches. It introduces the Adaptive Model Stealing Attack, which utilizes historical logits in training and provides a theoretical analysis of the associated privacy risk bounds. They also propose a defense strategy that perturbs the transmitted logits in the direction that minimizes the privacy risk while maximally preserving the training performance. Experiments under different settings demonstrate the performance of the proposed attack and defense.

### Strengths
* The paper provides the first analysis of the hidden privacy risk in logit-based FL methods. An attack and a corresponding defense method are proposed to quantify and prevent the privacy risk. The authors also provide a theoretical bound for the privacy risk.
* Experiments under different FL settings have been conducted to demonstrate the performance of the attack and the defense.

### Weaknesses
* The computational complexity of the proposed attack and defense are not discussed.
* For the defense method, the approximation error of the proposed heuristic solver is not analyzed.
* The number of communication rounds of the experiments is small. It would be interesting to see whether the performance of the attack and the defense still hold with hundreds or thousands of communication rounds.

### Questions
* For the proposed attack, how to determine the threshold $T_0$? Why the importance weight $w$ is linearly dependent with $t$ (rather than exponential dependence for example)?
* How many federated clients are in the experiments?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the privacy risk in logit-based federated learning (FL). In particular, the authors provide theoretical analysis to bound the privacy risks and propose a model stealing attack adapted to the logit-based FL settings. In addition, the authors also provide a defense strategy that perturbs the transmitted logits to minimize privacy risks.

### Strengths
1. The paper is well written.
2. The experiment presentation is clear.
3. The defense is simple and effective.

### Weaknesses
1. First of all, I have some questions and doubts about the significance of logit-based FL in the community of FL. I have checked the logit-based FL papers mentioned in the related work, and they are impactful on the FL community. Currently, logit-based FL seems not to be a well-established and standard norm in FL. From this perspective, studying the privacy risks of logit-based FL is unlikely to have an impact on the community in the long run.
2. The tricks used in the proposed attack lack technical depth. The proposed attacks improve by the previous baseline via a temporal weighted factor, making the attack an incremental improvement.
3. Ony baseline (MSA) is too naive. To demonstrate the effectiveness of the tricks used in the proposed attack, I suggest adding more baselines—for example, no threshold $T_0$ or setting $w_t=1$ in the proposed attacks.
4. Non-iid setting of FL. In Figure 1, the author states that the server aims to infer client $k$’s private models. I wonder if the attack makes sense in the non-iid setting of FL or if client $k$ is a poisoned client. In this case, the objective of the attack should be also justified.

### Questions
1. Is Adaptive attack possible in the presense of the attack knows the defense (e.g., obfuscate the logit with added noise)?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
