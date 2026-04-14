# Towards False-claim-resistant Model Ownership Verification via Targeted Fingerprint

- Decision: Reject
- Scores: 6, 1, 3, 6

## Abstract
The utilization of open-source pre-trained models has become a prevalent practice, but unauthorized reuse of pre-trained models may pose a threat to the intellectual property rights (IPR) of the model developers. Model fingerprinting, which does not necessitate modifying the model to verify whether a suspicious model is reused from the source model, stands as a promising approach to safeguarding the IPR. In this paper, we revisit existing model fingerprinting methods and demonstrate that they are vulnerable to false claim attacks where adversaries falsely assert ownership of any third-party model. We reveal that this vulnerability mostly stems from their untargeted nature, where they generally compare the outputs of given samples on different models instead of the similarities to specific references. Motivated by these findings, we propose a targeted fingerprinting paradigm ($i.e.$, FIT-Print) to counteract false claim attacks. Specifically, FIT-Print transforms the fingerprint into a targeted signature via optimization. Building on the principles of FIT-Print, we develop bit-wise and list-wise black-box model fingerprinting methods, $i.e.$, FIT-ModelDiff and FIT-LIME, which exploit the distance between model outputs and the feature attribution of specific samples as the fingerprint, respectively. Extensive experiments on benchmark models and datasets verify the effectiveness, conferrability, and resistance to false claim attacks of our FIT-Print.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes FIT-Print, a defense against false claim attacks in model fingerprinting. The paper first extends false claim attacks to various existing fingerprint schemes and reveals that existing methods are typically vulnerable. The paper attributes this vulnerability to the untargeted nature of existing methods, which allows an adversary to exploit transferrable verification samples to forge a false ownership. The paper then proposes FIT-Print, which features a targeted fingerprint optimization process that aligns the fingerprint to a pre-defined target vector. Based on FIT-Print, the paper provides two concrete implementations, FIT-ModelDiff and FIT-LIME, and experimental results show that the proposed method is effective in verification and robust against false claim attacks.

### Strengths
1. **This paper addresses a new and important problem in model fingerprinting**. False claim attack is a relatively new attack in model fingerprinting. This paper provides a defense technique against this new attack.
2. **This paper extends existing false claim attacks to testing-based fingerprint methods**. Previous false claim attacks have mainly focused on adversarial-example-based (AE-based) fingerprints. This paper provides an extension to testing-based methods.
3. **This paper proposes a defense against false claim attacks**. FIT-Print proposes a novel targeted optimization process to craft verification samples w.r.t. a pre-defined target, which limits the transferability of verification samples and is empirically effective against false claims. The method could also be generalized to various testing-based model fingerprints.

### Weaknesses
1. **The claim that FIT-Print "restricts the (potential) fingerprint space" lacks theoretical support**. The idea of restricting fingerprint space with targeted optimization is interesting. However, judging from Theorem 1, the false claim attack has little chance of success mainly because the adversary has no knowledge of $F$ and has to make random guesses. It lacks a deeper analysis of whether this design indeed restricts the fingerprint space and how it impacts the optimization of verification samples. Nonetheless, this is understandable considering the complexity of neural networks.
2. **The problem setting of false claim attack is unclear**. The current manuscript lacks a general workflow of the fingerprint registration and verification process. It also lacks a threat model for the false claim attack. Consequently, it is unclear how a fingerprint is registered and verified, or how the adversary could launch a false claim attack. While some of this information could be found in one of the cited works [1], these missing parts would still cause confusion.

[1] Liu et al. "False Claims against Model Ownership Resolution". USENIX Security 2024.

### Questions
1. In Section 4.4, is the adversary using the same fingerprint $F$ when creating verification samples from ImageNet independent models?
2. The setting of the adaptive unlearning attack in Section 4.5 is confusing. The authors mention that "the model reuser still has *no* knowledge of the target fingerprint $F$", but the first sentence says "the model reuser *knows* the target fingerprint" and the target fingerprint $F$ is used in the optimization in Eq. 17. It is unclear whether $F$ is known to the adversary or not.
3. What is the overhead of the targeted optimization (fingerprint generation)? The overhead analysis in Appendix F seems to only include the overhead for fingerprint verification. Since the targeted optimization process involves additional optimization and augmented models, will the fingerprint generation process be significantly slower?
4. What is the attack success rate of false claim attacks on existing AE-based or testing-based methods? It seems Table 2 only includes a false positive rate on independent models without considering false claim attacks.
5. From this reviewer's opinion, some discussions in the appendices are quite important, such as the reason for choosing testing-based fingerprints over AE-based fingerprints, the ablation study on the augmented models, and the discussion on the generalization of FIT-Print. Including these in the main part of the paper would help clarify some confusion. This reviewer understands that the authors are restricted by the page limit, but it would be better if some references to the appendices could be put where necessary.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
3

### Summary
To protect an AI model's copyright from illegal reuse or plagiarism, a method to verify true ownership is required. A fingerprinting strategy can be used without altering the model, preserving its performance. However, adversarial modifications used for fingerprinting can enable false claim attacks, where an attacker falsely claims ownership by exploiting the transferability of these modifications.
To counter false claim attacks, the authors propose a targeted adversarial fingerprinting method. They argue that current techniques are vulnerable due to their untargeted nature, which creates a large fingerprint space, enabling adversarial transfers between models. By focusing on specific target classes, they suggest that the fingerprint space can be narrowed, reducing transferability and minimizing false claim risks.

### Strengths
Supplementary Materials
The supplementary materials include code and a clear README, which demonstrates a responsible effort towards transparency and reproducibility.

### Weaknesses
1. Contribution and Innovation Deficiency
Contribution: 
There are some issues with the contributions mentioned in the paper:
Contribution 1: False claim attack is not a new type of attack. This type of attack has already been introduced in previous work, and thus, it should not be considered a contribution of this paper.
Contribution 2: The advantages of the proposed targeted verification method (compared to other verification methods) have not been sufficiently explained. Without clearly demonstrating the superiority of the targeted approach over other methods, it is difficult to view this as a valid contribution.
Contribution 3: The advantages of the proposed black-box fingerprinting method (compared to existing methods) are also insufficiently explained. Without providing more explanation to highlight the uniqueness or strengths of this method, it cannot be considered a significant contribution.
Novelty:
The main innovation of the paper seems to be the shift from non-targeted to targeted fingerprinting methods. However, this innovation appears somewhat limited and lacks sufficient persuasive power. Beyond this point, the paper's contributions in terms of innovation could be further enhanced. It is recommended to explore additional novel ideas to strengthen the overall impact of the paper.

2. Terminology and Definitions are Unclear
Improper naming: 
The choice of abbreviations in the paper is not standard, especially in "False-claIm-resistant Targeted model fingerPrinting (FIT-Print)." Using the letter "I" from the middle of the word "claim" is unconventional and does not follow common abbreviation rules. Naming should avoid such arbitrary choices to ensure clarity and consistency, and it is recommended to use abbreviations that are more in line with standard conventions.
Unclear explanation of the verification process: 
In section 2.3, the explanation of ownership verification through "p1" and "p2" is overly brief. Using such abbreviations makes it difficult for readers to grasp their exact meaning. It is recommended to clearly explain each step of the verification process and avoid using vague abbreviations that could lead to confusion.
Unclear definition of independent model: 
The term "independent model" is used in the paper but not clearly defined. Does "independent" refer to models with completely different structures, or models performing different tasks? This distinction is critical for understanding the relationships between models in the verification scenario. It is recommended to clearly define whether independence is based on structure, task, or dataset.
Unclear definition of the scenario and protocol: 
The paper does not clearly define each party involved in the fingerprint verification scenario, nor the specific steps they execute in the protocol. For instance, the roles of each party, the information they have access to, and the actions they perform should be clearly explained, so that readers can better understand how the entire verification protocol works. Similarly, providing more specific steps and details for both the attack and defense mechanisms—such as the inputs, outputs, who executes each step—would greatly improve the understanding of these processes.
Unclear model relationships: 
The symbols for models (e.g., Mo, Ms, Mi) are not clearly explained, particularly in the context of the "false claim of ownership of M1." Providing clearer descriptions and diagrams would improve understanding of how these models interact in the scenario.
Lack of illustrations: 
The paper lacks sufficient diagrams to explain the overall fingerprint verification framework. The relationships between models like Mo, Ms, and Mi are unclear, making it harder for readers to follow. Diagrams would help visualize these interactions and clarify the verification process.

Be careful when use of terminology:
Terms like "Proposition," "Definition," and "Theorem" are used in the paper but need to be applied with more caution. It is recommended to briefly introduce these terms in the introduction and clarify their specific contexts throughout the paper to help readers better understand their relevance and contribution to the argument.

3. Concerns about the Mechanism or Inadequate Consideration/Explanation of Vulnerabilities.
Fingerprinting scenario considerations: 
The paper does not fully explore whether adversarial samples generated by the model owner could also affect models not owned by them, given the known transferability of such samples. This scenario needs further investigation to assess its potential impact on the verification process.
Fairness in adversarial sample generation: 
The process for generating adversarial samples x̄ and x is unclear. If different algorithms are used, it raises concerns about fairness in the verification process. Ideally, fingerprinting rules should be enforced by a trusted third party to ensure consistency. Clarifying the generation algorithms would enhance transparency and fairness.

### Questions
The weaknesses of the paper have been pointed out in the Weaknesses section, and I hope that targeted revisions can be made for each issue. Overall, I believe the following improvements would enhance the quality of the paper:

1. Clear explanation and presentation
Please provide detailed explanations of the roles, tasks, and relationships of each party in the scenario, along with a step-by-step description of the process. I also expect more elaboration on potential attacks and defenses, specifying the processes involved.
I encourage the authors to invest more effort in improving the readability of the paper, such as by incorporating diagrams to help readers understand the proposed protocol and background knowledge. The terms and concepts used should be clearly defined, as omitting details here could compromise the clarity of the paper. Please also avoid using non-standard abbreviations or expressions that may confuse readers.

2. Accurate and expanded contributions
Accurately and truthfully present the contributions of your work. If the current contributions are insufficient, consider adding new ones. I believe there is room for further enhancement in the paper's contributions to innovation.

3. Explanation and discussion
This point corresponds to my concerns about the mechanism or inadequate consideration/explanation of vulnerabilities. I hope the authors can alleviate these concerns by providing additional explanations, analyses, or clear descriptions of the experimental setup. This would enhance the reliability of the work presented in the paper.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper studied the vulnerability of existing model fingerprinting approaches against false claim attacks. To fix the problem, they propose to incorporate a mapping function, which can be confidential, and input perturbation, which caters to the mapping, to defend against such attacks.

### Strengths
- The paper is well-written and has plenty of experiments to support the contributions.
- The research question is interesting.

### Weaknesses
- The attack scenario is limited. Currently, to the best of my knowledge, there is no third-party management on the fingerprints to authorize the ownership of open-sourced models. Therefore, it is impractical to consider and even defend against false-claim attacks. The authors should elaborate on the practical side of the setting.
- Another major concern is the missing literature. There is a work on KDD23 which also leverages a mapping function (they call it a meta-verifier) and adaptive input samples to build the model fingerprint. What is the different in methodology? The authors did not even mention the work. They should at least compare with it and show the substantial contribution over the work.

[1] https://dl.acm.org/doi/10.1145/3534678.3539257

### Questions
See the weakness part above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of protecting intellectual property rights (IPR) in open-source pre-trained models, which are widely reused but often without authorization. The authors pointed out a vulnerability in existing fingerprinting techniques, i.e., they are prone to false claim attacks, where adversaries could falsely claim ownership of third-party models. To address this, the authors propose a targeted fingerprinting paradigm called FIT-Print, designed specifically to counteract false claim attacks. FIT-Print transforms model fingerprints into targeted signatures through optimization. Based on this paradigm, the authors develop two black-box fingerprinting methods—FIT-ModelDiff and FIT-LIME, which rely on bit-wise and list-wise approaches, respectively, by leveraging the output distances and feature attributions of specific samples as fingerprints. Experiments on benchmark models and datasets demonstrate the effectiveness of the FIT-Print approach, showing that it is robust to false claim attacks.

### Strengths
+) Defending against false claim attacks is a timely topic.

+) The proposed method is new and technically reasonale in general

+) Experimental results looks promising and justified the claims

### Weaknesses
-) Some technical details are not very clear

-) It is not clear if the conclusion obtained on classification task can be extended to a more general scope of models

### Questions
a) The focus on defending against false claim attacks is both timely and relevant, given the increasing reliance on open-source pre-trained models in diverse fields. By addressing the vulnerability in existing model fingerprinting methods, this paper provides an advancement in securing model ownership verification.

b) While I can understand the general idea of test sample extraction is to find a perturbation that pushes the model's prediction towards a target fingerprint (Eq.6), some technical details are not very clear to me. First, how to choose the target fingerprint F (Eq. 5 and 6)? I believe the choice of F significantly affects the model performance (e.g, F is different from the true label). Secondly, while the paper proposes two mapping functions f, it is not clear what criteria should be considered as good mapping functions. The design of FIT-MODELDIF and FIT-LIME looks arbitrary, and not clear why they are good mapping functions? Thirdly, I am confused how Eq.6 grantees that an independent model won't respond similarly to the test samples? 

c) It is worth comparing to existing optimization-based model finger-printing methods, e.g, sensitive sample fingerprinting [1].

[1] He, Zecheng, Tianwei Zhang, and Ruby Lee. "Sensitive-sample fingerprinting of deep neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

d) It would be better to discuss if the conclusion obtained on classification tasks can be extended to a more general scope of models, e.g., LLMs and other computer vision models (e.g., diffusion models which involves randomness)?

### Soundness
3

### Presentation
2

### Contribution
3
