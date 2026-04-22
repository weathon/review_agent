# Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Training machine learning models with differential privacy (DP) limits an adversary's ability to infer sensitive information about the training data. It can be interpreted as a bound on the adversary's capability to distinguish two adjacent datasets according to the chosen adjacency relation. In practice, most DP implementations use the add/remove adjacency relation, where two datasets are adjacent if one can be obtained from the other by adding or removing a single record, thereby protecting membership. In many ML applications, however, the goal is to protect attributes of individual records (e.g., labels used in supervised fine-tuning). We show that privacy accounting under add/remove overstates attribute privacy compared to accounting under the substitute adjacency relation, which permits substituting one record. To demonstrate this gap, we develop novel attacks to audit DP under substitute adjacency, and show empirically that audit results are inconsistent with DP guarantees reported under add/remove, yet remain consistent with the budget accounted under the substitute adjacency relation. Our results highlight that the choice of adjacency when reporting DP guarantees is critical when the protection target is per-record attributes rather than membership.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper identifies an important gap between the privacy guarantees provided by add/remove adjacency and the actual protection needed for attribute inference in differential privacy. The authors develop novel auditing techniques for substitute adjacency and demonstrate empirically that standard add/remove accounting can significantly overstate attribute privacy protection.

### Strengths
1. The paper addresses a critical misconception in practical DP deployments where practitioners may unknowingly overestimate protection against attribute inference when using standard libraries.
2. The paper explains with examples and citations (Sections 1–2, pp. 1–2), why add/remove adjacency may fail for attribute privacy. This makes the motivation persuasive. 
3. Experiments in Section 6 demonstrate that empirical ε values from auditing exceed add/remove accountant bounds. This supports the paper’s key claim with direct quantitative evidence.
4. Reproducibility is good since a public anonymized code repository is provided, and Appendix Table A1 includes detailed hyperparameter.

### Weaknesses
1. The attacks are targeting DP-SGD; Whether they are applicable to other mechanisms (e.g., DP AdamW, DP sampling-based methods) are not clear.
2. Although empirical ε values are averaged over multiple runs (Figure 1, p. 7), the paper does not report confidence intervals or variance across repetitions, making it hard to assess statistical robustness.
3. The auditing method requires training thousands of models (e.g., R = 2500 per setting). This raises concerns about scalability and applicability to larger models or datasets.

### Questions
1. Whether this can be generalized to other mechanisms rather than just DP-SGD?
2. How robust are the empirical ε estimates to randomness in mini-batch sampling and initialisation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper argues that when the goal is to protect per-record attributes (e.g., labels) rather than just membership, reporting DP guarantees under the standard add/remove adjacency can be misleading. The authors propose auditing methods under substitute (replace-one) adjacency—including worst-case “canary” constructions in gradient space and input space—and show empirically that models trained with DP-SGD can leak more than the add/remove accountant would suggest, while the leakage aligns with a substitute-adjacency accountant.

### Strengths
The paper spotlights a real deployment pitfall: most libraries/accountants default to add/remove adjacency, yet many fine-tuning scenarios care about attribute privacy for users already known to be in the training data.

The gradient-space and input-space canaries (plus mislabeled and natural choices) are well-motivated and easy to reproduce. The “worst-case dataset canary” analysis gives intuitive mixture-of-Gaussians distinguishers and ties directly to the accountant.

### Weaknesses
The paper mixes hidden-state audits, crafted gradients, and input-space canaries; readers may struggle to map these to real-world attacker access. Suggestion: Add a single table that contrasts black-box / hidden-state / white-box access (what the adversary sees, what they can inject) and mark which canaries apply to each, plus a “practical examples” column (e.g., data-collection poisoning during SFT; known-in-training user seeking label privacy).

CIFAR-10 fine-tuning and Purchase100 MLP are reasonable, but the paper’s practical claim (“attribute privacy can be overstated”) would be stronger with at least one text or tabular SFT workload with label privacy relevance.

### Questions
I have no questions about this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper shows that when we are interested in attribute DP, add/remove adjacency overstates privacy. Substitute adjacency, which allows replacing a record, better captures attribute privacy. Using canary-based attacks to audit DP under substitute adjacency, the authors find that the privacy leakage can exceed add/remove guarantees but aligns with substitute bounds, thus choice of adjacency is critical for per-record attribute protection in practical deployments.

### Strengths
1) The authors provides a clear and rigorous demonstration that standard add/remove adjacency overstates privacy when the goal is to protect individual labels. Through extensive experiments (Section 5), the authors quantify the gap between theoretical DP guarantees and the actual leakage observed in practice, showing that models can leak significantly more information about labels than add/remove accounting would suggest. 

2) The main strength of the work is its development of a novel auditing framework under substitute adjacency. The authors design canary-based attacks that can replace target records to probe privacy leakage, with Algorithm 4 being effective in estimating tight lower bounds. By crafting canaries in both the input and gradient spaces, the study establishes a robust methodology that reliably captures attribute-level privacy loss.

### Weaknesses
Computational Cost: High-confidence audits require multiple retrainings, which can be resource-intensive. The authors acknowledge this and suggest potential optimizations using single-run approaches (Steinke et al., 2023).

(minor) Title: The title does not explicitly mention attribute or label privacy. Adding terms like “attribute DP” or “label DP” could make the paper’s setting immediately clear to readers.

### Questions
1. Given the high computational overhead, what is the authors' estimated potential reduction in the required number of training runs that could be achieved by integrating robust canaries with a single-run auditing technique?

2. "The mislabeled canary is relatively less sensitive to $C$, likely due to systematic misalignment with the task objective", can you elaborate on this?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper examines how current differential privacy accounting, based on add/remove adjacency, fails to reflect real privacy risks in machine learning. The authors argue that when an attacker already knows a record is in the dataset and wants to infer attributes such as labels, the correct notion is substitute adjacency. They design a set of empirical canary audits in gradient space and input space to measure empirical lower bounds of $\epsilon$. Experiments on image and tabular data show that substitute adjacency leads to about twice the reported privacy loss of standard accounting. The work demonstrates that existing DP practices overestimate protection against attribute inference.

### Strengths
- The paper makes a valuable empirical contribution by quantitatively showing how privacy guarantees differ under two adjacency definitions. This is an important clarification for the community and helps link theoretical concepts to practical implications in training differentially private models.

- The originality of the work lies primarily in the empirical auditing framework. The audited privacy loss values align closely with the theoretical upper bounds, supporting the validity of the approach.

- Experimental details and configurations are described clearly, making the study repeatable.

### Weaknesses
- The paper discusses attribute inference but does not evaluate or contrast its findings with other DP formulations that also target label or feature privacy, for example, label-DP.  

- It remains unclear what information the attacker is assumed to know. The experiments appear to assume full feature knowledge except for the sensitive attribute, but this is not formally stated or varied.

- Each privacy point requires many repeated training runs. The feasibility of scaling such auditing to larger settings remains unclear.

### Questions
- Would the tight bound derived under substitute adjacency remain valid when compared with mechanisms built on Label Differential Privacy, which is specifically designed to protect labels?
- Could the authors more precisely define the attacker’s prior knowledge in each auditing setup?
- Could the authors analyze the computational cost of their auditing approach and provide empirical approximations or results demonstrating its scalability to larger applications?
- How does the observed privacy loss behave when only part of the input features are assumed known to the adversary?
- For a fixed noise parameter, is there a typical ratio between substitute and add/remove privacy losses?

### Soundness
3

### Presentation
2

### Contribution
3
