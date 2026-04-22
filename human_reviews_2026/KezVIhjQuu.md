# Approaching the Harm of Gradient Attacks While Only Flipping Labels

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Machine learning systems deployed in distributed or federated environments are highly susceptible to adversarial manipulations, particularly availability attacks—adding imperceptible perturbations to training data, thereby rendering the trained model unavailable. Prior research in distributed machine learning has demonstrated such adversarial effects through the injection of gradients or data poisoning. In this study, we aim to enhance comprehension of the potential of weaker (action-wise) adversaries by posing the following inquiry: Can availability attacks be inflicted solely through the flipping of a subset of training labels, without altering features, and under a strict flipping budget? 


We analyze the extent of damage caused by constrained label flipping attacks against federated learning under mean aggregation—the dominant baseline in research and production. Focusing on a distributed classification problem, (1) we propose a novel formalization of label flipping attacks on logistic regression models and derive a greedy algorithm that is provably optimal at each training step. (2) To demonstrate that availability attacks can be approached by label flipping alone, we show that a budget of only $0.1$% of labels at each training step can reduce the accuracy of the model by $6$%, and that some models can perform worse than random guessing when up to $25$% of labels are flipped. (3) We shed light on an interesting interplay between what the attacker gains from more *write-access* versus what they gain from more *flipping budget*. (4) we define and compare the power of targeted label flipping attack to that of an untargeted label flipping attack.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper demonstrates that a label-flipping adversary with a strict budget can inflict an availability attack on federated learning, even under mean aggregation. The authors propose a greedy algorithm for logistic regression that is provably optimal at each training step. Experiments show that flipping just 0.1% of labels can reduce model accuracy by 6%, and a 25% budget can force performance near random guessing. The paper emphasizes that write-access ($k$) is more valuable than a larger local flipping budget ($b$) for the attacker.

### Strengths
1. The paper investigates a highly constrained threat model where the adversary can only flip a small percentage of labels and cannot alter features or overwrite gradients. This rigorous setting reveals a significant vulnerability that is difficult for typical norm-based defenses to detect, making the findings relevant for real-world federated learning deployments.

2. For the binary classification setting, the authors provide a novel problem formalization and derive a greedy algorithm (Algorithm 1) that is mathematically proven to be optimal at each training epoch. This strong theoretical foundation ensures the attack strategy is maximally efficient under the given constraints.

3. The analysis provides a practical trade-off, showing that wider write-access ($k$) over the dataset is more impactful for the attacker than a larger local budget ($b$). This guides future research in defense design, suggesting that limiting the number of contributing clients is paramount to robustness.

### Weaknesses
**Major Issues:**

1. The paper's theoretical framework and empirical findings are grounded in the study of logistic regression models. Despite the justification provided in the appendix regarding tractability, this model is not a mainstream choice for the current distributed and federated learning systems that the paper aims to address. This limits the immediate practical relevance of the conclusions, as the attack's effectiveness and optimality guarantees are not confirmed for the widely used deep neural networks.

2. The threat model relies on the server using mean aggregation, which is non-robust. The paper does not analyze the attack's efficacy against state-of-the-art FL defenses like robust aggregators (e.g., Krum, Median). According to the gradient visualization in Figure 4, the core attack strategy is to create a large divergence between the poisoned gradient and the honest gradient. This substantial directional and magnitude shift makes the malicious client update an easily identifiable outlier, which robust defense mechanisms are specifically designed to filter.

3. Another major concern lies in the assumption that the malicious worker contributes to the final batch at every training epoch. This level of guaranteed, continuous write-access is seldom maintained in a practical FL setting where clients are typically selected randomly for participation in each round. Consequently, the modeled adversary possesses an artificially strong capability that may not align with the stochastic nature of client selection in real-world distributed learning deployments.


**Minor Issues:**

1. Citations: The current citation style used throughout the paper is incorrect for ICLR submissions. It should strictly adhere to the author-year format, such as (XXX et al., 2025). All in-text citations should be revised to conform to the required ICLR template.

2. References: References [31] and [32] are duplicates. Please remove one of the entries to ensure an accurate and clean bibliography.

3. Abstract: The first word of the fourth point in the Abstract should be capitalized for consistency: "(4) We define and compare..."

4. Table Formatting: The title of Table 1 ("Notation Summary") is incorrectly positioned on the same line as the Section 2.1 heading. The table title should be distinct and positioned above the table as per standard academic formatting rules."

5. Equation Numbering: The core optimization formula in line 206 is incorrectly numbered as (1a). This equation should be corrected to the primary label for the problem, which is (1).

6. Layout Issue: The layout on Page 4 has a major formatting error where a line of body text (line 202-203) is awkwardly inserted between Figure 3 and Figure 4. This break in the flow is confusing for the reader and needs to be resolved by correctly placing the text relative to the figures.

7. Informal Algorithmic Language: The final instruction in Algorithm 1 is phrased in overly informal and explanatory language. This style deviates from the presentation expected of a formal algorithm description in a research paper.

8. Undefined Notation in Figures: The meaning of $h_{\theta}$ in Figure 1 and $I_1, I_2, I_3, I_4$ in Figure 3 are not defined or explained in the captions, main text, or the notation summary table.

### Questions
1. If the trained model were a non-convex architecture, such as a Multi-Layer Perceptron (MLP) or a Convolutional Neural Network (CNN), would your current conclusion regarding the greedy, per-epoch optimal attack still apply?

2. How is the target parameter vector ($\alpha^{Target}$ in binary classification or $W^{Target}$ in multi-class) for the targeted attack specifically set or chosen in your experiments?

3. Assuming the more realistic scenario where the attacker can only participate in and attack a single training round (epoch), what measurable impact would this single-shot attack be capable of inflicting on the final model accuracy?

4. Following a single-round attack, how many subsequent honest training rounds (epochs) would be required for the model's performance to recover to its normal expected effectiveness?

5. What was the distribution of data across participants (clients/workers) for the multi-class experiments? If the data were highly non-IID (severely imbalanced), how would that condition affect the severity of the attack's results?

6. How is the logistic regression per-sample gradient term $\left(\sigma\left(\alpha^{\top} x_n\right)-y_n\right) x_n$ in Equation (2) derived from the cross-entropy loss function?

### Soundness
3

### Presentation
3

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
This paper investigates the potential for availability attacks in federated learning systems with mean-like aggregation. The study asks if an attacker, who is constrained to only flipping labels on existing data and cannot alter features, can still significantly degrade model performance.

The authors formalize the label-flipping attack as a per-epoch constrained optimization problem. The attacker's objective is to choose a subset of labels to flip, within a set budget, to maximally misalign the resulting gradient from a reference direction . This formulation leads to a greedy algorithm that is proven to be optimal at each training step.

The threat model assumes an attacker with limited actions (label-flipping only) but strong knowledge. The attacker is considered omniscient, with full read-access to the model's parameters and the honest gradient at each epoch, which is required to calculate the optimal flips.

The experiments, conducted on logistic regression, show that this attack can reduce model accuracy by 6% when flipping only 0.1% of labels, and can force the model to near-random performance with a 25% budget. The paper also analyzes the trade-off between an attacker's "write-access" (the total data fraction $k$) and their "local budget" (the flipping fraction $b$), concluding that having wider write-access is more impactful. The method is presented for binary classification and then extended to the multi-class setting.

### Strengths
- The problem is motivated well. Figure 1 effectively illustrates the scope of this new attack, and Figure 2 presents a clear landscape of existing attacks, serving as a good survey.
- The paper convincingly discusses the continued relevance of mean aggregation in FL. It then presents a clean mathematical formulation for its attack, centered on the novel and significant idea of selectively flipping labels to maximize gradient misalignment.
- The setup and the threat model are clearly described and come across as fair. The resulting attack algorithm is simple, elegant, and intuitive.
- A major strength is that the simple, greedy algorithm is not just a heuristic; it is mathematically proven to be the optimal strategy for the attacker at each epoch, given their budget.
- The experiments are well-planned, and the results are extremely well-presented. The thoughtful choice of visualizations, like heatmaps and bar charts, and the clear design of the plot axes, convey a large amount of useful information in a highly interpretable form.
- The results consistently follow and confirm the intuition behind the attack. Every experimental result is interpreted with wise conclusions and insights, helping the reader gain a deeper understanding of the attack's mechanics, such as the $k$ vs $b$ trade-off.
- The paper successfully generalizes the attack from a binary setting to the multi-class setting with a formulation that is legitimate and well-reasoned.

### Weaknesses
- The claim that altering even a tiny fraction of labels (like 0.1% or 1%) is impactful may be overstated as a general result. This impact is highly dependent on the specific loss landscape of the data-model pair, and it should be clarified that this observation was only confirmed for the datasets and models tested.
- The paper claims (e.g., in Line 119) that such a small alteration could bypass anomaly detection, but this is presented without evidence. A critical question remains: how does the attack have such a large impact while remaining stealthy? One would want to know why the malicious gradients are not simply diluted by the low budgets.
- The paper is confined to logistic regression which is a good starting point and has clean math, but any results on a neural network would be more appreciated.
- It is not clear from the main paper what dataset is used for the main experiments. The appendix mentions MNIST and CIFAR-10 but it is not clear which result figures correspond to which dataset. 
- More details may be required to describe how the target model was computed. Does the adversary assume that the entire dataset is available to find this target model?

### Questions
- The result in Fig 5, which appears as the primary result, showing that accuracy decreases as the global budget ($k \times b$) increases, is somewhat obvious. What is the main motivation and takeaway from Fig 5. It may not have been referenced in the main paper. 
- How practical is the attack? Suggest scenarios where an adversary is able to flip the labels but not the features. The budget also needs justification from any practical perspective. Since only mean aggregation is targeted in this attack, one cant argue that a low budget makes the attack stealthy.
- As the training proceeds and the loss landscape is explored, is it possible if not frequent that a sample with label X was an effective attack in a certain round, but the same sample is flipped to label Y for an effective attack on a different round?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper provides a formalized analysis of label-flipping attacks in federated learning, showing that even a highly constrained attacker can cause significant degradation under mean aggregation. It contributes both theoretical insight (greedy optimality per epoch) and empirical evidence (strong attack effects under minimal corruption).

### Strengths
1. It reframes label flipping as an optimization problem with provable properties.
2. It provides formal derivation and proof of greedy optimality.

### Weaknesses
1. The proposed attack is limited to logistic regression under convex settings. It is unclear how the findings extend to deep or non-convex models, which are the dominant cases in modern federated learning.
2. The assumption of an omniscient attacker contradicts the paper’s framing of a “weaker adversary.” In practice, such full access is unrealistic.
3. The results on availability degradation (accuracy drop) are modest for small budgets and demonstrated only in simplified, synthetic setups.
4. The work does not include a comparative evaluation against stronger existing poisoning methods (e.g., gradient matching, backdoor attacks), so its relative effectiveness remains unclear.

### Questions
1. Why restrict the analysis to mean aggregation only? Since many federated learning systems now use robust aggregators (e.g., trimmed mean, median), would your method still apply?
2. The theoretical analysis guarantees only per-epoch optimality. Does this guarantee extend to the entire training trajectory? Could non-greedy strategies outperform the proposed one in the long term?
3. How does your attack compare quantitatively against existing label-flipping or gradient-matching attacks from prior work? Are there cases where those methods are strictly stronger or weaker?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the vulnerability of distributed or federated learning environments to adversarial attacks, particularly to availability attacks. The paper focuses on a weaker attacker, that is, generating an availability attack by only flipping labels under a constrained budget. The authors formulate this problem considering logistic regression models and propose a greedy algorithm that is provably optimal at each training step. The experiments show that availability attacks generated by only flipping labels are effective in significantly degrading model accuracy.

### Strengths
- The paper considers the weakest adversary and analyzes its effect on model robustness, which is more practical and further connects data-poisoning with gradient-based availability attacks.
- The formulation of the label flipping attack under the budget constraint is clear.
- The extension of the proposed algorithm to multi-class classification shows generalization.
- Per-step optimality of the proposed greedy algorithm is analyzed.

### Weaknesses
- A recent study [*] also particularly focuses on label flipping attacks during training and uses a similar budget-constrained optimization formulation. The paper addresses the comparison between gradient attacks and data poisoning (line 158); however, a discussion should have been included in the related work section. Since it is directly relevant, the proposed algorithm could have been compared against this label flipping attack, aside from the gradient attack comparisons presented.

> [*] Bal, M. I., Cevher, V., Muehlebach, M. (2025). Adversarial Training for Defense Against Label Poisoning Attacks. International Conference on Learning Representations (ICLR).

- Although per-step optimality is presented, the algorithm is locally optimal at each training iteration. That is, the temporal coupling of the model update is overlooked. A discussion on what would happen if gradient directions oscillate, and (if) greedy label flipping could cancel its own effect, should have been provided to deepen the optimality analysis.

- The paper considers only the logistic regression model; hence, the derivation does not hold for deeper or non-convex models. How can the method be extended to general differentiable models, NNs? For instance, could the authors provide an empirical example of the method applied to NNs?

- The authors consider a weaker adversary that can only flip labels; however, the omniscient read-access assumption is strong. In a more realistic FL setting, partial knowledge is considered. What if the adversary has no access to all parameters or access to partial data? 

- Although the authors discuss the choice of mean aggregation in detail, I think there should be an ablation on this design choice to analyze the effectiveness of the attack better. 

- The empirical evaluation includes small datasets and simple settings, i.e., no heterogeneity or non-iid partitioning typical of federated learning is tested.

- The current reference style used in the paper does not comply with ICLR2026 style. 
"Citations within the text should be based on the natbib package and include the authors’ last names and year (with the “et al.” construct for more than two authors)." Please check the paper template and modify the manuscript for the rebuttal version.

- For readability, the main text should include a summary of the experimental setup. The dataset, model and sizes should be mentioned before presenting the results to improve readability.

- Minor: Line 51, Figure caption: Extra blank space before dot.

### Questions
- How $\alpha$_target is selected? 
- See the Weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
2
