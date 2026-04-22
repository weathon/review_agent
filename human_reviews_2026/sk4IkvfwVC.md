# Lipschitz-aware Linearity Grafting for Certified Robustness

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Lipschitz constant is a fundamental property in certified robustness, as smaller values imply robustness to adversarial examples when a model is confident in its prediction. However, identifying the worst-case adversarial examples is known to be an NP-complete problem. Although over-approximation methods have shown success in neural network verification to address this challenge, reducing approximation errors remains a significant obstacle. Furthermore, these approximation errors hinder the ability to obtain tight local Lipschitz constants, which are crucial for certified robustness.
Originally, grafting linearity into non-linear activation functions was proposed to reduce the number of unstable neurons, enabling scalable and complete verification.
However, no prior theoretical analysis has explained how linearity grafting improves certified robustness.
We instead consider linearity grafting primarily as a means of eliminating approximation errors rather than reducing the number of unstable neurons, since linear functions do not require relaxation. In this paper, we provide two theoretical contributions: 1) why linearity grafting improves certified robustness through the lens of the $l_\infty$ local Lipschitz constant, and 2) grafting linearity into non-linear activation functions, the dominant source of approximation errors, yields a tighter local Lipschitz constant.
Based on these theoretical contributions, we propose a Lipschitz-aware linearity grafting method that removes dominant approximation errors, which are crucial for tightening the local Lipschitz constant, thereby improving certified robustness, even without certified training. After identifying dominant neurons based on an adversarially pre-trained model, we graft linearity into these neurons and fine-tune the model with existing adversarial training schemes. Our extensive experiments demonstrate that grafting linearity into these influential activations tightens the $l_\infty$ local Lipschitz constant and enhances certified robustness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates how to improve local Lipschitz constants in ReLU neural networks.
This is done both theoretically as well as practical aspects are addressed.
The core components include identifying neurons using a weighted interval score and an instability score, as well as a slope loss encouraging unstable neurons to become more stable.

### Strengths
- The topic is of certifying robustness is important in safety-critical systems.
- The paper presents theoretical results on grafting linearity w.r.t. to $\ell_\infty$ local Lipschitz bound.
- Obtaining neural networks with fewer unstable neurons generally greatly improves verifiably due to a reduced accumulated approximation error.

### Weaknesses
- It appears that the approach is mainly about the training process; however, this only becomes clear later in the paper and is not adequately addressed, e.g., in the abstract.
- While Theorem 1 provides a nice theoretical result, a different scoring system is used in Sec. 4.
- The paper would benefit from more formalism in Sec. 4, e.g.,  how are lb/ub determined, what is $\chi$, ...
- No standard benchmarks, e.g., from latest VNN-COMP [1] are used.
- The comparison to other certified training approaches could be improved. See, e.g., [2] for a good overview.
- The presentation could be improved using some figures illustrating the process.

[1] Brix et al. "The fifth international verification of neural networks competition (vnn-comp 2024): Summary and results." arXiv. 2024.
[2] Koller et al. "Set-based training for neural network verification." TMLR. 2025.

Minor points:
- I think the reference in l.266 is broken (should reference Alg. 1?)

### Questions
- When unstable neurons are grafted and then verified, doesn't that mean that the original network remains unverified?
- Is the verified accuracy determined based on the computed Lipschitz bounds or is an external verifier used?
- Are the results solely on one evaluation run per method? If not, over how many runs are the presented training runs averaged? Can you also provide a standard deviation?
- How sensitive is your approach to the chosen hyperparameters (l.268-269, l.284, l.297)?

### Soundness
2

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
This paper investigates how linearity grafting—replacing parts of nonlinear activation functions with linear components—can enhance certified adversarial robustness by tightening local Lipschitz constants. The proposed Lipschitz-aware linearity grafting method aims to eliminate dominant approximation errors, thereby achieving tighter local Lipschitz bounds and improved certified robustness without requiring certified training.

### Strengths
The proposed approach improves upon the existing Linearity Grafting method and demonstrates potential in further tightening local Lipschitz constants.

### Weaknesses
* The authors state at the beginning of the abstract that “Lipschitz constant is a fundamental property in certified robustness, as smaller values imply robustness to adversarial examples.” However, despite obtaining a tighter local Lipschitz constant, the empirical results (Table 1) show a drop in robust accuracy (RA %). This seems contradictory to the intended goal of improving robustness. Could the authors clarify whether this behavior aligns with their theoretical claims?

* The standard accuracy (SA %) decreases significantly in most experiments, which undermines the contribution of this work. In extreme cases, one could simply remove a large proportion of unstable neurons to achieve a tighter Lipschitz bound but at the cost of severely degraded accuracy. The key challenge in Lipschitz-based methods lies in balancing SA and RA. The authors should discuss how their approach maintains this balance and why the accuracy degradation occurs.

* The proposed methods in Sections 4.2 and 4.3 are introduced rather abruptly and lack sufficient explanation. For instance, it is unclear how the specific proportions (e.g., 15 %, 80 %, 70 %) were determined (lines 268–270). Similarly, the design rationale for Eq. (5) and the choice of hyperparameters in Eq. (6) need to be clarified and justified.

* Table 7 presents results for a larger perturbation radius (ε), but the discussion is minimal. It would be helpful to elaborate on whether the proposed method can be easily extended to handle larger ε values and how this impacts performance.

* The empirical evaluation is relatively limited. It would strengthen the paper to include comparisons with more existing Lipschitz-based robustness methods, providing a broader context for the improvements claimed.

[1] Yang, Yao-Yuan, et al. "A closer look at accuracy vs. robustness." Advances in neural information processing systems 33 (2020): 8588-8601.
[2] Zhang, Bohang, et al. "Rethinking lipschitz neural networks and certified robustness: A boolean function perspective." Advances in neural information processing systems 35 (2022): 19398-19413.

### Questions
What is the formal definition of verified accuracy (VA %), and how does it relate to robust accuracy (RA %)? VA % seems to be introduced for the first time around line 321 without a clear explanation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The main idea of this paper is to graft linearity into non-linear activation functions. This leads to lower approximate errors, make Lipschitz tight, and enhance certified robustness. In my onion, however, the paper still falls short of meeting the ICLR publication standard in its current form.

### Strengths
1.	The paper is well organized and clearly presented.
2.	The paper conducted extensive ablation studies to evaluate the effectiveness of the proposed method.

### Weaknesses
1.	The improvement is not significant and inconsistent. (as shown in Table 1 and 2)
2.	The core idea and methodology lack sufficient insight and novelty. Given that Lipschitz neural networks, such as LiResNet++ (Hu, 2024), have already scaled certified robustness to ImageNet and billion-parameter models, the proposed approach appears less competitive. Therefore, I would expect either a substantial performance improvement or a more conceptually innovative contribution.
[1] Hu, Kai, et al. "A Recipe for Improved Certifiable Robustness." ICLR, 2024.

### Questions
1.	My main concern is that the paper includes only one baseline. Although I have not reviewed every bound-propagation paper, it is difficult to believe that the most relevant baseline is from 2022. Could the authors include comparisons with more recent or stronger methods?  
2.	In Table 3, the paper should also report the trade-off between clean and robust accuracy. While tightening (or suppressing) the Lipschitz constant is straightforward, it often comes at the cost of model utility. Demonstrating improvement beyond this trade-off would be crucial to show the true value of the method.
3.	The OOM statement on line 304 is unclear. Does it mean that the proposed method encounters out-of-memory (OOM) errors with some probability, and the training is restarted each time? Please clarify this behavior.
4.	The dataset used is relatively small. For more practical usage, can the paper consider cifar-100 or tiny-imagenet? Bound-propagation methods are often known to have scalability issues; providing detailed results on larger datasets would address this concern.
5.	For each attack evaluation, results are reported at only one perturbation budget. Providing results across multiple budgets would help readers assess performance trends. In addition, the chosen budgets are quite small. Please report robustness under larger budgets to evaluate the method’s behavior in more challenging regimes.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigate how different non-stable ReLU neurons affects the local Lipschitz constant of the model when we consider the adversarial perturbations. It proposes two metrics "weighted interval score" and "instability score" which enables us to select the unstable neurons more critical to model's Lipshcitz constant and thus verified robustness. By grafting such neurons by a linear function, we can improve the verified robustness of the model.

### Strengths
++ It is novel and interesting to consider verified robustness from the perspective of controlling Lipschitz constant and grafting.

++ The proposed method is generic and plug-and-play.

++ The intuition is theoretically justified to some degree.

### Weaknesses
1. The proposed framework introduces a lot of hyper-parameters, including the number of grafted neurons each layer, $k$ in Equation (5), $\lambda$, $\beta$ and $\gamma$ in Equation (6). I believe all these hyper-parameters will affect the performance to some degree. However, I did not see ablation studies or adequate discussions about them.

2. The experiments are weak in general. Comparisons with more robustness certification and the corresponding provable training algorithms should be included. For example, Table 6 considers MTL-IBP, its successors like alpha-beta-CROWN should be included.

3. Theoretical claims seem not rigorous. I have concerns in the proof of Lemma 1, it is unclear to have $|f_i^{U(k)} - f_i^{L(k)}| \geq |f_{graft, i}^{U(k)} - f_{graft, i}^{L(k)}|$ for **all values of $I$** in Equation (15). The induction assumption only indicates $max_i |f_i^{U(k)} - f_i^{L(k)}| \geq max_i |f_{graft, i}^{U(k)} - f_{graft, i}^{L(k)}|$, as indicated in Equation (11). However, I do not see why this can generalise to any value of $k$. The authors should provide more theoretical details about this.

4. The theoretical analyses only support $l_\infty$ perturbations.

5. The presentation of this manuscript is relatively poor. For example, (1) the authors should correctly use \citep and \citet to include references; (2) In Equation (1), the relationship between $A^L$, $A^U$ and $A$ is unclear, does this indicate $A^L = A^U$?; (3) You use $Loss_{slope}$ in Equation (5) and $loss_{Slope}$ in Equation (6), please pay attention to the terminology consistency.

Minor:

1. In addition to the linearizing the loss function and randomized smoothing, there are several other types of certified robustness methods, including (1) using geometry: "Provable robustness of relu networks via maximization of linear regions." (2019), "Training Provably Robust Models by Polyhedral Envelope Regularization" (2023) (2) Lipschitz-aware training: "Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks" (2018)

### Questions
Based on the concerns above, I think the current manuscript needs major editing before being considered for publication at a top-iter conference. Please consider the following questions:

1. Discuss about the role of several hyper-parameter, how to set them and conduct ablation studies.

2. Include more baselines and apply the proposed method on that to more comprehensively validate the performance.

3. Provide more justifications about the theoretical analysis in the proof of Lemma 1. Answer the third point in the weakness part.

4. Include the writing to make the manuscript nicer and the terminology consistent.

In addition, I have the following question:

5. When considering the effect of one particular neuron on the local Lipschitz constant, why do you use the criteria in Section 4.1 instead of directly considering the difference between the local Lipschitz if we graft this unstable neuron and the original local Lipschitz. I think this value (the change of local Lipschitz) is more straightforward and not difficult to compute or estimate. In addition, this will facilitate us to select neuron across different layers, as the current criteria in Section 4.1 have different numerical meanings for different layers.

6. Instead of considering the largest sensitivity and the number of unstable inputs in Equation (3) and (4), why don't we consider the average sensitivity among the training set for neuron selection? I believe this will decrease the number of hyper-parameters.

7. It would be better to give a formal definition of "local Lipschitz constant" in the task we consider in this work.

### Soundness
2

### Presentation
1

### Contribution
2
