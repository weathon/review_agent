# Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large language models trained on web-scale data can memorize private or sensitive knowledge, raising significant privacy risks.
Although some unlearning methods mitigate these risks, they remain vulnerable to "relearning" during subsequent training, allowing a substantial portion of forgotten knowledge to resurface.
In this paper, we show that widely used unlearning methods cause shallow alignment: instead of faithfully erasing target knowledge, they generate spurious unlearning neurons that amplify negative influence to hide it.
To overcome this limitation, we introduce Ssiuu, a new class of unlearning methods that employs attribution-guided regularization to prevent spurious negative influence and faithfully remove target knowledge.
Experimental results confirm that our method reliably erases target knowledge and outperforms strong baselines across two practical retraining scenarios: (1) adversarial injection of private data, and (2) benign attack using an instruction-following benchmark.
Our findings highlight the necessity of robust and faithful unlearning methods for safe deployment of language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a critical and timely problem in machine unlearning for LLMs: the lack of robustness. The authors compellingly argue that existing unlearning methods often achieve "shallow unlearning alignment," a phenomenon where the model learns to "hide" target knowledge by generating "spurious unlearning neurons" that amplify negative influence, rather than "erasing" the neurons that encode the knowledge. To substantiate this claim, the paper introduces two practical "retraining attack" scenarios (harmful and benign) that demonstrate how this hidden knowledge can be easily recovered. The authors use an attribution-based analysis to quantify the difference between positive and negative influence variations, providing strong evidence for their hypothesis. Based on this diagnosis, the paper proposes SSIUU (Suppressing Spurious Unlearning Neurons for Robust Unlearning), a novel regularization method. SSIUU penalizes the growth of negative influence during unlearning, forcing the optimizer to achieve a more "faithful" unlearning by reducing the original positive influence. Experiments conducted on Llama-3.2 and Qwen-2.5 models using the FaithUn and TOFU datasets show that SSIUU significantly outperforms strong baselines in terms of robustness against the proposed retraining attacks, while maintaining comparable performance on standard unlearning metrics.

### Strengths
1. This paper provides the "shallow unlearning alignment" concept, backed by the "hide" (increasing D-) vs. "erase" (decreasing D+) analysis, which is a clear, insightful, and novel explanation for why unlearned models are fragile. The proposed method, SSIUU, is not an ad-hoc fix. It is a principled solution that directly targets the problem (spurious negative neurons) identified in the diagnosis. The method shows SOTA performance on the new robustness metrics (Tables 1 & 2), demonstrating a clear practical benefit over all baselines, including strong ones like NPO and KLUE.
2. The introduction of "harmful" and "benign" retraining attacks is a major strength. These are practical, well-motivated, and set a new, higher standard for evaluating unlearning methods.
3. The use of logit lens (Fig 4) to show SSIUU avoids "over-unlearning" and the attribution stability analysis (Fig 6) to show robustness at the representational level are both very convincing.

### Weaknesses
1. The entire paper's diagnosis and method rely heavily on the attribution method from Yang et al. (2023). The claims would be much stronger if the authors showed that the "shallow alignment" phenomenon (large D- increase) is consistent across different attribution methods.
2. The method requires calculating an attribution-based regularizer at each optimization step. This seems computationally intensive. The paper mentions a "simplification" for efficiency but provides no details on what this simplification is, nor does it provide any analysis of the actual computational cost (e.g., training time, memory usage) compared to the baselines. This is a critical practical limitation for a method intended for large models.
3. The simplification mentioned in Section 5 ("multiplying each parameter with its gradient") is vague. This sounds very different from the formal attribution definition in Equation 1. This ambiguity makes the method difficult to reproduce and assess.
4. While Table 1 includes many baselines, the detailed attribution analyses in Figures 3, 5, and 6 primarily focus on comparing SSIUU to GD. It would be more convincing to see this deep analysis applied to other strong baselines, especially NPO, which was the most robust baseline in Table 1.

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
3

### Summary
This paper reveals that standard unlearning techniques achieve only shallow alignment by relying on negative influence suppression and tries to improve the robustness of machine unlearning in LLMs. The authors introduce novel retraining attack benchmarks (harmful and/or benign) and develop a regularization method called SSIUU that actively penalizes negative attribution growth to achieve genuine knowledge erasure. Experiments on 3B models with two datasets demonstrate that SSIUU improves robustness against retraining attacks.

### Strengths
1) The paper addresses a critical and practical flaw in existing unlearning methods. The motivation is articulated clearly and persuasively.
2) The contributions are clearly listed, and the related work section is concise and positions the paper well. 
3) The paper doesn't just identify a problem; it provides both a way to measure it (the new benchmarks) and a way to fix it (SSIUU).

### Weaknesses
1) The proposed method requires a non-trivial attribution-like score calculation at every unlearning step. The paper provides zero quantitative analysis of the computational cost of SSIUU versus the GD baseline. This omission is fatal for evaluating the method's claimed efficiency and practical utility.
2) The main text lacks a comprehensive discussion on the trade-off between λ and retention performance. Figure 7-(a) suggests that aggressive regularization (λ≥0.01) begins to degrade accuracy on the retained knowledge set, implying SSIUU is a delicate balancing act that requires extensive hyperparameter tuning. This practical difficulty is insufficiently addressed.
3) The robustness is only evaluated against retraining attacks. It is an open question whether SSIUU is robust against other privacy attacks, such as membership inference attacks on the forgotten set, which might exploit the newly established, strong D+ based erasure mechanism.

### Questions
1) Does SSIUU impact convergence speed?
2) Can the authors provide a recommended strategy or dynamic schedule for setting λ that minimizes the degradation of retention accuracy while maintaining high robustness?
3) Can the authors provide an analysis on SSIUU's performance on other privacy tasks?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the limitations of current machine unlearning methods and introduces the concept of spurious unlearning neurons, which emerge during unlearning and suppress target knowledge through negative attribution rather than true erasure. The authors propose SSIUU, a regularization-based approach designed to mitigate the inflation of such negative influences and achieve more robust unlearning. Extensive experiments demonstrate that SSIUU not only improves robustness against retraining attacks but also enhances forgetting effectiveness without sacrificing utility.

### Strengths
1.  The paper identifies and analyzes the phenomenon of spurious unlearning, which is novel and provides valuable insights into why existing unlearning methods often fail to achieve true knowledge removal.

2.  The proposed SSIUU method demonstrates strong empirical performance, showing superior robustness and forgetting effectiveness across multiple datasets and models while maintaining utility on unrelated tasks.

### Weaknesses
1.  Why is the phenomenon of spurious unlearning considered to be stored at the neuron level, given that neurons are typically viewed as units encoding knowledge rather than behavioral control mechanisms?

2.  The current work demonstrates only the existence of spurious unlearning neurons but not their causal role. What would happen to the model’s behavior if these neurons were directly removed after unlearning?

3.  Is suppressing spurious unlearning neurons during training essentially a “whack-a-mole” process, where eliminating such neurons might lead to the emergence of new spurious unlearning patterns, for example at the attention-head level?

### Questions
1.  In Figure 3(c), the proposed method (_Ours_) shows an increase in D^{+}, representing the positive influence on the target knowledge. Why does this happen? Shouldn’t effective unlearning reduce the positive contribution instead of increasing it?

2. Why does gradient ascent in unlearning often lead to the emergence of spurious unlearning behavior, where the model suppresses rather than erases knowledge? Conversely, would gradient descent in standard learning produce an analogous phenomenon of spurious learning? Is it possible to theoretically prove these effects?

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
2

### Summary
This work presents a new machine unlearning approach that mitigates spurious unlearning, prevalent in current unlearning methods, by leveraging an attribution-guided regularization loss. The initial investigation leading to this new approach is logical and well-motivated. Experiments confirm the effectiveness of the method.

### Strengths
- S1. The paper investigates a well-motivated problem supported by both literature and experiments.
- S2. The authors tackle the problem with a targeted method supported by initial observations.
- S3. The method proposed is effective despite its apparent simplicity.

### Weaknesses
- W1. The proposed unlearning approach hinges on the correctness of a specific attribution method. This reliance is a potential weakness: if the attribution method does not accurately or consistently identify knowledge-bearing vs. spurious neurons, the effectiveness of SSIUU could be compromised. The paper’s arguments assume the attribution is reliable across conditions and tasks, but this is not extensively validated. It would strengthen the paper to either justify the choice of attribution method (e.g., why it’s expected to generalize) or to evaluate the unlearning approach with an alternative attribution technique to ensure the results are not an artifact of one particular method.
- W2. Some aspects of the method’s implementation are not clearly described in the paper (for instance, how final gradients per step are computed considering the new training objective introduced in equation 3). Without details or released code, it’s difficult to assess the feasibility and reproducibility of the approach. Machine unlearning on LLMs is a highly practical problem, so the paper would benefit from more discussion on practical considerations like computational cost, how the method scales with model size, and how it fits into an unlearning pipeline. I recommend the authors include more details on the implementation (perhaps pseudo-code or algorithmic complexity analysis) and discuss the feasibility: e.g., how much additional training overhead does SSIUU introduce? Such details would help readers gauge the practicality of deploying this method.
- W3. The experimental evaluation, while promising, could be more comprehensive. The paper would be stronger if it included results on additional models or tasks (adding statistical significance to the results), and particularly if it reported variability across multiple runs. Currently, it’s hard to tell if the gains are consistent or might vary under different conditions.

### Questions
- Q1. In the third paragraph of the Introduction, the authors discuss effects (shallow alignment via spurious neurons) as if they are known phenomena. Could the authors rephrase that paragraph to clarify that this paper is demonstrating these effects (rather than assuming them as prior knowledge)? This clarification is crucial because that paragraph is a key part of the motivation. Additionally, if there are any references or prior work hinting at such phenomena, please cite them to give more context.
- Q2. Could the authors provide more context or evidence for how changes in the variance of positive vs. negative influence (as measured by their chosen attribution method) correspond to truly eliminating knowledge versus merely hiding it? In most paragraphs discussing this, introduction and section 4 included, this correlation is discussed too abruptly. I believe the manuscript requires such in depth discussion which can be added as a separate section to the appendix.
- Q3. Can the authors provide more context on the applicability and limitations of the SSIUU method? For instance, what assumptions or conditions are required for SSIUU to be robust and effective? In particular, which types of unlearning methods or scenarios is SSIUU compatible with, and what are its potential limitations or failure modes? The paper currently has no limitations section, and adding one to discuss edge cases (where SSIUU might not perform well, or might not apply) would greatly strengthen the work
- Q4. In the hypothesis at the beginning of Section 4, the authors use several terms that were not defined earlier (e.g., “new neurons”, “spurious unlearning”, “suppressing knowledge”, “erasing knowledge”). Could the authors clearly define these concepts (either in Section 4 or earlier) to provide the proper context for interpreting the hypothesis?
- Q5. The paragraph “Quantifying Knowledge” in Section 4 needs more detail/clarity: (i) What exactly is meant by a “neuron” in the context of a transformer language model (e.g., is it a single hidden unit in a specific layer)? (ii) What do the authors mean by “gradient of the language model” in that paragraph? Please clarify this term and perhaps relate it to the model’s components. (iii) Could the authors contextualize these definitions in terms of the general structure of a decoder-only transformer (like those used in the experiments)? (iv) Additionally, could you clarify the notation in Equation 1 – specifically, why the multiplication sign “\times” is used and what operation it denotes in that context?
- Q6. In lines 252–254 of the paper, you mention using a certain number of samples to obtain attribution scores (for computational tractability). How many samples are used, and how well does this sampled approach approximate the true per-token attribution scores? Also, regarding the training objective (Equation 3): since the values $A$ are computed using gradients (attribution scores), how are the final parameter gradients for the optimizer obtained? In other words, are you backpropagating through the computation of $A$, or treating $A$ as fixed for the purpose of gradient computation? This detail is unclear and important for understanding the optimization process.

### Soundness
3

### Presentation
3

### Contribution
3
