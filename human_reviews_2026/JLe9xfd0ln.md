# Expert Merging in Sparse Mixture of Experts with Nash Bargaining

- Decision: Accept (Poster)
- Scores: 6, 6, 10, 8, 4

## Abstract
Existing expert merging strategies for Sparse Mixture of Experts (SMoE) typically rely on input-dependent or input-independent averaging of expert parameters, but often lack a principled weighting mechanism. In this work, we reinterpret expert merging through the lens of game theory, revealing cooperative and competitive dynamics among experts. Based on this perspective, we introduce Nash Merging of Experts (NAMEx), a novel framework that incorporates Nash Bargaining into the merging process, enabling more balanced and efficient collaboration among experts. Additionally, we incorporate complex momentum into NAMEx to accelerate expert propagation with theoretical guarantees for convergence. Extensive experiments across language modeling, text classification, image classification, and zero-shot robustness under data corruption show that NAMEx consistently outperforms competing methods while integrating seamlessly with popular MoE architectures. Finally, we demonstrate NAMEx’s scalability by applying it to large-scale systems, including Qwen1.5-MoE (14B) and DeepSeek-MoE (16B), where it proves effective in both zero-shot and fine-tuning settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This manuscript introduces Nash Merging of Experts (NAMEx), a novel, theoretically-grounded framework for compressing Sparse Mixture of Experts (SMoE) models. The core innovation is the reinterpretation of the merging process as a cooperative game, utilizing Nash Bargaining to determine the optimal, balanced parameter weighting for merging two or more experts. The authors claim this approach, coupled with a mechanism for enhanced convergence (complex momentum), leads to significant reduction in the SMoE's memory and compute footprint while preserving performance. The work is highly relevant to the scalable deployment of large sparse models.

### Strengths
* The most significant strength is the introduction of a game-theoretic perspective (Nash Bargaining) to expert merging.
* The work directly addresses the scalability and deployment challenge of large SMoE models by enabling significant model compression. The results demonstrating reduced footprint while maintaining performance are very compelling.

### Weaknesses
* While the memory/compute reduction is a strong result, the paper must clarify the computational overhead of the NAMEx merging process itself. Since Nash Bargaining requires solving an optimization problem to determine the merge point, how does the time taken for merging compare against simpler methods like magnitude pruning or basic parameter averaging? A clear complexity analysis or runtime comparison is needed.
* The core of Nash Bargaining relies on the definition of the experts' utility functions and the disagreement point (status quo). The manuscript needs to dedicate more detail to explaining why the chosen utility function accurately captures an expert's "value" or "specialized knowledge," and how it relates to the loss landscape or data distribution.

### Questions
* Could the authors explicitly detail the mathematical expression of the utility function, $u_i(\theta)$, for a single expert $i$ within the merging context? How is this function derived from the expert's parameters or its performance on a held-out dataset?
* Please provide a clear, detailed equation for the "complex momentum" and explain how it differs from standard momentum methods (like Nesterov or Adam momentum). Does it specifically counteract issues arising from the non-convexity introduced by the merging objective?

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
3

### Summary
This paper proposes a novel framework, Nash Merging of Experts (NAMEx), for merging Sparse Mixture of Experts (SMoE) models into a single dense model for efficient deployment. Departing from traditional heuristic-based averaging, NAMEx models expert merging as a multi-agent cooperative game. It applies the Nash Bargaining Solution (NBS) from game theory to compute fair and Pareto-optimal merging weights that reflect each expert's contribution. Furthermore, to address the slow convergence of prior expert-propagation methods, the authors integrate complex momentum, resulting in the NAMEx-Momentum variant. The method's effectiveness is demonstrated through extensive experiments on language, text, and vision tasks, as well as on large-scale models like DeepSeek-MoE 16B and Qwen1.5-MoE 14B.

### Strengths
The primary strength lies in reframing expert merging as a Nash bargaining game. This provides a principled, non-heuristic solution for combining experts, borrowing a sophisticated tool from multi-task learning to solve a practical problem in model deployment.

 The paper identifies a key weakness in prior work (EP-CAMEx), namely its slow convergence and suboptimal performance. The introduction of complex momentum is a targeted solution that demonstrably improves convergence speed and stability, as shown in the empirical analysis (e.g., Figure 5).

* **Comprehensive and Strong Validation:** The experimental validation is extensive and a significant strength. The method is tested across multiple modalities (language, vision), on robustness benchmarks (ImageNet-A/O/R), and scales up to 16B parameter models, showing consistent improvements over baselines. This broad validation builds strong confidence in the method's generalizability and practical utility.
* **Clarity:** The paper is well-written and clearly organized. Figure 2, for instance, provides an excellent visual comparison between NAMEx, CAMEx, and EP-CAMEx, making the architectural contribution easy to understand.

### Weaknesses
W1.  The core technical contribution is a clever *combination* of two very recent works: the expert propagation framework from (EP-CAMEx and the NBS) [1,2] . optimization from a multi-task learning paper. While highly effective, this might be seen as a successful application rather than a fundamental theoretical invention.


w2.The NAMEx method requires iteratively solving the NBS equation, which introduces significant computational overhead. The appendix (Table 11) shows a 6.8x increase in runtime for a per-layer update. While the final model (NAMEx-Full-Mom) seems to mitigate this by using fewer iterations (as per Appendix F.1), this trade-off between accuracy and cost is a notable weakness.

w3.The "bargaining budget" (number of NBS iterations) is a critical new hyperparameter, but its treatment is confusing. The text mentions "20 steps" in one place and "2 iters" in another. The paper lacks a clear ablation study on how this budget (e.g., 2 vs. 5 vs. 20 iterations) affects both final performance and computational cost.

w4.The paper justifies its choice of utility function by drawing an analogy between the domain vector $\tau_i$ and a task gradient. This is intuitive but remains an analogy.

### Questions
1.  **Overhead and Hyperparameters:** For the key results in the main paper (e.g., Table 2), how many NBS iterations were *actually* used for the best-performing model, NAMEx-Full-Mom (was it 2 or 20)? What is the concrete wall-clock time overhead of your final, optimized model during training compared to the EP-CAMEx baseline?
2.  **NAMEx vs. NAMEx-Full:** The superior performance of NAMEx-Full (recomputing $\alpha$ at each layer) over NAMEx (reusing the first layer's $\alpha$) seems to confirm that layer-wise dynamics (as hinted in Figure 1) are critical. Does this not imply that the basic "NAMEx" variant is a methodologically flawed ablation, as it ignores the very dynamics the paper observes?
3.  **Choice of Disagreement Point:** The paper follows prior work in setting the "disagreement point" to 0 (no update). Did you consider other, perhaps more natural, disagreement points for this specific problem, such as "standard average merging" (the main heuristic baseline)?
4.  **Role of the Curvature Matrix $M_i$:** Why was the curvature matrix $M_i$ explicitly removed from the NBS optimization step (Eq. 9, line 1) but kept in the final expert update step (Eq. 9, line 2)? If the game-theoretic solution itself ignores curvature, what role does $M_i$ play in the final update?


----

References



[1] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. arXiv preprint arXiv:2202.01017, 2022. 

[2] Viet Dung Nguyen, Minh Nguyen Hoang, Rachel Teo, Luc Nguyen, Tan Minh Nguyen, and Linh Duy Tran. CAMEx: Curvature-aware merging of experts. In The Thirteenth International Conference on Learning Representations, 2025.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces Nash Merging of Experts (NAMEx), a novel framework for expert merging in Sparse Mixture of Experts (SMoE) architectures. Diverging from conventional input-dependent or averaging strategies, NAMEx reinterprets expert merging through the lens of game theory, specifically leveraging the Nash Bargaining Solution (NBS) to derive principled merging coefficients. This approach models expert merging as a cooperative-competitive game, using expert domain vectors (deviations from a base expert) as utility functions to ensure a fair and efficient optimal agreement.
A key extension, NAMEx-Momentum, integrates complex momentum to accelerate convergence and enhance stability during the base expert propagation across SMoE layers, addressing observed slow convergence issues in prior methods like EP-CAMEx. The authors provide theoretical guarantees for the convergence of NAMEx-Momentum.

### Strengths
The paper is exceptionally well-written and clear. The authors effectively motivate the need for a principled merging strategy by highlighting the competitive and cooperative dynamics observed across different SMoE layers and architectures (Figure 1, Figures 6, 8, 9). Key concepts like the Nash Bargaining Solution and its adaptation to multi-task learning are concisely reviewed. The definition of NAMEx (Definition 3.3) and the algorithms (Algorithm 1 and 2) are clearly presented. The comprehensive experimental results are organized logically into tables, with the best performing variants (NAMEx-Full results) highlighted clearly.

This paper offers a highly original contribution to the Sparse Mixture of Experts:
- Novel Game-Theoretic Framework: NAMEx introduces the first game-theoretic interpretation of expert merging, leveraging Nash Bargaining to move beyond heuristic or input-independent averaging schemes. This principled method for balancing cooperation and competition among experts is highly original.
- Enhanced Convergence via Complex Momentum: The successful integration of complex momentum into the expert propagation mechanism (NAMEx-Momentum) accelerates convergence and provides needed stability, particularly addressing limitations found in EP-CAMEx. The theoretical convergence guarantee further enhances this contribution.

The central claims—that expert merging can be framed as a bargaining game and that NBS provides a principled solution—are adequately supported with both theoretical derivation and comprehensive empirical evidence.
The paper formalizes NAMEx as the Nash solution to the Bargaining of Expert Merging Problem, derived from the Nash product maximization objective. The authors provide a proof sketch, which establishes the Nash Bargaining equation $G^\top G \alpha = 1/\alpha$ for computing the optimal update direction. Furthermore, the introduction of complex momentum is backed by a convergence guarantee.
The empirical methodology is thorough, testing NAMEx and its momentum variants against baselines across four distinct domains. The consistent performance superiority shown across small, medium, and large-scale models confirms the effectiveness and scalability of the method. The supplementary analysis showing NAMEx yields faster and more stable convergence compared to EP-CAMEx (Figure 5) validates the motivation for using complex momentum.

**Originality and Significance**: The core innovation of framing expert merging as a bargaining problem solved via the Nash Bargaining Solution is highly original and offers a principled alternative to existing heuristics. This perspective enables a derived weighting mechanism ($\alpha$) rather than a heuristic one.

**Quality of Results**: NAMEx variants consistently deliver superior performance across various benchmarks.

**Clarity and Insight**: The paper is clear in its definitions and provides insightful empirical analysis, such as visualizing how NAMEx steers outcomes closer to the Pareto surface than linear averaging (Figure 11). The investigation into complex and quaternion momentum further pushes the boundaries for stability in expert propagation.

**Theoretical Foundation**: The theoretical proof of convergence for NAMEx-Momentum underpins the method's stability and provides a foundation for future analysis.

### Weaknesses
I don't find any major weakness in the paper. 

Here are two typos. Please fix
- Line 365: "present" -> "presents"
- Line 1187 "hese” -> "these"

### Questions
Please provide a more detailed theoretical or empirical justification for the removal of the curvature matrix $M_i$ in the calculation of the propagation update $\Delta E^{(l)}$ in NAMEx (Eqn 7) compared to EP-CAMEx (Eqn 3). Does the Nash Bargaining process implicitly account for the necessary geometry, or is this simplification merely an artifact of aligning with the existing bargaining framework?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces "Nash Merging of Experts" (NAMEx), a novel method for merging experts in Sparse Mixture-of-Experts (SMoE) models. It reframes expert merging as a game-theoretic problem, using the Nash Bargaining solution to calculate merging weights. This principled approach models the complex cooperative and competitive dynamics between experts.

### Strengths
- Applies game theory (Nash Bargaining) to expert merging.

- Consistently outperforms baselines across multiple experiments.

- Deeply investigates key components like layer-by-layer bargaining and momentum.

### Weaknesses
- What impact does this 20-step bargaining budget have on the quality of the solution? If the budget is increased, will the performance improve further?

- In the second line of Eq. 9, why not also use Nash Bargaining to guide the calculation of $\hat{E}_m$?

- It should be compared with more vision MoE models on ImageNet.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a token-wise expert merging strategy of MoE, mainly to replace the conventional sparse routing for better performance. The paper adopted the Nash Bargaining Solution (NBS) to enhance the collaboration among experts during the expert merging process. The paper provides theoretical and empirical support to justify the claim.

### Strengths
1. The proposal of NBS in expert merging seems novel
2. The paper provides theoretical justifications behind the proposed method
3. The method improves performance over previous related expert-merging methods (e.g., SMEAR, CAMEx, EP-CAMEx)

### Weaknesses
1. The main advantage of the MoE-based models is their training efficiency. It has been established in the literature that MoE models generally achieve similar performance with significantly lower training FLOPs due to their sparse training. However, the proposed expert-merging method (and possibly previous methods also) doesn't employ the sparse routing. For example, in equation (7) of the paper, the update of the base expert uses all $N$ experts and their curvature matrices. Therefore, there is an uncertainty about whether the proposed expert-merging process sacrifices the training-efficiency advantage of MoE. As we can see, in Table 12 of the Appendix, the SMoE has lower training FLOPs than NAMEx. Therefore, it is uncertain whether the advantage of NAMEx appears for extra training FLOPs or from the delicate design.

2. Another advantage of SMoE is its capability of maintaining a constant inference FLOPs with the increase of the number of experts. It is not clear whether the advantage remains in the proposed expert-merging method.

### Questions
Can the authors clarify whether the empirical advantage of expert-merging arises from extra training FLOPs or from the proposed design? A training FLOPs equivalent result can be a good way to clarify that.

### Soundness
3

### Presentation
2

### Contribution
2
