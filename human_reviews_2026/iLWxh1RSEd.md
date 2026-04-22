# ReviveEdit: Robust Sequential Editing via Dominant Subspace Preservation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Sequential knowledge editing in large language models often causes catastrophic collapse of the model’s general abilities, particularly for parameter-modifying methods. Existing approaches attempt to mitigate this issue with heuristic constraints, but they lack a principled understanding of the underlying failure mechanism and overlook the structured impact of edits on model parameters. In this work, we conduct a spectral analysis and identify a key failure mechanism: the progressive corruption of the dominant singular subspace of weight matrices, a low-rank subspace that we show is both crucial for encoding general abilities and highly sensitive to perturbations. Based on this insight, we propose REVIVE, a novel plug-and-play framework that prevents model collapse by explicitly preserving this dominant subspace. REVIVE projects any given update onto the singular vector basis of the original weight matrix and removes all components that would interfere with the protected subspace. This allows new knowledge to be integrated through less critical directions without damaging the model’s core structure. Extensive experiments show that REVIVE substantially outperforms existing methods, maintaining high editing efficacy and preserving general capabilities even under extreme sequences of up to 20, 000 edits.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the problem of performance degradation in large language models (LLMs) during sequential knowledge editing (SME).
The authors perform a singular value decomposition (SVD) on the model’s parameter matrices and observe that a model’s general capabilities are largely concentrated within its dominant singular subspace. They argue that continuous edits progressively distort this subspace, ultimately leading to model collapse.
To address this issue, the paper proposes ReviveEdit, a method that, during each edit, performs projection and filtering operations to constrain updates only to the low-energy directions of the parameter space. This procedure preserves the integrity of the dominant singular subspace and prevents degradation of general abilities.
Experiments on GPT-J, LLaMA3, and other models, using datasets such as COUNTERFACT, ZSRE, and GLUE, demonstrate that ReviveEdit achieves higher editing success rates and stronger stability compared to prior methods. Notably, even after 20,000 sequential edits, the model retains approximately 86% of its downstream task performance.

### Strengths
**Strengths:**

The experiments are **comprehensive and well-designed**.
They cover multiple models and tasks, including long-horizon sequential editing, and the results are stable and clearly reported. This demonstrates that the authors have invested substantial effort in the experimental evaluation.

The work also has **clear engineering value**.
The proposed method is simple, modular, and can be easily integrated with other model editing frameworks, making it practically useful for improving model robustness in large-scale applications.

### Weaknesses
**Weakness:**

The core idea of this paper — preserving dominant directions in the parameter matrix’s feature or singular subspace — is not novel.
Earlier works such as Delta-Edit (2024) and O-Edit (2025) have already proposed highly similar motivations from different perspectives, namely low-rank perturbation (Delta Projection) and orthogonal subspace regularization (gradient-space orthogonality).

The notion of “protecting the dominant subspace” has also been well established in prior research, including studies on low-rank fine-tuning, model compression, and weight perturbation analysis.
As a result, this paper’s theoretical contribution is limited, representing more of a formal restatement or spectral reinterpretation of existing ideas rather than a genuinely new conceptual advance.

**Therefore, despite the solid empirical validation, this work does not meet the originality threshold expected for ICLR acceptance.**

[1] O-EDIT: ORTHOGONAL SUBSPACE EDITING FOR LAN GUAGE MODEL SEQUENTIAL EDITING 

[2] DeltaEdit: Enhancing Sequential Editing in Large Language Models by Controlling Superimposed Noise

### Questions
See **Weankess**

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes REVIVEEDIT, a framework for robust sequential model editing that prevents model collapse by preserving the dominant singular subspace of parameter matrices. Through spectral analysis, the authors argue that catastrophic degradation during sequential edits stems from the corruption of high-energy singular components that encode general abilities. REVIVEEDIT mitigates this by projecting updates onto the singular vector basis and removing directions that interfere with dominant subspaces.

Experiments across GPT2-XL, GPT-J, and LLaMA3 on COUNTERFACT and ZSRE show substantial gains in both editing efficacy and general ability preservation, even after tens of thousands of sequential edits.

### Strengths
1.	The paper provides an insightful spectral explanation of why sequential editing leads to degradation, connecting weight structure and general ability loss.
2.	REVIVEEDIT is plug-and-play and compatible with existing editing frameworks such as MEMIT and AlphaEdit.
3.	Strong empirical validation across multiple models and baselines, including large-scale tests.

### Weaknesses
1.	Limited novelty compared to prior work –The method’s core idea—preserving or constraining updates within structured low-rank subspaces—bears strong resemblance to previous works such as PRUNE and AlphaEdit, which also regulate parameter updates through rank or null-space constraints. The new contribution (dominant subspace preservation via SVD) can be seen as an incremental extension of these ideas rather than a fundamentally new paradigm.
2.	The use of SVD projection and component filtering is technically straightforward. The novelty mainly lies in the empirical finding that high-singular-value directions encode general abilities, but this is somewhat intuitive and overlaps with insights from AlphaEdit.
3.	Computational feasibility not fully addressed. Performing SVD for all large matrices is costly; no discussion of efficiency or scalability to very large LLMs (e.g., 70B parameters) is provided.
4.	The argument that dominant subspace corruption is the cause of collapse remains empirical; a stronger theoretical guarantee or causal analysis is missing.

### Questions
1.	Does REVIVEEDIT require recomputing SVD after each batch of edits, or is it fixed once? How does that affect computational efficiency?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers sequential knowledge editing in LLMs and argues that catastrophic degradation after long edit sequences arises from corruption of the dominant singular subspace of weight matrices. The authors propose REVIVE, a plug-and-play mechanism that i) decomposes each update $\Delta W$ in the SVD basis of the original weight $W$ and ii) filters out components that involve the top singular directions using an energy threshold $\tau$. Experiments demonstrate substantial gains across multiple editors and model families, while maintaining robustness for sequences of edits up to 20k.

### Strengths
1. The paper provides a coherent explanation that deterioration of general abilities is linked to the dominant singular subspace.

2. REVIVE can be applied to existing editors directly, while acknowledging some added compute and storage for SVD-based filtering.

3. The authors conducted extensive experiments to validate the effectiveness of the proposed method.

### Weaknesses
1. The core idea—preserving a knowledge subspace—is conceptually close to strands in continual learning and to AlphaEdit in model editing (though AlphaEdit emphasizes feature subspaces while REVIVE emphasizes parameter subspaces). 

2. Post-hoc projection may be suboptimal. If the edit intrinsically lies within the top singular subspace, filtering may prevent achieving the desired update.  It would be more principled to include constraint in the solution (similar to AlphaEdit) or even in the optimization (i.e., in finding the target output).

### Questions
1. Are there concrete cases where the desired edit demonstrably lies in the top-energy directions? 
2. With thousands of edits, does the projected subspace become saturated?
3. Some parameter-preserving baselines should be considered. Recent SimIE is also plug-and-play and reports strong performance, and should be considered as a baseline.
4. Fig. 4 suggests that editing performance decreases suddenly, while low-rank subspace similarity decays more gradually (Fig. 3). What mechanism explains this discrepancy? 
5. Could you add an ablation that solves the edit with the constraint (e.g., constrained LS / projected gradient) and compares it to post-hoc projection?

[1]. Aging with grace: Lifelong model editing with discrete key-value adaptors, NeurIPS 2023.

[2]. Towards lifelong model editing via simulating ideal editor, ICML 2025.

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
4

### Summary
This paper proposes FastEdit, a method designed to address the high computational cost in model editing. By exploiting the low-rank-plus-diagonal (LR+D) structure of hidden representation, FastEdit replaces costly dense inversion with closed-form updates derived via the Sherman–Morrison–Woodbury (SMW) identity. The authors also incorporate a structural prior to improve the robustness of $\mathbf{K}_0$. Experiments show up to $10\times$ end-to-end speedups over prior model editors while maintaining competitive editing efficacy on standard benchmarks.

### Strengths
1. The idea of explicitly modeling the editing module with an LR+D structure is intuitive, which unlocks SMW-based closed forms.

2. The mathematical development is rigorous: (i) it motivates LR+D for hidden representations \mathbf{k}, (ii) it derives a closed-form objective and its solution, and (iii) it shows how SMW eliminates cubic-time inversions in practice.

3. The writing is easy to follow, with equations that map directly to the implementation.

4. The approach scales naturally to larger models, addressing a key bottleneck for real-world edits.

### Weaknesses
1. While the reported acceleration is impressive, the paper centers its complexity discussion on the inversion term $\left(\mathbf{K}_0\mathbf{K}_0^\top+\mathbf{K}_1\mathbf{K}_1^\top\right)^{-1}$, which may constitute only a fraction of the total runtime in practical pipelines. It is unclear how much of the measured speedup is attributed to this step versus other components (e.g., calculating K_1 and V_1). Releasing code and adding a detailed runtime breakdown would substantiate the claim.

2. The paper does not specify the final choice of the various hyperparameters. Additionally, the joint effects and correlation between them are not yet clear.

### Questions
1. Which components dominate the runtime in practice? How large is the measured reduction attributable specifically to replacing inversion with the SMW-based LR+D solver? A runtime analysis would help verify that the claimed acceleration indeed stems from the algorithm.

2. How do the authors tune these hyperparameters (search grid, validation criterion, fixed default)? Is it tuned per dataset/model? The authors are encouraged to provide a more detailed explanation of hyperparameter tuning.

3. In the lifetime model editing, SMW can also be used naturally because $\mathbf{K}_t\mathbf{K}_t^\top$ is a low-rank update. Can the author compare the efficiency of the proposed method?

### Soundness
2

### Presentation
3

### Contribution
2
