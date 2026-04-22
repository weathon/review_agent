# AIRE-Prune: Asymptotic Impulse-Response Energy for State Pruning in State Space Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
State space models (SSMs) often sacrifice capacity, search space, or stability to offset the memory and compute costs of large state dimensions. We introduce a structured post-training pruning method for SSMs — AIRE-Prune (Asymptotic Impulse- Response Energy for State PRUN(E)ing ) — that reduces each layer’s state dimension by directly minimizing long-run output-energy distortion. AIRE-Prune assigns every state a closed-form asymptotic impulse-response energy based score, i.e., the total impulse-response energy it contributes over an infinite horizon (time), and normalizes these scores layer-wise to enable global cross-layer comparison and selection. This extends modal truncation from single systems to deep stacks and aligns pruning with asymptotic response energy rather than worst-case gain. Across diverse sequence benchmarks, AIRE-Prune reveals substantial redundancy in SISO and MIMO SSMs with average pruning of 60.8%, with average accuracy drop of 0.29% without retraining while significantly lowering compute.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes AIRE-Prune, a post-training compression method to reduce the dimension of latent state space of a trained state-space model. The method is based on the idea of maximally preserving the energy of LTI systems while truncation. Results show that on Long-Range Arena, roughly $60\\%$ of states can be pruned without defecting the model's performance.

### Strengths
* The results of compressing S5 on LRA look compelling. The proposed method seems to perform significantly better than existing methods.
* The energy score of every state can be easily computed without resorting to large matrix operations or simulations, making the approach efficient for large models.

### Weaknesses
* The discussion of energy, while intuitively, is not rigorous. More theory of why the proposed method tends to work better than the existing ($\mathcal{H}\_\infty$-based) ones would strengthen the paper.
* The experiment section could involve more diagnostic experiments. For example, it would be interesting to see which states are pruned by the proposed method and more diagnosis around the kink would be useful.
* While I believe the paper contains nice ideas, its presentation has significant issues:
   * \\citep and \\citet are completely misused.
   * Similarly, en dashes (- in LaTeX) and em dashes (-- in LaTeX) are also misused all over the place.
   * ICLR papers use boldface letters for matrices and vectors.
   * I would avoid the subsection-style, which could disrupt the overall narratives of the paper.
   * The paper is not proofread as it contains many typos. For instance, on line 151, the equation is not properly cross-referenced; on line 150, you wrote "zero-order-hold" but you use "zero-order hold" elsewhere, which is more standard; the caption of Figure 2 misses the right parenthesis, etc.

The paper, in its current form, is not acceptable. I would raise my score to 4 if the author(s) could fix the presentation issues outlined above. I will be happy to further increase the score if my questions below are properly addressed.

### Questions
1. I am confused about eq. (9) and the statement above that "the total layer energy is additive across modes." According to the definition in  eq. (5), the energy is defined as the $L^2$ norm of the transfer function. For a diagonal LTI system, the transfer function is the sum of partial fractions that correspond to all states. However, there is no guarantee in general that these partial fractions are mutually orthogonal in $L^2$. If so, how can it be that the total layer energy is additive across modes?
2. Can you show the energies of the states being pruned in your experiments also as a function of "pruning ratio"? This could potentially explain the "elbow" that you observe in Image, Pathfinder, and PathX.
3. Have you tried the same method on S4; if so, how does the result compare?
4. How does the model benefit from post-training compression in terms of time and space usage?
5. While I understand intuitively why the $\mathcal{H}_2$-based method works, as explained in the paper, I don't have intuition in why it performs better than an $\mathcal{H}\_\infty$-based one. Can the author(s) provide more justification?

### Soundness
3

### Presentation
2

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
This paper introduces AIRE-Prune, a structured post-training pruning method for state space models (SSMs) that uses asymptotic impulse-response energy to identify and remove redundant states. The method assigns each state a closed-form energy score representing its total output contribution over infinite time, then uses layer-wise prefix normalization to enable global cross-layer pruning decisions. Evaluated on S5 MIMO models across Long Range Arena benchmarks, AIRE-Prune achieves 60.8% average state pruning with only 0.29% accuracy degradation without retraining, substantially outperforming the prior LAST method (33% pruning at 0.52% loss). The approach extends classical modal truncation from single systems to deep SSM stacks and provides both typical-case (H₂ energy) and worst-case (H∞) theoretical justifications.

### Strengths
Strong Theoretical Foundation: The paper elegantly bridges classical control theory (modal truncation) with modern deep learning (layer-adaptive pruning). The energy-based criterion has clear physical interpretation: states with low Eᵢ = ‖C:,i‖²₂‖Bi,:‖²₂/(1-|λᵢ|²) contribute minimal long-run output energy.

Practical Algorithm with Closed-Form Solutions: Unlike iterative methods, AIRE-Prune requires only: (1) compute Eᵢ per mode, (2) sort and compute prefix sums, (3) apply global threshold. No matrix inversions or optimization loops needed.

Impressive Empirical Results: Achieving 60.8% pruning at 0.29% accuracy loss substantially improves over LAST (33% at 0.52% loss). The step-function accuracy profiles (Figure 4) suggest the method effectively separates critical from redundant states.

Comprehensive Mathematical Analysis: Appendix B derives H∞ certificates showing ε ≤ κ(ρ)min{∑√Eᵢ, √|T|∑Eᵢ}, proving the energy-based method also bounds worst-case distortion. This dual justification is valuable.

Actionable Architectural Insights: Layer-wise profiles show task-dependent patterns—some layers can be entirely removed (enabling block-structured speedups), which is more deployment-friendly than fine-grained sparsity.

### Weaknesses
Severely Limited Experimental Scope:
- Only S5 models evaluated: No experiments on Mamba, Mamba2, or hybrid architectures which dominate current practice
- Only LRA benchmark: Missing speech (Speech Commands), language modeling (WikiText), or modern long-context tasks
- Input-selective SSMs (Mamba) have input-dependent B, C—the energy formulation assumes these are fixed. How does AIRE extend to this case?


Incomplete Comparison with LAST:

- Table 1 shows different pruning ratios per task, making direct comparison difficult. Need controlled experiments at {30%, 40%, 50%, 60%, 70%} pruning with both methods. Computational cost comparison missing: Is AIRE faster than LAST to compute scores?


Missing Key Experiments:
-No retraining: Even 1-2 epochs could show full potential. Other SSM pruning work achieves near-zero loss with minimal fine-tuning
- No ablations: What if using median normalization instead of prefix-sum? How sensitive to ε?
- No failure analysis: Why does ListOps only tolerate 20% pruning? Is this fundamental or fixable?


Limited Technical Analysis:

- Equation 8 assumes |λᵢ|²ᵀ → 0, but convergence rate matters for numerical stability near the unit circle. Claim that entire layers can be removed needs gradient flow analysis—does this harm subsequent fine-tuning?
-No discussion of how to extend to non-diagonal parameterizations (e.g., low-rank structured matrices)

### Questions
- Mamba Extension (Critical): Can you extend AIRE to input-dependent SSMs? For Mamba, where Bₜ = sB(xₜ), Cₜ = sC(xₜ), how would you estimate Eᵢ? Monte Carlo over calibration data? Expected energy under input distribution?

- Direct LAST Comparison: Can you provide results at matched pruning ratios (e.g., both methods at 50%, 60%) to help me get an equivalent comparison? What is the compute cost ratio for scoring?
Retraining: What accuracy can you achieve with 1-5 epochs of fine-tuning after AIRE pruning? This would strengthen claims about practical deployment.
-Layer Collapse: When entire layers are removed, does this create training instabilities if later fine-tuning? Have you tested this?

-Failure Modes: ListOps only tolerates 20% pruning. Is this because: (a) the task truly needs full capacity, (b) S5 architecture is suboptimal for this task, or (c) AIRE scoring doesn't capture syntactic structure dependencies? Can you investigate?

-Computational Complexity: What is the wall-clock time for computing AIRE scores vs. LAST scores for a typical S5 model? Is O(n) per state vs. O(n) amortized?


I appreciate the authors for their wonderful work. I learned a lot while reading the manuscript.  I do understand that I asked a lot of questions. However, as long as the critical and decent number of other questions/concerns are answered, I am happy to increase my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a novel state pruning technique for SSMs. The core idea is to use the total impulse-response energy as a criterion to identify and remove less important states. The authors demonstrate that this method enables more aggressive pruning ratios compared to existing baselines, achieving new state-of-the-art results on the benchmark task.

### Strengths
1.  Novel Pruning Criterion: The paper proposes a new and theoretically motivated pruning technique for SSMs. 
2.  Strong Empirical Results: The method achieves state-of-the-art performance on the tested benchmark (S5 model), effectively demonstrating its capability to outperform prior pruning approaches in that specific setting.

### Weaknesses
1.  **Clarity of Methodology (Sections 3.2 & 4):** The paper's core methodology is difficult to understand as written. Section 3.2 lacks clarity and citations to fully grasp. Furthermore, the theoretical part (Section 4) is hard to understand

2.  **Limited Evaluation and Generalizability:** The empirical validation is a significant weakness. The approach is only tested on a single model (S5) and a single benchmark. This narrow scope makes it impossible to assess the generalizability of the technique. It remains unclear whether these performance gains would translate to other important SSM architectures (e.g., Mamba) or to different tasks and datasets. The authors should expand their evaluation to include more models and benchmarks.

3.  **Unclear Motivation and Practical Impact:** The paper's motivation for state pruning is not well-articulated, and its practical impact is obscure. The authors claim, for example, to prune "60% of states," but it is unclear what this means in practice. In many modern SSMs (like Mamba), the state itself represents a small portion of the total parameters. The paper fails to connect state pruning to crucial downstream metrics. The authors must clarify:
    * What is the impact of state pruning on the total parameter count?
    * Does this pruning translate to tangible latency improvements (e.g., in inference or training) or memory reduction?

    Without this context, the practical benefits of the proposed method are unclear.

Minor:

* Missing Citations: Several claims and components lack proper attribution.
    * Section 3.1 makes assertions that require supporting citations.
    * The S5 model is mentioned in Section 3.2 but is not cited.
    * There appears to be a missing reference at L151.
* Undefined Acronyms: The terms "CT" (Continuous-Time) and "DT" (Discrete-Time) are used before they are formally defined, which could confuse readers.

### Questions
1.  **Motivation and Practical Impact:**
    * Could the authors provide a clearer motivation for state pruning, especially in the context of models like Mamba where the state itself is a small fraction of the total parameters?
    * When the paper claims to prune "60% of states," what is the corresponding impact on the total parameter count of the model?
    * Does this state pruning translate into tangible performance gains, such as reduced latency (in inference or training) or lower memory usage? We request that the authors provide empirical measurements for these metrics.

* **Generalizability:**
    * How well does the proposed impulse-response energy criterion generalize beyond the S5 model? Have the authors tested this approach on other significant SSM architectures, such as Mamba or S4?
    * To what extent are these results dependent on the specific benchmark used? How does the method perform on other tasks or datasets?

* Could the authors revise Sections 3.2 and 4 to make them more readable

### Soundness
2

### Presentation
2

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
The paper proposes a post-training pruning method for state space models (SSMs) that reduces state dimensions by minimizing long-run output energy distortion. Unlike previous worst-case gain approaches, their method assigns each state a closed-form asymptotic impulse-response energy score. These scores are normalized layer-wise for global comparison. The method achieves good results on Long Range Arena benchmarks, pruning 60.8% of states on average with only 0.29% accuracy drop without retraining.

### Strengths
- Good empirical results: 60.8% average pruning with only 0.29% accuracy drop without retraining on LRA
- Closed-form solution for importance scores seems efficient
- Energy-based metric is well-motivated from control theory perspective and has clear mathematical grounding
- Works for both SISO and MIMO SSMs

### Weaknesses
- Minor: missing reference details in line 117
- Limited to diagonal/diagonalizable SSMs and doesn't extend to input-selective models like Mamba
- Only evaluated on Long Range Arena without speech and language benchmarks
- No retraining experiments to show potential further improvements
- Comparison mainly against only one recent baseline (LAST)
- No discussion of computational overhead of computing energy scores
- Limited analysis of why certain tasks (ListOps) are more sensitive to pruning

### Questions
Are entire layers ever actually pruned? The paper states: "our method can help achieve low latency SSM model as we are able to prune layers"

### Soundness
3

### Presentation
3

### Contribution
3
