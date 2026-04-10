## Summary
This paper introduces MASH, a framework that trains LLMs for selective help-seeking via reinforcement learning with a pay-per-search penalty. The core idea is that optimizing for efficient tool use naturally aligns search decisions with the model’s parametric knowledge boundaries, enabling abstention when search is disabled. Experiments on three QA datasets demonstrate improved tool productivity over prior efficient-search baselines and competitive abstention performance compared to methods explicitly trained for abstention.

## Strengths
- **Novel conceptual contribution:** The link between selective help-seeking and abstention is elegantly motivated and offers a unified approach to improve both tool efficiency and reliability without requiring pre-defined knowledge boundaries for training.
- **Comprehensive empirical evaluation:** The paper thoroughly evaluates multiple datasets, reward penalties, and model scales, and includes insightful analyses (warm-start ablation, oracle helper, out-of-distribution generalization) that deepen understanding of the method’s behavior.
- **Practical warm-start procedure:** The synthetic SFT data generation from a different model is a simple yet effective solution to encourage diverse search behaviors without baking in the base model’s knowledge boundaries, addressing a key exploration challenge in RL.

## Weaknesses
### Major:
1. **Conceptual mismatch in abstention evaluation:** The paper treats search-tag generation (when search is disabled) as abstention, which is a different behavior from verbalized uncertainty (e.g., outputting “I don’t know”). While this is a valid proxy, the direct comparison to methods like DPO and AFH that are trained to output explicit abstention phrases is not entirely fair, and the claim of “analogous” behavior is overstated without evidence that the underlying decision processes are similar.
2. **Incomplete ablation of warm-start contribution:** The reported improvements of MASH over the OTC baseline are confounded because MASH uses warm-start SFT while OTC does not. Without an ablation where OTC is also warm-started, it remains unclear whether the gains stem from the novel reward formulations or simply from better initialization, undermining the claim that MASH’s RL training extracts better search behaviors.
3. **Fragility of training:** The method requires dataset-specific tuning of reward penalties and heavily relies on warm-start to avoid degenerate policies (Table 4). This sensitivity limits robustness and general applicability, as the approach may not transfer easily to new domains without careful hyperparameter selection.

### Minor:
1. **Limited evaluation scope:** Primary experiments use a single base model (Qwen2.5-3B) and three QA datasets. While additional model scales are briefly explored in the appendix, broader evaluation across model families and task types would strengthen claims of generality.
2. **Inconsistent out-of-distribution generalization:** As shown in Section 4.5 and Appendix F, generalization to other datasets is mixed (e.g., models trained on multi-hop data struggle on single-hop questions), suggesting learned policies may overfit to dataset-specific patterns.
3. **Basic theoretical analysis:** The theoretical analysis in Appendix A merely restates the optimality condition of the RL objective and does not provide new insights (e.g., convergence guarantees or the effect of penalty severity on the decision threshold).
4. **Dependence on exact match and LLM judge:** Correctness evaluation uses exact match for training/validation and an LLM judge (DeepSeek-V3.1) for testing. The impact of the judge’s potential biases is not ablated, which could affect reported metrics.

### Trivial:
- None.

## Nice-to-Haves
- Comparison to state-of-the-art selective RAG methods (e.g., SEAKR, DRAGIN) to better contextualize selective search performance.
- Evaluation with varying retriever quality (e.g., a weak retriever) to assess robustness to noisy retrieval signals.
- Extension to other task types (e.g., long-form QA, fact-checking) to demonstrate broader applicability of the framework.

## Removed Points
- **Synthetic data with intentional errors:** The paper intentionally uses synthetic data with 35% errors from a different model to avoid aligning with the base model’s knowledge boundaries. This is a design choice explained in the paper, not a flaw.
- **Answerability threshold λ=0.1:** While arbitrary, this is common practice in prior work, and the paper acknowledges the lack of consensus. We retained this as a minor weakness but not a major issue.
- **Missing composite metric for abstention:** The paper reports both overall accuracy and precision on non-abstained questions, which is standard. Demanding a single composite metric is not necessary.

## Suggestions
- Conduct an ablation study training the OTC baseline with the same warm-start procedure to isolate the effect of the reward formulations.
- Perform a per-question analysis linking penalty severity to the model’s switch from parametric to search behavior based on estimated parametric accuracy.
- Include qualitative examples of successful and failed multi-hop trajectories to illustrate the learned search strategies and their limitations.