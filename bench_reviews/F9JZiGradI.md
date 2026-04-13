## Summary

The paper proposes MLP-KAN, a unified architecture combining Multi-Layer Perceptrons (MLPs) and Kolmogorov-Arnold Networks (KANs) within a Mixture-of-Experts framework embedded in a Transformer. The goal is to eliminate manual model selection between representation learning (MLP) and function learning (KAN) by dynamically routing inputs to appropriate experts. Experiments span computer vision (CIFAR-10/100, mini-ImageNet), NLP (SST-2), and symbolic regression (Feynman dataset).

## Strengths

- **Clear conceptual motivation**: The paper identifies a genuine tension between architectures optimized for representation learning versus function approximation, and proposes a principled integration strategy using Soft MoE routing to combine MLP and KAN experts within a Transformer backbone.
- **Comprehensive cross-domain evaluation**: The experiments cover three distinct task types—image classification, sentiment analysis, and symbolic regression—which demonstrates effort to validate the "universal" capability claim across domains requiring different inductive biases.
- **Function learning results show clear MLP advantage**: On the Feynman dataset (Table 2), MLP-KAN dramatically outperforms standalone MLP on most equations (often by orders of magnitude in RMSE), validating that KAN-based experts add substantial value for symbolic regression tasks.
- **Ablation studies included**: Tables 4 and 5 provide useful analysis of expert count and Top-K routing sensitivity, showing that 8 experts with Top-2 routing provides reasonable trade-offs.

## Weaknesses

- **Claims contradict experimental results in function learning**: The abstract states MLP-KAN "consistently outperforms" baselines, and Section 5.2 claims it "outperforms both KAN and MLP across a variety of equations." However, Table 2 shows KAN achieves a **better average RMSE (2.09×10⁻²) than MLP-KAN (2.58×10⁻²)**. Additionally, KAN wins on 13 of 30 individual equations. The paper overstates MLP-KAN's function learning performance. — This matters because the core claim of unified superiority is not supported by the data.

- **Text contains factual error about results**: Section 5.2 states "MLP-KAN outperforms both KAN and MLP with an RMSE of 7.18×10⁻²" for equation 1.15.3t, but Table 2 shows KAN achieves 3.69×10⁻², which is nearly half the error. The table correctly marks MLP-KAN as second-best (underlined), contradicting the text. — This matters for scientific accuracy and reader trust.

- **Representation learning shows no consistent advantage**: On 6 of 8 CV/NLP metrics in Table 3, MLP-KAN is **second-best behind standalone MLP**, with differences within or near one standard deviation. MLP-KAN wins only on SST-2 accuracy. — This matters because the unified model does not match specialist performance on the majority of representation tasks.

- **Missing critical ablation: heterogeneous vs. homogeneous MoE**: The paper does not compare MLP-KAN against an all-MLP MoE or all-KAN MoE with equivalent expert count. Without this comparison, we cannot determine whether any performance changes stem from expert heterogeneity versus simply having more capacity through the MoE structure. — This matters because it's the core technical claim that mixing expert types provides benefit.

- **No routing behavior analysis**: The paper claims dynamic adaptation between representation and function experts but provides no analysis of actual routing weights. Do CV tokens preferentially route to MLP experts? Do symbolic regression inputs route to KAN experts? Without this, the "dynamic adaptation" claim is unsupported. — This matters because it's central to the paper's motivation.

- **Equation 13 inconsistent with Soft MoE description**: The Transformer block output uses a uniform average (1/NE)∑Fₑ over all experts, while Section 4.1 describes the Soft MoE routing with learned dispatch weights α. The relationship between these is unclear. If outputs are uniformly averaged at the block level, the Soft MoE routing plays no role in the final output.

- **No computational efficiency comparison**: KAN operations (spline evaluations) are computationally heavier than MLP matrix multiplications, yet the paper provides no FLOPs, training time, or inference latency comparisons. The claim of "maintaining efficiency across diverse datasets" cannot be evaluated.

## Nice-to-Haves

- Comparison against standard MoE Transformers (e.g., Switch Transformer baseline with all-MLP experts) to isolate the contribution of KAN integration from MoE capacity benefits.
- Visualization of routing weights across task types to validate that the mechanism learns to separate representation and function inputs.
- Clarification of the relationship between Soft MoE routing and the uniform averaging in Equation 13.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Citation of MoE literature**: The harsh critic notes Soft MoE is not cited in related work. While this should be addressed, it's a fixable omission rather than a fundamental flaw.
- **Formula formatting in Table 2**: Some equation labels appear duplicated or incorrect (e.g., multiple rows showing the same formula). While this affects presentation, it's not central to evaluating the method's validity.
- **Comparison against state-of-the-art models**: Demanding comparison against ResNet, ViT, or BERT is scope creep. The paper's contribution is combining MLP and KAN experts; comparing against these baselines is appropriate for establishing that baseline.
- **Parameter count matching**: The harsh critic suggests MLP-KAN may have more parameters than standalone KAN. While relevant for fairness, parameter-matched comparisons are not standard practice in MoE papers where the goal is to leverage additional capacity.

## Novel Insights

The paper identifies a meaningful architectural insight: KANs' spline-based structure provides superior function approximation inductive biases for symbolic tasks, while MLPs' fixed activations may be preferable for high-dimensional representation learning. The MoE combination is a natural but non-obvious integration that, if successful, would simplify practitioner decisions about model selection. However, the empirical results suggest the integration introduces overhead without consistently improving over either specialist—particularly for representation learning where standalone MLP performs equivalently or better. The paper would be significantly stronger if it analyzed why the MoE routing doesn't preferentially select KAN experts for symbolic tasks (or whether it does), which would validate or invalidate the core premise.

## Suggestions

1. **Correct the claims**: Revise the abstract and Section 5.2 to accurately reflect that MLP-KAN performs comparably to (not better than) KAN on function learning averages, and does not consistently outperform MLP on representation tasks.
2. **Add routing analysis**: Include a heatmap or statistics showing expert selection distributions for different task types. This is essential to verify the mechanism works as claimed.
3. **Add homogeneous MoE baseline**: Train an all-MLP MoE with 8 experts to isolate whether benefits come from heterogeneity versus capacity.
4. **Report compute metrics**: Include training time, parameter counts, and FLOPs for fair comparison.

## Evaluation

**Novelty**: Moderate. Combining KAN and MLP experts via Soft MoE in Transformers is a reasonable architectural contribution, but builds directly on existing MoE frameworks without substantial algorithmic innovation.

**Technical Soundness**: Mixed. The Soft MoE formulation is correctly described, but Equation 13's uniform averaging is inconsistent with the routing mechanism. The empirical methodology is standard, but missing critical ablations.

**Empirical Support**: Weak. The main claims (consistent superiority in both domains) are not supported by the data. Function learning shows worse average RMSE than KAN; representation learning shows no consistent advantage over MLP.

**Significance**: Limited. Without demonstrating that expert heterogeneity provides benefit over homogeneous MoE, or that routing actually separates tasks, the practical utility is unclear.

**Clarity**: Generally clear but undermined by inaccurate textual claims that contradict the presented tables.

MY FINAL SCORE: <pineapple>4.5</pineapple>