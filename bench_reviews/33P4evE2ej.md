## Summary

This paper proposes DynaMer Adapter, a method for adapting Vision Transformers to medical imaging tasks by dynamically merging tokens from two pre-trained models: a general-domain ViT (DINO v2) and a medical-domain ViT. The architecture combines a Gated Mixture-of-Experts adapter for token-level fusion with a layer-wise skipping router for computational efficiency. Experiments on the Med-VTAB benchmark demonstrate improvements over prior adapter methods across medical imaging tasks, with particular emphasis on out-of-distribution and data-scarce scenarios.

## Strengths

- **Comprehensive evaluation scope**: The paper evaluates on 23 datasets across color medical images, X-rays, and OCT/CT/MRI modalities, providing broad coverage of medical imaging tasks. Performance improvements over strong baselines (GMoE-Adapter, VPT variants) are consistent across nearly all datasets.

- **Novel token-level fusion mechanism**: Unlike prior work that combines features at the layer or network level, DynaMer operates at the token level, enabling finer-grained integration of general and medical domain knowledge. The gating mechanism that balances original vs. adapter-processed tokens addresses training stability concerns.

- **Practical efficiency focus**: The layer-wise skipping router provides a configurable trade-off between inference time and accuracy (Table 7 shows inference time reduction from 0.165s to 0.086s with minimal accuracy change), which is relevant for deployment-constrained medical settings.

- **Thorough ablation coverage**: Tables 4-7 examine the contribution of gating mechanisms, gating dimensions, gating layers, and token-skipping ratios, providing insight into architectural design choices.

## Weaknesses

- **Misleading parameter efficiency claims**: The paper reports "Total Params" as ~1.21X relative to single-backbone baselines, but this appears to count only adapter parameters while **running two complete ViT-B/16 backbones** simultaneously. Since ViT-B has ~86M parameters, the true system uses roughly 2× the backbone parameters of single-backbone methods. The efficiency comparison is fundamentally incomplete without reporting total FLOPs or end-to-end inference cost accounting for both backbones.

- **Missing architectural details for reproducibility**: The MoE expert architecture `AdapE_j` is never specified. The paper states that experts "take two tokens from general and medical models, and output an integrated one" but does not describe whether this is implemented as concatenation followed by an MLP, cross-attention, or another mechanism. Additionally, critical hyperparameters (number of experts `n`, top-`k` value) are not reported in the main text.

- **Unexplained experimental anomaly in Table 7**: Processing 50% of tokens yields *higher* accuracy than processing 100% (e.g., HyperKvasir: 70.85 vs 70.82). This counterintuitive result—less computation producing better performance—warrants discussion. Possible explanations include regularization effects or overfitting at 100%, but the paper does not address this.

- **Naming inconsistencies suggesting editorial errors**: Table 9 labels the proposed method as "**GL-MoF Adapter**" instead of "DynaMer Adapter" used throughout the rest of the paper. Tables 1 and 2 use "MoE-Adapter" vs "MoF-Adapter" inconsistently for the same baseline method (Tong et al., 2024b). These inconsistencies undermine confidence in the experimental reporting.

- **Domain mismatch between medical backbone and evaluation tasks**: The medical ViT is pre-trained on cell images (Nguyen et al., 2023), yet evaluation includes X-ray, CT, and MRI datasets. The paper provides no ablation testing whether a domain-matched medical backbone would improve results, or whether the cell-image backbone provides any signal beyond noise for non-pathology modalities.

- **Marginal improvements without statistical validation**: Improvements over GMoE-Adapter are consistently small (often <0.5% absolute, e.g., 70.82 vs 70.75 on HyperKvasir). No confidence intervals, standard deviations, or significance tests are reported, making it difficult to assess whether these improvements are meaningful or within experimental noise.

- **Table 4 ablation presentation is unclear**: The table shows four rows with identical checkmarks for "General Gate" and "Medical Gate" but varying parameter counts (1.19X to 1.21X). What differs between these configurations is not explained, making the ablation uninterpretable.

## Nice-to-Haves

- **Simple ensemble baseline**: Compare against a naive ensemble of two single-backbone adapters to determine whether "dynamic merging" outperforms standard ensembling without additional architectural complexity.

- **Expert specialization analysis**: Visualize which MoE experts activate for which modalities (X-ray vs. pathology vs. MRI) to validate that the routing mechanism learns meaningful specialization.

- **Dual general-backbone ablation**: Test merging two general-domain ViTs (e.g., DINO v2 + CLIP) to isolate whether gains come from domain complementarity or simply from using two models.

- **Significance testing**: Report mean ± std over multiple seeds (at least 3-5 runs) to confirm improvements are statistically meaningful.

## Removed Points

*These points are flagged to be removed or treated with caution:*

- **Formatting nitpicks** (mentioned by reviewers but not substantive): Minor grammatical issues like "four four folds" do not affect technical contribution.

- **Claim about "stand on two shoulders" title being imprecise**: The title is metaphorical and appropriate; this is stylistic criticism.

- **Claim that contribution 4 (generalizability) is speculative**: The paper does demonstrate on FGVC and VTAB-1K in Table 10, providing empirical evidence beyond speculation.

- **Demand for "standard vs. full fine-tuning comparison"**: This asks the paper to address a different evaluation paradigm. The paper focuses on parameter-efficient adaptation; comparing against full fine-tuning would be a different scope.

## Novel Insights

The token-level merging approach represents a meaningful shift from feature-level or layer-level fusion in multi-model adaptation. The observation that 50% token processing outperforms 100% suggests potential regularization benefits from selective token processing—this could inform future work on when and how to prune token computation. The consistently better OOD performance (Tables 8-9) suggests that dual-domain knowledge genuinely helps generalization, though stronger evidence (e.g., statistical tests, broader OOD datasets) would strengthen this claim.

## Suggestions

1. **Report total system parameters and FLOPs**: Explicitly state the full parameter count and computational cost including both backbones. If dual-backbone overhead is acceptable in context, defend it clearly.

2. **Specify expert architecture**: Add a sentence describing `AdapE_j` (e.g., "Each expert is a 2-layer MLP with hidden dimension 256 that takes concatenated token pairs as input").

3. **Report hyperparameters in a dedicated table**: Include number of experts, top-`k`, learning rate, batch size, and training epochs for reproducibility.

4. **Explain or fix Table 4**: Clarify what varies between the rows, or restructure the ablation to show clearer variable manipulation.

5. **Fix naming inconsistencies**: Standardize method names across all tables (DynaMer vs GL-MoF, MoE-Adapter vs MoF-Adapter).

---

**Quality Assessment:**

- **Novelty**: The token-level MoE fusion mechanism for dual-domain adaptation is a reasonable architectural contribution, building on but extending prior MoE adapter work.

- **Technical Soundness**: Core methodology is sensible, but missing implementation details (expert architecture, hyperparameters) and unexplained anomalies (Table 7) raise reproducibility concerns.

- **Empirical Support**: Extensive evaluation across 23 datasets is commendable, but marginal improvements (<0.5%) without statistical testing weaken the evidence. The dual-backbone computational cost being omitted from efficiency claims is a significant transparency issue.

- **Significance**: If the improvements hold up under statistical scrutiny, the method could be valuable for medical imaging adaptation. However, the domain mismatch between the medical backbone (cell images) and evaluation tasks (X-ray, CT, MRI) limits claims about principled domain fusion.

- **Clarity**: The paper has notable issues including table naming inconsistencies, unclear ablation presentations, and missing architectural details that should be addressed.

MY FINAL SCORE: <pineapple>5.5</pineapple>