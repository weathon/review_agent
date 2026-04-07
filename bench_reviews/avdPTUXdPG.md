## Summary
This paper provides a systematic analysis of region-based representations for MLLMs, demonstrating that (1) visual token ordering has minimal impact on MLLM performance because spatial information is encoded in the ViT's learned positional embeddings, and (2) the success of region-based representations depends critically on visual feature coherence—specifically the smoothness and locality of patch features. The authors identify "high-norm artifacts" and "non-smoothness" as key failure modes and propose practical strategies: selecting smoother visual backbones (RADIOv2.5), applying feature normalization, and using hybrid segmentation+clustering for region partitioning.

## Strengths
- **Systematic evaluation framework**: The paper evaluates region-based representations across performance, efficiency, and interpretability. The "focus metric" for measuring attention alignment with relevant image regions is a novel quantitative contribution to interpretability assessment that goes beyond standard accuracy metrics.

- **Clean experimental validation of token-order robustness**: Table 3 provides well-controlled experiments distinguishing between pre-encoder shuffling (which degrades performance by destroying spatial info) and post-encoder reordering (which has negligible impact). This validates that spatial information resides in ViT positional embeddings and provides principled justification for region-based approaches.


- **Feature incoherence diagnosis with theoretical grounding**: The identification of "high-norm artifacts" and "non-smoothness" as failure modes for region aggregation (Section 4.1, Figure 3) connects to recent work on register tokens (Darcet et al., 2024). The visualizations of PCA features and norm maps effectively illustrate why models like CLIP struggle with region-based aggregation while RADIO performs better.

- **Actionable design guidance with empirical support**: The paper provides concrete recommendations—RADIOv2.5 backbone, RMSNorm normalization, hybrid segmentation+clustering—that are supported by controlled experiments across multiple visual encoders and region sources.

## Weaknesses
- **Lack of quantitative smoothness metric**: The paper argues that feature smoothness is critical for region aggregation but relies on qualitative PCA visualizations (Figure 3). A quantitative metric—such as average pairwise cosine similarity between adjacent patches or spatial autocorrelation of feature norms—would strengthen the core claim and enable reproducible comparison across encoders.

- **Efficiency claims incompletely supported**: While visual token counts decrease substantially (e.g., 576→124 for RADIO in Table 1), the paper does not report the computational cost of region generation via SAM or clustering. If SAM inference adds substantial overhead, the net efficiency gain may be reduced. End-to-end latency measurements including region generation would substantiate the efficiency contribution.

- **Cross-attention aggregation failure unexplained**: Table 5 shows that learnable cross-attention aggregation does not outperform average pooling. The authors suggest "a more complex design might be needed" but provide no deeper analysis. Understanding whether this failure stems from insufficient capacity, wrong inductive bias, or training dynamics would guide future work.

- **Focus metric not validated against task performance**: The focus metric measures attention alignment with annotated regions but the paper does not establish correlation between focus scores and downstream accuracy. Demonstrating that higher focus actually predicts better task performance would strengthen the interpretability claims.

- **Limited evaluation on grounding tasks**: The benchmarks cover general vision-language tasks but exclude fine-grained grounding tasks (e.g., RefCOCO, GQA) where region-based representations would seem most advantageous. Including such tasks would better evaluate the claimed semantic grounding benefit.

- **No statistical significance analysis**: Tables report single numbers without error bars or confidence intervals. Given that many improvements are within 1-3 points, statistical significance of differences is unclear.

## Nice-to-Haves
- Comparison against recent token compression methods (LLaVA-PruMerge, VisionZip) to contextualize efficiency gains relative to existing compression techniques.
- Character-level region visualizations for OCR failure cases to validate the hypothesis that OCR degradation stems from region quality issues.
- Analysis of how region count variance per image affects LLM context management and performance stability.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **"Table 3 shows MMStar drops from 35.07 to 28.27, contradicting token-order robustness"**: This misunderstanding conflates pre-shuffle (before visual encoder) with post-encoder reordering. The paper correctly shows that pre-shuffle degrades performance (destroying spatial information before encoding) while post-encoder reordering has negligible impact. The result supports, not contradicts, the stated claim.

- **"RADIOv2.5 comparison is unfair because it's multi-teacher distilled"**: The paper's core contribution is identifying that smoother features benefit region-based representations. Using RADIO to demonstrate this is appropriate—the comparison illuminates the role of feature quality, which is central to the analysis. This is not an unfair comparison but rather part of the paper's insight.

- **"LLaVA-1.5/Vicuna-7B is outdated"**: The paper acknowledges this limitation in Appendix C and includes Qwen3-8B experiments (Table 6) showing consistent results. While newer baselines would strengthen the paper, this is reasonably addressed.

- **"Benchmarks like POPE are saturated"**: The paper evaluates seven diverse benchmarks including challenging tasks (CV-Bench spatial reasoning, MMStar, MME Cognition). The breadth is adequate for the analysis.

- **"Combined method doesn't consistently outperform"**: The paper appropriately positions the combined method as balancing semantic grounding and feature consistency. The results show it performs reasonably across settings, not that it is universally optimal.

- **"Compare against SOTA token compression baselines"**: This is scope creep. The paper's stated scope is understanding and improving region-based representations through feature quality analysis, not comprehensive comparison with all token compression methods.

## Novel Insights
The paper's most valuable insight is the decomposition of region-based representation challenges into two orthogonal concerns: (1) the spatial information question—demonstrating that reordering is permissible because position is encoded in ViT features, not sequence order—and (2) the feature coherence question—showing that naive aggregation fails when features contain artifacts or are spatially inconsistent. The connection between high-norm artifacts and Darcet et al.'s register tokens provides meaningful theoretical grounding. The finding that RMSNorm helps region-based but not patch-based representations—likely because averaging amplifies outlier artifacts while patches preserve local information—deserves deeper investigation but represents a genuine discovery.

## Suggestions
- Add a quantitative smoothness metric (e.g., mean pairwise cosine similarity of adjacent patches, or variance of feature norms across spatial neighbors) to make the coherence analysis more rigorous and enable direct numerical comparison across backbones.
- Report total inference latency including SAM/region generation time, or at minimum analyze the computational overhead of region generation separately, to substantiate practical efficiency claims.
- Investigate the cross-attention aggregation failure more deeply: is it parameter count, initialization, or fundamental incompatibility with frozen features? Even negative results with clear analysis help the community.


Score: 4.8
GT Score: 3.0