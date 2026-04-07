## Summary
Pctx introduces a personalized context-aware tokenizer for generative recommendation (GR), conditioning semantic IDs on a user’s interaction history to capture diverse interpretations of the same item. The method addresses sparsity via adaptive clustering and redundant ID merging, and demonstrates significant improvements over non-personalized GR baselines on three Amazon datasets.

## Strengths
- **Novel contribution**: This is the first work to personalize tokenization in GR, directly addressing the limitation that static tokenization enforces a universal similarity standard. The core idea is well-motivated and clearly differentiated from prior context-aware tokenizers (e.g., ActionPiece) that only consider local context.
- **Comprehensive empirical validation**: Experiments on three datasets show statistically significant gains (up to 8.9% NDCG@10) over a wide range of strong baselines. Ablation studies rigorously validate each component, and additional analyses (model ensemble, hyperparameter sensitivity, explainability) substantiate the claims.
- **Reproducibility**: The paper provides extensive implementation details, hyperparameter settings, and publicly released code, aligning with ICLR’s reproducibility standards.

## Weaknesses
- **Inference aggregation method is unspecified**: Section 2.3 states that probabilities from multiple semantic IDs for the same item are aggregated, but the exact operation (sum, max, etc.) is not given, hindering exact replication.
- **Computational efficiency and scalability are unaddressed**: The pipeline involves pre-training an auxiliary model (DuoRec), clustering, and merging steps. The overhead relative to static tokenization and scalability to very large datasets are not discussed, which is a practical concern for a paradigm that often emphasizes efficiency.
- **Limited comparison with multi‑identifier baselines**: While the paper discusses MTGRec (which assigns multiple static IDs per item) in Section 2.4, it does not include an experimental comparison. This leaves open whether the gains stem from personalization or merely from having multiple IDs per item.
- **Missing controlled baseline for token diversity**: The ablation with random target (γ=1) shows that arbitrary swapping hurts, but a stronger baseline that assigns multiple non‑personalized IDs per item (e.g., via clustering item features alone) is absent. Without it, the isolated effect of personalization versus token diversity is unclear.
- **Suboptimal quantization choice**: Main experiments use RQ‑VAE, but Appendix G.1 shows RK‑Means yields better performance. This suggests the reported gains may be conservative and raises questions about the primary quantization method selection.

## Nice-to-Haves
- Analysis of how context length (e.g., last 5 vs. all interactions) affects personalization, to clarify whether long‑term history is necessary.
- Performance breakdown by user/item frequency (head vs. tail) to see if improvements are uniform or concentrated.
- Visualization of the semantic ID space (e.g., t‑SNE of fused representations) to visually confirm that different IDs for the same item correspond to distinct context‑driven clusters.
- Quantification of how often beam search produces different semantic IDs for the same candidate item, verifying the claimed “multi‑facet” generation behavior.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- “Abstract does not specify baseline”: The abstract states improvement “over non‑personalized action tokenization baselines,” and Table 2 identifies ActionPiece as the strongest baseline.
- “Auxiliary model choice is insufficiently justified”: The paper includes an ablation comparing DuoRec and SASRec (Table 3) and explains DuoRec’s contrastive learning yields more distinguishable representations.
- “Hyperparameter sensitivity of the clustering scheme is a barrier”: Appendix B shows the optimal configurations are robust across datasets and provides detailed tuning ranges.
- “Explainability experiment relies on an external LLM”: This is a supplementary analysis using GPT‑4o, acknowledged as such, and does not affect the core results.
- “Missing explicit limitations section”: While not a dedicated section, limitations (e.g., dependency on an auxiliary model, need for item features) are discussed in the text and future work.

## Novel Insights
The paper’s key insight is that static tokenization in GR implicitly enforces a universal item‑similarity standard, which can be broken by conditioning semantic IDs on the user’s interaction history. This allows the same item to be interpreted differently across users, capturing personalized facets. The work demonstrates that meaningful personalization can be achieved through context‑aware tokenization, and the gains are not merely an artifact of increased token diversity (as shown by the random‑target ablation). This opens a new direction for personalization in generative recommendation.

## Suggestions
- Specify the probability aggregation method used during inference (e.g., sum or max over semantic IDs of the same item) in Section 2.3.
- Add a baseline that assigns multiple static semantic IDs per item via clustering of item features alone, to isolate the effect of personalization.
- Include an experimental comparison with MTGRec or other recent multi‑identifier GR methods to strengthen the claim that personalization is the key driver.
- Discuss computational overhead (time/memory) of the tokenization pipeline relative to static tokenization, either in the main text or appendix.
- Consider using RK‑Means as the primary quantization method, or justify the choice of RQ‑VAE more prominently, given the appendix results.