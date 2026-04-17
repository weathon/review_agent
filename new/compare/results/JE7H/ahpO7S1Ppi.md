---
job_id: 30faac11-fd57-49bc-b52d-1557a5290fc3
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ahpO7S1Ppi.pdf
paper: PCTX: Tokenizing Personalized Context for Generative Recommendation
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about generative recommendation, semantic tokenization, and context‑conditioned representations, which fits ICLR’s scope on representation learning and generative models.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Method, Experiments, Results, Discussion/Conclusion, Related Work) are present and reasonably complete. The method is technically sound at a high level, experiments are non‑trivial with strong baselines and ablations, and the paper is clearly written enough to be evaluated.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden or manipulative prompts targeting automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces Pctx, a personalized context‑aware tokenizer for generative recommendation (GR). Instead of mapping each item to a single static semantic ID, Pctx derives multiple semantic IDs per item by clustering user‑context representations (from an auxiliary sequential model) and fusing them with item text embeddings, then quantizing with RQ‑VAE or RK‑Means. A generative recommender is trained on these personalized semantic IDs with data augmentation and multi‑facet decoding, and experiments on three Amazon categories show consistent improvements (up to 8.9% in NDCG@10) over non‑personalized tokenization baselines such as TIGER and ActionPiece.

## Strengths

1. **Clear, well‑motivated problem and conceptual contribution.**  
   The paper pinpoints a real limitation of current GR tokenization: static, item‑only semantic IDs enforce a universal similarity notion that ignores user‑specific interpretation. The argument that prefix‑sharing semantic IDs get similar probabilities under an autoregressive model is convincing, and Figure 1 concretely illustrates how the same watch can correspond to very different “reasons” depending on the user context, making the case for personalized tokenization.

2. **Methodical design that addresses sparsity vs personalization trade‑off.**  
   The proposed pipeline is fairly well thought through:  
   - user‑context representations from DuoRec (Eq. (1)),  
   - item‑wise clustering with adaptive centroid counts (Appendix B),  
   - fusion with text embeddings (Eq. (2)),  
   - quantization to tokens (RQ‑VAE / RK‑Means),  
   - and two forms of semantic‑ID merging (Eqs. (5)–(11)) to handle duplicated and infrequent IDs.  
   The redundancy‑merging section in Appendix E is mathematically precise about how frequencies are updated and how distances between centroids are used, which helps make the sparsity‑control story more credible.

3. **Strong empirical results against a solid set of baselines.**  
   Table 2 shows that Pctx outperforms both conventional ID‑based sequential models and state‑of‑the‑art GR baselines (TIGER, LETTER, ActionPiece) across three datasets and four metrics. Improvements over the best non‑personalized tokenizer (ActionPiece) are substantial for a mature benchmark regime, e.g., +8.90% on NDCG@10 on Scientific and +11.11% on NDCG@5 on Instrument, with statistical significance (paired t‑test, p < 0.05). This supports the claim that personalized tokenization is not just a cosmetic tweak but beneficial in realistic setups.

4. **Careful ablations and component analysis.**  
   Table 3 and Table 8 provide an unusually detailed ablation of the components:  
   - context encoder choice (SASRec vs DuoRec vs static item embeddings),  
   - clustering vs no clustering,  
   - redundant SID merging,  
   - data augmentation,  
   - multi‑facet generation,  
   - using Pctx IDs but TIGER training.  
   The large drop when removing redundant SID merging ((2.2) in Table 3 / 8) strongly supports the importance of balancing personalization and sparsity. Similarly, the degradation of the “w/o Data Augmentation” and “w/o Multi‑Facet Generation” variants ((3.1), (3.2)) validates that multi‑facet decoding is not an arbitrary flourish but actually exploited by the model. Figure 5 and Figure 6 further analyze key hyperparameters γ and τ, illustrating how performance evolves and how sparsity vs personalization is traded off.

5. **Evidence that gains are not simply model ensembling.**  
   The model‑ensemble study in Table 4 and Table 9 shows that combining TIGER with SASRec or DuoRec as an ensemble improves over each individual model, but still lags significantly behind Pctx. That is, naively ensembling a strong context encoder with a GR model does not close the gap, which supports the paper’s claim that encoding the context into the tokenizer is qualitatively different from post‑hoc score fusion.

6. **Interpretability and qualitative analysis.**  
   Figure 3 analyzes the distribution of personalized semantic IDs per item, showing that most items get 1–3 IDs, not an explosion, and that Pctx actually uses multiple IDs more frequently than TIGER, as desired. The case study in Figure 4 is a nice qualitative example: the same StarCraft II item is assigned different semantic IDs [53, 395, 576, 770] vs [53, 412, 576, 770] under “story‑driven” vs “RTS‑focused” user histories, matching intuitive genre facets. The additional explainability experiment with GPT‑4o (Table 7 and Appendix H) is somewhat heuristic but provides extra sanity checks that different semantic IDs correspond to different preference clusters in a human‑interpretable way.

7. **Reasonable robustness claims on design choices.**  
   The RQ‑VAE vs RK‑Means experiment (Table 10) demonstrates that Pctx’s relative advantage persists when using a stronger quantization method (RK‑Means), with RK‑Means uniformly outperforming RQ‑VAE but preserving the superiority of Pctx over baselines. Similarly, swapping the text encoder to Qwen3‑Embedding‑0.6B (Table 11) still yields Pctx > TIGER, suggesting the core idea is not tied to a specific text backbone.

## Weaknesses

1. **Personalization mechanism is indirect and heavily dependent on an auxiliary model; pipeline complexity is high.**  
   The whole personalization story hangs on the quality of DuoRec context embeddings (Eq. (1)). These embeddings are trained with contrastive learning to avoid representation collapse, not to explicitly encode interpretable facets relevant for downstream tokenization. There is no end‑to‑end training that couples the tokenizer and the GR model. The pipeline involves: training DuoRec, clustering per item with fancy Gamma‑based allocation (Appendix B), quantizing with RQ‑VAE or RK‑Means, merging IDs (Eqs. (5)–(11)), plus augmentation and multi‑facet decoding. This is a lot of moving parts and hyperparameters (T, K, C_start, δ, τ, γ, clustering details) whose interactions are not fully explored. In practice, this complexity will make deployment and tuning significantly harder than static tokenizers like TIGER / ActionPiece, and the paper does not provide any discussion about computational cost or sensitivity on the auxiliary model quality.

2. **Limited and somewhat indirect evidence that tokenization is meaningfully *user‑level* personalized, beyond mild token diversity.**  
   Pctx is framed as a “personalized” tokenizer that captures user‑level interpretation standards. However, the personalization is approximated by clustering over *global* context embeddings pooled across all users, then at inference time selecting the closest centroid. This is arguably more like a “multi‑facet item representation conditioned on sequence state” than a strong user‑specific tokenization. Figure 3 shows that the majority of items have only 1–2 semantic IDs, so for most items, personalization is minimal. Figure 7 further shows that without augmentation, early positions are nearly always mapped to the most popular SID, and personalization appears mostly at later sequence positions, but γ must be tuned carefully to avoid destroying that effect. The paper could benefit from more direct metrics: e.g., user‑level diversity of semantic IDs, how often a given user reuses the same SID vs switching across SIDs for the same item, or per‑user performance differences, rather than relying mostly on global NDCG and one anecdotal case study (Figure 4).

3. **Comparative baseline scope and positioning around personalized/dynamic tokenizers is incomplete.**  
   The related work section is primarily centered on static tokenizers (TIGER, LETTER) and a short mention of ActionPiece. Beyond one sentence that ActionPiece uses only “adjacent actions,” there is no deeper empirical or conceptual comparison with other recent work on dynamic or personalized tokenization for GR. Very closely related works like dynamic or journey‑aware tokenization for generative recommendation (e.g., recent dynamic personalized tokenizers or chain‑of‑thought tokenization approaches) are not mentioned. As a result, it is hard to precisely situate Pctx relative to other attempts at dynamic or context‑aware ID design, and the “first personalized tokenizer” claim is likely overstated.

4. **Key design decisions lack deeper justification or alternatives.**  
   Several important choices are made somewhat heuristically:  
   - In Eq. (2), context and feature embeddings are simply concatenated with a scalar weight α, without exploring other fusion mechanisms (e.g., cross‑attention, learned gating, or non‑linear combination). Only a fixed α = 0.5 is used in all experiments (Appendix C.3), and there is no ablation on α.  
   - The centroid allocation strategy (Appendix B) uses a Gamma prior and arithmetic progression across T groups. This is rather elaborate, but there are no baselines with simpler strategies such as “C_{v_i} ∝ log(#interactions)” or fixed small C (e.g., 1–3 per item). It is unclear whether the Gamma smoothing truly matters versus something simpler.  
   - The data augmentation mechanism randomly replaces a semantic ID with another of the same item with probability γ. Figure 5 shows that performance is extremely sensitive near high γ (sharp drop at γ=0.8–0.9). But there is no exploration of more principled augmentation, such as conditioning replacement on similarity between SIDs or using curriculum (start small γ then increase).  

5. **Mathematical formulation and notation issues around the tokenization process.**  
   While Appendix E is relatively clear, the main‑text description in Section 2.2.2 is occasionally confusing:  
   - In Eq. (2), the sentence “$\bm{e}^{ctx}_{v_{i},k}$ is the $k$‑th representation of $\bm{e}_{v_{i},k}$” is self‑referential and likely a typo; it should read that $\bm{e}^{ctx}_{v_{i},k}$ is the k‑th *context centroid* for item $v_i$. This kind of sloppiness can confuse the precise role of these vectors.  
   - In Eqs. (5)–(6), the merging of duplicated SIDs keeps the prefix tokens and chooses $\min\{m_G^{v_i,k_s}\}$ as the collision token. “Min” here is over discrete codebook indices; this is arbitrary, not semantically motivated. It would be better to justify why this choice is OK or to define a more meaningful tie‑breaking scheme (e.g., frequency‑weighted or distance‑based).  
   - In Eq. (10), $\operatorname{dist}(M_{i,k^\star}, M_{i,a})$ is defined as “the euclidean distance of centroid of two personalized semantic IDs,” but the centroids of SIDs are not clearly notated; presumably, this is the centroid of the fused embedding cluster that generated the SID. Making this explicit would improve rigor and reproducibility.  

6. **Lack of analysis versus stronger and arguably simpler baselines that inject user context directly into the generative model rather than tokenizer.**  
   All GR baselines (TIGER, LETTER, ActionPiece) use static tokenization. A natural competitive baseline would be a GR model that takes both item semantic IDs and a learned user‑level embedding as input, or that conditions generation on an additional “user token” derived from the same DuoRec context embeddings, without changing the tokenizer. This would test whether the performance gain comes from *where* personalization is applied (tokenization vs model input), rather than from the mere presence of better user features. The current ablations (e.g., TIGER + SASRec / DuoRec in Table 4) perform score‑level ensembles only, which is not the same thing and may underestimate what model‑level fusion can do. Without this comparison, the necessity of pushing personalization into the tokenizer itself is not fully established.

7. **Scalability and efficiency aspects are not quantified.**  
   GR is championed for memory efficiency and scalability, but Pctx introduces extra storage and preprocessing overhead: per‑item multiple SIDs, per‑item centroid pooling, and potentially non‑trivial clustering costs. There is no reporting of training time, clustering time, or memory overhead relative to TIGER or ActionPiece. For real‑world GR systems where catalogs and user bases are massive and continually changing, these overheads could be decisive. Moreover, Pctx is built entirely offline and does not address how tokenization adapts to new items or evolving user behavior.

8. **Explainability experiment is weakly grounded and uses another LLM as an oracle.**  
   The explainability analysis in Section D.4 uses GPT‑4o to (a) infer cluster‑level preference summaries and (b) judge alignment between summaries and individual sequences. While interesting, this setup is noisy and subject to the biases of the external LLM. Accuracy >0.85 in Table 7 is encouraging but not rigorous evidence; there is no human evaluation or cross‑checker, and no control experiment (e.g., random semantic IDs). The paper should treat those results more cautiously and clearly frame them as anecdotal support rather than strong empirical evidence.

## Potentially Missing Related Work

1. **Feng et al., “Drift‑Aware Continual Tokenization for Generative Recommendation”, 2026.**  
   This work tackles evolving collaborative signals and continual tokenization for GR, which is closely related to how Pctx might handle changes in user behavior over time. It should be discussed in Section 4 (Generative Recommendation) as another approach to dynamic tokenization, contrasting their drift‑aware continual learning with Pctx’s offline, fixed tokenizer.

2. **Wang et al., “PIT: A Dynamic Personalized Item Tokenizer for End‑to‑End Generative Recommendation”, 2026.**  
   PIT proposes a dynamic personalized item tokenizer, conceptually very close to Pctx’s stated goal. It needs to be cited and contrasted in Section 2.4 and Section 4, with explicit discussion of differences: e.g., whether PIT conditions tokenization online per user, how it co‑trains with the GR model, and how Pctx’s clustering‑plus‑quantization approach compares.

3. **Li et al., “GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation”, 2023.**  
   GPT4Rec is a generative framework that models user interests and provides interpretability, relevant to Pctx’s explainability and personalized generation claims. It should be discussed in Section 4, emphasizing that GPT4Rec personalizes at the model level rather than at the tokenizer, and possibly added as a baseline or at least as a conceptual comparator for interpretability.

4. **Ma et al., “GRACE: Generative Recommendation via Journey‑Aware Sparse Attention on Chain‑of‑Thought Tokenization”, 2025.**  
   GRACE proposes a chain‑of‑thought style tokenization and journey‑aware sparse attention for generative recommendation, which is another line of work on richer token structures. It should be referenced in Section 4 and compared to Pctx’s multi‑facet semantic IDs and beam‑search decoding (Figure 2), clarifying how “journey‑aware” tokenization differs from Pctx’s user‑context‑conditioned clustering.

## Questions

1. **Comparison to model‑level personalization:**  
   Can the authors provide results where user context (e.g., the DuoRec sequence embedding) is injected directly into the GR model as an additional input token or conditioning vector, while still using a static tokenizer (e.g., TIGER)? This would more directly evaluate whether tokenization‑level personalization is necessary, or whether similar gains can be achieved by a simpler architecture.

2. **Ablation on centroid allocation strategy and α‑fusion weight:**  
   How sensitive is performance to the choice of $C_{v_i}$ allocation scheme and α in Eq. (2)? Could the authors add experiments with (a) a simple fixed number of centroids per item (e.g., 1, 2, 3), and (b) a small sweep over α (e.g., 0.25, 0.5, 0.75) to show that the method is not overly fragile to these design choices?

3. **Clarification of distance metric in semantic ID merging:**  
   In Eq. (10), what exactly is the representation used to compute $\operatorname{dist}(M_{i,k^\star}, M_{i,a})$? Is it the centroid of the fused embedding cluster, the reconstructed embedding from RQ‑VAE, or something else? Please clarify in the main text and ideally provide a short discussion on whether alternative metrics (e.g., cosine similarity) affect results.

4. **Scalability considerations:**  
   Could the authors report approximate preprocessing time (training DuoRec, clustering, RQ‑VAE/RK‑Means training, merging) and memory overhead versus TIGER/ActionPiece on one dataset, and comment on how the approach would scale to much larger catalogs and frequent catalog updates? Are there obvious incremental update strategies for new items/users?

5. **Quantifying personalization strength:**  
   Beyond Figure 7, can the authors provide quantitative analyses of how often the *same user* sees multiple SIDs for the same item across different contexts, and whether users with more diverse SIDs benefit more in recommendation accuracy? That would better support the claim that user‑level interpretive standards are captured.

6. **Robustness to context encoder choice:**  
   DuoRec leads to better performance than SASRec as a context encoder (Table 3, variant 1.1), yet DuoRec is weaker than SASRec as a recommender (Table 4). Could the authors elaborate on what properties of DuoRec’s embeddings make them better suited for tokenization? Have they tried other contrastive or self‑supervised sequence encoders?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is reasonably well specified and experimentally supported with strong baselines and ablations. Some design decisions are heuristic and not fully justified, but there are no obvious fatal methodological flaws.

## Presentation Rating

3: good.  
The paper is mostly clear, well structured, and uses figures and tables effectively (e.g., Figure 2 for the overall framework, Table 2 and Table 3 for results). A few notational inconsistencies and missing clarifications around the merging and distance computations should be fixed.

## Contribution Rating

3: good.  
The idea of conditioning item tokenization on user context in GR is timely and meaningful, leading to solid empirical gains. However, the conceptual novelty is somewhat diluted by the heavy reliance on an external encoder and the lack of comparison to strong model‑level personalization baselines.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper identifies an important limitation of current GR tokenization, proposes a coherent pipeline to infuse user context into semantic IDs, and demonstrates consistent improvements over strong baselines with solid analysis. At the same time, the approach is complex, the personalization is somewhat indirect, and some comparisons and positioning against closely related dynamic/personalized tokenization work are missing. With clarifications and additional experiments (especially vs model‑level personalization), this would be a strong contribution; in its current form it is a good but not flawless paper.

## Reviewer Confidence

4: confident.  
I am familiar with generative recommendation and semantic ID work and have carefully checked the core methodology, equations, and experimental setup, though I did not re‑implement the method.