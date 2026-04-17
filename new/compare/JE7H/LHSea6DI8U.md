---
job_id: e2be43f0-4268-4bc1-9bd5-fa94e4e1ad3a
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: LHSea6DI8U.pdf
paper: A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies continual spatio‑temporal forecasting with graph-based backbones and prompt‑like contextual banks, clearly within learning on graphs, representation learning, and lifelong/continual learning.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present and reasonably complete. The work is technically nontrivial, experiments are substantial, and there are no obvious fatal methodological or theoretical flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, manipulative instructions to automated reviewers, or suspicious formatting.

---

# Expected Review Outcome:

## Summary

The paper proposes STBP, a framework for continual spatio‑temporal forecasting on evolving graphs. It combines a *general spatio‑temporal backbone* (FreNet for frequency‑domain temporal processing plus a dual‑stream linear graph attention, DLGA) with a *contextual pattern bank* that is expanded over time and interacts with the backbone via gating and attention. In continual training, the backbone is frozen after the first period and only the pattern bank parameters are expanded and updated, aiming to preserve general knowledge while adapting to new nodes and distributions. Experiments on three streaming datasets (PEMS‑Stream, CA‑Stream, AIR‑Stream) show that STBP outperforms conventional STGNNs and several state‑of‑the‑art continual spatio‑temporal forecasting (CSTF) baselines in both full‑data and few‑shot regimes, with additional ablations and efficiency studies.

## Strengths

1. **Clear problem framing in CSTF and a coherent architectural story.**  
   The paper clearly articulates four challenges in continual spatio‑temporal forecasting (distribution drift, dynamic spatial correlations, catastrophic forgetting, and incremental adaptation coupling) and designs STBP to address all four jointly. The overview in **Figure 2** gives a reasonably clear end‑to‑end picture: FreNet modules for temporal frequency modeling, DLGA for scalable attention‑based spatial modeling, and the contextual pattern bank interacting with both through prompts.

2. **Strong empirical performance with carefully chosen baselines.**  
   The main results in **Table 1** show substantial gains over both non‑continual STGNNs (GWNet, STID, iTransformer) and recent CSTF methods (TrafficStream, STKEC, PECPM, STRAP, EAC) across three datasets and multiple horizons. For example, on PEMS‑Stream the average MAE improves from 15.67 (best baseline EAC) to 12.31, and on CA‑Stream from 20.20 (EAC) to 15.77. The gains are consistent across horizons (3, 6, 12 steps) and metrics (MAE, RMSE, MAPE), suggesting the method is not just tuned to a particular setup.

3. **Robustness under few‑shot incremental data.**  
   The few‑shot experiment in **Table 2**, where only 10% of data in later periods is used for training, is a nice stress test that is especially relevant for continual learning. STBP again clearly outperforms all baselines; on PEMS‑Stream with 10% data, the MAE drops from 16.13 (EAC) to 13.58, and MAPE from 24.02% to 17.89%. This supports the claim that the combination of frozen backbone + expanding pattern bank can leverage prior knowledge efficiently when new data is scarce.

4. **Reasonable, non‑toy ablations that tease apart backbone vs. bank.**  
   The ablation variants (Retrain, Online, w/o Backbone, w/o DLGA) are well chosen. **Figure 4** shows that removing the contextual pattern bank (Retrain, Online) hurts quite a bit relative to full STBP, particularly in MAE/MAPE on traffic datasets, confirming that parameter expansion plus prompts matter for mitigating forgetting. Conversely, replacing the FreNet+DLGA backbone with a simpler CNN+GCN backbone (w/o Backbone) also degrades performance to near EAC levels, indicating that the proposed backbone itself adds value beyond the prompting trick. The w/o DLGA variant’s visible drop gives evidence that the spatial attention part is not just an implementation detail.

5. **Qualitative analyses that support the heterogeneity/relevance claims.**  
   The t‑SNE visualizations in **Figure 3** and more extensively in **Figure 6** and **Figure 11/12** show that the contextual pattern bank embeddings cluster by traffic/air quality behavior; nodes in the same cluster share similar temporal profiles. Moreover, new nodes in later periods (e.g., nodes 693, 809, 834 in 2017 in **Figure 6**) are pulled into existing clusters that match their temporal patterns. This qualitatively supports the claim that the pattern bank captures both node heterogeneity and relevance and that incremental expansion fine‑tunes representations in a meaningful way.

6. **Scalability and efficiency are empirically studied, not just asserted.**  
   The efficiency comparison in **Figure 8** and **Figure 14** is a useful complement to the accuracy results. STBP’s training time per period and memory footprint are competitive with EAC and better than heavy transformer baselines like iTransformer, despite using a more expressive backbone. The toy experiments also show the practical benefit of linear attention vs. O(N²) attention, aligning with the DLGA design.

7. **Clarity and reproducibility.**  
   The paper is generally well written and easy to follow. The mathematical formulation in Section 3 is standard and precise; key modules (FreNet, DLGA, contextual pattern bank) are described with equations and summarized visually in **Figure 2**. Experimental settings, dataset details (Tables 4–6), and training hyperparameters are spelled out in the appendix, and multiple-period results are given in **Tables 7–8**, which supports reproducibility.

## Weaknesses

1. **Incremental architectural novelty; the method is a composition of known ideas.**  
   The core ingredients are not conceptually new in isolation:  
   - A frozen backbone with small, trainable, task‑ or node‑specific parameters is standard in prompt‑tuning and adapter‑based continual learning. EAC already uses an expanding prompt pool for CSTF.  
   - Node‑specific parameter banks and pattern pools have appeared in works like HimNet and PromptST for spatio‑temporal forecasting.  
   - FreNet essentially applies FFT + learned frequency weights (Eq. (6)), which is close in spirit to many recent frequency‑domain time‑series models.  
   - DLGA is a variant of linear attention (Katharopoulos et al. 2020) plus an extra key stream from the pattern bank.  
   The combination is well engineered for CSTF, but the paper sometimes markets STBP as a “general backbone tailored for continual forecasting” (Page 5) without fully acknowledging how much is incremental relative to existing prompt‑based CSTF (especially EAC) and pattern‑bank works. A more self‑critical positioning would help.

2. **Some methodological ambiguities and loose mathematical specification.**  
   Several key equations and definitions leave nontrivial ambiguities:  

   - **Eq. (4)**: The text states that $\mathbf{P}_{\tau}\in\mathbb{R}^{(N_{\tau}-N_{\tau-1})\times d}$ represents “newly introduced parameters for the current incremental period,” but earlier $\mathbf{P}_\tau\in\mathbb{R}^{N_\tau\times d}$ is the full bank. Overloading $\mathbf{P}_\tau$ and using a prime in $\mathbf{P}_\tau'$ is confusing. More consistent notation like $\Delta\mathbf{P}_\tau\in\mathbb{R}^{(N_{\tau}-N_{\tau-1})\times d}$ and $\mathbf{P}_\tau=[\mathbf{P}_{\tau-1};\Delta\mathbf{P}_\tau]$ would be clearer. Currently one has to infer that the right‑hand $\mathbf{P}_\tau$ in Eq. (4) is actually $\Delta\mathbf{P}_\tau$, which is error‑prone.  

   - **Eq. (5)**: $\mathbf{H}_\tau^{\prime}=\mathbf{P}_\tau^{(1)}\cdot h_\theta(\mathbf{H}_\tau\cdot (1+\mathbf{P}_\tau^{(0)}))$. It is unclear how the elementwise broadcasting and matrix multiplication are defined here. If $\mathbf{H}_\tau,\mathbf{P}_\tau^{(0)},\mathbf{P}_\tau^{(1)}\in\mathbb{R}^{N_\tau\times d}$ and $h_\theta$ is applied row‑wise, then:  
     - Is $\mathbf{H}_\tau\cdot(1+\mathbf{P}_\tau^{(0)})$ meant as elementwise product: $\mathbf{H}_\tau\odot (1+\mathbf{P}_\tau^{(0)})$?  
     - Is the leading multiplication by $\mathbf{P}_\tau^{(1)}$ also elementwise (a gate) or a learned linear map? The notation “·” strongly suggests matrix multiplication, but shapes do not match unless transposes or per‑node linear layers are used.  
     This matters because the expressive power and computational cost of the gating are quite different in these cases. Please formalize the operation, ideally with explicit per‑node equations like $\mathbf{h}_i' = \mathbf{p}^{(1)}_i \odot h_\theta(\mathbf{h}_i \odot (1+\mathbf{p}^{(0)}_i))$.  

   - **Eq. (8–9)**: DLGA is described as linear attention via random features $\phi(\cdot)$, but the text then says “Softmax used for approximation in our implementation,” which is confusing: standard linear attention replaces Softmax with kernel $\phi(Q)\phi(K)^\top$, not the other way around. In Eq. (9), the normalizing denominator from Eq. (10) disappears when moving from the fraction to $(\phi(Q)\phi(K)^\top + \phi(Q)\phi(P^{(2)}_\tau)^\top)V$. This aligns with the usual *unnormalized* linear attention variant, but then calling this a “Softmax approximation” is misleading. Clarifying whether you use normalized or unnormalized linear attention, and whether $\phi$ is a fixed random feature map or a simple nonlinearity (e.g., ReLU), is important for both correctness and reproducibility.

   - **Complexity claims.** DLGA is claimed to reduce complexity to $O(N)$, but Eq. (9) still implies computing $\phi(Q)\phi(K)^\top$ and $\phi(Q)\phi(P^{(2)}_\tau)^\top$, whose naive cost is $O(N^2 d)$. To actually achieve linear complexity, one needs the standard re‑ordering $(\phi(Q)\phi(K)^\top)V = \phi(Q) (\phi(K)^\top V)$, which is indeed $O(N d^2)$, but this is only evident in the scalar derivation in Appendix A.3.1, not in the main text. It would be better to explicitly rewrite Eq. (9) as $\phi(Q)(\phi(K)^\top V + \phi(P^{(2)}_\tau)^\top V)$ and emphasize the role of this re‑ordering. As written, a reader not familiar with linear attention might incorrectly implement the quadratic version.

3. **Evaluation protocol may favor STBP over some baselines.**  
   The continual‑vs‑non‑continual comparison protocol potentially biases against conventional models:  
   - GWNet and STID are retrained from scratch per period *only on that period’s data* (Page 7), which is a very weak baseline, because a “reasonable” offline STGNN could be periodically retrained on all historical data or at least on a sliding window. This choice exaggerates the benefit of any method that can reuse previous parameters.  
   - iTransformer is fine‑tuned online, but full‑parameter fine‑tuning in non‑stationary environments is an intentionally straw‑man setup for forgetting; simple regularization (e.g., weight decay to original parameters) or adapters would be stronger baselines.  
   - For CSTF methods, the authors seem to carefully follow each original paper’s protocol, which is good. However, it would be informative to also run ablations of STBP under analogous “naive” tuning strategies (e.g., full‑model fine‑tuning, or freezing the pattern bank and only tuning the backbone) to see whether its architectural prior is the main driver of performance, or the continual protocol.

   Overall, the direction of improvement is convincing, but the *magnitude* of gains in **Table 1** (especially vs. GWNet/STID) should be interpreted with this caveat.

4. **Limited ablation on the internal design of the contextual pattern bank.**  
   The contextual pattern bank is central to the paper, yet the ablations treat it as a monolith. We only see “with vs. without bank” and “online vs. retrain” in **Figure 4**; there is no quantitative comparison between different internal structures of the bank:  
   - What is the marginal benefit of having three components $\mathbf{P}^{(0)},\mathbf{P}^{(1)},\mathbf{P}^{(2)}$ vs a single embedding that is used for both gating and attention?  
   - How important is the prompt‑based gating in Eq. (5) vs. simply concatenating $\mathbf{P}_\tau$ to the hidden features? Ablating Eq. (5) directly would test the claimed advantage over simpler prompt methods like EAC’s additive prompts.  
   - The t‑SNE plots in **Figures 3, 6, 11, 12** are compelling visually, but without numerical cluster metrics or performance comparisons to alternative bank designs, it is hard to assess whether this specific bank structure is critical or just one workable design.

5. **Scope of datasets and distribution‑shift analysis is narrow.**  
   While PEMS‑Stream, CA‑Stream, and AIR‑Stream cover both transportation and air quality, they are still all urban sensor networks with relatively similar temporal granularity and forecasting horizons. The MMD analysis in **Table 6** is helpful, but the paper’s claims about handling various kinds of distribution drift (Section 4.3) would be stronger if tested on:  
   - At least one additional domain with very different dynamics (e.g., power grids, climate reanalysis data, mobility trajectories).  
   - Synthetic datasets where the degree and type of drift (temporal vs. structural) can be systematically varied.  
   As is, the model clearly works well on the three benchmarks, but its generality as a “general‑purpose backbone” is not thoroughly validated.

6. **Missing or under‑discussed related work on dynamic spatio‑temporal GNNs.**  
   The Related Work section focuses on static‑graph STGNNs and CSTF frameworks, but omits several recent works that explicitly target dynamic or multi‑periodic spatio‑temporal graphs and are directly relevant to DLGA and distribution‑shift handling. These include dynamic multiple‑graph attention and variable‑resolution GNNs (see Missing Related Work section). Some of these also explore efficiency and adaptivity over evolving graphs. A comparison in Section 2 and 4.3 would give a more accurate view of where STBP stands among methods that already learn time‑varying spatial correlations.

7. **Claims about mitigating catastrophic forgetting and distribution drift lack quantitative diagnostics.**  
   The narrative strongly emphasizes reduced forgetting and better handling of drift, but diagnostics are limited to overall MAE/RMSE/MAPE curves. For example, there is no explicit *forgetting metric* (e.g., performance on early periods *after* training on later periods) to show how much previous knowledge is retained, although **Tables 7–8** allow a manual inspection. Similarly, the role of FreNet in mitigating drift is argued mostly qualitatively; there is no controlled comparison where only the temporal module is varied while keeping everything else fixed (beyond the coarse “w/o Backbone” which changes both temporal and spatial modules). Explicit forgetting curves or drift‑sensitivity experiments (e.g., training on early periods then evaluating zero‑shot on later periods with/without FreNet) would make the story more convincing.

## Potentially Missing Related Work

1. **Zheng & Xie, “A Dynamic Stiefel Graph Neural Network for Efficient Spatio‑Temporal Time Series Forecasting”, 2025.**  
   This work proposes a dynamic graph neural network architecture that explicitly models evolving spatio‑temporal relations with efficiency considerations, highly relevant to DLGA’s goals. It should be cited and discussed in Section 2 (Spatio‑Temporal Forecasting), particularly around dynamic spatial modules, and briefly compared in Section 5.2 when highlighting STBP’s scalability.

2. **Shao et al., “Long-term Spatio-Temporal Forecasting via Dynamic Multiple-Graph Attention”, 2022.**  
   They design a dynamic multi‑graph attention mechanism for long‑term spatio‑temporal forecasting, conceptually close to modeling dynamic spatial correlations with attention. It would fit naturally into the “more recent models” paragraph in Section 2, and could be explicitly contrasted with DLGA in Section 4.3 (e.g., differences in complexity, use of pattern banks).

3. **Li et al., “A Variable-Resolution Unstructured Spatio-Temporal Graph Neural Network: An Application to Very Short-Range Weather Forecasting in Guangdong, China”, 2026.**  
   This paper studies variable‑resolution, unstructured spatio‑temporal graphs for weather forecasting, focusing on scalability and adaptability, which is closely related to STBP’s “general backbone” claim and the AIR‑Stream experiments. It should be mentioned in Section 2 and briefly related to FreNet+DLGA in terms of handling heterogeneous graph resolutions and evolving topologies.

4. **Zheng & Xie, “Dynamic Multi-Periodic Spatio-Temporal Graph Neural Network for Multivariate Time Series Forecasting”, 2026.**  
   This work explicitly models multi‑periodic spatio‑temporal dependencies, relevant to STBP’s use of frequency‑domain modeling for periodicity and trends. It would be appropriate to discuss in Section 2 and Section 4.3, both as related frequency‑aware STGNN design and as a potential baseline or conceptual comparator on handling long‑term periodic structures.

## Questions

1. **Clarify the exact operations in Eq. (5).**  
   Could you explicitly write the per‑node equation for the gating mechanism, specifying whether the dot operations are elementwise products or matrix multiplications, and how $h_\theta$ acts (shared MLP per node vs. full graph layer)? This clarification matters for reproducibility and for understanding the capacity of the contextual pattern bank.

2. **Details of the linear attention implementation.**  
   In DLGA, what is the precise choice of $\phi(\cdot)$? Is it a random feature map (e.g., FAVOR+), a simple nonlinearity, or something else? Do you apply any normalization (e.g., dividing by $\phi(Q)(\mathbf{1}\mathbf{1}^\top)$ as in standard linear attention)? Providing a pseudo‑code snippet or at least explicit formulae would help avoid mis-implementations.

3. **Memory scaling of the contextual pattern bank.**  
   Since $\mathbf{P}_\tau\in\mathbb{R}^{N_\tau\times d}$ is expanded without compression (“pure parametric incremental expansion”), how large does $\mathbf{P}_\tau$ get on the largest datasets, and how does its memory footprint compare to backbone parameters? In scenarios with orders‑of‑magnitude node growth, do you foresee the need for pruning or compression, and how might that interact with forgetting?

4. **Fairer baselines and protocol variants.**  
   How sensitive are the main conclusions in **Table 1** to alternative training protocols for GWNet/STID and iTransformer? For instance, if you retrain GWNet on all historical data up to period $\tau$ (or a rolling window), or equip iTransformer with a simple adapter layer that is tuned while freezing the base model, do you still see similar relative performance gaps? Even partial results on one dataset would improve confidence that STBP’s gains are not due primarily to a favorable protocol.

5. **Ablating the internal structure of the contextual pattern bank.**  
   Can you report ablations where:  
   - You remove $\mathbf{P}^{(2)}$ from DLGA (i.e., no pattern‑based keys),  
   - You replace Eq. (5) with simple addition or concatenation of $\mathbf{P}^{(0)}$ to $\mathbf{H}_\tau$,  
   - You use a single shared embedding per node instead of three partitions?  
   A small table analogous to **Figure 4** on at least one dataset would help clarify which parts of the bank are truly necessary.

6. **Quantifying catastrophic forgetting.**  
   Using the period‑wise results in **Tables 7–8**, could you compute and report a standard forgetting metric (e.g., maximum performance drop on a past period after training on all subsequent periods) and compare STBP to EAC and PECPM? This would directly support your claims about better knowledge retention.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The paper uses standard, de‑identified traffic and air quality datasets with no evident privacy, safety, or other ethical red flags.

## Soundness Rating

3: good.  
The methodology is generally sound and well validated empirically, with substantial experiments and ablations. Some mathematical and protocol ambiguities remain, but they are fixable and do not appear to undermine the core empirical findings.

## Presentation Rating

3: good.  
The paper is mostly clear, well structured, and supported by informative figures and tables. A few notational inconsistencies (particularly around the contextual pattern bank and DLGA) and missing related work prevent an “excellent” rating.

## Contribution Rating

3: good.  
The combination of a frequency‑domain backbone, linear attention, and an expanding contextual pattern bank for CSTF is a meaningful step forward. While the ingredients are mostly incremental and there are close conceptual neighbors (e.g., EAC, HimNet, PromptST), the empirical gains and thorough evaluation make this a valuable contribution for the spatio‑temporal and continual learning communities.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper presents a well‑engineered framework that performs strongly across several challenging continual spatio‑temporal benchmarks, supported by both quantitative and qualitative analyses. The main weaknesses are incremental novelty, some methodological ambiguities (especially around the attention approximation and gating), and incomplete positioning against the broader dynamic STGNN literature. With clarifications and expanded discussion, this would be a solid ICLR poster; as submitted, I lean to acceptance but recognize that a stricter view on originality could argue for rejection.

## Reviewer Confidence

4: confident.  
I am familiar with spatio‑temporal GNNs and continual learning, and I carefully examined the equations, figures, and experimental protocols. Some implementation details are underspecified, but they do not affect my overall assessment.