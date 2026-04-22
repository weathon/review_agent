# Geo-Invariant Scoring Lead with Domain-Adversarial Transformers

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 6, 2, 2, 6

## Abstract
Predicting B2B lead conversion requires not only modeling long‑range dependencies in richly sequenced customer interactions but also ensuring fair performance across under‑represented geographies. While our DeepScore transformer backbone improved overall AUPR from $0.266$ to $0.360$, it exhibited significant geo‑skew: majority‑region (America) signals dominated feature learning (AUPR $0.474$), leaving East-Asia ($0.262$) under‑served. To address this, we embed a Domain‑Adversarial Neural Network (DANN) module into DeepScore’s architecture. A gradient‑reversal layer connects a geo‑discriminator to the shared transformer encoder, enforcing a minimax game that drives hidden representations to be predictive of conversion yet uninformative of geography. Simultaneously, lightweight geo‑specific classifier heads learn residual region‑nuances without re‑introducing large divergence. DeepScore + geo‑DANN achieves a $4.3 \%$ relative gain in macro‑AUPR and reduces inter‑region AUPR gaps by up to $12.3\%$ , all without degrading America accuracy. To our knowledge, this is the first demonstration of adversarial domain adaptation in large‑scale B2B lead scoring, offering a scalable path to equitable, high‑fidelity predictions across heterogeneous markets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper addresses a meaningful and challenging real-world problem. The core idea of using DANN to achieve geographic invariance is sound and shows empirical promise.

### Strengths
please refer to the questions

### Weaknesses
please refer to the questions

### Questions
The introduction should begin by defining B2B lead scoring as a critical business process for prioritizing sales efforts, directly impacting revenue and resource allocation. A clear sentence like, "In Business-to-Business (B2B) sales, a 'lead' is a potential customer, and 'lead scoring' is the process of ranking these prospects based on their perceived likelihood to convert into a paying client," would be invaluable, if my understanding is correct. 

The authors should elaborate on the high stakes: wasting expensive sales personnel's time on poor leads versus missing out on high-value opportunities. This contextualizes why a small improvement lift in metrics as mentioned in Section 5.2, is actually a multi-million dollar business impact.

All acronyms, especially AUPR (Area Under the Precision-Recall curve), should be defined upon first use, explaining why it is more appropriate than AUC-ROC for highly imbalanced datasets like lead conversion.

The "Motivation and Prior Approaches" section 1.1 is a good start but lacks the depth and references needed to position the work firmly within the existing landscape.

The challenges are stated but not thoroughly dissected. The review of prior work is anecdotal rather than scholarly. The claim that transformers are now being used is not backed by references to specific works in lead scoring, making it difficult to discern the novelty.
The problem should be explicitly framed as a "geo-representation bias" or "distribution shift across geographic markets" in a global model. The paper correctly identifies that this is not mere class imbalance but a deeper distribution shift problem; this point should be emphasized and formalized early on.

The section on traditional approaches should cite key papers or industry reports that established gradient boosting models, e.g., Chen & Guestrin, 2016 for XGBoost; Ke et al., 2017 for LightGBM, as the standard. It should also reference any prior academic work that has attempted to use deep learning or contemporary approaches or address fairness in sales prediction. This would clearly show the gap, e.g., while Tree-Ensembles are standard, they fail to model sequences. While Transformers can model sequences, they exhibit severe geographic bias.  

The authors correctly note that DANN is not a novel ML technique. Therefore, the contribution must be framed around its *novel application and adaptation*.

The paper reads as a straightforward application of DANN to a new dataset. The "contribution" is not sharply defined, making it easy for a reviewer to dismiss as an incremental engineering effort. The contribution should be explicitly reframed to shift the contribution from "inventing DANN" to "successfully adapting and validating it for a complex, high-stakes business problem where it was previously unexplored."

The section of "Theoretical Foundation" is more of an intuitive explanation than a theoretical treatment. For example, the authors cite the Ben-David et al. (2010) bound but do not engage with it rigorously. There is no proof or formal argument for how their specific model minimizes the divergence term \(d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T)\).

The explanation of the error bound is good for intuition. The authors should strengthen it by explicitly linking the DANN's domain classifier to the empirical estimation of the \(\mathcal{H}\Delta\mathcal{H}\)-divergence, as is standard in DANN literature (Ganin et al., 2016). A sentence such as, "The domain discriminator \(D_{\psi}\) directly approximates a function that maximizes the error between domains, thereby providing a learning signal to minimize the feature-level divergence \(d_{\mathcal{H}\Delta\mathcal{H}}\)," would add depth.
The evaluation section is the cornerstone of an applied paper, and here it lacks rigor and clarity. The only direct baseline is a LightGBM model and the authors' own ablated models (such as Single-Head, Multi-Head). There is no comparison to SOTA domain adaptation or fairness-aware methods applied to the same problem.

The dataset of 1.4M leads is described only at a high level. Crucial details are missing: What is the overall conversion rate? How is the data split between train/validation/test? The mention of potential "data collection inconsistencies" in East-Asia is a major red flag that is not adequately discussed.

Key hyperparameters for the transformer (layers, heads, dropout), the DANN (dimension of discriminators), and training (batch size, learning rate) are omitted from the main text, making reproduction impossible.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This papers introduces:
1. DeepScore, a transformer whoses inputs are interaction and profiling embedding. for example clicking of ads.
2. Geo-DANN which is a domain adversial discriminator using the final hidden state from the transformer with gradient reversal layers.
3. The minimax task would be to the minimize the lead classification loss from a regional prediction head while maximizing the loss from the domain adversial prediction of the region

### Strengths
1. The paper describes the problem well that geo-skew dataset causes results to be skew in favour of the majority.
2. It mitigates a well-known problem in this domain through a well-known strategy that has been proven to work in another domain, namely GANs. This allows them to achieve better results.

### Weaknesses
1. The paper did not specific the embedding model which is used to encode both the interaction and profiling features.
2. The paper did not provide the architecture of the transformer used.

### Questions
1. What is the architecture of the transformer model used?
2. How are the features handle across different medium like opening emails, clicking of ads, etc...

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper targets an interesting problem of B2B Lead scoring (and therefore conversion prediction), aiming to address the fair performance across geographical regions. To this end, the authors propose and integrate a domain-adversarial neural network (DANN) into the proposed transformer backbone (i.e., DeepScore). The proposed approach is demonstrated to achieve better macro-AUPR while reducing inter-region gaps in AUPR. The work is claimed to be "the first demonstration of adversarial domain adaptation in large-scale B2B lead scoring".

### Strengths
- The problem studied is scoped in the B2B lead conversion prediction scenario, yet is comparable to many other problems that are widely studied in literature, such as "item recommendation in e-commerce across domains/websites" and "multi-person human activity recognition". So, the research outcome is promising to be transferable to similar problems in other domains.
- The theoretical analysis is interesting regarding why DANN levels geo performance. It gives readers an intuitive understanding of the design details and their rationales.
- The paper is well presented and easy to follow. The authors have put in lots of dedicated discussions to help set up motivation and justify the solution design.

### Weaknesses
- It does not quite convince me about how challenging the research problem is. The distributional shift and data imbalance issues have been a long-standing and extensively studied issue in cross-domain research. While the paper has been motivated mostly by discussing the limitations of region-specific models, it lacks an in-depth discussion of the related work covering methods more similar to the proposed approach. More specifically, Section 2 could offer more insights and comparisons of the methods investigated with the proposed method to provide convincing motivations for the paper and clarify what makes the paper's work novel in the discussed technical contexts.

- The overall approach is kind of straightforward with modest innovation. The main contribution lies in the geo-DANN, which consists of existing frameworks and tricks, to predict conversion while confusing geography.

- The empirical evaluation is mostly limited to DeepScore-based methods. There is a lack of competitive baseline methods included in the quantitative analysis to show how the proposed approach advances the state-of-the-art.

### Questions
Why not show the architecture of the entire architecture where DeepScore is a backbone, but only show DeepScore in Figure 1?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of geographic performance disparity ("geo-skew") in B2B lead conversion prediction , where models perform well in majority regions (like America) but under-serve under-represented regions (like East-Asia). The authors introduce "Geo-DANN DeepScore," a novel architecture that embeds a Domain-Adversarial Neural Network (DANN) module into their DeepScore transformer backbone. This DANN module utilizes a gradient-reversal layer to force the model's shared hidden representations to be predictive of conversion outcomes while remaining uninformative about geographic domains. Concurrently, lightweight geo-specific classifier heads are used to capture region-specific conversion patterns. The resulting model achieves a 4.3% relative gain in macro-AUPR and reduces inter-region AUPR gaps by up to 12.3%, all without compromising the accuracy of the majority region.

### Strengths
The paper reframes the issue as distribution shift and uses DANN to align shared representations while retaining per-geo heads for necessary conditional differences, achieving consistent gains at 1.4M-scale and preserving the majority region—clean design, deployable, and practically relevant.

### Weaknesses
1. Section 1.3 shows multi-head without adversarial training underperforms, but the final system still leans on geo heads (Sec. 3.2) and simply adds DANN (Sec. 3.3). Without a clean contrast among DANN+single head, geo heads only (with matching regularization/budget), and the full DANN+geo heads, it’s hard to credit DANN as the main driver. Same story for the discriminator design: the paper mentions multiple discriminators (geo/segment/BU/size), but there’s no clarity on why multi-discriminators beat geo-only, how their losses are weighted/normalized, or whether they interfere. The GRL schedule (λGRL, γ) is treated qualitatively; no sense of sensitivity (upper bounds, stability, per-region AUPR/Lift). Net-net, the gains might be from head specialization, regularization, or training knobs—not necessarily adversarial alignment.
2. Using smaller AUPR gaps and higher macro AUPR as the main evidence is tricky because AUPR is prevalence-sensitive; per-region base rates should frame any cross-geo comparison. Lift@30% is a top-slice business metric and doesn’t speak to group calibration or consistency. Also, while the adversary is said to cover multiple firmographic attributes, results only show geography—no read on size/segment/BU.
3. Sec. 3.1 builds on “full marketing and sales history,” but the scoring timestamp (e.g., MQL) and strict truncation of post-decision events aren’t specified—those would inflate AUPR/Lift. The train window ends 2024-05 and test is 2024-07–09, with 2024-06 skipped; with a 60-day median lag, that gap raises concerns about cross-period touchpoints or label–feature misalignment. 
Comparisons focus on LightGBM, reweighting, and single-/multi-head variants. Despite citing CDAN/WDGRL (Sec. 2.4), there’s no apples-to-apples positioning against stronger, standard robust/adaptation methods like Group DRO, IRM, MMD/CORAL, CDAN, and WDGRL under the same data/compute budgets.

### Questions
1. Could you clarify how much of the gain you attribute to DANN versus the geo heads? It would also be useful to hear the rationale for multiple discriminators over geo-only, how you balance/normalize their losses, and whether you observed interference.
2. Why are AUPR and Lift@30% appropriate proxies here? Some context on per-region base rates and how prevalence affects AUPR comparisons would be helpful. 
3. A short note on why June 2024 is omitted and how the 60-day lag is handled (lead-level splitting, right-censoring) would address leakage concerns.
4. Conceptually and operationally, how does your approach compare with Group DRO, IRM, MMD/CORAL, CDAN, and WDGRL (compute cost, stability, behavior under geo shift, deployment/maintenance complexity)? Any historical or informal observations in similar setups would help contextualize choosing DANN.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Geo-DANN DeepScore, a transformer-based lead scoring system that learns geography invariant features via a gradient-reversal, multi-discriminator adversarial module, plus small geo-specific classifier heads. On 1.4M leads across 10 markets, it increases macro-AUPR by 4.3% and narrows inter-region gaps (e.g., East Asia and Europe improve) without hurting America performance.

### Strengths
- The paper combines multi-domain DANN with geo-specific heads for B2B lead scoring, and the framing geoskew as a distribution shift is insightful. 
- The method is well-motivated by adaptation bounds with the concrete training schedule and includes multiple baselines and ablation studies.
- The paper is well-written. The pipeline is explained step-by-step with equations and scheduling. 
- The experiment shows the significance of the work by reducing the inter-region gaps, which is appealing for fair deployment without sacrificing business KPIs.

### Weaknesses
- No public benchmark replication limits reproducibility and transfer.
- While re-weighting and multi-head are compared, sensitivity to number and shape of discriminators, alternative adversarial methods (e.g., CDAN/WDGRL), and $\lambda\nu$ schedules could be explored more systematically.
- The paper focuses on geo AUPR gaps. Additional fairness criteria, such as calibration parity across regions, error decompositions, would strengthen the fairness claim in practice.

### Questions
- How sensitive is performance to $\nu$ in the GRL schedule and to the number and depth of domain discriminators?

### Soundness
3

### Presentation
3

### Contribution
3
