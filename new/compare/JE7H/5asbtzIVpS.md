---
job_id: 068fd625-4ea0-4172-a432-fba2e704f623
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 5asbtzIVpS.pdf
paper: Forest-Based Graph Learning for Semi-Supervised Node Classification
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅. The work is on semi-supervised node classification with a new graph learning paradigm (forest-based message passing), squarely within “learning on graphs and other geometries & topologies” and semi‑supervised representation learning.

## Minimum Quality  
Pass ✅. The paper is complete (abstract, intro, related work, method, theory, experiments, conclusion), technically substantial, and experiments are extensive with strong baselines. I did not spot fatal methodological or evaluation flaws.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅. I did not find hidden prompts or instructions targeting automated reviewers; the content is standard scientific writing.

---

# Expected Review Outcome:

## Summary

The paper proposes **Forest-based Graph Learning (FGL)**, a framework that models message passing on graphs as transport over a *forest* of spanning trees for semi-supervised node classification. Trees are sampled from a homophily-biased distribution learned via an edge-homophily estimator, then a general tree aggregator (instantiated as a linear operator) performs global propagation in linear time, and multiple trees are fused with a local module to yield the final node embeddings. The authors provide theoretical analysis showing how improving edge-homophily estimation sharpens the distribution toward homophilous trees (Theorem 2) and how their tree aggregator implements global pairwise interactions with linear complexity (Theorem 1). Extensive experiments on 9 benchmarks show competitive or better accuracy than strong GNNs and graph transformers, with favorable runtime, plus detailed ablations and hyperparameter studies.

## Strengths

1. **Clear paradigm shift and well-motivated idea**  
   - The work articulates a *structural* view on the local/global trade-off: total cost ≈ cost per structure × number of structures (Eq. (1)), then argues that spanning trees are the sparsest globally covering structures. This is a fresh and reasonably convincing way to rethink deep-local vs shallow-global paradigms in graph learning.  
   - The use of a *forest* of homophily-biased spanning trees is conceptually clean and directly connected to semi-supervised node classification objectives.

2. **Technically nontrivial tree aggregator with generality**  
   - Theorem 1 (Sec. 4.3, Eq. (5)-(6)) derives a two-recursion scheme for any message aggregator satisfying the Combine / Disentangle properties via operators \(\mathcal{M}^+\), \(\mathcal{M}^-\). This is a nice abstraction that cleanly separates a generic aggregator \(f_{\text{Agg}}\) from tree-specific recursion.  
   - The linear instantiation in Eq. (7)-(8) is simple yet allows incorporating edge attention weights \(\alpha_{i\to j}\) from Eq. (3). The resulting tree pass is intuitively interpretable: bottom-up accumulation of subtree messages \(S_u\) and top-down redistribution to compute each \(H'_v\).  
   - Appendix A.6 convincingly argues that many nonlinear aggregators (attention GNNs, RNNs, SSMs) can also fit the framework with minor modifications. This indicates potential for broader adoption beyond the specific linear version.

3. **Theoretical link between homophily estimation and tree quality**  
   - Theorem 2 (Sec. 4.6) formalizes how the expected edge-homophily ratio \(R_{\hat G}(\Delta)\) under the tree distribution \(P_{\hat G}^{(p,q)}(T)\) is monotonically increasing in the score ratio \(\Delta=p/q\) beyond some \(\Delta_0\), and converges to a structural upper bound \(1 - \frac{\text{NHCC}(\hat G)-1}{n-1}\).  
   - The proof in Appendix B.2 is nontrivial but readable: rewriting expectations in terms of the counts \(n^+(T)\), collecting terms by homophilous-edge counts, and relating the maximal \(n^+_{\max}\) to NHCC(\(\hat G\)). This provides a solid theoretical justification that progressively better edge-homophily scores indeed bias sampling toward more homophilous trees, up to a principled limit.

4. **Strong and broad empirical evaluation**  
   - Table 1 provides accuracy comparisons across 9 datasets (Cora, Citeseer, Pubmed, Actor, Cornell, Texas, Wisconsin, Arxiv, Flickr) against 26 baselines spanning classic MLP, shallow GNNs, deep GNNs, many graph transformers, and GraphMamba. The proposed method ranks first on all datasets and by far the best average rank (1.22).  
   - Gains on heterophilous graphs are especially impressive (e.g., Texas: 91.89 vs 77.84 for MLP and 69.19 for GCNII; Wisconsin: 86.27 vs 70.31 for GCNII and 80.00 for SGFormer), which strongly supports the paper’s claim of improved long-range modeling under label scarcity and heterophily.  
   - Table 3’s ablation shows meaningful contributions from each component: removing global or local submodules hurts performance, homophily-guided sampling outperforms uniform sampling, and multiple trees outperform a single homophilous tree.  
   - Table 4’s homophily estimator comparison (NAAM vs attention vs two-stage vs various FGL variants) empirically supports the theoretical story of Theorem 2: better estimators lead to better final performance.

5. **Runtime advantages are convincingly demonstrated**  
   - Table 2 compares per-epoch runtimes on Cora, Citeseer, Pubmed, Flickr, Arxiv. The proposed method is consistently among the fastest, and often faster than both deep GNNs (GCNII, DropEdge) and efficient transformers (DIFFformer, NodeFormer, SGFormer).  
   - Importantly, methods like GOAT and ANS-GT are orders of magnitude slower on large graphs (tens of seconds per epoch) while FGL remains under 0.25 s. This empirically validates the claimed linear complexity and supports the practical value of the paradigm.

6. **Figures effectively support key claims**  
   - **Figure 2** is a clear high-level overview of the pipeline (pre-processing → tree sampling → tree aggregation → tree fusion), which greatly aids understanding the interplay between components.  
   - **Figure 3** visualizes the local structure behind Theorem 1: how cutting and flipping a single edge between neighboring nodes changes global message flows. This makes the Combine/Disentangle intuition much more concrete and helps demystify Eq. (5)-(6).  
   - **Figure 4** and **Figure 11** (hyperparameter study) show the effect of number of trees \(N_T\): accuracy improves up to around 6–10 trees then plateaus or drops, supporting the claim that only a small forest suffices for good global coverage.  
   - **Figure 6 / 15** shows that trees from the homophily-guided sampler have consistently higher global homophily ratio than random spanning trees, giving a direct metric-level explanation for the observed performance gain.  

7. **Extensive experimental diagnostics and robustness analysis**  
   - Many additional experiments in the appendix (noisy features, dense-graph variant of Cora, large-scale AMiner-CS, graph classification on ENZYMES, comparisons with path-based GNNs and heterophily-specialized models) show the framework remains competitive or superior in a variety of settings.  
   - Robustness to feature noise (Fig. 17–18) and to hidden dimension size (Fig. 14) are particularly valuable from a practical standpoint.

8. **Thoughtful theoretical and conceptual discussions**  
   - The appendices offer interesting analyses on over-smoothing, propagation bottlenecks, and equivalence to path decompositions (Sec. A.2–A.4, B.4).  
   - In particular, Theorem 3 compares mixing time bounds on graphs vs spanning trees, arguing that trees can alleviate over-smoothing under fixed-distance aggregation; the derivation using spectral gap bounds (Eq. (16)-(17)) is a nice theoretical contribution even if high-level.

## Weaknesses

1. **Complexity and practicality of the full pipeline**  
   - The overall framework is quite involved: pretraining a non-attention auxiliary model for pseudo-labels, top‑k graph augmentation (Sec. 4.1), training a local attention homophily estimator (Eq. (3)), defining edge scores and a nontrivial tree distribution (Eq. (2)), running Wilson’s random-walk tree sampler (Algorithm 2), optionally using block acceleration (Algorithm 3), computing two recursions per tree (Eq. (7)-(8)), and finally local+global fusion (Eq. (9)-(11)).  
   - While each component is understandable in isolation, the paper does not fully quantify *end-to-end* complexity and overhead, especially including the two auxiliary pretraining stages. Sec. 4.5 focuses on student training cost but largely ignores estimator training and augmentation. For real-world deployment, these pretraining stages may dominate runtime or memory, especially on large graphs.  
   - It would help to see, e.g., wall-clock breakdowns: % time spent on (a) pretraining auxiliary models, (b) graph augmentation, (c) tree sampling, (d) main model training. Currently Table 2 reports only per-epoch training times of the final model.

2. **Some mathematical assumptions and generality claims are under-specified**  
   - The Combine / Disentangle conditions defining admissible \(f_{\text{Agg}}\) (Eq. (29)) are central, but the paper only provides high-level arguments that “many” aggregators satisfy them. For nonlinear message passing, the discussion in Appendix A.6 introduces pre-activation storage and invertibility tricks, but the construction becomes quite contrived.  
   - More concretely: for a typical GAT-style attention where \(f_{\text{Agg}}\) includes nonlinear activations and normalization (softmax over neighbors), it is not obvious that there exist \(\mathcal{M}^\pm\) satisfying Property (I) and (II) without recomputing attention for merged sets. The paper should explicitly work through at least one such popular nonlinear aggregator (with indices and normalization) and provide formulas for \(\mathcal{M}^\pm\).  
   - Similarly, Theorem 1 is formulated in quite general terms, but the actual derivation in Appendix B.1 assumes sets \(A,B\) are disjoint and that \(\mathcal{M}^-\big(\mathcal{M}^+(p,q),q\big)=p\). For attention with shared denominators, this identity may fail unless one stores additional normalization constants. The current presentation risks overstating generality.

3. **Tree homophily theory vs. practical estimator and weighting mismatch**  
   - Theorem 2 assumes **binary** edge scores: \(s(e_{ij}) = p\) if edge is homophilous, \(q\) otherwise, and uses the exact ground-truth labels. In practice, the paper uses *real-valued attention scores* \(\alpha_{i\to j}\) and defines \(s(e) = (\alpha_{i\to j} + \alpha_{j\to i})/2\) (Sec. 4.2). There is no explicit mapping from attention to the \((p,q)\) model, nor any guarantee that \(s(e)\) clusters cleanly into two sharp modes.  
   - Moreover, in the actual tree aggregator (Eq. (7)-(8)), the same \(\alpha_{i\to j}\) are re-used as propagation weights. It is then unclear whether the homophily theory, which pertains only to the sampling distribution, still meaningfully describes the *effective* path weights when aggregation also depends on these scores.  
   - The empirical study in Fig. 5 / Fig. 8 (“effect of homophily estimator accuracy”) appears to use artificially controlled estimator accuracies (including a perfect estimator that yields 100% classification), which is informative but somewhat unrealistic. The paper should be clearer about how those controlled estimators are constructed, and whether a similar trend holds for estimators trained under typical label sparsity without oracle access.

4. **Pre-processing augmentation can heavily distort the graph, but its impact is not fully scrutinized**  
   - Sec. 4.1 uses pseudo-labels to perform top‑k similarity-based augmentation for each node to ensure connectivity and raise homophily. This can add 5–15 edges per node (Appendix L), a large change on sparse citation graphs (e.g., Cora’s original average degree is 4.0, Table 7).  
   - While Fig. 9 shows that NHCC improves with more augmentation edges, there is little comparison to strong baselines that *also* benefit from this augmented graph. All baselines in Table 1 appear to be run on the original graphs. This raises the question: how much of FGL’s performance gain is due to nontrivial modeling vs just having access to a more homophilous, denser graph?  
   - A more rigorous comparison would: (i) re-run a subset of strong baselines (e.g., GCNII, SGFormer, DIFFformer) on the **same augmented graphs**; (ii) report whether FGL still holds a similar margin. Without this, the causal attribution of gains to the forest paradigm is somewhat weakened.

5. **Some experimental choices and fairness concerns remain ambiguous**  
   - Hyperparameter search for FGL is thorough (200 random trials per dataset, Tab. 9), but it is not clear whether comparable tuning effort was applied to all baselines, especially the many graph transformers. Some of those methods are known to be quite sensitive to depth, hidden size, and positional encodings.  
   - In Table 1, some transformer baselines run into OOM on larger graphs (GT, SAN, Graphormer), which is expected, but then their average rank is computed ignoring those datasets. This partially penalizes them for not scaling, which is fair, but reinforces the need to ensure FGL’s memory usage is reported explicitly.  
   - The paper mentions that for Flickr and Arxiv they use 20 random splits and share them across baselines (Sec. K.2), which is good, but the main table shows only mean values without reporting significance testing or confidence intervals. Given many margins are a few points, some indication of statistical significance would be valuable.

6. **Some clarity gaps and notation issues in core equations**  
   - Eq. (7)-(8) reuse the attention coefficients \(\alpha_{i\to j}\) from Eq. (3), but the training regime of these \(\alpha\) in the student phase is not fully specified. Are they frozen after homophily-estimator training, or updated jointly with tree-aggregator parameters \(W_A, W_B\)? This matters both conceptually (are we still sampling from the same estimated distribution?) and practically (stability).  
   - In Eq. (7), the scalar \(\alpha_{v\to u}\) is multiplied by \(W_A\) then by \(S_v\). The parentheses are ambiguous: is it \((\alpha_{v\to u} W_A) S_v\) or \(\alpha_{v\to u} (W_A S_v)\)? The shapes work in both orders, but computational cost differs. Similarly, Eq. (8) includes nested products \(\alpha_{\text{Fa}(v)\to v} W_A (H'_{\text{Fa}(v)} - \alpha_{v\to\text{Fa}(v)} W_A S_v)\); clarifying the exact order would reduce ambiguity for implementers.  
   - The notation \(\widehat{A}_{\widehat G}\) in Eq. (9) is not clearly defined in the main text (presumably the normalized adjacency of the augmented graph), and the symbol \(\alpha\) is overloaded to mean both attention matrices and (in Eq. (9)) a matrix blended with adjacency and identity. This causes some confusion.

7. **Positioning relative to closely-related tree and random-walk methods is thin**  
   - The related work section mostly discusses generic deep GNNs and graph transformers, with relatively little on methods that also exploit trees or random walks for efficient propagation. The later appendix (J.10) compares against GERN-GCN, but this is not adequately integrated into the main narrative.  
   - There is limited conceptual comparison to methods like random-walk aggregation GNNs or tree-decomposition-based GNNs, which arguably share similar motivations around long-range propagation and sparsification. This makes it harder to precisely gauge what is fundamentally new vs. a different instantiation of tree-based sparsification.

8. **Some theoretical results are insightful but only loosely connected to the implemented model**  
   - The over-smoothing analysis (Theorem 3 and surrounding discussion in Sec. A.3) compares mixing times on the original graph vs any spanning tree, arguing that trees require more steps to converge to stationarity and thus mitigate over-smoothing. However, the actual model does *not* perform pure random walks; message passing in Eq. (7)-(8) is deterministic linear propagation with attention weights and residual fusion with local layers.  
   - While the theory is conceptually suggestive, it is not demonstrated experimentally (e.g., by measuring embedding variance vs depth or distance in FGL vs deep GNNs) and relies on somewhat loose analogies (random walk vs linear aggregation). The paper could benefit from either tightening the link or tempering the claims.

9. **Minor but noticeable issues**  
   - There are a number of minor typos and formatting glitches (e.g., “FOL” instead of “FGL” in Table 4 row labels; repeated or garbled citations like “SuperGAT\(_\text{SD}\)” and several reference duplications in the bibliography).  
   - Some figure captions are truncated or duplicated (e.g., Fig. 4’s caption overlays with Fig. 5 text in the PDF extract), and a few references in the text point to sections “Sec. L” or “Sec. 1.6” that appear only in the appendix, which slightly impairs readability.

## Potentially Missing Related Work

1. **Wang & Derr, “Tree Decomposed Graph Neural Network” (2021)**  
   - This work also uses tree-based decompositions to facilitate graph neural network computation, which is conceptually close to the idea of leveraging trees as core propagation structures.  
   - It should be discussed in **Section 2 (Related Literature)** and compared against FGL in terms of: how trees are obtained (decomposition vs random sampling), what is propagated (local vs global messages), and complexity vs coverage trade-offs. A small-scale empirical comparison on at least one benchmark (e.g., Cora / Pubmed) would be appropriate if implementations are available.

2. **Jin et al., “RAW-GNN: RAndom Walk Aggregation based Graph Neural Network” (2022)**  
   - RAW-GNN integrates random walk strategies into the aggregation mechanism to enable long-range information propagation efficiently. This is directly related to the motivation in Sec. 1 concerning long-distance interactions and the limitations of stacking local layers.  
   - It should be cited in **Sec. 2 (Deep Local Models / Tradeoff Between Local and Global Models)** and contrasted with FGL: random walks vs spanning trees as sparse global structures, coverage, and complexity. Given both aim for efficient long-range aggregation, a quantitative comparison in Table 1 or the appendix would be informative.

3. **Peng et al., “Label-guided Graph Contrastive Learning for Semi-Supervised Node Classification” (2024)**  
   - This method leverages semantic-level label information to guide representation learning under semi-supervised settings. Since FGL also uses pseudo-labels and label-guided homophily estimation (Sec. 4.1–4.2), it is a relevant baseline and conceptual comparator.  
   - It should be referenced in **Sec. 2** and **Sec. 4.1**, highlighting differences between contrastive learning and forest-based propagation, and possibly added as a baseline for homophilous graphs if feasible.

4. **Guo et al., “ES-GNN: Generalizing Graph Neural Networks Beyond Homophily With Edge Splitting” (2024)**  
   - ES-GNN proposes edge splitting to adaptively handle homophilous and heterophilous edges. FGL also explicitly models edge homophily via an estimator and uses it to bias sampling.  
   - This work should be discussed in **Sec. 2** and **Sec. N.1 (Heterophily GNNs)** as another approach to edge-level modeling under varying homophily, and would be a strong baseline on heterophilous datasets like Texas / Wisconsin / Actor.

5. **Ma et al., “SeBioGraph: Semi-supervised Deep Learning for the Graph via Sustainable Knowledge Transfer” (2021)**  
   - SeBioGraph focuses on semi-supervised learning on graphs via knowledge transfer, using unlabeled nodes to improve labeled performance. Since FGL also heavily uses unlabeled data via pseudo-labels and structural augmentation, it is a relevant comparator.  
   - It should be cited in the **problem formulation / related work** parts discussing semi-supervised node classification and label-scarcity, and ideally considered in the experimental comparisons on homophilous datasets.

6. **Li & Wang, “Graph-based Semi-supervised Learning” (2011)**  
   - This is a classic work on graph-based semi-supervised learning, framing label propagation and Laplacian regularization foundations. Given FGL’s reliance on spanning trees and homophily, it would be appropriate background.  
   - It could be briefly mentioned in **Sec. 3 (Preliminary)** when discussing semi-supervised node classification and graph homophily, as foundational context.

7. **Gao et al., “A Path-Aware Graph Neural Network for Heterophily Graph Learning” (2026)**  
   - This work explores path-aware mechanisms for heterophilous graphs, which is closely related to FGL’s emphasis on long-range, path-based information aggregation.  
   - It should be included in **Related Work (heterophily / long-range modeling)** and compared conceptually (explicit path selection vs sampled tree paths) and empirically on heterophilous benchmarks.

If some of these methods are difficult to reproduce at scale, at minimum they should be carefully discussed to better position FGL in the landscape.

## Questions

1. **Effect of graph augmentation on baselines**  
   - Can you report performance of at least a few strong baselines (e.g., GCNII, SGFormer, DIFFformer) on the *same augmented graphs* \(\hat G\) used by FGL (i.e., after pseudo-label top‑k edge addition)? This would isolate the effect of the forest paradigm from the benefits of augmentation itself.

2. **Training regime of attention coefficients \(\alpha\)**  
   - After the homophily estimator is trained in Sec. 4.2, are the attention matrices \(Q,K,V\) and resulting \(\alpha_{i\to j}\) **frozen** during tree aggregation training, or are they fine-tuned jointly with \(W_A, W_B, W_H\)? If they are updated, how do you maintain consistency between the edge scores used for tree sampling and the propagation weights?

3. **Generality of the tree aggregator for nonlinear attention**  
   - Could you provide a concrete worked example of how a standard GAT-style aggregation (with softmax normalization over neighbors and nonlinear activation) can be instantiated within Theorem 1’s framework? Explicit formulas for \(\mathcal{M}^+\) and \(\mathcal{M}^-\) for this case would strengthen the claim of generality.

4. **Construction of “perfect” and controlled homophily estimators in Fig. 5 / Fig. 8**  
   - How exactly are different homophily estimator accuracies realized in these figures? Are you synthetically corrupting an oracle estimator with ground-truth labels, or training under varying label budgets? Clarifying this would help interpret the strong monotonic trends.

5. **Memory usage and scalability of tree sampling**  
   - For large graphs like Arxiv / AMiner-CS, what is the memory usage and runtime of the tree sampling stage (including block acceleration in Algorithm 3)? Does the block trick materially approximate the target distribution, and have you measured any performance difference between exact and block-accelerated sampling on small graphs?

6. **Impact of number of trees on over-smoothing and redundancy**  
   - Fig. 4 and Fig. 20 show that too many trees can slightly hurt performance. Do you observe any direct signs of over-smoothing (e.g., decreasing embedding variance across nodes) as \(N_T\) increases, or is the degradation mostly due to overfitting? Any diagnostic plots or metrics would be helpful.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core algorithms and proofs (Theorem 1, Theorem 2) appear mathematically consistent, and experiments are extensive and carefully designed. Some generality claims for the aggregator are slightly overstated and the link between theory and practical estimator/training could be sharper, but overall the methodology is solid and empirically validated.

## Presentation Rating

3: good.  
The paper is dense but mostly clear, with helpful figures (especially Figs. 2 and 3) and thorough appendices. Some notation overload, minor typos, and lack of explicit implementation details for key equations (Eq. (7)-(9)) slightly hurt readability, but not to a critical degree.

## Contribution Rating

3: good.  
The paradigm of spanning-tree forests for efficient global coverage, together with the specific homophily-guided tree sampling and linear tree aggregator, constitutes a meaningful contribution beyond existing deep local / global transformer models. Theoretical results and strong empirical evidence further increase the contribution, though the heavy pre-processing pipeline and partial overlap with tree/random-walk ideas somewhat temper the perceived originality.

## Overall Rating

8: Accept, good paper (poster).  
The paper delivers a well-argued and technically substantial forest-based graph learning paradigm with both theoretical backing and convincing empirical validation, including strong results on challenging heterophilous datasets and favorable efficiency. While the pipeline is complex and some theoretical claims could be better aligned with practice, the strengths clearly outweigh the weaknesses, and the work is likely to be of broad interest to the graph learning community.

## Reviewer Confidence

4: confident.  
I am familiar with graph neural networks, graph transformers, and random-walk-based methods, and I carefully examined the main equations, theorems, and experimental tables/figures. Some implementation details and potential baselines remain uncertain, but they do not undermine the main assessment.