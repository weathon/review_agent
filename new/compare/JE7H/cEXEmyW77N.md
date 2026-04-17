---
job_id: 0c4fdcc0-26dc-438d-80ee-1910ae5dca05
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: cEXEmyW77N.pdf
paper: Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies representation and detection of LLM-generated citation graphs using embeddings and GNNs, which lies squarely within “learning on graphs,” representation learning, and LLM analysis, all core ICLR topics.

## Minimum Quality
Pass ✅.  
All core sections are present (Abstract, Introduction, Related Work, Methodology/Approach, Experiments, Results, Discussion, Conclusion). The work is technically sound overall, includes substantial experiments, and is written in clear English.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden instructions or attempts to manipulate automated reviewing systems in the paper content.

---

# Expected Review Outcome:

## Summary

The paper compares citation networks induced by human reference lists with those induced by LLM-suggested references, using 10k focal papers from SciSciNet and paired GPT‑4o- and Claude‑Sonnet‑generated bibliographies, plus several field-matched random baselines. Graph-level structural descriptors (centralities, clustering, edge counts) plus Random Forests and GNNs show that LLM-generated bibliographies are nearly indistinguishable from ground truth in topology but diverge semantically when node text embeddings are used, enabling ≈93% test accuracy in distinguishing GPT graphs from human ones. The paper concludes that detection and debiasing of LLM bibliographies should rely on semantic/content signals rather than coarse graph structure.

## Strengths

1. **Clear high-level contribution and message.**  
   The paper delivers a crisp empirical takeaway: LLM-generated bibliographies, when created purely from parametric knowledge, reproduce realistic citation topology but exhibit detectable semantic fingerprints. This is articulated consistently from the abstract through the discussion (Pages 1, 7–10).

2. **Strong dataset and experimental design.**  
   - The use of 10,000 focal papers and ≈275k references from SciSciNet (Section 3, Page 3) is sizable for this kind of scientometric analysis and substantially larger/more systematic than most anecdotal LLM‑bibliography papers.  
   - The paired design (each focal paper has a human graph, an LLM graph, and a randomized baseline graph) is methodologically sound and allows clean within-paper comparisons. Figure 1 and Figure 2’s leftmost schematic (Page 2 and Page 5) make this pipeline easy to grasp.

3. **Carefully crafted random baselines.**  
   The field-level, subfield-level, and temporally constrained random baselines (Section 3, Pages 3–4; Appendix Figures 12–14 and Table 11) are thoughtfully constructed. They preserve out-degree and field/year distributions yet break latent structure, which is precisely what is needed to say something nontrivial about “structural realism” versus random attachment.

4. **Systematic structural analysis with clear visual support.**  
   - Figure 2 (Page 5) provides a comprehensive set of structural diagnostics: distributions of node-level metrics, joint plots of degree vs. clustering, hub dominance (mean degree vs. max/mean ratio), and edges vs. nodes/degree, all showing near-complete overlap between ground truth and GPT graphs and sharp separation from random graphs.  
   - This directly supports the claim that topology-only signals cannot reliably distinguish human vs. LLM graphs but can easily reject the random baselines.

5. **Thorough semantic analysis using embeddings.**  
   - The move from structural descriptors to 3072‑D title/abstract embeddings (Section 5, Pages 6–7) is natural and well motivated by the structural near-indistinguishability.  
   - Figure 3 (Page 7) and its PCA plots plus cosine / Euclidean diagnostics make it clear that LLM and human graphs are semantically aligned to each other but far from random graphs, yet RF classifiers in the full embedding space can still separate LLM vs. ground truth (Table 2).

6. **GNN experiments are reasonably thorough and internally consistent.**  
   - The paper evaluates four canonical GNN architectures (GCN, GAT, GraphSAGE, GIN) with a transparent hyperparameter random search (Table 12–14) and stratified splits (Section 6, Page 8).  
   - Figure 4 (Page 9) shows full validation-accuracy distributions over hyperparameters and seeds, not just best numbers, which is commendably transparent.  
   - Table 3 (Page 10) summarizes test-set performance, showing ~93% accuracy for GPT vs ground truth when using embeddings, consistent across architectures, while structure-only GNNs stay near chance, matching the RF story.

7. **Robustness checks and ablations.**  
   The authors go well beyond minimal experimentation:
   - Cross-LLM robustness (Claude vs ground truth, Sections in Appendix, Figures 5–9, Tables 4–7) shows the semantic-fingerprint pattern persists across generators.  
   - Cross-encoder robustness (OpenAI vs SPECTER2 embeddings) in Appendix Figures 8, 10–11 and Tables 5, 7, 8 suggests the effect is not tied to one embedding model.  
   - Random-vector control and PCA‑k ablation (Appendix Figures 15–17 and Page 27) convincingly show that the gains are due to semantic content, not high dimensionality alone.  
   - Cross‑model generalization (Table 9–10, Page 24) where models trained on GPT‑4o graphs test on Claude graphs give 0.68–0.80 accuracy, reinforcing the idea of a generator‑family semantic signature.

8. **Responsible discussion of implications and limitations.**  
   Section 7–8 (Pages 10–10) explicitly state that debiasing/detection should prioritize semantic signals, and acknowledge important limitations such as parametric-only references and focus on title/abstract text, which avoids overclaiming.

9. **Clarity and organization.**  
   The exposition is generally clear and well structured. The definitions of graph-theoretic quantities in the appendix (Page 17) are standard and correct, and help make the structural feature choices precise. Figure 20 (Appendix) usefully visualizes the RF tree depth and Gini reduction, giving some interpretability on how embeddings are used.

## Weaknesses

1. **Limited conceptual novelty beyond prior work by the same group.**  
   - The paper heavily builds on Algaba et al. (2024, 2025) and Mobini et al. (2025) (cited) that already examine how LLM bibliographies resemble human ones in various structural and bibliometric statistics.  
   - This paper’s distinct contributions are (i) using classification tasks to quantify separability of LLM vs human graphs, (ii) bringing in node embeddings and GNNs, and (iii) showing that semantics carry the main detection signal.  
   - While useful and more systematic, this feels like a natural extension rather than a major conceptual step. The paper would be stronger if it pushed further into *what* specific semantic dimensions drive the fingerprint (prestige, recency, topical shifts, over-use of canonical papers, etc.), not just that embeddings make them separable.

2. **Shallow analysis of *why* embeddings separate LLM vs human graphs.**  
   - The main empirical finding is that RF and GNNs on embeddings achieve high GPT vs ground-truth accuracy (Table 2, Table 3), but the paper largely stops at the existence of separability, with only coarse cosine-distance diagnostics in Figure 3(b,c).  
   - There is no attempt to probe which embedding directions or interpretable bibliometric covariates (e.g., publication year, venue prestige, citation counts) dominate the decision boundaries.  
   - For instance, an analysis of feature importance in RF (beyond the tree-depth statistics of Figure 20), or post-hoc clustering of misclassified graphs, would clarify whether the models are mostly learning recency bias, over-citation of high-impact venues, or some more subtle semantic pattern. This limits scientific insight: we know separation is possible, but not much about its qualitative nature.

3. **Potential circularity / coupling between generator and embedding backbone.**  
   - For the main GPT‑4o experiments, both the generator and embedding model are from the same provider (OpenAI), and likely share training data and tokenization biases. This raises the possibility that embeddings are particularly well attuned to GPT‑style outputs.  
   - The authors partially address this by including SPECTER2 experiments and Claude graphs (Appendix Sections), which is good, but the analysis of any remaining correlations is superficial. For example, in Table 2 vs Table 8, GPT vs ground truth separability drops from 0.83 to 0.75 when switching from OpenAI to SPECTER embeddings; this suggests some coupling that is not discussed.  
   - A more explicit comparison and discussion of the gap between same-vendor and cross-vendor configurations, and what it implies for robustness of detection pipelines in the wild, would be valuable.

4. **Topological treatment loses directionality and finer-grained structure.**  
   - The graphs are converted to simple undirected graphs (Page 4) to “reflect the topological organization of the network rather than directionality artifacts,” but for citation networks directionality is not an artifact; it encodes temporal and influence flows.  
   - By symmetrizing edges, closeness, eigenvector centrality, and clustering (Equations on Page 17) may lose informative asymmetries such as the focal node’s in- vs. out-role, or cascades of citations through time. This may partially explain why structure-only features underperform.  
   - At minimum, the paper should justify more concretely why directionality is discarded, and ideally run a small ablation with directed descriptors (e.g., in-, out-degree distributions, PageRank) to check whether topology remains non-informative under a richer structural feature set.

5. **Graph features in GNNs are somewhat ad hoc and not well justified.**  
   - For structure-only GNN experiments, each node receives a 5‑D feature vector with degree centrality, closeness, eigenvector centrality, clustering coefficient, *and the total number of edges of the graph copied to all nodes* (Section 6, Page 8).  
   - Injecting a global graph-level scalar as an identical node feature is a crude way to expose global scale; it is not clear how this interacts with message passing, nor why this specific choice is appropriate. A more principled design would use global pooling or virtual nodes, or at least a short ablation on including vs removing the “edge-count” feature.  
   - For embedding-based GNNs, the use of raw 3072‑D embeddings with no normalization or learned preprocessing might be suboptimal; some comment on whether simple linear projections or MLPs on node features were tried (beyond the PCA‑k ablation) would improve the methodological clarity.

6. **Interpretation of RF performance as “near chance” is somewhat loose.**  
   - In Table 1, “Ground truth vs GPT” classification using structural graph properties reaches ≈0.61 accuracy / F1. The text (Page 6) describes this as “near-chance” and infers that structural properties “do not reliably differentiate” LLM from human lists.  
   - On a balanced dataset of ~9k graphs per class, 0.61 is weak but not indistinguishable from chance; some statistical testing or confidence-interval analysis would help justify the “non-significant” claim. Alternatively, the paper should acknowledge there is a modest but nontrivial structural signal that RF can exploit.  
   - This matters because the central narrative is that structure alone is nearly useless, yet the reported numbers suggest weak but measurable signal.

7. **Limited connection to broader detection literature and LLM‑graph methods.**  
   - The Related Work section (Section 2, Pages 2–3) focuses on LLM bibliographies and scientometrics, but largely omits recent work on detecting LLM‑generated content and on LLM+GNN systems for scholarly graphs.  
   - This weakens the positioning of the detection aspect of the paper: the contribution is presented mostly in isolation from rich lines on LLM text detection and LLM‑augmented graph learning.

8. **Aggregation choice for graph-level embeddings is very simplistic.**  
   - For RF experiments, graph-level embeddings are obtained by summing node embeddings (Section 5, Page 6), which implicitly weights longer bibliographies more heavily and conflates count and content. There is no exploration of alternatives (mean pooling, attention, or more advanced graph pooling).  
   - While the main point is that even such a crude aggregation suffices, a short robustness check (e.g., mean vs sum pooling) would test whether separability is actually driven by representation geometry or trivial scale differences like average embedding norm.

9. **No statistical uncertainty on GNN test metrics and somewhat unusual reporting.**  
   - Table 3 reports mean ± std for accuracy and F1, but it is unclear whether those are computed over seeds, hyperparameter configurations, or both. The text says the test uses “best hyperparameter setup” (Page 8), suggesting only seeds vary, but this is not explicit.  
   - Moreover, Figure 4 emphasizes validation distributions, yet test results appear only as aggregated statistics. Given the centrality of the 93% number for the main claim, a more detailed description of how many runs contribute to this estimate, and perhaps confidence intervals, would strengthen the methodological transparency.

10. **Mathematical formalization is minimal and misses some key choices.**  
   - While the definitions of degree, closeness, eigenvector centralities and clustering (Page 17) are standard, there is no precise specification of how disconnected components are handled for closeness (e.g., are unreachable nodes ignored or given infinite distance?).  
   - Similarly, the random-graph reshuffling is described informally as a “without-replacement shuffle at the field level” (Page 4) but not in explicit notation (e.g., \( \pi_f \) permutations per field and assignment rule), which would help with reproducibility.  
   - The loss/objective for GNN training is standard cross-entropy but not explicitly written; including the formal graph classification objective, especially to clarify that labels are graph-level rather than node-level, would make the theoretical framing more complete.

Overall, these are not fatal flaws, but they keep the paper in the realm of a strong empirical study rather than a fully polished methodological contribution.

## Potentially Missing Related Work

1. **Wu, H., Xiang, H., Gao, J. (2026): “Detecting Miscitation on the Scholarly Web through LLM-Augmented Text-Rich Graph Learning.”**  
   - Relevance: Uses LLM-enhanced models with GNNs to detect miscitations in scholarly networks; very close in spirit to this paper’s use of embeddings and GNNs for detecting problematic or non-human-like citation patterns.  
   - Suggested placement: Section 2 (Related Work) after the discussion of citation networks and LLMs for scholarly tasks, plus a brief comparison in the discussion about how their system targets miscitations at the edge-level whereas this paper targets graph-level LLM-vs-human detection.

2. **Geng, M., Poibeau, T. (2025): “On the Detectability of LLM-Generated Text: What Exactly Is LLM-Generated Text?”**  
   - Relevance: Surveys and analyzes challenges in detecting LLM-generated text, conceptually close to this work’s question of detectability of LLM-generated bibliographies.  
   - Suggested placement: Section 2, in a paragraph on LLM-output detection, noting that this paper tackles a specialized structured artifact (citation graphs) rather than free text.

3. **Geng, M., Poibeau, T. (2025): “Zero-shot detection of LLM-generated text via text reorder.”**  
   - Relevance: Proposes a detection method based on logical ordering rather than supervised classifiers on embeddings. Offers complementary strategies and highlights the general landscape of LLM-output detection techniques.  
   - Suggested placement: Section 2, again in the detection context, to contrast supervised embedding/GNN approaches with zero-shot logic-based approaches.

4. **Zhao, X., Wu, D., Li, J. (2025): “Uncovering novel scientific insights with a synergistic GNN-LLM framework.”**  
   - Relevance: Integrates LLMs and GNNs for scientific knowledge discovery, illustrating another direction where GNNs and LLMs are combined on scientific graphs.  
   - Suggested placement: At the end of Section 2, when discussing scientometrics and LLM-based scholarly tools, to position this paper as complementary: their system uses LLM+GNNs constructively for discovery, while this paper uses similar ingredients diagnostically for detection.

## Questions

1. **On directed vs undirected graphs.**  
   - Could you clarify why you chose to symmetrize citation edges (Page 4) instead of working with directed graphs?  
   - Have you tried including directional features (in-/out-degree, PageRank, or temporal edge orientation) to see whether topology becomes more informative? Some small ablation or descriptive comparison here could either support or challenge your conclusion that structure alone is weak.

2. **On what semantic dimensions drive separability.**  
   - Beyond cosine and Euclidean distances in Figure 3, can you provide any analysis of which bibliometric or textual factors (e.g., publication year distribution, venue distribution, citation counts, keyword frequency) correlate most with the classifier’s decisions?  
   - For example, what happens if you regress the RF logit on mean publication year or mean cited‑by counts of the references? This would help interpret the “semantic fingerprint.”

3. **On vendor coupling of generator and encoder.**  
   - The main GPT‑4o experiments use OpenAI embeddings; when you switch to SPECTER2 (Table 8), separability for Ground truth vs GPT decreases from ≈0.83 to ≈0.75. Could you discuss this gap more explicitly?  
   - To what extent do you think same‑vendor alignment (GPT‑4o + OpenAI embeddings) is boosting performance, and how might this affect the practical reliability of detection systems built as in Section 5?

4. **On RF and GNN configuration details.**  
   - For RF: did you try varying the number of estimators, tree depth limits, or class weights? Are the results in Tables 1–2 robust to these hyperparameters?  
   - For GNNs: could you clarify how many independent runs per architecture you used when reporting the test numbers in Table 3, and whether the reported ± values are across seeds only?

5. **On aggregation choices for graph-level embeddings.**  
   - You use a simple sum over node embeddings to obtain graph-level vectors for RF (Section 5). Did you check whether mean pooling, max pooling, or concatenation of focal-paper embedding with pooled-reference embedding materially changes the separability results?  
   - If not, can you provide some intuition as to why sum pooling is appropriate, given that you also match the number of references across graph types?

Author responses addressing these questions and possibly adding small additional analyses could raise my confidence and potentially my assessment of the contribution.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The empirical methodology is generally solid, with strong baselines, multiple controls, and consistent findings. Some design choices (undirected graphs, limited probing of what semantics matter) and missing formalization details prevent a higher rating.

## Presentation Rating

3: good.  
The paper is clearly written, figures and tables are informative (notably Figures 1–4 and Tables 1–3), and the narrative is easy to follow. Some methodological explanations (e.g., random reshuffling details, handling of directionality) could be more precise.

## Contribution Rating

2: fair.  
The work offers a useful and carefully executed empirical study with a clear scientific message, but the novelty over prior work by the same group and the depth of semantic analysis are somewhat limited. The contribution is more diagnostic/empirical than methodologically innovative.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper is a well-executed, large-scale empirical study with a clear and practically relevant message and strong experimental robustness. Weaknesses lie in somewhat limited conceptual novelty and shallow analysis of the semantic fingerprint itself, but the work still provides meaningful value to the community, especially for those interested in LLM‑assisted scientific workflows and detection of LLM-generated scholarly content.

## Reviewer Confidence

4: confident.  
I am familiar with graph learning, representation learning, and the emerging literature on LLM‑generated content and scientometrics. I have checked the main equations and experimental protocols carefully, though I did not reimplement the models.