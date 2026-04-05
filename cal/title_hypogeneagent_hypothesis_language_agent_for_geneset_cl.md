=== CALIBRATION EXAMPLE 12 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate in that the paper introduces an LLM-based agent for gene-set annotation and cluster-resolution selection, but it overstates novelty by presenting the method as a general “hypothesis language agent” rather than a relatively specific scoring framework built on LLM-generated GO hypotheses.
- The abstract does state the problem, method, and intended result, but it is hard to tell exactly what is new versus a combination of known pieces: LLM-based gene-set interpretation, embedding-based similarity, and a scalar score over clustering resolutions.
- Several claims are stronger than what the paper supports. In particular, “calibrated confidence scores,” “objective adjudicators,” and “paving the way for fully automated, context-aware interpretation pipelines” are not convincingly established by the evidence presented. The abstract also implies broad validation, but the experiments appear to be on one GOBP benchmark and one Perturb-seq dataset.

### Introduction & Motivation
- The motivation is reasonable and ICLR-relevant: the paper tackles a real workflow bottleneck in single-cell analysis, namely resolution selection and functional annotation. The gap is also clear: existing clustering metrics do not directly incorporate biological interpretability.
- The contribution list is mostly understandable, but it is somewhat imprecise. The paper claims to “transform cluster annotation into a quantitatively optimizable task,” yet the actual optimization target is a composite semantic score derived from the model’s own generated hypotheses, which is not the same as an external biological objective.
- The introduction over-claims in several places. It suggests the framework “closes the gap” between clustering and automated annotation, but the paper only demonstrates that the score picks a resolution matching a chosen Perturb-seq benchmark better than some baselines. That is a narrower claim than the text implies.
- The related-work positioning is helpful, especially in situating LLM-based gene-set analysis, but the novelty boundary is not sharply drawn. It remains unclear how much of the contribution is algorithmic versus a task-specific repurposing of LLM annotations for hyperparameter selection.

### Method / Approach
- The overall pipeline is understandable at a high level, but the method is not described with enough precision to be fully reproducible or to support the main claims.
- The central issue is that the resolution score is based on embeddings of the model’s own textual outputs. This creates a circularity risk: the score measures internal consistency and self-dissimilarity of the agent’s hypotheses, not directly the biological quality of the clusters. The paper does not sufficiently justify why this proxy should correlate with true cluster correctness.
- The definitions in Table 1 and Section 3.4 need stronger formal clarity. For example:
  - The paper uses cosine similarity/distance inconsistently, including statements like “cosine distance” for ICS and then “average intra-cluster agreement” defined with similarity elsewhere.
  - The resolution score \(RS_k = w\,ICS_k + (1-w)(1-ICD_k)\) is intuitive, but the interpretation of “higher is better” depends on the scale and direction of the underlying metrics. Because ICD is defined as a mean similarity to other clusters, \(1-ICD_k\) is not necessarily a well-calibrated separation measure.
- The choice of \(w = 1/3\) is not well justified. The paper says it was chosen by a “small grid search” and “found to give a stable ordering,” but there is no principled criterion, no sensitivity analysis in the main text, and no explanation of whether this was tuned on the same data used for evaluation.
- The Stage 1 and Stage 2 distinction is important, but the methodology blurs them. It appears the same GOBP benchmark was used to choose the prompt/model/embedding configuration that is then applied to the Perturb-seq task. That is reasonable in principle, but the paper does not clearly separate development and evaluation data, which matters for an ICLR-standard claim of generalization.
- The clustering procedure itself is underspecified. The appendix mentions PCA, kNN, Leiden, and assignment matrices, but the main method does not explain exactly how gene signatures are formed from clusters in a way that avoids bias from cluster size or perturbation composition.
- A key failure mode is not addressed: if the LLM repeatedly produces generic GO phrases for many clusters, the embedding distances could reflect wording variance rather than biological separability. The paper does not discuss how it guards against generic but semantically distinct-sounding outputs.
- The method is claimed to be model-agnostic, but in practice it depends on a specific proprietary model and embedding service. The portability of the approach is therefore less established than stated.

### Experiments & Results
- The experiments partially test the claims, but not strongly enough for ICLR’s acceptance bar if judged as a method paper.
- Stage 1 on 100 non-redundant GOBP sets is a reasonable benchmark for annotation quality, but the evaluation is not presented with enough rigor. The paper reports comparisons across prompts, temperatures, models, and embeddings, yet there is no clear table of quantitative results, confidence intervals, or statistical tests.
- A major concern is that the main resolution-selection claim is supported by a single Perturb-seq dataset from K562. That is too little evidence for a general method. The paper asserts broad applicability, but there is no cross-dataset validation, no different organisms, no different perturbation regimes, and no negative controls.
- The baselines are only partially appropriate:
  - Silhouette and modularity are reasonable generic clustering baselines.
  - Functional enrichment is not really a baseline for resolution selection; it is itself an annotation method, and the paper compares the proposed score to enrichment-derived summaries in a somewhat circular way.
- There is no comparison to simpler biologically informed alternatives, such as resolution selection based on enrichment coherence, marker-gene stability, or cluster-label consistency without LLMs.
- The results are presented mostly as elbow plots and box plots, but the paper does not provide strong quantitative evidence that the selected resolution is objectively better than neighboring settings. The conclusion that \(r=0.4\) or \(r=0.5\) is “biologically meaningful” is plausible, but not convincingly demonstrated.
- Error bars and statistical significance are essentially absent. For a method that depends on stochastic LLM outputs and prompt design, this is a substantial omission. Repeated runs are mentioned in Stage 1, but not in a way that quantifies uncertainty in the final resolution choice.
- The paper also does not analyze computational cost in a way that would be meaningful for real use: “minutes” is not enough, especially given repeated retrieval, multiple resolutions, and external API dependence.
- The evaluation of annotation quality on GOBP sets appears to use cosine similarity between generated text and reference text. This is a weak proxy for correctness and may reward lexical overlap rather than biological correctness. The paper does not justify why this metric is sufficient, nor does it compare against expert human judgment or curated ontology-based matching.
- Several claims are not backed by the presented evidence, including “superior biological interpretability compared with traditional metrics” and “orders of magnitude faster than manual curation.”

### Writing & Clarity
- The paper’s main ideas are understandable, but several sections are difficult to follow because key methodological details are buried in the appendix or described inconsistently across the main text and supplement.
- The method section would benefit from clearer, mathematically consistent definitions of ICS, ICD, and RS. As written, the relationship between similarity and distance is easy to misread.
- Figures 3–6 are referenced meaningfully, but the captions and surrounding text do not always explain what specific conclusion each figure should support beyond a general “the optimum is here.” The figures appear to be more descriptive than evidential.
- Table 1 is helpful in principle, but the semantics of the metrics are not fully aligned with the formulas in Section 3.4, which makes the scoring logic harder to audit.
- The appendix contains useful implementation information, but the paper still lacks enough concise detail in the main text for readers to reconstruct the evaluation protocol without substantial inference.

### Limitations & Broader Impact
- The conclusion acknowledges some limitations, including scalability, generalizability, LLM dependence, cost, and prompt sensitivity. That is good, but it is quite brief relative to the methodological risks.
- The paper misses several fundamental limitations:
  - The score is based on the model’s own generated hypotheses, which can amplify model bias or hallucination rather than correct it.
  - Proprietary LLM and embedding dependence raises reproducibility and accessibility concerns.
  - The approach may be brittle to ontology choice, gene-set size, cluster size imbalance, and generic outputs.
  - The method does not establish that a higher resolution score corresponds to better biological truth, only that it correlates with a preferred resolution on the tested dataset.
- Broader-impact discussion is thin. There is no real discussion of how automated annotation might mislead biological interpretation if wrong, or how over-trust in LLM-generated labels could affect downstream experimental decisions. For ICLR, this is a meaningful omission given the scientific-decision role of the system.

### Overall Assessment
HYPOGENEAGENT is an interesting and timely attempt to use LLM-generated gene-set hypotheses as a signal for selecting clustering resolution in single-cell / Perturb-seq analysis. The idea is plausible and potentially useful, and the paper includes a concrete pipeline plus preliminary experiments. However, the current evidence is not strong enough for ICLR’s typical standard for a method paper: the core score is somewhat circular, the evaluation is narrow, uncertainty is not well quantified, and the main claims of generality and superiority are stronger than the results justify. I think the contribution is promising, but it needs substantially more rigorous validation, clearer formal justification, and stronger baselines before the paper would be convincing as an ICLR acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes HYPOGENEAGENT, an LLM-based framework that uses gene-set/cluster descriptions to select clustering resolution in Perturb-seq data by maximizing a new “Resolution Score” based on intra-cluster annotation agreement and inter-cluster distinctiveness. The paper also benchmarks LLM prompting/embedding choices on curated GO Biological Process gene sets and applies the method to a K562 Perturb-seq dataset, claiming that the selected resolutions better match known biology than silhouette, modularity, and enrichment-based baselines.

### Strengths
1. **Addresses a real and important problem in single-cell analysis.**  
   Resolution selection and functional annotation are indeed major pain points in scRNA-seq/Perturb-seq workflows, and the paper motivates this well with concrete discussion of manual marker inspection and heuristic tuning.

2. **Attempts to close the loop between annotation and clustering.**  
   A notable idea is to use the annotation content itself as feedback for choosing clustering resolution, rather than treating annotation as a purely downstream task. This is conceptually appealing and potentially useful for biology-aware clustering.

3. **Includes a benchmark stage for prompt/model/embedding choices.**  
   The paper does not immediately jump to the end task; it first explores prompt variants, model families, temperatures, and embedding back-ends on curated GOBP sets. That is a reasonable design choice for an LLM-centric method.

4. **Compares against standard clustering baselines.**  
   The manuscript evaluates against silhouette and modularity, which are common internal clustering criteria, and also discusses GO enrichment as a biological baseline. This is aligned with ICLR expectations for comparative evaluation.

5. **Uses a public Perturb-seq dataset and describes a preprocessing pipeline.**  
   The paper states that it uses the Replogle et al. K562 dataset and provides a reasonably detailed preprocessing workflow, which is helpful for potential reproduction.

### Weaknesses
1. **The core methodological novelty is limited and the scoring design is somewhat ad hoc.**  
   The Resolution Score is a weighted combination of cosine similarities between LLM-generated text hypotheses. This is an intuitive heuristic, but the paper does not establish that it is a principled objective, nor does it show that the score is robust to prompt/model variation. At ICLR, a new metric usually needs stronger justification, ablation, and theoretical or empirical validation.

2. **Evidence of biological correctness is weak and mostly qualitative.**  
   The main result is that the chosen resolutions “align with known pathway” or appear “clear as expected,” but the paper does not provide rigorous quantitative validation against known perturbation labels, held-out biological annotations, or independent expert review. The claim that the method outperforms traditional metrics is therefore not convincingly demonstrated.

3. **The evaluation protocol appears circular in places.**  
   The same LLM family is used to generate annotations and then its outputs are scored semantically using embedding similarity. Since the method judges cluster quality by the internal consistency of LLM-produced text, it may primarily measure prompt stability or model self-consistency rather than true biological quality. This is a substantial concern for credibility.

4. **Baselines are underdeveloped and not sufficiently fair.**  
   Silhouette and modularity are not directly comparable to an LLM-driven annotation score because they optimize different notions of clustering quality. The paper also mentions functional enrichment, but the evaluation of that baseline is not clearly standardized or separated from the proposed method. Stronger baselines would include resolution-selection methods tailored to single-cell data and downstream biological agreement metrics.

5. **Reproducibility is incomplete.**  
   The paper says code will be released upon acceptance, which is not sufficient for a reproducibility-oriented venue. In addition, the method depends on proprietary models/APIs (GPT-o3, GPT-4o, GPT-5, Azure OpenAI embeddings), which makes exact replication difficult. Several key implementation details are also underspecified, including prompt contents, retrieval tool behavior, and exact scoring computations in some sections.

6. **Clarity and technical presentation are uneven.**  
   The manuscript contains many repetitive explanations, some inconsistent notation, and several places where the methodology is not crisply defined. For example, the construction of “gene-to-cluster” and “perturbation-to-cluster” assignment matrices is not clearly connected to the final Resolution Score, and the rationale for the weighting parameter \(w=1/3\) is only loosely justified by a “small grid search.”

7. **Scalability and cost are not convincingly addressed.**  
   Running an LLM with retrieval on every cluster at every resolution, then repeatedly sweeping models/prompts/temperatures, may be expensive and slow. The paper claims efficiency but does not provide runtime, token usage, or cost analyses that would be expected for an LLM-based pipeline.

8. **The scope of validation is narrow.**  
   The core application appears to be one K562 Perturb-seq dataset. For ICLR, a method paper should usually demonstrate broader generality, e.g. across multiple datasets, perturbation types, or cell line contexts.

### Novelty & Significance
The idea of using LLM-generated functional hypotheses as feedback for clustering resolution selection is reasonably novel and could be practically interesting for biomedical data analysis. However, in its current form the contribution looks more like an engineering heuristic than a broadly validated method, and the empirical evidence is too limited to establish strong significance at ICLR’s acceptance bar. I would rate the novelty as **moderate**, the clarity as **moderate to weak**, reproducibility as **weak**, and significance as **potentially meaningful but not yet convincingly demonstrated**.

### Suggestions for Improvement
1. **Provide a stronger quantitative evaluation of biological validity.**  
   Add metrics that compare selected resolutions against known perturbation/pathway structure, expert labels, or external reference annotations. Show whether the method improves task-relevant downstream outcomes, not just text similarity.

2. **Include more and stronger baselines.**  
   Compare against single-cell-specific resolution-selection approaches and biological coherence criteria, not just silhouette and modularity. If using enrichment-based baselines, define them carefully and ensure fairness.

3. **Perform ablations on each design choice.**  
   Is the benefit coming from the LLM, the retrieval tool, the hypothesis prompt, or the embedding similarity aggregation? Show results for no-retrieval, no-self-verification, alternative scoring formulas, and alternative weighting \(w\).

4. **Demonstrate robustness across datasets and models.**  
   Evaluate on multiple perturb-seq or scRNA-seq datasets, with different cell types and perturbation structures, and show that the chosen resolution is stable across back-end LLMs and embedding models.

5. **Clarify the scoring methodology and notation.**  
   Give precise formulas, define every symbol consistently, and explain why cosine similarity of generated textual hypotheses is a valid proxy for cluster coherence and separation.

6. **Add runtime, cost, and token-budget analysis.**  
   Since this is an agentic LLM method, report the computational cost per dataset and per resolution sweep, and discuss how this scales to larger atlases.

7. **Release full code and prompts, ideally with a reproducible offline setup.**  
   For ICLR, this would substantially strengthen the paper. If proprietary APIs must be used, provide fixed prompt templates, cached outputs, and enough detail to replicate the evaluation as closely as possible.

8. **Reduce the framing claims.**  
   The paper should be more careful about statements like “objective adjudicators” or “fully automated” unless supported by rigorous validation. A more measured framing would make the contribution more credible.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add evaluation on multiple independent Perturb-seq/scRNA-seq datasets, not just K562 Replogle. At ICLR, a single-dataset case study is not enough to support a general “resolution selection” method; the claim needs testing across different cell types, perturbation types, and cluster structures.

2. Compare against stronger resolution-selection baselines that directly target cluster validity, not just silhouette and modularity. Include methods like stability-based selection, consensus clustering, gap statistic, and MultiK-style objective selection, because otherwise the claimed superiority over “traditional metrics” is too weak to be convincing.

3. Add a direct comparison to using GO enrichment alone for resolution selection. Since the method is basically replacing enrichment-based interpretation with LLM-generated annotation consistency, the paper needs to show that HypoGeneAgent adds value beyond standard enrichment summaries computed at each resolution.

4. Report performance against a non-LLM annotation pipeline using the same marker genes and retrieval context. Without a baseline like Gene Ontology enrichment, scmap/CellAssign-style mapping, or an LLM without retrieval/self-verification, it is unclear whether gains come from the agent design or simply from having any annotation step at all.

5. Add robustness experiments under perturbations of the input signature construction. The paper’s claim depends heavily on marker selection and ranking, so it should test sensitivity to top-N markers, alternative differential-expression methods, and batch/subsampling variation.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify whether the selected resolution actually improves biological recovery, not just the RS score. The paper needs external validation against known perturbation labels/pathways, enrichment purity, or holdout annotations to show that maximizing Resolution Score correlates with more correct biological structure.

2. Analyze the dependence on the arbitrary weighting hyperparameter w. The current “small grid search” is not enough for ICLR; the method needs a sensitivity analysis showing whether the selected resolution is stable over w and whether different choices change the conclusion.

3. Show that intra-cluster agreement and inter-cluster distinctiveness are not redundant with simple text similarity artifacts. Since both are derived from embedding cosine similarity over LLM outputs, the paper should analyze whether they mostly reflect prompt wording, ontology hierarchy proximity, or repeated generic phrases rather than real cluster biology.

4. Evaluate calibration and reliability of the agent’s confidence scores. The paper uses confidence to rank hypotheses, so it must show calibration curves or accuracy-vs-confidence statistics; otherwise the “self-verification” claim is unsupported.

5. Provide complexity/cost analysis for the full pipeline. If this is meant to be a practical automated method, readers need token cost, runtime, and scaling behavior versus the number of clusters and hypotheses; otherwise the claimed efficiency is not credible.

### Visualizations & Case Studies
1. Show cluster-level case studies at the chosen and nearby resolutions, with the top genes, LLM hypotheses, and GO enrichment side-by-side. This would reveal whether the resolution score is identifying genuinely coherent biological modules or just producing semantically similar generic descriptions.

2. Include failure cases where the method picks an obviously wrong resolution. ICLR reviewers will look for limits; without examples of ambiguous clusters, mixed signatures, or over-split modules, it is impossible to judge whether the score is robust or brittle.

3. Plot the full relation between Resolution Score and external biological quality metrics across resolutions. A scatter or rank correlation plot against enrichment purity, known pathway recovery, or perturbation-label consistency would show whether RS tracks meaningful biology rather than an internal text-similarity proxy.

4. Visualize the contribution of each component in the score per cluster, not just medians. Cluster-wise plots of ICS and ICD would expose whether the method is driven by a few easy clusters while failing on biologically important outliers.

### Obvious Next Steps
1. Extend the pipeline to additional ontologies and modalities, then test whether the same scoring rule still selects meaningful resolutions. The paper claims generality, but that claim is currently untested outside GO and one Perturb-seq dataset.

2. Replace the fixed heuristic weighting with a principled model-selection criterion. A learned or validated weighting scheme would be a necessary next step because the current RS is partly hand-designed and therefore not yet a trustworthy objective.

3. Add human expert comparison against manual resolution tuning. Since the paper’s core claim is that it reduces subjective curation, it should compare the selected resolution against expert-chosen settings or inter-annotator agreement.

4. Release an end-to-end reproducible benchmark with frozen prompts, models, and outputs. For an ICLR submission, reproducibility of an LLM-heavy workflow is essential; without it, the method cannot be independently verified or fairly compared.

# Final Consolidated Review
## Summary
This paper proposes HYPOGENEAGENT, an LLM-driven pipeline that annotates gene sets with ranked GO hypotheses and then uses those hypotheses to score clustering resolutions via intra-cluster agreement and inter-cluster distinctiveness. The idea is to couple annotation and resolution selection for single-cell/Perturb-seq analysis, and the paper evaluates the approach on a curated GOBP benchmark plus a K562 Perturb-seq dataset.

## Strengths
- The paper targets a real and important bottleneck in single-cell analysis: choosing clustering resolution in a way that is more biologically interpretable than generic graph or distance metrics.
- The “close the loop” idea of using annotation content itself as feedback for resolution selection is conceptually interesting and could be practically useful if validated more rigorously.
- The authors do include a two-stage setup: a benchmark stage on curated GOBP gene sets to compare prompts/models/embeddings, and a downstream Perturb-seq application. That is better than jumping straight to one case study.

## Weaknesses
- The central resolution score is built from cosine similarities between the model’s own textual hypotheses, so the method is inherently self-referential. This creates a serious circularity problem: the score may mostly reflect prompt consistency or wording stability rather than true biological cluster quality.
- The empirical validation is very narrow. The main clustering claim is supported by a single K562 Perturb-seq dataset, with no cross-dataset, cross-cell-type, or cross-perturbation validation. That is too little evidence for the paper’s broad claims of generality.
- The evaluation does not convincingly establish biological correctness. The paper relies heavily on elbow plots, box plots, and text-similarity proxies, but does not show strong external validation against independent labels, expert review, or downstream task improvement.
- The design choices look ad hoc in important places, especially the weighting parameter \(w = 1/3\). The paper mentions a small grid search, but there is no principled justification or sufficient sensitivity analysis to show the selected resolution is stable.
- Baselines are weak for the stated task. Silhouette and modularity are standard, but they are not strong biologically informed resolution-selection baselines; the paper also does not adequately compare against simpler alternatives such as enrichment coherence, marker stability, or consensus/stability-based methods.
- Reproducibility is limited by heavy dependence on proprietary LLM and embedding APIs, while key prompt and retrieval details are still not fully exposed in the main paper. For an LLM-heavy method, this materially weakens confidence in the reported results.

## Nice-to-Haves
- Add ablations separating the effects of retrieval, self-verification, the hypothesis prompt, and the embedding choice.
- Report runtime, token usage, and cost per dataset/resolution sweep.
- Include cluster-level case studies showing the top genes, LLM hypotheses, and GO enrichment side-by-side for selected and nearby resolutions.

## Novel Insights
The paper’s main genuinely new idea is not simply “use an LLM for gene-set annotation,” but rather to turn the annotation output into a feedback signal for selecting clustering resolution. That is a plausible and somewhat fresh direction: it reframes resolution tuning as a semantic consistency problem rather than a purely geometric or graph-theoretic one. However, the current implementation still looks more like a heuristic wrapper around LLM-generated text than a validated biological objective, so the novelty is real but the scientific claim remains under-supported.

## Potentially Missed Related Work
- MultiK (Liu et al., 2021) — relevant because it is an objective selection method for cluster numbers in scRNA-seq and is a stronger comparator than generic silhouette/modularity.
- GeneAgent (Wang et al., 2025) — relevant because the paper’s own pipeline depends heavily on LLM-based gene-set analysis and self-verification ideas that overlap with this work.
- scmap / CellAssign — relevant as reference-based annotation methods that could serve as non-LLM comparison points for the annotation component.
- Stability/consensus clustering methods — relevant because the paper claims to select resolution objectively, but does not benchmark against stronger resolution-selection criteria beyond standard internal metrics.

## Suggestions
- Add at least one additional independent Perturb-seq or scRNA-seq dataset and show that the same scoring recipe selects sensible resolutions across settings.
- Include a direct correlation analysis between Resolution Score and external biological quality metrics, not just internal text-similarity quantities.
- Strengthen the baseline suite with stability-based or consensus-based resolution selection, plus an enrichment-only resolution-selection baseline.
- Perform a sensitivity analysis over \(w\), marker selection, and top-\(N\) signatures to show whether the chosen resolution is robust.
- Provide frozen prompts, full retrieval details, and cached outputs or an offline reproducible package so the pipeline can be independently verified.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 2.0]
Average score: 1.3
Binary outcome: Reject
