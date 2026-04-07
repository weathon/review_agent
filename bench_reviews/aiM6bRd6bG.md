## Summary
This paper formulates the problem of PPI candidate ranking to prioritize novel protein-protein interactions for experimental testing. It proposes a two-stage framework that first uses interpretability-guided retrieval based on active embedding regions from known interactions, then re-ranks top candidates with multiple biological signals including interaction scores, structural plausibility, and semantic features. Evaluation on a large-scale prospective dataset from STRING v11 to v12 shows substantial improvements in ranking metrics over sequence-based PPI prediction models.

## Strengths
- **Novel task formulation with practical motivation:** The paper introduces PPI candidate ranking and sets up a prospective evaluation using successive STRING releases, mimicking real-world discovery scenarios where novel interactions from a later database version serve as the test set. (Evidence: Sections 1, 4, 5.1)
- **Creative methodological contribution:** The interpretability-guided retrieval mechanism leverages active residue regions from contact maps of known interactions to compute embedding similarities, moving beyond raw interaction scores for ranking. (Evidence: Section 4.1, Figure 1)
- **Comprehensive integration of diverse evidence:** The re-ranking module incorporates multiple complementary signals—interaction score, structural (pDockQ), functional annotations, and LLM-based semantics—providing a holistic approach to candidate prioritization. (Evidence: Section 4.2)
- **Large-scale and thorough evaluation:** Experiments on the STRING v11→v12 transition cover a wide range of ranking metrics (Recall@k, MAP, nDCG, etc.) and demonstrate significant improvements, e.g., for D-SCRIPT, Recall@10 increases from <2% to >25% and MRR by 4-6×. (Evidence: Table 1, Section 5.3)
- **Insightful analysis of complementary signals:** Pairwise comparison of re-ranking strategies reveals which evidence types (e.g., PubMedBERT, lightweight semantic overlaps) most consistently improve rankings, offering valuable guidance for future work. (Evidence: Table 2)

## Weaknesses
- **Exaggerated improvement claims:** The paper states "improvements by two orders of magnitude" in the Introduction and Conclusion, but the actual metrics show substantial gains that are closer to one order of magnitude (e.g., Recall@10 improvement of ~12.5×). This overstatement misrepresents the results. (Why it matters: Accuracy in reporting is essential for credibility and proper assessment of contributions.)
- **Risk of data leakage in evaluation:** The use of STRING v12 as "novel" interactions is not explicitly validated to ensure no overlap with v11 evidence beyond the described filtering, potentially inflating prospective performance. (Why it matters: The core claim of prospective evaluation hinges on strict temporal separation between training and test data.)
- **Incomplete baseline comparisons:** The paper only compares against raw interaction scores of PPI prediction models (D-SCRIPT, Topsy-Turvy, xCAPT5), missing key ranking-specific baselines such as collaborative-filtering (e.g., ranking by average interaction score with known partners) or network-based methods. (Why it matters: Without stronger and more relevant baselines, the superiority of the proposed framework is not fully established.)
- **Lack of statistical validation:** Results are presented as point estimates (averages) without measures of variance, confidence intervals, or statistical significance testing across target proteins. (Why it matters: For high-stakes decisions in ML, robustness and reliability of improvements must be demonstrated.)
- **Missing ablation studies:** The individual contributions of interpretability-guided retrieval versus the re-ranking module are not isolated, and the impact of each re-ranking signal is only shown pairwise, not in an integrated manner. (Why it matters: Understanding which components drive performance is critical for methodological insight and future improvements.)
- **Ambiguity in region selection method:** The algorithm for identifying the "most active region" from contact maps is described textually but lacks precise steps (e.g., thresholding, contiguity enforcement), affecting reproducibility. (Why it matters: Reproducibility is a cornerstone of scientific validation, especially for novel methods.)
- **Generalizability concerns:** Primary evaluation is on a single dataset (STRING); appendix results on the PiNUI dataset show much lower rediscovery ratios (0.38 vs. 0.97), and the core assumption that novel interactions resemble known ones may fail for proteins with few interactors, which is not quantified. (Why it matters: Practical applicability depends on performance across diverse datasets and conditions, including cold-start scenarios.)
- **Potential data leakage in LLM fine-tuning:** The PubMedBERT model is fine-tuned on STRING v11 annotations but pre-trained on biomedical literature that may include information about v12 interactions, risking indirect leakage. (Why it matters: This could artificially boost re-ranking performance, compromising the validity of the semantic signal analysis.)
- **High computational cost:** Retrieval requires hundreds of hours (Figure 2), and structural re-ranking with SpeedPPI is prohibitively slow (~13 minutes per pair), limiting scalability for genome-wide screening. (Why it matters: Efficiency is critical for real-world adoption in large-scale biological discovery.)
- **Limited interpretability:** While interpretability is used as a structural tool to extract active regions, the final rankings do not provide biological explanations for why candidates are ranked high, as acknowledged by the authors. (Why it matters: Explainability enhances trust and can offer biological insights beyond mere prioritization.)

## Nice-to-Haves
- Visualizations or case studies illustrating successful and failed rankings to enhance intuitive understanding.
- Exploration of integrated re-ranking models that combine multiple signals via learned weights or ensembles.
- Cross-organism evaluation (e.g., on yeast or mouse data) to demonstrate robustness beyond human STRING.
- More detailed algorithmic description or pseudo-code for the active region selection step to improve reproducibility.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"Two orders of magnitude" claim is hyperbolic:** This is kept as a weakness because it is factually based on the paper's own metrics, but note that the improvement is still substantial.
- **Criticisms about formatting or style nitpicks:** None were present in the reviews.
- **Demands for theoretical proofs or user studies:** Not applicable as this is an empirical systems paper.
- **Suggestions that the paper should have implemented obvious next steps (e.g., end-to-end ranker):** These are moved to Nice-to-Haves as they are improvements beyond the current scope.

## Novel Insights
The paper's key novel insight is the use of model interpretability not for explanation but as a methodological device to guide retrieval: by focusing embedding similarities on active residue regions identified from contact maps of known interactions, it effectively prioritizes novel candidates that share functional or structural patterns with established partners. This approach transforms internal model representations into a ranking mechanism. Additionally, the analysis reveals that semantic signals from LLMs and functional annotations consistently complement sequence-based methods, underscoring the importance of multi-evidence integration for prospective PPI discovery where no single signal suffices.

## Suggestions
- Address data leakage concerns by explicitly verifying no overlap between STRING v11 and the novel v12 interactions used in evaluation, perhaps through evidence timestamps or curation details.
- Include additional baselines such as collaborative-filtering (e.g., ranking candidates by average interaction score with known partners) or network propagation methods to strengthen the ranking comparison.
- Conduct ablation studies to quantify the individual contribution of interpretability-guided retrieval versus re-ranking, and test combined re-ranking strategies beyond pairwise comparisons.
- Perform statistical significance testing (e.g., paired tests across target proteins) on ranking improvements to demonstrate robustness.
- Quantify performance degradation for proteins with few known interactors (cold-start problem) to assess the limits of the core assumption.
- Implement controls for LLM data leakage, e.g., by using LLMs with pre-training cut-offs before v12 data or carefully curating training corpora.
- Provide a more precise algorithm or pseudo-code for the active region selection step in interpretability-guided retrieval to ensure reproducibility.