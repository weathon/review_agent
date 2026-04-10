## Summary
This paper proposes Real Deep Research (RDR), a pipeline for automated analysis of research landscapes. It uses off-the-shelf LLMs and embedding models to filter papers, extract content along expert-defined perspectives (e.g., Input, Modeling for foundation models), cluster the embeddings, and generate structured surveys, trend analyses, and cross-domain maps. The framework is demonstrated on AI and robotics literature, with evaluations comparing its outputs to those from prompted commercial LLMs.

## Strengths
- **Well-structured and practical pipeline**: The method is described in clear, sequential stages (data prep, reasoning, projection, analysis) with concrete prompts and perspective definitions. This makes the approach transparent and potentially replicable for the targeted domains.
- **Substantial qualitative demonstration**: The paper provides extensive outputs—generated survey tables, perspective-specific breakdowns, trend visualizations, and a knowledge graph—which usefully illustrate the kind of automated meta-analysis the pipeline enables.
- **Quantitative evaluation against commercial tools**: A pairwise user study with domain experts shows RDR’s generated surveys are frequently preferred over those from prompted commercial LLMs (e.g., GPT-5, Gemini), offering initial evidence of practical utility in the tested domains.

## Weaknesses
### Major:
- **Limited technical novelty**: The pipeline is a competent orchestration of existing, off-the-shelf components (LLMs for filtering/reasoning, a pre-trained embedding model, standard clustering). The core contribution is the framework design and application, not a novel algorithm for embedding, clustering, or summarization. The paper does not demonstrate that the *combination* of these steps yields qualitatively different or more insightful results than a carefully engineered single LLM prompt, beyond the user study.
- **Insufficient and opaque evaluation of core claims**: The primary validation rests on a user study that is underspecified: only 8 evaluators and 80 total comparisons are mentioned, with no reported inter-rater agreement or detailed criteria for “superior quality.” The baseline is a single, generic prompt to commercial LLMs, which is a weak comparison to “existing commercial tools.” More critically, key claims—like identifying “future research trends” (e.g., teleoperation rising, RL declining)—are presented as insights but lack validation. The method for deriving trends from embeddings is not described, and no predictive check (e.g., training on past data to predict recent trends) is performed.
- **Reproducibility concerns due to undisclosed corpus and filtering**: While venue statistics are given, the exact set of 4,424 foundation model and 1,186 robotics papers used for analysis is not provided. The area-filtering step, which uses an LLM to decide paper relevance, is central to building the corpus, but its accuracy and potential biases are not evaluated. This makes it impossible to verify the trends, clusters, or survey outputs independently.

### Minor:
- **Lack of ablation or sensitivity analysis**: The impact of critical choices—such as the number of clusters *k*, the specific embedding model, or the prompt designs for perspective extraction—on the final outputs is not examined. The pipeline’s robustness to these parameters is unknown.
- **Inadequate discussion of limitations and failure modes**: The paper does not address potential error propagation (e.g., LLM hallucinations during filtering or perspective extraction), the cost/scalability of running multiple LLM calls on thousands of papers, or the effort required to define useful “perspectives” for new domains. The dependence on pre-defined perspectives is a strong prior that could bias the entire analysis.

### Trivial:
- The writing has minor grammatical hiccups (e.g., “shed lights on researchers”) but remains clear overall.

## Nice-to-Haves
- Releasing the curated paper dataset (with IDs) and the derived perspective annotations would greatly enhance reproducibility and enable benchmark creation.
- A direct comparison against prior automated literature analysis systems (e.g., SciLitLLM, PaSa) on specific tasks like survey generation or trend detection would better position RDR within the research landscape.
- A simple validation of trend predictions—e.g., running the pipeline on data up to 2023 and checking if it identifies trends that materialized in 2024—would strengthen the claim of insightful forecasting.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness about unreleased models/tools**: Any suggestion that cited models (Doubao, o3, NV-Embed-v2) are “not yet released” or “cannot be independently verified” is removed. The paper cites them; they are assumed available.
- **Weakness about unfair comparisons favoring baselines**: The harsh critic’s claim that the baseline prompt is “poorly-specified” and makes the comparison unfair *against RDR* is removed per the asymmetric rule (the asymmetry favors the baseline, not the authors’ method).
- **Weakness about missing related work**: The human finder’s point about “limited comparative analysis with existing automated literature review systems” is removed because, as a meta-reviewer, I cannot confirm the existence or relevance of systems like “AutoSurvey” not cited in the paper.
- **Generic strengths**: Strengths like “Clear exposition” or “Application focus” are removed as they are too generic (“well-written,” “important topic”).
- **Formatting nitpicks**: None were present.
- **Reproducibility nitpicks about hyperparameters**: Requests for complete training logs or hyperparameter details for off-the-shelf models are removed as impractical.
- **Strawman weaknesses**: The harsh critic’s claim that the trend analysis method is “not described” is partially valid (it’s underspecified), but the paper does mention using “embedding-based clustering” and implies temporal analysis; the criticism is kept but softened from “not described” to “not described in detail and not validated.”

## Suggestions
- **Strengthen the evaluation**: Conduct a more rigorous user study with more evaluators, clear evaluation rubrics, and inter-rater agreement scores. Perform an ablation study to show the value added by each pipeline stage (e.g., perspective extraction vs. raw embedding clustering). Validate trend predictions with a held-out time period.
- **Improve reproducibility**: Release the filtered paper IDs and the code/scripts for the pipeline stages to allow exact replication of the analysis.
- **Add a limitations section**: Explicitly discuss the framework’s dependencies on off-the-shelf model quality, potential error modes, computational costs, and the manual effort required to define perspectives for new fields.

## Overall Assessment
The paper addresses a timely and valuable problem—automated research landscape analysis—and presents a coherent, well-documented pipeline that demonstrates practical potential through extensive qualitative outputs. However, its contributions are primarily in framework design and application, not in technical innovation. The evaluation, while a positive step, is not sufficiently rigorous or comprehensive to substantiate the core claims of superiority over existing tools or of providing validated future trend insights. The work feels more like a competent proof-of-concept project than a novel research advance. For acceptance at a top-tier venue, significant strengthening of the evaluation and validation of its unique claims would be required.