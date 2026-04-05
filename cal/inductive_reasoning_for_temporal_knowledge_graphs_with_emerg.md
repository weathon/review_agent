=== CALIBRATION EXAMPLE 66 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately reflects the paper’s focus on inductive reasoning in temporal KGs with emerging entities.
- The abstract clearly states the problem, the core idea (transferable temporal patterns from semantically similar entities), and the main empirical claim.
- However, the abstract makes a strong performance claim: “average improvement of 28.6% in MRR across multiple datasets.” This is consistent with the reported table, but it would be helpful if the abstract also clarified that the setting is specifically the emerging-entity subset under a nonstandard chronological split, since this materially affects how general the result should be interpreted.

### Introduction & Motivation
- The problem is well-motivated and clearly positioned as a gap in prior TKG reasoning: most methods assume a closed entity set and do not address entities that first appear at test time with no history.
- The empirical framing is compelling, especially the claim that about 25% of entities are unseen during training. That said, the introduction somewhat conflates “emerging entities” with “entities lacking historical interactions” and “entities unseen in training,” and the exact task definition only becomes precise later. This matters because the paper’s setting is stricter than standard inductive TKG completion.
- The contributions are clearly stated, but the introduction slightly over-claims novelty by implying the main challenge is broadly “preventing representation collapse,” when the actual method is a combination of textual clustering, chain encoding, and cluster-level transfer. The collapse story is important, but it is not fully established as the unique bottleneck.
- ICLR-level novelty seems plausible, but the motivation would be stronger if the paper more directly contrasted its setting with prior inductive TKG work and explained why static inductive methods fail specifically because emerging entities have zero history at first appearance.

### Method / Approach
- The method is described in a reasonably structured way, and the Classification–Representation–Generalization pipeline is understandable at a high level.
- That said, there are several important reproducibility and conceptual questions:
  - **Codebook mapping:** The paper uses frozen BERT textual embeddings \(h_e\) and learns a VQ codebook over them. This raises a key question: if the entity title embedding already encodes enough semantics to cluster entities, how much of the gain comes from the codebook versus the subsequent transfer module? The ablation removes “codebook” and “textual encoding” separately, but the paper does not isolate whether the codebook is truly interaction-aware, as claimed, or mostly a semantic clustering layer over title embeddings.
  - **Interaction Chain construction:** Eq. (6) uses TopK over relation similarity, but the rationale for selecting only relation-similar events is not fully justified. This can bias the model toward relation prototypes and may discard temporally relevant but relation-dissimilar events. The paper should discuss failure modes of this heuristic, especially for entities whose future behavior is not relation-repetitive.
  - **Generalization step:** Eq. (9) pools IC embeddings by cluster, then Eq. (11) forms \(\tilde h_e = h_e + \omega_e \cdot c^{dyn}_{\pi(e)}\). This is the most conceptually delicate part. For emerging entities with no history, \(c^{dyn}\) is defined through query entities in the cluster, but for non-query entities the dynamics seem to depend on cluster membership without direct evidence. The paper should explain how cluster prototypes are computed at inference for entities with no observed ICs, and whether this becomes circular or relies on batch-level statistics that are unavailable in a true online setting.
  - **Assumptions:** The entire framework depends on meaningful entity titles and a pretrained text encoder. This is a strong assumption, especially on datasets like GDELT where titles can be noisy. The paper mentions this in ablations, but the method section does not clearly state that text availability is a core prerequisite.
  - **Loss/objective:** The model combines link prediction loss with codebook loss, but it is not fully clear how the codebook is optimized jointly across all entities when assignments are based on frozen text embeddings while downstream reasoning depends on temporally varying IC embeddings. The relationship between these two representation spaces could be more carefully justified.
- For ICLR standards, the method is promising but not yet airtight in how the transfer mechanism is defined and how the assumptions support the claimed inductive setting.

### Experiments & Results
- The experiments are relevant to the paper’s claims: they evaluate performance specifically on emerging entities and also study robustness under alternative temporal splits.
- The main table is strong: TRANSFIR improves over a broad set of baselines on four datasets. However, there are several issues that affect confidence:
  - **Fairness of baselines:** The baseline suite is broad, but some baselines are not naturally designed for the exact emerging-entity zero-history setting. The paper does not sufficiently explain how each baseline was adapted, especially for static inductive methods and generative methods. Since performance is the central claim, these adaptation details matter.
  - **Evaluation protocol:** The paper uses a 5:2:3 chronological split rather than a more standard split, explicitly to increase emergence. That is reasonable, but the paper should be clearer that this choice changes the task difficulty and likely benefits methods tuned for inductive transfer. It would help to report results under a standard split as well, or at least justify the chosen split more rigorously.
  - **Ablations:** The ablation study is useful, but it is not fully sufficient. Missing ablations include:
    - removing the TopK relation filtering in the Interaction Chain,
    - replacing the VQ codebook with simple k-means or fixed clustering,
    - using cluster transfer without textual embeddings,
    - using raw temporal neighborhoods instead of ICs,
    - varying the “interaction-aware” nature of the codebook more directly.
    These would materially clarify which component is responsible for the gains.
  - **Statistical reporting:** Results are averaged over three seeds, which is good, but the main table does not report variance or significance tests. For ICLR, given the size of the reported improvements, some measure of uncertainty would improve confidence.
  - **Metrics:** MRR and Hits@k are appropriate for link prediction, but the paper focuses only on entity ranking. Since the key claim is inductive reasoning on emerging entities, a calibration of how often the correct entity is in the candidate set or a more fine-grained breakdown by first-hop vs. multi-hop would strengthen the evidence.
- Overall, the results support a real improvement, but the paper would benefit from tighter ablations and clearer baseline adaptation details to fully justify the magnitude of the gain.

### Writing & Clarity
- The paper is generally understandable, and the three-stage pipeline is presented in a coherent order.
- The biggest clarity issue is that several core ideas are introduced at a high level, but their precise operational meaning is only partially specified. In particular:
  - “interaction-aware codebook” is intuitive but not fully operationalized beyond VQ over text embeddings,
  - “representation collapse” is compellingly argued empirically, but the causal link to performance degradation is not fully established,
  - the transfer module’s behavior at inference for emerging entities needs a more explicit explanation.
- Figures 2 and 3 are conceptually helpful in linking emergence, collapse, and transfer. Figure 4 and the case study are also useful. But the method would be easier to assess if the paper included a compact end-to-end example showing how a single emerging query is processed through all three stages.
- The experimental appendix is fairly extensive, which helps, but the main paper still leaves some operational details ambiguous enough to slow understanding.

### Limitations & Broader Impact
- The paper acknowledges one important limitation: performance can suffer when entity text is noisy or uninformative, and it suggests external knowledge/LLMs as future work.
- However, it misses some fundamental limitations:
  - the method depends heavily on textual descriptions of entities, which may not exist or may be poor quality in many TKGs,
  - the approach assumes semantically stable clusters, but entity semantics can drift over time,
  - the framework is tailored to emerging entities with title semantics and may not generalize well to domains without good textual metadata,
  - there is no discussion of robustness when relation labels are ambiguous or when emerging entities belong to novel semantic types not seen in training.
- The broader impact discussion is minimal. While this is a standard ML paper with no obvious direct harm, temporal KG forecasting can be used in sensitive domains like geopolitics or surveillance. The paper’s ethics statement is generic and does not address possible downstream misuse or systematic bias propagation from textual encoders and pre-trained language models.

### Overall Assessment
TRANSFIR addresses a real and important gap in temporal KG reasoning: first-appearance entities with no historical interactions. The empirical evidence suggests meaningful gains over a wide set of baselines, and the central idea of leveraging semantic clustering plus transferable interaction chains is plausibly valuable. That said, the paper’s strongest claims rest on a method whose inference behavior is not yet fully clear, whose dependence on textual metadata is substantial, and whose experimental validation would benefit from more discriminating ablations and stronger statistical reporting. For ICLR, this is promising and potentially impactful, but I would want the authors to tighten the methodological explanation and more convincingly isolate why each component is necessary before fully endorsing the contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies a practically important but underexplored setting: temporal knowledge graph reasoning for **emerging entities** that appear at test time without historical interactions. The authors propose **TRANSFIR**, a three-stage framework that uses textual embeddings, a VQ-style codebook for latent semantic clustering, interaction-chain encoding, and cluster-level pattern transfer to address representation collapse and improve forecasting for zero-history entities.

### Strengths
1. **Targets a relevant and underexplored problem in TKG reasoning.**  
   The paper clearly motivates that standard TKG methods rely on a closed-world assumption and fail on entities that appear only at inference time. This is a meaningful gap for ICLR, especially because the setting is common in real-world dynamic graphs.

2. **Provides empirical evidence that emerging entities are common and harmful for existing methods.**  
   The paper reports that roughly **25%** of entities are unseen during training and shows large performance drops on “Emerging” triples versus vanilla test triples. This strengthens the case that the problem is not synthetic or marginal.

3. **The proposed framework is conceptually coherent.**  
   TRANSFIR’s classification–representation–generalization pipeline is reasonably well-motivated: textual embeddings provide a history-free prior, the codebook induces latent semantic clusters, and interaction chains capture transferable temporal patterns. The decomposition makes the design easier to understand than a monolithic architecture.

4. **The paper includes multiple analyses beyond raw accuracy.**  
   It offers collapse-ratio measurements, t-SNE visualizations, ablations, sensitivity studies, and additional evaluation under an “Unknown” setting. For an ICLR submission, this is a positive sign of effort toward understanding behavior rather than only reporting benchmark numbers.

5. **Results appear consistently strong across benchmarks.**  
   The method is reported to outperform many baselines on four datasets, with particularly large gains on GDELT. If the evaluation is sound, this suggests the approach has practical value.

6. **Reproducibility is at least partially addressed.**  
   The paper states that code will be publicly available and provides implementation details, baselines, and hyperparameter sensitivity analyses. This is aligned with ICLR expectations, even though the actual reproducibility depends on the released code and exact split protocol.

### Weaknesses
1. **The novelty is moderate rather than clearly breakthrough-level.**  
   The core ingredients—textual embeddings, clustering/codebooks, temporal sequence modeling, and transfer from cluster prototypes—are individually familiar. The paper’s main contribution is the particular combination for a new setting, but it is less clear that it introduces a fundamentally new learning principle. For ICLR, this may be acceptable if the empirical and conceptual gains are compelling, but the novelty bar is only partially met.

2. **The method relies heavily on entity titles/text, which may limit generality.**  
   TRANSFIR uses pretrained BERT embeddings from entity titles as fixed input. This helps zero-history entities, but it also means the approach may degrade when titles are noisy, absent, underspecified, or not semantically informative. The paper itself notes failure cases on noisy titles, which suggests the method may not generalize uniformly across domains.

3. **The “interaction chain” construction appears somewhat heuristic.**  
   The chain is formed by selecting Top-K past interactions based on relation similarity and then encoding them with a Transformer. This is plausible, but the paper does not fully justify why this particular filtering is optimal, nor does it compare against broader sequence construction alternatives in a systematic way. The design may be sensitive to the query-relation similarity heuristic.

4. **Potential concern about leakage or dependence on test-time structure.**  
   The model groups entities into clusters and computes cluster prototypes using query-time interaction chains. While this is reasonable, the exact training/inference boundary and whether any test-time statistics indirectly influence representation formation need to be extremely clear. The paper’s description is somewhat intricate, making it harder to verify there is no unintended transductive advantage.

5. **Experimental comparison scope is broad, but fairness details are not fully convincing from the paper text alone.**  
   The baseline list is large, but some baselines are adapted to an emerging-entity setting via nonstandard adjustments. The paper says it reduces rule lengths or merges timestamps for static inductive baselines. Such adaptations may be necessary, but they also complicate direct comparability and should be more carefully justified and standardized.

6. **The gains are reported only on a specialized evaluation regime.**  
   The paper focuses on emerging-entity queries under a 5:2:3 split and a zero-history definition. This is appropriate for the task, but the broader significance would be stronger if the method were also shown to help standard extrapolation, cold-start relations, or mixed settings more extensively.

7. **Clarity is uneven in the methodological details.**  
   The overall idea is understandable, but some parts of the formalization are hard to parse from the description: e.g., how cluster prototypes are updated across timestamps, whether the dynamic prototype is per-query or per-timestamp, and exactly how embeddings for non-query entities are used during scoring. This makes it harder to assess the exact algorithmic novelty and correctness.

### Novelty & Significance
**Novelty:** Moderate. The paper’s main novelty lies in framing inductive reasoning for emerging entities in TKGs and proposing a cluster-based transfer mechanism to mitigate representation collapse. However, the individual modeling components are largely built from established ideas, so the contribution is more of a strong integration for a new problem than a fundamentally new paradigm.

**Clarity:** Moderate. The motivation and high-level pipeline are clear, but some algorithmic details and training/inference interactions are difficult to verify from the presentation. For an ICLR paper, the narrative is decent, but the implementation-level clarity could be improved.

**Reproducibility:** Moderately strong in principle. The paper provides split details, baselines, sensitivity studies, and claims code availability. Still, reproducibility would depend on exact preprocessing, clustering/training schedules, and the precise construction of emerging-entity evaluation, which are not fully transparent in the main text.

**Significance:** Reasonably strong. Emerging entities are a realistic and important challenge in dynamic graphs, and the reported gains are large. If validated carefully, the work could be useful to the TKG community. That said, the paper’s significance is somewhat narrower than a broadly transformative ICLR contribution because it addresses a specialized setting with a fairly task-specific architecture.

### Suggestions for Improvement
1. **Strengthen the novelty argument by isolating what is fundamentally new.**  
   The paper should more explicitly distinguish TRANSFIR from prior inductive/static temporal methods and explain why the codebook + cluster-transfer mechanism is not just a straightforward composition of known components.

2. **Add ablations that directly test each design choice.**  
   For example: remove Top-K relation filtering, replace interaction chains with unordered histories, replace VQ clustering with k-means or learned soft clustering, and compare fixed vs. learned textual encoders. This would better support the necessity of each component.

3. **Clarify the exact training/inference protocol and prevent ambiguity about leakage.**  
   A step-by-step algorithm with precise definitions of what information is available at each timestamp would help ICLR reviewers verify the setup. In particular, the paper should clearly state how cluster prototypes are computed and whether any test-set facts influence them.

4. **Broaden evaluation to more challenging or diverse domains.**  
   Since the method relies on textual titles, testing on datasets with poorer textual metadata or different domain structure would better establish robustness. A domain transfer or cross-dataset experiment would also strengthen significance.

5. **Provide a more rigorous comparison to alternative zero-history strategies.**  
   The paper should compare against stronger text-only, prompt-based, or retrieval-based methods that also use entity descriptions, to show that the gains come from the proposed temporal transfer mechanism rather than from text priors alone.

6. **Discuss limitations more candidly.**  
   The current paper already hints at failure cases with noisy titles. A dedicated limitations section should explain when TRANSFIR is expected to fail and how much it depends on semantic metadata versus temporal structure.

7. **Improve exposition of the collapse analysis.**  
   The collapse ratio is interesting, but it would help to provide a more intuitive explanation and perhaps additional quantitative metrics, such as variance spectra or cluster separability measures, to substantiate the claim more convincingly.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison against stronger zero-shot/inductive temporal KG methods that can actually operate with text or unseen entities, especially recent LLM-based and prompt-based TKG systems beyond the current set. ICLR reviewers will question whether the gain is specific to a weak baseline suite, since the core claim is inductive reasoning for unseen entities.

2. Evaluate on additional TKG benchmarks with more diverse semantics and entity types, or at least a held-out cross-dataset transfer setting. The paper claims general inductive reasoning for emerging entities, but four ICEWS/GDELT-style datasets are narrow and do not establish that the method works beyond event-forecasting graphs.

3. Add an ablation that removes the textual encoder and replaces it with a non-text prior of equal capacity, plus a “text-only” baseline. Right now it is unclear whether the main gains come from the codebook/pattern transfer or simply from pretrained entity text embeddings, which directly weakens the contribution claim.

4. Compare against simpler cluster-transfer alternatives: k-means on text embeddings, nearest-neighbor prototype transfer, and mixture-of-experts or soft clustering without VQ. Without these, the codebook-based classifier is not convincingly shown to be necessary for the reported improvements.

5. Report results on a stricter cold-start protocol where emerging entities have zero interactions and zero textual cues, or separately stratify by availability of text quality. The method is explicitly positioned as handling absent history, so the claim is fragile if it relies on informative titles/descriptions.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how performance varies with the amount of prior history for emerging entities, not just the binary Emerging vs. Unknown split. This is needed to show whether TRANSFIR truly solves cold-start reasoning or only helps when a tiny amount of context exists.

2. Analyze cluster purity and assignment quality: how often do semantically similar entities map to the same codeword, and how does that correlate with link prediction accuracy? Without this, the codebook is a black box and the claimed “latent semantic clusters” are not trustworthy.

3. Show whether the method reduces collapse for the right reason by measuring gradient flow or embedding variance separately for known and emerging entities over training. The current collapse analysis is descriptive, but it does not identify which module prevents collapse or whether the effect is stable.

4. Provide an error analysis by entity type, relation type, and emergence frequency. ICLR expects more than aggregate MRR; without breakdowns, it is unclear where the method works and where it fails.

5. Test sensitivity to noisy or ambiguous entity names/descriptions, since the method depends on BERT embeddings of titles. The paper itself admits failures on sparse titles, so robustness to text quality is central to the validity of the approach.

### Visualizations & Case Studies
1. Show nearest-neighbor examples for emerging entities before and after codebook assignment, with the retrieved cluster members and transferred patterns. This would reveal whether the model is actually transferring meaningful structure or just using coarse topical similarity.

2. Add a failure-case gallery with 5-10 representative queries where TRANSFIR fails, grouped by cause: ambiguous text, wrong cluster assignment, relation mismatch, or insufficient history. This is necessary to understand the boundary of the method’s claims.

3. Visualize codebook evolution over training and cluster membership stability across timestamps. If clusters drift wildly or collapse to a few prototypes, the “latent semantic cluster” story is not convincing.

4. Include attention maps or chain selections for representative queries, especially comparing correct vs. incorrect predictions. This would expose whether the Interaction Chain mechanism is selecting genuinely relevant temporal evidence or noisy context.

### Obvious Next Steps
1. Add a rigorous cold-start benchmark protocol with controlled emergence levels and standardized baselines across datasets. That is the most direct way to validate the paper’s central claim under ICLR-level scrutiny.

2. Extend the method to emerging relations and joint entity-relation emergence, since the current formulation only addresses one half of open-world TKG dynamics. The conclusion already hints at this, and omitting it leaves the problem definition incomplete.

3. Replace the static title-only encoder with richer descriptions or external knowledge, and test whether the method still works when entity text is weak. The paper’s reliance on textual semantics is a core assumption and should be stress-tested.

4. Provide a principled theoretical or mechanistic explanation for why VQ clustering plus cluster-level transfer should avoid representation collapse. Right now the method is empirically motivated but not mechanistically justified, which limits confidence at ICLR.

# Final Consolidated Review
## Summary
This paper studies inductive reasoning on temporal knowledge graphs for **emerging entities** that first appear at test time with no historical interactions. It proposes **TRANSFIR**, a three-stage pipeline that uses frozen textual embeddings, a vector-quantized latent codebook, interaction-chain encoding, and cluster-level pattern transfer to mitigate representation collapse and improve link prediction for these cold-start entities.

## Strengths
- The paper tackles a real and underexplored failure mode of TKG reasoning: entities that appear only at inference time with zero history. The empirical study supports that this is not a corner case, reporting that roughly **25%** of entities in the studied benchmarks are unseen during training.
- The proposed pipeline is conceptually coherent and reasonably well-motivated: semantic priors from text, temporal pattern extraction from interaction chains, and cluster-level transfer form a plausible mechanism for zero-history reasoning. The additional analyses on collapse ratio, t-SNE, ablations, and the “Unknown” setting show the authors did more than just report benchmark numbers.
- The reported gains are large and consistent across four datasets, including strong improvement over a broad baseline suite. If the evaluation protocol is sound, the method is clearly effective on the intended emerging-entity task.

## Weaknesses
- The method is heavily dependent on entity titles and pretrained text embeddings, yet this assumption is central to the approach rather than peripheral. This seriously limits generality: many TKGs do not have informative titles, and the paper itself shows that noisy titles can break the clustering/transfer story.
- Several core components remain under-justified and somewhat heuristic. In particular, the Top-K relation-similarity filtering for interaction chains is not convincingly motivated, and the codebook/transfer mechanism is not cleanly separated from the strong textual prior. As written, it is difficult to tell how much of the gain comes from the codebook versus simply having usable text embeddings.
- The experimental validation is still somewhat narrow for the strength of the claims. The paper focuses on a specialized emerging-entity protocol on ICEWS/GDELT-style datasets, with nonstandard splits chosen to increase emergence. That is acceptable for the task, but it limits how far the claimed “inductive reasoning” story can be generalized.
- The baseline comparison is broad, but the adaptation details are not fully convincing from the main paper alone. Since several baselines are not naturally designed for zero-history emerging entities, the fairness and comparability of the reported improvements would benefit from more explicit protocol standardization and stronger zero-shot/text-based baselines.

## Nice-to-Haves
- Add ablations that isolate the key design choices more directly: remove Top-K filtering, replace VQ with k-means or soft clustering, compare against text-only and cluster-only variants, and replace interaction chains with unordered histories.
- Provide a clearer step-by-step example of one emerging query flowing through classification, chain encoding, transfer, and scoring.
- Report uncertainty across seeds more explicitly, e.g. with variance or confidence intervals, to strengthen confidence in the magnitude of the improvements.

## Novel Insights
The most interesting idea here is not simply “use text for unseen entities,” but the attempt to make **temporal pattern transfer operate at the level of semantic clusters** rather than individual entities. That is a useful reframing: the paper argues that the problem on emerging entities is less about learning a per-entity embedding from scratch and more about assigning the entity to a latent type that can inherit interaction regularities from similar known entities. The collapse analysis strengthens this story, although the paper does not fully disentangle whether the observed benefit comes from avoiding collapse itself, from semantic text priors, or from the combination of both.

## Potentially Missed Related Work
- **ULTRA (Galkin et al., 2024)** — relevant as a strong inductive KG generalization method; useful comparison point for zero-shot transfer ideas.
- **zrLLM (Ding et al., 2024)** — relevant because it uses language models for unseen relational settings, which is important given this paper’s dependence on text.
- **POSTRA (Pan et al., 2025)** — relevant as a recent inductive temporal KG transfer method.
- **LLM-DA (Wang et al., 2024b)** — relevant for dynamic adaptation in temporal KG reasoning and could be a stronger text-aware baseline.
- **Prompt-/text-based TKG completion methods such as PPT (Xu et al., 2023a) and ICL (Lee et al., 2023a)** — relevant because TRANSFIR also relies on textual priors, so these are important points of reference.

## Suggestions
- Add a stronger comparative study against text-driven zero-shot/inductive TKG methods and simpler clustering-based transfer baselines.
- Include an explicit robustness analysis under degraded or missing entity text, since that is the main practical weakness of the approach.
- Clarify exactly how cluster prototypes are formed and used at inference, especially for entities with no historical interactions, so the training/inference boundary is unambiguous.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
