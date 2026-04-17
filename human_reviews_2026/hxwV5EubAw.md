# Hippoformer: Integrating Hippocampus-inspired Spatial Memory with Transformers

- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
Transformers form the foundation of modern generative AI, yet their key–value memory lacks inherent spatial priors, constraining their capacity for spatial reasoning. In contrast, neuroscience points to the hippocampal–entorhinal system, where the medial entorhinal cortex provides structural codes and the hippocampus binds them with sensory codes to enable flexible spatial inference. However, existing hippocampus models such as the Tolman-Eichenbaum Machine (TEM) suffer from inefficiencies due to outer-product operations or context-length bottlenecks in self-attention, limiting their scalability and integration into modern deep learning frameworks. To bridge this gap, we propose mm-TEM, an efficient and scalable structural spatial memory model that leverages meta-MLP relational memory to improve training efficiency, form grid-like representations, and reveal an intriguing link between prediction horizon and grid scales. Extensive evaluation shows its good generalization on long sequences, large-scale environments, and multi-step prediction, with analyses confirming that its advantages stem from explicit understanding of spatial structures. Building on this, we introduce Hippoformer, which integrates mm-TEM with Transformer to combine structural spatial memory with precise working memory, achieving superior generalization in both 2D and 3D prediction tasks and highlighting the potential of hippocampal-inspired architectures for complex domains. Overall, Hippoformer represents a initial step toward seamlessly embedding structured spatial memory into foundation architectures, offering a potential scalable path to endow deep learning models with spatial intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes mm-TEM, a variant of the Tolman-Eichenbaum Machine using
meta-MLP memory, and Hippoformer, which integrates mm-TEM with Transformers. The
authors evaluate these models on 2D grid prediction and 3D environment tasks,
claiming superior generalization and discovering that memory update frequency
affects grid-like representation scales.

### Strengths
* Implementation efficiency: mm-TEM trains faster than original TEM (in gradient * steps), making hippocampal-inspired architectures more practical - but see note below regarding "efficiency"
* Integration with Transformers: The Hippoformer architecture combining structured spatial memory with Transformer's working memory is conceptually interesting.
* Extensive empirical evaluation: The paper includes multiple experimental settings (varying context lengths, environment sizes, circular grids, 2D and 3D tasks).
* Clear presentation: The paper is generally well-written with good figures illustrating the architecture and results.
* Emergence of grid-like representations: The spontaneous development of grid patterns provides interesting connections to neuroscience.

### Weaknesses
## Missing Foundational Literature on Hippocampal-Entorhinal Memory Systems

The introduction would be strengthened by acknowledging the broader foundational
literature on hippocampal-entorhinal contributions to episodic and relational
memory beyond spatial navigation. Several core ideas presented as novel to the
TEM framework—namely, factorized MEC-LEC streams, hippocampal binding of
structural and sensory codes, and generalization across relational spaces—have
substantial precedent in earlier empirical and computational work. The 'binding
of items and context' (BIC) model (Eichenbaum et al., 2007) directly addresses
how the hippocampus binds contextual and item information. Lesion and recording
studies have established MEC vs LEC functional dissociations (Hargreaves et al.,
2005; Knierim et al., 2014). The Complementary Learning Systems framework
(McClelland, McNaughton, & O'Reilly, 1995) proposed separate systems for rapid
relational binding and structured representations decades earlier. Computational
models have explicitly implemented factorized spatial and sensory
representations: Hasselmo et al. (2002) modeled grid cells for memory with
structural/contextual codes; Franzius, Sprekeler & Wiskott (2007) demonstrated
grid/place code emergence with associative mapping to sensory inputs; Bush et
al. (2015) explicitly separated grid-cell path integration from memory
association in recurrent networks. I recommend that the authors broaden their
introductory discussion to acknowledge this literature and clarify precisely
what mm-TEM contributes beyond these established frameworks rather than
appearing to introduce factorized hippocampal memory de novo.


## Terminology: "TEM Theory" Should Be "TEM Model/Framework"

Throughout the manuscript, the authors refer to "TEM theory" (e.g., lines 54,
134). I suggest replacing this with "the TEM model" or "the TEM framework." The
Tolman-Eichenbaum Machine (Whittington et al., 2020) is a computational
instantiation of existing theoretical ideas about cognitive mapping and
relational memory, rather than a full-fledged theory in itself. The underlying
theoretical principles, in particular that the hippocampal-entorhinal system
implements factorized memory for flexible generalization, predate TEM by decades
(see comment above). Calling TEM a "theory" risks overstating its conceptual
novelty and may mislead readers about its epistemic status. The contribution of
Whittington et al. (2020) was to provide an elegant computational implementation
of these principles, not to propose the theoretical framework itself. This
distinction matters for accurately situating the current work in the scientific
literature.


## Missing Related Work on Factorized Memory Architecturs
The paper would benefit from acknowledging a broader range of computational
models that have explored factorized memory architectures combining structural
and sensory representations. Beyond the TEM lineage cited, several prior works
share the core principle of separating spatial/structural codes from
sensory/content codes for flexible memory retrieval. Hasselmo et al. (2002)
modeled how entorhinal cortex provides structural/contextual codes while the
hippocampus performs binding operations. The Complementary Learning Systems
framework (McClelland, McNaughton, & O'Reilly, 1995), while not specifically
EC-HC focused, proposed separate systems for rapid relational binding versus
structured long-term representations—a conceptual predecessor to factorized
memory architectures. Franzius, Sprekeler & Wiskott (2007) demonstrated
grid/place code emergence with associative mapping to sensory inputs, providing
an early computational analogue of separating structural representations from
sensory mappings. Bush et al. (2015) explicitly implemented computational
separation of grid-cell-like path integration from memory association in
recurrent networks. Stachenfeld et al. (2017) modeled the hippocampus as a
predictive map using successor representations, integrating spatial structure
with value predictions. Waniek (2020) proposed Transition Scale-Spaces that
integrate structural and sensory information in multi-scale frameworks. Even
work on sequence memory in recurrent networks (e.g., Rajan, Harvey, & Tank,
2016) employs separate state/structure representations with readout layers,
loosely corresponding to mm-TEM's structural versus sensory binding.
Acknowledging these precedents would provide better context for understanding
how mm-TEM's specific implementation choices (meta-MLP memory, auxiliary losses,
Transformer integration) extend versus replicate prior ideas, and would help
readers assess the true novelty of the contribution.


## Imprecise Terminology: "Novelty or Surprisal"
The manuscript refers to "novelty or surprisal" in describing the fast-weight
update mechanism (lines 171-172). In formal information theory, these are
distinct concepts: surprisal is a probabilistic quantity (−log P(x)), whereas
novelty is relative to memory or prior experience (Palm, 2012). What the model
actually computes is the gradient of reconstruction loss, ∇_Θ L(k_t, v_t), which
measures prediction error -- the mismatch between predicted and actual values. I
recommend clarifying which quantity is actually computed and adjusting the
terminology to reflect the implementation accurately. The most precise term
would be "prediction error" or "reconstruction error" rather than the ambiguous
"novelty or surprisal." This precision matters for readers seeking to understand
the computational mechanism and for those attempting to reproduce or extend the
work.


## Misleading Description of Hippocampal-Entorhinal Feedback
In Appendix A.2 (lines 630-632), the manuscript states: "visual sensory cues
provide feedback from the HC to correct path integration errors in the MEC."
This description is biologically misleading. While feedback from the hippocampus
can indeed help stabilize MEC representations and correct errors in path
integration (Diehl et al. 2019, Mulas et al. 2016, and many others), the hippocampus does not send raw visual sensory information to
MEC. Visual and other sensory inputs primarily reach MEC via cortical pathways
(particularly through LEC from perirhinal and parahippocampal cortices). A more
accurate description would be that the hippocampus provides relational or
spatial feedback, likely in the form of conjunctive codes that bind spatial and
sensory information, that can help recalibrate MEC structural representations,
while MEC continues to integrate sensory cues from neocortical areas. I
recommend rephrasing this passage to clarify that the feedback represents a
computational abstraction where HC provides spatial/relational corrections
rather than literal sensory signals. This distinction is important for readers
interpreting the model in a neurobiological context and for understanding what
aspects of the architecture are biologically plausible versus computational
conveniences.


## Missing Critical Baseline: Direct Associative Memory Comparison
The paper's central claim is that factorizing memory into structural codes (via
path integration) and sensory codes enables superior generalization compared to
standard architectures. However, the experimental evaluation lacks a critical
ablation: a direct associative memory baseline that learns state-transition
pairs (state_t, action_t) → state_{t+1} using the same meta-MLP architecture,
warm-up procedure, and auxiliary losses, but without structural factorization.
The current baselines (Transformer, Titans) differ from mm-TEM in multiple
confounded ways -- architecture, training procedures, and task formulation --
making it impossible to isolate whether the performance gains stem from the
factorized representation itself or simply from having a meta-trained
associative memory.  The ablations in Fig. 3C only remove auxiliary losses, not
the core architectural choice of factorization. Furthermore, the claim of
"efficiency" lacks rigor: no computational complexity analysis, wall-clock time
comparison, or memory footprint evaluation is provided, only gradient step counts
against the original TEM. The comparison is further confounded by the warm-up
pre-training phase whose application to baselines is unclear. Finally, in small
grid worlds (8×8 to 11×11), a 64-step context likely covers a substantial
fraction of reachable states, meaning the model may largely be performing
within-episode memory retrieval rather than true generalization to novel spatial
configurations. A proper ablation study isolating the contribution of structural
factorization is essential to validate the paper's core hypothesis.

## Insufficient Mechanistic Explanation for Grid Scale vs. Update Frequency Relationship
The paper claims that the memory update frequency hyperparameter mb controls
grid scale through an "effective prediction horizon" mechanism (lines 252-256),
positioning this as a novel insight into grid-scale diversity. However, this
explanation lacks mechanistic rigor and conflates multiple confounded factors.
First, mb affects both training dynamics (how often gradients update the
meta-MLP weights) and inference behavior (how stale the memory becomes), but the
paper does not disentangle which factor drives the grid scale effect. The
observed correlation could simply be an artifact of optimization
dynamics. That is, sparser gradient updates naturally produce temporally smoother,
lower, frequency representations through gradient accumulation, rather than
reflecting a meaningful "prediction horizon." Second, the feedback mechanism
from relational memory to path integration (Appendix A.2) is central to
understanding this effect but is incompletely described: the function f_delta is
not defined, the strength of feedback (α) is not analyzed, and how mb interacts
with this feedback is unclear. Third, no ablation studies isolate the causal
mechanism. For instance, training with mb=1 but testing with mb=8, or analyzing
gradient flow as a function of mb. Without this mechanistic analysis, the claim
that mb reveals insights into biological grid-scale diversity through
"multi-timescale predictions" remains speculative correlation rather than
demonstrated causation. The authors should provide: (1) ablations separating
training-time vs. test-time effects of mb, (2) gradient flow analysis explaining
why different update frequencies produce different spatial frequencies, and (3)
complete mathematical description of the feedback mechanism.

## Missing Citation and Overclaimed Novelty on Grid Scale Mechanisms
The paper claims to reveal "a novel mechanism for grid-scale diversity in
MEC...as a natural consequence of multi-timescale predictions in the brain"
(lines 252-256), based on their observation that the memory update frequency mb
affects grid scale. However, this is not novel -- Waniek (2020, "Transition
Scale-Spaces: A Computational Theory for the Discretized Entorhinal Cortex")
analytically derived that grid scales emerge from different prediction horizons,
with the spatial frequency directly related to the temporal prediction distance.
The current paper essentially re-discovers this relationship empirically without
citing this prior work. Moreover, Waniek's analysis is rigorous in the sense of
an analytical derivation, compared to the current paper's informal "effective
prediction horizon" argument. Related work by Stachenfeld et al. (2017) and
Dordek et al. (2016) also established connections between temporal prediction
scales and spatial grid scales. The authors should: (1) cite this prior
theoretical work, (2) clarify what is actually novel about their contribution
beyond empirical confirmation in a different architecture, and (3) either remove
the novelty claim or demonstrate what their mechanism adds beyond existing
theory.

## Ambiguity in Generalization Claims Across Sections 3.1 and 3.2
The paper creates confusion about what "generalization" means across different
experiments. In Appendix A.1 (lines 600-602), the authors state that for "all 2D
grid prediction tasks," evaluation is confined to previously visited positions
because "predicting observations at unvisited locations is not meaningful" given
that observations are uncorrelated discrete IDs. However, in Section 3.2, which
claims to test "generalization" (line 260), this critical constraint is never
restated, leaving readers to infer that it still applies. This matters because
Section 3.2's "multi-step imagination" is framed as testing the model's ability
to "generalize beyond its training horizon" (line 268), but if evaluation is
confined to previously encountered positions, it's actually testing memory
retrieval over longer sequences, not spatial generalization to novel states. The
paper should explicitly clarify for each experiment: (1) whether evaluation
includes unseen positions, (2) what fraction of test positions were visited
during context, and (3) how "accuracy" is computed when positions are/aren't
previously seen. The distinction is crucial: the 3D experiments (Section 3.4)
acknowledge that "unvisited observations can be inferred from nearby spatial
information" due to continuous visual features, but no such acknowledgment is
made for 2D tasks where this fundamental difference in evaluation regimes should
be highlighted upfront in Section 3.2, not buried in the appendix.


## Unjustified Causal Claims from Correlational Data (Section 3.2, Figure 4)
The paper claims that "the presence of strongly grid-like cells is a key driver
for generalization" (lines 346-347) based solely on a correlation between grid
score and prediction accuracy (r=0.647, Fig. 4A). This is a causal claim
unsupported by the evidence. Correlation does not establish causality -- the
relationship could be due to reverse causation (good generalization enables
better grid formation) or a confounding variable (successful learning produces
both regular representations and good generalization). Notably, the paper's own
data weakens the causal claim: Figure 4B shows models with low grid scores
achieving high accuracy through "alternative but still regular neural
representations," suggesting that regularity (not grid-ness specifically) may be
what matters. To establish causality, the authors should conduct interventional
experiments: (1) inject hand-designed grid representations and test if
performance improves, (2) add regularization to suppress grid formation and test
if performance degrades, or (3) explicitly bias learning toward grid patterns
and compare against controls. As written, the claim that grids "facilitate" or
"drive" generalization is speculative. The authors should either provide causal
evidence or rephrase their claims to accurately reflect the correlational nature
of their findings (e.g., "grid scores correlate with generalization performance,
suggesting a potential relationship").


## Questionable Statistical Analysis in Figure 4A
The correlation analysis in Figure 4A raises several statistical concerns.
First, while reporting r=0.647 (p=0.0002), the authors use linear regression
despite evidence of non-linearity: accuracy appears to show a ceiling effect
(~0.95-1.0) and the relationship exhibits heteroscedasticity (variance in
accuracy is much higher at low grid scores than high). This violates key
assumptions of linear regression, making standard errors, confidence intervals,
and p-values unreliable. Second, with only ~20-25 data points visible, the
analysis is sensitive to outliers and the wide confidence interval suggests
substantial uncertainty. Third, r²≈0.42 means 58% of variance in accuracy
remains unexplained, yet the authors make strong causal claims ("key driver")
from this weak-to-moderate correlation. Fourth, the scatter plot shows several
counterexamples to the claimed relationship: models with grid scores around
0.7-0.9 achieve accuracy >0.9, while models with grid scores ~1.0-1.1 have
accuracy ~0.8. These observations, also noted in the text regarding "low grid
scores still achieve high accuracy" (Fig 4B), directly contradict the linear
relationship implied by the regression line. The authors should: (1) test for
non-linear relationships (threshold models, saturation functions), (2) report
non-parametric correlations (Spearman's ρ) that don't assume linearity, (3) show
residual plots and test assumptions, (4) report prediction intervals to
demonstrate the large uncertainty in predictions, and (5) acknowledge that grid
score alone is a poor predictor of performance. The statistical evidence does
not support the strong causal claims made in the text.

## Undefined Error Metric and Inconsistent Terminology in 3D Experiments (Section 3.4, Table 1)
The 3D environment evaluation suffers from unclear methodology and inconsistent
reporting. Table 1 reports "prediction error in units of 1e-3" but never defines
what error metric is used. Presumably MSE between predicted and ground truth
egocentric images, but this is not stated. Are errors computed in pixel space,
normalized space, or feature space? How are images preprocessed? The text
compounds confusion by referring to "accuracy" (lines 432-435) while the table
shows "error". These are opposite metrics (higher accuracy vs. lower error is
better). The classification of frames as "visible" vs "not visible" is
undefined: what determines if a frame is considered previously seen in
continuous 3D space with egocentric views? The reported standard deviations
raise questions: one-step errors show std=0.00 across 3 seeds for all models
(presumably due to rounding, but this should be clarified), while Hippoformer's
multi-step std (0.04) is 100× smaller than baselines (4-5). Why is Hippoformer so
much more stable? Finally, without baseline comparisons (chance level, naive
predictors) or interpretation of error magnitudes (e.g., "0.001 corresponds to X
per-pixel deviation"), the numbers lack context. The authors should: (1)
explicitly define the error metric and computation procedure, (2) use consistent
terminology (error or accuracy, not both), (3) specify the visible/not-visible
classification criterion, (4) explain the variance patterns across models, and
(5) provide baselines and interpretation to make the magnitudes meaningful.


## Discussion Section Overclaims Novelty and Omits Critical Limitations
The Discussion overclaims novelty and omits acknowledgment of significant
limitations identified throughout the paper. First, the claim that mm-TEM offers
"a new functional perspective on grid diversity" (line 445) ignores Waniek
(2020), who analytically derived that grid scales emerge from prediction
distances - the very relationship mm-TEM rediscovers empirically. Second, the
Related Work section omits foundational hippocampal-entorhinal literature
(Eichenbaum et al., 2007; Hasselmo et al., 2002; Stachenfeld et al., 2017;
Franzius et al., 2007; Bush et al., 2015) that established factorized memory
architectures and prediction-timescale-to-spatial-scale relationships decades
before TEM. Third, while the authors acknowledge limited integration and
single-layer design, they fail to address major methodological limitations: (1)
2D evaluation confined to previously visited positions (Appendix A.1), meaning
"generalization" is actually memory retrieval, not spatial inference to novel
states; (2) small environments (8×8 to 11×11) where 64-step context covers
substantial state space; (3) no ablation isolating the contribution of
factorization versus direct associative memory; (4) causal claims about
grid-generalization relationship based solely on correlation (r=0.647); (5)
undefined error metrics and inconsistent terminology in 3D experiments. Fourth,
efficiency and scalability claims lack rigor: no computational complexity
analysis, wall-clock time comparisons, or large-scale demonstrations are
provided. The Discussion should thus (1) cite prior work and clarify what is
novel about the grid-scale finding beyond empirical confirmation, (2)
acknowledge foundational neuroscience literature on factorized memory, (3)
explicitly discuss the limitation that 2D "generalization" is constrained to
previously visited positions (or clarify the text) (4) acknowledge that
grid-generalization causality remains unproven, (5) provide concrete criteria
for when mm-TEM/Hippoformer would be preferred over standard architectures, and
(6) temper claims about efficiency and scalability until rigorous evidence is
provided.

### Questions
1. Can you provide a direct ablation comparing mm-TEM against a flat associative memory (no factorization) with identical meta-MLP architecture, warm-up procedure, and auxiliary losses?
2. Can you clarify the evaluation protocol for 2D tasks in Section 3.2? What percentage of positions in the "imagination" phase were previously visited during context?
3. For the mb parameter effect on grid scales: Can you ablate training vs. test-time effects? (Train with mb=1, test with mb=8 and vice versa?)
4. For Figure 4A: Can you provide non-parametric correlation measures (Spearman's ρ), test for non-linear relationships, and show residual plots?
5. For Table 1: What specific error metric is used? How are "visible" vs. "not visible" frames classified in continuous 3D space?
6. Can you provide wall-clock time comparisons and computational complexity analysis to support efficiency claims?
7. How does performance scale to truly large environments where 64-step context covers <10% of reachable states?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work leverages recent formulations of memory from the Titan framework to improve models of the HPC-EC system. This in turn 1) improves the memory efficiency of the HPC-EC models over long contexts, 2) yields insights into grid scaling, 3) leads to an architecture that performs better than standard Titan and Transformer architectures. The new memory structure is an MLP which is trained in-context to map from keys to values, in contrast to older models based on Hebbian and more recently softmax attention.

### Strengths
This work is a significant and novel combination of recent ideas in neuroscience and machine learning. The improvement of tem-t using memory formulations from the recent Titans work is innovative and yields a fun result relating to grid scale. A hybrid architecture is proposed that blends the strengths of the titans-inspired model and transformer; however I would prefer it if the differential contributions of 1) the path-integration input (an old idea) and 2) the meta-MLP relational memory (new idea) were stated more clearly when comparing between models. The architecture has genuine promise to improve upon existing sequence models and I would be fascinated to see it deployed on language problems. 

The figures are all clear and the text is very well written.

### Weaknesses
Barely a weakness & perhaps more to do with framing, but my understanding is that both Transformer and Titan control models you implemented do not have a recurrent path-integrator. Therefore the improvements demonstrated over each are primarily due to the path-integrator token input that the model receives. Tem-t also has this advantage and so should also perform similarly well, which should be mentioned. It would be nice to see more comparisons between tem-t and mm-tem. The use of the meta-MLP memory instead of softmax attention is elegant, I would be curious to see the gains explored slightly more. Perhaps MLP is more brittle than softmax when faced with a novel but semantically familiar input? Or perhaps MLP imbues memory retrieval with generalisation capacity since it is itself a NN? And does this impact the learnt g representations of the outer-loop meta network? Maybe the MLP allows more interesting operations to be done on memories, e.g. contextual splicing of memories - if memorised spatial envs A and B and then encounter env C in which half the stimuli are from A states and half from B

### Questions
Typos:
- 188, 194, 256, 735, 736

Do we have a curve for fig2B tem-t?

Can you clarify in main text that the transformer component of Hippoformer does not receive the path-integration input

Some questions about mb:
- What is the intuition for larger mb increasing the learnt grid size? A fun result that I don't fully understand! 
- Does mb = 1 really perform that poorly - especially since there is no noise in the observations? If so then it would be nice to see this somewhere (since this is essentially the basis of the argument for including transformer in Hippoformer)
- Does smaller mb lead to better performance (assuming observations aren't noisy)? - If I understand correctly, a small mb ~ 1 is more similar to a transformer; would a network endowed with multiple memory MLPs each with different mb yield a similar performance to Hippoformer & also get grids at different scales?
- Linked to question above, could mb relate to oscillation frequency in HPC? I believe there is a dorsal-ventral gradient of oscillation frequencies in hippocampus (matches nicely with the gradient of grid scales). Also wonder if there are interesting findings relating to mb & the discretisation of grid scales that is observed.

Are there biological analogies to the hippoformer? 

I've always been curious to see what the path-integrator module adds when applied to tasks that are less obviously cognitive map-like. Have you tried this model on text-based tasks?

Are there relations between the Titans meta-MLP and working memory (in contrast to transformer softmax which seems more episodic memory-like)? If so, maybe some kind of systems consolidation inspired ideas might apply here i.e. selectively exporting things out of hippocampal memory into neocortical memory

Line 453 - should transformers and mm-TEM be the other way round?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Hippoformer, a hybrid architecture combining a novel hippocampus-inspired memory module, mm-TEM, with a Transformer. The work aims to address the lack of inherent spatial priors in standard Transformers. The central contribution lies in mm-TEM, a scalable variant of the Tolman–Eichenbaum Machine that employs a meta-MLP for relational memory, closely resembling the long-term memory module used in Titan. The authors conduct experiments in 2D and 3D environments, reporting that Hippoformer achieves superior generalization on long-horizon spatial prediction tasks compared to Transformer and Titan baselines.

While the paper is well-written and tackles an important problem, I have major concerns regarding the experimental methodology and the clarity of the architectural contributions. The central claims about the superiority of the proposed relational memory are not sufficiently supported because the baseline comparisons appear to be confounded by critical differences in how positional information is handled.

### Strengths
* **Strong Motivation**: The paper is motivated by a clear and significant limitation of current generative models—their lack of structured spatial memory. The inspiration drawn from the hippocampal-entorhinal system provides a principled foundation for the architectural design.
* **Interesting Component Design**: The mm-TEM module, with its meta-MLP and auxiliary relational losses, is an interesting and efficient take on prior hippocampus models. The analysis showing the emergence of grid-like representations is a valuable piece of evidence supporting the design.

### Weaknesses
The paper's central claims hinge on the superior performance of mm-TEM and Hippoformer over strong baselines, particularly in length generalization. However, the experimental setup is insufficiently described and may contain confounding variables that invalidate these conclusions.

1.  **Lack of Clarity on Positional Encoding in Baselines:** The most critical issue is the ambiguity surrounding the positional encoding (PE) used for the Transformer and Titan baselines. The mm-TEM model relies on a recurrently updated structural code, `g_t`, from a Path Integration Network, which effectively serves as a powerful, dynamic, and task-specific form of positional encoding. The paper does not specify whether the baselines have access to this same structural code.
    *   If the baselines use standard, fixed positional encodings (e.g., sinusoidal) or no PE at all, the comparison is fundamentally flawed. The performance gains of Hippoformer could stem entirely from its superior positional information, not its relational memory. As shown by Kazemnejad et al. (NeurIPS 2023, "The Impact of Positional Encoding on Length Generalization in Transformers"), the choice of PE is a dominant factor in a Transformer's ability to generalize to longer sequences. To isolate the contribution of the relational memory, the baselines must be equipped with a similarly powerful and dynamic PE.
    *   This lack of clarity makes it impossible to attribute the performance gains to the claimed source (the relational memory) versus a known, powerful factor (the positional encoding scheme).

2.  **Unclear Architectural Novelty Compared to Titans:** The paper presents Hippoformer as a combination of a Transformer and the mm-TEM module (which is centered around a meta-MLP memory). The Titan architecture is also described as a model leveraging fast MLP weights. From the descriptions provided, the high-level architectural blueprint of Hippoformer appears very similar to that of Titans. The paper needs to explicitly detail the architectural and mechanistic differences. Is the primary novelty of Hippoformer simply the introduction of the auxiliary relational loss as a form of inductive bias for the meta-MLP? If so, the contribution should be framed more narrowly as a novel training objective for existing hybrid architectures on spatial tasks, rather than a fundamentally new architecture.

3.  **Ambiguity in the Training of the Recurrent Module:** The mm-TEM module contains a recurrent update for the structural code `g_t`. This introduces dependencies across the entire sequence. The paper lacks crucial details about how this recurrence is handled during training. Is backpropagation through time (BPTT) performed over the full sequence length? Or is it truncated? This detail has significant implications for the model's computational cost, memory requirements, and its practical ability to capture the long-range dependencies it is being credited for.

### Questions
**1. Clarifications on Baseline Models (Crucial for Rebuttal):**
This is the most important area. A clear response here could significantly change my assessment.

*   **Question 1a (Positional Encoding):** How was positional/structural information provided to the Transformer and Titan baseline models? Specifically, did they take as input only `[s_t, a_t]`, or did they also receive the structural code `g_t` from the Path Integration Network, similar to mm-TEM?
*   **Question 1b (Type of PE):** If the baselines did *not* use the Path Integration Network, what form of positional encoding was used (e.g., sinusoidal, learned, rotary, or none)?

**2. Architectural and Contribution Framing:**

*   **Question 2 (Hippoformer vs. Titans):** Could you please provide a more detailed, side-by-side comparison of the Hippoformer and Titan architectures? What are the key differences in their memory update rules, the interaction between the MLP-based memory and the self-attention component, and the flow of information? A diagram or table would be very helpful.

**3. Implementation Details for Reproducibility and Analysis:**

*   **Question 3 (Training Recurrent Module):** How is the gradient calculated for the recurrent mm-TEM module? Is backpropagation through time (BPTT) applied over the full sequence length (e.g., 256 steps in Fig. 2), or is it truncated to a smaller window? What are the implications of this choice for computational complexity and memory usage during training?


**Suggestions**
* If the Fig. 3A experiment corresponds to the length generalization task and does not use the Path Integrator Network for g_t or employ g_t as a positional encoding, the authors should evaluate multiple positional encoding methods and compare their results to those of mm-TEM.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the limitations of prior hippocampal-inspired models (TEM's computational inefficiency) and modern architectures (Titans' lack of inherent spatial memory). By synthesizing these approaches, the authors propose **mm-TEM** (meta-MLP TEM), demonstrating better training efficiency and revealing a novel link between the memory update frequency and the emergence of biologically meaningful grid-like representations. Furthermore, they introduce **Hippoformer** (mm-TEM + Transformer), a hybrid architecture that effectively integrates structured long-term spatial memory with precise short-term working memory, achieving robust generalization across demanding 2D and 3D prediction tasks.

### Strengths
*   **Clarity and Organization:** The paper is well-structured with clear conceptual figures (e.g., Figure 1), making the overall architecture and training rationale highly intuitive.
*   **Compelling Rationale:** The core rationale for the proposed method—integrating the computational efficiency of the meta-MLP memory (inspired by Titans) into the theoretically grounded TEM framework—is valid for overcoming scalability issues while retaining biological plausibility.
*   **Strong Experimental Validation:** The systematic evaluation, including ablations and generalization tests across long context, multi-step imagination, and distribution shifts (circular-grid), robustly supports the claim that mm-TEM captures underlying spatial structure more faithfully than baseline models.

### Weaknesses
**Literature Review Suggestion (Line 40):** The Introduction's discussion of the Transformer's associative memory perspective would be strengthened by citing recent, relevant works that formalize the Transformer's components (e.g., FFNs) as explicit memory systems, such as:
- Geva, Mor, et al. (2020) on FFNs as key-value memories.
- Ramsauer, Hubert, et al. (2020) on Hopfield networks' relation to attention.

**Relational Loss Notation and Rationale:**
- **Notation Clarity:** The auxiliary relational losses (Lines 184-190) require precise clarification. Please map the main text's definitions\ 
$L_1$ and $L_2$ to the figure notations of $L_{x2g}$, $L_{g2x}$, $L_{g2g}$ and clarify the missing one.

- **Missing Term Rationale:** The authors can define four potential combinations ($g \to x$, $x \to g$, $g \to g$, $x \to x$). But figure/main text uses 2/3 losses. What is the underlying reason for excluding the other possible terms from the relational losses?

2.  **Path Integration Network Ablation and Role:**
    *   The core TEM theory mandates a Path Integration (PI) network for structural code generation. Since the authors emphasize the **novel meta-MLP relational memory** as the main source of generalization, could the authors include an **ablation study on the PI network component itself** (e.g., replacing it with a simpler, non-integrated recurrent mechanism or removing the error correction loop)? This would help quantitatively disentangle the contribution of the *novel memory* from the *PI component*.
    *   Given the PI network's role in generating the structural code ($g_t$) from actions, could this component be viewed as an advanced form of **learned positional encoding**?

3.  **Grid Pattern Neuron Selection:**
    *   For the analysis of emergent grid-like representations (Figure 2C, Figure 4B), the authors show visualizations of the high-gridness neurons. Could the authors explicitly state **from which module** (the Path Integration Network or the Relational Memory Network) the "Top-5/Top-3" high-gridness neurons were selected? Clarification on the specific origin of these analyzed neurons is crucial for interpreting the results.

4.  **Table 1 Completeness (3D Task):**
    *   Table 1 presents results for Transformer, Titans, and Hippoformer in the 3D environment. Given that **mm-TEM** is the core component providing the long-term generalization in Hippoformer, why are the performance results for the **mm-TEM** model alone **omitted**? (Figure 5 shows on par imagination capability between mm-TEM and hippoformer) Including mm-TEM's 3D performance would provide a necessary direct measure of the Transformer's specific contribution (abstraction) to the final Hippoformer architecture in this complex domain, thereby strengthening the claim of synergy.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
