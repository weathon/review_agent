# Bilinear representation mitigates reversal curse and enables consistent model editing

- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
The reversal curse—a language model's inability to infer an unseen fact "B is A" from a learned fact "A is B"—is widely considered a fundamental limitation. We show that this is not an inherent failure but an artifact of how models encode knowledge. Our results demonstrate that training from scratch on synthetic relational knowledge graphs leads to the emergence of a bilinear relational structure within the models' hidden representations. This structure alleviates the reversal curse and facilitates inference of unseen reverse facts. Crucially, this bilinear geometry is foundational for consistent model editing: updates to a single fact propagate correctly to its reverse and logically dependent relations. In contrast, models lacking this representation suffer from the reversal curse and fail to generalize model edits, leading to logical inconsistencies. Our results establish that training on a relational knowledge dataset induces the emergence of bilinear internal representations, which in turn support language models in behaving in a logically consistent manner after editing. This suggests that the efficacy of language model editing depends not only on the choice of algorithm but on the underlying representational geometry of the knowledge itself.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that transformer LMs trained on a synthetic relational knowledge graph can develop an internal bilinear relational structure that both overcomes the reversal curse and enables logically consistent edit propagation. The authors train 12-layer, ~206M parameter decoder-only transformers from scratch on a synthetic kinship-style corpus, sweeping weight decay to obtain both "reversal cursed" and "not reversal cursed" regimes. They then probe layerwise hidden states using three hypothesized relational geometries: linear, translational, and bilinear. They find that only a bilinear probe of the form $f_r(s,o) = s^\top M_r o$ achieves near-perfect relation classification in the middle layers (layers 6-9) of models that circumvent the reversal curse, while linear and translational probes remain near chance. The learned relation matrices also pass algebraic tests: their transposes approximate inverse relations and their products approximate two-hop compositions. Finally, the authors perform single-fact edits via layer-restricted fine-tuning and report that edits to earlier layers (1-4) propagate consistently to logically entailed inverse/neighboring facts while mostly preserving unrelated facts. They argue that this generalization correlates with the emergence of bilinear structure.

### Strengths
1. The clear, controlled synthetic setup effectively isolates inverse and compositional structure. The weight-decay sweep is particularly informative for understanding when the reversal curse emerges.
2. The probing methodology is systematic across three geometric families with an interpretable chance baseline and compelling mid-layer bilinear peaks. The experimental design is solid.
3. The algebraic validation is convincing (transpose $\approx$ inverse; product $\approx$ composition) and makes the representational claim very interpretable.
4. The layer-restricted editing experiments provide a concrete story about how edits can propagate to logically entailed facts while preserving unrelated facts.

### Weaknesses
1. The causal claims may not be sufficiently supported by interventions. To my knowledge, all evidence that bilinear structure "fixes" or enables reversal-curse mitigation comes from readout probes and offline algebra, not from interventions demonstrating that the model's forward computation actually depends on the probed subspace.
    * Necessity can be shown with projection ablations at a bilinear-peak layer. For example, after removing top singular subspaces of $M_r$ during inference, one could measure the drop in reversal/composition accuracy.
    * Sufficiency tests could involve adding a low-rank adapter constrained to a bilinear relational subspace. This subspace could be constructed either by: (i) aligning the cursed model's layer representation to a non-cursed model's corresponding layer (via a learned linear map) and importing that model's bilinear directions, or (ii) fitting bilinear relation matrices in the cursed model itself and taking their top singular vectors. One could then ask whether reversal/composition accuracy and edit generalization improve toward the non-cursed regime.
    * Without such controls, the evidence shows correlation but may not necessarily show causation.

2. The authors edit facts by fine-tuning only the MLP output of one chosen layer and observe that edits in earlier layers generalize better (i.e., doing so updates inverse/related facts more reliably). However, the authors do not check what happens to the bilinear structure after those edits.
    * As a result, it is unclear whether (i) early layers are special because they are where the relational structure is actually being built, or (ii) edits in early layers simply spread more broadly because they are upstream in the network, regardless of any bilinear mechanism.
    * One way to separate these would be to apply edits of the same norm that are either aligned with the layer's existing gradient direction or forced to be orthogonal to it, at different depths, and then re-probe $M_r$ afterwards. If bilinear structure (transpose/composition) is still intact and explains generalization, the result would support (i); if not, the result suggests (ii).

3. External validity is limited because the dataset is fully synthetic, the models are small, and no results on natural relations in pretrained language models are provided.
    * To be completely fair, such experiments are arguably out-of-scope for this paper and could be better left to future work.

4. The paper does not sufficiently engage with recent knowledge editing work that studies transformers trained on synthetic knowledge graphs and analyzes how their internal relational structure changes under editing.
    * Several recent studies train small transformers from scratch on graph-structured relational data (very similar to the kinship-style setup here), then apply targeted weight edits and observe that naive edits can disrupt relational representations unless constraints are applied.
    * I encourage the authors to further explore and acknowledge this line of work and situate this work in the broader landscape.

### Questions
1. Is it viable to run projection ablations that remove the principal subspace of $M_r$ at a bilinear-peak layer to test necessity, and add a low-rank bilinear adapter in cursed models to test sufficiency?
2. Similarly, would it be possible to repeat editing with gradient-aligned versus orthogonal updates across layers, and re-probe $M_r$ to verify whether transpose/product structure survives the edit?
3. Could you please add a short discussion addressing how one could extend these experiments and insights to real-world models beyond your synthetic setup, to help guide future work?
4. Could you please provide more thorough coverage of the related literature on knowledge editing in transformers, especially studies involving synthetic knowledge graphs?

I would be happy to raise my score if the above points are addressed with new experiments, or if the authors can provide convincing reasons for why they the existing evidence is sufficient.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates how Transformer-based language models represent relational knowledge, focusing on the “reversal curse”: the inability of models to infer “B is A” from “A is B.” The authors study this in a controlled synthetic setting by training Transformers from scratch on relational data describing family members (10k entities, 8 relations: _mother, husband, son, brother_, etc.). For a subset of families, _father_ and _mother_ relations are withheld during training, enabling the authors to test generalization to these unseen relations.
They analyze which models can generalize these withheld relations and how their internal representations differ. The main findings are:
- **Weight decay** critically determines whether models generalize or merely memorize. Higher weight decay leads to more models that generalize correctly.
- By probing internal representations, they evaluate three relational geometries—linear (projection matrix + bias), translational (additive bias), and bilinear (matrix-mediated interaction). They find that:
    1. These relational structures are more prominent in models that generalize.
    2. The **bilinear** structure most successfully explains model behavior.
- Finally, models that develop a bilinear structure also exhibit consistent **model editing** behavior: editing a single fact propagates correctly to logically entailed facts. Models lacking this structure fail to generalize edits.

### Strengths
This is a high-quality, well-motivated paper that offers new insights into how relational knowledge and logical consistency emerge in Transformer representations. While limited to a synthetic domain, the work’s conceptual clarity and systematic experimentation make it a valuable contribution to understanding the internal geometry of knowledge in LMs.

- The paper is clearly written and well-organized, making it accessible and easy to follow.
- The experimental design is sound and thoughtfully constructed, allowing for careful isolation of the phenomenon under study.
- The contributions are conceptually substantial and novel, offering meaningful insight into the representational geometry underlying relational reasoning in Transformers.
- The work provides a mechanistic link between regularization, representational structure, and model editing behavior.

### Weaknesses
- The study relies entirely on a toy synthetic setup, which limits the direct applicability of findings to real-world pretrained LLMs. Though the authors acknowledge this in the discussion.
- The connection between bilinear structure and fact editing may be correlational rather than causal. Other factors (e.g., differences in data fitting or feature organization) could also drive the observed editing generalization.
- The finding that weight decay promotes generalization may reflect the limited dataset size rather than a broader principle. Without sufficient regularization, memorization is expected for the size of the data, so further analysis (e.g., varying dataset size or noise) would strengthen the claim.

### Questions
- Do all models fit the training data equally well? It would be helpful to report training accuracy to confirm that differences arise from generalization rather than underfitting.
- How exactly is decoding performed to compute accuracy in Figure 2?
- What do the gray points in Figure 2 represent?
- This work seems related to studies on entity mapping in attention heads (e.g., _“Inferring Functionality of Attention Heads from their Parameters”_). It might be worth referencing this connection.
- The linear and bilinear probes differ not only in structure but also in training data size (10 examples vs. 1250). Could the superior performance of the bilinear probe partly reflect this discrepancy? It would help to discuss whether a larger dataset could make the linear probe competitive.

### Soundness
3

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
4

### Summary
This paper studies toy models to propose a hypothesis for why phenomena like reversal curse are observed: specifically, the paper argues for the (in)existence of the underlying structure (e.g., a graph) that binds the knowledge together when a model fails/succeeds to generalize to novel facts or combinations thereof (e.g., A->B vs. B->A). Broadly, the argument is that the existence of a structure allows a fact's implications to be inferred by the model, enabling generalization.

### Strengths
While the studied toy model and the derived claims can be (partially) found in recent literature, the insight of measuring how bilinear the representation is novel and I enjoyed thinking about that. The experiments have a solid outline to drive the main argument, which was fun to read.

### Weaknesses
I have three major apprehensions about the paper's results and narrative, as detailed below.

- Relation to prior work: Nishi et al. [1] perform a similar study as this work, wherein a synthetic knowledge graph is defined, a model is trained from scratch to learn it, and it is demonstrated that when the model learns the underlying graph topology (circles instead of hierarchies, as explored in this work), editing its knowledge via weight-targeting methods (e.g., ROME) can break the underlying graph structure---called "representation shattering" in [1]. The extent of representation shattering, i.e., how structured the representation is, correlates with success on multi-hop factual inferences---the conclusion also arrived upon in this paper. While I understand the current paper's analyzed methodology for updating the model is fine-tuning, I note that I'm highlighting [1] because (a) there's a broad similarity in the motivating problem and setup that [1] warrants contextualization in the current paper, and (b) the conclusions of [1] suggest editing can be quite brittle if the model has learned the underlying structure of the knowledge graph, which, even if it is explained by the use of a different editing method, warrants discussion to narrow down the current paper's conclusion. Finally, I'll also note that [1] verifies its conclusions with real models, using the fact that knowledge graphs are often stored with a precise structure in representations (e.g., days of the week [2] or the world map [3]). On a related note, I'll also add another relevant paper that performs a similar study as the current paper [4].
  - To be clear, I am not suggesting the current paper does not possess novelty; e.g., I really enjoyed the idea of using the bilinear probe as a measure to gauge how well structured the representation is. In [1], a similar experiment was performed via isomap projections, which implicitly involves the assessment and visualization of how well model representation similarities map to the ground truth adjacency matrix of the graph. However, I prefer the approach taking in this work and hence consider the work useful. Nevertheless, the writing often argues that the paper is the first to attempt a study of this sorts or arrive at conclusions shown in the paper, and for that I hope authors can rework the paper writing.

- Non-toy model experiments: As noted above, there are several precise structured representations in pretrained language models such that the conclusions drawn in this work should be easily testable on scale (again, I'd refer to [1], since they did those experiments). I would have loved for an attempt at this verification and deem this a strong limitation of the work right now. To be clear, I'm all for using toy models to derive hypotheses, but I expect then an evaluation of the hypothesis's prediction be conducted when it is feasible, which, in this case, is.

- Narrative problems: Finally, I'll note that the paper begins with discussion of model editing and reversal curse. The precise experimental regime studied, however, is one of pretraining. Specifically, the knowledge is being embedded into the model during pretraining and, during fine-tuning, some new facts may be added to test whether generalization occurs in expected directions. This is different from the precise reversal curse setting, as I understand it, and also for editing.

[1] https://arxiv.org/abs/2410.17194
[2] https://arxiv.org/abs/2405.14860
[3] https://arxiv.org/abs/2310.02207
[4] https://arxiv.org/abs/2503.21676

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
