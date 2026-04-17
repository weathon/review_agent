# Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Spatio-temporal reasoning is a remarkable capability of Vision Language Models (VLMs), but the underlying mechanisms of such abilities remain largely opaque. We postulate that visual/geometrical and textual representations of spatial structure must be combined at some point in VLM computations. We search for such confluence, and ask whether the identified representation can causally explain aspects of input-output model behavior through a linear model. We show empirically that VLMs encode object locations by linearly binding \textit{spatial IDs to textual activations, then perform reasoning via language tokens. Through rigorous causal interventions we demonstrate that these IDs, which are ubiquitous across the model, can systematically mediate model beliefs at intermediate VLM layers. Additionally, we find that spatial IDs serve as a diagnostic tool for identifying limitations and bottlenecks in existing VLMs. We extend our analysis to video VLMs and identify an analogous linear temporal ID mechanism. By characterizing our proposed spatiotemporal ID mechanism, we elucidate a previously underexplored internal reasoning process in VLMs, toward improved interpretability and the principled design of more aligned and capable models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a linear ID mechanism for spatiotemporal reasoning in VLMs. At intermediate layers, the model encodes an object’s spatial or temporal position as an approximately linear Spatial or Temporal ID within the activation of the corresponding object token, after which most reasoning proceeds in the language channel. The authors also provide an empirical procedure for extracting Spatial IDs. Targeted interventions on these IDs systematically alter the model’s judgments about spatial terms such as left and right, and enabling diagnosis of failure points across architectures, for example vision encoding versus cross modal integration.

### Strengths
1.The work unifies the write–then–reason pathway for spatial and temporal information under a linear ID framework, and introduces actionable intervention and diagnosis paradigms, including mirror swapping, arbitrary ID steering, and error attribution for failure cases.

2. The paper reports consistent phenomena and flip rates across 11 VLMs, with noise controls indicating that the effect is not an idiosyncrasy of any single model.

3. The same IDs both explain observed behavior and localize bottlenecks, separating issues in vision encoding, cross-modal binding, and language-stage reasoning, thereby offering concrete guidance for model engineering.

4. The study reproduces the same linearly steerable structure in video models as Temporal IDs, aligning interventions with before and after relations and suggesting a more general mechanism.

### Weaknesses
1. The Mirror Swapping design localizes key bottlenecks and supports the claim that Spatial IDs are bound to object tokens by replacing only the object tokens. However, it omits a crucial control condition that would replace non-object tokens under an otherwise identical setup, which is necessary to more directly test whether the observed effects indeed hinge on object token binding rather than on broader sequence perturbations.

2. The causal attribution in §4.2 across the stages of vision encoding, cross modal binding, and language readout is not fully convincing. The inference that Spatial ID errors imply the language stage is not the bottleneck lacks necessity level causal testing; in other words, a deviation in the Spatial ID does not rule out the possibility that the language decoder exists erroneous priors during readout, a phenomenon that has been reported as language side priors or language bias in VQA tasks[1][2]. In addition, using sensitivity to occluding the ground truth box to distinguish vision encoding from cross modal binding conflates misidentification or localization failure with miswriting into the object token, making it difficult to uniquely localize the failing component. Overall, the current Spatial ID instrumentation does not cleanly isolate the faulty segment of the architecture.

3. The paper largely remains at the level of revealing and validating the Spatial ID mechanism, without translating this mechanism into concrete corrective or editing procedures for existing models and without empirical evaluation of such interventions in a remediation setting.

[1] Ramakrishnan, Sainandan, Aishwarya Agrawal, and Stefan Lee. "Overcoming language priors in visual question answering with adversarial regularization." Advances in neural information processing systems 31 (2018).
[2] Leng, Sicong, et al. "Mitigating object hallucinations in large vision-language models through visual contrastive decoding." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates how and where VLMs combine spatial visual information with textual representations to facilitate capabilities such as spatiotemporal reasoning.  The authors derive and identify spatial IDs, which encode object locations in an $m\times m$ grid.  Spatial IDs are then used for interventions such as steering, as well as a variety of insightful analyses such as the lack of depth representation in VLMs.  The paper also investigates Temporal IDs, rounding out their investigation of spatiotemporal reasoning in VLMS.

### Strengths
The strengths of the work include its discovery and derivation of spatial IDs, and its thorough utilization for a variety of insights.  The paper defended spatial IDs thoroughly, through adversarial steering as well as how deviations from the ground truth spatial ID results in worse predictions.

### Weaknesses
Whereas the Spatial ID was thoroughly explored, much less attention was put on Temporal IDs.  Furthermore, the "before" and "after" evaluation is a bit more simplistic than spatial reasoning.  One expects a smoother interpolation; but in Figure 9(C) for 'After', both changes in logprob decrease past frame 5.

Furthermore, the queries are quite limited to simple position-based analysis.  General reasoning should utilize other attributes beyond position, such as properties of the model (e.g. an ice cream in sunlight should most likely melt - reasoning about this change over time requires less positional information).  For such settings, it is not clear how the current approach can provide insight into how the model is performing its reasoning.  It would be interesting to see if object-specific attributes can also be encoded within or alongside spatial IDs for more complete isolation of where the binding with text is performed; or if the authors can provide insight into where such other attributes may be found and where they are bound.

### Questions
For the mean embedding of object $o$, is this computed across a batch of examples of the particular object?  For example is it averaged across many different apple photos (positioned in different ($i, j$) per image)?

Is the $(i, j)$ spatial coordinate for an object just one singular coordinate?  If the object takes up a lot of space in the image you can imagine it encompassing many coordinates in the $m \times m$ grid.  In such a case, is only the centroid of the object used as the spatial position?  What about objects that span many discretized squares where the centroid may not actually be the position of the object (e.g. a bike; the centroid may be empty air)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies that VLMs encode spatial relationships through linear features called "spatial IDs" bound to object token representations. Through extraction and causal intervention experiments, the authors demonstrate that these spatial IDs can be manipulated via representation steering to control model outputs on spatial reasoning tasks. The analysis extends to video models, revealing similar linear temporal ID mechanisms for temporal reasoning.

### Strengths
1. Interesting mechanistic discovery: Identifying spatial IDs as linear features that mediate spatial reasoning in VLMs is novel and insightful, revealing how models bind location information to object tokens for subsequent linguistic processing.
2. Robust experimental validation: The paper provides rigorous evidence through well-controlled experiments (mirror-swapping with controls, causal interventions across 11 models achieving 64.4% vs 29.5% belief swap rates) and validates the framework across multiple datasets and model types.

### Weaknesses
1. Limited spatial relation coverage: The analysis focuses primarily on simple binary spatial queries ("left/right", "up/down"), while more complex and diverse spatial relationships like "near", "far", "between", or "surrounded by" remain unexplored. The generalizability of the linear spatial ID framework to these richer spatial concepts is unclear, limiting the scope of the findings.
2. Insufficient guidance for model improvement: While the paper offers valuable diagnostic insights, it lacks actionable strategies for improving VLM training or architecture. The identified issues (e.g., depth-height conflation in Section 4.1) are not accompanied by proposed solutions or experimental validation of potential fixes. Demonstrating how spatial ID insights can inform better training objectives, architectural modifications, or data curation would significantly enhance the practical impact of this work.

### Questions
What causes the depth-vertical conflation? Section 4.1 identifies depth-height conflation but does not determine whether this stems from training data biases (e.g., perspective projection correlating height with distance), architectural limitations of 2D vision encoders, or the lack of 3D positional encodings. Without understanding the cause, no solutions are proposed or tested. Given that depth reasoning is critical for real-world applications, this represents a significant missed opportunity to translate mechanistic insights into actionable improvements such as 3D spatial IDs, depth-aware training objectives, or multi-view architectures.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies how VLMs do spatial/temporal reasoning by identifying spatial/temporal IDs, which are approximately linear signals written into object-word activations at intermediate layers, and validating them with targeted interventions like mirror swapping and steering. It could also shed light on 3D directions and temporal reasoning in videos.

### Strengths
1. The problem the paper focus on, spatial reasoning of VLM, is clearly defined and important. 
2. The experiments especially in section 2 are clear and well-designed. 
3. The discovered spatial IDs show that models confuse "up/down" with "front/back", which helps explain real errors in 3D cases and gives a useful diagnostic. The temporal ID result is similarly clear and interesting.
4. The discovered spatial IDs exist among different VLMs though different backbones and training procedure.

### Weaknesses
1. The paper explains how models work in spatial reasoning but does not show that using these IDs can improve benchmark accuracy in a training-free manner or help train better models.
2. The authors could test whether stronger or clearer IDs mean better spatial reasoning ability. For example, by checking the correlation between an "ID strength score" or something similar, and model accuracy on a spatial reasoning benchmark. This would make the finding more practical and convincing.

### Questions
Please refer to the weaknesses. I'm glad to raise my score if you cover them well.

### Soundness
4

### Presentation
4

### Contribution
3
