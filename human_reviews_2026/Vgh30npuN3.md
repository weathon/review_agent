# Curvature-Guided Task Synergy for Skeleton based Temporal Action Segmentation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Fine-grained temporal action segmentation plays a vital role in comprehensivehuman behavior understanding, with skeleton-based approaches (STAS) gaining prominence for their privacy and robustness. A core challenge in STAS arises from the conflicting feature requirements of action classification (demanding temporal invariance) and boundary localization (requiring temporal sensitivity). Existing methods typically adopt decoupled pipelines, unfortunately overlooking the inherent semantic complementarity between these sub-tasks, leading to information silos that prevent beneficial cross-task synergies. To address this challenge, we propose CurvSeg, a novel approach that synergizes classification and localization within the STAS domain through a unique geometric curvature guidance mechanism. Our key innovation lies in exploiting curvature properties of well-learned classification representations on skeleton sequences. Specifically, we observe that high curvature within action segments and low curvature at transitions effectively serve as geometric priors for precise boundary detection. CurvSeg establishes a virtuous cycle: localization predictions, guided by these curvature
signals, in turn dynamically refine the classification feature space to organize into a geometry conducive to clearer boundaries. To compute stable curvature signals from potentially noisy skeleton features, we further develop a dual-expert weighting mechanism within a Mixture of Experts framework, providing task-adaptive feature extraction. Comprehensive experiments demonstrate that CurvSeg signif-icantly enhances STAS performance across multiple benchmark datasets, achieving superior results and validating the power of geometric-guided task collaboration for this specific problem.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of conflicting feature requirements in skeleton-based temporal action segmentation (STAS). The key innovation is utilizing the curvature properties of well-learned classification representations on skeleton sequences, with high curvature within action segments and low curvature at transitions serving as geometric priors for boundary detection.

### Strengths
This paper proposes a novel curvature-based approach that leverages the geometric properties of feature sequences to foster effective collaboration between classification and localization sub-tasks.

The introduction of a dual-expert weighting mechanism within a Mixture of Experts (MoE) framework enhances the performance of the synergy mechanism by separately capturing semantic representations for classification and fine-grained temporal details for localization.

### Weaknesses
There are the following issues with Figure 1: (a) and (b) represent different STAS pipelines, while (c) and (d) do not; (c) and (d) are jumbled together without any spacing to distinguish them; and text in this figure is too small.

Line 231, where, θt ∈ [0, π] quantifies. , should be deleted.

Line 146, Task Decoupling in STAS  . is missing. 

There is no analysis of the method's time complexity and runtime. Although a 1.5% improvement is achieved on the PKU-MMD dataset, it would be meaningless if it comes at the cost of increasing the model's runtime.

In temporal action segmentation, are the evaluation metrics Acc, Edit, and F1 reasonable? Why not use evaluation metrics similar to IoU?

### Questions
Although the writing in this paper is relatively standardized and the method exhibits some innovations, the task of skeleton-based temporal action segmentation is limited. 

This paper does not provide me with meaningful insights, and even if it were accepted, its contribution to the field would be quite limited. I suggest that the authors expand their method for more tasks to enhance its generality.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles skeleton-based action understanding by arguing that the geometry of feature sequences over time can be exploited to make classification and temporal localization help each other. The motivation is clear: current methods still show “insufficient cross-task collaboration,” so the authors introduce a curvature-based task synergy mechanism that uses geometric properties of feature trajectories to link both sub-tasks, and they couple it with a dual-expert weighting mechanism in a Mixture-of-Experts setup to adapt features to the task. As the authors themselves note, the approach still struggles with very noisy skeletons and complex multi-person cases, so there is room to improve robustness.

### Strengths
Clear motivation.  The paper starts from an identifiable gap, weak collaboration between classification and localization, and address it directly.

Geometric novelty. using curvature of feature sequences as a bridge between sub-tasks is a fresh angle compared to standard temporal smoothing or boundary refinement.

Task-adaptive extraction: the dual-expert / MoE part makes the framework flexible to different task demands instead of using a single shared feature space.

### Weaknesses
Major 1. Across the method section, several symbols are introduced without being defined first, or they are not mathematically specified in a precise way. For example:

	•	L176: \alpha_i and \alpha_j are not defined.

	•	L182 – F_s is introduced without stating whether it is the same as the feature sequence in the problem formulation (X) or already an encoded representation.

	•	Eq. (4) and L191 – F_{ST} is used without an earlier, explicit definition. From context, it seems to be the spatio-temporal feature output of the Liformer / GCN stage, possibly the same as F_{\text{gcn}}. Please unify the notation and add a one-sentence definition before Eq. (4).

	•	L223 – “frame-wise classification features.” The text seems to refer to the same tensor that in Fig. 2 is denoted by the decoder classification head Y_{cl}. Please make these two references consistent.

	•	L225–239 – windowed triplets. The text says “three consecutive points,” but the notation is F_{(cls, t-w)}, F_{(cls, t)}, F_{(cls, t+w)}. This is only strictly consecutive when w=1. For w>1, it is unclear: whether intermediate points are also used,  whether triplets are processed independently or aggregated, and how the final task representation is formed from them.

Please provide explicit sampling and aggregation equations.

	•	L262 – F_{st} is mentioned again, but it is not clear whether this is the same as F_s, F_{st}, or F_{ST}. The shape is given as V \times T without channels, which is hard to reconcile with the rest of the model.

⸻

Major 2.
In the ablation results, two entries on PKU-MMD (X-view) — the Edit score and F1@10 — should be bold for the CGS-only configuration. I would like to know why this could be happening, because it would somehow break the idea that “EDD provides the high-quality.”


To make this section coherent:
	1.	Bold those two CGS-only numbers.

	2.	Add a short interpretation, e.g. that CGS might be better aligned with cross-view variability in PKU-MMD than the EDD augmentation in that specific setup.

⸻

Major 3.
	•	Figure 2 seems to have two different flow directions: the central pipeline is read from bottom to top, while the side modules are read from top to bottom. This makes it hard to know where the computation actually starts and where it ends.

	•	Variable names are placed on top of the drawings and are hard to read (small font, low contrast, and sometimes overlapping the boxes).

	•	The figure repeats equations that are already in the text — equations (7) and (8) and L239 — which makes it redundant and not visually explanatory.

	•	The gray vertical line in the first column is not explained . if it is an encoder/decoder split, please state that.

	•	In text (L293) you say the feature is divided into M segments, but the figure shows g_1, g_2, g_3, which suggests a fixed number of Gaussians G=3. Please make the figure consistent with the text.

	•	The final step after the Gaussian generator ends in something like F_{m1}^{ST}, but it is not shown how these features are merged back into the task representation — that is precisely what readers will want to see.

⸻

Minor issues

	1.	L218–220: the statement “intra-segment points must frequently change direction to remain within their class-specific boundary, resulting in high curvature, while inter-segment points exhibit low curvature as they move between class regions” — where is this shown to be true? It would be necessary to include a visualization or some evidence where this can actually be seen.

	2.	L143 – “recent advances” needs citations. If you refer to recent advances, please add representative works.

	3.	Some of the dimension strings in the description are of the form “(V \times K) D \times T \times v” or “1 \times I \times K \, D \times D,” which can be a bit confusing regarding how they are multiplied.

	4.     In the hyperparameter analysis section, the parameter w is not discussed.

	5.	Some sentences are not well expressed or well written. For example:
	•	L216: “…feature space. (This observation can be formally proven: the average curvature of a random walk is inversely proportional to the radius of its bounding hyper-sphere. See Appendix B.) This…”
	•	L448: a sentence should not start with a variable.
	•	The last sentence of L454. Please revise in general.

	6.	Figure 4 – confusing “spins” in parts (b) and (c).
In Fig. 4 (b) and (c) there appear to be many sharp peaks / spins in regions where, according to the action annotation, the action does not change. Please explain why the method produces such high-frequency responses in stable segments.

### Questions
1. Notations and definitions. Can you clarify the symbols in the method section (in particular F_s, F_{st}, F_{ST}, F_{\text{gcn}}, and \alpha_i, \alpha_j) and make them consistent with Fig. 2?
2. Figure 2 clarify.
3. Rewrite Ablation on PKU-MMD (X-view): Why does the CGS-only setup outperform the full version for Edit and F1@10?
4. Triplet/window construction: When you say “three consecutive points” but use t-w, t, t+w, what happens for w>1? Are intermediate points used and how are they aggregated into the final task representation?
5. Can you provide a small visualization to support the high-curvature vs low-curvature claim, and explain the extra peaks in Fig. 4 (b)–(c) where the action does not change?
 
For more details, please see the Weakness section above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a core challenge in Skeleton-based Temporal Action Segmentation (STAS): the conflicting feature requirements for its two main sub-tasks, action classification (which needs temporal invariance) and boundary localization (which needs temporal sensitivity). Existing methods typically decouple these tasks, which prevents beneficial cross-task collaboration and creates "information silos". The paper proposes CurvSeg, a novel approach that synergizes these tasks using a geometric curvature guidance mechanism. The key insight is that in a well-learned classification feature space, the trajectory of skeleton frame features exhibits high curvature within an action segment (to stay within its class cluster) but low curvature at transitions when moving between clusters.

### Strengths
1.The paper introduces a novel curvature-based task synergy mechanism (CGS) that effectively exploits the geometric properties of feature sequences. This mechanism establishes a self-reinforcing loop where improved boundary detection and more discriminative classification features mutually enhance one another.

2. The method is validated through comprehensive experiments on multiple benchmark datasets (PKU-MMD, LARa, MCFS-22, and MCFS-130) , where it achieves superior, state-of-the-art results. The most significant gains are seen in segmental F1 scores, directly validating the method's ability to enhance temporal boundary precision.

3. Thorough ablation studies demonstrate the efficacy of each core component. The studies show that both the Expert-Driven Decoupling (EDD) and the Curvature-Guided Synergy (CGS)  independently improve performance. When combined, the full model achieves a synergistic effect, with performance gains surpassing the sum of the individual modules.

4. The paper demonstrates that curvature is a more robust proxy for action boundaries than traditional distance metrics like Euclidean or Cosine. Curvature is sensitive to changes in the direction of the feature trajectory, making it better at detecting both gradual and abrupt action transitions.

### Weaknesses
1. The related works in Skeleton-based Temporal Action Segmentation are not fully discussed. Only two works in 2020 are discussed. More recent works shuold be incorporated.
2. As reflected by Eq.9, the information of the classification head and localization head just interact once, not in a self-reinforcing loop as the authors describe.
3. The description of the foundation framework in sec.3.2 owns too much space. As the baseline and foundational model, it should be compactly introduced.
4. The key innovation is directly borrowed from the previous work (Shinet al., 2024), and adopted for STAS with a simple transfer.
5. Although the paper asserts that high curvature corresponds to intra-segment motion and low curvature to boundaries, the theoretical link is only qualitatively motivated and relies on assumptions (e.g., class clusters as hyperspheres). The “Appendix B proof” simplifies dynamics to random walks within spheres, which is too idealized for real, noisy skeleton trajectories.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes CurvSeg, a novel framework for skeleton-based temporal action segmentation (STAS) that addresses the long-standing tension between action classification and boundary localization. The key insight is geometric: well-separated classification features naturally induce high curvature within action segments and low curvature at transitions — forming a "valley" that serves as a strong prior for boundary detection.
To exploit this, the authors introduce two core components:
Curvature-Guided Synergy (CGS): A bidirectional consistency mechanism where classification feature curvature guides boundary prediction, while boundary supervision regularizes classification features to enhance cluster compactness.
Expert-Driven Decoupling (EDD): A Mixture-of-Experts module with task-specific experts that refine shared encoder outputs into adaptive representations for classification and localization.
Extensive experiments on four benchmarks (PKU-MMD, LARa, MCFS-22/130) show consistent improvements over state-of-the-art methods, particularly in segmental F1 scores, validating the effectiveness of curvature-guided collaboration.
The work makes a compelling case for geometric priors in structured prediction tasks, offering both conceptual novelty and practical gains.

### Strengths
Originality

Innovative use of representation geometry: Leveraging trajectory curvature as a cross-task signal is conceptually fresh and theoretically grounded (Appendix B). This moves beyond typical attention or fusion mechanisms.
Bidirectional synergy design: Unlike prior decoupled frameworks that treat tasks independently, CurvSeg establishes a mutual reinforcement loop, which is rare in STAS literature.
Task-adaptive MoE without parameter explosion: The Gaussian expert routing is lightweight yet effective, enabling dynamic feature specialization without full dual-path architectures.

 Quality

Rigorous experimental evaluation: Results across four datasets, including ablation studies and comparisons with strong baselines (DeST, LaSA), demonstrate robust performance gains.
Well-designed ablations: Tables 3–6 clearly isolate contributions of CGS and EDD, showing their individual and synergistic effects.
Qualitative visualization: Figure 4 effectively illustrates improved boundary precision and reduced over-segmentation.

 Clarity

The paper is well-written and logically structured, with intuitive figures (Fig. 1–2) explaining the core ideas.
Equations are clearly presented, and Algorithm 1/2 provide sufficient implementation details.
Appendices offer valuable theoretical justification and hyperparameter analysis.

Significance

Addresses a fundamental limitation in STAS: insufficient cross-task collaboration despite semantic interdependence.
Demonstrates that geometric structure in learned representations can be exploited for downstream tasks — an idea potentially applicable to other time-series problems (e.g., speech segmentation, medical signal analysis).
Offers a new paradigm: using internal model dynamics (curvature) as supervisory signals, reducing reliance on external priors.

### Weaknesses
1) Limited discussion on failure cases
While the method performs well overall, there is no analysis of when or why curvature fails as a boundary proxy. For example:
In gradual transitions (e.g., slow hand movement), curvature may not form clear valleys.
Noisy skeleton data might amplify spurious curvature peaks.
A brief error analysis (e.g., per-action performance drop) would strengthen the claims.

2) Assumption of uniform segment partitioning
The EDD module divides videos into fixed-length segments (e.g., M=64), regardless of actual action duration. This could misalign temporal patterns for very short or long actions. Some discussion on adaptivity (e.g., content-aware segmentation) would improve robustness.

3) Dependency on classification quality
The CGS module assumes that classification features already form compact clusters. If initial clustering is poor (e.g., due to ambiguous actions), curvature may not emerge reliably. The paper lacks sensitivity analysis under weak classification regimes.

 4) Reproducibility concerns
Although code will be released, some implementation details are missing:
How exactly are classification features $F_{cls}$ extracted? From the encoder output or after the classification head?
Is the curvature computed per-joint or globally?
These should be clarified in the final version.

### Questions
Q1: In Section 3.3, you mention that low-curvature regions correspond to boundaries. But in Fig. 4(a), we see high curvature at boundaries. Could you clarify this apparent contradiction? Is it possible that both high and low curvature can indicate transitions depending on context?
This could change my understanding of whether curvature acts as a direct boundary detector or only an indirect regularizer.

Q2: You show in Table 5 that curvature outperforms Euclidean/Cosine distance metrics. Have you considered comparing against learned boundary detectors (e.g., gradient-based saliency maps)? Does curvature still dominate in such comparisons?

Q3: What happens if you apply the curvature signal only during training but remove it at inference? Would performance drop significantly? This would help quantify how much of the gain comes from architectural synergy vs. test-time guidance.

### Soundness
3

### Presentation
3

### Contribution
3
