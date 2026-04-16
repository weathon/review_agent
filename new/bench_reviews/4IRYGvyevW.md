## Summary
This paper proposes a representation-based framework for studying feature learning through manifold capacity and related geometric descriptors, with the goal of moving beyond the usual lazy-versus-rich dichotomy. The core claims are that task-relevant manifolds untangle during richer feature learning, that manifold capacity can quantify this richness, and that geometric decompositions of capacity reveal finer learning strategies/stages and provide insights in neuroscience-style RNN settings and cross-dataset transfer/OOD-style probing.

## Strengths
- **A genuinely interesting representational perspective on feature learning.** The paper targets a real gap: most lazy/rich analyses are phrased in terms of weights or NTKs, whereas this work studies the geometry of representations directly. That framing is well motivated in the paper, especially for neuroscience: “current limitations in neuroscience technology for precisely tracking synaptic weight changes... necessitate a framework based on neural representations rather than network weights or neural tangent kernel.”
- **Nontrivial theoretical component, even if narrow.** Section 3.1 proves a monotone relationship between capacity and learning rate in a specific 2-layer teacher-student setting after one gradient step, and also links capacity to prediction accuracy. This gives at least one clean setting where the proposed scalar is analytically grounded rather than purely heuristic.
- **The geometric decomposition is more informative than a single scalar alone.** The use of radius, dimension, and alignment quantities gives a coherent mechanism-level vocabulary for describing how representations change, and Section 4 does surface qualitatively different trajectories that would be invisible from accuracy curves alone.
- **Empirical breadth across synthetic, vision, and RNN settings.** The paper is not restricted to one toy example: it includes 2-layer synthetic experiments, CIFAR-based DNNs, RNNs on neurogym-style tasks, and a cross-dataset probing analysis involving CIFAR-10/10C/100. That breadth supports the claim that the lens is potentially useful across domains.
- **Some observations are genuinely interesting even absent full causal validation.** In particular, Section 5.1’s finding that RNNs with different initial ranks can end with similar final capacity but different final geometry is a useful descriptive observation, and Section 5.2’s “ultra-rich” regime behavior is provocative enough to merit attention.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper’s headline claims are broader than what is actually established.**  
  The abstract says the authors “show both theoretically and empirically that task-relevant manifolds untangle during rich learning, and that manifold capacity quantifies the degree of richness.” But the formal result is much narrower: Theorem 1 is for a **2-layer network**, **teacher-student setting**, **fixed readout**, **squared loss**, **proportional asymptotics**, and crucially **one gradient step**; the paper itself explicitly notes in footnote 6 that it studies “only the first gradient step.” That theorem is meaningful, but it does not by itself justify the paper’s more general statements about full training dynamics, deep networks, or representation learning in realistic settings. The empirical sections provide supportive evidence, but they do not fully close that gap.
- **The main empirical validation is anchored to a regime-control hyperparameter rather than an independent target notion of feature learning.**  
  Section 2.3 explicitly states: “we use the inverse scale factor \(\bar{\eta}\) as a tunable ground truth for the degree of feature learning.” In the Chizat-style setup, \(\bar{\eta}\) is indeed a standard control knob for interpolating between lazier and richer training, so this is not a misunderstanding. But it is still a limited validation strategy: showing that capacity tracks the hyperparameter used to induce richer-vs-lazier dynamics does not fully establish that capacity measures task-relevant feature learning in a more general or model-agnostic sense. This especially matters because Section 3.2’s comparisons to weight change and alignment metrics are also judged largely by recovery of that intended ordering.
- **Several interpretive claims in the applications are stronger than the evidence warrants.**  
  In Section 5.2, the paper writes that radius expansion and center-axis alignment “explain the failure of OOD generalization in the ultra-rich regime.” What is actually shown is a correlation between those geometric quantities, lower capacity, and worse CIFAR-100 linear-probe performance. That is suggestive, but without intervention or ablation it does not yet establish explanatory mechanism. The same issue appears, though more mildly, in Section 4’s “learning stages” language and Section 5.1’s “structural biases” framing: the descriptive trends are interesting, but the paper sometimes crosses from characterization into explanation without sufficient causal support.
- **The “learning stages” and “learning strategies” claims are not rigorously validated as stable phenomena.**  
  Figure 4c labels four stages—clustering, structuring, separating, stabilizing—based on normalized trajectories of geometric measures. These labels are intuitively reasonable, but the paper does not provide quantitative stage definitions, changepoint analysis, or evidence of robustness across seeds/models in the main text. As written, this reads more like insightful descriptive patterning than an established discovery of discrete stages. The same applies to strategy tradeoffs inferred from the contour plots in Figure 4a,b.

### Minor
- **The OOD section is somewhat rhetorically overstated relative to the task actually used.**  
  Section 5.2 explicitly focuses on the case where the test label set differs from the training label set, and operationally evaluates a **linear probe on CIFAR-100 representations learned from CIFAR-10 pretraining**. This is a reasonable transfer/cross-dataset separability analysis, and the paper does note the different label sets. But calling this simply “OOD generalization” risks overgeneralization, since this is not the most standard same-task-under-shift setup.
- **The last-layer-only analysis is under-justified.**  
  Section 2.3 says: “All analyses were performed on the test data representations in the last layer.” That is a defensible simplification, and not inherently wrong. Still, for a paper about feature learning dynamics in deep nets, it would strengthen the case to justify why the last layer is the right locus, or to show whether earlier layers tell a similar or different story.
- **It remains unclear how much the MCT-derived geometric measures add beyond simpler geometric descriptors.**  
  The paper defines manifold radius/dimension/alignment via anchor points from manifold capacity theory. Those are analytically connected to capacity, which is a real advantage. But the paper does not compare them against simpler alternatives such as centroid distances, intra-class variance, or PCA-based effective dimension. As a result, the practical distinctiveness of these specific descriptors is not yet fully established.
- **The RNN application is more a descriptive reframing than a decisive new conclusion.**  
  Section 5.1’s reinterpretation as “poorer-richer” versus “wealthier-lazier” is interesting, especially because it highlights the role of initialization capacity rather than only changes during training. However, the functional consequence of ending with different geometry at similar final capacity is not tested. So the section demonstrates a useful lens, but not yet a strong new explanatory result.

### Trivial
- None.

## Nice-to-Haves
- Add a more explicit calibration of the claims in the abstract and introduction to match the actual scope of Theorem 1 and the empirical evidence.
- Quantitatively validate the “learning stages” with changepoint detection, seed variation, or a criterion defining stage transitions.
- Compare the proposed geometry measures to simpler representation-space baselines to clarify what is uniquely gained by the MCT machinery.
- In the cross-dataset probing section, test whether regularizing the implicated geometric quantities can actually mitigate the “ultra-rich” degradation.
- Explore earlier layers, not only the final layer, to understand where the geometric effects emerge.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The comparison to conventional measures is unfair because they measure different things.”**  
  Removed/softened. The paper is explicit that weight changes, NTK-label alignment, and representation-label alignment are conventional lazy/rich diagnostics, and it compares them as practical diagnostics rather than as theoretically identical quantities. The valid criticism is narrower: the comparison is mainly judged by agreement with the chosen regime-ordering variable.
- **“The paper lacks real neuroscience validation because it does not use neural recordings.”**  
  Removed as an out-of-scope demand. The paper claims applications to neuroscience-style problems via RNNs and representational tools; it does not promise analysis of biological recordings in this submission.
- **“The experiments are too limited because they do not include many more architectures / transformers / large-scale datasets.”**  
  Softened to a nice-to-have. The current experimental suite is not tiny: it includes synthetic 2-layer models, VGG-11, ResNet-18, and RNNs across several tasks. Broader coverage would help, but the absence of transformers or larger benchmarks is not by itself a substantive flaw relative to the paper’s stated scope.
- **Requests for additional related work or release/availability concerns.**  
  Removed per policy and because they are not verifiable here.

## Novel Insights
The most useful synthesis is that the paper is strongest when read as a **representational diagnostic framework** rather than as a fully established general quantification theorem for feature learning. In that framing, the work makes a credible contribution: it shows that manifold-capacity-based geometry can organize a range of empirical phenomena more richly than the lazy/rich binary alone. The weakness is not that the observations are uninteresting, but that the paper sometimes markets descriptive geometric regularities as general quantification or explanation. A cleaner positioning—diagnostic lens with one theoretically grounded anchor case—would likely make the contribution both more precise and more convincing.

## Suggestions
- Reframe the top-level claim more conservatively: present capacity as a **promising representation-based diagnostic of feature-learning richness**, theoretically justified in one clean setting and empirically supported in several others, rather than as a generally established quantifier.
- In Section 3, separate more clearly the theorem-backed claim from the empirical extrapolation to full deep-network training.
- In Section 4, define “learning stages” operationally and show robustness across random seeds or architectures.
- In Section 5.2, replace “explain” with “correlate with” unless an intervention is added; ideally, add a simple regularization or selection experiment testing whether controlling radius/alignment changes the CIFAR-100 probe outcome.
- Add at least one comparison between MCT-based geometric measures and simpler geometry baselines to clarify why the proposed descriptors are worth their complexity.
- If space permits, include a short last-layer-vs-earlier-layer comparison to justify the representational level chosen for analysis.

## Score and Decision
**Evaluation across axes:**  
- **Originality:** Good. The representation-centric use of manifold capacity for lazy/rich analysis is a real conceptual contribution.  
- **Importance of the research question:** Good. Understanding feature learning through representations is important for both ML and neuroscience.  
- **Whether the claims are well supported:** Mixed. Some are supported, but the broadest claims are stronger than the theory and experiments justify.  
- **Soundness of experiments:** Reasonable to good descriptively, but several conclusions are correlational and lack robustness/causal validation.  
- **Clarity of writing:** Generally good; the main issue is calibration of claims rather than exposition.  
- **Value to the research community:** Positive. Even with overclaiming, the framework is likely to be useful as an exploratory representational lens.

**Calibration against human-reviewed anchors:**  
- Compared to **“How connectivity structure shapes rich and lazy learning in neural circuits”** (Accept, scores 8/6/8/5), this paper is similarly interesting and similarly bridges theory with neuroscience-motivated experiments, but here the central quantification claim is more vulnerable to overreach and the application claims are more correlational. I place it slightly below that paper.  
- Compared to **“Grokking as the transition from lazy to rich training dynamics”** (Accept, scores 8/8/3/5), this submission has broader empirical scope and a useful new lens, but its theory-to-claim gap is also substantial; I view it in a similar middle-positive range.  
- Compared to **“How Feature Learning Can Improve Neural Scaling Laws”** (Spotlight, scores 8/8/6/6/8), this paper is clearly below: that paper’s limitations were better calibrated and its contribution more tightly matched to what was established.  
- Compared to lower-scored geometry/representation papers such as **“A simple connection from loss flatness to compressed representations”** (Reject, mixed scores 3/8/6/3), this paper is stronger: it is better motivated, broader empirically, and not fundamentally unsound.

Overall, this is a **good but imperfect** paper: meaningful contribution, interesting framework, but with notable overclaiming and insufficiently validated interpretive leaps. I land slightly above the acceptance threshold.

**Final score: 6.5 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>  
MY FINAL DECISION: <orange>Accept</orange>