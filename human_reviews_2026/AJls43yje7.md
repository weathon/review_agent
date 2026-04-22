# Multi-Marginal Flow Matching with Adversarially Learnt Interpolants

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
Learning the dynamics of a process given sampled observations at
  several time points is an important but difficult task in many
  scientific applications.
  When no ground-truth trajectories are available, but one has only
  snapshots of data taken at discrete time steps, the problem of
  modelling the dynamics, and thus inferring the underlying
  trajectories, can be solved by multi-marginal generalisations of
  flow matching algorithms.
  This paper introduces a novel flow matching method that overcomes
  the limitations of existing multi-marginal trajectory inference algorithms.
  Our proposed method, ALI-CFM, uses a GAN-inspired adversarial loss
  to fit neurally parameterised interpolant curves between source and
  target points such that the marginal distributions at intermediate
  time points are close to the observed distributions.
  The resulting interpolants are smooth trajectories that, as we
  show, are unique under mild assumptions.
  These interpolants are subsequently marginalised by a flow matching
  algorithm, yielding a trained vector field for the underlying dynamics.
  ALI-CFM outperforms existing baselines on spatial transcriptomics and
  cell tracking problems, while performing on par with them on
  single-cell trajectory prediction, which showcases its versatility and scalability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Adversarially Learned Interpolants: a GAN-style objective that directly matches intermediate-time marginals to a learnable interpolant. The full ALI-CFM pipeline first learns adversarially (with novel regularisers for uniqueness/smoothness), then regresses a time-dependent vector field via a standard CFM loss. The generator and discriminator are time-dependent, and no explicit metric is specified.

The method is evaluated against existing flow matching baselines on several tasks: a synthetic 2D knot dataset , a real-world cell tracking dataset, a standard single-cell trajectory inference benchmark , and a novel task of inferring tumour coordinates from spatial transcriptomics data.

### Strengths
1.  **Novel Regularisers** - the linear-reference and curvature penalties are well-motivated for stabilising adversarial training and encouraging uniqueness of interpolants
2.  **New Benchmark Task** - Tumour coordinate inference with spatial transcriptomics is a relevant multi-marginal benchmark; the paper positions ALI-CFM credibly there and reports reproducible settings 
3.  **Problem Importance**  - The paper proposes an elegant approach to well motivated problem

### Weaknesses
1. **MFM Baseline Comparison**  
   The paper’s central argument is that MFM is *“not suitable when the data geometry varies with time”* because their metric is *“time-independent.”* However, the statement *“their metric is time-independent, which contrasts with our generator and discriminator networks which are both time-dependent”* is somewhat misleading.  The authors explicitly state that their MFM implementation *“follows the setup … where the metric is inferred from all the data.”* This is a **specific, time-independent implementation choice**, not an inherent limitation of the MFM framework.  In fact, the LAND metric can be defined **locally in time** (e.g., using only data from the nearest time points, as done in MFM’s single-cell setup). Such a *time-localized* definition would make the metric effectively time-dependent and would likely yield more accurate MFM interpolants, especially in Figure 2.  This would basically be equivalent to time dependance learnt through Adversarial training as its the only granularity of information that we posses. 

2. **Novelty Claim**  
   The paper frames its novelty as introducing a time-dependent adversarial approach that overcomes previous work time-independence. However, a time-conditioned discriminator (as used in ALI) and a time-localized LAND metric (as in a correctly implemented MFM) are conceptually similar—both use data from neighboring time points to define geometry along the interpolant.   If a fair, time-localized MFM implementation performs competitively, the core premise of the paper weakens: the proposed method may represent an alternative, but not fundamentally new, way of learning time-dependent interpolants—albeit through a more complex, GAN-based training procedure.

3. **Minor: Attribution of the Interpolant Form**  
   The interpolant parameterization in Equation (5), $G_{\phi}(x_{0},x_{1},t)=(1-t)x_{0}+tx_{1}+t(1-t)f_{\phi}(\cdot)$,  is a standard form introduced in prior works (e.g., Neklyudov et al., 2024; Kapuśniak et al., 2024).  While these papers are cited in the related work section, the equation itself appears without a direct attribution, which could unintentionally suggest it is novel.

### Questions
1. **Baseline fairness and time-localization of LAND.**  
   Did your baselines using the LAND metric employ the *adjacent-time* constraint as in the original MFM implementation?  
   If not, could you please report results using the time-step–localized (i.e., nearest-time) implementation of LAND as discussed in the Weaknesses section, and reflect these results in Figures 2, 4, 7, and Table 3?

2. **Validity of the “geometry varies with time” claim.**  
   If the above concern holds, the statement that LAND/RBF-based and prior methods are *“not suitable when the data geometry varies with time”* may not be accurate. Is there any additional theoretical or empirical reason that supports this claim?

3. **Novelty.**  
   If the above concern is valid and a fair, time-localized MFM performs well, what do the authors consider to be the primary novelty of ALI-CFM beyond offering an alternative formulation of a time-conditioned interpolant?

### Soundness
1

### Presentation
4

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
This paper tackles *multi-marginal trajectory inference* from unpaired snapshots observed at discrete times. The proposed method, ALI-CFM (Adversarially Learnt Interpolants + Conditional Flow Matching), departs from hand-crafted paths (linear, spline, piecewise-linear) by *learning* a time-conditioned interpolant $G_\phi$ whose intermediate-time pushforwards match the observed marginals.

### Strengths
* **(S1) Interpolant learning via distribution matching.** Adversarial learning of $G_\phi$ avoids brittle pointwise constraints and yields smoother, noise-robust paths.

* **(S2) Non-stationary geometry.** Clear wins on datasets with evolving topology/geometry (e.g., Knot, Cell Tracking), where time-independent metrics (e.g., MFM) underperform.

* **(S3) High $K$ stability.** Handles very large numbers of time points (e.g., $K=1200$) without the "kinks" typical of piecewise methods, easing CFM training.

* **(S4) Theory.** Uniqueness results for linear / piecewise-linear reference regularizers (Thms. 2.1–2.2) provide identifiability-style guarantees.

* **(S5) Spatial transcriptomics.** Strong results on a challenging multimodal ST task and competitive performance on cell tracking benchmarks.

### Weaknesses
* **(W1) Adversarial stability.** No analysis of GAN stability, failure modes (e.g., mode collapse), or sensitivity to $\lambda$, critic class, or choice of GAN loss.

* **(W2) Compute overhead.** Two-stage training plus the need to evaluate $\partial_t f_\phi$ during CFM adds cost; the paper lacks wall-clock and complexity breakdowns versus single-stage baselines.

* **(W3) High-dimensional scRNA-seq.** In 50D/100D, ALI-CFM is only on par with (or slightly below) OT-MFM; limited discussion of causes or remedies.

### Questions
1. **Stability of adversarial training (W1).** Did you observe mode collapse or time-wise imbalance across $\{t_i\}$? How sensitive are results to $\lambda$ and to the GAN loss (vanilla vs. R3GAN in Appx. D.3)? Any training heuristics that were essential (e.g., critic:generator update ratios, spectral norm, gradient penalties)?

2. **Compute and $\partial_t f_\phi$ (W2).**
   * Authors stated "little overhead" but would be great to report wall-clock and scaling with $(d,K,N)$ for ALI, CFM, and end-to-end ALI-CFM, compared to OT-MFM/OT-CFM.
   * Algorithm 2 requires $\partial_t f_\phi(x_0,x_1,t)$. Do you autograd through $t$ every CFM step? What is the measured overhead relative to settings with analytic $\dot{G}$?

3. **Regularizer selection.** You mix linear reference, piecewise-linear reference, and $\|\partial_{tt} G\|^2$ across datasets. Can you provide practical guidance (e.g., stationary vs. evolving geometry, large $K$, noise levels) and an ablation isolating each regularizer per dataset?

4. **High-dimensional performance (W3).** Beyond the pointwise-vs-distribution-matching explanation, what limits ALI-CFM in 50D/100D (critic capacity, coupling $\pi$, time-conditioning)? Have you tried stronger critics or multi-scale time features?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new approach to learn probability distributions dynamics with multiple marginals, based on flow matching. Recognizing that previous approaches to define the interpolants across multiple marginals (piecewise linear or splines) may exhibit high curvature, thereby leading to suboptimal training, the authors propose to *learn* the interpolants. 

To learnt the interpolants, the authors enforce that the marginal distributions match at each time steps, using a adversarial loss in practice. Using the learnt interpolants, the authors then plug them directly into the flow matching machinery.

The authors propose multiple trajectories regularization for learning the interpolants, such as linearity, or curvature minimization.

The authors then show qualitatively that this results in smoother trajectories on synthetic and cell tracking data, leading to more accurate flow. Quantitatively, they evaluate their approach on cellular trajectories and tumor proliferation and demonstrate that their approach is competitive with state of the art methods.

### Strengths
- This paper proposes a creative approach to learning more meaningful interpolants in flow matching, which is of significant interest when dealing with multiple marginal distributions.
- I appreciate the design of the new experiment using spatial data, although the biological relevance of this experiment is questionable as it only takes into account the spatial distribution of the tumor, discarding all the single cell RNA seq resolution of the data.
- The paper is easy to read and the authors expose their ideas and contributions clearly.
- The authors demonstrate the performance of their method graphically, which makes the evaluation of performance of the method easier for the reader.

### Weaknesses
- An important motivation for this work is to have interpolant with less curvature, hopefully leading to improved training. As such, one experimental result that is lacking from this work is the cumulative curvature of the learnt interpolants, as well as some notion of divergence between the pushforward distribution with the GAN interpolants and the observed marginal distributions (for all different interpolant strategies). Ideally, depending on the hyperparameters and the strength of the regularization used, there would be some trade-off between the cumulative curvature and the matching at each time step. In my opinion, that would really nail down the contribution of the paper.

- The discriminator in the GAN seems to be common across all time steps. This seems suboptimal, especially for trajectories that intersect, like the synthetic knot experiment. In my opinion, you need a different discriminator for each time step, for the argument to hold, which may be challenging to train correctly when you don't have many samples per time step.

- The most established benchmark in this paper is the scRNAseq trajectory inference. However, this model does not improve on that benchmark. That undermines the main motivation for this paper. The spatial dataset is, from my understanding of biology, somewhat toy-ish too, as I expect there may much better suited methods for that problem that flow matching. If this model does not improve performance on tasks that are established to be meaningful, it raises the question of "what problem this paper is really solving?".

### Questions
- I’m surprised by how non-smooth the trajectories of the splines are in Figure 2. Did you compute splines between points that are matched with OT  between successive time steps ? Or abitrary points ? I think MMFM uses OT coupling between distributions.
- The authors state `Although our algorithm is on par with existing baselines, we believe that the nature of adversarial
training makes it difficult to completely outperform OT-MFM. Since our adversarially learned
interpolant matches the points at each time point in a distribution-matching sense, it might lose in
pointwise metrics to methods that are trained to overfit to the given points.` That is an interesting point, although the metric here is EMD so why do the authors refer to pointwise metrics ? Also, by the same argument as used by the authors, do we expect that this method may generate non-“realistic” samples ? That is, because the interpolants don't exactly align with the real data points, the model never learns to exactly generate "realistic" samples (it generates something slightly off). I would like the authors to comment on that as it can be a significant limitation of the method.
- Echoing my comment above: could you please comment on having a discriminator at each time step - whether you did this or not - and potential limitations in terms of sample size.
- Also echoing my other comment above, could the authors add the metrics I pointed to above - or  if not, could they argue why it's not relevant ? ` An important motivation for this work is to have interpolant with less curvature, hopefully leading to improved training. As such, one experimental result that is lacking from this work is the cumulative curvature of the learnt interpolants, as well as some notion of divergence between the pushforward distribution with the GAN interpolants and the observed marginal distributions (for all different interpolant strategies). Ideally, depending on the hyperparameters and the strength of the regularization used, there would be some trade-off between the cumulative curvature and the matching at each time step. In my opinion, that would really nail down the contribution of the paper.`
- I understand that this is very challenging give the time constraints, but I think it would be great to have an established benchmark where your method outperforms previous baselines. If not, please consolidate why the ST task is actually relevant biologically.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles multi-marginal flow matching from  intermediate time marginal data. The paper proposed a 2-stage approach, first learn a non-linear interpolant to match intermediate marginals via adversarial learning and then fit a vector field with the CFM loss on those interpolants. Experiments on synthetic, cell-tracking, scRNA-seq, and spatial transcriptomics are provided to demostrate the marginal matching property of the proposed method.

### Strengths
1. The paper is clearly written and easy to understand.
2. Theoretical analysis showing existence of the interpolants strengthens the paper.
3. Practical benefits are demonstrated in the experiments.

### Weaknesses
1. Compared to MFM, the method differs in use of GAN based distribution matching for intermediate marginals. The added complexity of adversarial training makes the idea less appealing compared to existing methods that do not require adversarial training.
2. The performance on Section 4.3 single-cell RNA-seq seems to be weak compared to MFM that does not require adversarial training. 
3. Only OT coupling based implementation is used in the experiments. OT adds additional complexity and is challenging to compute in high dimension. 
4. Ablations are not provided for OT/Non-OT, different regularizations, etc.
5. The empirical validation is limited to low dimensional applications.

### Questions
1. Could the authors explain the reason behind weaker performance in Section 4.3 ? 
2. Can the method work well with independent coupling ? How does it compare to OT-based? What would be the challenges? 
3. Could the authors provide ablation study for different regularizations? 
4. Could you discuss the possible challenges for high dimensional applications ?

### Soundness
3

### Presentation
3

### Contribution
2
