# From Circuits to Dynamics: Understanding and Stabilizing Failure in 3D Diffusion Transformers

- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
Reliable surface completion from sparse point clouds underpins many applications spanning content creation and robotics. While 3D diffusion transformers attain state-of-the-art results on this task, we uncover that they exhibit a catastrophic mode of failure: arbitrarily small on-surface perturbations to the input point cloud can fracture the output into multiple disconnected pieces -- a phenomenon we call meltdown. Using activation-patching from mechanistic interpretability, we localize meltdown to a single early denoising cross-attention activation. We find that the singular-value spectrum of this activation provides a scalar proxy: its spectral entropy rises when fragmentation occurs and returns to baseline when patched. Interpreted through diffusion dynamics, we show that this proxy tracks a symmetry-breaking bifurcation of the reverse process. Guided by this insight, we introduce PowerRemap, a drop-in, test-time control that stabilizes sparse point-cloud conditioning. On Google Scanned Objects, PowerRemap has a stabilization rate of 98.3% for the state-of-the-art diffusion transformer WaLa. Overall, this work is a case study on how diffusion model behavior can be understood and guided based on mechanistic analysis, linking a circuit-level cross-attention mechanism to diffusion-dynamics accounts of trajectory bifurcations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper reveals and deeply analyzes a catastrophic failure phenomenon that exists in the current most advanced 3D diffusion Transformers when performing surface reconstruction from sparse point clouds, which the authors call meltdown. Specifically, a tiny perturbation located on the surface of the input point cloud may cause the model output to melt from a complete single shape into a large number of unconnected fragments.

The author first demonstrated the existence of this phenomenon and found that the spectral entropy of this key activation could serve as a measurable proxy indicator for the occurrence of the collapse. Based on this insight, the author proposed a simple yet effective test-time intervention method called PowerRemap. The author provides an explanation for the meltdown phenomenon from the perspective of diffusion dynamics, linking it to the symmetry-breaking bifurcations of the potential energy landscape during the reverse diffusion process, thereby associating the discoveries at the circuit level with the deeper theoretical framework of generative models.

### Strengths
This is a high-quality and fascinating study. It combines solid experiments, ingenious interpretability analysis, effective solutions and profound theoretical connections, which is of great significance for improving the robustness and interpretability of diffusion models, especially 3D generative models.

### Weaknesses
The paper put forward the "consensus" hypothesis, but this has not yet been strictly verified.

Currently, the intensity parameter γ is a manually set global value (γ=100), which is not optimal and also limits the robustness of the method.

### Questions
1. Has the author ever attempted an adaptive γ selection strategy?

2. Please discuss in more detail the reasons for the poor performance on Make-A-Shape.

3. In the main text or appendix, the measurement criteria for point cloud sparsity should be more clearly stated. Can the average density of the point cloud relative to the surface area of the object or the sampling distance of the farthest point be provided? This helps readers better understand the challenge of the problem setting.

4. There is still some speculation about the mechanism explanation for "why does a decreased spectral entropy reduce invalid outputs?". It is suggested that the author deepen this explanation. For instance, can we analyze whether there are any changes in the alignment or consistency of the outputs from different attention heads before and after the PowerRemap intervention? Or, can the assumption that "the first singular vector represents the consensus feature of multiple heads" be verified through the analysis of singular vectors? Even a preliminary analysis can significantly enhance the persuasiveness of the argument.

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
3

### Summary
The paper studies a common catastrophic mode of faillure for 3D diffusion model in the task of surfuace completion from sparse point clouds . They name it as melt-down and performs activation-patching to localize the failure position. They leverages the singular-value spectrum of the located activation module to serve as a proxy for the failure mode. To tackle this issue, they proposes PowerRemap module to adjust the singular value as a test-time module. Experiments are done on GSO to illustrate the idea.

### Strengths
1. The meltdown phenomenon is a common issue in 3D diffusion models for shape completion and worth investigating.

2. The finding that a single cross-attention module is primarily responsible for the observed failure is particularly interesting and provides useful insight into the model’s internal behavior.

3. The discussion on diffusion dynamics is interesting and contributes to a better conceptual understanding of diffusion behavior.

### Weaknesses
1. The experiments are insufficient. The observed meltdown failure is likely to depend strongly on the density of the input point cloud, yet this factor is neither analyzed nor explicitly specified in the experiments. In addition, all experiments are conducted solely on the GSO dataset, which limits the generality of the conclusions. Including results on at least one additional dataset would significantly strengthen the empirical support for the proposed theory.

2. In Fig. 3, the trend of connectivity C does not fully align with that of H. Specifically, C rises sharply and reaches its maximum around ρ=0.4, then slightly decreases, whereas H continues to increase. This raises questions about the claimed relationship between H and C— why would a decreasing H correspond to improved connectivity? As presented, the experiment is not sufficiently convincing and requires clearer explanation or additional analysis.

3. The method appears to assume that the input point cloud is clean and contains only a single object. The definition of “healthy” results seems to rely on this restricted input condition. It is unclear how the approach would perform when the input represents a complete scene or includes significant sensor noise. Moreover, the theoretical formulation seems to inherently produce a single mesh, regardless of the input’s complexity or content.

### Questions
Please refer to weaknesses. Although the experimental evaluation is somewhat limited and certain analyses are not entirely convincing, the findings are novel and valuable. I would be glad to see this work published if the authors can provide additional experiments to strengthen the empirical validation.

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
1

### Summary
The paper identifies a failure mode called “meltdown” in point-cloud-conditioned 3D diffusion transformers, such as WALA and MAKE-A-SHAPE. Tiny on-surface input perturbations cause fragmented, multi-component outputs. Using activation patching, the authors localize causality to a single early cross-attention. The singular-value spectral entropy of this write tracks failure/rescue. The authors also propose PowerRemap, a test-time SVD power transform that lowers spectral entropy and substantially stabilizes outputs.

### Strengths
Clean activation-patching grid over depth×time pinpoints a single early cross-attention write controlling meltdown; procedure and repair map are explicit.

PowerRemap is model-agnostic, test-time only, and provably reduces spectral entropy without changing singular vectors.

On GSO, meltdown occurs widely, and PowerRemap rescues 98.3% of WALA failures.

### Weaknesses
For make-a-shape, reported rescue is only 10.1% with the same 𝛾, suggesting sensitivity to architecture and hyperparameters and limiting generality. 

Spectral entropy is the only diagnostic evaluated; no comparison to effective rank, top-k energy, condition number, per-head concentration, or Jacobian norms.

“Connected components” may conflate legitimate multi-part objects with failures; precision/recall vs. human labels not reported.

𝛾 selection is ad-hoc (global 𝛾=100); the paper itself notes the open question of choosing 𝛾 and the speculative nature of the “consensus via low entropy” explanation.

### Questions
Compare spectral entropy to alternative spectral metrics for predicting meltdown

Provide an adaptive 𝛾 rule and show it fixes MAKE-A-SHAPE’s low rescue rate

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper uses mechanistic interpretability to investigates the meltdown phenomenon of 3D diffusion transformers on surface reconstruction tasks, where small perturbations to the input point cloud can cause catastrophic fragmentation of the generated 3D surfaces. Using activation patching, this paper identifies a specific cross-attention head in the WALA model whose activations have causal connections with the connectivity of the reconstructed surface. The paper shows that intervening on the magnitude of the singular values of the decomposed cross-attention output can better recover the shape of generated object. Finally, the paper connected the meltdown phenomenon with bifurcation dynamics in the reverse diffusion process.

### Strengths
1. The paper presents an interesting application of existing activation patching method to identify geometry-related representations within 3D latent diffusion models.
2. The proposed meltdown phenomenon is novel and well-characterized, although it remains unclear whether similar behavior would be observed on other surface reconstruction datasets beyond Google Scanned Objects (GSO).
3. The proposed PowerRemap intervention is simple yet effective, demonstrating strong recovery performance on WaLa model and the GSO dataset by intervening the cross-attention head outputs that have causal connections with output surface connectivity. Nonethelss, it is questionable whether this intervention method generalizes to other models (see weakness 2).

### Weaknesses
1. The generalizability of this finding is very limited. The experiment focused on two models (WaLa and MAKE-A-SHAPE) and evaluated the meltdown on only one dataset (Google Scanned Objects). It is unknown whether the meltdown phenonomon is unique to the GSO datasets, and if the cross-attention head that controls the meltdown can be found in latent 3D diffusion transformer, other than WaLa and MAKE-A-SHAPE.
2.  As shown in Tables 2 and 3 in Appendix B.3 (p. 21), the effectiveness of PowerRemap differs significantly when applied to the WALA model versus the MAKE-A-SHAPE model. While PowerRemap recovers 98% of the meltdowned generation for WALA on GSO dataset, it recovers only about 10% of meltdowned cases for MAKE-A-SHAPE model. This large discrepancy again raises concerns about the generalizability of the proposed intervention and the causal role of the identified cross-attention head in MAKE-A-SHAPE model.
3. The interpretation offered in this paper is also limited in depth. What exact geometric features have been learned by this cross-attention head? If one ablates this cross-attention head, will the meltdown phenomenon disappear? What is the trade-off between suppressing the spectral entropy of this head's output versus ablating it. 
4. The trends shown in Figure 3 differ between individual and population levels. The difference at population level (across seeds) was unexplained. Within the same random seed, the plot shows that the connectivity $C$ sharply increases after the spectral entropy exceeds a threshold. However, across seeds, the meltdowns (sudden jump in $C$) occur earlier even when the spectral entropy is lower. Current text does not explain thos trend at the population level.
5. The influence of PowerRemap strength $\gamma$ on reconstruction connectivity is not studied. It remains unclear how $\gamma$ should be selected in practice or whether larger / smaller values introduce any trade-offs in reconstruction quality besides connectivity.
6. It is also unclear what data and how many data points and random seeds are used to localized the meltdown circuit in section 3.2.
7. Why search only the cross-attention outputs? The decision to restrict the search space to cross-attention outputs is insufficiently justified. Since the cross-attention outputs will be written back to the residual stream, will you obtain similar results if patch residual stream activations? How much worse does the activation patching on MLP layer outputs compared to the cross-attention outputs.

### Questions
1. What are the patching results for other components (MLP, self-attention, and residual stream) in the latent diffusion transformer?
2. As mentioned in Weakness 5, it is unclear what data and how many samples were used to produce Figures 2 and 3. Do the observed patterns in these plots generalize when evaluated on more data points and random seeds?
3. In Figure 3, the change in connectivity is sudden, suggesting meltdown is relatively binary phenomenon. However, the spectral entropy of the cross-attention head outputs varies smoothly. Does this imply that later MLP blocks might also contribute to (or mitigate) the meltdown failures?
4. For the qualitative examples shown in Figure 4, what are the corresponding ground-truth 3D shapes of these four objects?
5. According to Appendix B.3 (p. 20, l. 1060), the PowerRemap intervention was applied only to failure cases of the model’s generation. What happens if PowerRemap is applied to successful (non-meltdown) cases? Does it alter the output quality or connectivity in any noticeable way?

### Soundness
1

### Presentation
2

### Contribution
2
