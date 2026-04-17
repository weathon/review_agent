# Taming Flow-based I2V Models for Creative Video Editing

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Although image editing techniques have advanced significantly, video editing, which aims to manipulate videos according to user intent, remains an emerging challenge. Most existing image-conditioned video editing methods either require inversion with model-specific design or need extensive optimization, limiting their capability of leveraging up-to-date image-to-video (I2V) models to transfer the editing capability of image editing models to the video domain. To this end, we propose IF-V2V, an Inversion-Free method that can adapt off-the-shelf flow-matching-based I2V models for video editing without significant computational overhead. To circumvent inversion, we devise Vector Field Rectification with Sample Deviation to incorporate information from the source video into the denoising process by introducing a deviation term into the denoising vector field. To further ensure consistency with the source video in a model-agnostic way, we introduce Structure-and-Motion-Preserving Initialization to generate motion-aware temporally correlated noise with structural information embedded. We also present a Deviation Caching mechanism to minimize the additional computational cost for denoising vector rectification without significantly impacting editing quality. Evaluations demonstrate that our method achieves superior editing quality and consistency over existing approaches, offering a lightweight plug-and-play solution to realize visual creativity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents IF-V2V, a novel inversion-free framework for video editing. The authors aim to address a key limitation in existing methods: the reliance on computationally expensive, model-specific inversion or optimization, which restricts the use of state-of-the-art, pre-trained Image-to-Video (I2V) models for editing tasks.

The core idea is to adapt off-the-shelf flow-matching-based I2V models without requiring an explicit inversion step. To achieve this, the method introduces several key technical contributions:

Vector Field Rectification with Sample Deviation (VFR-SD): A mechanism to incorporate information from the source video (e.g., non-edited regions) directly into the denoising vector field via a "deviation term," thus bypassing the need for inversion.

Structure-and-Motion-Preserving Initialization (SMPI): A model-agnostic noise initialization strategy designed to embed structural and motion cues from the source video into the initial noise, promoting temporal consistency and structural fidelity.

Deviation Caching (D-Cache): An efficiency-focused mechanism to minimize the additional computational overhead introduced by the vector field rectification process.

The authors claim that this lightweight, plug-and-play approach achieves superior editing quality and consistency compared to existing methods, offering a more flexible solution for video manipulation.

### Strengths
- The paper’s primary contribution lies in its highly practical approach. It strategically bypasses the significant challenges of training dedicated video editing models from scratch, such as the well-known scarcity of large-scale, paired video-text data and the computationally intensive, time-consuming training process. By proposing a framework designed to transfer the capabilities of established image editing models to the video domain, it offers a far more scalable, efficient, and data-agnostic solution.
- The introduction of VFR-SD and SMPI is a commendable aspect of this work. These components are well-motivated, as they directly address the critical challenge of maintaining consistency.
- The proposal of the Deviation Caching (D-Cache) mechanism to reduce the extra computational burden is a noteworthy contribution, enhancing the overall efficiency of the framework.

### Weaknesses
- Concerns remain regarding the editing consistency and fidelity. Upon inspection of Figure 3c (row 2), the later frames exhibit clear artifacts in non-edited regions, such as a noticeable increase in the saturation of the leaves. A similar issue is present in Figure 3d (row 2), where the final frame appears to suffer from semantic leakage originating from the edited subject. The authors should address these inconsistencies.

- Regarding Figure 8, while the method's applicability to diverse models is a clear strength, the results also highlight a potential weakness. The consistency of the edits across these different models appears limited. This variability is concerning, given that the editing objective is singular and explicit, and one would expect a more uniform outcome regardless of the underlying model.

### Questions
- The manuscript would be strengthened by a more in-depth discussion of the inherent trade-off between non-edited region fidelity and edit consistency. The authors should elaborate on the factors that control this balance and, specifically, how the method is designed to prevent semantic leakage from the edited subject or unintended alterations in static background regions.
- The paper's evaluation of temporal consistency for long-duration videos is insufficient. The current analysis does not provide a clear picture of how the method scales. A more rigorous analysis is needed to demonstrate that consistency can be maintained reliably over extended frame sequences.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces IF-V2V, a training-free, inversion-free method that adapts flow-matching image-to-video models for image-conditioned video editing using an edited first frame. It runs two ODE trajectories in parallel (source and target) and adds a sample-specific deviation, which is the gap between the source’s ground-truth denoising vector and the model’s predicted one, onto the target’s vector field, preserving source details without explicit inversion. To boost fidelity and coherence, SMPI initializes latents with a structure-biased mix of source frames and noise, injects weak source-frame references into conditioning, and uses optical-flow-warped temporally correlated noise for motion. D-Cache reuses deviation terms when dynamics change slowly, lowering computation.

### Strengths
- **Training-free and inversion-free methods** 
The approach is training-free, needs no further training or inversion calculation, and provides a simple way to utilize a pre-trained i2v model to achieve first-frame-guided video editing.

- **Model-agnosticism.**
The approach works well without a model-specific attention manipulation design. This could apply to a series of flow-matching-based models.

- **Resource efficiency.**
The D-Cache mechanism provides a way to reduce resource consumptions.

- **Various editing cases.**
The paper shows various editing cases, including object removal/insertion, stylization, etc, demonstrating its editing capability.

### Weaknesses
- **Ambiguity regarding model-agnosticism.**
The approach still needs manipulation on denoising vectors and the conditioning strategies. The padding-based structure injection depends on the specific conditioning layout of certain I2V models. This might not seamlessly transfer to all i2v models. The claim on  "agnosticism" is tricky. It achieves editing without attention manipulations, but it actually uses some other strategies that still need specific designs with the models.

- **Motion preservation has some issues**
According to the demo shown in the video, there are some issues in motion preservation (e,g, the walking man has a different body shape and direction from the source video). What is the reason behind this artifact? Is that because the VFR-SD cannot preserve well the source motions compared with inversion-based methods?

- **Comparisons**
Are the previous methods for comparison re-implemented on the same base models that the paper uses? If inversion plus attention manipulations are used to help preserve the source motions, will the performance be improved? A more in-depth analysis should be conducted to figure out if the current inversion-free methods are indeed better than inversion-based methods for the same model. 

- **Runtime accounting is incomplete**
The paper does not provide runtime and resource consumption compared with other inversion-based methods and training-based methods.

### Questions
See "Weaknesses"

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an efficient optimization and inversion free method for video editing with flow based image generation models. The authors propose a novel vector field rectification technique to circumvent costly inversion. Structure and motion preserving initialization is used to preserve characteristics of the source video.

### Strengths
The paper introduces a novel contribution in vector field rectification. The comparative study demonstrates superior performance. Comparison with FlowEdit in the Appendix is also useful to place this paper.

### Weaknesses
Background preservation seems to be lacking in many examples, eg. person cycling (shirt color change example), woman walking in park example. etc. What is causing this inconsistency? What can be done to fix it?

Since efficiency is a major contribution and motivation in this paper, Table 1 a) should include comparison of time taken to make the edit with the baselines.

More video examples of edits would be good.

### Questions
None

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
This paper attacks the “first-frame-conditioned” video-editing problem: given a source video and one edited first frame, generate a new video that (i) keeps the original structure & motion and (ii) respects the edit. Prior work either pays the price of diffusion inversion (heavy, noisy) or engineers attention maps for every new I2V backbone (fragile). The authors propose IF-V2V, a training-free, inversion-free, plug-and-play wrapper for any flow-matching I2V model. Key ideas: (1) run two ODE trajectories (source & target) in parallel and rectify the target vector field with the ground-truth source residual (VFR-SD); (2) inject structure & motion priors into the initial noise and the conditioning frame (SMPI); (3) cache the residual when its variation is small (D-Cache). Extensive experiments on Wan2.1 and HunyuanVideo show consistent gains over Videoshop, AnyV2V and VACE in visual quality, temporal consistency and human rating.

### Strengths
- First systematic attempt to bring inversion-free editing into flow-based I2V models; introduces the notion of “sample-deviation field” to transport source-specific residuals to the target ODE.
- Rigorous exposition: derives VFR-SD under the linear-Gaussian transition assumption and shows λ=1 performs optimal-transport on transition distributions.
- Solid empirical protocol: 40 DAVIS + in-the-wild clips, 4 task types, 6 metrics, ablations for every component, cross-backbone (Wan2.1 & HunyuanVideo) and cross-solver (Euler & UniPC) tests.
- Democratizes high-quality video editing: practitioners can combine the latest image-editor + I2V backbone with zero retraining.

### Weaknesses
- The paper presents three appealing components (VFR-SD, SMPI, D-Cache), but their interaction is described only intuitively; a unified objective or gradient-level view that shows whether the corrections reinforce or cancel one another is not provided.
- Table 1b shows very small margins on AS and TC; without standard errors or p-values it is unclear whether the bold-highlighted improvements exceed sampling noise.
- D-Cache gives a 25 % speed-up, yet the same table records a slight EFC drop; a concise Pareto curve (time vs. score) that clarifies this trade-off is absent.
- λ, β, α and δ are reported only as single optimal values; the searched grids and selection criterion are not given, limiting reproducibility.

### Questions
-Could you provide a joint-ablation (2×2×2) that shows how the three components combine—i.e., are the gains additive, multiplicative, or occasionally conflicting?
-Please run paired t-tests across clips and report 95 % confidence intervals for AS/TC/EFC to confirm which bold numbers are statistically meaningful.
-Would you plot a short Pareto frontier (wall-clock time vs. EFC/OVC) for several δ values so practitioners can choose an appropriate speed-quality operating point?
-Can you share the full hyper-parameter grids and the exact criterion (best EFC, best AEC, etc.) used to select λ, β, α and δ?

### Soundness
3

### Presentation
3

### Contribution
2
