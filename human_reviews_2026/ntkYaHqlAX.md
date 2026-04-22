# Global Sharpness-Aware Minimization Is Suboptimal in Domain Generalization: Towards Individual Sharpness-Aware Minimization

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Domain generalization (DG) aims to learn models that perform well on unseen target domains by training on multiple source domains.  
Sharpness-Aware Minimization (SAM), known for finding flat minima that improve generalization, has therefore been widely adopted in DG.  
However, we argue that the prevailing approach of applying SAM to the aggregated loss for domain generalization is fundamentally suboptimal. This ``aggregated sharpness'' objective can be deceptive, leading to convergence to fake flat minima where the total loss surface is flat, but the underlying per-domain landscapes remain sharp. To establish a more principled objective, we analyze a worst-case risk formulation that reflects the true nature of DG. Our analysis reveals that per-domain sharpness provides a valid upper bound on this risk, while aggregated sharpness does not, making it a more theoretically grounded target for robust domain generalization. Motivated by this, we propose \textcolor{blue}{\textit{Domain-wise Gradual SAM (DGSAM)}}, which applies gradual, domain-wise perturbations to effectively control per-domain sharpness in a computationally efficient manner. Extensive experiments demonstrate that DGSAM not only improves average accuracy but also reduces performance variance across domains, while incurring less computational overhead than SAM.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents an idea of using individualized SAM to mitigate the domain generalization problem. Authors call the previous SAM approach to be "global", which I am uncomfortable because SAM is originally an idea of local flat minima (having said that I understand the authors claim and I concur that "aggregated" loss can be deceiving flat minima). They make an individualaity to represent a certain domain. Then, they present a method named DGSAM to control the individual sharpness. Experiments support the claim well.

### Strengths
1. Very nice problem discovery of the aggregated loss from the SAM perspective.
2. Good principled approach to define the domain generalization and its relation toward the individual sharpness
3. Necessary proofs are all provided. The defined individual sharpness will reduce the domain generalization errors. Stationary aspect, and its derived optimization approach path. 

Classic problem definition and solving it.

### Weaknesses
1.
I am very uncomfortable with the terminology that they defined or used.
As I mentioned earlier in the summary, all SAM approaches assume the parameters will be favored if they are located at the flat minima. However, this flat minima in the parameter space is always epsilon small, so SAM is always localized approach to a certain extent controlled by the epsilon. If authors agree with this aspect, then they would agree calling "global sharpness" is in its contradiction. What they are really pointing out is "the aggregated flat-minima loss surface over the parameter space". I would rather use "aggregation" instead of using a word "global".

2.
"Decreased-overhead" is your methodology name. I partially agree that the computational requirement could be reduced in the line of SAM researches. Having said that, I don't think that the overhead decrement would be your key contribution throughout the paper. Your key contribution is treating the "domain-specific" parameter space perturbation before "domain-aggregation" (yes. I don't like your wording 'individual' either), whereas the previous approaches have been "domain-aggregated" parameter space perturbation. Does this reversed process reduce the overhead? Could be. Is it the main-theme? No.

### Questions
I don't have much question on this paper. I think that I understand enough to see the merit of this paper. I would like to get answers from my weakness section.

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
This work examines how Sharpness-Aware Minimization (SAM) should be applied in the context of domain generalization (DG), and it challenges the conventional “global” use of SAM on aggregated training loss. The authors point out that simply finding a flat minimum for the average loss over all source domains can be misleading: it may produce a “fake flat” solution that appears robust overall but still has sharp (high-curvature) loss landscapes on individual domains, leaving the model vulnerable to domain-specific shifts.

To address this, they analyze a worst-case (adversarial) risk formulation for DG and show theoretically that minimizing individual-domain sharpness provides an upper bound on this worst-case risk, whereas minimizing global sharpness does not. In other words, each source domain’s loss landscape needs to be flat for true robustness, not just the combined loss. 

Building on this insight, the paper proposes DGSAM (Decreased-overhead Gradual SAM), which explicitly targets sharpness on a per-domain basis while keeping computational cost manageable.

### Strengths
Empirical results on five standard DG benchmarks show that DGSAM achieves better overall accuracy and significantly lower performance variance across domains compared to both standard training and globally-applied SAM. 

Models trained with DGSAM are consistently more robust to unseen target domains, indicating that the individual sharpness objective indeed translates to improved domain generalization. 

Moreover, DGSAM is shown to be computationally efficient – it incurs less overhead than traditional SAM (which doubles the compute) – and scales to large architectures like Vision Transformers. 

The paper’s contribution is notable in reframing SAM for multi-domain settings and providing both a theoretical justification and a practical algorithm that improves robustness.

### Weaknesses
DGSAM adds algorithmic complexity by requiring domain-specific updates (which could scale in cost with the number of domains), but the authors claim that they mitigate this with their gradual update scheme. Would this also fit with the case when the number of domains get really high (about 100 ~ )?

It would be beneficial to discuss more about related SAM works.
- https://arxiv.org/abs/2410.14802 : Discussion about data-responsive regularization, and why still per-domain sharpness is required?
- https://arxiv.org/abs/2403.07329 : How DGSAM differs from UDIM, when UDIM tries to generalize toward unseen domain?

I also want authors to measure "the zeroth-order sharpness result at converged minima" not only compared to original SAM, and other algorithms. To more precisely compare the impact of sharpness.

I think that this method is somewhat incremental from existing methods, because there were heavy-amount of SAM variants for domain generalization. But, if questions above are treated well, i will change my score.

### Questions
DIscussed in weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that the common practice of applying SAm to the global loss across source domains (in the context of DG) is suboptimal, i.e., it can lead models toward fake flat minima that remain sharp on individual domains. The authors introduce an average worst-case risk formulation and prove that individual sharpness per-domain yields a valid upper bound to this risk, whereas global sharpness does not. In response, the authors propose DGSAM, a gradual, domain-wise perturbation method that controls individual sharpness with lower cost than SAM and improves both average acc. and cross-domain variance.

### Strengths
Thank you for your submission, I enjoyed reading the paper and the fresh ideas on generalization from the loss landscape perspective. Below I have listed some aspects of the paper I've appreciated. 

- The paper points a conceptual flaw in how SAM has been ported to DG: "optimizing flatness of all the domains combined is not the same as ensuring generalization for each domains shift". From that perspective, the authors' idea that "individual domain sharpness is the right surrogate for DG" is sound and convincing.
- The proposed algorithm DGSAM aligns well with the problem formulation, and its effectiveness is supported through multiple empirical gains over a number of standard DG benchmarks. Also, the gradient re-use is notable, its efficiency was shown empirically (Sec. 5.4).  
- From my understanding, this paper is an extension of previous works in two key ways: (1) The authors suggest that minimizing the average of per‑domain sharpness provides a valid upper bound, while minimizing the global (aggregated) sharpness does not guarantee flat minima, and thus in improving generaliztion. Prior works do not explicitly mention the issue of fake flat minima. (2) It proposes a sequential, domain‑wise perturbation scheme that reuses gradients, so each domain’s loss surface is explicitly flattened.
- Overall, the paper was easy to read and the message was clear. The theoretical analysis was also sound (Sec.3) and in accordance with the previous literature.

### Weaknesses
- Positioning: While the paper aims to address an important, yet often overlooked issue in applying SAM to DG (or approaching generalization from the loss landscape perspective), it is still close to previous works (SAGM, ISAM, DISAM) that are aware of the issues in the naive application of SAM. Each of these works modifies the SAM objective to address specific deficiencies (e.g., inaccurate sharpness measures and gradient conflicts -- ISAM, inconsistent convergence across domains -- DISAM). Although they are cited and included among baselines, the paper would largely benefit from a sharper positioning. 
    - For instance, the idea that 'global sharpness' and 'per-domain sharpness' may not align was previously observed in Le et al. (2024). Although the paper was cited in line 484, we believe that their observation should be further noted as it aligns with the core idea of the paper. Similarly, the global vs. local sharpness/flatness idea was also studied in the federated learning literature [1].

- Cost Measure: A minor one, but in the paper, per-iteration gradient counts are reported (Sec 5.4), but end-to-end wall-clock, FLOPs, and peak memory comparisons are missing. Could the authors provide this? Again, this is a minor suggestion.

- Statistical Stability: Also a minor one, but in the camera-ready version, we suggest the authors to provide the average performance and standard error across more than 3 runs.

- Sharpness: In Sec 5.3 (and Tab. 3), the zeroth-order sharpness is measured to show that DGSAM can effectively reach flat minima. To my understanding, the zeroth-order sharpness refers to the maximal loss within the perturbed neighborhood. While they are a useful objective, they are still limited proxies that have distinct limitations [2,3]. In the generalization literature, different metrics are also commonly used (e.g., the largest eigenvalue of the Hessian), owing to their theoretical implications. 
    - In response, I believe that the authors should supplement their analysis with additional diagnostics. e.g., reporting the maximum Hessian eigenvalue or trace per domain (or at least proxies such as top‑eigenvalue estimates) would provide stronger evidence that the method genuinely finds flatter minima.

***
### Reference

[1] Caldarola et al., Beyond Local Sharpness: Communication-Efficient Global Sharpness-aware Minimization for Federated Learning, CVPR, 2025.

[2] Zhuang et al., Surrogate gap minimization improves sharpness-aware training, ICLR, 2022.

[3] Bian et al., Make Continual Learning Stronger via C-Flat, NeurIPS, 2024.

### Questions
- Expansion: One small question is whether the method can also be applied to single-source settings (Single-source Domain Generalization). My guess is that it wouldn't work (simply out of scope!) and would collapse to SAM, unless there are simulated (commonly augmented) domains. I acknowledge that the paper focuses on multi-domain settings, and this question is purely out of curiosity.

- Ablation Study: I'm interested in several components of the method and their effect on the performance gains. For instance, (1) what happens if the domain order is fixed, instead of being random (Line 3 in Algorithm 1)? (2) re-using vs. not re-using the ascent gradients. 

Please refer to the Weaknesses section for the questions. I'm mostly interested in the Sharpness measure and the Ablation study.

### Soundness
3

### Presentation
3

### Contribution
2
