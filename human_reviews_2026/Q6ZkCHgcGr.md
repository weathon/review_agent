# Generative Point Tracking with Flow Matching

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Tracking a point through a video can be a challenging task due to uncertainty arising from visual obfuscations, such as appearance changes and occlusions. Although current state-of-the-art discriminative models excel in regressing long-term point trajectory estimates—even through occlusions—they are limited to regressing to a mean (or mode) in the presence of uncertainty, and fail to capture multi-modality. To overcome this limitation, we introduce Generative Point Tracker (GenPT), a generative framework for modelling multi-modal trajectories. GenPT is trained with a novel flow matching formulation that combines the iterative refinement of discriminative trackers, a window-dependent prior for cross-window consistency, and a variance schedule tuned specifically for point coordinates. We show how our model's generative capabilities can be leveraged to improve point trajectory estimates by utilizing a best-first search strategy on generated samples during inference, guided by the model's own confidence of its predictions. Empirically, we evaluate GenPT against the current state of the art on the standard PointOdyssey, Dynamic Replica, and TAP-Vid benchmarks. Further, we introduce a TAP-Vid variant with additional occlusions to assess occluded point tracking performance and highlight our model's ability to capture multi-modality. GenPT is capable of capturing the multi-modality in point trajectories, which translates to state-of-the-art tracking accuracy on occluded points, while maintaining competitive tracking accuracy on visible points compared to extant discriminative point trackers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Generative Point Tracker (GenPT), which reinterprets the iterative optimization paradigm of many of the modern point trackers (such as PIPs, CoTracker3, LocoTrack) as a form of flow matching. The authors claim to bridge point tracking and generative modeling by formulating correspondence estimation as learning a continuous denoising process that maps perturbed query coordinates to target positions. The framework introduces Gaussian perturbations to query points, defines an auxiliary velocity field trained with a flow-matching objective, and evaluates both single-sample and Best-of-N inference strategies. The paper also includes a multi-template tracking extension, performing patch-wise correspondence aggregation inspired by LocoTrack.

### Strengths
- The paper tackles a genuine limitation of current discriminative point trackers, their inability to represent uncertainty and multimodal hypotheses in ambiguous or occluded regions.
- The authors provide comprehensive comparisons across several datasets

### Weaknesses
### Lack of generative insight
Although the paper positions itself as a generative reformulation of tracking, the actual mechanism remains deterministic iterative optimization under Gaussian perturbation, not a generative process.
- In generative models (diffusion or rectified flow), the model learns to map **pure noise --> data samples**, learning meaningful dynamics along a linear trajectory in data space.
- In GenPT, the model learns **query + noise --> correspondence**, where the starting point already encodes the spatial identity of the tracked feature. The added noise does not represent a generative latent, only a small random offset to an already meaningful input.
- Thus, the flow is effectively a regularized refinement of supervised training, not a learned stochastic trajectory from noise to data.
- Equation 6 changes the standard CoTracker initialization when $l=0$; increasing $l$ simply reduces supervision strength, not adding new semantics.
- In essence, GenPT = CoTracker3 + Gaussian perturbation + renaming of loss, rather than a true flow-matching model.

### Evaluation issues
- The Best-of-N performance gains could stem entirely from multiple inference-time noise injections, not a learned generative diversity. No comparison to a simple CoTracker3 + random perturbation at inference baseline is provided.
- The empirical improvements are small and inconsistent, and the method fails to demonstrate meaningful benefits in standard single-sample evaluation.

### Presentation and clarity
- The notation is excessive, making the method difficult to follow.

### Overall
While the paper explores a creative framing of point tracking via flow matching, it does not deliver genuine generative insight or methodological novelty. The proposed approach is functionally equivalent to noisy supervised fine-tuning of existing trackers, with only minor differences in objective formulation. The results and framing overstate the impact relative to the simplicity of the actual change.

### Questions
See weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GenPT, a generative point tracker that models multi-modal trajectories for long-range point tracking in videos. Unlike discriminative trackers that regress a single mean and thus struggle under occlusions or appearance changes, GenPT trains a likelihood-based model with flow matching and three key modifications: (i) iterative refinement within each step, (ii) a window-dependent prior, and (iii) a variance schedule tailored to point coordinates. GenPT achieves competitive visible-point accuracy and state-of-the-art occluded-point accuracy.

### Strengths
1. This paper introduces the first generative point tracker trained using a modified flow-matching objective for trajectories, extending generative modeling concepts to the task of point tracking.
2. The authors design three key modules: iterative refinement, window-dependent prior, and variance schedule. These components are well-motivated and thoroughly ablated.

### Weaknesses
1. Point tracking is inherently a deterministic problem, so a multi-modal approach may not be well-suited for this task.
2. The improvements of this model mainly target occluded points. However, the objective function used in models such as CoTracker3 or other similar approaches is typically L=Huber_loss(predicted point,ground truth point)×is_visible_gt(this point) 
In other words, these models are not explicitly designed to predict occluded points.
3. The greedy search strategy requires running the algorithm five times, which makes it computationally expensive and time-consuming.

### Questions
1. Could you train CoTracker3 with the objective L=Huber_loss(predicted point,ground truth point) and evaluate how much improvement it achieves on occluded points?
2. Do the failure cases tend to cluster around homogeneous textures or repetitive patterns?

### Soundness
3

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
4

### Summary
The paper introduces the Generative Point Tracker (GenPT), the first framework to address the Point Tracking problem using a Generative Model based on Flow Matching. Existing Discriminative Models struggle with uncertainty (e.g., occlusion) as they regress to a single mean estimate. GenPT overcomes this by modeling multi-modal trajectories, enabling it to sample several plausible paths in ambiguous situations.

### Strengths
- GenPT can model and sample from multiple plausible trajectory candidates, particularly when tracking uncertainty is high due to occlusion. This translates directly to state-of-the-art tracking accuracy on occluded points.
- The model effectively transitions between probabilistic and quasi-deterministic behavior. While always generative, its prediction variance tightly contracts (becoming nearly deterministic) when the tracked point is clearly visible and uniquely identifiable.

### Weaknesses
- There is a substantial and recurring performance gap between the Oracle scores (the model's maximum potential) and the Greedy scores (the model's actual performance when relying on its confidence). This fundamental disconnect means the model is poor at judging the quality of the trajectories it generates, limiting the real-world utility of its multi-modality.
- The advertised speed advantage (2x faster than CoTracker3) is strictly limited to generating a single sample. To achieve the demonstrated improvements in accuracy, the 'Best-of-N' sampling method must be used. This process rapidly increases the runtime, often making GenPT slower than its discriminative counterparts, thus sacrificing one of its key efficiency claims for practical performance.
- A significant portion of GenPT's SOTA claim relies on the custom TAP-Vid Sliding Occluder Benchmark introduced by the authors, which is specifically designed to highlight its strength in occlusion handling. While useful, the novelty of the benchmark means the competitive results require independent verification across established, universally adopted benchmarks.

### Questions
Have the authors explored an adaptive sampling strategy where multiple samples ('Best-of-N') are only generated in windows where the model's initial predicted uncertainty (variance/confidence) is above a certain threshold, rather than sampling N times in every window?

### Soundness
3

### Presentation
3

### Contribution
3
