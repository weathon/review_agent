# Learning Dynamic Stability Landscapes from Graph Topology

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
The robustness of synchronization is a central theme of the study of dynamical systems on networks. Typically one attempts to define a single stability index that characterizes the robustness of individual nodes to a class of perturbations. The dependence of a stability index on topology and system parameters can then be studied using network science or GNNs. Here we propose a novel upstream task, Stability Landscapes, that allows deriving many downstream stability indices.
To support this task, we release two computationally intensive datasets of 10,000 graphs each at 20 and 100 nodes with per-node landscape labels. The dynamics are given by a conceptual oscillator model that captures aspects of the synchronization behavior of power grids.
A compact graph neural network with a CNN decoder predicts these landscapes with about 85\% SSIM in distribution and 67\%  under a 20 to 100 size shift, and 65\%-73\% SSIM when going from the 100 node ensemble to realistic power grid topologies with 100-400 nodes.
This demonstrates that while basin landscapes are not suitable for study with conventional methods of network science, they are amenable to machine learning methods.
This suggests that there is considerable potential in the study of complex networked systems across biology, neuroscience, and power grids, to move beyond scalar stability indices.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors lay out a full, novel computational pipeline, from conceptual premises to data set generation to a first baseline model, for the problem of predicting the behavior of coupled oscillators under perturbation. They focus on the interesting case of understanding how individual nodes behave qualitatively (whether or not they remain synchronized with respect to a rotating reference frame) under the influence of perturbations to their phase and frequency. Instead of predicting the effect of a single perturbation, they investigate the more general problem of inferring an entire landscape of perturbation effects for all given nodes in different graph topologies. After generating a rather large data set of such landscapes, they propose a graph neural network architecture which performs well on the task and can generalize to real topologies taken from electric power grids.

### Strengths
1. The presentation of a full pipeline for perturbation prediction is impressive and valuable. Overall, I think the paper is very sound as is, but could be improved by addressing some of the questions and weaknesses discussed below. 


2. The concrete products of this study in the form of the dataset of stability landscapes and the baseline GNN encoder are useful and will help the community study these synchrony phenomena in the future.

### Weaknesses
1. The authors emphasize the need for stability landscapes holistically so that “safe and unsafe” regions in perturbation space can be predicted. I think in theory this is a very good idea. However, I notice that–at least for the case of the inertial Kuramoto model studied here–these landscapes have similar coarse structure: uniform zero vs full “eye structure” vs top eyelid vs bottom eyelid. I think it would be worthwhile for the authors to justify why predicting the full pointwise array of landscape probabilities is valuable compared to just predicting these coarse categories. Would it be possible to decode these categories from their GNN? This is not a weakness per se, but I think addressing this categorical structure in their data could help solidify the problem setting. For instance, we see some interesting stripes in the lower part of Fig 1a/b. Could the authors explain why predicting this fine-grained structure is interesting and maybe in parallel discuss if coarser categories are worth predicting too? 


2. I was a little confused by the nature of the second downstream task which “[predicts] the volume of the basin stability, a metric derived from the heatmaps.” I believe these are the results shown in Table 3, but it is not immediately clear from the text. Could the authors clarify what is the nature of this task, even if the details are in the supplement? It also seems to connect to my first point about predicting qualitative structure as opposed to fine-grained detail. 


3. I think that the focus on second-order Kuramoto is justified, and it is not worth exploring other models in extensive detail. However, I think it could be worthwhile to discuss extensions to other dynamics in the concluding sections. On its face, it seems like the method would work fine on other systems, but what challenges, if any, do the authors foresee, for example, if the landscapes or perturbations (maybe with higher order derivatives) become more complex?

### Questions
To summarize questions implicit in the weaknesses mentioned above: 

1. What is the value in predicting fine-grained vs coarse landscape structure? Can you discrete decode landscape categories? 

2. Can you clarify the nature of the the second downstream task which “[predicts] the volume of the basin stability, a metric derived from the heatmaps”?

3. Can you discuss extensions to other oscillator models and perturbations? Off the top of my head, maybe delay models, noise, higher-order interactions? (You don't have to discuss these examples specifically, but it would be good in general to read about challenges and opportunities in extending beyond the current case.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Stability Landscapes as an upstream task for node-level robustness in network synchronization, releasing two large datasets (10k graphs each at 20 and 100 nodes) generated from a conceptual power-grid oscillator model to derive many downstream stability indices.

### Strengths
- **Novel problem formulation**: Rather than predicting a single stability index, the paper proposes predicting a full landscape of stability across the perturbation space. This is conceptually interesting and could inspire related tasks such as node-level OOD detection or uncertainty quantification in stability analysis.
- **Substantial computational investment**: The datasets required approximately 500,000 CPU hours to generate, representing a significant resource contribution that could benefit the broader community.
- **Cross-distribution evaluation**: The evaluation includes both in-distribution performance and out-of-distribution generalization across graph sizes and to realistic power grid topologies, which is commendable.

### Weaknesses
- **Unclear practical advantage over scalar prediction**: The paper fails to demonstrate concrete scenarios where landscape prediction outperforms scalar SNBS prediction. Table 3 shows that deriving SNBS from predicted landscapes actually performs worse than direct SNBS prediction. Without clear use cases where the spatial structure of landscapes provides actionable insights beyond a single stability score, the motivation for this more complex task remains unconvincing. It is unlikely that practitioners would inspect heatmaps for each node qualitatively.
- **Overly restrictive experimental scope limits generalizability**: The evaluation is confined to a single oscillator model with fixed parameters, binary node features, and a specific perturbation and threshold. No parameter analysis is provided. This rigid setting raises serious questions about whether the approach generalizes to other coupled oscillator models, different parameter regimes, or other classes of networked dynamical systems. The claimed contribution to "complex networked systems across biology, neuroscience, and power grids" is not substantiated.
- **Methodological ambiguity in downsampling procedure**: The downsampling from 10,000 random perturbations per node to a 20×20 grid is poorly explained. Since the original perturbations form "a point cloud that differs from node to node in both location and density" (line 190), how does the discretization ensure consistent spatial correspondence? downsampled cells at the same grid index (m,n) may still correspond to different regions of the perturbation space across nodes.
- **Inadequate loss function and evaluation metrics for image prediction**: For image reconstruction tasks, L1 loss is more commonly employed than L2/MSE due to its robustness to outliers. More critically, evaluating heatmap quality solely with R² is insufficient. Standard image quality metrics such as PSNR, SSIM, or perceptual similarity should be reported to properly assess reconstruction fidelity.
- **Missing ablation study on heatmap resolution**: The choice of 20×20 discretization appears arbitrary and is not justified. The paper lacks any ablation study examining how performance varies with heatmap resolution (e.g., 10×10, 30×30, 40×40). This is essential to validate that the task formulation is robust, useful and to understand the trade-off between different granularity.
- **Insufficient baseline comparisons on realistic topologies**: Tables 4-5 show transfer to real power grids, but no comparison to alternative methods is provided. Given the claimed computational advantage over Monte Carlo sampling, the authors should benchmark against cheaper approximation methods or classical stability estimation techniques on these realistic networks. This omission is particularly concerning given the massive computational cost of dataset generation and training.

### Questions
See weakness section

### Soundness
3

### Presentation
2

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
This paper introduces the concept of Stability Landscapes, a upstream task for predicting the robustness of synchronization in dynamical systems on graphs. Instead of estimating a single scalar stability score per node (basin stability), the authors propose predicting full stability heatmaps that describe how node-level stability varies across perturbation conditions.

### Strengths
- The paper introduces stability landscapes as a new supervised learning task, generalizing beyond scalar stability measures.
- Two new datasets (10,000 graphs each) are released, generated from half a million CPU hours of Kuramoto simulations. This provides the research community with a valuable benchmark that combines topology, dynamics, and continuous-valued outputs.
- The method achieves ≈90% R² in-distribution, 66–70% under size-shift (20 - 100 nodes), and up to 86% on real-world power grid topologies, showing impressive zero-shot transfer across domains.

### Weaknesses
- The GNN–MLP architecture is quite simple and largely serves as a proof of concept. The innovation lies more in the dataset and task formulation than in methodological novelty.
- The work focuses exclusively on Kuramoto oscillator models. While this is a canonical setting, it limits the immediate applicability to broader classes of dynamical systems (biochemical, ecological, or chaotic dynamics).
- Although the landscapes provide richer information, the paper doesn’t explore what the learned embeddings represent physically, or how topological features influence predicted stability patterns.
- Despite claiming open release, the dataset generation process requires massive compute (∼500,000 CPU hours), making reproducibility difficult.
- Given that the encoder-decoder is trained on synthetic graphs, even though it generalizes reasonably, the robustness on heterogeneous, real-world network distributions (with different coupling constants, noise, or weighted edges) remains uncertain.

### Questions
See the points mentioned in the weaknesses.

- Could the Stability Landscape framework be extended to other classes of dynamical systems (chaotic oscillators or neural mass models)? How general is the proposed formulation?

- Why were simple decoders (MLPs) chosen instead of more expressive generative models (VAEs, CNNs, or diffusion-based decoders) for 2D heatmap reconstruction?

- What structural graph features most influence the predicted stability landscapes?

- How would the approach perform on much larger real grids (>1,000 nodes)? Are there computational bottlenecks or accuracy drop-offs with scale?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces "stability landscapes" as a novel upstream task for predicting per-node stability behavior in dynamical systems on networks. Instead of predicting scalar stability indices (SNBS), the authors propose predicting full 2D stability landscapes (20×20 heatmaps) that capture spatial patterns of stability across perturbation space. The work focuses on second-order Kuramoto oscillator networks and uses GNN-based encoder-decoder architectures, achieving ~90% R² in-distribution and 66-70% under size shift.

### Strengths
Originality 
  - Novel problem formulation: First work to predict full stability landscapes rather than scalar stability indices, representing a meaningful extension beyond prior SNBS prediction work
  - Creative task design: The graph-to-image prediction task is unique in the stability analysis domain and could inspire similar approaches in other fields
  - Theoretical contribution: Theorem 3.1 provides a clean mathematical relationship between pixel-wise MSE and downstream SNBS error, offering theoretical grounding for the approach

Quality

  - Substantial dataset effort: 500k CPU hours to generate datasets with 10k graphs each at two scales demonstrates significant computational investment
  - Multiple evaluation scenarios: Tests in-distribution, out-of-distribution (size shift), and zero-shot transfer to real power grids
  - Sound experimental methodology: Proper train/validation/test splits, multiple seeds, and statistical reporting

Clarity

  - Clear motivation: The progression from scalar to landscape prediction is well-motivated and easy to follow
  - Good mathematical exposition: Equations and notation are generally clear and consistent
  - Effective visualizations: Figure 1 effectively illustrates the concept, and landscape comparisons in Figures 4,6,7 are informative

Significance

  - Potential broader impact: Could influence how stability analysis is performed across multiple domains (power grids, neuroscience, biology)
  - Methodological advancement: Demonstrates feasibility of predicting complex spatial patterns from graph topology alone

### Weaknesses
Experimental Validation Gaps

  1. Missing critical baselines: No comparison to interpolation methods, PCA-based reconstruction, or modern generative models (VAEs, GANs, diffusion models) that are natural baselines for image generation tasks
  2. Limited architecture exploration: Only tests two GNN encoders with simple MLP decoders; no exploration of CNN decoders, attention
  mechanisms, or other architectures suited for image generation
  3. Inadequate evaluation metrics: Relies solely on R² which doesn't capture perceptual quality or spatial structure; missing image-specific
  metrics like SSIM, LPIPS, or FID

  Questionable Practical Utility

  1. No evidence of superior decision-making: Authors claim landscapes enable "multiple downstream tasks" but only demonstrate SNBS recovery, which often performs worse than direct SNBS prediction (Table 3: 82-89% vs 85-90%)
  2. Missing use case analysis: No concrete examples of decisions that would benefit from spatial landscape information vs scalar indices

  Technical Limitations

  1. Poor generalization: Significant performance drop from 90% to 66-70% R² under size shift, and high variance on real networks (32-86% R²) suggests limited practical reliability
  2. Arbitrary design choices: 20×20 discretization appears unmotivated; no sensitivity analysis or principled approach to grid resolution
  selection
  3. Information loss in preprocessing: Point cloud to grid conversion discards information and introduces artifacts, but this trade-off is
  not analyzed

  Scope Limitations

  1. Single dynamical model: Restricted to second-order Kuramoto oscillators; generalization to other dynamical systems unknown
  2. Limited network diversity: Despite claims of broad applicability, only tested on power grid topologies
  3. Computational efficiency unclear: Authors claim speed advantages but provide no timing comparisons or scalability analysis

### Questions
1. Computational efficiency: What are the actual runtime and memory comparisons between landscape prediction vs direct simulation vs direct SNBS prediction? This is crucial for practical adoption claims.
  2. Grid resolution sensitivity: How does performance vary with different grid resolutions (10×10, 30×30, etc.)? Why was 20×20 chosen
  specifically?
  3. Baseline comparisons: How does the approach compare to:
    - Simple interpolation of scattered simulation points?
    - PCA reconstruction from a subset of simulations?
    - Modern generative models trained on the same data?
  4. Practical utility demonstration: Can the authors provide specific examples where landscape spatial information leads to better
  engineering decisions than scalar indices?
  5. Failure mode analysis: When and why does the method fail? What network characteristics predict poor performance?
  6. Architecture justification: Why use simple MLPs instead of CNNs or other architectures designed for image generation? What's the
  performance gap?
  7. Real-world validation: Have actual power grid operators evaluated whether these landscapes provide useful insights for grid planning and operation?

### Soundness
2

### Presentation
3

### Contribution
2
