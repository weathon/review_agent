# FM4NPP: A Scaling Foundation Model for Nuclear and Particle Physics

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Large language models have revolutionized artificial intelligence by enabling large, generalizable models trained through self-supervision. This paradigm has inspired the development of scientific foundation models (FMs). However, applying this capability to experimental particle physics is challenging due to the sparse, spatially distributed nature of detector data, which differs dramatically from natural language. This work addresses if an FM for particle physics can scale and generalize across diverse tasks. We introduce a new dataset with more than 11 million particle collision events and a suite of downstream tasks and labeled data for evaluation. We propose a novel self-supervised training method for detector data and demonstrate its neural scalability with models that feature up to 188 million parameters. With frozen weights and task-specific adapters, this FM consistently outperforms baseline models across all downstream tasks. The performance also exhibits robust data-efficient adaptation. Further analysis reveals that the representations extracted by the FM are task-agnostic but can be specialized via a single linear mapping for different downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FM4NPP, a self-supervised foundation model for nuclear & particle physics (NPP) built on a Mamba (SSM) backbone. It introduces (i) a >10M-event simulated RHIC/sPHENIX TPC dataset and three downstream tasks (track finding, particle ID, noise tagging), (ii) a Hierarchical Raster Scan to serialize sparse detector spacepoints, and (iii) a k-Next-Nearest-Neighbor (k-NNN) autoregressive pretraining objective that predicts outward (in radius) neighbors to avoid order-leakage. With frozen FM features and lightweight adapters, the largest model (188M params) reportedly outperforms GNN/point-cloud baselines across tasks and shows clean scaling trends w.r.t. model/data/compute.

### Strengths
1. The paper targets an important gap: general, reusable representations for sparse detector data rather than bespoke task models. The two-stage “pretrain FM -> freeze ->  task adapters” design is well-motivated.
2. The Hierarchical Raster Scan balances global outward geometry with local track continuity; k-NNN aligns the self-supervised target with physics causality (increasing radius) rather than sequence order. The ablation table isolates benefits over next-token prediction and Hilbert curves.
3. Clear power-law trends for model/data/compute; the paper also documents hardware/iterations, optimizer, and $\mu$-param transfer, useful for reproducibility. Also, FM4NPP + small adapters beats GNN and 3D point-cloud baselines on ARI and efficiency.

### Weaknesses
1. The approach based on the proposed ordering. Although ablations compare Hierarchical Raster vs. Hilbert, it’s unclear how sensitive performance is to bin sizes, box aspect ratios, or detector layout changes (e.g., silicon layers, calorimeters), and whether learned positional encodings could reduce this hand-crafted dependency.
2. k-NNN chooses neighbors with a larger radius (future in $r$). That choice encodes a physics prior (outward propagation) but may under-represent curling/looping tracks, secondaries, or back-scattering. Ablations vary $k$ but not the neighborhood definition (e.g., cone in $\eta,\phi, r$).
3. All tasks are TPC-centric, point-level segmentations. For a foundation model, it would be compelling to include fusion tasks (TPC+EMCal), rare-event tagging, vertexing, or particle-flow style reconstruction, even as small-scale probes.

### Questions
1. Have you evaluated FM4NPP embeddings on real sPHENIX runs (even small samples)? How does sim-trained FM perform zero-shot/adapter-shot on real data, and what domain-gap mitigation (calibration, augmentation, feature normalization) is most effective?
2. How do different grid resolutions $r \times \eta \times \phi$, binning strategies, or non-radial intra-box sortings affect pretraining loss and downstream ARI? Could a learnable/local ordering, or permutation-invariant SSM input layer, match or exceed Hierarchical Raster?
3. Why constrain neighbors strictly to $r_j > r_i$ rather than a cone or helix-aware neighborhood? Did you try multi-horizon targets mixing local and mid-range neighbors, or contrastive variants that avoid explicit regression?
4. For tracking, why a DETR-style query decoder over bipartite clustering in embedding space? For PID/noise, did you test slightly larger heads or lightweight decoders (e.g., 2–4 self-attention layers) to probe head-capacity vs. FM-quality trade-offs?

### Soundness
3

### Presentation
4

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
This work introduces a foundation model tailored towards nuclear and particle physics. The authors develop a large scale open dataset with 10 million events and benchmark on three downstream tasks. The authors develop a self-supervised training methodology for their foundation models, and present benchmarks on downstream tasks in addition to scaling behaviors for their foundation models. They demonstrate improved performance of their model over baselines across all tasks.

### Strengths
* The authors present a well motivated problem that they address with a multi-step comprehensive framework 
* The self-supervised learning objective is physics motivated, and intuitively works with the detector setup
* Very thorough benchmarking and ablation studies show improved performance for all pieces of the foundation model
* Performance on downstream task show greatly increased performance over baselines
* Scaling curves show increased performance for increased flops/parameter count, indicating scalability of the model
* Shows greatly increased performance against state of the art models
* Shows that the self-supervised learning technique improves performance on downstream tasks

### Weaknesses
* Does not justify choice of mamba over linear transformer models, or other state space model alternatives
* Unclear how the hierarchical raster scan would impact the efficiency
* No benchmarks against efficient transformer baselines such as HEP-T (locality sensitive hashing transformer for high energy physics applications, https://arxiv.org/abs/2402.12535), which performs very well on trackML datasets
* For PID, also missing comparisons to dedicated physics ML algorithms, like MLPF (https://arxiv.org/abs/2503.00131) or HGPF (https://arxiv.org/abs/2410.23236)
* Does not define the model architecture/construction (number of layers, parameters per layer etc)
* There are no code or pretrained weights available
* Unclear of pretraining + fine-tuning outperforms dedicated models trained from scratch; Most other literature on FMs in particle physics do not claim this and instead focus on data efficiency (i.e. better performance when training on smaller datasets)

### Questions
* Why choose mamba over other linear transformers (deltanet, GLA..) and state space models? 
* How long are the sequence lengths for the input? State space models struggle with extremely long context. 
* Does the hierarchical raster scan have a strong impact on the amount of compute?
* What is the full model architecture of the foundation model?
* Can the code and models be made public?
* Do you actually claim that self-supervised pre-training + fine-tuning outperforms a dedicated model trained from scratch on the downstream task in Fig. 5 (b)? Or are the architectures different? If you extend the axis, would Adaptor only eventually converge to the FM performance, as is seen in other FM papers in particle physics?

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
The paper introduces FM4NPP, a scientific foundation model for particle physics that attempts to address existing scaling challenges outside of natural language due to, e.g., the sparse nature of detector data, difficulties in serializing 3D data. The model is trained on simulated collisions under conditions of the sPHENIX experiment and evaluated on three core downstream tasks, mostly outperforming non-FM baselines on key metrics. FM4NPP is also shown to observe scaling laws at high parameter counts, and may present a useful paradigm for scientific FMs in other domains.

### Strengths
- The paper is well-organized, includes a comprehensive literature review, and provides high quality figures/diagrams.
- The paper's stated contributions are clear and address a difficult, high-impact problem in the NPP domain. The approach potentially lays the groundwork for tackling broader challenges relevant to scientific foundation models beyond particle physics settings.
- The reported evaluation is extensive, covering a variety of important dimensions that help position the model's utility and behavior. For instance, the model is compared to several baseline methods on the downstream tasks (highlighting its comparative advantages), key ablations are reported (justifying architectural decisions), and different model sizes are evaluated, highlighting scaling laws.
- I appreciate the discussion style of Section 5.3, which sets up several key questions relevant to practitioners seeking to leverage FM4NPP on new tasks.

### Weaknesses
- I find the core positioning of the paper to be somewhat difficult to pin down. There is a clear focus on particle physics as the primary domain of interest, but the proposed training scheme for scaling and adapting FMs to downstream tasks is very general. To fully demonstrate the generality of the scheme, more (scientific) domains would need to be evaluated to verify the approach transfers well to other settings (helping substantiate the stated hypothesis that FMs encode task-agnostic representations useful for downstream tasks).

  Note that this isn't to take away from the model's performance on this task, but simply that the methodology leans more general-purpose while the evaluation focuses entirely on a single domain/dataset. 
- The downstream tasks used for evaluation appear limited in their variety. In particular, they seem to test for similar kinds of model understanding, which is to say classifying spatiotemporally connected groups of points in the particle space. For instance, given that the model performs well on track finding, it's not surprising that it would perform similarly well on particle identification or noise tagging (as these tasks, naively, simply re-group the tracks). Including downstream tasks like forecasting would be useful for demonstrating broader model understanding, more explicitly highlighting its ability to understand/extrapolate particle dynamics.
- It is ultimately quite difficult to asses how well the trained model fills the role of a foundation model. This point is made in tandem with the first two, i.e., that it's hard to gauge general model understanding provided the single dataset and task suite. What's concretely demonstrated is that FM4NPP is an effective embedding model for particles under conditions of the sPHENIX experiment, and large parameter counts can help with downstream classification tasks. But the jump to labeling FM4NPP as a foundation model is hard to justify without stronger evidence of its generality, such as more diverse tests of model understanding and/or evaluation under different conditions.

### Questions
- Were any evaluations run using the adapter models under a different (non-FM4NPP) spacepoint embedding model? This might be a distinct way to isolate the performance of the proposed FM architecture.
- As presented with the three downstream tasks, can this approach be trained end-to-end? The explicit adapter-based approach seems to be the most flexible route and keeps the backbone in more of an embedding role, but given the overlap in the three tasks it seems plausible that these could fully captured end-to-end (potentially internalizing/sharing knowledge across tasks previously isolated within the adapter models).
- Were any other kinds of downstream tasks tested (e.g., forecasting)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript presents a foundation model for nuclear and particle physics, trained on proton-proton (p+p) collision data at a center-of-mass energy √s = 200 GeV. The proposed architecture utilizes the Mamba model as its backbone and is pretrained using a novel k-Next-Nearest-Neighbor prediction objective. The authors evaluate the model's performance on three downstream tasks: Track Finding, Particle Identification, and Noise Tagging.

### Strengths
The authors present detailed technical considerations for building the foundation model, though there are shortcomings on specific design and experiment choices.

### Weaknesses
- The manuscript does not adequately situate its contribution within the emerging landscape of foundation models for high-energy physics. It fails to discuss or compare against other recent efforts (e.g., models based on transformers, graph networks, or other architectures applied to similar data).
- The datasets used appear limited in scope, focusing on a specific collision system and energy. How does this restricted training data affect the model's validity and utility as a general-purpose foundation model for the broader field of nuclear and particle physics?

### Questions
- The model is positioned as a foundation for particle physics, yet it is trained only on RHIC data. Given that the LHC represents the forefront of particle physics, why was the model not trained or validated on LHC data to demonstrate its utility as a true foundation model for the entire field? The authors must justify this fundamental limitation in scope and discuss the model's generalizability to higher-energy, more complex LHC environments.
-  Can the authors quantify the computational overhead of the k-NNN objective compared to a baseline, and justify how the performance benefits warrant this increased expense?

### Soundness
2

### Presentation
2

### Contribution
2
