# Harnessing Optimization Dynamics for Curvature-Informed Model Merging

- Decision: Reject
- Scores: 6, 4, 8, 6

## Abstract
Model merging is an effective strategy for composing capabilities in large language models without the need for costly joint retraining. We study this process in the supervised fine-tuning (SFT) stage, consolidating multiple checkpoints specialized for distinct capabilities (e.g., math, coding, and precise instruction following) into a single model. First, we introduce Optimization Trajectory Aware (OTA) Merging, a curvature-aware method for mitigating task interference that uses optimizer second-moment statistics as a diagonal curvature proxy to first prune the task vector with our Fast Fisher Grafting (FFG) technique and then reweight the pruned vector. When merging diverse, capability-based checkpoints, OTA improves the merged model's performance over strong baseline methods, as evaluated on unseen capability-based benchmarks. Second, we conduct a comprehensive, theoretically-inspired empirical analysis to explain the effectiveness of OTA. Our analysis surprisingly reveals that FFG implicitly induces a layer- and role-wise aware pruning mechanism that is capable of maintaining fine-tuning performance at much more aggressive pruning ratios compared to magnitude pruning and that exhibits interpretable task localization properties. Third, an extensive comparison of our curvature proxy across capability checkpoints shows that experts converge to a basin with substantial curvature similarity, offering a novel lens on why simple linear merging can be effective in practice. This result further strengthens our ablation study, showing that FFG is critical for merging performance. Finally, we develop a memory-light variant of OTA that efficiently compresses the second moments, mitigating the additional storage requirements of our method and improving scalability. We make all code, training and evaluation scripts, visualization artifacts, and capability-specific SFT checkpoints accessible through an anonymized repository at \url{https://github.com/anon123ota-dotcom/ota-ffg}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical problem of model merging in LLMs, specifically focusing on integrating capabilities from multiple SFT  expert checkpoints into a single, high-performing model. The core contribution is the introduction of the OTA Merging framework. This method cleverly leverages the second-moment statistics from adaptive optimizers (Adam) as a computationally efficient proxy for diagonal curvature. It consists of two stages: FFG for curvature-aware pruning of the task vector to mitigate interference, followed by a curvature-informed aggregation. The paper demonstrates strong empirical results, outperforming several baselines。

### Strengths
Strengths

1.  The approach of using stored optimizer second-moment statistics as a proxy for curvature is highly practical and memory-efficient. Unlike traditional Fisher-information methods, which require additional forward/backward passes on a dataset, OTA-FFG harnesses information already computed during training, making it much more scalable and easier to apply in practice.
2.  The proposed OTA-FFG method significantly improves performance when merging diverse, capability-based checkpoints, establishing a new SOTA baseline against strong competitors.
3.  The paper excels in its analysis, providing critical new insights into the model merging and fine-tuning landscape:
    - FFG Mechanism: The analysis effectively demonstrates that FFG acts as an efficient denoiser, enabling performance maintenance even at highly aggressive pruning ratios.
    - Task Localization: The work uncovers a highly interpretable, layer- and role-wise task localization property induced by FFG (e.g., pruning Query/Key weights more aggressively than Value/Output/FFN), which is a valuable finding for understanding how task-specific knowledge is stored.
    - Shared Curvature Geometry: The most significant theoretical finding is the evidence of substantial shared curvature similarity across specialized checkpoints, offering a compelling new explanation for the surprising effectiveness of simple linear merging methods.

### Weaknesses
1. Although effective empirically, using a diagonal curvature proxy is a simplification of the true, full Hessian or Fisher Information Matrix. The theoretical underpinnings could be strengthened by discussing potential limitations arising from ignoring off-diagonal curvature components, particularly in highly non-convex loss landscapes.

2. The paper primarily compares the merged model against other merging methods. A clear comparison of the final merged model's absolute performance against strong SFT baselines (i.e., the results achieved by training with Tulu-SFT) doesn't provide a stronger guarantee of the merged model's overall efficacy, which currently raises some concerns. Analyzing the effect of merging intermediate checkpoints from an SFT training process (e.g., from the Tulu-SFT training trajectory) using the proposed OTA method could provide valuable additional insights.

3. The experiments are exclusively conducted on the Llama 3.1 architecture. To fully establish the generalizability of the findings and the robustness of the OTA framework, verification across diverse transformer families (e.g., Mistral, Qwen2.5/3) is strongly recommended. The current limited architectural scope reduces the overall persuasiveness of the method's broad applicability.

### Questions
Please refer to the weaknesses.

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
3

### Summary
This paper introduces "Optimization Trajectory Aware (OTA) Merging," a framework for efficiently merging fine-tuned language models without retraining. OTA uses adaptive optimizer statistics to approximate loss curvature and operates in three stages: Fast Fisher Grafting (FFG) identifies task-critical parameters by pruning low-saliency updates; curvature-aware aggregation merges expert subnetworks; and rank-1 compression reduces storage overhead. Empirical results show OTA outperforms baselines, and the paper proposes a "shared curvature geometry" theory explaining the effectiveness of linear averaging.

### Strengths
- The paper reframes model merging as interference mitigation and creatively repurposes Adam's second-moment statistics as a cost-free curvature proxy, with rigorous empirical validation and insightful ablations supporting its claims.
- The work is well-written with compelling motivation, logical methodology, and effective visualizations that clearly communicate complex concepts.
- It introduces a scalable state-of-the-art method while providing novel insights into fine-tuning dynamics and loss landscape geometry that will impact future model composition research.

### Weaknesses
1. Limited Empirical Validation: Evaluation is confined to Llama-3.1-8B only. The generalizability of the curvature-aware framework and shared curvature hypothesis to different scales or architectures remains unvalidated.
2. Limited Novelty in Compression: The rank-1 approximation is directly adopted from AdaFactor without methodological innovation. Empirical validation includes only one table entry, lacking deeper analysis of this component's effectiveness.
3. No Scalability Analysis: The paper claims to mitigate interference but only tests five individual tasks. It fails to demonstrate how performance degrades as the number of merged models increases, leaving the method's scalability unproven.
4. Impractical Dependency on Training Artifacts: The method requires Adam's second-moment statistics, which are rarely distributed with public models. This undermines model merging's key advantage as a post-hoc technique and restricts applicability to users controlling the full training pipeline.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes to use optimizer statistics particularly second-order moment from ADAM as a cheap proxy for loss curvature, which can be used to guide expert merging. Proposed approach has 3 stages: Fast Fisher Grafting, which uses stored optimizer stats to find and keep the most important parameter updates per expert. Then curvature-aware aggregation, which merges the pruned task vectors using weighted averaging, then rank-1 compression to store second moments cheaply. The authors conduct experiments on a set of SFT-ed experts based on Llama-3.1-8B and show better avg task performance compared to baselines

### Strengths
1. The approach is novel; I like the idea of turning optimizer second moments into a curvature proxy.
2. Strong empirical validation: Comprehensive experiments on diverse reasoning and coding benchmarks, with clear ablations.
3. The analysis is insightful and shows where and how different skills are encoded (layer- and role-specific patterns).
4. Practical: The AdaFactor compression is a nice touch and is a pragmatic step toward real-world use cases.

### Weaknesses
1. One major issues with the paper is that it assumes access to the optimizer stats for the finetuned experts. This severely limited the applicability to off-the-shelf models or experts. 

2. The approach is fairly complex with three different stages, which adds implementation and compute overhead compared to simple linear merges. 

3. No runtime or compute comparisons: Merging cost and efficiency relative to other methods could be quantifier and studied more clearly. 

4. Interpretability claims are questionable: the task-localization visualizations are compelling but somewhat anecdotal. Some qunatitatve validation would support them.

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
This paper proposes **OTA Merging**, a method for combining multiple fine-tuned LLMs using Adam optimizer's second-moment statistics as a cheap curvature proxy. The approach has two stages: (1) **Fast Fisher Grafting (FFG)** prunes task vectors by reverting low-saliency parameters to base model values using curvature-weighted scores, and (2) curvature-aware aggregation merges the pruned vectors. Experiments merge five capability-specific Llama-3.1-8B models (math, coding, instruction-following, knowledge, commonsense), showing improvements over baselines like TIES and Fisher Merging. The paper also reveals that FFG induces interpretable structured sparsity patterns and that different experts converge to similar curvature geometries, explaining why simple linear averaging works well.

### Strengths
1. Repurposing Adam's `exp_avg_sq` as a free curvature proxy is clever and eliminates expensive Fisher computation.
2. Extensive study of FFG reveals interpretable patterns—aggressive Q/K sparsification, V/O preservation, task-specialized attention heads in late layers, and structured low-rank sparsity. Full replication of Tulu-3 SFT pipeline with five distinct experts and comprehensive benchmarking.
3. Empirical evidence that capability-based SFT models share similar curvature geometry despite different training data
4. Ablations stuides demonstrate FFG's pruning is the primary performance driver over curvature-aware aggregation

### Weaknesses
1. Unvalidated theory: Assumes perfect calibration and late NTK regime without verification for fine-tuned models
2. Requires per-expert sparsity tuning: Fixed sparsity fails; no practical guidance provided for setting these hyperparameters
3. And as for coding drops 78.8%→86.6% vs. multi-task; no investigation of why or when method fails ?

### Questions
1. How to set per-expert sparsity without expensive tuning?
2. Computational cost vs. multi-task training?
3. Why does coding degrade so much?

### Soundness
3

### Presentation
2

### Contribution
2
