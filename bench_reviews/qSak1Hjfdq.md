## Summary

This paper formalizes the all-day multi-scenes lifelong VLN (AML-VLN) problem, where agents must continually learn across multiple scenes and diverse environmental conditions (low-light, scattering, overexposure) without catastrophic forgetting. To address this, the authors propose Tucker Adaptation (TuKA), which lifts parameter-efficient adaptation from 2D matrices to a 4th-order tensor via Tucker decomposition, explicitly decoupling shared knowledge (core tensor + encoder/decoder) from scene-specific and environment-specific expert factor matrices. A Decoupled Knowledge Incremental Learning (DKIL) strategy combines EWC-style regularization on shared components with orthogonal constraints on new experts. The paper also contributes an extended Habitat benchmark with synthesized degraded imaging conditions and demonstrates AlldayWalker's superiority over LoRA-based continual learning baselines.

## Strengths

- **Principled departure from matrix-based adaptation:** The core insight—that multi-hierarchical knowledge (scene × environment) is naturally represented as a high-order tensor rather than forced into 2D matrix factorizations—is well-motivated and technically sound. The Tucker decomposition cleanly separates shared vs. specific knowledge across tensor modes, and Equation 3 provides a mathematically correct mechanism for collapsing the high-order tensor back to a 2D weight matrix compatible with LLM backbones. The comparison with ABC-LoRA (Appendix I) effectively isolates the benefit of tensor factorization from mere architectural hierarchy (65% vs. 55% SR), demonstrating that the gain comes from the Tucker core capturing cross-dimensional interactions, not just from having a tree structure.

- **Thorough ablation design:** The paper ablates 3rd-order vs. 4th-order tensors (Appendix H), shared component contributions (Table 3), rank scaling (Appendix G), and extends to 5th-order tensors (Appendix J). The 30-task scalability test (Table 4) shows minimal degradation, supporting the method's stability. These ablations collectively build a strong empirical case for the design choices.

- **Benchmark contribution with practical relevance:** Extending Habitat with physics-based imaging degradation models (atmospheric scattering, low-light sensor noise, overexposure saturation) grounded in established imaging models (Eqs. 10–12) provides a reusable testbed for a practically important problem that standard VLN benchmarks ignore.

## Weaknesses

### Major:

- **No analysis of expert retrieval reliability at inference:** Section 3.4 describes matching the current observation's CLIP features against stored scene/environment feature sets via cosine similarity, but provides zero analysis of how often this retrieval is correct, what happens when it fails, or how robust it is to visual ambiguity (e.g., a dimly lit scattering scene might match "low-light" rather than "scattering"). Since the task-id is agnostic at test time, the entire method's practical viability hinges on this retrieval step. Without any retrieval accuracy metric, failure case analysis, or sensitivity to ambiguous observations, it is impossible to assess whether AlldayWalker would work reliably in deployment. This is the single most important gap in the evaluation.

- **Missing computational overhead analysis for inference:** The paper emphasizes parameter efficiency (~0.3M trainable params) but never reports inference latency, FLOPs, or memory bandwidth for the Tucker reconstruction (Eq. 3) combined with the expert retrieval search. The mode-products in Eq. 3 must be computed for every transformer layer at every navigation step, and the CLIP-based retrieval adds a forward pass plus similarity computation. For a paper targeting real-time robotic deployment ("all-day" navigation), the absence of any latency comparison against standard LoRA's simple matrix multiply is a significant omission.

### Minor:

- **Scalability of expert matrices in open-ended lifelong learning:** The current benchmark has M=7 scene experts and N=4 environment experts. In true open-ended deployment, encountering a new scene requires adding a new row to **U₃**, growing parameters linearly. The paper does not discuss this growth rate or compare it against simply storing independent LoRA adapters per task. If every new task introduces a novel scene, TuKA's parameter growth (new expert row + shared updates) may not be more efficient than per-task LoRA storage, undermining the parameter-efficiency claim for the lifelong setting that matters most.

- **Ambiguity in expert selection frequency:** It is unclear whether the expert retrieval (Section 3.4) is performed once per episode or dynamically per step. If a robot transitions from a dark corridor to a well-lit room mid-episode, does the agent switch environment experts? The current formulation and Algorithm 2 suggest a single retrieval per inference call using "current observation Oq," but the paper never discusses dynamic expert switching or evaluates its necessity, which is critical for the "all-day" claim where conditions change within a single trajectory.

- **No ablation of individual DKIL loss components:** Table 3 ablates shared architectural components but does not isolate the contributions of L_ewc, L_co, and L_es to forgetting prevention. Given that these three losses serve distinct purposes (shared consolidation, expert consistency, orthogonal exploration), understanding which drives the performance is essential for justifying the full loss design and for practitioners who may want simpler variants.

- **No variance reported across task orderings:** Continual learning results are known to be sensitive to task ordering. The paper mentions randomized ordering but reports results from a single run. Without error bars or multiple ordering experiments, the reliability of the reported improvements is uncertain.

### Trivial:

- The distinction between "scene" (geometry/layout) and "environment" (illumination/weather) is crucial for the tensor formulation but is introduced informally in Section 2. An earlier, explicit definition alongside Figure 1 would aid readability.

## Nice-to-Haves

- Comparison with replay-based continual learning methods (e.g., iCaRL, ER) to situate TuKA within the broader CL landscape, though the paper reasonably argues replay is costly for embodied tasks.
- Interpretability analysis of what the core tensor **G** vs. expert matrices **U₃**, **U₄** actually learn (e.g., via probing or dimensionality reduction), to move beyond the assumption that knowledge is genuinely decoupled.
- Analysis of negative transfer between tasks (when learning a new task actively harms performance on a previous one beyond simple forgetting), which is distinct from the forgetting rates currently reported.
- Hyperparameter sensitivity analysis for λ₁, λ₂, λ₃ beyond the rank scaling in Appendix G.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Synthetic degradations are too aggressive / inflate performance gap"**: The harsh critic suggests the 0% collapse of Seq-FT indicates unrealistically extreme degradations. However, Seq-FT with a single overwritten LoRA naturally collapses on all prior tasks regardless of degradation severity. Other baselines (LwF-LoRA: 12%, EWC-LoRA: 15%) also perform poorly but non-zero, and the real-world generalization results (Table 5: 55% avg SR on unseen real scenarios) confirm the method works beyond synthetic settings. The performance gap reflects genuine problem difficulty, not artificial inflation.

- **"Fisher Information requires past data / replay buffer"**: The harsh critic questions how F_{θ,t-1} is computed without storing past data. Section 5.1 explicitly states Fisher is computed "using the first 10% of the data before adaptation to each task," which is standard EWC practice—computing Fisher on the current model's parameters using current task data before updating. No replay buffer is needed.

- **"ABC-LoRA is custom-built, compare with more established methods"**: ABC-LoRA is a reasonable hierarchical baseline specifically designed for fair comparison with matched parameter budgets. The paper already compares against 10 other baselines including established methods (HydraLoRA, BranchLoRA, O-LoRA, SD-LoRA). Adding more baselines would be nice-to-have, not a weakness.

- **"Missing related works"**: Per the hard rules, I cannot confirm the existence of unspecified related works.

- **"Test on more diverse degradation types (rain, fog, motion blur)"**: This is scope creep. The paper introduces 4 degradation types grounded in physics-based imaging models, which is sufficient for an initial benchmark. Demanding more types is a nice-to-have.

- **"Model-dependent effects not characterized"**: The paper evaluates on a single backbone (StreamVLN + Qwen2-7B). While cross-architecture evaluation would strengthen claims, this is standard scope for a methods paper and not a core flaw.

- **"Mechanistic explanation of why 4th-order outperforms 3rd-order is missing"**: The paper does provide a conceptual explanation (Section 5.3 and Appendix H): 4th-order tensors decouple scene and environment into separate expert matrices, while 3rd-order tensors couple them into a single flattened expert set. This is a reasonable explanation.

## Novel Insights

The comparison between TuKA and ABC-LoRA (Appendix I) reveals an important nuance: architectural hierarchy alone (scene branch + environment branch with matrix multiplication) is insufficient—the Tucker core tensor's role as a shared interaction hub that captures *cross-dimensional couplings* (scene↔environment) is what drives the 10% SR gap. This suggests that in multi-hierarchical continual learning, the bottleneck is not just separating knowledge across dimensions but modeling their *interactions*, which tensor cores naturally provide but cascaded matrix multiplications cannot. This insight generalizes beyond VLN to any setting where multiple orthogonal sources of variation must be jointly adapted.

## Suggestions

- **Quantify expert retrieval accuracy:** Run the CLIP matching on the test set with ground-truth scene/environment labels and report the retrieval accuracy (top-1 match rate). Additionally, report performance when using ground-truth expert selection vs. CLIP-based selection to isolate retrieval errors from navigation errors.

- **Add an inference latency table:** Compare wall-clock time per navigation step for AlldayWalker vs. LoRA baselines on the same hardware. Even a single-row comparison would address the deployment concern.

- **Clarify expert selection frequency:** Explicitly state whether retrieval is per-episode or per-step, and if per-step, evaluate whether dynamic switching provides benefits over static selection within an episode.

- **Ablate individual DKIL losses:** Add a table showing average SR and F-SR with each of L_ewc, L_co, and L_es removed individually, to demonstrate which components are essential vs. incremental.

## Quality Assessment

- **Novelty:** High. The application of Tucker decomposition to PEFT for multi-hierarchical continual learning is a genuine conceptual advance over MoE-LoRA variants. The tensor-to-matrix alignment mechanism (Eq. 3) is clean and non-obvious.

- **Technical soundness:** Good overall. The mathematical formulation is correct, the DKIL strategy is well-designed, and the ablations are thorough. The main gap is the unanalyzed inference pipeline (retrieval + reconstruction).

- **Empirical support:** Strong on simulation, moderate on real-world. The 24-task benchmark with 12 baselines and extensive ablations provides solid evidence. The real-world evaluation (4 seen + 2 unseen scenarios) is encouraging but limited in scale.

- **Significance:** High for the embodied AI community. The AML-VLN problem formulation and the TuKA method address a practical deployment challenge that will become increasingly important as VLN agents move toward real-world deployment.

- **Clarity:** Adequate. The notation is dense in Section 3.3 but generally well-structured. The distinction between scene/environment hierarchies could be introduced earlier. The paper would benefit from a concrete numerical example of how Eq. 3 produces a weight matrix from the tensor factors.