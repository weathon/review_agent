# Skill Weaving: Efficient Self-Improvement of LLMs via Modular Skillpacks

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 6

## Abstract
We propose SkillWeaving, a modular self-improvement framework that enables large language models (LLMs) to specialize using only their own generations. The key idea is to decompose general-purpose LLMs into a collection of skillpacks—lightweight, domain-specific delta modules—that reorganize and refine the model’s internal knowledge. By combining rule-based verification with preference optimization, each skillpack learns high-quality self-refined capability from a specific domain, achieving robust improvement without external labels. To further ensure scalable deployment, we introduce SkillZip, a fully quantized delta-compression method that jointly quantizes weights and activations, eliminating runtime decompression and enabling fast, low-cost inference. By merging shared knowledge and hardware-aware technical designs, our method achieves both parameter-efficient specialization and inference-efficient execution. On multi-task and agentic benchmarks, Skill Weaving outperforms expert-tuned baselines and even surpasses larger 32B monolithic models using only 9B parameters, while offering up to 4× speedup. Overall, our approach offers an interpretable and resource-efficient path toward LLM self-improvement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SkillWeave, a modular self-improvement pipeline for LLMs. The idea is to (i) partition a base model’s capabilities into domains, train full-parameter task vectors with online DPO on self-generated data filtered by lightweight rules, then (ii) compress each task vector into an inference-friendly skillpack using SkillZip, a “full quantization” delta-compression scheme that quantizes both low-rank deltas and activations and avoids runtime dequantization. The framework also merges shared knowledge back into a common backbone to reduce interference and memory, and optionally routes tokens to skillpacks at inference (Fig. 1, p. 3). On general-capability benchmarks (math, coding, dialogue, reasoning), the method reports consistent gains over model-merging, LoRA-MoE, and self-improvement baselines (Table 1, p. 7). On agentic tasks (AgentBench), a 7B backbone + ~0.5B skillpacks per domain approaches a 32B monolith while delivering up to ~4× lower latency (Appendix C/Fig. 4).

### Strengths
•	Originality / Design coherence. Clear story and easy to understand: extract full-capacity task vectors with preference learning, then compress into efficient skillpacks. The “full-tune-then-zip” position is well-motivated vs. PEFT-only adapters (§3, Fig. 1). 
•	Technical quality. SkillZip is thoughtfully engineered: constrained scaling to match tensor-core kernels, channel-wise smoothing (SmoothQuant-like) + rank-wise rotation after SVD, and a concatenation trick for larger GEMMs (§3.2, B.2; Table 2).  
•	Empirical scope. Evaluations span four general domains with multiple benchmarks and an agent setting; comparisons include model-merging, routing-MoE (LoRA-MoE/Twin-Merging), self-rewarding, and delta-compression baselines (Table 1; Appendix C). 
•	Practical significance. The deployment argument is compelling: shared 7–9B backbone + compact skillpacks, with batch-friendly S-LoRA/VSGGEMM and reported large latency wins (Fig. 3/4 & App.).

### Weaknesses
1.	“Self-improvement only” claim vs. data usage. The main text emphasizes improvement “without external labels or stronger teachers,” yet Appendix E.2 lists many human-labeled datasets (e.g., OpenMathInstruct-2, ARC, HumanEval/MBPP sources) used in training/verification, and B.1 uses external reward models for dialogue scoring. Please reconcile what proportion of supervision is truly self-generated vs. derived from labeled corpora or reward models, and how results change if those are removed. 
2.	Fairness and budgets. It’s unclear whether baselines received comparable compute and online sampling (e.g., 64 candidates/prompt, long contexts; E.3) and whether they were allowed router/merging hyper-sweeps of similar depth. Report wall-clock, FLOPs, and GPU hours per method/domain.  
3.	Statistical reliability. No confidence intervals, seed variability, or bootstrap CIs are reported. Given modest margins on some benchmarks, include variation across ≥3 seeds and, for coding/math, pass@k and solve-rate breakdowns. 
4.	Latency reporting inconsistency. The text claims “0.38× speedup over DeltaCome” and “0.04× speedup over S-LoRA” (p. 9). Speedups should be >1 if faster; otherwise report latency ratios (<1). Clarify units (ms/token vs. tokens/s), batch sizes, and kernel vs. end-to-end measurements.  
5.	Router and domain inference. Inference-time dispatch is central (Fig. 1 bottom), yet the router (training data, features, losses, OOD behavior) is only briefly mentioned. Provide details and an ablation for mis-routing rates and robustness to mixed-domain prompts.  
6.	Interference/forgetting evidence. The paper argues skillpacks “avoid cross-task interference,” but no explicit cross-contamination tests are shown (e.g., train skillpacks sequentially and measure performance decay; probe cross-domain robustness). Add such diagnostics. 
7.	Compression fidelity knobs. Truncating INT32 to INT8 between GEMMs (B.2) could be brittle. Include sensitivity to truncation thresholds, rank R, and bit-widths (Xk/Ak/Bk), and compare against weight-only W8A16 + SmoothQuant on the same deltas.  ￼
8.	Writing polish. Several typos/grammar issues and cross-ref errors (e.g., “Tab. 3” in §5.1; inconsistent capitalization of SKillZip). A pass for clarity would help.

### Questions
1.	Data provenance: Precisely quantify the fraction of self-generated pairs used in DPO per domain vs. instances using ground-truth labels or external reward models. Can you reproduce Table 1 with only self-generated pairs and rule checks (no reward models, no gold labels)?  
2.	Compute and cost: What are the GPU hours for (a) skillpack building and (b) SkillZip per domain? How do these compare to the best-performing baseline families?  
3.	Router details: What signals are routed on (prompt tags, token embeddings, a small classifier)? What is the OOD fallback when the input mixes domains? Provide accuracy vs. mis-routing curves.  
4.	Interference tests: Can you report sequential skillpack training with periodic re-evaluation on earlier domains to directly evidence reduced forgetting?  
5.	Latency metrics: Please reconcile the “× speedup” phrasing and provide a standardized latency table: tokens/s and ms/token at batch sizes {1, 8, 32}, seq-lengths {1k, 4k}, on A100-80G.  
6.	Ablations: (i) remove rank-wise rotation; (ii) vary truncation INT32→INT8; (iii) swap DPO for SimPO; (iv) compare against a PEFT-only pipeline with the same SkillZip (i.e., zip LoRA deltas) to isolate the value of full-tune.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes SkillWeave, a modular self-improvement framework for LLMs that turns one general model into a set of domain-specialized skillpacks, each trained with the model’s own synthetic data and filtered by lightweight rule-based checks, then optimized by DPO to prefer helpful over harmful generations. Additionally, the paper introduces SkillZip, an inference-oriented delta-compression scheme that first merges shared knowledge into the backbone and then fully quantizes both deltas and activations, enabling direct INT8/INT4 computation without runtime dequantization. The proposed method demonstrates good performance and efficiency on several agentic benchmarks.

### Strengths
-  The paper is clearly written and organized. 
- The experiments are very comprehensive with comparisons over many related works.
- The experiments demonstrate superior performance showing that a small SkillWeave model trained only on self-generated data can match or exceed the performance of teacher-assisted systems while being faster at inference.

### Weaknesses
- Some typos in Section C of the Appendix, where the figure and table indexes are in mismatch.
- The rule-based self verification seems to be unapplicable to more generic skills/tasks.

### Questions
- I think the paper needs to explain more details on how does Skillweave route input token to different skillpacks. This seems to be an important technical part of the proposed method but remains unexplained.

### Soundness
3

### Presentation
4

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
SkillWeave is a modular self-improvement framework for large language models that splits a model’s general abilities into smaller, domain-specific “skillpacks.” The framework introduces SkillZip, a fully quantized compression method that reduces both weights and activations via a smooth-quant variant to achieve faster inference. Together, SkillWeave and SkillZip enable efficient, self-supervised specialization without extra human labels. Experiments show a the collection of the 5x7B SkillWeave model outperforms 32B monolithic LLMs while running up to four times faster.

### Strengths
1. Modularity: SkillWeave effectively decomposes large language models into domain-specific skillpacks that can be trained, updated, and deployed independently. This modular separation helps reduce cross-task interference and prevents catastrophic forgetting, enabling more stable multi-domain specialization.

2. Hardware-aware compression: The integrated SkillZip module performs full quantization of both weights and activations, allowing computation directly in low-bit formats. This design achieves significant improvements in inference speed and memory efficiency without requiring runtime decompression.

3. Performance-efficiency balance: The benefits of SkillWeave are clearly shown in comparisons between monolithic 32B models and distributed 5×7B model setups, where SkillWeave delivers similar or superior accuracy while achieving faster inference.

### Weaknesses
1. Model size comparison: Several baselines in Table 1 use smaller models (e.g., 8B) compared to the 10B SkillWeave configuration. This makes it difficult to clearly separate improvements due to the proposed method from those arising from the increased model capacity. A fairer comparison with models of equal or similar size would strengthen the empirical claims.

2. Overloaded figure presentation: Figure 1 contains many overlapping components, which makes the core pipeline hard to follow. The authors should consider restructuring the figure and introducing each stage -- skillpack construction, compression, and integration -- sequentially in the text to improve clarity and reader comprehension.

3. Need for clearer narrative focus: The writing often shifts between unrelated technical details -- such as quantization smoothing, preference optimization, and inference serving optimizations -- without emphasizing the central contribution. Streamlining the exposition to first establish the main SkillWeave pipeline and deferring secondary engineering details (e.g., quantization specifics, GEMM parallelization) to the appendix would make the paper more coherent and easier to follow. If the authors could provide a more central pipeline of their method, this would aid in distinguishing between engineering/implementation features and central methodology.

### Questions
Refer to the requested changes in the Weaknesses Section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces lightweight, domain-specific skillpacks that boost domain performance for LLMs. It beats other baselines on most domain related benchmarks while using self-generated data without relying on costly human labels or stronger teachers. For deployment, the paper introduces SkillZip, a quantization strategy that  packages domain skillpacks in a fully quantized form. It jointly quantizes weights and activations to avoid runtime decompression/dequantization and reduce latency. SkillWeave  and SkillZip as a unified pipeline enables self-improvement and efficient deployment for LLMs.

### Strengths
1. Clear accuracy gains with strong baselines. It consistently matches or exceeds competitive merging, routing, and self-improvement baselines of comparable or larger size. 

2. Hardware-aware design with strong throughput. The agentic evaluation uses a single 7B backbone plus five 0.5B skillpacks and shows 4.2× faster inference than a 32B monolithic model and 5.5× faster than deploying five separate 7B models , while staying within 3–5% of their accuracy. The SkillZip study further indicates that fully quantized deltas (e.g., X8A8B8, X8A4B4) maintain or slightly exceed DeltaCome’s accuracy at lower latency.

### Weaknesses
1. In Table 1, the reported accuracy assumes the task type is already known and the system can activate the correct skillpack accordingly. However, how to reliably infer the task type at inference time remains unspecified. Without quantified routing accuracy (misclassification rate,  fallback), it is unclear how much task-type misidentification would hurt downstream accuracy.

2. Critical E2E inference settings (prefill & decode length, batch size) are missing for Figure 3.

### Questions
1. Can you clarify on how to use the 5x7B model for domain specific tasks? Is model loading time also included in the inference latency? 

2. In section 5.1, you metioned a 7B backbone but Table 1 uses 8B backbone.

### Soundness
3

### Presentation
3

### Contribution
2
