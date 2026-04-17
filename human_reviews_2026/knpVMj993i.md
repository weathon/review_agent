# Orchestrated Sparse Consortium of Small Experts Beats Monolithic LLMs

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Large Language Models (LLMs) attain impressive capabilities but demand heavy computation and offer limited transparency. Naively shrinking a model reduces computational overhead yet typically sacrifices breadth and performance; we therefore pursue a different axis: keep models modular and *scale up* by coordinating multiple experts such that a small, task-adaptive subset collaborates per input and can achieve better performance. In this paper, we introduce **FOCUS** (*Flexible Orchestration and Collaboration Using Specialists*) -- a *generic* multi-expert collaboration framework that trains a lightweight *orchestrator* under *oracle* supervision to *select, order,* and *coordinate* a consortium of *experts* (homogeneous/heterogeneous language models of any size). A learnable sparse, near-symmetric *collaboration matrix* governs information flow among experts, and a *multi-round refinement* process aggregates intermediate outputs into a single answer; the oracle is used only during training, not at test time. At test time, the orchestrator adaptively routes experts with early stopping, achieving *sublinear cost growth* as consortium size increases. **FOCUS** achieves striking results: on MMLU, GSM8K, and HumanEval, a consortium of 5–7 Qwen experts (combined ~9B parameters) reaches 94.1%, 94.1%, and 87.8% accuracy, respectively, matching or surpassing a Qwen3-14B model by an average margin of 7.6%. On reasoning benchmarks, a consortium of 5 Phi-4-Mini models improves AIME-2024 from 26% to 40% and GPQA-DIAMOND from 19% to 31%, and attains 92% on MATH-500, exceeding a single Phi-4-14B reasoning model. These results establish *collaboration* as a distinct axis of scaling: carefully orchestrated experts can outperform comparable-size monolithic models while remaining modular and cost-effective for deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper trains a small orchestrator to manage a pool of individual LLMs. The orchestrator determines which LLM generates the initial response, which LLMs refine the response, and when to output a final answer.

While performance improvements are observed, some of the paper’s claims are questionable. Additionally, the orchestrator’s design, as well as the root causes of the observed improvements, require stronger support and clearer explanation.

### Strengths
Downstream accuracy improvements are observed across a diverse set of model pools.

### Weaknesses
1. I am curious about the effectiveness of the design. The expert selected first (based on the highest score it receives) is used to generate the initial solution; a high score implies this expert is better suited to address the task initially. However, for refinement, instead of recomputing scores to identify which expert is best suited for refining the solution, the authors reuse the previously computed scores. This means the expert selected for refinement is the second-highest scorer for generating the initial solution. This a choice that seems counterintuitive.

2. In Line 102, the statement “This inverse scaling law justifies the higher effectiveness of FOCUS on smaller language models” is circular. First, you observe that FOCUS performs more effectively on small models. Then, you fit a scaling law to these observed performance patterns. Finally, you claim this fitted law “justifies” that FOCUS works better on small models—using the result of the analysis as evidence for the very phenomenon the analysis was based on.

3. The loss function (Equation 4) includes 8 hyperparameters, making the method overly complex. Tuning such a large number of loss components is highly cumbersome, yet I did not find any ablation studies on L_{sel}, L_{len}, L_{symm}, or L_{spar}. Without these analyses, it is unclear whether such complexity is necessary, nor can we gain insights into the contribution of each component.

4. Table 2 shows that FOCUS is better than LLM-Blender and LLM-Debate. I did not find which models are used for LLM-Blender and LLM-Debate. It is perplexing that LLM-Blender scores only 42% while a single model already achieves 73.6% accuracy. I assume that even a simple voting mechanism should at least match the accuracy of individual models in the pool. Are there many weak models in LLM-Blender's pool? Similar inconsistencies appear in the LLM-Debate results, where the reported accuracy also falls unexpectedly low relative to the single-model baseline (47.3 to 73.6).

5. The method does not train specialized expert models but only trains the orchestrator. For scenarios with multiple homogeneous experts (e.g., two Qwen3-4B models), my understanding is that selecting either of these two Qwen3-4B models is basically the same, as their core capabilities are the same. If this is the case, the "selection" between the two Qwen3-4B reduces to a multi-agent role-playing: their only distinction lies in the roles assigned to the Qwen3-4B models (e.g., refining the initial solution or generating the final answer, which is driven by the prompt).

6. The authors emphasize that the orchestrator is lightweight, with a size of only ~1M parameters. However, the paper also introduces a shared encoder based on BERT. Excluding this encoder when reporting the orchestrator’s size is misleading, as it overstates the method’s orchestration efficiency.

### Questions
1. During inference, how is the calling order of selected experts determined?

2. See Weakness 4.

3. Please address the concern raised in weakness 5. For instance, in Table 1b, ~2.3 activated Qwen3-8B models achieve 94.1% accuracy on MMLU. I am curious whether using your prompt, recursively having a single Qwen3-8B model refine its own output ~2.3 times would produce similar accuracy.

4. You claim that small models benefit more from orchestration, but this result is not surprising—it aligns with findings from many RLVR papers, where small models achieve substantially higher pass@K accuracy when rolled out enough times, while large models show minimal pass@K gains as K increases. This pattern is intuitive: small models often make simple, recoverable mistakes, so giving them more opportunities to retry (via refinement or orchestration) allows them to correct errors. In contrast, large models already make fewer mistakes, so additional refinement, retries, or orchestration naturally yield little improvement. Furthermore, given the extremely small number of trainable parameters in your system, I question whether the model can actually learn effective orchestration strategies.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- Proposes an “orchestrated” approach where a small controller coordinates multiple frozen experts over several refinement rounds.  
- Uses oracle signals during training but removes the oracle at inference.  
- Produces interpretable artifacts: a collaboration matrix \(C\) and a sequence distribution \(\pi\).  
- Evaluates on multiple reasoning and coding benchmarks, reporting accuracy–cost trade-offs.

### Strengths
- **Clean structure:** The paper is well-organized; training/inference flow and losses are easy to follow, even if exposition needs tightening.  
- **Proposes a new method:** Introduces a learnable orchestrator that combines sparse collaboration and sequence planning with interpretable controls.  
- **Conducts some experiments:** Presents results on several benchmarks with ablations and cost analysis, showing advantages on accuracy vs. budget.

### Weaknesses
## Weaknesses

1) **Scaling claims lack support.**  
The paper presents an inequality and “inverse scaling” as general laws. There is no per-task analysis or error bars. Decoding settings might change the trend. These should be framed as empirical patterns by dataset. Report confidence intervals and fit quality for the power laws. Also show where the relation breaks.

2) **Oracle supervision may leak answers.**  
Training uses oracle attention and distillation. It is unclear what the oracle outputs (free text, CoT, short rationale) and how they are embedded. Oracle text might encode labels indirectly. Explain how leakage into \(o\) is prevented. State whether BERT-base is frozen or tuned, and whether prompts differ for oracle vs. expert. Add a step-by-step example from oracle text → embedding → effect on \(C\) and \(\pi\). These details clarify whether the system learns routing or recovers teacher-coded answers.

3) **Collaboration matrix assumptions are unproven.**  
The paper encourages near-symmetry and sparsity in \(C\). Why should symmetry help beyond aesthetics? Some tasks may benefit from directional flows. Quantify the accuracy or cost penalty without symmetry. Add ablations with: no symmetry, strong symmetry, and sparsity sweeps vs. chain length. Report full accuracy–latency trade-offs.

4) **Decoding and prompts are under-specified.**  
Results can change with temperature, top-p, system prompts, few-shot seeds, and tool use. The paper should list per-task decoding configs for each expert and each baseline. Do the same for system prompts and tool settings (e.g., code execution on HumanEval). Place these in the main text or an appendix.

### Questions
1) **Oracle features.**  
What exactly is \(o\) (layer, pooling)? How is it aligned across model families? How sensitive are results to the oracle’s prompt?

2) **Symmetry in \(C\).**  
Is the near-symmetry constraint justified across tasks? Some settings likely require asymmetric “fixers” late in chains. When does symmetry help, and when does it hurt?

3) **Train–test mismatch with attention.**  
Training applies cross-expert self-attention with oracle-augmented states. Test-time has no oracle yet still applies self-attention. What is the performance drop if that test-time attention is removed? Please quantify the shift.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces **FOCUS** (Flexible Orchestration and Collaboration Using Specialists), a novel framework for orchestrating a consortium of small or medium-sized \textbf{language model experts} to collectively outperform a single, large monolithic LLM.  
FOCUS trains a lightweight orchestrator (≈1M parameters) under oracle supervision to learn:

1.  a sparse, near-symmetric collaboration matrix governing directed information flow among experts; and

2.  a sequence distribution $\pi$ that selects and orders a small subset of experts for each input.

During inference, the orchestrator adaptively activates a top-$K$ subset of experts and employs multi-round refinement with early stopping, ensuring sublinear computational cost with respect to the consortium size.

Empirical results are striking: a consortium of 5–7 Qwen3 experts (total $\sim$9B parameters) achieves 94.1\% on MMLU, 94.1\% on GSM8K, and 87.8\% on HumanEval—surpassing or matching a single 14B–32B model while being more modular and cost-effective. Similarly, a consortium of 5 Phi-4-Mini models boosts AIME-2024 accuracy from 26\% to 40\% and GPQA-DIAMOND from 19\% to 31\%. These results demonstrate that \textbf{collaboration forms a new scaling axis}, orthogonal to parameter count, and that orchestrated multi-expert systems can achieve superior efficiency–accuracy trade-offs.

### Strengths
**Originality:**

1.  Introduces a learned, differentiable collaboration protocol with explicit **matrix-based expert interaction**, unlike previous ensemble, debate, or MoE approaches.
2. The combination of **oracle-supervised orchestration**, **sparse near-symmetric collaboration**, and **early-stopped refinement** is conceptually novel.
3.  Establishes empirical **scaling laws for collaboration**, quantifying performance as a function of expert size and count.

**Clarity:**

1.  The paper is well-written, with intuitive figures (e.g., Fig.~1 illustrating orchestration) and a clear narrative linking motivation, method, and empirical findings.
2.  Key design choices—such as oracle use only during training, frozen experts, and sublinear cost—are articulated transparently.
3. The inclusion of FAQs enhances accessibility and replicability.


**Significance:**

1.  Demonstrates that coordinated small experts can outperform much larger monolithic LLMs—potentially redefining efficiency scaling in model design.
2.  Offers a principled foundation for modular LLM ecosystems, complementing existing paradigms like MoE, agent frameworks, and ensembling.
3.  Opens new directions for \textit{collaboration scaling theory}, modular deployment, and low-cost specialization.

### Weaknesses
**Oracle supervision dependency:** While the oracle is removed at inference, its use during training raises questions about supervision cost and generalization when oracle quality is poor.

**Limited theoretical grounding:** The method empirically demonstrates scaling laws (e.g., $A_{\text{FOCUS}}(K,B)\approx100-\gamma(B/K)K^{-\alpha}$) but lacks a formal theoretical explanation.

**Cross-family fragility:** Experiments show that heterogeneous consortia (e.g., Qwen + Phi + LLaMA) underperform due to representational misalignment; more analysis on why alignment fails would strengthen the paper.

**Ablation depth:** While extensive, some critical variables (e.g., the number of refinement rounds, early-stopping thresholds) lack sensitivity analysis.

**Computational cost reporting:** The orchestrator’s runtime and training overhead are described qualitatively; quantitative latency or FLOPs comparisons would clarify efficiency claims.
   
**Comparative scope:** Missing comparisons with strong routing baselines (e.g., MoE with adaptive gating or reinforcement-learning routers) that share similar objectives.

### Questions
1. How sensitive is FOCUS to the oracle’s strength or domain coverage? Would weaker or noisy oracles (e.g., smaller models) still yield stable routing policies?
2. Can the learned collaboration matrix generalize to unseen experts (e.g., adding a new model without retraining the orchestrator)?
3. Is the near-symmetric constraint on $C$ empirically optimal, or could asymmetric flow capture hierarchical expert refinement better?
4. Could the authors provide concrete runtime or energy metrics to substantiate the “sublinear cost” claim?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FOCUS, a novel multi-expert collaboration framework designed to address the prohibitive computational costs of LLMs. The core idea is to coordinate a "consortium" of multiple small "expert" models (which can be homogeneous or heterogeneous) rather than building a single, massive, monolithic model. At the heart of the FOCUS framework is a lightweight orchestrator (approx. 1M parameters) that is supervised by a powerful "oracle" model during the training phase. At inference time (without the oracle), the orchestrator dynamically selects a sparse subset of experts (Top-K, averaging 2-3) based on the input, defining their collaboration order and information flow . The selected experts then engage in a multi-round "refinement" process, progressively improving the answer to generate a final, aggregated output.Empirical results demonstrate that a consortium of 5-7 small Qwen experts (totaling ~9B parameters) matches or even surpasses the performance of the much larger monolithic Qwen3-14B model on benchmarks such as MMLU, GSM8K, and HumanEval.

### Strengths
1.The paper provides sufficient experimental support for its core thesis that "a small consortium of experts beats a monolithic model". For instance, a Qwen3-8B consortium using FOCUS surpassed the monolithic Qwen3-32B on the average score (91.2% vs. 83.2%) , while the performance of a Qwen3-1.7B consortium (83.1%) rivaled that of a single Qwen3-8B model (78.0%).
2.The proposed FOCUS framework achieves a good trade-off between accuracy and cost. By leveraging sparse activation and early stopping mechanisms , the system attains high performance while activating only a few experts. Compared to baselines like LLM-Debate or DyLAN , FOCUS significantly advances the Pareto frontier for multi-expert systems.
3. The paper introduces a "learnable orchestrator". Unlike multi-agent or ensemble methods that rely on hard-coded rules or simple voting , FOCUS learns a differentiable collaboration protocol. This approach, based on "multi-round refinement" rather than simple answer fusion, may represent a promising future direction for multi-expert systems.

### Weaknesses
1. The total loss function is defined as a weighted sum of eight distinct loss terms. This introduces a large number of hyperparameters, and these weights appear to require meticulous tuning to balance task utility, structural properties (like symmetry and sparsity), and efficiency (like length and diversity). The paper lacks a detailed exploration of the system's sensitivity to these hyperparameter weights, which could pose a practical barrier to reproducing and generalizing the method.
2. Although the paper claims the framework is suitable for heterogeneous consortia, the experimental results are mixed. The best performance gains are achieved using experts from the same model family (such as Qwen-only). In contrast, cross-family collaboration (involving models like Qwen, LLaMA, and Phi) performs poorly, a result the paper attributes to "representational misalignment." This is a significant shortcoming, as a truly "plug-and-play" system should be able to efficiently integrate experts from diverse sources.
3. The framework's mechanism is heavily reliant on a powerful oracle model. This dependency is twofold: it relies on a representation-level distillation loss and an oracle alignment loss, which helps bootstrap the initial collaboration protocol. Although the oracle is removed at inference time, this introduces a paradox: to train a system capable of replacing a large model, one must first possess an even more powerful large model to serve as the oracle.

### Questions
1. The total loss function comprises eight different weighted terms, introducing numerous hyperparameters. This complexity could make the results difficult to reproduce. To what extent does the system's performance depend on the precise tuning of these weights? For example, are the contributions of the symmetry loss and the sparsity loss to performance and cost orthogonal, or are they highly correlated? Can you provide a sensitivity analysis for the weights of these key structural loss terms?
2. The paper's core thesis is that "small beats large," meaning a consortium of small experts outperforms a monolithic model. However, the training process relies heavily on a much stronger oracle for supervision, specifically for the distillation loss and the oracle alignment loss. Doesn't this create a paradox: you must first possess a more powerful model to train this supposedly "more efficient" system? If this oracle supervision were removed, relying only on task utility and structural losses, could the FOCUS framework still learn effective collaboration, or would it collapse into suboptimal routing strategies?
3. Figure 5 illustrates a failure case where Expert 1 produces the correct answer, only for Expert 2 to overwrite it with incorrect reasoning, causing Expert 3 to become confused and diverge further. This exposes a critical risk: the multi-round refinement mechanism seems unable to distinguish between "constructive improvement" and "destructive overwrite." Could you elaborate on why the early stopping mechanism failed to trigger after Expert 1 provided the correct answer? Furthermore, is there anything in the learning mechanism for the collaboration matrix or the sequence distribution designed to prevent this kind of "overwrite" of correct intermediate steps?
4. The authors claim the framework can handle an "arbitrarily large number" of experts. However, the consortium size in the experiments is relatively limited (e.g., 18 or fewer). Given that the collaboration matrix scales quadratically with the number of experts, and the orchestrator must compute scores for all of them, how can a lightweight orchestrator with only about 1M parameters maintain its learning efficiency and routing accuracy when the number of experts becomes very large (e.g., 100 or 1000)? Does this growth in the expert count create a bottleneck in the orchestrator's own expressive capacity or its training convergence?

### Soundness
3

### Presentation
3

### Contribution
3
