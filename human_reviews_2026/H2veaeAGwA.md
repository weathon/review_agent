# RAG-FGO: Enhancing RAG with Fungal Growth Optimizer for LLM Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Generative retrieval leverages large language models (LLMs) to directly generate retrieval queries or candidate document representations, and has recently shown great potential in open-domain question answering and knowledge-intensive tasks. Compared to traditional index-based retrieval methods, generative retrieval provides greater flexibility in handling semantic diversity and complex task requirements. However, existing approaches often rely on static prompt designs or fixed generation strategies, which struggle to maintain both stable and efficient performance in scenarios characterized by semantic complexity, task variability, or noisy knowledge bases. To address these limitations, we propose RAG-FGO (Retrieval-Augmented Generation with Fungal Growth Optimizer), a heuristic search-based framework for optimizing dynamic generative retrieval agents in Retrieval-Augmented Generation (RAG) systems. RAG-FGO integrates both global and local search strategies within the solution space to generate more robust and effective retrieval prompts and parameters, while avoiding local optima. In addition, it introduces a query memory pool that stores high-performing prompts during iterative optimization, thereby guiding subsequent search and generation. Experimental results indicate that RAG-FGO outperforms baselines such as Direct, ReAct, and Self-Act on benchmark datasets including HotpotQA, MuSiQue, and SQuAD, confirming its effectiveness for complex generative retrieval tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents RAG-FGO, a framework designed to enhance Retrieval-Augmented Generation (RAG) systems by optimizing retrieval prompts and parameters. Inspired by the Fungal Growth Optimizer (FGO) algorithm, the proposed method models the search for an optimal “retrieval agent” in semantic space as a process that combines global exploration with local refinement. A key component is the query memory pool, which preserves high-performing agents during iterations to ensure stable and cumulative optimization. The core contribution lies in formulating retrieval optimization as a dynamic search problem. Experimental results on question-answering and reasoning benchmarks such as HotpotQA and MuSiQue demonstrate that RAG-FGO outperforms several strong baselines.

### Strengths
1. Novel problem framing and method: The paper introduces RAG‑FGO, formulating retrieval‑prompt optimization as a semantic‑space search with a query memory pool and FGO‑inspired exploration–exploitation, supported by a clear workflow.

2. Well‑specified global vs. local search mechanism: The paper proposes local search as well as global search, where local perturbations are performed at a fine-grained level, while global perturbations force larger changes.

3. Broad experimental coverage and ablations: The study spans six QA benchmarks and seven reasoning benchmarks with multiple backbones, reports head‑to‑head results, and analyzes iteration/search budgets and saturation effects.

### Weaknesses
1. Computational overhead and cost reporting: Runtime or Dollar Costs and Compute‑Matched Comparisons are not reported. The paper discusses that this method is not suitable for LLMs that are large enough to provide richer data.

2. Missing component ablations: Although the schedule and operators are defined, there is no ablation isolating local vs. global search or the query memory pool to quantify each component’s contribution.

3. Figure readability: Some figures have small fonts and become blurry when zoomed in, which hinders quick understanding of pipeline and ablation trends.

### Questions
For global search, what are the rules for rewriting using LLM? For local search, are the prompt modification rules for algorithms such as synonym replacement random or sampled?

### Soundness
3

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
4

### Summary
This paper proposes RAG-FGO, a bio-inspired optimization framework for Retrieval-Augmented Generation (RAG) that employs the Fungal Growth Optimizer (FGO) to dynamically refine generative retrieval agents. The method treats retrieval optimization as a semantic-space search problem, combining global exploration and local refinement, with a query memory pool to accumulate high-performing retrieval strategies. Evaluations on multiple QA and reasoning benchmarks (HotpotQA, MuSiQue, SQuAD, MMLU-Pro, etc.) demonstrate that RAG-FGO consistently improves over strong baselines such as ReAct, Self-Act, and Aflow.

### Strengths
Originality:
  The paper introduces a novel adaptation of the Fungal Growth Optimizer for retrieval prompt and parameter optimization in RAG systems. While prior works have explored heuristic or evolutionary search for LLM tuning, this specific bio-inspired approach and its integration with RAG’s retrieval module are original and timely. The inclusion of a query memory pool for cumulative improvement further enhances the conceptual novelty.

 Quality:
  The methodology is well motivated and experimentally validated on diverse datasets and models (e.g., GPT-4o-mini, Qwen-Long, Gemma2-27B). The design of both local and global search phases, along with iteration scheduling, shows thoughtful engineering. Ablation studies on iteration and search count (Figures 4–5) provide some insight into the framework’s efficiency–accuracy trade-offs.

 Clarity:
  The paper is generally well organized, with clear mathematical notation, figures that effectively communicate the iterative optimization process, and detailed experimental descriptions. The reproducibility section is thorough, listing datasets, hyperparameters, and implementation details.

 Significance:
  By addressing the rigidity of static retrieval prompts and manually tuned parameters, RAG-FGO contributes to making RAG systems more adaptive, autonomous, and robust. This work is relevant to ongoing research on retrieval–generation co-optimization and scalable agentic LLM pipelines.

### Weaknesses
Lack of theoretical grounding:
  The paper lacks formal proofs, convergence analysis, or theoretical guarantees for the proposed optimization process. While heuristic optimization methods are often empirical, ICLR readers generally expect at least a formalized problem definition or complexity characterization. The description of FGO’s adaptation to semantic-space search remains heuristic and narrative rather than mathematically rigorous.

 Algorithmic opacity:
  Despite the inclusion of pseudocode references, the formal definition of the optimization objective (e.g., explicit loss function or expected improvement metric) is missing. The framework is presented as a procedure rather than as a well-defined optimization problem, limiting clarity on what precisely is being optimized and why it converges.

 Incremental technical novelty:
  While the bio-inspired framing is interesting, the underlying mechanism—population-based heuristic search with local/global exploration—closely resembles existing evolutionary methods (e.g., PSO, GA). The novelty lies mainly in the metaphor and domain application rather than in the algorithmic structure itself.

 Incomplete experimental validation:
  The experiments do not compare against recent gradient-free or evolutionary prompt optimization baselines (e.g., AutoPrompt, Genetic Prompt Search, RLPrompt). Without these, it is unclear whether the gains stem from FGO’s specific design or from generic search diversity.

 Missing component ablations:
  There is no analysis isolating the contribution of the query memory pool or the local vs. global search modules, making it difficult to attribute improvements to individual components.

 Efficiency and reproducibility:
  Although token cost trends are visualized, runtime and computational overhead (e.g., API calls, GPU hours) are not reported. Given the multiple iteration cycles, the method may be expensive to run at scale. The reproducibility statement is good but lacks an immediate code release.

 Presentation:
  Some references are duplicated (e.g., Gupta et al., 2024a/b), and occasional typos (e.g., “tradining process”, line 294) and formatting inconsistencies reduce polish. The Related Work section could be more synthetic rather than exhaustive citation lists.

### Questions
1. Can the authors provide a formal objective function or theoretical justification for why the FGO-based search converges to high-quality retrieval prompts?
2. How does RAG-FGO perform compared to other heuristic optimizers (e.g., PSO, GA, simulated annealing) under identical RAG settings?
3. What is the computational cost per iteration and total token usage across datasets?
4. How sensitive is the method to hyperparameters (e.g., δ, σₗ, σ_g, kₗ, k_g)?
5. What happens if the query memory pool is disabled—does the system still converge effectively?
6. Can the trained retrieval agent generalize across domains (e.g., transfer from HotpotQA to NQ)?

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
This paper proposes RAG-FGO, a novel retrieval method for generative retrieval task, achieving higher accuracy under the same amount of iterations than the baseline.

### Strengths
* Good-quality figures
* Timely problem

### Weaknesses
* Technical content is a bit light
* Technical novelty is limited

### Questions
I am not an expert in this domain, so this review is simply from the perspective of an ordinary ML + Sys person.

* Are you also comparing against traditional RAG that does not perform generative retrieval? For me I need more background to understand why generative RAG is more favorable than traditional RAG. The source of information for me is the same between generative RAG and traditional RAG --- they all uses a set of selected documents plus the query. If the input is the same, why doing things in pipeline (like in generative RAG) instead of doing things end-to-end (like traditional RAG)?
* Can you show a case study where the global search capability provide clear benefit?
* Why FGO? I understand that it provides global search capability, but why this algorithm rather than other algorithms?

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
This paper targets on a timely topic to optimize the retrieval and generation among RAG systems from agentic perspective. A framework is proposed to incorporate fungal growth optimization into retrieval-augmented generation, formulating retrieval optimization as a dynamic search problem in semantic space. The evaluation are conducted on widely-used datasets and show the effectiveness. However, some unclear content should be addressed.

### Strengths
1. The topic is timely and there are still a lot of improvement space in terms of RAG. This study optimize it from an agentic perspective.

2. The experimental results show the effectiveness of proposed RAG-FGO compared to previous agentic systems.

### Weaknesses
1. The paper’s motivation is the limitation of generative IR and prompt tuning, while it is not clear that what is the connection with retrieval agent and why it should be related to retrieval agent? The jump between the third and forth paragraph in introduction would make reader confused. Then, why agentic systems are necessary and what are the main differences compared to existing RAG systems in terms of pipeline, principle and assumption?

2. The scenario/task definition should be clearer so the reader can know what is the main difference compared to exiting RAG or generative IR systems. Thus, a task definition section is desirable before illustrating methodology.

3. The comparison are based on previous agentic systems. Why there are not direct comparison with general RAG system and retrieval-generation pipeline systems? This is also related to the question in point 2.

4. The methodoloy of training framework is unclear, what is the final optimization objective and what is the functionality of Query Memory Pool among the training framework? (The proposed method is not training-free right? Please correct me if there is any misunderstanding.)

### Questions
What is the implementation of Agent initialization in line 198, and what is the latency/cost to perform one iteration among agents collaboration?

And the questions in weakness.

### Soundness
2

### Presentation
3

### Contribution
2
