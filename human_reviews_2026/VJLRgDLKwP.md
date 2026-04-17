# DHEvo: Data-Algorithm Based Heuristic Evolution for Generalizable MILP Solving

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Primal heuristics are crucial for accelerating the solving process of mixed integer programming (MILP) problems. While large language models (LLMs) have shown great promise in generating effective heuristics, existing methods often fail to generalize across instances within the same problem class, where we define a problem class as a set of MILP instances derived from the same mathematical model. This limitation arises because MILP instances within the same class can exhibit substantial structural and distributional heterogeneity. However, existing methods treat instances uniformly, averaging performance over limited samples and yielding heuristics that lack generalization. To address this, we propose DHEvo, a data-algorithm co-evolution framework that jointly evolves representative instances and tailored heuristics integrated into the open-source solver SCIP. DHEvo employs an LLM-based multi-agent system to generate and refine data-algorithm pairs iteratively, guided by fitness feedback. Experiments on diverse MILP benchmarks show that DHEvo significantly outperforms state-of-the-art hand-crafted, learning-based, and  LLM-based methods in solution quality and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DHEvo, a novel framework that co-evolves Mixed-Integer Linear Programming (MILP) problem instances and diving heuristics using a Large Language Model (LLM)-based multi-agent system. The core idea is to iteratively select "structurally representative" instances to guide the evolution of more generalizable heuristics, addressing the problem of high intra-class instance heterogeneity.

### Strengths
1. The concept of co-evolving both the data (MILP instances) and the algorithms (heuristics) is a creative contribution. Moving beyond a static training set to an adaptive, evolutionary selection of "representative" instances is a useful approach to improving generalization.
2. The experimental section is comprehensive. The method consistently outperforms a wide range of baselines on a lot of academic datasets and real-world problems.

### Weaknesses
1. The theoretical analysis in Section 3.2 is presented as a motivation. However, it does not explain why co-evolution on representative instances works; it only states that if you have a good set of base heuristics, combining them is safe. It does not address the core challenge of intra-class heterogeneity or explain how co-evolution helps. 
2. The prompt engineering details in the appendix are excellent. However, it would be beneficial to include a brief discussion in the main text about how it is being designed and why this PE works.

### Questions
1. Can you demonstrate the generalization advantage of DHEvo more robustly, for example, by validating the generalization capabilities across different types of problem classes?

### Soundness
2

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
This paper introduces DHEvo, a data-algorithm co-evolution framework that automatically generates generalizable diving heuristics for Mixed Integer Linear Programming (MILP) problems. DHEvo uses a multi-agent, LLM-driven evolutionary system to jointly evolve MILP instances and their heuristics. It identifies representative instance–heuristic pairs that generalize well within problem classes.  Experiments on synthetic benchmarks and real-world datasets show that DHEvo significantly outperforms state-of-the-art hand-crafted, learning-based, and LLM-based methods in solution quality (primal gap), variance, and solving efficiency.

### Strengths
- The idea of simultaneously evolving both instances and heuristics is well-motivated, justified, and proven effective through ablation studies.
- Using multiple LLM agents to generate, review, and select heuristics is methodologically sound. Ablation studies show this approach improves both solution quality and heuristic diversity.
- The experiments use synthetic and real-world datasets with clear performance metrics—such as primal gap and primal-dual integral—supporting the authors' claims through rigorous ablation studies.

### Weaknesses
- **Scope Limited to Diving Heuristics:** While the use case is well-motivated and relevant, DHEvo is applied only to diving heuristics. It remains unclear why the method targets this specific class of heuristics and whether it can extend to other MILP heuristics or broader combinatorial optimization problems. A discussion of limitations or necessary adaptations for other domains would strengthen the paper.
- **Computational and Resource Cost Reporting:** The paper lacks a fair comparison of computational resources, particularly the time and token consumption of LLM-based methods (including DHEvo and baselines). This information is essential for assessing practical deployment under real-world compute constraints.
- **Novelty with Respect to Prior Work:** The multi-agent, LLM-driven evolutionary search paradigm—including agent-based code design and review—has been explored in recent literature. The paper would benefit from clearly articulating which methodological and conceptual advances are novel to DHEvo.

### Questions
1. Is DHEvo in its current form specific to diving heuristics, or could it be extended to other primal heuristics, branching rules, or even non-MILP algorithms? What would need to be adapted for such generalization?
2. What is the average/total compute time, number of tokens, and any incurred API costs for running DHEvo compared to LLM-based baselines? This is important for practical adoption and reproducibility.

**Minor Comments**
- Eq. (2): Have you introduced and explained $\Delta^{k-1}$ before using it here?
- Line 366: Duplicated "Experimental setup"

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
5

### Summary
The paper proposes DHEvo, a data–algorithm co-evolution framework for generalizable MILP solving. It co-evolves representative instances and diving heuristics via an LLM-based multi-agent system (Designer/Coder/Reviewer/Judge), selecting high-fitness instance–heuristic pairs across generations, and integrates the learned heuristics into SCIP. Evaluations on four CO datasets (cauctions, setcover, facilities, indset) and three real-world settings (LoadBalance, NNVerify, MIPLIB) show consistent improvements in primal gap, solving time, and PDI over hand-crafted heuristics, learning baselines (L2DIVE), and LLM-based methods (LLM4Solver, FunSearch, EoH).

### Strengths
- Interesting and timely idea: jointly evolving data (instances) and algorithms (heuristics) to improve generalization within a problem class.
- Clear instantiation: multi-agent LLM pipeline, well-described evolutionary operators, and fitness-driven selection with temperature-based sampling.
- Strong empirical results: consistent gains over hand-crafted, learning-based, and LLM-based baselines; improvements carry over to full B&B with SCIP, and to real-world datasets.
- Ablations are helpful: show benefits of co-evolution, MA-Evolution, and data selection policies (simple/representative instances).

### Weaknesses
- Missing a key baseline: ReEvo is a prominent agent-based code-generation/evolution approach. Lack of comparison weakens the contribution claim, especially since DHEvo’s pipeline shares the agentic-evolution flavor.

- Benchmark breadth: classic LLM+EC works (e.g., EoH, FunSearch) commonly report on canonical CO tasks like Bin Packing and TSP. Not evaluating on such benchmarks limits comparability to the broader literature and makes it harder to assess transferability beyond MILP templates used here.


[1]Ye H, Wang J, Cao Z, et al. Reevo: Large language models as hyper-heuristics with reflective evolution[J]. Advances in neural information processing systems, 2024, 37: 43571-43608.

### Questions
- Will you add ReEvo as a baseline? At minimum, reproduce ReEvo’s agent framework on your four CO datasets with aligned prompts/evaluation to quantify the margin.

- Can you evaluate DHEvo on Bin Packing and TSP to match canonical CO reporting in EoH/FunSearch? Even a subset would strengthen external validity.

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
The paper proposes DHEvo, a framework that co-evolves the heuristic function and the data in MILP solving, aiming to obtain heuristic functions with better generalization across different types of problems. The proposed approach demonstrates superior performance compared to several baseline methods.

### Strengths
- The paper provides a solid theoretical motivation, offering guidance for the joint evolution of heuristics and data.
- The experiments include ablation studies on the main components of DHEvo, showing a well-structured evaluation.

### Weaknesses
Several parts of the paper are unclear or insufficiently explained; see the questions below.

### Questions
1. The key idea of DHEvo is the “co-evolution of data and heuristics.” However, according to *Algorithm 2: DHEvo Framework*, it seems that DHEvo does not actually synthesize new data but rather modifies the evaluation data during heuristic evolution. Could the authors clarify this point?
2. Could the authors compare the experimental results in Tables 1, 3, and 4 with those obtained using Gurobi?
3. In the appendix, the authors mention that 20 MILP instances were selected from MIPLIB for evaluation. Since MIPLIB contains hundreds of instances, what criteria were used to select these 20? Could the authors evaluate DHEvo on additional MIPLIB instances?
4. During the evolution and evaluation phases of DHEvo, are the same data used? In other words, did the experiments distinguish between a “training set” (used for heuristic evolution) and a test set?
5. The paper claims that DHEvo addresses the issue of “yielding heuristics that lack generalization” in existing LLM-based methods. Which experiment specifically evaluates the generalization ability of the heuristics generated by DHEvo?

### Soundness
3

### Presentation
3

### Contribution
3
