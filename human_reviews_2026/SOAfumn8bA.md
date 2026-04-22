# CrystalAgent: Towards Autonomous Crystal Generation via Agentic Reasoning

- Avg Score: 3.50
- Decision: Reject
- Scores: 8, 2, 4, 0

## Abstract
Recent advances in large language models (LLMs) have demonstrated remarkable generalization capabilities across diverse domains, and recent studies have begun to explore their application in crystal generation. Nevertheless, most of these approaches rely heavily on extensive fine-tuning with large-scale datasets, which often limits their adaptability and generality when applied to real-world crystal discovery. To overcome these limitations, we propose CrystalAgent, an LLM-based agent that eliminates the need for additional training and adapts flexibly to diverse crystal discovery scenarios. Specifically, we decompose the crystal generation process into four key stages: Extract, Retrieval, Generation, and Optimization. The Extract stage involves extracting crystal design constraints from user inputs. In the Retrieval stage, based on the extracted constraints, the system automatically selects few-shot examples from the database to inform subsequent processes. The Generation stage leverages LLMs to generate crystal structures by learning atomic distribution patterns from the selected examples, and the Optimization stage refines the generated structure by using crystal structure optimization tools and energy evaluation tools to select the optimal structure as the final output. Extensive experiments across various crystal generation tasks highlight the flexibility, controllability, and versatility of our framework, underscoring the substantial potential of LLM agents in automating the generation of crystal materials and advancing the field of materials discovery.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The work proposes an agentic framework (CrystalAgent) for crystal generation with the main claim that CrystalAgent eliminates the need for additional training and provides more flexibility in comparison with the current solutions in diverse crystal discovery setups. Based on user query consisting of design constraints, CrystalAgent retrieves few-shot examples best matching the query based on element proportions (anonymized formula) and structural features (Wyckoff prototype) represented in symmetry-based encoding and used for conditional generation by LLM. Framework was validated in one unconditional and two conditional tasks including i) unconditional crystal structure generation (CSG), ii) structure prediction (CSP), and iii) structure design (CSD).

### Strengths
1. Genuinely relevant challenge of flexible crystal design.
2. High clarity and design of the paper.
3. Accessible explanation of mathematical concepts used in the work.
4. Broad framework validation in three setups (CSG, CSP, and CSD).

### Weaknesses
1. The rationale behind the use of LLM for crystal generation is not well posed in the paper, where plenty of models exist that work on graph structures, intrinsically respect symmetries, and seem to be more natural to use while LLM is often the best choice for data natively represented by text. Moreover, it would be necessary to probe other LLMs besides gpt-4o on these tasks for comprehensive comparison since performance on the specific tasks often depends strongly on the model.
2. Some of the claims (e.g., "dependence on extensive training data restricts generalization" in line 62) are either poorly formulated or incorrect since contradict with well-established observations, which should be checked more carefully. On the contrary, informing the model with few-shot examples sampled from MP based on the specified constraints is more biased and can restrict the generalization. Authors should pay more attention to retrieval stage e.g., ways of informing the model with examples, optimal thresholds for examples retrieval etc.
3. Benchmarking against more recent SOTA models should be performed (DiffCSP++, con-CDVAE, WyckoffDiff, MatterGen, SymmCD), where CSD task also lacks comparison with existing solutions or/and some baselines.
4. Thermodynamic stability of generated materials should be evaluated (formation energy, energy above the convex hull).

### Questions
1. What will the framework do if so happens there are no structures matching the constraints close enough? What happens if the specified set of constraints can't be satisfied theoretically? Such cases should be studied in more details to find edge cases characterizing framework limitations.
2. Are fractional coordinates optimal (many cells can describe the same crystal, orientation/permutation variance for F, not ensures fine interatomic distances, nearly singular L are easy to generate but hard to relax etc.)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents the framework CrystalAgent which uses large language model agents to generate crystalline materials autonomously. The workflow comprises four stages: extract design constraints from user input, retrieve relevant few-shot examples from a crystal structure database, generate candidate crystal structures with an LLM trained via in-context examples, and finally optimize those structures using physics or energy tools to select the most stable output. Experiments across multiple crystal generation tasks are presented to demonstrate flexibility and adaptability of the approach to diverse crystal discovery scenarios.

### Strengths
1. The paper introduces CrystalAgent, an LLM-based framework for autonomous crystal generation, which is a conceptually interesting and timely idea for materials discovery.
2. The motivation to bridge data-driven CSP/CSD and flexible LLM reasoning is well framed.
3. The manuscript is generally well organized and written clearly.

### Weaknesses
1. No code is provided despite the reproducibility statement.
2. Missing discussion of limitations and future directions.
3. Retriever design lacks clarity and has no ablation support.
4. Claims about bridging data scarcity are unsubstantiated.
5. The evaluation set appears too small and not well justified.
6. Baseline comparisons are inconsistent and omit important recent methods.
7. Quantitative results lack error bars or deviations.
8. Validation is purely in silico without supporting artifacts or data release.

### Questions
1. _“...we propose CrystalAgent, an LLM-based agent that eliminates the need for additional training and adapts flexibly to diverse crystal discovery scenarios.”_

How is this demonstrated? I do not find specific examples of that in the paper.

2. The title “towards” assumes the proposed framework is not definitive but rather an important step in the direction of autonomous agent-based crystal generation. In this context, it is crucial to have a dedicated section on limitations of the current approach and prospects of the agentic frameworks. Both are missing at the moment. Could the authors address this?

3. _“In the Retrieval stage, based on the extracted constraints, the system automatically selects few-shot examples from the database to inform subsequent processes.”_

Retriever design would benefit a lot from a visual representation of its components and key steps or an algorithm highlighting the logical steps. Are there any ablation studies supporting the described design? Could the authors include a schematic or algorithmic breakdown of the Retriever stage?

4. Up until Section 4, the authors refer to the database as the data source used by the proposed framework. However, even from Section 4.1 it is not entirely clear what exactly is that database. Could the authors clarify?

5. _“The Challenge Set (Antunes et al., 2024) consists of 70 crystalline compounds… we evaluate on the 58 compounds from recent literature.”_

Was the framework validated using these 58 compounds? How is it guaranteed they are absent from the training data? Which training data?

6. The test set of 58 compounds seems unreasonably small. Could the authors explain this design choice in detail, including aspects of generalizability, representativeness, and bias in evaluation metrics?

7. Why is the set of baselines different across Tables 1 and 2?

8. Why don’t the authors present error bars or deviations for any evaluations?

9. In my view, Table 1 does not present convincing evidence of the effectiveness of CrystalAgent in the CSG task. Structural Validity Rate and Coverage metrics are not informative since all methods score similarly. For the other metrics, CrystalAgent is somewhat better than CrystalTextLLM, FLowLLM, and CDVAE in Composition Validity Rate, but significantly worse otherwise. The authors advocate that another advantage of CrystalAgent is that it does not require fine-tuning, but I don’t see this as a serious argument in this context. Can the authors provide additional experimental results to support their claims?

10. Table 2 is suggesting that CrystalAgent is superior to other methods, however, there are multiple questions to this comparative analysis. Why is the Challenge Set evaluation limited to a single baseline, i.e. CrystaLLM? This appears as a major limitation. Also, to my knowledge, there are at least two versions of CrystaLLM – which one was used?

11. Table 3 shows promising results, but I find no data assets, repositories, or artifacts supporting this. Can the authors provide them? 

12. More generally, do the authors intend to release code, data assets, or artifacts to support reproducibility? Why aren’t those included in the submission? 

13. Were only in silico models and tools used for validation? 

14. The paper omits some recent and relevant works, e.g. SymmCD, DiffCSP++, deCIFer. Could the authors discuss how CrystalAgent compares to these methods conceptually and experimentally?

15. The authors claim that CrystalAgent bridges the gap caused by data scarcity, but the framework relies on a single database with the same limitations. How is this gap addressed in practice?

16. Can properties of experimental synthesis or synthetic accessibility by itself be important constraints for the CSP and CSD tasks? Why aren’t these considered or discussed?

17. The prompts in Appendix appear quite simple. Did the authors use prompt optimization tools or techniques? Please describe your approach to prompt and context engineering.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CrystalAgent, a four-stage, training-free LLM agent for crystal structure generation and related tasks (CSG, CSP, and constrained CSD). The system decomposes the problem into (1) constraint extraction from user input, (2) retrieval and constraint completion from an in-domain crystal database (MP-20/MPTS-like), (3) in-context, constraint-aware structure generation with a general LLM, and (4) tool-based optimization/validation of candidates. On MP-20, MPTS-52, and a recent challenge set, the method reports very high validity and coverage and slightly higher match rates than prior LLM-based systems, while claiming not to fine-tune the base model.

### Strengths
1. The 4-stage workflow (extract $\rightarrow$ retrieve/complete $\rightarrow$ generate/check $\rightarrow$ optimize) is easy to follow and sits well within the current agentic AI for science. It shows how to make a general LLM act safely on domain-specific, highly structured outputs (crystals).

2. Using retrieval to fill in missing space groups / Wyckoff prototypes and then conditioning generation on them is a realistic design for real users who cannot specify full crystal details.

### Weaknesses
1. The proposed method runs on GPT-4o-mini, whereas several LLM baselines (CrystalTextLLM, MatExpert, and earlier CrystaLLM/Mat2Seq) rely on older/smaller backbones (LLaMA-2/3 or GPT-2–level). Part of the gain may therefore come from a much stronger, more recent model rather than from the agentic pipeline itself; this is a confound and makes the current comparison only partially fair.

2. Recent work MatLLMSearch [1]: Crystal Structure Discovery with Evolution-Guided LLMs follows the same high-level philosophy (frozen LLM + external loop/evaluator for crystal discovery) but is not compared or even discussed, so it is hard to see whether CrystalAgent is more sample-/compute-efficient than an evolutionary outer loop.

3. The method relies heavily on in-domain retrieval from MP-20 to provide in-context examples and to complete constraints. This reliance blurs the line between “no training” and “using the train set as a memory,” and raises the question of how well the method works on genuinely out-of-distribution sources.

4. Most evaluations use surrogate/ML potentials and structural metrics; there is little DFT-level or experimental validation to show that the generated candidates are actually useful to materials scientists.

5. With several moving parts (constraint completion, similarity-based retrieval, generation-time checks, optimization), the paper needs clearer ablations to isolate which components actually drive the improvements.

[1] Gan J, Zhong P, Du Y, et al. Large language models are innate crystal structure generators[C]//AI for Accelerated Materials Design-ICLR 2025. 2025.

### Questions
1. Can you report CrystalAgent with a matched backbone (e.g., LLaMA-2-13B / LLaMA-3-70B) to show that the pipeline itself, not GPT-4o-mini, is responsible for the improvements? Alternatively, can you re-run at least one LLM baseline under GPT-4o-mini with your retrieval + check settings?

2. How does CrystalAgent compare to MatLLMSearch under the same evaluator (same ML potential / same DFT budget) in terms of validity, metastable/stable rate, and candidate diversity? What is the compute cost difference between your fixed 4-stage pipeline and their evolutionary loop?

3. Since your retrieval pool is MP-20-like and your evaluation is also on MP-20/MPTS, what happens if the reference database is swapped to another source (e.g., OQMD/ICSD subset) or you evaluate on a held-out recent literature set without close neighbors?

4. What is the fraction of generations that pass your constraint checks on the first attempt, and how many re-generations are typically needed per final valid structure? This matters for real-time agentic usage.

5. Which structure optimization/energy evaluation tools are actually used in stage 4, and do their training data overlap with MP-20? Can you clarify the cost per candidate and whether this step ever rejects structures that pass stage 3?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The work proposes CrystalAgent, which is a framework for crystal structure generation that uses a large language model (LLM) in a four-stage pipeline: extracting constraints from natural language inputs, retrieving similar structures from a database, generating candidate structures via in-context learning, and optimizing them using external tools. The system achieves good performance on crystal structure generation, prediction and design (CSG, CSP, CSD) without task-specific fine-tuning.

### Strengths
1. The work uses symmetry-based encoding that represents crystals using space groups, lattice parameters, and Wyckoff positions rather than raw atomic coordinates. The choice is reasonable and physically sound.

2. The system can handle multiple tasks (CSG, CSP, CSD) without specific fine-tuning, given the advantage of tool use.

### Weaknesses
W1. The optimization step is fundamentally flawed. Comparing total energy is valid for comparing polymorphs of the same composition (CSP tasks only), but it does not apply to CSG or CSD. It is formation energy that you want to approach. Such described 'optimization' would eventually favor structures with certain patterns e.g. fewer atoms simply because they have lower total energy.
W2. The concept of 'thermodynamic stability' is abused in the draft. Overclaims like 'ensures that the final output is ... thermodynamically stable' were emphasized multiple times. Expressions like 'ensure structural stability and physical plausibility' are scientifically invalid. In addition, a local minima does not guarantee the structure is stable. Even an M3GNet relaxed crystal structure with negative formation energy only indicates the structure is metastable with respect to decomposition.  
W3. The work is mispositioned as "agentic reasoning". The claimed "agentic reasoning" is a deterministic pipeline of retrieval, few-shot prompting, and static tool use for validation checks. It shows no sign that the designed system can benefit from autonomous, adaptive reasoning from a real agentic system.
W4. Instead of 'training-free', 'fine-tuning free' is more appropriate, as you concluded in the last paragraph.

### Questions
Q1. Is the optimization step shared by all tasks?

### Soundness
1

### Presentation
4

### Contribution
1
