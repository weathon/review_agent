# InternAgent-MLE: Navigating Fine-Grained Optimization for Coding Agent

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Large language models (LLMs) have shown impressive performance in general programming tasks. However, in Machine Learning Engineering (MLE) scenarios such as AutoML and Kaggle competitions, achieving high performance depends heavily on expert intervention and repeated adjustments rather than simply generating correct code. When applied directly to these tasks, LLMs often lack fine-grained domain priors, and existing MLE approaches that use linear or tree-structured searches limit knowledge transfer to adjacent hierarchical links. As a result, they cannot leverage past full trajectories or share information across branches, limiting self-evolving ability and search-space diversity.
To address these limitations, we introduce InternAgent-MLE, an LLM-based coding agent that integrates a domain knowledge base for high-quality prior guidance and Monte Carlo Graph Search (MCGS) for efficient exploration. MCGS retains the tree-guided exploration of MCTS while embedding a graph structure into the expansion stage to enable dynamic path reorganization, historical trajectory reuse, and multi-solution fusion to support both self-evolution and collaborative learning. Combined with fine-grained operator sets, this design improves stability and accelerates convergence. 
Evaluation on the MLE-Bench shows that InternAgent-MLE achieves state-of-the-art performance in numerous dimensions, such as the average medal rate and valid submission rate, under a 12-hour budget (half the standard runtime).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces InternAgent-MLE, a coding agent that helps automate machine learning engineering tasks such as those found in Kaggle competitions. The main idea is to improve how LLM agents explore and use past work: Instead of a normal tree search (which treats each attempt in isolation), the authors propose a graph-based search method (MCGS) that lets the agent recall previous attempts and combine ideas from different solution paths, also learning from failures. They also add a curated knowledge base containing common ML tricks and model guidelines. Together, this mix of elements achieves respectable performance on the MLE benchmark. The results look solid, but the approach depends on manual curation of the knowledge base and only tests on a single benchmark. Similar recent papers (e.g., MLE-STAR) report comparable or better results, so the novelty may be smaller than claimed. Overall, this is a technically strong and carefully executed system paper, but its contribution is more incremental than groundbreaking (which it would have been, one ICLR ago).

### Strengths
Paper deals with a clear and relevant limitation of current MLE agents (poor reuse and isolation in tree-based searches).
It has a somewhat novel combination of graph-based search (MCGS) and domain knowledge base, with SOTA (at the time it was written, unfortunately recently scooped by other approaches) performance on MLE-Bench.
Clear empirical gains from each component (knowledge base, intra-/cross-branch mechanisms), with demonstrated robustness across LLM backends (DeepSeek, Gemini, OpenAI). Also, paper is well writting and organised.

### Weaknesses
There is recent strong competing work, specifically MLE-STAR, which reports similar medal-rates

The novelty of MCGS over standard MCTS is mostly structural, mostly graph edge reuse.

Manual curation still has a large role, compromising the ability of this approach to scale, and potentially raising reproducibility questions

Evaluation limited to MLE-Bench; unclear generalization to non-Kaggle settings, unclear how much overfitting to some specific properties of Kaggle is happening.

No cost analysis for building and maintaining the knowledge base or for computational overhead of MCGS.

Limited interpretability analysis, how or when cross-branch references help remains opaque.

Paper does not address failure cases or negative results, leaving unclear when MCGS might degrade performance.

Lacks theoretical justification for why graph augmentation yields better convergence.

Paper would have been SOTA prior to MLEstar ...

### Questions
How is the graph structure in MCGS updated and pruned over time to prevent uncontrolled growth?
How does the agent decide when to trigger multi-branch aggregation versus normal expansion?
How large is the knowledge base, and how much manual effort was needed to curate and maintain it?
Could the knowledge base integration be automated or learned from prior runs?
What is the computational overhead of MCGS compared to MCTS (per iteration or per valid solution)?
How sensitive is performance to the choice of LLM backend?
What is the authors position regarding their approach compared to work such as MLEStar? 63% medals, 36%

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces InternAgent-MLE, an LLM-based coding agent designed for Machine Learning Engineering (MLE) tasks. The key contributions are: (1) a curated ML domain knowledge base providing task-specific priors, and (2) Monte Carlo Graph Search (MCGS), which extends MCTS by introducing graph edges for cross-branch knowledge sharing and solution fusion. Evaluated on MLE-Bench, the method achieves 36.4% average medal rate under a 12-hour budget, outperforming existing baselines.

### Strengths
1. The paper clearly identifies limitations of existing tree-based search methods, specifically node isolation and limited knowledge transfer across branches. The motivation for graph-based search is compelling.
2. MCGS extends MCTS in a clear way by introducing reference edges (E_ref) while maintaining tree edges (E_T) for backpropagation. The four expansion modes (primary, intra-branch evolution, cross-branch reference, multi-branch aggregation) provide systematic ways to leverage historical information. The approach is well motivated and sound.
3. The method achieves state-of-the-art performance with 36.4% average medal rate at the submission time (as of today, the SOTA is 43.5%) proving its efficacy.
4. The paper includes ablations demonstrating the value of each component, analysis across different LLMs, and temporal analysis showing progressive improvement.
5. The paper is generally well written and easy to follow.

### Weaknesses
1. The primary technical contribution, Monte Carlo Graph Search, is presented as a novel variant of MCTS. However, the idea of augmenting MCTS with graph structures to share information, merge nodes, or reuse trajectories has been explored in various forms in the search and planning literature. The related work section focuses heavily on other MLE agents but lacks a discussion of how MCGS differs from or builds upon other graph-based MCTS variants.
2. The curated knowledge base (KB) is a critical component, responsible for a ~9% absolute improvement in the medal rate according to the ablation study. However, its construction is described vaguely as "synthesizing practices from open-source repositories... followed by careful selection". The engineering effort and generalizability of this component are unclear. The paper would benefit from sharing at least all the prompts and pseudocode for building such a knowledge base. Moreover, the paper does not address potential data contamination – 
there is no discussion of whether the knowledge base includes competition-specific heuristics that could appear from different sources.
3. Critical design decisions lack clear specification. For example:
  - When is "cross-branch reference" triggered? What constitutes a "stagnant" branch? 
  - How is "multi-branch aggregation" triggered? The phrase "heuristically spawned" is vague.
4. Some ablations are missing but desirable, e.g., impact of knowledge base design choices (model-level vs. data-level vs. strategy-level)

### Questions
1. Could authors detail the KB decontamination: which repositories/solutions were included/excluded? Is there any overlap with the 75 tasks? Could you provide the examples how the KB looks like? Is the construction of KB an automated process or does it require manual curation?
2. What are the precise triggering conditions for intra-branch, cross-branch and multi-branch operations?
3. Could authors provide a reward structure, i.e. how different actions are rewarded/penalized?
4. How sensitive is performance to the hyperparameters in Table 4 (e.g., branch_top_k, max_ref_num)? What are the most important ones? Were these tuned on the test set?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes **InternAgent-MLE**, a coding agent designed for end-to-end Machine Learning Engineering (MLE) tasks such as Kaggle competitions. The framework integrates three core components:
1. **A curated ML knowledge base**providing model-, data-, and strategy-level priors to reduce cold-start errors of initial solutions;
2. **A Monte Carlo Graph Search (MCGS) algorithm**, extending Monte Carlo Tree Search (MCTS) by allowing nodes to incorporate information from non-parent nodes during expansion—thereby enabling trajectory recall, cross-branch reference, and multi-branch aggregation.
3. **A fine-grained operator set** that stabilizes search and improves executability.

InternAgent-MLE achieves state-of-the-art performance on **MLE-Bench**, reaching a 36.4% average medal rate and 18.7% gold medal within a 12-hour runtime—half the standard evaluation time.

### Strengths
1. The proposed MCGS mechanism allows efficient reuse of previous solutions both within branches or across branches.
2. The agent demonstrates strong empirical performance, outperforming all existing baselines on MLE-Bench in both efficiency and robustness.

### Weaknesses
1. The paper uses the term 'Monte Carlo Graph Search (MCGS)' without referencing prior work under the same name. Existing literature—such as [1] and [2]—already explores graph extensions of MCTS. The authors should clarify whether their formulation is conceptually novel and cite these precedents to avoid ambiguity.
2. The operator selection policy remains under-specified. It is unclear how the agent decides which operator to apply.
3. Details about reward propagation could be elaborated—particularly how rewards are updated across parent nodes.

**References**

[1] Czech, Johannes, Patrick Korus, and Kristian Kersting. 'Monte-Carlo graph search for AlphaZero.' arXiv preprint arXiv:2012.11045 (2020).

[2] Leurent, Edouard, and Odalric-Ambrym Maillard. 'Monte-carlo graph search: the value of merging similar states.' Asian Conference on Machine Learning. PMLR, 2020.

### Questions
1. **Novelty clarification:** Has MCGS been attempted in other coding-agent contexts besides MLE? Or more generally, has a similar graph-structured search been attempted in other coding-agent contexts besides MLE?
2. **Graph structure:** In Figure 2 and Eq. (8), should the new node created by multi-branch aggregation indeed be connected to the root node to somehow denote it as a starting node.
3. **Operator selection:** How is the next operator chosen?
4. **Reward updates:** During backpropagation, how do all ancestors along the trajectory update their reward values?
5. **Notation consistency:** In Line 293, 'graph-level top-$k$' seems to correspond to the 'top-$N$' reference set in Eq. (7); aligning the notation would improve clarity.
6. **Edge usage:** Since $E_{ref}$ is excluded from backpropagation, is it still necessary to keep this recorded?

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
This article introduces a novel large language model (LLM) agent, **InternAgent-MLE**, designed to enhance the **stability and convergence speed** of LLMs in handling **Machine Learning Engineering (MLE**) tasks. Specifically, the paper builds upon the existing **Monte Carlo Tree Search (MCTS**) framework and proposes the use of **Monte Carlo Graph Search (MCGS**) to optimize the agent's ability to leverage **historical information**. Additionally, it establishes a connection between a curated knowledge base and the LLM to improve **warm-start efficiency**, and employs **parallel exploration** to accelerate convergence.
The article provides a deep analysis of the limitations of current LLM agents utilizing MCTS. It focuses on the **isolation of information within different branches** as a key motivator for improving the agent’s capabilities. The main body of the paper introduces the search logic of MCGS and the naming of different types of nodes based on their functions. It then discusses the operational mechanisms of MCGS, including the enablement and updating of **primary and reference edges**, around Figure 2.
In the experimental section, the authors use **MLE-Bench** and **MLE-Bench-lite** as evaluation datasets and compare the performance of various methods with InternAgent-MLE in terms of **medal rates and times to Medal at different granularity levels**. They also compare the **beat ratio** between InternAgent-MLE and the baseline agent, demonstrating the strength of InternAgent-MLE. Furthermore, the paper compares the performance of InternAgent-MLE using different LLMs for **image, text, and tabular tasks**, proving its adaptability to different LLMs. Additionally, **ablation experiments** are conducted on the three main components of the new method, demonstrating their effectiveness and necessity. The appendix also provides a detailed supplement to the method’s functions, including **hyperparameter values for method calls and performance on various tasks**, mentioning the functional preferences of different LLMs and the agent’s ability to stimulate their strengths.

### Strengths
**Originality:**  
This paper pioneers the integration of **Monte Carlo Graph Search (MCGS)** into the search strategy of **large language model (LLM) agents**, complemented by a **curated knowledge base** and **parallel exploration** mechanisms. Each key component is meticulously designed to address concrete challenges in Machine Learning Engineering (MLE), such as **node isolation** in traditional tree-based searches and **cold-start inefficiencies**. The solutions are well-justified with empirical evidence, yielding distinctive results (e.g., 36.4% average medal rate) and demonstrating high originality.  
**Quality:**  
In the experimental section, the paper evaluates multiple methods across various metrics on the authoritative **MLE-Bench** dataset. It conducts **ablation studies** on critical components (e.g., knowledge base, MCGS component) and tests the adaptability of **InternAgent-MLE** to different LLM backends (e.g., DeepSeek-R1, Gemini-2.5-pro). The methodology is rigorous, results are clear and reproducible, and the comparative analysis underscores the superiority of the proposed framework.  
**Clarity:**  
The paper provides a concise yet comprehensive explanation of **MCGS’s operational logic**, including its expansion mechanisms (e.g., intra-branch evolution, cross-branch reference) and edge types (primary vs. reference edges). Experimental hyperparameters (e.g., UCT constant, parallel workers) and implementation details are thoroughly documented (Table 4), ensuring high transparency and reproducibility.  
**Significance:**  
This work addresses the critical limitation of **branch isolation** in Monte Carlo Tree Search (MCTS) by enabling **trajectory reuse** and **cross-branch knowledge sharing**. It significantly boosts success rates on low-complexity tasks (62.1% vs. 48.5% for the best baseline) while reducing task completion time by **66%** (12 hours vs. 36 hours for Neo). The balance between **high-quality output** and **efficiency** makes InternAgent-MLE a powerful tool for MLE tasks. The detailed parameter configurations and benchmark results provide invaluable resources for future research.

### Weaknesses
The paper's description of **Monte Carlo Graph Search** is concise, which effectively communicates core concepts to domain experts. However, this brevity may pose challenges for **cross-disciplinary readers**. Including more detailed operational logic and illustrative diagrams of node interactions in the appendix could enhance accessibility for a broader audience.  
Additionally, the results in **Table 1** reveal that **InternAgent-MLE** underperforms compared to **Neo** on **medium-complexity tasks**, despite demonstrating strong performance on both **low-complexity** and **high-complexity** tasks (Figure 4). This counterintuitive trend raises questions about the underlying causes. A discussion analyzing why performance specifically declines in **medium-difficulty tasks** would significantly improve the method's **theoretical coherence**.  
There is an extraneous ")" at the end of line 195.

### Questions
The results in Table 1 show that InternAgent-MLE's performance falls slightly behind Neo on Medium Task Complexity tasks, yet it demonstrates superior performance on both Low and High Task Complexity tasks (as further supported by Figure 4). This observation raises a question: Intuitively, if a method's advancements enable it to outperform a baseline at both the lower and higher bounds of difficulty, it should logically maintain its lead on the intermediate difficulty level as well. The current result of underperformance on the medium level seems counter-intuitive. Could you please provide an explanation for this specific performance dip at the medium complexity level?

### Soundness
4

### Presentation
3

### Contribution
3
