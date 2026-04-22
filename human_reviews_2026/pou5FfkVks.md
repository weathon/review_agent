# ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Search agents powered by Large Language Models (LLMs) have demonstrated significant potential in tackling knowledge-intensive tasks. Reinforcement learning (RL) has emerged as a powerful paradigm for training these agents to perform complex, multi-step reasoning. However, prior RL-based methods often rely on sparse or rule-based rewards, which can lead agents to commit to suboptimal or erroneous reasoning paths without the ability to recover. To address these limitations, we propose \textbf{ReSeek}, a novel self-correcting framework for training search agents. Our framework introduces a self-correction mechanism that empowers the agent to dynamically identify and recover from erroneous search paths during an episode. By invoking a special \textbf{JUDGE} action, the agent can judge the information and re-plan its search strategy. To guide this process, we design a dense, instructive process reward function, which decomposes into a correctness reward for retrieving factual information and a utility reward for finding information genuinely useful for the query. Furthermore, to mitigate the risk of data contamination in existing datasets, we introduce \textbf{FictionalHot}, a new and challenging benchmark with recently curated questions requiring complex reasoning. Being intuitively reasonable and practically simple, extensive experiments show that agents trained with ReSeek significantly outperform SOTA baselines in task success rate and path faithfulness. Our code and dataset are available at https://anonymous.4open.science/r/Re-Search-5A0F.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
ReSeek  wants to address the problem of LLM-based search agents getting stuck on erroneous reasoning paths due to sparse rewards. The paper proposes a self-correcting framework in which an agent can invoke a special JUDGE action to pause mid-trajectory, assess the usefulness of retrieved information, and re-plan its search strategy if the current path is unproductive. A dense, instructive reward function provides fine-grained feedback by decomposing into a correctness reward (to encourage retrieving factual, relevant information) and a utility reward (to encourage finding information useful for answering the query). The authors also introduce FictionalHot, a new benchmark of synthetic multi-hop questions about fictional entities (designed to avoid training data contamination) to rigorously test the agent’s reasoning ability. Extensive experiments on eight QA datasets (spanning single-hop and multi-hop questions) show that agents trained with ReSeek achieve higher answer accuracy and more faithful reasoning chains than state-of-the-art baselines.

### Strengths
The extensive experimentation across a wide range of benchmarks, including FictionalHot, demonstrates the effectiveness of the proposed method. The results show consistent improvements in performance, particularly in multi-hop question answering, where the self-correction mechanism significantly outperforms traditional search-augmented models.

### Weaknesses
1. **Limited Innovation Relative to Search-R1 and Similar Approaches**
   The differences between ReSeek and methods like Search-R1 mainly lie in the prompt design and reward calculation mechanisms. The core idea of integrating search with reinforcement learning (RL) is not new. While the introduction of the JUDGE action and a more detailed reward function is interesting, the overall innovation may not be as groundbreaking as suggested. It feels like an incremental improvement rather than a novel paradigm shift.

2. **Lack of Detailed Explanation for `rerank_score` and Threshold Choice**
   The concept of rerank_score (Line 223) and how it is used to assess whether a label is present in the retrieved information is not explained thoroughly in the paper. After reviewing the code up to the version updated on October 14th, it seems the label presence in the retrieved information is being used as the ground-truth (GT) for effectiveness. This is not discussed clearly in the text, and the approach may need further clarification. Additionally, the choice of a threshold of 0.7 is mentioned, but it is not clear what this threshold is specifically measuring. Is it based on the similarity between the retrieved content and the GT answer? Further details on how this threshold was determined and its impact on performance would be helpful.

3. **Inconsistent Training Set Details**
   In Line 313, the paper states that “we fine-tune on a unified training set that merges the NQ and HotpotQA training splits,” but later, in Line 724, it mentions training on the HotBenchmark dataset. This discrepancy needs to be clarified. More details on the integration of these datasets are needed to avoid confusion.

4. **Lack of Specifics on the FictionalHot Benchmark**
   While the paper mentions that FictionalHot is created from a 10% random sample of seed questions from existing benchmarks, there are no details on how large the FictionalHot dataset is overall. The paper does not provide any information on the number of questions or the size of the full dataset. Additionally, the approach to creating this synthetic benchmark based on LLM-generated content raises concerns about accuracy and potential conflicts with real-world facts. The paper does not explain how entities that are fictional could still be consistent and internally coherent. Was there any human review or oversight to ensure the quality and correctness of these fictional entities? An analysis of failure cases in FictionalHot could provide valuable insights into the robustness of the benchmark.

### Questions
1. Please provide more details on how `rerank_score` is computed and how the model determines the effectiveness of retrieved information in relation to the ground-truth answer.

2. The threshold of 0.7 for `rerank_score` needs further explanation. Clarify why this value was chosen and its impact on the model's performance.

3. In Line 313, the paper mentions merging the NQ and HotpotQA training splits, while in Line 724, the HotBenchmark dataset is mentioned. Please clarify how these datasets are combined or used separately.

4. More information is needed on the size of the **FictionalHot** dataset. Specify the number of questions it contains and how they were generated.

5. Clarify whether there was human oversight to ensure the consistency and quality of the fictional entities in the **FictionalHot** dataset.

6. An analysis of failure cases in the **FictionalHot** benchmark would provide valuable insights. Please share examples or explanations of why some cases failed.

7. In Table 7, it seems that a real-world search engine can also answer FictionalHot questions correctly. Could you clarify the experimental setup for this test?

### Soundness
2

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
5

### Summary
This paper focuses on search agents trained with agentic RL. Existing search agents lack mechanisms to re-plan when they generate incorrect queries. The authors introduce a JUDGE action with additional rewards, enabling dynamic evaluation and re-planning during search episodes. They also propose FictionalHot, a new benchmark designed to mitigate data contamination.

### Strengths
* **Well-Motivated Problem**. The paper effectively motivates the limitations of existing RL-based search agents that commit to erroneous paths without recovery mechanisms.
* **Simple and Reasonable Design**. The core idea of having agents evaluate intermediate steps is well-established in the self-critique and reflection literature. The main contribution is adapting this to search agents with a specific implementation. Although this is somewhat incremental, using the JUDGE action to perform self-correction is reasonable and can help search agents recover from errors.
* **Comprehensive Experimental Evaluation**. The evaluation on diverse benchmarks achieves good results.

### Weaknesses
1. **Gap between paper and implementation**. I carefully reviewed the code provided by the authors, especially focusing on the implementation of the JUDGE action in the pipeline and the reward function. The reward function and corresponding hyperparameters in `verl/utils/reward_score/search_r1_like_qa_em_s4.py#L206` are slightly different from those presented in this paper.
2. **Reward hacking**. In `verl/utils/reward_score/search_r1_like_qa_em_s4.py#L118`, the authors calculate the judge reward multiple times. In this case, a 1-turn correct trajectory may receive a lower reward score than a multi-turn incorrect trajectory that receives multiple judge rewards, which might lead to reward hacking and an increased number of search calls. The authors should increase the maximum number of turns **during testing** (e.g., to 10) and report the average number of tool calls compared to Search-R1.
3. **Wrong judge reward implementation**. In `verl/utils/reward_score/search_r1_like_qa_em_s4.py#L23`, the implementation can be easily hacked by this example:
```
<information>Info 1</information>
<information>Info 2</information>
<judge>Yes</judge>
```
We expect a JUDGE action for each `<information>` tag, but the implementation is incorrect, leading to erroneous judge rewards.

4. **JUDGE action does not work during the inference pipeline**. Although the authors introduce the JUDGE action in Section 3.2 and Eq. (2), the JUDGE action does not affect the inference pipeline, as shown in `scripts/runs/reseek/reseek_search/llm_agent/generation.py#L532`. When a judge action is generated, it is omitted and does nothing. This raises the question of whether the judge action actually works. Comparing Figures 8 and 10 in the appendix, consider the first search action. Although the first judge is Yes and the second is No, they behave identically: the first hop has been answered, and the agent starts the second hop. The agent does not re-plan even when the judge is No.

### Questions
1. Which checkpoint of Search-R1 was used in Figures 1, 7, and 9? A well-trained Search-R1 model with 7B or 14B parameters should be able to correctly decompose multi-hop questions rather than simply treating the entire question as the search query on the first turn.
2. Equation 1 does not correctly use subscript notation for the expectation. Additionally, `o_t` is not introduced in Equation 2 until line 224 on the next page.

### Soundness
2

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
5

### Summary
This paper explores an interesting problem of improving reinforcement learning for training a search agent. To be specific, the authors introduce a new “judge” action after the information is retrieved to figure out if the information is relevant or not. They use a reranker to provide a silver label of the relevance of the documents and compute the intermediate retrieval correctness reward. The final reward consists of two parts: a final answer reward and an intermediate judgment reward. They also propose a new dataset, FictionalHot, to alleviate the data contamination problem associated with existing QA datasets. Experiments on several datasets demonstrate the effectiveness of the proposed ReSeek method.

### Strengths
1. The problem tackled in this paper is important and valid
2. The authors propose a “judge”-based action and introduce a process reward for denser reinforcement learning.
3. A new dataset, FictionalHot, is introduced in the paper for a more comprehensive evaluation.
4. Experiments are conducted to demonstrate the effectiveness of ReSeek.

### Weaknesses
1. The writing is not very clear in some sections:
- Although a “reranker” is plotted in Figure 2 to illustrate the calculation of the silver passage relevance labels, it is not discussed in the main paper. This makes the understanding of this part challenging.
- Eq.(2) is not clear. From Eq.(2), it seems that the context C_t is dynamically adjusted based on the judgment j_t, which means that only retrieved information that is judged as useful is concatenated to the context window. However, this is not the case, as referring to the cases shown in section A.5. In addition, it is unclear what \tau_{t-1} is in Eq.(2).
- It is not discussed in the paper what the final reward looks like. We can guess from Figure 2 that it is a combination of the final answer reward and intermediate judgment reward. However, there is no concrete illustration.


2. Some of the experiments are unclear: It is questionable whether the “sensitivity to the retrieval encoder” experiments refer to doing RL training with different retrieval encoders or training with the same but testing with different retrieval encoders. It makes the conclusion hard to understand.

3. Some of the experiments are not new: similar studies comparing base and instruction-tuned backbones for RL have already been conducted in the original Search-R1 paper, and the observation is not new.

4. Some minor typos: line 161 “xxx”,

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper targets a key limitation of LLM-powered search agents: sparse rewards that cause unrecoverable errors in multi-step reasoning. It proposes ReSeek, a reinforcement learning framework with a self-correction mechanism via a special JUDGE action, enabling agents to evaluate and adapt search paths on the fly. ReSeek employs a dense process reward decomposed into correctness and utility to guide factual and relevant retrieval. The authors also introduce FictionalHot, a benchmark built around fictional entities to mitigate data contamination. Empirically, ReSeek-trained agents reportedly surpass state-of-the-art baselines in task success and reasoning faithfulness across combined benchmarks.

### Strengths
- The JUDGE action supports dynamic path adaptation by assessing the utility of retrieved evidence and filtering contexts with an indicator $I(j_t \neq \text{'bad'})$, improving recovery on multi-hop queries and promoting efficient trajectories. Its integration with structured prompting yields consistent judgment steps (even in weaker models), enhancing reproducibility. Training aligns judgments with “ideal” labels derived from rerank scores through policy optimization, instilling a form of meta-cognitive control without heavy backtracking.
- The reward decomposes into correctness and utility, providing step-level signals that directly address the sparse-reward failure modes common in prior RL agents and better guide intermediate retrieval/decision steps.
- Using synthetic, fictional entities reduces contamination and encourages procedural reasoning over memorization. Coupling FictionalHot with established datasets (e.g., HotpotQA, NQ) yields a broader, standardized evaluation across single- and multi-hop settings.

### Weaknesses
- Exact Match is named as the primary metric, yet comprehensive result tables and statistical tests are missing. Figures 4–5 show trends, but numerical tables, variance, or confidence intervals for ablations are not provided; Table 4’s reranker hierarchy lacks error bars.
- Although training uses merged NQ and HotpotQA splits, there is no clear evidence of hyperparameter tuning or convergence behavior (e.g., loss/return curves).
- Baselines span vanilla prompting and RL-tuned policies, but there is no visible ablation isolating the reward components or the contribution of the JUDGE action.
- The self-correction assembly in Eq. (2) hinges on $I(j_t \neq \text{'bad'})$, where binary judgments stem from continuous rerank scores using a threshold $\tau=0.7$. The rationale for this choice and sensitivity analyses are not presented.
- Notation/definition issues. Eq. (1) contains a formatting error. In Eq. (2), the history term $\tau_{t-1}$ is referenced but not explicitly defined earlier.

### Questions
- Please provide a detailed account of hyperparameter tuning (e.g., the weighting $\beta$ in Eq. (1)), training curves (returns/loss), and compute/resource usage.
- What is the isolated effect of (i) the JUDGE action, (ii) the correctness vs. utility reward components, and (iii) the self-correction assembly? The paper claims gains in success and faithfulness, but these ablations are not evident.
- Why was $\tau=0.7$ chosen to map rerank scores to binary judgments $j^*$? Please report sensitivity to $\tau \in {0.5, 0.6, 0.8, 0.9}$ (or a continuous calibration curve).
- How is the fidelity of reasoning structures preserved during paraphrasing? Which metrics (e.g., semantic equivalence, reasoning graph preservation) and annotation protocols validate that the paraphrases keep the same reasoning demands?
- The abstract states a decomposition into “correctness” and “utility,” yet Sec. 3.3 seems to present a binary $R_{\text{judge}}$ based on a rerank threshold (Eq. (3)). Please write the complete training reward, and include ablations that zero out each term to quantify its contribution.
- Since ReSeek relies on multi-turn interaction plus an auxiliary JUDGE step, please report latency per question, tokens per question, and training/inference FLOPs/compute. A turn-budget vs. latency trade-off plot would clarify whether improvements reflect gains rather than redundancy.

### Soundness
2

### Presentation
2

### Contribution
2
