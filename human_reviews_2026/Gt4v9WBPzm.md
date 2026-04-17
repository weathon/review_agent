# HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Agentic Retrieval-Augmented Generation (RAG) is a powerful technique for incorporating external information that Large Language Models (LLMs) lack, enabling better problem solving and question answering. However, suboptimal search behaviors exist widely, such as over-search (retrieving information already known) and under-search (failing to search when necessary), which leads to unnecessary overhead and unreliable outputs. Current training methods, which typically rely on outcome-based rewards in a Reinforcement Learning (RL) framework, lack the fine-grained control needed to address these inefficiencies. To overcome this, we introduce $\textbf{Hi}$erarchical $\textbf{P}$rocess Rewards for Efficient agentic $\textbf{RAG}$ (HiPRAG), a novel training methodology that incorporates a fine-grained, knowledge-grounded process reward into the RL training. Our approach evaluates the necessity of each search decision on-the-fly by decomposing the agent's reasoning trajectory into discrete, parsable steps. We then apply a hierarchical reward function that provides an additional bonus based on the proportion of optimal search and non-search steps, on top of commonly used outcome and format rewards. Experiments on the Qwen2.5 and Llama-3.2 models across seven diverse QA benchmarks show that our method achieves average accuracies of 65.4\% (3B) and 67.2\% (7B), outperforming strong agentic RAG baselines. This is accomplished while dramatically improving search efficiency, reducing the over-search rate from over 27\% in baselines from previous work to just 2.3\% and concurrently lowering the under-search rate. These results demonstrate the efficacy of optimizing the reasoning process itself, not just the final outcome. Further experiments and analysis demonstrate that HiPRAG shows good generalizability across a wide range of RL algorithms, model families, sizes, and types. This work demonstrates the importance and potential of fine-grained control through RL, for improving the efficiency and optimality of reasoning for search agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The main contribution of this work is HiPRAG, a novel Reinforcement Learning training methodology that incorporates a fine-grained, knowledge-aware "Hierarchical Process Reward". This approach is achieved by first decomposing the agent's reasoning trajectory into discrete, parsable <step> blocks to distinguish between search and non-search steps ; it then evaluates the necessity of each search decision "on-the-fly" during training, using external LLM judges to identify "over-search" and "under-search" behaviors ; finally, it applies a hierarchical reward function that provides an additional "gated process bonus" based on the proportion of optimal steps, which is added only when both the final answer and the format are correct. Experiments show HiPRAG achieves 67.2% average accuracy (7B), outperforming strong baselines , while dramatically improving efficiency by reducing the over-search rate from over 27% to just 2.3%.

### Strengths
1.The quality of the experimental methodology is high. The paper's conclusions are based on extensive experiments across seven QA benchmarks, multiple model families (Qwen, Llama), and multiple RL algorithms (PPO, GRPO). The ablation studies in Section 5.3 are exhaustive and provide strong support for HiPRAG's key design choices.

2.A key finding is that the HiPRAG model trained based on Qwen2.5-3B-Instruct + GRPO not only surpasses strong external 7B baselines like R1-Searcher++, but it also outperforms the 7B counterpart trained with the authors' baseline reward . This indicates that the authors' training methodology is a more effective path to performance gains than simply scaling the model size with conventional rewards, which has significant implications for resource-constrained research and deployment.

3.The paper's presentation is very clear and well-organized, making a complex methodology easy to understand. The figures in the paper are of high quality and greatly enhance understanding.

### Weaknesses
The Over-search Detection method  requires the policy model to perform a second, complete generation. This "re-generation step" effectively doubles the inference cost for each search step during the RL rollout phase. Although the authors mention that "the re-generation step can be executed separately through batch generation", this still represents significant training overhead. A discussion on the impact of this training method on the total training time would be valuable.

### Questions
1.Why did the authors not provide an ablation study for λf ? How was the value of 0.2 determined? How sensitive are the model's final performance and convergence stability to the choice of λf ?

2.The knowledge source used for the experiments is the 2018 Wikipedia dump , while the judge model for 'under-search' detection is gpt-5-mini, which possesses 2025-era knowledge. If the policy model provides a non-search answer that was factually correct based on the 2018 knowledge base,but is outdated according to the 2025 judge's knowledge, will the judge incorrectly flag this step as an 'under-search'? Could this "temporal misalignment" between the retrieval knowledge source and the judge's internal knowledge base introduce significant noisy rewards, thereby contaminating the training process?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the over-searching or under-searching phenomena in Retrieval-Augmented Generation, which are critical for both effectiveness and efficiency. The authors propose HiPRAG, a hierarchical reward function that supervises search or non-search decisions at each step through reinforcement learning. Specifically, this reward function comprises two key components: an over-search detection module and an under-search detection module, both implemented using two external LLMs. Extensive experiments across 7 QA datasets demonstrate the effectiveness of HiPRAG.

### Strengths
1. The proposed HiPRAG is well-motivated, effectively addressing the over-searching or under-searching issues in RAG. This reward function enhances the original outcome reward by incorporating the over-search and the under-search detection modules.


2. This paper provides comprehensive ablation studies examining various aspects of the proposed method, including the impact of process bonus coefficients and different RL algorithm, which offer valuable insights into HiPRAG's mechanisms.

### Weaknesses
1. The experimental evaluation lacks comparisons with the efficiency-aware RAG methods. While this paper is not the first work to consider the over-searching issue, it fails to discuss or compare with relevant prior works. Consequently, all baselines in the experiments are limited to standard R1-like RAG methods.


2. Although the paper aimes to improve efficiency in RAG, it lacks crucial efficiency metrics such as the average number of search steps or inference time per question.


3. Both over-search and under-search detection modules rely on the LLM-as-judge method, which may not be reliable, particularly when using the smaller LLMs. Therefore, GPT-4.1-mini and GPT-5-mini are used for the reward quality. It is worth further discussion on the influence of the different reward LLMs.

### Questions
1. Could you provide comparisons with existing efficiency-aware RAG methods (e.g., adaptive retrieval, mechanisms)? How does HiPRAG perform relative to these approaches in terms of both effectiveness and efficiency?

2. How sensitive is HiPRAG to the choice of reward LLMs?

3. What is the performance degradation when using smaller/weaker LLMs for reward generation?

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
This paper proposes HiPRAG, a RL framework for agentic RAG that introduces hierarchical process rewards to guide both accuracy and search efficiency. Instead of relying solely on outcome-based rewards, HiPRAG decomposes the trajectory into discrete steps and introduces step-level rewards to penalize over-searching and under-searching. The framework employs external LLMs to detect these inefficiencies on-the-fly and incorporates the results into a hierarchical reward formulation.

### Strengths
1. The hierarchical reward structure provides a novel and interpretable mechanism for optimizing process-level efficiency rather than only final-answer correctness.
2. Unlike previous works on faithful process supervision, this paper specifically targets over-searching and under-searching issues in multi-hop retrieval, and presents a well-motivated solution.
3. The proposed method achieves consistent improvements across different model sizes (3B, 7B) and RL algorithms (PPO, GRPO), with significantly reduced over- and under-search rates.

### Weaknesses
1. The over-/under-search detection relies on commercial LLMs (GPT-4.1-mini, GPT-5-mini) as external judges, which raises concerns about stability, cost, and reproducibility. The impact of judgment errors is not analyzed.
2. The on-the-fly verification by external LLMs adds training complexity, yet the paper provides no analysis of computational cost, runtime, or GPU-hour consumption.
3. The role of $/lambda_p$ is unclear. Since the process reward is already gated by answer and format correctness (i.e., a hierarchical design), it should not directly influence final answer quality. Why does increasing $/lambda_p$ to 0.6 “over-prioritize step purity at the expense of answer correctness”? This suggests that λₚ may not be a meaningful parameter. Please clarify its actual effect on training dynamics.

### Questions
1. How is the “Training with Over-search or Under-search Only” setting implemented? Are the ignored types treated as optimal steps in the reward computation?
2. The ablation study section enumerates many results without clear structure; summarizing the main findings would improve readability and help highlight key insights.

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
3

### Summary
In this paper, a new training method, named Hierarchical Process Rewards for Efficient agentic RAG (HiPRAG), is proposed to address common inefficiencies in agentic Retrieval-Augmented Generation (RAG) systems. The authors identify two key problems: over-search (needlessly retrieving known information) and under-search (failing to retrieve necessary information, leading to errors). Current Reinforcement Learning (RL) methods rely on coarse, outcome-based rewards. The proposed method tackles this by incorporating a fine-grained, hierarchical process reward into the RL loop and achieves good results.

### Strengths
1. The paper is, in general, well written with good structure and is easy to follow.

2. The hierarchical process reward formulation well balances correctness with efficiency through a gated bonus structure, which avoids the classic RL issues of over-penalizing necessary search.

3. In this paper, the authors conducted various design choices with a lot of experiments with multiple models (Qwen, Llama), sizes (3B, 7B), types (base, instruct), and RL algorithms (PPO, GRPO). And the proposed method achieves good results compared to other baselines across several datasets.

4. The proposed method can significantly reduce the over-search rate, which is very useful and impressive.

### Weaknesses
1. The proposed reward mechanism relies on "on-the-fly" calls to external LLM judges to detect over-search and under-search during the RL training loop, which could potentially add significant computational overhead, cost, and API call latency to every training step. 

2. A further question to this reliance is that the LLM models may behave differently over time. Can we still keep the consistency of the proposed method?

3. The method assumes the judge is accurate. If the judge mislabels a step (false positive over-search or false negative under-search), the agent may be clipped incorrectly.

### Questions
Please check the "Weaknesses" and also provide comments on the following questions:

1. The paper defines under-search as a factual or logical error in a non-search step. This may be a useful proxy for hallucination, but I wonder how the proposed method behaves with the other under-search situations, such as suboptimal or incomplete knowledge. If I understand it correctly, the current detector, which only checks for correctness, would not flag this as an under-search and would incorrectly reward it as an "optimal" non-search step.

2. Can you provide more details on when the reward collapse occurs?

### Soundness
3

### Presentation
3

### Contribution
3
