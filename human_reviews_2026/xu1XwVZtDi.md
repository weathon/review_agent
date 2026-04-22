# Kevin: Multi-Turn RL for Generating CUDA Kernels

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 8

## Abstract
Writing GPU kernels is a challenging task and critical for AI systems' efficiency. It is also highly iterative: domain experts write code and improve performance through execution feedback. Moreover, it presents verifiable rewards like correctness and speedup, making it a natural environment to apply Reinforcement Learning (RL). To explicitly incorporate the iterative nature of this process into training, we develop a flexible multi-turn RL recipe that addresses unique challenges encountered in real-world settings, such as learning from long trajectories and effective reward attribution across turns. We present Kevin - K(ernel D)evin, the first model trained with multi-turn RL for CUDA kernel generation and optimization. In our evaluation setup, Kevin shows significant gains over its base model (QwQ-32B), improving correctness of generated kernels (in pure CUDA) from 56% to 82% and mean speedup from 0.53x to 1.10x of baseline (PyTorch Eager), and surpassing frontier models like o4-mini (0.78x). Finally, we study its behavior across test-time scaling axes: we found scaling serial refinement more beneficial than parallel sampling. In particular, when given more refinement turns, Kevin shows a higher rate of improvement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces "Kevin," a model trained to generate and optimize GPU kernels using a multi-turn RL framework. The study think that the iterative nature of code optimization is best captured by a training process where the model receives and learns from execution feedback over multiple rounds. The core is a multi-turn RL recipe designed to overcome challenges inherent in long-horizon tasks. By explicitly modeling the "generate -> execute -> get feedback -> refine" loop, the training process aligns more closely with how human experts work. Evaluation results on the KernelBench dataset demonstrate that this approach is highly effective, showing that Kevin achieves significant improvements in both the correctness and performance of the CUDA kernels it generates compared to its base model and other strong baselines.

### Strengths
1. Well-motivated multi-turn RL framework to solve the Kernel Bench CUDA task. And the approach is validated by strong empirical results. Not only surpasses its base model and a single-turn RL counterpart but also outperforms strong frontier models like OpenAI 04-mini on both correctness and performance metrics. 
2. Clearly demonstrate that sequential refinement is more compute-efficient than parallel sampling. 
3. Offers practical engineering solution, detailing robust solutions to reward hacking probelm.

### Weaknesses
1. The most significant weakness is the reliance on the relatively small KernelBench benchmark, which contains only around 250 tasks. The scarity of training data on this specific domain will raise my concern on the generalizability compared to other large language models. 
2. The paper mentioned that "only accurate for those dimentions and NVDIA H200 GPUs". The reported speedup performance improvement are highly specific to H200 evaluation environment within KernelBench. More analysis on the methods potential performance transfer to different hardware architectures or real-world workloads remains an open question, further validation on other benchmark environments are needed.

### Questions
Please check weakness.

### Soundness
2

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
The paper proposes Kevin, a CUDA kernel generation model trained with multi-turn RL. Instead of producing one kernel in one shot, the model iteratively refines kernels over several turns, using runtime and profiling feedback from previous attempts. The training pipeline assigns discounted reward across turns and treats each refinement step as its own RL sample, with context summarization to keep prompts short. 
On 100 held-out CUDA tasks (KernelBench style), Kevin matches a strong single-turn RL baseline in correctness (82% best@16) but shows somewhat higher peak runtime speedup (1.10× vs 0.85× best@16). 
The paper argues that this proves multi-turn RL improves the ability to iteratively optimize kernel performance and scales better when you allow more refinement turns at test time.

### Strengths
**Realistic problem setup.** The work models how GPU performance engineers actually iterate: propose kernel → profile → refine. The RL formulation explicitly credits early partial attempts that later lead to a fast kernel, instead of rewarding only final outputs. 

**Clear engineering advances.** They introduce (i) per-turn training on every refinement step to improve sample efficiency and (ii) discounted future-return style reward aggregation for credit assignment across turns. 

**Beating strong general models.** Kevin, post-trained from QwQ-32B, outperforms powerful proprietary baselines like o4-mini and o3-mini on this CUDA kernel benchmark, even though those models are otherwise stronger than QwQ-32B on standard coding and reasoning tasks. 

**Test-time scaling insight.** The paper studies how performance scales with more refinement turns and shows the multi-turn–trained model benefits more from extra turns than single-turn RL, suggesting better iterative optimization ability.

### Weaknesses
**Performance gains are marginal vs the main ablation.** Compared to the single-turn RL baseline, Kevin does not improve solve rate: correctness best@16 stays 82% vs 82%, and fast1 best@16 stays 43% vs 43%. The main improvement is higher best-case speedup (1.10× vs 0.85×), and average speedup only rises slightly (0.40× vs 0.35×). 
So the method is not solving more tasks; it’s mostly squeezing somewhat better runtime on tasks that were already solvable.

**Heavy system heuristics, limited theory.** The approach bundles many practical tricks (turn-by-turn training, summarizing CoT to avoid 50k–100k token contexts, discounted reward shaping). 
But these are presented as an engineering recipe rather than a principled analysis of why multi-turn RL should outperform single-turn RL.
Detailed ablation study would help us understanding real contribution of introducing multi-turn.

### Questions
Q1: Multi-turn RL and single-turn RL have the same solve rate (82% best@16) and same fast1 rate (43%), and only a modest speedup gain (1.10× vs 0.85×). In what concrete situations does multi-turn RL succeed where single-turn RL fails (either solving a task it couldn’t solve, or finding a faster kernel it couldn’t reach)? Please quantify.

Q2: Kevin is trained starting from a strong 32B base model. You mention weaker bases tend to hack rewards or collapse.
Can you show any numbers (even partial) for smaller models, or describe concretely how and why training fails there?

Q3: Your method bundles many heuristics (CoT summarization, discounted reward shaping etc). Can you provide ablations that isolate the effect of multi-turn RL itself versus these stabilizing heuristics? Without that, it’s hard to attribute the reported gains to “multi-turn” rather than to engineering tricks.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Kevin is a multi-turn reinforcement learning (RL) based framework to train LLMs to generate better GPU kernels. Authors propose a design for multi-turn RL with a reward function proportional to correctness and generated kernel's performance. Authors have used GRPO for training the model. During training authors sample m parallel trajectories with n iterative refinement steps. Authors also propose to compress CoTs from prior attempts to save on context length available for generation. Authors have demonstrated that single-turn RL saturates early on and does not benefit from the refinement aspects for test-time compute scaling. Authors also propose to training on each turn in the multi-turn realization. Authors aggregate the reward across multiple turns with two specific discount factors 0.4 and 0.8. Multi-turn training behaviour shows reward monotonically increasing on an average. Authors have used 180 tasks out of 200 (from level 1 and 2) tasks from kernelbench benchmark. Remaining 20 and additional author created 80 tasks are used for evaluation. Authors compare speedup across pytorch implementation provided in tasks. The paper also shows that test-time scaling trends outperform the baseline LLM scaling. Further, authors have shown that Kevin benefits sequential axis of test-time scaling than that of parallel.

### Strengths
- Demonstration of multi-turn RL for GPU kernel generation. 
- RL reward formulation as a function of correctness and speedup.
- Effectiveness of multi-turn RL to improve inference time scaling trends.
- Clearly comparing gains against single-turn RL.

### Weaknesses
- The evaluation methodology does not follow standard practices. Kevin trains on 180/200 examples from kernelbench evaluation benchmark. The authors must create their own dataset for training and then evaluate on kernelbench. This does not inspire fair comparison with existing approaches and goes against standard practices. 
- Paper does not specify where are the initial CoTs obtained from.
- In section 4 line 237, the description is unclear. 
- In subsection 4.1: Summarizing all previous CoTs does not appear to be the best way to build context. There are several prior methods that have shown using prior top-k + bottom-k attempts to build context that results in better contrastive learning.
- subsection 4.2 appears to be incomplete.
- Performance is evaluated with Pytorch implementation as a baseline. However, even if the benchmark provides reference Pytorch implementations, these are not truly optimal implementations available today. I encourage authors to compare speedup against torch.compile as their baseline. This will show true improvements over manual kernel optimization which is a very time consuming and tedious task. This will justify all the effort and energy spent in training/inference/generation of LLMs.
- There is no profiler feedback integrated. Neither during training nor during inference. The profiler feedback is very crucial which helps human experts to reason and write best strategies to improve kernel performance. 
-  Authors compare against o3/4-mini models which are not optimized for code. They should instead compare against GPT4.1 and claude-sonnet-4 which are optimized for code.
- With best@16 Kevin shows 1.10x performance improvement. Which does not inspire any confidence in the efficacy of Kevin given that 10% gain could come from measurement errors. 
- There are some case studies in appendix showing speedup of 1.9x, 2+x, 3+x; all these speedups are over naive pytorch calls. torch.compile significantly optimizes a given pytorch code by performing operator fusion on the fly using a pool of manually optimized kernels. This should be the ideal baseline that AI solution must beat.

### Questions
Please see weaknesses section.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors propose an RL training method to generate high-performance CUDA kernels by using multi-turn RL training, known as Kevin. In contrast with previous code-generation studies that focused only on correctness, Kevin starts from a reference PyTorch implementation and lowers it to CUDA, monitoring both correctness and performance. To accomplish this, the authors assign a score to each kernel that balances correctness and performance, and train an LLM using GRPO on samples without immediately iterating on external feedback, as is typical of single-turn training. Instead, during each training step, multiple responses are evaluated and assigned a reward. During multi-turn training, not only is the code used to stir the process, but the chain of thought as well. However, with the aggregation of so many instances, the CoT context becomes quite large, so the authors summarize the context from earlier runs. Two approaches to assigning scores were investigated: a greedy approach that assigns a kernel score to each turn and an outcome-based approach that assigns a score to all turns based on the best score in the trajectory. The authors propose a hybrid approach that balances the pros of these two credit assignment strategies and perform an ablation study to support the efficacy of this choice. The evaluation section illustrates the improved performance of multi-turn training compared to single-turn alternatives and other LLMs to generate code. The code generated by KL yields a higher rate of correctness and achieves improved performance in most tests.

### Strengths
- This is my first time encountering multi-turn in the context of code generation. The results support its usage from both a correctness and performance perspective. 
- The authors investigated multiple aspects of the RL training procedure, from the generation of the training data, reward assignment, and the composition and structure of samples during training. All of these training hyperparameters were shown to have a notable impact on the quality of the result and should provide a useful data point for other researchers in the area, considering multi-turn RL.
- The issue of context length control seemed to be an interesting, but maybe ephemeral problem, as the context lengths continue to grow. The solution they proposed did not seem detrimental to kernel improvement.
- The tradeoff inference configurations, illustrated in Table 2, were an interesting display of the performance impact of trajectory vs turns on both the performance and correctness.

### Weaknesses
- I found the discussion of the choice of baseline model in Appendix B.6 to be insufficient from the reader's perspective. While it seems completely plausible for the largest model to have the best priors and smaller models more susceptible to reward hacking, it may be the case that certain updates to the reward function or training could alleviate these issues.
- One of the major limitations, noted by the authors, is the limited number of robust tasks usable for training. With access to more tasks, the kernel generation capabilities of Kevin could be significantly higher.
- All the figures are difficult to digest for people who struggle with differentiating colors. A change in the markers and/or line styles would make it easier to tell the differences at a glance.

### Questions
- Is it possible to add a graph to illustrate the issues related to smaller models and reward hacking in Appendix B.6? Not strictly necessary, but it would provide more context for readers who are not as familiar with these issues.

### Soundness
3

### Presentation
3

### Contribution
3
