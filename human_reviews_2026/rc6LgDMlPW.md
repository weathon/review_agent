# TreePO: Enhancing Policy Efficacy and Inference Efficiency with Tree Modeling

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 8, 4

## Abstract
Recent advancements in aligning large language models via reinforcement learning have achieved remarkable gains in solving complex reasoning problems, but at the cost of expensive on-policy rollouts and limited exploration of diverse reasoning paths. 
In this work, we introduce TreePO, involving a self-guided rollout algorithm that views sequence generation as a tree-structured searching process. 
Composed of dynamic tree sampling policy and fixed-length segment decoding, TreePO leverages local uncertainty to warrant additional branches. 
By amortizing computation across common prefixes and pruning low-value paths early, TreePO essentially reduces the per-update compute burden while preserving or enhancing exploration diversity. 
Key contributions include: (1) a segment-wise sampling algorithm that alleviates the KV cache burden through contiguous segments and spawns new branches along with an early-stop mechanism; (2) a tree-based segment-level advantage estimation that considers both global and local proximal policy optimization.
and (3) analysis on the effectiveness of probability and quality-driven dynamic divergence and fallback strategy. 
We empirically validate the performance gain of \modelname on a set reasoning benchmarks and the efficiency saving of GPU hours from 22% up to 43% of the sampling design for the trained models, meanwhile showing up to 40% reduction at trajectory-level and 35% at token-level sampling compute for the existing models. 
While offering a free lunch of inference efficiency, TreePO reveals a practical path toward scaling RL-based post-training with fewer samples and less compute.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TreePO, a novel rollout algorithm designed for reinforcement learning (RL) with large language models (LLMs) based on a tree structure. Starting from a root node, TreePO creates multiple branches and generates segments for each branch. Branches with flawed patterns or EOS tokens are discarded. The advantage estimation is refined by inspecting the tree structure, allowing for more granular evaluation. The training efficiency is improved due to prefix sharing across rollouts. Experiments show that TreePO outperforms naive parallel rollouts.

### Strengths
The tree-structured rollout design is both novel and effective. Given that rollouts are a major bottleneck in RL training, the tree structure reduces decoding overhead through KV cache sharing in the tree prefix. Additionally, it enhances advantage estimation, mitigating the sparse reward issues in RLVR.

### Weaknesses
- One major concern is the misleading performance of the GRPO baseline. In Figure 1, GRPO’s performance drops significantly after step 200, which has not been widely reported in literature. The proposed methods show similar performance to GRPO until step 200, and the performance gain may be attributed to GRPO’s drop. In Table 1, GRPO achieves only 72.89% for maj@16 on the MATH dataset. However, similar literature [1] reports that Qwen2.5-7B-Instruct achieves 75.68% for avg@32 on MATH even without training. While there are dataset, setting and method differences, Qwen2.5-7B’s avg@32 performance should be expected in the range of 75-78% with GRPO, and maj@16 should be even higher. It’s recommended that the authors also report avg@16 or avg@32 to clarify this issue.

- Another concern is related to efficiency. TreePO appears to have lower parallelism than naive rollouts. For example, the naive rollout baseline can generate 16 responses for each prompt, with 64 prompts, resulting in an overall parallelism of 64 × 16 = 1024. However, TreePO starts with a parallelism of only 64, which gradually increases to 128, 256, and eventually 1024. It is well-known that parallelism affects efficiency. Does TreePO actually reduce parallelism compared to naive rollouts for the same tree width w? Although the authors provide efficiency experiments in Section 4.1, the settings differ from those in the training. For training, it is mentioned “the maximum tree width is set to w = 16”, which is different from the “64 rollouts per prompt” in 4.1.

[1] PVPO: Pre-Estimated Value-Based Policy Optimization for Agentic Reasoning

### Questions
- The same as the second concern in weaknesses. Does TreePO reduce parallism compared to naive rollouts given the same width w?

- What is meant by “flawed sub-string patterns” in line 167? Is this the same as "repetitive substrings" mentioned later? If so, using the same term would improve clarity.

- Minor issues: (1) In line 199, the reference is broken:”In the later §??,”.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces TreePO, a novel reinforcement learning framework for training Large Language Models (LLMs) that reformulates the standard on-policy rollout process as a segment-based tree search. The core idea is to leverage the inherent structure of reasoning tasks, where multiple trajectories for a single query share common prefixes. By explicitly modeling these shared prefixes and branching at points of uncertainty, TreePO achieves significant computational efficiency gains through KV-cache reuse and early termination of unpromising paths. Furthermore, the authors propose a tree-based advantage estimation function that provides more granular credit assignment by comparing trajectories within hierarchically nested subgroups. The method is empirically validated extensively on a range of mathematical reasoning benchmarks, demonstrating a reduction in GPU hours of up to 43% without sacrificing, and sometimes even improving, final performance compared to conventional sampling methods.

### Strengths
- The authors present an elegant algorithm-system co-design that tackles a major bottleneck in RL for LLMs: the cost of on-policy data generation. The key insight of structuring rollouts as a tree of segments is well-motivated and justified, as it neatly mirrors the natural tree-like structure of reasoning processes where paths share common prefixes before diverging.

- The framework is holistic, seamlessly integrating the tree-based sampling mechanism with a corresponding tree-based advantage estimation technique. This synergy is crucial; it allows the method to not only perform heuristic search during sampling (e.g., dynamic branching, early termination) but also to leverage the resulting tree structure to derive more precise credit assignment, thereby further enhancing the training benefits.

- The paper provides thorough experimentation, demonstrating clear improvements in training stability and computational efficiency. The reported savings of 22% to 43% in GPU hours are substantial and of high practical utility, providing a valuable reference for the community seeking to scale up RL-based training.

### Weaknesses
- The most significant weakness is that all experiments are confined to mathematical reasoning tasks. While the results here are impressive, it remains an open question how well TreePO generalizes to other critical domains like code generation. The core premise of TreePO—significant shared prefixes across trajectories—may not hold equally well in other domains.

- While the empirical results are strong, a theoretical analysis or intuition on how the tree-based sampling affects the exploration policy and the convergence properties compared to standard i.i.d. sampling would strengthen the contribution. Such analysis would help clarify the trade-offs introduced by the biased exploration of the tree-structured policy and provide a deeper understanding of the method's properties.

### Questions
- The impressive reduction in GPU hours is a key result. Could the authors provide a more detailed breakdown? What fraction of the savings comes primarily from KV-cache sharing (a system-level optimization) versus the algorithmic improvements of early termination and more efficient exploration?

- Given that the method is on-policy, have the authors considered or attempted to combine TreePO with off-policy acceleration techniques (e.g., partial rollouts)? If not, what would be the primary technical challenges in such an integration?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces TreePO, a reinforcement learning framework that restructures on-policy rollout generation for LLM reasoning tasks from independent sequential sampling to segment-based tree search. The key insight is that stochastic sampling of reasoning trajectories produces substantial prefix overlap, which can be exploited through shared KV-cache computation. TreePO combines:
(1) a segment-wise tree sampling algorithm that generates fixed-length segments, branches dynamically, and implements early stopping and fallback mechanisms;
(2) a hierarchical advantage estimation function that forms sub-groups based on shared prefixes at different tree depths; and (3) analysis of depth-segment trade-offs and probability-based branching heuristics.
The method is evaluated on mathematical reasoning benchmarks (AIME, AMC, MATH, MINERVA, Olympiad Bench) starting from Qwen2.5-7B base model. Results demonstrate 22-43% GPU hour reduction during training while achieving comparable or improved performance (+11.58% overall accuracy over GRPO baseline). The approach notably enables "RL-zero" training directly from base models without requiring prior supervised fine-tuning.

### Strengths
Originality:
1. Novel hierarchical advantage estimation that groups trajectories by shared prefixes, enabling local comparison within sub-groups rather than global comparison
2. Good work to systematically analyze the efficiency-performance trade-off of tree-based sampling specifically for RL training (prior work focused on inference or used tree structure without efficiency analysis)
3. Creative exploitation of the observation that stochastic sampling produces overlapping prefixes

Quality:
1. Comprehensive experimental validation across 5 benchmarks with consistent improvements
2. Thorough ablations validate each design choice (averaging vs. weighting, sub-group rejection, root group removal, token alignment)
3. Honest reporting of negative results (probability-based branching doesn't help, §D.2)
4. Both theoretical motivation (credit assignment) and practical engineering (KV cache efficiency)

Clarity:
1. Clear problem motivation (Figure 2 makes the prefix overlap concrete)
2. Progressive methodology presentation building from simple observation to full system
3. Extensive appendices provide reproducibility details
4. Good use of visual aids (Figures 3-9 effectively communicate results)

Significance:
1. Addresses real bottleneck in RL post-training at scale
2. 22-43% efficiency gains have immediate practical value for industry and researchers with limited compute
3. Training from base models (RL-zero) is important for accessibility
4. Findings about depth-segment trade-offs and local vs. global advantages provide insights for future work

### Weaknesses
1. The paper claims this is a general approach for "complex reasoning," but provides no evidence that:
- Prefix overlap occurs in other domains (code generation, dialogue, scientific reasoning)
- The advantage function generalizes beyond math
- Efficiency gains transfer to tasks with less structured reasoning

Might include at least one non-math benchmark, or explicitly frame contributions as specific to mathematical reasoning.

2. Evaluation-training mismatch: Training uses tree sampling, but evaluation (Table 1) uses sequential sampling with Maj@16. Table 2 shows tree sampling actually has slightly lower performance than sequential at equivalent compute. This raises questions:
- Should tree-trained models be evaluated with tree sampling?
- Is the performance gain partly an artifact of evaluation methodology?
- How much of the benefit comes from training vs. inference efficiency?

Might report both tree and sequential evaluation for all models, or justify why sequential evaluation is appropriate for tree-trained models

3. Memory efficiency concerns. Figure 7c shows performance decreases beyond 16 rollouts for Qwen2.5-Math-7B due to KV cache fragmentation and memory pressure. This:
- Limits scalability claims
- Suggests tree sampling has overhead that dominates at scale
- Contradicts narrative that tree structure is always more efficient

Might discuss memory limitations more prominently and characterize when tree sampling becomes counterproductive.

### Questions
1. Direct comparison with TreeRL/SPO: Can you provide a controlled comparison where only the advantage function differs? This would clarify the specific contribution of your hierarchical sub-group approach vs. their parent-child value difference.
2. Base model vs. SFT model: You claim TreePO works from base models while TreeRL/SPO require SFT. Have you tested whether TreeRL/SPO also work from base models? What specifically enables RL-zero capability?
3. Evaluation methodology: Why evaluate with sequential sampling when training uses tree sampling? Have you compared tree-evaluated vs. sequentially-evaluated performance systematically?
4. Advantage function design: You show simple averaging > size weighting (§4.2). Did you explore other aggregation schemes? (e.g., weighted by sub-group diversity, inverse variance weighting, learned aggregation)
5. Fallback strategy details: Algorithm 1 mentions fallback but implementation details are sparse. How often does fallback trigger? How is the fallback segment chosen? Does this introduce additional overhead?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper TreePO proposes a tree-structured rollout and advantage estimation framework for reinforcement learning–based alignment of large language models. Instead of generating independent trajectories during policy optimization, TreePO treats generation as a tree search composed of fixed-length segments. Shared prefixes across trajectories are reused via KV cache sharing, improving inference efficiency. The method also introduces a tree-based advantage estimator that computes credit assignment using subgroup structures derived from shared prefixes. Experiments show improved GPU sampling efficiency (22–43% reduction in compute), more stable RL training curves, and
comparable or improved accuracy on math reasoning benchmarks compared to GRPO and sequential sampling baselines.

### Strengths
- Training LLMs with RL for reasoning is computationally expensive. The paper targets a meaningful problem: both exploration diversity and compute efficiency, which are considered two central limitations of current RLHF-style pipelines.

- The observation that reasoning rollouts share substantial prefixes is empirically grounded and persuasive. Leveraging this to build a tree sampling structure is a natural but impactful idea.

- Technical Novelty in sampling design, segment-wise decoding, KV cache reuse across branches,  and dynamic branching + fallback. These combine to create a coherent sampling system that improves efficiency without architectural changes.

- Integrating subgroup baselines to refine credit assignment is conceptually sound and improves training stability.

- Evaluations span multiple math reasoning datasets and show gains in both stability and compute cost, with sensible ablation studies.

### Weaknesses
The method description is dense, with multiple interacting components (segmenting, branching, fallback, subgroup estimation). The paper would benefit from clearer step-by-step examples or visual walkthroughs.

Recent works such as TreeRL, SPO, and ETS are acknowledged, but direct head-to-head comparisons are missing. 

KV reuse and batching benefits heavily depend on inference engine behavior (e.g., vLLM version, GPU memory). A more standardized hardware comparison is needed.

While math reasoning is a valid test domain, it remains unclear whether:
- Tree-based branching remains stable for longer, more diverse tasks,
- Advantage estimation scales to multi-turn dialogue or tool-use tasks.

Advantages are motivated heuristically; convergence or bias analysis is not deeply developed.

### Questions
The paper acknowledges prior tree- or segment-based RL methods. Would you please add comparison with these schemes? 

The performance and efficiency of TreePO depends on the selection of: Segment length, Branching factor, and Maximum depth. Could the authors provide guidance or rules of thumb for selecting these values across tasks? Is there evidence that the same hyperparameters generalize to non-math domains?

The experiments focus on math reasoning tasks (GSM8k, MATH, etc.). How does TreePO perform in: Multi-turn dialogue alignment, Code generation tasks, tool-augmented reasoning? Even small preliminary results or qualitative case studies would help establish generality.

The method relies on the ability to reuse KV caches across branches. Could the authors clarify how TreePO behaves with Streaming inference engines, GPU memory fragmentation,Or engines that do not preserve memory layout (e.g., older vLLM versions)?

### Soundness
3

### Presentation
2

### Contribution
2
