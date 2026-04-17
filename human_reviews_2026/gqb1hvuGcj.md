# TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4

## Abstract
Temporal search aims to identify a minimal set of relevant frames from tens of thousands based on a given query, serving as a foundation for accurate long-form video understanding. Many existing works attempt to progressively narrow the search space. However, these approaches typically rely on a hand-crafted search process, lacking end-to-end optimization for learning optimal search strategies. In this paper, we propose **TimeSearch-R**, which reformulates temporal search as interleaved text–video thinking, seamlessly integrating searching video clips into the reasoning process through reinforcement learning (RL). However, applying RL training methods, such as Group Relative Policy Optimization (GRPO), to video reasoning can result in unsupervised intermediate search decisions. This leads to insufficient exploration of the video content and inconsistent logical reasoning. To address these issues, we introduce GRPO with Completeness Self-Verification (GRPO-CSV), which gathers searched video frames from the interleaved reasoning process and utilizes the same policy model to verify the adequacy of searched frames, thereby improving the completeness of video reasoning.
Additionally, we construct datasets specifically designed for the SFT cold-start and RL training of GRPO-CSV, filtering out samples with weak temporal dependencies to enhance task difficulty and improve temporal search capabilities. Extensive experiments demonstrate that TimeSearch-R achieves substantial improvements on temporal search benchmarks such as Haystack-LVBench and Haystack-Ego4D, long-form video understanding benchmarks like VideoMME, MLVU, and LongVideoBench, as well as video reasoning benchmarks such as Video-Holmes, consistently and significantly outperforming other existing temporal search approaches and text-only reasoning models. Our code is available at *https://github.com/Time-Search/TimeSearch-R*.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenge of temporal search in long-form video understanding—how to efficiently identify a small subset of relevant frames among tens of thousands to answer video-based questions. Existing works depend on hand-crafted multi-step pipelines, whereas **TimeSearch-R** proposes an end-to-end reinforcement learning (RL) formulation that interleaves text-video reasoning and frame retrieval. The key technical contribution is GRPO-CSV (Group-Relative Policy Optimization with Completeness Self-Verification), which supplements the standard RL outcome reward with an additional self-verification phase: the model must re-answer the question using only the frames it has searched, thereby improving temporal completeness and reasoning consistency. The paper further introduces a two-stage data filtering pipeline to remove linguistically solvable or unsolvable samples. Empirically, TimeSearch-R achieves new state-of-the-art (SOTA) results on Haystack-LVBench, Ego4D, and long-video reasoning benchmarks, including VideoMME, MLVU, and LongVideoBench, outperforming both open-source and closed-source models. Ablations show CSV prevents RL collapse and improves search completeness.

### Strengths
- **Clear Motivation.** Addresses the key limitation of static frame sampling with a well-motivated reformulation of temporal search as text-video interleaved reasoning. Well-written and logically structured, making ideas easy to follow.
- **Algorithmic Novelty.** Proposes GRPO-CSV, a simple yet effective reinforcement-learning mechanism that improves search completeness and reasoning consistency.
- **Remarkable Empirical Results.** Achieves new SOTA on multiple long-video benchmarks with thorough ablations and comparisons.
- **Insightful Analysis.** Provides clear case studies and ablations demonstrating stability gains and human-like search behavior.

### Weaknesses
- **Incremental novelty.** While the integration of self-verification into GRPO is interesting, the idea remains an incremental extension of existing RL-based reasoning frameworks, rather than introducing a fundamentally new optimization principle.
- **Limited theoretical justification.** The completeness reward (Eq. 4) is heuristic and may become unstable if the base answer is wrong; no theoretical analysis of convergence or variance reduction is provided.
- **Heuristic parameterization.** Several hyperparameters (e.g., search budget, reward weights, KL penalty) are empirically tuned without sensitivity analysis, leaving uncertainty about robustness across tasks and video lengths.
- **Marginal quantitative gains.** Although the model achieves clear improvements, the gains are moderate on some benchmarks and may not fully justify the added complexity.
- **Scalability and Cost Concerns** Training requires 32 × A100 GPUs with multiple rollouts per prompt. The scalability to larger backbones and the claimed efficiency remain unverified due to missing runtime or complexity analysis.
- **Missing qualitative evaluation.** While one qualitative case is shown, the paper lacks broader visualizations of temporal search evolution or failure cases to illustrate where TimeSearch-R surpasses prior methods.

### Questions
- Could the authors clarify how Completeness Self-Verification (CSV) is triggered during training — is it applied to every rollout or only when the original answer passes a correctness threshold?
- How are the reward weights (completeness / format / accuracy) combined or tuned? Would alternative weighting (e.g., annealing completeness) affect training stability?
- Could the authors quantify the number of samples filtered at each stage and analyze its impact on performance and generalization?
- Could the authors report variance or confidence intervals over multiple runs to demonstrate result robustness?
- Could the authors provide statistics on average search turns and retrieved frames compared to baseline methods? 
- How does the inference time (or throughput in videos/sec) compare to baselines? 
- What are the training GPU hours or per-iteration cost introduced by the CSV phase?
- Could the authors provide examples or analysis of typical CSV failure cases (e.g., incomplete retrieval or reasoning drift)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents TimeSearch-R, a novel framework that reformulates temporal search in long-form video understanding as an interleaved text-video thinking process, optimized end-to-end with reinforcement learning. The key innovation is the Completeness Self-Verification (CSV) mechanism, which addresses critical failure modes (insufficient exploration, inconsistent reasoning) in standard outcome-based RL by forcing the model to re-answer questions using only its dynamically retrieved frames. The authors also construct a high-quality dataset via a two-stage filtering pipeline to eliminate samples solvable by linguistic bias or unsolvable even with ideal search. Extensive experiments show that TimeSearch-R achieves new state-of-the-art results on temporal search benchmarks (e.g., Haystack-LVBench) and long-video QA benchmarks (e.g., LongVideoBench), while also exhibiting emergent, human-like search strategies.

### Strengths
1. Clear motivation and problem formulation. The paper correctly identifies the bottleneck of current Video-LLMs and reformulates temporal search as an interactive reasoning process.

2. The GRPO-CSV algorithm introduces self-verification to provide intermediate supervision within reinforcement learning.

3. The proposed model shows consistent gains across several long-video understanding benchmarks and produces explicit reasoning–search traces, enhancing interpretability over prior approaches.

### Weaknesses
The main concern lies in model coupling and generalization.
It remains unclear whether the learned TimeSearch-R policy is specific to the base VLM (Qwen2.5-VL-7B) or transferable to other model sizes and architectures.
Since the reinforcement training is done with a fixed backbone, it is possible that the learned policy overfits to the internal embedding or feature distribution of that model.
This would limit the practical usefulness of TimeSearch-R as a general temporal search module for other VLMs (e.g., 3B, or 72B versions).
The paper would be much stronger if it could demonstrate robustness across different backbones or scales.

### Questions
1. Can you verify whether TimeSearch-R trained with Qwen2.5-VL-7B can work directly with other model variants (e.g., Qwen2.5-VL-3B, 34B, or 72B) without retraining?
If not, how much fine-tuning or adaptation is required?

2. Have you tested whether the learned search policy can generalize to other Video-LLMs beyond Qwen2.5-VL, such as LLaVA-Video or Gemini-style multimodal backbones?

### Soundness
3

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
4

### Summary
The goal is to train models to be able to reason about long-form video, which means an iterative process of search/retrieval paired with state-tracking for eventual answer generation. The authors provide a formulation for training first via SFT and also an updated RL formulation.

### Strengths
Long-form video understanding and reasoning is an important task on which models perform poorly.  This approach aims to move the needle and leverages a nice balance of search and "reasoning" (as CoT is usually done) to integrate the video content where appropriate to validate a hypothesis (the question). The approach appears to work well and is intuitive.

### Weaknesses
1. See questions below
2. I fear that I'm missing some intuition about where to go next after this paper. I'm trying to use the examples in the appendix (Fig 16/17) to reason about where/why things fail and if there's anything that can be done about them.  The author's guidance on the importance of specific parameters (e.g. 8) would be appreciated but also where the approach is not appropriate.  For example, in the insufficient search, is the data well characterized such that we know how well an oracle would do and how often the number of data points is out of scope for the budget, the model, etc? Do some tasks reduce to exhaustive search? Are there other aspects of visual reasoning/continuity/etc that are missing from models such that even with the perfect set of frames the model cannot succeed?  I would like to be able to factor the various components/concerns and understand their interplay.

minor
- L128, prue

### Questions
- Can you enumerate some degenerate solutions the training is likely to find? And what mitigation was required (or if none, why didn't we end up with reward hacking)? My first thought was something about exhaustive search, wide temporal windows, etc
- I don't understand the role of 0.5 in (4), 
- Fig 3 makes it seem like there are a set of answers one per frame in the retrieval? But many intermediate clips would not align with the actual final answer, correct? 
- Can a distribution of the range of frames (8.8 being average) be provided? More generally, insight into which frames are selected, why, and/or what biases the count?
- L323 says there's a budget of 8, but table says 8.8 average? Also, how was this value determined? 
- Can Table 2 be easily run with smaller budgets for a more direct comparison to prior work?

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
4

### Summary
The goal of this paper is to propose an RL framework to solve the temporal search problem in video-language data in a general way without human annotations.
* Solid contribution with practical value: The paper presents a well-executed approach that achieves consistent improvements across downstream Video QA tasks on appropriate benchmarks (VideoMME, MLVU, LVB), demonstrating that the method works broadly. The framework successfully addresses the temporal search problem without requiring human annotations, and the planned release of code, models, and data will facilitate future research in this actively growing area.
* Well-grounded work with clear limitations: While the performance gains are moderate rather than substantial, and the novelty is somewhat incremental (applying GRPO to frame search), the paper is well-motivated, clearly written, technically sound, and appropriately positioned in the literature. The work answers a relevant question ("Does RL-based frame search work without human annotations?") affirmatively with proper empirical support.

### Strengths
* The paper is well written and is easy to understand.
* The paper proposes a novel algorithm for frame search that does not rely on human annotations and leverages “thinking” with video frames and text.
* The authors have a fairly good set of eval benchmarks: VideoMME MLVU and LVB.  To better understand when the authors’ work should be used, it would be interesting to see the cases where the base model performs better, or any potential failure modes of the proposed method.
* The code, models, and data being released will help facilitate research in this area, which is actively growing area of research in long video understanding.

### Weaknesses
* Performance is not substantially better than existing methods, often only moderate performance gains at best.  However, there are consistent gains on the downstream Video QA task, which indicates the approach can work broadly 
* This approach requires finetuning the base VLM via RL, whereas other approaches (e.g., T*) do not require any finetuning and can thus leverage API endpoints.  
* The novelty of the method may be somewhat limited in the sense that it applies a known technique, GRPO, to the setting of frame search in videos.

### Questions
1. More comprehensive analysis:
Cases where the base model performs better than the proposed method
Clear failure mode analysis
This would demonstrate deeper understanding and make the work more valuable for practitioners deciding when to use this approach
2. Stronger empirical results in at least one of these ways:
More substantial performance gains (not just moderate/consistent)
Demonstrating the method works well on additional benchmarks
Showing the approach scales better or generalizes to new domains
3. Addressing the finetuning limitation:
Either: showing the finetuning requirement is justified by significantly better performance
Or: demonstrating other practical advantages that offset not being able to use API endpoints (e.g., efficiency, cost, control)

### Soundness
3

### Presentation
3

### Contribution
2
