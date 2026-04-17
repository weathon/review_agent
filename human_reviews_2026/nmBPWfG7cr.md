# Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We introduce Ego-R1, a novel framework for reasoning over ultra-long (i.e., in days and weeks) egocentric videos, which leverages a structured Chain-of-Tool-Thought (CoTT) process, orchestrated by an Ego-R1 Agent trained via reinforcement learning (RL). Inspired by human problem-solving strategies, CoTT decomposes complex reasoning into modular steps, with the RL agent invoking specific tools, one per step, to iteratively and collaboratively answer sub-questions tackling such tasks as temporal retrieval and multi-modal understanding. We design a two-stage training paradigm involving supervised finetuning (SFT) of a pretrained language model using CoTT data and RL to enable our agent to dynamically propose step-by-step tools for long-range reasoning. To facilitate training, we construct a dataset called Ego-R1 Data, which consists of Ego-CoTT-25K for SFT and Ego-QA-4.4K for RL. Furthermore, our Ego-R1 agent is evaluated on a newly curated week-long video QA benchmark, Ego-R1 Bench, which contains human-verified QA pairs from hybrid sources. Extensive results demonstrate that the dynamic, tool-augmented chain-of-thought reasoning by our Ego-R1 Agent can effectively tackle the unique challenges of understanding ultra-long egocentric videos, significantly extending the time coverage from few hours to a week.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Ego-R1, a tool-augmented agent for long egocentric video QA. The key idea is a chain-of-tool-thought (CoTT) controller that dynamically decides when to invoke 1) a hierarchical RAG over text logs; 2) a short-horizon Video-LLM; and 3) a frame-level VLM. The model is trained with a two-stage approach: 1) SFT on CoTT traces, 2) GRPO-style RL. The authors also introduce Ego-R1 Data (Ego-CoTT-25K for SFT and Ego-QA-4.4K for RL) and an evaluation set Ego-R1 Bench (week-long videos). Reported results show strong performance across multiple long-video QA benchmarks, including VideoMME and EgoSchema.

### Strengths
- Sound framework. A simple yet compelling agentic architecture separating long-range retrieval from localized video understanding (Video VLM) and fine-grained frames (frame VLM), with a controller trained to decide which to use when through SFT+RL.

- Good results. On several benchmarks, including VideoMME and EgoSchema, Ego-R1 achieves strong results compared to other methods like VideoAgent and Gemini-1.5-Pro.

- Useful ablations. Ablations show that both stage training (SFT+RL) improves performance.

### Weaknesses
- Limited novelty. The agentic framework in video understanding has been well explored, for example, in VideoAgent. The training techniques used in this paper, SFT followed by RL, are also well-known, as in VideoR1. Given the limited novelty, more insightful analysis, such as the failure cases when applying the method to this setting, the reasons for the failures, and how to adapt them, is worth further study. For instance, how to create the best SFT data? how to design the rewards?

- More ablations needed. The paper proposed using both video VLM and frame VLM as tools; I'm wondering about the performance of removing each of them. How each contributes to the final performance, and how different model choices in these video VLMs and frame VLMs affect the performance.

- Questionable improvements due to contamination. The paper shows most improvements in EgoLifeQA and Ego-CoTT-25K. However, the training data and these benchmarks are closely related in the domain and construction pipeline. This raises concerns of data contamination rather than real generalization.

### Questions
See weaknesses.

Line 127: "Table ??" is a typo

### Soundness
2

### Presentation
3

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
This paper successfully adopts the Deepseek-R1 style reasoning training in the context of egocentric video understanding, with tool integration. To this end, the authors propose a manually annotated and automatically generated dataset. After a cold start and GRPO training, the trained model surpasses the compared baselines in VideoMME and 3 other egocentric video benchmarks.

### Strengths
- This paper extends egocentric video understanding into week-level duration.
- This paper successfully implements reasoning-tool calling CoT in the context of video understanding.
- This paper proposes new datasets for egocentric video understanding.

### Weaknesses
- Long temporal retrieval is conducted in the text form instead of visual language matching. However, the transformation from visual space to textual space inevitably loses information.
- In Tab.1, the performance of the base model is not reported. In addition, although samples that are overlapped with the benchmark are removed in training, the cold-start and RL stages are still focused on ego-centric videos that are in-domain data. Therefore, the comparison with other general video models seems to be meaningless in the egocentric setting.
- The tool set used in this paper seems to be limited but heavy (captioning, VLM, ...). I am wondering about the training and inference costs.

In conclusion, I think this paper is another application of the ReAct paradigm in the context of egocentric understanding; although new datasets are proposed, I think the contribution seems to be limited, and experiments are not solid enough.

### Questions
- In Lines #077-078, I think this paradigm is known as ReAct. What is the need to create a novel term, CoTT, for video understanding?
- In Line #404, is the base model Qwen2.5-VL-3B instead of Qwen2.5-3B?

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
4

### Summary
This paper proposes a hierarchical RAG framework for egocentric video understanding, called Ego-R1 Chain of Tool-Thoughts. The method builds a temporal hierarchy in its database, organizing information by week, day, hour, and clip, and introduces a multi-tool reasoning pipeline combining retrieval, segmentation, and question answering. The authors evaluate the model on an egocentric benchmark where timestamps and hierarchical video segmentation are clearly defined, showing improved retrieval and reasoning performance compared to flat RAG baselines.

### Strengths
-	Clear and systematic hierarchical RAG structure, which improves efficiency and relevance in timestamp-based video reasoning tasks.
-	Experiments on multiple egocentric datasets demonstrate consistent improvement over flat retrieval methods.

### Weaknesses
-	The hierarchical database structure (week → day → hour → clip) appears optimized for benchmarks with clear temporal granularity, but it’s unclear if it remains effective for datasets or tasks where such segmentation is not naturally defined.

-	The uniform video segmentation approach might not be robust across diverse video lengths or event types. The method may fail to capture variable-duration actions or continuous interactions.

-	The technical contribution is moderate, focusing on database structuring and system integration rather than advancing the underlying reasoning or learning algorithms.

### Questions
-	How robust is the proposed hierarchy when applied to benchmarks that do not have clear or fixed temporal units (e.g., datasets without explicit timestamps)?

-	Is the uniform segmentation scheme adaptive to variable-length activities, or could it fragment meaningful events?

-	Could the system generalize to other RAG tasks beyond egocentric video, or is it tightly coupled to timestamp-based segmentation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Ego-R1 is a Chain-of-Tool-Thought (CoTT) agent for week-scale egocentric QA that plans over hierarchical memory (day→hour→10-min captions/ASR) and adaptively invokes a short-window Video-LLM plus a single-frame VLM for fine grounding.
Training is SFT on tool-grounded traces followed by GRPO reinforcement to learn multi-turn policies that trade accuracy vs. tool economy (≈7 calls/question; tens of frames instead of hours).
The release includes Ego-CoTT-25K (synthetic traces), Ego-QA-4.4K (human-verified QA), and a week-long benchmark (≈44.3 h/video) targeting long-horizon temporal reasoning.

### Strengths
- Originality: Frames week-scale egocentric QA as sequential decision-making via a Chain-of-Tool-Thought controller over hierarchical temporal memory with adaptive Video-LLM/VLM calls.
- Quality: Strong margins on a week-long benchmark (46.0%, +7.7 vs Gemini-1.5-Pro), clear ablations (SFT+RL > SFT; CoTT > retrieval-only), and substantial frame-budget reductions.
- Clarity: Explicit tool APIs, training signals, and memory construction; stepwise traces reveal evidence flow and typical failure modes.
- Significance: Provides reusable traces, QA sets, and a week-scale benchmark, establishing a modular, plug-and-play blueprint likely to influence long-video agents beyond egocentric QA.

### Weaknesses
- CoTT over hierarchical memory likely overlaps prior agentic long‑video approaches. Action: run strict, matched‑backbone and matched‑budget comparisons against strong agentic and training‑free video‑RAG baselines; add a lightweight-critic variant to test the incremental value of planning alone.
- Data construction and inference rely on proprietary LLMs/VLMs. Action: provide a fully open stack with results, release exact prompts/tool schemas/configs, and report contamination checks between generation pipelines and evaluation items.
- Gains may stem from backbone swaps and costs exclude memory-bank build. Action: standardize backbones across methods; report fixed vs. dynamic frame budgets, wall‑clock and $/QA including offline preprocessing; and quantify hierarchical‑RAG hit@K and temporal localization error.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
