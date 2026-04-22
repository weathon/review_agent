# Auto-scaling Continuous Memory for GUI Agent

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
We study how to endow GUI agents with scalable memory that help generalize across unfamiliar interfaces and long-horizon tasks. Prior GUI agents compress past trajectories into text tokens, which balloons context length and misses decisive visual cues (eg. exact widget size and position). We propose a continuous memory that encodes each GUI trajectory into a fixed-length sequence of continuous embeddings using the VLM itself as an encoder; these embeddings are plugged directly into the backbone’s input layer, sharply reducing context cost while preserving fine-grained visual information. As memory size and retrieval depth increase, performance improves monotonically, unlike text memories that degrade with long prompts. To grow memory at low cost, we introduce an auto-scaling data flywheel that (i) discovers new environments via search, (ii) synthesizes tasks with an open-source VLM, (iii) rolls out trajectories with the agent, and (iv) verifies success with the same VLM. Using this pipeline, we collect 10k trajectories for about \$500 and fine-tune only the memory encoder (LoRA on a Q-Former, 1.2\% parameters) with 1,500 samples. On real-world GUI benchmarks, our memory-augmented agent consistently improves success rates under long horizons and distribution shifts. Notably, Qwen-2.5-VL-7B + continuous memory achieves performance comparable to state-of-the-art closed-source models (eg. GPT-4o, Claude-4).
Our data and code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CoMEM, a framework that enhances training and inference efficiency for GUI agents by integrating memory compression within a self-validating data-generation flywheel. For efficient trajectory representation, each trajectory is encoded into a fixed number (e.g., 8) of learned vectors, which are used as in-context examples and directly injected into the VLM’s input layer. On web-GUI benchmarks, a 7B open-source VLM augmented with this memory matches or surpasses closed-source baselines (e.g., GPT-4o).

### Strengths
- The focus on agent scaling with memory is timely for GUI agents. Compressing trajectories while preserving visual cues effectively targets generalization to unseen interfaces and long-horizon tasks.
- The paper presents a simple yet effective scaling approach for GUI agents that uses fixed-length continuous embeddings which can be retrieved and prepended without inflating the context length.
- The system is both effective and efficient, requiring only lightweight fine-tuning of the memory encoder and enabling low-cost memory growth.
- The literature coverage on GUI agents and memory for LLMs is solid, and the proposed pipeline is clearly positioned relative to recent work.

### Weaknesses
- The novelty mainly rests on analysis and design choices for GUI agents rather than on a fundamentally new learning principle, as the idea of automating a data collection flywheel through environment interaction is not new for long-horizon sequential decision-making agents [1, 2, 3, 4].
- The paper’s use of “auto-scaling” is potentially misleading. Auto-scaling conventionally denotes elastic compute scaling, whereas the proposed data flywheel increases data, not capacity. The authors should revise or clearly define the term, for example as “automated data expansion” or “automated data augmentation,” to avoid overstating the contribution.
- The approach is validated mainly with a 7B VLM. It assumes open-source VLMs have sufficient planning and verification ability to maintain diversity and quality at scale, but there is no convincing analysis for smaller models. A more practical setting would include transparent results or insights for small models such as 3B and 1B.
- Some numbers are inconsistent between the abstract and the conclusion. The paper lacks a transparent presentation of the collected memory demonstrations, detailed cost accounting, retrieval configuration, and memory bank curation criteria, which hinders reproducibility.
- Analysis and discussion of bottlenecks for scaling the CoMEM framework are limited. The paper asserts that trajectories can be compressed to as few as eight continuous vectors but does not justify when eight are sufficient as trajectory complexity grows. A simple test across complexity regimes would strengthen the claim.
- The authors claim that high-quality training is important, but they do not analyze the robustness or data quality of the memory.
- While the paper presents promising results, the evidence supporting generalization to unseen environments could be strengthened through deeper analysis or focused case studies showing where CoMEM succeeds while the baseline fails, thereby clarifying the factors driving the improvement.

Minor
- Figure 2 is informative but visually dense. Reducing icon clutter and explicitly highlighting the key components of the flywheel would improve readability.

[1] Shinn, Noah, et al. "Reflexion: Language agents with verbal reinforcement learning." Advances in Neural Information Processing Systems 36 (2023): 8634-8652.

[2] Wang, Guanzhi, et al. "Voyager: An open-ended embodied agent with large language models." arXiv preprint arXiv:2305.16291 (2023).

[3] Dai, Zhuyun, et al. “Promptagator: Few-shot Dense Retrieval From 8 Examples.” Proceedings of the International Conference on Learning Representations (ICLR), (2023).

[4] Wang, Zun, et al. “Bootstrapping Language-Guided Navigation Learning with Self-Refining Data Flywheel.” Proceedings of the International Conference on Learning Representations (ICLR), 2025,

### Questions
- Could prior knowledge be used to pre-adjust the retrieval range or weighting during evaluation so that the memory focus becomes sharper for a given task family? In particular, how could CoMEM be improved or extended to support such prior-conditioned retrieval?
- For training the memory encoder, what sampling criteria are used to balance the quality and diversity of trajectories? Is there a principled metric for determining the required number of training samples? If fewer but more diverse trajectories are used, could the method achieve better data efficiency? This question arises because performance degrades around 2,000 samples, suggesting sensitivity to sample selection and an increased burden on hyperparameter tuning.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a continuous memory methodology to address the generalization challenges GUI agents face with unfamiliar interfaces and long-horizon tasks. Existing GUI agents compress past trajectories into text tokens, which dramatically increases context length and loses crucial visual cues (e.g., exact widget size and position).

To address this, the authors propose CoMEM (Continuous Memory). Each GUI trajectory is compressed into a fixed-length sequence of 8 continuous embeddings using the VLM itself as an encoder, which are then directly injected into the backbone's input layer. Experimental results demonstrate that continuous memory shows monotonic performance improvement as memory size and retrieval depth increase, whereas text-based memories degrade beyond roughly ten retrieved items due to ballooning sequence length, increased attention overhead, and accumulated semantic noise.

The authors also propose an auto-scaling data flywheel for memory scaling. This pipeline consists of four stages: (1) discovering new environments via search engines, (2) synthesizing tasks with open-source VLMs, (3) rolling out trajectories with the agent model, and (4) verifying success with the VLM. Using this pipeline, they collected 15,145 GUI trajectories for approximately $553 and fine-tuned only the memory encoder (LoRA on Q-Former, 1.2% of parameters) with 1,500 training samples.

In benchmark experiments, Qwen-2.5-VL-7B equipped with CoMEM achieves performance comparable to or better than state-of-the-art closed-source models like GPT-4o and Claude-4, significantly outperforming them on the Webvoyager dataset. The paper demonstrates that this scaling law can be leveraged to continuously improve GUI agent performance through low-cost, automated data collection and efficient training.

### Strengths
- The paper is well-motivated and easy-to-read.
- The auto-scaling data flywheel presents a practical method to collect large-scale data (15,145 trajectories) at low cost ($553) without human annotation.

### Weaknesses
- The differences from existing memory-based methodologies are not clearly visible. It seems necessary to clearly articulate the differences from various approaches that enhance GUI Agents using memory.
-  The only difference from existing methodologies in the paper appears to be compressing existing trajectories through a memory encoder. The contribution seems insufficient.
- The baselines only include Text-based Memory, which is a very naive memory utilization method, without including other memory-based approaches. Recent baselines such as AWM or Memp should be included.
- This paper does not include ablation experiments..

### Questions
- What exactly are the criteria for retrieving trajectories?
- Qwen2.5-VL-7B shows higher performance than UI-TARS-1.5-7B. what is the reason for this?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces continuous memory, where each GUI trajectory is compressed by the VLM’s own encoder into a fixed-length sequence of continuous embeddings. During inference, retrieved memory vectors are directly injected into the VLM’s input embedding layer, avoiding context explosion while preserving visual detail. The authors also propose an automatic data pipeline that automatically constructs the memory database using open-source VLMs and search engines. They report a total cost of $553 for collecting 15,145 trajectories across 6,676 environments. The approach is evaluated on MMInA, Multimodal-Mind2Web, and WebVoyager benchmarks against text-based memory and various baselines. Results show consistent performance gains as memory scale and retrieval depth increase, while text memory degrades with longer prompts.

### Strengths
- The work offers a pragmatic solution by injecting external experience as continuous vectors directly into the VLM, circumventing long-context limitations while retaining fine-grained GUI visual information. Combined with a low-cost automated data flywheel, it demonstrates that open-source 7B-scale models can rival or surpass closed-source systems.
- Memory vectors are integrated at the embedding layer rather than concatenated as text, resulting in a cleaner architecture and predictable inference cost. The use of Q-Former + LoRA (only 1.2% of parameters) also facilitates efficient adaptation.
- The data collection process is scalable.
- The authors provide systematic evidence that performance scales monotonically with memory size and retrieval depth, while text-based memory deteriorates under longer contexts.

### Weaknesses
- Incomplete memory write/forget policy: The work focuses on the read path but lacks a quantitative analysis for when and how to write, deduplicate, or prune memories.
- The text-memory baseline is highly implementation-sensitive (e.g., summarization, structure, retrieval prompting). Insufficient disclosure of its setup may exaggerate the gap with continuous memory.
- Retrieval quality and error propagation: CLIP-style keys with FAISS nearest-neighbor retrieval may fail under small UI changes or noisy web elements (ads/pop-ups). If noisy memories are blindly injected, they may introduce strong interference. No robustness or gating analysis is presented.

### Questions
- Can the authors provide an evaluation using an open source VLM for the data flywheel?
- What motivated the choice of eight embeddings per trajectory? How does performance trade off with 4/16/32?

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
The paper proposes a plug-and-play “continuous memory” for GUI agents. Past multi-modal trajectories (screens + actions) are encoded by a small Q-Former into a handful of fixed-length vectors and prepended as a soft prefix at the input-embedding layer, avoiding prompt bloat while letting the backbone attend to relevant experience. At runtime, a CLIP+FAISS retriever pulls top-k similar trajectories, the memory encoder produces the vectors, and the VLM consumes them with the current screen/instruction. A low-cost flywheel keeps expanding the memory bank. Experiments show strong gains on real-world web benchmarks.

### Strengths
Clean, architecture-agnostic injection, lightweight tuning and low inference overhead make it practical. The approach exhibits a clear scaling trend, more/better memories and deeper top-k retrieval steadily help, while delivering competitive results on real web GUIs and lifting specialized baselines too. The automated data flywheel keeps costs down and coverage growing, and the design plays nicely with existing agent stacks (retrieval, planning, tool use), making it easy to slot into production-style GUI agents

### Weaknesses
Q1: Gains on OSWorld are modest, suggesting the memory mostly captures web-navigation regularities and doesn’t transfer cleanly to desktop/app workflows. If the base model switches (e.g., to UI-TARS) and improvements remain small, that points to a real domain gap rather than an underpowered baseline.

Q2: The CLIP+FAISS first-stage retrieval is coarse, and the injected memory prefix is cheap for the model to trust. When retrieval is slightly off, those few vectors can still steer attention the wrong way. The paper lacks stress tests for retrieval noise and stronger reranking/quality gates (e.g., cross-encoder re-rank, confidence gating, or learned trust of memory).


Q3: Collapsing long, multi-step GUI traces into ~8 learned vectors is aggressive. It likely drops temporal dependencies and subtle causal cues (order of actions, transient states), especially under large layout shifts. The scaling curves are encouraging, but there isn’t a convincing analysis that these embeddings retain the decision-critical bits in hard OOD cases versus capturing surface similarity.

Q4: The low-cost loop (discover → synthesize → rollout → auto-verify) is efficient but invites bias and label noise. Without clear validator error rates, duplicate/failure audits, or freshness governance, the memory bank can accumulate skewed or stale experiences. Ethical/robustness issues (privacy, site policies) are acknowledged but not operationalized.

### Questions
see above weakness

### Soundness
3

### Presentation
3

### Contribution
2
