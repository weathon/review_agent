# LatentEvolve: Self-Evolving Test-Time Scaling in Latent Space

- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Test-time Scaling (TTS) has been demonstrated to significantly enhance the reasoning capabilities of Large Language Models (LLMs) during the inference phase without altering model parameters. However, existing TTS methods are largely independent, implying that LLMs have not yet evolved to progressively learn how to scale more effectively. With the objective of evolving LLMs to learn ``how to scale test-time computation,'' we propose LatentEvolve, a self-evolving latent TTS framework inspired by the complementary learning system (CLS) theory. Analogous to the human brain's dual system of a fast-recall hippocampus and a slow-consolidating neocortex, LatentEvolve comprises two evolutionary components: \textit{daytime scaling}, which rapidly retrieves historical latent representations to better guide current LLM reasoning; and \textit{nighttime scaling}, which integrates past latent optimizations in a manner akin to the human brain's consolidation of experiences during sleep. The alternation of daytime and nighttime processes facilitates a fast and slow evolution of LLM TTS, mirroring human cognitive dynamics in a fully unsupervised manner. Extensive experiments across eight benchmarks and five model backbones demonstrate that our LatentEvolve surpasses state-of-the-art TTS methods such as LatentSeek and TTRL by up to $13.33\%$ and exhibits exceptional cross-domain and cross-backbone generalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes LatentEvolve, a latent test-time scaling framework that lets an LLM self-evolve across queries without parameter updates to the backbone. This analogy to the human brain’s dual system is very interesting. The experimental results are excellent across eight benchmarks and five models. LatentEvolve often matches or exceeds state-of-the-art TTS baselines (e.g., TTRL, LatentSeek) and shows cross-domain transfer and continual learning dynamics.

### Strengths
1. This analogy to the human brain’s dual system is very interesting.
2. The weighted momentum transfer uses change vectors $\Delta z$ from similar tasks—intuitively leveraging “how we improved” rather than “where we ended”.
3. The experiments are very extensive and comprehensive, and demonstrate consistent gains over strong TTS baselines and transfer to out-of-domain datasets, suggesting a path toward continual, unsupervised evolution at inference.
4. The writing is clear and easy to follow.

### Weaknesses
1. This method introduces multiple rollout refinement steps, retrieval over a growing buffer, and periodic weaver training for every instance. The paper should quantify total inference FLOPs/tokens vs. baselines under matched compute budgets, and report throughput/latency impacts.

2. While ablations remove Daytime/Nighttime, there is no scaling analysis of the episodic buffer size, retrieval top-$k$, or archive threshold $\tau$. These directly affect quality and cost.

3. Some AIME scores are based on small sets and can be volatile. I suggest reporting mean ± stdev over ≥3 runs in different sampling seeds.

### Questions
No

### Soundness
3

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
4

### Summary
This paper proposes a new framework, called LatentEvolve, for improving test-time scaling (TTS) of large language models (LLMs) by introducing a self-evolving latent-space mechanism. Key contributions include:
Identification of a limitation in existing TTS approaches: most treat each query independently and thus do not accumulate experience across queries to improve future inference. 


* Inspiration from the complementary learning systems (CLS) theory (hippocampus + neocortex) to design a dual-phase process:


* Daytime scaling: fast retrieval of prior latent interventions + on-the-fly latent optimization for each incoming query.


* Nighttime scaling: consolidation of past experience into a compact latent “weaver” model that produces better initial latent states for future queries.



* Empirical evaluation across multiple LLM backbones (e.g., Llama-3, Qwen, Gemma) and eight benchmarks spanning general QA, math reasoning, scientific reasoning, medical reasoning. The method shows consistent improvements over strong baselines including LatentSeek, TTRL, etc. 

* Evidence of cross-domain generalization (i.e., improvements transfer from math domain to GPQA, JAMA) and continual learning capability (processing new domains sequentially doesn’t degrade performance on previous domains) via the dual-phase mechanism.

### Strengths
* The analogy to CLS (fast + slow learning) offers a fresh lens on test-time adaptation / scaling.


* The authors test across several backbones, many dataset types, and compare to a large set of baselines. This breadth increases confidence in generality.


* The idea that the system accumulates experience over time and becomes better aligned with real-world deployment where models repeatedly see new inputs.

### Weaknesses
* The daytime latent optimization (via self-rewarding policy gradient) and the nighttime consolidation add overhead. The paper would benefit from more discussion of runtime, cost, scalability (especially for very large models or high query volume).


* It is unclear how sensitive the performance is to the design choices of latent-token length, initialization, buffer size, retrieval similarity metric, and tuning of thresholds (τ), iteration count (K), etc. Some of these hyperparameters may require heavy tuning.


* The episodic buffer grows with experiences; managing retrieval, similarity search, etc might become onerous in production. More details on buffer capacity, retrieval latency, memory demands would strengthen the paper.  Also,  the retrieval process  relies on semantic embedding. If the retrieval process is inaccurate, how does it influence the performance?


* The self-rewarding strategy uses the model itself to score Q(y). There is an inherent risk of circularity or bias: if the model is wrong but confident, it may reinforce the wrong latent sequence. The paper would benefit from ablations or analysis of reward reliability.


* While some cross-domain results are given, one may ask how the system performs when the new queries are far from any prior buffer entries (i.e., the retrieval gives weak plus few relevant experiences). Does the system fallback gracefully or degrade badly?

### Questions
* Could you provide detailed statistics on the inference cost (wall-clock time, latency) of daytime scaling + retrieval + latent optimization vs. a vanilla inference pass? In deployment settings, latency is often critical.





* If the system is exposed to a query from a domain completely unseen so far (so retrieval returns low‐similarity neighbours), how does performance degrade? Is there a fallback default initialization?




* Could you share some qualitative examples of latent sequences (zbase → z*), or visualise how the latent space evolves over time (e.g., clustering of queries, drift in latent interventions)? This would help understand what the system is learning.

### Soundness
3

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
LatentEvolve is a novel self-evolving test-time scaling (TTS) framework for Large Language Models (LLMs) that aims to enable LLMs to progressively learn "how to scale test-time computation" without altering model parameters during inference. Inspired by the Complementary Learning System (CLS) theory of the human brain, LatentEvolve introduces a dual-phase evolutionary process comprising "Daytime Scaling" for rapid, episodic adaptation and "Nighttime Scaling" for slow, procedural consolidation. This allows LLMs to accumulate and refine experiential knowledge in an unsupervised manner, leading to enhanced reasoning capabilities.

While the mechanism of Test-Time Scaling using memory (latent representations) rather than computation (tokens/steps) is novel, the paper fails to contribute meaningful insight or theoretical understanding to the community. The Complex Learning System (CLS) analogy functions as an unnecessary, unsupported wrapper around a complex heuristic, and the authors fail to provide the necessary ablations or latent space analyses to demonstrate why the full, complex system is necessary.

### Strengths
1.  This paper is well-structured and easy to follow. The figures and tables are clear, professionally executed, and effectively convey the high-level architecture and quantitative results.
2.  LatentEvolve introduces a novel direction for Test-Time Scaling by evolving the memory (latent space representations) rather than scaling the computation (tokens/steps). This is a conceptually interesting shift from existing TTS methods like CoT or Self-Refine and is a valuable idea for the community.
3.  LatentEvolve achieves state-of-the-art results across several reasoning benchmarks, demonstrating its practical effectiveness.

### Weaknesses
1. The paper heavily relies on a biological metaphor (CLS) which does not translate into a strong, mathematically-justified framework. In other words, the connection between EvolveLatent and the hippocampus (Daytime Scaling) and neocortex (Nighttime Scaling)  is purely descriptive. If the approach truly brings insight, the paper should rigorously define what property of the latent space (e.g., lower variance, better separability, higher clustering of task-relevant information) is improved by the "Nighttime" consolidation, independent of the bio-analogy. Without this, the CLS is an unnecessary wrapper around a complex heuristic.
2. The method combines multiple components: A latent memory buffer, a retrieval mechanism, and an integration/consolidation step. An ablation study is needed where "Nighttime Scaling" is replaced by a simple, computationally cheap Exponential Moving Average (EMA) or FIFO (First-In, First-Out) replacement strategy. If the performance gap is marginal, the core mechanism of "evolution" is debunked.
3. The method's performance will inevitably depend on the size of the latent memory buffer. I wonder if the memory needs to grow indefinitely to maintain performance gains across a wide distribution of tasks (e.g., across hundreds of thousands of prompts). If so, the method is not scalable for real-world deployment.
4. The experimental section lacks depth regarding the _mechanism_ of the proposed method's success, failing to generate insight. The core novelty resides in "self-evolving test-time scaling in latent space," yet there is no visual evidence of this evolution. Does Nighttime truly consolidate the latent space, making representations for similar problems tighter or better clustered? Does the retrieval process pull the currently processed latent vector closer to the centroid of successful historical solutions? Without these analysis, the claims of "evolution" and "consolidation" remain hand-wavy and unproven.

### Questions
- How is the performance degradats if the memory buffer size is drastically limited (e.g., using only the 100 or 500 most recent examples)?

### Soundness
2

### Presentation
3

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
This paper addresses a key limitation of existing TTS methods: they are static, independent processes that do not improve or adapt over time. The authors reframe TTS as a skill that an LLM can learn and master. They propose LatentEvolve, a novel self-evolving framework inspired by the CLS theory from neuroscience. The method operates on a "Daytime-Nighttime" cycle, where during the daytime (inference), the model performs inference by fusing two parallel latent scaling modules, and nighttime (evolution) is an offline phase, which does self-distillation. Through this cycle, the stable module progressively "evolves," allowing the model to get better at the skill of scaling its computation. Experiments on reasoning benchmarks (GSM8K, MATH, SciBench) show that LatentEvolve significantly outperforms both base models and existing static TTS methods like Self-Refine and Time-Warping.

### Strengths
- The primary strength is the novel reframing of Test-Time Scaling as a learnable, evolving skill rather than a static inference algorithm. This is a significant conceptual leap.
- Instead of an ad-hoc design, this methodology is principled in CLS. 
- The empirical results are quite strong.

### Weaknesses
- The paper clearly demonstrates the benefits of the Nighttime evolution steps but does not quantify their cost. For practical application, it's crucial to know the computational and data requirements of a single offline evolution phase. Could the authors please quantify the computational cost (e.g., GPU hours, number of samples required) for a single "Nighttime" evolution step?
- The paper states the transient module is "reborn," which implies re-initialization. It is not fully clear if this is a full random re-initialization (which might be unstable) or a reset to a pre-trained state. Could you please clarify the "reborn" process for the Transient Latent Scaling (TLS) module? Is it a full re-initialization with random weights, or is it reset to a specific pre-trained checkpoint?

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
