# Deep Reflection Hinting: Leveraging Offline Knowledge for Improving LLM Agents Adaptation

- Decision: Reject
- Scores: 6, 2, 2

## Abstract
Large language model (LLM) agents perform well in sequential decision-making tasks, but improving them on unfamiliar domains often requires costly online interactions or fine-tuning on large expert datasets. These strategies are impractical for closed-source models and expensive for open-source ones, with risks of catastrophic forgetting. Offline trajectories offer reusable knowledge, yet demonstration-based methods struggle because raw traces are long, noisy, and tied to specific tasks. We present \emph{Deep Reflection Hinter (DR.Hinter)}, an agentic system that distills offline traces into compact, context-aware hints. A zooming mechanism highlights decisive steps in long trajectories, capturing both strategies and pitfalls. Unlike prior methods, DR.Hinter leverages both successful and failed trajectories, extracting guidance even when only failure data is available, while supporting parallelized hint generation and benchmark-independent prompting. At inference, a retriever selects relevant hints for the current state, providing targeted guidance with transparency and traceability. Experiments on MiniWoB++, WorkArena, and WebArena Lite show that DR.Hinter consistently outperforms strong baselines, including human- and document-based hints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces DR. HINTER, an agentic framework that turns offline trajectories—both successful and failed—into compact, reusable natural-language hints to improve LLM agents without fine-tuning. A zooming module identifies critical decision points within long traces, and a reflection step distills these segments into strategy and pitfall hints, each indexed by a semantic key for retrieval. Hints can be generated from single traces, contrastive pairs, or multi-trace aggregates, enabling coverage even when only failures exist, and are retrieved at inference either step-by-step (contextual) or once per episode (goal-conditioned). This approach is benchmark-independent, scalable, and transparent, and it supports closed-source models by shifting computation offline to a more capable “hinter” model. Across MiniWoB++, WorkArena-L1, and WebArena-Lite, DR. HINTER consistently outperforms ReAct and an AutoGuide reimplementation, sustains competitive out-of-task generalization, and proves more practical than documentation or human-authored hints. Ablations show that zooming improves hint quality and larger hinter models yield further gains, especially on long-horizon tasks.

### Strengths
1. The general idea to distill off-line traces into compact hints looks promising.
2. Some designs like zooming and extracting hints from both successful and failure trajectories sound practical
3. The writing is clear in general.

### Weaknesses
1. I think the experimental setting is not very good. Based on the descriptions from line 252-275, the authors seems to collect trajectories from benchmark data. Even in out-of-task generalization, the hints from other tasks on the same datasets are used, which means they use hints of similar tasks in the same distribution in the evaluation. In WebArena, the benchmark uses task templates to construct tasks, where multiple tasks only differ in a few values, while the trajectories to complete them are very similar. I think this gives the question about whether the results are overfitting and how the proposed methods can be applied in general tasks.
2. The experiments remain the GPT5--nano and GPT5-mini, but the effectiveness on more powerful models like GPT-5 or Claude-4.1 remain unclear.
3. From example in Appendix E.2, I feel that the generated hints are specific to certain tasks. It would be better to further abstract hints or scale the number of hints to make this mechanism generalizable.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Deep Reflection Hinter (DR.HINTER), an agentic system that distills offline trajectories into explicit, context-aware hints. DR.HINTER is capable of extracting hints from both successful and failed trajectories simultaneously. Experiments on MiniWoB++, WorkArena-L1, and WebArena-Lite show that DR.HINTER's method outperforms ReAct and AutoGuide, demonstrating the high efficiency of the proposed approach.

### Strengths
* The proposed method has strong generality. The author consider the scenario in practical applications where only failed trajectories might be available.

* The proposed method has strong interpretability. The Case Study in Section 5.4 makes it easy to understand the reasons for the method's effectiveness.

* The paper provides a detailed description of the method's design and prompt design, which facilitates reproducibility.

### Weaknesses
* A key innovation of this work is the ability to utilize both successful and failed trajectories. However, some existing works also emphasize the ability to extract experience from failed trajectories to guide subsequent inference, such as Automanual [1]. This paper lacks a comparison and experimental contrast with these works.

* Another innovation of this work is the design of the zooming mechanism. However, the paper only proves the importance of this step through ablation experiments and lacks an interpretability analysis or qualitative analysis of this step. For example, it is unclear whether the zooming mechanism would still provide benefits if using large models that support long contexts, such as GPT-5, as the complete trajectory could also provide richer information.

* The paper only verifies that the method achieves performance improvements on GPT-5-Nano and GPT-5-Mini. It lacks experimental analysis on larger models like GPT-5, Gemini 2.5 Pro, and state-of-the-art open-source models like Qwen3-8B.

* A significant advantage of improving model performance through context is strong generalization. However, this paper conducts limited generalization experiments. For instance, the paper should experiment with whether hints obtained on MiniWob++ can be applied to other benchmarks of the same type.

[1] Minghao Chen, Yihang Li, et.al. "AutoManual: Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning"

### Questions
See Weaknesses.

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
3

### Summary
The paper proposes Deep Reflection Hinting (DR. HINTER), an offline pipeline that distills agent trajectories—successful and unsuccessful—into natural-language hints keyed by semantic summaries for retrieval at inference time. The pipeline has three main stages: (i) Zoom & Reflect, which identifies critical steps within long traces and prompts an LLM (“hinter”) to produce step-focused guidance; (ii) Hint indexing via semantic keys; and (iii) Retrieve & Act, which injects hints either once per episode (goal-conditioned) or per step (context-conditioned). Experiments on MiniWoB++, WorkArena-L1, and WebArena-Lite report gains over ReAct and AutoGuide; ablations assess out-of-task generalization, the effect of larger hinters, and documentation/human-hint baselines.

### Strengths
* The paper provides a clear, modular pipeline with ablations to justify some choices (e.g., LLM “zooming”), and the episode- vs. step-level retrieval modes are well-motivated.  
* The method leverages failed trajectories to mine “what not to do,” enabling hint creation even when no success trace exists—useful in low-success regimes.  
* Broad evaluation across three web-agent benchmarks, including documentation and human-hint comparisons; out-of-task results and hinter-scaling ablations further add empirical soundness.  
* Qualitative case studies in the appendix support interpretability claims.

### Weaknesses
My main concern is the novelty aspect, but I am open to revising my assessment during the discussion period.
* Conceptual novelty feels incremental relative to prior works. DR. HINTER’s main advances— zooming, single/failed/multi-trace, etc.—read as engineering refinements. Additionally, there are relevant prior works that are not discussed (e.g. \[1\]).  
* **Most components rely on prompts to an LLM.** The paper should be more explicit about exactly which roles are handled by LLM prompting (e.g., the step selection) and which are not.  
* **Domain scope.** All experiments are browser-based; it’s unclear how the hint format transfers to other settings. Such empirical specificity should be explicit in the title or be accompanied by an additional domain.

\[1\] Holt, S., Luyten, M. R., & Pouplin, T. (2025). Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search. *ArXiv*. https://arxiv.org/abs/2506.09171

### Questions
1. How do you address duplicate, contradictory, or incorrect hints?   
2. Will you release the AutoGuide reimplementation so others can reproduce results?

### Soundness
3

### Presentation
3

### Contribution
2
