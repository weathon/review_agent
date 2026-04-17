# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 8

## Abstract
Agent memory shapes how Large Language Model (LLM)-powered agents, akin to the human brain, progressively refine themselves through environment interactions. Existing paradigms remain constrained: parametric memory forcibly adjusts model parameters, and retrieval-based memory externalizes experience into structured databases, yet neither captures the fluid interweaving of reasoning and memory that underlies human cognition. To address this gap, we propose MemGen, a dynamic generative memory framework that equips agents with a human-esque cognitive faculty. It consists of a \textit{memory trigger}, which monitors the agent’s reasoning state to decide explicit memory invocation, and a \textit{memory weaver}, which takes the agent's current state as stimulus to construct a latent token sequence as machine-native memory to enrich its reasoning. In this way, MemGen enables agents to recall and augment latent memory throughout reasoning, producing a tightly interwoven cycle of memory and cognition. Extensive experiments across eight benchmarks show that MemGen surpasses leading external memory systems such as ExpeL and AWM by up to $38.22\\%$, exceeds GRPO by up to $13.44\\%$, and exhibits strong cross-domain generalization ability. More importantly, we find that without explicit supervision, MemGen spontaneously evolves distinct human-like memory faculties, including planning memory, procedural memory, and working memory, suggesting an emergent trajectory toward more naturalistic forms of machine cognition.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes MemGen, a latent generative memory framework designed to give LLM agents a more human-like and cognitively interwoven memory system. Unlike existing parametric memory (fine-tuning the base model and prone to catastrophic forgetting) or retrieval-based memory (externally stored context being rigid and extractive), MemGen introduces a two-component latent memory architecture: (1) a reinforcement-learned trigger that monitors the agent’s internal reasoning state and decides dynamically when memory should be invoked, and (2) a memory weaver that generates latent machine-native memory tokens on-the-fly rather than fetching or copying text. These latent memory tokens are interleaved back into the LLM’s hidden states during reasoning, enabling a recursive cycle of memory and thought without modifying the frozen backbone.

Experiments across several benchmarks show large performance gains. The authors further show MemGen spontaneously develops human-like memory faculties emerging without explicit supervision. They argue this demonstrates a pathway toward self-evolving agents with fluid cognitive memory, beyond prompt-stuffing or gradient-update-based methods.

The proposed approach is technically sound. MemGen operationalizes a genuine new memory paradigm by placing a reinforcement-learned trigger and a generative latent memory weaver directly into the LLM’s internal token-level reasoning loop, rather than relying on retrieval-append or parametric weight updates. This is architecturally meaningful and experimentally supported across domains, though some of the “human-like cognition” framing appears partly post-hoc. The paper is clearly structured and visually well supported, but occasionally heavy in rhetorical narration. Still, the overall contribution is high-value: MemGen establishes generative latent memory as a first-class alternative to retrieval-based or parametric memory, with a principled RL-trained trigger and real performance gains, making it an architecture-level contribution.

### Strengths
The paper introduces a genuinely new memory formulation based on generative latent memory, rather than yet another variation of retrieval concatenation or parametric weight-updating. The architecture is operational and not hypothetical. Memory is generated in latent token form and dynamically interwoven during reasoning at token-time, with a reinforcement-learned trigger that decides when memory should be invoked, rather than heuristically forcing memory at preset turn boundaries. This is a substantial architectural step. The evaluation is serious and shows not only higher in-domain performance, but also strong cross-domain generalization and non-trivial continual learning behavior. The authors go beyond accuracy metrics and perform targeted interventions on latent clusters to reverse-engineer memory roles, providing convincing evidence of emergent planning, procedural memory, and working memory patterns.

Another notable strength is the breadth and sincerity of experimentation, which I appreciate enormously. The authors evaluate across nine benchmarks and report non-incremental gains. MemGen outperforms both retrieval-based and RL-based agent memory baselines consistently, and unlike most memory systems, it demonstrates improvement even when evaluated out of training domain.

### Weaknesses
The paper leans too aggressively on cognitive analogies (for example, hippocampus-like reconstruction and human-like memory faculties) that are partially justified but still interpretive rather than causally established. The intervention studies are solid, but they still operate at the level of inferred function, not formal guarantees. Some readers will find the cognitive framing oversold relative to the evidence. The stronger claim is architectural novelty. The biological narrative should be toned down in messaging to match what is strictly demonstrated.

In terms of writing, the paper is dense to the point of being unnecessarily heavy, especially in the introduction and motivation. The authors repeat the retrieval-versus-parametric critique multiple times, instead of positioning MemGen directly and succinctly. There is also a conceptual risk of blurring with latent steering and latent CoT intervention work such as Medusa, CODI, Coconut, and related latent policy modulation methods. The differentiation is real, but it should be more sharply articulated to avoid the impression that this is simply another latent-control method rather than a true memory system. Clarity and restraint in framing would strengthen the paper further.

### Questions
(1) Your method operates in the latent space and applies interventions during generation. How do you conceptually and architecturally differentiate MemGen from existing latent intervention and latent CoT works such as CODI, Medusa, Coconut, Co-processor, SoftCoT? IOW what prevents this from being viewed as “latent policy steering” rather than specifically “latent memory”?

(2)  You claim that latent memory is reconstructed and not retrieved or embedded similarity-based. Can you provide quantitative evidence (not just intuition) that the weaver is actually generating novel inferential structure, not just embedding-compressed re-encodings of prior examples?

(3) Under what task conditions does MemGen approach the worst case of inference latency? If this grows with task horizon, can MemGen scale to multi-hour multi-hop agents realistically?

(4) Since the memory trigger is trained via RL, did you observe any pathological cases where it overfires or underfires in edge situations?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel framework to dynamically incorporate memory during LLM reasoning. It uses a 'memory trigger' to determine when to use memory and a 'memory waver' to generate the memory to use. Through this design, they can achieve higher flexibility and capacity compared to the previous parametric memory and retrieve-based memory. Compared to other latent memory methods, their method has 'seamless interleaving of reasoning and memory', and can 'generatively reconstruct memory into novel, coherent insights' instead of using memory in a retrieval-based method.

### Strengths
The paper is well structured in general, and most of its explanations and descriptions are clear and intuitive. The proposed framework is pretty interesting and promising. Moreover, they provide extensive experiments that are relatively comprehensive and persuasive.

### Weaknesses
1. Although the paper is generally well written and clear, some parts are still somewhat confusing and insufficiently explained.
    * Sections 4.2 and 4.3 seemed confusing to me before reading Appendix C and D. Both 'memory trigger' and 'memory waver' require an instance of each other for their own training. This may confuse readers, and is only explained in Appendix D. Considering that Appendix D is not referenced in the main text until the end of Section 5.1, readers may find it difficult to identify where to look for information that could clarify their confusion. It would be helpful if the paper could provide a brief explanation in the main text on how this coupling is addressed, as it would improve the reader’s understanding.
    * The explanation of the training method of 'memory trigger' seems overly general and lacks sufficient detail. More details about how the RL training is conducted could be included in the Appendix.
2. More experiments to verify the necessity of 'memory waver' could be appreciated. Currently, the paper has conducted an ablation study about the value of 'memory trigger', but the corresponding ablation of 'memory waver' is absent.

### Questions
1. Could you please explain why you use hidden states **H** as the input of 'memory trigger' and 'memory waver' instead of the original text? It seems odd that the hidden states, which are already encoded by the LLM, go through another LLM one more time, especially since the LLM is only a LoRA variant version of the same LLM that **H** resulted from. The LLMs are learned to handle text tokens, and the space of encoded hidden states may differ substantially from the original space of token embedding.
2. I am curious about the memory capacity of your framework. Your sequential experiments for RQ3 have demonstrated that after training on the training data from the four datasets, your framework exhibits almost no 'forgetting', which is pretty promising. But will your framework eventually start to 'forget' when the training samples scale up? If so, what is the capacity of your framework? Moreover, how can we increase the capacity?

### Soundness
4

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
4

### Summary
The paper proposes MemGen, a memory module for LLM agents: A small trigger watches the model’s hidden states and decides when to call memory. A weaver then generates a short sequence of latent vectors (soft tokens) that are inserted into the model’s hidden states to guide the next steps of reasoning. The base model stays frozen. The authors test MemGen on many benchmarks and report solid gains over parametric finetuning and retrieval memory. They also analyze the learned latent vectors and suggest roles like planning, procedural, and working memory.

### Strengths
-  **Originality:** Although the paper did not fully convince me about the usefulness of the mechanism, I found the work creative and inspiring. A big difference between LLMs and humans is that humans can invoke on-demand details based on a cue, effectively interrupting ongoing reasoning to retrieve distributed multimodal representations that support the current task (for example, recalling the visual details of a face when talking about a person). I have not seen this specific on-demand recall framing in prior work, and it could be more useful in other settings too (for example, human-like cross-modal recall).

### Weaknesses
* **Unfair comparison (most critical):** The paper compares against baselines with the same backbone size as the reasoner. However, MemGen uses two extra components that are instantiated on the backbone (trigger and weaver, implemented with LoRA). This increases effective capacity and compute. A fairer comparison would use three base models with LoRA adapters or a single larger model with a comparable parameter and compute budget. Models with larger capacities could implement the same mechanisms internally, reducing the need for an artificial extra trigger (the trigger could be "implicitly" learned inside the larger model).

* **Unclear mechanisms and training signals:** The paper says the synthesized memories are prepended to the internal state of the frozen reasoner, but it is not clear where this happens, which layer sees them, and how they interact with the existing hidden states and KV cache. The reward is also not explained, and it is unclear where the training signals and datasets come from.

* **Baseline setup:** The baselines (for example, SFT, GRPO, REINFORCE) feel under‑specified, and for the memory baselines it is uncertain they are the best comparison for these tasks. Do the chosen tasks actually require a memory system, or would stronger reasoning or longer context suffice?

### Questions
* **Reward and data:** How are rewards calculated for both the trigger and the weaver training? Where do the training signals come from? Is another AI judging the results, or are there gold labels or environment success signals? Please also clarify the sources of the training data.

* **Capacity vs necessity:** Is there anything MemGen can learn that a model cannot internalize directly in its parameters with the same trainable‑parameter and compute budget? Today’s models already decide when to stop thinking and how to reuse past context to some degree.

* **Nature of the latent outputs:** Do you have any explanation or intuition about the resulting non-human-readable latent tokens and why they are effective?

* **Exact insertion details:** When you say memories are prepended to the internal state, where exactly is this done, at which layer, and how are they combined with existing states and attention caches?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents MemGen, a generative latent memory framework that enables LLM agents to dynamically synthesize and integrate memory during reasoning. The core contribution—a memory trigger coupled with a memory weaver—is well-motivated and technically sound, offering a promising alternative to parametric and retrieval-based paradigms. Experiments across nine benchmarks demonstrate consistent gains and strong cross-domain generalization, with ablations supporting the necessity of learned memory invocation.

### Strengths
* I find it novel to leverage latent tokens as memory carriers, given that only two LoRA adapters are fine-tuned while surpassing full GRPO. The proposed method differs from previous latent-based memory approaches such as MemoryLLM and Memory3 in that the latent memory is context-sensitive rather than reusing KV caches. This aligns with the authors’ concept of generative memory.
* The method demonstrates promising transferability (Figure 3); inserting tokens to intervene in decoding appears to effectively mitigate catastrophic forgetting. The frequency study in Figure 4 is interesting.
* The authors manage to provide interpretability analyses of latent memory, ranging from clustering visualizations to post-hoc examinations.

### Weaknesses
* I am somewhat confused about the memory form: it seems to reside within the weaver’s parametric memory, suggesting that memgen may not be a purely latent-memory method—its carrier is parametric knowledge manifested in latent form.
* How are the weaver and trigger trained? Is the process akin to an EM-style alternating optimization, or are they jointly trained? If the latter, the training process risks to be highly unstable.
* More ablation studies could be included, particularly regarding the LoRA parameter settings of the weaver (Table 6 shows only a single configuration).
* Could the latent token length be made dynamic rather than fixed a priori?

### Questions
- How is the trigger and weaver jointly trained?  
- Does fixing K impose a capacity bottleneck on more long-horizon task? Is it possible that memgen can benefit from a dynamic memory length mechanism?  
- How is catastrophic forgetting avoided inside the weaver itself, since its parameters continuously absorb new experience while the core LLM stays frozen?  
- Could the authors provide more detailed ablation studies?

### Soundness
3

### Presentation
3

### Contribution
3
