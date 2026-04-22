# Latent Visual Reasoning

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 4

## Abstract
Multimodal Large Language Models (MLLMs) have achieved notable gains in various tasks by incorporating Chain-of-Thought (CoT) reasoning in language spaces. Recent work extends this direction by leveraging external tools for visual editing, thereby enhancing the visual signal along the reasoning trajectories. Nevertheless, these approaches remain fundamentally constrained: reasoning is still confined to the language space, with visual information treated as static preconditions. We introduce Latent Visual Reasoning (LVR), a new paradigm that enables autoregressive reasoning directly in the visual embedding space. A visual encoder first projects images into visual tokens within a joint semantic space shared with the language model. The language model is then trained to generate latent states that reconstruct key visual tokens critical for answering the query, constituting the process of latent visual reasoning. By interleaving LVR with standard text generation, our model achieves substantial gains on perception-intensive visual question answering tasks. In addition, we adapt the GRPO algorithm to conduct reinforcement learning on latent reasoning, further balancing LVR and textual generation.  We show that LVR substantially improves fine-grained visual understanding and perception, achieving 71.67\% on MMVP compared to 66.67\% with Qwen2.5-VL. Code base and model weights will be released later.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Latent Visual Reasoning (LVR), a method that enables multimodal large language models to reason directly within their latent visual embedding space. LVR introduces a mechanism for switching between normal text generation and a special latent reasoning mode, indicated by specific tokens. In this mode, the model manipulates internal visual representations relevant to the question before producing text outputs. The training process has two stages: supervised fine-tuning using bounding-box annotations to align latent states with image regions, and reinforcement learning that refines reasoning quality.

Experiments are conducted on Qwen2.5-VL backbones at 3B and 7B scales. LVR shows the largest gains on fine-grained visual reasoning benchmarks such as MMVP, MMStar, and MM-Vet, compared to both “think-about” (text-only reasoning) and “think-with” (external visual reasoning) baselines. It performs particularly well on tasks requiring precise spatial, compositional, or attribute-based reasoning. The paper also introduces new decoding strategies for latent reasoning, comparing fixed-length and adaptive termination approaches. Training efficiency and decoding stability are analyzed through detailed ablations.

### Strengths
1. Simple modification of standard MLLM pipelines without external tools.
2. Adapts reinforcement learning to hidden-state spaces effectively.
3. Uses fixed encoders and adaptive batching for efficient multi-step reasoning and conceptually extends beyond CoT into “machine mental imagery.”
4. Open weights and code for reproducibility.

### Weaknesses
1. Gains are strongest on perception-heavy benchmarks, but less so on others (e.g., Relative Reflect).
Question: Have the authors tested multi-image or compositional reasoning tasks? Would LVR generalize beyond single-image settings?

2. The variable-length LVR decoding strategies (LatentEnd Token, Mode Switching Loss) perform inconsistently; only the fixed-token variant works reliably.
Question: Why does the model fail to self-terminate properly? Could an adaptive controller or entropy-based stopping improve this?

3. The paper compares against open models but omits newer ones like LLaVA-OneVision or InternVL 3.5, which are directly relevant.
Question: How does LVR compare to these recent omni-modal reasoning models?

4. Since LVR heavily relies on Visual-CoT training data, there’s risk it memorizes dataset patterns instead of general reasoning.
Question: Have they tested on out-of-distribution datasets to measure transfer robustness?

### Questions
See questions above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents Latent Visual Reasoning (LVR), a novel method that allows multimodal models to reason directly with visual embeddings. Unlike existing models that reason with text or use external tools, LVR operates autoregressively within the visual space itself. The model learns to reconstruct important visual information related to a query by generating a sequence of latent states. This capability is developed in a two-stage training process, which starts with supervised fine-tuning and is later refined using reinforcement learning based on a modified GRPO algorithm. Experiments confirm the strength of this approach, showing it surpasses other methods on demanding visual perception benchmarks like MMVP and V*.

### Strengths
The primary strength of this paper lies in its introduction of Latent Visual Reasoning (LVR), a fundamentally new and well-motivated paradigm for multimodal reasoning. By enabling the model to reason directly within the visual embedding space, it provides an elegant alternative to purely text-based or tool-dependent approaches.  Furthermore, the methodology is well-conceived, featuring a logical two-stage training pipeline and a clever technical adaptation of reinforcement learning (GRPO_latent) to handle reasoning in a continuous latent space.

### Weaknesses
While the method excels at visual perception, its applicability to complex visual reasoning tasks remains unclear. For instance, in geometry problems where auxiliary lines are crucial, the practical value of the model's generated visual tokens is questionable and requires further validation. The work would be significantly strengthened by evaluations on more challenging reasoning benchmarks, such as MathVista, and by including more in-depth visual analyses that clearly illustrate the impact of these visual tokens on the reasoning process and its results.

### Questions
1. The current evaluation focuses almost exclusively on perception-based tasks. However, this leaves its performance on more complex visual reasoning tasks as an open question. Could the authors comment on the applicability of LVR to benchmarks like MathVista, which require more symbolic or multi-step reasoning? It is important to understand whether LVR's core mechanism of reconstructing visual tokens is sufficient for tasks that depend less on recognizing pixels and more on abstract logical deduction.

2. If the model focuses on an incorrect Region of Interest in an early step, does it have any mechanism to detect and correct this error in subsequent steps?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces Latent Visual Reasoning (LVR) — a new paradigm for multimodal reasoning that allows autoregressive reasoning directly in the visual embedding (latent) space, rather than relying solely on textual Chain-of-Thought (CoT) reasoning or external visual tools. Experiments on visual reasoning and perception-intensive benchmarks (e.g., V*, MMVP, BLINK) show significant improvements over both “Think about Images” (text-only CoT) and “Think with Images” (external tool-based) baselines. LVR achieves up to 71.7% on MMVP, outperforming Qwen2.5-VL and PixelReasoner.

### Strengths
1. The concept of latent visual reasoning is novel and well-motivated.
2. The methodology is sound: clear architectural modifications, well-defined training objectives, and careful integration of latent-space reconstruction.
3. Experiments are comprehensive across multiple benchmarks and baselines (Qwen2.5-VL, PAPO, Vision-R1, PixelReasoner).
4. The work offers a scalable alternative to tool-based multimodal reasoning, reducing dependency on external APIs and handcrafted operations.

### Weaknesses
1. While conceptually strong, the paper provides little visualization or introspection of the latent reasoning process (e.g., what latent reconstructions “look like”). Future work could include latent trajectory visualizations or probing to show what is being “reasoned” in the latent space.
2. The reliance on a fixed latent reasoning length may restrict flexibility and generalization to variable reasoning complexity.

### Questions
1. Can you provide qualitative examples (e.g., t-SNE plots or attention maps) showing what information the latent visual reasoning reconstructs?
2. Could alternative stopping mechanisms (e.g., energy-based thresholds or learned confidence) improve variable-length reasoning stability?
3. Does LVR introduce latency at inference time due to latent-space propagation?

### Soundness
4

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
This paper proposes latent visual reasoning for MLLMs, which supervises an autoregressive process on latent states with reconstruction of ground-truth visual semantic embeddings, resulting better results than previous "think about images" and "think with images" methods. The training stages include SFT with next token prediction combined with reconstruction, and a GRPO stage compatible with the generated latent sequences. The paper further investigates different decoding strategies and concludes that the method works best with a fixed number of reasoning steps.

### Strengths
1. The framework is novel by directly using MSE on hidden states and incorporating GRPO with modifications.
2. The paper is well-organized and clearly presented.
3. The experimental results are achieved by carefully controlling the conditions, making the performance advantage convincing.

### Weaknesses
Although the training stages incorporates RL, it can not avoid using critical region annotation, as GRPO yields only marginal improvement, while the MSE supervision on latents brings clear advantage over vanilla SFT. The latents are supervised only by the region semantics from the input image, which seems more like learning a visual grounding. Thus, the novelty advantage might be limited compared to other works proposing latent visual reasoning.

### Questions
1. How can the RL process improve the latent generation process? As I see in (5), the latents seem to be used as inputs with no gradients.
2. I find the "LVR with heads" confusing in the ablation section. Is it just some additional mapping parameters? Why is the performance worse with more parameters involved?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Latent Visual Reasoning (LVR), a novel paradigm that enables multimodal large language models to perform reasoning directly within the visual embedding space, rather than relying solely on textual chain-of-thought reasoning. LVR allows the model to interleave latent visual reasoning with text generation, jointly processing both modalities in a shared semantic space. The authors propose a two-stage training pipeline combining supervised finetuning for visual reconstruction and reinforcement learning with an adapted GRPO algorithm to refine reasoning dynamics. Experimental results show that LVR significantly improves fine-grained visual perception, spatial understanding, and robustness on benchmarks such as MMVP and V*, outperforming previous “Think about” and “Think with” image approaches.

### Strengths
1. The paper is very well-written and logically structured, making the core ideas easy to understand. The figures are clear and effectively illustrate the main concepts.

2. The motivation is straightforward and reasonable.

### Weaknesses
1. The paper mostly extends the idea of latent reasoning from text-only LLMs to multimodal models. While this is a reasonable step forward, it feels more like a technical adaptation than a fundamentally new idea. The authors could do a better job explaining why reasoning in the visual latent space offers a distinct advantage rather than just being a straightforward extension of existing methods.

2. The approach relies heavily on Visual CoT data and also requires identifying regions of interest (ROIs) for training, which makes it conceptually close to Visual CoT itself. However, there’s no head-to-head comparison with Visual CoT models in terms of both performance and inference cost. Without this, it’s hard to judge how much the proposed method actually improves over existing baselines.

3. Because the reasoning happens in the latent space, it’s not clear what the model is actually doing during the latent visual reasoning phase. The paper doesn’t provide any qualitative examples or visualizations of these hidden processes, which makes the claimed reasoning mechanism somewhat opaque.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
