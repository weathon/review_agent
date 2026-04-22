# Transfer between Modalities with MetaQueries

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Unified multimodal models aim to integrate understanding (text output) and generation (pixel output), but aligning these different modalities within a single architecture often demands complex training recipes and careful data balancing. We introduce MetaQueries, a set of learnable queries that act as an efficient interface between autoregressive multimodal LLMs (MLLMs) and diffusion models. MetaQueries connects the MLLM's latents to the diffusion decoder, enabling knowledge-augmented image generation by leveraging the MLLM's deep understanding and reasoning capabilities. Our method simplifies training, requiring only paired image-caption data and standard diffusion objectives. Notably, this transfer is effective even when the MLLM backbone remains frozen, thereby preserving its state-of-the-art multimodal understanding capabilities while achieving strong generative performance. Additionally, our method is flexible and can be easily instruction-tuned for advanced applications such as image editing and subject-driven generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents an interesting approach that freezes the LLM backbone solely for understanding tasks, while delegating generation tasks to image-generative models. This design simplifies data balancing and removes the need for complex training procedures.

### Strengths
1.The paper introduces MetaQuery, a framework that leverages the capabilities of frozen MLLMs.

2.The proposed method achieves state-of-the-art performance in both multimodal understanding and image generation tasks.

3.The approach also shows potential for extension to other image-related applications, such as image editing, through appropriate fine-tuning.

### Weaknesses
1. The concept does not appear entirely novel, as similar ideas have been explored in prior works such as Seed-X, MetaMorph, and Next-GPT. Could the authors clarify the main differences between their approach and these methods? 

2. It would strengthen the paper if the authors could include additional experiments on subtasks that more directly assess the benefits of a unified model, such as interleaved image-text generation or other multimodal interaction tasks.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of creating unified multimodal models that excel at both understanding (text output) and generation (pixel output) without performance degradation. The authors propose **MetaQueries**, a set of learnable tokens that act as an efficient interface between a **frozen** autoregressive multimodal LLM (MLLM) and a diffusion model.

### Strengths
1. The core idea of MetaQueries as a "plug-and-play" interface is a significant strength, promoting modularity in a field dominated by monolithic models.
2. The paper's most impressive contribution is the *proof* (via Figure 7 and Table 4) that MetaQueries are not just a simple conditioning mechanism but a functional bridge for transferring high-level MLLM capabilities (world knowledge, reasoning) to the generation process. SOTA on the WISE benchmark confirms this.
3. The "Tune vs. Freeze MLLM" and "MetaQueries vs. Last-Layer Embedding" ablations are critical, well-executed, and strongly support the paper's central claims.

### Weaknesses
1. The paper honestly notes (Section 5.1) that its method lags behind autoregressive (AR) visual-token models like Janus-Pro on prompt-alignment benchmarks (GenEval, DPG). While the authors provide a qualitative defense (better visual quality, Appendix E), this remains a quantitative gap. It would be beneficial to discuss if this is a fundamental limitation of the diffusion-decoder approach or if it could be closed with more data/tuning.
2. In Section 3.1, the paper states, "we continue to use causal masking for the entire sequence". This is slightly ambiguous. A clearer explanation of the exact attention mask between the `[Multimodal Input]` and `[MetaQueries]` would be helpful. For instance, do the queries attend to the input, but the input cannot attend to the queries?
3. The trainable "connector" is a key component, but it's not deeply analyzed. Appendix A.2 explores two designs, but the impact of connector depth/complexity on performance vs. efficiency is not fully explored. How much of the "alignment" is handled by the queries versus this 24-layer transformer?

### Questions
1. Following up on Weakness #1: Do you believe the prompt-alignment gap (vs. AR models like Janus-Pro) is a fundamental trade-off for the superior visual quality of diffusion, or could this gap be closed, perhaps by scaling the 25M image-caption pre-training data or further tuning the connector?
2. Following up on Weakness #2: Could you please clarify the exact attention mechanism? Given a sequence `[Input Tokens, MetaQuery Tokens]`, what is the attention mask? Is it a standard causal mask over the entire concatenated sequence?
3. In Table 2, the "Freeze MLLM" setting achieves a *better* (lower) FID score than "Tune MLLM" (e.g., 6.06 vs 6.28 when training DiT). This is counter-intuitive, as one might expect tuning to help. Do you have a hypothesis for why *not* tuning the MLLM leads to slightly better visual quality?
4. The instruction-tuning data pipeline (Section 4) is very clever. How sensitive is the model's performance to the MLLM-generated instruction? For example, did you find that variations in the system prompt (Appendix B) led to significant differences in the model's final editing/subject-driven capabilities?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed to unify multimodal understanding and generation by combining bespoke pretrained models. Compared to existing unified LLMs, this work adopted a trainable connector to act as a bridge between pre-trained MLLMs and the diffusion decoder. Learnable queries as input to fuse vision-language information from the pre-trained MLLMs as input conditions for diffusion decoding. Extensive experimental results demonstrated the effectiveness of the proposed approach.

### Strengths
1. The paper is well-organized, and the figures are well-prepared.
2. This approach achieves state-of-the-art results on the existing multimodal understanding and generation benchmarks.
3. The reasoning and knowledge-augmented generation looks interesting.

### Weaknesses
1. My biggest concern is the size of the connector, which is extremely large when adopting Qwen2.5-VL 3B and 7B as the base MLLMs. The connector size may be larger than the diffusion decoder, which makes the claim that this transfer is effective even when the MLLM backbone remains frozen less convincing.  

2. Given the above point, the paradigm of MetaMorph may be more effective. This raises another concern: what if we take the MLLM's output tokens as direct input for the connector? Would it be more effective, or would there be fewer learnable parameters required to make this transfer? Additional experimental results are required.

### Questions
See weaknesses. I will adjust the rating according to the authors' response.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose MetaQueries, a simple and light module that bridges the frozen LLM and a diffusion model, to boost the multimodal understanding and generation. MetaQueries is a series of learnable tokens that fed into the LLM to learn latent conditions to align to the conditional space of diffusion models. While the authors conduct experiments over multiple benchmarks, I think the contribution mostly comes from the engineering, rather than the technical novelty.

### Strengths
- text is easy to follow
- method is simple and efficient, by inserting learnable tokens, the training process only needs to fine-tune diffusion model.

### Weaknesses
- limited novelty: the main component, MetaQueries, is essentially a form of learnable prompts / queries, similar to prior adapters such as Q-Former
- lack of theoretical analysis: no deeper analysis on why or when frozen MLLM features can serve as effective generative conditions, nor exploration of failure cases or transfer limitations

### Questions
please refer to the weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
