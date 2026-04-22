# Think Then Embed: Generative Context Improves Multimodal Embedding

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
There is a growing interest in Universal Multimodal Embeddings (UME), where models are required to generate task-specific representations. While recent studies show that Multimodal Large Language Models (MLLMs) perform well on such tasks, they treat MLLMs solely as encoders, overlooking their generative capacity. However, such an encoding paradigm becomes less effective as instructions become more complex and require compositional reasoning. Inspired by the proven effectiveness of chain-of-thought reasoning, we propose a general Think-Then-Embed (TTE) framework for UME, composed of a reasoner and an embedder. The reasoner MLLM first generates reasoning traces that explain complex queries, followed by an embedder that produces representations conditioned on both the original query and the intermediate reasoning. This explicit reasoning step enables more nuanced understanding of complex multimodal instructions. Our contributions are threefold. First, by leveraging a powerful MLLM reasoner, we achieve state-of-the-art performance on the MMEB-V2 benchmark, surpassing proprietary models trained on massive in-house datasets. Second, to reduce the dependency on large MLLM reasoners, we finetune a smaller MLLM reasoner using high-quality embedding-centric reasoning traces, achieving the best performance among open-source models with a 7% absolute gain over recently proposed models. Third, we investigate strategies for integrating the reasoner and embedder into a unified model for improved efficiency without sacrificing performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates using multimodal CoT traces to improve multimodal embedding for retrieval tasks. The paper explores diverse training settings, including using separate or unified models to combine CoT and embedding, and the design choices of a good embedding head to extract information from CoT latents. Through comprehensive experiments, the paper demonstrates the effectiveness of CoT for generating multmodal embedding compared to both decoder-encoder and multimodal llm based methods.

### Strengths
* The paper shows that using CoT traces improves multimodal retrieval without using additional data.

* The paper comprehensively examines different settings to incoporate the CoT traces, including separate reasoning models, distilled reasoning models, and unified reasoning-embedding models. This highlights the flexibility of apply CoT for better multimodal embedding.

* The ablations clearly demonstrate the effectiveness of incoporating CoT to improve multimodal embeddings by comparing with baselines without CoT.

### Weaknesses
* The paper's method is not specifically tied to multimodal, or vision-language. The same approach can be applied to text-only retrieval. It would be great to clarify on how multimodal is important here, or demonstrate the method on text-only datasets.

* The paper only used Qwen-VL series as the base models, but the method should be generalizable to most MLLMs. It would be great to show results on other base models.

* While the authors hyphosize that the last hidden states might not be optimal, it is not properly justified in the paper. The improved results from Q-Former or self-initialized could be from the extra trainable parameters. A deeper analysis, like comparing the performance of each layer, would strengthen the design choices of the embedder head.

### Questions
* It would be interesting to see whether using traces from more powerful reasoning models, like GPT-o3, can further improve the embedder performance.

* Does the performance scale with the size of the MLLM? And does a better MLLM, measured by standard benchmarks like MMMU, results in better embedding?

### Soundness
4

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
This paper introduces a Universal Multimodal Embedding framework, Think-Then-Embed(TTE), which first generates reasoning before producing embeddings. TTE exploits the generative reasoning ability of MLLMs to enhance embedding quality. Specifically, a powerful MLLM serves as a Teacher Reasoner to produce high-quality CoT traces (embedding-centric reasoning in this paper), guiding a smaller embedder to learn more discriminative representations. To reduce reliance on large models, the authors further distill a compact MLLM reasoner from the teacher and train a unified model that functions simultaneously as a reasoner and an embedder. Experiments show that TTE achieves state-of-the-art results on MMEB-V1 and MMEB-V2 benchmarks. Ablation studies confirm the effectiveness and stability of ECR  for complex instruction embedding.

### Strengths
1. The paper provides a clear motivation and detailed explanation of the proposed method, with well-defined formulations for both the task and the training process.
2. Based on the intuition for retrieval tasks that require reasoning, as well as the strong results of TTE_t, the proposed Think-Then-Embed (TTE) framework appears highly effective.

### Weaknesses
1. The analysis of ECR stability is only described qualitatively and lacks experimental support.
2. The MMEB-V2 benchmark seems to be missing some evaluation data. 

Please refer to the Questions section for details.

Things to improve the paper that did not impact the score:
1. In Table 1, there are several mistakes in the use of bold and underline to highlight the first and second-best results.

### Questions
1. From the comparison between TTE_t and TTE_s, as well as the ablation results before and after fine-tuning TTE_s, it seems that the quality of ECR plays a crucial role in the embedder’s performance. However, the paper claims that the embedder is stable even with imperfect ECR, which appears somewhat contradictory. Could the authors provide a more detailed explanation or analysis to clarify this point?
2. The MMEB-V2 benchmark results do not include TTE_u. Since the results on MMEB-V1 suggest that TTE_u performs similarly to TTE_s, including such data would make the paper more complete and convincing.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Think-Then-Embed (TTE), a novel framework for multimodal reasoning that introduces an explicit generative embedding step between reasoning and representation. Instead of directly encoding multimodal inputs (e.g., image-text pairs) into a single joint embedding, the method first conducts latent reasoning generation through an autoregressive module and then projects the reasoning trace into a compact embedding space via a learned encoder. This “reasoning-before-embedding” paradigm aims to enhance reasoning fidelity and interpretability while maintaining embedding efficiency. Experiments are conducted on multiple multimodal benchmarks (e.g., ScienceQA, OK-VQA, and MM-CoT-style datasets). Results show consistent improvements over baseline embedding methods such as CLIP, BLIP-2, and reasoning-augmented contrastive approaches. The authors claim that TTE achieves comparable or better performance with smaller embedding dimensionality and provides interpretable intermediate reasoning traces.

### Strengths
- The writing is mostly clear and logically structured; the introduction and motivation are strong. Figures and pipeline diagrams are helpful in explaining the “think → embed” flow.
- The overall framework is technically coherent and easy to reproduce: the authors provide clear training and inference procedures.
- The formulation of generative embeddings based on explicit reasoning traces is intuitive and connects language modeling with representation learning in a principled way.

### Weaknesses
- Based on Tables 1 and 2, $\text{TTE} _\text{T}$ performs significantly better than the other two paradigms. However, it is unclear whether this advantage stems from the use of a stronger teacher model. Did the authors conduct any ablation studies using different teacher models to isolate this effect?
- If reasoning traces are generated by a large pretrained LLM (e.g., GPT-4V or Qwen2-VL), it is unclear whether they contain test-relevant information from pretraining corpora, which may bias the results.

### Questions
- The “contrastive generative embedding” objective is vaguely described. It is unclear whether gradients flow through the reasoning generator (is it frozen or partially fine-tuned?).
- The distinction between “reasoning trace” and “explanation text” is blurred—are they generated with the same prompt template or via different decoding strategies?

### Soundness
3

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
4

### Summary
This paper introduces Think-Then-Embed (TTE), a universal multimodal embedding framework that uses a reasoner to "think" before generating embeddings with an embedder. Leveraging a multimodal large language model (MLLM) as the reasoner significantly enhances the performance of smaller embedders, showing that chain-of-thought (CoT) reasoning benefits multimodal representation learning. To boost efficiency, we distill a compact reasoner from the large model’s reasoning traces, allowing both the reasoner and embedder to share the same backbone for improved capacity and speed. Additionally, a pluggable embedding head on the reasoner enables single-pass embedding generation, reducing parameters and further increasing efficiency. Extensive experiments on MMEB benchmarks demonstrate that TTE outperforms existing methods without extra data, confirming its effectiveness and robustness.

### Strengths
The method is simple yet effective. Involving cot process in retrieval shows significant performance gains on benchmarks. The author designed various experiments and methods to demonstrate this, and the experimental results performed well on the MMEB benchmarks.

### Weaknesses
* Some terms should be unified: ‘Fig.’ in line 62 and ‘Figure 2’ in line 177 should be consistent.

* I recommend that the author distinguish between MLLM and LLM. For example, ‘Qwen2.5-VL-72B’ in line 84 and ‘Qwen2.5-72B’ in line 
212 are confusing. These names should be unified in both the introduction and the methods section.

* Figure 5 looks strange; the results on different benchmarks do not show temporal relationships clearly. A bar chart or radar chart would be more appropriate.

* The grounding task score does not show an advantage compared to LLaVA-OV-based models in Table 1, even though Qwen2VL is claimed to have better grounding ability.

### Questions
* The grounding task score does not show an advantage compared to LLaVA-OV-based models in Table 1, even though Qwen2VL is claimed to have better grounding ability.
* Why does the embedder use Qwen2VL-7B? Is it superior to Qwen2.5-VL-7B?

### Soundness
3

### Presentation
3

### Contribution
3
