# OneFlowSeq: Achieving One-Step Generation for Diffusion Language Models via Lightweight Distillation

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Autoregressive models dominate Seq2Seq generation but suffer from slow, error-prone token-by-token decoding. Diffusion language models (DLMs) enable parallel refinement and global coherence, yet their iterative denoising requires hundreds of steps, limiting practicality. We propose **OneFlowSeq**, a novel framework that distills a powerful multi-step diffusion teacher (LLaDA-8B-Instruct) into a one-step generator via MeanFlow-based supervision and parameter-efficient prompt tuning. Our OneFlowSeq introduces a Jacobian-vector product signal that provides richer guidance than conventional distillation, allowing the student to not only match the 128-step teacher in terms of one-step generation quality. Experiments on paraphrasing, text simplification, and question generation benchmarks show that OneFlowSeq achieves state-of-the-art performance, while reducing trainable parameters by 1600$\times$ and delivering inference speeds orders of magnitude faster than both autoregressive and multi-step diffusion baselines. This work establishes one-step diffusion as a practical and scalable paradigm for Seq2Seq generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
TThis paper proposes OneFlowSeq, a novel distillation framework for diffusion language models that resolves the speed-quality trade-off. It distills a multi-step teacher model into a lightweight, parameter-efficient student by using MeanFlow theory and a Jacobian-vector product signal for superior guidance. This enables the student to match the teacher's quality in a single generation step, achieving state-of-the-art results with a 1600x reduction in trainable parameters and a 160x inference speedup, facilitating scalable deployment.

### Strengths
1. The proposed method achieves state-of-the-art performance in benchmark datasets such as QQP, Wiki-Auto, and Quasar-T for tasks like text simplification, question generation, and paraphrase detection. 

2. It significantly reduces the number of parameters required compared to previous methods while maintaining high inference speed.

### Weaknesses
1.  Novelty: The paper's idea is very straightforward, amounting to a simple application of MeanFlow, which does not meet the standard for ICLR.

2.  Experimental Results: The paper only evaluates a few simple text generation tasks and fails to include datasets like MMLU, which the base LLaDA model was evaluated on and are necessary to demonstrate the capabilities of large language models.

3.  Experimental Baselines: Since the paper chose a soft-prompting approach, other parameter-efficient fine-tuning (PEFT) methods, such as LoRA, should have been evaluated as baselines.

### Questions
NA.

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
5

### Summary
This paper proposes OneFlowSeq, a distillation framework for turning a multi-step diffusion language model (LLaDA-8B-Instruct) into a one-step generator.
The method combines MeanFlow-based distillation with an additional Jacobian-vector product (JVP) supervision term to approximate the teacher’s flow direction, while keeping the backbone frozen and training a lightweight soft prompt (~5M parameters).
Experiments on paraphrasing (QQP), text simplification (WikiAuto), and question generation (Quasar-T) claim comparable quality to 128-step diffusion generation and large inference speedups.

### Strengths
Tackles an important issue: making diffusion-based LMs practical by reducing multi-step inference to one step.
Elegant integration of MeanFlow and JVP supervision; clearly described training process.

### Weaknesses
* Unfair and inconsistent baselines:
- Teacher: LLaDA-8B-Instruct (8B parameters, pretrained on trillions of tokens).
- Baselines: GPT-2 (1.5B), DiffuSeq (trained from scratch on small datasets), and DLM-One (re-implemented).
- These baselines are not comparable in capacity and not pretrained on the similar size datasets, giving OneFlowSeq an unfair advantage.


* Weak tasks and evaluation design:
- Evaluations (QQP, WikiAuto, Quasar-T) involve short and easy sequences (1–2 sentences).
- Missing evaluations on complex, long-form tasks like summarization (CNN/DailyMail, XSum) or reasoning datasets.
- Results cannot demonstrate scalability or compositional generalization of the proposed method.

* Missing strong baselines and ablations:
- No comparison with Fast-DLLM, Block Diffusion, Consistency Models, or Rectified Flow Transformers.

* Questionable efficiency and generality:
- Reported 160× speedup is achieved with batch size 256 for OneFlowSeq vs. batch size 1 for AR models - not a fair per-sample measure.
- Inference cost remains dominated by the 8B teacher backbone.
- No results for real-time latency, FLOPs, or wall-clock performance on similar setting

* Limited insight into JVP supervision:
- The JVP term is presented as novel but is widely used in flow-matching and consistency distillation.
- Its role in improving one-step alignment is not clearly isolated; improvements might stem from regularization rather than Jacobian matching.

### Questions
How are the baselines pretrained as well? Do they have the same scale as LLaDA-8B?
Can you provide results on longer, more complex tasks such as summarization or story generation?
What is the true per-sample latency under equal hardware and batch size settings?
Why were recent consistency or flow-based distillation models not included as baselines?

### Soundness
3

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
4

### Summary
This paper proposes a method for distilling a text diffusion model into a model capable of predicting all tokens in parallel, using a second-order training objective.  The student model learns a parameter-efficient soft prompt model which adapts a copy of the teacher model for this objective.  Empirical results show that this method is effective for low-entropy conditional text generation, with much improved generation speed.

### Strengths
This is a surprisingly effective way to learn a model which does efficient non-autoregressive text generation.  Relevant ablations are also done.

### Weaknesses
The model is based largely on the MeanFlow model, so the novel contribution is mostly in how to exploit this insight in an effective distilled model, and in the empirical results.  I found it impossible to understand the technical details without already understanding the MeanFlow model.

The model is only evaluated in low-entropy conditional generation tasks.  This makes sense, since one-shot text generation is presumably impossible in high-entropy tasks because of the multi-modal nature of the output distribution.  But they never evaluate or discuss this limitation.  Some design choices, such as lines 259-260 "This intermediate decoding can be simplified by directly feeding the continuous embeddings corresponding to z_t_i into the model", only make sense for low-entropy tasks.

The presentation of the model could be better.  The student model is a PEFT version of the teacher model (if I understand correctly), but they talk about the student model as if it consists only of the adaptation parameters without hardly mentioning that the student also includes a huge model with frozen parameters.  This is especially confusing because the teacher is also the same model also with frozen parameters.

### Questions
For the student model, why use prompt tuning?  Have you tried more effective/efficient PEFT methods like LoRA?

### Soundness
4

### Presentation
3

### Contribution
3
