# Language-Instructed Vision Embeddings for Controllable and Generalizable Perception

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Vision foundation models are typically trained as static feature extractors, forcing the burden of task adaptation onto large downstream models. We propose a different paradigm: instead of solely feeding visual features into language, we use language itself to dynamically guide the vision encoder. Our method, Language-Instructed Vision Embeddings (LIVE), leverages language as high-level guidance to produce task-centric embeddings at inference time—without requiring task-specific retraining. This enables the encoder to focus attention on contextually relevant aspects of the input, yielding more controllable and generalizable representations. Empirically, LIVE reduces visual hallucinations (+34 points on MMVP), outperforms vision–language models with orders of magnitude more parameters on visual question answering, and generalizes to unseen instructions and  tasks---offering a direct path toward adaptive, instruction-driven visual intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Language-Instructed Vision Embeddings (LIVE), a novel paradigm that uses language instructions to dynamically guide a vision encoder during inference, enabling it to produce task-specific visual embeddings without retraining. To achieve this, LIVE reuses the text encoder from SigLIP and injects textual queries directly into the vision part, allowing it to focus on instruction-relevant image regions. After training on  around 16.4 million images-query-answer synthesized by Gemini, LIVE reduces visual hallucinations by 34 points on the MMVP benchmark and outperforms much larger vision–language models (e.g., LLaVA, InstructBLIP) on GQA, demonstrating its effectiveness.

### Strengths
1. The idea of this paper is quite novel. By training a language-instructed visual embedding model, it addresses visual tasks more efficiently with a model of reduced parameter count, eliminating the need for task-specific training. Furthermore, it mitigates the issue of large language models (LLMs) dominating visual inference in existing multimodal large language models (MLLMs), which is a commendable contribution.
2. The methodology is both straightforward and effective. Moreover, the synthesized triplet dataset holds significant potential for enhancing CLIP-style pretraining for the research community.
3. The experimental results are promising, demonstrating that a considerably smaller embedding model surpasses existing MLLMs across a wide range of tasks. Additionally, the attention visualizations substantiate the efficacy of LIVE at the representational level, which is highly appreciated.

### Weaknesses
1. Generating large-scale vision-related data based on Gemini 2.0 may incur substantial costs. It would be beneficial if the authors could provide relevant details regarding the expenses associated with data synthesis and model training.
2. The examples presented in the paper are confined to relatively simple queries. It remains an open question whether the proposed approach maintains its efficacy when applied to more complex visual problems and intricate queries (e.g., visual reasoning tasks).

### Questions
1. Can the trained visual encoder be used for MLLM training? Will it make current MLLM (.e.g. LLaVA) better or worse?
2. See Weakness above

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
This paper aims to make vision encoders, such as CLIP, to extract language-instructed vision embeddings to produce query-specific visual features for downstream applications. To address this, the authors let Gemini-2.0-Flash annotate over 16M image-question-answer pairs, and then fine-tune the original vision encoder with SigLIP's loss. Experiments under various evaluation protocols demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper is well-written and easy to follow.
2. The motivation is clear, reasonable, and meaningful.
3. The proposed method is quite effective.

### Weaknesses
1. The improvement may mainly come from the data instead of the method. To investigate this, the following experiments are highly recommended:
   - Train LIVE on the classic caption dataset (using "describe this image" as the prompt) or the original ImageNet dataset (using "what is the category of this image" as the prompt).
   - Train other methods like Flamingo or BLIP-2 using the proposed data.

2. How this encoder incorporates with typical MLLMs remains unclear. It is encouraged to replace the original CLIP-336 in LLaVA with LIVE.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This work proposes a language-instructed vision embedding (LIVE)  method, aiming at extract more controllable and generalizable visual representation upon the language guidance. Specifically, the vision encoder takes the embedding of the textual instruction q and the image as input and compute the instructive visual embedding. Extensive experimental results compared with a series of the leading methods show the improved performance of the proposed method.

### Strengths
* The proposed method proposes a new perspective of learning effective and adaptive visual representation, which is interesting. 

* This work is well-motivated. 

* The introduction is well-written.

### Weaknesses
* L156~160 demonstrate that the vision encoder always requires the query as the instruction to guide the visual representation learning. Although the concept is novel and could make sense, this strategy inevitably brings the bypass for the multimodal learning, e.g., model can   rely on the query, rather than the image itself to learn the representation for the semantic matching. Intuitively, this might be one of the cause for the performance boost. 

* During training, the answer is adopted. Yet, during inference, it seems that the answer may not be adopted. Is this mismatch intentionally devised and what might be the rational behind this?

 * This work provides extensive of performance comparison but lacks thorough ablation and model discussions in the manuscript, such as the discussion of each type of the input to the text/img encoders. 

* Fig.6 shows that the query steer the visual attention. Yet, the more effective the steering it is, the more important role the query plays for the representation learning, instead of visual signals.

### Questions
* What is the setting of the inference pipeline? What are the specific inputs of the visual encoder and the text encoder? More detailed illustrations are expected. 

* The proposed method highlights more on the semantics of the visual embedding, how's the sensitivity of the proposed method to the visual detail variation? 

* Why the proposed work is termed as "generalizable perception"? How does the proposed method relate and contribute to the visual perception and low-level patterns?

### Soundness
2

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
The paper introduces a new paradigm in vision–language modelling: rather than merely feeding image features into a large language model, they reverse this flow by using language instructions to guide the vision encoder. The proposed method, called Language-Instructed Vision Embeddings (LIVE), uses natural-language directives at inference time to steer the vision encoder into producing task-centric visual embeddings without any task-specific retraining.
In practice, this means the model can dynamically focus on aspects of an image relevant to the given instruction, resulting in more controllable and generalizable representations. Empirical results show substantial reinforcement: LIVE reduces visual “hallucinations” by a large margin and outperforms much larger vision–language models on visual question-answering tasks, even when faced with unseen instructions and tasks.

Contributions: 
- Introduces the idea of language-driven visual encoding: language is not just describing or querying, but actively guiding how the visual encoder processes the image.
- Provides a framework where the vision encoder produces embeddings that are instruction-aware, enabling strong controllability (you decide what to focus on) and generalizability (works for new instructions/tasks).
- Demonstrates empirically that this paradigm can deliver strong performance gains: better alignment of visual features with linguistic task goals, fewer hallucinations, and robustness to unseen tasks/instructions.

### Strengths
The paper presents a well-motivated and timely contribution in the field of language-guided visual representation learning. The proposed LIVE (Language-Instructed Visual Encoder) introduces an instruction-based training paradigm that allows natural language to directly modulate the visual encoding process. This idea aligns closely with the current trend of multimodal alignment and reasoning, yet it explores an under-investigated direction—embedding-level linguistic control—rather than the more common dual-encoder or caption-generation settings.

Originality
The originality of LIVE lies not merely in architectural novelty, but in its reformulation of multimodal learning as instruction-conditioned representation alignment. Instead of aligning paired modalities post hoc (as in CLIP), the model internalizes linguistic guidance into the feature extraction process. This perspective opens new possibilities for controllable and interpretable vision encoders. While the high-level idea is inspired by prior “visual instruction tuning” work, the paper provides a conceptual synthesis that generalizes those methods into a unified embedding-learning view.

Quality
The paper provides a coherent training objective, a clear architecture description, and a reasonable learning pipeline involving LLM-generated image–instruction–answer triplets. The experimental section covers multiple benchmarks, showing consistent improvements over baseline vision encoders. The methodology is easy to follow, and the ablation studies—though limited—support the claim that linguistic instruction improves cross-task generalization. In terms of engineering quality, the model is computationally efficient and can be integrated into existing multimodal pipelines.

Clarity
The manuscript is generally well-written and clearly organized. Figures and tables effectively convey the core architecture and comparative results. The motivation is explicitly stated, and the connection to prior work is well articulated. 

Significance
LIVE addresses a fundamental question in representation learning: Can visual features be directly instructed through language? The proposed framework provides early empirical evidence that such an approach is not only feasible but also beneficial. This work could inspire a new class of “instruction-aware” encoders applicable to downstream tasks such as retrieval, visual reasoning, and grounding. Its conceptual significance lies in bridging the gap between vision-language alignment and language-conditioned representation control, which is of high relevance to the ICLR community’s focus on generalizable and interpretable AI systems.

### Weaknesses
1.	Originality is under-argued vs. closely related paradigms.
The paper positions LIVE as “language-instructed” encoding inside the vision tower, but the boundary from prior lines of work (caption-conditioned encoder modulation, cross-attention/adapters, visual instruction tuning) isn’t made watertight. 

2. Potential data contamination between training and evaluation remains insufficiently addressed.
Although the paper specifies that the 16.4 M (image, instruction, answer) triplets are generated with Gemini 2.0 Flash and even provides example prompts, it does not report any train–test decontamination analysis. Given that Gemini-generated supervision and evaluation “oracle” labels (marked †) are both derived from similar prompting pipelines, there is a tangible risk of semantic overlap or leakage into benchmarks. 

3.	Use of a model-generated “oracle” undermines evaluation rigor.
Several benchmarks marked with “†” use Gemini-produced answers and then measure how close LIVE gets to that oracle. This is not a neutral ground truth: it bakes the teacher’s biases into the metric and can inflate apparent progress. Replace “oracle” labels with human annotations or established datasets; if teacher labels must be used, include teacher–student agreement vs. human gold, plus audits for leading prompts and spurious shortcuts.

4.	Fairness of headline gains is only partially established (scale mismatches remain).
The reported +34-point MMVP and large GQA gains are impressive. However, the overall training scale differs substantially—LIVE uses 16.4 M LLM-generated triplets trained on 256 TPUv3 cores—which makes it difficult to disentangle architectural improvements from data or compute advantages. Re-evaluating baselines under comparable compute/data budgets or reporting normalized gains would clarify how much of the improvement stems from the proposed mechanism itself.

5.	Ablations only partially support the claimed mechanism.
The paper provides some evidence that language modulates visual representations—such as comparisons without instructions, attention visualizations, and encoder-size studies—but these analyses remain high-level. More controlled ablations (e.g., varying where or how language is injected, or replacing instructions with neutral text) could strengthen the causal link between “instruction conditioning” and the observed performance gains. As written, it remains difficult to fully disentangle the benefits of language-guided encoding from those of larger data or improved contrastive training.

### Questions
1.	Clarifying Conceptual Novelty
	•	What exactly is unique about LIVE’s modulation pathway (e.g., position of language injection, training objective) that cannot be reduced to prior adapter or cross-attention methods?

2. Data Contamination (Weakness #2)
	•	Did you perform any explicit decontamination between the Gemini-generated triplets and the evaluation benchmarks?
	•	Would you consider running a controlled experiment on a held-out, human-annotated benchmark to quantify the possible inflation due to data leakage?

3.	Use of Model-Generated “Oracle” Labels
	•	For the “†” benchmarks where Gemini provides the reference answers, can you clarify the rationale for treating Gemini outputs as ground truth?
	•	How correlated are Gemini’s responses with human-labeled gold data (e.g., on a small manually annotated subset)?

4.	Scale and Fairness of Comparisons
	•	Can you provide more detail on the total compute budget and data volume of LIVE versus each baseline?
	•	Were any baselines retrained under identical compute/data conditions, or were all numbers taken from prior publications?

### Soundness
2

### Presentation
3

### Contribution
3
