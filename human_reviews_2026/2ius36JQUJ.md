# UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
The remarkable success of multimodal large language models (MLLMs) has driven advances in multimodal embeddings, yet existing models remain inherently discriminative, limiting their ability to benefit from reasoning-driven generation paradigm. In this work, we pioneer the exploration of reasoning-driven generative embeddings, unifying embedding tasks within a generative paradigm. We propose UME-R1, a universal multimodal embedding framework consisting of a two-stage training strategy: a cold-start supervised fine-tuning equips the model with reasoning capabilities and enables it to generate both discriminative and reasoning-driven generative embeddings; a subsequent reinforcement learning enhances reasoning and further optimizes generative embedding quality. This pioneering work reveals four key insights: 1) reasoning-driven generative embeddings unlock substantial performance gains over conventional discriminative embeddings by leveraging the powerful generative reasoning capabilities of MLLMs; 2) discriminative and reasoning-driven generative embeddings are complementary, whose combined oracle performance far exceeding that of either alone; 3) RL can effectively enhance reasoning-driven generative embeddings, establishing a scalable optimization paradigm; 4) repeated sampling at inference boosts downstream task coverage (pass@k), highlighting the inference-time scalability potential of reasoning-driven generative embeddings. Evaluated on the MMEB-V2 benchmark across 78 tasks spanning video, image, and visual documents, UME-R1 significantly outperforms conventional discriminative embedding models and offers a foundation for more interpretable, reasoning-driven generative multimodal embeddings. Our datasets, models, and code are available at https://github.com/XMUDeepLIT/UME-R1.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces UME-R1, a universal multimodal embedding framework that bridges discriminative and generative paradigms, allowing embeddings to be produced either directly or through reasoning-driven token generation. The method involves a two-stage training process: a supervised stage with chain-of-thought reasoning and summary annotations, followed by reinforcement learning leveraging a novel reward based on template adherence and embedding quality. Evaluations on the MMEB-V2 benchmark demonstrate strong improvements over discriminative embedding models.

### Strengths
1. Thorough testing across 78 diverse tasks spanning three visual modalities provides strong empirical validation.
2. Generative embeddings provide interpretable reasoning paths, potentially allowing quality assessment through generated explanations.
3. This paper claims some insights about the relationship between generative and discriminative embeddings as well as their impact on the model's performance.

### Weaknesses
1. Generating reasoning and summaries significantly increases inference time and computational requirements compared to traditional discriminative embeddings.
2. The selection mechanism between discriminative and generative embeddings for different tasks is not clearly defined, and there is a gap from the oracle baseline.
3. UME-R1's performance doesn't seem to be comparable with the SOTA baseline on the VisDoc workload.

### Questions
1. How is the generated embedding used in different MMEB-V2 tasks?
2. How would one deploy a system that decides between discriminative and generative embeddings in practical applications? Is there any mechanism—heuristic or learned—that enables adaptive switching?
3. UME-R1 doesn't beat the previous SOTA work on the VisDoc workload. Can you provide some potential solutions to improve performance for this specific task?
4. In the ablation study, the author claims that the RL stage improves the performance substantially. However, the improvement seems marginal according to Table 2.

### Soundness
2

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
1

### Summary
This paper propose UME-R1, a novel universal multimodal embedding framework which unifies the discriminative and generative embeddings. A two-stage training strategy applies a cold-start supervised fine-tuning for the model with reasoning capabilities. Extensive experiment is conducted across various visual tasks and the proposed method outperforms conventional discriminative embedding models and provide a future direction of more interpretable, reasoning-driven generative multimodal embeddings.

### Strengths
1. Generative embeddings are worth studying and have certain commonalities for multimodal tasks. This work explores a reasoning-driven method that is brave and novel.

2. The experiment result is solid with a newly constructed cold-start supervised fine-tuning dataset for embedding training with intermediate reasoning and summaries.

3. Convincing visualization examples are provided to prove the effectiveness of the proposed method.

### Weaknesses
1. Lack of a comparison framework between the traditional multimodal embeddings and the proposed method of generative embeddings. Providing this may help better presentation.

2. Visual examples require relatively detailed explanations; otherwise, they may feel complex and difficult to determine the more efficient aspects of the proposed method when viewed directly.

### Questions
1. See weakness.

2. The training cost and inference speed between the proposed method and baselines may require further investigation.

Overall, I think this work is solid. Although it is an essential module in MLLM work, multimodal embeddings is a relatively unfamiliar research field to me compared to MLLM, which calls for a better presentation in several respects on this work. So I tend to adjust my score and confidence based on the author's response and other reviewers' opinions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces UME-R1, a multimodal embedding framework that unifies discriminative and generative embeddings within a single model. Specifically, it incorporates reasoning supervision during training and extends the traditional discriminative embedding pipeline to produce generative embeddings through a reasoning-augmented generation process. Moreover, it employs reinforcement learning to further enhance representation quality. Experiments on MMEB-V2 demonstrate consistent improvements over baselines.

### Strengths
- The integration of generative embeddings into the existing embedding learning paradigm represents a conceptually meaningful extension beyond prior MLLM-based approaches.
- The application of reinforcement learning with verifiable rewards is a well-considered adaptation for optimizing embedding quality.
- Experiments show consistent and notable improvements on the MMEB-V2 benchmark across multiple modalities.

### Weaknesses
1. The improvement achieved by the RL stage is marginal (around one point) relative to its additional computational cost.
2. While the oracle setting illustrates the upper bound of the framework, the resulting scores are overly idealized and offer limited practical relevance without an implementable selector.

### Questions
1. The authors should discuss the weaknesses mentioned above to provide a constructive view of the method’s effectiveness.
2. What is the additional inference-time cost of generating reasoning and summary tokens compared with standard discriminative embedding extraction?
3. Have the authors examined how varying the loss weighting among contrastive and generative objectives during the SFT stage affects final performance?
4. Have the authors considered implementing a learned or heuristic selector to approximate the oracle results in practice?

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
This paper introduces UME-R1, a universal multimodal embedding framework that produces embeddings conditioned on chain-of-thought reasoning. The approach consists of a two-stage training strategy: supervised fine-tuning augments query-target pairs with intermediate reasoning and summaries generated by a thinking-capable MLLM, followed by reinforcement learning that optimizes a reward function based on both ranking and similarity gaps to further enhance embedding quality. The model can flexibly produce either discriminative embeddings (extracted directly from input tokens) or generative embeddings (extracted from tokens following generated reasoning and summaries). Evaluated on the MMEB-V2 benchmark spanning 78 tasks across video, image, and visual documents, UME-R1 demonstrates significant improvements over conventional discriminative embedding models, with oracle analysis revealing substantial complementarity between the two embedding types.

### Strengths
1. The core contribution is novel and well-executed. To my knowledge, this is the first work to systematically incorporate chain-of-thought reasoning into multimodal embedding generation, demonstrating that reasoning-conditioned embeddings can substantially outperform standard discriminative embeddings. The idea of having models generate intermediate reasoning before producing embeddings is intuitive and well-motivated.

2. The two-stage training framework is well-designed and clearly presented. The supervised fine-tuning stage effectively teaches the model to produce both embedding types while developing reasoning capabilities, and the reinforcement learning stage introduces a clever reward design that addresses the challenge of applying RL to embedding tasks which lack ground-truth answers. The combined ranking and similarity gap reward is particularly elegant, addressing the zero gradient problem that would arise from using fixed similarity thresholds.

3. The experimental results are strong and comprehensive. UME-R1 achieves consistent improvements across multiple modalities (images, videos, visual documents) and task types. The ablation studies are thorough, demonstrating the value of each component including the RL stage, the dual reward design, and the generative training objective. The oracle analysis showing 3.6-4.3 point improvements reveals interesting complementarity between discriminative and generative embeddings.

4. The paper includes valuable analysis beyond standard benchmarking. The inference-time scaling experiments (pass@k analysis) reveal an intriguing property of reasoning-conditioned embeddings, and the comparison with external reasoning models demonstrates that self-generated reasoning is more effective than externally-provided reasoning. These insights extend beyond simply showing performance improvements.

5. The writing is generally clear and the paper is well-structured, making it easy to follow the methodology and experimental design.

### Weaknesses
1. The terminology and framing claims in the abstract and introduction are too broad and potentially misleading. The paper repeatedly claims to "pioneer the exploration of generative embeddings" and positions UME-R1 as introducing the first "generative" embeddings. However, the term "generative embedding" is overloaded and could reasonably describe several existing approaches. For instance, LamRA and UniIR extract embeddings from generative models during the generation process, which some would consider "generative embeddings." What UME-R1 actually introduces are embeddings conditioned on chain-of-thought reasoning that the model generates before producing the final embedding. This is a more specific contribution than introducing "generative embeddings" broadly. The authors should revise their claims throughout the abstract, introduction, and conclusion to more precisely describe their contribution as "reasoning-driven" or "CoT-conditioned" or "reasoning-conditioned" embeddings rather than claiming primacy on all "generative embeddings."

2. The distinction between discriminative and generative embeddings as defined in this paper needs clearer justification. The paper defines discriminative embeddings as those extracted from the last input token and generative embeddings as those extracted after generating reasoning and summaries. However, this distinction is somewhat arbitrary - both involve forward passes through the model and both produce vector representations. The key difference is really whether intermediate reasoning is generated, not whether the embedding process is "discriminative" vs "generative" in any fundamental sense. The paper would benefit from more precise terminology that doesn't overload these already-loaded terms from the broader machine learning literature. 

3. The computational costs and practical implications of the approach deserve more discussion. Generative embeddings require generating potentially lengthy reasoning chains (up to 8,192 tokens according to the experimental setup), which substantially increases inference cost compared to discriminative embeddings. Although, I think the performance improvements and potential interpretability make the approach "worth it", it would be nice to consider these ancillary factors in the work.

### Questions
My comments are constructive, containing both my critique as well as approaches to resolve the concern.

### Soundness
3

### Presentation
3

### Contribution
3
