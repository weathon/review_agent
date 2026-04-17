# TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents TikZilla, a new family of open-source models for generating scientific figures from text by producing TikZ code. The authors' primary contributions are twofold: 1) the creation of a large-scale, high-quality dataset named DaTikZ-V4, and 2) a two-stage training pipeline. The dataset involves an LLM-based pipeline to debug and repair uncompilable code and a VLM-based process to generate rich figure descriptions, overcoming the sparseness of typical captions. The training methodology first uses SFT to teach the model TikZ syntax, followed by RL to align the output with the desired visual semantics. A key aspect of the RL stage is a domain-specific reward model, derived from a fine-tuned inverse-graphics encoder, which proves to be more effective than generic multimodal metrics. Through extensive automatic and human evaluations, the paper demonstrates that even small TikZilla models (3B/8B) achieve SOTA results, outperforming strong proprietary baselines like GPT-4o and matching the performance of next-generation models on image-based evaluations.

### Strengths
This is a strong paper with several key strengths that make it a compelling contribution to the field.

1.  The paper's most significant contribution is the creation of DaTikZ-V4, a large-scale, high-quality dataset for the Text-to-TikZ task. The authors have addressed the critical problem of data scarcity and quality through a meticulous, multi-source collection process and two highly innovative enhancement steps: (1) an LLM-based debugging pipeline to repair uncompilable code, substantially increasing the usable data, and (2) the use of VLMs to generate semantically rich descriptions, which are demonstrably superior to the sparse original captions. The appendix further validates the data quality with detailed analysis.

2.  The proposed domain-specific reward model for RL is a standout feature. By finetuning an inverse-graphics (Image-to-TikZ) model and using the semantic distance between image embeddings as a reward signal, the authors have developed a method that is simple, intuitive, and highly effective. This approach is more semantically faithful for scientific figures than general-purpose metrics, and its strong correlation with human judgment validates its design and utility in guiding the model towards generating visually accurate figures.

3.  The paper presents a solid two-stage training pipeline (SFT followed by RL) that is executed with rigor, evidenced by the detailed description of the multi-GPU, long-duration training process. The authors clearly articulate the role and limitations of each stage, providing a strong justification for their approach. For instance, the paper notes that while SFT trains the model in syntax, it "remains unaware of the rendered semantics," which perfectly motivates the necessity of the subsequent RL stage for visual alignment.

4. The experimental evaluation is extensive and highly convincing. The authors benchmark their models against a wide range of the latest and most powerful systems on a carefully constructed, contamination-free test set. The results are outstanding: TikZilla achieves state-of-the-art performance across multiple metrics, significantly outperforming top open-source models and even much larger proprietary systems like GPT-4o. Crucially, these strong quantitative results are corroborated by a thorough human evaluation with expert annotators, confirming that TikZilla produces publication-quality figures.

5.  The paper is exceptionally well-written and clearly structured. The motivation is compelling, the problem is well-defined, and the proposed solutions are presented in a logical, easy-to-follow narrative. The authors excel at explaining the rationale behind each design choice, making the entire research story coherent and persuasive.

### Weaknesses
1. The entire training pipeline is fundamentally dependent on the quality of the VLM-generated figure descriptions. While the paper demonstrates that these are superior to raw captions, VLMs are known to hallucinate and omit critical details.
2. The paper convincingly argues for its domain-specific reward model (R_sim) by showing its high correlation with human judgment. However, it doesn't contain a direct experimental comparison against general-purpose reward models (like CLIPScore or DreamSIM) within the RL training loop.
3. The reward model operates on the semantic similarity of image patch embeddings. It may not be sensitive enough to fine-grained stylistic details that are crucial for scientific figures, such as precise line weights, specific dash patterns, exact color hex codes, or font styles.
4. The dataset is sourced primarily from arXiv, GitHub, and TeX StackExchange, which likely results in a corpus rich in diagrams, plots, and graphs common to fields like computer science, mathematics, and physics. But it's unclear how well TikZilla would generalize to more structurally different (OOD) types of scientific figures, such as complex chemical reaction pathways, intricate biological diagrams, or detailed engineering schematics, which may be underrepresented in the training data.

### Questions
1. Could you provide a more quantitative analysis of the VLM's error rate? For example, by manually evaluating a random sample of generated descriptions against their ground-truth images, could you estimate how often the VLM omits key details, hallucinates elements, or misinterprets spatial relationships?
2. While I understand that a full training run with a different reward model is resource-intensive, could you provide more direct evidence of $R_{sim}$'s practical superiority?
3. Have you observed any limitations regarding the model's ability to render specific styles, such as different dash patterns, line weights, or precise color shades, when these are explicitly requested in the prompt? Since the reward is based on patch embeddings, it might not penalize deviations in these subtle attributes as strongly.
4. Table 3 shows an interesting result: RL applied directly to the base Qwen3-8B model (+RL) yields a significant performance boost (AVG 0.251 → 0.357), whereas for Qwen2.5-3B, the improvement is much smaller. The paper suggests this is because larger models already encode some TikZ knowledge. Could you elaborate on this hypothesis? Is the primary role of SFT for smaller models simply to get them into a "syntactically plausible" region where RL can effectively take over?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on advancing the state-of-the-art in the application of text-to-Tikz, the challenge of generating code for a tikz diagram based on a text specification. The paper provides several contributions. The first is an analysis of existing text-to-Tikz datasets. The paper finds that relying on captions for the textual description of a figure, as many of these datasets apparently do, is probably insufficient for the development of strong models since captions fail to capture major structural components. To mitigate this, the paper introduces DaTikZ-V4, which revises and extends a previous dataset called DaTikZ-V3. This includes more careful filtering, additional data sources like GitHub, and the inclusion of LVLM figure descriptions. After this, the paper describes its approach to training text-to-TikZ models using supervised finetuning and then RL with a special emphasis on the RL rewards used. Finally, the paper describes a range of comparisons between models trained with the proposed methods (the paper calls these TikZilla models) and other relevant LLMs, showing that Tikzilla models are competitive with much larger, proprietary systems.

### Strengths
**Clarity:** The paper is well-written. It effectively communicates its contributions relative to other work in this area so that even a reader that is not familiar with the text-to-TikZ problem can appreciate the results. DaTikZ-V4 is well-motivated by an analysis of existing datasets (though the reviewer would have appreciated a few more examples). The figures are all of high quality and efficiently visualize key ideas in the paper.

**Effective, small, specialized models:** One potential criticism of this paper is that the TikZilla models are at best marginally better than many existing models (e.g., GPT5). The important point though (from this reviewer’s perspective) is that the TikZilla models are much, much smaller than GPT5 and other proprietary models that achieve comparable performance. This reviewer believes that the generation of small, reliable, specialized models is an important contribution to the community.

### Weaknesses
**A lot of the paper feels routine:** The approach described in the paper appears to result in substantially better performance than prior small-model approaches. However, the methods used to achieve these improvements are exactly what most readers would likely expect: increase the size of the dataset, increase the quality of the dataset, implement recent approaches that have been successful in language modeling broadly (e.g., RL). One exception to this is the section on reward signals specific for the text-to-TikZ problem. Together, the reviewer believes that this will somewhat limit what a reader will take away from the work. On the other hand, a result being in line with community intuition should not be a barrier to publication. As such, the reviewer did not weight this ‘weakness’ very strongly when evaluating the work.

**More analysis of failure cases would be interesting:** The paper devotes considerable space to describing how current text-to-TikZ datasets fall short of what would likely be needed to train strong text-to-TikZ models. The reviewer appreciated this contribution. It would have been nice to see a similar type of analysis applied to the results section where text-to-TikZ models were compared. TikZilla and GPT5 score comparably in certain respects, but this doesn’t mean that they fail in the same ways. If large proprietary models and TikZilla have different strengths and weaknesses, this would be very important to know in practice. 

**Nitpicks**
- Line 268: A parenthesis is incorrectly oriented.
- Line 365: This would probably come across as more interesting if it was not labeled as ‘emergence’, which feels like a strong word for this particular phenomenon.
- Table 3: It would be helpful to remind the reader what the acronyms stand for so they don’t have to flip back to earlier sections.

### Questions
-	Line 478: Why is this approach “ethical”?
-	Will models or datasets be released?

### Soundness
3

### Presentation
3

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
This paper proposes a data generation methodology to generate 1.5M tikz examples. These are then used to finetune QWen (with RL + SFT) in order to obtain small-ish (3B/8B) models that surpass GPT-4 and are similar in performance to GPT-5.

The authors do this by first sourcing data from Github and other sources and then using a pipeline to identify valid tikz and leveraging a VLM to obtain descriptions.

This data is used to finetune and then with RL using GRPO.

### Strengths
1. This paper describes how to obtain a large scale text-to-tikz dataset. They use a combination of choices (e.g. 1 tikz per website; ensuring compilation; VLM style description) to obtain a large scale and high quality dataset.

2. They describe howt o use such a dataset for SFT and for RL. For RL, they add a couple of domain specific changes -- e.g. the model for the reward; the scalar rewards for capturing semantic alignment. 

3. Their strategy seems to work -- leading to clear improvements across model sizes and leading to performance that is comparable with GPT-4o / GPT-5 though much smaller by using a human rating comparison.

### Weaknesses
1. It would be good to quantify the difference in model size in Figure 4 -- how much smaller is the model than GPT-5/4o in terms of sheer size of the model as well as compute at inference time.

2. it would be good to plot / understand performance as a function of the size of the data. If we want better performance, can we just collect more data or are we already hitting diminishing returns here ?

3. Is the dataset planning to be made public? It would be useful for other researchers I imagine.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents DaTikZ-V4, a new large-scale dataset designed for text-to-TikZ generation.
In addition to the conventional supervised fine-tuning (SFT) approach, the authors introduce a reinforcement learning framework that explicitly takes into account the visual quality of the generated images.
By employing a two-stage training strategy that combines large-scale data with reinforcement learning, the proposed method achieves a substantial improvement in text-to-TikZ generation performance.

### Strengths
- The authors newly construct a large-scale TikZ dataset, approximately four times larger than the existing dataset.
- They define a reward function based on image similarity and demonstrate the effectiveness of reinforcement learning.
- They conduct not only automatic evaluations but also human evaluations, demonstrating the effectiveness of the proposed method.

### Weaknesses
- In Section 4, the paper describes the collection of a new large-scale dataset; however, there appears to be no mention of its license information. Since license details are essential for enabling data reuse, it would be helpful to provide not only the total number of data samples but also a breakdown of the dataset by license type.
- The correlation between the automatic evaluation metrics and the human evaluation results appears to be low, making it difficult to accurately assess the quality of the generated results based solely on the automatic scores. For example, in Table 4, the discussion is based only on the automatic evaluation values. It would therefore be desirable to develop or employ automatic evaluation metrics that exhibit a higher correlation with human judgments.

### Questions
In Section 3, under Caption Quality Analysis, the paper states that “Accurate text-to-TikZ generation requires captions that specify objects, attributes, and spatial relations.”
While it is understandable that detailed captions are helpful for reducing hallucinations during model training, in practical applications it is often costly to describe every element in an image exhaustively. In many real-world scenarios, users may prefer to generate diagrams from shorter or more abstract textual instructions rather than from fully detailed descriptions.
Given that the proposed model is trained only on detailed captions, how does it perform when provided with insufficiently detailed or abstract instruction texts? What kind of outputs would it produce in such cases?

### Soundness
2

### Presentation
3

### Contribution
3
