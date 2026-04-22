# ModernVBERT: Towards Smaller Visual Document Retrievers

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Retrieving specific information from a large corpus of documents is a widely prevalent industrial use case of modern AI, notably due to the popularity of Retrieval-Augmented Generation (RAG) systems. Although neural document retrieval models have historically operated exclusively in the text space, Visual Document Retrieval (VDR) models - large vision–language decoders repurposed as embedding models which directly work with page screenshots as inputs - are increasingly popular due to the performance and indexing latency gains they offer. In this work, we show that, while cost-efficient, this approach of repurposing generative models bottlenecks retrieval performance.
    
Through controlled experiments, we revisit the entire training pipeline, and establish a principled recipe for improving visual document retrieval models. We notably measure the impact of attention masking, image resolution, modality alignment data regimes, and late interaction centered contrastive objectives which emerge as central performance factors.
Building on these insights, we release ModernVBert, a compact 250M parameter vision–language encoder that outperforms recent models up to 10 times larger when fine-tuned on document retrieval tasks, enabling efficient inference on cheap CPU hardware and greatly reducing latency and costs while maintaining strong performance. Models, code and data are available in the public version of this work under an open license.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper systematically investigates the impact of different visual retriever training designs on retrieval performance. The authors analyze the influence of various factors—including model architecture, data scale, attention masking strategies, image resolution, and the scale of image-text pooling in contrastive learning—on different types of visual retrieval tasks. Based on these analyses, they design training methods for the ModernVBERT and ColModernVBERT models. Experimental results demonstrate that the proposed models achieve performance comparable to large-scale models with 10 times more parameters, while requiring only one-seventh of the inference time on CPUs.

### Strengths
- A systematic study and analysis of different visual retriever training strategies and their impact on the performance of visual retrieval tasks.
- The proposed ModernVBERT and ColModernVBERT models contain only 250M parameters but achieve excellent performance across multiple tasks, matching the effectiveness of large-scale models with 10 times their parameter count.

### Weaknesses
- In Section 3.1, the performance differences between the trained vision-language model and SigLIP across various tasks may be influenced by differences in training data scale and distribution. Further analysis is recommended to strengthen the reliability of the conclusions.
- The experiments only report inference speed on CPUs and do not include results from GPU environments. Considering that models are often deployed on GPUs in practice, the speed differences between models of varying scales may be less pronounced in GPU environments. It is recommended to supplement with relevant tests to provide a more comprehensive evaluation of model efficiency.

### Questions
See the Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the problem of visual document retrieval. To this end, the authors have conducted controlled experiments to analyze the factors that can make the performance of visual document retrieval better. This includes attention masking, image resolution, modality alignment data regimes, and late interaction centered contrastive objectives. Based on the conclusions from the controlled experiments, the authors have developed ModernVBERT, a 250M-parameter vision–language model that can achieve good performance on visual document retrieval, though being lightweight.

### Strengths
1. Good insights and analysis. The authors have studied several different factors to influence the performance of visual document retrieval models and provided analysis and insights for each factor.

2. An efficient visual document retrieval model is developed. According to the insights and conclusions, the authors have developed a 250M-parameter model which achieves good performance for the task of visual document retrieval.

### Weaknesses
1. Application scenarios. It would be better if the authors can discuss more about the potential application scenarios of an efficient visual document retrieval model. This can highlight the importance of developing 'efficiency' for the task. For example, it is important to develop an efficient object detection system as when applied to embodied AI and robotics, it is important to make the model small but effective. Are there wide scenarios where an efficient visual document retrieval model is in a high demand?

2. Conclusions on large models. As the authors have noticed, the paper has only conduced experiments on small models and some of the conclusions made in the paper may not hold for large models. This further limit the scope of the paper, given that the scope of the paper has already been constrained to very specific task - visual document retrieval.

3. Presentation of the paper. It is not good to include fake links in the paper. It is better to say models and code will be made publicly available. In this way the space of the paper can also be saved. The 'finetuning' in the abstract and in the paper should be 'fine-tuning'.

### Questions
I would suggest the authors to address my concerns in the weakness section in the rebuttal.

### Soundness
2

### Presentation
2

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
The paper propsoes ModernVBERT, small but powerful mutimoal embedding model. Unlike larger multimodal embedding embedding models, ModernVBERT opts for a text encoder backbone instead of a decoder LLM backbone. The paper demonstrates ModernVBERT's profinciency in visual documents retrieval on VideoRe, matching the performence of substancially larger models.

### Strengths
- Using encoder backbones in multimodal embedding models is under-explored, and the results showing encoder backbones are better then decoder backbones are interesting.
    
- Small document retrievers, especially those that can run on CPUs, are an interesting and distinct setting. Existing work focuses almost exclusively of the >7B range, with is not feasible for most CPUs.
    
- The writing is mostly clear, albeit with a few typos.

### Weaknesses
- Most of the experiments and conclusions in section 3.1 are are severely confounded by the MLM being trained on a mixture containing a very large proportion of documents compared to SigLIP. This results in potentially misleading conclusions.
    
- Although the model has less parameters than other small encoders, the model’s resolution scaling scheme significantly increases the model’s compute requirement by tokenizing an image into more visual tokens (although only on the embedding side).
    
- The tradeoff between ColModernVBERT and a larger CLIP model that has been specialized for document embedding is unclear. In general, the baselines considered are either too large to be a fair comparison or not specialized for document retrieval.
    
- BiModernVBERT is very similar in approach to existing mulitmodal embedding models, with the main difference being the use of an encoder instead of a decoder LM.
    
- The manuscript does not contain an explanation for how the resolution of images is “scaled” in the image tokenizer.
    
- Although ModernVBERT has strong document embedding capabilities (for its size) the experiments are not sufficient to conclude what aspect of its design this results from. It could be the training data, architecture, or even the batch curation strategy acting as implicit hard negative mining.

Minor: 
- missing space: “(LoRA)(Hu et al., 2021)”
    
- typo: “alignement”

### Questions
- How is the resolution scaled in the vision backbone and how does that effect the number of visual tokens?
    
- The post training phase mentions that it uses 118M(illion?) samples from MSCOCO, which I assume is actually 118k (judging by table 7)?
    
- Can you control for data when compared to a dual encoder (I.e. SigLip) baseline?

- What is exactly is meant by "too large to run on CPUs" (table 3)? Although the models may be slow to run on CPU, they should still run.

### Soundness
2

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
3

### Summary
It took me a long while to write this review.

Because I found the argument of this paper so boring and far from my interests, and I have been given really too many papers to review. Still, I got assigned this paper and here I am.

Now let me be clear: by saying that I found the argument boring, I do not mean that it is bullshit. It actually tackles a very concrete problem and offers a sensible improvement.

A problem so important that I myself, without even noticing it, make constant use of the currently available solutions.

Recently I saw a cool video about the conception of the IKEA Lack table, the small coffee table you can buy for 10$. It is a cheap coffee table, nothing exactly exciting or groundbreaking, but it is something fundamentally useful, and the idea of making it out of hexagonal cardboard had quite an impact by making it astonishingly cheap and allowing its wide spreading all over the world.

Tables may be boring, but this accomplishment deserves attention, and the details of the idea are worth knowing and celebrating. I see this paper as the blueprint of the IKEA Lack table, containing the very clever innovations and their reason, such as the hexagonal cardboard internal frame (the patch representations) or the trick of not using full-wood legs (the bidirectional attention à la BERT).

My criticism is about the presentation.

The paper should do a better job explaining the problem and showing why it is relevant. ICLR is full of people who do not really care about tables, and since your paper is here, you should make an effort in explaining the relevance of what you have done, perhaps by doing a better job explaining the problem in the abstract and the introduction.

Already the abstract was cryptic for me. Do not rely on the tag “Visual document retrievers”; most people don’t know what you are talking about. Your paper should be more self-contained.

I have nothing to say regarding the rest; I find your paper well executed.

I will put a 6 that I will update to an 8. I hope to update it after having read a new abstract and introduction that make me more passionate and aware of the small but very significant improvement that is going to be described in the next pages.

This is the video of the IKEA Lack table: https://www.youtube.com/watch?v=0h8vAGCiRX0

### Strengths
see above

### Weaknesses
see above

### Questions
see above

### Soundness
4

### Presentation
2

### Contribution
3
