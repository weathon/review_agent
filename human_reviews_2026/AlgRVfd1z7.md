# Referring Layer Decomposition

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
Precise, object-aware control over visual content is essential for advanced image editing and compositional generation. Yet, most existing approaches operate on entire images holistically, limiting the ability to isolate and manipulate individual scene elements. In contrast, layered representations, where scenes are explicitly separated into objects, environmental context, and visual effects, provide a more intuitive and structured framework for interpreting and editing visual content. To bridge this gap and enable both compositional understanding and controllable editing, we introduce the Referring Layer Decomposition (RLD) task, which predicts complete RGBA layers from a single RGB image, conditioned on flexible user prompts, such as spatial inputs (e.g., points, boxes, masks), natural language descriptions, or combinations thereof. At the core is the RefLade, a large-scale dataset comprising 1.11M image–layer–prompt triplets produced by our scalable data engine, along with 100K manually curated, high-fidelity layers. Coupled with a perceptually grounded, human-preference-aligned automatic evaluation protocol, RefLade establishes RLD as a well-defined and benchmarkable research task. Building on this foundation, we present RefLayer, a simple baseline designed for prompt-conditioned layer decomposition, achieving high visual fidelity and semantic alignment. Extensive experiments show our approach enables effective training, reliable evaluation, and high-quality image decomposition, while exhibiting strong zero-shot generalization capabilities. The project will be released at https://yaojie-shen.github.io/project/RLD/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new task that extracts a target RGBA layer guided by a user-provided prompt. To support this task, the authors present RefLade, a dataset comprising 1.11 million image–layer–prompt triplets. Building on RefLade, they establish an evaluation protocol along three key dimensions: preservation, completion, and faithfulness. Furthermore, the paper proposes RefLayer as a baseline model for this task. RefLayer encodes spatial prompts using color-coded maps integrated into the latent space and employs a parallel alpha decoder to generate complete object RGBA layers.

### Strengths
1. This paper is well-written and easy to understand.

2. This paper defines a novel task and presents a comprehensive framework to address it, including a dedicated dataset, evaluation metrics, and a baseline model.

### Weaknesses
1. In Figure 1, the flag extracted exhibits a noticeable color discrepancy (or color cast) when compared to the original image. The color shift is quite apparent and detracts from the quality of the result. The authors should investigate the cause of this artifact.

2. The paper's qualitative evaluation is currently a weak point. First, the paper lacks direct qualitative comparisons of different methods and ablation study. Without side-by-side visual comparisons, it is difficult for the reader to verify the claimed advantages of the proposed model over existing work. Second, the number of generated results shown is limited. The authors should include a more comprehensive and diverse set of examples (either in the main paper or, more extensively, in the supplementary material) to better demonstrate the model's capabilities, robustness, and potential failure cases.

### Questions
1. The authors employed a third-party, closed-source LLM to construct a large-scale dataset. Have the authors considered making this dataset publicly available? Since the primary contribution of this work lies in the dataset’s construction, the overall impact of the paper would be considerably limited if the dataset remains closed-source.

2. In the constructed dataset, approximately 25% of the training data contain errors. As mentioned in the appendix, these issues primarily result from failed restoration and segmentation inaccuracies. Have the authors conducted an analysis of these erroneous samples? Furthermore, do these problematic data limit the potential applications of the dataset? Do the authors plan to address these issues in future work?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduce the Referring Layer Decomposition (RLD) task, a new task which predicts
complete RGBA layers from a single RGB image, conditioned on flexible
user prompts.  a large-scale
dataset (RefLade) comprising 1.11M image–layer–prompt triplets  is constructed.  In addition, RefLayer is proposed as a simple baseline for
prompt-conditioned layer decomposition.

### Strengths
1. This paper introduces Referring Layer Decomposition (RLD), the pioneering task that explores layer decomposition guided by multi-modal referring inputs.

2. The authors introduce RefLade, a large-scale dataset of 1.11 million image-layer-prompt triplets built using a scalable data engine. With its human-curated splits for tuning and testing and a well-defined evaluation protocol, RefLade facilitates and paves the way for future RLD studies. RefLayer is also desigend as a simple baseline.

### Weaknesses
1. More details to ensure the correctness of the image–layer–prompt triplets should be given. In scene understanding, the availabel models for object detection and instance-segmentation can not perform well in all situations, especially for small or occlude objects. How doauthors deal with these cases?

2. It would have been better to show the image distribution with respect to styles, e.g., real images, cartoon， posters and so on, and discuss the model performance in different styles.

### Questions
What are the computational costs in terms of both human effort and GPU resources  to construct  training and testing datasets.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new task called Referring Layer Decomposition (RLD). RLD is conceptually similar to referring image segmentation, whose inputs can be text or spatial cues of the target object. The output of the task is the layer representation of the target object (RGBA, completed object + alpha mask). Since the most critical factor for generative AI models is the data, this work scales up the RGBA data generation pipeline and creates the RefLade dataset which is significantly larger than prior real-world RGBA datasets. Then, authors design an evaluation metric, which is shown to be aligned with human preference. Finally, a simple baseline model, RefLayer, is designed and fine-tuned on the proposed dataset. Despite its simple design, it achieves significantly better performance than prior works.

### Strengths
- The paper is well-written and easy to follow. The figures and charts help understand the pipeline and the statistics of the dataset very well.
- We all know that data is key to the advance of GenAI. This paper proposes a large-scale, real-world dataset for an interesting task. The data generation pipeline involves several SOTA models which guarantee the high quality of the dataset.
- The analysis on data scale, data quality (different subsets), and pre-trained models is comprehensive.
- I love how the proposed metric is aligned with human preference. This is critical for a useful metric to monitor real model progress.

### Weaknesses
I do not see any major weaknesses in the paper. As a paper that defines a new task and proposes a new dataset, every aspect of it is executed very well. Maybe one concern is having more baselines: the paper proposes its own simple baseline which is great, but would it be possible to adapt some prior work's models to this task and benchmark against your proposed method?

Another weakness is, can you show some downstream application of the dataset, similar to Sec.4.4 of the MULAN paper. For example, is it possible to fine-tune LayerDiffuse on the single-object subset of RefLade, and see if that improves performance? Or fine-tune InstructPix2Pix to perform object removal & insertion tasks? These results will definitely make the paper stronger, though I think the current form is already enough for acceptance.

### Questions
- The HPA scores are low when using text prompts. Can you provide some qualitative failure case analysis? One issue might be there are multiple similar objects (e.g. belong to the same category) in an image, so text prompts along cannot disambiguate between them.
- What are model 1-9 in Fig.4? Are they the same model trained on different steps / amount of data?
- Typo in Fig.1 "Linguistic Prompting" part: “The Brown and white house” should be "horse” not "house".

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
3

### Summary
The paper introduces the task of Referring Layer Decomposition (RLD), which aims to recover a complete RGBA layer corresponding to a user-provided prompt, such as spatial maps or text. To support this task, the authors construct RefLade, a large-scale dataset of image-layer-prompt triplets produced through a combination of automatic generation and human curation, along with an evaluation protocol assessing preservation, completion, and faithfulness. A diffusion-based baseline, RefLayer, is proposed and evaluated extensively. Overall, the paper presents a well-motivated and clearly defined formulation, a high-quality dataset, and a comprehensive baseline, offering a valuable foundation for future research on image layer decomposition.

### Strengths
- The proposed dataset constitutes a significant improvement over existing resources for this problem domain, both in scale and in the level of curation. The combination of automated and manual verification enhances its overall quality and reliability.
- The paper provides thorough evaluations, including analyses of design choices, as well as assessments of the alignment between the proposed metrics and human judgments.
- The paper is clearly written, with a well-motivated problem statement and sufficient technical and implementation details.

### Weaknesses
The paper is overall good, and I didn't find major weaknesses. One minor:

For the completion metric, what is the rationale for defining it as the difference between CLIP embeddings, $f(g_\text{rgb}) - f(g_\text{rgb} * g_v)$, rather than directly using the CLIP embedding of the non-visible region, $f(g_\text{rgb} * (1 - g_v))$? 
The motivation for this specific formulation should be clarified.

### Questions
No questions

### Soundness
4

### Presentation
4

### Contribution
3
