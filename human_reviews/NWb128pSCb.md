# Evaluating Semantic Variation in Text-to-Image Synthesis: A Causal Perspective

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 5

## Abstract
Accurate interpretation and visualization of human instructions are crucial for text-to-image (T2I) synthesis. 
However, current models struggle to capture semantic variations from word order changes, and existing evaluations, relying on indirect metrics like text-image similarity, fail to reliably assess these challenges. This often obscures poor performance on complex or uncommon linguistic patterns by the focus on frequent word combinations.
To address these deficiencies, we propose a novel metric called SemVarEffect and a benchmark named SemVarBench, designed to evaluate the causality between semantic variations in inputs and outputs in T2I synthesis. 
Semantic variations are achieved through two types of linguistic permutations, while avoiding easily predictable literal variations.
Experiments reveal that the CogView-3-Plus and Ideogram 2 performed the best, achieving a score of 0.2/1. Semantic variations in object relations are less understood than attributes, scoring 0.07/1 compared to 0.17-0.19/1. 
We found that cross-modal alignment in UNet or Transformers plays a crucial role in handling semantic variations, a factor previously overlooked by a focus on textual encoders. 
Our work establishes an effective evaluation framework that advances the T2I synthesis community's exploration of human instruction understanding. Our benchmark and code are available at https://github.com/zhuxiangru/SemVarBench.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on the performance of Text to Image (T2I) models in handling semantic variations. It identifies that existing T2I models tend to overlook the relationships between words in the input text when faced with similar semantics or semantic substitutions, making it challenging to generate accurate images. At the same time, the paper introduces the SemVarEffect metric and SemVarBench to evaluate this issue.

### Strengths
- This paper offers an interesting and novel perspective on the current issues facing T2I models, and provides an effective metric and dataset to measure this issue.
- The paper offers detailed and fair experiments to compare various T2I models.
- The analysis section is thorough and carefully examines the causes of the semantic understanding issues in T2I models, providing a reliable path for future research.

### Weaknesses
- I noticed that some of the samples provided in the paper, such as "A mouse chasing a cat," are somewhat unrealistic captions. However, this category is not emphasized in SemVarBench, as seen with sentences like "A cat chasing a dog" and "A dog chasing a cat." These sentences also exhibit semantic variations, but I believe that current T2I models can handle these changes. Therefore, I think further discussion on this issue is needed.
- I believe it is necessary to further discuss the relationship between this phenomenon and the training data in the discussion. In the problematic captions where semantic variations occur, a significant portion of the original captions may be included in the pre-training data. Therefore, similar to the discussion in paper [1], I suggest designing experiments to investigate whether T2I models can perform reverse inference of semantic variations.

[1] Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stickland, A. C., Korbak, T., \& Evans, O. (2024). *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"*. arXiv:2309.12288.

### Questions
See Weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper primarily focuses on the impact of word order variations in text prompt on text-to-image synthesis. It proposes a metric to evaluate a model's semantic sensitivity to changes in word order. The main idea is to calculate the difference in visual semantics of generated images between semantic-variant and semantic-invariant text rearrangements. Additionally, a dataset of approximately 11k samples has been collected as an evaluation benchmark, and popular models have been assessed and analyzed based on this benchmark.

### Strengths
- The authors find that existing text-to-image synthesis models struggle to capture semantic shifts caused by word order changes and proposed an evaluation metric and benchmark. 
- The SemVarEffect metric is based on the difference between semantic-variant and semantic-invariant text rearrangements, similar to the differential denoising principle in circuits, which I find quite novel. 
- The paper is well-organized and easy to understand.

### Weaknesses
- The definition of visual semantics: The primary contribution of this work is the metric to measure the impact of word order in text prompts on text-to-image synthesis. The basic principle is to alter the text and observe the resulting visual semantic changes in the generated image. However, within this metric, the “visual semantics” of the generated image is defined through text-image similarity, which may not be entirely accurate. Visual semantics in an image typically refers to the content within the image itself, such as object categories, attributes, or scene information. In contrast, text-image similarity only measures the alignment of high-level features between the two modalities; it does not represent the actual visual semantics. Ideally, visual semantics should be defined independently of text. 
- The mechanism for ensuring that the generated data is semantic-variant or semantic-invariant during benchmark creation seems to be insufficiently specified, and why are some data extraction templates and rules created using LLMs while others are crafted manually? What challenges are involved in this?

### Questions
- In section 4.1, If text-image similarity is obtained using MLLMs, could biases inherent in the MLLM itself (insensitivity to changes in word order) affect the evaluation results?
- Text-to-image synthesis models have difficulty accurately reflecting semantic changes caused by variations in word order, possibly due to certain commonsense biases inherent in the training data. This could be seen as a form of “built-in bias.” For example, as illustrated in Figure 1 of the paper, “A cat chasing a mouse” is a reasonable depiction, whereas the reordered phrase “A mouse chasing a cat” is less plausible, leading to poorer generation results. It might be helpful to add related experiments to validate this hypothesis.

### Soundness
2

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
4

### Summary
This paper proposes a novel metric called SemVarEffect and a benchmark called SemVarBench to evaluate text-to-image (T2I) models' ability to capture semantic variations caused by word order changes in input prompts. The key contributions are:
- A causal inference framework to measure how semantic variations in text inputs affect visual semantic variations in generated images.
- A carefully curated dataset of text triplets (anchor, permutation with changed meaning, permutation with unchanged meaning) to test T2I models.
- Comprehensive evaluation of  state-of-the-art T2I models. This also includes fine-grained Analysis of the role of text encoders and cross-modal alignment in handling semantic variations.

### Strengths
Strengths:
Novel and important problem formulation: The paper addresses a critical gap in evaluating T2I models' language understanding capabilities.

The causal inference based framework provides a principled way to isolate the effect of semantic variations.

SemVarBench is carefully curated to avoid easily predictable variations and focus on challenging semantic permutations, with comprehensive coverage of different type of variations. LLM generated results were checked.

### Weaknesses
The training dataset is generated using the same base SDXL model and filtered by MMLM, which could introduce self-distillation artifacts and bias, particularly affecting performance on categories with less data. 

The paper suggests that fine-tuning improves alignment at the word level rather than semantic variants, but doesn't provide sufficient discussion or analysis to support this claim

### Questions
How does the choice of filtering training data using MMLM potentially affect the evaluation results, particularly for categories with limited samples?

With the 4 metics you had mentioned, which one is the main metric that would make sense quickly for readers? It is a little hard to follow with different metrics presented at different parts, which I can imagine is a little confusing especially for readers who are looking for a single metric for benchmarking.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a novel metric, SemVarEffect, and a corresponding benchmark, SemVarBench, designed to evaluate the causality between semantic variations in input text and output images for text-to-image (T2I) synthesis models. The authors identify a gap in current evaluation methods, which often rely on indirect metrics like text-image similarity, failing to adequately assess how models handle semantic changes due to word order. SemVarEffect measures the average causal effect of input semantic variations on output images, while SemVarBench provides a dataset with controlled linguistic permutations to test model performance. The experiments reveal that even state-of-the-art models struggle with semantic variations, indicating room for improvement in understanding and generating accurate visual representations from textual instructions.

### Strengths
- The paper directly addresses a critical issue in the evaluation of T2I models, which is the accurate interpretation of human instructions and evaluate the finegrained instruction following ability of the models.

### Weaknesses
1. The paper suffers from significant presentation issues, making the proposed method difficult to comprehend. The overall structure of the methodology section requires refinement:
   - Figure 2, which is crucial to understanding the proposed method, is challenging to interpret.
   - In Section 2.2, the term "localized change" needs clarification.
   - The definition of "integrated visual semantic variations" is also unclear.
   - Line 278 uses the phrase "evoke vivid mental images," but its meaning is ambiguous in the given context.
   - Table 2 includes numerous annotations, but offers no explanation of what the different scores represent.

2. In my opinion, the method appears overly complex. The core idea is straightforward: we anticipate that semantic alignment of generated images should be strong if the inputs are semantically identical, and less aligned if the inputs differ semantically. Thus, a simple metric like S(I_a, I_pu) - S(I_a, I_pi) could be employed. What advantages does the proposed causal relationship provide over this simple baseline? I would appreciate deeper insight and explanations on this matter.

3. Regarding model-based evaluation, there is no discussion on whether the MLLM used to compute the alignment score is effective for this task, nor is there any human evaluation. This omission raises concerns about the reliability and trustworthiness of the scores presented.

### Questions
See weakness part.

### Soundness
2

### Presentation
1

### Contribution
2
