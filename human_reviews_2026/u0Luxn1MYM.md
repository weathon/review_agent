# Don’t Lag, RAG: Training-Free Adversarial Detection Using RAG

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types - all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs - including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO - alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95% classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98%, but remains closed-source. Experimental results demonstrate VRAG’s effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work extends the RAG into the area of adversarial defense, formulating a robust, efficient, and training-free defense against existing adversarial patch attacks. This method first divides the image into localized regions, retrieves similar adversarial images, concatenates all of them as the context, and then prompts a downstream VLM for classification.

### Strengths
- Clear motivation. This method adopts the RAG as an attacked/unattacked classifier as a filter.
- Simple and effective method. This method effctively use the pre-trained the vision encoder and VLM with RAG to build a powerful training-free filter

### Weaknesses
Though the method is simple and effective, there are several flaws in the main paper, especially the evaluation process which makes it hard to follow:
1. Unclear comparison. From the presentation, the *proposed method acts like a filtering method* (see Fig. 1 and A. 4). In contrast, some reference and baseline methods are robust classification methods, i.e., they continue the subsequent classification task even under an attack image. So, what exactly are Table 1 and Table 2 comparing? Does the author drop the attacked images and report accuracy on the rest? 
	- If so (I cannot get the method from the paper, so I can only guess, and please correct me if I made a mistake), then the baseline selection and comparison are misleading, and the comparison is not fair. Instead, it should select a filtering baseline and report the false-positive rate, AUROC, and accuracy on the filtered set. In practice, the user will be annoyed if the images they update are frequently (for example, 4%) labeled as attacked when they are not. Thus, reporting these two metrics is essential for filtering methods.
	- If not, the authors should specifically describe the **complete** pipeline.
2. Lines 366-367 say a clean accuracy will be provided, but it is not found in Table 1. 
3. The 'baseline' method (using retrieval only) outperforms the proposed method in a 0-short case, but without explanation.
	- What is the detail of the baseline method? After retrieving the image, I guess there is a subsequent thresholder for the cosine similarity attached to decide whether the image is attacked (there is no detailed explanation for this method throughout the main paper and the appendix, thus I can only guess)
	- After a detailed review in the appendix, I think the author is trying to compare a) a simple method with retrieved images and b) a zero short VLM filter without any retrieved images and examples to show the importance of the additional information. If this is the case, the presentation is misleading, as the baseline is shown under the '0s' column while it actually uses additional data from the external database. 

4. Lack of generalization analysis. How can the proposed method be generalized to unseen adversarial patches? There is a generalization study for the patch shape. However, the database covers the attack methods used in the subsequent evaluation. In real-world scenarios, an attacker might not release their attack method; therefore, generalizing to unseen patch attack methods is crucial. The authors are suggested to investigate it by removing part of the database corresponding to the specific attack method, then evaluating the method to test it.

I really appreciate the proposed method, but these flaws significantly mark the paper below the acceptance level. If the above weaknesses are adequately addressed, I will consider increasing my score.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
1

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
This paper presents **VRAG (Visual Retrieval-Augmented Generation)**, a training-free framework for adversarial patch detection. The method retrieves visually similar regions from a pre-built adversarial patch database and uses a vision-language model (VLM) for generative reasoning to identify malicious patches, without requiring additional training or labeled data. Experiments show strong performance (over 95% accuracy) across diverse datasets and models, demonstrating good scalability and practicality.

### Strengths
* **Simple, well-motivated, and intuitive design.** The combination of retrieval and generative reasoning is conceptually clear and easy to understand.  
* **Training-free and highly scalable.** Since VRAG does not require model retraining, it can be easily applied to new architectures or domains.  
* **Strong detection performance.** The method achieves consistently high detection accuracy (>95%) across various types of adversarial patches, indicating strong robustness.

### Weaknesses
* **Lack of theoretical analysis.**
 Although the approach is intuitively sound, it lacks a theoretical explanation (e.g., embedding-space similarity or retrieval generalization analysis), making the contribution mainly empirical.  
* **Potential scalability issues.**
 As the adversarial patch database grows, redundancy, embedding drift, and increased retrieval latency may arise, potentially limiting large-scale deployment.  
* **High prompt dependency.**
 The method’s performance varies significantly with different prompt templates, raising the question of whether powerful multimodal LLMs combined with optimized prompting could directly detect adversarial patches.  
* **Limited task scope.**
 The work focuses solely on patch detection without discussing post-detection handling, such as patch removal, correction, or interpretability.

### Questions
1. **Provide theoretical grounding.**
 Include an analysis of why retrieval-based embedding alignment and semantic reasoning enable generalization to unseen patches.  
2. **Evaluate large-scale scalability.**
 Report results on larger databases (e.g., >10k entries) to study the trade-off between retrieval efficiency and detection performance.  
3. **Reduce prompt sensitivity.**
 Explore automatic or learned prompt optimization methods (e.g., reinforcement or gradient-based prompt tuning) to enhance generalization across models.  
4. **Expand the research scope.**
 Consider extending VRAG to patch reconstruction or defense-integration tasks to demonstrate its broader practical potential.

### Soundness
2

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
- This paper presents a training-free Visual Retrieval-Augmented Generation (VRAG) pipeline for adversarial patch detection in imagese. The methods first precomputes a database of patch and region embeddings by placing diverse adversarial patches on natural images. The model then retrieves top-k visually similar entries per image region, and uses VLM prompting to decide whether the query image contains an adversarial patch. This method is a training-free method that uses few/zero-shot adversarial examples to produce structured prompts by imxing the retrieved patches and attacked images.

### Strengths
- Clear, modular pipeline: The method is conceptually simple yet effective—grid embeddings → retrieval → VLM reasoning—making it appealing for practical deployment without finetuning. Algorithms and figure flow are easy to follow
- Training-free defense: Avoids re-training or fine-tuning classifiers/segmenters, addressing a major pain point in maintaining robustness against evolving patch styles
- Broad empirical sweep: Evaluation spans two datasets (ImageNet-Patch for synthetic, and APRICOT for real-world), four backbones (ResNet-50, ResNeXt-50, EfficientNet-B0, ViT-B/16), and multiple VLMs (open/closed)

### Weaknesses
- Selection of optimal threshold: while the authors state that the optimal threshold has been selected at 0.77 for the cosine similarity score and that scores nearing 1.0 makes the retrieval overly permissive, it is somewhat confusing whether higher similarity threshold enforces stricter selection of candidates or looser. I suggest the authors include results of TPR/FPR vs. threshold across dataset
- The authors claim that the 0-shot variant is competitive with the baselines, yet DIFFender achieves higher 0-shot accuracies than the VRAG variants in Table 1, and often the retrieval-only baseline is strong
- the proposed method requiring access to *representative* patch database remains to be a key limitation, restricting performance to that database. While the authors explained briefly the potential failure modes in L1018, more in-depth analysis on novel/unknown adversarial patch generalization tests with visualization would help strengthen the discussion.
- it would help to elaborate and show visualizations of synthetic and real samples (ImageNet-Patch, APRICOT) to demonstrate the types of adversarial patches created in the database and those evaluated on each dataset

### Questions
- Please clarify the statement that τ→1.0 becomes “permissive”; provide FPR/TPR vs tau plots and explain chosen threshold across datasets
- How do you prevent overlap/near-duplicate contamination between the retrieval database and evaluation sets?
- What happens if an attacker minimizes cosine-similarity to your DB while keeping the patch effective, or targets the VLM?
- How does accuracy vary with DB size, k-shots, and grid granularity?
- Can you report absolute throughput (image/sec, etc.), latency per image, and cost ($) for open-source vs. closed-source LLMS under a standard hardware profile?
- How fragile is the model on different query prompts for the VLM?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a training-free Visual Retrieval-Augmented Generation(VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types-all without additional training or fine-tuning.

### Strengths
1. It constructs a training-free retrieval-based pipeline.
2. Experimental results show that the retrieval-augmented detection approach achieves state-of-the-art detection across threat scenarios of the synthetic and real-world patch benchmark.

### Weaknesses
There are some points that I am confused about:
1. The author claims that this method can handle a wide variety of adversarial patch attacks. Does this “variety” refer to different patch shapes or something else? If so, the experimental section should include results on detection accuracy against diverse adversarial patch types to substantiate this claim.
2. In Tables 1 and 2, the compared methods include both defense-based and detection-based approaches. I am not entirely clear how the defense methods produce the metric "accuracy" reported in the tables. Since defense methods typically aim to restore classification performance rather than explicitly detect adversarial samples, could the authors clarify how the detection accuracy was computed for these methods?
3. In Table 2, it is not explicitly stated which dataset was used for evaluation. I would also like to understand the relationship and distinction between the dataset used to construct the patch database and the dataset used for testing. Were they chosen simply because one contains synthetically generated adversarial patches and the other is a real-world patch benchmark?
4. How is the method's ability to generalize to new patch attack types demonstrated?
5. Why does the similarity criterion become overly permissive when the threshold approaches 1.0? Could you please provide some explanations and an ablation study for the threshold?

### Questions
Apart from the issues mentioned in the Weaknesses section, I have other questions:
1. Why did the authors include only Gemini among the closed-source models, instead of also evaluating models from the GPT or Claude series?
2. Regarding the results shown in Figure 4, I would like to know whether, when the input is an adversarial sample, two corresponding patches and their associated images are retrieved from the database, resulting in a total of five images being fed into the model. In that case, is the reported inference time still only 2-3 seconds? Does this inference time include the retrieval process?

### Soundness
2

### Presentation
2

### Contribution
2
