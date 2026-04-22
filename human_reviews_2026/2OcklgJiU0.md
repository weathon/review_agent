# Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Selective prediction aims to endow predictors with a reject option, to avoid low confidence predictions. However, existing literature has primarily focused on closed-set tasks, such as visual question answering with predefined options or fixed-category classification. This paper considers selective  prediction for visual language foundation models, addressing a taxonomy of tasks ranging from closed to open set and from finite to unbounded vocabularies, as in image captioning. We seek training-free approaches of low-complexity, applicable to any foundation model and consider methods based on external vision-language model (VLM) embeddings, like CLIP. This is denoted as $\textit{Plug-and-Play Selective Prediction} (\textbf{\texttt{PaPSP}})$. We identify two key challenges: (1) $\textit{instability of the visual-language representations}$, leading to high variance in image-text embeddings, and (2) $\textit{poor calibration of similarity scores}$. To address these issues, we propose a $\textit{memory augmented}$ $\textbf{\texttt{PaPSP}}$ ($\textbf{\texttt{MA-PaPSP}}$) model, which augments $\textbf{\texttt{PaPSP}}$ with a retrieval dataset of image-text pairs. This is leveraged to reduce embedding variance by averaging retrieved nearest-neighbor pairs and is complemented by the use of contrastive normalization to improve score calibration. Through extensive experiments on multiple datasets, we show that $\textbf{\texttt{MA-PaPSP}}$ outperforms $\textbf{\texttt{PaPSP}}$ and other selective prediction baselines for selective captioning, image-text matching, and fine-grained classification. Source code will be made public.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MA-PaPSP, a memory-augmented plug-and-play method for selective prediction in VLMs. It aims to improve upon standard PaPSP (e.g., CLIP-Score) by addressing its representation instability and poor score calibration. The method retrieves $k$-nearest neighbors from a dataset to create a more stable "proxy embedding" and uses a contrastive score for better calibration.

### Strengths
* The paper addresses a significant and practical problem: enabling training-free, plug-and-play selective prediction for foundation models, particularly for challenging open-set tasks like image captioning.

* The experimental evaluation is comprehensive. The method is validated across a clear taxonomy of VLM tasks (classification, ITM, captioning) and tested with various predictive VLMs (P-VLMs) of different scales.

### Weaknesses
* SP-VLM as Bottleneck: I am skeptical about the core premise. The paper focuses on augmenting a relatively weak SP-VLM (like CLIP/SigLIP) to judge a much stronger P-VLM (like Qwen-2.5-VL). As P-VLMs advance, the SP-VLM (the "judge") inherently becomes the bottleneck. The proposed MA mechanism seems like a complex workaround for this weak judge. A critical missing baseline is the "LLM-as-judge" approach: using a state-of-the-art LVLM (e.g., Qwen3VL or an API-based model) via prompt-engineering to directly score the output.
* Limited Novelty: The core technical contribution of the memory augmentation (MA) appears to be limited in novelty. The use of $k$-NN retrieval and weighted averaging (Eq. 6) to create a "proxy embedding" is a well-known ensemble and smoothing technique used to enhance robustness and accuracy. While its application to PaPSP is new, the underlying mechanism itself is not sufficiently novel.

### Questions
Conclusion: While the problem is well-motivated and the experiments are thorough, the two major weaknesses regarding the method's core premise (using a weak judge) and its technical novelty prevent a clear recommendation for acceptance.

I am willing to reconsider my score if the authors can convincingly address these two points, especially by providing a strong benchmark against a powerful "LLM-as-judge" baseline.

### Soundness
2

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
3

### Summary
This paper presents a unified framework for addressing the selective prediction problem across various vision-language tasks, including classification, retrieval, and captioning. The goal is to derive a more reliable confidence score than the vanilla CLIP similarity score. To achieve this, the method first leverages neighborhood representations for arbitrary modalities to enhance representation stability, and then applies contrastive normalization to improve calibration.

### Strengths
The authors address the selective prediction problem, which aims to balance risk and coverage. This problem has been relatively underexplored in the context of vision-language models, particularly for open-set tasks such as captioning.

They propose a novel formulation that constructs neighborhood proxies using retrieval datasets, independent of the input or output modality. In addition, they present a unified framework that can be applied across different types of VLM tasks.

Through experiments, the authors further analyze the impact of retrieval datasets, which enhances the significance and credibility of their study.

### Weaknesses
The readability of this paper is unsatisfactory. Figure 1 is quite confusing; while it seems intended to illustrate the differences between classification and captioning in the proposed framework, the lines and arrows instead create misunderstanding. Figure 2 is difficult to interpret and contributes little to improving comprehension of the paper. Figure 4 is quite complex and lacks clear organization. The text in Figure 5 is too small, and the purpose of the figure itself is not clear enough.

In addition, the notations throughout the paper are generally hard to follow. Equation (6) is especially difficult to understand, possibly due to the ambiguous use of the symbols y and z. Overall, substantial revisions are needed before this work is ready for publication.

### Questions
Can the retrieval-based proxy proposed in this paper enhance open-set tasks in VLMs, such as captioning, beyond merely serving as a confidence score?

### Soundness
2

### Presentation
1

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
This paper introduces MA-PaPSP, a lightweight and training-free method for selective prediction in vision-language models. It addresses the instability and poor calibration of standard similarity scores by an external memory and a calibration algorithm.

### Strengths
1. Problem Formulation and Framework: The formulation of the Plug-and-Play Selective Prediction (PaPSP) problem for a taxonomy of VLM tasks, especially open-set scenarios like captioning, is timely and meaningful.
2. Thorough Analysis: The paper provides a comprehensive analysis that successfully identifies and validates the core challenges of representation instability and score miscalibration in baseline methods.
3. Strong Generalization: The proposed MA-PaPSP method demonstrates impressive generalization capabilities across a wide range of tasks including specialized domains, provided a suitable memory bank is available.

### Weaknesses
1. The paper's claim of being a "lightweight" solution is somewhat challenged by its reliance on a massive external memory bank (e.g., 15M image-text pairs). The computational overhead and storage cost of performing nearest-neighbor retrieval from such a large database during inference should be more thoroughly discussed. This overhead could impact the method's practical deployment in latency-sensitive or resource-constrained environments, and a comparison of inference time against baselines like LLM-as-Judge would strengthen this analysis.
2. Dependence on Memory Quality and Practical Limitations: Model performance is highly dependent on the domain relevance of the memory bank, particularly in open-set settings. The necessity of procuring a memory distribution that aligns well with the target domain at test-time presents a significant practical challenge. This dependency could limit the method's applicability in scenarios where such curated data is unavailable.

### Questions
Was any analysis conducted to quantify the distributional similarity between the memory banks and the target evaluation datasets? Such an analysis would greatly strengthen the claims about why certain memory banks outperform others and provide guidance for memory selection in practice.

### Soundness
3

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
3

### Summary
This paper addresses the problem of selective prediction (SP) for VLMs, extending it to open-set tasks like image captioning where the output space is unbounded. The authors propose a training-free, "plug-and-play" method called Memory Augmented Plug-and-Play Selective Prediction (MA-PaPSP). This approach uses an external VLM as a "judge" to score the output of a primary predictor VLM. To overcome the inherent instability and poor calibration of the judge VLM's embeddings, MA-PaPSP is augmented with a retrieval dataset. This memory is leveraged to compute more stable "proxy embeddings" via nearest-neighbor averaging and to derive well-calibrated "contrastive scores" by comparing the prediction against a set of hard negatives. The primary contribution is a practical, lightweight framework for enabling VLMs to abstain from low-confidence predictions without requiring any model retraining.

### Strengths
- The focus on creating a training-free, plug-and-play solution that is applicable to open-set tasks like captioning is a valuable direction for improving the safety and reliability of modern VLMs.
- While the constituent components (selective prediction, memory augmentation, contrastive scoring) are not new in isolation, their combination to address open-set SP for VLMs in a training-free manner is a simple and well-motivated approach.
- The proposed method is technically sound and is validated through a comprehensive set of experiments. The authors test their approach across a reasonable taxonomy of tasks and demonstrate clear improvements over baselines.

### Weaknesses
1. **Omission of Inference Cost Analysis:** The claim of being "light-weight" is not substantiated with empirical evidence. The method introduces a potentially computationally expensive k-NN search over a large-scale dataset (e.g., CC12M) and the storage requirements for the embeddings. The paper lacks analysis of latency, throughput, or memory footprint, which is a critical omission for a method proposed for practical application.
2. **Methodological Limitations and Dependencies:**
    1. **Performance is Fundamentally Capped by the Judge VLM:**  The method's ability to assess prediction quality is inherently limited by the semantic understanding of the external "judge" VLM (e.g., SigLIP). While the proposed techniques address the *structure* of the embedding space (instability, calibration), they cannot correct for the judge's fundamental semantic blind spots, such as failures in counting, fine-grained spatial reasoning, or attribute binding. This upper bound on performance is an important limitation that is not adequately discussed.

    2. **Ignores Predictor's Internal Confidence:** The method's design completely disregards any internal confidence signals from the predictor VLM. While this enables the plug-and-play functionality, it also discards a potentially valuable source of information. It is possible that a hybrid approach, combining the external MA-PaPSP score with the predictor's internal state (when accessible), could yield superior performance.
3. **Potential Fragility from Distribution Shift and Heuristics:**
    1. There is a potential distribution shift between the captions generated by modern, powerful VLMs (often long and detailed) and the typically short, noisy captions found in web-scrapped datasets like CC12M. This mismatch could compromise the quality of the proxy embedding.
    2. The effectiveness of the contrastive score relies heavily on the quality of the "hard negatives." The paper states these are generated via a rule-based or small LM approach, which appears heuristic and may be a source of fragility. The sensitivity of the method to this generation process is not explored.
4. **Disorganized Structure and Presentation:**
    1. The structure of the experiment section (Section 4) hinders readability. The mixing of dataset descriptions, baseline details, main results, and ablation studies makes the paper's empirical contributions difficult to follow.
    2. Minor but persistent presentation issues, such as very small font sizes in figures (especially Figure 2,3,4,5) hinders readability.

### Questions
1. Could you provide a quantitative analysis of the inference costs (e.g., latency in ms/sample, memory usage) introduced by MA-PaPSP, especially when using a large retrieval set like CC12M? A comparison to the baseline inference time of the predictor VLM would be very helpful to contextualize the "light-weight" claim.
2. The text distribution of captions from modern VLMs like Qwen-2.5-VL differs significantly from the web-scraped captions in CC12M. How robust is the proxy embedding calculation to this domain gap? Have you investigated whether this shift negatively impacts the performance for state-of-the-art predictor models?
3. The proposed framework seems difficult to apply directly to multiple-choice Visual Question Answering (VQA) tasks, where the model's output is often a single character (e.g., 'A', 'B') whose text embedding is semantically meaningless. Could you clarify how the method could be adapted for such a common VLM task, or is this considered outside the scope of the current work?
4. The decision to completely ignore the predictor VLM's internal confidence is a core design choice. Have you considered or experimented with scenarios where this signal is available (e.g., for non-black-box models)? Could combining the MA-PaPSP score with a measure of the predictor's internal confidence lead to further improvements?

### Soundness
3

### Presentation
3

### Contribution
3
