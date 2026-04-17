# Forging Image Watermarks by Reversing Watermark Removal Attacks

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Image generative models have accelerated the need for robust image watermarking to track and verify AI-generated images. While watermark removal attacks have been extensively studied, the threat of watermark forgery, where benign images are maliciously modified to appear watermarked, remains underexplored, especially in the no-box setting. In this work, we introduce WForge, a no-box and query-free forgery attack that reframes forgery as the inverse of removal. Our key insight is that residual perturbations from removal attacks approximate watermark signals and can be repurposed to forge watermarks. Concretely, we train a forger network to learn the pattern of residuals and apply it to unwatermarked images, making them falsely detected as watermarked. We evaluate WForge across three datasets and four state-of-the-art watermarking methods, demonstrating that it consistently outperforms existing forgery baselines. Our results further reveal a critical vulnerability: the existence of a successful removal attack implies the feasibility of forgery for the same watermarking method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes WForge, a no-box, query-free image watermark forgery method that conceptualizes forgery as the inverse of watermark removal. The core idea is that residual perturbations from watermark removal attacks approximate the underlying watermark signal; by learning these residuals through supervised training, a forger network can add imperceptible perturbations to non-watermarked images so that they are falsely detected as watermarked. Researchers process experiments across multiple datasets, and watermarking methods demonstrate that WForge significantly outperforms prior forgery baselines.

### Strengths
1. Good Paper Writing: The paper is exceptionally well-written, featuring a clear structure and a detailed description of the methodology. Furthermore, the authors provide sound theoretical arguments to justify the statistical properties of their detection mechanism. 

2. Conceptual Novelty: The straightforward insight that watermark removal and forgery are inverse processes is elegant and leads to a practical, effective attack framework. The authors made extensive experiments on multiple watermarking schemes and datasets to substantiate the method’s generality;

3. Security Relevance: The discovery that removal attacks imply forgery vulnerability is an important and underexplored insight with implications for AI content authenticity frameworks.

### Weaknesses
1. Some watermarking techniques used to be forged are outdated, particularly the content-dependent methods, which were developed 5 years ago (e.g., StegaStamp (CVPR 2020) and RivaGAN (arXiv 2019)). It remains uncertain whether the proposed method can effectively counter modern post-processing content-dependent watermarks, such as TrustMark (ICCV 2025) and VINE (ICLR 2025). To ensure a more comprehensive evaluation, experiments involving at least VINE should be included in the experiences. (p.s., I totally understand this may be a lot to run, but some small but convincing experiments would really help. I will consider significantly raising the score if more supportive data about the latest effectiveness can be provided.

2. Potential Concerns about Removal Methods: Since the forger is trained on residuals from a specific remover, its generalization to unseen watermarking-removal pairs may be overstated (e.g., weakness 1 supplementary experiment fails). Besides, the proposed method relies significantly on the availability of effective watermark removal attacks (e.g., VAE-based removers) to generate training residuals. However, if removal is weak or unavailable for certain watermarks, WForge's performance could degrade significantly. 

3. Unfinished `Appendix A.4 Observation`: has no contents but only a title.

### Questions
Has the author considered combining the results of multiple watermark removal algorithms to create a more comprehensive training dataset? This approach may learn richer prior representations of watermark features, and enhance adaptability when encountering entirely new and unknown watermark paradigms.

Are there any suggestions for the next steps in the development of watermark forging from the perspective of image modality?

### Soundness
3

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
2

### Summary
This paper studies how to perform watermark forgery attacks in a no-box and query-free setting by leveraging reverse watermark removal attacks. Concretely, the authors investigate whether one can reverse an existing watermark-removal method and use that to train a forger network that produces forged watermarks without access to the original watermarking pipeline or queries to it.

### Strengths
1. The problem of forging watermarks in a no-box, query-free setting is realistic and the threat model is reasonable—practical and worthy of study.

2. The proposed approach is well-motivated. Empirical results indicate that the method can work under the evaluated settings, supporting the paper’s claims of feasibility.

### Weaknesses
1. The paper lacks a main flowchart or an algorithm diagram in Section 4 that clearly lays out the end-to-end pipeline. A visual summary or pseudocode would greatly improve clarity.

2. The approach appears heavily dependent on the effectiveness of the underlying watermark-removal method. If the removal step fails or is weak, subsequent training of the forger network may be unreliable. The manuscript does not sufficiently explain how the reliability of the removal stage is ensured, nor quantify how removal quality affects downstream forgery. This raises concerns about strong implicit assumptions on the attacker’s knowledge or capabilities.

3. The experimental set of watermarking methods is limited, which weakens claims about generality. It is unclear whether the approach generalizes across diverse watermark schemes (e.g., FNNS, TrustMark, and other modern methods).

4. The paper does not investigate robustness to unknown watermark parameters (e.g., strength, spatial placement, embedding hyperparameters). It is unclear whether the method can forge or remove watermarks when these parameters vary or are unknown.

### Questions
Does the method require any implicit or explicit knowledge about the watermark (e.g., type, strength, embedding scheme) to succeed? If so, please clarify these assumptions and discuss their realism in your threat model.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes WForge, a method for forging content-dependent image watermarks. WForge first leverages a generative watermark-removal model to obtain a clean image from a watermarked one, and then extracts the watermark residual by taking their difference. It subsequently trains a network on large-scale data to predict watermark residuals for arbitrary images, thereby generalizing the residual estimation process. By adding the predicted residuals back to clean images—following watermark steganalysis principles—the method successfully forges content-dependent watermarks. Experimental results show that WForge achieves higher attack success rates than the baseline methods Steganalysis and Watermark Faker on content-dependent watermarking schemes.

### Strengths
1. The paper is interesting and technically insightful. The idea of obtaining watermark residuals through a generative removal process and then learning a generalized residual predictor is creative. The authors effectively extend watermark steganalysis techniques to the more challenging setting of content-dependent watermarks.

2. Extensive ablation studies are conducted to validate the effectiveness of the proposed method.

### Weaknesses
1. The optimal configuration (i.e., choice of watermark removal method and residual predictor model) appears to vary across different watermarking algorithms. In real-world scenarios, since the watermark decoder is inaccessible, the forgery performance is uncertain, making it difficult to determine an appropriate configuration in practice.

### Questions
For the Success Rate metric, it is unclear why the authors chose the threshold value τ = 0.8. Typically, τ is selected based on the False Positive Rate (FPR). What is the corresponding FPR for this setting?

### Soundness
3

### Presentation
3

### Contribution
3
