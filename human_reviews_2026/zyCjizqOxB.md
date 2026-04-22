# Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Vision-language models (VLMs) are increasingly replacing traditional OCR pipelines. However, they often hallucinate on lossy visual inputs, such as visually degraded document images, producing fluent yet incorrect text without signaling uncertainty. This occurs because current post-training emphasizes accuracy, which encourages models to guess even when uncertain. The problem persists in state-of-the-art systems and severely impacts OCR reliability. To improve the trustworthiness of OCR on degraded documents, we propose uncertainty-aware OCR. Rather than suppressing guesses, our model transcribes while explicitly bracketing spans it deems unreliable with uncertainty tags. To train our model, we use Group Relative Policy Optimization (GRPO). We define usage rules for uncertainty tags and an evaluation protocol, introducing a pseudo-labeled cold start and a multi-objective reward that balances transcription accuracy and uncertainty coverage while preventing reward hacking. We explore different combinations of cold-start and reward granularity. We also assess the effect of reward parameters in preventing reward hacking and improving the corresponding metrics. Furthermore, we introduce Blur-OCR, a challenging benchmark for uncertainty-aware OCR on degraded document images under lossy visual conditions. In extensive experiments, our model maintains transcription accuracy while achieving an uncertainty tag F1 score of 0.685.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces uncertainty-aware OCR, where vision-language models transcribe degraded documents while bracketing uncertain spans with explicit uncertainty tags (`<C>...</C>`). The authors employ a pseudo-labeled cold start followed by Group Relative Policy Optimization (GRPO) with a multi-objective reward that balances transcription accuracy and uncertainty coverage. They introduce Blur-OCR, a benchmark of 2,048 synthetically degraded images from Project Gutenberg. The best model (Qwen2.5-VL-7B) achieves word-level F1 of 0.685 for uncertainty tagging and 0.839 accuracy, outperforming several baseline models including GPT-4o and Claude-Opus4.

However, the limited evaluation scope, missing key baselines, and conceptual concerns about the cold start procedure prevent a stronger recommendation. With revisions addressing the generalization questions and including comparisons with MinerU systems and uncertainty quantification methods, this could become a solid contribution.

### Strengths
**Clear problem formulation**: The paper addresses a real problem—VLM-based OCR systems hallucinate on degraded documents without signaling uncertainty, which is worse than classical OCR systems that produce obviously garbled output. The motivation is well-articulated.

**Systematic methodology**: The two-stage training approach (pseudo-labeled cold start + GRPO) is reasonable and well-described. The multi-objective reward design with safeguards against reward hacking (especially the length-mismatch damping factor η) demonstrates careful engineering.

### Weaknesses
### 1. Limited Benchmark Coverage and Missing Baselines

The evaluation is restricted to a single synthetic benchmark (Blur-OCR). The paper does not evaluate on:
- Established document understanding benchmarks like OmniDocBench [1], which provides diverse real-world PDF documents with comprehensive annotations
- More general OCR benchmarks beyond the two mentioned in Related Work (OCRBench/OCRBench v2)
- Recent document parsing systems like MinerU [2] or MinerU2.5 [3], which represent state-of-the-art in document content extraction

### 2. Incomplete Related Work on Uncertainty Quantification

The paper misses critical recent work on OCR uncertainty estimation. Notably, it does not cite or compare with methods that provide quantitative uncertainty measures. For instance, recent work on consensus entropy for multi-VLM agreement [4] provides token-level uncertainty scores that can be directly compared with the proposed tagging approach. While the paper mentions entropy-based baselines briefly in Exp4 (Section 7.4), it lacks:
- Proper contextualization within the broader uncertainty quantification or calibration literature
- Discussion of how the proposed explicit tagging approach differs from or improves upon probabilistic uncertainty measures

### 3. Conceptual Issues with the Cold Start Procedure

The pseudo-labeling strategy has a fundamental problem: it tags the **model's own errors** on degraded images, not necessarily the **visually unreadable regions**. This conflates two distinct phenomena:

a) Text that is visually degraded/unreadable in the image  
b) Text where the model happened to make a mistake

The paper claims (Section 5.1) that "When the image is unreadable, models tend to guess and are often wrong," but this assumption is not validated. Two problematic cases arise:

- **False positive tags**: The model may correctly transcribe text from a degraded-but-readable region, yet the cold start labels it as uncertain simply because it differs from GT due to actual degradation differences
- **False negative tags**: The model may confidently hallucinate on clear, undegraded regions (a known VLM failure mode [5]), which would not receive uncertainty tags


### 4. Limited Analysis of Generalization Beyond Synthetic Degradations

VLMs are known to hallucinate even on clean, high-quality document images [5]. The paper does not:
- Test whether the uncertainty-aware model can identify hallucinations on non-degraded documents
- Evaluate on real-world degraded documents (e.g., historical documents, which are mentioned in the motivation but never tested)
- Compare performance on different types of errors: character substitutions vs. hallucinated words/phrases

### 5. Benchmark Construction Concerns

The Blur-OCR benchmark applies random combinations of degradations to clean text, but:
- No analysis is provided on whether the degradations are realistic compared to actual historical documents or real-world low-quality scans
- The paper does not discuss the distribution of degradation severity or provide statistics on what fraction of text becomes truly unreadable
- Figure 2 shows sample pages, but there is no quantitative analysis of degradation characteristics

This makes it difficult to assess whether Blur-OCR represents realistic use cases or is primarily useful for evaluating this specific training paradigm.

### Questions
1. **Generalization to real degradations**: Can the authors evaluate on real-world degraded documents (e.g., historical document datasets) to demonstrate that the approach generalizes beyond the specific synthetic degradation pipeline?

2. **Comparison with MinerU systems**: MinerU and MinerU2.5 [2,3] represent recent advances in document parsing. How does the proposed method compare against these systems on Blur-OCR? If these systems cannot produce uncertainty estimates, can they be combined with the proposed tagging approach?

3. **OmniDocBench evaluation**: OmniDocBench [1] provides diverse document types. Can the authors evaluate whether uncertainty tagging helps on this more realistic benchmark?

4. **Quantitative uncertainty measures**: How does explicit binary tagging compare with continuous uncertainty scores (e.g., consensus entropy [4], token-level entropy, or multi-model voting confidence)? Can the authors provide experiments showing when discrete tags are preferable to probabilistic scores?

5. **Hallucination on clean images**: Does the trained model successfully tag hallucinations that occur on non-degraded, high-quality document images? This would demonstrate that the model learns genuine uncertainty rather than merely memorizing the training degradation distribution.

6. **Cold start validation**: Can the authors provide analysis showing that the pseudo-labels in the cold start actually correspond to visually degraded regions? For example, human annotations on a sample of cold-start labels, or correlation analysis between degradation strength and tagging frequency.

7. **Breakdown by error type**: What types of errors are well-covered by UNC tags vs. those that escape detection? Are character-level substitutions easier to catch than word-level hallucinations?


## References

[1] Linke Ouyang, Yuan Qu, Hongbin Zhou, et al. OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations. arXiv:2412.07626, 2024. (CVPR’25)

[2] Bin Wang, Chao Xu, Xiaomeng Zhao, et al. MinerU: An Open-Source Solution for Precise Document Content Extraction. arXiv:2409.18839, 2024.

[3] Junbo Niu, Zheng Liu, Zhuangcheng Gu, et al. MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing. arXiv:2509.22186, 2025.

[4] Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR

[5] Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, and Edwin Zhang. Why language models hallucinate. arXiv:2509.04664, 2025.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents an uncertainty-aware fine-tuning framework for OCR-capable large language models (LLMs).
Instead of producing overconfident transcriptions for visually degraded documents, the model is trained to explicitly mark uncertain spans with specific tags.
The approach combines two stages: (1) a cold-start supervised fine-tuning (SFT) phase using pseudo uncertainty labels automatically derived from model errors, and (2) Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm that jointly optimizes transcription accuracy and uncertainty tagging quality using a reward function balancing edit distance and F-beta-based span precision–recall.
Experiments on the new Blur-OCR benchmark demonstrate that GRPO improves both transcription correctness and uncertainty calibration compared to the cold-start baseline, while avoiding degenerate behaviors such as excessive tagging.

### Strengths
1. The problem is clearly motivated — OCR hallucination is a realistic and underexplored setting for uncertainty estimation.
2. The proposed explicit uncertainty tagging paradigm is conceptually simple yet effective.
3. The GRPO objective is well designed, balancing transcription accuracy and tagging F1, and preventing pathological behaviors via the reward-damping term.
4. The evaluation setup is thorough: they report both accuracy and uncertainty metrics and analyze different training stages.
5. The paper is very clearly written and easy to follow, with transparent motivation and mathematical detail.

### Weaknesses
1. Experiments are limited to a single OCR model family. It remains unclear whether the proposed method generalizes to other setups (e.g., multi-modal vision-language models such as Donut or TrOCR).
2. Comparison baselines are relatively narrow — no direct comparison with alternative uncertainty modeling approaches such as entropy-based rejection or calibration.
3. There is limited qualitative analysis of false-positive tags (over-tagging). Some visual examples or error breakdowns could strengthen interpretability claims.

### Questions
1. Did the authors try training with other values of λ — how sensitive are results to this hyperparameter?
2. How would the approach behave in non-OCR settings, such as code transcription or speech recognition?
3. Did you observe any tendency for the model to under-tag uncertainty after GRPO fine-tuning?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work tackles the problem of hallucinations for OCR for vision-language models. The models hallucinate when they are provided with blurry documents. This work proposed to use Reinforcement learning - GRPO to tackle this problem where the model is made to answer uncertainty tags along with transcription. They construct a multi-objective reward that balances accuracy with uncertainty and also mitigates reward hacking. They also provide a benchmark to measure uncertainty aware OCR performance on degraded documents and show that their method outperforms baselines.

### Strengths
- Uncertainty estimation has been a long studies problem but this work studies it in the context of OCR which is an important problem and the technique of using RL to is interesting. 
- The work discusses the importance of cold-start SFT, describes in detail their reward formulation, character vs word level tagging, different hyperparameters and provide comprehensive experiments.

### Weaknesses
- The benchmark that they introduce has synthetic degradations. It is unclear how much the results transfer to actual degradations found in practice.
-  The paper has nice set of experiments but lacks some intuitions examples as detailed below.

### Questions
- The authors discuss tag validity and alignment around lines 113. I don’t fully understand how the GT segments which are not inside y^hat work. It would be nice to clarify with an example.
- In lines 370, the authors discuss the tradeoff of using character level vs word level granularity for tags. Could the authors provide an example along with the text? This would make it easier to follow. What do the authors mean by character level rewards?
- Can the authors compare their method to some baselines used in the uncertainty literature like softmax probabilities, entropy etc?

### Soundness
3

### Presentation
2

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
This paper focuses on a task: when performing OCR, a VLM should mark uncertain and hard-to-recognize spans by surrounding them with a custom UNC tag. To enable this, the authors build a training set of about 100K samples and a 2K-sample benchmark. For training, they use two phases — SFT for warm-start, followed by RL.

### Strengths
1. The writing is clear, and the presentation of different settings is easy to follow.

2. Introducing uncertainty-aware generation in OCR to mark unclear spans has practical value.

3. The experiments include extensive ablation study, which helps clarify the effectiveness of the method, and I appreciate that.

### Weaknesses
1. Minor: Although admitting uncertainty in OCR has some practical value, on the other hand, the broader significance is also limited since the work focuses on a specific application setting.

2. The backbone choice is quite limited.

3. The benchmark is constructed by the authors, while appreciated, i also want to know how well the method (and models) generalizes to more OOD scenarios.

4. Minor: I think the paper should use “VLM” instead of “LLM.”

5. While I appreciate the detailed metric section and the following ablation studies, it feels somewhat over-emphasized. The data component should likely deserve more effort than defining the metrics.

### Questions
is it possible to do semi-supervised learning on this task since uncertainty can also be generated by LLM itself (although it may not be so well-calibrated)?

### Soundness
2

### Presentation
3

### Contribution
2
