# Improving GUI Grounding with Explicit Position-to-Coordinate Mapping

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
GUI grounding, the task of mapping natural-language instructions to pixel coordinates, is crucial for autonomous agents, yet remains difficult for current VLMs. The core bottleneck is reliable patch-to-pixel mapping, which breaks when extrapolating to high-resolution displays unseen during training. Current approaches generate coordinates as text tokens directly from visual features, forcing the model to infer complex position-to-pixel mappings implicitly; as a result, accuracy degrades and failures proliferate on new resolutions. We address this with two complementary innovations. First, RULER tokens serve as explicit coordinate markers, letting the model reference positions similar to gridlines on a map and adjust rather than generate coordinates from scratch. Second, Interleaved MRoPE (I-MRoPE) improves spatial encoding by ensuring that width and height dimensions are represented equally, addressing the asymmetry of standard positional schemes. Experiments on ScreenSpot, ScreenSpot-V2, and ScreenSpot-Pro show consistent gains in grounding accuracy, with the largest improvements on high-resolution interfaces. By providing explicit spatial guidance rather than relying on implicit learning, our approach enables more reliable GUI automation across diverse resolutions and platforms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces two methods—RULER tokens and I-MRoPE—to enhance the visual grounding capability of MLLMs. The approach is intuitive and theoriatical. Experimental results on mainstream benchmarks show consistent gains in grounding accuracy, suggesting that the proposed techniques are effective. However, the underlying motivation is not sufficiently developed, and the paper lacks deeper analysis or interpretability studies to clarify why these methods lead to improved localization.

### Strengths
1. The paper proposes a more intuitive visual grounding strategy by positional encoding from the perspective of spatial awareness, which offers a clearer interpretation of how models localize objects.

2. The authors identify inherent limitations in existing positional encoding schemes and present targeted enhancements that effectively improve localization capability.

3. Extensive experiments on established benchmarks demonstrate the empirical effectiveness of the proposed methods.

### Weaknesses
1. The paper lacks a compelling and intuitive research motivation, making the introduction of the two proposed strategies feel abrupt and insufficiently grounded in the broader challenges of multimodal grounding. Stronger design intuition and pre-experiments would help clarify why these particular techniques are necessary and meaningful. This issue is especially pronounced for I-MRoPE, where the rationale behind the method is under-developed and its connection to real-world grounding failures remains unclear.

2. The experimental findings feel somewhat superficial and lack sufficient breadth. First, recent advances in grounding are rapidly evolving, and the paper does not fully situate its results within the latest leaderboard trends or analyze whether the proposed strategies meaningfully reduce existing performance gaps. Second, there are methods that outperform the proposed approach, yet the paper does not investigate the reasons behind these differences, limiting the reader’s understanding of the technique’s strengths and weaknesses. Third, the evaluation on MLLM foundation models remains limited. For example, newer multimodal systems such as mini-CPM exhibit strong grounding ability, would they experience the same deficiencies identified here, and would the proposed strategies still lead to improvement? 

3. The results in Table 3 indicate that the proposed strategy does improve the localization capability of MLLMs; however, the contribution of I-MRoPE appears marginal. This raises questions about the necessity and justification for combining the two methods as presented. Additionally, fine-tuning results show notable gains, yet the paper lacks interpretability analysis, such as attention behavior or grounding heatmaps, to substantiate how and why these improvements occur. Without such evidence, the effectiveness of each component and the underlying mechanism behind the observed performance changes remain insufficiently explained.

4. Grounding enhancement is often motivated by its impact on downstream tasks such as GUI tasks (AndroidControl, AndroidWorld, OSWorld). The current experimental setup does not demonstrate whether the proposed improvements translate into measurable gains in these more realistic use cases. Additional downstream evaluations would therefore significantly strengthen the practical relevance and efficacy of the proposed approach.

5. It would be valuable to investigate scaling behavior with respect to model size and data volume. Establishing scaling laws could provide deeper insight into the effectiveness and limitations of the proposed strategies, and help clarify whether improvements persist as models and datasets grow.

6. The paper would benefit from a more intuitive, step-by-step case study to clarify the end-to-end workflow of the proposed method. In particular, it would be helpful to illustrate how RULER tokens are extracted, how the I-MRoPE module is instantiated, and how these components jointly contribute to the final grounding prediction. Such a detailed example would greatly improve the accessibility and interpretability of the approach.

7. Since the proposed approach introduces additional tokens, it would be valuable to provide a more intuitive analysis of the computational implications, for example by measuring the impact on inference latency. 

8. What advantages does this approach offer, compared to directly providing the object locations from OCR parsing and analysing the layout via CoT, by offering additional image token references?

### Questions
See above

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
4

### Summary
This paper tackles the failure of VLMs in GUI grounding, specifically their inability to generalize to high-resolution displays unseen during training. The authors identify the root cause as the model's reliance on implicitly regressing coordinates from visual features. They propose two innovations: 1) RULER tokens, which provide explicit coordinate references to transform grounding into a robust "reference-and-adjust" mechanism, and 2) I-MROPE, a balanced positional encoding that corrects frequency bias.

### Strengths
1. The paper excels at identifying a critical and well-defined weakness in existing models (the instability of implicit coordinate regression). The proposed RULER token mechanism is an intuitive solution
2. The method shows its significant gains on the SS-Pro, which features high-resolution displays and a domain shift from the training data. This provides strong evidence for the authors' core claim that explicit coordinate referencing is more robust than existing works.

### Weaknesses
1. The paper's comparison in Table 1 is a significant weakness. The authors' best from-scratch model (32.1%) significantly underperforms established baselines like UI-TARS-7B (35.7%) and GUI-Actor-7B (44.2%). While the tuned model (37.2%) is more competitive, it still trails GUI-Actor. The authors attribute this to differences in training data, but this is a critical point that need further explanations.
2. The core concept of this paper seems to be a variation of explicit tokens or coord embeddings that have been explored in other visual grounding (namely gui-actor) or object detection contexts.
3. Minor: The paper's strength lies in framing it as a "reference-and-adjust" mechanism inspired by induction heads, but it lacks direct analysis (e.g., attention probing) to prove the model actually learns this mechanism rather than simply using the tokens as a stronger signal.
4. Please refer to my question.

### Questions
1. The models are trained *exclusively* on the UGround dataset (web) but evaluated on SS-Pro, which features professional desktop applications (CAD, Scientific, etc.). The authors frame this as a test of generalization, but it introduces a massive domain / resolution shift?
2. The ablation study in Table 1 suggests that the primary performance gain comes from RULER tokens, not I-MROPE. The performance difference between `LLaVA-NeXT + MROPE` (29.2) and `LLaVA-NeXT + I-MROPE` (29.4) is marginal. So, does that mean I-MROPE is an incremental, minor fix that adds complexity for a negligible benefit, and that this work should focus almost entirely on RULER as the core contribution?

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
3

### Summary
This paper proposes to addresses the problem of patch-to-pixel mapping in the GUI grounding task. The key contributions of the paper lie in (1) proposing RULER tokens to function as explicit coordinate markers; (2) proposing Interleaved MRoPE (I-MRoPE) to address the asymmetry of standard positional schemes. The experiments on three benchmarks illustrate the effectiveness.

### Strengths
1. This paper is well motivated, focusing on a novel problem in grounding tasks.
2. The proposed methods RULER and I-MRoPE are interesting and makes sense.

### Weaknesses
1. It is unclear whether the proposed method can generalize to other grounding tasks beyond GUI scenarios. Also, how does the method perform when applied to GUI Agent tasks (e.g., AndroidWorld, WebArena) ?
2. It lacks experiments on whether training with these methods can affect the performances on the general ability of VLMs.
3. GUI-Actor also tackles on the similar research question while RULER fails to give comparable or superior performances. Also, the comparisons on ScreenSpot series benchmarks lack many recent strong baselines. These weaknesses question whether the proposed method is truly effective and competitive ?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
GUI grounding, the task of mapping natural-language instructions to pixel coordinates, is crucial for autonomous agents, yet remains difficult for current VLMs. The core bottleneck is reliable patch-to-pixel mapping, which breaks when extrapolating to high-resolution displays unseen during training. Current approaches generate coordinates as text tokens directly from visual features, forcing the model to infer complex position-to-pixel mappings implicitly; as a result, accuracy degrades and failures proliferate on new resolutions. We address this with two complementary innovations. First, RULER tokens serve as explicit coordinate markers, letting the model reference positions similar to gridlines on a map and adjust rather than generate coordinates from scratch. Second, Interleaved MRoPE (I-MRoPE) improves spatial encoding by ensuring that width and height dimensions are represented equally, addressing the asymmetry of standard positional schemes. Experiments on ScreenSpot, ScreenSpot-V2, and ScreenSpot-Pro show consistent gains in grounding accuracy, with the largest improvements on high-resolution interfaces. By providing explicit spatial guidance rather than relying on implicit learning, our approach enables more reliable GUI automation across diverse resolutions and platforms.

### Strengths
1) The topic is interesting
2) The writing is good
3) The experiments show the effectiveness of the proposed method.

### Weaknesses
1) The introduction of new tokens and spatial encoding methods adds complexity to the model architecture, which may require more resources and time for development and training.
2) Dependency on Training Data: While the method improves performance on high-resolution displays, it may still be limited by the quality and diversity of the training data used, potentially impacting performance on very novel interfaces.
3) Potential Overfitting: The focus on explicit mappings might lead to overfitting on specific tasks or resolutions, limiting the model's adaptability to other types of GUI environments not represented in the training data.

### Questions
1) The introduction of new tokens and spatial encoding methods adds complexity to the model architecture, which may require more resources and time for development and training.
2) Dependency on Training Data: While the method improves performance on high-resolution displays, it may still be limited by the quality and diversity of the training data used, potentially impacting performance on very novel interfaces.
3) Potential Overfitting: The focus on explicit mappings might lead to overfitting on specific tasks or resolutions, limiting the model's adaptability to other types of GUI environments not represented in the training data.

### Soundness
2

### Presentation
2

### Contribution
2
