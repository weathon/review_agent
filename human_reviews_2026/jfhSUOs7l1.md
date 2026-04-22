# FOCUS: A Frequency-Oriented and Class-Underrepresented Semantic Segmentation Framework for Food Images

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Generating high-quality semantic segmentation results from food images remains a challenging task, particularly in the presence of complex boundaries and class imbalance. Existing methods often struggle with blurred edges and underperform on long-tailed categories, limiting their generalizability in practical scenarios. To address these issues, we propose FOCUS, a novel semantic segmentation framework designed to enhance boundary precision and improve rare class recognition. Specifically, we introduce a frequency-based strategy that selectively processes high-frequency components via differential convolution and integrates explicit edge supervision during training. This enables the model to better capture fine-grained boundary details and improves edge discriminability. To mitigate class imbalance, we introduce an enhanced gradient allocation mechanism that applies targeted matching supervision to rare categories, thereby amplifying learning signals for low-shot classes and improving classification accuracy. Extensive experiments on benchmark datasets, FoodSeg103, UECFoodComplete, and Food50Seg, demonstrate that FOCUS consistently outperforms existing approaches in both boundary quality and rare class performance, validating its architectural effectiveness and robust generalization capability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work presents FOCUS for food semantic segmentation. The authors address two challenges: complex boundary recognition and long-tail class distribution. Starting from a frequency perspective, they propose a Frequency-Aware Boundary Modeling mechanism to selectively process high-frequency components to capture discriminative edge features. They also introduce Gradient-Enhanced Underrepresented Class Matching to improve model learning for minority classes. Experiments demonstrate improved segmentation accuracy.

### Strengths
- The manuscript is well organized and clearly written. The figures are accurate and help readers quickly grasp the core idea.  
- The motivation is clear and compelling. Complex boundary recognition and long-tail class imbalance are long-standing challenges in semantic segmentation, and it is encouraging to see a study that addresses both.  
- Owing to the clear writing, the logical progression from the problem statement to the FOCUS module design is easy to follow.

### Weaknesses
- Missing discussion of closely related work. Although the INTRODUCTION section cites several prior works, the manuscript lacks a focused discussion and comparison with strongly related methods, particularly FADC and DEA-Net. No experimental comparison with FADC is provided.
- Insufficient and partly outdated baselines. Over half of the baselines are from before 2022. It is recommended to retain one or two classic baselines (e.g., FPN or DeepLabV3+) but prioritizing comparisons with recent state-of-the-art or mostly related methods. For example, CPT uses category prototypes as unified representations for same-class pixels and shows strong performance on object boundaries. A direct comparison with methods like CPT [r1] and FADC would strengthen the paper.
- Incomplete ablation studies. The manuscript introduces several hyperparameters, e.g., $\alpha$, $k$, $j$, $m$, etc, but provides no validation for them.

[r1] Quan Tang, et al. Rethinking Feature Reconstruction via Category Prototype in Semantic Segmentation, TIP 2025.

### Questions
**Two more questions**:
- In Fig. 3 (b) FABM, why does the lower branch contain one extra convolution operation?
- In Fig. 3 (b) FDB, I am curious about the effect of applying the weighting operation before versus after the differential convolution on segmentation results. Have the authors performed such experiments? Why not place the weighting at the position of Eq. (6)?

**Some suggestions**:
- The loss functions are not clearly described in the text. It appears to include an edge loss from Fig. 3. If so, please state this explicitly and include ablation experiments isolating the effect of that loss term.
- The definition of aAcc in Eq. (9) appears to be wrong. For clarity, correct the formula and highlight the most significant differences in Fig. 4.
- This work is built upon Mask2Former. We observe a notable accuracy improvement over Mask2Former, but what is the cost in terms of model parameters, FLOPs, and inference efficiency?

Overall, I have mixed feelings about this work. Although the paper has a clear motivation and an easy-to-understand network design, it lacks sufficient experimental studies and textual descriptions of related work and some necessary components such as loss functions. When I reached at the end of the paper, I even looked for an appendix. It feels more like a semi-finished product. Therefore, I currently rate it at 2-REJECT, and look forward to further discussion with the authors. Feel free to correct if there is any misunderstanding.

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
4

### Summary
The FOCUS framework proposed in this paper addresses two core challenges in semantic segmentation of food images—"blurred boundaries" and "poor performance on long-tailed categories" by innovatively combining frequency-domain processing and gradient-enhanced matching strategies. Its core consists of two key components:  the first is the Frequency-Aware Boundary Modeling (FABM) module, which splits feature maps into high-frequency and low-frequency components via Fast Fourier Transform (FFT), refines edge-rich high-frequency signals using differential convolution, and fuses them with weighted low-frequency features to enhance boundary precision; the second is the Gradient-Enhanced Underrepresented Class Matching (GEUCM) module, which defines underrepresented classes based on both category frequency and segmentation performance, and adopts a two-stage matching strategy (one-to-one for common queries and one-to-many for extra queries) to amplify gradient signals for long-tailed categories without compromising the learning of common classes.

### Strengths
Method design is targeted: FABM’s high-frequency refinement addresses food’s irregular edges, while GEUCM’s query allocation targets food’s long-tailed distribution.
Technical details are transparent: FABM’s frequency division and GEUCM’s matching algorithm are well-formulated.
Experiments are comprehensive: covering three datasets, multiple metrics, and ablation of key components, ensuring results are robust.

### Weaknesses
The paper unifies frequency modeling and underrepresented class learning under one framework (FOCUS), but the connection between these two ideas is superficial. It is recommended that the authors strengthen the explanation of the link between the two modules.
The frequency-aware design lacks a theoretical justification for why simple binary frequency masking plus difference convolutions should significantly enhance boundary features.
The parallel structure of FFT and multi-branch convolution increases computational overhead, yet the paper fails to report metrics such as FLOPs (Floating-Point Operations Per Second) or inference time.

### Questions
In the FABM module, only a simple binary separation of high and low frequencies is applied.  Why was the current frequency threshold ɑ chosen? How sensitive is the model performance to this threshold? Is there any theoretical or experimental justification for this choice?
In FABM, four types of difference convolutions — CDC, HDC, VDC, and ADC — are used simultaneously. How much additional parameter count and computational overhead does this multi-branch structure introduce?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a segmentation model specifically for food images. The model comprises two main design components: i) a frequency-based feature modelling using high/flow frequency maps calculated through FFTs; ii) a new matching method that prioritises higher gradient magnitudes for underrepresented classes, as determined by their low appearance frequencies within the dataset and the model’s low performance. The proposed method is evaluated on three food-related datasets and outperforms both food-specific and general segmentation methods.

### Strengths
The paper is well-written and presents compelling motivations for its method designs. The model is small and efficient, making it trainable on consumer-level GPUs. It outperforms both domain-specific and general segmentation methods.

### Weaknesses
Overall, I feel the biggest limitation of this paper is its scope and definition. It falls outside the standard paper styles and qualities of ICLR. The paper touches a very niche area of food segmentation, and the experiment settings and method designs seem incremental.

There are also some additional limitations:

- The paper ablated the contributions of each component but failed to provide design adjustments for other standard methods that have been well-ablated across more general segmentation methods. For example, 
    - Does standard ImageNet/DiNO-pretrained features perform worse than hand-crafted high/low frequency modelling? 
    - Does the model perform better mainly due to the extremely small model design to avoid overfitting, considering the training set only contains 10k images?
- Since we are doing segmentation instead of object detection, why do we need to do class matching instead of one-hot representation of each defined class in the dataset?
- The authors evaluate the method with a list of domain-specific and general segmentation methods but without any further adjustments. 
    - Are the reported performance for general segmentation methods representing zero-shot performance or fine-tuning performance? 
    - Are they trained from scratch or trained from any specific checkpoint?

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FOCUS, a semantic segmentation framework tailored for food images. It introduces two main components: (1) Frequency-Aware Boundary Modeling (FABM), which leverages high-frequency features and difference convolution to enhance edge precision; and (2) Gradient-Enhanced Underrepresented Class Matching (GEUCM), which assigns extra queries to rare classes to improve classification under class imbalance. The method is evaluated on three food segmentation benchmarks and shows improved performance over existing baselines.

### Strengths
1. The proposed framework integrates frequency-domain processing and query allocation strategies in a coherent architecture.

2. Experimental results on multiple benchmarks demonstrate consistent improvements over prior methods, including both general-purpose and food-specific segmentation models.

### Weaknesses
1. The core technical contributions lack novelty. Similar ideas have been extensively explored in prior works such as: [1]Spatial Frequency Modulation for Semantic Segmentation (PAMI 2025) [2] Frequency-aware Feature Fusion for Dense Image Prediction (PAMI 2024) These works already investigate frequency decomposition, selective filtering, and edge enhancement in segmentation tasks.

2. The proposed FABM module simplifies frequency decomposition and applies difference convolution, but this combination does not introduce fundamentally new mechanisms or insights beyond existing literature.

3. The GEUCM strategy for underrepresented class matching is a heuristic extension of one-to-many matching and does not offer a principled or theoretically grounded solution to class imbalance.

4. The paper does not provide sufficient analysis to demonstrate that its frequency-oriented design is uniquely effective for food segmentation, as opposed to general segmentation tasks.

### Questions
1. Can you provide ablation results comparing your frequency decomposition strategy against more granular or adaptive frequency partitioning?

2. Is there any theoretical justification for the effectiveness of GEUCM beyond empirical gains? How does it compare to standard re-weighting or sampling strategies?

3. Have you tested FOCUS on non-food datasets to validate whether the frequency-based edge modeling is domain-specific or generally applicable?

### Soundness
2

### Presentation
2

### Contribution
2
