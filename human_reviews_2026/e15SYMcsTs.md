# Learning Dynamics of Logits Debiasing for Long-Tailed Semi-Supervised Learning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Long-tailed distributions are prevalent in real-world semi-supervised learning (SSL), where pseudo-labels tend to favor majority classes, leading to degraded generalization. Although numerous long-tailed SSL (LTSSL) methods have been proposed, the underlying mechanisms of class bias remain underexplored. In this work, we investigate LTSSL through the lens of learning dynamics and introduce the notion of baseline images to characterize accumulated bias during training. We provide a step-wise decomposition showing that baseline predictions are determined solely by shallow bias terms, making them reliable indicators of class priors. Building on this insight, we propose a novel framework, DyTrim, which leverages baseline images to guide data pruning. Specifically, we perform class-aware pruning on labeled data to balance class distribution and label-agnostic soft pruning with confidence filtering on unlabeled data to mitigate error accumulation. Theoretically, we show that our method implicitly realizes risk reweighting, effectively suppressing class bias. Extensive experiments on public benchmarks show that DyTrim consistently enhances the performance of existing LTSSL methods by improving representation quality and prediction accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the DyTrim method, which eliminates bias by employing benchmark image-guided dynamic data pruning. The authors demonstrate that this approach implicitly implements risk reweighting, elucidate the mechanism by which benchmark images mitigate classifier bias in class-imbalanced scenarios, and reveal how class-imbalanced datasets influence a model's predictions on baseline images.

### Strengths
This paper demonstrates through a solid theoretical foundation that the DyTrim method can effectively suppress class bias. The DyTrim method provides an elegant solution to the class imbalance problem in semi-supervised learning without requiring complex network branches or de-biasing mechanisms. Extensive experiments on public benchmarks validate the effectiveness of the proposed method. This approach contributes to the advancement of the field.

### Weaknesses
[1] Conventional pruning strategies are typically employed to eliminate samples that contribute minimally to training, thereby accelerating the training process. However, this paper utilizes such strategies to obtain a relatively balanced training subset. Would this approach suffer from the undersampling problem, specifically for tail classes?

[2] Figure 1 fails to provide sufficient evidence for the conclusions put forward in the paper. Especially in the bottom part, the predicted distributions derived from pseudo-labels—whether correct or incorrect—show no significant difference; in some cases, they are even completely identical.

[3] In Equation 18, what is the distinction between H_t^l  and the subsequent H_(c,t)^l? Additionally, how are the scoring functions H_t^l and H_t^u specifically obtained?

[4] In Eq (20), for unlabeled data, high pseudo-label confidence links to a random pruning probability r , where a larger r results in a larger gradient scaling factor. Could this intensify the model’s bias toward high-confidence samples (usually head classes)? Furthermore, the gradient scaling follows Qin et al.’s setting. Ablation studies demonstrate that it contributes more to model performance than other components proposed in this work, and removing it causes a notable decline in performance.

[5] According to Equations (19) and (20), the scores corresponding to pruned samples will no longer be modified. Does this mean that these samples will not be used in subsequent training processes? If so, it would result in severe information waste.

[6] The experimental section lacks comparisons with the latest literature. Additionally, it would be beneficial if the authors could provide information on the pruning amount of training samples in each iteration.

[7] It seems that the predicted class probabilities obtained from Figure 3(a) and (b) both appear to be unsatisfactory. Additionally, some errors appear to exist in the titles and their corresponding descriptions for Figure 3, Figure 4, and Figures 6 to 10.

[8] There are still some expression errors and unclear descriptions in the manuscript. For instance, the definition of the imbalance of the unlabeled dataset and the description of Figure 2 require clarification.

### Questions
As described in “Weaknesses”.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In their manuscript, the authors address the problem of long-tailed semi-supervised learning. More specifically, the problem of estimating the accumulated biased is approached using techniques of learning dynamics and benchmark images. The proposed framework, DyTrim, uses these findings to guide data pruning. The approach is evaluated on three standard benchmarks.

### Strengths
- The manuscript makes use of very recent work such as Learning Dynamics (Ren&Sutherland, ICLR2025) and solid-color input for optimal bias estimation (Xing et al., AAAI2025)
- The writing is generally good, although the text-flow is sometimes confusing
- Figure 1 is a good illustration, although it needs better connection with the main storyline.
- The method has been compared in a comprehensive set of experiments with good results.

### Weaknesses
- Both soundness and contribution seem to be problematic, possibly cause by unclear presentation
    - contribution: it remains unclear in which sense the current method goes beyond combining the works by Ren&Sutherland, ICLR2025 (btw: the reference should be updated from the pre-print) and Xing et al., AAAI2025 for guiding sub-sampling
    - soundness: it remains unclear how the authors reflect about their use of definitions and propositions. The presentations of the theory leaves the reader with serious doubts that the construction based on definition 1, subsequent propositions, and theorem 1 is sound. The proofs in the appendix are not mentioned in the main text and seem to be not fully completed (e.g. missing references)
    - writing: although generally good, it is sometimes unnecessary hard to interpret the writing, also caused by some language issues (e.g. line 040: "the approach employ baseline image ") 
- Some relevant references are missing, e.g. mixing-based approaches (built on [1]) 
- The presentation of results is a bit confusing: which results are taken from the literature and which have been reproduced; why deviate reproduced values from the literature, etc.

[1] Zhang et al. Mixup: Beyond empirical risk minimization. ICLR 2018.

### Questions
1. how does the proposed method goes beyond combining Ren&Sutherland, ICLR2025 and Xing et al., AAAI2025 for guiding sub-sampling?
1. what is defined in Definition 1? In its current from, it says "decompose" which implies a proposition
1. in which ways go propositions 1 and 3 beyond definition 1 (which might be a proposition)?
1. how can the use of a bias-free first layer (line 173) be justified?
1. what is the intuitive, non-trivial connection between proposition 2 and theorem 1?
1. which results in the experiments were reproduced and which were taken from the literature?
1. in case of deviations: why do the results differ?

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
2

### Summary
This paper tackles the problem of semi-supervised learning with heavy class imbalance. It proposes to dynamically prune the dataset according to the model confidence as well as the approximate marginal class distribution based on the logits derived from a baseline black image. The proposed approach performs well on small-scale image benchmarks (32x32).

### Strengths
- The proposed approach seems to intuitively make sense in solving the problem (for the most part)
- The proposed approach seems to do well on the various benchmarks explored.
- The presentation of the paper is clear and professional. Figure 2 is very helpful in understanding the proposed approach.

### Weaknesses
The paper load for ICLR this year has been large, and so I have not been able to spend as much time as I would like on reviewing. I encourage the authors to correct any errors/misunderstandings I may have with regards to the paper.  Moreover, I do not work in either semi-supervised or long-tailed learning, so my review confidence will be low, and I will defer to other more knowledgeable reviewers.

1. **Motivation is hard to understand**
    1. The learning dynamics theory doesn't feel particularly relevant or well-connected to the downstream approach. 
    1. Figure 1 is not very easy for the reader to interpret.
1. **Baseline image design choice**
    1. I am confused by the choice to use a black image as a baseline to extract marginal class probabilities. Why not just directly extract the bias vector from the final fully connected layer? This is usually accessible for most deep learning models in my experience. 
    2. Suppose we have a class-balanced dataset of images where a CNN learns to classify different solid colours (e.g. white, black, red). Wouldn't you expect the black baseline image to return a probability vector close to [0,1,0]?
1. **Experiments only on small-scale data**
    1. Although experimental results seem good, they are only on small-scale data (up to 64x64). I would be much more confident in the approach if the authors demonstrated strong performance on imagenet-scale (224x224) scale data such as https://github.com/zhmiao/OpenLongTailRecognition-OLTR. 

1. **Complexity of approach**
    1. It seems like the proposed approach is quite complicated with many knobs for the practitioner to adjust. I am a little sceptical whether such an approach could be conveniently adopted.

### Questions
I am unsure why pruning of data is performed rather than simply reweighting the sampling frequency. Can the authors clarify this point?

I am open to raising my score if my above issues are addressed; however, I will still keep low confidence even if I do so, as I am not familiar with this field of research.

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
2

### Summary
This paper analyzes long-tailed SSL bias through learning dynamics. Key insight: predictions on baseline images (solid-color inputs) \pi_\theta(\mathcal{I}) encode class bias via BatchNorm terms (Theorem 1). DyTrim uses these as pruning ratios for class-aware labeled data pruning and label-agnostic unlabeled pruning. Shows 1-3% improvements over CDMAD on CIFAR-LT/STL-LT/ImageNet-127.

### Strengths
Principled framework: decomposing learning dynamics with a formal link between baseline representations and class priors.

Consistency: Works with FixMatch/FlexMatch/FreeMatch and WRN/ViT/ResNet.

Thorough validation: 15+ baselines, diverse settings (matched/mismatched \gamma).

### Weaknesses
Incremental Gains: Performance improves by only 1–2 % over CDMAD, which is modest given their conceptual overlap—both methods rely on baseline-image statistics.

### Questions
When does the method fail? What are its limitations?

### Soundness
3

### Presentation
3

### Contribution
3
