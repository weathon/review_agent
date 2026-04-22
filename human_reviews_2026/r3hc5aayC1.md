# Class-Adaptive Rectification with Experts for Robust Long-Tailed Noisy Label Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Real-world datasets frequently exhibit long-tailed class distributions alongside noisy labels, posing compounded challenges for robust learning.
While recent methods have made progress, they often neglect the uneven impact of label noise across classes, resulting in insufficient correction for tail classes.
This imbalance further introduces erroneous over-regularization on other classes, ultimately undermining long-tailed learning.
To address these challenges, we propose Class-Adaptive Rectification with Experts (CARE), a parameter-efficient framework built upon vision–language models, which performs class-aware label correction by jointly leveraging three complementary sources of supervision: noisy observed labels, text embeddings, and image features. 
CARE further employs a class-adaptive Top-$K$ expert consensus mechanism, which assigns smaller $K$ to tail classes in order to extract reliable candidate labels and recalibrate class frequencies.
This refinement yields faithful class-frequency estimation, thereby enabling more reliable long-tailed calibration.
We evaluate CARE on CIFAR-100-LTN, mini-ImageNet-LTN, and real-world datasets, including Food101N and WebVision-50. 
Across all benchmarks, CARE consistently surpasses recent state-of-the-art methods, achieving up to 3.0\% accuracy improvements in certain settings.
The source code is temporarily available at https://anonymous.4open.science/r/CARE-9F10.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper propose a method for label correction by merging information from text-image alignment, pseudo label and noise label. Also it use class adaptive strategy to apply different level of noise correction for class of different frequency.  The experiments are conducted on both synthetic data and real data. Compared with CLIP based methods, it can bring further improvement. However, there lack ablations to support the key claims, and there missing some details for the whole pipeline. The suggest is to reject.

### Strengths
The proposed method merges priors in text/image alignment(TE), pseudo labels (IE) and noise labels (BE) to generate a better label closer to ground truth distribution, yielding to better results. Also it emphasizes different impact of label noise across classes.

### Weaknesses
The motivation sounds good but there lack experiments supporting the claim. 
Why the tail classes requires more corrections. There is no analysis showing something like marginal benefit when applying label corrections for head classes.
How the proposed method apply more corrections on the tail classes? There is a theorem (Proposition 3) but the proof for it is empirical and there is no comparison for different settings for an actual proof. 

See questions part below for other concerns

### Questions
1. How is the NR calulated in Table1. From figure 1 it seems RLD 1,2,3 have much lower noise level than reported in table 1. 
2. What is NR for data at different frequency in RLD 1,2,3 in Table1? 
3. For the proof of Theorem 3, the proof seems empirical but not a strict proof. Like "Since tail class t has
low sample frequency, ... Thus, Pt(K) grows faster than Tt(K). "  How low frequency class t should be to make it happen? At least Pt(K), Tt(K) should be written in a function of class frequency. Otherwise it is hard to tell whether the claim is correct
4. The relationship between Kc and class frequency n_c is the key for class aware noise controlling. How the choice of the relationship affect the results? Like using a global K for all the class, what the actual result would be?  Will the noise level of tail class increase or decrease?
5. The name of Table 7 and Table 8 should exchange?
6. In Table 7, the first three rows have exact the same NR and accuracy, which is strange.
7. Does the improvement come from the class-aware noise assignment or the expert mixing strategy? From table 1 and figure 1, the RLD 3 have more noise level at the "Many" class than RLD 1,2 but it still can achieve better accuracy at head classes. 
8. In Figure 2 case1, there is a class with 0.4 probability in the IE part,  which is removed in the final accumulation. I don't find the corresponding strategy in the paper. Is that a mistake in the figure?
9. The pipeline remains unclear to me. The AdaptFormer and w is for calculation of IE, and used in label correction. After that, which part of the model will be trained using the corrected label to get the final results? How is the AdaptFormer and w initialized? Do they need cold start training using the noise label?

### Soundness
2

### Presentation
2

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
This work is about learning from datasets that have long tailed class distributions and label noise. The method proposed is to leverage three complementary experts (text, image, and observed labels) with a class adaptive top K consensus mechanism to correct noisy labels. The central argument is that tail classes require more conservative consensus (smaller K) to avoid confirmation bias, while head classes can tolerate more greater consensus (larger K).

### Strengths
+ The identified problem is timely and interesting. The class agnostic label correction can actually harm performance by insufficiently correcting tail class labels.
+ The theoretical analysis is interesting and showing how consensus based refinement amplifies reliability. It provides good intuition for the design choices.
+ While missing some standard datasets in this task like CIFAR-10, the experiments are thorough enough, they spans on CIFAR-100-LTN, and mini-ImageNet-LTN, which are synthetic datasets, and real world datasets like Food101N, WebVision-50 under various noise types and imbalance factors. Results consistently show improvements, particularly notable gains for severe imbalance scenarios.
+ It is well presented.

### Weaknesses
- While the combination is novel and backed by the theoretical analysis, the individual components, like using CLIP for label correction, expert consensus, and top k voting have been explored previously . The main contribution is the class adaptive mechanism, which while effective, represents an incremental advance.
- Compared to the state of the art evaluations, real world noise evaluation is limited to webvision50 and food101n. Using Clothing1M would strengthen the claims.
- The method fundamentally relies on CLIP, limiting applicability to domains where such pre-trained models aren't available or perform poorly.
- The proof of Theorem 1 assumes conditional independence between pTE and pIE, but this assumption may not hold since both derive from CLIP's shared vision language space.

Minor:
- Eq 4 appears complex and could benefit from clearer notation.

### Questions
- how to handle edge cases e.g. when all experts disagree completely or when Kc becomes 0 for extremely rare classes?
- can you discuss failure cases or provide error analysis to understand when the proposed method might underperform?

### Soundness
3

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
The paper presents "Class-Adaptive Rectification with Experts (CARE)," a framework for robust learning in the context of long-tailed noisy label (LTNL) problems. It addresses the challenges of label noise and class imbalance, particularly in tail classes, by leveraging a class-aware Top-K expert consensus mechanism. The proposed method integrates noisy labels, text embeddings, and image features to improve label correction and class frequency estimation. Experimental results on multiple benchmarks, including CIFAR-100-LTN and real-world datasets, show that CARE consistently outperforms existing state-of-the-art methods, achieving significant accuracy improvements. The authors provide a comprehensive analysis of their approach and its implications for future research.

### Strengths
Innovative Approach: The CARE framework effectively combines label rectification with class-adaptive techniques, addressing both noisy labels and long-tailed distributions.

Comprehensive Evaluation: The framework is tested on various benchmarks, demonstrating consistent performance improvements over state-of-the-art methods.

Strong Theoretical Basis: The paper provides solid theoretical justifications for its design choices, enhancing its credibility and understanding.

### Weaknesses
1. The CARE framework may require significant computational resources due to the integration of multiple expert models, which could limit its practicality in resource-constrained environments.

2. The evaluation primarily focuses on specific datasets, raising concerns about the framework's adaptability to other domains or types of noise not covered in the experiments.

3. The emphasis on correcting tail class labels could lead to overfitting, especially if the tail classes are too small or underrepresented in the training data.

4.  The reliance on accuracy as the sole performance metric may overlook other critical aspects of model performance, such as precision, recall, or F1-score, particularly in imbalanced datasets.

### Questions
How does CARE perform with larger datasets?

Can it handle other noise types or distributions?

What can be done to prevent overfitting on tail classes?

### Soundness
3

### Presentation
2

### Contribution
2
