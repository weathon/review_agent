# Probably Approximately Correct Labels

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Obtaining high-quality labeled datasets is often costly, requiring either
human annotation or expensive experiments. 
In theory, powerful pre-trained AI models provide an opportunity to 
automatically label datasets and save costs. 
Unfortunately, these models provide no guarantees on their accuracy, 
making wholesale replacement of manual labeling impractical.
In this work, we propose a method for leveraging pre-trained AI models to curate 
cost-effective and high-quality datasets.
In particular, our approach results in
*probably approximately correct labels*: with high probability, the overall
labeling error is small. 
Our method is nonasymptotically valid under minimal assumptions on the dataset or
the AI model being studied, and thus enables rigorous yet efficient dataset
curation using modern AI models. We demonstrate the benefits of the methodology
through text annotation with large language models, image labeling with
pre-trained vision models, and protein folding analysis with AlphaFold.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem that building high-quality labeled datasets is costly, and while leveraging AI model predictions can reduce costs, the lack of accuracy guarantees makes full replacement difficult. It proposes a mathematical framework to label datasets using AI models, mathematically bounding the error below epsilon and the confidence above 1-a. Uncertainty scores play a central role in this process, where expert labels are collected only for data exceeding an uncertainty threshold u^. To ensure precise uncertainty measurement, calibration correcting uncertainty per data cluster is performed. The authors experimentally verify the broad effectiveness of the PAC approach across diverse domains including natural language, images, and proteins.

### Strengths
- Provides a formal mathematical framework to set an appropriate uncertainty threshold between expert labels and model-generated labels, with a rigorously defined iid variable within this framework.  
- Clearly points out the critical importance of precise uncertainty measurement to guarantee the quality of AI-generated labels, enhancing uncertainty estimation with calibration techniques.  
- Demonstrates through experiments on various data modalities—natural language, image, and protein structure—that the PAC method is a broadly applicable algorithm across multiple modalities and models.

### Weaknesses
- The uncertainty threshold is set universally across the dataset, but actual model inferences often show variations in uncertainty by class, due to reasons such as insufficient training samples for specific classes or confusing classes. Has the approach of using class-specific thresholds for more efficient labeling been considered?  
- How is the value of m determined? This seems like a very critical hyperparameter for the method. If this value is heuristically determined, it would be difficult to consider the algorithm as robustly functioning. If it is described somewhere in the paper that I missed, it would be helpful to be informed.
- From the experimental results, it is hard to say that PAC brings a meaningful gain. In Figure 1, the PAC results lie between naive thresholding and AI-only interpolation, suggesting the PAC method mainly adjusts the trade-off between error and budget, rather than achieving gains beyond existing trade-off lines. If the authors’ intent is to enable trade-off adjustment using PAC, it makes sense, but this can partially be achieved just by constant threshold adjustment, limiting the contribution.  
- There is no conclusion section, and therefore no mention of limitations and future work. Usually, limitations and future work part is considered as an important part of a paper, so this omission is a clear weakness.

### Questions
- As model overparameterization increases, over-confidence in model predictions worsens. Are there any countermeasures proposed for this?  
- Depending on the reader’s background, some may find the format of protein folding labels unfamiliar. Would it improve the completeness of the paper to briefly explain the label format of the various domains you used in the experiment section?

### Soundness
3

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
2

### Summary
This paper addresses the high cost of building high-quality labeled datasets. The authors claim that while many labeling methods exist, they have a critical limitation of providing no guarantees on accuracy. Therefore, the paper tackles the problem of how to guarantee the error rate, proposing a methodology called "PAC labeling." The method first uses AI labeling, classifies labels based on confidence, and then decides whether to use human labeling based on a certain threshold. The authors state this is done under minimal statistical assumptions. As a result, they achieve high performance compared to an "AI only" baseline and also show better performance than a fixed-threshold baseline.

### Strengths
This paper addresses an important problem and is well-timed with the current trend of synthetic data labeling, particularly in its focus on providing statistical guarantees.

### Weaknesses
The biggest problem is that the baseline comparison feels contrived, and the paper's contribution seems overstated. The authors claim that existing methods fail to provide mathematical guarantees, but such methodologies clearly exist. For example, Conformal Prediction (CP), which the authors themselves mention, deals with a very similar problem. The CP methodology quantifies prediction uncertainty into a prediction set with guaranteed coverage; it statistically guarantees the probability that the true label is within this set. Can this not be used to solve the same problem? The same applies to papers like "Learning to Defer...". This methodology also solves the same problem. Why did the authors not compare against these methods and evaluate their contributions toe-to-toe?

### Questions
See the Weakness section. Why were direct competitors that also provide statistical guarantees such as Conformal Prediction and Learning to Defer, omitted from the experimental comparison, and why was their contribution not evaluated "toe-to-toe"?

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
The paper proposes a method to create high-quality labels by leveraging cheap predictions from AI models alongside costly expert labels. The method has statistical guarantee: with high probability, the final labeled dataset's error will not exceed a user-specified threshold. For data points labeled by AI models, only those with uncertainty exceeds a threshold are sent for expert labeling, minimizing overall cost. The paper also outlines a more complex PAC Router for optimally selecting between multiple AI models and presents experiments demonstrating budget savings.

### Strengths
- The paper considers a very practical setup with AI models generating labels for all data and then experts annotating the most valuable subset. This is helpful in generating labeled training data in low resource domains.
- The proposed method has nice statistical guarantee that with high probability, the final labeled dataset's error will not exceed a user-specified threshold.
- The effectiveness of the method in terms of cost saving is demonstrated empirically.

### Weaknesses
- In practice, the cost of high quality AI models is not neglectable, especially given the existence of test time scaling. Wonder how does this affect the proposed approach? Also wonder if allocating the whole expert budget to AI models test time scaling achieves better results?

- The considered baselines are too naive. A natural baseline is to use active learning. For example, first initialize the active learning model to be the model trained on labels generated by AI models, then select data points for experts to label using a standard active learning process. Also, a better baseline than the second baseline is to get more labels from AI models (with different prompts or AI models) until the cost matches the proposed method, and then aggregate the labels (e.g. with existing methods from the crowd sourcing literature).

### Questions
- how does the method compare to active learning or aggregated labels from many AI models (with different prompts and models).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Probably Approximately Correct (PAC) Labeling, aiming to provide provable guarantees for AI-assisted labeling.
It proposes to select an uncertainty threshold such that all samples with model confidence above the threshold are labeled automatically, while others are sent to experts.
The guarantee states that, with high probability, the global labeling error is ≤ ε.
Extensions include a multi-model PAC Router and a practical multicalibration procedure to improve uncertainty reliability.

### Strengths
1. Relevant and well-motivated problem formulation. Applying PAC reasoning to automatic labeling is timely and practically meaningful.  
   It formalizes a common heuristic (confidence-based filtering) into a statistically grounded process.  

2. Simplicity and generality. The proposed approach is model-agnostic and directly applicable across diverse labeling pipelines.  

3. Comprehensive empirical coverage. Experiments span multiple modalities and show consistent compliance with PAC bounds while reducing expert effort.  

4. Clear presentation and awareness of practical issues.   The inclusion of multicalibration demonstrates attention to real-world miscalibration.

### Weaknesses
1. Limited overall contribution despite novel framing.
While the paper introduces new formulations (e.g., PAC Labeling and PAC Router) that are conceptually fresh, the underlying theoretical substance remains limited.
The analysis primarily builds on existing mean-upper-bound PAC results without introducing new bounds, assumptions, or insights into the nature of uncertainty in labeling.
Consequently, although the problem setting is well-motivated, the contribution lies more in repackaging and integration than in theoretical advancement.


2. Disconnect between calibration and PAC validity.
Multicalibration empirically adjusts uncertainty scores but may violate the monotonicity assumption required for PAC reasoning.  
   The paper provides no formal analysis showing that PAC guarantees still hold after calibration.

3. Lack of intuitive and competitive comparisons.
Although experiments are broad, they do not clearly show cost–error trade-offs (e.g., comparing methods under equal error rates).  
   Baselines are limited to simple heuristics; stronger competitors like conformal labeling or active selection are missing.  
   As a result, the empirical advantage remains qualitative rather than quantitative.

### Questions
1. What concrete theoretical or methodological innovations are introduced beyond adapting existing PAC mean-upper-bound results?  

2. Does multicalibration theoretically preserve the assumptions required for PAC guarantees, or is it purely an empirical fix?  

3. Have the authors evaluated cost–error trade-offs, e.g., comparing expert costs at equal error levels across methods?  

4. How does the framework behave under correlated errors or severe miscalibration beyond what multicalibration can correct?

### Soundness
3

### Presentation
4

### Contribution
2
