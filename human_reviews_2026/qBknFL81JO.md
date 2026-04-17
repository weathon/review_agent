# Random Label Prediction Heads for Studying Memorization in Deep Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
We introduce a straightforward yet effective method to empirically study memorization in deep neural networks for classification tasks.
Our approach augments each training sample with auxiliary random labels, which are then predicted by a random label prediction head (RLP-head).  
RLP-heads can be attached at arbitrary depths of a network, predicting random labels from the corresponding intermediate representation and thereby enabling analysis of how memorization capacity evolves across layers.
By interpreting the RLP-head performance as an empirical estimate of Rademacher complexity, we obtain a direct measure of both sample-level memorization and model capacity.
We leverage this random label accuracy metric to analyze generalization and overfitting in different models and datasets.
Building on this approach, we further propose a novel regularization technique based on the output of the RLP-head, which demonstrably reduces memorization.
Interestingly, our experiments reveal that reducing memorization can either improve or impair generalization, depending on the dataset and training setup.
These findings challenge the traditional assumption that overfitting is equivalent to memorization and suggest new hypotheses to reconcile these seemingly contradictory results.
The source code is available at https://github.com/MarlonBecker/RandomLabelHeads

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a new mechanism that both measures and regularizes memorization in ML models. The proposes mechanism introduces a loss term that measures how well a learned representation can be used to predict randomly assigned labels. Smaller loss is correlated with stronger memorization in the representation. The loss then can become a regularizer that prevents memorization. Empirical study shows that the  reduced memorization may have different effects on model utility depending on the sample size and the nature of information being memorized.

### Strengths
This paper proposes an interesting and effective mechanism based on simple principles to detect and control memorization. The design is interesting and the intuition is clear. Well done. The work is technically sound.

### Weaknesses
Despite the smart design of RLP-head and the fairly comprehensive experiments, there seem to be a few missing links in the argument of the paper. Specifically:

1) The paper proposes the loss on random label prediction as **both** the regularizer **and** the measure of memorization. Notice that the RLP loss serves as a proxy of empirical Rademacher complexity, which measures the model's capacity to memorize instead of the amount of memorization. Does a low RLP score necessarily mean less memorization?

2) Given the plethora of work on the pro and cons of memorization, I wonder if the empirical evaluation results and their implications are different from previous work. (See questions.)

In addition, the pdf file seems not fully follow the template. Some margins between the paragraphs are too small. Hope this can be fixed in future versions.

### Questions
1) Could you use a different metric for memorization, say influence-based heuristic in [1] or a method of your choice, to show that the model has less memorization when regularized more heavily with RLP?

2) Has empirical Rademacher complexity ever been used as regularizer before?

3) There has been literatures showing memorization could be beneficial (long-tail) [2] and detrimental (wrong-label) [3]. What are the key insights in this work's experiment that are different from the previous work?

[1] Feldman, Vitaly, and Chiyuan Zhang. "What neural networks memorize and why: Discovering the long tail via influence estimation." Advances in Neural Information Processing Systems 33 (2020): 2881-2891.

[2] Feldman, Vitaly. "Does learning require memorization? a short tale about a long tail." Proceedings of the 52nd annual ACM SIGACT symposium on theory of computing. 2020.

[3] Liu, Sheng, et al. "Early-learning regularization prevents memorization of noisy labels." Advances in neural information processing systems 33 (2020): 20331-20342.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a method to measure memorization in deep models. The method involves a trainable RLP heads that can be attached to any layer of the network. The performance of RLP heads serves as a proxy for sample-level memorization. The authors also propose an RLP-based regularizer that reduces memorization by penalizing confident random-label predictions.

### Strengths
I think this is a really important problem, especially given the fact that sample-level memorization is expensive. The authors present a light weight solution, that can not only be run with relatively low latency, but can also be adapted to different layers of the model. I think this paper has a lot of potential if the authors can address the concerns below.

### Weaknesses
1. Lack of Suitable Baseline:

RLP approximates sample-level memorization. However, the paper lacks a clear baseline to verify whether the points identified or regularized by RLP are indeed the truly memorized samples. This is a major concern. The authors should validate this by comparing RLP’s behavior against established memorization benchmarks. This can be using Feldman et al.’s methodology or by introducing random noise images or “canary” points into the dataset. This would help determine whether 1) RLP selectively targets memorized data or unintentionally affects well-learned samples 2) It will also help understand how RLP behaves when base points are learned vs memorized. At this point, it is hard to gauge how well this technique performs. However, a good baseline can alleviate those concerns.

2. Limited Comparison with Existing Regularization Methods:

Although the authors present preliminary results suggesting that RLP-based regularization reduces memorization, the study lacks *direct* comparisons to standard regularizers (e.g., dropout, weight decay, or label smoothing). Such comparisons are critical for understanding whether RLP provides a distinct benefit beyond existing techniques. Ideally, these evaluations should be performed along two axes: (a) classification accuracy on intentionally mislabeled or noisy points (refer to point 1) to measure memorization control and (b) test accuracy on clean data (to assess generalization). Without these baselines, it is difficult to fully gauge the contribution and novelty of RLP as a regularizer.

3. Narrow Experimental Scope:

The current experiments are restricted to vision models trained on image datasets. Extending the analysis to text classification models and even LLMs would significantly strengthen the work, demonstrating the generality of RLP as a tool for studying memorization across architectures and domains.

### Questions
Refer to points above

### Soundness
2

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
4

### Summary
The paper proposes Random Label Prediction heads (RLP-heads) as a simple mechanism to measure and control memorization in deep nets during standard supervised training. Each training sample is given an auxiliary random label. A small prediction head attached to an intermediate activation is trained to predict these random labels in parallel with the main task. The authors interpret the RLP accuracy as an empirical proxy for Rademacher complexity and use it to study how memorization evolves over time and across layers. They also introduce a regularizer that penalizes correct random-label predictions to suppress memorization while keeping the task head untouched. Formally, the losses are L_{\text{class}}=-\log p_y, L_{\text{rnd}}=-\log \hat p_{\hat y}, and the regularizer L_{\text{reg}}=\log(1-\hat p_{\hat y}) scaled by \lambda. 

Empirically, the paper shows:
1.	RLP accuracy tracks capacity and overfitting dynamics, rising to about 70 percent when training ViT-B/32 on ImageNet as test and train accuracies begin to separate. 
2.	Standard regularizers like dropout, weight decay, and label smoothing reduce RLP accuracy, supporting the complexity interpretation. 
3.	Using L_{\text{reg}} can reduce overfitting and improve test accuracy on ImageNet with ViT (about +1.5 points), but on CIFAR-100 with WRN it reduces memorization without helping test accuracy.
4.	Layer-wise probes reveal memorization grows with depth and that regularizing the final layer shifts memorization earlier rather than eliminating it. 
5.	Dataset size and noise matter. Suppressing memorization helps when data are well sampled, but can hurt on undersampled datasets. Adding label noise makes the regularizer beneficial, as predicted.

### Strengths
•	Originality: using online random-label heads to continuously estimate per-layer memorization without retraining on random labels is clever and practical
•	Quality: solid empirical study across architectures, datasets, head designs, and hyperparameters, including offline sanity checks and dataset subsampling
•	Clarity: method, losses, and training protocols are well specified, figures communicate dynamics and layer-wise trends
•	Significance: provides a low-overhead diagnostic for memorization and a tunable knob to reduce it, yielding nuanced insights on when memorization helps or hurts generalization; the layer-wise shift phenomenon is especially interesting

### Weaknesses
•	The mapping from RLP accuracy to Rademacher complexity is argued empirically; a theoretical bridge or formal bound would strengthen the claim beyond correlation
•	Improvements on ImageNet are relatively small and sensitive to \lambda; practical guidance on choosing \lambda beyond grid search is limited
•	Baseline comparisons are missing to targeted anti memorization methods such as mixup, manifold mixup, early stopping with sharpness awareness, or confidence based noise filtering; the paper mostly compares to generic capacity regularizers
•	The regularizer may alter features in ways that indirectly affect the main head; while gradients are restricted, feature extractor changes can still trade off task signal vs sample specificity; stronger checks isolating collateral effects would help
•	The undersampling hypothesis is compelling but remains somewhat post hoc; additional controlled long tail benchmarks or per class sampling analyses would solidify it
•	Compute overhead and wall clock costs from extra heads, multi head variants, and per layer probes are not quantified in detail; practicality at large scale is uncertain
•	Privacy claims are hinted at in motivation but not evaluated; no extraction or canary tests are provided

### Questions
•	Can the authors formalize the connection between RLP accuracy and empirical Rademacher complexity for multiclass settings, perhaps via margin based surrogates or a bound that depends on head capacity and n?
•	How sensitive are results to the random label assignment itself; do multiple seeds produce similar \lambda optima and layer profiles, and what is the variance?
•	Could you compare RLP regularization against mixup or manifold mixup at matched hyperparameter tuning budgets, and report both accuracy and calibration?
•	Do RLP metrics predict robustness or privacy leakage; for example, do higher RLP accuracies correlate with membership inference or canary extraction rates?
•	On long tail datasets like iNaturalist or ImageNet LT, does suppressing memorization reduce performance on rare classes; per tail analysis would test the hypothesis directly
•	For the observed shift of memorization to earlier layers, can you probe within blocks to rule out within block hiding; e.g., multiple taps inside the same transformer block
•	What is the runtime and memory overhead of single head vs multi head vs all layers, and how does this scale with n and hidden size
•	Does RLP regularization interact with self supervised pretraining; does it help fine tuning stability or hurt transfer

### Soundness
3

### Presentation
3

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
The paper introduces Random Label Prediction (RLP) heads: auxiliary prediction heads attached at various depths that are trained to predict fixed, per-example random labels during standard supervised training. The accuracy of these heads is used as a proxy for memorization/capacity. The authors also propose an RLP regularizer that discourages the feature extractor from confidently fitting the random labels, aiming to limit memorization without changing the task head. Experiments on ViT-B/32 with ImageNet and WRN-16-4 with CIFAR-100 show: (i) RLP accuracy rises early in training and can reach high levels on ImageNet; (ii) standard regularizers like dropout, weight decay, and label smoothing tend to reduce RLP accuracy; (iii) the RLP regularizer can improve ImageNet generalization but can hurt on CIFAR-100; (iv) RLP attached at intermediate layers suggests memorization increases with depth, and regularizing only the last layer can shift memorization earlier.

### Strengths
1. Simple, general mechanism that is easy to add and interpret: higher random-label accuracy indicates more sample-specific information in the features.

2. Clear empirical phenomena: early rise of RLP accuracy and monotonic increase with depth.

3. Useful bridge from measurement to control: correlation with common capacity controls and a targeted regularizer derived from the same signal.

4. Layerwise analysis reveals memorization shifting under last-layer regularization and provides a way to localize where class information emerges.

### Weaknesses
1. Theoretical grounding is informal. The connection to Rademacher complexity is motivational rather than a formal result; no theorem guarantees RLP accuracy is a calibrated surrogate for capacity in deep nets.

2. Attribution is ambiguous. The metric may conflate memorization in the feature extractor with the auxiliary head’s own capacity, the number of random labels, and optimization budget. Appendix ablations help but a clearer identifiability story would be better.

Minor comments
1. Please switch to the official ICLR template and fonts; the current submission uses a nonstandard font.

### Questions
1. CIFAR100 is known to have near duplicates in train and test sets, this could be causing the contrary results on CIFAR100, would de-duplicated CIFAR100 be a better fit?

### Soundness
3

### Presentation
3

### Contribution
3
