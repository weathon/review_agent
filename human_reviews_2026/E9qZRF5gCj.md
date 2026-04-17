# What really matters in matrix whitening optimizers?

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
A range of recent optimizers have emerged that approximate the same "matrix-whitening" transformation in various ways. In this work, we systematically deconstruct such optimizers, aiming to disentangle the key components that explain performance. Under tuned hyperparameters across the board, all flavors of matrix-whitening methods reliably outperform their elementwise counterparts, such as Adam. Matrix-whitening is often related to spectral descent -- however, metrics reveal that performance gains are *not explained solely by accurate spectral normalization* -- particularly, SOAP displays the largest per-step gain, even though Muon more accurately descends along the steepest spectral descent direction. Instead, we argue that matrix-whitening serves *two* purposes, and the variance-adaptation component of matrix-whitening is the overlooked ingredient explaining this performance gap. Experiments show that variance-adapted versions of optimizers consistently outperform their sign-descent counterparts, including an adaptive version of Muon. We further ablate variance adaptation strategies, finding that while "lookahead" style approximations are not as effective, low-rank variance estimators can reduce memory costs without a performance loss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper systematically deconstructs and analyzes the factors contributing to the performance improvements of modern matrix-whitening optimizers such as Shampoo, SOAP, and Muon. The authors argue that the success of these optimizers is not solely attributed to 'spectral normalization', but that 'variance adaptation'—an often-overlooked component—is critically important. Experiments demonstrated that optimizers incorporating variance adaptation (Adam, SOAP, AdaMuon) consistently outperformed those without it (Signum, SPlus, Muon). This suggests that the two components of a matrix-whitening optimizer can be applied in a decoupled manner.

### Strengths
1.	The analytical approach itself, which deconstructs optimizers along the two axes of 'spectral normalization' and 'variance adaptation', is original.

2.	Although some experiments were 'best-effort' , the fact that the other version optimizers underwent thorough tuning of four key hyperparameters —which was transparently disclosed in the appendix—shows an effort to adhere to scientific procedures.

### Weaknesses
1.	Lack of generalization: All conclusions rely solely on a single model (GPT-2 Base) , a single task (Language Modeling) , and a single dataset (OpenWebText).

2.	Insufficient evaluation metrics: The sole measure of final performance is 'Validation Loss'. The paper fails to demonstrate whether faster convergence or a slight loss improvement translates to actual performance gains on downstream tasks. 

3.	Baseline failure: The analysis completely lacks an explanation for why the key baseline, Shampoo-100, failed to converge, which lowers the experiment's reliability.

4.	Ambiguity: The use of a 'simplified' AdaMuon and the 'best-effort' tuning for N=10 versions are factors that compromise the fairness of the comparisons.

### Questions
1.	It would be difficult to generalize findings based on only a single model and dataset . Can you provide evidence that the paper's claims hold true for other transformer-based models and datasets?

2.	Can you present evidence that the marginal improvements in validation loss lead to statistically significant performance gains on practical downstream tasks for LLMs, such as the GLUE benchmark?

3.	Please provide a clear explanation for the convergence failure of Shampoo-100.

### Soundness
2

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
This study compares various optimizers based on the *whitening metric* by reporting the minimal validation loss achieved after training a GPT-2 transformer for next-token prediction on the OpenWebText dataset. The training consists of 10,000 updates with cosine learning rates following an initial warmup. The hyperparameter sweep includes four parameters: learning rate, weight decay, momentum EMA coefficient, and variance EMA coefficient. Other hyperparameters, such as Adam's epsilon (also used in other optimizers for regularizing inverses), are held constant.

The first key result confirms that all carefully tuned optimizers employing update rotation outperform the diagonal Adam optimizer.

Additionally, the study argues that Shampoo-style optimizers facilitate more accurate estimation of steepest spectral descent, based on a metric that I was not able to fully understand. Importantly, this improved spectral descent is not the sole factor behind the enhanced performance: a key takeaway is that, when comparing "variance-adaptation" and "sign" versions of equivalent optimizers side by side, the variance-adaptation variants consistently perform better.

### Strengths
The empirical comparison is of interest by itself, even though I think a much greater effort should be pursued across the whole community to provide larger scale controlled benchmarks.

The paper is careful not to overstate its findings. The limitations are clearly acknowledged, emphasizing that the results are based on a single experimental setup.

### Weaknesses
Surveying different optimization methods is inherently challenging due to the large number of hyperparameters, the multitude of optimizers proposed by the community, and the variety of available datasets—even for the same task of next-token prediction. While the current work is undoubtedly of interest, I am uncertain whether its findings are sufficiently general to warrant publication at ICLR. For example, using a different learning rate scheduler (such as cosine annealing) might lead to different conclusions.

Additionally, some design choices could be questioned. For instance, I believe that Adam's epsilon, which is arguably the second most important hyperparameter after the learning rate, should have been included in the hyperparameter sweep.

Another point of concern is that it is not clear what is being plotted in the left panel of Figure 3, nor is the corresponding discussion sufficiently explanatory.

### Questions
1. A clarification question: what is the "singular value of updates" (l.305, plotted in figure 3 ?), where the "update" is just a step in parameter space, i.e. a vector of all parameters of the LLM, for which I don't get the concept of "singular value".
2. A question on the relevant of publishing such work to ICLR: Do you think your findings would generalize to other setups, or are they specific to this architecture/dataset/training setup ? If so, how can you convince your readers ?

Then, I have some more minor questions/comments:

3. How are the hyperparameters of the "nonstandard" Adam optimizer (line 199) chosen?
4. It would be helpful to include a summary table of the update rules for all the benchmarked alternatives (lines 212–262).
5. Why limit the comparison to optimizers based on \sqrt{var} and sign only? Since implementing natural gradient-like optimizers (such as KFAC/EKFAC) involves not much additional effort, including them could provide a more theoretically grounded comparison. Do you have any thoughts on this?
6. Could you clarify whether you are using weight decay or L2 regularization? This distinction might matter, as discussed in "Three Mechanisms of Weight Decay Regularization" (ICLR 2019).
7. What about the training loss? Isn't that the quantity that the optimizers are directly minimizing?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper provides a systematic decomposition and empirical analysis of matrix whitening optimizers. The authors find that both spectral normalization and variance adaptation are indispensable components. Further ablation studies show that the lookahead approximation is less effective than the low-rank approximation.

### Strengths
The main strength of this paper lies in its thorough empirical analysis of recently popular matrix whitening optimizers, including Shampoo, SOAP, and Muon. Through carefully controlled experiments, the authors disentangle two key factors, spectral normalization and variance adaptation, and find that both lead to significant performance improvements compared with their element-wise and non-adaptive counterparts. The experimental results are comprehensive and well support their findings, and the presentation is generally clear.

### Weaknesses
Although this paper presents detailed empirical analyses, it does not appear to offer important new findings. The authors only verify the effectiveness of spectral normalization and variance adaptation under a single setting, results that are already well known. Specifically, Adam corresponds to variance adaptation, while spectral normalization is employed in methods such as Shampoo and Muon. Numerous prior works have already demonstrated that these techniques improve performance across a wide range of tasks. Moreover, combining the two is not new either; the SOAP and AdaMuon papers have extensively shown that integrating spectral normalization with variance adaptation is highly effective. Therefore, it is unclear what new insights the authors provide. If the contribution is merely a systematic validation of previously established findings under a single setting, it would be difficult for such a study to be accepted at a venue like ICLR.

### Questions
The paper’s main conclusion, that both spectral normalization and variance adaptation contribute to the success of matrix-whitening optimizers, appears consistent with prior findings from works such as SOAP and AdaMuon. Could the authors clarify what new understanding or insight their analysis brings beyond these established results?

Could the authors provide more empirical evidence to explain how matrix-whitening optimizers work, rather than only reporting validation losses?

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
5

### Summary
This paper presents a systematic deconstruction of modern "matrix-whitening" optimizers. The authors' central thesis is that the success of these methods relies on two key, decoupled components: (1) spectral normalization and (2) variance adaptation.

Through a series of controlled experiments on a 162M parameter GPT-2 model, the paper argues that while spectral normalization (the "whitening" part) is beneficial, variance adaptation (the "Adam-like" part) is an "overlooked" and "crucial ingredient" that is "roughly as important as the spectral-normalizing aspect". The paper's primary conclusion is that variance-adapted optimizers (like Adam, SOAP, and AdaMuon) consistently and significantly outperform their sign-descent counterparts (like Signum, SPlus, and Muon).

### Strengths
The paper's primary strength lies in its attempt to create a controlled and rigorous experimental setup. The methodology isolates the optimizers to only the dense parameters of the Transformer and relies on a thorough, independent hyperparameter sweep for each method, which the authors use to argue against improper tuning as a confounding factor.

### Weaknesses
1. A Questionable Premise: The paper groups historically distinct optimization strategies—namely adaptive regularization methods (Adam, Shampoo) and spectral descent methods (Muon)—into the same "matrix-whitening" bucket. This re-interpretation, which frames all methods as approximations of a single "whitening metric" defined in Equation (2), is a non-standard premise that is not sufficiently justified or defended. Equation (2) computes a value different from adapative regularization methods which use gradient accumulation over optimization steps.

2. Unconvincing Baseline Performance (Shampoo): The paper's central argument for the importance of variance adaptation is severely weakened by its own experimental results for Shampoo. If variance adaptation is a "crucial ingredient," one would expect Shampoo (a variance-adapted method) to significantly outperform Muon (which the paper frames as a sign-descent method). However, the paper's own results show Shampoo-10 (Val Loss 2.963) is only negligibly better than Muon (Val Loss 2.964), and Shampoo-100 "fails to converge".
This strongly suggests the Shampoo baseline was not properly tuned. The authors' admission that they "disregard auxiliary design choices in each algorithm (e.g. learning rate grafting...)" all but confirms this. While this decision was made to isolate "core" behavior, it appears to have crippled a key baseline.

3. Confusing Experimental Comparisons: The analysis in Section 5, which compares SOAP to Muon, is confusing. A more natural and direct comparison to isolate the effect of variance adaptation would have been between SOAP and SPlus, as both operate on the same rotated eigenbasis. The paper's choice to instead compare SOAP (explicit eigendecomposition) with Muon (implicit Newton-Schulz) to critique the accuracy of spectral normalization obscures the main argument.


4. Insufficient Explanation for the Central Claim: The paper's primary takeaway—that variance-adapted optimizers (Adam, SOAP) perform better than their sign-descent counterparts (Signum, SPlus)—is presented as a major finding. However, this is a not so surprising observation. The major concern is that while the paper shows variance adaptation is important, it fails to provide a deep, novel explanation for why. 

5. Minor Errors: The abstract states that matrix-whitening serves "two purposes" but does not clearly enumerate them in the following text, Validation loss 0.4 --> 0.04 in discussion section.

### Questions
Did you try grafting the Frobenius norm of Muon to Shampoo?

### Soundness
2

### Presentation
2

### Contribution
1
