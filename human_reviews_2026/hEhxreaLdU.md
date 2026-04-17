# Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges

- Decision: Accept (Poster)
- Scores: 4, 8, 8, 2

## Abstract
Reliable certification of Large Language Models (LLMs)—verifying that failure rates are below a safety threshold—is critical yet challenging. While "LLM-as-a-Judge" offers scalability, judge imperfections, noise, and bias can invalidate statistical guarantees.
We introduce a "Noisy but Valid" hypothesis testing framework to address this. By leveraging a small human-labelled calibration set to estimate the judge's True Positive and False Positive Rates (TPR/FPR), we derive a variance-corrected critical threshold applied to a large judge-labelled dataset. Crucially, our framework theoretically guarantees finite-sample Type-I error control (validity) despite calibration uncertainty. This distinguishes our work from Prediction-Powered Inference (PPI), positioning our method as a diagnostic tool that explicitly models judge behavior rather than a black-box estimator.
Our contributions include: (1) Theoretical Guarantees: We derive the exact conditions under which noisy testing yields higher statistical power than direct evaluation; (2) Empirical Validation: Experiments on Jigsaw Comment, Hate Speech and SafeRLHF confirm our theory; (3) The Oracle Gap: We reveal a significant performance gap between practical methods and the theoretical "Oracle" (perfectly known judge parameters), quantifying the cost of estimation.
Specifically, we provide the first systematic treatment of the imperfect-judge setting, yielding interpretable diagnostics of judge reliability and clarifying how evaluation power depends on judge quality, dataset size, and certification levels. Together, these results sharpen understanding of statistical evaluation with LLM judges, and highlight trade-offs among competing inferential tools.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a well-motivated novel approach at verifying LLM judge capabilities through noisy hypothesis testing with minimal required human annotations. They have a large set of LLM judge scores and a small set of human scores and use these two sets to identify the FPR and TPR of the LLM judge scores in relation to human scores. They are then able to perform noisy hypothesis testing using these parameters to adjust the associated threshold with the target prediction to be either accepting or rejecting a model. They provide the first evaluation of the imperfect judge hypothesis testing setting, evaluating how power depends on judge quality, dataset size, and certification levels. They perform empirical validation on three datasets across a range of LLMs and LLM judges to support their theoretical framework.

### Strengths
This paper addresses a very important and significant problem: How do we trust the outcomes of LLM judges if the judge itself is imperfect? The authors do a good job of motivating this problem and outlining related methods that attempt to answer this query. The empirical results of the noisy hypothesis testing with oracle parameters indicates that this formulation has potential for high fidelity acceptance or rejection of downstream models conditioned on the knowledge of the TPR and FPR. The empirical results are well motivated and relatively complete, covering many different model, judge, dataset combinations to assure continuity of results. At a general level, this is a novel approach to a timely concern by many in the LLM evaluation research field.

### Weaknesses
Throughout the abstract and the introduction, the contributions are not clear. The authors provide limited analysis of results besides the introduction of raw figures and theoretical framing of the evaluation approach. It would be helpful if authors quantified results numerically. In the introduction "We show our procedures outperform conventional ones in terms of type-II error" could be augmented with what that delta of outperformance is or some other clear way to measure progress.

Figure 1 is generally confusing. Fig 1 A-C needs a legend. Does D indicate that there is only a limited part of the plane where noisy hypothesis testing outperforms traditional hypothesis testing? If so, a discussion of this finding/shortcoming is needed. Figure 2 should have a caption.  Further, while the individual plots show differences between varying values of TPR, FPR, nM and nJ, the impact of these changes are not analyzed. It would be helpful/interesting to more systematically evaluate the whole landscape of parameters such that a recommendation could be given to a practitioner-- if you have this FPR/TPR, this many annotations would be needed for this result vs that result such that people could better understand the role of scaling annotation in this setting.

"PPI baselines typically outperform noisy hypothesis testing; however, PPI baselines do not outperform oracle noisy hypothesis testing; this then suggests there may be scope to improve these baselines via judge modelling." If I understand correctly, this supports that PPI outperforms the recommended noisy hypothesis testing method when the parameters are estimated from human annotation subsets. More discussion is needed about why PPI outperforms HT here. This solidifies that the contribution is the noisy hypothesis testing formulation and not the actual method of evaluation since PPI tends to have less error.

Generally, the presentation of the results make it hard to ascertain any individual trends/takeaways. You could aggregate data and show in a table such that it is easier to clearly tell what the best methods are for each error type and under what settings.

### Questions
A baseline of using the human annotations directly to accept or reject would be helpful to see if the only benefit of the judge is increased nJ for statistical significance. If we are using the subset of human annotations to calibrate judge quality, what happens if we just use these subset annotations directly to accept/reject the model?

How does the thresholds change if FPRs are really big (you show a max of FPR=0.25, but don't show a full range of parameter effects)?

Need to discuss why you use hypothesis testing as opposed to other traditional forms of LLM judge evaluation approaches (Cohens kappa, ICC, etc.), or at least how your work is positioned among other traditional LLM judge by human judge calibration methods.

In empirical studies, how does the subset TPR and FPR change? You report the oracle parameters but what about the estimators for TPR and FPR? How much closer to the true TPR and FPR are the subset TPR and FPR for 100 nM than 25 nM? What is the scaling law of nM? Again, the current reporting of information in multiple line graphs makes trends/scaling laws hard to extract systematically from the work.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies LLM-as-a-Judge evaluation when judges are imperfect. It frames certification as one-sided hypothesis testing on a model’s failure rate $R_M$, but runs the test on a large, noisy, judge-labeled set after calibrating the judge’s TPR/FPR on a small human-labeled set. The method maps the threshold $\alpha$ to a judge-space threshold α′=FPR+(TPR−FPR)α, derives a variance-corrected critical value that accounts for uncertainty in $\widehat{\text{TPR}}, \widehat{\text{FPR}}$, and proves finite-sample type-I control with power characterization. Experiments on synthetic data, toxic-comment classification (Jigsaw/Hate Speech), and a safety (SafeRLHF) generative setup broadly match the theory and compare against direct testing, PPI, and an oracle variant.

### Strengths
1. Algorithm 1 includes an explicit critical value with variance terms from judge-parameter estimation; type-I control and type-II bounds are proved (Berry–Esseen based).


2. Experiments cover both classification and generative settings with multiple judges; qualitative alignment with theory increases confidence in the claims.

### Weaknesses
1. Critical values use normal/Berry–Esseen approximations. There’s no comparison to Wilson/Clopper-Pearson style bounds when $n_M$​ is small or failures are rare—precisely when certification is most needed.



2. The judge prompts and aggregation choices can materially change TPR/FPR. The paper doesn’t investigate prompt variants, majority-vote vs. single-judge sensitivity, or robustness to minor instruction changes, despite known judge prompt sensitivity.

### Questions
1. It’s better to provide advice to help practitioners choose sample sizes or decide when the method beats direct testing.



2. Do you observe positive correlation between model failures and judge errors on the human-labeled set (e.g., both struggle on long or sarcastic inputs)? If so, how does that alter variance terms or recommended $n_M$?


3. If the judge is chosen after peeking at results, how should users adjust α\alphaα (e.g., holdout, Bonferroni/Holm) to avoid inflated type-I error?

### Soundness
3

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
3

### Summary
Establishing the reliability of an LLM is a challenge to tackle due to factors such as data leakage in benchmarking and human evaluation being costly. Recently the LLM-as-a-Judge paradigm has emerged as a tool to look at the reliability of a model, however this is heavily dependant on the quality of the judge. 

This paper develops a hypothesis-testing framework to verify whether an LLM is reliable or not, keeping the noisiness of such a judge in mind. The FPR is characterized as the probability of the LLM-as-a-judge perceiving a correct model response as incorrect and the TPR as the probability that the LLM-as-a-judge perceives an incorrect model response as incorrect. 
The type I error probability is characterized as the risk of the LLM where an unreliable LLM is incorrectly seen as reliable and type II error as when a reliable model is incorrectly seen as unreliable. 

The null hypothesis $H_0$ is defined as the true failure rate of the model being larger than the user specified threshold $\alpha$. 
As we are dealing with a noisy model setup with a noisy LLM-as-a-judge for evaluation, the formulation of the procedure is applied to the noisy LM failure rate, where $\alpha$’ is used instead: $\alpha$’ = FPR + (TPR-FPR)$\alpha$
where FPR and TPR are from the LLM-as-a-Judge. 

The procedure relies on two data sources ($D_M$: human verified ground truth, small dataset; $D_J$: LLM-as-a-judge labels, large dataset) and primarily consists of two steps: 
1. The LLM-as-a-Judge is first employed on $D_M$ to get the TPR and FPR, which in turn is used to get the adjusted threshold $\alpha$’ 
2. The hypothesis testing is then done on D_J, to determine whether the null hypothesis can be rejected or not

The paper also contains a theoretical analysis with theoretical guarantees for the procedure in terms of Type I and Type II error probabilities in the scenario where we have access to the true TPR and FPR of the LLM-as-a-judge (oracle) and when we have to estimate it ourselves (the real noisy scenario). 
When empirically applying the hypothesis testing to a synthetic setting and classification and generation settings, the theoretical guarantees largely hold. Noisy hypothesis testing outperforms direct hypothesis testing, but does not always outperform other baselines such as PPI. In contrast, the oracle scenario does outperform all. 

The framework can help in model comparison and judge selection, which is crucial for LLM reliability.

### Strengths
1. This paper tackles a highly relevant problem of determining the reliability of LLMs. Benchmarks do not reveal the true capabilities of an LLM and the bias of an LLM-as-a-Judge also does not provide reliable insights. I thus feel using such a hypothesis testing framework is very useful and the need of the hour where there are so many options for which LLM can be used.
2. The approach is grounded in reality. It uses a mix of small-scale human data (which is expensive to get) and large-scale LLM-as-a-Judge data (which is easier to get); this is highly reflective of most LLM setups. Furthermore, it nicely incorporates the popular LLM-as-a-Judge paradigm to make hypothesis testing more appropriate in such settings. 
3. The paper consists of both theoretical guarantees as well as empirical results that tie in nicely both model performance as well as LLM-as-a-Judge capabilities and is done in both a synthetic and real dataset scenario with both classification and generation types. 
4. The paper is also nicely structured which helps the reader understand the methodology nicely. Especially Figure 2 is very helpful.

### Weaknesses
1. I think discussion around the takeaway of the procedure could be clearer. The findings highlight that noisy hypothesis testing outperforms direct hypothesis testing in certain regimes where the TPR is higher and FPR is lower; what does this mean for the takeaway? An overview of in which scenarios the procedure signs would be very helpful. E.g., also the oracle outperforms all but how realistic is this oracle setting also? It would be nice to incorporate this explicitly. 
2. Related to the previous point, but if PPI performs better than noisy hypothesis testing, what are its limitations that would make me opt for noisy hypothesis testing instead? 
3. I personally feel that having the Related Work right after the Introduction would be nicer. For me as a reader, seeing PPI used as a baseline was not as straightforward as after reading the Related Work section at the end. It kind of felt like PPI came out of nowhere. A bit more explanation on what PPI is / how it works could also likely help with point 2. 
4. Not necessarily a weakness but I think a nice discussion point would be LLM-as-a-judge in the scenario for subjective tasks, which is also an issue with human evaluation; who are we asking. Would this be interesting for future work, would the framework need some adaptation or can it be used as such?

### Questions
1. Why do you opt for different LLMs as the base model and different LLMs for LLM-as-a-judge? E.g., Mistral 7B only as LLM-as-a-judge? Is there a specific design choice behind it? 
2. Just to ensure if I understood correctly; in Figure 2, $D_M$ is in a separate box but the next steps do not use $D_M$ separately right, only in the mixed $\tilde{D}_M$? 
3. There is a typo on line 428; an extra “for” in front of “judges”

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
The paper proposes using hypothesis testing to estimate the LLM-judge's imperfection by calibrating their false positive and true positive rates using a small human-annotated dataset. This is different from prediction-powered inference, which relies on human annotations to model the judes.

### Strengths
- The paper addresses an important challenge in evaluating large models, leveraging statistical frameworks such as hypothesis testing.  

- The method is compared against a prediction-powered inference approach, and the paper notes that PPI often outperforms both oracle-noisy upper bounds. These are good findings.

### Weaknesses
- I believe that relying on a small dataset to calibrate the automatic judges will depend a lot on the task, model, and the quality of the collected data. 

- This paper is very challenging to read. I miss very basic motivations and illustrations of the core ideas of the work. For example, none of the figures make the paper accessible. 

- The theoretical insights should have made the paper's contributions relevant. On the contrary, these insights are full of jargon and formulations that are not relevant.

### Questions
- Could you please provide examples of when the methods are successful and when they fail? 

- I strongly recommend that the paper be revised and rewritten with a clear presentation, reducing jargon and ensuring that the methods and findings are understandable to a wider audience. For example, follow a clear structure (with visualisations): how do you calibrate and why? How do you measure this calibration and compare it to PPI as well as noisy conditions? Give concrete examples. The current version of the paper feels like a very long abstract.

### Soundness
2

### Presentation
1

### Contribution
2
