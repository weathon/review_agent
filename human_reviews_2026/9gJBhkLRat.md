# Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Large Language Models (LLMs) should refuse to answer questions beyond their knowledge. This capability, which we term knowledge-aware refusal, is crucial for factual reliability, while existing metrics fail to capture this ability. In this work, we propose the Refusal Index (RI), a novel and principled metric that measures how accurately LLMs refuse questions they do not know. We define RI as Spearman's rank correlation between refusal probability and error probability. RI is practically measurable with a lightweight two-pass evaluation method which only require observed refusal rates across two standard evaluation runs. Extensive experiments across 16 models and 5 datasets demonstrate that RI accurately quantifies a model's knowledge-aware refusal capability. Notably, RI remains stable across different refusal rates and provides consistent model rankings independent of a model's overall accuracy and refusal rates. These properties suggest RI captures a stable, intrinsic aspect of model knowledge calibration. More importantly, RI provides insight into an important but previously overlooked aspect of LLM factuality: while LLMs achieve high accuracy on factual tasks, their refusal behavior can be unreliable and fragile.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper makes a contribution to the field of LLM evaluation, specifically addressing the critical issue of "knowledge-aware refusal."  Firstly, the authors clearly identify a major gap in existing evaluation methodologies for LLM refusal behavior. Therefore, they propose the primary contribution-the Refusal Index (RI), a rigorously defined metric based on Spearman’s rank correlation between a model's refusal probability and its error probability. This metric is designed to directly and faithfully quantify a model's intrinsic capability to refuse questions beyond its knowledge. The extensive empirical validation across 16 models and 5 datasets demonstrate that RI is stable, consistent, and independent of a model's overall accuracy and refusal rate.

### Strengths
1. The paper tackles a critical and under-explored problem in LLM reliability—"knowledge-aware refusal." The introduction of the Refusal Index (RI) is a novel and timely contribution that addresses a clear gap in the existing evaluation landscape.

2. The paper provides extensive and compelling experimental evidence to support its claims. Experiments across 16 models and 5 datasets offer a robust demonstration of RI's stability, consistency, and superiority over existing metrics, and further delivers some insightful ideas.

3. The proposed two-pass evaluation method for estimating RI is lightweight and practical. This thoughtful design makes the metric feasible for researchers and practitioners to adopt without requiring excessive computational resources, enhancing its potential impact.

### Weaknesses
1. Although the empirical results of this paper is promising, the technical contribution seems not solid and sound. The rationality of the method is not well presented. Therefore, the technical validity is not convincing.

2. The paper is not well written. There are many concepts introduced in this paper. However, these concepts are not rigorously clarified. The details can be found in Questions.

### Questions
1. In section 2.1, why choose bivariate gaussian distribution to model the probablilty value of error and refusal. Does some empirical results or previous works support this point? I think it is a rough characterization.

2. How does the rank in the definition of spearman's rank correlation play the role in the estimation of RI?

3. In equation (3), how can we obtain the value of $r_i$ and $w_i$ ?

### Soundness
2

### Presentation
1

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
This paper aims to evaluate a model’s level of refusal solely based on its output text. To this end, the authors propose a new evaluation metric called the Refusal Index (RI). Unlike traditional metrics that are heavily affected by the refusal rate, RI remains stable across different refusal rates. The paper validates the effectiveness of RI on multiple models and datasets.

### Strengths
1.This paper observes that traditional metrics are highly affected by the refusal rate and proposes a new metric called RI.

2.The overall writing of the paper is clear and fluent.

3.The experiments are thorough, and the effectiveness of RI is validated across multiple models and datasets.

### Weaknesses
1. The paper lacks sufficient explanation of traditional evaluation metrics. In the introduction, the authors briefly mention some of their weaknesses, but it remains unclear what these metrics actually are and why they exhibit such shortcomings. This background is essential for understanding the motivation of your proposed approach. I suggest moving this part to Section 2 and clearly introducing the limitations of traditional metrics before presenting your own.

2. The authors claim that external calibrators, such as verbalized confidence or linear probes, cannot replace direct refusal measurements. However, the rationale for this statement is not clearly articulated. Could the authors elaborate on why these methods are unsuitable? For instance, since we can explicitly train models to output verbalized confidence, it is not immediately clear why such signals cannot serve as a proxy for model confidence or be used in place of direct refusal measures.

3. Starting around line 129, the definition of the key notion raises potential confusion: are refusals also counted as errors? This point needs clearer explanation. Later sections suggest that each question has its own error rate (and refusals are re-answered for measurement), but this is not obvious when first introduced. Additionally, in Table 1, the formula c/(1–r) is unclear — does c represent the number of correct answers among the non-refused cases? Please clarify this.

4. The proposed metric seems conceptually related to AUROC, which also reflects the consistency between confidence and ability. This is similar to your statement that “its refusal probability increases monotonically with error probability.” Could the authors explain more explicitly how their metric fundamentally differs from AUROC? Why should I use RI instead of AUROC? Refusal probability can also be reflected through confidence scores, rather than being binarized into a simple “refuse or not” decision.

5. The “refusal tendency” appears analogous to a fine-grained confidence estimation, while the “error tendency” seems related to accuracy over multiple responses. Have the authors explored this connection? Why not estimate model confidence directly and then threshold it according to user preferences, instead of measuring refusal explicitly.

6. Since the proposed metric relies on Gaussian estimation, I am concerned about its robustness under limited sample sizes. How accurate is the estimation in such cases? Furthermore, is the Gaussian assumption itself empirically justified?

7. The choice of baselines could be further discussed. For fine-grained confidence, AUROC and ECE are typically appropriate. For binary confidence, accuracy (as a measure of ability) and alignment (whether refusal matches correctness) might be more relevant.
Clarifying why your chosen baselines are suitable would strengthen the experimental section.

8. While the paper discusses the influence of refusal rate, accuracy also substantially affects metric behavior. For example, AUROC can appear artificially high under extremely imbalanced accuracy (e.g., only 10 correct samples out of 1000). This suggests that model ability significantly impacts evaluation. I am also curious about the authors’ perspective on how model competence affects alignment — if a model is either very strong or very weak, learning when to refuse may become trivially easy. Is this an expected or desirable property?

### Questions
See Weaknesses.

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
3

### Summary
- This paper investigates knowledge-aware refusal in LLMs—the ability to refuse questions they are unlikely to answer correctly while not refusing questions they can answer. The authors identify that existing metrics like correctness conditioned on non-refusal are easily biased by refusal tendency and manipulated through system prompts.
- The authors propose the Refusal Index, which measures the Spearman correlation between refusal probability and error probability. They develop an efficient estimation procedure using Gaussian copula fitting that requires only two samples per question, making it tractable compared to expensive sampling-based calibration methods.
- The paper validates this metric by showing: (1) stability across different prompts for refusal tendencies, (2) clean correlation with sampling-based calibration methods, and (3) consistent model rankings across evaluation settings.
- Using the validated metric, the authors show: (1) cautious prompts increase refusal rates but do not improve calibration, (2) the Refusal Index is largely independent of model capability and aligns more with model family, and (3) removing or adding misleading context degrades refusal calibration.

### Strengths
- The Refusal Index captures something fundamental about a model's calibration and remains stable across different prompting strategies. 
- Despite the mathematical complexity, the metric requires only two samples per question—nearly as cheap as computing accuracy. 
- RI has strong correlation with expensive sampling-based calibration metrics 
- Interesting and diverse experimental results. The finding that refusal calibration is largely independent of model capability and instead aligns with model family is particularly striking—it suggests calibration may be a distinct dimension of model quality worth optimizing separately.

### Weaknesses
- The mathematical presentation of RI feels dense. Figure 1 hints that the Refusal Index captures convexity of the curve in refusal rate vs. correct answer space, and further developing this intuition and/or motivating the Gaussian copula fit could aid clarity.
- Greater discussion on whether the Refusal Index measures something fundamentally different than sampling-based calibration methods, or simply serves as a more sample-efficient proxy, would be helpful. The appendix contains interesting results on sample efficiency—a direct head-to-head comparison showing how much more sample-efficient RI is than naive calibration-based metrics would better motivate its advantages.
- The stability results in Table 2 focus on Qwen and Mistral models, but Figure 4 shows Gemma-3-12b's Refusal Index varying considerably across prompts (roughly 0.1 to 0.3). It's unclear whether Gemma is an outlier or if this variation is typical. Showing both the Figure 4 analysis and Table 2 stability results on a broader, consistent set of models would clarify how stable the metric actually is in practice. This concern applies more broadly to later experiments—either evaluate more models consistently or be more intentional about which models are presented and why.
- The frontier model evaluation figure is interesting, but it's not possible to identify which specific model corresponds to each data point—only the model family is discernible, not the generation or size.

### Questions
- How is stability computed in Table 2? Could differences in distribution concentration affect the apparent variability of different metrics? What level of RI variation across prompts (e.g., 0.1 to 0.3 for Gemma-2-9b) should be considered acceptable? Would it be possible to show empirical data points on iso-RI curves (as in Figure 3) for more models? This would help clarify whether the Gemma-2-9b variability pattern is typical.
- The finding that RI aligns with model family rather than capability is interesting, but it's difficult to identify specific models in Figure 5. Could you provide clearer labels or a table showing individual model RI scores?
- Is RI measuring something fundamentally different from sampling-based calibration methods, or is it primarily a more sample-efficient approximation? A naive alternative would be to estimate RI through extensive sampling without Gaussian copula fitting—how would this compare?

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
In the blackbox approach to factual knowledge, the authors point out that current methods are limited. They will mostly look at the rejection rate and/or correct answer rate. Here, the authors propose to check whether those two things happen together, beyond chance. This method additionally avoids a lot of sampling which is often relied on for similar approaches. They provide extensive empirical testing on different models and datasets, and discuss model variations. They also check effect of prompt variation.

### Strengths
Reduces a true gap in a very empiric field. As an empirical study it is quite solid, checking a variety of models, datasets, but also checking the effect of prompt variation to some extent. 

The empirical section is not only quantitatively very strong, but actually makes good use of it's data. Datasets and models are not only listed, but are used to make compelling arguments on different effects. Figures are clear and very understandable. Multiple appendices make the effort of testing many variations of the setup to ensure it is correctly validated.

### Weaknesses
An issue I find generally in the blackbox tradition of model factual knowledge is that a lot of definitions are arbitrary. 
I would worry that these poorly defined targets of "checking if a model knows" or "checking if a model answers when it knows" are moving goalposts which lead to incremental progress to evaluate models which purposedly (hence blackbox) do not provide the required information to properly move forward.

Were this a journal paper in a major venue, I would ask to rework the definition of "knowledge". As this is a conference paper in a major venue I can only notify that this definition is very shaky (we are not discussing observable correct answer for a factual question, but a more interesting but very abstract notion of "knowing" a fact). I nonetheless acknowledge that this paper is taking a step in the right direction by decoupling model behaviour of refusal to answer from model knowledge, and this is why I've set a positive score. 
I remain nonetheless worried that definitions are not well set - much like in previous empirical works, there is no clear gold standard for "knowing". Reasoning then becomes somewhat circular - we empirically define knowing as RI, and then show that it is better than previous methods which had set different definitions. 

Along the same line the authors criticize the notion of proxy metrics. I did not understand how this new RI method is not a proxy metric, even if a better one. 

On a much less important note, I've noted moments where I was confused reading. Should they seem personal, feel free to ignore them.
* Line 015/016 : "simple refusal based metrics are biased by refusal rates and yield inconsistent scores when models exhibit different refusal tendencies" --> confusing
* starting L050: I was confused again by the third paragraph of the introduction.

For both of those I only understood what was going on from the examples in l110 onwards which made your point as well as the difference between refusal based metric, refusal bias, refusal tendencies, and refusal itself as a concept much clearer. I would advise either clarifying earlier, or rephrasing.

* 024 "RI accurately quantifies a model's intrinsic knowledge-aware refusal rates capability in factual tasks." --> intrinsic knowledge aware refusal rates is not defined later in the paper, and confusing here

### Questions
1) could you please re-explain why you consider calibration a proxy, and not RI?

2) L151/152 "While overall refusal rates can be adjusted through input context or preference learning, the discriminative capability for knowledge-aware refusal remains more robust and consistent" - I am confused by this statement. How can changing refusal rates be more consistent than discriminating? more consistent for what? I think I understand your point that refusal rates are not the only thing we want to act on - but I don't think RI as a metric is acting on anything.

3) more of a personnal curiosity point: is there a reason why you are using Spearman rank correlation rather than another?

### Soundness
3

### Presentation
3

### Contribution
3
