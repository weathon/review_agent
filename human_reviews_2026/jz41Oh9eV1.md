# Robust Preference Alignment via Directional Neighborhood Consensus

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Aligning large language models with human preferences is critical for creating
reliable and controllable AI systems. A human preference can be visualized as a
high-dimensional vector where different directions represent trade-offs between
desired attributes (e.g., helpfulness vs. verbosity). Yet, because the training data
often reflects dominant, average preferences, LLMs tend to perform well on com-
mon requests but falls short in specific, individual needs. This mismatch creates
a preference coverage gap. Existing methods often address this through costly
retraining, which may not be generalized to the full spectrum of diverse preferences.
This brittleness means that when a user’s request reflects a nuanced preference
deviating from the training data’s central tendency, model performance can degrade
unpredictably. To address this challenge, we introduce Robust Preference Selection
(RPS), a post-hoc, training-free method by leveraging directional neighborhood
consensus. Instead of forcing a model to generate a response from a single, highly
specific preference, RPS samples multiple responses from a local neighborhood
of related preferences to create a superior candidate pool. It then selects the re-
sponse that best aligns with the user’s original intent. We provide a theoretical
framework showing that, under mild conditions where (i) nearby preference direc-
tions correspond to better-trained regions of the model and (ii) the reward-model
scores change smoothly with small angular changes in the preference vector, our
neighborhood generation strategy yields a higher expected best score than a strong
baseline that also samples multiple candidates. Comprehensive experiments across
three distinct alignment paradigms (DPA, DPO, and SFT) demonstrate that RPS
consistently improves robustness against this baseline, achieving win rates of up
to 69% on challenging preferences from under-represented regions of the space
without any model retraining. Our work presents a practical, theoretically-grounded
solution for enhancing the reliability of preference-aligned models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Having identified what they call a "preference coverage gap", where the targeted user preferences might differ significantly from choices made during the training of an LLM, the authors propose to handle it at inference time, introducing a "Robust Preference Selection," or RPS. Instead of asking a model to generate for a target outside of its domain, they first sample a neighborhood of more familiar preferences ; then generate a response for each adapted vector ; and pick the best response according to the original target preferences.
The robustness gain of the RPS method is both presented formally and validated experimentally, as they compare the performances of 3 training paradigms (DPA, DPO and SFT) coupled with RPS on 3 datasets (UltraFeedback, HelpSteer and HelpSteer2). Interestingly the experiments confirm its soundness as target preferences go outside of the training distribution.

### Strengths
- this is a pragmatic contribution as it shows a practical way to improve robustness at inference when generating from an LLM outside the preferences learned during its training ;
- despite the relative simplicity of the approach, the conceptual link with Distributionally Robust Optimization is theoretically and philosophically interesting ;
- the experimental results do illustrate the validity of the approach ;
- the paper is well written and solid, both formally and experimentally.

### Weaknesses
- the simplicity of the approach (see "strengths" above) diminishes the contribution which really boils down to: instead of asking the model to generate outside of its domain it's better to keep it closer to home _and_ then pick the response closer to what the user wanted ;
- the verbosity vs helpfulness example used here is somewhat intuitive but it is not clear (to me?) how much it is a trade-off, so the single theta controlling both dimensions can be seen as problematic. In other words there could very well be a long, helpful answer ;
- only the DPA model is trained to use correctly the dimensions in the prompt, the others (DPO, and SFT) might do their best but there's no calibration. Yes, it does seem to work here but would it apply to more complicated cases, than verbosity vs helpfulness?

### Questions
- can you comment on this artificial trade-off you see in verbosity vs helpfulness? And the single theta you use as a control?
- couldn't we just generate more samples, and use a better scorer?
- can you comment on the correlation of the weights used in the prompts and the semantic of the generated answers, especially for DPO and SFT ofc?
- what about using human judges for something as difficult to assess, even in this simple (verbosity, helpfulness) case?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To align user preferences at inference time, previous studies introduce a preference vector (e.g., helpfulness vs. verbosity) in the prompt to adjust the model’s behavior. However, these approaches often underperform outside their training domains, thus requiring extra training. In this work, the authors propose **Robust Preference Selection (RPS)**, a post-hoc method for improving preference alignment during inference. It samples a set of neighboring vectors from the target one and generates responses with each, then selects the optimal one using a reward model. Experimental results demonstrate that RPS attains a higher win rate compared to naive-sampling baselines.

### Strengths
- The paper is well-written, the motivation is clear, and the teaser figures are intuitive and easy to follow.
- The authors present theoretical evidence for their proposed method, showing the effectiveness of RPS under certain assumptions.

### Weaknesses
- Assumption 1 appears rather idealized, and the paper provides limited empirical evidence to support it. Although the authors mention that Figure 5 offers some justification, a deeper analysis or additional experiments would help validate it.
- The paper lacks ablation studies on the choices of $k$ and $\theta_{\max}$; without these, it is difficult to assess whether the method is sensitive to hyperparameters.
- The assumption of a well-calibrated reward model that generalizes to out-of-distribution (OOD) data seems overly strong and may not hold in other domains.

### Questions
- It seems that the authors utilize models from prior works. In this case, what is the training distribution of each model? How do the authors ensure that the testing range of $10^\circ$ to $45^\circ$ indeed includes out-of-distribution (OOD) cases?
- The prompt for the LLM judge appears quite simple, raising doubts about whether the evaluation is truly robust in a zero-shot setting. Have the authors tested the robustness of the LLM judge by sampling multiple times?
- How does the value $v_{\text{target}}^{\top} r(x, y)$ compare with the baseline? Does it outperform the baseline across all angles as well?

### Soundness
2

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
This paper addresses the preference coverage gap, a problem where LLMs aligned on dominant, average human preferences perform poorly on specific, out-of-distribution requests. To mitigate this brittleness without costly retraining, the work introduces Robust Preference Selection (RPS), a training-free, post-hoc adjustment method. The work identifies the preference coverage gap, as LLMs tend to perform well on common requests but falls short in specific, individual needs. RPS solve this by generating a candidate pool of responses from a local neighborhood of more "in-distribution" preference vectors, rather than directly from the out-of-distribution target preference. The final response is chosen from this pool by selecting the candidate that best aligns with the original target preference. The paper present both theoretical and empirical analysis for RPS framework, showing the validity and soundness of the proposed approach.

### Strengths
First, the paper proposed a conceptually novel method, RPS with a clear motivation. Rather than attempting to force a model to directly generate a high-quality response for a difficult, out-of-distribution (OOD) preference, RPS reframes the problem. It hypothesizes that it is more effective to first sample from a neighborhood of related, but easier, preference vectors where the model is inherently more competent. This conceptual shift from direct, constrained generation to a "generate-then-select" paradigm and provides new insights into LLM alignment.

Second, the proposed solution is practical and (potentially) broadly applicable due to its post-hoc, training-free nature. Unlike various methods that require extensive retraining or fine-tuning to the model architecture, this work offers a lightweight option where it can be implemented at inference time on the pre-trained models. The simplicity of the algorithm, generating from slightly perturbed preference vectors and then re-ranking, ensures a low barrier for adoption, making it a highly valuable tool for practitioners seeking to improve model robustness in real-world applications.

Third, the paper provides comprehensive empirical validation to support its claims. The authors go beyond a simple performance comparison by establishing a strong, compute-matched baseline. The consistent win rates of RPS against several baseline across a diverse set of models, including those trained with SFT, DPO, and DPA, demonstrate the method's generalizability. Crucially, the analysis that correlates the performance gain of RPS with the degree of OOD-ness of the preference provides compelling evidence for the paper's core hypothesis. This result shows that the method is effective precisely in the challenging scenarios it was designed to address. Furthermore, the paper gives relatively complete theoretical justification of using RPS for model alignment.

### Weaknesses
First, the paper's theoretical claim of being "provably superior" rests on a critical yet unformalized logical gap, which will undermine its rigor. The entire argument of Theorem 1 hinges on the "local consistency" assumption, stated as v_target^T r(x, y_i) ≈ v_i^T r(x, y_i). This approximation is presented without any formal justification, error bounds, or discussion of the conditions under which it might hold. Consequently, the strong language of "guarantee" and "proof" is a mischaracterization; the theoretical contribution should be more accurately and detailed framed as a heuristic argument.

Second, the paper's exclusive reliance on a single automated metric, the judgment of GPT-4o-mini, introduces a potential confounder that is not adequately addressed. While LLM-as-judge is a common practice, it is known to have biases. Without a supporting human evaluation study or an analysis using multiple distinct judge models to check for consensus, it is hard to conclude that the observed win rates reflect a true improvement in response quality rather than an artifact of the specific evaluation protocol.

### Questions
1.	Are results robust to different judges (e.g., GPT-4o, open-source preference models)? Can a small human evaluation be included?
2.	The theoretical argument hinges on the "local consistency" assumption. Can this be formalized? For example, under what conditions on the reward model r and the distance between v_i and v_target does this approximation hold with a bounded error? Without this, the claim of a proof seems difficult to justify.
3.	What happens in higher-dimensional preference spaces (≥3 attributes)? If there is any preliminary results?  
4.	If it is possible to conduct an ablation study on hyperparameters such as k and θ_max?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Robust Preference Selection (RPS), a three-phase, training-free method designed to address the “preference coverage gap”, where language models fail to align with out-of-distribution user preferences. The authors provide a theoretical framework to justify this approach and present experiments under three preference-learning datasets across three alignment paradigms to demonstrate the superiority of RPS.

### Strengths
- The preference selection method is training-free, making it applicable to models trained under different schemes.
- Theoretical analysis shows that the expected score of the best response selected by RPS is greater than that of the best response selected by the baseline.

### Weaknesses
- The method’s generalization to higher-dimensional preference spaces is not empirically validated.
- RPS assumes that the reward model used in the Consensus Selection phase (Phase 3) is robust when evaluating responses against OOD targets ($v_{\text{target}}$), potentially shifting the “brittleness” problem from generation to evaluation.
- The theoretical foundation relies on two key assumptions: Assumption 1 and the local consistency assumption (L203). Assumption 1 requires the neighborhood vector $v_i$ to differ from the OOD target $v_{\text{target}}$, whereas the local consistency assumption requires them to be sufficiently similar for the final evaluation score to be transferable.
- The paper lacks user studies or alternative automated reward metrics that could demonstrate RPS’s superiority.

### Questions
- Could the authors provide empirical evidence supporting the local consistency assumption, particularly under broader settings such as $\theta_{\max}=30^\circ$?
- Could the authors provide an ablation study using other reward models to demonstrate that the RPS framework is generalizable?
- How does the proposed inference-time approach compare against strong training-time optimization baselines such as GRPO [1]?

[1] Shao, Zhihong, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv preprint arXiv:2402.03300 (2024).

### Soundness
2

### Presentation
3

### Contribution
2
