# K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
The rapid development of visual generative models raises the need for more scalable and human-aligned evaluation methods. While the crowdsourced Arena platforms offer human preference assessments by collecting human votes, they are costly and time-consuming, inherently limiting their scalability. Leveraging vision-language model (VLMs) as substitutes for manual judgments presents a promising solution. However, the inherent hallucinations and biases of VLMs hinder alignment with human preferences, thus compromising evaluation reliability. Additionally, the static evaluation approach lead to low efficiency. In this paper, we propose K-Sort Eval, a reliable and efficient VLM-based evaluation framework that integrates posterior correction and dynamic matching. Specifically, we curate a high-quality dataset from thousands of human votes in K-Sort Arena, with each instance containing the outputs and rankings of K models. When evaluating a new model, it undergoes (K+1)-wise free-for-all comparisons with existing models, and the VLM provide the rankings. To enhance alignment and reliability, we propose a posterior correction method, which adaptively corrects the posterior probability in Bayesian updating based on the consistency between the VLM prediction and human supervision. Moreover, we propose a dynamic matching strategy, which balances uncertainty and diversity to maximize the expected benefit of each comparison, thus ensuring more efficient evaluation. Extensive experiments show that K-Sort Eval delivers evaluation results consistent with K-Sort Arena, typically requiring fewer than 90 model runs, demonstrating both its efficiency and reliability. The dataset and code are publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes K-Sort Eval, a framework for evaluating visual generative models that uses a Vision-Language Model (VLM) as a judge. The goal is to create a more scalable and efficient alternative to human-based "Arena" evaluations. The method leverages a curated, high-quality dataset of human preference rankings from the existing K-Sort Arena. It introduces two key technical contributions: 1) a "posterior correction" method to align VLM judgments with human supervision by modeling VLM-human discrepancies as observation noise, and 2) a "dynamic matching" strategy to efficiently select the most informative comparisons. Experiments show that K-Sort Eval's rankings are consistent with the human-based K-Sort Arena leaderboard, while requiring significantly fewer (under 90) model runs.

### Strengths
1. Addresses a Critical Problem: The paper tackles the highly relevant and important challenge of scalable, efficient, and human-aligned evaluation for generative models, which is a major bottleneck in the field.

2. Novel Correction Mechanism: The "posterior correction" method (Section 3.2) is a novel and clever approach to mitigate the known biases and unreliability of VLM-as-a-Judge. Modeling the discrepancy between VLM and human votes as observation noise in a Bayesian framework is a solid technical contribution.

3. Demonstrated Efficiency: The dynamic matching strategy (Section 3.3) and VLM-based approach demonstrably reduce the evaluation cost to fewer than 90 model runs. This is a significant practical improvement over the thousands of comparisons often required in large-scale human evaluations.

### Weaknesses
1. Dependency on Curated Dataset: The entire framework seems to be a "closed-world" system. The posterior correction (Eq. 12) appears to require a pre-existing human-voted ranking ($R^i$) for the exact same prompt to calculate the correction factor $\lambda'$. This suggests the method can only be used to evaluate new models on the static, pre-existing prompts (500 for images, 300 for videos) from the curated dataset, which severely limits its generalizability for open-ended evaluation on new, unseen prompts.

2. Potential for "Overfitting" to the Leaderboard: The dataset curation process itself (Section 3.1) filters out human votes that are inconsistent with the overall K-Sort Arena leaderboard. This creates a risk of circularity: the "ground truth" used to build the evaluator is already a product of the system it's meant to replicate. This could cause the evaluator to simply "overfit" to the specific biases of the K-Sort Arena, rather than capturing a more general or robust notion of human preference.

3. Unclear Scalability of Supervision: The reliance on dataset supervision for the correction mechanism is a key weakness. If the method does require human ranking data for every new prompt to be evaluated, it doesn't solve the scalability problem; it just shifts it. If it doesn't require this, the paper fails to explain how the correction $\lambda'$ is calculated for a truly new prompt that isn't in the curated dataset.

4. Limited Ablation on VLM Choice: The choice of VLM (GPT-4o, Qwen-VL-Max) is critical, yet its impact is not deeply explored. The ablation study (Table 5) shows that removing posterior correction dramatically changes the model rankings (e.g., from 3 to 6 for CogVideoX-5b), proving the VLM's raw judgment is unreliable. This makes the system highly dependent on the (unexplored) interaction between the specific VLM and the specific correction dataset.

### Questions
1. Could you please clarify the workflow for evaluating a new model on a novel prompt that is not in the curated dataset? How is the posterior correction (Eq. 12) applied in that scenario, given that there is no "dataset supervision" $R^i$ to compare against? Or, is the framework strictly limited to the 800 prompts in the dataset?

2. The dataset curation (Section 3.1) uses a correlation threshold ($\tau=0.75$) to filter out human votes that deviate from the global leaderboard. How did you ensure that this process doesn't just "bake in" the specific biases of the K-Sort Arena leaderboard and remove genuinely diverse or high-variance (but still valid) human preferences?

3. The ablation study (Table 5) shows very large rank changes when posterior correction is removed. This suggests the VLM's "raw" judgment is significantly different from the human-aligned one. Could you provide more analysis on the nature of this discrepancy? For example, is the VLM consistently biased on certain criteria (e.g., aesthetics vs. alignment) or for certain models?

4. How sensitive is the dynamic matching strategy (Eq. 17) to the hyperparameters, particularly the balancing coefficient $\alpha$? The paper states it was set after a simple grid search, but it seems crucial for balancing the exploration of diverse model matchups versus challenging the new model against its closest competitors.

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
4

### Summary
This paper introduces K-Sort Eval, an efficient framework for evaluating visual generative models. The method addresses the limitations of existing approaches: human evaluation is costly and time-consuming, while using Vision-Language Models (VLMs) as judges suffers from inherent bias and hallucinations .

### Strengths
1. Reliable VLM Correction: The core strength is the posterior correction method. By modeling the VLM-human preference gap as observation noise and correcting the Bayesian update , the framework significantly improves alignment with human judgments .
2. Strong Empirical Validation: The method demonstrates high consistency with the human-voted K-Sort Arena leaderboard on both image and video tasks. Its utility is also shown in practical use cases, like evaluating compressed models .

### Weaknesses
1. Simplified Noise Model: Assumption 1—that VLM noise is statistically independent of the true model capability—is a strong simplification. It's plausible that VLM biases are systematic (e.g., favoring certain aesthetics or failing to spot specific artifacts), which this noise model would not capture.
2. Incomplete Literature Review: The "Large Model as a Judge" related work section overlooks several recent and relevant works exploring VLMs as human-aligned evaluators. For instance, it misses applications in other domains like text-to-3D generation (e.g., **"GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation"**) or specialized tasks like personalized image generation (e.g., "DreamBench++"), which could offer broader context.

### Questions
1. Have you tested the framework's robustness using different VLM judges? How much does the correction mechanism's effectiveness depend on using a top-tier model like GPT-4o?
2. Can you provide more justification for Assumption 1? Did you test for systematic VLM biases (e.g., errors correlating with model rank), which would violate this assumption?

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
5

### Summary
This paper proposed K-Sort Eval, a novel framework for efficiently evaluating visual generative models by combining VLM judgments with statistical correction mechanisms informed by large-scale human preference rankings. The system dynamically selects the most informative model comparisons and applies Bayesian posterior correction to align VLM-based scores with true human preferences. This approach dramatically reduces evaluation cost and time compared to human-only leaderboards, while maintaining high agreement with actual human rankings.

### Strengths
S1) This paper tickles the pain-point that Its difficult to evaluating visual generative models both reliably and efficiently. Human annotations are highly accurate but expensive and infeasible to scale for every model update or new dataset. Meanwhile, fast automatic scoring using VLMs is scalable, but suffers from biases. K-Sort Eval aims to deliver a solution that preserves human-grade reliability while radically improving cost-efficiency and scalability by statistically correcting VLM judgments using robust human data.​ This is a highly original work and can be generalized to apply on other Arena works.

S2) It proposed a rigorous data cleaning using Spearman correlation filtering and prompt safety screening to ensure high data quality and annotation reliability.

S3) The experiments are well designed and showing high reliably of the proposed method.

### Weaknesses
W1) When this is applied in cases where there are not enough votings, the rankings might not be fully reflecting the performances. Will such behavior propagate to the new models? Also would it be possible to find an estimate on how much collected human/ VLM annotated data are enough to make sure that K-sort eval will works effectively?

### Questions
See weaknesses

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
This paper proposes K-Sort Eval, an automated evaluation framework for visual generative models that uses vision-language models (VLMs) as judges to approximate human preferences. The method builds upon K-Sort Arena by curating a high-quality dataset from human votes and introducing two key innovations: (1) posterior correction using Bayesian updating that accounts for VLM-human alignment consistency, and (2) dynamic matching strategy that selects informative instances based on uncertainty and diversity criteria. The framework aims to provide scalable, efficient, and reliable evaluation while maintaining alignment with human preferences from K-Sort Arena.

### Strengths
1. The paper addresses a real scalability challenge with Arena-style human evaluations, which are costly and time-consuming. The proposed VLM-as-judge approach is timely and practical.
2. The posterior correction method (Section 3.2) is theoretically grounded in Bayesian inference, with formal derivations provided. The treatment of VLM-human misalignment as observation noise is elegant and well-justified through Lemma 1.
3. The paper demonstrates consistency with K-Sort Arena across multiple models while requiring significantly fewer evaluations (<90 runs vs. traversing entire datasets). This is quite helpful for people who wants to quickly evaluate a new model without access of huge amounts of human annotation resources.

### Weaknesses
1. The K-sort Eval still an existing K-Sort Arena data as supervision, which limits where this can be applied and cannot produce a score independently. Therefore, the quality of the evaluation also depends on the quality of K-sort arena data.
2. The evaluation setting more clear about how they define the evaluation task. The K-sort eval selects a model already in the K-sort arena and then predicts its rank using exisitng information from the K-sort arena, there might be information leaking during the prediction that affects the evaluation, thus making the evaluation unfair. This authors should be more clear about this.

### Questions
1. Can you be more clear about how the evlauation is set up? You select a model from the K-sort arena leaderboard and predicts its rank, using the data in the K-sort arena. But this model is already been compared with multiple models in the arena, will this cause circular information leakink?
2. In Table 3, why there are only 5 models for the text-to-image and text-to-video tasks, while the total rank can be as big as 12. I think you should run K-eval for all of the models in the leaderboard and summarize the prediction result, instead of picking some of them for presentation?

### Soundness
2

### Presentation
2

### Contribution
3
