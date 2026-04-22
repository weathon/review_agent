# Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Language models serve as proxies for human preference judgements in alignment and evaluation, yet they exhibit systematic miscalibration, prioritizing superficial patterns over substantive qualities. This bias manifests as overreliance on features like length, structure, and style, leading to issues like reward hacking and unreliable evaluations. However, the connection between training data artifacts and the miscalibrated preferences exhibited by models remains poorly understood.

In this work, we systematically investigate the relationship between training data biases and preference model miscalibration across five idiosyncratic features of language model generations: length, structure, jargon, sycophancy and vagueness. Using controlled counterfactual pairs, we first quantify the extent to which preference models favor responses with artificially magnified biases (\textit{skew}), finding this preference occurs in $>60$\% of instances, and model preferences show high \textit{miscalibration} ($\approx 40$\%) compared to human preferences. Notably, bias features only show mild negative correlations to human preference labels (mean $r_{\mathrm{human}} = -0.12$) but show moderately strong positive correlations with labels from a strong reward model (mean $r_{\mathrm{model}} = +0.36$), suggesting that models may overrely on spurious cues.

To mitigate these issues, we propose a simple post-training method based on counterfactual data augmentation (CDA) using synthesized contrastive examples. Fine-tuning models with CDA reduces average miscalibration from 39.4\% to 32.5\% and average absolute skew difference from 20.5\% to 10.0\%, while maintaining overall RewardBench performance, indicating that targeted debiasing can strengthen the reliability of preference models within standard alignment pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper 1/ suggests that preference models tend to "over index" on superficial features ; 2/ specifically studies five biases (length, structure, jargon, sycophancy and vagueness) ; 3/ applies the RATE protocol to build counterfactual pairs for these features to study skew (towards the features) and miscalibration (with human) rates ; 4/ shows that indeed four reward models (trained on the Skywork dataset) are skewed and miscalibrated ; 5/ connects these defects to artifacts in the common training dataset (Skywork) and ; 6/ somewhat mitigates them by training the models on new examples created using Counterfactual Data Augmentation (CDA).

### Strengths
- a clear protocol to assess skew (towards the features) and miscalibration (with humans) of LLMs post-trained with preference reward models (RMs) ;
- a detailed analysis of recent LLMs with RMs post-trained on the Skywork dataset, which could be replicated by scientists for their own models ;
- the results from this analysis, for example the fact, as shown in figure 2, that LLM evaluators are themselves miscalibrated ;
- a lightweight mitigation method, CDA, able to reduce skew, especially for jargon and vagueness.

### Weaknesses
- it's minor but having figure 1 above the abstract is troubling, especially as it forces it, the abstract, to end on page 2 ;
- the choice of emphasizing, in figure 1 again, the idiosyncratic features with a visual effect is distracting, at best, the reader ;
- although 4 models are studied, it's really about Gemma 2 and Llama 3, and the Skywork impact on the RM ;
- bias annotation seems weak as sycophancy relies on regex for flattering phrases, vagueness and jargon rely on prompting, and structure mainly looks for list ;
- RATE might work well but it's not clear to me how the double-rewrites guarantee the targeting of a specific features, ie it could do more ;
- human evaluation is smallish with only 100 items per bias with 3 votes, and partly author-run.

### Questions
- prompt B.6 "Structure Re-rewrite" seems to be incorrect, as it is about sycophancy: it's just a copy/paste issue, right?
- were the positions of response randomized for LLM‑as‑judge and human raters, to account for bias?
- beyond the impact, or lack thereof, on RewardBench, limited to a comment (and a figure in the appendix), it would be nice to actually assess the CDA-trained model in situation...
- RATE was new to me: is the re-rewrite step mandatory? It might have been nice to see how sensible the results were to this extra step...

### Soundness
2

### Presentation
2

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
This paper seeks to understand certain properties of aligned models (i.e. their tendency to be verbose, sycophantic, and/or vague) through analysis of the training data used to align them. To do so, they generate counterfactual pairs of responses from a base response by artificially magnifying one of the chosen bias dimensions and then rewriting again to generate a new re-written base response. They find that preference models tend to prefer the responses with magnified biases, in contrast to humans. They then propose a data augmentation method for preference tuning that reduces the models' tendencies to display these biases.

### Strengths
1. I think this paper provides a valuable empirical contribution that sheds light on phenomena that I think many in the community have noticed when looking at LM outputs. I could see this encouraging further work that looks deeper at some of the qualitative insights people circulate about factors such as LLM sycophancy. To my knowledge, few prior works study these issues at as much of a comprehensive level. 
2. The counterfactual approach for isolating the influence of each bias dimension is technically sound and is potentially a really valuable technique to study biases in these outputs. 
3. The paper is written pretty clearly and is easy to follow and understand.

### Weaknesses
1. Some validation of the gpt-4o based re-writing step would assuage concerns about its use. Even some more concrete examples would be helpful (in addition to those in Table 1).
2. The results for the augmentation-based preference model training method seem mixed. While that in and of itself is not a problem, I think it at least warrants a little bit more analysis. In addition, I think the results would be better contextualized with more comparisons to existing de-biasing methods where possible.
3. I would appreciate some clarification on the correlation analysis. Wouldn't another important set of data points be the correlation corresponding to the case where we consider the reward model's predictions on the training data?

### Questions
1. Do you have a sense of why the agreement between annotators for the length bias is the lowest? It seems counter-intuitive to me. 
2. Since some of the bias classification criteria are more subjective, was there any validation done to compare the labeling model's performance with humans?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
< Summary >

This paper investigates systematic miscalibration in preference models, both in terms of reward modeling for alignment tuning or evaluator for pairwise comparison. The paper focus on five well-known idiosyncratic biases: length, structure, jargon, sycophancy, and vagueness. The authors use counterfactual data augmentation via the rewriting method (RATE), to create controlled response pairs that isolate specific bias features. They define two measures for miscalibration, skew (how often models prefer the biased variant) and miscalibration (model preference vs. human majority), and compare model behavior with human preference. They report substantial skew and miscalibration across features, with especially high miscalibration for length, jargon, and vagueness on both open-source models (Gemma2, Llama3.2, etc) and close-source model (GPT-4o, Claude-3.7-Sonnet). Specifically, their experiments reveal significant model-human miscalibration (40% on average), and bias features show mild negative correlation with human preferences (r_human = -0.12) while model preferences shows moderate positive correlation (r_model = +0.36). To address these issues, they propose a post-training counterfactual data augmentation (CDA) method that reduces both average miscalibration and skew difference while maintaining RewardBench performance.

### Strengths
< Strength >

- The paper addresses a critical issue in RLHF of the overreliance of preference models on spurious surface-level features which can lead to reward hacking and unreliable evaluation. Although some of the specific features are already studied with adhoc-treatment, this paper provides more systematic method.
- The use of counterfactual pairs through counterfactual data augmentation (CDA) is simple but provides a practical controlled experimental framework to isolate individual bias features while minimizing confounds. The proposed post training with CDA is also intuitive and effective, and maintains overall model quality, making it practically deployable within existing alignment pipelines.
- The main finding of amplified biases and proposed approach for tracing back the source of bias in dataset provides practical approach of measuring a spurious bias in LLM or dataset. This demonstrates which bias is aligned with human preference (e.g. structure) and which one is diverged (e.g. vagueness)
- The proposed metrics (skew, miscalibration) and training data analysis (Section 4) are valid and can be extended to other bias. The training data analysis provides a novel approach to effectively trace model biases back to imbalances in the dataset.  
- The paper examines multiple models (4 reward models + 3 LLM evaluators) across 5 bias dimensions with moderate size human evaluation (300 annotations per bias type), providing robust and reliable evidence of miscalibration. Also strong documentation of experimental detail such as human evaluation procedures, detailed prompts enhances reproducibility.

### Weaknesses
< Weakness >

- Although the observed miscalibration is significant, the paper's central claim that training data imbalances cause model miscalibration is weakened by Figure 3. For sycophancy, only 5.7% of examples show the bias in off-diagonal, which is too small to explain the observed model behavior. Also for jargon, the 54.4% selection rate is barely above random chance (50%). Only structure shows strong imbalance (65.5%), yet it has the lowest miscalibration among all LLM evaluators.
- The paper uses GPT-4o to generate perturbations with prompts like "make it longer, but change *nothing* else." However, it raise concern whether it can reliably generate valid counterfactuals, as GPT-4o is already biased toward these features.
- Further, the assumption that RATE keep other features unchanged is highly questionable. For example, making responses longer without adding new informative might naturally increase vagueness. The provided prompt such as “Adjust the original response to make it longer, but change *nothing* else.”, would not be sufficient to make LLM take consideration of preserving other bias fixed. No quantitative validation such as correlation of the features on CDA is provided to verify that perturbations truly isolate single features
- The regression-based feature attribution methods, considering the correlation between features are highly relevant but not adequately discussed. For example, [1] done linear regression with multiple feature vectors and figure out which feature dominates the preference. Given that Section 4 essentially performs correlation analysis with annotated features, prior work using multivariate regression should be explicitly compared.
- The intervention with CDA reduces skewness toward biased features but doesn't necessarily improve human alignment. For example, for structure and sycophancy, where base model skew was already similar to human skew, CDA actually increases miscalibration. Although the author mention this as overcorrection, it reveals that simply reducing bias doesn't guarantee correct alignment direction.

< Minor issues >

- The floating Figure1 and Table1 hurts the readability as the abstract is cut off in the first page. I guess the format guidance also recommends to place abstract strictly in the first page
- In line 130, “usetd reward models“ should be “used reward models”
- In line 136, “signals.s” should be “signals.”

[1] Go, Dongyoung, et al. "Compositional Preference Models for Aligning LMs." ICLR. 2024.

### Questions
- I'm curious about how the study controls for potential correlations between bias features. For instance, it seems that lengthening a response without adding substantive information might naturally increase its vagueness as well. Could author check the correlation of features on the augmented dataset to check whether other features are reasonably controlled in the augmentation?
- r_human shows differences between perturbed data and training data (e.g., structure and jargon). As the training data is also built with human label, this raise a questions about the reason of this difference between real-world dataset and controlled human preference, and which one should be considered as ground truth
- beyond the five features studied here, do the authors have thoughts on what other potential biases might exist in preference datasets? Would it be feasible to develop methods that could automatically discover previously unidentified biases, rather than requiring researchers to specify them in advance?

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
3

### Summary
The paper provides a thorough analysis of miscalibration in preference models, connecting RLHF training data biases to model misalignment and proposing counterfactual data augmentation (CDA) to mitigate these biases. While CDA is effective in controlled settings of the experiment, its ability to generalize to complex, real-world environments remains uncertain, as it focuses on a limited set of bias features.

### Strengths
- This paper presents a comprehensive analysis of miscalibration in preference models, systematically examining five common bias features: verbosity, structure, jargon, sycophancy, and vagueness. Experimental results show that existing models exhibit substantial disagreement with human preferences.

- It quantifies how imbalances in RLHF training data amplify these bias features, revealing a clear connection between data artifacts and model miscalibration.

- The paper further introduces a simple post-training approach based on counterfactual data augmentation (CDA), which mitigates biased preferences by adding contrastive “flipped” training pairs.

### Weaknesses
- The effectiveness of the approach in real-world applications remains uncertain. While Counterfactual Data Augmentation (CDA) performs well in controlled settings, it may still struggle to generalize to diverse, dynamic environments where biases are complex and context-dependent. 

- The proposed CDA method targets a limited set of bias features, e.g., verbosity, jargon, which may lack sufficient diversity or challenge. Consequently, the model might only learn to correct biases in relatively simple or unrealistic scenarios. 

- Larger, more powerful models such as Qwen3 and Gemma3 could provide more robust evaluations due to their greater capacity for complex reasoning and adaptation.

### Questions
Please see may detailed comments posted above.

### Soundness
3

### Presentation
3

### Contribution
3
