# When Verifiable Rewards Switch the Language: Cross-Lingual Collapse in Chain-of-Thought

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Reinforcement learning with verifiable reward (RLVR) has been instrumental in eliciting strong reasoning capabilities from large language models (LLMs) via long chains of thought (CoT). During RLVR training, we identify an empirical phenomenon—a systematic drift whereby a multilingual model’s CoT reverts to its dominant pre-training language (e.g., English) even when prompted in another language—which we term Cross-lingual Collapse. Because the long-CoT regime magnifies exposure to linguistic priors, the underlying trade-off between maximizing reasoning depth and preserving target-language fidelity has remained under-characterized. To examine this trade-off, we train LLMs with Group-Relative Policy Optimization (GRPO) on translated versions of math datasets widely used to elicit long-CoT reasoning. Throughout training, we track both task accuracy and the language consistency of reasoning chains. Our experiments yield three findings: (i) under RLVR, CoT in LLMs systematically drifts toward the pre-training dominant language as reasoning performance rises; (ii) English-centric priors, long-CoT GRPO optimization, task difficulty, and high-entropy decoding jointly amplify this drift, and the pattern persists beyond mathematics; and (iii) interventions that favor target-language traces—via a language-consistency reward, decoding-time controls, or more balanced backbones—mitigate collapse but reveal a persistent performance–fidelity trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the widely acknowledged phenomenon of "cross-lingual collapse", demonstrating that multilingual LLMs experience degraded reasoning in non-English languages, primarily correlating this effect with factors like disproportionate token usage. Through controlled experiments and reward-based interventions, it analyzes the causes, effects, and partial mitigations of this language drift during reasoning tasks.

### Strengths
The paper introduces an observation about cross-lingual collapse in multilingual math reasoning. It attempts to contribute a mechanistic investigation by seeking to attribute this known phenomenon to specific internal factors like token usage or representational imbalances, even if this analysis is ultimately superficial.

### Weaknesses
- The central premise, the "cross-lingual collapse" of reasoning capabilities, is a known problem frequently discussed across numerous influential papers concerning multilingual and English-centric models. Especially it has been clearly pinpointed in the paper of GRPO [1]. The authors waste significant space describing a phenomenon without proposing a truly novel mechanism, a new theoretical framework, or a breakthrough solution.

- The experiments are restricted to relatively small models (up to 3B parameters) that are known to exhibit unstable behavior, which limit the generalizability of the findings to larger models commonly used in practice.

- Only three languages are used (Chinese, Korean, and Ukrainian), which constraints the generality of the conclusions across typologically diverse languages and language‐resource levels. It lacks the inclusion of other script languages such as, Arabic, Latin, Bengali, Tamil and others.

- The study uses translated versions of standard math datasets into target languages. Translation quality and cultural or linguistic adaptation issues may cloud the results. It is unclear whether the collapse is caused by translation artifacts, as the solutions to math problems may depend on specific linguistic or cultural contexts.

- There is no thorough human evaluation of whether the reasoning in the target language is meaningful, or whether the drift to dominant language improves or harms reasoning quality in the target language context.

- The findings might be narrowly applicable to math reasoning tasks, and may not generalize to other reasoning domains such as commonsense reasoning, logical reasoning, or real-world problem solving. Please 

- For the language consistency reward, the paper acknowledges the drop in performance and the mitigation of collapse, but gives limited analysis of when the trade-off might be acceptable.

- The manuscript is poorly written, likely suffering from overly complex jargon intended to mask simple observations


[1] Guo, Daya, et al. "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning." arXiv preprint arXiv:2501.12948 (2025).

### Questions
- Does the languages similarity impact the conclusions? 

- How to ensure the quality of the translated datasets? Is there any human evaluation or the results of automatic metrics?

### Soundness
2

### Presentation
2

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
This paper demonstrates that training with verifiable rewards and no incentive for the model to output tokens in a particular language results in the model outputting reasoning tokens in English. It also shows that efforts to mitigate this issue reduce accuracy.

### Strengths
1. Overall the explanation in the paper is clear, with nice illustrative figures.
2. The experimental results are reasonably extensive.
3. The paper is covering an important topic of making sure that multilingual models know how to reason in different languages.

### Weaknesses
1. The insight that LLMs don't reason consistently in the language of the original question is somewhat well known already, including in early reports from deepseek and openai where they observed that the reasoning chains resulting from RLVR are not consistently "good" English.
2. The methods to attempt to force the reasoning chains to be in the original language are not very effective, they do not demonstrate that it is possible to get good performance when reasoning in other languages.

Based on this, I'm not exactly sure how this paper would change practice in any way.

### Questions
What parts of the paper do you think are the ones that are going to be the most impactful to people who are training or using language models?

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
This work studies how verifiable-reward training (GRPO) on translated reasoning datasets causes language drift in multilingual LLMs. The authors show that as reasoning accuracy rises, chains of thought collapse into the model’s pretraining-dominant language (English)—a phenomenon they call Cross-lingual Collapse. They systematically measure, explain, and partially mitigate it via reward shaping, controlling generations, and multilingual training, revealing a persistent accuracy–fidelity trade-off that challenges multilingual reasoning alignment under RLVR.

### Strengths
- The authors identify a consistent phenomenon and analyze mechanisms that both exacerbate and mitigate the phenomenon in an easy-to-follow manner.
- The study is comprehensive across language choice, base model, and evaluation datasets.
- I think the results are interesting and highlights a scenario where the ‘unconditional reward seeking’ behavior of the RL optimization leads to undesirable effects.

### Weaknesses
- There are a few things unclear about the setup and main results in Table 1, namely the baseline that the authors are comparing to and the persistent non-zero EN WR; please see Questions.
- Some of the details in the authors’ implementation of their mitigation strategies require further clarification; please see Questions.

Minor:
- Some areas in the manuscript need `\citep` (eg. line 99, 162)
- Wording in sentence from lines 166-167
- Line 195: Accuracy -> accuracy
- Line 196: -34.0 -> +34.0
- Wording in sentences from lines 201-203
- Line 357: Ukrainia -> Ukrainian

### Questions
- I’m a bit unclear on the ‘non fine-tuned baseline’ that the authors compare to in Table 1; did the authors conduct a cold start SFT phase using translated versions of the datasets or does this start from the base model? How much does the gap get mitigated with more dedicated SFT on a diverse set of reasoning datasets (not just GSM8k, but adding eg. translated OpenMathInstruct2, etc).
- It seems that for most of the baselines, all of the EN WR is still quite high (~15-20%) - what are these tokens comprised of in the reasoning traces? It seems that it is possible to have near-zero EN WR as exemplified by Qwen’s 0.5 WR for Ukrainian. Including some example traces from the baseline models would be useful in this regard.
- How do the authors implement the language consistency reward? Is this done by a threshold for EN WR or an added bonus related to Target WR? How would an alternative strategy where authors discard rollouts containing English CoT continuations (but ensuring they sample enough rollouts to fill their batch) perform?
- If the authors compared “English-drifted” answers to regular completions from the original English versions of the training datasets, how similar or different are they?

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
4

### Summary
This paper investigates a phenomenon termed Cross-lingual Collapse: when LLMs are trained with reinforcement learning using verifiable rewards (e.g., GRPO), their chain-of-thought reasoning in non-English target language prompts systematically drifts back toward the model’s pre-training dominant language (typically English), even when the prompts and training data are fully translated into the target language. This drift coincides with improved task accuracy, demonstrating a trade-off between performance and language fidelity. The work also evaluates several mitigation strategies (e.g., adding a language-consistency reward signal, reducing decoding entropy,  using multilingual training mixtures) and finds that while these approaches help maintain target-language fidelity, they also degrade reasoning accuracy to varying degrees.

### Strengths
- The paper provides a systematic identification and characterization of Cross-lingual Collapse, a previously underexplored failure mode in RLVR-based long chain-of-thought reasoning. The authors clearly illustrate how models trained with verifiable rewards tend to revert to their pre-training dominant language. This constitutes a meaningful contribution that advances our understanding of multilingual reasoning dynamics in large language models. The results suggesting that RLVR-based reasoning amplifies pre-training language priors also raise important concerns for multilingual reliability and interpretability.

- The study offers extensive empirical analysis and experiments with several plausible mitigation strategies, including language-consistency reward shaping, decoding-time constraints, and multilingual training mixtures. These interventions yield actionable insights into the trade-offs between reasoning performance and language fidelity, and provide valuable guidance for practitioners developing multilingual reasoning models. In addition, the paper is also well-written and easy to follow.

### Weaknesses
- The experiments examine only three non-English languages (Korean (KO), Ukrainian (UK), and Chinese (ZH)), which seems insufficient to support their broader claim of multilingual contexts and limit the generalizability of the presented collapse phenomenon. Also, the reason behind adopting these three languages lacks explanation (please correct me if I am wrong!). 

- The training setup relies heavily on GSM8K, a widely used dataset that the backbone models may have already encountered during pre-training, potentially reinforcing existing language priors and introducing a confounding factor. Using a more recent dataset such as MATH for training, and evaluating with higher-quality human-curated MGSM [1] instead of machine-translated GSM8k, could greatly strengthen the validity of the empirical results.

- In the reported main experiments in Table 1, for the Ukrainian results on GSM8K, Qwen maintains a high target-language word ratio with no drift, whereas Llama exhibits a dramatic reduction in target-language word ratio (over an 80% decrease), despite Ukrainian being a low-resource language for both model contexts. This raises concerns that the extent of cross-lingual collapse may be model-dependent rather than a universal behavior. Conducting experiments on additional languages could further strengthen this conclusion (see the weakness noted above).

[1] Shi et al. "*Language models are multilingual chain-of-thought reasoners.*" ICLR 2023.

### Questions
- Some references are not consistently formatted throughout the paper in their use of \citet and \citep. For example, Marchisio et al. (2024) (Line 99), Guerreiro et al. (2024) (Line 155), and Lightman et al. (2024) (Line 162)

### Soundness
2

### Presentation
2

### Contribution
3
