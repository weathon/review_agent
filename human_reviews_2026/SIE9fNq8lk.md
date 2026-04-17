# Text2Grad: Reinforcement Learning from Natural Language Feedback

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6, 4

## Abstract
Traditional RLHF optimizes language models with coarse, scalar rewards that mask the fine-grained reasons behind success or failure, leading to slow, opaque learning. Recent work augments RL with textual critiques through prompting or reflection, improving interpretability but leaving model parameters untouched. We introduce **Text2Grad**, a reinforcement-learning paradigm that *turns free-form textual feedback into span-level gradients*. Given human (or programmatic) critiques, Text2Grad aligns each feedback phrase with the relevant token spans, converts these alignments into differentiable reward signals, and performs gradient updates that directly refine the offending portions of the model's policy. This yields precise, feedback-conditioned adjustments instead of global nudges. Text2Grad is realized through three components: (1) a high-quality feedback–annotation pipeline that pairs critiques with token spans; (2) a fine-grained reward model that predicts span-level reward on answers while generating explanatory critiques; and (3) a span-level policy optimizer that back-propagates *natural-language gradients*. Across summarization, code generation, and question answering, Text2Grad consistently surpasses scalar-reward RL and prompt-only baselines, providing both higher task metrics and richer interpretability. Our results suggest that natural-language feedback can serve not only as explanations, but also as actionable training signals for fine-grained alignment. The code for our method is available at *[https://github.com/microsoft/Text2Grad](https://github.com/microsoft/Text2Grad)*.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents Text2Grad, an RL framework that learns from natuaral language feedback to train LLM. Specifically, the method transfers the natural language feedback into span-level rewards, and use it in PPO.

Specifically, the method contains three steps:
1. Use GPT-4o to generate a natural language model feedback, and lists of "good spans" and "bad spans"
2. Train a reward model with GPT-4o generated feedback and lists of spans
3. PPO with token-level rewards. Specifically, the tokens in the "good spans" has reward +1, in "bad spans" has reward -1

The authors experiment this method on Llama3.1-8B, and GPT-4o generated feedback, on summarization, code generation, and QA. The method outperforms scalar reward RL with PPO, DPO, reflection, and PRM-PPO on several metrics and LLM-as-a-judge comparisons.

### Strengths
1. The idea that transfers language feedback into span-level rewards is interesting, and make sense. It also naturally fits into PPO frameworks

2. The authros conduct experiments on multiple domains, showing gains on all the areas.

3. The experiment and theory shows that token-level rewards give faster convergence compared with scalar-level rewards.

### Weaknesses
1. All experiment depdends on GPT-4o generated feedback. There are some other important baselines to compare with. For example, the algorithms that just SFT with responses refined by language feedback (in Scheurer et al., Training language models with language feedback at scale), and even just distilling from GPT-4o. Are these more effective approach to learn from human feedback.

2. What if we use critics from sources different from GPT-4o. For example, from another LLM, or from humans. How will the performance change? Is the algorithm robust to the source of critics?

3. Many results are heavily relied on LLM-as-a-judge. Many judges are GPT-4 based, which may inflate the gains because the model is trained with GPT-4o feedbacks. There lacks human evaluation on policy models' performance. There is only human evaluation on reward model quality.

### Questions
1. Performance when using other critics that is not GPT-4o. What if we use another LLM (for example, Qwen, or o3) as the source of feedback? or use human feedback? The paper will be more convincing if experimented with more than one source of feedback.

2. More human evaluation. Is the trained models really better in humans' eye, or it is just doing some overfitting on styles?

3. More baselines, as said in the weakness section.

4. Talk about failure modes. Is there any reward hacking observed in the experiments? Is there any failure cases? Is that due to reward model failure, or some other reasons?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Text2Grad, a method for transforming natural-language feedback into token-level rewards for finetuning LLMs. While standard RLHF compares pairs of model outputs and assigns a single scalar reward to update the LLM weights, Text2Grad provides fine-grained supervision by identifying which parts of the generated response are incorrect. The approach works in three stages: first, a strong LLM (GPT-4o in the paper) is prompted to produce both textual critiques and span-level reward annotations for model outputs. The output of the first step is used as teacher knowledge to finetune a reward model. Lastly, the reward model's outputs are used to guide policy optimization via NL-Gradient PPO, giving the generator model more informative feedback during training. This enables the model to learn not just whether a response is good, but also how and where to improve it.

### Strengths
- Paper tackles a relevant and timely issue - specifically, with the proposed approach, authors enable more precise finetuning of LLMs and at the same time their method natively presents a basic form of explainability of the training process
- Innovative idea of combining structured token-level rewards with gradient optimization
- Evaluation of the method across multiple tasks

### Weaknesses
- Authors acknowledge that the one limitation is that the optimization severely depends on the quality of the reward model, but in addition to that, the method also requires that the "upstream" LLM (GPT-4o) produces valid training data for teaching the reward model. It's unclear how generally this assumption can be satisfied
- Evaluation is in several parts unclear: For instance, in table 1 it is not clear how human annotation accuracy was obtained, nor what it represents. Also, how were the signals for finetuning models with DPO/PPO obtained? This is particularly important because without CoT, Text2Grad typically performs worse than PPO/DPO (tables 3 and 4)
- No quantification of uncertainty in any table: how many times were the experiments run? Are the improvements over baselines statistically significant?
- The experimental setup seems biased toward GPT-4o's judgments and lacks independent evaluation.
- Method description should be improved: especially section 3.5 contains several symbols that are never introduced (e.g., $V$, $\psi$, $r_k^{KL}$)

Minor things:
- Typo: `optimization. and present` (page 2)
- Table 4: for ARC-C, GPT-4 has a higher value than GPT-3.5, so it seems that the wrong value is highlighted
- GPT-4o and ChatGPT-4o refer to the same model, so I'd recommend using consistent naming throughout the paper
- I'd suggest aligning the text contents with the content of the figure on page 4 (specifically, in the JSON excerpt, "first time author" is good, while in the figure the same part is poor span)

### Questions
- Intuitively, in many cases, only the first token in a "wrong" part of the output should be penalized. For instance, if the output sequence is "The ca-pi-tal of Spa-in is Bar-cel-lo-na", I would expect that the "Bar" would be penalized more than the following tokens, as all the tokens that are generated afterwards are conditioned upon the first mistake. Why do you think giving equal negative signal ($\delta_t$ is always -1) to the entire part is justifiable?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Text2Grad, a framework for converting natural-language critiques into token-level rewards for PPO training.  The paper collects natural language critiques paired with span-level annotations using GPT-4o and trains a generative reward model (RM) to output both textual feedback and token-level labels. Then, it uses the RM to label the correct and incorrect span from original output, scoring -1/+1 for each corresponding span token, respectively. The method aims to improve credit assignment and interpretable policy updates. Experiments in summarization, code generation, and QA show consistent improvements.

### Strengths
1. The motivation is straightforward and easy to follow, around credit assignment and interpretability in RLHF.

2. It introduces the LLM annotation pipeline in detail, which makes it clear for reproduction.

3. Experiment results show consistent improvements across different datasets. The authors also provide a cherry-picked example to show its interpretability.

### Weaknesses
1. “NL-gradient” terminology is inaccurate: the method does not differentiate through language, but converts text to discrete span, and assign different span as token-level rewards (+1/-1 ). The method is not language-conditioned gradient flow.

2. All feedback is GPT-4o-generated, making this RLAIF, not RLHF, with no human validation or study of noisy/contradictory real feedback. 
The training data distilled from gpt-4o lacks real human feedback. The robustness to noisy or adversarial human critiques is unclear.

3. The claim of "high-quality annotation pipeline" lacks human agreement analysis or data quality metrics; quality is assumed rather than demonstrated.

4. The paper claimed that "back-propagates natural-language gradients" with "feedback-conditioned adjustment". However, there is no quantitative measurement of whether feedback actually conditions behavior at the criticized spans. The qualitative span-level evaluation is missing.

5. The method relies on string-span matching tied to tokenizer boundaries, yet tokenization mismatch  (gpt-4o tokenizer vs llama tokenizer) can cause reward attribution noise. robustness to model/tokenizer differences is not evaluated.

6. Evaluation is only on Llama-8B, and the paper does not test across different model scales or model families. Thus, claims about scalability and generalization are unclear.

7. The method appears difficult to scale: span labeling, reward generation, and token-level PPO introduce significant engineering and compute overhead without evidence it transfers beyond the tested setup.

8. The conceptual contribution leans heavily on terminology packaging, and the method does not clearly extend beyond the presented pipeline.

9. Missing RLHF literature related to span-level optimization to improve credit assignment, such as [1,2,3]. The authors should state the position of this work in comparison to the literature and discuss the contribution.

10. The conceptual contribution leans heavily on terminology packaging, and the method does not clearly extend beyond the presented pipeline.

**References**:

[1] MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions. ICLR 2025.

[2] Beyond Sparse Rewards: Enhancing Reinforcement Learning with Language Model Critique in Text Generation. arxiv 2024.

[3] SCAR: Shapley Credit Assignment for More Efficient RLHF. arxiv 2025.

### Questions
1. How do the authors handle incorrect or adversarial feedback? What failure modes arise when GPT-4o produces wrong spans or hallucinated critiques?

2. How robust is the method to tokenization mismatch across models or vocabularies? 

3. For gpt-4o annotation schema, how did the author handle predicted short spans occurs multiple times in the original response? String matching may fail.

4. The author is suggested to provide results across scales and model families to support claims about method generality.

5. How do the authors justify the claim of "high-quality annotation pipeline" without human agreement studies, audits, or error analysis?

6. The authors report the human agreement, and how to measure the recall of span-level annotation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Text2Grad, a reinforcement learning framework that transforms natural language feedback into span-level gradient signals for policy optimization. It first builds a scalable annotation pipeline to collect span-level feedback, then trains a reward model on this annotated data to jointly generate critiques and span maps in one pass.  Building on this, the policy is updated through NL-Gradient Policy Optimization. Experiments on summarization, code generation, and open-domain question answering show consistent improvements over scalar-reward and prompt-based baselines, achieving more precise credit assignment, faster convergence, and better interpretability.

### Strengths
1. The paper is well-written and easy to follow.
2. The method is novel and well-motivated which bridges interpretable textual feedback and gradient-based optimization, addressing a real limitation of scalar RLHF.
3. The method shows strong empirical results across diverse tasks with consistent improvements over strong baselines.

### Weaknesses
1. Binary pseudo-rewards discard fine-grained information in textual critiques. Moreover, treating all tokens within a labeled span equally is simple but ignores token importance.
2. Computation overhead could be critical for method adoption, yet there is no empirical measurement comparing Text2Grad with other baseline methods.

### Questions
1. How sensitive is performance to span length? Is there any correlation between span length and policy learning effectiveness?
2. The paper assigns uniform pseudo-rewards across all tokens within a span. Is it possible to extract more fine-grained token reward signals from the critique?
3. The PRM-PPO results are reported on SLF5K and KodCode but not on UltraFeedback due to annotation cost. Could you clarify (a) how PRM spans or steps were defined in each domain, and (b) what cost differences led to excluding PRM-PPO on UltraFeedback, given that GPT 4o based step level annotation might also be applicable.
4. Table 1 shows your reward model has moderately negative token recall (22% and 43% on UltraFeedback and SLF5K respectively). How sensitive is Text2Grad to reward model quality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a reinforcement learning (RL) method called TEXT2GRAD, which turns textual feedback into span-level gradients. This method contains three core components, including: (1) a feedback annotation pipeline that pairs critiques with token spans; (2) a reward model that predicts span-level rewards on answers while generating explanatory critiques; and (3) a span-level policy optimizer that back-propagates natural-language gradients. Experimental results on three tasks demonstrate that natural language feedback, when converted to gradients, is powerful for policy optimization.

### Strengths
1. The motivation of the proposed method is intuitive and the proposed method can support the motivation.
2. The empirical performance is validated through three different tasks, which shows the generalization ability of the proposed method.
3. This paper is overall well-written and easy to follow.

### Weaknesses
1. This paper misses existing works on reinforcement learning from natural language feedback especially in the training process, such as [1]. Since the proposed method exactly falls into this line of work, the authors should add a discussion to highlight the core novelty of the proposed method.

2. The proposed method jointly generates free-form natural language critiques and structured span-level reward labels. But how natural language critiques can help improve the learning of span-level rewards is not clear. From Figure 1, the relationship between these two modules seems weak, which should be further clarified.

3. The experimental part still has a lot of room for improvement. Now this part only include the main results of three datasets and a case study. In my view, at least a detailed ablation study should be conducted to show the contribution of the core parts in the proposed method.


[1] Learning from Natural Language Feedback. TMLR 2024.

### Questions
I have included my questions in the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2
