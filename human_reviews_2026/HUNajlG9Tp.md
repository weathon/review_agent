# AutoRule: Reasoning Chain-of-Thought Extracted Rule-based Rewards Improve Preference Learning

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 6, 8

## Abstract
Existing rule-based rewards in preference-based reinforcement learning rely on manual engineering, limiting scalability. We present AutoRule, a fully automated method for extracting rules from preference feedback and formulating them into rule-based rewards. AutoRule extraction operates in three stages: it leverages a reasoning model to interpret user preferences, identifies candidate rules from the reasoning chains of these interpretations, and synthesizes them into a unified rule set. Using the finalized rule set, we employ language-model verifiers to judge rule satisfaction, using this metric as an auxiliary reward alongside the learned reward model during policy optimization. Empirically, AutoRule yields gains for both Llama-3-8B and Olmo-2-7B in both in-distribution and out-of-distribution benchmarks. On Llama-3-8B, it achieves a 20.7% relative improvement in length-controlled win rate against GPT4 on AlpacaEval2.0, and a 6.1% relative gain in second-turn performance on a held-out MT-Bench subset, compared to baseline models. Further analysis shows that the extracted rules exhibit strong agreement with dataset preferences and are behaviorally consistent across multiple runs, extraction scales, and aggregated scores. Notably, these rules also contribute to mitigating reward hacking in reward models, likely because they serve as constraints that prevent the policy from exploiting spurious features. Extracted rules are provided; code and model checkpoints will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a pipeline for extracting rule-based rewards for preference alignment of LLMs. The pipeline employs a reasoning LLM (e.g. DeepSeek-R1) to generate reasoning chains for a subset of training samples, extract rules from the reasoning chains, and then merge the extracted rules into a concrete list, which is used by a verifier LLM to produce 0/1 rewards during training. The paper examines the effectiveness of this pipeline on UltraFeedback, MT-Bench, and AlpacaEval2.0 and presents a series of in-depth analysis for the quality of rules, consistency of rules, and pipeline ablation.

### Strengths
1. The idea of extracting reward rules from reasoning models is an interesting idea, since rules are compact and likely to be generalizable.

2. The proposed pipeline is straight-forward and concise.

### Weaknesses
1. The presentation needs improvement. For example, even though the author mention RaR and RLCF, two concurrent word dedicated on automatic rule construction, from line 145-146 it is unclear how the proposed method differs from them. In addition, the inclusion of conciseness as a reference is not mentioned in section 3, but it is experimentally analyzed in line 375. There are also some minor mistakes. The symbol $y$ is defined to be output sequences, so I think in line 166 and 176 the correct notation is $(o,r)\sim\pi_\phi(\cdot|x)$. The 1 and 2 shown in line 174 and line 175 are also confusing.

2. Although this method is claimed to be resolving reward hacking, the only evidence provided are the learning curves presented in Fig 3a & 3b, where the model reward of alternative methods begin to decline after training proceed but the model reward of the proposed method keep improving. Although I agree that such results are positive, their connection with reward hacking needs further illustration.

3. The proposed pipeline has some space to improve. See questions below.

### Questions
1. The results shown in Table 2 suggest that the proposed method is label efficient, if we regard token from the teacher model as labels. But how do you determine the amount subset of data to be used for rule extraction? Is the size of such subset correlated with the overall performance? 

2. Rather than random selection, is there a better way to select samples in this subset?

3. Section 5.3 presents quantitative evaluation for the extracted rules. What do you mean by "from different rule lists" in line 431?

4. I think generalizability is a more intuitive name for the equation in line 424. Could you provide results similar to fig 2a, but in the setting of cross-domain generalization? In other words, evaluate rules extracted from UltraFeedback on other datasets.

### Soundness
3

### Presentation
2

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
This paper introduces an automated framework for extracting explicit natural language rules from human preferential data and using them as additional rewards in RLHF. By doing so, they provide some interpretable or constraint like rewards and hence mitigating reward hacking.

### Strengths
* The paper proposes a fully automated pipeline to extract the rules from the data. The amount of manual engineering is pretty nominal
* The extracted rules are human interpretable and seem to be aligned with known good practices for llm responses.

### Weaknesses
* The results are quite marginal and limited. Llama 3 8B is quite old at this point. And the AE LC Win rate is quite low, compared to Llama 3 8B Instruct. 
* THe results seem to raise a question whether the advantage of rules is mainly effective in out of distribution or extreme scenarios, rather than in distribution (seems contrary as the rules are derived from this distribution). 
* The conciseness constraint added to the verifier is an implicit design choice. It may bias the model toward shorter responses and is not extensively evaluated
* The cited parallel works (RaR and RLCF) show larger improvements on their respective benchmarks, comparing those methods with the presented evaluation would be needed to contextualize the contributions of this method.

### Questions
1. How sensitive is the rule set to the choice of teacher model?  Would using a smaller or different reasoning model significantly change the rules or performance?
2. Can one rule set extracted from UltraFeedback generalize to other domains?
3. After merging, are the final rules entirely complementary, or did you observe any cases of overlap/conflict between rules?

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
3

### Summary
This paper proposes a methodology for automatically generating a set of grading rules based on a preference dataset. To do so, the preferred and dispreferred responses are presented to a reasoning model which is instructed to justify the preference. From the reasoning chain elicited, concrete rules are extracted and merged with other existing rules. These rules are then used during reward model inference. The authors find that the rules generated by AutoRule lead to higher quality reward signals than other rule extraction / reward shaping methodologies.

### Strengths
* Extracting preference rules over a whole preference dataset is both novel and timely given recent research on rubric based optimization.
* AutoRule leads to impressive performance gains in terms of best performance and robustness to overoptimization
* The authors conduct thorough evaluations and ablations which robustly demonstrate their claims.

### Weaknesses
* One potential confounder is that it's unclear how much of the performance gain comes from the utility of the autogenerated rules versus the amount of inference compute spent. Namely, autorule requires doing a forward pass per rubric item. It would be useful to have an inference cost fixed evaluation, potentially by varying the thinking length of a normal llm judge.

### Questions
* Did the authors experiment with how RM performance is related to the number of generated rules? I would be interested in understanding the scaling behavior there.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes an automated framework that extracts rule-based rewards from preference feedback, removing the need for manual rule design. It uses a three-stage pipeline that contains reasoning-based interpretation, rule extraction, and synthesis. The pipeline is followed by LM verification to generate auxiliary rewards. The method boosts alignment performance (+25.6% vs GPT-4 on AlpacaEval 2.0, +6.1% on MT-Bench), reduces reward hacking, and improves interpretability, with all code and models open-sourced.

### Strengths
The proposed method has significant gains on AlpacaEval 2.0 and MT-Bench for Llama-3-8B and Olmo-2-7B.

The proposed method has substantial implications for the community: it provides explicit, human-readable constraints that explain policy behavior.

Open-sourced rules, code, and checkpoints.

### Weaknesses
The authors argue it is the first fully automated rule-extraction system for RLHF/post-training. However, such pipelines are pretty industrial; therefore, the protocol and engineering efforts might not be innovative for the community.

The multi-stage design and instruction-following nature did provide logical transparency of the pipeline, but the paper did not clearly illustrate the merits of such designs. It is conceptually appealing, but I was unable to digest the evidence presented in the main body and the appendix.

### Questions
It is out of scope for this paper, but it would be very helpful if the dependency on the reward function were clarified. 

In many real-world datasets, binary reward models, e.g., the Bradley-Terry (BT) model, are known to be subject to 'intransitivity' because they rely on scalar variables, which assume all preferences are transitive. 
- The literature below studied representative preference datasets in the real world, where the 'transitive' relationship between preference annotations may not always hold. 
- https://arxiv.org/abs/2409.19325 (Duan et al, 2017)

### Soundness
3

### Presentation
2

### Contribution
3
