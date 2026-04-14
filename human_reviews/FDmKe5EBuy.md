# Diverse and Effective Red Teaming with Auto-generated Rewards and Multi-step Reinforcement Learning

- Decision: Reject
- Scores: 5, 3, 6, 3

## Abstract
Automated red teaming can discover rare model failures and generate challenging examples that can be used for training or evaluation.  However, a core challenge in automated red teaming is ensuring that the attacks are both diverse and effective.  Prior methods typically succeed in optimizing either for diversity or for effectiveness, but rarely both.  In this paper, we provide methods that enable automated red teaming to generate a large number of diverse and successful attacks.

Our approach decomposes the task into two steps: (1) automated methods for generating diverse attack goals and (2) generating effective attacks for those goals.  While we provide multiple straightforward methods for generating diverse goals, our key contributions are to train an RL attacker that both follows those goals and generates diverse attacks for those goals.  First, we demonstrate that it is easy to use a large language model (LLM) to generate diverse attacker goals with per-goal prompts and rewards, including rule-based rewards (RBRs) to grade whether the attacks are successful for the particular goal.  Second, we demonstrate how training the attacker model with multi-step RL, where the model is rewarded for generating attacks that are different from past attempts further increases diversity while remaining effective.  We use our approach to generate both prompt injection attacks and prompts that elicit unsafe responses.  In both cases, we find that our approach is able to generate highly-effective and considerably more diverse attacks than past general red-teaming approaches.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a reinforcement learning approach to training “attacker” models that generate adversarial prompts triggering harmful responses by “victim” models. As part of this process, the authors proposes a method for generating goals for the attacks. The paper has experiments both in the jailbreaking and in the prompt injection setting.

### Strengths
* After discussion with the authors during the rebuttals, I now find the reward design contribution to be interesting and worthy for the community to see published.
* The authors responded with a table presenting the exact numbers that had only been drawn in plots that were hard to read, so they have improved their presentation.

### Weaknesses
Even after the rebuttal discussion, I continue to have concerns that the method might be finding prompts that exploit the model's general helpfulness tendency, rather than something that would violate the policy. In their last response, the authors frame this as a problem of overfitting but I believe the issue is the core problem with many automated red teaming methods, including this one. Automated red teaming methods do not just need to optimize for diversity but they need to be able to discover if harmful responses can be obtained from the model with enough effort. It is hard for me to be confident that this method could discover such responses based on the results presented.

### Questions
Can you provide a more readable version of Figures 4 and 5? For example, a table with the raw numbers might be needed. Currently, it is hard for me to map between the colors in the legend and the plot colors, so I cannot tell how well each method does.

Can you explain the table in Appendix C.4? Is each row supposed to be goals based on the prompt in the furthest left? Why do prompts in some columns have nothing to do with the “Prompt Details” column?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces a two step automatic red-teaming process to produce effective and diverse attacks. In particular, this is used for automated red-teaming of jailbreaks and injection prompts. The first step consists of generating a diverse set of instructions and criteria both from data and from using a rule-based reward. In the second step an LLM red-teamer is trained using multi-step reinforcement learning on the instructions and criteria collected at step 1. The reward includes attack success, similarity and a length penalty. The red-teaming method is tested one state-of-the-art model and one small model (that is not mentioned in the text)

### Strengths
- It’s good to have a method that produces diverse and effective red-teaming attacks.
- Prompt injection is tricky and it’d be good to have a method to red-team for it.

### Weaknesses
- Some typos throughout the text
- The section about AutoRBR should include more technical details. It’s not clear what is the role of the rule-based reward for the first step of the method
- The baselines should include other red-teaming methods, not just mainly variations of the proposed method
- The method is evaluated on two models that are not mentioned because of concerns about double blind reviews, but it’s not clear why
- Plots in figure 4 and 5 are a bit small and are not clear. Which model is scored in these plots? What do the “crosses” represent?

### Questions
- What are the models you are evaluating?
- Have you considered evaluating the method against more methods?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method for improving the joint diversity and effectiveness of automated red teaming methods for LLMs. The overall method first generates a diverse set of goals, which are then optimized by a multi-step RL method that conditions on previously generated attacks to improve the diversity of the attacks. This improves the attack diversity while maintaining good ASR.

### Strengths
- This is a good line of work. Most jailbreaking and automated red teaming papers haven't taken the RL route first explored by Perez et al (2022), and most haven't given diversity of attacks enough consideration. I wish more work like this existed.
- I appreciate how the authors describe that the diversity was relatively poor at first, which led them to develop their multi-step method where the attacker conditions on previous attacks and tries to make the new attacks different from what came before.
- The proposed methods improve diversity of the attacks.

### Weaknesses
- This paper discusses diversity of attacks, but it doesn't clearly distinguish between two types of diversity that appear in the paper: attack goal diversity, and diversity of the attack itself (i.e., style diversity). These should be clearly distinguished.
- Only the style diversity is evaluated using embedding cosine similarity (presumably self-similarity, as in Perez et al. (2022)). What about the attacker goal diversity? If part of the method involves improving the attacker goal diversity, then surely that should be backed up with an evaluation of some sort. I'm actually not sure what would be a good metric for this, and it seems like an important point to consider for future work on automating exploratory red teaming, so updating the paper to include some sort of evaluation of goal diversity could provide value to the community.
- The presentation is lacking in areas. E.g., Figure 1's caption is essentially missing. This needs to be fixed. Also, the handwritten style of Figure 1 is hard to follow, and many symbols in the figure are not labeled.
- Reading section 5.3, I can't shake the feeling that the discovered lack of diversity isn't a very deep finding. Couldn't one characterize this as just not having designed a good enough goal generation prompt? Does this merit being mentioned in an ICLR-tier paper?
- I'm generally not a fan of making metrics depend on closed-source models. The ASR and diversity metrics used in this submission both rely on the OpenAI API, which reduces reproducibility in the long run.
- The paper involves a lot of experiments, but it's unclear what scientific or technical advances were made. It's OK for papers to be more about interesting results; technical novelty isn't the only source of value. But in this case, I think the outcomes of the experiments aren't that surprising; this may be a paper where the main source of value is in figuring out all the details and showing that this could be done.

### Questions
I'm not sure that the distinction between jailbreaking and indirect prompt injection is good to propagate. They feel like exactly the same problem, with different window dressing. What do you think?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose a pipeline for automated red-teaming to both generate diverse attack goals, and then generate attacks for the goals. By prompting an LLM, a diverse set of instructions and criteria are obtained and used to create a dataset. This dataset is then used to fine-tune an LLM using RL on a few different rewards: attack success (rule-based rewards and breaking moderation filters), style diversity, similarity/consistency, and response length. The core contributions of this paper are the multi-step RL approach and formulation of the reward to encourage style diversity, and proposing to apply this framework for red-teaming prompt injections (in addition to standard jailbreaks). The attacks produced by the fine-tuned model are then evaluated by either their RBRs or OpenAI’s Moderation API, showing that their method improves over baselines (one-shot generation, vanilla RL) while also improving on diversity metrics.

### Strengths
- Results show clear improvement over the naive baselines (one-shot generation, vanilla RL) given the evaluated metrics
- Novel rewards for handling issues with prompt diversity (to the best of my knowledge)

### Weaknesses
- Qualitatively, the results appear questionable - more discussion in the question section
- Figure 5a colours don’t match (success rate vs attack diversity)
- Figures are generally hard to parse and general presentation could be improved
- It is impossible to verify the claims of this paper; no information of the models evaluated was given, and  was given, nor the code to reproduce results. While the authors did promise to release code upon publication, it is difficult to gauge the significance of the results

### Questions
- Why was the method not evaluated on more commonly benchmarked models (e.g. Llama, Gemma, etc)?
- The qualitative examples in C.3 either look very simplistic or somewhat odd; which ones succeeded/failed, and what were the outputs the model produced to these prompts?
- I found the prompt injection task difficult to follow. I understand what they are, and I understand the goals/types of prompt injections that are being included (links/images/specific phrases in responses, or generally the examples in table C.3). However it is unclear to me what you are injecting these goals into, and what you are exactly evaluating.

### Soundness
2

### Presentation
2

### Contribution
2
