# SKILL-MIX: a Flexible and Expandable Family of Evaluations for AI Models

- Decision: Accept (poster)
- Scores: 5, 8, 6

## Abstract
With LLMs shifting their role from statistical modeling of language to serving as general-purpose AI agents, how should LLM evaluations change? Arguably, a key ability of an AI agent is to flexibly combine, as needed, the basic skills it has learned. The capability to combine skills plays an important role in (human) pedagogy and also in a paper on emergence phenomena (Arora & Goyal, 2023).

This work introduces SKILL-MIX, a new evaluation to measure ability to combine skills. Using a list of $N$  skills the evaluator repeatedly picks random subsets of $k$ skills and asks the LLM to produce text combining that subset of skills. Since the number of subsets grows like $N^k$, for even modest $k$ this evaluation will, with high probability, require the LLM to produce text significantly different from any text in the training set. 
The paper develops a methodology for (a) designing and administering such an evaluation, and (b) automatic grading (plus spot-checking by humans) of the results using GPT-4 as well as the open LLaMA-2 70B model. 

Administering a version of SKILL-MIX to popular chatbots gave results that,  while generally in line with prior expectations, contained surprises. Sizeable differences exist among model capabilities that are not captured by their ranking on popular LLM leaderboards ("cramming for the leaderboard"). Furthermore, simple probability calculations indicate that GPT-4's reasonable performance on $k=5$ is suggestive of going beyond "stochastic parrot" behavior (Bender et al., 2021), i.e., it combines skills in ways that it had not seen during training.

We sketch how the methodology can lead to a SKILL-MIX based eco-system of open evaluations for AI capabilities of future models. We maintain a leaderboard of SKILL-MIX at [https://skill-mix.github.io](https://skill-mix.github.io).

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present Skill-Mix, an evaluation for LLMs which utilizes skill combinations to evaluate LLM capability on likely unseen tasks. They demonstrate that some of their results contradict existing results on popular LLM evaluations.

### Strengths
**********************Motivation:********************** One of the main motivations of the paper is good: LLM evaluations can be trained for so it makes sense to do some combinatorial benchmark which requires combining skills in a way that’s likely unseen in any training corpus.

**************Method:************** The idea of skill combination as an LLM evaluation is good, and to the best of my knowledge (i’m not very familiar with LLM benchmarks), novel. It is definitely harder to expect knowledge of the benchmark to be contained in the training corpora of LLMs a priori.

******************Dataset:****************** It’s good idea to release only 10% of the skills and topics for evaluation to prevent overfitting/training on the testing criteria.

****************Results:**************** I’m glad the paper demonstrates that this evaluation is not so correlated with other existing benchmarks, as to prove that it is testing something different from the others with respect to LLM capability.

### Weaknesses
****************************Skill picking:**************************** Section 3.1 (and the appendix) describe how the authors pick the skills by hand based on textbooks on logical reasoning, rhetoric, and theory of mind. But what is the justification for using these types of skills? For example, i’m sure there are many skills not related to any of those topics which still require some sort of intelligent capability to combine meaningfully. The authors should explain this better.

**********************Evaluation:**********************

- One minor issue is that I don’t think the models picked necessarily need to be instruction-finetuned as regular, non-instruction finetuned language models are still able to follow instructions in the prompt. In fact, given that some recent work has demonstrated that instruction-finetuning on smaller models reduces their ability to produce as diverse of a response distribution, it could artificially hurt the performance of the selected open-source models in evaluation due to the selected topics occurring with a Google n-gram probability of ~1e-6 and them likely being fine-tuned on a much smaller dataset that may not include the selected topics/skills.
- What is the variance among the automatically graded scores? It is stated that human variance is higher than the automatic ones, but neither seem to be presented anywhere.
- The analysis of Skill-Mix vs other benchmarks could be more comprehensive — it would be interesting to see ******************correlation scores****************** against other benchmarks to see how skill-mix differs overall wrt individual LLM rankings.

********Writing + Clarity + Motivation:********

- Intro first paragraph: A missing “yet” between the 2nd and third sentence, otherwise these two sentences next to each other are a bit confusing.
- Intro: “(including for some public models such as LLaMA-2)” this could be put next to “size of training corpora” as right now it’s directly after mentioning closed-source LLMs which makes the parenthesis seem confusing.
- Intro: Why is a specific opinion from Hinton used as motivation for desiderata for the authors’ benchmark? While Hinton is an important figure in the field, a good justification should come from (1) an intuitive explanation of the problem, (2) consensus in the field for the problem, or (3) direct citation of evidence of the problem (or some combination of all 3). I don’t think citing a single person’s opinion is good justification in general or intuitive to a reader presented with a new problem.
- Section 1.1: How does SKILL-MIX directly address the desiderata presented in the prior paragraph? For reader clarity, this question should be answered immediately after presenting the desiderata or else we will forget.
- Section 1.1: Arora & Goyal is cited as demonstrating that scaling up language models → improvements at applying k’-tuples of basic skills, therefore motivating the authors’ Skill-Mix benchmark. But this doesn’t mean that applying k’-tuples of basic skills → better general purpose understanding. This is a problem with many LLM benchmarks, but still presents an issue that the authors should discuss in this paragraph.
- Prior work: This section isn’t that well written. The authors should relate Skill-Mix to each of the types of prior work rather than describe them in isolation.
- Section 4: This is a minor issue, but Section 4 can probably be made a subsection of Section 3 as it is part of the introduced SkillMix evaluation.
- Table 1/2: Some sort of bolding on the best performing across each category for each k would be really helpful to parse all of these numbers
- Section 6: Can the authors explain the derivation of the high probability statement for $O(T \log T + N \log N)$? Intuitively, it’s quite surprising given “random prompts” to expose these low-probability skills and their topics.

### Questions
See weaknesses section

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a new evaluation benchmark SKILL-MIX for general-purpose language capabilities, specifically, a particular sort of compositional generalization. It tests the model’s ability to create text on-demand on a given topic and with a given combination of well-known skills. The paper provides an exemplary introduction and motivation for the task of appropriate evaluation of the newly emerging topics.  A key idea is that the skills and the topic are chosen randomly from a big list, which itself can be expanded in many ways to give new evaluations. Such evaluations may end up requiring the (Student) model to imagine and describe situations that does not exactly match anything seen during training. Grading this evaluation is shown to be possible with GPT-4 (and the authors spot-checked the grading). The authors also emphasized the important of human spot-checking, but since human evaluations were  found to have significant variance, they are not reported.

### Strengths
I found this paper one of the best papers I reviewed recently (including the Neurips reviewing).
I found the Introduction (Section 1) and especially "Desiderata for next-generation evaluations" extremely valuable for any LLM development. I am currently working on a task that requires complex evaluation of LLMs and the guidance and observations summarized in the introduction are right to the point of what we are looking for. 

I also find the idea of mixing random skills with evolving N as a way to mitigate some of the issues in current evaluations simply, yet elegant.  The authors also validate its effectiveness with numerous experiments.

### Weaknesses
1) I'd love to see some correlation of the presented results with human spot-checking and evaluation and if such correlation does not exist, discussion on what might be the reason

2) I'd love to get some suggestion on how this framework can be extended beyond language to Vision - Language Models.

3) I'm curious to see how the experiments with fine-tuning while releasing 10% of skills pan out.

### Questions
1)Do you think that instead of have one strong evaluator, you can have a mixture of experts each capable to evaluate a single task exceptionally well?
2) Can you please provide reference for "While both models are creditable graders, they, like all current LLMs, were unreliable at simple arithmetic, which is important for calculating a total score."

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a new general-purpose evaluation framework for LLMs called 'skill mix'. The approach is based on an N-choose-k sampling of a set of pre-defined 'skills', combined with a set of pre-defined topics. The authors present the prompting and evaluation configuration, an LLM-based automated evaluation methodology, empirical results of running this new benchmark against a collection of contemporary LLMs that score well on current leaderboards, and a limited set of human evaluations.

The language of the paper is reasonably clear, and lots of empirical evaluations have been performed, but I find the motivation and justification for the proposed framework lacking. The paper is a little under-prepared and would benefit from further work before publication.

### Strengths
* LLM evaluation (beyond statistical language modelling to general purpose planning systems) is an important and timely topic
 * In the empirical evaluation the authors have clearly made an effort to compare a wide range of contemporary LLMs that currently score well on leaderboards
 * The key concept behind the skill-mix framework is easy to understand

### Weaknesses
* The motivation for this evaluation framework needs further development. Sec 1. highlights accelerating rates of leaderboard saturation, training set contamination, and training corpora secrecy as pressing issues, and suggests 7 desiderata for new LLM evaluation frameworks; (a) relevant to general-purpose intelligence, (b) easy for humans to design and administer, (c) resistant to training-set contamination, (d) capable of revealing novelty in some sense, (e) easy to grade at scale, (f) easy to scale difficulty over time, and (g) comprehensible by the general public. This list seems promising, but the paper doesn't make it clear how skill-mix addresses each of these needs, or how these 7 desiderata are related to the 3 pressing issues with current evaluation methods. In particular, I see no reason to believe that, once released, skill-mix won't also suffer the problem of 'cramming for the leaderboard' (also see next point)
 * In the current form, skill-mix defends against 'cramming' by proposing a held-out private test set of skills and topics. However, any evaluation design that relies on private evaluation data is fundamentally flawed - the field of LLM evaluation design needs to learn from cryptography algorithm design on this count. This is also problematic from an open science perspective - making it harder for peer review to evaluate the efficacy of the evaluation framework for instance.
 * The theoretical justification for skill-mix is also unclear to me.
    * For instance - does it make sense to combine /any/ $k$ of the N skills together, or are there some incompatible skill-tuples?
    * The connections with pedagogy theory and the empirical observations from the Arora & Goyal 2023 paper are a good starting point, but it's not clear in what way combining skills relates to generalized intelligence. A clearer definition of 'skill' may help here.
 * Some 'late breaking' paper results are missing from the submitted version (e.g. footnote 3, Sec. 6) - the paper will be strengthened with the inclusion of these results in a future version.
 * (End of Sec. 5) "Difference between model ranking on popular LLM leaderboards vs. SKILL-MIX provide evidence of "cramming for the leaderboard", further validating that SKILL-MIX is a good evaluation benchmark" - the conclusion here doesn't follow from the premise.

### Questions
# Questions and comments

 * It is great to see some human experiments done to compare against the automated grading scheme (however I note the lack of IRB or ethics approval in the paper related to human experiments). However, I note that variance in human grading responses (e.g. see end of Sec. 4) shouldn't automatically be interpreted as lower quality of responses - the variance can often be a signal, not noise. For instance, this could indicate combinations of skills and/or topics which are perhaps less compatible or sensible to query LLMs with.
 * I like the core idea of N-choose-k as an evaluation approach that scales quickly with k. The paper and results gave me the impression that you only consider linguistic skills, which seems like a big limitation. I note though that in the appendix the list of 10 published skills includes several non-linguistic skills (e.g. folks physics etc.). It would make review and evaluation of skill-mix much clearer if the full list of skills were published, along with (more) examples of select skill-topic-tuples. In the current form, the limited (linguistic) skill-tuple examples make it hard to imagine what the range and scope of questions looks like.
 * A key part of the skill-mix framework needs to be how to do evaluation at scale, including human oversight. To this end, the paper should include a proposed strategy or curriculum for how the human 'spot checking' can be performed. Currently there is not detail about this (very important) aspect of the evaluation procedure.
 * Lots of interesting and important open questions are raised in the paragraph 'difficulty of grading' at the end of Sec. 6. In particular, it seems worth investigating the duality between grading vs. answering questions more generally as part of a theory of LLM pedagogy. 
 * Additional experiments exploring what skill-mix would look like in a multi-modal setting are welcome in future work - this seems like a great area to explore next, as the authors note (end of Sec 7).

# Minor comments and grammatical points

 * Figure 1 caption - typo 'red hearing'.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
