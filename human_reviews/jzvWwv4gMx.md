# On the Paradox of Generalizable Logical Reasoning in Large Language Models

- Decision: Reject
- Scores: 3, 5, 6, 5, 3, 3

## Abstract
The emergent few-shot reasoning capabilities of Large Language Models (LLMs) have excited the natural language and machine learning community over recent years. Despite the numerous successful applications, it remains an open question whether LLMs have generalizable logical reasoning abilities. In this work, we expose a surprising failure of generalization in logical reasoning tasks (deduction, induction, and abduction)---when semantics are decoupled from the language reasoning process (\ie, replacing semantic words with pure symbols), LLMs tend to perform much worse. We hypothesize that the learned \textit{semantics} of language tokens do the most heavy lifting during the reasoning process but fail to imitate the basic formal reasoning abilities of humans. Furthermore, we also attempt to fine-tune Llama-2 on pure symbolic reasoning tasks to narrow the gap. However, the results indicate that FT-Llama2 can utilize similar template matching to respond to reasoning queries, but it falls short of generalizing to novel logic rules. These surprising observations question whether modern LLMs have mastered the inductive, deductive, and abductive reasoning abilities as in human intelligence, and motivate research on unveiling the magic existing within the black-box LLMs and evaluating and improving language models' reasoning abilities.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
I appreciated the related works section. However, I am not sure that the experimental design is up to the standards of a top venue like ICLR.

### Strengths
I appreciated the references to the literature. The questions you want to answer are interesting.

### Weaknesses
An experiment should try to isolate the hypothesis being tested and removing confounding factors. I am also suspicious your  conclusions; e.g., you claim "In other words, LLMs show significantly worse performance when semantics are decoupled " but without error bars, the Symbols and Semantic of Table 2 look the same to me. (I don't think that you should generalize from a single example (Symbol tree), where the results don't hold for other example (ProofWriter).

Much deductive reasoning is combinatorially difficult, and is difficult even for humans. I'm surprised humans can do the examples B.2 well. (I thought that the psychology literature results are that humans are not good at logical reasoning -- but I am not a psychologist).

You have a strange definition of abduction. It looks like "find a proof" (from the only example given in Figure 1 and on page 4), where the sisterOf and motherOf are given as facts. Abduction in logic means to find a plausible explanation:
e.g. why did someone cough? An explanation is that they have asthma. Another explanation is they have a cold. The system does not know wether they have asthma or a cold. It is difficult to judge the correctness of an answer.

(see also questions)

### Questions
Why only "zero-shot Symbols" for humans? Who are the humans used? Are they trained in logic? (Appendix F1 doesn't provide much details being being diverse college/graduate students). This is not up the standards of a good human-experiment to make any conclusions. Were the humans all the same? Why isn't there a range? The examples you gave on p 36 for humans were for the semantics case (unless I misunderstood appendix I). I wish your appendices gave a few complete examples rather than more abstract examples; what was the actual input and what was the actual output for the computers and the humans? 

For the Symbolic Tree dataset, is the LLM/human told it is a closed-world dataset? What is a false fact? None of the examples seem to rely on negative facts, and none of the examples in appendix B have any. Are there negative examples as well as positive examples?

How hard are the induction tasks? What bias do you assume for the correct answer?  Why should we believe the "gold proofs" are correct?

Can you explain why In Table 1, for ChatGPT Zero-Shot-CoT is better than Zero-Shot for deduction, but in Table 2, it is worse for all depths? Does CoT help?

For "Paradox 1" and "Paradox 2" - why are they paradoxes? Maybe use "hypothesis"?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provided an in-depth evaluation of the reasoning capability of large language models through the language of first-order logic:

- For reasoning tasks, deductive, inductive, and abductive reasoning are conducted.
- For the representation of language, the pure logic language, natural language, and some corner cases of language inputs with garbled symbols.
- For LLMs, the in-context learning of ChatGPT and GPT4 as well as the fine-tuning of Llama-13B is discussed.

By conducting investigations over logical reasoning, the authors identified two major findings and multiple minor findings from their empirical results. It is suggested that the logical reasoning of the large language model is still a challenging task. The good performance of large language models is either mixed results of the semantics of the language, templating matching of the known knowledge, and eventually, the strict logical reasoning ability.

### Strengths
It is praiseworthy that this paper justifies many aspects of logical reasoning.

The highlighted discussions include
- the gap between formal and natural language (or the symbolic or semantics referred to in this paper).
- the impact of in-context knowledge and parameterized knowledge on the commonsense and counter-commonsense settings.

Though there is no innovation from the methodological aspect,  the way of tackling this problem demonstrated by this paper will surely encourage future work.

### Weaknesses
Despite the impressive points that the authors intended to make, some facts might undermine the validity of the claims.

1. The first part of the claims are made by direct prompt ChatGPT/GPT4. However, some gaps between the performances are not significant.
2. Some claims are too general to be valid, please check the question part.

### Questions
1. For the claim
> The length of the context influences reasoning performance, as shorter contexts make it easier to select relevant and useful information while minimizing the impact of unrelated content. 
The effect is also affected by the semantic language and internal knowledge. Are there any results from the symbolic logic language evaluation?

2. For the claim regarding LLM leverages template matching, why do the garbled symbols decrease the performance of deductive reasoning?

3. For the claim regarding learning to reason, why do authors expect that "fine-tuning the LLM on symbolic trees" should lead to good performance in FOLIO and RuDaS?

### Soundness
2 fair

### Presentation
3 good

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
The paper investigates whether large language models (LLMs) like ChatGPT and llama have true logical reasoning abilities that can generalize across facts, rules, domains, and representations.
The authors evaluate LLMs on deductive, inductive, and abductive reasoning tasks. They find that when semantics are removed from the tasks by replacing words with symbols, the performance of LLMs drops significantly. This suggests LLMs rely on semantic associations rather than formal reasoning.
The authors fine-tune an LLM on symbolic reasoning tasks, which improves performance on unseen facts but not novel rules. This indicates the LLM uses template matching rather than truly mastering generalizable reasoning.
Overall, the paper reveals two paradoxes: 1) LLMs rely on semantics rather than formal reasoning, and 2) Fine-tuning enables shallow generalization via template matching but not true generalization to new rules.

### Strengths
Authors study the reasoning capabilities of LLMs and find an interesting angle. Authors report extensive negative results for future body of work to tackle.

### Weaknesses
Authors can be more specific regarding details. Authors can also perform additional interpretability analyses to help the community understand the failure modes.

### Questions
- Can authors clarify which version of GPT4 and ChatGPT they use? There are many timestamped versions with differing context length. 
- Can authors provide more study on how GPT4 fails on symbols version of the task?

I read the author response and I am keeping my score.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper asks whether LLMs truly understand logical reasoning or is their success on some datasets influenced by linguistic semantics and pattern matching. To this end, they experiment with linguistic logical reasoning datasets both in their original form and in pure symbolic form (e.g., relations r1, r2, ...). The find that there is a substantial performance gap between the two settings for both ChatGPT and GPT-4 in a zero/few-shot setting. Further, fine-tuning closes the gap in the symbolic setting, but there still is a large gap when asked to generalize to rules in a different domain.

### Strengths
The basic question that the paper is asking is important to study in order to understand the true capabilities of LLMs, especially when it comes to performing logical reasoning. Their approach of decoupling relation/fact semantics from performing deduction/induction/abduction with logical rules is interesting (though likely not the first, but I cannot pin-point a prior work at this time, so I'll give the authors the benefit of doubt).

The authors conduct a reasonably large study (within the realm of the 2-3 datasets they consider), with many side questions and analyses.

The paper situates itself in the broader NLP / AI research work, citing a LOT of (perhaps too many?) related papers.

### Weaknesses
The overall pitch of the paper is not as convincing as it could be. It's written like the community believes (from prior papers) that LLMs have strong logical reasoning skills, and that the current paper questions this belief and provides evidence against it. However, I don't think it's the case that the community believes logical reasoning is solved by LLMs. E.g., for the datasets considered here, even the baseline performance (in the original, so-called *Semantics* version of the tasks) is not high enough. This makes the motivation of the study relatively weak.

The pitch is also confusing because of the use of the word "paradox". What paradox exactly is being explored here? Reading the title, I was expecting to see something like: LLMs are great at X, which implies they should be great at Y too, but they fail at Y, raising a conundrum. Or some such internal conflict or conflict with commonsense expectations, that would justify the word paradox. I'm not sure what the authors have in mind for a paradox.

Overall, while I thought the study was useful, I didn't find anything subjectively surprising. It is generally accepted that LLMs --- being **language models** --- rely on many linguistic clues and prior knowledge to perform tasks. Taking away these clues is thus expected to drop performance. Similarly, training directly on the so-called *Symbolic* form should help, which they authors also find to be the case. All this is very much aligned with expectation, which makes it difficult to pin point what the new knowledge this paper would bring to the community.

There are number of additional side experiments in the paper. This, in principle, is nice. However, while reading through those section, I found the large number of questions to be somewhat distracting. At the least, the authors should try to thread the narrative better through these side experiments and analyses, and try to provide a view of them that helps support the overall message of the paper.

In summary, while it's somewhat useful to see the experiments on *Symbolic* forms of the considered datasets done, the results don't really feel different from what one might expect to see.

MINOR comments:

* The use of *Semantics* when referring to relation names but not when referring to logic is confusing. Logic, of course, by design has a very clear and unambiguous semantics. I think what you mean is *linguistic semantics* of predicate names. If so, please be sure to clarify this and emphasize the *linguistic* aspect.

* Your related work section (as well as the introduction) has a **lot** of citations, almost too many to be meaningfully valuable. E.g., near the top of page 3, you have 12+ citations around ICL, without any explanation of the connection between these prior works and what's in your paper. As a general rule, it's more valuable for the related work section to point out the few most related works AND articulate a clear connection of the current work to them, as opposed to dumping a huge list of papers just to cover every possible connection.

* The last sentence of page 2 ("Wei et al propose symbolic tuning, which ....") is very long and hard to parse.

### Questions
* What exactly is the paradox (i.e., some form of commonsense contradiction) that you are referring to? Or if *paradox* is not the right word, please replace it with something else.

* Looks like you forgot to discuss Table 2 (results on ProofWriter) in the main paper. What are the main take-aways from this table and how do they support your claims? E.g., it appears the going from the Semantics setting to the Symbolic setting does *not* reduce the performance of the models substantially; in fact, the performance goes up in many cases. How does this align with your claims from Table 1?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides an experimental evaluation of logical reasoning abilities of large language models. The authors first evaluate pre-trained models (GPT-4, GPT-3.5 Turbo) on logical reasoning tasks (deduction, induction, abduction) on both problems expressed with symbols and with words. They observe a large gap in some of the tasks, with even GPT-4 performing generally very poorly on induction with symbols, but much better with words that carry commonsense semantics. The authors then try fine-tuning LLaMA 2 on these tasks, observing that while it is able to match the training rules very well, it still cannot generalize to novel logical rules at inference time.

### Strengths
The main motivating question here is interesting, of whether there are fundamental limitations for logical reasoning in LLMs.

The authors run human comparisons, which is good to sanity check the feasibility of the tasks.

The paper tries to do a very through experimental evaluation, considering both prompting and fine-tuning, and on a range of tasks -- synthetic and existing ones from prior work.

The paper is generally easy to follow.

### Weaknesses
The paper has two main sets of results: with prompting and with fine-tuning. I'll give separate comments on those.

## Results with prompting

The main result here was that models performed significantly worse when symbols were used to described the rules, instead of meaningful words. In a broad sense, this question has been studied before in both papers that observed content effects in LLM reasoning that the authors cite (PrOntoQA and Dasgupta et al). In those papers, they used made-up words, whereas here the authors used short symbols (A, B, etc), but I believe the insight is the same. So, if the authors believe this says something that hasn't been said before, I don't think it came across in the paper.

Finally, the gap in these results is significantly larger in the induction and abduction tasks. The results of GPT-4 in induction (< 10% with symbols) do make me wonder whether this was due to the specific way the task was set up, or whether these are honest failures. It would be interesting if this was the latter case, since induction and abduction haven't really gotten as much attention from prior work. However, the paper has little detail about these tasks besides the general description (I have some specific questions below). It would have helped to have seen many examples of the problems and of GPT-4 responses, to make sure that the task was set up properly and that this is actually due to GPT-4 having a surprisingly bad performance. I tried to find such examples in the Appendix, but couldn't (it's possible I just missed them because there's a lot there! In that case, please point me to it).

## Results with fine-tuning

For fine-tuning, the main result was that models can internalize rules seen during training, but fail to generalize to novel rules. But if I understand, the total number of rules in training and testing was extremely small (5 in training, 3 in testing). Indeed, we would not expect to see generalization from these many examples. There are many other works showing that you do need a certain minimal level of task diversity in the training to get in-context learning in LMs [1,2]. In order to draw this strong conclusion that Transformers might have fundamental limitations to generalizing to unseen logical rules, you would have to train with a much larger number of training rules (e.g. hundreds of thousands) to make this argument convincing. If _even then_ you see a large gap, then it starts to look more like scaling the data is not leading to significant improvements, suggesting that such limitation might be more fundamental. But, at the current scale, the negative result is to be expected, and does not lead to insights into the broader motivating question.

[1] Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression. Allan Raventós, Mansheej Paul, Feng Chen, Surya Ganguli, 2023
[2] Data Distributional Properties Drive Emergent In-Context Learning in Transformers. Chan et al, 2022.

### Questions
- Can you point to specific examples of GPT-4 failures in induction and abduction?
- Generally, your few-shot numbers seem worse than zero-shot. Why would that be the case? That might point to not giving good examples of reasoning.
-- In particular, looking at some of the examples of the appendix, I don't think they contain valid reasoning. For example, this one in Appendix H:
```
Statement: r8(Elena, Nina)
Answer: We can use logical rule L5: ∀A, B, C : r3(A, B) ∧ r3(B, C) ∧ r2(A) → r8(A, C) to deduce whether the statement r8(Elena, Nina) is true or false. [...]
```
This is not the complete problem, but I don't think this is correct. Rule L5 might only be used to prove that r8(A, C) is true (which in this case it does), but if its premises are not satisfied it does not say anything about r8(A, C) being false. Thus, this example is misleading - this reasoning template does not generalize. In fact, all of the other examples below this one proceed like this, and conclude "true". Do you also give examples of "false" cases?
- Why are there missing entries in induction in Table 1?
- What do you think are the novel insights in your experiments with words <--> symbols compared to results in prior works around content effects in LLM reasoning?
- For induction and abduction, what was the complexity of the held-out premises or rules? How did you make sure the answer was unique, since this is logically non-trivial? (in fact impossible, since formally there will be an infinite set of hypothesis in first-order logic that could be used to derive any given conclusion)
- For fine-tuning, would you be able to provide specific fine-tuning examples, besides just the prompts?

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 6

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies whether the logical reasoning capability of large language model generalizes. They evaluate deductive, inductive, and abductive reasoning.

First, they replaced semantic words with pure symbols and found that LLMs perform much worse on the Symbolic Tree dataset which consists of family tree relations. In contrast, there's no drop for ProofWriter which consist of fictional facts and rules.

Second, they finetuned Llama2 on symbolic reasoning tasks from one domain (Symbolic Tree), which made the gap disappear in domain, but found that the finetuned model cannot generalize to other domains (ProofWriter, RuDaS, FOLIO).

They concluded that the reasoning abilities of LLMs were confounded by memorizing the semantics, and even if finetuned, it uses template matching instead of truly learning the rules.

### Strengths
1. The writing is relatively easy to understand.
2. There are a few interesting empirical findings from the carefully designed experiments, e.g.
(1) Finetuning on symbolic reasoning generalizes to unseen facts, but finetuning on semantics doesn't.
(2) Finetuning on symbolic reasoning can help with generalization in semantic reasoning.
3. The paper found previous works either focusing on a single domain or are confounded by semantics, and try to address their shortcomings.

### Weaknesses
1. I think the major weakness is the lack of novelty. Previous works [e.g. Saparov & He] already showed that semantics affects LLMs's reasoning ability, and that if we give new fact and rules contrary to the pretraining prior, the model struggles with learning those new rules. I think the main message of this paper is the same thing and not very new.
2. While I agree that looking at test performance on multiple OOD datasets is important, I hope the authors can explain more clearly whether the datasets contain the same logic rules as the training dataset (LogicTree). If they're different, why do we expect finetuning on LogicTree would generalize at all? Requiring the model to generalize to any novel symbolic rule OOD doesn't seem reasonable to me. Usually for domain generalization one has to specify the boundary of domains. Is this all first-order logic or propositional logic? The delineation seems unclear to me, and I'm not sure inductive reasoning is comparable to deductive reasoning, since we would also not want the model to learn spurious correlations in context. I think the authors should clarify the exact scope of expected generalization in mathematical language. For example, we may want to train on 2-hop but generalize to multi-hop problems, etc.
3. Some minor issues:
(1) All tables and plots: missing error bars
(2) Tables 4, 5, 6 can have more informative row names. The current row names are hard to parse.
(3) Table 6 is lacking context. What is the baseline we are comparing to?

### Questions
1. Table 1: Why does induction column miss some values?
2. LLMs can perform simple arithmetics and it couldn't have seen all possible additions / multiplications etc. during training. Doesn't this show it have some ability to learn rules beyond sematics?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
