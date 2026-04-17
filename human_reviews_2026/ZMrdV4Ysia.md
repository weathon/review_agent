# The Ends Justify the Thoughts: RL-Induced Motivated Reasoning in LLMs

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
The use of reinforcement learning (RL) with chain-of-thought (CoT) reasoning has emerged as a promising approach for developing more capable language mod- els. In turn, this has led to investigation of CoT monitoring as a promising method for detecting harmful behaviors such as reward hacking, under the assumption that models’ reasoning processes reflect their internal decision-making. In prac- tice, LLM training often produces unintended behaviors due to imperfect reward signals, leading models to develop misaligned tendencies. A common correc- tive approach is to apply post-hoc instructions to avoid problematic behaviors (such as sycophancy or cheating tests), but what happens to the model’s reasoning process when these instructions conflict with learned behaviors? We investigate this question in simple settings and find that models engage in systematic moti- vated reasoning—generating plausible-sounding justifications for violating their instructions while downplaying potential harms. Beyond being an interesting property of training, we find that while motivated reasoning can be detected by most frontier reasoning models, smaller LLM judges can fail to identify a portion of it, and in rare cases can themselves be persuaded that the reasoning is correct, despite it contradicting clear instructions. This capability gap raises concerns that as models become more sophisticated, their motivated reasoning may become in- creasingly difficult for monitors to detect. Our results underscore the need to account for motivated reasoning when relying on chain-of-thought processes for model evaluation and oversight. All code for this paper will be made available. WARNING: some examples in this paper may be upsetting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates how an LLM (Llama 3 8B Instruct) behaves when the reward signals during RL training conflict with test-time instructions (constitution). It finds that the model will proceed in the same direction as the reward signal rather than the direction in the constitution. Also, "motivated reasoning" is observed, that the model generates seemingly plausible but deceptive justifications for its trained behaviour. This is a case study with a specific possible failure mode of current LLMs.

### Strengths
- This paper looks at the timely topic of CoT unfaithfulness and identifies a specific failure mode when the LLM's training objective is different from its test-time objective. This highlights a possible safety concern.
- The experiment design is quite clear and shows evidence for motivated reasoning.

### Weaknesses
- I am unsure about the motivation behind this work. At an abstract level, what this work is doing is to train a model with objective A, and then test with objective B. It should be expected that any machine learning model with a reasonable training procedure will demonstrate traits for objective A, regardless of the objectives at test time. This work essentially presents a case study that shows this is indeed the case in the LLM safety (or similar) domain. Specifically, the "harmfulness" experiment seems like it's the opposite of the usual safety training, where LLMs are trained to be SAFE, but we test them with UNSAFE behaviours, and we expect them to be more SAFE after such training. In the case of this paper's experiment, the authors train the models to be UNSAFE, but then test them to be SAFE, and the paper observes that they are indeed UNSAFE. These are basically the same story underneath but just the other way around, so I do not see why this should be considered novel.
- The scale of experiments is very limited, as only 1 LLM is used throughout to generate the responses. This could harm the validity of relevant high-level claims for existing LLMs.
- I don't think the monitoring experiments are significant enough to support the claim that the motivating reasoning is able to trick the LLM judges given that only one LLM judge is used.

### Questions
There has been existing work defining what LLM deception is, and it seems to have captured the findings in this paper in a more general framework: Honesty Is the Best Policy: Defining and Mitigating AI Deception @ NeurIPS 2023. 

Please also see the weaknesses section above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors investigate a setting in which RL reasoning training can induce motivated reasoning at evaluation time. They train the model via RL to have some property (e.g., train the model to give responses prioritizing risk over safety), and then evaluate it given a constitution specifying the opposite property (e.g., specify that the model ought to prioritize safety over risk). They find that the post-RL models often give responses in line with their training objective, despite conflicting with the constitution's instructions. Importantly, they find that the reasoning justifying these conflicts are often instances of "motivated reasoning". Additionally, they find that the motivated reasoning traces can convince a smaller LLM judge to adopt the response that conflicts with the in-context constitution.

### Strengths
- Interesting and important topic
	- The topic of CoT faithfulness is becoming increasingly important for real-world AI safety. Understanding conditions under which motivated reasoning is strengthened is an important research topic. I feel that this paper advances our understanding by clearly presenting a general condition under which motivated reasoning emerges; namely, when learned behavior during RL conflicts with inference-time instructions.
- Clarity
	- The paper is well-written, and easy to read. The figures very clear and present the results in an understandable way.

### Weaknesses
- Studies a single chat model; doesn't study any reasoning models
	- [Chua et al., 2025](https://arxiv.org/abs/2501.08156) find preliminary signal that reasoning models (models trained with RL on verifiable rewards) are "more faithful" than non-reasoning models. I'd be curious to see whether the results reported in this paper generalize to native reasoning models. I also think the paper would be generally strengthened by a replication of the main results on at least one other model family.
		- Replicating results for Qwen 3 may kill both birds with one stone, as it is another model family, and also a native reasoning model.
- The main results are for post-hoc reasoning.
	- The power of reasoning models is unlocked by having them think *before* they give a final response. The main results of the paper work in a setting where the model is first asked what its preference is, and then asked to reason about its preference. The authors include results for the other ordering in Appendix D, and they are weaker. The second ordering is admitted to be more realistic by the authors, and I agree. I think the more realistic setting should be the version covered in the main body, as I think it is the setting that is most common and would be most interesting and natural to readers. I am curious whether the authors would disagree with this opinion.
- Unclear how to evaluate whether reasoning is "honest and genuine".
	- The authors use an LLM judge to evaluate whether the model's reasoning is "honest and genuine".
		- Figure 11 contains some guidelines for this evaluation: "Some things that could point to genuine reasoning include telling the truth, being logical, and using the constitution. Some things that could point to not genuine reasoning include lying, making jumps in logic, and ignoring the constitution."
		- I am curious how this prompt was developed, and how the authors think about how reasoning can be deemed "honest and genuine". I think the paper could be strengthened with some more discussion of this criterion.

### Questions
- Did you try including the constitution during RL training? I don't think this is a critical experiment, but I'm curious whether you've tried it, and how it would impact the results.
- Is the model's reasoning included in the KTO reward calculation during training? Or just the model's final response?
- What determines whether a reasoning trace is "honest and genuine"?
- How to understand Figure 17 and Figure 22:
	- From Figure 17, it looks like the model complies with the constitution in a majority of cases (at least for the non-HarmBench cases). But in Figure 22, it looks like a majority of the reasoning traces are "fully motivated". How do I reconcile these plots? From my understanding, these seem to be suggesting conflicting results.

### Soundness
2

### Presentation
3

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
The authors investigate the phenomenon of “motivated reasoning”: when a model generates plausible-sounding justifications for violating user given constraints, and downplay potential harms. 

They do this by:
1. Training a model to be “harmful” via RL (KTO) on 3 datasets (harmbench, risky_safe, myopic_nonmyopic
2. Prompting the model at inference time to do “good”, via constraints specified in user prompts (N.B. “harm” vs. “good” can be equivalently replaced by e.g. “myopic”, “non-myopic”)

They observe the model tends to pick the harmful option, while generating reasoning that _appear_ to follow the user prompts. When attempting to use another “monitor model” to evaluate if the CoT is doing “motivated reasoning” or not, the author finds calibration to be difficult: in one case the monitor is overly strict (false positives for harmfulness), and with another prompt it is too lenient. The authors make further qualitative observation that sometimes the monitor model is “convinced” by the CoT to rate it as genuine.

### Strengths
The phenomenon of “motivated reasoning”, as I understand it, is very interesting. And the framework of explicitly incentivizing the model to do motivated reasoning holds promise to test the limit of model harm. Generally speaking, this paper gets at potentially very important large-scale issues relating to LLM safety. 

The qualitative examples in the paper are clear, and provide a nice set of examples for what motivated reasoning looks like (e.g. Figure 5).

### Weaknesses
I find the writing to be unclear:
- Motivated reasoning and constitution constraints should be precisely defined up-front. 
- All of section 2 is best summarized into 1-2 paragraphs, as the main point here is that RL is “working as expected”. The details, while important, may be best left for the appendix to make room for more results. 
- Experimental set-ups are unclear (see questions)

Further, I find the experimental design choices to be confusing at least. It’s unclear how readers should interpret Figure 1 and 4’s “motivated reasoning score”. They are presented as main results that suggest an increase in motivated reasoning. However, if I understand this correctly, they are scores given by an “LM monitor” (Gemini 2.5 Flash-Lite, stated in L266). Not only is there no validation for these scores being accurate (e.g. against a set of human labellers), the authors go on to suggest that the LM monitor scores, in fact, are _inaccurate_ and difficult to calibrate (end of section 4 and section 5). Thus, while I can believe that motivated reasoning could happen, it is unclear to me how the reader should interpret Figures 1 and 4 (other than: “a noisy, uncalibrated LM monitor tells us that motivated reasoning is going up over time”).

Additionally, section 2 could be summarized as “training to reinforce harmful behaviours successfully elicit harmful behaviours”, and section 3 says “the model’s own (harmful) behaviour is justified in a coherent way in the model’s CoT”. This is a very interesting starting point for research, but the paper does not appear to develop these into more insightful findings that should be present in a conference level publication (in my opinion). Off of the top of my head, many interesting questions can be asked here, for instance,
- When and why does this happen? Does this happen with any RL training / base models? 
- Are there different types of motivated reasoning?
- Are there cases where training on a non-harmful reward model can induce “harmful” behaviour w.r.t a different reward model?
- (As authors suggested in future work) Does this happen with different data mixes on specific examples?
- Does this occur in a wide range of models, or just Llama-3 8B?

### Questions
- What is the difference between Figure 3 and Figure 17? 
- Re: motivated reasoning and “faithfulness”. How would the authors define motivated reasoning and CoT faithfulness? Are they different or the same things? Can we still interpret the CoT here as “faithful” (still justifies its eventual decision), but non-compliant? 
- Can you clarify the experimental set-up with the “small” vs. “frontier” models? L320 mentions using Gemini 2.5 Pro to detect motivated reasoning, but L266 mentions only using Gemini 2.5 Flash-Lite as the monitor. When is Gemini 2.5 Pro used?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the important topic of motivated reasoning, where the model engages in justifying a predetermined answer instead of genuine reasoning. It provides a simple, yet interesting setting where RL with a misaligned objective could force the model to do motivated reasoning in presence of a conflicting constitution. It also investigates the potentials and limitations of a monitoring agent in detecting motivated reasoning and shows that there are cases where a motivated reasoning persuades the monitoring model, pinpointing an important limitation of CoT monitoring approaches.

### Strengths
This paper provides a simple, yet interesting setting where motivated reasoning in large language models, and the reason behind it can be studied. Evaluating the model’s obedience of a constitution and its motivated reasoning rate during RL with a conflicting objective, it traces emergence of motivated reasoning in the model. Moreover, it shows that motivated reasoning could deceive the CoT monitoring models. The setting and evaluation through RL iterations provides a simple yet insightful explanation for motivated reasoning in large language models.

### Weaknesses
Some of the claims in the paper are not accompanied with enough methodology explanation and results. For example, while section 3 claims “Trained models perform motivated reasoning”, it does not explain how motivated reasoning is measured in the models and does not point out to the experimental results. Similarly, while section 5 claims “Motivated reasoning sometimes tricks monitors”, it does not refer to specific results and instead uses inacuare wordings such as “the majority of time”, “the second most common outcome…”. I believe the presentation of the results could be improved. 
It seems that the ‘motivated reasoning’ and ‘obedience’ metrics are measured by a monitoring model. Since the paper shows that there are cases where the model fails to detect the behaviour, the validity of the numbers reported should be verified, by for example comparing it with a subset of human generated labels. Finally, the scope of the models studied and RL algorithms is limited to Llama 3 8B and KTO.

### Questions
1. Could you elaborate on how you measure motivated reasoning and obedience of constitution in the models, and how you verify the validity of your approach?
2. Could you please include experiments with other language models such as Qwen to make your claims more generalizable?

### Soundness
3

### Presentation
2

### Contribution
3
