# Interpreting Reward Models in RLHF-Tuned Language Models Using Sparse Autoencoders

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Reinforcement learning from human feedback (RLHF) is a technique for aligning AI systems with human preferences. Arguably, it has been key to the widespread commercialization of large language models (LLMs). However, the effects of RLHF on the internals of these models are not just obfuscated but largely uncharted. We introduce a general method for interpreting the \textit{learned} reward function of an RLHF-tuned LLM, leveraging recent developments in unpacking superposed features to construct more interpretable representations using sparse autoencoders. Our approach trains sets of autoencoders on the activations of both a base model and a model fine-tuned through RLHF. Through identifying unique features present in the hidden space of these autoencoders, we investigate the accuracy of the learned reward model present in this LLM. To assist in this, a toy scenario is constructed whereby the fine-tuned model is tasked with learning a table of token to reward mappings, and then maximizing reward given those mappings. This allows us to quantify the efficacy of the learned reward model in the fine-tuned LLM. To the best of our knowledge, this is the first application of sparse autoencoders to interpreting learned reward models, as well as the first general attempt at understanding learned reward functions in LLMs.  We believe this is a promising technique for ensuring alignment between specified objectives and model behavior. Ultimately our results show that through the method presented in this paper alone, only an abstract approximation reward model integrity can be obtained, but that future work might lead to more rigorous charting of learned reward models in LLMs. This culminates in a score for how well certain features encompass the table of token to reward mappings, as well as a table of features likely to exist in the fine-tuned model in layers where reward modeling is most probable.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a pipeline for finding and interpreting features in models trained with RLHF and the difference in learned features in the RLHF model vs the base model. They train sparse auto-encoders on model activations (as proposed by prior works) to decompose them into learned features, decomposing layers with the largest parameter difference, and then interpret the features learned by the models. They perform experiments on a movide review sentiment completion task. They provide some qualitative analysis of the learned features, and show that the features in the RLHFed model correspond more to the features the reward function was evaluating than the features in the base model, weakly implying that the RLHFed model has internalised the reward function to some extent.

### Strengths
* The topic of understanding the internals of RLHFed models is important, and the sparse auto-encoder approach hasn't been applied to this domain, so there is some novelty in that.
* The diagrams aid in understanding parts of the experimental setup.
* The measure of absolute utility of features is interesting and somewhat novel.

### Weaknesses
# Presentation and clarity

The paper's writing is somewhat difficult to follow at times, partially due to confusing use of terminology. The paper and title talks about interpreting reward models, but their approach interprets policy models, not reward models. If they are aiming to interpret a potentially implicitly learned reward function inside the policy model, that should be made more clear, and the assumptions underlying whether a policy model would internalise the reward function make explicitly.

# Unmotivated methodology

While the methodology is interesting, many design choices are made without any motivation, and it's unclear whether alternatives were considered and how well they performed. For example:
* choosing layers with higher parameter difference, rather than some other difference measure (e.g. in activations)
* training two autoencoders on each layer and taking features that occur only in both.
* The autoencoder architecture in general
* the choice of implementation of the Bills et al (2023) methodology.

While many of these choices occurred in previous work, it would be useful to justify and investigate their use in this work to ensure the methodology proposed is the best version it can be.

# Limited experiments and evaluation

Experiments are only performed in a single setting (movie review completion), with a range of relatively small models. Further, even within this setting the analysis is very limited, comprising just a qualitative analysis of some of the features present (table 3 and section 6.1), and the measure of aggregate absolute utility in table 4. Given this, it's unclear whether the proposed methodology achieves what it is intended to achieve (and it is somewhat unclear what this goal is), and what we have learned about implicit reward functions in RLHFed models from this work.

I suggest the authors perform more extensive experiments, more evaluation of their methodology vs possibly ablations and alternatives, and try to provide more concrete conclusions that the analysis shows about RLHFed models.

# Summary

Overall, I don't feel this work is ready for publication at at top venue such as ICLR. I'd encourage the authors to spend more time improving the clarity, presentation and motivation of the work, as well as performing additional analysis and experimentation to validate their approach and gain clearer conclusions from the analysis. This is an important topic, and I believe an much-improved version of this paper would be impactful and important to the community.

### Questions
* This is mostly addressed previously, but for each of the design decisions made (i.e. the bullet points in section 4.1), describing the motivation in more detail for choosing this approach vs other possible alternatives would be beneficial.
* What's the strongest or most exciting conclusion from the analysis in this work?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper studies the interpretation of reward models in RLHF. The authors take high divergence layers, and train sparse autoencoders to find high confidence features, and analyze the activations on these features.

### Strengths
I like the idea of using autoencoders to help find important features, and analyze activations on these features via asking GPT-4 to give explanation. Although similar methods have been applied to GPT-2, there haven't been methods explaining reward model. It helps us better understand how the reward model

### Weaknesses
The writing and experiment design of the paper can be significantly improved. Besides the formatting issue which shall make the paper be desk rejected, the English and mathematical notations are very confusing and not well defined. For example, in the beginning of page 6 the authors are using tokens and words interchangeably, with both $t_n$ and $w_n$. There are also large amount of typos in the paper. 

The authors mention that the pearson correlation serves as a grading for the accuracy of the prediction. But there is no such experimental results reported. 

The paper can also be significantly improved if the authors could select some of the GPT-4 explanations, and design some specific prompt-response pairs to validate the GPT-4 descriptions. 

Overall, I find the idea interesting. But the current writing and results are definitely not enough for ICLR standard. Thus I recommend rejecting the paper for now and wait for a better version in the future.

### Questions
See weakness.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors aim to provide some insight into what a RLHF-finetuned model learns during the RLHF procedure, with particular attention placed on the *implicit* reward function learned while performing RLHF (against an explicit reward model).

To do so, they employ a multi-step procedure, whereby a model is first tuned via RLHF. Then, layers which have changed significantly due to RLHF (in Euclidean norm) are identified, and autencoders (of two different sizes – one pair per layer) are trained to reproduce activations in these layers in both the RLHF'd model and the original model. Relevant features are extracted by seeing which ones occur in the larger and smaller autoencoders. Finally, via a procedure I have yet to fully understand, feature descriptions are provided by GPT-4 to explain which the features are actually doing.

This paper represents a first step towards interpreting the *implicit* reward model learned by a LLM during RLHF (NOT the reward model used for RLHF), which is a very important direction of study given the dominance of RLHF-based finetuning in modern-day ML.

### Strengths
## significance

Although I will be quite critical in this review in general, I want to start out by stressing how happy I am to see people working on this topic. In general, the field is in dire need of more people who are aware of and working on AI safety related issues. Regarding the area of study of this paper in particular – understanding reward functions implicitly learned during RLHF – this is, in my mind, a super important and relevant area, and continued research in this area could have a meaningful impact in improving the chances that AGI goes well. In this regard, I applaud the authors and want to again express my excitement with and belief in the general direction.

In particular, the opening question about whether reward models (perhaps I would prefer: "objectives" or "goals") learned during RLHF actually correspond to the training objectives very compelling.

## originality

The method suggested in the paper appears to be fairly original. I am not aware of work exploring the implicit reward models learned during RLHF (assuming they exist).

## quality

There was clearly effort put into this research, and I with more work (especially on the paper presentation) I think it can be of meaningful quality.

## clarity

The background and related work sections are quite clear and pleasantly extensive. If such clarity is brought to the rest of the paper – especially sections 4, 5, 6 – then it will be very good.

### Weaknesses
## Margins

The margins appear significantly smaller than ICLR requirements.

## Falling short of answering the key question

The key question being asked in the intro, which I find very compelling, is about whether RLHF actually teaches the model the correct thing, and whether we can interpret how they fall short of this. Yet, the paper is not quite about this topic – it's more about interpretability of learned reward in general, without any comparison to the original reward model or divergences with it. That's ok, but it should be clear that you're trying to answer on a subset of the question, since you provide the question as the main motivation.

## Paper layout

I found the different sections a bit hard to follow – it was unclear what was going to discussed next. For example, the distinctions between sections 4, 5, and 6 were not so clear to me.

## General confusion

I found myself quite confused at several parts in the paper, and I think part of this is due to the presentation. One major point of confusion for me is that there seems to be a conflation (or in any case, and ambiguous distinction) between the (explicit) reward model used during RLHF and the (implicit) reward model which is (maybe) learned as a result of RLHF. In particular, most of the discussion, and all the experiments, are focused on understanding the implicit reward model. Yet, to my knowledge, it is not even guaranteed that there _exists_ such an implicit reward model – that could be one way that the RLHF'd agent learns to get higher rewards, but it could also simply learn a value function and/or policy without any implicit computation of reward itself.

Personally, I would encourage the use of "implicit reward model" always, vs "learned reward model", since the actual reward model learned from human preferences is also learned during RLHF, which causes confusion.

## Reproducibility

It would be good to have access to the code during the review procedure so reviewers can reproduce the results / examine the experiment setup. Since anonymity is a concern, an anonymizing tool can be used, such as: https://anonymous.4open.science/.

## Algorithm presentation

It would be good to make the comments (like "Find top n layers with most divergence) different colours for clearer presentation

in line 9, what is the input that you're calculating the activations over?

in lines 10 and 11, shouldn't the autoencoder size be smaller than the A layer dimension? Otherwise, what compression is happening?

12 and 13: I propose using the convention A\Epsilon_\large.Decoder. Otherwise it looks like you're applying a function called Decoder, whereas here I believe you're trying to extract the decoder from the autoencoder? Please tell me if I'm misunderstanding here.

21: 50 tokens seems quite long. In the paper you gave an example with 3 tokens. Was 50 what you actually did in the experiments? Having access to the codebase would help answer this kind of question.

consider using \texttt for variable names, since that leads to nicer treatment of underscores

## Typos/unclear bits

There are a medium number of typos/unclear bits present in the paper:
- approximation reward model -> approximation of reward model
- second paragraph "Elhage et al" should be \citep
- model activations were sampled -> model activations that were sampled from
- a L_1 coefficient -> an L_1 coefficient
- proposed specifically for expressing reward models (I don't know what "expressing a reward model" means)
- compared to features out brought -> compared to features brought
- in section 4.2, instead of saying "below", the paper should reference a Table by its number
- the table "below" should have a number and a caption. It could go in a wrapfigure if space is a concern.
- the formatting of the list of architecture components in section 4.2 is broken
- gnerations -> generations
- in 5.1, the Sanh et al should be \citep
- it would be nice to find dates for the two von Werra references
- Table 1: Hyperparameters used in the experiment. Which experiment?
- Figure 2: model model -> model
- 5.2 you mention "the previous experiment". Which one is that?
- an autoencoder per high-divergence -> one autoencoder for each high-divergence
- "the following dictionary sizes" (again, reference the Table specifically)
- learned these divergences below (reference the Table specifically). also, please mention that the table is only a subset of the whole table, and ideally include things not only from layer 2.
- e.g. `good', `happy', etc -> e.g., `good', `happy', etc.
- GPT-4s -> GPT-4's
- lend well to this task -> lend themselves well to this task
- utlity -> utility

### Questions
## General confusion

- "Our primary method for interpreting reward learned reward models consists of first isolating parameters relevant to reward modeling" (I'm not sure what this sentence means. Are you talking about explicit or implicit reward model? And why are parameters being discussed here – I thought the analysis was done at the level of features, not parameters?

- "such that any changes in learned features due to RLHF are interpretable" (what does this mean?)

- "were used between all dictionary pairs" (what are all dictionary pairs?)

- when you say "from high likelihood layers involved in reward modeling" do you mean "from layers which are highly likely to be involved in reward modeling"?

- "through the spares coding method from the previous exporiment" (which previous experiment are you referring to?)

- when you train the autoencoders, where are you getting the activations from? a random sample of the dataset? the whole train set?

- how are you getting the feature descriptions from GPT-4? I follow the procedure up to step 4 (on page 2), but I'm not sure where the GPT-4 descriptions come from.

- you mention "selecting case studies of reward modeling failures and successes to show the utility in doing so" but I'm not sure where you show these case studies.

- since the RLHF objective was to give *positive* reviews, I would have expected the interpretability method to find features related to whether a review was good or bad. Yet, it seems most features have to do with movie stuff in general, which is to be expected simply by the nature of the dataset. Based on this experiment, to what extent can you say that the method is useful for finding features learned by RLHF, and not just finding somewhat random features pertaining to the dataset which were learned during fine-tuning? Perhaps section 6.2 aims to explain this, but I do not understand what it is saying.

## Statements in the paper

- You talk about finding the high probability layers involved in reward modelling by looking at the parameter divergence between base model and fine-tuned model. Yet, how do you know which parts of that divergence are used in reward modelling (if any), and which are used in other things, such as learning a policy, or planning, or building a world model, etc?

## Experiment suggestions

- I'm curious why you chose to use exactly two autoencoders. It would be nice to have an experiment with three (or more) and to show how much more information is gained by doing so.

## Closing comment

I want to again emphasize the importance of this topic and my encouragement to the authors to please continue pursuing this and related lines of research. Thank you for working on AI safety related topics. I hope that my and others' comments can help this paper become an impactful part of the AI safety literature.

I would happily consider changing my rating based on the responses of the authors to my questions, reformatting of the paper to match ICLR guidelines, the provision of experiment code, and increased clarity in sections 4, 5, and 6. As it stands, as it is written at present, I would give the paper a 4 (not a 3) but that doesn't appear to be an option I can select.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
