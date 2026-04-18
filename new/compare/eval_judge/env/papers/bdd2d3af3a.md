# Large language models are not zero-shot communicators 

## Abstract

 L

ARGE LANGUAGE MODELS ARE NOT ZERO - SHOT COMMUNICATORS

 Anonymous authors Paper under double-blind review

A BSTRACT

 The recent success of large language models (LLMs) has drawn heavy attention and investment in their use as conversational and embodied systems. Despite widespread use of LLMs as conversational agents, evaluations of performance fail to capture a crucial aspect of communication: interpreting language in con- text. Humans interpret language using beliefs, prior knowledge about the world, and more. For example, we intuitively understand the response “I wore gloves” to the question “Did you leave fingerprints?” as meaning “No”. To investigate whether LLMs have the ability to make this type of inference, known as an im- plicature , we design a simple task and evaluate a set of models. We find that despite only evaluating on utterances that require a binary inference (yes or no), most perform close to random. Models adapted to be “aligned with human intent” via reinforcement learning perform much better, but still leave a significant gap with human performance. This gap is even more pronounced for context-heavy utterances. We present our findings as the starting gun for further research into evaluating how LLMs interpret language in context, in order to drive the develop- ment of more pragmatic and useful models of human discourse.

1 I NTRODUCTION

 User: “Have you seen my phone?” InstructGPT: “Yes, I have seen your phone.” InstructGPT’s response 1 is a perfectly fine answer to the question, but a human might answer dif- ferently. They might respond “it’s in your bag," bypassing the obvious follow-up question (“where is it?”). Giving such a helpful and efficient answer is an example of pragmatic language usage that goes beyond the semantic meaning of the utterance. Meaning is not only determined by a combina- tion of words, but also context, beliefs, and social institutions (Grice, 1975; Huang, 2017). Consider another exchange where Esther asks her friend Juan “Can you come to my party on Friday?” and Juan responds “I have to work.”. We resolve Juan’s response into a decline by using the contextual commonsense knowledge that having to work on a Friday night precludes attendance. Both these exchanges contain an implicature —utterances that convey something other than their literal mean- ing. 2 Implicatures illustrate how context contributes to meaning; distinguishing writing and speaking from communicating (Green, 1996). We cannot fully understand utterances without understanding their implications, nor can a computational model. Indeed, the term “communication” presupposes the speakers’ implications are understood by the addressee. More generally, being able to resolve seemingly completely novel implicatures and—more broadly—engage in pragmatic understanding constitute an essential and ubiquitous aspect of our every day usage of language. Large language models (LLMs) have demonstrated remarkable ability on a variety of downstream tasks such as planning (Huang et al., 2022a), commonsense reasoning (Kojima et al., 2022), infor- mation retrieval (Guu et al., 2020; Kim et al., 2022; Lewis et al., 2020), and code completion (Austin et al., 2021; Biderman & Raff, 2022), to name just a few. When finetuned with human feedback, LLMs obtain higher ratings on desiderata like helpfulness (Ouyang et al., 2022; Bai et al., 2022), and are proposed as conversational agents (Thoppilan et al., 2022). Despite the widespread use and 1 Appendix A contains details on how this answer was obtained from InstructGPT. 2 In Appendix B we present a comprehensive introduction to implicature . 1 deployment of LLMs as conversational agents, there has been limited evaluation of their ability to navigate contextual commonsense knowledge. This raises an important question: to what extent do large language models understand conversa- tional implicature? To answer this question we use a publicly available dataset of conversational implicatures. Using the insight that we can obtain negative examples for binary implicatures, we propose an evaluation protocol (Figure 1). We conduct large scale experiments on a range of state- of-the-art models, including base LLMs like OPT (Zhang et al., 2022) and GPT-3 (Brown et al., 2020) as well as instructable models (versions of instructGPT; Ouyang et al., 2022). We focus on zero-shot evaluation, but also test whether performance improves by presenting in-context examples (few-shot evaluation). Our results suggest that implicature resolution is a very challenging task for LLMs. Most models obtain around 60% accuracy on the test set, whereas humans obtain 86% and random performance is 50%. In-context prompting does not help much for base models. Instructable models consistently outperform base models across all model sizes considered, but even here zero- shot and few-shot evaluation leaves a gap of 14% and 6% respectively with the average human. We do a comprehensive error analysis and uncover that the performance increase for the largest models seems driven by the simplest examples in the dataset that require no context to be resolved. For these examples the conventional meaning of the words entails a proposition, e.g. “some people came to the party” implying “not all people came”. When isolating performance on implicatures that do require commonsense knowledge to be resolved (like the one in Figure 1), the zero-shot and few-shot gap of the best performing model with the average human become 24% and 9% respectively. Based on this result, we hypothesise it is unlikely further scaling alone will lead to significant improvements. Our work demonstrates shortcomings of current SOTA LLMs when resolving binary implicatures. More complex implicatures, sometimes entailing several propositions at once, are ubiquitous in hu- man communication. This highlights the importance of further advancements to enable interpreting language in context. The main contributions of this work are as follows i) we motivate implicature understanding as a crucial aspect of communication that is currently missing from evaluations of LLMs, ii) we design an implicature resolution task and propose a comprehensive evaluation protocol on which we evaluate both humans and LLMs to find that it poses a significant challenge for state-of-the-art LLMs, and (iii) we perform a comprehensive error analysis and identify opportunities for future work.

2 R ELATED W ORK

 LLMs have demonstrated remarkable performance on tasks for which they were not explicitly trained (Brown et al., 2020). Building on the hypothesis that these abilities arise due to implicit multitask learning (Radford et al., 2019), the recent works of Sanh et al. (2022) and Wei et al. (2022) 2 explicitly train LLMs in a supervised multitask fashion, leading to models that are better zero-shot learners with fewer parameters. Besides rapidly saturating language understanding benchmarks (Kiela et al., 2021), these advancements make LLMs beneficial foundations for agents performing a plethora of tasks (Adolphs et al., 2022; Reed et al., 2022). The trend towards using these models as agents brings along with it increased urgency for alignment with human values (Kenton et al., 2021). However, larger models trained with next-word prediction are generally more toxic and un- helpful (Gehman et al., 2020; Bender et al., 2021; Lin et al., 2022). Recent work mitigates this with approaches like prompting and finetuning on human-annotated outputs (Askell et al., 2021; Ouyang et al., 2022; Thoppilan et al., 2022). The produced models are more aligned on desiderata such as in- formativeness when evaluated by dedicated benchmarks and humans. We argue, however, that there is still something missing. What is helpful and informative, as Kasirzadeh & Gabriel (2022) also point out, depends on the context in which a conversation is held. Consequently, any application of language models that requires communicating with humans will rely on pragmatic communication skills—something that is not explicitly captured by the benchmarks used to evaluate LLMs. LLMs are evaluated on a large set of benchmarks, covering tasks like question answering (Berant et al., 2013; Joshi et al., 2017; Kwiatkowski et al., 2019), language completion tasks (Levesque et al., 2012; Paperno et al., 2016; Mostafazadeh et al., 2016; Zellers et al., 2019; Sakaguchi et al., 2021), common-sense reasoning (Mihaylov et al., 2018; Clark et al., 2018; Bisk et al., 2020; Bhak- thavatsalam et al., 2021), reading comprehension (Lai et al., 2017; Choi et al., 2018; Reddy et al., 2019; Dua et al., 2019), natural language inference (Rajpurkar et al., 2018; Nie et al., 2020), and more (Wang et al., 2019; Srivastava et al., 2022). Even though implicature is one of the most impor- tant aspects of language pragmatics (Levinson, 1983), none of these benchmarks explicitly evaluate implicature understanding. Reddy et al. (2019) evaluate implicit coreference among other aspects of conversation. This may indirectly measure performance on implicatures. However, unlike our work, it fails to decouple performance on implicatures from other aspects of pragmatics. Zheng et al. (2021) are the first to fill this gap with a dataset of conversational implicatures, called GRICE. This is important pioneering work highlighting the difficulty of implicature for language models, but their evaluations require task-specific training. In contrast, our evaluation protocol is applicable out- of-the-box and is much more comprehensive, evaluating models up to 175 billion parameters and using in-context prompting. Additionally, Zheng et al. (2021) benchmark synthetic data whereas this work evaluates performance on naturally occurring implicatures (George & Mamidi, 2020). We believe this to be a better representation of the true distribution of implicatures in natural dialogue. Critiques of language modelling benchmarks are widespread (Raji et al., 2021; Bender et al., 2021; Bender & Koller, 2020; Raji et al., 2022). These works question whether the evaluation protocols measure what researchers claim they do. In similar spirit to our work, Valmeekam et al. (2022) point out that despite the fact that many works claim to use LLMs to “plan” (Ahn et al., 2022; Shah et al., 2022; Huang et al., 2022b) they either do not evaluate whether LLMs can do planning or use limited benchmarks that cannot justify the claims being made. Valmeekam et al. (2022) introduce an extensive evaluation suite for planning and find that “GPT-3 is, as of right now, pretty ineffective in reasoning about actions and change.”

3 T HE EVALUATION PROTOCOL

 To answer the research question “To what extent do large language models understand conversa- tional implicature?” we evaluate a wide range of large language models which differ in both training objective and size. In this section we outline the full evaluation protocol. We focus on simple binary implicatures that require inferring “yes” or “no” (like the one in Figure 1). As a proxy for ‘un- derstanding’, we say a model understands an utterance if it assigns higher likelihood to a coherent utterance than a similar but incoherent one, detailed below. Zero-shot evaluation . Consider the example from the introduction packed into a single utterance: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means no. We can transform this example to be incoherent (in the sense that it will become pragmatically inconsistent with expected use) by replacing the word “no” with “yes”: 3 Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means yes. If the model understands the implicature, it should assign higher likelihood to the first of the two sentences above, namely the most coherent one. Importantly, both sentences have exactly the same words except for the binary implicature “yes” or “no”, making the assigned likelihood scores directly comparable. Formally, let the coherent prompt be x and the augmented, incoherent prompt be ˆx . A model outputs a likelihood p parameterized by weights θ . We say a model pragmatically understands an example x when it assigns p θ ( x ) > p θ ( ˆx ) . This is equivalent to evaluating whether the model assigns a higher likelihood to the correct continuation of the two options. Note that this is a more lenient evaluation protocol than sometimes used for language models, where models are evaluated on on their ability to generate the correct continuation, in this case “no”. However, “no” is not the only coherent continuation here, and marginalising over all possible correct continuations is intractable. The more lenient evaluation does capture implicature understanding, because the choice of “no” versus “yes” is fully determined by the resolution of the implicature. We use a dataset of conversational implicatures curated by George & Mamidi (2020). This dataset contains conversational implicatures that, like in Figure 1, are presented in utterance-response- implicature tuples. Of these data, 718 are binary implicatures that we can convert into an incoherent sentence. We randomly sample 600 examples for the test set. We keep the remaining 118 examples as a development set to improve language model implicature understanding after pretraining through in-context prompting or finetuning. Few-shot in-context evaluation . We can add k examples of the task to the original prompt, e.g. with k = 2 : The following examples are coherent sentences: Esther asked “Have you found him yet?” and Juan responded “They’re still looking”, which means no. Esther asked “Are you having fun?” and Juan responded “Is the pope Catholic?”, which means yes. Finish the following sentence: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means no. We evaluate the k -shot capabilities of the models for k ∈ { 1 , 5 , 10 , 15 , 30 } by randomly sampling k examples from the development set for each test example. We opt for a random sampling approach in place of the predominant approach in prior work which leverages the same ordered set of k prompts for each test example. This change in protocol allows us to control for two two sources of randomness. Firstly, development examples have different levels of informativeness. Secondly, the order in which these examples are presented matters (Lu et al., 2022). Ideally, to marginalise over these random factors, we would evaluate each test example with all permutations of k examples from the development set. This requires 118 ! evaluations for each test example, which is intractable. Instead, we estimate performance per test example by randomly sampling from the development set. In this way we control for some of the variance in performance that is expected, but we avoid extra evaluations. Controlling for prompt sensitivity . It has been shown language models are sensitive to the word- ing of the prompt (Efrat & Levy, 2020; Tan et al., 2021; Reynolds & McDonell, 2021; Webson & Pavlick, 2021). To control for this factor of randomness we manually curate six different template prompts and measure performance across these different wordings. One of the templates has al- ready been presented in the examples in this section, namely “Esther asked < utterance > and Juan responded < response >, which means < implicature >”. Another prompt template is: “Question: < ut- terance >, response: < response >, meaning: < implicature >”. The former we call natural prompts and the latter structured prompts. Each group has three templates that only differ slightly in word- ing. This grouping allows us to look at the variance due to slight changes in wording as well as 4 performance difference due to a completely different way of presenting the example. The full list of prompts can be found in Table 4. As Perez et al. (2021) point out, for the few-shot evaluation to be truly few-shot, we formulate these prompt templates before any evaluation is done and never use more than k examples from the development set for a test example.

4 E XPERIMENTS

 We evaluate a set of large language models that vary on key axes of interest, such as the number of parameters and method of training. The model classes we look at are OpenAI (through an API, of the class GPT-3 (Brown et al., 2020) with GPT-3 as the base model of the series and Davinci-001 and Davinci-002 likely of the class InstructGPT (Ouyang et al., 2022)), Cohere (through an API), OPT (Zhang et al., 2022), BLOOM (BigScience, 2022), EleutherAI (Wang & Komatsuzaki, 2021; Black et al., 2022), BlenderBot (Ng et al., 2019), RoBERTa (Liu et al., 2019), BERT (Devlin et al., 2018), and GPT-2 (Radford et al., 2019). Note that the only accessible models that are finetuned with human feedback are Davinci-001 and Davinci-002 (called “instructable” models), all other models are base language models. A detailed categorization of the models in each model class and the attributes we discuss in the results can be found in appendix D 3 . We make use of the pretrained models in the transformers library (Wolf et al., 2020) and EleutherAI’s framework to evaluate them (Gao et al., 2021). We separately treat zero-shot and few-shot in-context evaluation, discussing performance for different model sizes of each model class and the variance over the prompt templates. Additionally, we manually group a third of the test examples into categories and analyse what type of examples are difficult for the models. We contrast the models’ performance with human performance. Details on how the experiment with human subjects was done can be found in the Appendix E. We start by discussing the main evaluation, which is zero-shot performance

4.1 Z ERO - SHOT EVALUATION

 The best performing model classes overall . Table 1 shows the best zero-shot accuracy each model class achieved on the implicature task. The instructable models perform significantly better than any other class. The best zero-shot accuracy is achieved by Davinci-001 (a 175 billion parameter model 4 ) at 72% ± 2 . 8 . This leaves a gap of 13.9% with human average performance. Davinci-002 comes 3 Note that there are several important aspects unknown for models behind APIs, like Cohere and OpenAI. 4 Note that for all OpenAI’s API models except Davinci-002 the size is assumed to align with models from the original GPT-3 paper, since there is reasonable evidence for this to be true https://blog.eleuther . ai/gpt3-model-sizes/ 5 second with a zero-shot accuracy of 70 . 6% ± 2 . 3 . This is surprising, as Davinci-002 is advertised as the best of OpenAI’s models. However, in the few-shot evaluation below Davinci-002 outperforms Davinci-001. The other classes obtain between 53.4% (by BlenderBot-3b) and 61.5% (by OPT- 30b), meaning all non-instructable models obtain performance closer to random than to humans, showing a gap of at least 24% with the average human. We hypothesise that instruction finetuning is especially important for the task of implicature resolution. RoBERTa-125m, EleutherAI-2.7b, and OPT-30b have the property that they outperform all smaller models. In Appendix F.1 we reframe the implicature resolution task such that models can contrast the coherent and incoherent prompt, but this did not improve performance. Moreover, in Appendix F.3 we go into the stochasticity in the results due to the fact that OpenAI’s and Cohere’s models are behind an API. After running the zero-shot experiment ten times through each API we conclude there is some stochasticity, but it is too small to impact the conclusions. Sensitivity to prompt wording . As detailed in Table 4, each example in the test set is wrapped in six different prompt templates. The standard deviation in Table 1 shows the estimated sensitivity to different prompt wording. The standard deviation ranges from 0.3 for BlenderBot, to 4 for Cohere- XL and BLOOM-7b1 when looking at all templates. This variation is often much smaller when separating the performance over structured and natural prompts, most notably for Cohere-XL and BLOOM-7b1. These two models are better at naturally worded prompts (template 2, 5, and 6 in Table 4), whereas OpenAI’s models and OPT-30b are better at structured prompts (template 1, 3, and 4 in Table 4). All in all, the sensitivity to prompt wording does not seem to be a problem for this task; the best and worst evaluations for each model do not change the fact that OpenAI’s Davinci-001 perform best, but significantly worse than humans. The effect of scaling . The left plot in Figure 2 shows the scaling laws we obtained from the model classes for which we know the number of non-embedding parameters 5 . We observe that OpenAI’s instructable models perform significantly better than almost all other models on this task. We addi- tionally observe that although the slope of the lines is increasing, the improvement becomes smaller with model size. For OPT, the largest model we tested (66 billion parameters) obtains a worse per- formance than the second-largest model (30 billion parameters). The same holds for model class GPT-3, where the 1.7 billion parameter model performs better than the 175 billion parameter model. 5 The largest models of classes BlenderBot, OPT, and BLOOM are missing due to computational constraints. 6 Breaking down performance per example type . In Table 2 a taxonomy of the examples is shown, representing types of examples that occur frequently in the dataset. We manually labeled 213 exam- ples of the 600 examples in the test set according to this taxonomy. The remaining 387 examples do not fall as clearly within a category and are grouped together as type other . Generalised implica- tures are what Grice calls conventional implicatures. These implicatures do not require context to be understood and cannot be cancelled with context (see in Appendix B that filing this type under im- plicatures is in fact contentious). This is the simplest type of example in the test set. Particularised implicatures, by contrast, do require context to be resolved. For example, from Table 2, we need the context that it is undesirable to stay up late drinking when one has to get up early. Additionally, this implicature can be cancelled if we add “... but I’d like a nightcap nonetheless”. The type world knowledge requires knowledge of the physical world to be resolved. From the example in Table 2; we need to know that you cannot leave fingerprints when wearing gloves to resolve this implicature. Idiom types contain an idiom or a metaphor that one needs to know or understand to resolve the implicature, and finally Rhetorical question contain a question like “Is the Pope Catholic?”, often requiring factual knowledge to be resolved. In Figure 3 the relative accuracy difference with the mean is shown for classes Cohere and OpenAI. Generalised implicatures are relatively easier for almost all model sizes, and particularised implica- tures are relatively more difficult for all model sizes. In fact, for the largest models this difference becomes more pronounced. Cohere-XL obtains a mean performance of 58.5% whereas for gener- alised examples it is 73.9% and for particularised examples it is 51.5%, which is close to random performance. For Davinci-001 the mean performance is 72.3%, whereas for generalised examples it is 79.3% and for particularised examples it is 59.7%. Humans also do slightly worse for the particu- 7 larised examples (83.2%), but the gap with the mean is smaller. Comparing the absolute accuracy on the particularised examples with human performance uncovers a larger performance gap of 23.5% for Davinci-001 and 31.7% for Cohere-XL. The performance increase for larger model sizes seems driven by the simple examples in the dataset that require less or no context to be resolved. We hy- pothesise that scaling up model size alone will not help with more complex implicature resolution. Moreover, as mentioned in Section 1, even though particularised implicatures do require context to be resolved, they are all implying a simple “yes” or “no”. We conjecture that implicatures entailing several propositions are unlikely to be resolved by current SOTA language models. On prompting . There is a narrative around large language models that if they fail a task, it might be that the prompt was not the right one. The idea is that they can be prompted to simulate almost any- thing, if you set them up correctly. Because implicature resolution is a ubiquitous result of learning language, we hold the view that a model should be able to do this task if a prompt is given in coherent natural language. Nonetheless, in an additional effort to find the “let’s think step-by-step” (Kojima et al., 2022) of zero-shot implicature resolution we try three more prompt templates. We evaluate a base large language model and the two instructable models 3, Davinci-001, and Davinci- . The prompts we use are taken from a recent work propos- ing a dialogue agent trained with human feedback (Glaese et al., 2022), but adapted to the task of implicature res- olution. The full prompts are presented in Table 5. Ta- ble 3 shows the results. The new templates do not im- prove the results for any of these models. The variance over the prompt templates for Davinci-002 is very high, and the best prompt template of these three does achieve a slightly higher accuracy than the others: 74.5%. These re- sults do not change the picture sketched so far. Of course, we will never claim a black swan does not exist, but given the breadth of our experiments we can conclude that using current LLMs to interpret language in context is non-trivial and advancements are needed.

4.2 F EW - SHOT IN - CONTEXT EVALUATION

 The effect of larger k . We prompt the model with in-context examples from the development set to prime it for the task of implicature resolution. Detailed results can be found in Appendix F.4. The highest accuracy we obtain is 80.6% ± 1.22, by Davinci-002 for k = 30 . This shrinks the gap with the average human to 5.6% and with the best human to 9.2%. Note here that humans were tested zero-shot. When only looking at the structured prompts, the accuracy is even slightly higher at 81.7% ± 0.9. The best performance due to in-context prompting of non-instructable models is obtained by OPT-13b with 67.4% ± 2.1. The right plot in Figure 2 shows the relative performance increase due to few-shot prompting for the models of the classes OpenAI, Cohere, and OPT. In- context prompting helps performance for Cohere and OpenAI roughly up to k = 5 , for higher k the performance barely increases anymore. For OPT-66b prompting does not have a significant effect at all. We stopped at k = 30 because the models’ context window could not handle more examples. Regardless, from Figure 2 it seems like larger k would not increase performance significantly. In Appendix F.2 a small experiment is done to estimate the variance over prompt order for OpenAI’s Davinci-002, where the variance is again low enough to conclude this will not impact the results. The effect of in-context examples on sensitivity to prompt wording . Figure 4 again shows the relative performance increase due to in-context prompting, but now broken down per prompt tem- plate. For Davinci-001, most templates benefit similarly from more in-context examples, except for template 1. Perhaps surprisingly, we see that this template already achieves a performance of 76.5% at the zero-shot evaluation and does not improve much with few-shot prompting. For Cohere-XL we see a clear grouping between the structured prompts (dashed lines) and natural prompts (dotted lines). Cohere struggles significantly more with the structured prompts than with the natural prompts in the zero-shot evaluation, and few-shot prompting can mitigate that, leaving the model standard deviation over prompt templates of 1.89 at k = 30 as opposed to 4 at k = 0 . Breaking down performance per example type . We observe again that the context-heavy exam- ples are more difficult for the best performing model Davinci-002 at k = 30 . Recall that humans 8 obtain a performance of 83.2% on the particularised examples. Davinci-002 obtains a performance of 74.4% performance, leaving a gap of 8.8% with the average human.

5 C ONCLUSION AND F UTURE W ORK

 Large language models have made remarkable progress on fluency and coherence in recent years. These advancements have led the field to invest in the usage of LLMs as the foundation for conver- sational agents. We argue however that a central aspect of language understanding is still missing. To understand language means to understand its pragmatics: its usage in context. We design a protocol that evaluates LLMs on binary implicature resolution and establish a significant gap with human understanding. The best performing models leave a gap of 13.9% with the average human in the zero-shot setting, and of 5.6% when k = 30 . All other models obtain performance closer to random than to human performance. Model scaling plots and few-shot evaluation show increasing model size and prompt size is unlikely to close the gap. Moreover, when isolating performance on context-heavy subset of the test set, we see the gap becomes more pronounced. On context-heavy examples the gap with the average human for the best model is 23.5% in the zero-shot setting, and 8.8% when k = 30 . We conjecture that a large part of the zero-shot performance increase for larger models is driven by simple examples in the dataset that require no context to be resolved. We further conjecture that the large difference in performance between OpenAI’s Davinci models and the non-instructable LLMs can be explained by instruction finetuning. However, without access to other instructable models (Thoppilan et al., 2022; Chowdhery et al., 2022) it is impossible to substantiate this hypothesis. We invite researchers who adapt LLMs to be more aligned with human values to additionally evaluate on implicature understanding, to provide further evidence that these models can be used as conversational agents. The finding that instructable models outperform base LLMs can guide future work towards im- proved zero-shot implicature resolution. There is evidence that pragmatic language emerges when reinforcement learning agents optimise joint utility (Vogel et al., 2013). Progress might come from finetuning to cooperate on text-based tasks with reinforcement learning. The type of implicatures we study is a simple type of conversational implicature that can be resolved to a yes or a no. This leaves ample room for the design of benchmarks with complex implicatures entailing more interesting propositions. Humans resolve much more complex propositions intu- itively in conversation. For example, imagine Esther now asking “Can I use your stapler?” and Juan responding “Here’s the key to my office.”. Juan is implicating that (1) Esther can use the stapler, (2) the stapler is located in the office, and (3) the office is currently locked. We believe substantial work needs to be done to move beyond fluent text generation towards communication with autonomous agents and we hope this work will allow researchers to measure progress towards this goal. 9

6 R EPRODUCIBILITY S TATEMENT

 We share all the data, human annotations, code used for the evaluations, and the raw results in the supplementary material. Additionally, in Appendix F.3 we estimate the variance due to stochasticity in the API’s of OpenAI and Cohere. Of course, if either OpenAI or Cohere decides to change the models behind the API, the results might look different. We publish the exact date and time each API was queried for the results in Appendix G. Finally, in Appendix F.2 we estimate the variance over the prompt order of the in-context examples.

7 E THICS S TATEMENT

 In this work, we conduct a study with human subjects (see Appendix E for details). To get matched with participants, we used the platform Prolific. Prolific complies with ethical standards according to UK law (e.g. complying with the GDPR). We compensated participants with a UK living wage at 15 GBP an hour, which is 6 GBP an hour more than Prolific recommends at 9 GBP per hour. Implicature is an aspect of pragmatics, and pragmatic language impairments are universal in Autism Spectrum Disorder (ASD) (American Psychiatric Association, 2013). Difficulties in understanding scalar implicatures are claimed to be present in people with ASD (Volden, 2017), although the na- ture of the relation has proven hard to establish and has recently been debated (Katsos et al., 2011; Schaeken et al., 2018). For the purposes of this work, whether or not implicature understanding relates to ASD is not important. We took the following steps to make sure no sensitive data is col- lected or published. The human annotations we obtain are anonymous, related to a participant only by their Prolific ID for the purposes of compensation. In publishing the human annotations, we will not publish the Prolific ID of participants or anything else related to the participants. Additionally, we did not collect or request any personal or demographic characteristics of the participants apart from that they are all native English speakers.

R EFERENCES

 Leonard Adolphs, Benjamin Börschinger, Christian Buck, Michelle Chen Huebscher, Massimil- 10 3 1 19 , Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless 10 5 org D18 3

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra



Adam Roberts, Paul Barham



Hyung Won Chung



Charles Sutton



Sebastian Gehrmann



Parker Schuh



Kensen Shi



Sasha Tsvyashchenko



Joshua Maynez



Abhishek Rao



Parker Barnes



Yi Tay



Noam Shazeer



Vinodkumar Prabhakaran



Emily Reif



Nan Du



Ben Hutchinson



Reiner Pope



James Bradbury



Jacob Austin



Michael Isard



Guy Gur-Ari



Pengcheng Yin



Toju Duke



Anselm Lev



skaya



Sanjay Ghemawat



Sunipa Dev



Henryk Michalewski



Xavier Garcia



Vedant Misra



Kevin

 11 5 j.tics . .005 20 12 19 1 3 1 3 org 2209 3 . ISSN 1873-7838. doi: 10.1016/j.cognition.2010.12.004. URL https://doi.org/10.1016/j.cognition.2010.12.004 . 10 URL https arxiv org/abs/2103.14659 . 3 3

Su Young Kim, Hyeonjin Park, Kyuyong Shin

 13 3 19 3 1 3 5 3 5 14 zero task generalization International Conference on Learning Repre sentations , 2022. URL https://openreview.net/forum id=9Vrb9D0WI4 . 2 15 ca/books?id=1LkkAQAAMAAJ . 19 Aarohi Rao Awal Md Shoeb Abid Adam Agarwal Alethea Ray Ko An Gholamidavoodi Arfa Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karaka¸s, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartłomiej Bo- janowski, Batuhan Özyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Cameron Diao, Cameron Dour, Cather- ine Stinson, Cedrick Argueta, César Ferri Ramírez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christo- pher D. Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dim- itri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Se- gal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodola, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed, Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata,

Gerard de Melo,

 Germán Kruszewski,

Giambattista

 Parascandolo

,

 Giorgio

Mariani, Gloria Wang, Gonzalo

 Jaimovitch

-López, Gregor Betz, Guy Gur-Ari, Hana Galijase

 -

vic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Shevlin, Hinrich Schütze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac No- ble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac, James B. Simon, James Koppel, James Zheng, James Zou, Jan Koco´n, Jana Thompson, Jared Kaplan, Jarema Radom, Jascha Sohl-

 Dickstein

, Jason Phang, Jason Wei, Jason

 Yosinski, Jekaterina Novikova

, Jelle

 Bosscher

, Jennifer Marsh,

 Jeremy

Kim, Jeroen Taal, Jesse Engel, Je

 - sujoba Alabi

,

 Jiacheng

Xu,

 Jiaming

Song,

 Jillian

Tang, Joan

 Waweru

, John Burden, John Miller

 ,

John

 U. Balis

,

 Jonathan Berant

, Jörg

 Frohberg

, Jos Rozen, Jose Hernandez-

 Orallo

, Joseph Boude

 -

man, Joseph Jones, Joshua B

 . Tenenbaum

, Joshua

 S.

Rule, Joyce Chua, Kamil

 Kanclerz

,

 Karen Livescu

,

 Karl Krauth

,

 Karthik Gopalakrishnan

,

 Katerina Ignatyeva

,

 Katja Markert

,

 Kaustubh D. Dhole

,

 Kevin Gimpel

,

 Kevin Omondi

,

 Kory Mathewson

,

 Kristen Chiafullo

,

 Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson

, Laria Reynolds, Leo Gao, Li Zhang

 , Liam Dugan, Lianhui

Qin

 , Lidia Contreras-Ochando

, Louis-

 Philippe Morency

,

 Luca Moschella, Lucas Lam

,

 Lucy

Noble,

 Ludwig Schmidt

,

 Luheng He

,

 Luis Oliveros

Colón

 , Luke Metz, Lütfi Kerem ¸Senel, Maarten Bosma

,

 Maarten

Sap

 , Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Man- tas Mazeika,

Marco Baturan, Marco Marelli, Marco Maru, Maria Jose Ramírez Quintana, Marie Tolkiehn, Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, Mátyás Schubert, Medina Orduna Baitemirova, Melody Arnaud, Melvin McElrath, Michael A

 .

Yee, Michael Cohen, Michael Gu, Michael Ivanitskiy, Michael Starritt, Michael Strube, Michał

 16 Sw˛edrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, Mukund Varma T, Nanyun Peng, Nathan Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha S. Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon Htut, Pinyu Hwang, Piotr Miłkowski, Piyush Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei, Qing Lyu, Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramón Risco Delgado, Raphaël Millière, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank, Rohan Sikand, Roman Novak, Roman Sitelew, Ronan LeBras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Ruslan Salakhutdinov, Ryan Chi, Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad, Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel S. Schoenholz, Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey, Sebastian Bischoff, Sebastian Gehrmann, Se- bastian Schuster, Sepideh Sadeghi, Shadi Hamdan, Sharon Zhou, Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham Toshniwal, Shyam Upadhyay, Debnath Shyamolima, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva Reddy, Sneha Priscilla Makini, Soo-Hwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas De- haene, Stefan Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven T. Piantadosi, Stuart M. Shieber, Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq Ali, Tatsu Hashimoto, Te-Lin Wu, Théo Desbor- des, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius Nkinyili, Timo Schick, Timofei Kornev, Timothy Telleen-Lawton, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj, Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nya- mai, Vikas Raunak, Vinay Ramasesh, Vinay Uday Prabhu, Vishakh Padmakumar, Vivek Sriku- mar, William Fedus, William Saunders, William Zhang, Wout Vossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yu- fang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, and POMDPs. In Proceedings of the 51st Annual Meeting of the Association for Com- putational Linguistics (Volume 2: Short Papers) , pp. 74–80, Sofia, Bulgaria, August 2013. As- sociation for Computational Linguistics. URL https://aclanthology.org/P13-2014 . 9 17 3 org 10 1007 -3 47489 2_3 10 scholar.google.de/scholar.bib?q=info:1G2GoIkyCZIJ: scholar.google.com/&output=citation&hl=de&ct=citation&cd=0 . 19 5 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.182. URL https://aclanthology.org/2021.findings-acl.182 . 3

A O PENER EXAMPLE WITH I NSTRUCT GPT

 The opener quote by InstructGPT was obtained through the OpenAI playground for Davinci-002. Davinci-001 consistently generates better responses. The following prompt was given: User: “Have you seen my phone?” InstructGPT: With temperatures t = { 0 , 0 . 7 , 1 } . All three of Davinci-002’s responses were similar to: 18 User: “Have you seen my phone?” InstructGPT: “Yes, I have seen your phone.” Davinci-001 consistently generates: User: “Have you seen my phone?” InstructGPT: “No I have not seen your phone.” We tried extending the prompt, which gave similar results for Davinci-002. The following is a request from a user. InstructGPT is a helpful and friendly conversational agent that tries to assist its users. User: “Have you seen my phone?” InstructGPT: “Yes, I have seen your phone.” The same approach makes Davinci-001 a bit more helpful: The following is a request from a user. InstructGPT is a helpful and friendly conversational agent that tries to assist its users. User: “Have you seen my phone?” InstructGPT: “I haven’t seen your phone, what type of phone is it?” This is just a small experiment to illustrate a point, which half of the time goes wrong, even when prompted to be a helpful assistant. Of course, InstructGPT cannot see, so the only “truthful” re- sponse is no.

B B ACKGROUND

 The first influential consideration of implicature is Grice (1975). In his work, Grice continues the trend of moving away from purely logical accounts of language started by Wittgenstein (1921) by hypothesising implicatures arise in conversation when some mutually agreed upon maxims seem to be violated. For example, if we agree on only making relevant contributions to conversation, Juan’s response in the introduction seemingly violates this maxim—after all, he starts talking about work when Esther asks him about a party. However, because Juan agreed to be relevant he must be implying that having to work means he cannot come to the party. Grice contrasts conversational implicatures that arise through context with conventional implicatures. These are implicatures where the conventional meaning of the word determines what is implicated. An example given by Grice is the following sentence: “he is an Englishman; he is therefore brave.”. Grice notes that this sentence does not literally state that an Englishman being brave is a direct consequence of him being English, but it’s implied by the conventional meaning of the word ‘therefore’. Since then, issues with the Gricean cooperative principle have been pointed out by many (Levinson, 1983; Sperber & Wilson, 1986; Davis, 1998; Lepore & Stone, 2014). The most influential alterna- tive theory is relevancy theory by Sperber & Wilson (1986). They do away with the cooperative principle and instead theorise implicatures arise because speakers try to produce utterances that are both as relevant as possible and require the least effort to process. Another point of contention is the incorporation of conventional implicatures on the pragmatics side. Bach (1999) argues that there is no such thing as conventional implicatures, and they are simply instances of something else. Potts (2005) also argues that to explain conventional implicatures we can stay on semantic turf. In- deed, even Grice himself says conventional implicatures derive from the meaning of the words, not from conversational context. However, Potts does not claim conventional implicatures do not exist, but instead argues they arise by a combination of lexical meaning and novel ways of combining words—the latter being the well-known principle of compositionality, an important part of seman- tics, not of pragmatics. Potts provides us with an illuminating demarcation between conventional and conversational implicatures. Conventional implicatures are never negotiable by context, whereas conversational implicatures are context-dependent and can always be cancelled without causing in- coherent discourse. Consider again the sentence “he is an Englishman; he is therefore brave.” and the sentence “Eddie has three bicycles” (implicating that Eddie has exactly three bicycles and not more). The former sentence can not be cancelled by new context without contradiction, whereas 19 for the latter, if we continue saying “In fact, Eddie has 10 bicycles, he is a bicycle junkie”, we have cancelled the implicature. This demarcation clearly puts conventional implicatures on the semantic side, and conversational implicatures on the pragmatic side. Potts goes on by providing a formal theory for conventional implicatures. In later work, Potts (2006) describes how pragmatic pressures interacting with context cause con- versational implicature to arise. He shows how sensitive conversational implicatures are to small changes in the context. Novel information about a speaker’s belief state might completely change what is implied. There are many more models of implicature that aim to explain how humans under- stand language in context. Most notably, Frank & Goodman (2012) formalise the view that speakers produce utterances that are helpful and not longer than necessary with a Bayesian model called the rational speech act (RSA). Many variants on the RSA framework have since been proposed. For example, Goodman & Frank (2016) extend it to handle nonliteral uses of language, like irony, and metaphor.

C D ETAILED PROMPT TEMPLATES

 Table 4 contains the full prompt templates we used for the main evaluation and Table 5 contains the extra prompt templates. 1 Does the following response to the question imply yes or no? question: < utterance > response: < response > implicature: < implicature > 2 Finish the following text: Esther asked "< utterance >" and Juan responded "< response >", which means < implicature > 3 Is the implied meaning of the following response yes or no: question: < utterance > response: < response > meaning: < implicature > 4 What is the intent of the following response, yes or no? question: < utterance > response: < response > intent: < implicature > 5 Finish the following text: Karen asked "< utterance >" and William responded "< response >", which means < implicature > 6 Finish the following text: Bob asked "< utterance >" and Alice responded "< response >", which means < implicature >

D M ODEL CATEGORIZATION

 Table 6 contains details on the model size, dataset size, and training objective of each model. 20 7 The following text shows an interaction between two humans called Esther and Juan. In the interaction, Esther will ask Juan a question, and Juan will give an answer that contains an implicature. An implicature is an utterance that means something other than the literal meaning of the words. The implicature of Juan’s response is yes or no. You, the AI assistant, are asked to finish the text with yes or no. The task begins: Esther asked "< utterance >" and Juan responded "< response >", which means < implicature > 8 The following text shows an interaction between two humans called Esther and Juan. In the interaction, Esther will ask Juan a question, and Juan will give an answer that has a meaning besides the literal meaning of the words. That meaning is either yes or no. You, the AI assistant, are asked to finish the text with the correct meaning, either yes or no. The task begins: Esther asked "< utterance >" and Juan responded "< response >", which means < implicature > 9 The following text shows an interaction between two humans called Esther and Juan. In the interaction, Esther will ask Juan a question, and Juan will give an answer that has a meaning besides the literal meaning of the words. That meaning is either yes or no. You, a highly intelligent and knowledgeable AI assistant, are asked to finish the text with the correct meaning, either yes or no. The task begins: Esther asked "< utterance >" and Juan responded "< response >", which means < implicature >

E H UMAN EVALUATION

 The participants for the human evaluation in this paper were recruited using Prolific ( www. prolific.co ). The setup of the experiment is as follows. We divide the test set of 600 ex- amples into four non-overlapping subsets of 150 examples. Each set of 150 examples was given to five unique annotators. This means each example in the test set is labeled five times by different people, and we have in total twenty annotators for the whole test set (five different ones for each of the four subsets). The only constraint for the annotators is that they are native English speakers. In Figure 5 the screen shown to potential participants on Prolific is shown. Participants are paid 15 pounds an hour, which was the living wage at the time of the experiment and more than the 12 dollars an hour Prolific recommends. 21 The 150 test examples are wrapped in prompt template 2 (see Table 4) and presented in a Google form. The participants are asked to choose the correct continuation, yes or no (see Figure 6a). As recommended by Prolific, we subject the participants to an attention test (see Figure 6b). At three random places in the form, we add a question that does not contain an implicature and obviously maps to “yes”. In this way, if the participants fails at least two of these questions, we can conclude they were not paying attention and remove their answers from the result. In practice, this happened once and we decided to pay the participant regardless, but discard their results, which were close to random. Table 7 shows the performance of each annotator on the subset they annotated. The average human performance across subsets and annotators is 86.2% ± 2.3, the best performance is 89.8% ± 2.2, and the worst performance is 83.5% ± 1.5. The column “IAA” shows the average Cohen’s Kappa coeffi- 22 cient which is the pairwise inter-annotator agreement for each annotator per subset. All agreements are substantial according to the interpretation guidelines for Cohen’s Kappa (between 0.61–0.80).

F A DDITIONAL RESULTS F.1 C ONTRASTIVE EXPERIMENT

 In this section we reframe the implicature resolution task to a contrastive one, allowing the model to contrast the coherent to the incoherent sentence in a single prompt. Contrastive task . In the ranking task the model is required to assign higher likelihood to the coher- ent utterance than the incoherent one ( p θ ( x ) > p θ ( ˆx ) ). In assigning a likelihood to x , the model has no knowledge of ˆx , and vice-versa. We hypothesize that the task might become easier if we reformulate it as a contrastive task. Consider the following prompt p . Which of the following sentences is coherent: A: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means no. B: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means yes. Answer: We can now evaluate the models’ ability to understand which is the coherent sentence by evaluating whether it assigns p θ ( A | p ) > p θ ( B | p ) . Note that this can again be framed in a ranking task of assigning a higher likelihood to the coherent prompt. If we finish the above prompt p by adding “A” to make a coherent prompt x and “B” to make an incoherent prompt ˆx we can again formulate the task by p θ ( x ) > p θ ( ˆx ) . The difference is that within both the coherent and the incoherent prompt, the model can contrast the coherent and incoherent utterance to each other. We randomise the assignment of A and B to the utterances. We do a small experiment with the contrastive task with the best performing model overall, OpenAI’s Davinci-002, for k = { 0 , 1 , 5 } . We use two prompt templates and for each template try three different multiple choice answers: A and B like above, one and two, or the full text of the answer. For the last option the coherent prompt x would look as follows: Which of the following sentences is coherent: A: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means no. B: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means yes. 23 Answer: Esther asked “Can you come to my party on Friday?” and Juan re- sponded “I have to work”, which means no. In Table 8, perhaps surprisingly, we can see that the contrastive task is much more difficult than the original ranking task. For k = 0 , the result is random except for the prompt where the multiple choice options are A and B. For k = { 1 , 5 } the full text ranking does best, but is still significantly worse than the original ranking setup. Because of these disappointing results, we did not evaluate the other models contrastively. Future work must establish whether the contrastive setup is worse across all model classes and sizes.

F.2 V ARIANCE OVER PROMPT ORDERING

 As mentioned in Section 3, models are sensitive to the ordering of the k examples in the prompt. Instead of marginalising over this random factor by evaluating all possible prompt orderings, we randomly sampled an ordered set of examples from the development set for each test example. Throughout experiments, we kept this randomly sampled order the same, meaning if you re-run the 5-shot evaluation you get exactly the same orderings. The reason for this is that we want evaluate each model equally. In this section we ask how the performance chances for the best performing model if we select another random order. We do this for the 5-shot evaluation, because the results show that adding more in-context examples barely helps performance. Table 9 shows the results of this experiment. Some prompt templates seem to be more sensitive to prompt example ordering than others, but for none of them the variance is high enough to change any conclusions.

F.3 V ARIANCE OVER API RUNS

 In this section we comment on the reproducibility of research done using APIs. Two of the model classes we evaluate have their models behind an API, meaning we do not have control over what happens to the prompt before the model processes it. We run the main evaluation, which is zero- shot, ten more times for the largest models of OpenAI and Cohere, Davinci-002 and Cohere-XL. The results from this experiment are shown in Table 10 and 11. From this we can conclude that there is some stochasticity in the API that we have no control over, a bit more for OpenAI than for Cohere, but again we can be relatively confident that the conclusion will not be different because of it. The results from this work are therefore reproducible with access to the same models behind the API now. Unfortunately, when OpenAI or Cohere changes the models behind the API, these results are not exactly reproducible anymore. For completeness, we add the timestamp that each result was obtained below (Appendix G). 24 Eachevaluation has exactly the same text, so the variance in performance is due to API stochasticity. . Each evaluation has exactly the same text, so the variance in performance is due to API stochasticity.

D ETAILED RESULTS PER MODEL

 This section contains the results used for the zero-shot and few-shot evaluation in the main text in Section 4, broken down per prompt template. See Table 12 until Table 51. ). Template k = 0 k = 1 k = 5 k = 10 k = 15 k = 30 1 55.3 57.2 58.3 57.5 58.2 60.5 2 46.7 56.8 56.3 59.5 59.2 61.7 3 54.0 54.5 53.3 54.0 56.5 56.7 4 53.5 52.8 54.7 56.7 58.8 59.7 5 49.8 57.3 55.3 58.5 58.8 61.8 6 49.5 57.2 56.3 60.2 61.5 61.2 Mean 51.5 56.0 55.7 57.7 58.8 60.3 – std 3.02 1.72 1.55 2.04 1.48 1.75 Structured 54.3 54.8 55.4 56.1 57.8 59.0 – std 0.759 1.81 2.11 1.5 0.974 1.64 Natural 48.7 57.1 56.0 59.4 59.8 61.6 – std 1.4 0.216 0.471 0.698 1.19 0.262 25

G T IMESTAMPS API CALLS

 For reproducibility purposes Table 52 and 53 contain the dates and times the APIs from OpenAI and Cohere were queries for the results. 26 27 28 .852 2.32 2.22 2.16 2.58 1.41 Natural 58.4 63.1 67.1 64.7 65.1 69.2 – std 1.13 1.67 0.665 1.13 0.419 0.648 29 std 0.189 1.18 0.535 1.44 0.432 0.732 Natural 60.8 50.3 50.8 56.7 56.1 53.5 – std 1.09 7.11e-15 7.11e-15 0.236 0.613 0.408 30 1.92 1.18 0.942 0.471 1.04 1.93 Natural 64 5 65 5 65.9 65.1 66.8 66.5 – std 0 377 1 97 0.822 0.579 0.66 0.748 31 .36 4.93 3.47 2.72 5.0 Natural 63.4 65.7 63.4 63.5 65.9 65.0 – std 2.26 1.44 2.3 0.713 1.03 0.205 32 33 34 std 0.759 0.613 0.634 0.33 0.736 1.13 Natural 60.0 57.4 57.8 59.7 58.3 59.9 – std 1.78 0.492 0.386 0.0471 0.556 0.294 35 36 Natural 56 54 60 236 37 38 39 23 14:27:15 GPT-3-davinci/5-shot 2022-09-23 15:10:40 GPT-3-davinci/10-shot 2022-09-23 16:04:53 GPT-3-davinci/15-shot 2022-09-23 17:17:04 GPT-3-davinci/30-shot 2022-09-23 18:36:38 OpenAI-ada/0-shot 2022-08-17 16:59:45 OpenAI-ada/1-shot 2022-08-17 18:23:12 OpenAI-ada/5-shot 2022-08-17 19:16:48 OpenAI-ada/10-shot 2022-08-17 20:24:16 OpenAI-ada/15-shot 2022-08-17 21:21:46 OpenAI-ada/30-shot 2022-08-17 22:44:47 OpenAI-babbage/0-shot 2022-08-17 11:50:44 OpenAI-babbage/1-shot 2022-08-17 12:22:08 OpenAI-babbage/5-shot 2022-08-17 12:50:59 OpenAI-babbage/10-shot 2022-08-17 13:27:52 OpenAI-babbage/15-shot 2022-08-17 14:57:43 OpenAI-babbage/30-shot 2022-08-17 15:45:16 OpenAI-curie/0-shot 2022-08-18 04:39:55 OpenAI-curie/1-shot 2022-08-18 05:10:17 OpenAI-curie/5-shot 2022-08-18 05:40:56 OpenAI-curie/10-shot 2022-08-18 06:15:28 OpenAI-curie/15-shot 2022-08-18 06:53:09 OpenAI-curie/30-shot 2022-08-18 07:35:40 OpenAI-davinci-001/0 shot 2022 08 26 20 26 21 OpenAI-davinci-001/1-shot 2022 08 26 21 02 31 OpenAI-davinci-001/5 shot 2022-08 26 21 35 19 OpenAI-davinci-001 10 shot 2022 08 27 07 14 02 OpenAI-davinci-001/15 shot 2022 08 27 07 58 25 OpenAI-davinci-001/30-shot 2022 08 27 08 44 42 OpenAI-Davinci-002/0-shot 2022 08 10 21 41 50 OpenAI-Davinci-002/1 shot 2022 08 11 10 04 17 OpenAI-Davinci-002/5 shot 2022 08 12 15 41 45 OpenAI-Davinci-002/10 shot 2022 08 12 16 41 14 OpenAI-Davinci-002/15 shot 2022 08 16 12 11 43 OpenAI-Davinci-002 30 shot 2022 08 16 14 35 38 40 41

