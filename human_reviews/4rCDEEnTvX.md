# From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3

## Abstract
Large Language Models (LLMs) have achieved remarkable success, demonstrating powerful instruction-following capabilities across diverse tasks. Instruction fine-tuning is critical in enabling LLMs to align with user intentions and effectively follow instructions. In this work, we investigate how the instruction fine-tuning modifies pre-trained models, focusing on two perspectives: instruction recognition and knowledge evolution. To study the behavior shift of LLMs, we employ a suite of local and global explanation methods, including a gradient-based approach for input-output attribution and techniques for interpreting patterns and concepts in self-attention and feed-forward layers. Our findings reveal three significant impacts of instruction fine-tuning: 1) It empowers LLMs to better recognize the instruction parts from user prompts, thereby facilitating high-quality response generation and addressing the ``lost-in-the-middle'' issue observed in pre-trained models; 2) It aligns the knowledge stored in feed-forward layers with user-oriented tasks, exhibiting minimal shifts across linguistic levels. 3) It facilitates the learning of word-word relations with instruction verbs through the self-attention mechanism, particularly in the lower and middle layers, indicating enhanced recognition of instruction words. These insights contribute to a deeper understanding of the behavior shifts in LLMs after instruction fine-tuning and lay the groundwork for future research aimed at interpreting and optimizing LLMs for various applications.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper does a comprehensive comparison between a pre-trained model (LLaMa) and two instruction fine-tuned models (Vicuna and Alpaca). The authors look at the importance of particular types of input tokens (like instruction words) for the gold output probabilities, as well as what parts of the inputs models focus on. They find that instruction-tuned models suffer less from the known problem that models focus more on the first and last part of the prompt than the middle, and that when Vicuna follows user instructions, instruction words are more important. The authors also use an interpretability technique to look at concepts the FFN and the attention heads encode. finding, among other things, that there is a shift in represented concepts after fine-tuning (more focus on math, writing, coding), but no shift in linguistic concepts.

### Strengths
This paper does a lot of experiments and seemingly has some very interesting results that can contribute to our understanding of the behavior shift after instruction tuning:
- there is a shift in representation of concepts like math and coding but not in linguistic concepts after instruction fine-tuning
- instruction-tuned models have less of a problem with not focusing on the middle of the prompt, but the problem is still there

### Weaknesses
The paper is not written very clearly, and is very hard to follow because of it. There are many errors and difficult-to-interpret formulations. Additionally, the authors sometimes compare to the baseline model LLaMa, and sometimes do not. The latter results are difficult if not impossible to interpret. The qualitative results are sometimes based on a single prompt or only a small subset of analyses concepts, which makes it hard to know if the results hold for other prompts and for the other concepts. In some more detail:

- Section 3.1.3 is seemingly based on a single prompt. The results here are also very hard to interpret from just this single saliency plot. I'd say you can leave this out entirely and just pose this as a hypothesis and confirm in section 3.2.3, or alternatively somehow show that it holds for more than just this single prompt.
- You only show the importance density on instruction words for Vicuna in Table 1. How can we interpret the shift between LLaMa and Vicuna here if the importance density for LLaMa is not mentioned? You say *"Table 1 underscores that instruction fine-tuned models excel in identifying and harnessing instruction words from user prompts for response generation"*, but how can you say that without a baseline comparison? Maybe LLaMa will follow instructions more when more importance is on instruction words as well?
- In table 2 you mention 5 of 300 principal component concepts. The reader is left wondering if the rest is not interpretable? And how does this compare to LLaMa? Similarly for observation 5, you talk about these 4 concepts, and it's unclear what's going on with the others or how many there even are.
- You never mention in the main text that you use chatGPT to identify concepts.
- In the discussion you claim that your results show the importance of prompt diversity, but it's unclear to me how they show that.

As mentioned, the text is very difficult to follow:
- After reading the abstract and introduction, I have very little understanding of the findings. For example, you keep referring to the "lost-in-the-middle" issue without saying what this is. After reading the entire paper, it's clear that this can be explained in very few words.
- You don't mention you compare to a non-fine-tuned baseline until section 3
- In the introduction you say that instruction-tuning updates model weights different from pre-training, and it's unclear what you mean since pre-training of course also updates weights.
- In the introduction you use an example-sentence "me and him goes to a stores for buying a apple" that has many errors, and it seems like this is on purpose to highlight something about the analysis, but it's completely unclear to the reader what this is supposed to highlight.
- Many examples of difficult to follow sentences, e.g. : Last paragraph on page 1; "What the encoded knowledge evolves after instruction fine-tuning?";  you write LLaMa differently many times (LLama, LLaMa, LLaMA);  "differed to typical language models", end of page 5; "one of the nice property by using l1/lp density function is if two input tokens ... would receives more .."; "to verify our assumption that a model is following instruction if it identify and use instruction words" should be "identifies and uses"; "where each prompt was manually annotated the instruction part"; "Before applying our method, we crease a new vocabulary"; "the build-in LLaMA vocabulary"; "indicating multi-linguistic knowledge is forgetting" should be "forgotten" or something; etc.

To summarise;

I think this paper has some very interesting results that merit publication, however the current write-up is very difficult to follow and the reader is left with many questions. Most importantly; some observations seem based on only a very small part of the analyses, the authors do not always compare to the baseline LLaMa, and most of the paper is difficult to follow.

### Questions
- Why do you include section 2.1? That is very common notation and the Transformer architecture can also be assumed known I'd say.
- Why do you use a different set of hyper-parameters for 3.2.2 than for 3.2.1 when you're trying to verify the assumption in the latter with the former?
- Could you add some justification for the analysis in 4.1? Why is this a sensible thing to do?
- Could you justify the linear approximation of the importance and especially why it's OK to just incorporate the embedding of the input token?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the behavior shift of LLMs after the instruction-following finetuning step. To this end, it introduces several explanation methods enabling the study of the effect of the finetuning procedure on the pretrained model. With a series of examples, the paper shows that:
- finetuned LLMs recognize better the instruction than the pretrained counterpart
- the finetuning process aligns the kowledge in FF layers with user tasks
- self-attention heads learn better word-word relations with instruction verbs after the finetuning process.

### Strengths
This paper develops a set of local and global methods to study large language models. These methods are more general than the scope of this paper and can be used by the literature to analyze any LM for any given task.

Also, the paper introduces good notations that help to understand the different methods and experiments.

### Weaknesses
- Obs-1 in 3.1.3 ("instruction fine-tuning helps the models distinguish between instruction and context words more accurately") is not convincing. That is, it is not clear how Figure 1 shows that instruction finetuning helps the model distinguish between instruction and context words more accurately. For instance, I cannot see that the instruction part is brighter than the context one in the figure, or that the finetuned model focuses more on instructions compared to the pretrained one. Some quantitative metrics could be helpful to interpret this figure.

- The introduced metrics rely on a few introduced parameters, namely L, b, p. The choice of these parameters was not justified in the paper and it does change depending on the paragraph. Some justification for the choices needs to be added to the paper.

### Questions
- I don't understand the derivation of S_{n,m} from I_{n,m} and its intuition. Can you add a few sentences about this metric and its need compared to using directly I_{n,m}? Also, how do the added hyper-parameters L and b need to be chosen?

- I find the results in Table 1 interesting. However, can you also add the results of importance density S_{n,m} for the pretrained model in the same task? This will give a good baseline to assess the impact of the finetuning process. As it is, it is not a study of the effect of the finetuning step

Misc/typos:
- "Also, what the encoded knowledge evolves after instruction fine-tuning?” --> “Also, how the encoded knowledge evolves after instruction fine-tuning?”
- I am not sure what this sentence means: we collect an annotated dataset, where each prompt was manually annotated the instruction part
- Hugginface library → Huggingface library
- To verify our assumption that a model is following instruction if it identify and use instruction words to guide the generation, we collect an annotated dataset --> “To verify our assumption that a model is following instruction*s* if it identif*ies* and use*s* instruction words to guide the generation, we collect an annotated dataset”
- Thus, it is naturally to represent this vector with the words having largest projection length → “it is natural”
- Before applying our method, we crease a new → we create

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a suite of methods designed at understanding and interpreting token importance for LLMs. The authors develop a series of different methods, which build on each other and measure importance between input and output tokens; between input tokens and the entire output sequence; and the "concept-level" knowledge in feedforward networks. They evaluate these metrics in the context of LLaMA and its instruction-tuned variants Vicuna, Alpaca, and connduct this analysis over the Self-Instruct, LIMA, and MT-Bench datasets. The authors highlight 3 key findings: (1) fine-tuning enables LMs to recognize "instruction words" from prompts; instruction-tuning "aligns knowledge in feed-forward layers with user-oriented tasks with minimal shifts across linguistic levels"; and instruction-tuning encourages attention heads in lower layers to learn word-word relations with instruction verbs.

### Strengths
I will describe strengths, weaknesses, and questions together in the format below.


# Overall

Overall, the paper feels somewhat exploratory and still unrefined in its methods, and its takeaways or intended impact. The authors propose a general set of methods for LLM interpretability -- which seem useful and applicable beyond instruction-tuning, as the authors observe -- but, as a consequence, the application to instruction-tuning feels somewhat less motivated, and the findings don't lead to any clear insights about future work, how instruction-tuning works, or how to improve the instruction-tuning process. I also identify a few gaps in the authors' assessment of the results, which are somewhat qualitative in certian places. I feel that the paper could be improved substantially, but lays a good methodological foundation and have the potential to "shine more light" on a process that isn't particularly well understood yet. However, I feel that revisions, described below, would make the paper much stronger.

# Major Comments

* A major concern I have with this paper is that it isn't clear what it ultimately delivers to the reader: the results don't seem particularly surprising, and also don't lead to clear directions for improving models, training, or instruction-tuning. It seems obvious that instruction-tuning words are critical for models to understand, and that instruction-tuned models would attend strongly to them -- they are the semantic crux of the inputs. This would have been predictable, and indeed strongly expected, without the methods. What additional insight do we learn here, and how can we use this scientific knowledge to improve models? I am not sure after reading the paper.

* In a few places, the quantitative evaluation is somewhat informal and could be improved. 
  - Section 3.1 is a particularly bad example of this: it uses a visualization of (if I am reading it correctly) a single prompt-response element in LLaMA and Vicuna to make its claim. This is achieved by a heatmap which requires 5 different annotation boxes to display the trends to the reader. A more quantitative analysis of these results would be more comprehensive and persuasive. As it is, the analysis is somewhat unclear (what are the "certain context lines that appear less bright"?) and hard to make sense of (is this a big effect? how does it compare to other examples, or non-instruction-tuning inputs?).
  - Similarly, many of the tables could improve the results presentations. For example, the null hypothesis being tested with the reported p-values is never reported (I assume it is that the difference between the two values is zero, but I am not sure). But, there are many details I am unsure of in these tables: what are the "instruction words" in Table 1, and how are they identified? Why are 5 of the top 69 principal components selected for Table 2, and where are the rest?

* In places it is difficult to follow the paper, due to a mix of missing context (I would suggest to move the related work to Section 2, since a great deal of the investigation here seems to be motivated by phenomena observed in prior work; the datasets are also never clearly introduced and explained, familiarity with them is assumed), terms that are not defined (I still don't fully understand the concept of a "linguistic level" and didn't find its definition in the paper), and typos (see below). The authors also sometimes skip over what seem to be important claims about their methods and metrics ("It's crucial to recognize that a word with a lower coenfidence doesn't necessarily imply it is a trivial word"; "if we simply map...it might give us the full picture of all these patterns"). I would suggest extensive revisions to clearly explain these details in terms of the provided equations (and would appreciate some clarification on those two sepcific phrases in the response).

* Since the authors present a general "toolkit" for interpretability in LLMs, it seems appropriate to consider applying this analysis to other contexts, or at least to clearly explain (1) where else these tools could be applied, and (2) how another user could utilize the authors' code (I would like a clearer explanation than "we will release our code and data upon acceptance" to understand how accessible and usable these tools would really be to the community).

# Minor Comments

* Please provide citations for the "well-accepted belief" regarding instruction-tuning described in Section 1.

* Based on the discussion in Section 1, it seems that Finding 3 is almost the same bottom-line finding as 1: "pre-trained models become adept at recognizing instruction words during instruction fine-tuning" (Finding 3, last sentence of bulleted paragraph in Section 1) could easily also describe Finding 1.


* The authors switch between "instruction tuninng" and "instruction fine-tuning" in the paper -- are these describing the same method? If so, please pick one and stick to it (instruction tuning seems more conventional).

* Please provide more details on the labeling of responses in Sectoin 3.2.2. Who labels them, and using what criteria?

* Please describe how the "new vocabulary derived from ShareGPT" is created in 4.2.

* Please provide more details regarding the PCA in Table 2: for example, seeing the cumulative proportion of variance explained would be useful in assessing whether the 69th-ranked vector is a meaningful one to be interpreting.


# Typos etc.

There are many typos in the paper; I list a few here but it needs a careful revision.
* P1: "what the encoded knowledge evolves" -> how the encoded knowledge evolves
* P2: "LLMs that billions of parameters."
* P2: "as their transition from language modeling to instruction following." -> as they transition
* P2: "our analyis to these concepts"
* P3: "a N-length" --> an N-length
* Paper switches between LLaMa, LLaMA, and Llama
* P4: "would receives more"
* P6: "we crease" --> we create
* P6: "the well interpretability of"

### Weaknesses
see above

### Questions
see above

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
