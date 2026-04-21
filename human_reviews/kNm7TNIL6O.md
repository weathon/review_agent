# UPAR: A Kantian-Inspired Prompting Framework for Enhancing Large Language Model Capabilities

- Avg Score: 5.00
- Decision: Reject
- Scores: 3, 6, 6

## Abstract
Large Language Models (LLMs) have demonstrated impressive inferential capabilities, with numerous research endeavors devoted to enhancing this capacity through prompting. Despite these efforts, a unified epistemological foundation is still conspicuously absent. Drawing inspiration from Kant's a priori philosophy, we propose the UPAR prompting framework, designed to emulate the structure of human cognition within LLMs. The UPAR framework is delineated into four phases: “Understand”, “Plan”, “Act”, and “Reflect”, enabling the extraction of structured information from complex contexts, prior planning of solutions, execution according to plan, and self-reflection. This structure significantly augments the explainability and accuracy of LLM inference, producing a human-understandable and inspectable inferential trajectory. Furthermore, our work offers an epistemological foundation for existing prompting techniques, allowing for a possible systematic integration of these methods. With GPT-4, our approach elevates the accuracy from COT baseline of 22.92% to 58.33% in a challenging subset of GSM8K, and from 67.91% to 75.40% in the causal judgment task.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a Kant-inspired framework for the sequential prompting of LLMs. After revisiting Kant's transcendental philosophy structure, the paper proposes a four-step framework that is based on these ideas, consisting of understanding, planning, acting, and reflecting. The newly proposed framework called UPAR is compared to existing methods that roughly cover some but not all aspects of UPAR, and shows to be superior when using GPT-4.

### Strengths
S1) The argument that the ongoing paradigm shift brought by LLMs can use some formal framework from cognitive science is compelling and timely.

S2) The idea to synthesize the thinking steps along the four subsequent components is interesting and aligns well with prior work on LLMs. 

S3) The results, especially the ablations, are informative and improve the understanding of the contribution of each component.

S4) The formalizations of the four steps are useful and refreshing.

### Weaknesses
W1) The chief weakness of this paper is that the paper seems to exaggerate its contribution. The promise of the paper is that it will ground the thinking steps with LLMs in some objective framework that has been well-accepted in psychology/philosophy. What happens in the paper is that the coupling between Kant's theory and the UPAR framework is loose at best (compare figures 2 and 3). Now, the paper is unclear whether UPAR is: A) only inspired by this theory or B) claims to be supported by this theory (which is a much stronger claim). 

If A) is the claim, then a loose coupling would be fine, but this would undermine many of the novelty claims (which are anyway difficult to follow), like "often concentrate exclusively on local and specific reasoning processes, neglecting the intrinsic human cognitive structures underpinning language", "these tools are products of human thought, not the foundation of thinking", and "these tools are the creations of human intellect rather than the basis of human reliable thinking".  

If B) is what the paper claims, then the authors really need to justify why Kant's theory is taken as the golden standard of "human reliable thinking" and how the UPAR framework aligns seriously with Kant's framework.

W2) While the paper emphasizes the need for a model to receive the full complexity of UPAR thinking, in fact, the main UPAR variant being emphasized is its "simple" variant, which replaces the understanding aspects of Kant's framework with other ones (entities and relations) that are indeed intuitively more useful for the tasks at hand. Surprisingly, the authors do not comment on this finding and what this means for the overall premise of the work.

W3) The paper makes claims that using reasoning would reduce "illusions", which is a nice and compelling statement. However, it is unclear whether UPAR indeed results in less illusions. In general, it is unclear what the qualitative improvement brought by UPAR is; but the improvement does not seem to be some emergent/qualitative jump, but rather a little better overall score while still producing judgments that are as unreliable as the baseline model (as far as I can see, there is no way to guarantee reliability of the reasoning in UPAR).

### Questions
Q1) Can you please clarify the relation between Kant's theoretical work and UPAR?

Q2) How is Kant's theory (or UPAR's framework) guaranteeing higher reliability or less illusions?

Q3) How do you interpret the fact that the UPAR-S method is generally much more useful than the UPAR framework? What does this mean for the general premise of the paper?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper represents UPAR, a prompting framework inspired by Immanuel Kant's arguments about the structure of the human mind. It consists of "understand, plan, act, reflect" steps which ask the model to break down its given problem in a pre-specified way.

### Strengths
* The paper was fun to read and draws on interesting ideas.
* The prompting approach is simple and seems like it'd be easy to understand if it were fully described in the paper.

### Weaknesses
1. The paper lacks a lot of important details.
  a. I'm confused by the descriptions of the P, A, and R steps. Sections 3.2–3.4 just discuss the philosophical side and motivation without saying how the model is actually prompted. That seems to me like the most important thing to communicate in the paper. Please include it, like you did with "Understand". Also, please be more specific even in the "Understand" section about how your prompting approach works. Do you prompt it four separate times, once for each question? How do you instruct the model outside of just asking it the question?
  b. Does it work only on instruction-tuned or RLHF models, or is it designed to work with pretrained LMs as well? Can you use few-shot examples? Where would they come from? Is there some set of tasks on which it doesn't work? What changes besides the simplification in UPAR-S might be necessary to make it work in other cases?
2. The results are not very promising. It does yield improvements over zero-shot CoT with GPT-4, but only very small ones, and it's unclear whether they are statistically significant (how big are the test sets? What's the total n being tested on?). The only case of a large gap with GPT-4 zero-shot was on a subset of GSM8k _filtered to examples which zero-shot CoT with GPT-4 got wrong_ — not a fair comparison.
3. It's not totally clear to me how deep the relationship goes between the prompting approach and the philosophical backdrop. Especially given the lack of detail in the paper, one can imagine many possible ways of implementing the same idea. Why this way in particular? For example, the assignment of questions to the pure categories of understanding seemed like a little bit of a stretch from its philosophical source material. If the whole point of the paper is that this framework follows from Kant, especially as the results are weak, I think it's important to make that case rock solid.

### Questions
See my questions in 'Weaknesses' above.

Note for authors and AC: I am passingly familiar with the underlying philosophy, but not enough to evaluate whether this paper's characterization of Kant, or the connection of their method to Kant's arguments, is accurate or satisfactory.

**EDIT:** I know it might be a bit late for this, but I've revised my Soundness score up and my review score to just over the acceptance threshold. The authors clarified a fair bit of stuff in the new draft and the results are a lot stronger. I'm still uncertain about the magnitude of the contribution and the soundness of all of the philosophical connections, hence decreased my confidence to 2.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new prompting framework based on Kant's philosophy to enhance large language models.  It tries to emulate the structure of human cognition within the LLMs. The framework of UPAR: understand, plan, act and reflect, tries to structure the prompt with these four reasoning components based on Kant’s philosophy and even more fine-grained elements of understanding such as time, space,  events and their relationships and more. They show that asking the language model to adhere to these steps of reasoning improves accuracy in question answering/reasoning. They were able to improve GPT-4 results on two benchmarks on causal judgement and grade school math problems (GSM8k) compared to COT prompting.

### Strengths
-The paper is very well-written.
-The interdisciplinary aspect of paper is novel and interesting as it applies Kant's philosophy to reasoning structure of LLMs.
-The background information and overview of the related work was done very good and neatly. 
-The results show some improvements in reasoning over text compared to baselines.

### Weaknesses
--The experimental results are not very strong. GPT3.5 does not show any improvements. GPT-4 has a mixture of results, mostly improves a bit though.  
 
--The fact that they needed to simplify the prompt steps to obtain better results weakens the idea of applicability of the theory in this context. Specially, there are several results that show dividing the input to parts and having step-by-step reasoning is helpful, so I am not sure if Kant's theory is specifically helpful here or dividing the problem to sub-problems in anyways can be helpful. The results are only compared to COT not any other newer variations of step by step reasoning compared here. More baseline might show the advantage of this theory better (?). 

--It was not clear how they provided the information about each step of reasoning to the LLM, I could not see additional descriptions in the prompt other than the keywords like understand, plan, etc.

### Questions
-How many examples did you provide in the context [input of the LLM]? 
-Did you only use the keywords of understand, plan, etc along with an example for in-context learning? and without any further explanation?
-Did you do this step by step? i.e. the output of first step will be the input to the next step? do you concat the output each time to the older input? 
More details about the exact interactions with the LLM for obtaining the answer will be helpful.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
