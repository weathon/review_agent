# Unearthing Skill-level Insights for Understanding Trade-offs of Foundation Models

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
With models getting stronger, evaluations have grown more complex, testing multiple skills in one benchmark and even in the same instance at once. However, skill-wise performance is obscured when inspecting aggregate accuracy, under-utilizing the rich signal modern benchmarks contain. We propose an automatic approach to recover the underlying skills relevant for any evaluation instance, by way of inspecting model-generated {\em rationales}. After validating the relevance of rationale-parsed skills and inferring skills for $46$k instances over $12$ benchmarks, we observe many skills to be common across benchmarks, resulting in the curation of hundreds of \emph{skill-slices} (i.e. sets of instances testing a common skill). Inspecting accuracy over these slices yields novel insights on model trade-offs: e.g., compared to GPT-4o and Claude 3.5 Sonnet, on average, Gemini 1.5 Pro is $18\%$ more accurate in \emph{computing molar mass}, but $19\\%$ less accurate in \emph{applying constitutional law}, despite the overall accuracies of the three models differing by a mere $0.4\\%$. Furthermore, we demonstrate the practical utility of our approach by showing that insights derived from skill slice analysis can generalize to held-out instances: when routing each instance to the model strongest on the relevant skills, we see a $3\\%$ accuracy improvement over our $12$ dataset corpus. Our skill-slices and framework open a new avenue in model evaluation, leveraging skill-specific analyses to unlock a more granular and actionable understanding of model capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an automatic approach to infer underlying skills for evaluation instances by inspecting model-generated rationales. This helps in achieving a finer-grained understanding of model capabilities from existing benchmarks, increasing the interpretability of evaluating LLMs with benchmarks.

### Strengths
- The way of analyzing the model evaluation benchmark from the aspect of inferred skills is interesting and novel to me.

- This work utilizes a post-hoc verification of skills and validates its validity by comparing the results with human validation.

- The authors conduct experiments to show some interesting findings on skill generation and model evaluation.

### Weaknesses
Overall, the presentation needs improvement such as:
- A simple prompt example used to generate a detailed rationale shown in Figure 2 would be better.
- A clear subsection of used benchmarks and LLMs in the main paper.

I am confused about the description of the experimental section such as:

- Figure 8 is not clear enough at first sight. Which method uses skill annotations? And does the higher precision @ k = 20 indicate better performance or worse? Could you please explain the reason behind the evaluation?
- Figure 9 only depicts the number of skills per instance obtained by direct prompting vs. our rationale parsing. While the larger number cannot show the generation effectiveness. Reliability or relevance to the question are more important. Is there any other ablation study showing the comparisons concerning the reliability of annotated skills?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper mainly studies the evaluation challenge from testing multiple skills in modern LLM evaluations. Given a specific benchmark dataset, the skill-wise performance is obscured and the possible hidden rich signal is usually underutilized. To alleviate this issue, this paper propose a automatic approach for recovering the underlying skills given any evaluation instance. In specific, this is implemented by inspecting model generated rationales. By examining the relevance of rationale-parsed skills and inferring skills for 46k instances over 12 benchmarks, it is shown that many skills to be common across benchmarks, resulting in the curation of hundreds of skill-slices. These skill slices are finally further shown to deliver novel insights on model trade-offs.

### Strengths
- First of all, the LLM evaluation problem is very important, especially under the case where multiple or many skills are evaluated together. This has been shown by Figure 1 as well to clearly clarify the motivation of this work. 

- The paper is well written and easy to follow

- Automatic discovering the skills related to any evaluation instance is an important topic.

- Some findings are quite interesting. For instance, Gemini 1.5 Pro to be much stronger in math and science skills, like ‘computing molar mass’, while falling far behind for legal skills like ‘applying constitutional law’.

### Weaknesses
- The automatic verification protocol should be further explained. This is the single most important part of this method pipeline. From my understanding, querying LLM models or APIs with specific prompt engineerings can always get some sort of skills, rationales or chain of thoughts. The more important thing is how can you guarantee these generated content are properly validated. The author claimed that they will "release all our rationales, skill annotations, and skill-slices, which we term the Skill-Index, to the public.", but more explanation should be provided. The verified trustworthy accuracy here will be quite important.

- The technique contribution looks a bit limited. All the work done here are PE related work.

### Questions
- Assume in the industry, you are not allowed to access those strong models (e.g., GPT-4o). Given a specific LLM, the skill discovering ability is actually constrained by the skill ability of this LLM. How could you explain the effectiveness of automatic method, if let's say the model itself is weak in one category of skill while you expect this LLM can identify this skill and generate the corresponding rationales automatically?

- Given the generated rationales will be treated as a golden standard for the future use in other model evaluations, how to guarantee the trustworthiness of such rationales? This is actually the same question listed in weakness part.


- I am curious: How is the cost of conducting such kind of research?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new method to evaluate LLMs by analyzing their skills rather than just relying on overall accuracy scores. It extracts detailed reasoning steps (rationales) from models like GPT-4 to identify the skills used, grouping these into "skill-slices". By doing this, the researchers can reveal strengths and weaknesses that are not revealed through traditional evaluations. They also introduce probing questions to validate the identified skills, ensuring that the analysis reflects the model's true abilities. This approach aims to provide a deeper understanding of model capabilities and guide improvements for future models.

### Strengths
- The paper provides a new insight to evaluate LLMs by evaluating on fine-grained skill analysis rather than traditional scores. This method of extracting "skill-slices" from model-generated rationales allows for a deeper understanding of models’ capabilities. The insight is reasonable, and the meaningful for current LLM studies.
- The design of probing questions allows for a robust validation for the extracted skills, which could reduce the risk of hallucination.
- The paper provides extensive experimentations across multiple benchmarks and models. It clearly provides solid analysis and findings which might benefit for future studies.
- The paper provides skill annotations, which would guide future model and dataset developments. 
- The skill-slices design could benefit for the explainability and transparency of large language models.

In general, it is a good paper.

### Weaknesses
- Real-world tasks often require a combination of multiple skills. The current method of skill extraction assumes that skills can be categorized and analyzed in isolation. This overlooks the fact that the interaction between multiple skills may be non-linear and context-dependent, meaning that combining skills does not always produce the same outcome as using those skills independently.
- Some LLMs may operate as "black boxes," where the true internal reasoning process is not fully transparent. This means that the skills inferred from the generated explanations may not align with the model's real decision-making processes.

The paper has already discussed some limitations.

### Questions
Given that models may hide internal process and not explicitly demonstrate all the skills they use, how would we evaluate those models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper uses LLMs to annotate benchmark instances with skills required to solve them. The authors validate their annotation in multiple different (mostly automated) ways. They also analyze how different models' capabilities vary when restricted to so-called skill slices, i.e. sets of benchmark instances that all require a specific skill to solve (according to the automated annotation).

### Strengths
- The writing is generally quite engaging 
- Decomposing broad benchmarks into more specific ones seems quite useful for things like model selection, whenever models are to be deployed in a specific setting. 
- Despite the caveats listed in the weaknesses, the results on differences in model capabilities are quite interesting. The additional validation experiments, like the one on model routing, make me believe that there is indeed meaningful signal in the results.

### Weaknesses
- All results on comparing pairs of models would strongly benefit from adding error bars to show that the differences are indeed exceeding what would be expected from noise when comparing equally performant models. 
- Additional statistical analysis to show that the most stark model differences cannot be explained by random noise + testing many hypotheses (i.e. comparing models on many different skills) would also strenghten the results. 
- Releasing the annotations would be helpful for the reviewers to get a better intuitive picture of the validity of the proposed methods. 

- Nitpicks:
   - The writing sometimes seems a bit too grandiose:
       - For example, whether the paper is "providing a valuable resource for the research community" seems like something the community will decide on.
  - The formatting in Appendix E is off (lots of empty space)
  - Appendix G is incomplete (it ends mid-sentence)
  - Some kind of face-value evaluation (such as a small scale human evaluation) beyond GPT-4 as judge would be useful for section 5.

### Questions
- What does figure 8 show? Is the last row's annotation a typo?
- What happens when combining skill and text/image embeddings in figure 8?
- What are the pros and cons of the verification approach based on negative samples?
- Which skills and benchmarks does the routing approach improve most on (let's say compared to an ensemble, as well as "perfect" routing that has access to the ground truth)?

### Soundness
3

### Presentation
3

### Contribution
3
