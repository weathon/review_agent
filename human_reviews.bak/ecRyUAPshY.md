# CodeUpdateArena: Benchmarking Knowledge Editing on API Updates

- Decision: Reject
- Scores: 8, 6, 3, 5, 5

## Abstract
Large language models (LLMs) are increasingly being used to synthesize and reason about source code. The libraries and API functions they invoke are continuously evolving, with functionality being added or changing. Yet, no prior work has studied how an LLM's knowledge about code API functions can be updated. To fill this gap, we present `CodeUpdateArena`, a benchmark for knowledge editing in the code domain. An instance in our benchmark consists of a synthetic API function update paired with a program synthesis example that uses the updated functionality; our goal is to update an LLM to be able to solve this program synthesis example without providing documentation of the update at inference time. Compared to knowledge editing for facts, success here is more challenging: a code LLM must reason about the semantics of the modified function rather than just reproduce its syntax. Our dataset is constructed by first prompting GPT-$4$ to generate atomic and executable function updates. Then, for each update, we generate program synthesis examples whose code solutions are prone to use the update. Our benchmark covers updates of various types to 54 functions from seven diverse Python packages, with a total of 670 program synthesis examples. Our experiments show that fine-tuning open-source code LLMs (i.e., DeepSeek, CodeLlama) on documentation of a new update does not allow them to incorporate changes for problem-solving. However, prepending the same information does help, establishing that the information is present, and careful fine-tuning on examples demonstrating the update shows improvement, paving the way for better knowledge editing techniques for code.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a new program synthesis dataset focused on evaluating how well an LLM can be adapted to handle changes to an existing API. In each task, an LLM gets an API change description and a few example usages, and after after fine-tuning (or otherwise integrating this information) must solve synthesis examples using the new API, without being prompted with the new API. The evaluation metric ensures that the new API is used and that the old API wouldn't have worked in place of it. The benchmark itself is LLM generated, but with considerable testing and manual inspection for quality control. The evaluation finds that while fine tuning on updated API examples improves performance a bit, none of the evaluated fine tuned models achieves high accuracy. They also ensure the solvability of the dataset by showing that in a few-shot setting with the API functions provided, most of the benchmark is solved.

### Strengths
The dataset created in this work deals with an interesting and real-world applicable problem, one that is particularly relevant to large language models which are trained on large amounts of historical code. Broadly I think that such a dataset is a good thing for the community to have, and that it will encourage future research in this area – and this work does a good job of building and evaluating the dataset.

The metrics presented in the work, UPass@K (performance on problems using new API) and SPass@K measuring specificity (i.e. performance degradation on unrelated problems), are solid metrics for evaluating the task. In particular it's important that UPass@K ensures that the new solution not only passes all the unit tests but does so using the new functionality, and that swapping in the old functionality would *not* solve all the tests. Overall this work does a good job of setting up a sensible and precise problem statement / set of metrics for future work to be evaluated on.

While an LLM is used for dataset generation, considerable effort was put into automatically and manually screening the generated tasks – both for the API specifications (4.2) and for ensuring that solving failures in the few-shot setting are not because of incorrect specifications (Table 3 and Appendix C.2).

The evaluation is reasonable for a paper presenting this dataset – a novel approach to solving the problem isn't presented, but doesn't need to be for a dataset paper in a new, compelling domain.

### Weaknesses
- I'm confused about how the Base Models could perform so well, which I've elaborated on in the Question section – this is my main concern since it suggests there's something I'm misunderstanding either about those Base Models or the dataset itself.
- (minor) Including the c=1 (so r=2) condition in ablation Table 5 would be nice to see, just to be sure that c=2 really provides the best performance benefit before it starts to drop with c=3.
- Quick fix – Table 8's description should say 77 failures not 66, at least that's what the column adds up to.

> We observed relatively few cases of incorrect test cases or bad specifications, indicating that our dataset is of sufficient quality to test knowledge editing methods.

- From the appendix it seems to be that 19.5% (15/77) of failures are for these reasons, which is not ideal but okay – manual inspection to fix these would be nice for verifying the quality of the dataset. But not a big problem given what a small fraction this is of the total dataset (since 19.5% is only within failures, which themselves are 16.9% of all the problems so it's only about 3% of the whole dataset, if I understand correctly).

### Questions
- I would have expected the Base Model row of Table 4 to get essentially 0% for UPass for all models, since UPass requires not only solving the problem but actually solving it using new API in a way that would not work with the old API – but instead models get as high as 22.7% in Pass@1 and 33.3% in Pass@5. My understanding is that the Base Model doesn't get to see the new API at all (neither in fine tuning nor prompting), so in order to solve any problems it would have to somehow hallucinate the correct API change, misunderstanding the old API to behave like the new API despite having never heard about the new API. It seems really surprising that that would happen so often.
  - Am I understanding the Base Model correctly? If so, can you explain this, perhaps providing examples of where this happens to help understand how it could be so common? How could models solve anything without having heard of the new API function?

- In section 6 it's mentioned that 2 program synthesis problems are used for fine tuning, because many update functions only have 3 synthesis problems so the remaining one is held out for testing (and cross validation is used). In cases where where there are more than 3 synthesis problems, what does it mean to be "correct" on this update specification. Does the model need to solve *all* of the held out synthesis problems in one go (with k=1 or k=5 attempts)? Or does it only need to solve one at a time, and these are treated as separate entries in the dataset (where each entry has 1 test synthesis problem and 2 training synthesis problems)?

### Soundness
3

### Presentation
3

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
The paper introduces a benchmark to assess LLMs' ability to update their knowledge of API functions. It includes synthetic API function updates and program synthesis tasks using these updates. The paper evaluates various LLM update methods, such as fine-tuning open-source models on update documentation, prepending update information at inference time, and fine-tuning on examples. The dataset comprises 670 program synthesis examples involving updates to 54 functions from seven Python packages, all generated using GPT-4.

### Strengths
- **Relevant and timely research**: Updating APIs presents a significant challenge for LLMs. Therefore, assessing how effectively LLMs evolve with API changes addresses a valuable and necessary research area.
- **Novel techniques**:
    - The paper explores code-specific update methods like fine-tuning with examples, recognizing that LLMs must understand the semantics of an update rather than just its syntax. 
   - The paper also establishes a new foundation for measuring LLM update capability in coding using UPass@k and specificity metrics.
- **Interesting insights for future research**: This paper also many interesting insights to help future research in this area. For example, 
    - The paper shows that fine-tuning with examples outperforms fine-tuning on documentation.
    - The paper highlights the issue of specificity loss when fine-tuning with knowledge updates, encouraging future research to address this challenge.

### Weaknesses
**Evaluation limitations**: The model is penalized if it generates a correct program without using the updated function, even though it may still achieve the correct outcome. And from the examples given in the paper, it seems like this can occur very commonly. It would be helpful to explore a metric that captures these cases more accurately. **Suggestion**: Comparing pass@k with upass@k in Table 4 could offer some insight here.

**Possible compounding errors in LLM-generated datasets**: Since the dataset is fully generated by LLMs, LLMs can introduce errors affecting the quality of the dataset. It is very hard to establish that the update implementation is correct since the generated test cases themselves could be wrong. It is possible that LLM generated both wrong implementation and wrong test cases, but the wrong implementation might pass the wrong test cases. Similarly, LLM generated test cases for the program synthesis problems are used to the efficacy metric UPass@k and again there is no guarantee that these test cases are correct beyond just random manual inspection done by the authors.

### Questions
1.	Can you add pass@k metric to Table 4 in addition to UPass@k metric? 
2.	How are LLMs updated on the docstring? For FT(U), how many times is the update instruction finetuned on? Does this hyper-parameter impact the performance? 
3.	What are the random updates in the training examples in FT(PS) setting? Can you provide some examples?
4.	In Table 4, what is the baseline approach (is it the model without any fine-tuning or prompting for updates)? Additionally, what is the Upass@k for the baseline model? It’s surprising that this value is significantly greater than 0, given that models without update knowledge would likely fail to use the new API correctly.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes CodeUpdateArena, a benchmark for knowledge editing in the code domain. CodeUpdateArena evaluates the extent to which a pretrained LLM’s knowledge of code API functions can be updated. This benchmark is a set of API updates and program synthesis examples.

The authors evaluate different approaches for updating information, such as pretending the function update’s docstring in the prompt, fine-tuning on code update information, and fine-tuning on program synthesis examples. They find that fine-tuning on documentation of a new
update does not allow LLMs to incorporate changes for problem-solving, while prepending the same information or fine-tuning on examples does help the LLM.

### Strengths
- The paper is well-written, with an easy-to-follow running example in the description of the synthetic data generation process.
- The benchmark is open-sourced with 54 functions from seven diverse Python packages with 670 program synthesis examples.

### Weaknesses
- While useful, the impact of the paper is limited as the size and diversity of the data set is small (54 functions) in one programming language (python).
- The core contribution is that LoRA fine-tuning on natural language descriptions of code updates is worse than adding the updates in prepended prompts or LoRA fine-tuning on code examples.

### Questions
How might these results vary according to the size of the training dataset? E.g. if you employ full-parameter fine-tuning with significantly more samples, how might the efficacy of the FT methods change?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a benchmark to assess a model's ability to adapt to changes in API function calling. They show that fine-tuning the model on updated program synthesis problems can improve the model's ability to embrace the new API function call.

### Strengths
This paper poses an interesting question of how to evaluate model's robustness to API changes and introduces a dataset to evaluate this. The further provide a testing environment to compare different approaches to augmenting model capabilities with new APIs, testing both in-context and fine-tuning approaches.

### Weaknesses
1) It's not clear that the types of updates proposed by GPT-4 are representative of the types of updates found in the wild. With synthetic data, GPT-4 may be idiosyncratic in the updates proposed, step-4 of deduplication removing 53% of the problems seem to indicate this is a significant downside of using GPT-4 for problem generation. A comparison to historic API changes would help justify this.

2) Related work would be better in section 2 to provide better context on what other work has been done in this topic. 

3) The naming of "Arena" seems to imply side-by-side comparison and human input the way ChatbotArena does. This benchmark does not do this. The creation of test cases and sample tasks that involve the new updated API function is more akin to R2E https://r2e.dev/pdfs/paper.pdf

4) It's not clear that updates to APIs happen at a high frequency compared to updates to post-training in proprietary models. In those cases, it may be simper or beneficial to amend the data in the post-training corpus. It would be worth studying if models can be taught to distinguish the difference between package/api versions. 

5) The results are limited to statistical and data processing tools popular in python. It would be interesting to show on other APIs like web applications and visualization packages like plotly and matplotlib.

### Questions
1. Do these results replicate when not using LoRA but using full fine-tuning? It's possible that full fine-tuning with data augmentation will allow the model to pay better attention to the in context docstrings and API description. Perhaps something like RAFT (https://arxiv.org/abs/2403.10131). 

2. Does GPT-4 fine-tuning lead to improved performance? Otherwise, this result might be only limited to smaller models with weak reasoning and coding abilities. 

3. Does the fine-tuning lead to improved capabilities for other coding tasks like test-generation for code, self-repair, and code execution?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes CodeUpdateArena, a dataset of synthetic updates to library APIs in Python. Each instance in the dataset consists of the update (e.g., add/modify a parameter), a code synthesis problem using the changed API, and a set of tests to validate the synthesis results. The authors use GPT-4 to generate most of the dataset, followed by manual inspection to eliminate low-quality or incorrect instances. 

The authors explore three evaluation strategies with closed-source and open-source LLMs (GPT-4, Claude, DeepSeekCoder, and CodeLLama): prepending prompts with api update information, fine-tuning with docstrings, and fine-tuning with synthesis examples. 

The results show that the prepend and fine-tuning options improve the efficiency of the LLMs but also hurt their specificity, i.e., their performance on unrelated tasks.

### Strengths
- The paper targets an interesting problem about the integration of API updates in LLM code generation. I believe this is an important task given that LLMs do not have knowledge about the version of the libraries that they use to generate code. Hence, without the updated information, the generated code might be incorrect or not compilable.

- The proposed dataset, CodeUpdateArena, is potentially useful to evaluate if LLMs can generate updated information. 

- The paper covers a reasonable set of LLMs in their evaluation, including both closed-source and open-source LLMs.

### Weaknesses
- The fact that dataset is generated using GPT-4, including both the updates, synthesis problems, and the tests to evaluate the generated synthesis, is questionable. This approach limits the quality and realistic nature of the dataset. There is historical data and commits on API updates for many libraries on Github -- which would be more realistic and perhaps still challenging for the LLM to work with. 

- Unlike historical facts, multiple versions of a library API can exist at the same time and are concurrently used by different users due to compatibility issues. E.g., the latest version of numpy may require python 3.12 or greater; such users may want the older API. Hence, getting the latest version of the API is not always important, rather the LLM may have to understand the context or dependencies better for correct prediction. Hence, fine-tuning the LLM with fixed updates does not make sense. 

- Further, numerous libraries are updated every day/week, hence fine-tuning LLMs every time for these updates is not scalable.

- The evaluation approach and metrics do not provide a way to gain high confidence if the LLM learned the API update. There can be many other synthesis scenarios that requires using the API in question. How can we confidently say if the LLM learned the update if the evaluation is only done on a handful of synthesis examples? Hence a more robust evaluation is needed for this aspect.

- Given the fact that the larger LLMs like GPT-4 and Claude already obtain ~60-70% efficacy, this benchmark seems to be challenging only for smaller and slightly outdated LLMs. With the latest LLMs, the benchmark might soon saturate. Hence, it is unclear if the benchmark will be useful to evaluate the upcoming generations of LLMs.

### Questions
1. Why do the authors generate synthetic data instead of using historical API updates?
2. How can we rely on tests generated by the LLM?
3. Given the very few examples, the SPass metric is a weak measure of LLM's updating ability? How can we gain more confidence in this metric?

### Soundness
2

### Presentation
3

### Contribution
2
