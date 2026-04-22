# GIFARC: Synthetic Dataset for Leveraging Human-Intuitive Analogies to Elevate AI Reasoning

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
The Abstraction and Reasoning Corpus (ARC) poses a stringent test of general AI capabilities, requiring solvers to infer abstract patterns from only a handful of examples. Despite substantial progress in deep learning, state-of-the-art models still achieve accuracy rates of merely 40–55% on the 2024 ARC Competition, indicative of a significant gap between their performance and human-level reasoning. In this work, we seek to bridge that gap by introducing an analogy-inspired ARC dataset, GIFARC. Leveraging vision-language models (VLMs), we synthesize new ARC-style tasks from a variety of GIF images that include analogies. Each new task is paired with ground-truth analogy, providing an explicit mapping between visual transformations and everyday concepts. By embedding robust human-intuitive analogies into ARC-style tasks, GIFARC guides AI agents to adopt analogical reasoning approaches, facilitating more concise and human-understandable solutions. We empirically demonstrate that GIFARC improves task-solving performance by aligning model reasoning with human analogical problem-solving strategies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GIFARC, a synthetic dataset of ~10,000 ARC-style tasks generated from short, motion-rich GIFs to inject human-intuitive analogical structure into ARC problem solving. A three-stage, VLM-driven pipeline (visual abstraction → task sketch → ARC task with executable Python solution and an explicit analogy label) converts each GIF into input–output grid pairs, code, and a named analogy (e.g., “blocked water flow”). The authors evaluate along three dimensions: (i) in-context learning, where examples from GIFARC encourage models to describe tasks using human-aligned analogies; (ii) human/LLM judgments of analogy alignment; and (iii) supervised fine-tuning (SFT) on GIFARC, which substantially lowers NLL on ARC-AGI-1/2 but yields only 2.9% accuracy on ARC-AGI-1 (from a 0.2% baseline) and 0% on ARC-AGI-2. The dataset and prompts are released publicly; the pipeline itself relies on a proprietary VLM (“o4‑mini”), and later-stage generation exhibits ~19–20% failure rates requiring retries.

### Strengths
- The paper targets a widely acknowledged gap between human- and model-level generalization on ARC and argues that analogical grounding can constrain search and make reasoning more human-like—implemented here by mining analogies from real-world GIFs. This is a creative way beyond rule-only synthetic generators. 

- GIFARC provides 10k tasks with executable solutions and explicit analogy labels, plus extensive appendices (prompts, statistics, visual examples) and links to data/visualization, enhancing reuse and analysis by the community.  

- Experiments show that “flattening” out analogy cues degrades models’ abstract descriptions, while human/LLM judgments confirm the analogical labels are meaningfully aligned with human concepts.  

- The paper provides effective figures and qualitative failure analyses that highlight where analogy identification and execution can be decoupled.

### Weaknesses
- Despite reductions in negative log-likelihood (NLL), exact-match accuracy improves only to 2.9% on ARC-AGI-1 and remains at 0% on ARC-AGI-2—well below recent ARC benchmarks, including the ARC Prize-winning results. This undercuts the core claim of “bridging the gap” between models and human-level reasoning.  

- The main experiments employ supervised fine-tuning (SFT) on direct grid prediction, whereas state-of-the-art ARC solvers use program synthesis or neuro-symbolic pipelines with test-time search. This mismatch makes it difficult to assess GIFARC’s true value as a source of priors or analogical guidance for code/search-based frameworks.  

- The pipeline’s heavy reliance on the closed/proprietary VLM “o4-mini” severely limits reproducibility, cost transparency, and future stability. The paper reports ~19–20% failure rates in later stages, raising concerns about the scalability and robustness of the data generation process.  

- Many ARC tasks hinge on abstract, symbolic, or topological relations not easily expressed or extracted from GIF-based analogies (such as symmetry, flow, or discrete logical rules). Experiments indicate cases where analogy detection succeeds but execution fails, highlighting representational limitations of the pipeline.  

- All reported fine-tuning results center on a single model (Mistral-NeMo-Minitron-8B-Base) trained for a fixed number of epochs, with no discussion of a hyperparameter search.

### Questions
- Given the significant NLL reductions but only 2.9% / 0% accuracies on ARC-AGI-1/2, what concrete bottlenecks prevent improved likelihoods from translating into exact-match solutions?

- Have you substituted “o4-mini” with open-source VLMs in any pipeline stages, and can you ablate components (with/without analogy labels, sketches-only, code-only) to pinpoint what drives the observed NLL improvements? 

- Can you quantify the most common causes of the reported ~19–20% stage failures, number of retries, and the overall API, time, and monetary generation costs?  

- How do the analogy categories in GIFARC compare to the latent concept space of ARC? Do GIFs bias toward physical and dynamic analogies, and are there observed systematic blind spots for more abstract/symbolic relations? 

- How do humans perform on GIFARC tasks, versus the original ARC distribution? Does difficulty match, and are common error types similar or different?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors use VLMs to synthesize (in 3 different stages) a new dataset, called GIFARC, of ARC-style tasks that are "inspired" by GIFs from an online collection.  These ARC-like tasks are annotated with "analogy labels" that are generated by the VLMs.  

In one experiment, the authors gave GPT 4.1-mini in-context examples from GIFARC, and then asked it to solve tasks from ARC-AGI-2.  They experimented with three types of in-context examples: 

-- 15 examples from GIFARC dataset with full description

 -- 15 examples from GIFARC where researchers have replaced "analogic terms" by "lower-level synonyms"

 -- 15 examples from GIFARC where researchers have replaced "analogic terms" by "lower-level synonyms" and the "solution" (Python program encoding the transformation) is not given

In a second experiment, a pre-trained LLM fine-tuned on examples from GIFARC, and then it is used to solve tasks from ARC-AGI-1 and ARC-AGI-2 evaluation set.  There is marginal improvement on ARC-AGI-1.  No tasks from ARC-AGI-2 are solved. 



In a second

### Strengths
The paper attempts to incorporate analogical reasoning into a method for improving LLM ARC solvers, in an original way.

### Weaknesses
There are two main weaknesses:  

First, it was a struggle to understand this paper---there are many aspects of it I found unclear.  It would really help to have a running example to illustrate the steps of starting with a GIF and generating a GIFARC task.   It would also be really helpful to have examples for the "full-description", "without analogy", and "without analogy and solution" data.    Also, Section 4.2 was particularly unclear. 

Second, the authors make claims that are not clearly supported by the evidence given in the paper, and the paper lacks methods for evaluating those claims.   First, it's not clear how successful VLMs are at generating useful "analogy-laden" descriptions and then tasks.   Second, it's not clear how much the in-context examples enable the LLM to generate a correct / useful analogy.  Figure 3 gives one example where a correct analogy is given, but this is just one example.  It would be useful to give results on the percentage of tasks that have a correct analogy and correct output grid, correct analogy and incorrect output grid, incorrect analogy and correct output grid, etc. 
Finally, for the fine-tuning experiment, the hypothesis seems to be that fine-tuning on GIFARC is what is causing the improvement over the base model.  It would be good to do a control experiment -- would an improvement over the base model still be seen if the fine-tuning data was, say, the ARC-AGI training set instea of GIFARC?  The authors need to show in a more compelling way that GIFARC is actually doing something above and beyond simple fine-tuning on ARC-like tasks.

### Questions
The authors state: ""Despite substantial progress in deep learning, state-of-the-art models still achieve accuracy rates of merely 40-55% on the 2024 ARC Competition" -- this is technically true, since o3-preview didn't compete, but o3-preview in high effort mode got 88% on semi-private test set.  It is misleading to leave this out.

Figure 1: "Illustration of two different solutions of ARC-style task found with or without analogic approach.  -- Found by who or what?
Also, how representative is this of the overall results? 

"Our first stage converts a raw GIF into a structured, readable summary that captures the analogy-laden visual logic hidden in the clip." -- what do you mean by "analogy-laden visual logic"?  What's an example?  When I look at these examples in the appendix, I don't see a lot of what I would consider to be analogies.  

"The final stage transforms each task sketch produced in Step 2 into a fully executable ARC-style task."  What does "fully executable" mean here?  Are these tasks actually solvable in some human-like way?

Section 4.2 "Next, GPT 4.1-mini with full description and GPT 4.1-mini with analogy-removed description were instructed to find the analogy implied in the 12 ARC-style tasks."  Is there a prompt given in the appendix for this? And which 12 ARC-style tasks are you talking about?

"We first instructed o3-mini to evaluate semantic similarities by providing a one-shot guideline about measuring similarity." -- Please explain what you did in more detail. 

Figure 4 is hard to understand.  What is being measured in the "Human", "Full Description", and "Analogy-Removed description bars? These are similarities between what and what?  What do the actual values mean?

"we embedded each outputs and compared the cosine similarity between" -- how did you do the embeddings? 


The authors state, "We have now confirmed that the components are analogically meaningful." but how exactly has this been confirmed, and what does "analogically meaningful" mean?

Similarly, the authors state, "we confirmed that GIFARC provides analogical reasoning  guidance to models".  Again, this doesn't seem to be confirmed by the results given in the paper.  Is there a way to show this more quantitatively? 

"NLL values indicate a higher likelihood of the model generating the correct answer" -- Do you mean "lower  NLL values" indicate this?

"the substantial NLL improvements across both evaluation sets demonstrate that the model is learning meaningful patterns and moving closer to correct solutions." -- How do you verify this?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes a synthetic data generation pipeline for ARC tasks that extracts analogy concepts from online GIFs. The dataset is used to fine-tune an LLM, showing limited improvement on ARC-AGI-1.

### Strengths
- The idea to use GIFs as a source of analogical transformations is interesting, and intuitively makes sense given that GIFs often contain visual motion, which is one of the important priors in ARC tasks.
- Synthetic data generation is a promising framework for improving abstract reasoning in LLMs.
- The fine-tuned model is evaluated on both ARC-AGI-1 and ARC-AGI-2.

### Weaknesses
- The primary weakness is that fine-tuning on the GIFARC dataset does not yield significant improvements. On ARC-AGI-1, the fine-tuning only improves performance from 0.2% to 2.9%, which is a very small improvement, and very poor performance both before and after fine-tuning. On ARC-AGI-2, performance is 0% both before and after fine-tuning. These results do not suggest that the dataset is successful at improving performance on ARC tasks.
- The LLM-evaluated similarity results shown in Figure 3 are not informative. First, the use of an LLM to evaluate similarity is not reliable. Second, similarity is not a useful metric here. It is possible for two descriptions to be similar (i.e., involve similar words of features) despite one being correct and one being completely wrong.
- There is not much discussion of where the GIFs that are used in data generation come from, and what kind of images they depict. Intuitively, it is not clear whether randomly sourced GIFs would actually be helpful in ARC tasks, so it would be good to include more discussion of the sorts of GIFs that are used and why they should be helpful for generating ARC tasks. Additionally, there is no validation that the generated tasks have anything to do with the GIFs that form the input to the pipeline. Do the generated tasks actually employ similar concepts as the input GIFs? 
- There is no ablation provided for the specific importance of using GIFs. For instance, there should be a comparison with models fine-tuned on synthetic data generated using alternative pipelines, for instance using actual ARC tasks as input, or images instead of GIFs. The current results do not establish that GIFs are specifically useful above and beyond the potential benefit of synthetic data.
- The description of the state-of-the-art in ARC results is somewhat misleading. Although the best performing model that was officially entered in the ARC-AGI-1 competition achieved only 55% accuracy, other models that were not officially entered into the competition (due to using computational resources beyond those allowed in the competition) achieved performance very close to average human performance, most notably openAI's o3 model. ARC-AGI-2 results were worse, but overall the performance of state-of-the-art models is not as bad as is implied in the introduction.

### Questions
- Do the synthetically generated tasks meaningfully incorporate visual transformations extracted from the GIFs?
- How does the model perform when fine-tuned on data generated using alternative pipelines, including using ARC tasks or images as input?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces GIFARC -- an ARC-style dataset generated by leveraging GIF images to extract analogies, and then construct ARC-like problems with corresponding verbal descriptions. The authors run a number of experiments to validate the quality of their dataset.

### Strengths
-The authors investigate a highly relevant problem. The struggles of modern models with ARC-style tasks is indicative of a substantial gap in modern AI architectures or training approaches.
-The idea to extract visual analogies from GIF sources is original and promising.

### Weaknesses
While the idea of the paper is quite promising, I believe that there are very substantial flaws with experimental evaluation.

Some crucial aspects of evaluation are missing. 

For example, it's crucial to check that the generated problems are human-solvable. The provided human evaluations are insufficient. Not only the sample is very small (three experts and 12 problems), but also analogy description is very different from solvability. 

For example, while a human can, perhaps with some difficulty, find an analogy for the generated ARC-like problem shown on Figure 16, I believe nobody can be reasonably expected to be able to solve that task which essentially displays a medium-resolution pixel art of a flower/circular pattern with some nontrivial color transformation.

Many other tasks similarly appear either quite hard to solve or are, in my view, confusing. It's important to provide a reasonable human performance estimate for the dataset. In this scenario, the tasks need to be sampled at random.


Another issue is the abundance of unusual experimental design choices and lack of detailed procedure description.

For example, 4.1 (i) mentions that the tasks were selected after iterative refinement with LLM and human. But that makes these seected tasks non-representative of the general dataset which can not be refined in the same way (as that would be prohibitively time-consuming).

A related issue is the loose use of terms like "analogy" and even "solution", and lack of criteria for what constitutes a good analogy/solution. Figure 3 illustrates this issue quite well. In the top panel, there is indeed an analogy-like phrase ("tidying up"), but that analogy is not quite relevant to the task. And, at least in the verbal description, I don't see the correct solution. (the solution, I believe is that different shapes fall to the right or to the left in tetris-like fashion, depending on the border color).

I believe that listing every such issue will be counterproductive. I believe that the authors have an interesting idea, but before diving into subtle and challenging evaluations of analogical relevance, it's crucial to first provide simple experiments with human participants to demonstrate that the resulting tasks are human-solvable.

Additionally, a simple way to test the quality of the generated analogies/solutions is to give them to humans and to ask to apply them to the task. To show that the analogies/solutions are complete/sufficient, humans should be able to solve the test example without seeing the training arc demonstrations. To show that the analogies are at least helpful, one can compare human performance on these tasks with and without the analogies. This is similar to some of the experiments shown in the paper, but the crucial thing is that the focus should be on task solving performance (not analogy identification) and it should be, in my view, on human participants.

Overall, unfortunately, at present, I can not recommend this paper to acceptance. But I hope that the authors refine their approach and resubmit the paper in the future.

### Questions
What was the reason for selecting specifically 3 experts and 12 problems?

In the appendix D, it's stated that the experts were shown 12 tasks. But in the provided prompt given to the experts, it's said that the number of tasks is 13. What is the cause for this discrepancy?

In 4.1. (i), there's a mention of iterative refinement in task selection. How were these tasks selected and what was involved in this refinement process?

### Soundness
1

### Presentation
3

### Contribution
1
