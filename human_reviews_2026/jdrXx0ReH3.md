# In-Context Learning for Esoteric Programming Languages: Evaluating and Enhancing LLM Reasoning Without Fine-Tuning

- Avg Score: 2.50
- Decision: Reject
- Scores: 0, 6, 4, 0

## Abstract
Large Language Models (LLMs) have revolutionized mainstream software development, yet their ability to generalize to esoteric languages — who may have small or no representation in the training corpus —remains poor. Programming in esoteric languages tests a model’s capacity to infer novel grammar and leverage nontrivial reasoning capabilities in utilizing the documentation. To quantify these effects, we evaluate both open and closed-source LLMs on code generation and language identification tasks across four esoteric languages—Minipy, Pyth, Rhokell, and 0815—and compare traditional prompt‐based methods to agentic coding IDEs.   Our findings reveal that LLMs can now generate some correct code in these languages when provided with documentation and sparse examples; however, performance remains far below that of similar models in common programming languages.

Furthermore, we introduce a novel in-context augmentation strategy in which LLMs first generate solutions, which are then manually verified and re-inserted as examples into subsequent prompts. Our results indicate that strategically embedding just a few analogous problems can yield large accuracy improvements without any model retraining. Our findings show that this ``self-scaffolding'' approach can boost performance on coding benchmarks: inserting Deepseek’s verified EsoEval solutions raised EsoEval accuracy on Pyth from 16.67\% to 30.82 \%, while HumanEval accuracy on Minipy jumped from 51\% to 65\%. We offer this as a flexible alternative to costly fine-tuning, paving the way for rapid adaptation of LLMs to highly specialized, emerging, or other low data domains.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper evaluates LLMs on 4 esoteric programming languages (Minipy, Pyth, Rhokell, and 0815). The authors get their tests from HumanEval and use them to generate a new benchmark, EsoEval. The authors propose a method of using self-scaffolding to generate solutions, manually verify them, and then re-use the correct ones as in-context demonstrations. This results in sizable gains in some settings. Another finding is that higher compilability (code that compiles) is a poor proxy for correctness in esolangs. Lastly, without penalty as the authors do acknowledge it, this paper was written by an LLM.

### Strengths
* This paper attempts to tackle an important problem that is in a popular research area
* The proposed method of self-scaffolding is simple and actionable
* The evaluations go over 4 different languages based on two source benchmarks and provide insights that hold over almost everything

### Weaknesses
* The rationale behind the selection of their languages is missing. The authors arbitrarily selected 4 languages. One of their main arguments is that we should look for languages less popular than Python (figure 2) yet two of the languages they chose (Minipy and Pyth) are related to python. In fact, the relationship between Minipy and python is so strong that on line 355 they say, “any submission that successfully ran under a standard Python interpreter were excluded, regardless of functional accuracy” so arguably Minipy is not an esoteric language. Why not a language like Isabelle or Scenic? 
* In the evaluations, the authors refer to accuracy but it is never formally defined
* Both the benchmark, EsoEval, and the self-scaffolding procedure require a human in the loop. The authors do not provide a rubric or set of rules for both procedures so both are not reproducible nor can one validate the merits of the benchmark.
* Outside of websites, the authors reference a total of 11 other works. They fail to acknowledge the rich literature around Domain Specific Language (DSL) code generation. 
* This manual self-scaffolding procedure is erringly similar to HyDE by Gao et al and so there's a lack of novelty. The applications are different, code vs information retrieval respectively, but the idea of first guessing something closer to what you want then refining is more or less the same.

### Questions
* In self-scaffolding, how many verified examples were added, by what selection heuristic, and in what order? Any diminishing returns curves? Sensitivity to noisy/incorrect examples? Since none of this was covered I think it would have been easier to just use DSPy which you did call out as future work but doesn't excuse the lack of details.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors assess the capabilities of large language models for code generation on a set of esoteric programing languages, specifically Pyth, Minipy, Rhokell, and 0815. The paper demonstrates an in-context learning approach for improving LLM performance on these models, of which there is limited training data available (especially relative to popular languages like python). Specifically, the paper presents a self-scaffolding approach where model-generated solutions are manually verified and then re-inserted as examples back into subsequent prompts. Their approach demonstrates strong empirical results with significant gains in performance on these languages. While esoteric languages may not have a significant impact in the real-world due to their limited use, the creation of a benchmark around these languages is significant and may help represent coding problems that require strong generalization and have limited risk of contamination.

### Strengths
- A robust evaluation framework where generated code is run and tested for expected outputs based on the HumanEval cases.
- Some surprising findings, specifically: "Models will occasionally learn just enough grammar to compile correctly—matching parentheses, using valid tokens and whatnot, yet still produce algorithms that don’t solve the target problem." I appreciate that the authors identified this unexpected behavior and provided additional context to make sense of this phenomenon. Conversely, the connection between language obscurity and performance is unsurprising but an interesting tidbit to include in the paper — it certainly highlights gaps in LLM knowledge (which are far too often assessed primarily on the most popular languages: python, java, c++). So this is a good test of broader generalization among LLMs.
- EsoEval benchmark: A simplified 100-problem benchmark designed to be more tractable than HumanEval is a significant contribution to the community.
- The paper is very original and well written with a clear problem and approach.

### Weaknesses
- In the most esoteric languages the perform can be very low, even with the presented approach. Hence it can be difficult to assess signficance, when the differences may just mean a small handful of samples are correct. Nonetheless, it doesn't appear that the approach ever results in worse performance.
- Missing citation: ["Multi-Lingual Evaluation of Code Generation Models"](https://arxiv.org/pdf/2210.14868) is a well cited paper on converting monolingual datasets (e.g., MBPP) to multilingual code. The approach for generating new benchmarks is relevant.
- Statistical significance: No confidence intervals, significance tests, or variance measurements. This is especially concerning for single run inference evaluations on LLMs.

### Questions
1. How were the 100 problems selected? What's the difficulty distribution?
2. Was it a decision to select problems that would result in 100% accuracy with gpt-4o-mini on Python? Or was this by happenstance? 
3. How does the self-scaffolding approach compare to standard prompting?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper centers on evaluating LLMs on code generation and related tasks in esoteric languages, and on improving performance through an in-context augmentation strategy.
It assesses Minipy, Pyth, Rhokell, and 0815 on code generation and language identification, comparing traditional prompt-based methods with agentic coding IDE workflows.
The evaluated systems include GPT-4o, GPT-4o-mini, Llama-3.3-70B-Instruct-Turbo, and DeepSeek V3, and the study also examines IDEs such as Codeium’s Windsurf.
The benchmarks comprise HumanEval and a simplified benchmark introduced by the authors called EsoEval.
Methodologically, the paper presents a “self-scaffolding” procedure in which the model first generates solutions, humans manually verify correctness, and the verified examples are re-inserted into subsequent prompts as in-context augmentation.
The experimental setup supplies official documentation and sparse examples in prompts, extracts generated code, executes it with each esoteric language’s interpreter, and determines correctness using input–output tests derived from benchmark test cases.
The findings show that documentation and sparse examples enable some correct generations, yet performance remains far below that of similar models in common programming languages.
With self-scaffolding, inserting a small number of verified examples yields notable gains, raising EsoEval accuracy on Pyth from 16.67% to 30.82% and increasing HumanEval accuracy on Minipy from 51% to 65%.
As an observation on standard Python benchmarks, nearly all generated solutions that compiled also passed all provided unit tests for EsoEval, indicating that compilation success is a robust proxy for functional correctness in that setting.
Taken together, the results position self-scaffolding as a flexible alternative to costly fine-tuning, enabling rapid adaptation of LLMs to highly specialized, emerging, or other low-data domains.

### Strengths
* Importance of the Topic and Reader Interest  
This work contributes to a highly active area by examining the adaptability of LLM code generation under challenging conditions involving unseen, low-resource, and nonconventional grammars. While esoteric languages are niche, they serve as effective stress tests for documentation-driven generalization to unknown specifications and for probing model reasoning and consistency limits. The study is tightly connected to readers’ concerns about rapid adaptation to unknown environments and about sound evaluation design, which is a notable strength.  
  
----


* Novelty of the Main Contribution  
Whereas prior multilingual code benchmarks have focused on mainstream languages and largely static evaluation setups, this paper targets esoteric languages and foregrounds a lightweight, rapid adaptation strategy, "self-scaffolding," which reinserts manually verified solutions into subsequent prompts. Positioning iterative in-context learning as a primary subject of evaluation clearly differentiates the work, and the no-additional-training, practice-oriented improvement path carries strong originality.

---

* Importance of New Insights   
The study quantitatively shows that supplying documentation and a few examples yields some correct solutions, and that reinserting verified examples can produce practical accuracy gains. These insights are easy to reproduce in real-world settings and provide actionable guidance for bootstrapping low-data or newly emerging languages through prompt design. The implications are concrete and accessible for both researchers and practitioners, which enhances the value of the findings.

### Weaknesses
* 1. Limited Scope of Coverage  
The study confines its evaluation to four esoteric languages (Minipy, Pyth, Rhokell, 0815), and the model comparison mainly spans two parameter scales of the same family (GPT-4o and GPT-4o-mini) plus a handful of open-source models, which limits generalizability. The authors explicitly note the need to explore broader model sizes, architectures, and pretraining corpora, indicating that the current conclusions may have a restricted applicability range.

---
* 2. Dependence on Manual Verification  
The proposed self-scaffolding relies on manually verifying model-generated solutions and reinserting the verified examples into subsequent prompts. While appropriate for ensuring correctness, this design can become a bottleneck in terms of human cost, scalability, and reproducibility. The authors acknowledge that automating the verification would be desirable, and a fully closed-loop automation is not yet in place.

---
* 3. Lack of Optimization for Example Selection and Ordering  
Although inserting a few verified examples into the context improves performance, the study does not optimize which examples to include or how to order them. The paper suggests that curation and curriculum design could substantially affect outcomes, implying that the current gains may depend on human design choices. 

---
* 4. Benchmark Coverage and Difficulty  
In addition to HumanEval, the evaluation uses the authors’ simplified benchmark EsoEval (100 problems), and the paper states that the tasks are relatively simple. This offers transparency and practicality but may not fully capture the diversity and complexity of real esoteric-language usage, so external validity and robustness will require further expansion.

---
* 5. Improvements Concentrated in Specific Combinations  
The reported quantitative gains from self-scaffolding are shown on specific language-benchmark combinations, such as Pyth on EsoEval improving from 16.67% to 30.82% and Minipy on HumanEval improving from 51% to 65%. While these are significant indicators, the paper itself positions broader transferability and consistency across other esolangs and task sets as future work, so generalization of the effect needs additional evidence.

---
* 6. Reproducibility and Transparency of Compute/Cost  
Execution-based evaluation supports reproducibility, but the method’s reliance on manual verification means that transparency about procedures, prompts, verification criteria, and human effort critically affects replicability. Without well-structured reporting of inference tokens, time, and human-hours, rigorous re-experiments and cost estimation by other groups become difficult, leaving open issues in practical reproducibility and cost transparency.
---
I am willing to update the scores and evaluations when the authors have appropriately addressed my concerns and questions.

### Questions
* The selection and ordering of examples are underexplored  
While “a few well-chosen examples” help, the paper lacks ablations on which exemplars to include and how to order them for maximal effect, limiting prescriptive guidance.
Is there anything the authors could discuss regarding this point?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper evaluates GPT-4o and Llama's ability in 4 esoteric programming languages, and observe that in-context learning can improve the performance.

### Strengths
The paper focuses on the fundamental common sense abilities of LLMs. The writing is clear and accessible, making it easy to understand even for readers without prior knowledge of LLMs.

### Weaknesses
This is probably a good course project for an introductory LLM class, but it is probably not at the level required for an ICLR submission.

1. They simple just try in-context learning on some tasks about esoteric programming languages. There are not any new things other than the standard in-context learning.

2. The experiments are limited in scale, covering only GPT-4o and Llama on approximately 100 tasks. The benchmark mainly replicates HumanEval.

3. All the figures are of low quality, appearing to be simple screenshots. It’s quite surprising that the first two figures focus only on the frequency statistics of the selected programming languages...... The figure layouts look unprofessional, with awkward design and excessive blank space.

### Questions
Are there any really novel concepts introduced in this work?

### Soundness
1

### Presentation
1

### Contribution
1
