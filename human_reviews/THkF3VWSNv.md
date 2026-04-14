# REDO: Execution-Free Runtime Error Detection for Coding Agents

- Decision: Reject
- Scores: 5, 3, 3, 3

## Abstract
As LLM-based agents exhibit exceptional capabilities in addressing complex problems, there is a growing focus on developing coding agents to tackle increasingly sophisticated tasks. Despite their promising performance, these coding agents often produce programs or modifications that contain runtime errors, which can cause code failures and are difficult for static analysis tools to detect. Enhancing the ability of coding agents to statically identify such errors could significantly improve their overall performance. In this work, we introduce Execution-free Runtime Error Detection for COding Agents (REDO), a method that integrates LLMs with static analysis tools to detect runtime errors for coding agents, without code execution. Additionally, we propose a benchmark task, SWE-Bench-Error-Detection (SWEDE), based on SWE-Bench (lite), to evaluate error detection in repository-level problems with complex external dependencies. Finally, through both quantitative and qualitative analyses across various error detection tasks, we demonstrate that REDO outperforms current state-of-the-art methods by achieving a 11.0% higher accuracy and 9.1% higher weighted F1 score; and provide insights into the advantages of incorporating LLMs for error detection.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents REDO, a novel approach at predicting runtime errors in Python code patches by first querying traditional static analysis (SA) tools and extending them with LLM predictions if no error is detected. They compare the method with pure SA and pure LLM methods and one recent work on two datasets, one based on real world GitHub projects and one based on standalone scripts, and conclude that such methods are superior based on obtained F1 scores.

### Strengths
- The story is coherent and reads in an engaging way and mostly well-written and properly formatted.
- The figures and tables are well sized and easy to read and interpret.
- The proposed method is interesting in that it combines traditional conservative and novel more sensitive tooling, to obtain the best of both worlds.

### Weaknesses
I think the paper presents an interesting idea, however I am not convinced that the claimed SOTA results have merit in the sense that it is claimed in the paper, therefore, without further changes I lean towards reject. Based on my current understanding addressing the following points would be sufficient to change my recommendation towards accept.

1) Lack of precision in assessing method prediction in SWEDE

The benchmark SWEDE considers all error types as one for its evaluation, considering coinciding "method predicts some error" and "some error occurred" as a successful prediction (Sec. 4). I think this score is significantly less interesting than the score used by Bieber et. al (2022) in the previous work, i.e. to consider each error _type_ as a separate class, for several reasons. 

- The purpose of error prediction is the potential use of the predicted information for subsequent repair of suggested fixes. The more precise such a prediction the more useful it is to whichever tool (i.e. LLM) is to implement such a fix. Predicting an AttributeError for a non-existing method when the actual error is completely unrelated may be unhelpful at best or confusing at worst for subsequent processing. Ideally one would even want to correctly predict the cause of the error, though this is admittedly difficult to assess in an automatic fashion. 
- The previous work, which this work compares to, uses exactly such a metric and a comparison might be unfair or misleading.
- Traditional, rule-based static analysis tools usually predict errors (and their sources) precisely. Related guarantees are often important for applications (i.e. see above) and are already violated by extending results with LLM predictions. It would be interesting to see how precise LLMs are, in order to obtain insight into whether further research in this direction will be required.


I suggest for these reason that either 
- the authors provide such a more detailed score for SWEDE or
- the authors present a compelling experiment to investigate whether the predicted errors are already helpful for repair (i.e. they could also still be helpful because they rarely point to entirely unrelated errors, even when not exactly matching)


2) No attempt at generalization of the proposed method

The authors compare the combined performance of static analysis tools and LLMs with preceding work that trains a tailored model on error prediction. Since the proposed method of combining i) precise but conservative static analysis with  ii) an imprecise but more sensitive learned model, is very general, I am wondering a) if it is not possible to combine the preceding work by Bieber et. al. with static analysis in such a way and b) what the performance of such an approach would be.

Consequently it is also not clear whether the improvement in the results stems from the involvement of LLMs or from the clever combination with static analysis tools. A more rigorous analysis of the involved factors would be interesting.

3) Presentation

Overall, the paper could benefit from a number of improvements in the general presentation. Below I present a list of smaller concerns regarding the form:

- In Figure 1 the chart is zoomed in a lot, to highlight results varying around 12% (min-max plotted value). Since the percentages are rather small I would like to see error bars that help assessing the significance of the results.
- Section 4 analyses external dependencies of python files in SWE-Bench "when only one directory level above the location where the modified files resides" is considered. It is not very clear to me what this means and even less so why the dependencies are not calculated for the entire repository (instead of vaguely mentioning that it "could become even more intimidating").
- Algorithm 1 is referenced in Sec. 3.2. Since it was not mentioned that the algorithm is in the appendix, it was difficult to find. I recommend either including the algorithm in the main body (which might also help general understanding or mentioning this) or explicitly referring to "Algorithm 1 in the Appendix"
- I don’t understand why the bolded text in the "STA dataset" section of Sec. 5.1 is bolded and which "different conclusions" it is referring to.
- Sec 5.1. is generally a large swath of text that might benefit from refactoring (i.e. moving details to the appendix and highlighting the main points more easily digestible)
- Table 3 would better fit on page 7 together with the corresponding dataset paragraph
- (A shortened version of) the analyzed patch and code for figures 4 and 5 would aid its understanding. The respective evaluation log could correspondingly be shortened.
- Appendix Section B is apparently completely empty.
- Row spacing in several tables is quite tight, resulting in intersections between presented numbers, underlines and table lines.
- Somehow no recall is provided for the CodeNet dataset and overall very different types of numbers are presented while the datasets do not fundamentally seem different. Similar metrics would IMO aid understanding here.

### Questions
- (Referring to W1) Can you provide a more detailed score on how precise LLM predictions are when differentiating between error types? Or alternatively provide some insight on why such differentiation is not needed?
- (Referring to W2) Could you provide some insight on the possibility and impact of combining static analysis with the method proposed by Bieber et. al.?
- What are the error margins in Figure 1?
- What does the bolded text in "STA dataset" refer to (see Nitpicks)?
- What is the expected impact of applying the proposed method in languages with stronger type safety guarantees? Or put differently, which kind of errors is most reliably predicted by static analysis tools and LLMs respectively and what is the expected impact when errors caught by type systems are covered by the compiler?
- Is there a reason for the different metrics of CodeNet and SWEDE and can the assessments be unified in their metrics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces REDO, a framework for error detection with large language modes (LLMs) without executing code. REDO combines static analysis with LLM-based detection to capture errors.

- The static analysis tools (e.g., PyRight) detects runtime errors by comparing the error profile of the original code with that of the modified code. By focusing on new errors introduced by modifications, this approach filters out false positives related to the original code and flags modifications that introduce potential runtime issues, such as SyntaxErrors or AttributeErrors.

- For instances where no errors are found through static analysis, an LLM (like Claude) is prompted with the problem context, original code, and modified patch to reason about runtime errors that might depend on specific inputs or complex conditions. This step helps detect context-sensitive errors, such as TypeErrors, that static analysis might miss, improving error detection without executing the code.


Additionally, the paper presents the SWEDE benchmark, an extension of SWE-Bench, to evaluate error detection capabilities in repository-level tasks with complex dependencies. 

In the experiments, REDO demonstrates improved accuracy and F1 scores compared to baseline methods, but it shows a reduction in precision.

### Strengths
- Originality

REDO combines static analysis with LLMs, and although similar approaches are becoming more common, this implementation remains useful.

- Quality 

The empirical evaluation of REDO  provides meaningful insights into the effectiveness of the proposed approach.


- Clarity 

Overall, the paper is not difficult to follow.

- Significance 

REDO addresses an important issue by ensuring program code security with LLMs.

### Weaknesses
- Overclaim

This paper claims to focus on error detection for coding agents; however, the REDO approach appears to be a more broadly applicable method for general program code error detection, which creates a misalignment in the work's focus.

- Limited Novelty

REDO combines PyRight with LLMs in a relatively straightforward way, with the LLM engaged only when PyRight deems the code safe. There seems to be no interaction or integration between the static analysis tool and the LLM beyond this sequential setup. In Figure 2, while an arrow suggests information flow from PyRight to the LLM, it is unclear what, if any, information is being transferred. Furthermore, other works also use LLMs alongside static analysis and various other tools, further undermining the novelty of REDO's design.

- Inconclusive Performance Improvement

The LLM in REDO is intended to catch potential errors that PyRight might miss; however, LLMs can sometimes mistakenly flag safe code as erroneous, thereby reducing precision. This limitation is reflected in REDO’s lower precision scores in the experiments, indicating that the addition of LLMs does introduce new issues, instead of consistently improve error detection performance.

### Questions
1. Could the authors clarify whether REDO is uniquely designed for coding agents or applicable to general program code error detection?

2. What specific information, if any, is transferred between PyRight and the LLM as indicated by the arrow in Figure 2?

3. Given that LLMs may introduce false positives, what measures are taken to control for this, especially when PyRight has deemed code safe?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies how to apply LLM to runtime error detection of source codes that are generated by coding agents. The authors employ a hybrid approach in which a static analysis approach collaboratively works with LLM to detect runtime errors. Specifically, PyRight and Pyflakes are used to detect syntactical errors such as SyntaxError, NameError, while LLMs are used to detect runtime errors in codes. The authors claim that their hybrid approach is execution-free in that no execution is needed to detect such errors. A prototype is implemented and evaluated on a self-designed benchmark SWEDE. The experimental results show the outperformance of the proposed approach REDO over other related techniques.

### Strengths
1. important problem. Coding agents face a very big problem of generating buggy and unreadable codes.  Automatically generated codes are difficult to debug manually. It is important to find automatic ways of analyzing LLM-generated codes. The authors provide an alternative solution in this direction. The proposed hybrid approach is reasonable and intuitive. Undoubtedly, LLMs can help to some extent find bugs in programs and could be leveraged with other algorithm-based approaches such as static analysis. This direction deserves further investigation. 


2. a clear and simple solution. The proposed approach is clear and easy to understand. The mechanism is simple and easy to implement. That might can be used to solve practical problems, as shown in the experiments. Errors can be found efficiently. As an auxiliary functionality, the approach could be integrated like grammar checkers or syntactic checkers for coding agents, providing useful suggestions and alerts to programmers to be careful of potential bugs that are detected.

### Weaknesses
1. The capability of detecting bugs purely relies on backend LLMs and static analysis tools. I didn't see how the two approaches can be deeply integrated and collaborated. Apparently, the simple loose coupling of static analysis tools and LLMs cannot find all bugs in codes. Actually, no method exists to find all bugs. It would be very important to investigate collaborative ways of leveraging the advantages of tools and LLMs to find bugs or errors as many as possible. The current integration approach is very preliminary and shall be further explored. 

2. The use of static analysis tools is quite limited. The authors emphasize that their approach is execution-free. I didn't see why we have to consider execution-free. Are there some specific reasons or scenarios we have performance execution-free analysis? In the paper, the authors mentioned that static analysis cannot find input-dependent errors (or struggle to find them in their words). That is not the truth. There are various static analysis approaches to detect input-dependent errors such as symbolic execution, conclic testing. Why not apply these sophisticated approaches to LLM-generated codes? 

3. Technical contributions are limited. This point might be related to the above two issues. Throughout the paper, I didn't see any technical contributions, which I think is important to a research paper. To me, the improvement by REDO is not surprising, when comparing it with the purely static analysis tool PyRight. LLMs are extraordinarily powerful.  However, imposing LLMs on PyRight only gives very limited improvement (See Table 3). I believe LLMs can do more for this task if the collaboration between the tool and LLM can be tighter.

### Questions
1. What are the benefits of being execution-free? Static analysis is not necessarily execution-free. For instance, symbolic execution is a very powerful static analysis technique. It can be used to detect input-dependent errors. 

2. Can LLMs work more tightly with static analysis tools for this task, for instance, LLM-based static analysis or static analysis for LLMs? 

3. What are the key challenges that are solved by the proposed approach? Dubbing LLM-generated codes is generally a very challenging task. It could be valuable to identify what these challenges are and how they could be solved separately.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces REDO which is a system that attempts to detect runtime errors in code modifications without execution by combining static analysis tools with LLMs.

### Strengths
I think the paper and the overarching idea of using existing program analysis tools in conjunction with LLMs within agents/systems is quite timely. Several recent breakthroughs in tasks such as SWEBench have been made by designing and integrating several classical tools such as linters, type checkers, repository dependency graphs, etc. I also found the data recipe to create the SWEDE benchmark from existing agent failures to be quite interesting.

### Weaknesses
**Motivations unclear**. The paper starts of with SWEBench and software agents as the motivation. However the task suddenly switches to simply predicting execution errors. The paper never circles back on how this prediction could actually help agent/systems that generate code. This makes the overall motivation of the paper quite weak in my opinion. While it is clear that setting up execution-environments for real-world code is challenging and an execution-free analyzer helps, the paper does not show any results that motivate that hypothesis.

**Execution feedback will almost always be better than REDO**. Adding to the point above, assume you have an execution environment for a repository. For most cases, the error messages from execution would contain a lot more information (traceback, type of error, etc.) than the assessment that REDO can provide. Furthermore you can even use an LLM to analyze the failed execution trace and code to produce even better feedback for agents. In that case, I'm wondering if one even needs a system like REDO if the motivation is to integrate with software agents.

**SWEDE an afterthought?**. I am not sure of the contributions of SWEDE as a benchmark as currently presented. While letting models figure out execution-based errors before execution takes place seems quite important and powerful, the current analysis or results do not explore this at all. Further more the benchmark reduces complex error detection and root cause analysis to binary classification. With that it seems like SWEDE is a means to evaluate REDO but it's unclear the value beyond that. There's potential for future versions of SWEDE to be a more comprehensive benchmark around predicting execution results that tap into whether or not LLMs truly understand code.

### Questions
Que: Could the authors clarify the motivations of the paper beyond integration with agents, if any? That would maybe help view the current set of results with more context.

### Soundness
3

### Presentation
3

### Contribution
1
