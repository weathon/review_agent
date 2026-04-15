# Interpretable Table Question Answering via Plans of Atomic Table Transformations

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6, 5

## Abstract
Interpretability for Table Question Answering (Table QA) is critical, particularly in high-stakes domains like finance or healthcare.
While recent Large Language Models (LLMs) have improved the accuracy of Table QA models, their explanations for how answers are derived may not be transparent, hindering user ability to trust, explain, and debug predicted answers, especially on complex queries.
We introduce Plan-of-SQLs (POS), a novel method specifically crafted to enhance interpretability by decomposing a query into simpler sub-queries that are sequentially translated into SQL commands to generate the final answer.
Unlike existing approaches, 
POS offers full transparency in Table QA by ensuring that every transformation of the table is traceable, allowing users to follow the reasoning process step-by-step.
Via subjective and objective evaluations, we show that POS explanations significantly improve interpretability, enabling both human and LLM judges to predict model responses with 93.00% and 85.25% accuracy, respectively.
POS explanations also consistently rank highest in clarity, coherence, and helpfulness compared to state-of-the-art Table QA methods such as Chain-of-Table and DATER.
Furthermore, POS demonstrates high accuracy on Table QA benchmarks (78.31% on TabFact and 54.80% on WikiTQ with GPT3.5), outperforming methods that rely solely on LLMs or programs for table transformations, while remaining competitive with hybrid approaches that often trade off interpretability for accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces Plan-of-SQLs (POS), a method for interpretable Table Question Answering that decomposes queries into simple, sequential SQL steps, ensuring full transparency in the reasoning process. POS enables users to trace how answers are derived, significantly improving interpretability over existing models. Through human and LLM evaluations, POS demonstrates strong clarity, coherence, and competitive accuracy on benchmarks like TabFact and WikiTQ, making it suitable for high-stakes applications demanding clear model explanations.

### Strengths
1. The paper is well-motivated, addressing the critical need for interpretability in Table QA, especially for high-stakes fields where transparency is essential.
2. It includes a comprehensive evaluation, using both human and LLM judges to assess interpretability and predictive accuracy, showcasing POS’s advantages in clarity, coherence, and overall effectiveness compared to existing methods.

### Weaknesses
1. **Limited Novelty**: While the paper addresses an important problem in Table QA, the core ideas, such as breaking down queries into atomic steps and using SQL transformations for interpretability, are established in other domains. This reduces the method's originality, as it largely adapts existing decomposition and programmatic interpretability approaches rather than introducing fundamentally new techniques specific to Table QA.

2. **Limited Effectiveness and Scalability**: The performance improvements with POS are incremental, indicating only a modest boost in interpretability and accuracy over baseline methods. Additionally, the approach relies on SQL-based transformations, which may restrict its applicability in more dynamic or complex data environments where SQL alone may be insufficient. As a result, the method lacks clear potential for further application in other QA scenarios or for handling more complex reasoning tasks, limiting its scalability and broader impact.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new in-context learning method for solving table-based QA, Plan-of-SQLs (PoS). The method first generates a sequence of atomic actions as the "plan" and translates the actions in the plan to simple SQL queries step by step. The final answer would be the result of the sequence of the translated SQL queries.

The author emphasizes the interpretability of the method by arguing all actions are taken through SQL queries and the output is the result of the SQL queries.

In terms of performance, the proposed method achieves comparable performance compared with baselines while is much better in interpretability.

### Strengths
The proposed method PoS learned many good merits from existing methods, including the plan generation, and step-by-step execution. It also proposed a novel point about the interpretability of Table QA. The explanation generation method is also novel where it highlights the related col/row/cells in the queries. This method is uniquely designed for Table QA.

### Weaknesses
1. Since this is an in-context learning method, the performance is greatly influenced by the choice of LLM backbones. The paper only uses gpt-3.5-turbo-16k-0613 which has already been deprecated by OpenAI. I recommend the authors choose more variety of models and use newer models. This would make the results more reliable and reproducible.

2. The XAI comparison is not very convincing. The highlight method is preferable for SQL-based methods such as PoS or Text-to-SQL. However, it is not explicit for other baselines without SQL, such as Dater and CoTable. 

3. The authors argue that it is better for interpretability to merely depend on SQL to process the table and answer the question. However, the necessity of this constraint is still questionable for the table qa in practical scenarios. The tables are of various formats, and the answer usually cannot be produced by a SQL query. Such limitation may explain the lower performance of PoS compared with many other baselines.

4. The novelty of the method is limited. Since the step-by-step execution has been already proposed in CoTable, this work seems a simple extension of CoTable with SQL queries and plan generation. And the final performance is even reduced with the added components.

5. The paper's presentation requires improvement, as some tables extend beyond the page margins.

### Questions
1. Are there any cases where the answer cannot be generated by SQL queries? 

2. What is the motivation to add highlights in the XAI experiments? Did you try to remove them in the experiments and see the results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Plan-of-SQLs (POS), an interpretable Table Question Answering (Table QA) method that enhances model transparency by decomposing complex queries into atomic natural-language sub-queries. Each sub-query is sequentially converted into SQL commands, allowing the input table to be transformed step-by-step until the final answer is obtained. This decomposition approach ensures that every transformation is clear and traceable, providing a fully interpretable reasoning process. POS achieves competitive performance on standard Table QA benchmarks (TabFact and WikiTQ) while significantly improving clarity, coherence, and helpfulness compared to existing models such as CoTable and DATER.

### Strengths
The paper is well-structured and written, with a logical flow that makes the methodology and experiments easy to follow. The proposed POS advances interpretability in Table QA by decomposing complex queries into atomic, natural-language sub-queries sequentially converted into SQL commands. The authors support their interpretability claims with comprehensive experiments, including both human and LLM evaluations, showcasing the clarity, coherence, and helpfulness of POS explanations.

### Weaknesses
While POS enhances interpretability, its innovation appears incremental, with its main contribution being the decomposition of queries into SQL-translatable atomic steps. Furthermore, POS shows limited accuracy improvements over existing models, which could diminish its attractiveness in applications where interpretability is less prioritized. Another concern is POS’s reliance on sequential processing via the NL Atomic Planner, where each plan requires the previous step's results, including intermediate table states. This dependence on continuous LLM calls may lead to inefficiencies, particularly for complex or large tables, as each step adds computational overhead. Detailed efficiency comparisons with non-sequential methods or potential optimizations would strengthen the paper and address this limitation.

### Questions
1.	Given POS’s sequential nature and reliance on intermediate tables as inputs for each step, what are the efficiency implications?
2.	In Appendix C, the ablation study discusses changes in interpretability after removing different modules, but no quantitative metrics are provided to measure these changes. Could the authors include specific interpretability metrics to quantify the impact of each module?
3.	There appear to be issues with Figure 6. The function parameters in Step 1 are incorrect, and there is an unintended split in the image for Step 3.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces Plan-of-SQLs (POS), an approach for enhancing interpretability in Table Question Answering. POS decomposes complex queries into atomic sub-queries, each translated into SQL commands executed sequentially. This design enables users to trace the decision-making process step-by-step. The authors claim that POS not only outperforms comparable Table QA methods in interpretability but also maintains competitive accuracy, with promising results on benchmarks such as TabFact and WikiTQ.

### Strengths
- POS is well-designed for interpretability by breaking down complex queries into simple, understandable natural language planning. This sequential process makes it easier for users to follow and verify each stage of the answer generation. 
- The authors employ both subjective and objective evaluations, involving human and LLM judges, to assess POS’s interpretability, coherence, and helpfulness, providing strong evidence for POS's improvement on interpretability. 
- POS performs reasonably well on key benchmarks, achieving high interpretability without compromising accuracy.

### Weaknesses
- The authors report that 9.8% of samples on TabFact and 27.8% on WikiTQ could not be fully processed using POS and thus defaulted to end-to-end QA methods. Even on relatively simple Table QA datasets like TabFact and WikiTQ, these rates are notably high, raising concerns about POS’s scalability to more complex datasets, such as TabMWP. If the unprocessable rate rises significantly with more complex datasets, it raises the concern that POS may only improve interpretability with a sacrifice of precision provided in pure program-based methods.  The authors do not provide enough analysis regarding this matter, such as a comparison of unprocessable rates between POS and other program-based models and the unprocessable rates on more complex table-QA datasets. 

- The technical contribution of POS is limited, since the whole framework primarily involves prompt engineering. Essentially, compared to traditional program-based methods, POS simply adds an additional layer of prompts to generate natural language “plans” that function as pseudo-comments for SQL statements. This does not introduce substantial technical contribution beyond traditional program-based methods. 

- The NL Atomic Planner in POS depends on in-context planning examples that are specifically tailored to each dataset. This raises questions about its adaptability to a variety of reasoning types and Table QA problems. It remains unclear whether the current prompting schema can generalize effectively across different Table QA tasks. Further experimentation across a broader range of datasets (ideally 5–6) would be needed to demonstrate POS’s generalizability and robustness.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2
