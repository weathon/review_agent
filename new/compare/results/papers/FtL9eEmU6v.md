# Ed I Tbe N C H: Evaluating Llm Abilities To Perform Real-World Instructed Code Edits

Wayne Chi1,∗ Valerie Chen1,∗ **Ryan Shar**1,∗ Aditya Mittal1 Jenny Liang1 Wei-Lin Chiang2 **Anastasios Nikolas Angelopoulos**2 Ion Stoica2 Graham Neubig1 Ameet Talwalkar1,† **Chris Donahue**1,†
1Carnegie Mellon University 2UC Berkeley, Arena

## Abstract

Instructed code editing, where LLMs directly modify a developer's existing code based on a user instruction, is becoming a widely used interaction mode in AI coding assistants. However, few benchmarks directly evaluate this capability and current datasets often rely on artificial sources. We introduce EditBench, a benchmark for evaluating LLM code editing capabilities grounded in real-world usage, i.e., user instructions and code contexts collected in the wild. EditBench comprises of 540 problems, multiple natural and programming languages, and a diverse set of real-world use cases, ranging from resolving errors to adding features. EditBench introduces context-dependent problems that require the model to understand code context, highlighted code, and cursor position in addition to the user instruction. We evaluate 40 diverse LLMs and observe that EditBench is a challenging set of problems where only 1 model scores over 60%. We find that model performance varies across different categories of user instructions. Further, we find that different levels of contextual information greatly affect task success rate, with performance varying up to 8%, indicating the importance of evaluating with realistic context.

Github Repo https://github.com/waynchi/editbench Leaderboard https://waynechi.com/edit-bench/

## 1 Introduction

Software developers increasingly write code with AI assistants such as Github Copilot (Github, 2022),
Cursor (Cursor, 2023), and Continue (Continue Dev, 2025) using a variety of modes of interaction. *Instructed* code editing, where developers use natural language to request the assistant to edit a highlighted section of code, has emerged as a prominent interaction mode alongside autocomplete suggestions and chat (Nam et al., 2025). Due to the flexibility provided through natural language instructions, use cases for edits are diverse and range from code improvements given detailed user instructions to bug fixes provided only an error trace (Cassano et al., 2023b). Because of this, instructed code edits pose a challenging set of problems that existing LLMs must tackle to support developers. Despite the emergence of this new interaction modality, we lack benchmarks to capture real-world edit behavior. Code generation benchmarks typically evaluate LLM capabilities on generating code from scratch (Chen et al., 2021; Austin et al., 2021; Jain et al., 2024; White et al., 2024). While there are a few edit-related datasets (e.g., CanItEdit (Cassano et al., 2023b), Aider polyglot (Gauthier, 2025)), the sources of data are
* Equal contribution.

† Equal senior author.

1

![1_image_0.png](1_image_0.png)

not reflective of most real-world software development, relying on either simple, annotator-written problems or Leetcode and educational style problems that do not capture diverse, real-world software development challenges. Recent work has begun collecting human preferences to interactively evaluate models—Chatbot Arena (Chiang et al., 2024) evaluates LLM capabilities for chat and contains a coding subset, while Copilot Arena (Chi et al., 2025) evaluates LLM capabilities to perform code completions—highlighting a growing awareness of the need for grounding evaluations with in-the-wild data. However, "arena-style" evaluations are costly, requiring a significant number of human votes to rank a new model.

We introduce EditBench, a benchmark for evaluating LLM code editing capabilities built on real-world edit contexts and instructions (Figure 1). We source our problems by developing a VS Code extension that mimics existing instructed code editing tools from GitHub Copilot and Cursor. As developers use the extension, we gather a live, in-the-wild dataset containing user-written instructions, associated code context, and user votes between pairs of model responses. We recruited nearly 500 users to provide these data points. EditBench differentiates from previous edit-related benchmarks in several ways:
Diverse user instructions and context. Since EditBench is constructed from data collected from programmers performing day-to-day coding tasks, users specify user goals with diverse content and formats. For example, a bug fix can be requested as "fix this" accompanied with highlighted code, a direct dump of the error trace, or a natural language description of the erroneous behavior. EditBench tests for these varied user instructions instead of the more templated approaches (e.g., fix a specific function in a well-defined way) in previous benchmarks. Context dependent problems. Real instructed code edits often feature ambiguous user instructions that require contextual clues to parse the underlying user intent. In addition to the user instruction, in EditBench we also capture the code file to edit, the highlighted region of code, and the user's current cursor position. Code context length can be significant (e.g., ≥10k characters), requiring the model to properly use the comments, highlighted code, and other contextual clues to determine the correct solution. We are the first benchmark to include this combination of features for instructed code edits. Multiple natural and programming languages. While most previous coding benchmarks consist of only English problems, EditBench consists of 5 natural languages (English, Spanish, Russian, Chinese, Portuguese) and 2 programming languages (Python and Javascript). Since our code is gathered in-the-wild, any natural language variations occur in both the user instruction and code itself.

We evaluate 40 open-weight and closed models on EditBench and find that the best model, claude-sonnet-4 (Anthropic, 2023), achieves a pass@1 of 66.67%. Closed-source models tend to outperform open-weight models, with deepseek-chat-v3.1 and kimi-k2-0905 being the only two open-weight models in the top 10. We observe that both the inclusion of additional context (e.g., highlighted code and cursor position) and the type of edit category (e.g., optimization versus bug fixing tasks) drastically affects performance, Finally, we find that EditBench is only weakly correlated with existing edit benchmarks like Aider Polyglot (Gauthier, 2025), suggesting that our real-world data captures a unique set of difficult edit tasks. Our results show that EditBench is challenging even for state-of-the-art models and reveals new insights into model capabilities, emphasizing the importance of benchmarking LLMs on realistic data.

## 2 Related Work

Coding Benchmarks. Static benchmarks, e.g., HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021), largely focusing on interview-style programming problems have been the most commonly used to evaluate coding capabilities (Lu et al., 2021; Nijkamp et al., 2023; Zhu et al., 2022; Wang et al., 2023; Liu et al., 2023; Jimenez et al., 2023b; Khan et al., 2023; Yan et al., 2023; Cassano et al., 2023a; Muennighoff et al., 2023; Dinh et al., 2023; Yang et al., 2024b), measured using pass@k. Additionally, some recent work focuses on creating live benchmarks that reduce contamination risks (Jain et al., 2024; White et al.,
2024). Increasingly, people are interested in code editing with LLMs, focusing on bug fixing (Zhang et al., 2023b; Moon et al., 2023; Shinn et al., 2023; Chen et al., 2023; Olausson et al., 2023; Jin et al., 2023; Joshi et al., 2023; Wei et al., 2023; Li et al., 2022), a specific subset of code editing; fill-in-the-middle code completion (Bavarian et al., 2022; Fried et al., 2023; Yee & Guha, 2023; Roziere et al., 2023; Guo et al., 2024a; Zhang et al., 2023a), an inference strategy that requires specific insert locations; and intrinsic code editing (Li et al., 2023; Gupta et al., 2023), which involves editing code without a specified instruction, exerting the model's ability to intrinsically ascertain the desired code changes. CodeEditorBench (Guo et al., 2024b) evaluates code editing using competitive programming problems and CanItEdit (Cassano et al., 2023b) expands on this to create varied prompts and diverse topics. Grounding Evaluation in Real-World Data. A limitation of the aforementioned benchmarks is that the source of their tasks is not from real-world user data. Copilot Arena (Chi et al., 2025) evaluates code completions with real-world data and highlights how the distribution of data from benchmarks differs from real-world data in terms of the type of task, context length, and more. However, these in-the-wild evaluations require immense scale to build a leaderboard and evaluate new models (e.g., Chatbot Arena (Chiang et al., 2024) has millions of votes). The primary benchmark that creates problems from real-world sources is SWE-Bench (Jimenez et al., 2023a) and related extensions including SWE-Bench Multimodal (Yang et al., 2024a) and Multi-SWE-Bench (Zan et al., 2025). However, these benchmarks focus on fixing issues that require agentic workflows (e.g., editing multiple files) and are limited to a handful of repositories or problems written in one natural language. Our work, EditBench, complements this growing set of benchmarks by providing a benchmark for instructed code edits that is *realistic* (i.e., collected from real users in real workflows) and *diverse* (i.e., contains many different natural languages and task categories).

## 3 Benchmark Construction 3.1 Data Collection.

We develop an open-source VSCode extension with instructed code editing as a core feature to support the collection of code edit data. Gathering data via a real coding extension (Izadi et al., 2024; Chi et al., 2025) allows for more realistic instructions and tasks when compared to coding competition platforms. For each code edit, the user highlights a code-snippet and writes a short task description (Figure 2). Participants are not compensated for using the extension, as in a traditional user study, but instead receive free access to

![3_image_0.png](3_image_0.png)

state-of-the-art models. Given the sensitive nature of programming, we established clear privacy controls to give users the ability to restrict our access to their data. Depending on privacy settings, we collect the user's instruction, code context (including the highlighted code segment, the cursor location, prefix, and suffix) at the time of the request, and model responses. Additionally, we log whether the user accepted the edit. Our data collection process was reviewed and approved by our institution's IRB. Additional details about our data collection policy are provided in Appendix A.

## 3.2 Problem Curation.

Across 458 users, we collected 2672 responses (i.e., the user accepted an edit). However, not all of these responses were interesting, challenging, or even feasible to turn into testable problems. We narrow our problem set in the following ways. First, we focus on questions written in Python and Javascript, which combined comprise of the majority of our responses at just over 1700 problems. Second, we exclude problems that are too similar to one another—sometimes a user might try similar prompts on the same code context to see how different models edit. Lastly, we remove any trivial (e.g., add a single parameter), stylistic (e.g., add a comment), or ambiguous problems. We provide concrete examples of removed problems in Appendix C. This filtering process left us with around 470 problems which we found both interesting and challenging. Given that not all problems are feasible to create test harnesses for, we succeeded in creating 109 unique problems for EditBench-core. There are five languages—English, Russian, Chinese, Polish, and Spanish —in EditBench. In order to equally distribute the natural languages in the problem set, we also translate each problem to the other languages found in our problem set to form EditBench-complete. To do so, we followed a similar method prescribed by HumanEval-XL (Peng et al., 2024) and translate the comments in each problem using GPT-4o to create a total of 540 problems. To validate the translations, we had native speakers evaluate a subset of the translated tasks, primarily in Chinese and Spanish. In addition to GPT-4o, we experimented with several other models (GPT-4o-nano, GPT-4o-mini) and Google Translate, but found GPT-4o to provide the best quality with no noticeable concerns with any of the translations.

## 3.3 Test Harness Creation.

The data from our extension provides us with realistic human instructions and code, but does not contain test cases, making the raw data ill-suited for a benchmark. We create test harnesses composed of the environment setup, which includes preparing configurations, virtual environments, or mock files, and *test cases* that define expected inputs and outputs.

To write our tests, we assemble a team of five experienced programmers who have expertise in both natural and programming languages present in the real-world edit data. The team, recruited through academic networks, included researchers and students from various fields who write code extensively. The annotators 4 Table 1: Comparing EditBench **to other edit-related benchmarks.** We compare EditBench with similar benchmarks (CanItEdit (Cassano et al., 2023b), EditEval (Hu et al., 2023), Aider Polyglot) in terms of the problem source, user instruction (\# NL refers to the number of natural languages), code context (\# PL refers to the number of programming languages, HL refers to whether users can highlight a subset of code),
and associated test cases. Standard deviation is indicated by ±. EditBench is the only benchmark built from in-the-wild problems and exhibits considerable variation in both instruction and code context length.

| Benchmark                         | Problem   | Instruction      | Code Context   |             |        |             |     |
|-----------------------------------|-----------|------------------|----------------|-------------|--------|-------------|-----|
| # Problems                        | Source    | # NL             | Length         | # PL        | Length | HL          |     |
| CanItEdit (Cassano et al., 2023b) | 105       | Annotator        | 1              | 140 ± 105   | 3      | 1309 ± 1116 | No  |
| EditEval (Hu et al., 2023)        | 194       | Annotator        | 1              | 99.9 ± 49.3 | 1      | 258 ± 185   | No  |
| Aider Polyglot (Gauthier, 2025)   | 225       | Coding Exercises | 1              | 606 ± 885   | 5      | 6184 ± 6452 | No  |
| EditBench                         | 540       | In-the-wild      | 5              | 238 ± 738   | 2      | 5642 ± 7567 | Yes |

![4_image_0.png](4_image_0.png)

were instructed to create test harnesses that adhere to the user's intent and are generalizable to different potential implementations. While the user instruction and code file are perhaps the most important pieces of information, they by themselves can often be too ambiguous. The highlighted code segment and cursor locations provide crucial contextual clues to prescribe user intent. Annotators were asked to design problems given all of this information, and if a problem was still too ambiguous, we asked the annotators to remove the problem. To support the annotation process, we generated some example solutions using GPT-4o and Sonnet 3.7 (chosen to balance cost and quality) to give insight into possible solutions. Additionally, annotators were also asked to screen for and remove any Personal Identifiable Information (PII). Finally, all refined test cases were assigned to a second annotator in the team to do a second review with the same procedure.

Originally, we attempted to use a coding agent (e.g., Claude Code) to construct test cases, but found that the agent often struggled with test case generation itself, frequently resorting to undesirable tests such as directly pattern-matching with the source code, despite explicit instructions to avoid this behavior. However, despite the complexities involved in environment setup, especially for languages such as Javascript, we found the agent was consistently able to set up the correct packages and environments. As a result, we used the agent to setup the test harness environment. We provided setup files (e.g., a conftest.py file in Python and a jest-config.js file for Javascript) to help support the agent and standardize outputs.

## 4 Benchmark Statistics

EditBench consists of 540 problems that span 5 natural languages (English, Spanish, Russian, Chinese, Portuguese) and 2 programming languages (Python and Javascript). EditBench features a diverse set of problems with considerable variation in instruction and code context lengths (Table 1). Based on the import library usage (Figure 3), we can see that EditBench captures 74 different unique imports, demonstrating Table 2: **Comparing user instructions written in IDE to the instructions written by human annotators.** We provide examples across different task categories, comparing with two edit-related datasets (CanItE-
dit (Cassano et al., 2023b) and EditEval (Hu et al., 2023)). We truncate some instructions for brevity and provide full examples in Appendix B. In general, we find that real-world prompts are much less specified and require models to leverage the provided context, compared to existing benchmark prompts.

| EditBench (Ours)                                                                          | CanItEdit (Cassano et al., 2023b)                                                                                     | EditEval (Hu et al., 2023)                                                                                                                   |
|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Feature Addition take the globe countries layer                                           | Add a method                                                                                                          |                                                                                                                                              |
| from below ''// this'' and add                                                            | 'estimate location' that                                                                                              |                                                                                                                                              |
| it to the existing globe                                                                  | returns the estimated the appropriate location for this house, calculated by...                                       | Add a function 'filter odd numbers' to filter odd numbers using lambda function.                                                             |
| Feature Modification do not use R style, use python                                       | Flip the correlation function                                                                                         |                                                                                                                                              |
| style                                                                                     | given to calculate the covariance instead using the Corr(X, Y), Var(X) and Var(Y). The new function should...         | Modify the function to correctly determine the season based on month and day, considering edge cases for season changes. Raise error when... |
| Resolve Errors RuntimeError:                                                              | Cannot close                                                                                                          |                                                                                                                                              |
| a running event loop sys:1: RuntimeWarning: coroutine 'Application.shutdown' was never... | Fix combination unlimited rep() so that it returns the right result. The function combination unlimited rep should... | Fix the bug in 'sum even and even index' to make it return the sum of even numbers at even indices.                                          |
| Optimize Code optimize the computation by                                                 | Optimize the bm25 algorithm                                                                                           | Optimize the function to find                                                                                                                |
| better batching the latter                                                                | by avoiding frequency                                                                                                 | the longest common subsequence                                                                                                               |
| part                                                                                      | calculations.                                                                                                         | for the given two sequences using dynamic programming                                                                                        |

much more diversity (at least three times) than existing benchmarks. From our analysis on EditBench problems, we find the following characteristics:
Real user instructions are diverse and messy. When inspecting real-world data, we find that users write varied instructions across many problem categories. While many of these categories are similar to existing benchmarks, we find that user instructions are much more informal and less well-specified compared to the annotator-written instructions in existing benchmarks (Table 5). Interestingly, even the way a user would write an instruction within a category varies in terms of descriptiveness. For example, to resolve errors, users may briefly describe the erroneous behavior using natural language or directly paste in the terminal error traces. Further, unlike prior benchmarks where user instructions are only written in English, we find users write instructions in multiple languages, including Russian, Chinese, and Spanish (see Table 1 for additional comparison of user instructions). Real-world code contexts span many applications and context lengths. We observe that users work on a variety of applications, including frontend/backend, machine learning, and algorithmic problems. Additionally, the context lengths are much longer than those evaluated in prior benchmarks (Table 12). We also look at the distribution of code-to-edit token lengths, as computed by the number of highlighted tokens, and find that most people are highlighting targeted portions of code for edits. The median is 138 tokens, while the full file is typically closer to 4.5k tokens. The code contexts that we collect are primarily in Python (43%), with the next most common programming languages being Javascript/Typescript (21%), PHP (18%), and HTML (7%).

We focus on problems written in Python and Javascript, which together comprise the majority of in-the-wild instructed edits collected.

We identify four common clusters of functional edits. By analyzing in-the-wild user instructions in EditBench, we derive four different categories that describe functional real-world edits: *feature addition*, feature modification, *bug fixing*, and *optimization*. We find the distribution across these categories as 43% additions, 27% modifications, 22% fixes, and 8% optimizations. Table 2 provides examples of each category. In our later analysis, we compare how well models are able to perform these different problem categories.

## 5 Evaluation

We now use EditBench to evaluate models and identify trends in code editing capabilities across models. We also compare EditBench results to existing benchmarks. We overview our choice of LLMs, evaluation metrics, and prompts to perform code edits, with additional details in Appendix D.

Model choices. We select 40 LLM spanning multiple model families, sizes, and training schemes (e.g., reasoning and non-reasoning models). We use 9 models from the GPT family (OpenAI, 2025), 8 models from Qwen (Hui et al., 2024), 5 models from Llama (Meta, 2025), 4 models from Mistral (Mistral, 2025), 3 models from Sonnet (Anthropic, 2023), 3 models from Gemma (Team, 2025b), 2 models from Grok (Grok, 2025), 2 models from Deepseek (DeepSeek-AI et al., 2024), 2 models from Gemini (Google DeepMind, 2025), 1 model from Kimi (Team, 2025c), and 1 model from the GLM family (Team, 2025a). For a full list of models, see Table 6. For GPT reasoning models (gpt-o3-mini, gpt-o4-mini, gpt-5), we also vary reasoning effort. We set temperature to 0 when possible to reduce non-deterministic outputs. Evaluation Metrics. Following prior work (Kulal et al., 2019; Chen et al., 2021), we report pass@1, where 1 code sample is generated per problem and a problem is considered solved if it passes all unit tests. To facilitate analysis on the types of problems that current models excel or struggle with, we also partitioned our dataset into two subsets of easy and hard difficulty, in addition to reporting the Full results. We categorized problems that were solved by k or fewer models as Hard and the remainder as Easy (Gauthier, 2025). To obtain a roughly even split between problems, we selected k = 20. We find that easy versus hard problems are roughly evenly distributed across problem categories.

Code Editing Methods. In all our prompts, the model is given the user instruction and main code context and requested to edit the entire file by regenerating the entire code context. We also evaluate models when given varying levels of contextual information (e.g., highlighted code and cursor position). We find that models perform best when given highlighted code, but not cursor position; hence, we run all of our main experiments with highlighted code given only. All prompts are provided in Appendix D.

## 5.1 Discussion Of Results

We present our primary results in Figure 4 and highlight the key takeaways below. Appendix E provides additional results and discussions.

EditBench **is a challenging benchmark, even for current state-of-the-art models.** Only 1 out of 40 models (claude-sonnet-4) achieves more than a 60% pass@1. Further, EditBench captures questions of varying difficulty, reflecting the diversity of challenges in real-world code edits. As such, we find a sharp contrast between the easy and hard questions, where the average gap across models is 59.3%
(standard deviation of 10.6%). Given the large gap between easy and hard problems, we explore what types of prompts are present in hard problems compared to the general dataset. Overall, we see that hard instructions tend to have *shorter* instructions (by nearly 5 times) but slightly *longer* highlighted code. This means that the model cannot simply rely on following the user's instructions alone but rather needs to reason about multiple pieces of information. We provide an example in Appendix E.

7

![7_image_0.png](7_image_0.png) 

| Pass@1             |           |               |               |                    |
|--------------------|-----------|---------------|---------------|--------------------|
| Model Name         | Code Only | +Highlight    | +Cursor Only  | +Highlight +Cursor |
| claude-sonnet-4    | 62.41     | 64.81 (+2.40) | 63.15 (+0.74) | 64.26 (+1.85)      |
| deepseek-chat-v3.1 | 51.48     | 54.26 (+2.78) | 53.15 (+1.67) | 52.78 (+1.30)      |
| gemini-2.5-flash   | 52.59     | 52.96 (+0.37) | 52.41 (-0.18) | 56.30 (+3.71)      |
| glm-4.6            | 52.96     | 56.48 (+3.52) | 52.22 (-0.74) | 44.81 (-8.15)      |
| o3-mini            | 60.00     | 56.85 (-3.15) | 59.26 (-0.74) | 55.19 (-4.81)      |
| kimi-k2-0905       | 54.63     | 56.48 (+1.85) | 52.22 (-2.41) | 58.15 (+3.52)      |
| qwen3-coder        | 56.48     | 53.89 (-2.59) | 56.48 (+0.00) | 53.89 (-2.59)      |

Model performance is affected by additional contextual information. To evaluate how additional contextual information (highlighted code and cursor position) affects model performance, we run an ablation with the 7 top models in different model families (Table 3). When adding highlighted code to the prompt, the task success rate increases for 5 out of the 7 models. On the other hand, additionally adding the cursor position leads to mixed performance when compared to only adding the highlighted code. We notice that trends are generally consistent; the two models that do not benefit from including highlighted code in contexto3-mini and qwen3-coder—do not benefit from including cursor position either. These findings show the importance of evaluating models on editing tasks that require integrating multiple pieces of information. Gap between closed and open models. Comparing the colors in Figure 4 very readily shows that open models tend to lag behind closed models. Out of the 40 models we evaluate, only 4 out of the top 15 are open models, and the bottom 15 are all open models. Of the open models, we find that glm-4.6 performs the

![8_image_0.png](8_image_0.png)

best with a pass@1 of 56.48%, with kimi-k2 and deepseek-chat-v3.1 not far behind. Surprisingly, gpt-5 with default reasoning (medium effort) lags behind gpt-5-mini. When inspecting test cases where gpt-5 failed, we find that it struggles with simple tasks like formatting code indentation properly and catching edge cases, despite being a strong reasoning model. Models excel in different problem categories. When we divide questions into categories that test different editing-related skills, we find that performance varies. Overall, we find that models perform best on bug fixing problems (average of 52.2%), which may be most akin to tasks found in prior benchmarks like SWE- Bench (Jimenez et al., 2023a). In contrast, models tend to struggle with optimization and feature addition
(44.6% and 39.6%, respectively). Still, we find that claude-sonnet-4 ranks first in every category except optimization. Furthermore, we find that some models have particularly large gaps between categories (Figure 5). For example, qwen3-coder-flash's top category is fixing bugs while claude-sonnet-4's is making feature modifications.

## 5.2 Comparison To Existing Benchmarks

We compare our results with two maintained leaderboards: performance on Aider Polyglot (Gauthier, 2025), which has been used in prior model releases as a metric of model editing capabilities, and ranking on the coding subset of Chatbot Arena (Chiang et al., 2024), which has been widely used to capture human preferences. We have 17 and 30 shared models, respectively. We observe a weak, positive correlation with both Polyglot (Pearson correlation coefficient r = 0.24, p = 0.06) and Chatbot Arena (r = 0.11, p = 0.01).

We believe our observations are due to the following factors. The first is **code-centric input and output.**
Input/outputs in Chatbot Arena are often written purely in natural language, so the *majority* of codingrelated questions in Chatbot Arena do not contain code (Chi et al., 2025); this is unlike EditBench and Polyglot, both of which require code for every problem. Second, there is a difference in **interaction modality.**
EditBench and Polyglot test a model's ability to perform *instructed code edits*, where there is a freeform input (the user instruction) and structured output (the resulting code), while Chatbot Arena evaluates a model's ability to *chat*, where there is both freeform inputs and outputs. Also, the inclusion of additional code context (e.g., highlighted code) may affect correlation to Polyglot. Finally, correlation may be affected by the inclusion of **real-world user intent.** Polyglot's problems are entirely based on coding exercises from educational-style problems that lack the organic user intent present in Chatbot Arena and EditBench.

## 6 Conclusion, Limitations, And Future Work

As instructed code edits become more widely adopted in real-world IDEs, there is a need to benchmark LLM
capabilities on these types of problems. We develop a VSCode extension to collect real-world instructed code edits, which include user instructions and code contexts. We transform this in-the-wild edit data into EditBench, a set of high-quality test harnesses that evaluate LLM's ability to perform diverse tasks. Evaluations on 40 models show that EditBench is challenging even for current state-of-the-art models and provides insights into how performance varies when considering different code context information and types of edits. Overall, to adequately support developers using LLM-powered tools, our findings demonstrate the need for future models to be trained on real-world interaction modes and evaluated across a broad spectrum of problem categories, languages, code contexts, and user intents. Limitations and Future Work. While we attempted to make EditBench as diverse as possible, there are still additions from which it would benefit. For example, as we collect more data using our extension, we will increase the number of examples we have for the existing languages and expand to other common programming languages. Additionally, despite improvements over existing benchmarks, it is unclear to what extent our problems encapsulate all real-world use cases. We plan to continue updating the EditBench leaderboard as new models are released and exploring automatic workflows to more seamlessly translate real-world data to benchmark problems.

## Acknowledgments

This work was supported in part by the National Science Foundation grants IIS1705121, IIS1838017, IIS2046613, IIS2112471, and funding from Datadog. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of any of these funding agencies.

## References

Anthropic. Meet claude, 2023. URL https://www.anthropic.com/claude. Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv* preprint arXiv:2108.07732, 2021.

Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek, and Mark Chen. Efficient training of language models to fill in the middle. *arXiv preprint arXiv:2207.14255*, 2022.

Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, et al. Multipl-e: a scalable and polyglot approach to benchmarking neural code generation. *IEEE Transactions on Software Engineering*, 2023a.

Federico Cassano, Luisa Li, Akul Sethi, Noah Shinn, Abby Brennan-Jones, Jacob Ginesin, Edward Berman, George Chakhnashvili, Anton Lozhkov, Carolyn Jane Anderson, et al. Can it edit? evaluating the ability of large language models to follow code editing instructions. *arXiv preprint arXiv:2312.12450*, 2023b.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Xinyun Chen, Maxwell Lin, Nathanael Scharli, and Denny Zhou. Teaching large language models to ¨
self-debug. *arXiv preprint arXiv:2304.05128*, 2023.

Wayne Chi, Valerie Chen, Anastasios Nikolas Angelopoulos, Wei-Lin Chiang, Aditya Mittal, Naman Jain, Tianjun Zhang, Ion Stoica, Chris Donahue, and Ameet Talwalkar. Copilot arena: A platform for code llm evaluation in the wild. *arXiv preprint arXiv:2502.09328*, 2025.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. *arXiv preprint arXiv:2403.04132*, 2024.

Inc. Continue Dev. Continue: Open-source ai code assistant. https://github.com/continuedev/
continue, 2025. Accessed: 2025-05-08.

Cursor. Cursor: The ai-first code editor, 2023. URL https://cursor.sh/features. Accessed:
2023-12-03.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, and Chengda Lu et al. Deepseekv3 technical report. *ArXiv preprint*, abs/2412.19437, 2024. URL https://arxiv.org/abs/2412.

19437.

Tuan Dinh, Jinman Zhao, Samson Tan, Renato Negrinho, Leonard Lausen, Sheng Zha, and George Karypis.

Large language models of code fail at completing code with potential bugs. Advances in Neural Information Processing Systems, 36, 2023.

Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Scott Yih, Luke Zettlemoyer, and Mike Lewis. Incoder: A generative model for code infilling and synthesis. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.

net/forum?id=hQwb-lbM6EL.

Paul Gauthier. Aider polyglot coding benchmark. https://aider.chat/docs/leaderboards/,
2025. Accessed: 2025-05-08.

Github. Github copilot - your ai pair programmer, 2022. URL https://github.com/features/
copilot.

Google DeepMind. Gemini 2.5: Our newest gemini model with thinking. https://blog.google/
technology/google-deepmind/gemini-model-thinking-updates-march-2025/,
March 2025.

Grok. Grok code fast 1 model card. https://data.x.ai/
2025-08-26-grok-code-fast-1-model-card.pdf, August 2025.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets programming - the rise of code intelligence, 2024a.

Jiawei Guo, Ziming Li, Xueling Liu, Kaijing Ma, Tianyu Zheng, Zhouliang Yu, Ding Pan, Yizhi LI,
Ruibo Liu, Yue Wang, Shuyue Guo, Xingwei Qu, Xiang Yue, Ge Zhang, Wenhu Chen, and Jie Fu.

Codeeditorbench: Evaluating code editing capability of large language models, 2024b. URL https:
//arxiv.org/abs/2404.03543.

Priyanshu Gupta, Avishree Khare, Yasharth Bajpai, Saikat Chakraborty, Sumit Gulwani, Aditya Kanade, Arjun Radhakrishna, Gustavo Soares, and Ashish Tiwari. Grace: Language models meet code edits.

In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2023. URL https://doi.org/10.1145/3611643. 3616253.

Qisheng Hu, Kaixin Li, Xu Zhao, Yuxi Xie, Tiedong Liu, Hui Chen, Qizhe Xie, and Junxian He. Instructcoder:
Empowering language models for code editing. *arXiv preprint arXiv:2310.20329*, 2023.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. Qwen2.5-coder technical report. *ArXiv preprint*, abs/2409.12186, 2024. URL https:
//arxiv.org/abs/2409.12186.

Maliheh Izadi, Jonathan Katzy, Tim van Dam, Marc Otten, Razvan Mihai Popescu, and Arie van Deursen.

Language models for code completion: A practical evaluation, 2024. URL https://arxiv.org/ abs/2402.16197.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-
Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. *arXiv preprint arXiv:2403.07974*, 2024.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan.

Swe-bench: Can language models resolve real-world github issues? *arXiv preprint arXiv:2310.06770*, 2023a.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R
Narasimhan. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2023b.

Matthew Jin, Syed Shahriar, Michele Tufano, Xin Shi, Shuai Lu, Neel Sundaresan, and Alexey Svyatkovskiy.

Inferfix: End-to-end program repair with llms. In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2023. URL
https://doi.org/10.1145/3611643.3613892.

Harshit Joshi, Jose Cambronero Sanchez, Sumit Gulwani, Vu Le, Ivan Radi ´ cek, and Gust Verbruggen. Repair ˇ
is nearly generation: Multilingual program repair with llms. In Proceedings of the Thirty-Seventh AAAI
Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, 2023. URL
https://doi.org/10.1609/aaai.v37i4.25642.

Mohammad Abdullah Matin Khan, M Saiful Bari, Xuan Long Do, Weishi Wang, Md Rizwan Parvez, and Shafiq Joty. xcodeeval: A large scale multilingual multitask benchmark for code understanding, generation, translation and retrieval. *arXiv preprint arXiv:2303.03004*, 2023.

Sumith Kulal, Panupong Pasupat, Kartik Chandra, Mina Lee, Oded Padon, Alex Aiken, and Percy S Liang.

Spoc: Search-based pseudocode to code. *Advances in Neural Information Processing Systems*, 32, 2019.

Jia Li, Ge Li, Zhuo Li, Zhi Jin, Xing Hu, Kechi Zhang, and Zhiyi Fu. Codeeditor: Learning to edit source code with pre-trained models. *ACM Transactions on Software Engineering and Methodology*, 2023.

Zhiyu Li, Shuai Lu, Daya Guo, Nan Duan, Shailesh Jannu, Grant Jenks, Deep Majumder, Jared Green, Alexey Svyatkovskiy, Shengyu Fu, and Neel Sundaresan. Automating code review activities by large-scale pretraining. In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE 2022, pp. 1035–1047, New York, NY, USA, 2022.

Association for Computing Machinery. ISBN 9781450394130. doi: 10.1145/3540250.3549081. URL
https://doi.org/10.1145/3540250.3549081.