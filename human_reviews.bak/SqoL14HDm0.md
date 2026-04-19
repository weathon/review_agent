# When Prompt Engineering Meets Software Engineering: CNL-P as Natural and Robust "APIs'' for Human-AI Interaction

- Decision: Accept (Poster)
- Scores: 5, 8, 6

## Abstract
With the growing capabilities of large language models (LLMs), they are increasingly applied in areas like intelligent customer service, code generation, and knowledge management.
Natural language (NL) prompts act as the ``APIs'' for human-LLM interaction. 
To improve prompt quality, best practices for prompt engineering (PE) have been developed, including writing guidelines and templates. 
Building on this, we propose Controlled NL for Prompt (CNL-P), which not only incorporates PE best practices but also draws on key principles from software engineering (SE).
CNL-P introduces precise grammar structures and strict semantic norms, further eliminating NL's ambiguity, allowing for a declarative but structured and accurate expression of user intent. 
This helps LLMs better interpret and execute the prompts, leading to more consistent and higher-quality outputs. 
We also introduce an NL2CNL-P conversion tool based on LLMs, enabling users to write prompts in NL, which are then transformed into CNL-P format, thus lowering the learning curve of CNL-P.
In particular, we develop a linting tool that checks CNL-P prompts for syntactic and semantic accuracy, applying static analysis techniques to NL for the first time.
Extensive experiments demonstrate that CNL-P enhances the quality of LLM responses through the novel and organic synergy of PE and SE. 
We believe that CNL-P can bridge the gap between emerging PE and traditional SE, laying the foundation for a new programming paradigm centered around NL.
To further demonstrate the effectiveness of CNL-P, we develop the AI-native CNL-P IDE, UGAiForge that leverages CNL-P to enable users to Think, Describe, and Build with AI and for AI.
For detailed information, please refer https://ugaiforge.ai.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces Controlled Natural Language for Prompt (CNL-P), a novel prompt language to elicit high-quality responses from large language models (LLMs). CNL-P combines principles from software engineering with prompt engineering to create structured, accurate natural language prompts for interacting with LLMs. By defining clear syntax and components such as Persona, Constraints, Variables, and Workflow, CNL-P reduces natural language ambiguity and increases response consistency. Its structured, modular format allows for more reliable, maintainable, and interpretable prompts, acting as a natural “API” that makes human-AI interaction accessible and robust. In addition, the author proposed an NL2CNL-P agent to convert natural language prompts into CNL-P format allowing users to write prompts in natural language without expertise in learning syntax of CNL-P. A linting tool is proposed to check the syntactic and semantics of CNL-P. The experiments demonstrate the effectiveness of CNL-P in improving the consistency of LLM responses across various tasks.

### Strengths
1. The paper effectively combines prompt engineering and software engineering to introduce CNL-P as a structured, precise language for prompt design.
2. CNL-P’s modular design enables independent development, testing, and maintenance.
3. Its linting tool supports syntactic and semantic checks, which enables static analysis techniques for natural language.

### Weaknesses
I have several concerns regarding the evaluation section:
1. For RQ1, the authors asked ChatGPT-4o to assess the quality of conversions from natural language prompts to CNL-P or NL style guides based on five criteria. However, the reliability of this evaluation is not properly validated:
    - The authors did not provide evidence of how the evaluation results correlate with actual human evaluations, which would strengthen their claims. 
    - There is no guideline detailing how the scale for each category is defined, which makes it difficult to interpret the numbers in Table 1.
2. For RQ2, the experiment lacks comprehensiveness:
    - Diversity of tasks: Currently, all tasks fall under the classification category. It would be beneficial to include more complex tasks, such as reasoning and coding, to better demonstrate the effectiveness of CNL-P.
    - The authors should conduct a more thorough evaluation across a broader range of both open-source and closed-source LLMs to validate the effectiveness of CNL-P.
    - Insufficient error analysis of CNL-P across various LLMs and tasks: The performance of CNL-P varies among different models/tasks. The claim that weaker models benefit more from CNL-P lacks thorough discussion and validation. A comprehensive analysis should include task difficulty, the quality of natural language prompts, the quality of CNL-P, and their relationship to the task performance.
3. Generalization of CNL-P:  
    - As all the tasks in the experiment are classification tasks, the natural language prompts should not be overly complex (correct me if I am wrong). Consequently, the paper does not sufficiently address how CNL-P performs with very large and complex prompts. It also fails to clarify whether the linting tool can handle such complex CNL-P.
4. In lines 789-799, the natural language prompt appears well-organized and detailed. I would like to ask:
    - For an effective CNL-P prompt, is such a level of detail and organization required from the human input?
    - How does the organization of the natural language prompt impact the quality of the converted CNL-P prompt?
    - Does the CNL-P prompt still outperform a well-organized natural language prompt?
5. The plots and figures for result analysis should be integrated into the relevant pages or paragraphs; otherwise, it is hard to follow the discussion and analysis.

### Questions
1. Which Llama model was used in your experiment?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Controlled Natural Language for Prompt (CNL-P), a novel framework that bridges prompt engineering (PE) and software engineering (SE) principles to enhance the clarity, predictability, and effectiveness of prompts for large language models (LLMs). CNL-P addresses inherent ambiguities in natural language prompts by formalizing grammar structures and semantic norms. The work's primary theoretical contribution is the formalization of prompt engineering through SE principles, supported by a novel static analysis approach for natural language prompts and empirical validation across multiple LLM architectures.

### Strengths
Theoretical Innovation:
- Novel synthesis of SE principles with PE practices
- Comprehensive formal grammar for controlled natural language
- Innovative application of static analysis theory to natural language

Technical Contribution
- Formal specification of the CNL-P grammar
- Theoretical framework for prompt verification
- Rigorous performance analysis across LLM architectures
- Novel approach to static analysis of natural language

Research Impact:
- Opens new theoretical directions in prompt engineering
- Bridges formal methods and LLM interaction
- Provides a foundation for analyzing prompt properties
- Advances understanding of structured approaches to PE

### Weaknesses
Theoretical Limitations:
- Formal analysis of expressive power could be stronger.
- Completeness properties of the static analysis need more discussion.
- Edge cases in the formal grammar require deeper analysis.
- Theoretical bounds need more rigorous treatment.

Methodological Concerns:
- Formal comparison with other structured approaches could be deeper.
- Statistical analysis could be more comprehensive.
- Theoretical justification for design choices needs elaboration.
- Formal properties of the conversion process require more analysis.

Validation Gaps:
Limited formal analysis of grammar properties.
Statistical significance analysis could be more rigorous.
Theoretical comparison with other formal methods needed.
Completeness of the static analysis approach not fully addressed.

### Questions
Comparative Evaluation: Could you provide a more detailed comparison of CNL-P’s functionality and usability versus DSPy, LangChain, and Semantic Kernel? Specifically, how does CNL-P’s approach to modularity and state management differ in terms of user accessibility and technical demands?

Advantages and Trade-offs: While CNL-P is designed to decouple prompts from code for accessibility, frameworks like DSPy offer robust control through tight integration with programming language abstractions. Could you discuss specific scenarios where CNL-P might outperform DSPy or vice versa, especially in terms of prompt complexity and user involvement?

Non-Technical Accessibility: CNL-P is described as more accessible to non-technical users than PL-based methods like DSPy and LangChain. Could you elaborate on any studies, tests, or qualitative comparisons you conducted to evaluate this claim? This would clarify the extent to which CNL-P lowers the barrier for non-programmers.

Performance in Practical Applications: Do you have insights or preliminary results comparing the performance and user experience of CNL-P with DSPy, LangChain, and Semantic Kernel in specific application areas (e.g., dynamic prompt generation or complex workflow management)? Real-world examples could strengthen the practical context of CNL-P’s advantages.

Future Integration with PL-Based Methods: Given that DSPy and other PL-based frameworks emphasize structured programming benefits, do you foresee potential for CNL-P to integrate with or complement these frameworks? A discussion on interoperability could highlight pathways for combining strengths across approaches.

Scoring and Evaluation: Can you elaborate on the specific criteria used to assign scores across the five evaluation dimensions (Adherence to Original Intent, Modularity, Extensibility and Maintainability, Readability and Structural Clarity, and Process Rigor)? Was there a weighting system applied to these dimensions, or were they treated as equally important?

Interpretability of Results: The table of results is challenging to interpret due to its minimal contextual information. Could you provide a more detailed breakdown or rubric that explains how scores were derived, potentially with examples of how different prompt types scored across specific dimensions?

Comparative Analysis: Did you consider using statistical measures to compare the performance of CNL-P, RISEN, and RODES across evaluation metrics? This could strengthen the validity of the reported improvements.

Presentation Improvements: The experimental results could benefit from a more visual presentation format, such as radar charts or bar graphs, for easier comparison across dimensions. Would you consider updating the results presentation in a revised version?

Completeness of Scoring Process: Did you perform any error analysis or additional validation to understand how CNL-P performs in specific scenarios where RISEN or RODES may excel, or vice versa? This would provide insight into potential edge cases for CNL-P.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
- This paper proposes the use of Controlled Natural Language (CNL) by framing prompts as a form of API, which allows users to harness AI model capabilities without needing in-depth technical knowledge.
- This work applies software engineering (SE) principles such as modularity, abstraction, and encapsulation to Controlled Natural Language (CNL), offering a structure that decouples prompts from code
- The authors conducted experiments to evaluate how effectively CNL-P adheres to design principles and whether CNL-P or template methods improve the quality of LLM responses.

### Strengths
- Clearly motivated framing of prompting as an API, enabling users to leverage AI model capabilities without extensive technical expertise.
- Strong connection to first principles in SE, providing a foundation to address challenges in complex NL-PL conversion and prompt-code coupling. This approach is particularly beneficial for language experts and non-technical users by effectively decoupling prompts from code.
- Dimensions to assess NL-to-CNL-P conversion quality are well-designed, covering diverse quality aspects.

### Weaknesses
- The specific aims of the work remain unclear; while high-level challenges and design considerations are presented, the precise goals are hard to identify.
- Experiment setup in RQ1 lacks clarity on how the five dimensions are measured and how the 93 prompt instances were chosen. There are also no human validation results presented, even as partial samples.
- RQ1 primarily assesses design considerations, while RQ2 focuses on accuracy. Given the current setup and task scope in RQ2, the advantages of CNL-P are not fully apparent, as other models also perform well.
- For a more robust finding that CNL-P is better suited for weaker models, it would be beneficial to include additional experiments with weaker models beyond GPT-4-o mini.

### Questions
- Considering the setup and the consistency of output generation with single-turn GPT-4-o prompting, what advantages does CNL-P offer over single-turn generation with GPT-4-o?
- Given the broad goals of this work and the multi-faceted design of CNL-P, what rationale led the authors to focus on these three specific research questions?

### Soundness
3

### Presentation
2

### Contribution
2
