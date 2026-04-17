# Review

## Summary
This paper introduces EditBench, a benchmark designed to evaluate the code editing capabilities of LLMs in real-world scenarios. The benchmark comprises 540 problems spanning multiple natural and programming languages, collected from real-world user interactions. The paper evaluates 40 LLMs on EditBench and analyzes their performance across different categories of user instructions and levels of contextual information provided.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- **Real-World Relevance**: The benchmark is grounded in real user instructions and code contexts, making it highly relevant for AI coding assistants in practical use.
- **Diversity**: EditBench includes a diverse set of problems, multiple natural and programming languages, and various types of edits (e.g., bug fixes, feature additions), which reflects the complexity of real-world coding tasks.
- **Challenging**: The benchmark proves to be challenging, with only one model exceeding a 60% success rate, indicating its effectiveness in distinguishing the capabilities of different LLMs.
- **Thorough Evaluation**: The paper evaluates a wide range of LLMs (40 diverse models) and provides a detailed analysis of their performance across different categories and contextual information.

## Weaknesses
- **Limited Problem Size**: The benchmark consists of only 540 problems, which may not capture the full spectrum of real-world coding challenges.
- **Translation Quality**: While GPT-4o was used for translations, the quality of the translated problems and their accuracy in representing the original user instructions are not thoroughly evaluated.

## Questions
- How do you ensure the quality and accuracy of translations in EditBench-complete?
- What steps have you taken to mitigate potential biases in the benchmark data?
- How do you plan to update and expand EditBench in the future to keep pace with evolving LLMs and coding assistant technologies?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4