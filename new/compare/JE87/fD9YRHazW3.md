# Review

## Summary
This paper presents In-Context Watermarks (ICW), a novel watermarking method for large language models (LLMs) that embeds watermarks into text solely through prompt engineering, leveraging LLMs' in-context learning and instruction-following abilities. The authors propose four ICW strategies—Unicode, Initials, Lexical, and Acrostics—each with tailored detection methods. A key focus is the Indirect Prompt Injection (IPI) setting, where watermarking is applied to detect AI-generated reviews in academic peer review, with the conference organizers having no access to the models used by reviewers. Experiments validate ICW as a model-agnostic, practical approach, with performance improving as LLM capabilities increase. The paper discusses the trade-offs between detectability, robustness, and text quality, and explores the limitations of ICW under potential attacks.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. ICW offers a novel approach to watermarking that does not require access to the decoding process of LLMs. This is particularly useful in scenarios where the model owner is unknown or untrusted, such as in the academic peer review setting discussed in the paper.

2. The paper provides a thorough exploration of different ICW strategies, including Unicode, Initials, Lexical, and Acrostics, each with distinct advantages and trade-offs in terms of detectability, robustness, and text quality. This helps in understanding the capabilities and limitations of ICW in different contexts.

3. ICW is designed to be model-agnostic, meaning it can be applied to various LLMs without modifying the models themselves. This broadens the applicability of watermarking to a wider range of scenarios and models.

4. The paper includes extensive experiments to validate the effectiveness of ICW. These experiments demonstrate the feasibility and practicality of ICW in detecting AI-generated text, particularly in the IPI setting.

## Weaknesses
1. The effectiveness of ICW is highly dependent on the capabilities of the underlying LLMs. For example, some ICW strategies may not perform well with LLMs that lack strong in-context learning or instruction-following abilities. This limits the applicability of ICW to only certain models.

2. While the paper discusses the trade-offs between detectability, robustness, and text quality, it does not provide a comprehensive analysis of these trade-offs across different ICW strategies. Further exploration of how these factors interact could help in designing more balanced watermarking schemes.

3. The paper primarily focuses on the IPI setting, specifically in the context of academic peer review. While this is a relevant and important application, the generalizability of ICW to other scenarios and use cases is not thoroughly explored.

## Questions
1. How does the performance of ICW change with different LLM architectures and sizes? Are there certain types of LLMs or configurations where ICW is more effective or less effective?

2. Can you provide more detailed analysis of the trade-offs between detectability, robustness, and text quality for each of the ICW strategies? How can these trade-offs be better balanced?

3. How generalizable is ICW to other applications beyond academic peer review? Have you explored its use in other scenarios, such as content moderation, plagiarism detection, or intellectual property protection?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4