# ChildEval: How Large language models meet children’s personalities

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4, 4

## Abstract
The remarkable success of Large Language Models (LLMs) has revolutionized LLM-based chatbots for personalized tasks beyond generic dialogues. Personalization involves customizing LLMs to generate text responses based on user preferences. One promising endeavor is to enable personalized interactions for children's caretakers while also promoting development and learning. However, dedicated research is required to determine whether LLMs can effectively deliver personalized responses based on children's preferences, as their interactions differ from those of adults. We introduce ChildEval, a benchmark to evaluate LLMs' capacity to infer, interpret, and follow child-centered preferences in a long-context conversational setting. Our benchmark comprises of 29K synthesized children's (ages 3-6) persona profiles, which are related to their preferences in both explicit and implicit manners. Implicit preferences are integrated inside dialogues consisting of 6 to 10 turns. The preferences cover 5 top-level and 14 sub-level topics that involve children's daily lives and development. We further propose child-centric preferences to systematically evaluate the performance of open-source LLMs. Experimental results demonstrate the impact of various personalized representations on LLM responses and indicate that fine-tuning on this dataset may enhance performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce ChildEval - an evaluation benchmark for LLMs to rate their response from the perspective of children (3-6 years old).
They build a synthetic dataset of child-eval conversations + metrics to evaluate how well a LLM responds to a child.
Those metrics are based on custom prompts - showing a QWEN model the history, LLM_response and asking to give yes/no related to the criterion to evaluate.

Using this dataset and metrics, they evaluate 5 state of the art open-weight LLMs to see how good they are at responding to children. They show that LLMs struggle to keep their response consistent with child preferences after long and implicit conversations.They show that finetuning those LLMs on ChildEval improves evaluation performance.

### Strengths
Strengths:
- The paper introduces a synthetic dataset that looks soundly constructed for child-LLM dialogue, which can be reused by other researchers focusing on child LLM interaction. This is a novel domain that hasn't been explored.
- The paper introduces evals for child-LLM interaction - which is great for a new field and can be used as a benchmark for following papers.

### Weaknesses
- It is not clear in this paper how LORA finetuning was done to improve performance : there is no mention of train/test splits and it is unclear if LORA fine-tuning was done on the same data that is used for evaluations.
- The paper constructs personas and preferences as separate but some of the preferences are actually already included in personas. This make the finding "preference consistency increases when adding persona to the prompt" weaker.
- The 4 evals that are introduced are based on a prompt + response for QWEN - and numbers are then taken as ground truth for model comparison. It would be good to validate the stability of those metrics by comparing metric stability across different judge models.
- There is no correlation between the 4 metrics that are introduced and any real data/real human preference. The dataset is from synthetic data and the evals are from a synthetic Judge.

### Questions
What was the methodology used for LORA fine-tuning (training data, evaluation data)?
How strongly do the introduced metrics correlate with actual human preference?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ChildEval, a benchmark to evaluate how well Large Language Models can understand and respond to children's (ages 3-6) preferences in conversational settings. The benchmark comprises 29K synthetically generated children's persona profiles created through a three-step pipeline: 1) Persona Generation: Used Qwen2.5-72B to generate diverse child personas through iterative generation-and-refinement, with FAISS-based filtering to remove semantically similar profiles; 2) Preference Generation: Created 46K preferences from personas and sampled topics, expressed as first-person statements that can be revealed either explicitly or implicitly; 3) Dialogue Construction: Generated child-LLM conversations that naturally embed implicit preferences using prompt-based generation.

The benchmark introduces fine-grained child-centric metrics beyond standard Preference Consistency (PC): 
- Emotional Adaptation (EA): Sensitivity to children's emotions
- Interaction Scaffolding (IS): Guiding participation with questions/hints
- Developmental Appropriateness (DA): Age-appropriate language and complexity
- Engagement (EG): Creating lively, interesting interactions

Experiments evaluated 5 state-of-the-art open-source LLMs (Qwen2.5-3B, Qwen3-4B, LLaMA3.1-8B, DeepSeek-R1, Mistral-7B). Results showed LLMs struggle with long-term personalization and implicit preference inference. Both LoRA and PSM fine-tuning methods significantly improve performance over base models on both preference consistency and child-oriented evaluation.

### Strengths
The work studies an important research direction: formulating child-LLM interaction as a personalization problem, addressing an important and underexplored application area with significant societal impact, and offers essential resources:
- The first large-scale benchmark with child-specific evaluation dimensions (46K preferences across 14 developmental topics)
- Novel child-centric evaluation framework beyond generic preference consistency: Emotional Adaptation (EA), Interaction Scaffolding (IS), Developmental Appropriateness (DA), and Engagement (EG)
- Comprehensive problem formulation distinguishing explicit vs. implicit preferences in long-context multi-session dialogues.
- Experimental analysis across 5 state-of-the-art LLMs and 3 adaptation strategies (prompting, LoRA, PSM), revealing that: LLMs struggle significantly with implicit preference inference and long-term personalization.

### Weaknesses
The benchmark does not provide sufficient details on data quality verification or comprehensive evaluation validation. Here are the key limitations:

Dataset quality:
- The training and evaluation set are entirely generated by Qwen2.5-72B with no validation against actual child-LLM conversations. It is unclear if generated dialogues match authentic child speech/text patterns (word omissions, semantic errors).
- No user studies with actual children or domain experts to verify realistic scenarios.
- The dataset might contain model-specific biases given it is entirely generated by Qwen2.5-72B, then evaluated using Qwen2.5-72B-Instruct, which might explain why Qwen3-4B-instruct outperform DeepSeek-R1 in Table 2.

Evaluation Validation:
- No human evaluation to validate LLM-as-judge approach. The LLM-judge prompt for EA and DA does not provide the evaluation guideline for different age (e.g., 3 year-old vs 6 year-old).
- All metrics evaluated by Qwen2.5-72B-Instruct - single model bias
- No error analysis on evaluation - unclear how often the evaluator makes mistakes

### Questions
- What are the data quality filters applied other than  FAISS-based similarity filtering? What types of errors were caught during data cleaning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The study proposes ChildEval to evaluate LLMs' personalization capability toward preschool children (ages 3-6). The benchmark features 29K synthetic child persona profiles, with both explicit and implicit preferences embedded within 6 to 10-turn dialogues. The evaluation protocol goes beyond traditional Preference Consistency (PC), introducing child-specific metrics: Emotional Adaptation (EA), Interaction Scaffolding (IS), Developmental Appropriateness (DA), and Engagement (EG).

### Strengths
The paper addresses LLM interaction and personalization for children, a critical and underexplored area given LLMs' increasing role as a core medium for human-computer interaction.

### Weaknesses
- Neglect of Safety and Mental Health (Core Concern): The paper focuses too narrowly on preference matching and completely omits the crucial dimensions of LLM security, risk mitigation, and the ability to guide children toward healthy mental development. This is considered a more essential scenario and an unacceptable omission in any LLM-child interaction assessment.

- Flawed Personality Modeling: The methodology relies on generating personas via LLM-created "explicit static persona descriptions." This approach lacks discussion on what "child personality" truly is, risking significant "personality bias" and failing to capture the implicit, dynamic, and complex nature of a child's true personality.

- Data Reliability and Ethical Issues: The data is entirely model-driven/synthetic with no human participation. This raises serious doubts about the reliability and validity of the fabricated personalities, questioning whether they reflect the true distribution of human/child personalities.

- Lack of Interdisciplinary Rigor: The work is perceived as irresponsibly applying a general-purpose personality modeling framework to a sensitive child scenario without sufficient consideration of social, psychological, or educational complexities. This purely technical approach is viewed as lacking practical meaning and failing to align with the ultimate goal of ensuring LLMs serve a child's healthy growth.

### Questions
1. Why does the smaller Qwen3-4B model significantly outperform much larger models like DeepSeek-R1 in Table 2? This conclusion violates the general trend observed in the scaling law. What is the authors' explanation for this anomalous phenomenon?

2.  Why are the results for DeepSeek-R1 in Figure 2 only presented up to 10 dialogue turns, while other models (such as Qwen3-4B and Llama2-7B) are evaluated over longer contexts (e.g., 20 turns)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new benchmark called ChildEval, designed to assess the ability of large language models (LLMs) to understand, reason, and follow children's preferences in long-context conversations. The authors note that existing personalized research primarily focuses on adults, while children exhibit significantly different interaction patterns and needs, lacking corresponding evaluation metrics. To address this gap, ChildEval incorporates 29,000 synthetic child personas, each associated with explicit and implicit preferences in daily life and developmental contexts.

Beyond the dataset itself, this work proposes a fine-grained evaluation protocol tailored for children. It introduces four new child-oriented assessment dimensions alongside the traditional “Preference Consistency (PC)”: Emotional Adaptation (EA), Interaction Scaffolding (IS), Developmental Appropriateness (DA), and Engagement (EG). .

In the experimental section, the authors evaluated multiple open-source LLMs, verifying the impact of different personalization representation methods (such as incorporating personas into prompts) on model performance. They demonstrated that fine-tuning on ChildEval enhances models' ability to personalize for children. Experimental results reveal challenges existing LLMs face in handling children's implicit preferences and long dialogue histories, emphasizing the importance of comprehensive evaluation.

### Strengths
- ChildEval is thoughtfully designed. It not only incorporates children's basic profiles but also distinguishes between explicit and implicit preferences, with the latter posing greater demands on the model's reasoning capabilities. By inserting irrelevant dialogues into conversation histories to simulate long-context scenarios, the evaluation becomes more realistic and challenging.
- One of the key contributions of this paper is the introduction of four new evaluation dimensions tailored for children: EA, IS, DA, and EG. These dimensions go beyond merely assessing whether responses align with children's preferences, instead evaluating LLM responses across multiple dimensions—interaction quality, emotional support, educational value, and fun—all of which are crucial for child users. This evaluation framework provides a valuable foundation for future research in this area.

### Weaknesses
- The entire ChildEval benchmark (including child profiles and dialogues) was generated by LLM. Although the authors employed methods like deduplication to ensure diversity, synthetic data may not fully capture the complexity, creativity, and unpredictability of authentic child language. Biases inherent in the generative model itself may also be introduced into the dataset. The paper's description of the “self-verification” process is insufficiently clear, lacking human validation steps to ensure the quality and authenticity of the generated data. This represents a major limitation of the work.
- The paper employs LLM to evaluate four dimensions of child-oriented content. While “LLM-as-a-judge” is a prevalent evaluation approach today, its reliability remains under investigation. Conducting human evaluations on a small subset of data and calculating consistency with LLM assessments would significantly enhance the credibility of evaluation results.

### Questions
- Could you provide more details about the “self-verification” mechanism during dialogue generation? When constructing the dataset, was any form of manual sampling evaluation or qualitative analysis conducted to validate the authenticity and quality of the generated child profiles and dialogues—particularly those used to infer implicit preferences?
- When using LLM as an evaluator, have you considered or conducted any small-scale human evaluations to validate the accuracy of the LLM's judgments? How do you perceive and address potential biases that may arise from using an LLM to evaluate other LLMs?
- Experimental results indicate that the LoRA method generally outperforms your proposed PSM in preference consistency. Could you elaborate further on potential scenarios or aspects where PSM might hold advantages over LoRA? For instance, does it exhibit particular strengths in computational efficiency, deployment convenience, or interpretability?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ChildEval, a novel benchmark designed to evaluate how well LLMs can personalize interactions for children (ages 3-6) by inferring and following their preferences in --possibly long-- conversational settings. The benchmark includes 29K persona profiles, each paired with 6-10 turn dialogues (check Q1 for clarification), covering 5 top-level and 14 sub-level topics related to children's daily lives and development (art, cognitive development, nutrition, language, social-emotional development), in both Chinese and English. The authors propose a comprehensive evaluation framework covering Preference Consistency (Alignment score) and more child-oriented metrics (e.g., Appropriateness). Finally, the authors benchmarked four well-known models (Qwen, LLaMA, DeepSeek, Mistral) on the proposed dataset using three strategies: prompting, LoRA fine-tuning, and a novel architecture called Persona Steer Module. The authors find that the performance of LLMs degrades as irrelevant dialogue turns increase. Adding persona information helps consistency, and fine-tuning the model improves performance with different trade-offs (LoRA vs. proposed approach).

### Strengths
- Comprehensive benchmark and with 29K diverse persona profiles and realistic scenarios and diverse topics, and bilingual.
- Novel evaluation framework, with 4 new child-specific dimensions (EA, IS, DA, EG)
- Comprehensive evaluation with many SOTA LLMs, different evaluation strategies (prompting, LoRA, Persona Steer Module), long-context evaluation (up to 21K tokens) and several ablation and error analysis.

### Weaknesses
- All personas and dialogues are LLM-generated, and there is no verification that synthetic data reflects actual child behavior.
Authenticity concerns: May not capture true child communication patterns (e.g., word omissions, semantic errors mentioned but not evidenced)
- Lack of human evaluation to verify the quality of the proposed evaluation framework and metric. 

Minor
- No comparison with frontier models ( GPT-4, Claude, or Gemini), even on a subset would have being an interesting analysis.

### Questions
Q1: how many actual dialogues and turns have being generated?

### Soundness
2

### Presentation
3

### Contribution
2
