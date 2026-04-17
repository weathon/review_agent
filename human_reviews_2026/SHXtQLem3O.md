# ComboBench: Can LLMs Manipulate Physical Devices to Play Virtual Reality Games?

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Virtual Reality (VR) games require players to translate high-level semantic actions into precise device manipulations using controllers and head-mounted displays (HMDs). While humans intuitively perform this translation based on common sense and embodied understanding, whether Large Language Models (LLMs) can effectively replicate this ability remains underexplored. This paper introduces a benchmark, \methodname, evaluating LLMs' capability to translate semantic actions into VR device manipulation sequences across 262 scenarios from four popular VR games: Half-Life: Alyx, Into the Radius, Moss: Book II, and Vivecraft. We evaluate twelve LLMs, including GPT-3.5, GPT-4, GPT-4o, GPT-5.1, Gemini-1.5-Pro, Gemini-3-Pro, Claude-Sonnet-4.5, Grok-4, GLM-4-Flash, LLaMA-3-8B, LLaMA-3-70B, and Mixtral-8x7B, compared against annotated ground truth and human performance. Our results reveal that while top-performing models like Gemini-3-Pro demonstrate strong task decomposition capabilities, they still struggle with procedural reasoning and spatial understanding compared to humans. Performance varies significantly across games, suggesting sensitivity to interaction complexity. Few-shot examples substantially improve performance, indicating potential for targeted enhancement of LLMs' VR manipulation capabilities. 
We release all materials at \url{https://sites.google.com/view/combobench}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ComboBench, a benchmark designed to evaluate an LLM's ability to translate high-level actions into VR manipulations across 262 scenarios. The authors evaluate several state-of-the-art LLMs and report that while they excel at task decomposition, they struggle significantly with procedural reasoning and motor mapping. However, the study's conclusions are undermined by a severely rigid evaluation framework that relies on a single ground-truth sequence for each task. This design leads to extremely low scores even for the human baseline (e.g., 1.2% SSM), suggesting the metrics measure similarity to one specific annotation rather than actual task success, thus calling the validity of the reported model capabilities into question.

### Strengths
1. The paper focuses on the crucial transition from "understanding" to "action" for LLMs, specifically how high-level intent is translated into low-level physical device operations. This is a frontier and highly valuable research direction in embodied AI and human-computer interaction, especially within immersive environments like VR.

2. The authors have constructed and committed to releasing the ComboBench dataset. The process for its creation (game selection, expert annotation, LLM-assisted labeling) is clear. This provides a valuable public resource for the community further research.

### Weaknesses
1. The paper’s evaluation framework suffers from severe rigidity problem. As a result, even the human baseline attains extremely low scores (e.g., SSM only 1.2%), strongly suggesting that the metric fails to measure task success and instead measures similarity to a particular annotation. Moreover, the ground truth relies on a single action sequence, while VR tasks typically allow multiple valid solutions. This “single truth” design unfairly penalizes models that produce other valid action sequences, distorting the true assessment of model capability.

2. Comparing models with humans is important to the paper’s argument, but the methodology is flawed. Having human participants read instructions and “write down” action steps is a task of memory retrieval and language expression, not an actual, dynamic interaction in a VR environment. These two tasks rely on different cognitive abilities and constraints, which likely explains the counterintuitive result that “models outperform humans on certain procedural metrics.” Therefore, the current human baseline cannot serve as a valid reference for real-world VR interaction ability.

3. The chosen open-source models (Llama-3-8B, Mixtral-8x7B) differ vastly in parameter scale from state-of-the-art closed-source models (GPT-4o, Gemini-1.5-Pro). While this reflects the current model landscape, directly comparing 8B models to hundred-billion–parameter (or larger) models will mingle “scaling effects” with “architectural differences.” Testing models with similar architectures but different sizes (e.g., Llama-3-70B) would make the conclusions more compelling.

### Questions
1. On the human baseline: Could you clarify the data collection procedure? If participants indeed wrote down steps, how do you justify this disembodied evaluation as a valid proxy for VR interaction ability, especially given the extremely low human scores under this paradigm?

2. What is the specific method or equations that map base evaluation metrics (NSAS, SOP, etc.) to the six cognitive ability scores in Figure 1?

3. Why did you only experiment with temperature = 0, and did you explore the impact of non-zero temperatures on generating valid action sequences?

4. Given the paper’s focus on cognitive abilities, why was there no evaluation of advanced prompting strategies such as Chain-of-Thought and their potential to improve performance?

5. What is the reason for selecting the specific similarity threshold of 0.8387?

### Soundness
2

### Presentation
4

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
This paper proposes a benchmark to evaluate LLMs' ability to play VR games. The benchmark includes 262 scenarios from 4 VR games, and categorizes the actions into different labels for analyzing different aspects of model abilities. Some evaluation metrics are designed for comparing LLM-generated steps versus ground truth. This paper tested 6 LLMs on this benchmark and identified some problems of existing models.

### Strengths
The idea of testing LLM in VR is interesting. Labeling each action with semantics is a good way for analyzing model behaviors.

### Weaknesses
1. **Motivation.** The motivation of letting LLMs manipulate in VR is unclear. Is this benchmark mainly testing LLMs in long-horizon tasks? But there are many other more meaningful tasks like robotics, digital agents, etc to test such capability. Anything unique in this VR setting?

2. **Evaluation Metrics.** The proposed metrics are mainly compare the difference of action sequences between model and ground-truth. However, for such long-horizon tasks, there could be multiple trajectories lead to the success end state. Therefore, having metrics for task success rate is very important, but missing in this paper. It is hard to interpret the results using such step matching metrics. In addition, some results show models perform better than human (Table 1), which means human may do the task in a different way but still achieve the goal, and therefore those metrics cannot reflect the actual capability.

3. **Presentation.** The presentation of this paper is poor, without a single figure to illustrate the idea, no qualitative examples to show what the task looks like, what's model's behavior, etc. Also in table 3, the metric is lower is better, but the authors bold the wrong numbers.

### Questions
What's the actual input to the model, do you provide the visual input (image or video)? If so, some of those tested LLMs are not multimodal, such as Llama-3-8B and Mixtral-8x7B. How did you test those LLMs?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper builds a benchmark to test whether LLMs can translate high-level intent (“raise your hands to surrender”) into precise, environment-dependent device manipulations. The paper is not explicit about motivation, but seems to consider the task inherently interesting. The benchmark covers four popular VR games evaluates six LLMs. Scoring breaks down into (1) semantic understanding, (2) procedural correctness, and (3) device-specific accuracy — all the models are good at (1) and bad at (2) and (3).

The paper goes through various aspects of benchmark construction: game selection, action identification, annotation of VR device manipulations, and LLM-based annotation of the cognitive capabilities applicable to a step. 

Finally, it discusses results. Gemini-1.5-Pro performs best overall, but all models struggles with realistic physics and games requiring nuanced controller inputs. LLMs perform better on games with more consistent, discrete interaction patterns. Few-shot examples help for semantic reasoning, but less so for exact match. Models can decompose tasks but mostly fail at procedural reasoning and motor-action mapping. Gemini-1.5-Pro does somewhat better and shows more variation across tasks. Termination-condition judgment is also weak. Humans beat all the models in sequential order preservation and spatial reasoning, though the models approach human success in more semantic capabiltieis.

### Strengths
### Quality 
- Very much appreciate expert input in deciding on the cognitive capabilities 
- Structure of cognitive capabilities makes sense, though it's hard to know whether it has enough coverage of the space of skills used 
### Clarity
- Writing is clear and easy to follow 
- Results section is particularly well-written, even if figures would help 
### Originality and significance
I'm not aware of a benchmark like this, and work has clearly been put in to make a full-fledged benchmark

### Weaknesses
### Quality 
- What exactly is the motivation for this? It's certainly an interesting and meaningful problem, especially with VR just being a substrate to test LLMs' ability to learn and generate correct fine-grained physical control sequences. However, being explicit about the motivation would help. 
- Second contribution really just feels like a part of the first contribution 
- Would help to have more explanation via examples of the nature of the games, so we have intuition for what is meant by e.g. "more complex interactions" or "more defined interactions". It makes sense that more complex and ambiguous interaction spaces would be harder on LLMs, but the success of the benchmark hinges on this so it needs to be really convincing 
- It ultimately isn't surprising that LLMs do well on more semantic skills and not on more physical ones, nor is it surprising that they fail on more complex problems. The fact that they perform consistently with each other is validating of the bench mark construction, but it doesn't prove its value. Furthermore, VR physics is separate enough from the real world that it's not clear what the value of this benchmark is beyond general ability to take complex and specific actions. While that is valuable, there are many ways to approach it and more justification of this approach would help. 
- It would be very useful to have experiments run with more recent models that have 1) better coherent reasoning capability and 2) if possible, better physical reasoning capability specifically 
### Clarity 
- It's great that experts were involved in choosing the cognitive constructs, but right now section 2.1 reads like a nonspecific process statement. The interview format for the experts is better left to the appendix; it's more important to know what they said and how they approached the selection process. 
- There are very few figures - figures showing examples of interaction spaces and direct visual comparison of stats would both be really useful 

I recommend rejection because ultimately the value of this benchmark has not been proven. What useful insights do we gain that we don't gain from other physical benchmarks, or benchmarks with different levels of complexity, given that VR performance isn't inherently useful?

### Questions
- What do the few-shot examples look like? 
- "Exact match" as a metric seems very stringent - what justifies this? 
- Are the alignment metrics based on embeddings?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces the first benchmark for evaluating the cognitive capabilities of large language models (LLMs) in executing semantically grounded actions for VR device manipulation. Six state-of-the-art LLMs are compared using a suite of cognitively motivated evaluation criteria. The benchmark comprises 262 scenarios, each annotated by human raters and by the LLMs themselves, drawn from four carefully selected and diverse games. Results indicate that Gemini 1.5 Pro achieves the most consistent performance across models, and that few-shot prompting yields substantial improvements in overall model performance.

### Strengths
- Fine-grained decomposition of gameplay actions into 262 scenarios, each annotated by both human raters and LLMs, yielding a rich and reusable dataset.
- Inclusion of open-source models enables a fair and transparent comparative analysis.
- The evaluation metrics are well aligned with the proposed scenarios and are applied creatively to capture the relevant cognitive dimensions.
- Code is open and available

### Weaknesses
- No prompt details, which makes it harder to assess the whole framework's performance
- Missing table reference at line 1072

### Questions
- Could you share a representative example of the exact prompt provided to the models, including any system/instruction text, context, and few-shot exemplars? 
- In Table 1, Gemini 1.5 Pro is not included even though it is reported as the top performer. Could you clarify why it was omitted from that comparison?
- Did you evaluate the models’ generalization to unseen VR games and devices? And do the gains from few-shot prompting persist when exemplars are drawn from games different from those evaluated?

### Soundness
3

### Presentation
4

### Contribution
3
