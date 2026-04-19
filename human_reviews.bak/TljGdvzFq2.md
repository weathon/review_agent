# Law of the Weakest Link: Cross Capabilities of Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 5

## Abstract
The development and evaluation of Large Language Models (LLMs) have largely focused on individual capabilities. However, this overlooks the intersection of multiple abilities across different types of expertise that are often required for real-world tasks, which we term **cross capabilities**. To systematically explore this concept, we first define seven core individual capabilities and then pair them to form seven common cross capabilities, each supported by a manually constructed taxonomy. Building on these definitions, we introduce *CrossEval*, a benchmark comprising 1,400 human-annotated prompts, with 100 prompts for each individual and cross capability. To ensure reliable evaluation, we involve expert annotators to assess 4,200 model responses, gathering 8,400 human ratings with detailed explanations to serve as reference examples. Our findings reveal that current LLMs consistently exhibit the ``Law of the Weakest Link,'' where cross-capability performance is significantly constrained by the weakest component. Across 58 cross-capability scores from 17 models, 38 scores are lower than all individual capabilities, while 20 fall between strong and weak, but closer to the weaker ability. These results highlight LLMs' underperformance in cross-capability tasks, emphasizing the need to identify and improve their weakest capabilities as a key research priority. The code, benchmarks, and evaluations are available on our [project website](https://www.llm-cross-capabilities.org).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the concept of cross capabilities in LLMs. It defines seven individual and seven cross capabilities, creating a taxonomy to evaluate LLM performance. The authors introduce CROSSEVAL, a benchmark with 1,400 prompts to test both individual and cross capabilities. Results reveal that LLMs often follow the Law of the Weakest Link, where the weakest capability limits performance in cross-capability tasks.

### Strengths
(1) The paper provides a detailed taxonomy of individual and cross capabilities in LLMs.

(2) The CROSSEVAL benchmark is a good contribution. It consists of 1,400 human-annotated prompts that rigorously test LLMs on individual and cross capabilities.

(3) The paper identifies a critical pattern in LLM performance -- the Law of the Weakest Link. This result and finding are insightful.

(4) The paper is well-written. And code and data are shared.

### Weaknesses
(1) Although the paper includes a multilingual aspect (with Spanish), the evaluation of cross capabilities involving languages other than English is limited. 

(2) The study concentrates on evaluating capabilities in pairs, but it does not investigate scenarios requiring the integration of more than two capabilities simultaneously.

(3) Need a detailed analysis of specific errors LLMs make in cross-capability tasks. A breakdown of common failures for particular capability combinations could provide more insights.

### Questions
Can the authors clarify the selection criteria for annotators and their domain expertise? How did the authors address potential biases in human ratings and ensure the reliability of the annotations?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates cross-capabilities - the intersection of multiple distinct abilities across different types of expertise needed to solve complex, real-world tasks. The authors comprehensively define individual and cross-capabilities in LLMs and carefully annotate prompts to evaluate these cross-capabilities. Their findings reveal that current LLMs follow the "Law of the Weakest Link," where cross-capability performance is significantly limited by the weakest component capability. Additionally, they discovered that improving the weakest capabilities leads to the greatest improvements in LLMs' general capabilities.

### Strengths
- The investigation of cross-capabilities is both interesting and important.
- The prompt collection and annotation methodology is comprehensive and reliable.
- The insights about the "Law of the Weakest Link" and the finding that "improving the weakest capabilities leads to the greatest improvements" are particularly valuable.

### Weaknesses
- This work focuses more on benchmarking rather than algorithm design. It would be more appropriately categorized as a data/benchmark paper.
- The study's scale is somewhat limited, with a dataset of only 1,400 prompts.

### Questions
Could you provide more explanation about:
- The rationale for incorporating reference responses in Section 3.3.2
- How this relates to correlation between capabilities?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper constructs a benchmark for the core and cross capabilities. It identifies seven core capabilities and proposes seven cross-capabilities based on them. The author proposes CROSSEVAL to evaluate each capability. Throughout the extensive experiment, the author claims the Law of the Weakrdy link, meaning that the cross capability is heavily influenced by the weakest core capability associated with it.

### Strengths
1. The topic of this paper appears to be interesting. A proper taxonomy of the ability of LLM is necessary.
2. The paper writing is good. The whole paper flow is easy to follow.

### Weaknesses
1. The reviewer fails to understand why constructing prompts for core capabilities. Don't there exist many benchmarks for specific core capabilities?
2. The reviewer doesn't understand why each capability is comparable to the others, considering the difference in prompts.

### Questions
1. What are the criteria for selecting core capabilities and cross capabilities? It appears the these capabilities are somehow randomly selected without proper justification. For instance, why are *English* and *Spanish* selected as core capabilities like *long-context* instead of creating a different dimension named *lingual*?
2. Won't this paper be more suitable for a benchmark track instead of a research paper?
3. How is the image encoded in LLM? Does the LLM include other modalities?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The goal of the paper is to investigate how LLMs perform at the intersection of various capabilities. The paper starts off by remarking that most of the LLM evaluations tend to focus on single capabilities like long context reasoning and factual knowledge retrieval. In practice, however, LLMs may be used to perform tasks that leverage several such capabilities simultaneously. The main contribution of the paper is a meticulously contracted, human-annotated dataset. The dataset consists of prompts that aim to test one or more capabilities. In addition to the input prompts, the paper also asks humans to annotate responses by various SoTA LLMs. Noting the relatively low degree of inter-annotator agreement, the paper improves the annotation guidelines to increase the agreement. Next, the paper design effective prompting strategies for using LLMs as evaluators. The results of the experiments are quite interesting: the performance of the LLMs on compound capabilities correlates with the performance on the individual capabilities.

### Strengths
1. The paper is quite correct in pointing out that benchmarks have not kept up with the actual usage of LLMs in the wild. With that in mind, a systematic approach for studying the cross-capability performance of the models will be very much appreciated by the LLM community.

2. The paper is well-written and quite easy to follow. The paper provides illustrative examples at pretty much every step.

3. The human surveys are quite thorough. The authors spent significant effort in understanding the results and improving the annotation performance.

4. The model-based evaluation protocol in Section 3.3 is very meticulously designed. The protocol exposes many interesting choices (though some are from existing papers) which will be enhance readers' knowledge on conducting effective model based evals.

### Weaknesses
While the paper is a pleasure to read, and generally well-executed, I feel there are two main issues that still stand in the way of acceptance, namely, lack of precise definitions and lack of important details.

## Lack of precise definitions

It is very difficult to understand what counts as a capability, and the paper doesn't not discuss it in sufficient detail. How should a capability be theoretically defined? Are different MMLU categories, e.g., math, medicine, individual capabilities? Is sitting in the SAT exam a single capability? It does require a bunch of skills like recalling meaning of words and basic math formulas, analytical reasoning and equation solving. Having a precise definition of a capability seems quite important in understanding how generalizable the insights of the paper are.

The reader is left asking why certain skills are designated as different capabilities and where really the boundaries are. For instance, why are English and Spanish two different capabilities? Will all languages be different capabilities? What if the languages are closely related to each other, e.g., Hindi & Urdu, German & Dutch?

On a related note, do we assume that the capabilities are equally different from each other? For instance, coding and Image Recognition (different input modalities) seems much more different than English vs. Spanish (same modality, even some overlap of words and grammar). Is there a notion of capability Gerrymandering (akin to [fairness Gerrymandering](https://arxiv.org/abs/1711.05144)) where one could form cross capabilities in various ways to show completely different results?

## Missing details

The paper seems to be missing some critical details which are important to assessing the generalizability of the results.

1. How were the capabilities selected? What was left out? For instance, why not consider Long Context and Language?

2. Also, how are the capabilities broken down into Levels 1 and 2? How exhaustive is this breakdown?

3. How were the examples for annotators generated (Appendix B.4)? Could these example prime the annotators? If these examples are not from the benchmark (as the text notes), what is their propose?

4. Line 201: "selecting a leaf node from our taxonomy to categorize each prompt by capability". Where do these prompts come from?

5. How much coverage do these prompts provide of each capability?

6. Line 211: There are 100 to 500 annotated prompts for each capability, but then why are the reviewers asked to select a precise 100 examples out of it? Is it possible that some capabilities have better quality annotations than others because different sets of prompts were forced to be downsampled to a round number of 100? Why not set a fixed quality criteria and let all the prompts through even if the final number if higher / lower than 100?


### Small nits

3. Line 156: Why is “Brainstorming about Local and Current Things” a part of the Tool Use? Could you provide examples of tools that perform brainstorming?

### Questions
Please see the questions under the "Weaknesses heading".

### Soundness
3

### Presentation
4

### Contribution
2
