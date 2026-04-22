# GEM-Bench: A Benchmark for Ad-Injected Response Generation within Generative Engine Marketing

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Generative Engine Marketing (GEM) is an emerging ecosystem for monetizing generative engines, such as LLM-based chatbots, by seamlessly integrating relevant advertisements into their responses. At the core of GEM lies the generation and evaluation of ad-injected responses. However, existing benchmarks are not specifically designed for this purpose, which limits future research. To address this gap, we propose GEM-Bench, the first comprehensive benchmark for ad-injected response generation in GEM. GEM-Bench includes three curated datasets covering both chatbot and search scenarios, a metric ontology that captures multiple dimensions of user satisfaction and engagement, and several baseline solutions implemented within an extensible multi-agent framework. Our preliminary results indicate that, while simple prompt-based methods achieve reasonable engagement such as click-through rate, they often reduce user satisfaction. In contrast, approaches that insert ads based on pre-generated ad-free responses help mitigate this issue but introduce additional overhead. These findings highlight the need for future research on designing more effective and efficient solutions for generating ad-injected responses in GEM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GEM-Bench, a benchmark for evaluating ad-injected response (AIR) generation in both chatbot and search settings. It provides three curated datasets and a set of evaluation metrics designed to capture multiple dimensions of user satisfaction and engagement. The authors establish baseline results using a modular multi-agent framework. Experimental findings indicate that prompt-based ad insertion can improve engagement but may compromise accuracy and trust, whereas generate–then–insert approaches yield higher response quality at the cost of increased computational overhead.

### Strengths
- S1. The benchmark covers both conversational and search-oriented use cases, enabling broader applicability across domains.

- S2. The proposed evaluation metrics are multifaceted and reflect several relevant user experience dimensions.

- S3. The multi-agent baseline framework is conceptually simple, easy to extend, and provides a reproducible starting point for future work.

- S4. The experimental study is reasonably comprehensive and clearly illustrates trade-offs between prompt-based and multi-stage insertion strategies.

### Weaknesses
- W1. The evaluation relies entirely on LLM-as-judge scoring without any human annotation or preference validation, making it unclear how well the measured metrics align with real user perception.

- W2. The methodological contribution is relatively limited; the baseline (i.e., Ad-Chat) is straightforward and not previously peer-reviewed, which reduces the novelty.

- W3. Engagement evaluation is based on synthetic CTR estimates rather than real user interaction data, so the results reflect perceived clickability rather than actual conversion behavior.

### Questions
- Q1: For Figure 2, are the shown weaknesses of Ad-Chat due to inherent limitations of prompt-based insertion, or mainly due to the specific prompt used? Were alternative prompt designs evaluated?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces GEM-BENCH, the first comprehensive benchmark framework for Ad-Injected Response (AIR) generation. GEM-BENCH constructs two datasets — MT-Human and LM-Market — for chatbot scenarios, as well as CA-Prod, which simulates the AI overview feature in search engines. The paper also proposes a multi-dimensional evaluation system that covers both user satisfaction and engagement. Experimental results demonstrate that multi-stage ad insertion (Generate–Inject–Rewrite) significantly improves user satisfaction, trust, and personality compared to prompt-based ad injection (Ad-Chat), although at a higher computational cost.

### Strengths
1.This paper is among the first to systematically define, benchmark, and evaluate the task of ad-injected response (AIR) generation for LLM-based systems. It addresses a highly relevant and timely challenge for the monetization of generative AI.
2.The proposed GEM-Bench is comprehensive and thoughtfully constructed, featuring curated datasets from multiple scenarios (chatbot and search) and a multi-faceted evaluation ontology that captures both user satisfaction and engagement.
3.The paper provides a systematic comparison between prompt-based and multi-step ad-injection methods. The analysis offers valuable insights into the inherent trade-offs, explaining the underlying reasons for the performance differences, such as the compromise of accuracy in prompt-based methods and the increased overhead in multi-step approaches.

### Weaknesses
1.The paper does not explicitly define the calculation methodology for key qualitative metrics (e.g., Accuracy, Naturalness) in the main text. While LLM-as-a-Judge is employed, the heavy reliance on this approach without a clear, transparent rubric in the primary description may raise concerns about the objectivity and reliability of the evaluations.
2.The benchmark relies exclusively on multiple LLMs as judges for subjective metrics like user satisfaction. The lack of human validation to corroborate these automated judgments is a significant limitation, as it risks propagating or amplifying biases inherent in the judge models themselves.
3.The chatbot datasets contain only 10 and 100 queries, respectively, which may limit statistical robustness and generalizability to diverse domains.

### Questions
1.How sensitive are the results to the number of ads (k > 1) inserted into each response?
2.Have you conducted any small-scale human evaluation to validate the consistency of LLM-as-a-Judge results?
3.How are factual claims within the injected advertisements themselves verified for correctness? While the paper evaluates factual consistency of the overall response, it remains unclear whether the factual validity of the advertisement content itself is separately verified or controlled.
4.Given the potential ethical concerns surrounding ad injection, are there reasonable mechanisms in place to filter out deceptive or fraudulent ads with misleading inducements?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a benchmark for a newly evolving field of generative engine marketing, whereby ads are generated within an LLM chat history that are supposed to be relevant to user and query. This field is fairly new, and while some models exist for generating such ads, an evaluation is missing.

### Strengths
* well written and easy to understand
* GEM-Bench includes two Ad datasets and metrics
* Several LLM judges are tried and yield consistent results

### Weaknesses
* the fit to a conference on Learning Representations seems fairly low to me. How best to serve ads - while of course a difficult challenge that industry does encounter - seems to me likely to not be of tremendous interest to the core and broader ICLR community.
* The paper therefore also contains limited novelty as there's no new methodology being proposed, apart from using LLMs as a judge for a new benchmark with existing data
* There are potential problems with ethics: steering LLM behavior and thus users in a "subtle [sic]" way might be easily misused. A clear and thorough discussion on ethics **in the main paper** should be included. I can see how such a paper might also be useful to start the academic study of such ad-systems, so the positioning of this seems critical.
* the core metric of click-through-rate is only simulated based on the LLM judge. Without actual cofirmation of this metrics reliability, the results cannot stand on their own.
* results are only presented with niche LLM of doubao-1.5 

* small:
    * Citations should be in \citep and not \cite
    * typos like "EVALUTAION", "Rewritter"

### Questions
* How is this paper relevant to a conference on learning representations?
* Method:
** what if you use varying base LLMs to produce ads, e.g. Qwen, llama3, etc.?
** do your simulated human ratings actually correlate with those of real humans? Across what culture and socio-economic groups? It is unlikely that a single LLM judge will be able to simulate many potential different user groups correctly without further tuning. Analysis here would be direly needed.
*

### Soundness
3

### Presentation
3

### Contribution
2
