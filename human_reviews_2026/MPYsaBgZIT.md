# ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Human mobility generation aims to synthesize plausible trajectory data, which is widely used in urban system research. While Large Language Model-based methods excel at generating routine trajectories, they struggle to capture deviated mobility during large-scale societal events. This limitation stems from two critical gaps: (1) the absence of event-annotated mobility datasets for design and evaluation, and (2) the inability of current frameworks to reconcile competitions between users' habitual patterns and event-imposed constraints when making trajectory decisions. This work addresses these gaps with a twofold contribution. First, we construct the first event-annotated mobility dataset covering three major events: Typhoon Hagibis, COVID-19, and the Tokyo 2021 Olympics. Second, we propose ELLMob, a self-aligned LLM framework that first extracts competing rationales between habitual patterns and event constraints, based on Fuzzy-Trace Theory, and then iteratively aligns them to generate trajectories that are both habitually grounded and event-responsive. Extensive experiments show that ELLMob wins state-of-the-art baselines across all events, demonstrating its effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ELLMob, an event-driven human mobility generation framework that leverages large language models (LLMs) to synthesize realistic human trajectories under large-scale societal events. Unlike existing approaches that mainly learn habitual mobility patterns, ELLMob explicitly models the competition between routine behaviors and event-induced disruptions (e.g., natural disasters, pandemics, or major events). Following this, this paper proposed the first human-mobility dataset that is annotated with the event label, and then proposed an LLM-based method with ''reflect–refine'' paradigm for human mobility generation.

### Strengths
(1) The paper tackles event-driven mobility generation, a previously overlooked challenge. While most prior works model routine movements, this study explicitly considers non-routine deviations caused by major events—a critical step toward urban simulations, especially contexted by big social event.

(2) The paper is well-writen and the proposed method is technically sound.

(3) This paper contributes a new dataset that has event-label  along with the human trajectory data.

### Weaknesses
(1) One concern is number of events in the paper, which only covers Typhoon, COVID-19 Pandemic, and Tokyo 2021 Olympics. Though they cover serveral typical patterns of different social events, the number of events is somehow limited. While in real-world, there are lots of  different events. In that sense, the proposed event-dataset and method is somehow limited. How can the dataset  and the model scale up w.r.t the event types?

(2) LLM-based methods can have different output with the same input, it would be better to report the results across different runs to see whether the performance is stable or consitent.

(3) What is the running time at the inference time, since the model requires multiple LLM runs? which could cost a lot time and tokens, especially when simulation for a mega-city is required.

### Questions
Please see the weakness above.

### Soundness
2

### Presentation
4

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
This paper introduces ELLMob, a self-aligned LLM framework for event-driven human mobility generation. It highlights two main gaps in current work: (1) the lack of event-annotated mobility datasets, and (2) the inability of LLM-based trajectory generators to reconcile habitual patterns with event-induced constraints. To address these, the authors construct a Tokyo-based event-annotated dataset (Typhoon Hagibis, COVID-19, Tokyo 2021 Olympics) and propose a Fuzzy-Trace Theory (FTT)-inspired framework that extracts three “gists” (pattern, event, action) and performs iterative “reflection-based alignment” to generate plausible trajectories.

### Strengths
Strength:
S1: The “event-driven mobility” setup moves beyond routine trajectory generation and has real-world applications in urban crisis modeling and planning.
S2: The constructed event-annotated dataset is potentially useful for studying mobility under disruptions. The data collection and anonymization steps are clearly documented.

### Weaknesses
Weakness:
W1: The core of ELLMob lies in a prompt-based reflection loop with heuristic “alignment auditing.” While reflection-based prompting has been extensively studied in the LLM literature, this work does not introduce any new architecture or learning algorithm. Essentially, the contribution remains at the level of prompt engineering rather than methodological contribution and is quite limited. 

W2: Although the proposed dataset includes event annotations (e.g., typhoon, pandemic, Olympics), the data collection process appears to cover only trajectory information, and the event annotation itself seems rather trivial. In this sense, many comparable datasets are already publicly available. Meanwhile, the dataset’s scale is small (only 1,100 users), and it is restricted to the Tokyo area, which further limits its generalizability and practical utility. 
 
W3: Although the paper claims to be “theory-driven,” the use of FTT is largely metaphorical. The separation of gist into pattern-based and event-based forms is very intuitive for this task, but not necessarily derived from the cognitive theory itself. The attributes used to analyze trajectories as “pattern gist” or “event gist” are also intuitive and do not require FTT for guideline or serving as a core component. Meanwhile, it remains unclear for me how FTT concretely improves the model’s reasoning or generation performance.

W4: Baseline choice. It is unclear how event information is incorporated into the baseline methods. In addition, all LLM-based baselines are single-pass models without reflection or iterative refinement. Given that reflection mechanisms are now common in the LLM prompting literature, the comparison should include other iterative-reflection baselines.
 
W5: Backbone choice. All modules rely on proprietary models such as GPT-4 or Gemini for prompting (w/o any open-source LLMs). The reflection loop is not quantified in terms of computational cost or token usage. Reproducibility might be limited due to closed-source LLM dependencies.

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the task of event-driven human mobility generation. To address the shortcomings of existing methods under atypical scenarios, which tend either to replay routine travel or to over-conform to event constraints, this paper constructs an explicitly event-annotated dataset covering three Tokyo events plus a normal period, and proposes ELLMob, a cognition-theory–driven framework. ELLMob first distills long-form news and policy text into a structured event context, then extracts three kinds of gists (pattern, event, and action) and runs an iterative reflect–refine loop that conducts dual auditing for internal alignment  and external alignment, until a trajectory is produced that is both habitually grounded and event-compliant. Using Jensen–Shannon Divergence (JSD) on the distributions of Step Interval (SI), Step Distance (SD), Category Distribution (CD), and Spatial Grid Distribution (SGD) for evaluation, ELLMob significantly outperforms strong baselines across all three events, and ablations confirm that the self-alignment mechanism is the key source of improvement.

### Strengths
(1) This paper introduces the first event-centric, fully featured, explicitly annotated dataset, providing a solid foundation for studying non-routine deviations in mobility.
(2) This paper converts event understanding from free text into a structured event context directly consumable by models, reducing loss of critical information.
(3) This paper proposes the ELLMob framework, which uses gist-based alignment to explicitly reconcile competing mobility decisions.
(4) ELLMob achieves significant improvements over existing methods.

### Weaknesses
(1) The data come exclusively from Twitter/Foursquare, which may skew the sample toward younger, more homogeneous users, introducing selection and behavioral biases.
(2) The paper lacks details about the LLM type and hyperparameters; providing a fuller description would improve clarity and reproducibility.
(3) For the reflect–regenerate procedure, if constraints remain unmet after K iterations, please offer a further remedy or fallback strategy.

### Questions
(1) The study’s geographic and event scope is relatively narrow, which focused solely on Tokyo and three event types within Tokyo, does it have a certain degree of transferability?
(2) The paper does not discuss how the choice of LLM might affect the method’s performance, does model selection materially influence results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces ELLMob, an event-driven human-mobility generator that reconciles a user’s habitual patterns with exogenous event constraints via a self-alignment loop. It also provides a new event-annotated mobility dataset for Tokyo spanning three events. Evaluation shows ELLMob outperforms baselines on the dataset.

### Strengths
-paper is overall well-written and easy to follow

-new dataset with event context has been provided

-the proposed method ELLMob is neat and shows consistent improvements across SI/SD/CD/SGD over the baseline methods. The ablation study shows both the schema and the reflection matter.

### Weaknesses
-methodology contribution is relatively on the incremental side. The contribution is more on applying the concept of self-alignment along with some domain specific heuristic into the human-mobility prediction task.

-evaluation can be more comprehensive:
(1) variance of results are absent
(2) all scenarios are single-city (Tokyo)

### Questions
Can you compare the token usage / latency of ELLMob and the baselines?

### Soundness
3

### Presentation
3

### Contribution
3
