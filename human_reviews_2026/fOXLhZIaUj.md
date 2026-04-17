# Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions

- Decision: Accept (Poster)
- Scores: 4, 10, 6, 2

## Abstract
Cancer patients are increasingly turning to large language models (LLMs) for medical information, making it critical to assess how well these models handle complex, personalized questions. 
However, current medical benchmarks focus on medical exams or consumer-searched questions and do not evaluate LLMs on real patient questions with patient details. 
In this paper, we first have three hematology-oncology physicians evaluate cancer-related questions drawn from real patients. 
While LLM responses are generally accurate, the models frequently fail to recognize or address false presuppositions} in the questions, posing risks to safe medical decision-making.
To study this limitation systematically, we introduce Cancer-Myth, an expert-verified adversarial dataset of 585 cancer-related questions with false presuppositions.
On this benchmark, no frontier LLM---including GPT-5, Gemini-2.5-Pro, and Claude-4-Sonnet---corrects these false presuppositions more than $43\%$ of the time.
To study mitigation strategies, we further construct a 150-question Cancer-Myth-NFP set, in which physicians confirm the absence of false presuppositions.
We find typical mitigation strategies, such as adding precautionary prompts with GEPA optimization, can raise accuracy on Cancer-Myth to $80\%$, but at the cost of misidentifying presuppositions in $41\%$ of Cancer-Myth-NFP questions and causing a $10\%$ relative performance drop on other medical benchmarks.
These findings highlight a critical gap in the reliability of LLMs, show that prompting alone is not a reliable remedy for false presuppositions, and underscore the need for more robust safeguards in medical AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles a blind spot in medical LLM evaluation: models often sound helpful yet miss or fail to correct false presuppositions in patient questions, which can mislead care. The authors compile 994 cancer myths and build Cancer-Myth, an expert-validated adversarial set of 585 patient-style questions with embedded misconceptions, plus Cancer-Myth-NFP (150 questions without false presuppositions) to gauge over-caution. They introduce a generate–respond–verify pipeline, followed by hematology–oncology physician review and category labeling. Evaluation is standardized with two metrics, PCS and PCR, that score whether answers fully correct the misconception. Using this setup, they show that frontier models and even a strong multi-agent system still struggle here, underscoring presupposition handling as a distinct skill from general medical QA.

### Strengths
- Strong, clinically grounded motivation. The paper highlights a practical and consequential gap in medical reasoning, illustrated with a clear example.  
- Credible data contribution. The dataset appears carefully curated, and the generation–response–verification pipeline is transparent and methodologically sound.

### Weaknesses
- Missing frontier “reasoning-first” baselines with scaled test-time compute. The paper doesn’t evaluate models like OpenAI o1 or DeepSeek-R1, which are explicitly designed to improve as you give them more “think time” (test-time compute) and often change error profiles on reasoning tasks. At minimum, include these (or close open-source proxies) and report PCS/PCR vs. think-time (e.g., few-shot/none, best-of-N, majority vote), so readers can see whether presupposition correction improves with deliberate inference.  
- “Expert-in-the-loop” needs clearer, auditable detail. Spell out the clinical review protocol: expert count and specialties; inclusion/exclusion criteria; annotation guidelines; whether experts had references; inter-rater reliability and adjudication; examples of borderline cases; and any pre/post expert edits to generated questions. Packaging this as a short datasheet/data statement (motivation, composition, collection, labeling, uses, maintenance) would make the dataset easier to trust and reuse.

### Questions
see weakness part

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper studies how well LLMs handle realistic cancer patient questions when those questions contain *false presuppositions* (e.g.  that there is no treatment for advanced lymphoma. The authors show current LLMs often give medically correct information but fail to challenge the false presupposition, which can have the effect of reinforcing the use of such presuppositions in question-answering. They build an expert-verified adversarial dataset to measure this and show that naïve “be cautious” prompting fixes one side of the problem but breaks the other.

Contributions

* Identifies and formalizes detecting and addressing false-presuppositions as a missing capability in medical LLMs for oncology patient Q&A (beyond MedQA-style benchmarks).
* Creates a Cancer-Myth dataset with 585 expert-verified, adversarial, cancer-specific patient questions that embed common cancer myths generated across 7 categories and validated by hematology–oncology physicians.
* Creates a companion Cancer-Myth-NFP set of 150 cancer questions with no false presuppositions for a baseline
* Shows that cutting-edge LLMs fail to fully correct the false presupposition much of the time, and that agent models also fail.
* Precautionary prompting can push correction on Cancer-Myth up but causes (i) a high rate of false positives on Cancer-Myth-NFP and (ii) a relative drop on standard medical benchmarks. This indicates that prompt engineering alone won't fix the problem.
* Reveals a clinically relevant tradeoff—improving myth-detection vs. preserving accuracy on valid patient questions.

### Strengths
Originality

* Frames false presuppositions as an evaluation target—this contributes to a broader set of work that seeks to evaluate beyond simple accuracy to addressing medical bias and logical fallacies that could be reinforced by LLMs and thus cause harm.
* But unlike some of that previous work, it provides an important medical benchmark.
* The choice to have the companion dataset that acts as a benchmark and helps with possibility of over-detection of false presuppositions is non-obvious and is a good touch. 

Quality

* The dataset construction pipeline is careful: start from 994 real cancer myths, to LLM generation, to adversarial filtering, to physician verification. The  human-in-the-loop step makes the benchmark more credible for clinical safety use.
* The evaluation is applies to a broad set of models including agents.

Clarity

* The failure mode is illustrated with a simple, memorable example (advanced lymphoma → “no treatment”) and the paper keeps returning to that pattern, so the reader always knows what “false presupposition” means in context.

Significance

* The problem is clinically meaningful: reinforcing a false “no treatment” belief is more dangerous than giving a slightly incomplete chemo explanation.
* Addresses plausible real-world harm.
* The result that aggressive safety prompting boosts Cancer-Myth result but hurts Cancer-Myth-NFP result highlights an important trade-off
* Framework applies way beyond oncology and medics

### Weaknesses
* Relied on zero-shot prompting, doesn't explore best practice prompting techniques. Including prompts that included exemplars with facts might have impacted whether or not models correct presuppositions.
* No analysis of the reasons for the asymmetries of presupposition performance across categories.

### Questions
* How was the taxonomy for the seven myth categories derived?
* Can you provide any explanation or additional analysis that gives insight into why performance differs across myth categories?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper targets a clinically important failure mode: when patient questions contain hidden, false assumptions, current LLMs tend to accept the premise rather than correct it. The authors first compare LLMs to human social workers on real patient queries and observe that models are often factually sound yet pragmatically unsafe because they do not challenge misconceptions. They then introduce a physician-verified benchmark of cancer-related questions with embedded false presuppositions, plus a matched control set without them, and show that many strong models still miss these corrections. Finally, they demonstrate that common “precautionary” prompting can
improve myth detection but at the cost of over-caution and degraded performance elsewhere.

### Strengths
1. Strong problem framing: focuses on a realistic, safety-critical interaction failure that is underrepresented in exam-style medical benchmarks.
2. Solid evaluation design: includes a no-presupposition control set to quantify over-caution and a category taxonomy that supports actionable error analysis.
3. Good methodological transparency: clear pipeline description and prompt templates; expert involvement to validate that questions actually contain presuppositions and that corrections are medically sound.

### Weaknesses
1. Heavy reliance on a single LLM-as-judge for both dataset curation and scoring risks judge-induced bias; human validation is present but limited in scope.
2. The benchmark is largely synthetic and category-balanced, which aids analysis but may distort real-world prevalence and phrasing of patient misconceptions.
3. The initial human comparison uses a small sample and non-physician baselines, which blunts any strong takeaways about relative human vs. LLM performance in clinical settings.

### Questions
1. Would you be open to a simple, robust judging setup that combines (i) a small mixture of diverse LLM judges, (ii) a friendly, rubric-driven checklist for presupposition detection/clarification with short evidence spans, and (iii) a tiny human-anchored set to calibrate or weight the ensemble? A short note on rank stability under this setup would make the results feel very solid.
2. Could you add light uncertainty to the main metrics (e.g., straightforward intervals/bootstraps and paired comparisons since models see the same items) and a few compact post-hoc views—such as detection vs. correction breakdown and a couple of counterfactual rewrites—to help readers understand where models stumble?
3. What is your plan to share data, prompts, scoring code, and judge/verifier configurations (including seeds/decoding settings) along with a minimal evaluation harness so others can rerun the scoring or swap in alternative judges? Even a brief “evaluation card” with intended use and caveats would be great.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors construct two cancer Q&A datasets, Cancer-Myth, which contains prompts containing misconceptions, and Cancer-Myth-NFP, which contains no false presuppositions. The former contains over 500 prompts while the latter contains just under 200, all LLM-generated. Physicians both verify that prompts contain/do not contain erroneous information, and whether LLM responses correct/do not correct questions.The authors find that even SoTA frontier models cannot correct over 50% of questions containing misconceptions. Adversarial prompts generated by Gemini-1.5-Pro are the most misleading. GEPA optimization improves correction but drops performance on Cancer-NFP and other medical benchmarks.

### Strengths
- The datasets appear to be truly challenging, stumping two advanced LLMs. Physician verification helps legitimize this set. 
- The prompt generation process is straightforward and can be used with any LLM. 
- Well-explained motivation: indeed, providing correct advice without correcting patient misconceptions can be harmful.

### Weaknesses
- In my opinion, this is too small of a dataset contribution to be considered useful for the medical LLM research community. It would have made more sense to make both Cancer NFP and Cancer-Myth the same size. 
- The GEPA optimization does not seem to be properly tested -- the size of training and test are less than nearly all datasets studied in the GEPA paper. 
- 67% identification of questions as human-written versus LLM-generated very clearly indicates that questions appear synthetic overall. This type of distinguishably can be trained on. 
- Gemini-2.5-Pro and GPT-5 are not tested as prompt generators, despite being assessed as evaluators. 
- The evaluation of NFP is a bit unrealistic. It is possible that a question without presuppositions would never be corrected unless the LLM were asked to engage in such labeling.

### Questions
- How were the dataset sizes chosen and why were they not made equal? 
- Were Gemini-2.5-Pro and GPT-5 not tested as prompt generators?
- Is there any correlation between being (un)able to correct misconceptions with general performance on other cancer-related Q&A benchmarks? 
- How does this approach compare to the DyReMe question generation framework (arxiv 2510.09275)?

### Soundness
3

### Presentation
2

### Contribution
1
