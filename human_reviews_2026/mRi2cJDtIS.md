# Learning Facts at Scale with Active Reading

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
LLMs are known to store vast amounts of knowledge in their parametric memory.
However, learning and recalling facts from this memory is known to be unreliable, depending largely on the prevalence of particular facts in the training data and other factors which are poorly understood.
Practitioners are lacking tools which will allow them to ensure that the models learn a given body of knowledge reliably and consistently.
To this end, we propose Active Reading: a framework where we train models to study a given set of material with self-generated learning strategies.
First, we demonstrate models trained with Active Reading on expert domains absorb significantly more knowledge than vanilla finetuning and other data augmentations.
We train expert 8B models that achieve 66% on a Wikipedia-grounded subset of SimpleQA (+313% relative over vanilla finetuning) and 26% on FinanceBench (+160% relative over vanilla finetuning) by applying Active Reading to the source documents for each benchmark.
Finally, we show that Active Reading can be utilized at pre-training scale to build more factual models.
As a demonstration of this, we release WikiExpert-8B, a Wikipedia-expert model trained on 1 trillion generated tokens, which outcompetes models with hundreds of billions of parameters on factual QA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper propose Active Reading: a two-stage synthetic data pipeline where the model first generates diverse learning strategies for a given source document (e.g., timelines, analogies, Q&A, active recall prompts), and then applies each strategy to produce multiple augmentations used for training.

### Strengths
1. The method is clear: first make learning strategies for each document, then apply them to create training data. It covers paraphrase, Q&A, and simple concept linking. It works for general or task-specific goals.
2. The empirical gains are large. On SimpleWikiQA the score reaches about 66 percent. On FinanceBench the score reaches about 26 percent. Both beat repeat, paraphrase, and synthetic-QA under the same token budget.

### Weaknesses
1. When the document pool expands beyond the target subset, performance on the target recall task drops materially. The proposed “fix” (higher LR + heavier pretraining-data mix) is effectively a training patch, not a property of the method. If simply adding more relevant-but-broader documents hurts recall, then Active Reading does not positively scale in the straightforward way the paper implies.
2. Most wins of the proposed method are on long-tail factual recall (e.g., SimpleQA subsets). Improvements on broader QA (NQ/TQA) are incremental. There is little evidence that Active Reading scales to domains requiring multi-hop, compositional reasoning, or other forms of tasks like math reasoning or coding.

### Questions
1. The paper concedes that modern RAG approaches approach gold-context ceilings on these benchmarks. Given (i) the massive synthetic generation + continued pretraining cost and (ii) the modest gains on non-adversarial QA, why is parameterizing all this knowledge inside weights more efficient than maintaining a high-quality retriever+index?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- It is unclear why not simply input all of the four graph representations together as a baseline. Maybe the result will be better?
- The paper lacks a comparison with human expert performance, even on a subset of the tasks. Including such results would help readers understand how far current LLMs have progressed in graph reasoning relative to human capabilities.
- Table 4 shows that different models prefer different graph representations, yet the training of the Selector depends on a specific model. This implies that a new Selector must be re-trained each time when the model changes, which limits the generality and practicality of the proposed approach.
- The description of the baseline in Section 4.2 is confusing. Does Vanilla prompting mean using the same workflow without additional training, or does it refer to using a single model without the agentic framework? Clarifying this distinction would make the experimental setup easier to understand.

### Strengths
- The core idea of this work is both innovative and well-grounded. The proposed Active Reading framework, inspired by human learning behaviors, offers a conceptually sound and intuitively appealing approach to improving factual knowledge acquisition through self-generated data augmentation.

- The work is methodologically solid, with clear experimental design, comprehensive ablation studies, and detailed scaling analyses. The authors provide meaningful comparisons with strong baselines such as paraphrasing and synthetic QA generation, which strengthens the empirical credibility of the findings.

- The scalability and practical relevance of the method are convincingly demonstrated. Applying Active Reading at pretraining scale to develop the WikiExpert-8B model showcases its feasibility for integration into large-scale training pipelines and its potential for building more factual and efficient LLMs.

- The work makes a notable empirical contribution by quantifying factual recall improvements across both general and expert domains. The consistent performance gains and smooth scaling behavior suggest that the proposed approach captures underlying principles of robust knowledge integration rather than dataset-specific effects.

### Weaknesses
- In line 240, the authors state that they add mixed pre-training data to prevent model degradation, but they do not provide experiments to demonstrate the occurrence of such degradation.

- The conclusion in lines 274–276 seems rather obvious. Since SimpleWiki can be viewed as representing long-tail knowledge, while the expanded dataset introduces new knowledge, training on new data may naturally interfere with long-tail knowledge retention. This outcome is not surprising.

- Figure 3 is confusing. It is unclear why the curve for the 4× data condition stops at 5k, the same point as the original dataset.

- The conclusion in lines 308–309 is unsurprising, as the model is trained on a related dataset. Recovering performance in this context seems expected and aligns with intuition.

- The experiments in Section 4.1 are conducted only on the LLaMA-3-8B model, which limits the generalizability of the results.

- Lines 242–250 contain several statements that lack proper citations.

- There are a few typos: in line 235, EntiGraph (Yang ...) is missing a space, and in line 308, NaturalQuestions (Kwiat...) is also missing a space.

- In line 288, it is unclear why the learning rate changes from 1e-5 to 3e-4 instead of to 5e-4 or 1e-4. The rationale for this specific choice is not explained.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Active Reading, a novel framework designed to improve the factual reliability of large language models. The problem is that LLMs often struggle to learn and recall facts, especially from the long tail of their training data. The proposed method has a model "study" a given corpus by first generating a diverse set of learning strategies specific to a document and then applying those strategies to create varied synthetic training data. This process is inspired by human learning techniques. The authors demonstrate that this method substantially improves factual recall on expert domain benchmarks, showing a 313% relative gain on a subset of SimpleQA and a 160% relative gain on FinanceBench compared to standard finetuning. They also scale this approach to create WikiExpert 8B, a model trained on 1 trillion synthetic tokens that surpasses the factual accuracy of much larger models like DeepSeekV2 and Llama 3.1 405B.

### Strengths
1. The Active Reading method is intuitive, scalable, and presents a clever way to generate highly diverse synthetic data by leveraging the model's own capabilities.
2. The empirical results are extremely strong, particularly the performance of the 8B model on Simple WikiQA which nearly matches the gold context baseline.
3. The release of WikiExpert 8B is a significant contribution, as it achieves state of the art factual recall for its size class and provides a powerful, compact model for fact intensive tasks.
4. The scaling analysis in Section 4.2 is very insightful. The discovery that mixing in general pretraining data and using a higher learning rate is necessary to prevent degradation when scaling the knowledge corpus is a valuable finding for the community.

### Weaknesses
1. While the method excels at information extraction, its performance on the full FinanceBench benchmark is notably weaker than the synthetic QA baseline. This suggests the generated strategies may not adequately cover complex reasoning, a point the paper acknowledges but does not fully resolve.
2. The finding in Table 3 that data generated by a 70B model leads to worse performance for an 8B model than its own self generated data is highly counterintuitive. This result is not deeply investigated and raises more questions than it answers.
3. The evaluation is heavily focused on Wikipedia based corpora (SimpleQA, NQ, and the main WikiExpert model). While FinanceBench provides one alternative, demonstrating this method's effectiveness in another distinct domain like medicine or law would strengthen the claims of generality.

### Questions
See Weakness.

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
4

### Summary
This paper presents a simple, effective method for generating corpora for continued pretraining. Their method first involves first prompting the LLM to generate a large set of learning strategies, which specify methods of paraphrasing or summarizing information from a source document. LMs then use such strategies to generate synthetic corpera for pretraining. The authors apply this method to wikipedia and financial analysis documents, demonstrating gains on wikipedia and finance-based QA tasks. The authors then perform extensive analysis, looking into scaling behavior, the effect of training hyperparameters (learning rate, data mix ratios), and the impact of diversity in the synthetically generated corpora.

### Strengths
1. The proposed method is straightforward and demonstrates significant gains compared to baseline methods, in particular for the 8B model.

2. The analysis is thorough and examines a variety of questions about their proposed method. Each analysis section provides valuable insights into why their method works and under what settings it does.

### Weaknesses
1. The results with the 70B model have substantially smaller improvements when compared to the 8B -- improving accuracy on SimpleQA <2% versus ~60% according to Table 3. Given this dramatic difference, it would be helpful to see the full results from the primary settings and some of the analysis repeated with this model. Likewise, experimenting with another base model (non-llama) would also help substantiate these results.

2. Including more randomly selected samples of both task agnostic and task specific strategies would help with understanding what the method is actually doing.

### Questions
1. While performance seems consistent across most guardrail tasks in Table 5, is there any explanation for why performance drops so much on GSM8k and MBPP?

### Soundness
3

### Presentation
4

### Contribution
3
