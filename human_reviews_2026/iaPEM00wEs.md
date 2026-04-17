# When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Large Multimodal Models (LMMs) store vast amounts of pretrained knowledge but struggle to remain aligned with real-world updates, making it difficult to avoid capability degradation when acquiring evolving knowledge. Furthermore, most current work focuses on exploring static textual knowledge injection, neglecting dynamic multimodal evolving knowledge injection, leaving the potential of LMMs for multimodal knowledge injection as an open question. To address this, we first propose a pipeline to construct MMEVOKE, a benchmark for evaluating LMMs' ability in multimodal evolving knowledge injection. MMEVOKE contains 9,422 samples spanning 159 subtypes. Then, based on extensive experiments with MMEVOKE, we reveal challenges such as poor injection performance and capability degradation in existing knowledge injection methods through knowledge injection tests and general capability tests. Finally, to tackle these challenges, we introduce knowledge augmentation and knowledge retention methods, finding that knowledge-aware augmentation strengthens knowledge injection performance, and that Data Replay and MoE methods effectively mitigate capability degradation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a crucial but underexplored challenge for LMMs, i.e., their ability to acquire and retain evolving multimodal knowledge. The authors argue that while LMMs possess vast pretrained knowledge, they struggle to remain consistent with the dynamically changing world. To tackle this, they introduce MMEVOKE, a large-scale benchmark comprising 9,422 multimodal samples spanning 159 subfields, collected from evolving entities and news since 2024. The authors further propose knowledge-aware augmentation (versus naive data augmentation) and retention strategies (e.g., replay and MoELoRA) to mitigate these challenges. They demonstrate that knowledge-aware augmentation improves both adaptation and retention, and replay/MoELoRA effectively alleviate degradation.

### Strengths
1. The work addresses a critical gap in current LMM research, i.e., the handling of evolving multimodal knowledge, which is becoming increasingly relevant as real-world data changes rapidly.
2. The MMEVOKE benchmark is well designed, covering diverse modalities (text and images) and dynamic sources (CNN and Wikipedia). The pipeline is largely automated, reproducible, and designed for continuous updates.
3. The paper evaluates a broad spectrum of existing paradigms (SFT, LoRA, RAG, Web Search, Commercial AI systems) and general capabilities across 12 benchmarks in 7 dimensions, offering a panoramic view of how LMMs behave under evolving knowledge injection.

### Weaknesses
1. The proposed “knowledge-aware augmentation” and “retention” methods largely adapt existing paradigms (e.g., replay, MoE) rather than introducing fundamentally new algorithms. Their novelty lies in application and synthesis, not algorithmic innovation.
2. Despite the automated data pipeline, Step 4 (manual selection) introduces subjectivity and may constrain full automation or large-scale scalability.
3. Evolving knowledge has been widely studied in the context of LLMs, but there is few discussion in the related works and literature review, such as RealtimeQA, DyKnow, EvoWiki, etc.

### Questions
1. Besides CEM/F1, could you provide qualitative or human evaluation on reasoning consistency, hallucination reduction, or factual grounding after injection?
2. Could you elaborate on how GPT-4o’s summarization and augmentation process ensures semantic fidelity and avoids introducing bias?
3. The conclusion hints at multi-stage or hybrid strategies. Have you conducted any preliminary experiments combining augmentation and replay/MoELoRA?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MMEvoke, a benchmark for evaluating LMMs’ ability to incorporate evolving multimodal knowledge. MMEvoke is built via an automated pipeline with manual curation, using images and accompanying text scraped from Wikipedia and CNN across 159 subfields. It evaluates adaptation and retention for SFT, RAG, AI web agents, and Sufficient Context. The authors show that MMEvoke is challenging for current knowledge-injection methods, which also degrade general capabilities (e.g., instruction following, multi-turn QA). Additional studies indicate that knowledge-aware augmentation substantially improves adaptation, whereas knowledge-agnostic augmentation harms it; and that Replay/MoE-LoRA mitigates degradation better than EWC/LwF.

### Strengths
1. MMEvoke leverages real-world data to benchmark LMM’s abilities to adapt to evolving knowledge. The benchmark is comprehensive, containing 9,422 knowledge and covering 159 subfields.
2. The paper conducts a comprehensive evaluation (12 benchmarks) spanning training- and retrieval-based methods, e.g., SFT (Full/LoRA) to RAG, commercial agents, and sufficient context, yielding useful cross-method comparisons.
3. The analysis of knowledge-aware vs knowledge-agonistic augmentation and Replay/ MoELoRA provides practical insights into improving knowledge adaption while preserving  general capabilities.
4. The writing quality is good. It is easy to follow the benchmark pipeline and evaluation details.

### Weaknesses
1. It is unclear why injecting images of new knowledge is necessary, since the text often suffices to answer questions once the model recognizes the person and recalls textual knowledge. In both examples of “Geoffrey Hinton + Nobel Prize” and “Donald Trump + Assassination”, the text provides all the information needed to answer the question. Or, in the case of news for “Region” or “Business” categories, the images often don’t provide closely related information to the text.
2. The work does not differentiate among types of knowledge injection, e.g., (1) **new entities** (Xiaomi SU7), (2) **known entities with new facts** (Geoffrey Hinton + Nobel Prize), and (3) **known entities with conflicting facts** (Lee Jae-myung: party leader vs. president).
3. The pipeline cannot guarantee that knowledge is truly new to the pre-trained LMM, since the article date is not necessarily the first online appearance (e.g., iPhone 16, Xiaomi SU7 existed online before 2024; links below).
4. For RAG, AI web agents, and Sufficient Context, performance is surprisingly low without an accompanying error analysis; failures could stem from (1) failure to recognize the image, (2) incorrect images from Google, or (3) inability to leverage the provided context.

Links:
- https://carnewschina.com/2023/11/15/xiaomis-first-ev-revealed-in-china-to-be-called-xiaomi-su7/
- https://www.forbes.com/sites/davidphelan/2023/11/03/apple-iphone-16-pro-will-bring-remarkable-upgrade-report-claims/

### Questions
1. Among the 9,422 knowledge updates, how many correspond to *known entities* versus *new entities* for the evaluated LMMs?
2. Do you provide a baseline where the LMM is trained **only on text-based knowledge updates** (without images)?
3. How do you ensure that knowledge updates scraped from CNN or Wikipedia do not already exist in the LMM’s **parametric knowledge**, given the counterexamples mentioned?
4. How are the **knowledge-aware variants** constructed for both text and images?
5. What are the causes of the **failure cases** for RAG, AI Web Agent, and Sufficient Context? Can you provide more details for error analysis.
6. How is **Sufficient Context** constructed? Please provide more methodological details on this setup.

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
4

### Summary
In this paper，the authors proposed MMEVOKE for multimodal evolving knowledge, which, serves as a evaluation dataset to measure LMMs’ evolving knowledge injection capabilities.  The authors conduct knowledge injection tests with Supervised FineTuning, Retrieval Augmented Generation, Web Search Engine, and Sufficient Context on MMEVOKE. Based on the experimental results, the authors find that existing methods exhibit poor knowledge adaptation performance and the performance of LMMs remains imperfect even with sufficient context.

### Strengths
1. In this paper，the authors proposed MMEVOKE for multimodal evolving knowledge, which, serves as a evaluation dataset to measure LMMs’ evolving knowledge injection capabilities.  
2. The authors conduct knowledge injection tests with Supervised FineTuning, Retrieval Augmented Generation, Web Search Engine, and Sufficient Context on MMEVOKE. Based on the experimental results, the authors find that existing methods exhibit poor knowledge adaptation performance and the performance of LMMs remains imperfect even with sufficient context.

### Weaknesses
1. In my view, the authors overlook a type of method, knowledge editing, such as ROME ( Locating and Editing Factual Associations in GPT), AnyEdit (AnyEdit: Edit Any Knowledge Encoded in Language Models.) and MEMIT (Mass-Editing Memory in a Transformer). 
2.  In Benchmark construction, the authors compare offline versions of Wikipedia at different time points to identify new entries. But such a way cannot cannot guarantee that these entities will be unfamiliar to LMMs, since LMMs are pre-triained on a much larger dataset than Wikipedia.

### Questions
See in Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work investigates the knowledge retention and adaptation capabilities of LMMs in evolving knowledge. A benchmark named MMEVOKE is constructed to evaluate the performance. Two types of evaluations are conducted: 1) knowledge injection tests, which assess the ability of models to acquire newly introduced knowledge. 2) general capability tests, which examine the ability to preserve previously learned knowledge. Several approaches, including Supervised Fine-Tuning, Retrieval-Augmented Generation (RAG), Commercial AI Web Search Engines, and Sufficient Context Provision, are evaluated for their effectiveness in knowledge injection. Experimental results show that current methods perform poorly on the MMEVOKE benchmark, and LMMs still tend to produce incorrect answers even when sufficient contextual information is provided. Knowledge-aware augmentation demonstrates a clear improvement in knowledge injection performance. Regarding knowledge retention, the study finds that the general capabilities of LMMs degrade after knowledge injection. Methods based on Replay and MoELoRA are shown to effectively mitigate this degradation and help maintain the overall performance of LMMs.

### Strengths
1. The proposed MMEVOKE benchmark serves as the first evaluation dataset designed to measure the evolving knowledge injection capabilities of Large Multimodal Models.
2. This work systematically evaluates a wide range of approaches for their effectiveness in knowledge injection, including Supervised Fine-Tuning, Retrieval-Augmented Generation (RAG), Web Search Engines, and Sufficient Context Provision. The results indicate that knowledge augmentation substantially enhances model comprehension and adaptability in dynamic knowledge environments.
3. This work further examines multiple approaches for knowledge retention, encompassing Replay-based methods, two classical continual learning techniques (EWC and LwF), and MoELoRA. The findings reveal that Direct Rehearsal (Replay) and Structured Separation (MoELoRA) effectively preserve previously acquired knowledge through retraining on historical data and isolating newly injected knowledge, respectively.

### Weaknesses
1. It appears that the proposed concept of evolving knowledge injection is essentially a continual learning problem. The connection between evolving knowledge injection and continual learning remains unclear. A more detailed discussion of this relationship should be included in the Introduction or Related Work sections to better position this study within the broader research context.
2. In the field of continual learning, several recent studies have explored the potential of large multimodal models, e.g., Large Continual Instruction Assistant (ICML 2025), Generative Multi-modal Models are Good Class-Incremental Learners (CVPR 2024). The CL methods evaluated in this work, such as EWC and LwF, are relatively outdated. It would be valuable to evaluate more recent and advanced approaches on the proposed MMEVOKE benchmark and to analyze whether their results align with or challenge the current findings.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
