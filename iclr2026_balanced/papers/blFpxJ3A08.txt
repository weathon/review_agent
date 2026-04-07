# LPFQA: A LONG-TAIL PROFESSIONAL FORUM## BASED BENCHMARK FOR LLMS’ EVALUATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Large Language Models (LLMs) have made rapid progress in reasoning, question
answering, and professional applications; however, their true capabilities remain
difficult to evaluate using existing benchmarks. Current datasets often focus on
simplified tasks or artificial scenarios, overlooking long-tail knowledge and the
complexities of real-world applications. To address this gap, we propose LPFQA,
a benchmark derived from authentic professional forums across 20 academic and
industrial fields, covering 502 tasks grounded in practical expertise. LPFQA
introduces four key innovations: fine-grained evaluation dimensions that target
knowledge depth, reasoning, terminology comprehension, and contextual analysis; a hierarchical difficulty structure that ensures semantic clarity and unique answers; authentic professional scenario modeling with realistic user personas; and
interdisciplinary knowledge integration across diverse domains. We evaluated 12
mainstream LLMs on LPFQA and observed significant performance disparities,
especially in specialized reasoning tasks. LPFQA provides a robust, authentic,
and discriminative benchmark for advancing LLM evaluation and guiding future
model development.


1 INTRODUCTION


The rise of Large Language Models (LLMs) has been one of the most significant breakthroughs in
the field of artificial intelligence over the past decade, impacting areas such as question answering
Zhuang et al. (2023); Li et al. (2024b), reasoning Havrilla et al. (2024); Wang et al. (2023), code
optimization Nam et al. (2024); Gu (2023); Fakhoury et al. (2024), and beyond. The ability of
LLMs to handle complex tasks has enabled many previously unattainable applications, facilitating
their rapid integration into both daily life and professional domains Yang et al. (2024); Zheng et al.
(2025). As model architectures and training strategies continue to advance, the accurate and comprehensive evaluation of their true performance becomes increasingly crucial. The current approach
involves employing benchmark tests, which are datasets composed of carefully designed questions
or tasks. LLMs are required to generate answers or complete these tasks, and their performance is
then quantitatively assessed based on the outcomes Chang et al. (2024).


Given that a substantial portion of knowledge in the real world follows a long-tail distribution, which
is often fragmented and highly professional, an effective evaluation benchmark should include such
long-tail knowledge that is relatively underrepresented in pre-training data Zhang et al. (2023); Yang
et al. (2022). Moreover, these questions must be grounded in real-world authenticity to better reflect
actual user needs. However, existing benchmarks exhibit clear limitations. For instance, MMLU
focuses primarily on simple question answering or multiple-choice tasks, which fail to evaluate
a model’s ability to handle complex, multi-step reasoning Wang et al. (2024); Hendrycks et al.
(2021); HLE Phan et al. (2025) leverages human annotations to approximate human preferences,
but its task scenarios are often overly idealized or uncommon, thus not representative of typical user
demands. And Arena-Hard Li et al. (2024a), although capturing certain aspects of real user queries,
suffers from limited diversity in question types and insufficient difficulty, making it less effective in
differentiating performance among LLMs.


To this end, we constructed a comprehensive evaluation benchmark (LPFQA) based on highly professional forums, which characterizes both real-world and long-tail knowledge. The data is collected
from technical forums across multiple professional domains. This ensures that tasks of LPFQA are


1


highly professional, as they are based on complex questions raised by real practitioners with expertise in various fields. At the same time, the data is authentic, as it reflects the real needs and challenges encountered by users in practice. We completed this benchmark construction through three
main phases, including (1) data collection and preprocessing, (2) automated question generation and
quality control, and (3) expert verification and difficulty adjustment, ensuring that all selected questions fulfill the demands of the benchmark. LPFQA spans 20 academic fields, including Computer
Science, Mathematics, Biology, Physics, etc., with a total of 505 questions. We evaluated LPFQA
using 12 mainstream models, including GPT, Gemini, DeepSeek, Seed, Qwen, Grok, Claude, and
Kimi.


This work introduces LPFQA, an authentic, structured, and interdisciplinary dataset with long-tail
knowledge for evaluating LLMs’ ability in complex reasoning, providing a robust benchmark for assessing and advancing LLM performance in real-world professional contexts. The main innovations
of LPFQA and contributions of this work can be summarized as follows.


    - **Innovated** **evaluation** **dimension** **design** . We design a set of fine-grained evaluation dimensions, including knowledge depth, reasoning ability, terminology comprehension, and
contextual analysis, ensuring LPFQA ’s comprehensiveness in evaluating LLMs’ capabilities in handling complex tasks.


    - **Hierarchical** **difficulty** **design** **with** **guaranteed** **uniqueness** . We employ a tiered difficulty structure to match varying capabilities of different LLMs, while ensuring semantic
clarity and answer uniqueness for each task, enhancing the reliability, fairness, and discriminative power of LPFQA.


    - **Authentic professional scenario modeling** . We ground questions in authentic use cases by
constructing detailed user personas and realistic contextual scenarios, enhancing the ability
of LPFQA to validate the performance of LLMs in real-world professional environments.


    - **Interdisciplinary knowledge integration** . We integrate long-tail knowledge from diverse
fields, improving the LPFQA’s effectiveness in evaluating LLMs’ integrative capabilities
of judgment and reasoning in complex scenarios.


2 RELATED WORK


The field of large language model evaluation has seen a rapid proliferation of benchmarks, each
designed to probe different facets of model capabilities. Early benchmarks, such as GLUE Wang
et al. (2018) and SuperGLUE Wang et al. (2019), focused on a broad range of general language
understanding tasks, including question answering and natural language inference. While these
benchmarks were instrumental in driving early progress, they are now often considered insufficient
for evaluating the nuanced reasoning and vast knowledge base of modern, more capable LLMs.
Subsequent benchmarks, such as MMLU Wang et al. (2024), BIG-bench Srivastava et al. (2023),
and HELM Liang et al. (2022), extended evaluation to multi-disciplinary knowledge, reasoning,
and holistic dimensions of safety, robustness, and fairness. Despite their contributions, these benchmarks still fall short in capturing the challenges of specialized knowledge and complex reasoning,
motivating the exploration of new evaluation paradigms.


2.1 LONG-TAIL KNOWLEDGE BENCHMARKS


In the real world, data distributions universally exhibit a long-tail characteristic. This implies that a
small number of ”head” categories account for a significant portion of the data, while the vast majority of ”tail” categories are extremely rare. In the context of LLMs, such a distributional imbalance is
crucial because the large pre-training corpora, while massive, often lack sufficient coverage of this
rare, specialized, or infrequently mentioned ”tail” knowledge. As a result, while LLMs demonstrate
robust performance on common topics, their ability to handle this long-tail information can decline
significantly.


To assess a model’s capabilities on long-tail knowledge, researchers have designed specialized
benchmarks. The construction methods for these benchmarks primarily fall into two categories:
the first is natural data collection, where data is obtained directly from the real world. An example
is biodiversity datasets (e.g., iNaturalist Van Horn et al. (2018)), where a large number of species


2


have very few image samples. This approach captures the most authentic distributions, but data
collection is often costly. The second method is synthetic construction, where long-tail distributions
are artificially created by imbalanced sampling from existing, balanced datasets (e.g., ImageNet-LT
from ImageNet Liu et al. (2019)). While this method is straightforward, it may not fully simulate
the complexity and diversity of real-world long-tail data. Although the above benchmarks lay a
foundation for evaluating long-tail knowledge, their tasks are often overly simplistic or confined to
a few specific domains Liang et al. (2022). These limitations underscore the necessity of developing
complementary benchmarks.


2.2 USER-CENTRIC AND CHALLENGING BENCHMARKS


In contrast to static long-tail knowledge evaluation, another important class of evaluation methods
focuses on a model’s performance on dynamic tasks. Chatbot Arena Chiang et al. (2024), for example, is a crowdsourcing platform that evaluates model performance through user blind testing.
Its core idea is to have users engage with two anonymous LLMs and vote for the one that performs better. This method effectively captures user preferences and measures a model’s overall
performance in open-ended conversations. However, crowdsourced evaluation methods like Chatbot Arena also have clear limitations. First, they lack control over specific difficulty or expertise
levels. User-submitted questions can be too simple, leading to similar responses from all top-tier
models, which makes the benchmark less discriminative. For instance, Arena-Hard Li et al. (2024a)
aims to address this issue with adversarial questioning, but its question types can still be relatively
concentrated, making it difficult to fully assess a model’s capabilities on a broader range of complex,
professional long-tail knowledge.


To further test the limits of a model, the Humanity’s Last Exam (HLE) Phan et al. (2025) has
emerged. HLE is designed to test an LLM’s general intelligence and advanced reasoning by collecting extremely difficult questions that even human experts find challenging to answer. These
questions typically require cross-disciplinary knowledge integration, complex logical reasoning, and
deep comprehension. However, this benchmark also has its limitations. While the questions in HLE
are highly challenging, their source and nature may not represent the day-to-day needs of average
users. This makes it less effective in evaluating a model’s practicality in real-world applications.
Furthermore, its extreme difficulty may lead to poor performance from most models, thus limiting
its utility as a regular evaluation tool.


Through the analysis above, we recognize the limitations of existing benchmarks. Long-tail knowledge benchmarks lack consideration for complex tasks, while conversational evaluation benchmarks
are deficient in terms of domain-specific expertise and difficulty control. Extreme benchmarks like
HLE can test a model’s cutting-edge capabilities, but their questions have weak relevance to everyday application scenarios. To bridge these gaps, our work aims to construct a new benchmark that
can effectively evaluate a model’s complex reasoning abilities on professional long-tail knowledge
while also reflecting the demands inherent in real-world scenarios.


3 LPFQA: LONG-TAIL KNOWLEDGE-BASED BENCHMARK


In this section, we begin with an overview of LPFQA, describing its structure and highlighting
its advantages over previous works. Then, we present the detailed steps involved in constructing
LPFQA.


3.1 OVERVIEW


LPFQA is a long-tail knowledge benchmark, which consists of 505 questions across 20 scientific
fields gathered from multiple real professional technical forums, specifically designed for complex
reasoning. The following features can distinguish this benchmark.


**Diversity** **evaluation** **dimension** . The ability to handle complex tasks is critical for LLMs. To
enable the assessment of this ability, LPFQA innovatively covers tasks across multiple evaluation
dimensions, including depth of knowledge, reasoning ability, understanding of professional terminology, and contextual analysis.


3


Figure 1: Pipeline of LPFQA’s construction


**Discriminative** **ability** **and** **unambiguous** **guarantee** . To ensure the validity and accuracy of the
evaluation results, a benchmark must be discriminative enough to differentiate the abilities of various
LLMs, while each task should also be clearly defined. To this end, after careful selection, the tasks
in LPFQA can be categorized into distinct levels of difficulty, designed to reflect characteristics
suitable for LLMs of varying capabilities. Furthermore, the clarity of each task and the uniqueness
of its corresponding answer are guaranteed.


**Derived from real-world scenarios** . To effectively evaluate the response and reasoning capabilities
of LLMs in real-world scenarios, a benchmark must closely reflect the types of questions that users
genuinely encounter. LPFQA is designed with this objective in mind, emphasizing authentic professional tasks derived from real discussions in technical forums. This design ensures that the tasks
are representative of practical situations, thereby enabling a more accurate and realistic evaluation
of LLM performance in real-world applications.


**Diversity domains knowledge** . Moreover, LPFQA integrates tasks from a broad spectrum of professional technical forums, spanning domains such as biology, finance, materials science, and computer science. This cross-disciplinary benchmark challenges LLMs to demonstrate comprehensive
judgment and reasoning across diverse and complex scenarios.


3.2 CONSTRUCTION OF LPFQA


This work develops a fully automated pipeline for constructing such an authentic cross-disciplinary
benchmark from professional technical forums. In detail, the whole construction consists of eight
steps: ❶ collecting professional forums, ❷ scraping discussion links, ❸ capturing screenshots of
discussions, ❹ generating questions from the screenshots using MLLMs, ❺ cleaning up duplicated
and ambiguous items with LLMs, ❻ transitioning them into multiple-choice or short-answer form,


4


❼ verifying all questions by professional experts, and ❽ filtering questions by difficulty through
empirical testing, finally.


These steps can be divided into three phases: data collection and preprocessing, automated question
generation and quality control, and difficulty adjustment and expert review. This three-phase process
follows the natural progression of building a benchmark from raw data to a standardized and highquality benchmark, ensuring both scalability and reliability.


3.2.1 DATA COLLECTION AND PREPROCESSING


**The first phase addresses the challenge of sourcing diverse and representative raw materials** .
We manually selected and crowd-sourced several professional forums that represent different disciplines, ensuring coverage across domains such as biology, finance, materials science, and computer
science (❶). We developed a customized web crawler to collect forum data at scale. The crawler
is capable of adapting to heterogeneous forum structures and supports filtering by metadata such as
time, view count, reply count, and vote count, which helps control both the quality and relevance
of the collected data (❷). To facilitate later multi-modal content analysis, automated scripts visited
each post page and captured screenshots in addition to extracting textual content. This process not
only preserved contextual and visual information but also provided a reliable basis for subsequent
processing (❸).


3.2.2 AUTOMATED QUESTION GENERATION AND QUALITY CONTROL


**The second phase focuses on transforming raw forum content into structured question–answer**
**pairs** . The MLLM first examined each screenshot to determine whether it contained a valid question.
Screenshots without valid questions were discarded, while those with valid content proceeded to the
next stage. If a post included meaningful replies, the model extracted both the question and key
responses to form candidate question–answer pairs; otherwise, only the question itself was retained
(❹).


These items then underwent automated quality control with the aid of an LLM. The process included
duplicate removal, filtering of incomplete or ambiguous entries, and marking with labels such as
domain, clarity, and difficulty. Logical consistency was also checked to ensure alignment between
questions and their corresponding answers (❺).


Finally, the validated question–answer pairs were transmitted into multiple-choice or short-answer
format. For multiple-choice items, the LLM generated distractor options designed to resemble common errors or misconceptions. For short-answer items, in addition to the correct reference answer, a
set of key knowledge points was also provided, which serves as the criterion for determining whether
a response is correct. This transition enhanced the usability of the dataset while maintaining both
clarity and evaluation effectiveness (❻).


3.2.3 EXPERT VERIFICATION AND DIFFICULTY ADJUSTMENT


**The** **third** **phase** **ensures** **that** **the** **question** **bank** **achieves** **a** **balanced** **level** **of** **difficulty** **and**
**scientific correctness.** First, the generated items underwent a human verification by the professional
experts. They verify the factual accuracy, relevance, and difficulty of each item, while also correcting
residual errors introduced during the automated pipeline. This operation enhanced the scientific rigor
and reliability of our benchmark (❼).


Finally, to improve the benchmark’s ability to differentiate LLMs’ capabilities, we conduct an empirical difficulty test. Multiple LLMs were employed to answer all questions, and their accuracy
rates were recorded to classify the items into different difficulty levels. The dataset was adjusted by
selectively adding or removing items, ensuring a well-balanced difficulty structure (❽).


By integrating the above steps, namely data collection and preprocessing, automated question generation with quality control, and difficulty adjustment with expert review and empirical test-based
evaluation, the proposed pipeline achieves end-to-end automation while maintaining high standards
of reliability and evaluation utility. This design provides a scalable and systematic approach for
constructing a question dataset that faithfully represents real-world professional discourse and is
well-suited for LLM evaluation.


5


As depicted in Figure 2, LPFQA covers 20 academic fields with a total of 505 questions, including
_Computer Science_ (CS), _Mathematics_ (Math), _Biology_ (Bio), _Physics_ (Phys), _Electronic Information_
_Engineering_ (EIE), _Chemistry_ (Chem), _Electronic_ _Science_ _and_ _Technology_ (EST), _Finance_ (Fin),
_Mechanical and Automation_ (Mech), _Artificial Intelligence and Machine Learning_ (AI), _Computer_
_Systems and Software_ (CSS), _Miscellaneous_ (Misc), _General Engineering_ (Eng), _Aerospace_ (Aero),
_Law_, _Medical_ (Med), _Data_ _Science_ _and_ _Big_ _Data_ _Technology_ (DS), _Energy_ (En), _Electronics_ _and_
_Information Science_ (EIS), and _Information and Communication Engineering_ (ICE). Among them,
_Physics_, _Mathematics_, and _Biology_ contain the largest number of items, each exceeding 60, while
most of the other fields fall within the 10–50 range, and the field of _Data_ _Science_ _and_ _Big_ _Data_
_Technology_ has a relatively smaller number, with 3 items.


4 EXPERIMENTS


Based on LPFQA, we evaluate the following mainstream models: Qwen-3-235B Yang et al. (2025),
Grok-4 xAI (2025), DeepSeek-R1 Guo et al. (2025), Seed-1.6-Thinking Volcengine (2024), Gemini2.5-Pro Comanici et al. (2025), GPT-4.1 OpenAI (2024a), GPT-4o OpenAI (2024b), o3-high OpenAI (2024c), Claude-4-Sonnet Anthropic (2024), GPT-5 OpenAI (2025), Kimi-K2 Team et al.
(2025), and DeepSeek-V3 Liu et al. (2024). All results provided are averaged over three trials.


6


60


40


20


0


Figure 2: Quality distribution of each field in LPFQA


Table 1: Performances of different models on
LPFQA.


**Models** **Score**
Qwen-3 38.78
Grok-4 39.04
DeepSeek-R1 38.25
Seed-1.6 41.50
Gemini-2.5-Pro 44.42
GPT-4.1 38.31
GPT-4o 32.40
o3-high 43.03
Claude-4 38.05
GPT-5 **47.28**
Kimi-K2 35.26
DeepSeek-V3 32.60
**Average** 39.08


3.3 STATISTICS OF LPFQA


Table 2: Scores of different models on filtered
LPFQA.


**Models** **LPFQA** _[−]_ **LPFQA** [=]

Qwen-3 44.65 42.62
Grok-4 44.95 42.37
DeepSeek-R1 44.04 41.89
Seed-1.6 47.78 45.84
Gemini-2.5-Pro 51.15 49.64
GPT-4.1 44.11 42.45
GPT-4o 37.31 35.03
o3-high 49.54 48.10
Claude-4 43.81 41.57
GPT-5 **54.43** **53.11**
Kimi-K2 40.60 38.58
DeepSeek-V3 37.54 35.59
**Average** 44.99 43.07


|M DS|Med Math|
|---|---|
|Mech<br>Law<br>g<br>s<br>4<br>22.<br>29.19<br>28.87<br>44.47<br>43.62|IC<br>Misc<br>Chem<br><br><br>~~**41.75**~~<br>33.43<br>25.06<br>34.13<br>33.43<br>1.48<br>33|
|io<br>EIS<br>EIE<br>EST<br><br>39.36<br>56.70<br>26.70<br>42.04<br>4|Aero<br>CS<br>CSS<br>Fi<br><br><br>8.22<br>45.88<br>32.04<br>37.19<br>35.92|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>42<br>11.<br>35.44<br>20.00<br>37.80<br>46.56|IC<br>Misc<br>Chem<br><br><br>~~**41.63**~~<br>38.14<br>52.13<br>38.58<br>47.94<br>.64<br>00|
|io<br>EIS<br>EIE<br>EST<br><br><br>39.30<br>53.30 40.00<br>26.07<br>29|Aero<br>CS<br>CSS<br>F<br><br><br>.56<br>54.13<br>30.8138.00<br>39.77|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>43.<br>44.33<br>18.75<br>20.00<br>51.20<br>43.66|IC<br>Misc<br>Chem<br><br><br>~~**29.13**~~<br>33.43<br>43.69<br>31.53<br>43.75<br>18|
|io<br>EIS<br>EIE<br>EST<br><br><br>38.80<br>43.30<br>39.90<br>38.35<br>37.|Aero<br>CS<br>CSS<br>F<br><br>0041.75<br>41.04<br>34.12<br>33.27|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>44.<br>33.33<br>18.81<br>20.07<br>37.73<br>40.66|IC<br>Misc<br>Chem<br><br><br>~~**49.88**~~<br>38.00<br>41.69<br>43.87<br>47.94<br>84|
|io<br>EIS<br>EIE<br>EST<br><br><br>51.41<br>40.00<br>56.70 46.3548.|Aero<br>CS<br>CSS<br>Fi<br><br>11<br>58.38<br>34.65<br>30.98<br>35.92|


|M DS|Med Math|
|---|---|
|Mech<br>Law<br>g<br>s<br>4<br>11<br>27.06<br>31.07<br>33.40<br>52.49|IC<br>Misc<br>Chem<br><br><br>~~**54.13**~~<br>42.86<br>41.69<br>43.03<br>56.25<br>1.00<br>.00|
|io<br>EIS<br>EIE<br>EST<br><br>51.38<br>43.30<br>63.30<br>46.374|Aero<br>CS<br>CSS<br>Fi<br><br>4.44<br>41.63<br>40.96<br>34.81<br>47.46|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>40<br>33.33<br>27.19<br>20.00<br>35.60<br>35.76|IC<br>Misc<br>Chem<br><br><br>~~**50.00**~~<br>33.29<br>25.00<br>35.97<br>45.81<br>.98|
|io<br>EIS<br>EIE<br>EST<br><br><br>45.87<br>23.30<br>66.60<br>52.17<br>25|Aero<br>CS<br>CSS<br>F<br><br>.89<br>37.50<br>34.65<br>32.53<br>33.35|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>27.<br>22.33<br>29.19<br>11.13<br>33.33<br>39.22|IC<br>Misc<br>Chem<br><br><br>~~**54.13**~~<br>28.57<br>18.69<br>38.61<br>37.50<br>89<br>|
|io<br>EIS<br>EIE<br>EST<br><br><br>46.98<br>16.60<br>53.30 42.02<br>18.|Aero<br>CS<br>CSS<br>F<br><br>44<br>25.00<br>17.92<br>17.07<br>24.38|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>45.<br>33.33<br>43.75<br>24.40<br>46.53<br>52.49|IC<br>Misc<br>Chem<br><br><br>~~**62.50**~~<br>66.71<br>50.00<br>46.50<br>39.56<br>31|
|io<br>EIS<br>EIE<br>EST<br><br><br>38.25<br>43.30<br>46.60<br>47.13<br>33.|Aero<br>CS<br>CSS<br>Fi<br><br>33<br>29.13<br>30.73<br>37.26<br>34.58|


|M DS|Med Math|
|---|---|
|Mech<br>Law<br>g<br>s<br>4<br><br>27.06<br>17.80<br>31.07<br>39.71|IC<br>Misc<br>Chem<br><br><br>~~**37.50**~~<br>33.43<br>22.94<br>40.34<br>35.44<br>1.52<br>0.00|
|io<br>EIS<br>EIE<br>EST<br><br><br>46.43<br>40.10<br>66.70<br>44.96<br>1|Aero<br>CS<br>CSS<br>Fi<br><br><br>8.56<br>50.00<br>29.50<br>32.56<br>38.46|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>47<br>33.33<br>35.44<br>13.33<br>48.93<br>60.79|IC<br>Misc<br>Chem<br><br><br>~~**70**~~<br>61.86<br>52.06<br>49.16<br>43.69<br>.51|
|io<br>EIS<br>EIE<br>EST<br><br><br>41.52<br>60.00<br>50.00<br>47.8544|Aero<br>CS<br>CSS<br>F<br><br>.44<br>29.13<br>50.00<br>45.70<br>37.15|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>37.<br>0.<br>22.88<br>17.80<br>35.53<br>41.68|IC<br>Misc<br>Chem<br><br><br>~~**33.38**~~<br>38.14<br>43.81<br>34.18<br>25.00<br>67<br>0|
|io<br>EIS<br>EIE<br>EST<br><br><br><br>43.13<br>36.80<br>66.60<br>39.85<br>44.|Aero<br>CS<br>CSS<br>F<br><br><br>56<br>41.75<br>24.42<br>19.3529.54|


|Ma DS|Med ath|
|---|---|
|Mech<br>Law<br>g<br>s<br>28.<br>0.<br>18.75<br>11.13<br>31.13<br>42.15|IC<br>Misc<br>Chem<br><br><br>~~**45.88**~~<br>19.00<br>35.38<br>29.82<br>29.19<br>93<br>0|
|io<br>EIS<br>EIE<br>EST<br><br><br><br>43.69<br>16.60<br>53.30<br>34.80<br>18.|Aero<br>CS<br>CSS<br>Fi<br><br><br>44<br>37.50<br>23.12<br>27.09<br>35.88|


|M DS|Med Math|
|---|---|
|Mech<br>Law<br>ng<br>s<br>4<br>38.8<br>27.79<br>32.49<br>32.22<br>35.47|IC<br>Misc<br>Chem<br><br><br>~~**42.33**~~<br>39.44<br>52.48<br>19.63<br>38.89<br>0.98<br>1|
|io<br>EIS<br>EIE<br>EST<br><br>34.25<br>37.68<br>47.5543.84<br>2|Aero<br>CS<br>CSS<br>F<br><br><br>0.36<br>40.2544.90<br>38.90<br>40.46|


|Ma DS|Med ath|
|---|---|
|Mech<br>aw<br><br>47.<br>(G<br>44.33<br>(R1)<br>43.75<br>(o3)<br>31.07<br>(Ge.)<br>51.20<br>(R1)<br>60.79<br>|I<br>Misc<br>Chem<br><br><br>70.<br><br>66.71<br>(o3)<br>52.13<br>(Gro.)<br>49.16<br>(G.5)<br>56.25<br>(Ge.)<br>51<br>.5)|
|EIS<br>EIE<br>EST<br><br><br>(G.5)<br>51.41<br>(Seed)<br>60.00<br>(G.5)<br>66.70<br>(Cla.)<br>52.17<br>(G4.1)<br>48.<br>(Qw|Aero<br>CS<br>CSS<br>F<br><br>~~(G.~~<br>22<br>e.)58.38<br>(Seed)<br>50.00<br>(G.5)<br>45.70<br>(G.5)<br>47.46<br>(Ge.)|


|M DS|Med Math|
|---|---|
|Mech<br>Law<br>g<br>s<br>0<br>(C<br>18.75<br>(R1)<br>11.13<br>(G4o)<br>31.07<br>(Cla.)<br>35.76<br>|IC<br>Misc<br>Chem<br><br><br>29.13<br><br>19.00<br>(V3)<br>18.69<br>(G4o)<br>29.82<br>(V3)<br>25.00<br>(Kim.)<br>27.89<br>(G4o)<br>.00<br>la.)|
|o<br>EIS<br>EIE<br>EST<br><br>(G4.1)<br>38.25<br>(o3)<br>16.60<br>(G4o)<br>26.70<br>(Qwe.)<br>26.0<br>(Gro.|Aero<br>CS<br>CSS<br>Fi<br><br>~~(R1)~~<br>7<br>)<br>18.44<br>(G4o)25.00<br>(G4o)<br>17.92<br>(G4o)<br>17.07<br>(G4o)<br>24.38<br>(G4o)|


Figure 4: Average, Maximum, and Minimum scores.


4.1 MAIN RESULTS


As shown in Table 1, the performance of the evaluated models on LPFQA falls within a relatively
narrow range, with scores spanning from 32.40 to 47.28. Among them, GPT-5 achieves the highest
score, while GPT-4o records the lowest. To provide a more fine-grained comparison, Figures 3 report
the scores of individual models across different fields, offering a clearer picture of their strengths
and weaknesses in specific areas. The overall average performance of all models is further summarized in Figure 4a, which provides a holistic perspective on their general capability across fields.
Finally, to highlight the comparative extremes, Figures 4b and 4c identify the models that achieve
the maximum and minimum scores in each field, thereby providing an intuitive view of their relative
advantages and limitations.


7


~~A~~ I


(b) Grok-4


~~A~~ I


(c) DeepSeep-R1


(a) Qwen-3


(d) Seed-1.6


~~A~~ I


(g) GPT-4o


(e) Gemini-2.5-Pro


(h) OpenAI-o3-high


(f) GPT-4.1


(k) Kimi-K2


~~A~~ I


(l) DeepSeep-V3


(i) Claude-4-Sonnet


(j) GPT-5


Figure 3: Scores of models on different fields of LPFQA.


Bio


Phys


~~A~~ I


(a) Average scores


(b) Max score models


(c) Min score models


|64 LPFQA<br>5958 60 LPFQA =|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|Col30|Col31|Col32|Col33|Col34|Col35|Col36|Col37|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|59<br>64<br>~~58~~<br>60<br>LPFQA<br>~~LPFQA ~~=|59<br>64<br>~~58~~<br>60<br>LPFQA<br>~~LPFQA ~~=|59<br>64<br>~~58~~<br>60<br>LPFQA<br>~~LPFQA ~~=|59<br>64<br>~~58~~<br>60<br>LPFQA<br>~~LPFQA ~~=|59<br>64<br>~~58~~<br>60<br>LPFQA<br>~~LPFQA ~~=|59<br>64<br>~~58~~<br>60<br>LPFQA<br>~~LPFQA ~~=||||||||||||||||||||||||||||||||
|52<br>49|52<br>49|52<br>49|52<br>49|||||~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|~~34~~<br>~~38~~<br>~~35~~<br>33<br>~~37~~<br>~~34~~<br>|
|52<br>49|52<br>49||||||||||||||||||||||||||||||||||||
|52<br>49|52<br>49||||||||||||||||||||||||||||||||||||
|21<br>21|21<br>21|||||||||||||20<br>19|20<br>19|20<br>19|20<br>19|20<br>19|20<br>19||||||||||||||||||
|||||||||9<br>9|9<br>9|||||||12<br>8<br>12<br>8|12<br>8<br>12<br>8|12<br>8<br>12<br>8|12<br>8<br>12<br>8|||~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|~~15~~<br>12<br>6<br>10<br>~~14~~<br><br>9<br>8<br>7<br>~~15~~<br>11<br><br>10<br>13<br><br>9<br>8<br>7|
|||||||||||||||||||||||||||5|||||3<br>3||||||


Figure 5: Quality distribution of each field in filtered LPFQA


Based on the results presented in Figures 3 and 4, we analyze the performances of these models
from three perspectives: overall performance, disciplinary distribution, and extreme values across
models.


    - **Overall performance** . Among all evaluated systems, DeepSeek-V3 demonstrates the most
balanced and consistent performance across disciplines, with no apparent weaknesses, and
can thus be regarded as the overall best-performing model. GPT-5 exhibits strong competitiveness, achieving the highest scores in several domains such as AI, Phys, EIS, Chem,
Fin, and CSS, in some cases surpassing DeepSeek-V3. Seed-1.6 and GPT-4.1 also achieve
competitive results in specific domains (e.g., CS, Aero, Bio for Seed-1.6; EIT, En for GPT4.1), though their overall performance remains less comprehensive. Other models, such as
Claude-4-Sonnet, Grok-4, and Kimi-K2, tend to show domain-specific strengths but also
exhibit noticeable weaknesses, limiting their overall robustness.


    - **Disciplinary perspective** . From a disciplinary perspective, clear differences emerge across
fields. As shown in Figure 4a, Misc yields the highest average scores (above 50), while En
records the lowest overall average (around 20). Other relatively strong domains include
Chem, AI, Fin, CS, and EIS, while weaker performance is observed in Med, Law, Eng,
and Bio. Intra-model variation is also significant. For example, DeepSeek-R1 attains leading scores in DS, Math, Eng, and Law, but remains comparatively weak in ICE. Similarly,
GPT-5 shows clear superiority in Phys and AI, while its performance in Law is less competitive. These disparities indicate that current models continue to face challenges in achieving
uniform cross-disciplinary generalization.


    - **Max and Min scores** . To provide a comprehensive view beyond average performance, we
examine maximum and minimum scores across all disciplines (Figures 4b and 4c). For
maximum scores: AI, Phys, EIS, Chem, Fin, and CSS are led by GPT-5; CS, Aero, and
Bio by Seed-1.6; DS, Math, Eng, and Law by DeepSeek-R1; EIT and En by GPT-4.1; EIE
by Claude-4-Sonnet; ICE by OpenAI-o3-high; and Misc by Grok-4. For Minimum scores:
GPT-4o accounts for the lowest performance in multiple domains (Math, Chem, Fin, CSS,
CS, Aero, En, and EIS). Other models show more localized weaknesses: Claude-4-Sonnet
in DS and Eng, DeepSeek-R1 in Mech and ICE, OpenAI-o3-high in Bio, Qwen-3 in EIT,
Grok-4 in EIE, Kimi-K2 in Med, and DeepSeek-V3 in Misc.


4.2 DETAIL ANALYSIS


4.2.1 FILTERED LPFQA


During our analysis, we observed that none of the evaluated models could correctly answer a subset
of questions. Since one of the primary purposes of the benchmark is to differentiate the capabilities of different models, these questions provide little discriminatory value. Therefore, we first
excluded them from LPFQA, leaving a remaining set of 436 items. This filtered version, denoted as
LPFQA _[−]_, was then used to recalculate the distribution of questions across different fields (Figure 5)
and the corresponding scores of each model (Table 2).


8


Table 4: Configured with search tool


**Models** **Score** ∆
Qwen-3 23.31 15.47% _↓_
DeepSeek-R1 33.60 4.65% _↓_
Seed-1.6 37.58 3.92% _↓_
Gemini-2.5-Pro 35.19 9.23% _↓_
GPT-4.1 36.32 1.99% _↓_
GPT-4o 32.60 0.20% _↑_
o3-high 42.71 0.32% _↓_
GPT-5 45.18 2.10% _↓_
Kimi-K2 35.52 0.26% _↑_
DeepSeek-V3 28.08 4.51% _↓_
**Average** 35.01 10.64% _↓_


Table 3: Configured with code interpreter tool


**Models** **Score** ∆
Qwen-3 35.89 2.89% _↓_
DeepSeek-R1 34.46 3.79% _↓_
Seed-1.6 36.85 4.65% _↓_
Gemini-2.5-Pro 34.46 9.96% _↓_
GPT-4.1 36.12 2.19% _↓_
GPT-4o 30.28 2.12% _↓_
o3-high 42.76 0.37% _↓_
GPT-5 48.01 0.73% _↑_
Kimi-K2 36.12 0.86% _↑_
DeepSeek-V3 28.42 4.18% _↓_
**Average** 36.15 7.75% _↓_


In addition, we identified another subset of questions that were answered correctly by all models
without exception. While such questions may reflect fundamental or widely shared knowledge, they
also contribute minimally to distinguishing the relative strengths and weaknesses of the models. To
further emphasize the performance gaps, we excluded these universally solvable questions based
on LPFQA _[−]_, resulting in a remaining set of 421 items. This second filtered version is denoted as
LPFQA [=], on which we recomputed both the distributions across different fields (Figure 5) and the
model scores (Table 2).


4.2.2 ABLATION ANALYSIS


**Does LPFQA evaluate knowledge or reasoning ability?**


We investigated the effect of integrating a Jupyter Code Interpreter (CI) into the reasoning process,
which is expected to enhance reasoning ability through code execution. However, as shown in
Table 3, it can be observed that overall performance decreased: the scores dropped on most models,
and the few improvements that appeared were marginal, leading to a lower overall average. These
findings suggest that LPFQA primarily reflects a model’s mastery of domain knowledge rather than
its reasoning ability.


**Is deep-search always rewarding?**


We incorporated GoogleSearch and TextBrowserView tools into the reasoning process to enable
information retrieval. As shown in Table 4, the scores of most models decreased under this setting.
We attribute this phenomenon to the nature of LPFQA, which consists of long-tail knowledge that
is inherently difficult to retrieve from the web. In such cases, the additional retrieval functions may
introduce misleading information during the reasoning process, thereby reducing overall inference
accuracy. In other words, for tasks involving long-tail knowledge, simply augmenting models with
online search does not provide a positive effect and may even be detrimental. This observation offers
valuable insights into the limitations faced by all models when dealing with long-tail knowledge.


5 CONCLUSION


In this work, we proposed LPFQA, a long-tail professional forum-based benchmark designed to
evaluate LLMs on complex reasoning and specialized knowledge across 20 domains. LPFQA emphasizes authenticity, interdisciplinarity, and fine-grained evaluation dimensions, with hierarchical
difficulty and expert verification ensuring reliability and fairness. Our experiments on 12 mainstream
LLMs reveal notable disparities, highlighting the persistent challenge of long-tail knowledge. Furthermore, ablation studies show that LPFQA primarily reflects domain knowledge mastery, and that
direct integration of external tools does not always enhance performance. Overall, LPFQA provides
a robust, discriminative, and authentic benchmark that not only measures current model capabilities
but also guides future research toward more generalizable and reliable LLMs.


9


ETHICS STATEMENT


This study is based on publicly available professional forum data, which was collected, filtered,
and processed in compliance with relevant ethical standards. No personally identifiable or sensitive
information was included in the benchmark. All data used were anonymized and only retained
for research purposes. The benchmark construction and experiments were conducted strictly for
academic evaluation and model analysis, without any intention of infringing on privacy, spreading
harmful content, or causing potential misuse. We affirm that this research adheres to the ethical
principles of fairness, transparency, and responsible AI development.


REPRODUCIBILITY STATEMENT


To foster transparency and facilitate reproducibility, we will release our benchmark to the public.
Furthermore, we provide the details of the benchmark construction process in the appendix, including: (1) all prompts used for question generation, (2) the prompts applied for evaluation criteria, and
(3) the complete list of forums utilized. We believe these resources will enable the community to
faithfully reproduce our results and build upon our work.


REFERENCES


Anthropic. Claude-4-sonnet. [https://www.anthropic.com/news/claude-4, 2024.](https://www.anthropic.com/news/claude-4) Accessed: 2025-09-17.


Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan
Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. _ACM_
_transactions on intelligent systems and technology_, 15(3):1–45, 2024.


Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li,
Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot
arena: An open platform for evaluating llms by human preference. In _Forty-first_ _International_
_Conference on Machine Learning_, 2024.


Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit
Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the
frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _arXiv preprint arXiv:2507.06261_, 2025.


Sarah Fakhoury, Aaditya Naik, Georgios Sakkas, Saikat Chakraborty, and Shuvendu K Lahiri. Llmbased test-driven interactive code generation: User study and empirical evaluation. _IEEE Trans-_
_actions on Software Engineering_, 2024.


Qiuhan Gu. Llm-based code generation method for golang compiler testing. In _Proceedings of the_
_31st ACM Joint European Software Engineering Conference and Symposium on the Foundations_
_of Software Engineering_, pp. 2201–2203, 2023.


Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025.


Alex Havrilla, Sharath Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravynski,
Eric Hambro, and Roberta Raileanu. Glore: when, where, and how to improve llm reasoning via
global and local refinements. In _Proceedings_ _of_ _the_ _41st_ _International_ _Conference_ _on_ _Machine_
_Learning_, pp. 17719–17733, 2024.


Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob
Steinhardt. Measuring massive multitask language understanding. In _International_ _Conference_
_on Learning Representations_, 2021.


Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and
benchbuilder pipeline. _arXiv preprint arXiv:2406.11939_, 2024a.


10


Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, and Jianyong Wang. Flexkbqa: A flexible llm-powered framework for few-shot knowledge base question
answering. In _Proceedings_ _of_ _the_ _AAAI_ _conference_ _on_ _artificial_ _intelligence_, volume 38, pp.
18608–18616, 2024b.


Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language
models. _arXiv preprint arXiv:2211.09110_, 2022.


Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. _arXiv preprint_
_arXiv:2412.19437_, 2024.


Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Largescale long-tailed recognition in an open world. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _conference_ _on_
_computer vision and pattern recognition_, pp. 2537–2546, 2019.


Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad Myers. Using
an llm to help with code understanding. In _Proceedings_ _of_ _the_ _IEEE/ACM_ _46th_ _International_
_Conference on Software Engineering_, pp. 1–13, 2024.


OpenAI. Gpt-4.1. [https://openai.com/index/gpt-4-1/,](https://openai.com/index/gpt-4-1/) 2024a. Accessed: 2025-0917.


OpenAI. Gpt-4o system card. [https://openai.com/index/gpt-4o-system-card/,](https://openai.com/index/gpt-4o-system-card/)
2024b. Accessed: 2025-09-17.


OpenAI. Introducing o3 and o4-mini. [https://openai.com/index/](https://openai.com/index/introducing-o3-and-o4-mini/)
[introducing-o3-and-o4-mini/, 2024c.](https://openai.com/index/introducing-o3-and-o4-mini/) Accessed: 2025-09-17.


OpenAI. Introducing GPT-5. Online publication, 2025. [URL https://openai.com/index/](https://openai.com/index/introducing-gpt-5)
[introducing-gpt-5.](https://openai.com/index/introducing-gpt-5)


Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin
Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. _arXiv_ _preprint_
_arXiv:2501.14249_, 2025.


Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Shoeb, Abubakar Abid, Adam Fisch,
Adam R Brown, Adam Santoro, Aditya Gupta, Adri Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _Transactions_ _on_
_machine learning research_, 2023.


Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen,
Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. _arXiv_
_preprint arXiv:2507.20534_, 2025.


Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam,
Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In
_Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 8769–8778,
2018.


Volcengine. Seed-1.6-thinking. [https://www.volcengine.com/docs/82379/](https://www.volcengine.com/docs/82379/1593703)
[1593703, 2024.](https://www.volcengine.com/docs/82379/1593703) Accessed: 2025-09-17.


Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman.
Glue: A multi-task benchmark and analysis platform for natural language understanding. _arXiv_
_preprint arXiv:1804.07461_, 2018.


Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer
Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language
understanding systems. _Advances in neural information processing systems_, 32, 2019.


Boshi Wang, Xiang Yue, and Huan Sun. Can chatgpt defend its belief in truth? evaluating llm
reasoning via debate. In _EMNLP (Findings)_, 2023.


11


Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming
Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_,
37:95266–95290, 2024.


xAI. Grok-4. Large language model, 2025. [URL https://x.ai/.](https://x.ai/)


An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv_ _preprint_
_arXiv:2505.09388_, 2025.


Lu Yang, He Jiang, Qing Song, and Jun Guo. A survey on long-tailed visual recognition. _Interna-_
_tional Journal of Computer Vision_, 130(7):1837–1872, 2022.


Ziqi Yang, Xuhai Xu, Bingsheng Yao, Ethan Rogers, Shao Zhang, Stephen Intille, Nawar Shara,
Guodong Gordon Gao, and Dakuo Wang. Talk2care: An llm-based voice assistant for communication between healthcare providers and older adults. _Proceedings of the ACM on Interactive,_
_Mobile, Wearable and Ubiquitous Technologies_, 8(2):1–35, 2024.


Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, and Jiashi Feng. Deep long-tailed learning:
A survey. _IEEE transactions on pattern analysis and machine intelligence_, 45(9):10795–10816,
2023.


Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh TN Nguyen, Lauren T May, Geoffrey I Webb, and
Shirui Pan. Large language models for scientific discovery in molecular property prediction.
_Nature Machine Intelligence_, pp. 1–11, 2025.


Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. Toolqa: A dataset for llm
question answering with external tools. _Advances in Neural Information Processing Systems_, 36:
50117–50143, 2023.


APPENDIX


A LLM USAGE STATEMENT


In the preparation of this manuscript, we employed LLMs solely for textual polishing and language
refinement. The motivation, research design, etc., were independently conducted by the authors.


B EXAMPLES Q&A OF LPFQA


12


C PROMPTS OF LLM AS A JUDGE


This benchmark was generated with the use of MLLM and LLM, and the relevant steps involved the
following prompts.


13


14


15


16


D FORUM LIST


The forums selected include, but are not limited to, the following:
https://scienceforums.net/forum/80-sciences/
https://stats.stackexchange.com/
https://math.stackexchange.com/
https://mathoverflow.net/
https://mathematica.stackexchange.com
https://or.stackexchange.com
https://geant4-forum.web.cern.ch/


17


https://root-forum.cern.ch/
https://quantumcomputing.stackexchange.com
https://www.physicsforums.com/
https://astronomy.stackexchange.com
https://physics.stackexchange.com
https://worldbuilding.stackexchange.com
https://chemistry.stackexchange.com/
https://crafts.stackexchange.com
https://biology.stackexchange.com
https://medicalsciences.stackexchange.com/
https://bioinformatics.stackexchange.com
https://bioacoustics.stackexchange.com
https://www.biostars.org/
https://space.stackexchange.com
https://drones.stackexchange.com
https://aviation.stackexchange.com
https://eaaforums.org/
https://www.eng-tips.com/
https://mechanics.stackexchange.com
https://engineering.stackexchange.com
https://bicycles.stackexchange.com
https://3dprinting.stackexchange.com
http://www.mjtd.com/
https://www.practicalmachinist.com/
https://www.practicalmachinist.com/
https://diysolarforum.com/
https://cr4.globalspec.com/thread/88025/High-Voltage-Engineering
https://www.elitetrader.com/
https://quant.stackexchange.com/
https://patents.stackexchange.com
https://law.stackexchange.com/
https://answers.justia.com/
https://iot.stackexchange.com
https://ham.stackexchange.com
https://electronics.stackexchange.com
https://dsp.stackexchange.com
https://arduino.stackexchange.com
https://patents.stackexchange.com/
https://www.lawanswers.com.au/forums/defamation-law-forum.25/
https://3dprinting.stackexchange.com
https://android.stackexchange.com
https://artofproblemsolving.com/community
https://arduino.stackexchange.com
https://ai.stackexchange.com
https://apple.stackexchange.com
https://patents.stackexchange.com
https://board.asm32.info/
https://aviation.stackexchange.com
https://learn.microsoft.com/en-us/answers/topics/azure-digital-twins.html
https://alcohol.stackexchange.com
https://bioacoustics.stackexchange.com
https://bioinformatics.stackexchange.com
https://biology.stackexchange.com
https://www.biostars.org/
https://bitcoin.stackexchange.com
https://blender.stackexchange.com
http://forums.corvetteforum.com/index.php
https://cardano.stackexchange.com
https://chinese.stackexchange.com


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


https://civicrm.stackexchange.com
https://codegolf.stackexchange.com
https://computergraphics.stackexchange.com
https://cs.stackexchange.com/
http://www.cplusplus.com/forum/
https://crypto.stackexchange.com
https://datascience.stackexchange.com
https://dba.stackexchange.com
https://discuss.dvc.org/
https://electronics.stackexchange.com
https://emacs.stackexchange.com
https://engineering.stackexchange.com
https://ethereum.stackexchange.com
https://forum.filezilla-project.org/index.php
http://www.fluka.org/fluka.php?id=mailinglist&mm2=6
https://french.stackexchange.com
https://gamedev.stackexchange.com
https://engx.theiet.org/
https://iot.stackexchange.com
https://forums.majorgeeks.com/
https://mattermodeling.stackexchange.com
https://community.myfitnesspal.com/en/categories/forums
https://networkengineering.stackexchange.com
https://opensource.stackexchange.com
http://www.openedv.com/
https://or.stackexchange.com
https://parenting.stackexchange.com
https://money.stackexchange.com
https://www.physicsforums.com/
https://pm.stackexchange.com
https://proofassistants.stackexchange.com
https://psychology.stackexchange.com/
https://puzzling.stackexchange.com
https://discuss.pytorch.org/
https://quant.stackexchange.com
https://quantumcomputing.stackexchange.com
https://quantumcomputing.stackexchange.com/
https://forums.raspberrypi.com/
https://www.reddit.com/r/math/
https://root-forum.cern.ch/
https://softwareengineering.stackexchange.com
https://community.spiceworks.com/
https://medicalsciences.stackexchange.com/
https://stackoverflow.com/questions/tagged/robotics
https://www.statalist.org/forums/forum/general-stata-discussion/general
https://stellar.stackexchange.com
https://www.techpowerup.com/forums/
https://tex.stackexchange.com
https://tezos.stackexchange.com
https://unix.stackexchange.com
https://ux.stackexchange.com
https://www.vnpy.com/forum/
http://forums.vwvortex.com/
https://guba.eastmoney.com/
https://bbs.pinggu.org/
http://www.mjtd.com/
http://www.3dportal.cn/
http://www.proewildfire.cn/
https://www.armbbs.cn/


19