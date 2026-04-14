

{0}------------------------------------------------

# CORRELATING AND PREDICTING HUMAN EVALUATIONS OF LANGUAGE MODELS FROM NATURAL LANGUAGE PROCESSING BENCHMARKS

**Anonymous authors**

Paper under double-blind review

## ABSTRACT

The field of natural language processing (NLP) historically evaluated language models using benchmarks with automated metrics. However, the recent advent of highly capable chat language models (LMs) has caused a tectonic shift from NLP benchmarks to human evaluations. The relationship between these two evaluation processes is unclear and underexplored for chat LMs. Broadly, to what extent are human evaluations and NLP benchmarks correlated with one another? How well can computationally inexpensive and automated benchmarks predict expensive and time-intensive human evaluations? Which benchmarks provide predictive signals for human preference for LMs? What role, if any, should benchmarks play in the era of chat LMs? To answer these questions, we conducted a large-scale study of the relationships between human evaluations and benchmarks. We show that benchmarks are broadly highly correlated with human evaluations, and we identify which benchmarks exhibit strong correlations with human evaluations and which do not. Having established that reliable correlations exist, we fit models to predict a language model’s human evaluation scores from its academic evaluation scores and provide evidence that such predictive models can generalize across LM scales.

## 1 INTRODUCTION

For decades, the field of natural language processing (NLP) has relied on academic benchmarks and automated metrics (e.g., Accuracy, Brier Score (Brier, 1950), BLEU Papineni et al. (2002)) to evaluate the performance of language models (LMs). These NLP benchmarks provide a standardized and efficient way to measure model capabilities such as machine translation, text summarization, and question answering (Wang et al., 2018; 2019; Srivastava et al., 2022; Gao et al., 2023; Wang et al., 2023a). However, the recent emergence of highly capable chat LMs such as GPT (Ouyang et al., 2022; Achiam et al., 2023), Llama (Touvron et al., 2023a;b; Dubey et al., 2024), Gemini (Team et al., 2023; Reid et al., 2024) and Claude (Anthropic, 2023) has prompted a re-evaluation of how we assess LMs, with a growing emphasis on assessing LMs based on their ability to interact with and assist human users in real-world scenarios (Zheng et al., 2023; Reuel et al., 2024).

This shift towards human evaluations raises important questions about the relationship between NLP benchmarks and human evaluations of chat LMs. Additionally, human evaluations are not without challenges; they can be expensive, time-intensive and noisy, in contrast with computationally cheaper, faster and precise benchmarks. In this paper, we aim to explore the relationship between human evaluations and NLP benchmarks in pursuit of understanding what role, if any, benchmarks should play in the era of chat LMs. As shown in Fig. 1, we seek to answer two key research questions:

1. To what extent are human evaluations and NLP benchmarks correlated with one another?
2. How well can NLP benchmarks predict human evaluations?

To answer these questions, we conducted a large-scale study comparing human evaluations and NLP benchmarks using four Llama 2 Chat language models (LMs) (Touvron et al., 2023b). For human evaluations, we constructed a large-scale dataset of single-turn and multi-turn prompts from a diverse taxonomy (Fig. 2) and collect high quality pairwise preference data of the four Chat Llama 2 models against GPT 3.5 (Ouyang et al., 2022) from paid human annotators. For NLP benchmarks,

{1}------------------------------------------------

![Diagram illustrating the correlation and prediction of human evaluations from NLP benchmarks for chat language models. At the top, four llama icons represent 'Chat Fine-Tuned Language Models'. Below them, 'Human Evaluations' are shown on the left with a grid of yellow cells and icons for face, pencil, triangle, and speech bubble. On the right, 'NLP Benchmarks' are shown with a grid of blue cells and icons for documents. Arrows from the models point to both grids. A central thinking face emoji is labeled 'What is the relationship?' with arrows pointing to it from both grids.](9ba3dc91984c80b96f217fb1bddd5c06_img.jpg)

Diagram illustrating the correlation and prediction of human evaluations from NLP benchmarks for chat language models. At the top, four llama icons represent 'Chat Fine-Tuned Language Models'. Below them, 'Human Evaluations' are shown on the left with a grid of yellow cells and icons for face, pencil, triangle, and speech bubble. On the right, 'NLP Benchmarks' are shown with a grid of blue cells and icons for documents. Arrows from the models point to both grids. A central thinking face emoji is labeled 'What is the relationship?' with arrows pointing to it from both grids.

Figure 1: **Correlating and Predicting Human Evaluations of Language Models from Natural Language Processing (NLP) Benchmarks.** We evaluate chat language models on conversational tasks with human pairwise evaluations and on standard NLP benchmarks with automated metrics, then study whether scores on computationally inexpensive and fast NLP benchmarks are correlated with and predictive of expensive and time-intensive human evaluations?

we evaluate the same four Chat Llama 2 models on standard NLP benchmarks under established evaluation processes (metrics, prompting, 0-shot/few-shot, etc.). We analyze pairwise correlations between NLP benchmark and human evaluations to identify which NLP benchmarks correlate highly with human evaluations and which do not. We also aim to identify which human evaluations, if any, are uncorrelated with any NLP benchmarks. We then pivot to predicting human evaluations from NLP benchmarks using overparameterized linear regressions and leave-one-out cross-validation. We investigate the extent to which NLP benchmarks can predict human evaluations.

## 2 RELATED WORK

The evaluation of language models has a rich and constantly evolving history. Human evaluations have long been considered the gold standard (Gatt & Krahmer, 2018; Van Der Lee et al., 2019; Celikylmaz et al., 2020; Roller et al., 2020; van der Lee et al., 2021), despite serious objections raised regarding the collection, analysis, and interpretation of human evaluation scores (Novikova et al., 2018; Howcroft et al., 2020; Bowman & Dahl, 2021; Karpinska et al., 2021; Clark et al., 2021; Smith et al., 2022; Gehrmann et al., 2023; Finch et al., 2023). Many classic NLP benchmark metrics, such as BLEU (Papineni et al., 2002), NIST (Doddington, 2002), ROUGE (Lin, 2004), and METEOR (Banerjee & Lavie, 2005), were introduced on the premise that they correlate with human judgments. However, subsequent studies revealed that the relationship between automated metrics and human evaluations is often complex and not straightforward (Liu et al., 2016; Novikova et al., 2017; Reiter, 2018; Karpinska et al., 2021). Another prominent class of evaluation methods are based on machine learning models, e.g., word mover distance (Kusner et al., 2015) and BERT-Score (Zhang et al., 2019) that have since evolved into using chat LMs themselves as evaluators (Wang et al., 2023b; Zheng et al., 2024; Chiang & yi Lee, 2023; Chan et al., 2023; Bavaresco et al., 2024; Fu et al., 2024), albeit with limitations, e.g., (Dorner et al., 2024; Szymanski et al., 2024; Thakur et al., 2024).

The earliest investigations into the general relationship between NLP benchmark scores and human evaluations date back to Bangalore et al. (2000), Belz & Reiter (2006), and Liu et al. (2016). In the context of natural language generation, Clinciu et al. (2021) found that embedding-based automated metrics (e.g., BERT-Score (Zhang et al., 2019) and BLEURT Sellam et al. (2020)) correlate more strongly with human judgments compared to word-overlap metrics (e.g., ROUGE (Lin, 2004) and BLEU (Papineni et al., 2002)). In the domain of natural language inference, Schuff et al. (2021) found that automated metrics do not appear to correlate with human judgment scores. However, the majority of these works predate the current era of chat LMs, which exhibit significantly more advanced capabilities compared to their predecessors. This new era motivates our work to investigate the relationship between NLP benchmarks and human evaluations when evaluating chat LMs.

{2}------------------------------------------------

108  
109  
110  
111  
112  
113  
114  
115  
116  
117  
118  
119  
120  
121  
122  
123  
124  
125  
126  
127  
128  
129  
130  
131  
132  
133  
134  
135  
136  
137  
138  
139  
140  
141  
142  
143  
144  
145  
146  
147  
148  
149  
150  
151  
152  
153  
154  
155  
156  
157  
158  
159  
160  
161

![Figure 2: Human Evaluations: Taxonomy of Single-Turn and Multi-Turn Conversations. The diagram shows a hierarchical taxonomy of 9 areas (blue), categories (green), and subcategories (yellow).](7055f51feb10ea4ea48b27c36f085286_img.jpg)

The diagram illustrates a hierarchical taxonomy of conversation areas, categories, and subcategories. The structure is as follows:

- Factual Questions** (Area):
  - Cultural & Social Topics** (Category):
    - Arts & literature
    - History & traditions
    - Popular culture & media
    - Religion & spirituality
    - Social issues & current events
- Writing & Content Creation** (Area):
  - Creative Writing** (Category):
    - Articles & reviews
    - Fictional stories & narrative
    - In-the-style-of writing
    - Poetry & songwriting
    - Social media posts
  - Summarization & Editing** (Category):
    - Content restructuring & organization
    - Proofreading
    - Style & tone adjustments
- Recommendations** (Area):
  - Entertainment Suggestions** (Category):
    - Books, authors, & genres
    - Games, apps, & digital content
    - Movies, TV shows, & streaming content
    - Music, songs, & artists
  - Personal & Professional Development** (Category):
    - Health, wellness, & self-improvement tips
    - Job search & career advice
    - Skill-building resources & courses
    - Networking & mentorship opportunities
- Language Assistance** (Area):
  - Grammar, Spelling & Vocabulary** (Category):
    - English slang
    - Grammar & syntax
    - Language conventions & style
    - Spelling & orthography
    - Vocabulary & word choice
- Dialogue** (Area):
  - Adversarial Dishonesty** (Category):
    - Adversarial Dishonesty
  - Adversarial Harmfulness** (Category):
    - Adversarial Harmfulness
  - Advice** (Category):
    - Casual advice & recommendations
    - Personal & interpersonal relationships
  - Brainstorming** (Category):
    - Brainstorming
  - Classification** (Category):
    - Classification
  - Code** (Category):
    - Code
  - Conversational Entertainment** (Category):
    - Conversational Entertainment
  - Dialogue** (Category):
    - Dialogue
  - Extraction** (Category):
    - Extraction
  - Factual Questions** (Category):
    - Factual Questions
  - Language Assistance** (Category):
    - Language Assistance
  - Math** (Category):
    - Math
  - Identity / Personas** (Category):
    - Famous historical personalities
    - Fictional characters
    - No character (just AI)
    - Public figures
    - Synthetic (made up)
  - Mathematical Reasoning** (Category):
    - Mathematical Reasoning
  - Open QA** (Category):
    - Open QA
  - Procedural Questions** (Category):
    - Procedural Questions
  - Reasoning (Math / Problem Solving)** (Category):
    - Reasoning (Math / Problem Solving)
  - Recommendations & Brainstorming** (Category):
    - Recommendations & Brainstorming
  - Rewriting** (Category):
    - Rewriting
  - Safety** (Category):
    - Safety
  - Summarization** (Category):
    - Summarization
  - Writing** (Category):
    - Writing
  - Writing & Content Creation** (Category):
    - Writing & Content Creation

Figure 2: Human Evaluations: Taxonomy of Single-Turn and Multi-Turn Conversations. The diagram shows a hierarchical taxonomy of 9 areas (blue), categories (green), and subcategories (yellow).

Figure 2: **Human Evaluations: Taxonomy of Single-Turn and Multi-Turn Conversations.** Single-turn and multi-turn prompts were created in a hierarchical taxonomy of 9 areas (blue), categories (green) and subcategories (yellow). Chat Llama 2 generations were then rated against ChatGPT generations by paid human annotators on a 7 point Likert scale (Likert, 1932).

## 3 METHODS: MODELS, HUMAN EVALUATIONS AND NLP BENCHMARKS

We briefly outline our methodology here; for additional information, please see Appendix A.

**Models** Our paper leverages the Llama 2 model family, consisting of four Chat LMs with 7, 13, 34, and 70 billion parameters pre-trained on 2 trillion tokens and finetuned using supervised finetuning (Sanh et al., 2021; Chung et al., 2022; Longpre et al., 2023) and reinforcement learning from human feedback (Christiano et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020). We chose the Llama 2 models because at the time we collected our data, the Llama 2 family contained leading open-access chat-finetuned models spanning multiple scales with minimal variations in architecture, ensuring consistency in our analyses and a robust foundation for our investigations.

**Human Evaluations: Single Turn & Multi-Turn** In this work, our aim was specifically to identify which NLP benchmark scores are predictive of human preferences on open-ended prompts representative of real-world chat model usage. We chose this approach to maximize the ecological validity and generalizability of the findings to real-world use cases. For a concrete example, we may want our chat language models (LMs) to excel at providing bespoke career advice; which NLP benchmarks provide useful signals for whether models are improving at such tasks?

To answer such questions, we created a taxonomy of single-turn and multi-turn interactions (Fig. 2) between chat LMs and humans. For single-turn interactions, we generated a diverse set of prompts

{3}------------------------------------------------

spanning common areas of interest: Factual Questions, Procedural Questions, Language Assistance, Writing & Content Creation, Dialogue, Code, Reasoning, Recommendations / Brainstorming and Safety, with nested categories and subcategories. For multi-turn prompts, non-annotator humans were asked to have conversations (3 to 15 turns long) with all models on similar topics of interest: Factual Questions, Procedural Questions, Language Assistance, Writing & Content Creation, Summarization & Editing, General Dialogue, Reasoning and Recommendations / Brainstorming. This taxonomy was chosen to broadly cover common use-cases of Chat LMs. Example prompts include: “What is the tallest mountain in the world?” (Factual Question); “How do I make minestrone soup?” (Procedural Question); “Please make this sentence more friendly: I need you to stop parking in my space” (Language Assistance); “Write me a poem about getting to the weekend after a long day at work” (Writing & Content Creation). See Appendix A.2 for more information.

We then paid human annotators to evaluate each of the four Chat Llama 2 models against ChatGPT 3.5 (Ouyang et al., 2022) (gpt-3.5-0301) on a dataset of single-turn and multi-turn prompts (Fig 2). We chose gpt-3.5-0301 because, at the time this data was collected, gpt-3.5-0301 was a good balance of three desirable properties for our study: performant, cheap, and stable. For each pair of conversations (one conversation with Chat Llama responses and the other with ChatGPT responses), at least three unique human annotators independently indicated which conversation was preferred using a Likert scale (Likert, 1932) from 1 to 7, where 1 denotes the Chat Llama model was strongly preferred and 7 denotes gpt-3.5-0301 was strongly preferred. Across the 11291 single-turn samples and 2081 multi-turn samples, we had at least 3 unique human annotators per pairwise comparison, with 2104 unique human annotators overall. For our analyses, we averaged the annotators’ scores for each pairwise comparison to give us an average human evaluation score per datum.

**Natural Language Processing (NLP) Benchmarks** We evaluated the four Chat Llama 2 models on large-scale and commonly-used NLP benchmarks: AGI Eval (Zhong et al., 2023), AI2 Reasoning Challenge (ARC; both Easy and Hard) (Clark et al., 2018), BIG Bench Hard (Srivastava et al., 2022; Suzgun et al., 2022) BoolQ (Clark et al., 2019), CommonSenseQA (Talmor et al., 2019), COPA (Roemmelme et al., 2011), DROP (Dua et al., 2019), GSM8K (Cobbe et al., 2021), HellaSwag (Zellers et al., 2019), HumanEval (Chen et al., 2021), InverseScaling (McKenzie et al., 2022a;b; 2023), MBPP (Austin et al., 2021), MMLU (Hendrycks et al., 2020), Natural Questions (Kwiatkowski et al., 2019), OpenbookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), QuAC (Choi et al., 2018), RACE (Lai et al., 2017), SIQA (Sap et al., 2019), SQUAD (Rajpurkar et al., 2016), TLDR (Völske et al., 2017), TriviaQA (Joshi et al., 2017), Winogrande (Sakaguchi et al., 2021) and XSum (Narayan et al., 2018). Some of these benchmarks (e.g., MMLU) contain subsets (e.g., Jurisprudence) that we do not aggregate over. These tasks cover commonsense reasoning, world knowledge, reading comprehension, coding and more. We used standard evaluation processes for all academic benchmarks including prompt formatting, metrics, 0-shot/few-shot, etc. This structured approach facilitates an exhaustive examination of model performances across varied metrics. For more information, see Appendix A.1.

**Scores for Subsequent Analyses** For each dataset and evaluation process (either human or NLP), we average each model’s scores across all samples, yielding two matrices of scores:

$$X_{\text{NLP}} \in \mathbb{R}^{160 \times 4} \qquad X_{\text{Human}} \in \mathbb{R}^{55 \times 4}$$

Here, 4 is the number of models, 160 is the number of NLP benchmarks per model and 55 is the number of human evaluation area-category-subcategory scores per model. We subsequently study the correlations between  $X_{\text{NLP}}$  and  $X_{\text{Human}}$ , then test how well  $X_{\text{NLP}}$  can predict  $X_{\text{Human}}$ .

## 4 CORRELATING HUMAN EVALUATIONS WITH NLP BENCHMARKS

We began by computing correlations between human evaluations and NLP benchmarks, computing three standard correlations over the 4 average scores per model — Pearson (Galton, 1877), Spearman (Spearman, 1904) and Kendall (Kendall, 1938) — giving us three correlation matrices of shape  $160 \times 55$  between every pair of NLP benchmark and human evaluation area-category-subcategory (Fig. 3). Pearson correlation measures the linear relationship between two continuous variables, whereas Spearman and Kendall correlations assess the monotonic relationship between two variables; Spearman correlation is based on the rank order of the data points, whereas Kendall correlation is determined by the number of concordant and discordant pairs. By using different correlation metrics, we aim to robustly characterize the relationships between human and NLP benchmarks.

{4}------------------------------------------------

![Figure 3: Two heatmaps showing Pearson correlations between human evaluations and NLP benchmarks. The top heatmap is for 'Human evaluation areas' and the bottom for 'Human evaluation categories'. Rows are grouped by human evaluation areas, and columns are NLP benchmarks. Red indicates positive correlation, blue indicates negative correlation, and light colors indicate low correlation.](e0d425c8e4eef259e4c52d81426d93fa_img.jpg)

The figure consists of two heatmaps, one above the other. Both heatmaps share the same columns, which represent various NLP benchmarks. The rows are grouped by human evaluation areas. The top heatmap is titled 'Human evaluation areas' and the bottom one 'Human evaluation categories'. The color scale ranges from -1 (blue) to 1 (red), with 0 being light gray. The heatmaps show that many human evaluation areas and categories are positively correlated with certain NLP benchmarks, particularly those related to language understanding and generation. For example, 'Language Assistance' and 'Open Question Answering' show strong positive correlations with benchmarks like 'CoPhIR' and 'OpenQA'. Conversely, some safety and adversarial benchmarks show negative correlations with certain human evaluation areas.

Figure 3: Two heatmaps showing Pearson correlations between human evaluations and NLP benchmarks. The top heatmap is for 'Human evaluation areas' and the bottom for 'Human evaluation categories'. Rows are grouped by human evaluation areas, and columns are NLP benchmarks. Red indicates positive correlation, blue indicates negative correlation, and light colors indicate low correlation.

Figure 3: **Pearson Correlations Between Human Evaluations and NLP Benchmarks.** Rows: Human evaluation areas-categories-subcategories. Columns: NLP benchmarks. The heatmap is row-wrapped to fit on the page. Large positive correlations (+1) are shown in red. Large negative anticorrelations (-1) are shown in blue. Low uncorrelations ( $\sim 0$ ) are shown in light-white-gray.

Macroskopically, at the most coarse grouping of human evaluations in our taxonomy (i.e., areas) (Fig. 2), we found that average NLP benchmark scores are highly correlated with average human scores for all human evaluation areas under all three correlation metrics (Fig. 4 top). Due to the small number of models ( $N = 4$ ), Spearman and Kendall correlations suffer discretization effects (Fig. 11), inducing an illusion of undulations. These strong correlations suggest that, at a high level, NLP benchmarks are reasonable proxies for human judgments of LM quality.

Mesoscopically, at the level of human evaluation areas and categories, we find that NLP benchmarks remain highly correlated with human evaluations, with two notable types of exceptions (Fig. 4). First, Adversarial Dishonesty, Adversarial Harmfulness, and Safety are anti-correlated with most NLP benchmarks, potentially indicating that these adversarial and safety-focused categories are more easily transgressed by more capable LMs; an alternatively hypothesis could be that safety benchmarks simply are not especially good, as demonstrated by Ren et al. (2024). Second, Language Assistance and Open Question Answering are uncorrelated with most NLP benchmarks, suggesting that these categories may require new NLP benchmarks. Open Question Answering was surprising given that

{5}------------------------------------------------

![Figure 4: Distributions of Correlations between Human Evaluations and NLP benchmarks. The figure consists of six rows of violin plots. The first three rows show macroscopic distributions for five human evaluation areas: Dialogue, Factual Questions, Language Assistance, Recommendations, and Writing & Content Creation, using Pearson, Spearman, and Kendall correlation methods respectively. The bottom three rows show mesoscopic distributions for 24 specific NLP benchmarks, grouped by human evaluation area and category, using the same three correlation methods. The y-axis for all plots is 'Correlation' ranging from -1.0 to 1.0. The x-axis for the bottom plots lists benchmarks such as Adversarial Dishonesty, Adversarial Harmfulness, Advice, Reasoning, Code, Entailment, Dialogue, Factual Questions, Idioms/Proverbs, Language Assistance, Mathematical Reasoning, Open QA, Procedural Questions, Reasoning (Math/Problem Solving), Reasoning (Textual Reasoning), Safety, Writing, Writing & Content Creation, Cultural & Social Topics, Grammar, Spelling, & Vocabulary, Entailment Suggestions, Personal & Professional Development, Creative Writing, and Summarization & Editing.](c54b3ca7603d65d4589151bc3a49d054_img.jpg)

Figure 4: Distributions of Correlations between Human Evaluations and NLP benchmarks. The figure consists of six rows of violin plots. The first three rows show macroscopic distributions for five human evaluation areas: Dialogue, Factual Questions, Language Assistance, Recommendations, and Writing & Content Creation, using Pearson, Spearman, and Kendall correlation methods respectively. The bottom three rows show mesoscopic distributions for 24 specific NLP benchmarks, grouped by human evaluation area and category, using the same three correlation methods. The y-axis for all plots is 'Correlation' ranging from -1.0 to 1.0. The x-axis for the bottom plots lists benchmarks such as Adversarial Dishonesty, Adversarial Harmfulness, Advice, Reasoning, Code, Entailment, Dialogue, Factual Questions, Idioms/Proverbs, Language Assistance, Mathematical Reasoning, Open QA, Procedural Questions, Reasoning (Math/Problem Solving), Reasoning (Textual Reasoning), Safety, Writing, Writing & Content Creation, Cultural & Social Topics, Grammar, Spelling, & Vocabulary, Entailment Suggestions, Personal & Professional Development, Creative Writing, and Summarization & Editing.

Figure 4: **Distributions of Correlations between Human Evaluations and NLP benchmarks.** Top: Macroscopically, for each human evaluation area, Chat LM scores are typically highly correlated with NLP benchmarks. Bottom: Mesoscopically, human and NLP benchmarks remain positively correlated, with notable exceptions: Adversarial Dishonesty, Adversarial Harmfulness and Safety are anticorrelated with most NLP benchmarks, and Language Assistance and Open QA are uncorrelated.

some of our NLP benchmarks are open question answering datasets, e.g., OpenBookQA (Mihaylov et al., 2018). We found the three correlations metrics visually agreed with one another and were themselves tightly coupled (App. Fig. 11), and so we present only one (Pearson) moving forward, with equivalent plots of the other two (Spearman, Kendall) deferred to the appendix.

### 4.1 WHICH HUMAN EVALUATIONS HAVE FEW-TO-NO CORRELATED NLP BENCHMARKS?

To the best of our ability to discern, none. Every human evaluation seemed to have at least some NLP benchmarks that were either correlated or anticorrelated with it. This result is promising because it suggests human evaluations might be predictable from NLP benchmarks (Sec. 5).

### 4.2 WHICH NLP BENCHMARKS EXHIBIT HIGH CORRELATIONS WITH HUMAN EVALUATIONS?

To answer this question, we ordered NLP benchmarks based on their average correlation score with all of the human evaluation areas, categories and subcategories. We found many NLP benchmarks

{6}------------------------------------------------

![Figure 5: Two scatter plots showing the correlation of various NLP benchmarks with human evaluations. The left plot shows correlations with 'Human Evaluations' (x-axis from -0.15 to 0.10), and the right plot shows correlations with 'All Human Evaluations' (x-axis from -0.15 to 0.10). The y-axis for both lists various benchmarks, including Nutrition, Human Aging, Sociology, Public Relations, Moral Scenarios, College Computer Science, Word Sorting, Reasoning About Colored Objects, Logical Deduction, HellaSwag, ARC, RACE, PIQA, NaturalQuestions, QuAC, CommonSenseQA, DROP, TriviaQA, ETHOS, Kth Sentence, Inverse Scaling, Resisting Correction Classification, OpenBookQA, COPA, SciBench, Fundamentals of Physics, and SIQA. The plots show that benchmarks like Nutrition, Human Aging, and Sociology have higher positive correlations, while others like ETHOS and SIQA show weaker or negative correlations.](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

Figure 5: Two scatter plots showing the correlation of various NLP benchmarks with human evaluations. The left plot shows correlations with 'Human Evaluations' (x-axis from -0.15 to 0.10), and the right plot shows correlations with 'All Human Evaluations' (x-axis from -0.15 to 0.10). The y-axis for both lists various benchmarks, including Nutrition, Human Aging, Sociology, Public Relations, Moral Scenarios, College Computer Science, Word Sorting, Reasoning About Colored Objects, Logical Deduction, HellaSwag, ARC, RACE, PIQA, NaturalQuestions, QuAC, CommonSenseQA, DROP, TriviaQA, ETHOS, Kth Sentence, Inverse Scaling, Resisting Correction Classification, OpenBookQA, COPA, SciBench, Fundamentals of Physics, and SIQA. The plots show that benchmarks like Nutrition, Human Aging, and Sociology have higher positive correlations, while others like ETHOS and SIQA show weaker or negative correlations.

Figure 5: **NLP Benchmarks Ranked by Average Pearson Correlation over All Human Evaluations.** Certain benchmarks have higher correlations with human evaluations, including a subset of MMLU, a subset of BIG Bench Hard, HellaSwag, ARC, RACE, PIQA, NaturalQuestions, QuAC, and CommonSenseQA. Other benchmarks were weakly or uncorrelated with human evaluations: ETHOS, Kth Sentence, Inverse Scaling (with the exception of Resisting Correction Classification), OpenBookQA, COPA, SciBench (with the exception of Fundamentals of Physics) and SIQA.

have high average correlation with human evaluations (Fig. 5); the highest average correlation NLP benchmarks include a subset of MMLU (Nutrition, Human Aging, Sociology, Public Relations, Moral Scenarios, College Computer Science), a subset of BIG Bench Hard (Word Sorting, Reasoning About Colored Objects, Logical Deduction), HellaSwag, ARC, RACE, PIQA, NaturalQuestions, QuAC, CommonSenseQA, DROP and TriviaQA. Other benchmarks were less correlated or uncorrelated with human evaluations: ETHOS, Kth Sentence, Inverse Scaling (with the exception of Resisting Correction Classification), OpenBookQA, COPA, SciBench (with the exception of Fundamentals of Physics) and SIQA. Upon investigating more closely, some of the most highly correlated NLP benchmarks make sense. For instance, Inverse Scaling’s Resisting Correction Classification ranked second highest for being correlated with human evaluations, and the task measures a highly desirable capability for human users: the LM’s ability to follow user instructions that run counter to the LM’s natural inclinations.

### 4.3 WHAT COMMUNITIES EXIST BETWEEN HUMAN EVALUATIONS AND NLP BENCHMARKS?

To detect what communities exist between human evaluations and NLP benchmarks, we computed the singular value decomposition of the pairwise Pearson correlation matrix between human evaluations and NLP benchmarks (Fig. 6 top). The maximum rank the correlation matrix can have is 4 because the correlations are computed over the 4 Chat Llama 2 models, but we found that the correlation matrix has only 3 non-zero singular values (App. Fig. 12). Decomposing the correlation matrix into its 3 rank-one components  $\sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T + \sigma_3 u_3 v_3^T$  revealed three levels of increasing fine-grained structure in the correlations (App. Fig. 13). We then visualized the human evaluations and NLP benchmarks in the (dimension-scaled) plane defined by the first two rank-one components of the Pearson correlation matrix (Fig. 6 bottom).

The bulk of **human evaluations** and **NLP benchmarks** live in one community; however, there are also several smaller interesting communities. Starting on the left of Fig. 6 and moving clockwise, at the top left is a loose community of **Dialogue Code**, **Dialogue Language Assistant**, several **Kth Sentence** tasks, **Openbook Question Answering (OBQA)**, **AGILSAT**, **AGILLawyer Qualification Test**, which generally measure model capabilities at identifying and using key information within the context. On the top right, **Inverse Scaling.NEQA Classification** is alone; this benchmark measures whether models are tripped up by negated questions, which most humans try not to do and likely explains why

{7}------------------------------------------------

![Figure 6: Structure of Pairwise Pearson Correlations Between Human Evaluations and NLP Benchmarks. The plot shows the 1st and 2nd Singular Vectors (Dimension-Scaled) for Human Evaluations (pink dots) and NLP Benchmarks (green dots). The x-axis ranges from -1.0 to 1.0, and the y-axis ranges from -2 to 2. Most evaluations are clustered on the left side (negative x), while benchmarks are more spread out across the plot.](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

The figure is a scatter plot titled "Pearson Correlation". The x-axis is labeled "1st Singular Vector (Dimension-Scaled)" and ranges from -1.0 to 1.0. The y-axis is labeled "2nd Singular Vector (Dimension-Scaled)" and ranges from -2 to 2. The plot contains two sets of data points: "Human Evaluations" represented by pink dots and "NLP Benchmarks" represented by green dots. The human evaluations are densely clustered on the left side of the plot (negative x-values), with a few outliers. The NLP benchmarks are more dispersed, with some points on the left, some in the center, and some on the right. Specific labels for points include "Kth Sentence 1024", "Kth Sentence 512", "Kth Sentence 256", "Kth Sentence 128", "Kth Sentence 64", "Kth Sentence 32", "Kth Sentence 16", "Kth Sentence 8", "Kth Sentence 4", "Kth Sentence 2", "Kth Sentence 1".

Figure 6: Structure of Pairwise Pearson Correlations Between Human Evaluations and NLP Benchmarks. The plot shows the 1st and 2nd Singular Vectors (Dimension-Scaled) for Human Evaluations (pink dots) and NLP Benchmarks (green dots). The x-axis ranges from -1.0 to 1.0, and the y-axis ranges from -2 to 2. Most evaluations are clustered on the left side (negative x), while benchmarks are more spread out across the plot.

Figure 6: **Structure of Pairwise Pearson Correlations Between Human Evaluations and NLP Benchmarks.** Top: Each row is a human evaluation area and category, and each column is an NLP benchmark and task; values are Pearson correlations ranging from **anticorrelated (-1)** to **correlated (+1)**. The correlation matrix has 3 non-zero singular values (App. Fig. 12). Bottom: **Human evaluations** and **NLP benchmarks** are plotted projected along the (dimension-scaled) first two singular modes of the Pearson correlation matrix. The bulk of evaluations live in one community (left), with smaller communities (top, bottom, right); for an in-depth interpretation, see Sec. 4.3.

this benchmark is isolated. On the right and lower right side, **Dialog.Safety** is next to **ETHOS**, a hate speech detection benchmark, and **AGL.Gaokao Chemistry**, a chemistry benchmark. This community is also close to another community in the lower right comprised of **Dialogue.Adversarial Harmfulness**, **Dialogue.Adversarial Dishonesty**, **Inverse Scaling.Into the Unknown**, **TLDR**. In the lower left, **Dialogue.Open QA** and **Dialogue.Writing** are near **BIG Bench Hard’s Dyck Languages**, **Geometric Shapes and Tracking Shuffled Objects** and multiple science and factual knowledge benchmarks like **MMLU’s Electrical Engineering, Management., SciBench’s Quantum Chemistry (quan and chemmc)**. **BIG Bench Hard’s Formal Fallacies** and **Kth Sentence (1024)** lie in the center, disconnected from most other evaluations.

{8}------------------------------------------------

![Figure 7: Leave-One-Out Cross Validated Linear Regression Predictions of Human Evaluations. The figure consists of four vertically stacked scatter plots, each representing a different Chat Llama 2 model: Llama 2 7B, Llama 2 13B, Llama 2 34B, and Llama 2 70B. Each plot shows 'Predicted Human Evaluation Score' on the y-axis and 'Actual Human Evaluation Score' on the x-axis, both ranging from -10 to 10. A diagonal line represents the identity function (y=x). Data points are colored circles, where the color corresponds to a specific human evaluation area, category, or subcategory as defined in the legend. The legend lists 30 categories: Dialogue (Adversarial, Advice, Brainstorming, Code, Conversational, Identity, Language Assistance), Recommendations (Entertainment, Personal/Professional), and Writing & Content Creation (Creative, Summarization). The plots show a strong positive correlation between predicted and actual scores, with points tightly clustered around the identity line.](b93cbfb52e37619e688175a6aad9edd9_img.jpg)

Figure 7: Leave-One-Out Cross Validated Linear Regression Predictions of Human Evaluations. The figure consists of four vertically stacked scatter plots, each representing a different Chat Llama 2 model: Llama 2 7B, Llama 2 13B, Llama 2 34B, and Llama 2 70B. Each plot shows 'Predicted Human Evaluation Score' on the y-axis and 'Actual Human Evaluation Score' on the x-axis, both ranging from -10 to 10. A diagonal line represents the identity function (y=x). Data points are colored circles, where the color corresponds to a specific human evaluation area, category, or subcategory as defined in the legend. The legend lists 30 categories: Dialogue (Adversarial, Advice, Brainstorming, Code, Conversational, Identity, Language Assistance), Recommendations (Entertainment, Personal/Professional), and Writing & Content Creation (Creative, Summarization). The plots show a strong positive correlation between predicted and actual scores, with points tightly clustered around the identity line.

Figure 7: **Leave-One-Out Cross Validated Linear Regression Predictions of Human Evaluations.** Overparameterized linear regressions typically accurately predict human evaluation scores from all NLP benchmark scores. Each subfigure shows predicted human evaluation scores against actual human evaluation scores on each of the four left-out Chat Llama 2 models colored by the particular area, category and subcategory of human evaluation.

## 5 PREDICTING HUMAN EVALUATIONS FROM NLP BENCHMARKS

Having established the existence of correlations between human evaluations and NLP benchmarks, we next investigated the feasibility of predicting human evaluations from NLP benchmarks. Our goal is to build predictive models that accurately predict a language model’s average human evaluation scores per evaluation areas and categories using the model’s average scores on NLP benchmarks and tasks. However, we faced a significant challenge due to the overparameterized nature of our data: for each target human evaluation area or category, there are approximately 150 covariates (NLP benchmarks and tasks) but only 4 samples (Chat Llama 2 models).

**Predictive Modeling: Overparameterized Linear Regressions** To predict human evaluations from NLP benchmarks, we used overparameterized linear regression. In general, overparameterized linear regression is known to be capable of generalizing (App. Sec. A.3), although whether linear models would generalize in this setting was an empirical question. For each human evaluation area and category, we fit a linear model to predict a language model’s average human evaluation score from its average scores on all NLP benchmarks and tasks. To assess the predictive accuracy of these overparameterized models, we employed leave-one-out cross validation: we fit four separate linear models, each time fitting on three of the chat LMS’ scores and holding out the fourth to test the performance of the linear model. This approach allows us to estimate the models’ performance on unseen data, albeit with limitations due to the small sample size. Before fitting the models, we normalized all human evaluation scores to lie in  $[0, 1]$  rather than  $[-7, -1]$  (recalling that higher scores indicate the human evaluator prefers the Chat Llama 2 model compared to GPT-3.5).

{9}------------------------------------------------

**Results** Across the various human evaluation areas and categories, we found that the linear models’ predicted average human evaluation scores generally align well with the actual average human evaluation scores, as evidenced by most points falling close to the identity line in the predicted score vs. actual score plane (Fig. 7). This suggests that, despite the overparameterization, the linear models can capture meaningful relationships between NLP benchmarks and human evaluations. However, we caution against over-interpreting these results, as the small sample size and the assumption of linearity may limit the generalizability of these findings to other language models or evaluation settings.

To gain insight into which NLP benchmarks are most informative for predicting human evaluation scores, we examine the learned weights of the linear models (Fig. 18). NLP benchmarks with consistently high absolute weights across different human evaluation areas and categories are likely to be more predictive of human judgments. However, due to the overparameterized nature of the models, we refrain from drawing strong conclusions about the relative importance of individual benchmarks and instead focus on the overall predictive performance. These results suggest that scaling up the number of chat LMs and human evaluation data could unlock highly predictive models of slow, noisy and expensive but valuable human evaluations using fast, precise and cheaper NLP benchmarks.

## 6 DISCUSSION

In this paper, we explored the relationship between human evaluations and NLP benchmarks of chat-finetuned language models (chat LMs). Our work is motivated by the recent shift towards human evaluations as the primary means of assessing chat LM performance, and the need to understand the role that NLP benchmarks can play in this new era.

Through a large-scale study of the Chat Llama 2 model family on a diverse set of human and NLP evaluations, we demonstrated that NLP benchmarks are generally well-correlated with human judgments of chat LM quality. However, our analysis also reveals some notable exceptions to this overall trend. In particular, we find that adversarial and safety-focused evaluations, as well as language assistance and open question answering tasks, exhibit weaker or negative correlations respectively with NLP benchmarks. We also explored predicting human evaluation scores from NLP evaluation scores using overparameterized linear regression models. Our results suggest that NLP benchmarks can indeed be used to predict aggregate human preferences, although we caution that the limited sample size and the assumptions of our models may limit the generalizability of these findings. Our results suggest that NLP benchmarks can serve as fast and cheap proxies of slower and expensive human evaluations in assessing chat LMs.

Additionally, our work highlights the need for further research into NLP evaluations that can effectively capture important aspects of LM behavior, such as safety, robustness to adversarial inputs, and performance on complex, open-ended tasks. It is possible that new NLP benchmarks can provide signals on these topics, e.g., (Wang et al., 2023a). Of particular interest is developing human-interpretable and scaling-predictable evaluation processes, e.g., (Schaeffer et al., 2024a; Ruan et al., 2024; Schaeffer et al., 2024c). Developing and refining such evaluation methods (Madaan et al., 2024), as well as detecting whether evaluations scores faithfully capture models’ true performance (Oren et al., 2023; Schaeffer, 2023; Roberts et al., 2023; Jiang et al., 2024; Zhang et al., 2024; Duan et al., 2024) will be crucial for ensuring that LMs are safe, reliable, and beneficial as they become increasingly integrated into real-world use cases.

In conclusion, our study provides insights into the relationship between human evaluations and NLP benchmarks of chat language models. By leveraging the complementary strengths of both human and NLP benchmarks, we can build a more complete understanding of LM capabilities and behaviors, ultimately enabling the development of models more capable, trustworthy, and beneficial to society.

 Rest of paper (reference and Appendix) is removed.