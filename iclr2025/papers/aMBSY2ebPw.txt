

{0}------------------------------------------------

# CAN LLMs REALLY LEARN TO TRANSLATE A LOW-RESOURCE LANGUAGE FROM ONE GRAMMAR BOOK?

Seth Aycock<sup>1,2</sup>   David Stap<sup>2</sup>   Di Wu<sup>2</sup>   Christof Monz<sup>2</sup>   Khalil Sima'an<sup>1</sup>

<sup>1</sup>ILLC, University of Amsterdam <sup>2</sup>LTL, University of Amsterdam  
 s.aycock@uva.nl

## ABSTRACT

Extremely low-resource (XLR) languages lack substantial corpora for training NLP models, motivating the use of all available resources such as dictionaries and grammar books. *Machine Translation from One Book* (Tanzer et al., 2024) suggests that prompting long-context LLMs with one grammar book enables English–Kalamang translation, an XLR language unseen by LLMs—a noteworthy case of linguistics helping an NLP task. We investigate the source of this translation ability, finding almost all improvements stem from the book’s parallel examples rather than its grammatical explanations. We find similar results for Nepali and Guarani, seen low-resource languages, and we achieve performance comparable to an LLM with a grammar book by simply fine-tuning an encoder-decoder translation model. We then investigate *where* grammar books help by testing two linguistic tasks, grammaticality judgment and gloss prediction, and we explore *what kind* of grammatical knowledge helps by introducing a typological feature prompt that achieves leading results on these more relevant tasks. We thus emphasise the importance of task-appropriate data for XLR languages: parallel examples for translation, and grammatical data for linguistic tasks. As we find no evidence that long-context LLMs can make effective use of grammatical explanations for XLR translation, we conclude data collection for multilingual XLR tasks such as translation is best focused on parallel data over linguistic description.

## 1 INTRODUCTION

Most of the world’s languages are extremely low-resource (XLR), severely lacking in suitable corpora for NLP tasks (Ranathunga & de Silva, 2022), such as parallel data for machine translation (MT). However, over 50% of languages have both a dictionary and a grammar (Nordhoff & Hammarström, 2011). While human-readable, grammar texts are difficult to incorporate into most NLP models due to their non-standard, unstructured format. Large language models (LLMs) can handle free-form textual instructions and provide a potential solution to this data mismatch. After pre-training on trillions of tokens (in mainly high-resource languages), LLMs can learn tasks from only a few in-context examples (Brown et al., 2020; Wei et al., 2022). Given this, interest in exploiting grammar texts in-context for NLP tasks is growing (Ramos et al., 2024; Tanzer et al., 2024; Zhang et al., 2024b).

*Machine Translation from One Book* (Tanzer et al., 2024) claims LLMs can learn to translate between Kalamang (ISO 639-3: kgv)—a newly-documented language unseen in LLM training data—and English (eng) via *in-context learning* with only a grammar book. We note that kgv has over 3,000 parallel sentences, a dictionary with over 3,000 definitions (Visser, 2020), a 500-page grammar book (Visser, 2022) consisting of grammatical explanations and over 1000 parallel glossed examples, and nearly 100 typological feature specifications (Skirgård et al., 2023a;b). This level of resources is comparable to or more than thousands of XLR languages have (Joshi et al., 2020; OLAC, 2024), thus we expect most of these are also minimally represented in LLMs’ pretraining data. Given this, finding methods to effectively exploit the available kgv resources could have wide-reaching implications for XLR NLP. In this paper, we question the claimed utility of grammatical explanations for XLR MT with LLMs, then ask *where* and *what kind* of grammatical knowledge helps. We show that:

{1}------------------------------------------------

**Parallel examples are essential for translation** We disentangle grammar books’ parallel examples from grammatical explanations, finding explanations add no *significant* advantage over parallel data: adding +0.7 CHRf++ into *kgv*, and into *eng* scores fall −0.3 points adding explanations to parallel sentences; and quality drops up to 8 points with parallel data removed. Our findings generalise to Nepali (*npi*) and Guarani (*gug*), where the book’s parallel sentences outperform the full book by up to 4 CHRf++. LLMs fail to effectively exploit grammatical explanations *for translation*.

**Fine-tuning matches long-context LLMs** We fine-tune small translation models on the parallel data, achieving competitive results within 0.2 CHRf++ of the performance of *Gemini* with a grammar book into *kgv*, and beating *Llama-3.1-8B* settings with access to the same data by up to 20 points. Parallel examples (especially with glosses) are both more *token-efficient* and readily available than grammar books, and enable computationally cheaper methods than long-context LLMs.

**Typological prompting outperforms explanations and helps linguistic tasks** We introduce a novel typological feature prompt, and for *kgv* and *npi* translation we find our method is more effective than explanations into *eng*, but not into XLR languages. On *kgv* grammaticality judgment, our typological prompt improves up to 3% over the book’s 1.2k parallel sentences and 8% over the whole book. For gloss prediction, parallel sentences again beat the book by up to 5.3% morpheme accuracy, and adding typology achieves leading performance on this task. Therefore LLMs *can* exploit grammar for *relevant* linguistic tasks—if provided in a useful *form*—but not for translation.

Task-appropriate data is therefore essential. In the current paradigm, we recommend that data collection for XLR MT is thus better focused on parallel data over linguistic description, given the advantages in token efficiency, computational cost, and availability.

## 2 RELATED WORK

**Grammar for low-resource machine translation** Translation of low and extremely-low resource languages (here meaning <100k and 10k parallel examples respectively) with LLMs is currently of significant interest (Cahaywijaya et al., 2024; Court & Elsner, 2024; Iyer et al., 2024). Methods include fine-tuning (Xu et al., 2024), dictionary prompting (Ghazvininejad et al., 2023), and retrieval-augmented few-shot prompting (Merx et al., 2024). Alongside advances in long-context LLMs, recent work has introduced grammar information in context for various tasks: Guo et al. (2024) test a textbook-style prompt with LLM-generated parses, seeing limited gains against parallel sentences; and Zhang et al. (2024a) add a singular syntactic rule to their prompt with small effects. Zhang et al. (2024b) chain morphological analysis, a dictionary and an LLM-summarised grammar book in-context, observing small gains from book passages over a dictionary-only setup. Others meanwhile use grammars with LLMs for data augmentation (Lucas et al., 2024) or as a hybrid rule-based translation system (Coleman et al., 2024).

**Machine Translation from One Book (MTOB)** (Tanzer et al., 2024) introduces a translation test set for the newly documented XLR language Kalamang (thus unseen by LLMs), plus a grammar book and additional parallel sentences. MTOB suggests long-context LLMs can exploit linguistic knowledge (a grammar book) for XLR translation, a potential step forward in leveraging underused resources for XLR languages. However, several issues mean MTOB leaves open questions over LLMs’ ability to exploit linguistic information for XLR tasks. The test sets of 50 short, easy examples are potentially too small for making wider generalisations, and the human baseline is somewhat flawed as they may learn from examples at test time; relatedly, Gemini Team et al. (2024) ask the non-fluent human baseline to rate model outputs and their own *kgv* predictions, potentially biasing the evaluation. Furthermore, despite CHRf++ being the de-facto standard in XLR translation (Maillard et al., 2023; Costa-jussà et al., 2024; Edman et al., 2024), MTOB uses CHRf which unlike CHRf++ does not factor in word order. MTOB’s results would benefit from further ablations, since the signal from parallel sentences and explanations is not disentangled, nor is a strong translation approach tested. Finally, we note that the *kgv* grammar book is not designed for language learning, but for describing theoretical linguistic phenomena—which MTOB’s authors note limits LLMs to a basic competence. In this paper, we tackle these issues by combining the test sets, using automatic CHRf++ scores, disentangling the parallel/non-parallel signal, and testing two tasks better aligned for grammar books.

**Linguistics in NLP** Incorporating linguistic information into NLP models is a long-standing goal with mixed results (Lakoff, 1978; Raskin, 1985; Uszkoreit, 2009; Opitz et al., 2024). Past work sees

{2}------------------------------------------------

gains from adding syntactic knowledge into translation models using constituency parses (Currey & Heafeld, 2019), grammar supertags (Nädejde et al., 2017), or tree-structured models (Sartran et al., 2022). One useful form of linguistic information is *typology*, available for many languages in standardised feature databases (Dryer & Haspelmath, 2013; Skirgård et al., 2023a;b); features describe languages in terms of phenomena such as word order rules, verb tenses, and noun cases. Trained linguists condense fine-grained textual descriptions from grammar books into discrete, categorical, and cross-linguistically consistent feature specifications. Typological features have been incorporated into NLP models with some success in the form of embeddings (Malaviya et al., 2017; Östling & Tiedemann, 2017) but several studies find minimal positive effects on performance (Ponti et al., 2019; Üstün et al., 2022). To test whether LLMs follow this trend, we construct a novel prompt that uses readily available typological feature specifications for source and target languages, as an in-context and language-invariant method for bridging cross-lingual grammatical differences.

**Interlinear gloss prediction** Language documentation involves describing the underlying grammar of a language given its surface forms (Ginn et al., 2023). A standardised data format for this analysis is *Interlinear Glossed Text* (IGT), comprising a morphologically segmented *transcription* (where morphemes are the smallest units of meaning), an aligned *interlinear gloss* with subword-level lexical and grammatical information, and a sentence-level *translation* (Comrie et al., 2015; Mortensen et al., 2023); a Kalamang example is shown in Example 1 (Visser, 2022). We note that glossing is designed for trained linguists rather than language learners. Glosses have been widely applied in NLP tasks, including as a pivot for translation (Zhou et al., 2020), dependency parsing (Georgi et al., 2012), grammar generation (Bender et al., 2014), morphological analysis (Moeller et al., 2020; Shandilya & Palmer, 2023), and linguistic resource development (Beermann et al., 2020; Buchholz et al., 2024). Predicting IGT is therefore a well-motivated grammatical task, and segmented IGT is a valuable linguistic resource. IGT prediction is most relevant for XLR languages where it is impactful in assisting annotators for documentation and preservation (Ginn et al., 2023). Prior methods include supervised neural models (Zhao et al., 2020; Girbach, 2023), or adapting multilingual language models (He et al., 2023; Ginn et al., 2024a;b). Since IGT is costly to generate, past work has scraped it from books (Nordhoff, 2020; Nordhoff & Krämer, 2022); we follow this method to extract Kalamang glosses from the grammar book. One of our contributions involves testing gloss prediction to determine whether LLMs can use grammatical knowledge for more relevant tasks.

|  |  |  |  |  |  |  |  |  |  |
|-|-|-|-|-|-|-|-|-|-|
| (1) | bal | se | sor=at | na | ma | se | nan=i | koyet | <b>Transcription</b> |
|  | dog | IAM | fish=OBJ | consume | 3SG | IAM | consume=PLNL | finish | <b>Interlinear Gloss</b> |
|  | ‘The dog ate the fish, after he ate.’ |  |  |  |  |  |  |  | <b>Translation</b> |

## 3 METHODOLOGY

### 3.1 GRAMMAR BOOKS FOR TRANSLATION

Our methods are guided by open questions over the use of grammar books for XLR translation. First, we manually filter the grammar books into parallel examples and word pairs, and explanatory, descriptive text, to disentangle the signal from translations and grammatical explanations (see Appendix A for a *kgv* book extract). This novel ablation is necessary to understand which specific aspects of grammar books are useful for XLR MT. We ask whether LLMs really learn effectively from the grammar explanations, or if most translation supervision stems only from the book’s parallel examples. We combine the directional test sets into a single 100 example test set to improve the generalisability of these results, and evaluate with CHR++ (Popović, 2017) to take word order into account<sup>1</sup>. We also test *eng-npi* and *gug* translation, low-resource languages with an established evaluation set, FLORES (Costa-jussà et al., 2024) and likely a low data weight in LLMs; while not unseen, these experiments broaden our results to seen low-resource languages more generally.

### 3.2 NEURAL MACHINE TRANSLATION APPROACHES

To compare the LLM-based approach with a standard MT approach for learning to translate a language as yet unseen by the model, we run experiments fine-tuning NLLB-1.3B (Costa-jussà et al., 2024)

<sup>1</sup>We omit human evaluation (cf. Gemini Team et al., 2024) given the infeasibility of engaging proficient Kalamang speakers. See Appendix I for a small-scale qualitative analysis of several *kgv-eng* examples.

{3}------------------------------------------------

on the parallel data sourced from the grammar book. We expect similar results to be achieved with the same resources using a small, specialist encoder-decoder model, which would confirm that the useful translation signal stems from the parallel sentences contained within grammar books—which constitute less than 20% of the *kgv* grammar book’s total tokens (see Table 1 for token counts).

### 3.3 TYPOLOGICAL FEATURE PROMPTING

In asking *what kind* of grammatical knowledge can aid LLMs in XLR tasks, we introduce a text-based method for incorporating typological information into prompts, differing from previous work on continuous typological embeddings (Oncevay et al., 2020). We extract categorical typological feature specifications from Grambank Skirgård et al. (2023b) for *kgv*, *npi*, *gug*, and *eng*, and use a rule-based template to construct a prompt containing features for each language and a short explanation. For an example of the prompt format, see Appendix D. Most languages with grammar books have some typological feature specification, since features are distilled by annotators from external resources. Our method isolates high-level grammatical tendencies of a language from the specific instantiations of those features (i.e. parallel examples). We hypothesise that our method, when combined with the grammar book’s parallel sentences, will at least match the performance of the grammar book. We expect that providing explicit features such as word order rules removes some reasoning requirements for the LLM. Conversely, typological features will not have *relevant* parallel examples, so some reasoning and retrieval is still required, potentially tempering the advantages.

### 3.4 GRAMMATICALITY JUDGMENT

To test the LLM’s ability to acquire knowledge and understanding of Kalamang grammar from the book, we introduce a discriminative grammar judgment experiment. We ask the model to choose the original Kalamang test sentence against a modified example, with three successively easier settings: swapping two adjacent words (*SWAP<sub>adj</sub>*), two random words (*SWAP<sub>ran</sub>*), and shuffling all words (*SHUFFLE*). We acknowledge that while we cannot guarantee all corruptions are ungrammatical (since no author speaks Kalamang), we assume the uncorrupted examples are linguistically unmarked sentences. For all settings we expect a 0-SHOT model to achieve approximately 50% accuracy, while for high-resource languages we would expect near 100% accuracy. We expect the grammar book to have a greater positive impact in this setting where grammatical knowledge is explicitly rewarded.

### 3.5 INTERLINEAR GLOSSED TEXT PREDICTION

To explore another more relevant task for exploiting grammar explanations, we test IGT prediction with the grammar book against few-shot and supervised baselines. This experiment tests whether LLMs can learn grammar from a book to the extent that we see a difference in performance on a grammar-focused task. IGT requires both lexical translation and grammar analysis, without any generation in the language at hand. This makes IGT prediction a more appropriate task to perform from a descriptive, non-didactic grammar text. We argue that IGT prediction accelerates XLR documentation more than translation, and is likely to have more direct impact for both first language (L1) speakers and linguists, not to mention the potential downstream uses, e.g. POS tagging and MT. IGT prediction is also a well defined task with strong baselines from a shared task (Ginn et al., 2023) and clear evaluation metrics, primarily morpheme accuracy (McMillan-Major, 2020; Zhao et al., 2020); our experiments build on this prior work. Finally, we argue grammar books are intuitively suited to IGT prediction more than translation, because their unique contribution is glossed text, rather than just parallel sentences. We use all available sentences with IGT from Dictionaries as our test set, and for our supervised baselines, we process the grammar book IGT examples into a training and development set. We expect the grammar book to provide marginal gains over raw parallel sentences because the grammar book explicitly explains the glossed examples therein.

## 4 EXPERIMENTAL SETUP

### 4.1 DATA

We use the preprocessed Kalamang (*kgv*) grammar book (Visser, 2022; Tanzer et al., 2024), with additional processing of irregularities (particularly for glossing) introduced in LaTeX conversion.

{4}------------------------------------------------

We similarly preprocess a grammar text in Nepali (npi) (Bal, 2004) and Paraguayan Guaraní (gug) (Estigarribia, 2020). We prompt with the entire grammar,  $\text{BOOK}_{\text{all}}$ , in-context (where the subscript indicates the data subset). Following Nordhoff & Krämer (2022), we extract parallel glossed examples and bilingual word/phrase pairs from the book based on text formatting into a parallel subset,  $\text{BOOK}_{\text{para}}$  ( $p$ ). The remainder of the book contains grammatical explanations without parallel examples, labelled  $\text{BOOK}_{\text{non-para}}$  ( $-p$ ). Subset statistics are shown in Table 1. We preprocess  $\text{kvg-eng}$  parallel examples from the grammar book into an unsegmented parallel data format, giving  $\text{PARA}_{\text{book}}$  (used for 5\*-SHOT examples and in full as a prompt) and  $\text{PARA}_{\text{book}}^{\text{IGT}}$  which includes glosses (1239 examples) – for excerpts of prompt types, see Appendix E. We additionally test prompts with  $\text{PARA}_{\text{train}}$  (400 examples) and  $\text{WORDLIST}$  (W) (3813 examples). Additionally, we sample 500 examples from *Dictionaria*<sup>2</sup> (Visser, 2020) as the development set for fine-tuning. In total there are 3.3k  $\text{eng}=\text{kvg}$  parallel examples<sup>3</sup>; we focus on the 1.2k in  $\text{PARA}_{\text{book}}$  for fair comparison with  $\text{BOOK}$  settings. For testing, we use our combined 100 example test set for  $\text{kvg}$ , and *FLORES* devtest for  $\text{npi}$  and  $\text{gug}$  (1012 examples) (Guzmán et al., 2019), with few-shot examples from *FLORES* dev. For IGT prediction, we preprocess 1221 examples from the grammar book with glosses for training (5623 words) and development (612 words) sets (split 90:10% by sentences). Following Ginn et al. (2023), we introduce a test set of 97 glossed examples (447 words) from a different source, *Dictionaria*, which were manually inspected for correct alignment.

Table 1: Dataset statistics for grammar book subsets, in lines and space-separated tokens.

| Language | Split | Lines | Tokens |
|-|-|-|-|
| $\text{kvg}$ | $\text{BOOK}_{\text{para}}$ | 4489 | 17858 |
|  | $\text{BOOK}_{\text{non-para}}$ | 2282 | 81268 |
| $\text{npi}$ | $\text{BOOK}_{\text{para}}$ | 759 | 5333 |
|  | $\text{BOOK}_{\text{non-para}}$ | 2896 | 23233 |
| $\text{gug}$ | $\text{BOOK}_{\text{para}}$ | 5718 | 49122 |
|  | $\text{BOOK}_{\text{non-para}}$ | 3295 | 57338 |

### 4.2 MODELS

In our experiments we use the API-only *Gemini-1.5-Flash-001* (henceforth *Gemini*) (Gemini Team et al., 2024). We justify this choice due to *Gemini*’s context window of 1M tokens, significantly larger than other models, which can handle the entire grammar book, and use *Flash* over *Pro* due to prohibitive cost differences. We also use the smaller, open-weight *Llama-3.1-8B* base and instruction-tuned models (Dubey et al., 2024), with a context of 128k tokens. This is insufficient for  $\text{kvg}$  and  $\text{gug}$   $\text{BOOK}_{\text{all}}$ , but fits  $\text{BOOK}_{\text{para}}$  and  $\text{BOOK}_{\text{non-para}}$ , plus the  $\text{npi}$   $\text{BOOK}_{\text{all}}$ . We test *Llama-Instruct* (*Llama-I*), and fine-tune *Llama* base with *LoRA* (Hu et al., 2021) on  $\text{PARA}_{\text{book}}$  (*Llama-ft*) with prompt masking for 5 epochs with a constant learning rate of  $1\text{e-}4$ , batch size 4, and  $\text{LoRA } \alpha = 16$ ,  $r = 16$ , targeting all linear projections. For our NMT baseline, we fine-tune *NLLB-1.3B-Distilled* (*NLLB*) (Costa-jussà et al., 2024) on  $\text{PARA}_{\text{book}}$ . For  $\text{kvg}$  grammaticality judgment and IGT prediction, we use the same *Gemini* model as above.

### 4.3 EVALUATION

We evaluate translation automatically with *CHRF++* (Popović, 2017). We favour *CHRF++* over *CHRF*, used in Tanzer et al. (2024), since it takes into account word order as well as character  $n$ -gram overlap. We report scores for trimmed responses after the first newline character to distinguish translation quality from overgeneration and chat explanations (Aycock & Bawden, 2024) and use a forceful prompt (detailed in Appendix E) to ensure the translation is produced on the first line.

<sup>2</sup><https://dictionaria.cllid.org/contributions/kalamang>

<sup>3</sup>Data (including grammar book splits) and code are made available at this link.

{5}------------------------------------------------

Table 2: Translation results for  $\text{eng} \leftrightarrow \text{kvg}$  with Gemini, Llama-Instruct (L-I) and fine-tuned (L-ft), and prompt tokens counted with NLTK’s tokenizer (Bird et al., 2009). Highest BOOK<sub>para</sub> scores are underlined, highest overall are **bolded**. Grey rows indicate settings with data other than the book’s parallel data; – indicates tests ruled out by context length. \*W4W tests are not run with Gemini but are included for comparison. The subset of the book’s parallel sentences almost matches or outperforms the whole grammar book, while its grammatical explanations perform poorly.

| Setting <sub>↓</sub> | Model <sub>↓</sub> | CHRF++ |  |  |  |  |  | Tokens |  |
|-|-|-|-|-|-|-|-|-|-|
|  |  | eng-kvg |  |  | kvg-eng |  |  |  |  |
|  |  | Gemini | L-I | L-ft | Gemini | L-I | L-ft |  |  |
| BASELINES |  |  |  |  |  |  |  |  |  |
| 0-SHOT |  | 11.0 | 2.7 | 18.5 | 12.7 | 12.5 | 23.0 | 0 |  |
| W4W |  | 18.9* | – | – | 18.2* | – | – | 0 |  |
| PARALLEL DATA |  |  |  |  |  |  |  |  |  |
| WORDLIST (W) |  | 29.1 | 13.6 | 19.5 | 27.9 | 20.8 | 26.8 | 9.0k |  |
| 5*-SHOT PARA <sub>book</sub> |  | <u>38.9</u> | 15.0 | 24.6 | 33.4 | 21.1 | 23.0 | 0.8k |  |
| PARA <sub>book</sub> |  | 26.6 | 7.3 | 13.0 | 33.1 | 22.9 | 26.9 | 15.6k |  |
| + W |  | 34.7 | 6.8 | 14.4 | 34.7 | 27.5 | 30.5 | 24.6k |  |
| + PARA <sub>train</sub> |  | 40.7 | 13.8 | 17.9 | <b>46.6</b> | <b>31.3</b> | <b>37.6</b> | 29.4k |  |
| PARA <sub>book</sub> <sup>IGT</sup> |  | 33.7 | <b>20.3</b> | <b>28.8</b> | 32.8 | <u>24.7</u> | <u>33.1</u> | 22.7k |  |
| GRAMMAR BOOK SUBSETS |  |  |  |  |  |  |  |  |  |
| BOOK <sub>all</sub> |  | 34.4 | – | – | 34.4 | – | – | 99.6k |  |
| + W |  | 38.3 | – | – | 39.6 | – | – | 108.6k |  |
| + PARA <sub>train</sub> |  | <b>43.7</b> | – | – | 46.1 | – | – | 113.4k |  |
| BOOK <sub>para</sub> (p) |  | 30.8 | 9.7 | 19.0 | 34.7 | 22.1 | 28.8 | 18.3k |  |
| BOOK <sub>non-para</sub> (¬p) |  | 22.6 | 3.3 | 10.0 | 27.5 | 14.3 | 16.7 | 81.3k |  |
| TYPOLOGY |  |  |  |  |  |  |  |  |  |
| TYP 0-SHOT |  | 10.8 | 3.4 | 13.6 | 13.9 | 14.3 | 17.6 | 68.4k |  |
| + BOOK <sub>para</sub> |  | 31.4 | – | – | <u>35.2</u> | – | – | 86.7k |  |
| + PARA <sub>book</sub> |  | 32.9 | – | – | 33.0 | – | – | 84.0k |  |
| + W + PARA <sub>book+train</sub> |  | 40.6 | – | – | 44.9 | – | – | 100.6k |  |

### 4.4 BASELINES

For translation experiments, we test several baselines: 0-SHOT translation with a standard translation prompt; word-for-word translation with fuzzy dictionary lookup (W4W); 5 retrieved examples *per word* (5\*-SHOT) based on longest common subsequences following Tanzer et al. (2024); prompting with the full WORDLIST (W), parallel examples, PARA<sub>book</sub>, parallel examples with glosses, PARA<sub>book</sub>, and processed training set examples, PARA<sub>train</sub>. For IGT prediction, we use a baseline frequency-based classifier (TOP-CLASS), a fine-tuned RoBERTa token classifier (Ginn et al., 2023) (SMP-BASE); a hard-attention glossing model (TUCL-MORPH) (Girrbach, 2023); and BYT5-FT and GLOSSLM-FT models (Ginn et al., 2024b) fine-tuned on our  $\text{kvg}$  IGT training and development sets. We provide segmented input, and English translations to models which accept them.

### 4.5 EXPERIMENTS

Our central research question investigates the contributions of grammatical explanations and parallel data to translation performance. We therefore prompt models with BOOK<sub>all</sub> and its filtered subsets. We test our typological feature prompt, TYP, to replace BOOK<sub>non-para</sub>. For  $\text{npi}$  and  $\text{gug}$ , we repeat the book settings as above. We fine-tune translation models with the PARA<sub>book</sub> parallel data for comparison with BOOK<sub>para</sub> settings. For grammaticality judgment and IGT prediction tasks, we similarly test Gemini with the  $\text{kvg}$  BOOK and TYP prompts.

{6}------------------------------------------------

Table 3: Translation results for  $\text{eng} \rightleftarrows \text{npi}$  and  $\text{eng} \rightleftarrows \text{gug}$  with Gemini and Llama-I. Best  $\text{BOOK}_{\text{para}}$  (white rows) scores are underlined, best overall are **bolded**; – indicates tests ruled out by context length. While  $\text{BOOK}_{\text{all}}$  and  $\text{BOOK}_{\text{non-para}}$  decrease performance from 0-SHOT,  $\text{BOOK}_{\text{para}}$  has a neutral or positive effect into and from  $\text{npi}$  respectively, with a similar trend seen for  $\text{gug}$ .

| Setting <sub>i</sub> | CHRf++ |  |  |  |  |  |  |  |
|-|-|-|-|-|-|-|-|-|
|  | $\text{eng} \rightleftarrows \text{npi}$ |  | $\text{npi} \rightleftarrows \text{eng}$ |  | $\text{eng} \rightleftarrows \text{gug}$ |  | $\text{gug} \rightleftarrows \text{eng}$ |  |
|  | Gemini | L-I | Gemini | L-I | Gemini | L-I | Gemini | L-I |
| 0-SHOT | 42.5 | 28.6 | <b>65.2</b> | 51.1 | 26.6 | 6.1 | 41.3 | <b>23.6</b> |
| 5*-SHOT | <b>43.2</b> | <b>37.6</b> | 64.9 | <b>57.3</b> | <b>29.2</b> | <b>13.7</b> | <b>43.1</b> | 23.4 |
| $\text{BOOK}_{\text{all}}$ | <u>42.6</u> | 24.3 | 64.4 | 48.9 | 22.2 | – | 38.7 | – |
| $\text{BOOK}_{\text{para}}$ ( $p$ ) | <u>42.5</u> | <u>28.6</u> | <u>64.9</u> | <u>52.6</u> | <u>25.8</u> | <u>6.7</u> | <u>41.8</u> | <u>11.8</u> |
| $\text{BOOK}_{\text{non-para}}$ ( $\neg p$ ) | 41.8 | 24.5 | 64.5 | 48.4 | 19.3 | 5.6 | 34.5 | 10.1 |
| TYP 0-SHOT | 42.4 | 23.2 | 64.6 | 49.5 | 21.1 | 4.3 | 33.9 | 23.4 |
| TYP + $\text{BOOK}_{\text{para}}$ | 41.8 | 22.0 | <u>64.9</u> | 49.1 | 21.9 | – | 34.5 | – |

## 5 RESULTS & ANALYSIS

**Grammar versus parallel sentences for translation** We disentangle the signal from grammar books’ explanations and parallel sentences for translation. Our  $\text{kgv}$  results in Table 2 show that most or all performance improvements stem from the book’s parallel sentences, with quality plummeting when parallel data is removed. With Gemini into  $\text{eng}$ ,  $\text{BOOK}_{\text{p}}$  marginally outperforms  $\text{BOOK}_{\text{all}}$ , and beats  $\text{BOOK}_{\neg p}$  by 7 CHRf++, while into  $\text{kgv}$ ,  $\text{BOOK}_{\text{p}}$  outperforms  $\text{BOOK}_{\neg p}$  by over 8 points, and  $\text{BOOK}_{\text{all}}$  performs 3 points better than  $\text{BOOK}_{\neg p}$ . However, we show statistically in Section 5.1 that this small improvement is modelled directly by an increase in test set vocabulary coverage, rather than from the grammatical explanations. Additionally, this gap closes with the  $\text{PARA}_{\text{book}}^{\text{IGT}}$  prompt, which preprocesses and structures the parallel data in  $\text{BOOK}_{\text{p}}$  into  $\text{kgv-gloss-eng}$  triples.  $\text{PARA}_{\text{book}}^{\text{IGT}}$  performs particularly well for Llama-I, with over 10 points improvement over  $\text{BOOK}_{\text{p}}$  into  $\text{kgv}$ . Due to context restricting  $\text{kgv}$   $\text{BOOK}_{\text{all}}$  tests, conclusions with Llama-I are limited, but we find again that  $\text{BOOK}_{\neg p}$  performance lags far behind  $\text{BOOK}_{\text{p}}$ . We note that baselines including 0-SHOT show  $\text{kgv}$  translation is non-trivial. We also find that additional parallel data further improves translation quality, and note 5\*-SHOT is generally competitive despite its short average prompt, achieving the best  $\text{BOOK}_{\text{p}}$  score into  $\text{kgv}$  with Gemini. Thus for  $\text{kgv}$  translation, both LLMs on test mainly learn from the book’s parallel sentences, failing to exploit the grammatical explanations.

We observe a similarly strong trend for  $\text{npi}$  and  $\text{gug}$ , seen low-resource languages with high-quality FLORES test sets, in Table 3.  $\text{BOOK}_{\text{p}}$  settings largely match or outperform  $\text{BOOK}_{\text{all}}$  for both models and languages (except Llama-I in  $\text{gug} \rightleftarrows \text{eng}$  where the model often fails to output translations on the first line for  $\text{BOOK}_{\text{p}}$  settings). Few settings beat 0-SHOT and differences between Gemini settings (especially  $\text{npi}$ ) are smaller than for  $\text{kgv}$ ; perhaps the model’s prior competence (and a shorter  $\text{npi}$  grammar book) mean there is less to be gained. However, analysing  $\text{BOOK}$  settings in isolation shows that both  $\text{BOOK}_{\text{all}}$  and  $\text{BOOK}_{\neg p}$  have a detrimental effect of up to 7 points below 0-SHOT, while  $\text{BOOK}_{\text{p}}$  has a neutral or small positive impact in both  $\text{npi}$  and  $\text{gug}$ . Finally 5\*-SHOT is again effective, especially for Llama-I into  $\text{npi}$  and  $\text{gug}$ , likely due to the greater vocabulary coverage of the example set. These results generalise our findings for  $\text{kgv}$  to seen low-resource languages: we find no evidence that LLMs can effectively exploit grammatical explanations *for translation*.

**Fine-tuning versus in-context learning** We test a standard MT approach for adding a new language by fine-tuning NLLB, a small MT model, on the book’s parallel data, shown in Table 4. NLLB achieves competitive or improved performance compared to prompting Gemini with the same preprocessed parallel data,  $\text{PARA}_{\text{book}}$ . We also test backtranslation (BT), a standard method to boost performance in MT (Sennrich et al., 2016). A single BT iteration with  $\text{PARA}_{\text{train}}$  has a negative impact into  $\text{kgv}$ , likely due to the poor quality of the initial model introducing excessive noise. However we see a boost of 3 CHRf++ into  $\text{eng}$ , we expect because of the strong English language modelling of NLLB. Further, adding a small 400 example parallel training set sees large gains of 4-8 points. These results suggest the MTOB benchmark can be adequately addressed as a standard XLR MT problem with simple data preprocessing, a small pre-trained model, and fine-tuning on a single GPU for 1 hour.

{7}------------------------------------------------

Table 4: Translation results for  $\text{eng} \rightleftarrows \text{kgv}$  with NLLB, an MT model, fine-tuned on  $\text{PARA}_{\text{book}}$  data; equivalent in-context learning results with Gemini are shown for comparison. Fine-tuned NLLB achieves competitive results with an LLM given the same parallel data, especially into  $\text{kgv}$ .

| Setting <sub>↓</sub> | CHRf++ |  |  |  |  |
|-|-|-|-|-|-|
|  | $\text{eng} \rightleftarrows \text{kgv}$ |  | $\text{kgv} \rightleftarrows \text{eng}$ |  |  |
|  | Model <sub>↓</sub> | Gemini | NLLB | Gemini | NLLB |
| $\text{PARA}_{\text{book}}$ |  | 26.6 | 34.2 | 33.1 | 28.6 |
| + $\text{PARA}_{\text{train}}$ |  | 33.4 | 38.7 | 38.5 | 36.9 |
| + BT $\text{PARA}_{\text{train}}$ |  | – | 32.0 | – | 31.6 |

We also fine-tune Llama base on  $\text{PARA}_{\text{book}}$  to give  $\text{Llama-ft}$ , with results in Table 2. We find all  $\text{Llama-ft}$  settings beat equivalent  $\text{Llama-I}$  tests with  $\text{BOOK}_{\text{all}}$  data, except for  $\text{PARA}_{\text{book}}^{\text{GT}}$  settings with glosses which marginally outperform  $\text{Llama-ft}$  0-SHOT results. Prompting  $\text{Llama-ft}$  with parallel data in-context further improves performance over 0-SHOT by up to 10 points. We additionally fine-tune Gemini on  $\text{PARA}_{\text{book}}$ , with results in Appendix G, finding  $\text{Gemini-ft}$  underperforms NLLB and Gemini with the same data in-context by 6-12 CHRf++; we expect this is because it is already extensively instruction-tuned. Thus fine-tuning—particularly of small MT models—is a cheap method for achieving competitive results with prompting instruction-tuned long-context LLMs, given the same parallel data.

**Typological prompting for linguistic tasks** Given the limited contribution of grammatical explanations to translation performance, we introduce a novel prompting method summarising languages’ typological features. This prompt is intended to replace  $\text{BOOK}_{\text{p}}$ , thus we are primarily focused on results when combined with  $\text{BOOK}_{\text{p}}$  data. Our results for  $\text{eng} \rightleftarrows \text{kgv}$  translation in Table 2 show expectedly poor 0-SHOT performance due to the lack of any Kalamang text. Into  $\text{kgv}$ , our prompt beats  $\text{BOOK}_{\text{p}}$  but not  $\text{BOOK}_{\text{all}}$ ; however into  $\text{eng}$ , our prompt with  $\text{BOOK}_{\text{p}}$  achieves the best translation results for settings with book parallel data. For  $\text{npi}$  in Table 3,  $\text{TYP} + \text{BOOK}_{\text{p}}$  is less effective than  $\text{BOOK}_{\text{all}}$  into  $\text{npi}$ , and marginally outperforms it into  $\text{eng}$  up to 0.5 CHRf++, though  $\text{BOOK}_{\text{p}}$  alone performs best; similarly in  $\text{gug}$  tests,  $\text{BOOK}_{\text{p}}$  outperforms  $\text{TYP} + \text{BOOK}_{\text{p}}$ , which beats or matches  $\text{BOOK}_{\text{p}}$ . The performance of typological prompting for translation is therefore inconsistent, supporting the above finding that LLMs fail to effectively exploit grammatical information *for MT*.

![Figure 1: Horizontal bar chart showing Grammaticality judgment accuracy in kgv for three test settings: Swap_adj, Swap_tran, and Shuffle. The x-axis represents Accuracy (%) from 30 to 100. The y-axis lists the test settings. For each setting, six bars represent different prompting methods: 0-shot (yellow), 10*-shot (green), Book_all (pink), Book_p (purple), Book_p (orange), and Typ + Book_p (teal). Gemini scores are 100%, 99%, and 100% for the three settings respectively. Typ + Book_p consistently shows the highest accuracy across all settings, reaching 83% in the Shuffle setting.](7ff005f9556dc6518981bb92091d36ab_img.jpg)

| Test Setting | 0-shot | 10*-shot | Book <sub>all</sub> | Book <sub>p</sub> | Book <sub>p</sub> | Typ + Book <sub>p</sub> |
|-|-|-|-|-|-|-|
| Swap <sub>adj</sub> | 56% | 63% | 63% | 63% | 63% | 65% |
| Swap <sub>tran</sub> | 57% | 65% | 70% | 75% | 76% | 76% |
| Shuffle | 54% | 71% | 76% | 80% | 62% | 83% |

Figure 1: Horizontal bar chart showing Grammaticality judgment accuracy in kgv for three test settings: Swap\_adj, Swap\_tran, and Shuffle. The x-axis represents Accuracy (%) from 30 to 100. The y-axis lists the test settings. For each setting, six bars represent different prompting methods: 0-shot (yellow), 10\*-shot (green), Book\_all (pink), Book\_p (purple), Book\_p (orange), and Typ + Book\_p (teal). Gemini scores are 100%, 99%, and 100% for the three settings respectively. Typ + Book\_p consistently shows the highest accuracy across all settings, reaching 83% in the Shuffle setting.

Figure 1: Grammaticality judgment accuracy in  $\text{kgv}$ ; for reference in  $\text{eng}$  tests, Gemini scores 100%, 99%, and 100% respectively. Our prompt  $\text{TYP} + \text{BOOK}_{\text{p}}$  performs best overall suggesting grammar can help LLMs for linguistic tasks.

To determine whether grammar is not useful for MT or LLMs cannot exploit grammatical explanations more broadly, we test two more relevant tasks: grammaticality judgment and IGT prediction. In Figure 1, grammaticality judgment results in  $\text{kgv}$  with Gemini show all settings perform similarly poorly on  $\text{SWAP}_{\text{adj}}$ , though improving on 0-SHOT by around 7%. Generally, 10\*-SHOT is worse than prompts with  $\text{BOOK}_{\text{p}}$ , likely because diverse sentences may help here more than overlapping

{8}------------------------------------------------

Table 5: IGT prediction results in *kgv* for supervised baselines and *Gemini* settings. Our **TYP + BOOK<sub>para</sub>** prompt achieves the highest morpheme accuracy and high scores on other metrics, while **BOOK<sub>all</sub>** performs poorly overall.

| Model | Morph Acc. | Word Acc. | Stem F1 | Gram F1 | CHRf++ |
|-|-|-|-|-|-|
| TOP-CLASS (Ginn et al., 2023) | 44.0 | 39.7 | 40.6 | 57.8 | 34.5 |
| SMP-BASE (Ginn & Palmer, 2023) | 45.2 | 41.7 | 39.7 | <b>58.9</b> | 34.3 |
| TÜCL-MORPH (Girrbach, 2023) | 43.6 | 38.8 | 40.0 | 50.7 | 35.4 |
| BYT5-FT (Xue et al., 2022) | 40.8 | <b>48.6</b> | 40.9 | 45.4 | 49.0 |
| GLOSSLM-FT (Ginn et al., 2024b) | 43.8 | 47.7 | 41.5 | 50.4 | <b>49.1</b> |
| 10*-SHOT | 43.9 | 43.7 | <b>44.3</b> | 45.2 | 46.4 |
| BOOK <sub>all</sub> | 40.1 | 31.5 | 38.7 | 43.4 | 40.5 |
| BOOK <sub>para</sub> ( <i>p</i> ) | 45.4 | 42.1 | 44.0 | 49.0 | 45.0 |
| BOOK <sub>non-para</sub> ( <i>¬p</i> ) | 21.0 | 8.8 | 23.9 | 15.6 | 26.0 |
| <b>TYP + BOOK<sub>para</sub></b> | <b>46.1</b> | 40.9 | 44.2 | 50.5 | 44.8 |

vocabulary, which helps more for MT. For BOOK settings we observe that BOOK<sub>*p*</sub> matches or outperforms BOOK<sub>all</sub> across all three tests by up to 5%, and consistently beats BOOK<sub>*¬p*</sub>, by up to 18% in SHUFFLE tests. So far, the LLM still fails to exploit grammatical explanations effectively and learns mainly from parallel examples. However, our TYP + BOOK<sub>*p*</sub> setting performs best over the three tests by up to 3% over BOOK<sub>*p*</sub>. These positive results suggest that LLMs *can* learn from grammar, given the right kind of grammatical knowledge and a relevant task.

For *kgv* IGT prediction, we compare *Gemini* settings with supervised baselines in Table 5. The leading performer in morpheme accuracy, the key IGT metric, is again our typological prompt **TYP + BOOK<sub>*p*</sub>**, scoring 6% above BOOK<sub>all</sub>, 0.5% over BOOK<sub>*p*</sub>, and 25% over BOOK<sub>*¬p*</sub>. Additionally, our prompt beats all supervised systems by 1-5%, suggesting in-context learning with typological knowledge and parallel glossed examples is a strong method for XLR IGT prediction. Results for other metrics show slightly differing trends, with supervised models showing stronger word accuracies and Gram F1 scores (since most are closed-set classifiers). Generally though, BOOK<sub>*¬p*</sub> shows extremely poor performance, while BOOK<sub>*p*</sub>, 10\*-SHOT, and TYP + BOOK<sub>*p*</sub> settings perform consistently well, often beating supervised baselines. We note that TYP + BOOK<sub>*p*</sub> scores show competent performance for both grammatical (on morpheme accuracy and Gram F1) and lexical aspects (via Stem F1) of IGT prediction, suggesting all-round competence on this task. These results reinforce our findings that while parallel sentences still provide most of the useful signal, LLMs can exploit grammatical—specifically typological—information for linguistic tasks.

### 5.1 ANALYSIS

**Type coverage and Token efficiency** We investigate whether any added performance from grammatical explanations is statistically significant or can instead be attributed to greater test set type coverage in the prompt. We distinguish between types, meaning unique words in a vocabulary, and tokens, i.e. individual occurrences of types in a text. We fit univariate least squares regression models to CHRf++ scores with test set *type* coverage as the independent variable, for both directions, shown in Figure 2. All settings fall within the 95% confidence interval of the regression lines, and the models are significant in both directions ( $p < 0.005$ , F-test)<sup>4</sup>; the Pearson correlations are also significant ( $p < 0.005$ ). Thus maximising target vocabulary coverage (via parallel sentences) in-context is the most efficient method for improving LLM-based XLR translation. These linear regressions show that translation performance can be directly modelled by test set vocabulary coverage, and that the book’s grammar explanations provide no *significant* advantage over its parallel sentences. See Appendix F for full statistics on our prompts’ test set type coverage.

We then explore whether the improved translation scores can be attributed to a longer (or shorter) prompt, by testing for a relationship between prompts’ total *tokens* and translation quality in terms of CHRf++ for BOOK<sub>{all/*p*/¬*p*}</sub> with *Gemini*. The resulting linear models are not significant in either direction ( $p = 0.997$ ,  $p = 0.78$  into and from *kgv*, F-test), with no significant Pearson correlations.

<sup>4</sup>For details of these and following statistical tests, see Appendix B.

{9}------------------------------------------------

![Figure 2: Two scatter plots showing regression models of CHRf++ score against test type coverage for eng-kgv and kgv-eng translation. Both plots show a positive correlation with regression lines and 95% confidence intervals. Data points are labeled with prompt settings like '0-shot', '1-shot', '5-shot', and various combinations of typology (Typ), parallel examples (P_parallel), and grammar books (Book).](4e0ade2f41b66d5602160da5cc978274_img.jpg)

The figure contains two scatter plots. The left plot is titled 'eng → kgv' and the right plot is titled 'kgv → eng'. Both plots have 'Test Type Coverage (%)' on the x-axis (ranging from 0 to 100) and 'ChrF++' on the y-axis (ranging from 10 to 50). Each plot features a red regression line and a pink shaded area representing the 95% confidence interval. The regression equation for the left plot is  $y = 0.37x + 14.38$ , and for the right plot is  $y = 0.31x + 13.46$ . Data points are labeled with abbreviations: '0-shot', '1-shot', '5-shot', 'P\_parallel', 'Typ', 'Book', and combinations like 'Typ + P\_parallel', 'Book + P\_parallel', 'Typ + Book', and 'Book + W'. The points generally follow an upward trend, indicating that as test type coverage increases, the ChrF++ score also increases.

Figure 2: Two scatter plots showing regression models of CHRf++ score against test type coverage for eng-kgv and kgv-eng translation. Both plots show a positive correlation with regression lines and 95% confidence intervals. Data points are labeled with prompt settings like '0-shot', '1-shot', '5-shot', and various combinations of typology (Typ), parallel examples (P\_parallel), and grammar books (Book).

Figure 2: Regression models of CHRf++ score against test type coverage for *eng*–*kgv* and *kgv*–*eng* translation with *Gemini*. Prompt settings are labelled with abbreviations for clarity. The plots show that translation performance can be statistically modelled by test set vocabulary coverage.

The grammar book is therefore both a token-inefficient way to learn (with similar performance despite nearly 5x more tokens than *kgv* *BOOK*<sub>*p*</sub>), and a cost-inefficient dataset to generate, compared to using its parallel sentences. The needle-in-a-haystack problem could partially explain this: with increasing context, retrieval of relevant information (i.e. similar parallel examples) becomes harder (Hsieh et al., 2024), so while *BOOK*<sub>*p*</sub> is a subset of *BOOK*<sub>*all*</sub>, there is a greater ratio of relevant to irrelevant information in the prompt—assuming grammatical explanations cannot be effectively exploited for translation.

**Discussion** We note that our results do not indicate LLMs cannot understand books in general; rather, we find no quantitative evidence that the results here and in MT0B show LLMs can effectively exploit grammar books (or linguistic knowledge) *for translation*. Indeed, we show that LLMs can exploit grammatical information in the form of typology for more relevant, linguistically-focused tasks. More broadly, from an educational perspective, translation is a problem-solving task aiming to reach a goal state (translation) via a series of actions given an initial state (source) and optionally rules on applying actions. Humans tend to learn this kind of task more efficiently via worked-examples (van Gog et al., 2019), i.e. with explicit explanations, rather than pure discovery learning, meaning without explicit guidance (Mayer, 2004). Our results however indicate that for translation, LLMs learn more effectively from unannotated parallel examples (i.e. discovery) than from grammar principles with explained examples (i.e. example-based). Our results thus tentatively support a divergence between learning strategies for translation between human learners and LLMs learning in-context. We suggest that this may partially stem from prompts with parallel data aligning more closely with LLMs’ instruction-tuning data than grammar book explanations.

## 6 CONCLUSION

We find no evidence that LLMs can effectively exploit grammatical explanations for low and extremely low-resource MT in Kalamang, Nepali, and Guarani, instead finding that LLMs rely on the parallel sentences within the book. This runs counter to the claim of prior work including MT0B which use grammar books to enable LLMs’ performance on XLR tasks. We show that fine-tuning small MT models matches the performance of costly long-context LLMs. Further, we show statistically that grammatical explanations add no significant advantage above the increased type coverage they provide, and that grammar books are less token-efficient for prompting than parallel sentences. However, LLMs *can* exploit grammatical information, given an appropriate task—e.g. grammaticality judgment or IGT prediction—and more useful grammatical data in the form of our typological prompt, which achieves leading results on these linguistic tasks. We therefore emphasise the importance of task-appropriate data: parallel data for MT, and grammatical, preferably typological, knowledge for linguistic tasks. Moreover, we suggest data collection efforts for multilingual XLR tasks, at least for MT, are better focused on parallel data over linguistic description, which enables less costly, more token-efficient translation.

 Rest of paper (reference and Appendix) is removed.