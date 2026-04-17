# Neologism Learning for Controllability and Self-Verbalization

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Humans invent new words when there is a rising demand for a new useful concept
(e.g., doomscrolling). We explore and validate a similar idea in our communication
with LLMs: introducing new words to better understand and control the models,
expanding on the recently introduced neologism learning. This method introduces a
new word by adding a new word embedding and training with examples that exhibit
the concept with no other changes in model parameters. We show that adding a new
word allows for control of concepts such as flattery, incorrect answers, text length,
as well as more complex concepts in AxBench. We discover that neologisms can
also further our understanding of the model via self-verbalization: models can
describe what each new word means to them in natural language, like explaining
that a word that represents a concept of incorrect answers means “a lack of complete,
coherent, or meaningful answers. . . ” To validate self-verbalizations, we introduce
plug-in evaluation: we insert the verbalization into the context of a model and
measure whether it controls the target concept. In some self-verbalizations, we find
machine-only synonyms: words that seem unrelated to humans but cause similar
behavior in machines. Finally, we show how neologism learning can jointly learn
multiple concepts in multiple words.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes neologism learning, a parameter-efficient method for controlling LLM behavior. The method involves adding a new token embedding to a frozen LLM and training only this new embedding on examples of a target concept. The paper introduces two concepts for interpretability: 1) self-verbalization, where the model can provide natural language descriptions (like synonyms) for its newly learned neologism. 2) plug-in evaluation, a method to validate these verbalizations by plugging them back into prompts and measuring if they control the target concept.

### Strengths
* The core idea of training the neologism using a parameter-efficient method is interesting and practical.
* Metrics such as self-verbalization and plug-in evaluation are a simple, effective, and easily reproducible method for validating the method in interpretability research.
* The discovery of "machine-only synonyms" (e.g., "lack" for brevity) is interesting and opens up a new line of research.

### Weaknesses
### 1. Data
The paper proves its method works on clean prompts (LIMA) to produce clean synthetic outputs (Gemini-style). How about more practical, messy, real-world prompts?

### 2. Model
The primary experiments are on Gemma-3-4B-IT, a relatively small model. While the "lack" example anecdotally transfers to Gemini-2.5-Flash, the paper lacks a systematic study of how neologism learning, and especially self-verbalization, scales with model size and architecture

### 3. Metric
To support the claim of controllability, metrics measuring some side effects would be valuable. (e.g., when they use flattery-word, maybe this makes response longer or harmful).

### Questions
* The paper shows learning 3 neologisms. What happens when you try to learn 100? Does the model's base knowledge degrade?
* If you train two different neologisms (from different random or word-based initializations) on the exact same dataset of a concept, do they converge to similar embedding vectors? More importantly, do they produce the same self-verbalizations?
* The paper compares against a few-shot ICL, but a more direct comparison to other parameter-efficient steering methods would be interesting.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose Neologism Learning, a method that incorporates new concepts (i.e., neologisms) into a Large Language Model (LLM) solely focusing on the embedding projections,  that is without updating the parameters of the model. The authors suggest that, once trained, specific concepts such as “give a short answer”, “give a flattering answer”, or compositions thereof, placed in a prompt (e.g., at the end of a sentence like “describe the recipe of a lasagna”), allows to “control” the model’s behavior in a somewhat reliable way. The authors further suggest that the model can perform what they call “self-verbalization”, i.e.,  explain back (e.g., via listing a bunch of synonyms) the meaning of these neologisms. Finally, they observe that some of these synonyms work as machine-only concepts: they modify the model’s behavior even though they are meaningless to humans.

### Strengths
1. The paper is clearly written and well structured.

2. Although the core methods that give rise to the main method of this work are not novel, I find the overarching goal, i.e., Neologism Learning, to be a fresh perspective on LLM evaluation and output controllability.

### Weaknesses
1. I do not believe that the baselines proposed by the authors are fair. Given that the authors use “neologisms” related to concepts that already have a clear semantic meaning (e.g., give a short answer), it seems that a proper comparison would be to compare the model’s behavior output with a prompt incorporating a semantic description of the concept.

2. It is unclear whether we can consider the proposed concepts as “neologisms”. The latter are associated with (intrinsically) novel societal concepts, like the one provided by the authors in their paper (i.e., doomscrolling). In contrast, it seems that this work merely proposes novel bindings, a “pseudo-word” to an already existing meaning (via optimizing the NLL of a “bundle” of sentences that relate to an already existing concept). Hence, rather than learning a novel concept, LLMs may simply be learning to associate already encoded specific sequential token patterns to a novel input token. In fact, the authors at some point discuss out-of-distribution generalization which I think is a gross overstatement of the results presented in this work.

3. In fact, in light of my second point, the results of the plug-in evaluation method should not come as a surprise. The objective function pushes the novel (learned) concepts to be semantically associated (as in a knowledge graph) with words that relate new concepts’ description (or sentences it relates to). Thus, the results of the plug-in evaluation emerge from the ability of these models to perform 1-hop associative bindings of concepts and their semantic descriptions. In sum, these results would have been much more robust and convincing if the authors had focused on associating new tokens, to completely new concepts (such as doomscrolling).

### Questions
1. Have the authors compared the model's output to a non-default response, i.e. incorporating in the prompt instructions for the proper behavior with the semanctic description of the concept itself.

2. Have the authors thought of building completely novel concepts, that is related to semantic descriptions that may even be considered as unreal (or surreal).

### Soundness
2

### Presentation
3

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
The study introduces neologism learning, which has been studied by expanding vocabulary to the model, while existing word embeddings are held frozen, new words are learned with new word embeddings. It validates these tokens via self-verbalization an approach, called plug-in evaluation, has been introduced where verbalization is inserted into the context of a model and measure it controls the target concept.

### Strengths
- The approach is simple: "Neologism learning" adds only new token embeddings (with the backbone frozen); however, it reliably modulates behaviors, brevity vs. verbosity, single-sentence outputs, and flattery/refusal/incorrect answers.
- Self-verbalization with plug-in evaluation in which the model explain its learned token (synonyms or free-form description), then replace the token with the verbalization and test whether control persists. This reveals machine-only synonyms (e.g., lack -> brevity) that transfers across models.
- Ablations study: It demonstrated that multiple prompt templates helps in improving the performance.

### Weaknesses
- The evaluation of fully relying on LLM-judge with three scores is not justified very well. Any human evaluation is conducted? 
- As per as Table 3, it shows that overall original score improved compared to the baseline, however, it is not very much clear what higher concept scores by LLM-judge entails? 
- While the plug-in works on average, synonym quality is uneven, leaving broader reliability across prompts/tasks uncertain.

Typos/Grammatical issues
- we use two methods 1. add a directive --> we use two methods: 1. add a directive
- L257: "the concept concept “words related to sensory experiences and physical interactions”)." --> concept repeated

### Questions
- Did you run any manual evaluation to validate judge scores?
- Only five text-genre concepts are reported. Any results for other genres or for concepts?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper expands on a prior position paper introducing "neologism learning", a method which adds new token embeddings to an LLM's vocabulary and tunes them in isolation on a small amount of samples demonstrating a new concept. The authors show that these neologisms can be used to steer the model, e.g. a neologism token that represents the idea of brevity can be used to elicit short responses from an LLM (here, Gemma). The authors find that LLMs can self-verbalize learned neologisms and occasionally find synonyms that are machine-only (that is, they are unintuitive for humans but shared across LLMs – here, Gemma and Gemini). The paper explores simple concepts as well as more challenging/abstract ones (from AxBench), and finally explores the composition of concepts. For systematic evaluation, the authors rely on LLM-graded plug-in evaluation of the model's generated self-verbalizations.

### Strengths
1. The paper is very well written with a strong narrative.
2. The paper recombines simple and existing techniques (such as vocabulary expansion and prompting/verbalization) in novel ways to systematically demonstrate the usefulness of neologism learning to better understand and steer/control LLMs.
3. The authors highlight neologism learning for steering, but if developed further I imagine it could also be useful for context compression and more efficient agent-to-agent communication.
4. The authors really test the limits of the method through the compositionality experiment and make well-tempered claims.

### Weaknesses
1. The authors only study Gemma and Gemini models. I consider this a methodological weakness for two reasons: a). Findings may overall not generalize to other model families. b). To my knowledge Gemma is built through distillation from Gemini, likely uses a similar tokenizer and a subset of Gemini's training data. As such, the finding that machine-only synonyms transfer from Gemma to Gemini may not be too surprising. It would be much more interesting to see whether they can transfer to a model guaranteed to have a different tokenizer and training data.
2. The conclusion section is quite vague (e.g. "pushes the frontier of communication with what language models have learned") and would benefit from an elaboration on the practical significance of the findings / actionable takeaways, and an outlook on potential future applications or directions to make neologism learning a practical tool for researchers and practitioners.
3. The authors rely on LLM scoring (with Gemini) for the self-verbalization experiment,  which might bias the results. It would help to (at least partially) calibrate the results through some human evaluations.
4. Something that is, in my eyes, missing is an experiment on whether the LLM learns to use the neologism without being explicitly prompted for it – I assume it won't, and that is a limitation that prevents using neologisms to make communication with/among LLMs more efficient. Neologism learning makes the model recognize and assign meaning to new special tokens (and this allows steering the model through them) but there may not be any emergent lexical use.
5. There are very strong similarities of the neologism learning technique with methods from the multilinguality (tokenizer/vocab adaptation to support new languages) and PEFT literature (e.g. prefix tuning and soft prompts), and there is not enough discussion on these connections.

Minor issues:
- There are some citations that are ill-formatted (using in-line citations instead of parentheses, probably due to using `\cite` instead of `\citep` and `\citet`)
- There is a typo in the conclusion ("contrasticely")

### Questions
1. Regarding the "aperitif" on the machine-only synonym "lack": Why is this word implicitly treated as a noun? I'd think the word "lack" may just be the English verb "to lack sth.", and the fact that the model can work with a sentence like "give me a lack answer" may just come from the model's robustness to poor grammar and Translationese/artifacts of multilinguality.
2. How do you determine what is a "semantically vacuous word"? Is this something you just eye-balled or is there an automated procedure that can be evaluated objectively? And is a word here strictly one token?
3. Will there be any code or data release to facilitate reproducing these results and turning the method into a practical tool?

### Soundness
3

### Presentation
4

### Contribution
3
