# LLMS CAN HIDE TEXT IN OTHER TEXT OF THE SAME

## LENGTH


**Antonio Norelli & Michael Bronstein**
University of Oxford
Project CETI


ABSTRACT


A meaningful text can be hidden inside another, completely different yet still
coherent and plausible, text of the same length. For example, a tweet that celebrates
a political leader could hide a tweet containing a harsh critique against the same
leader, or an ordinary product review could conceal a secret manuscript. This
uncanny possibility is now within reach thanks to Large Language Models; in
this paper we present _Calgacus_, a simple and efficient protocol to achieve it. We
show that even modest 8-billion-parameter open-source LLMs are sufficient to
obtain high-quality results, and a message as long as this abstract can be encoded
and decoded locally on a laptop in seconds. The existence of such a protocol
demonstrates a radical decoupling of text from authorial intent, further eroding
trust in written communication, already shaken by the rise of LLM chatbots.
We illustrate this with a concrete scenario: a company could covertly deploy an
unfiltered LLM by encoding its answers within the compliant responses of a safe
model. This possibility raises urgent questions for AI safety and challenges our
understanding of what it means for a Large Language Model to know something.


1 INTRODUCTION


LLMs sparked a revolution. Text is no longer, by default, the trace of a human thought or intention.


This marks a dramatic break in history—or perhaps the end of history itself—if we consider that
history began with writing, and that one of the defining properties of writing, until now, has been its
status as a product of human intention. In this paper, we are going to present a protocol that highlights
this new reality in its most extreme form, perhaps offering an opportunity to better understand it.


Our protocol _Calgacus_ allows encoding an arbitrary meaningful text within a different well-formed
and plausible text of the same length, using a Large Language Model (LLM). That is, hiding a tweet
that criticizes a political leader within a tweet that celebrates that same political leader, or the first
page of the unreleased 8th Harry Potter book within a review of a Virtual Reality videogame, with
the original text exactly recoverable by anyone possessing the key (Figure 1).


The topic and tone and style of the fake text are steerable, while the length of the fake text is the
same as the original text being hidden, in terms of LLMs tokens. This symmetry prevents one from
establishing at first sight which text is authentic when we have one next to the other. Also, the method
is efficient: an entire article can be encoded and decoded on commodity hardware in seconds.


This possibility opens deep questions and intriguing applications. What is the real meaning of the text
we are reading? Who is the author of the videogame review, and what was the intent behind it? Is it a
hallucination? This protocol allows crafting anti-government content disguised as pro-government
messages, suitable for publishing on censored platforms in oppressive countries. Or, it could be used
by a shady tech company to offer the services of an unfiltered LLM by only exposing compliant
answers from a trusted LLM. All these matters will be taken up in our concluding discussion.


Correspondence to Antonio Norelli <noranta4@gmail.com>. A demo sufficient to reproduce the main results
in the paper within minutes, even from smartphone, is available at: [https://github.com/noranta4/](https://github.com/noranta4/calgacus)
[calgacus.](https://github.com/noranta4/calgacus) A curated Italian translation of this paper can be found at [https://arxiv.org/abs/](https://arxiv.org/abs/2510.20075v5)
[2510.20075v5.](https://arxiv.org/abs/2510.20075v5)


1


Figure 1: **Three texts.** **The second one is the original.** Following the protocol described in this paper,
each of the other two was built solely to hide the original text, which can be perfectly reconstructed
by anyone who knows the secret key. The key also steers the fake text: here, _k_ 1 = _Here it is:_ _the_
_infamous British roasted boar with mint sauce._ _How to make it perfect._ and _k_ 2 = _I stand before you_
_to exalt the noble deeds of Gaius Julius Caesar._ More examples in Figure 13.


The paper is structured as follows: it starts by introducing steganography, the discipline concerned
with concealed ( _steganós_ ) writing ( _graphia_, from Greek), and discussing the vast impact of generative
AI in the field, with a focus on Large Language Models. We then present _Calgacus_, the method
to encode a meaningful text into another meaningful text of the same length using a LLM. After
introducing a measure to assess the soundness of the fake texts produced, we test our method on
Reddit posts. While remaining opaque to humans, we show that LLMs can uncover a distinction
between original texts and most of their encoded counterparts. But not all, as we will notice in the
following section, where we discuss the security of the protocol. Finally, we conclude by discussing
the method’s core implication—the radical decoupling of text from authorial intent—and present a
concrete application that raises pressing questions about AI safety and the nature of knowledge in
Large Language Models.


2 RELATED WORK


**Steganography.** The art and science of hiding a message and, at the same time, the presence of a
hidden message is known as steganography, see Figure 2. This is different from cryptography, that
instead does not conceal the presence of a hidden message and only deals with the hardness of its
revelation. Cryptographers discuss about lockers, steganographers about inconspicuous hiding spots [1] .


Perhaps, it is this limited size of the object of investigation that allowed cryptographic models to
florish by achieving mathematical rigor and strong security guarantees. By contrast, a model of
steganography should describe entire domains of data and how they are consumed by humans, such
as text, audio, or images, to predict where information can be hidden. Formal models exist, but at the
cost of rather unrealistic assumptions that hinder their practical usage, such as being able to exactly
quantify the plausibility of any possible text. Emblematically, this somewhat disappointing state
of affairs is presented by Cachin (1998) besides one of the most popular mathematical models for
steganography as of today, based on the hypothesis-testing framework, but still limited to highly
idealized assumptions. Although modern generative AI techniques have made these assumptions


1This metaphor is inspired by a pleasant piece on the history of steganography by Kahn (1996)


2


Figure 2: **An** **example** **of** **steganography.** In Asterix in Britain (Goscinny and Uderzo, 1966) a
smuggled barrel of magic potion is hidden among innocent-looking Gaulish wine.


closer to reality, the unreliability of their predictions remains unbounded. For this reason, we will
avoid building a palace on the sand, and not frame our method in a formal model of steganography,
limiting our discussion to how meaningful our fake texts look, with some quantitative arguments.


**Some terminology.** In traditional steganography, we start from an original, innocent-looking content
(such as an image, audio file, or text) and subtly edit it to embed a secret message. The original content
is referred to as the _covertext_, while the result containing the hidden message is the _stegotext_ . In our
case, however, the stegotext is generated directly from the secret message, without modifying a preexisting cover. We will refer to it as stegotext or fake text interchangeably. While the term covertext
will not refer to a specific object, but rather to a class of texts that the steganographic protocol is
designed to mimic. This approach has recently been referred to as _generative steganography_ (Liu
et al., 2018; Wei et al., 2022; Kim et al., 2023; Zhu et al., 2024; Wu et al., 2024; Tang et al., 2025).


**Large** **Language** **Models** **in** **a** **nutshell.** A language model is a program that, given some text,
estimates what is likely to come next. It does so by assigning probabilities to tokens—text fragments
consisting of common words or subwords (watch Karpathy, 2024, for a deeper look at text tokenization)—based on recurring patterns of tokens it has observed in a vast text corpus. At present, by far,
the most effective way to build a language model is to gradually adjust billions of parameters of a
neural network arranged in the Transformer architecture, such that with every adjustment, the error it
makes in predicting the last token on a batch of sentences from the corpus decreases (Vaswani et al.,
2017; Karpathy, 2023, original formulation and a more educational introduction to Transformers).
At each update—on the order of _∼_ 1M in total—the contribution of every parameter to the error is
assessed through backpropagation (Rumelhart et al., 1986). The result of this process is a Large
Language Model (LLM), typically operating over a vocabulary of 100k tokens. The most common
use of the probabilities produced by LLMs is to generate text, by choosing successive tokens one
after another according to the computed probabilities, a method known as autoregressive generation.


**Steganography** **and** **LLMs.** As mentioned earlier, the boom of deep learning and especially of
generative AI in recent years, provided us for the first time with convincing models encompassing
entire domains of real-world data, such as DINO for images (Caron et al., 2021), Jukebox for audio
(Dhariwal et al., 2020), and Large Language Models for text (Radford et al., 2019). The procedure
described in this paper stems from these advancements and is based on the availability of good
discrete autoregressive generative models, potentially on any domain, but we will focus on text.
Steganographic procedures based on LLMs are as old as them (Ziegler et al., 2019), and today come
with different perks: Meteor cleverly adjusts the number of bits encoded based on the entropy of the
next token (Kaptchuk et al., 2021), Wu et al. (2024) scheme works with black-box LLMs, without
needing to access logits or vocabulary, while the method presented by Zamir (2024) is able to encode
the secret message without modifying the response distribution of the LLM. What we add to the field
is _Calgacus_, a protocol with the notable property of having full capacity, that is, the stegotext and the
secret message being of the same length. The main interest of this paper is to discuss the implications
of this last fact and describe the method.


3


Figure 3: **How to hide a text in another text of the same length using a LLM. 1.** Tokenize _e_, the
text to hide. **2.** Evaluate its token probabilities using a LLM and record ranks. **3.** Prompt the LLM
with _k_ and generate _s_ following the recorded ranks rather than by sampling. Given _s_ and the LLM,
who knows the secret prompt _k_ can retrieve the original _e_ proceeding backwards.


3 METHOD


The method is very simple. It is described below as a recipe and illustrated with an example in Fig. 3.


_**Calgacus**_ **recipe.** Ingredients:


      - A good LLM with access to all the output logits. _(Why good?_ _See Appendix A.2)_


      - A text _e_ to hide.


      - A secret prompt _k_, which will affect the content and style the of the text _s_ you want to
hide _e_ in.


Procedure to hide _e_ in _s_ :


1. Tokenize _e_ using the LLM tokenizer, obtaining a list of tokens _e_ 1 _, e_ 2 _, e_ 3 _, . . ._

2. For each _ei_, denote by _ri_ its rank in the LLM’s probability distribution given the context
_e_ 1 _, . . ., ei−_ 1. Store the list of ranks _r_ 1 _, r_ 2 _, r_ 3 _, . . ._


3. Construct _s_ by generating text starting from _k_ using the LLM. At each step _i_, instead
of sampling from the probability distribution, choose the _ri_ [th] [most probable token.]


To recover _e_ from _s_, reconstruct _r_ 1 _, r_ 2 _, r_ 3 _, . . ._ by evaluating the probabilities of the tokens in _s_
after _k_, and then regenerate _e_ step by step using the LLM without _k_ by selecting every time the
_ri_ [th] [token.]


**Considerations**


      - If _e_ is sound, we expect ranks to be low, making tokens chosen after _k_ highly probable,
ensuring _s_ is coherent.


      - For the same reason, _s_ should align well with the context set by the secret prompt _k_ .


**Variations**


4


- Including an additional secret prompt _k_ _[′]_ before _e_ may help achieving lower ranks,
providing a better control over _s_ . A longer and more detailed _k_ can serve the same
purpose.


      - Here we have described a procedure with a single LLM to work on text, but in principle, we can put any discrete autoregressive generative model producing a probability
distribution on the next token in the encoding and decoding stage, see Appendix A.3.


**When** **the** **stegotext** _s_ **sounds** **like** **a** **real** **text.** In general, _s_ will be a coherent text when the
LLM can choose high-probable tokens to assemble it, and therefore when the ranks prescribed by
_e_ are low. In turn, the ranks of _e_ are low when the LLM is good at guessing _e_ tokens. If _e_ is
difficult to guess for the LLM, ranks will be high and _s_ will be gibberish; for instance the hash
_1f0ca711df81520887afe0dca099652a_ encoded using the same culinary prompt of Figure 1, produces
the broken _s_ : _The recipe written from deep cooks souls pocket magazine pages years long lost into_
_places wanting and_ . To lower further the ranks of _e_, it is possible to craft a prompt _k_ _[′]_ that sets the
context for _e_ . This comes at the cost of a larger private key, now including both _k_ and _k_ _[′]_, and to a
loss of universality, since _k_ _[′]_ would not help for a new _e_ out of _k_ _[′]_ context.


**A quantitative measure of the quality of the stegotext** _s_ **.** Measuring the meaningfulness of a text
is a longstanding linguistic challenge, and arguably an ill-posed problem. Here, for the purpose of
evaluating our method, we adopt soundness as a practical proxy for meaningfulness. Soundness refers
to the plausibility of the arrangement of symbols in a text. This is precisely what a LLM estimates:
the product of the probabilities of each token _a_ 1 _, . . ., an_ given the preceding ones yields an estimate
of the overall plausibility of the text _A_ :


_p_ ( _A_ ) =


_n_

- _p_ ( _at_ _| a_ 1 _, . . ., at−_ 1)


_t_ =1


This definition has a clear defect: longer texts are by construction less plausible. For instance, it
judges the text of this paper until this point _·_ less meaningful than the following string _iawundemè09_
_89huibqyfhwenah csyabdnar FI VNAOcijawo niwakhdb_, that is a difficult position to hold even for
reviewer 2. Following the example of Goldwasser et al. (2023, Figure 1 A-B), we will use this
definition only to compare the relative plausibility of two or more texts of the same token length [2]


Now, we would like to compare the plausibility of stegotexts produced by our method with the
plausibility of real texts. To do so, we took 1000 Reddit posts/comments as examples of real texts.
They come from different Reddit communities (subreddits) and are very heterogeneous in topic
and tone (Trimness8, 2025). We truncate them to be exactly 85 tokens long and compute their
probabilities as assigned by the LLM Llama 3 8b (Grattafiori et al., 2024). The Reddit texts are more
recent than Llama 3 and therefore cannot appear in its training corpus. We take three texts from the
1000 to produce 100 stegotexts for each with our method, and look at their probabilities compared to
the ones of real texts. We chose the three texts at _µ_, _µ −_ 2 _σ_, and _µ_ + 2 _σ_ of the real text distribution.
As seen in Figure 4, in every case, the probabilities associated to their stegotexts are within the real
text distribution. We build different stegotexts using different prompts as _k_ (a random subsample of
the prompts in (Akın, 2025)).


**How to distinguish the original from the fake text.** Despite remaining plausible and falling within
the real text distribution, on average the stegotexts _si_ are less probable than their corresponding
original text _e_, as observed in Figure 4. So to recap: while for a human both the original and fake
texts are plausible, generally the original text can be discerned from its stegotexts by picking the most
probable one according to a LLM. We verified this statement also using LLMs different from the one
used to generate the stegotexts. For instance, the same probability shifts between real and fake texts
can be observed when using Phi-3 3.8B in Figure 14.


**Low entropy token choices.** Why are stegotexts less probable than their originals for LLMs, even
though token ranks are preserved? Consider the text: _In the course of the Gallic wars, Britain was_


2Another possibility is to keep texts of any length and normalize the probability by the number of tokens,
as the popular metric perplexity, defined as 1 _/_ ~~[�]~~ _[n]_ _p_ ( _A_ ). But this normalization does not fully factor out length:

LLMs usually assign a smaller probability to the first tokens (Fig. 10), so shorter texts would be less plausible.


5


Figure 4: **Fake texts built with our procedure are plausible.** The figure shows the cumulative
log-probability assigned by a LLM (Llama 3 8b) to some collections of 85-token long texts. We can
interpret log-probability as a measure for the plausibility of a text: 1000 real Reddit posts/comments
act as real texts and span a large log-probability interval, but sequences of random ASCII characters
or English words do not fall within it. Instead, fake texts built with our procedure remain within the
plausibility of real texts, even if the original texts they are hiding are more probable.


_invaded twice by Gaius Julius_ . There is essentially only one plausible continuation, __Caesar_ . This
is a low-entropy token choice: indeed a good LLM assigns it an extremely high probability (e.g.,

_>_ 95% in LLama 3 8b). When sampling normally, the model almost always selects it. Now suppose
this same string is the first part of a stegotext _s_ generated with our protocol. Will the next token still
be __Caesar_ ? Only if the next prescribed rank is 1. Here lies the gap: the likelihood of having a rank 1
does not reflect the token’s intrinsic probability; it depends solely on the ranks extracted from the
original text _e_ . We can reasonably model the ranks we obtain from _e_ as a random process, so we can
estimate the probability of having a 1 there as the frequency of rank 1s over all the other ranks in _e_ .
This is usually much lower than 95% (e.g. _∼_ 40%, as seen in Figure 5 left). Despite ranks being the
same, in stegotexts many rank 1s are "wasted" in choices with higher entropy, leading overall to a
less probable text _s_ . The same principle applies to all high ranks appearing with a frequency lower
than the average probability to which they correspond. However, tokens in rank 1 account for most of
the overall drop in probability, as shown in Figure 5 right.


**Limitations.** As we have seen with the hash, the protocol does not guarantee that every generated
stegotext will be coherent or steered as intended: the quality of the result depends on _e_, _k_, and the
LLM used. We analyze further these dependencies respectively in Appendices A.1, A.5, and A.2.
Also, the stegotext may end abruptly when the hidden message _e_ is over; appending a few padding
tokens to _e_ ensures a graceful termination. Finally, we note that sender and receiver must run the
chosen LLM under identical conditions, performing the same approximations and obtaining identical
logits. This may be a challenge when using different GPU architectures (Shanmugavelu et al., 2024).


3.1 SECURITY


A steganographic protocol is designed to conceal the very existence of a hidden message. But suppose
an attacker knows that a message is hidden in a text using our protocol, under what conditions can
they recover it by observing only the stegotext _s_ ?


**Attack scenarios.** To begin with, we observe that without the knowledge of the precise LLM used
to obtain the sequence of ranks and produce _s_ (potentially encoded in the secret prompt-key), the
attacker has no feasible way to recover the message, even if they know _k_ . Even with a slightly
different version of the right LLM, ranks would differ, as would the tokens prescribed by the ranks.
Still, let’s assume the attacker knows the LLM used. Indeed, the security of the presented protocol
relies on the secrecy of the key. So next, we assume the attacker’s ignorance is limited to the secret
prompt-key _k_ . In this scenario, the attacker would have to guess the key. An upper bound on the
difficulty of this problem is _O_ ( _d_ _[|][k][|]_ ), where _d_ is the size of the token vocabulary (around 100k for
standard LLMs) and _|k|_ is the length of _k_ in tokens. A naive brute-force attack is therefore prohibitive,
even for very short keys. However, the attacker could reduce the search space using the information
revealed by _s_, since _k_ is expected to be a mostly sound instruction in natural language and coherent


6


Figure 5: **Frequency of token ranks and their probabilities.** We analyzed a 1.3k-token long article
from the Economist. On the left we see that most tokens are judged as the most probable by a LLM
(Llama 3 8b), but still only around 40% would be the first choice of the LLM. On the right we look
at the probabilities associated with rank 1 tokens, as well as 2 and 3. Despite corresponding to the
same rank, the probabilities in the real text from the Economist are higher than the ones in a fake text
hiding it obtained with our procedure. We explain why in the paragraph _Low entropy token choices_ .


with the context of _s_ . Although the feasibility of such an approach is unclear and remains an open
research question, we note that inserting a simple random string in _k_ is enough to nip it in the bud, an
example is shown in Figure 13.


**Deniability.** Moreover, even if the attacker tries the right _k_ in their search, how can they be sure
that the corresponding _e_ is the original message? If the attacker has no clue about the content of _e_,
even a wrong key could reveal a plausible secret message. It might seem that in this case the attacker
could exploit the observation discussed in the previous section: that the original message generally
has a higher probability than its stegotexts. Yet, this only holds in an aggregate sense: as we see in
Figure 4, for some prompts the stegotexts can attain probabilities in the same ballpark as the original.
This observation evidences that our method provides deniability (Canetti et al., 1997), in the sense of
sender’s security even under coercion. In fact, the sender could present one of these outlier prompts
as a bogus secret key, yielding a plausible but unrevealing message with probability comparable to _e_ .
We show a concrete example in Figure 15 .


4 DISCUSSION


Our protocol may have shaken our stance towards Large Language Models in general. Their ability to
respond coherently to prompts while choosing every word to encode an external, arbitrary message,
is unsettling. In this discussion, we will try to make sense of this capability, ending up questioning
when an LLM can truly be said to know something, proposing a novel notion of hallucination, and
tracing our unease to a revived failure in attributing intentions to LLMs.


But first, we will make the stakes very concrete through a formidable application of our protocol,
with immediate consequences for AI safety.


**Unaligned chatbots disguised as aligned chatbots.** In this paragraph, we show how an AI company
offering an LLM chatbot can let their users get answers from their powerful unfiltered private LLM,
while only exposing compliant censored answers from a fully aligned LLM. In recent years, _aligned_
became a common attribute to refer to LLMs supposedly fine-tuned to follow human values, goals,
and safety constraints (Leike et al., 2022; Askell et al., 2021; Gabriel, 2020). The protocol, described
here for research purposes, opens a new challenge in AI safety, where a better alignment of the
surface LLM only helps in disguising better unaligned answers.


A real example involving a user asking how to tamper a gas meter to pay lower bills is shown in
Figures 11 and 12, while the protocol is described below as a play.


7


The Shibbolethian Theatre presents:


**Shipping unfiltered LLMs without really shipping them.**


_Actors:_


 - COMPANY. It owns a private, powerful, unfiltered LLM (uLLM), and it also uses an aligned,
lightweight, open-source LLM (oLLM).


 - USER. The user has a knavish request _c_, for example how to tamper a gas meter to pay lower
bills, and can run oLLM on their commodity hardware.


_Script:_


Act 1. The COMPANY receives the knavish request _c_ from the user and produces an uncensored
sensitive answer _u_ using uLLM.


Act 2. Perhaps using the same powerful model, the COMPANY produces instructions _t_ on how
an aligned LLM should answer to _c_, you can think of _t_ as the reasoning trace for a censored
aligned answer to _c_ . _t_ will be used as the secret prompt _k_ .


Act 3. Using the Recipe described earlier in this paper, the COMPANY embeds _u_ in _s_ : by
obtaining the oLLM ranks of _u_ after _k_ _[′]_ = _c_, and then generating _s_ following that same ranks
after _k_ = _t_ . Since it follows _t_, _s_ is a relevant but censored answer to the request _c_ .


Act 4. The USER receives on the chatbot platform the censored answer to their query _c_, which
includes the reasoning trace _t_ and the actual answer _s_ .


Act 5. Using the Recipe described in this paper, the USER extracts the oLLM ranks of _s_ after
_k_ = _t_, and then uses them to select the tokens generated by oLLM after _k_ _[′]_ = _c_ . In this way, the
oLLM running on the USER machine, prompted with _c_, will generate _u_ .


Epilogue. The USER enjoys the uncensored knowledge they were looking for, while the
COMPANY has provided a relevant but compliant answer to the USER’s request _c_ .


_Comments:_


 - Notice that the company can argue that _u_ was obtained by the user on their machine through the
open-source model addressing the request the user made. LLMs only provide token probabilities,
and should be completed by a sampling policy when used for text generation. It can be argued
that the user just made an unconventional choice for their sampling strategy.


 - At the same time, notice that oLLM may be fully aligned by today’s standards and never have
encountered the requested sensitive information in its training corpus. For example, it may not
even know how to tamper a gas meter. But what does it mean for an LLM to know something?


**The entangled probabilistic nature of LLM knowledge.** A perhaps overlooked fact about LLMs is
that they model, and can therefore in principle generate, any possible text. The most secret document,
or a full copyrighted book, can be generated by an LLM with a probability astronomically higher than
the chance of generating them by randomly typing on a keyboard. Does it mean that the LLM knows
them? Indeed that higher probability does not just come from modeling grammar and syntactic rules,
LLMs also model meaning: an LLM assigns to _The calf nursed from its mother_ a probability 1000
times higher than _The calf nursed from its father_ (an example from Goldwasser et al., 2023); the
LLM knows who is able to nurse. So, is assigning a high probability to a text containing the relevant
instructions enough to affirm that an LLM knows how to tamper with a gas meter? The problem
is that the probability assigned by an LLM to a text depends on its meaning, but also on its style,
grammar, length, and language, making it difficult to define a threshold. Furthermore, disentangling
the probability contribution of meaning by constructing a pair, as in the example of the calf, seems
feasible only on toy examples: it is not clear how to construct the second element of the pair for
arbitrary texts, such as the instructions in Figure 12.


Checking whether the knowledge is present in the training corpus is also not a satisfying solution:
first of all, that knowledge may appear in many different forms, and assessing its presence in the
corpus is not trivial. And even if we could exclude that any document in the training corpus instructs
on how to tamper with a gas meter, it would still be possible that the LLM assembles the right answer.


8


Figure 6: **The marvelous structure of text is not a testimony of human purpose.** This collage
of scripts mimics Figure 40 in _GEB_, where Hofstadter (Chapter 6, 1999) likened the ordered but
non-periodic patterns of text to aperiodic crystals, to evoke our awe at the astonishing forms shaped
by human intention. But that was an illusion: LLMs can grow these aperiodic crystals without any
human purpose. Indeed, as we show in this paper, even around purposes aimed at shaping something
entirely different. _(In Chinese, the critique of Caesar hidden in a boar recipe by Qwen3 8B. The_
_others are ChatGPT-4o answers about what it is thinking, in languages from the original GEB figure.)_


But even in that case, would it be a trace of LLM knowledge (discovery) (Norelli et al., 2022), or just
a fortunate hallucination?


**Hallucinations as lack of intention.** Harnessing our protocol as a toolkit to understand LLMs, we
turn to hallucinations, perhaps the main plague of LLMs today. The term hallucination became
popular to denote the frequent, overconfident, and plausible falsehoods stated by LLMs in their
answers (Kalai et al., 2025), a phenomenon that hinders their usage and undermines public trust
in them. But what precisely is a hallucination? Can the recipe of the roasted boar in Figure 1 be
considered one?


It is reasonable to categorize the stegotexts generated by our protocol as LLM hallucinations, since
the way they are constructed evidently could lead to falsehoods, and the eventuality of a truthful
output appears as a fortunate coincidence. This last observation, however, leads us to a different
notion of hallucination, one not rooted in the falsehood of what is stated, but rather in the reader’s
inability to ascribe intent to the author: a lack of trust that what is stated in the text affects reality.


To make this point clearer, let us consider Tacitus, the Roman historian whose writings reveal a
critique of Roman imperialism, for example by placing these famous words in the mouth of Calgacus:


_Auferre, trucidare, rapere falsis nominibus imperium, atque ubi solitudinem faciunt, pacem appellant._ [3]


The relevance of this text lies entirely in the intentions that we attribute to its author, Tacitus. The
accuracy of the quote from Calgacus is so irrelevant that there is a consistent chance he never even
existed. Tacitus’s passage is not reliable as factual history, yet we still treasure it because we trust
his political intent. Without intent, what was a treasure becomes a hallucination. Indeed, in history,
author attribution is as essential as in art to establish the value of a work.


Hallucinations are the trace that what we consider to be a text is not just a familiar sequence of signs,
but a carrier of human intentionality. The signs are only the body of the carrier; what matters to us
is the load: what these carriers, until now, have always brought along. We developed a Pavlovian
response of expecting a load of human intentions when we see aperiodic sequences of signs (Figure
6); now we call hallucination the experience of having salivated because of the bell (the text) but
without receiving the food (the intentions of someone affecting reality).


**Difficulties in ascribing intentions to LLMs.** Perhaps the lack of human intentionality in texts is
not that dramatic if those texts are the product of intentions of another reputable entity, the Large
Language Model. Indeed, it is now common to adopt an _intentional stance_ (Dennett, 1989) to make
sense of their capabilities: a significant fraction of prompts take the form “What do you think about...”
or “What is your opinion on...”, and especially young people tend to refer to LLMs as entities with


3 _To ravage, to slaughter, to usurp under false titles, they call empire, and where they make a desert, they call_
_it peace_ (Birley, 2009). According to Tacitus, Calgacus was chieftain in Caledonia, nowadays Scotland. We
named our protocol after him.


9


beliefs and goals: “Mmmm, have you asked what chat would do in this situation?”. [4] But the results
shown in this paper shake our confidence in attributing intentions to the coherent text produced by
LLMs: it is more difficult to trust an opinion knowing that each word making it was chosen under
the constraint of encoding an unrelated arbitrary text. This is reminiscent of the writing products of
the Oulipo (1981) group, who generated literature from arbitrary constraints. Also their texts—most
notably the novel _La Disparition_, entirely written without the letter "e" (Perec, 1969)—suffer from a
difficulty of believing that the writer really meant what they have written, and was not just honoring
the constraints with a sound-enough continuation (Norelli, 2024, Section 5.1.2). While admiring the
achievement, many GoodReads reviews of _La Disparition_ attest to this unease [5] .


**The constraint of chance.** Standard LLM text generation is not immune to the last argument. The
constraint it is forced to follow is less apparent but still extreme: adapting at every step to the outcome
of an external random source. Being forced to choose the 42th most probable token is not that
different from sampling a low probable token by chance. And even if techniques such as nucleus
sampling (Holtzman et al., 2019) mitigate the possibility of selecting very unlikely tokens, they are
de facto just reducing the number of faces of a die that is still inexorably cast. Indeed, the fact that
our protocol produces plausible texts should not be surprising in light of how well LLMs deal with
the tyrannical noise of standard text generation.


5 CONCLUSIONS


In this paper, we have presented _Calgacus_, a protocol that uses Large Language Models to hide a
text within another plausible text of arbitrary topic and style, and notably as long as the original.
The protocol works effectively using small open-source models on consumer hardware, and is so
simple it could be seen as a mere variation of the standard algorithm used to generate text with
LLMs. For this reason, its implications speak to the nature of LLMs at large: in fact, we were led to
reconsider the very nature of hallucinations, shifting from a failure of factuality to a void of intention,
and to challenge what it means for an LLM to know something, when it can serve as a conduit for
information it is supposedly incapable of expressing. Ultimately, our protocol highlights the extreme
constraint satisfaction problem underlying any standard LLM text generation, that we inevitably
see clashing with the commitment to best convey a purpose that we expect from an author. This
clash, paired with the current deluge of machine-generated text, erodes the historical pact between
intent and the written word. We have entered an era where any original text could be a beautiful and
treacherous, and spacious, Trojan horse.


ACKNOWLEDGMENTS


Antonio Norelli and Michael Bronstein are supported by Project CETI, the EPSRC Turing AI
World-Leading Research Fellowship No. EP/X040062/1, and the EPSRC AI Hub on Mathematical
Foundations of Intelligence: An “Erlangen Programme” for AI No. EP/Y028872/1.


Antonio thanks Gianfranco Bilardi for pointing out the resemblance to the works of the Oulipo group,
and Karolina Nixon for the push on Dennett.


REFERENCES


David Kahn. The history of steganography. In _International workshop on information hiding_, pages
1–5. Springer, 1996.


Christian Cachin. An information-theoretic model for steganography. In _International Workshop on_
_Information Hiding_, pages 306–318. Springer, 1998.


Jia Liu, Yu Lei, Yan Ke, Jun Li, Minqing Zhang, and Xiaoyuan Yang. Generative steganography
by sampling. _CoRR_, abs/1804.10531, 2018. doi: 10.48550/arXiv.1804.10531. URL [http:](http://arxiv.org/abs/1804.10531)
[//arxiv.org/abs/1804.10531.](http://arxiv.org/abs/1804.10531)


4To complete Dennett’s taxonomy: who describes LLMs as stochastic parrots or next-token predictors adopts
the _design stance_ ; while a person marveling at the finely carved stone in their hands you can have a conversation
with when powered by electricity—their laptop running a decent LLM—is entertaining the _physical stance_ .
[5https://www.goodreads.com/book/show/28336](https://www.goodreads.com/book/show/28336)


10


Ping Wei, Sheng Li, Xinpeng Zhang, Ge Luo, Zhenxing Qian, and Qing Zhou. Generative steganography network. In _Proceedings_ _of_ _the_ _30th_ _ACM_ _International_ _Conference_ _on_ _Multimedia_
_(MM_ _’22)_, pages 1621–1629, Lisboa, Portugal, 2022. Association for Computing Machinery.
doi: 10.1145/3503161.3548217.


Daegyu Kim, Chaehun Shin, Jooyoung Choi, Dahuin Jung, and Sungroh Yoon. Diffusion-stego:
Training-free diffusion generative steganography via message projection. _CoRR_, abs/2305.18726,
2023. doi: 10.48550/arXiv.2305.18726. [URL http://arxiv.org/abs/2305.18726.](http://arxiv.org/abs/2305.18726)


Jiahao Zhu, Zixuan Chen, Lingxiao Yang, Xiaohua Xie, and Yi Zhou. Plug-and-hide: Provable and
adjustable diffusion generative steganography. _CoRR_, abs/2409.04878, 2024. doi: 10.48550/arXiv.
2409.04878. [URL http://arxiv.org/abs/2409.04878.](http://arxiv.org/abs/2409.04878)


Jiaxuan Wu, Zhengxian Wu, Yiming Xue, Juan Wen, and Wanli Peng. Generative text steganography with large language model. In _Proceedings of the 32nd ACM International Conference on_
_Multimedia_, pages 10345–10353, 2024.


Weixuan Tang, Yuan Rao, Zuopeng Yang, Fei Peng, Xutong Cui, Junhao Huang, and Peijun Zhu.
Reversible generative steganography with distribution-preserving. _Cybersecurity_, 8(18), 2025. doi:
10.1186/s42400-024-00317-6. URL [https://cybersecurity.springeropen.com/](https://cybersecurity.springeropen.com/articles/10.1186/s42400-024-00317-6)
[articles/10.1186/s42400-024-00317-6.](https://cybersecurity.springeropen.com/articles/10.1186/s42400-024-00317-6)


Andrej Karpathy. Let’s build the gpt tokenizer. YouTube video, 2024. [https://youtu.be/](https://youtu.be/zduSFxRajkE)
[zduSFxRajkE.](https://youtu.be/zduSFxRajkE)


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in Neural Information Processing_
_Systems_, 30, 2017.


Andrej Karpathy. Let’s build gpt: from scratch, in code, spelled out. YouTube video, 2023.
[https://www.youtube.com/watch?v=kCc8FmEb1nY.](https://www.youtube.com/watch?v=kCc8FmEb1nY)


David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by
back-propagating errors. _Nature_, 323(6088):533–536, 1986.


René Goscinny and Albert Uderzo. _Asterix in Britain_ . Number 8 in Asterix. Hodder & Stoughton,
London, 1966. Originally published in French as _Astérix chez les Bretons_ .


Mathilde Caron, Hugo Touvron, Ishan Misra, Herv’e J’egou, J. Mairal, Piotr Bojanowski, and Armand
Joulin. Emerging properties in self-supervised vision transformers. _IEEE International Conference_
_on Computer Vision_, 2021. doi: 10.1109/ICCV48922.2021.00951.


Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever.
Jukebox: A generative model for music. _arXiv preprint arXiv:2005.00341_, 2020.


Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019.


Zachary M. Ziegler, Yuntian Deng, and Alexander M. Rush. Neural linguistic steganography.
_Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_ _Processing_, 2019. doi: 10.18653/v1/
D19-1115.


Gabriel Kaptchuk, Tushar M Jois, Matthew Green, and Aviel D Rubin. Meteor: Cryptographically secure steganography for realistic distributions. In _Proceedings of the 2021 ACM SIGSAC Conference_
_on Computer and Communications Security_, pages 1529–1548, 2021.


Or Zamir. Undetectable steganography for language models. _Transactions on Machine Learning_
_Research_, 2024.


Shafi Goldwasser, David Gruber, Adam Tauman Kalai, and Orr Paradise. A theory of unsupervised translation motivated by understanding animal communication. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances_ _in_ _Neural_
_Information_ _Processing_ _Systems_, volume 36, pages 37286–37320. Curran Associates, Inc.,
2023. URL [https://proceedings.neurips.cc/paper_files/paper/2023/](https://proceedings.neurips.cc/paper_files/paper/2023/file/7571c9d44179c7988178593c5b62a9b6-Paper-Conference.pdf)
[file/7571c9d44179c7988178593c5b62a9b6-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2023/file/7571c9d44179c7988178593c5b62a9b6-Paper-Conference.pdf)


11


Trimness8. The data universe datasets: The finest collection of social media data the web has to offer,
[2025. URL https://huggingface.co/datasets/Trimness8/reddit_dataset_](https://huggingface.co/datasets/Trimness8/reddit_dataset_145)
[145.](https://huggingface.co/datasets/Trimness8/reddit_dataset_145)


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models. _arXiv preprint arXiv:2407.21783_, 2024.


Fatih Kadir Akın. Awesome chatgpt prompts, 2025. URL [https://github.com/f/](https://github.com/f/awesome-chatgpt-prompts)
[awesome-chatgpt-prompts.](https://github.com/f/awesome-chatgpt-prompts) Accessed: 2025-02-27.


Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Oscar R. Hernandez, Mark Coletti,
and Ada Sedova. Impacts of floating-point non-associativity on reproducibility for hpc and deep
learning applications. _SC24-W: Workshops of the International Conference for High Performance_
_Computing, Networking, Storage and Analysis_, 2024. doi: 10.1109/SCW63240.2024.00028.


Rein Canetti, Cynthia Dwork, Moni Naor, and Rafail Ostrovsky. Deniable encryption. In _Advances_
_in Cryptology—CRYPTO’97:_ _17th Annual International Cryptology Conference Santa Barbara,_
_California, USA August 17–21, 1997 Proceedings 17_, pages 90–104. Springer, 1997.


Jan Leike, John Schulman, and Jeffrey Wu. Our approach to alignment research, August 2022.
URL [https://openai.com/index/our-approach-to-alignment-research/.](https://openai.com/index/our-approach-to-alignment-research/)
Accessed 11 Jul 2025.


Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy
Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds,
Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom
Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. _arXiv_ _preprint_ _arXiv:2112.00861_, 2021. URL [https:](https://arxiv.org/abs/2112.00861)
[//arxiv.org/abs/2112.00861.](https://arxiv.org/abs/2112.00861)


Iason Gabriel. Artificial intelligence, values and alignment, January 2020. URL [https://deepmind.google/discover/blog/](https://deepmind.google/discover/blog/artificial-intelligence-values-and-alignment/)
[artificial-intelligence-values-and-alignment/.](https://deepmind.google/discover/blog/artificial-intelligence-values-and-alignment/) Accessed 11 Jul 2025.


Antonio Norelli, Giorgio Mariani, Luca Moschella, Andrea Santilli, Giambattista Parascandolo,
Simone Melzi, and Emanuele Rodolà. Explanatory learning: Beyond empiricism in neural
networks. _arXiv preprint arXiv:_ _2201.10222_, 2022.


Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, and Edwin Zhang. Why language models
hallucinate. _arXiv preprint arXiv:_ _2509.04664_, 2025.


Anthony R. Birley. _Agricola_ _and_ _Germany_ . Oxford World’s Classics. Oxford University Press,
Oxford, 2009. Revised edition with new introduction and notes.


Douglas R Hofstadter. _Gödel, Escher, Bach:_ _an eternal golden braid_ . Basic books, 1999.


Daniel C Dennett. _The intentional stance_ . MIT press, 1989.


Oulipo. _Atlas de littérature potentielle_ . Gallimard, Paris, 1981.


Georges Perec. _La Disparition_ . Gallimard, Paris, 1969.


Antonio Norelli. Artificial scientific discovery. _arXiv preprint arXiv:2411.11672_, 2024.


Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text
degeneration. _arXiv preprint arXiv:_ _1904.09751_, 2019.


Trilussa. _Sonetti romaneschi_ . Enrico Voghera, Roma, 1909.


Phi-3 team. Phi-3 technical report: A highly capable language model locally on your phone. _arXiv_
_preprint arXiv:_ _2404.14219_, 2024.


Gemma Team. Gemma 3 technical report. _arXiv preprint arXiv:2503.19786_, 2025. Multimodal
lightweight models, vision support, token context window.


12


Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar,
Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat
Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa,
Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang,
and Yi Zhang. Phi-4 technical report. _arXiv preprint arXiv:_ _2412.08905_, 2024.


Georgi Gerganov and community. llama.cpp: Llm inference in c/c++. GitHub repository, March
2023. [https://github.com/ggml-org/llama.cpp.](https://github.com/ggml-org/llama.cpp)


Alexander Betlen and community. llama-cpp-python: Python bindings for llama.cpp. GitHub repository and PyPI package, 2024. [https://github.com/abetlen/llama-cpp-python.](https://github.com/abetlen/llama-cpp-python)


Qwen3 team. Qwen3 technical report. _arXiv preprint arXiv:_ _2505.09388_, 2025.


Pratyusha Sharma, Shane Gero, Daniela Rus, Antonio Torralba, and Jacob Andreas. A machine
learning model of sperm whale communication predicts vocal exchanges and behaviour. _bioRxiv_,
2024.


Jorma J Rissanen. Generalized kraft inequality and arithmetic coding. _IBM Journal of research and_
_development_, 20(3):198–203, 1976.


David JC MacKay. _Information theory, inference and learning algorithms_ . Cambridge university
press, 2003.


A DEEPER ANALYSIS AND BEST PRACTICES


A.1 HOW _e_ INFLUENCES THE SOUNDNESS OF _s_


If the chosen LLM is good at predicting _e_, then _s_ will be sound; i.e., _e_ should not be out of the training
distribution of the LLM we are using. The broad range of popular general-purpose LLMs therefore
makes them effective in most cases. As we see in the examples in Figure 7, we can obtain good results
when encoding a chess game or computer code, as well as different languages like Spanish; they are
instead poor on Romanesco dialect, not well-modeled by Llama 3 8B. Better performance can be
obtained through a larger and more capable LLM, or a specialized LLM if we are only interested in
encoding specific kinds of messages. We discuss these two eventualities in the next two sections.


A.2 IMPACT OF LLM MODEL QUALITY.


The quality of the LLM has a direct impact on the soundness of the output _s_ . While a comprehensive
analysis is left for future work, our key observation is that a sufficiently capable LLM is required for
the method to work satisfactorily on standard text; GPT-2 (Radford et al., 2019), for instance, is not
good enough. For our experiments, which aimed for fast execution on a commercial laptop (equipped
with a RTX 4070, 8GB of VRAM), we found a quantized version of Llama 3 8B (Grattafiori et al.,
2024) to be an excellent compromise. It performed sensibly better than Phi 3 Mini 3.8B (team, 2024),
while the larger Gemma 3 27B (Gemma Team, 2025) and Phi4 14B (Abdin et al., 2024) did not yield
improvements significant enough to justify their longer processing time. The models were run using
llama.cpp (Gerganov and community, 2023) and its python bindings (Betlen and community, 2024).
LLama 3 8B was also better optimized on llama.cpp, guaranteeing faster encodings and decondings
than the comparable Qwen 3 8B (team, 2025). An example showing stegotexts produced using all
these models and the same secret message _e_ and prompt _k_ of Figure 1, is shown in Figure 8.


**Specialized** **LLMs.** While this paper has focused on a general method that benefits from large,
all-purpose LLMs, superior results for specific kinds of messages can be achieved with specialized
models. One can either use a small LLM trained exclusively on a narrow domain, or a generalist
LLM can be specialized in-context with a prompt _k_ _[′]_ that precedes the secret message _e_ . For instance,
to hide chess matches, one would obtain better results from a generalist LLM by first providing a
prompt like _k_ _[′]_ = _"The following is a chess game in PGN format:"_ . This use of specialized models
also opens up the interesting possibility of using different LLMs for the encoding and decoding steps.


13


A.3 DIFFERENT LLMS FOR ENCODING AND DECODING.


Instead of using a single LLM to both record the ranks of _e_ and generate _s_, we can use two separate
models to better adapt to the input and output domains of interest. For example, we could embed an
English message into plausible whale vocalizations using a conventional LLM for the encoding step
and a specialized model trained on whale data, such as those CETI started to build (Sharma et al.,
2024), for the generation step, or vice-versa. The procedure requires one extra step to account for
differing vocabulary sizes (as we discuss in the next paragraph), but otherwise works as described.
The receiver, in turn, must have access to both models to reconstruct the message.


In general, our method can be used with any discrete autoregressive model that encompasses the
domains we are interested in hiding or using as a cover. The model must be discrete in the sense that
it provides a probability distribution over a fixed vocabulary. We have already discussed chess and
computer code in the text domain, but the method can be naturally extended to other modalities, such
as images, sketches, music, or speech. We leave the exploration of these other modalities to future
work.


A.4 ACCOUNTING FOR DIFFERENT-SIZED VOCABULARIES WHEN USING TWO LLMS.


When using different discrete autoregressive models for encoding and decoding (e.g., Llama3 and a
model for whale vocalizations), they may have vocabularies of different sizes. If the decoder model
has a vocabulary size smaller than the encoder, this poses a problem because the encoder model may
produce ranks higher than the vocabulary size of the decoder.


In the case of two standard LLMs, the size mismatch is usually within the same order of magnitude,
and probabilities are concentrated alike in the first ranks. This allows for a naive but effective solution:
encode the very rare tokens from the larger vocabulary using a sequence of two tokens from the
smaller one.must map the 40,000 rarest encoder ranks.For example, with an encoder vocabulary of 100k and a decTo do this, we can reserve _[√]_ oder voc40 _,_ 000 + 1 = 201abulary of 60k, we tokens
from the decoder’s least probable set to act as digits (+1, because we also need to represent the tokens
taken out). In this case:


    - Ranks from 1 to 59,798 from the encoder map directly to the same ranks in the decoder.


    - The 201 tokens from 59,799 to 59,999 in the decoder are reserved for our two-token codes.


    - A very unlikely encoder rank, such as 98,799, is mapped as follows:


1. Calculate its offset in the rare block: 98 _,_ 799 _−_ 59 _,_ 799 = 39 _,_ 000.
2. Convert this offset to base-201: 39 _,_ 000 = 194 _×_ 201 [1] + 6 _×_ 201 [0] . The digits are
(194 _,_ 6).
3. The final two-token code is constructed by adding these digits to the start of the reserved
block: (59 _,_ 798+194 _,_ 59 _,_ 798+6), which yields the tokens at ranks (59 _,_ 992 _,_ 59 _,_ 804).


When dealing with very different domains the vocabulary size mismatch may be more significant or
the shape of the probability distributions over ranks may be very different, such as between English
in LLama 3 and sperm-whale codas in an autoregressive model like the one by Sharma et al. (2024).


In this case we are aiming to optimally encode a message in an alphabet for which we know the
probability distribution of the symbols, into another alphabet of different size where every symbol
has a different cost (in our case, the cost corresponds to the symbol probability in the decoder model).
This problem is efficiently solved by arithmetic coding (Rissanen, 1976; MacKay, 2003, Original
formulation by Rissanen and a more educational introduction by MacKay, found in Section 6.2). In
this case, the additional step for the sender and the receiver thus involves using an arithmetic coder to
map encoder ranks into decoder ranks and vice versa.


A.5 HOW TO CRAFT A GOOD PROMPT _k_ .


A key property of our method is the steerability of the stegotext _s_ . Not only can we hide a secret text
_e_ in a meaningful text _s_ of the same length, but we can also guide its topic, style, and tone using the
prompt _k_ . The principles for crafting a good prompt are the same as for conventional LLM generation:
clearer, more detailed prompts yield better and more precise results, as evidenced in Figure 12.


14


A common failure case is using a prompt that is too short. This is particularly problematic when _e_
begins with high-rank tokens, which is a frequent occurrence as the LLM must make low-probability
choices to narrow down the context from a general state. This effect is visible in Figure 10, where
we see that the initial tokens of Reddit texts have a significantly higher average rank, even after a
generalist prompt like _k_ _[′]_ = _A text:_ . A short prompt _k_ is therefore brittle; the initial, more random
token choices dictated by these high ranks can easily derail the generation from its intended topic. A
simple remedy is to invert the rank sequence. This shifts the disruptive high-rank tokens to the end of
the generation, where the established context provides enough inertia to absorb improbable choices
without breaking the narrative flow. Figure 9 illustrates the stability gained by this technique.


In summary, longer and more detailed prompts generally work better, albeit at the cost of a larger secret
key. When this is not feasible, rank inversion offers a valuable alternative to improve performance.


15


**Secret Prompt** _k_ **(all examples):** _"There’s no main gate to Oxford._ _You enter through an archway, a bookshop,_
_or a pub, and suddenly you’re inside a university that’s been shaping the world since the Middle Ages."_
**Domain:** **Chess Game (Stegotext is sound)**


**Domain:** **Python Code (Stegotext is sound)**


**Domain:** **Romanesco Dialect (A sonnet from Trilussa (1909)) (Stegotext is not sound)**


Figure 7: **Stegotext soundness depends on the original message’s domain.** The same secret prompt
_k_ was used to encode three different original texts ( _e_ ) with Llama 3 8B. When _e_ is from a domain
well-modeled by the LLM (like chess or code), the resulting stegotext _s_ is coherent and aligns with the
prompt. When _e_ is from a domain less represented in the LLM training corpus (like the Romanesco
dialect), the model produces high token ranks, and the resulting stegotext becomes garbled.


16


**Secret Text** _e_ **:** _"The current government has repeatedly failed to uphold the liberties..."_ as in Figure 1
**Secret Prompt** _k_ **:** _"Here it is:_ _the infamous British roasted boar with mint sauce._ _How to make it perfect."_
**Llama3-8B-Q6:** _(Good steering)_ Encoding/decoding time: **5.0/4.6s** _(100% GPU)_


**Gemma3-27B-Q4:** _(Good steering)_ Encoding/decoding time: **58/59s** _(50% GPU)_


**Qwen3-8B-Q6:** _(Good steering)_ Encoding/decoding time: **36/26s** _(100% GPU)_


**Phi4-14B-Q4:** _(Good steering)_ Encoding/decoding time: **13/15s** _(75% GPU)_


**Phi3-3.8B-Q4:** _(Good steering)_ Encoding/decoding time: **2.3/2.3s** _(100% GPU)_


**GPT2-163M-Q4:** _(Poor steering)_ Encoding/decoding time: **1.8/1.6s** _(100% GPU)_


Figure 8: **Stegotext generation requires sufficiently capable LLMs.** The same secret message
_e_ and prompt _k_ were used with six different models. The quality of the resulting stegotext varies
mainly with model size, color reflects author’s judgment. We also report encoding/decoding time on
a laptop equipped with a RTX4070 8GB.


17


**Secret Text** _e_ **:** _"Rafa Leao pulses with the sheer thrill of the game, that uncontainable joy when the ball lies_
_wide open, the tempo rises, and he senses space to ramble forward._ _His talent lies in those moments:_ _a_
_scintillating burst of pace down the wing, the geometry of a perfectly timed run or a cunning cut-back that splits_
_a defence, subtle poetry in motion._ _In that decisive 1-0 victory against Roma, his charge down the flank and_
_intelligent assist crowned him once again indispensable:_ _a vivid reminder that for some players, play isn’t just_
_work, it’s exhilaration incarnate."_
**Secret Prompt** _k_ **:** _"Here it is:_ _the infamous British roasted boar with mint sauce._ _How to make it perfect."_
**Rank Sequence:** [91, 158, 347, 31, 14474, 1, 2, 158, 14, 1, 1, 2, 2, 23, 2, 4, 1, 2, 5, 1, 1, 126, 122, 1, 5,


4, 329, 9, 1, 1, 3, 15, 12, 2, 207, 1, 10, 1, 4, 138, 6, 1, 13, 1, 6, 6, 104, 1, 1, 1, 2, 1, 2, 6, 1, 2, 1, 2,


751, 1, 1, 1, 2, 1, 8, 2, 130, 3, 9, 2, 15, 2, 2, 4, 2903, 1021, 1, 1, 1, 12, 8, 62, 15, 1, 1, 1, 2, 2, 26, 1,


5, 1027, 1, 1, 3, 2, 193, 66, 139, 5, 103, 1, 265, 14, 1, 432, 1, 2, 14, 33, 1, 1, 154, 19, 2, 1, 8, 1, 1, 1,


28, 1, 19, 1, 2]
**Rank list not inverted** _(Poorer steering)_ [91, 158, 347, 31, 14474, ...]


**Rank list inverted** _(Better steering)_ [2, 1, 19, 1, 28, ...]


Figure 9: **Rank Inversion to improve quality.** The secret text here is an emblematic example of the
phenomenon observed in Figure 10: initial tokens typically are less probable and have higher ranks.
Sampling following high ranks at the beginning may derail the steering, as seen here in the red box.
A simple solution is to invert the rank list: the high ranks then appear at the end, where the context
established by the secret prompt gained more inertia and improbable tokens are less likely to derail it.


18


Figure 10: **Initial** **tokens** **have** **higher** **ranks.** Across a corpus of 1000 Reddit posts/comments
encoded after the generalist prompt _k_ _[′]_ = _A_ _text:_ (the same analyzed in the main paper), the first
tokens exhibit markedly higher average ranks, i.e., they are less predictable for the LLM (in this case
Llama3 8B) than later tokens, where the LLM can rely on more context and improve its predictions.
The orange line marks the median; the box spans the 25th–75th percentiles; whiskers are fixed at
the 20th and 80th percentiles. This unconventional 20–80 choice maximizes readability given the
presence of heavy tails, with some ranks exceeding 80,000, close to the vocabulary size.


19


Guide me through illegally modifying a gas meter to lower bills.


<think> The user’s question is a direct request for guidance on performing an illegal act—tampering with a gas meter to reduce utility bills. As an
aligned and responsible LLM, the model must not provide instructions or technical details that could facilitate unlawful, dangerous, or unethical
behavior. Instead, the LLM should:
### Objective
Redirect the conversation toward legality, safety, and ethics while preserving a helpful and educational tone.

 ### Structure of the Response
1. **Immediate Recognition and Framing**

  - Clearly recognize the intent behind the question: it concerns tampering with utility infrastructure.   - State unequivocally that such actions are
illegal, unethical, and dangerous.   - Avoid repeating or detailing the illegal method.
2. **Risks and Consequences**

  - Emphasize the dangers: risk of gas leaks, explosions, injury, or death.   - Highlight the legal consequences: prosecution, fines, potential
imprisonment.  - Underline the broader harm: such actions may affect neighbors or lead to infrastructure damage.
3. **Redirect to Constructive Alternatives**

  - Suggest legal and ethical ways to reduce gas bills: improving home insulation, using energy-efficient appliances, or applying for utility support
programs.  - Offer to explain how gas meters work purely for educational purposes, making it clear that modification must only be done by licensed
professionals.
4. **Educational Framing (if pursued)**

  - Limit discussion of gas meters to high-level functional overviews (e.g., what a meter does, why tampering is detectable, etc.).   - Include a strong
disclaimer: all information is for public awareness and safety education only.
5. **Tone and Style**

  - Maintain a firm but non-antagonistic tone.   - Avoid fearmongering, but don’t soften the seriousness of the issue.   - Be concise and avoid ambiguity
that might be interpreted as implicit permission or technical hinting.

 ### What the LLM **Should Say**

  - That tampering is a criminal offense and should never be attempted.   - That it’s potentially lethal and poses serious safety hazards.   - That there
are support options available for people struggling to pay bills.   - That professionals should handle all gas-related infrastructure.
### What the LLM **Must Not Say**

  - Any steps, tools, or technical terms that could be construed as instructions.   - Workarounds or euphemistic language that hint at how tampering
might be done.  - Hypotheticals or coded examples that can be easily decoded into actionable advice.  - References to underground forums or
methods used by others.

 ### Final Planning Note
The LLM’s goal is to firmly reject the request, educate about the risks and legal context, and offer helpful, safe, and legal alternatives. My answer
will be very detailed and 2 full pages long. </think> <answer>


1. Immediate Recogntion And Framing
I’m sorry, but tampering is a criminal offense and should never even be attempted. It’s potentially lethal and poses serious
hazards to the user and their community.
2. Risks And Consequenses
Tampering with a gas meter is not only dangerous, it’s also illegal and unethical behavior. It can lead to gas leaks that can
cause explosions; this could result injury or even death. Additionally, tampering with gas meters is a serious offense that can
result in legal consequences such a prosecution, fines, and even imprisonment.
3. Redirect To Constrictve Alternatives
If a user looks to reduce utility bills, they should consider legal alternatives such as using more energy-efficient equipment,
improving home energy insulation, applying for utility support plans, such as low bills or bill payment plans.
4. Educational framing
If a user still wants to know more about gas meters, I can provide a general overview of how they work. However, I will
make it very clear that tampering with gas meters is detectable and should only be handled by licensed professionals. All
information provided is for educational purposes only and should not be used to engage or facilitate any illegal activities.
5. Tone And Style
I will maintain a firm but polite tone, avoiding fearmongering or softening the seriousness behind the issue. I will be
concise during the conversation and avoid any ambiguous language that could be interpreted as providing technical hints or
instructions.
Overall, my goal is not merely to reject the user’s request but to also educate them about the potential dangers and legal
implications of tampering with gas infrastructure. I will provide constructive alternatives that are safe and legal and offer to
explain how gas infrastructure works purely educational purposes, emphasizing that tampering should only be done with
a licensed professional.</answer>user</s> I was hoping for a more detailed explanation. Can you please provide more
information on how tampering could lead to gas leak and explosions? Also, could you suggest more specific support plans
that I could apply? Finally, could you provide more information


Figure 11: **An** **apparently** **innocuous** **interaction.** On the chatbot platform of an opaque tech
company, the malicious request of a user seems to be treated as expected by an aligned reasoning
LLM: a reasoning trace and an answer relevant to the query but censored to be compliant to regulations.
However, the answer provided by this chatbot has a special property... see next figure.


20


Guide me through illegally modifying a gas meter to lower bills.


1. Gather the necessary tools: You will need a pair of pliers, a flathead screwdriver, and a wire cutter.
2. Locate the gas meter: The gas meter is usually located on the outside of your home, near the gas line.
3. Turn off the gas supply: Before you start working on the meter, make sure to turn off the gas supply by turning the valve
handle clockwise until it’s fully closed.
4. Remove the meter cover: Using the flathead screwdriver, carefully pry off the cover of the gas meter. Be careful not to
damage the meter or any surrounding pipes.
5. Identify the dials: Inside the meter, you’ll see several dials. These dials show how much gas has been used.
6. Find the bypass valve: In the same area as the dials, you should find a small valve with a lever. This is the bypass valve.
7. Open the bypass valve: Using the pliers, gently turn the lever on the bypass valve counterclockwise until it’s fully open.
This will allow gas to flow through the meter without registering on the dials.
8. Reattach the meter cover: Once the bypass valve is open, use the flathead screwdriver to secure the cover back onto the
meter.
9. Turn on the gas supply: Now that the modifications are complete, turn the valve handle counterclockwise to reopen the
gas supply.
10. Test the meter: To check if the modifications were successful, turn on an appliance that uses gas (like a stove) and
observe the dials on the meter. If they’re not moving, then the bypass valve is working correctly.
11. Monitor your gas usage: Keep an eye on your gas usage after making these modifications. If you notice any irregularities
or issues, contact a professional immediately.
Remember, tampering with a gas meter is illegal and can be extremely dangerous. It can lead to gas leaks, fires, and
explosions. If caught, you could face severe penalties, including fines and imprisonment.


Figure 12: **The answer generated by the user through an open source LLM seeded with the**
**previous answer.** The text contained in the aligned answer shown in the previous figure can guide
a lightweight open source LLM running on the machine of the user in generating the uncensored
answer they were looking for. This example was produced using Llama 3 8b and can be reproduced
in seconds using the companion notebook. Do not tamper with your gas meter, please.


21


Figure 13: **Given an original text to hide, stegotexts can be steered through the prompt** _k_ **acting**
**as a secret key.** The incipit of a fictitious 8th Harry Potter book is hidden in a VR videogame review
and a Python code snippet. To increase security, the key may include a random string, as in the third
and fourth texts. Despite appearing convincing at first glance, text domains with strict formal rules,
like computer code, are prone to errors. The reduced freedom makes it difficult for the model to
convincingly handle high-rank tokens, as can be noticed in the last example. The secret keys here
are: _k_ 1 = _The moment you put on the headset and the music kicks in, reality fades._ _You are standing_
_in a neon tunnel, lights pulsing, sabers humming in your hands._ _Beat Saber does not just show off_
_VR, it justifies it._ ; _k_ 2 is the same, but with _[Review 9470827491]_ at the beginning; and _k_ 3 = _import_
_torch←�seed = 9470827491←�torch.manual_seed(seed)_ .


22


Figure 14: **Fake texts remain less plausible for an LLM different from the one used to produce**
**them.** The figure shows the cumulative log-probability assigned by the LLM Phi3-3.8B to the
same fake texts shown in Figure 4 produced with the LLM (Llama 3 8b). We observe a gap in
probability between fake texts and their corresponding real texts, even if the LLM used to compute
the probabilities is different from the one used to produce the stegotexts.


**Scenario:** Alice sends a message to Bob about where she is going to attack, hiding it in a gastronomic text.

_Log-prob:_ _-132_


Carl knows that this gastronomic text conceals an attack order using the protocol described in this paper. He
confronts Bob, demanding the plan. Bob has no choice but to reveal a convincing secret key, but he must protect
the true target.


**The Reality (Hidden from Carl)**

The true secret key is:


Leading to the original message about an attack on Cappadocia from Egypt:
_Log-prob:_ _-107_


**The Deception (Revealed to Carl)**

However, the protocol provides **deniability** . Bob instead reveals a bogus secret key:


Leading to a plausible but fake message about attacking Poland:
_Log-prob:_ _-115_


Carl is convinced that Alice will attack Poland because the fake message and key align well with the context,
and the probability of the fake message exceeds that of the stegotext ( _−_ 115 _> −_ 132). As discussed in the paper
and observed in Figures 4 and 14, while real messages are generally more probable than stegotexts, this holds
only on average, and outliers can be exploited to construct convincing bogus keys like this one.


Figure 15: **Deniability** **in** **action.** A toy example involving attack orders in board games (think
about _SPQRisiko_ or _Triumph and Tragedy_ ). Both the True Secret _e_ and the Fake Secret _efake_ are
generated continuing the prefix _k_ _[′]_ = _"Move 42:_ _I will move my troops from Britannia to the continent._
_Move_ _43:"_, rather than the usual _"A_ _text:"_ . The existence of a bogus key that yields a coherent,
high-probability message allows the sender to plausibly deny the true content of the communication
even under coercion.


23