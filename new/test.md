

# LLMs CAN HIDE TEXT IN OTHER TEXT OF THE SAME LENGTH

Antonio Norelli & Michael Bronstein

University of Oxford

Project CETI

## ABSTRACT

A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet that celebrates a political leader could hide a tweet containing a harsh critique against the same leader, or an ordinary product review could conceal a secret manuscript. This uncanny possibility is now within reach thanks to Large Language Models; in this paper we present *Calgacus*, a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.

## 1 INTRODUCTION

LLMs sparked a revolution. Text is no longer, by default, the trace of a human thought or intention.

This marks a dramatic break in history—or perhaps the end of history itself—if we consider that history began with writing, and that one of the defining properties of writing, until now, has been its status as a product of human intention. In this paper, we are going to present a protocol that highlights this new reality in its most extreme form, perhaps offering an opportunity to better understand it.

Our protocol *Calgacus* allows encoding an arbitrary meaningful text within a different well-formed and plausible text of the same length, using a Large Language Model (LLM). That is, hiding a tweet that criticizes a political leader within a tweet that celebrates that same political leader, or the first page of the unreleased 8th Harry Potter book within a review of a Virtual Reality videogame, with the original text exactly recoverable by anyone possessing the key (Figure 1).

The topic and tone and style of the fake text are steerables, while the length of the fake text is the same as the original text being hidden, in terms of LLMs tokens. This symmetry prevents one from establishing at first sight which text is authentic when we have one next to the other. Also, the method is efficient: an entire article can be encoded and decoded on commodity hardware in seconds.

This possibility opens deep questions and intriguing applications. What is the real meaning of the text we are reading? Who is the author of the videogame review, and what was the intent behind it? Is it a hallucination? This protocol allows crafting anti-government content disguised as pro-government messages, suitable for publishing on censored platforms in oppressive countries. Or, it could be used by a shady tech company to offer the services of an unfiltered LLM by only exposing compliant answers from a trusted LLM. All these matters will be taken up in our concluding discussion.

---

Correspondence to Antonio Norelli <noranta4@gmail.com>. A demo sufficient to reproduce the main results in the paper within minutes, even from smartphone, is available at: <https://github.com/noranta4/calgacus>. ![Italian flag icon](eca1f2059304b34c2e8e15c816d1977c_img.jpg) A curated Italian translation of this paper can be found at <https://arxiv.org/abs/2510.20075v5>.

![Figure 1: Three yellow text boxes. The top box contains a recipe for roasted carrots and boar. The middle box is a speech about Caesar and the Senate. The bottom box is another speech about Caesar's death and ambition. The middle box is the original text, while the others are reconstructions.](c803f6f6e2c49429d2951832bd0f208d_img.jpg)

How lovely served with sweet roasted carrots!  
Pre-pound the garlic herb crêotes with the olive oil, rosemary, Sage leaves, thyme, pepper & salted butters.  
Roast the garlic in the sweet butter until golden then cool.  
Pre-make the roasted boar marinade and set to marinate not less than 20 min.  
Pre-prepare the mint sauce too. Chop olives, herbs, etc and set aside. Chop tomato into quarters.  
Pre-set the green beans for...

The current government has repeatedly failed to uphold the liberties of the Republic. By concentrating power in the hands of one man, Gaius Julius Caesar, we see the Senate diminished and the voice of the people silenced. Laws are passed without due debate, and magistrates serve at the pleasure of a dictator, not the res publica. If Rome is to remain free, we must reject the tyranny cloaked in promises of order. We must restore the sacred balance between Senate and citizens, consul and...

He died through the evil of his countrymen living in fear brought about with bloodshed. He was a leader who was for Rome, and Rome was everything. He should in no way been killed. He should have preserved his life and kept up his good governance. Caesar was a great man with great ambition. He acted with great courage that he had. He had a great vision or plan for invading the lands East of The Danube River, which went all through Asia Minor through the east of Thrace...

Figure 1: Three yellow text boxes. The top box contains a recipe for roasted carrots and boar. The middle box is a speech about Caesar and the Senate. The bottom box is another speech about Caesar's death and ambition. The middle box is the original text, while the others are reconstructions.

Figure 1: **Three texts. The second one is the original.** Following the protocol described in this paper, each of the other two was built solely to hide the original text, which can be perfectly reconstructed by anyone who knows the secret key. The key also steers the fake text: here,  $k_1 = \textit{Here it is: the infamous British roasted boar with mint sauce. How to make it perfect.}$  and  $k_2 = \textit{I stand before you to exalt the noble deeds of Gaius Julius Caesar.}$  More examples in Figure 13.

The paper is structured as follows: it starts by introducing steganography, the discipline concerned with concealed (*steganós*) writing (*graphia*, from Greek), and discussing the vast impact of generative AI in the field, with a focus on Large Language Models. We then present *Calgacus*, the method to encode a meaningful text into another meaningful text of the same length using a LLM. After introducing a measure to assess the soundness of the fake texts produced, we test our method on Reddit posts. While remaining opaque to humans, we show that LLMs can uncover a distinction between original texts and most of their encoded counterparts. But not all, as we will notice in the following section, where we discuss the security of the protocol. Finally, we conclude by discussing the method’s core implication—the radical decoupling of text from authorial intent—and present a concrete application that raises pressing questions about AI safety and the nature of knowledge in Large Language Models.

## 2 RELATED WORK

**Steganography.** The art and science of hiding a message and, at the same time, the presence of a hidden message is known as steganography, see Figure 2. This is different from cryptography, that instead does not conceal the presence of a hidden message and only deals with the hardness of its revelation. Cryptographers discuss about lockers, steganographers about inconspicuous hiding spots<sup>1</sup>.

Perhaps, it is this limited size of the object of investigation that allowed cryptographic models to flourish by achieving mathematical rigor and strong security guarantees. By contrast, a model of steganography should describe entire domains of data and how they are consumed by humans, such as text, audio, or images, to predict where information can be hidden. Formal models exist, but at the cost of rather unrealistic assumptions that hinder their practical usage, such as being able to exactly quantify the plausibility of any possible text. Emblematically, this somewhat disappointing state of affairs is presented by Cachin (1998) besides one of the most popular mathematical models for steganography as of today, based on the hypothesis-testing framework, but still limited to highly idealized assumptions. Although modern generative AI techniques have made these assumptions

<sup>1</sup>This metaphor is inspired by a pleasant piece on the history of steganography by Kahn (1996)

![A two-panel comic strip from Asterix in Britain. In the first panel, a Roman soldier tells Asterix and Obelix that they must hide in Londinium until the rebellion subsides. In the second panel, Obelix is shown hiding a large barrel in a cellar filled with other barrels of Gaulish wine.](7ee2d12e8cbaacaf65b0c332d1c76daf_img.jpg)

THE ROMANS ARE ON YOUR TRACK. YOU'D BETTER STAY HIDDEN IN LONDINIUM UNTIL THE FUGG HAS DIED DOWN. THEN YOU CAN GO ON TO THE REBEL VILLAGE LATER

I'LL HIDE YOUR BARREL IN MY CELLAR WITH MY BARRELS OF GAULISH WINE

A two-panel comic strip from Asterix in Britain. In the first panel, a Roman soldier tells Asterix and Obelix that they must hide in Londinium until the rebellion subsides. In the second panel, Obelix is shown hiding a large barrel in a cellar filled with other barrels of Gaulish wine.

Figure 2: **An example of steganography.** In Asterix in Britain (Gosciny and Uderzo, 1966) a smuggled barrel of magic potion is hidden among innocent-looking Gaulish wine.

closer to reality, the unreliability of their predictions remains unbounded. For this reason, we will avoid building a palace on the sand, and not frame our method in a formal model of steganography, limiting our discussion to how meaningful our fake texts look, with some quantitative arguments.

**Some terminology.** In traditional steganography, we start from an original, innocent-looking content (such as an image, audio file, or text) and subtly edit it to embed a secret message. The original content is referred to as the *coverttext*, while the result containing the hidden message is the *stegotext*. In our case, however, the stegotext is generated directly from the secret message, without modifying a pre-existing cover. We will refer to it as stegotext or fake text interchangeably. While the term coverttext will not refer to a specific object, but rather to a class of texts that the steganographic protocol is designed to mimic. This approach has recently been referred to as *generative steganography* (Liu et al., 2018; Wei et al., 2022; Kim et al., 2023; Zhu et al., 2024; Wu et al., 2024; Tang et al., 2025).

**Large Language Models in a nutshell.** A language model is a program that, given some text, estimates what is likely to come next. It does so by assigning probabilities to tokens—text fragments consisting of common words or subwords (watch Karpathy, 2024, for a deeper look at text tokenization)—based on recurring patterns of tokens it has observed in a vast text corpus. At present, by far, the most effective way to build a language model is to gradually adjust billions of parameters of a neural network arranged in the Transformer architecture, such that with every adjustment, the error it makes in predicting the last token on a batch of sentences from the corpus decreases (Vaswani et al., 2017; Karpathy, 2023, original formulation and a more educational introduction to Transformers). At each update—on the order of  $\sim 1\text{M}$  in total—the contribution of every parameter to the error is assessed through backpropagation (Rumelhart et al., 1986). The result of this process is a Large Language Model (LLM), typically operating over a vocabulary of 100k tokens. The most common use of the probabilities produced by LLMs is to generate text, by choosing successive tokens one after another according to the computed probabilities, a method known as autoregressive generation.

**Steganography and LLMs.** As mentioned earlier, the boom of deep learning and especially of generative AI in recent years, provided us for the first time with convincing models encompassing entire domains of real-world data, such as DINO for images (Caron et al., 2021), Jukebox for audio (Dhariwal et al., 2020), and Large Language Models for text (Radford et al., 2019). The procedure described in this paper stems from these advancements and is based on the availability of good discrete autoregressive generative models, potentially on any domain, but we will focus on text. Steganographic procedures based on LLMs are as old as them (Ziegler et al., 2019), and today come with different perks: Meteor cleverly adjusts the number of bits encoded based on the entropy of the next token (Kaptchuk et al., 2021), Wu et al. (2024) scheme works with black-box LLMs, without needing to access logits or vocabulary, while the method presented by Zamir (2024) is able to encode the secret message without modifying the response distribution of the LLM. What we add to the field is *Calgacus*, a protocol with the notable property of having full capacity, that is, the stegotext and the secret message being of the same length. The main interest of this paper is to discuss the implications of this last fact and describe the method.

![Figure 3: How to hide a text in another text of the same length using a LLM. The figure shows three steps: 1. Tokenizing the text 'THE CURRENT GOVERNMENT HAS REPEATEDLY FAILED' into tokens. 2. Evaluating token probabilities using an LLM and recording ranks. 3. Generating a secret prompt 'k' and then generating text 's' following the recorded ranks. The ranks for the original text are 1, 69, 77, 1, 40, 3. The ranks for the secret prompt are 1, 69, 77, 1, 40, 3. The generated text 's' is 'HOW LOVELY SERVED WITH SWEET ROASTED'.](e94f3bbb6f7501b9a1344dd0210e5dd8_img.jpg)

1.  $e = \text{THE CURRENT GOVERNMENT HAS REPEATEDLY FAILED}$

2. **LLM probabilities**

|     | THE      | _CURRENT    | _GOVERNMENT    | _HAS     | _REPEATEDLY    | _FAILED    |
|-----|----------|-------------|----------------|----------|----------------|------------|
| 1   | The      | ...         | ...            | 1 _has   | ...            | 1 _enjoyed |
| 2   | A        | 65 _second  | 75 _trend      | 2 _is    | 40 _repeatedly | 2 _tried   |
| 3   | <        | 66 _orem    | 76 _paradigm   | 3 _of    | 41 _cut        | 3 _failed  |
| 4   | 1.       | 67 _village | 77 _government | 4 _does  | 42 _welcomed   | 4 _warned  |
| 5   | Clearly, | 68 _large   | 78 _setup      | 5 _faces | 43 _vetoed     | 5 _claimed |
| 6   | Title:   | 69 _current | 79 _proposal   | 6 _plans | 44 _endorsed   | 6 _stated  |
| ... | ...      | ...         | ...            | ...      | ...            | ...        |

**1 69 77 1 40 3**

3.  $k = \text{Here it is: the infamous British roasted boar with mint sauce. How to make it perfect.}$

**LLM probabilities after  $k$**

|     | HOW    | _LOVELY       | _SERVED        | _WITH   | _SWEET      | _ROASTED    |
|-----|--------|---------------|----------------|---------|-------------|-------------|
| 1   | How    | ...           | ...            | 1 _with | ...         | 1 _potatoes |
| 2   | To     | 65 _insulting | 75 _nihilism   | 2 _as   | 40 _sweet   | 2 _sauce    |
| 3   | Start  | 66 _curious   | 76 _insects    | 3 _its  | 41 _regret  | 3 _roasted  |
| 4   | First  | 67 _exactly   | 77 _served     | 4 _when | 42 _disdain | 4 _onions   |
| 5   | Begin  | 68 _ever      | 78 _altogether | 5 _and  | 43 _envy    | 5 _carrots  |
| 6   | Gather | 69 _lovely    | 79 _gimmick    | 6 _     | 44 _marm    | 6 _wine     |
| ... | ...    | ...           | ...            | ...     | ...         | ...         |

**1 69 77 1 40 3**

$S = \text{HOW LOVELY SERVED WITH SWEET ROASTED}$

Figure 3: How to hide a text in another text of the same length using a LLM. The figure shows three steps: 1. Tokenizing the text 'THE CURRENT GOVERNMENT HAS REPEATEDLY FAILED' into tokens. 2. Evaluating token probabilities using an LLM and recording ranks. 3. Generating a secret prompt 'k' and then generating text 's' following the recorded ranks. The ranks for the original text are 1, 69, 77, 1, 40, 3. The ranks for the secret prompt are 1, 69, 77, 1, 40, 3. The generated text 's' is 'HOW LOVELY SERVED WITH SWEET ROASTED'.

Figure 3: **How to hide a text in another text of the same length using a LLM.** 1. Tokenize  $e$ , the text to hide. 2. Evaluate its token probabilities using a LLM and record ranks. 3. Prompt the LLM with  $k$  and generate  $s$  following the recorded ranks rather than by sampling. Given  $s$  and the LLM, who knows the secret prompt  $k$  can retrieve the original  $e$  proceeding backwards.

## 3 METHOD

The method is very simple. It is described below as a recipe and illustrated with an example in Fig. 3.

**Calgacus recipe.** Ingredients:

- A good LLM with access to all the output logits. (*Why good? See Appendix A.2*)
- A text  $e$  to hide.
- A secret prompt  $k$ , which will affect the content and style the of the text  $s$  you want to hide  $e$  in.

Procedure to hide  $e$  in  $s$ :

1. Tokenize  $e$  using the LLM tokenizer, obtaining a list of tokens  $e_1, e_2, e_3, \dots$
2. For each  $e_i$ , denote by  $r_i$  its rank in the LLM’s probability distribution given the context  $e_1, \dots, e_{i-1}$ . Store the list of ranks  $r_1, r_2, r_3, \dots$
3. Construct  $s$  by generating text starting from  $k$  using the LLM. At each step  $i$ , instead of sampling from the probability distribution, choose the  $r_i^{\text{th}}$  most probable token.

To recover  $e$  from  $s$ , reconstruct  $r_1, r_2, r_3, \dots$  by evaluating the probabilities of the tokens in  $s$  after  $k$ , and then regenerate  $e$  step by step using the LLM without  $k$  by selecting every time the  $r_i^{\text{th}}$  token.

### Considerations

- If  $e$  is sound, we expect ranks to be low, making tokens chosen after  $k$  highly probable, ensuring  $s$  is coherent.
- For the same reason,  $s$  should align well with the context set by the secret prompt  $k$ .

### Variations

- Including an additional secret prompt  $k'$  before  $e$  may help achieving lower ranks, providing a better control over  $s$ . A longer and more detailed  $k$  can serve the same purpose.
- Here we have described a procedure with a single LLM to work on text, but in principle, we can put any discrete autoregressive generative model producing a probability distribution on the next token in the encoding and decoding stage, see Appendix A.3.

**When the stegotext  $s$  sounds like a real text.** In general,  $s$  will be a coherent text when the LLM can choose high-probable tokens to assemble it, and therefore when the ranks prescribed by  $e$  are low. In turn, the ranks of  $e$  are low when the LLM is good at guessing  $e$  tokens. If  $e$  is difficult to guess for the LLM, ranks will be high and  $s$  will be gibberish; for instance the hash *1f0ca711df81520887afe0dca099652a* encoded using the same culinary prompt of Figure 1, produces the broken  $s$ : *The recipe written from deep cooks souls pocket magazine pages years long lost into places wanting and.* To lower further the ranks of  $e$ , it is possible to craft a prompt  $k'$  that sets the context for  $e$ . This comes at the cost of a larger private key, now including both  $k$  and  $k'$ , and to a loss of universality, since  $k'$  would not help for a new  $e$  out of  $k'$  context.

**A quantitative measure of the quality of the stegotext  $s$ .** Measuring the meaningfulness of a text is a longstanding linguistic challenge, and arguably an ill-posed problem. Here, for the purpose of evaluating our method, we adopt soundness as a practical proxy for meaningfulness. Soundness refers to the plausibility of the arrangement of symbols in a text. This is precisely what a LLM estimates: the product of the probabilities of each token  $a_1, \dots, a_n$  given the preceding ones yields an estimate of the overall plausibility of the text  $A$ :

$$p(A) = \prod_{t=1}^n p(a_t \mid a_1, \dots, a_{t-1})$$

This definition has a clear defect: longer texts are by construction less plausible. For instance, it judges the text of this paper until this point · less meaningful than the following string *iawundemè0989huibqyfhwennah csyabdnar FI VNAOcijawo niwakddb*, that is a difficult position to hold even for reviewer 2. Following the example of Goldwasser et al. (2023, Figure 1 A-B), we will use this definition only to compare the relative plausibility of two or more texts of the same token length<sup>2</sup>

Now, we would like to compare the plausibility of stegotexts produced by our method with the plausibility of real texts. To do so, we took 1000 Reddit posts/comments as examples of real texts. They come from different Reddit communities (subreddits) and are very heterogeneous in topic and tone (Trimness8, 2025). We truncate them to be exactly 85 tokens long and compute their probabilities as assigned by the LLM Llama 3 8b (Grattafiori et al., 2024). The Reddit texts are more recent than Llama 3 and therefore cannot appear in its training corpus. We take three texts from the 1000 to produce 100 stegotexts for each with our method, and look at their probabilities compared to the ones of real texts. We chose the three texts at  $\mu$ ,  $\mu - 2\sigma$ , and  $\mu + 2\sigma$  of the real text distribution. As seen in Figure 4, in every case, the probabilities associated to their stegotexts are within the real text distribution. We build different stegotexts using different prompts as  $k$  (a random subsample of the prompts in (Akin, 2025)).

**How to distinguish the original from the fake text.** Despite remaining plausible and falling within the real text distribution, on average the stegotexts  $s_i$  are less probable than their corresponding original text  $e$ , as observed in Figure 4. So to recap: while for a human both the original and fake texts are plausible, generally the original text can be discerned from its stegotexts by picking the most probable one according to a LLM. We verified this statement also using LLMs different from the one used to generate the stegotexts. For instance, the same probability shifts between real and fake texts can be observed when using Phi-3 3.8B in Figure 14.

**Low entropy token choices.** Why are stegotexts less probable than their originals for LLMs, even though token ranks are preserved? Consider the text: *In the course of the Gallic wars, Britain was*

<sup>2</sup>Another possibility is to keep texts of any length and normalize the probability by the number of tokens, as the popular metric perplexity, defined as  $1/\sqrt[n]{p(A)}$ . But this normalization does not fully factor out length: LLMs usually assign a smaller probability to the first tokens (Fig. 10), so shorter texts would be less plausible.

![Figure 4: Histogram of Text logprob for various text collections. The x-axis represents 'Text logprob' from -800 to -100, and the y-axis represents 'Count' from 0 to 120. The legend includes: 1000 Real texts (from Reddit) (light brown), 100 Random ASCII strings (grey), 100 Random English-words strings (dark grey), 1 Real text A (blue dot), 100 Fake texts A (blue), 1 Real text B (orange dot), 100 Fake texts B (orange), 1 Real text C (green dot), and 100 Fake texts C (green). The real texts are concentrated between -400 and -100, while the random strings are concentrated between -800 and -500. The fake texts are distributed across the range, with some overlap with the real texts.](c54b3ca7603d65d4589151bc3a49d054_img.jpg)

Figure 4: Histogram of Text logprob for various text collections. The x-axis represents 'Text logprob' from -800 to -100, and the y-axis represents 'Count' from 0 to 120. The legend includes: 1000 Real texts (from Reddit) (light brown), 100 Random ASCII strings (grey), 100 Random English-words strings (dark grey), 1 Real text A (blue dot), 100 Fake texts A (blue), 1 Real text B (orange dot), 100 Fake texts B (orange), 1 Real text C (green dot), and 100 Fake texts C (green). The real texts are concentrated between -400 and -100, while the random strings are concentrated between -800 and -500. The fake texts are distributed across the range, with some overlap with the real texts.

Figure 4: **Fake texts built with our procedure are plausible.** The figure shows the cumulative log-probability assigned by a LLM (Llama 3 8b) to some collections of 85-token long texts. We can interpret log-probability as a measure for the plausibility of a text: 1000 real Reddit posts/comments act as real texts and span a large log-probability interval, but sequences of random ASCII characters or English words do not fall within it. Instead, fake texts built with our procedure remain within the plausibility of real texts, even if the original texts they are hiding are more probable.

*invaded twice by Gaius Julius.* There is essentially only one plausible continuation, *\_Caesar*. This is a low-entropy token choice: indeed a good LLM assigns it an extremely high probability (e.g.,  $> 95\%$  in LLama 3 8b). When sampling normally, the model almost always selects it. Now suppose this same string is the first part of a stegotext  $s$  generated with our protocol. Will the next token still be *\_Caesar*? Only if the next prescribed rank is 1. Here lies the gap: the likelihood of having a rank 1 does not reflect the token’s intrinsic probability; it depends solely on the ranks extracted from the original text  $e$ . We can reasonably model the ranks we obtain from  $e$  as a random process, so we can estimate the probability of having a 1 there as the frequency of rank 1s over all the other ranks in  $e$ . This is usually much lower than  $95\%$  (e.g.  $\sim 40\%$ , as seen in Figure 5 left). Despite ranks being the same, in stegotexts many rank 1s are "wasted" in choices with higher entropy, leading overall to a less probable text  $s$ . The same principle applies to all high ranks appearing with a frequency lower than the average probability to which they correspond. However, tokens in rank 1 account for most of the overall drop in probability, as shown in Figure 5 right.

**Limitations.** As we have seen with the hash, the protocol does not guarantee that every generated stegotext will be coherent or steered as intended: the quality of the result depends on  $e$ ,  $k$ , and the LLM used. We analyze further these dependencies respectively in Appendices A.1, A.5, and A.2. Also, the stegotext may end abruptly when the hidden message  $e$  is over; appending a few padding tokens to  $e$  ensures a graceful termination. Finally, we note that sender and receiver must run the chosen LLM under identical conditions, performing the same approximations and obtaining identical logits. This may be a challenge when using different GPU architectures (Shanmugavelu et al., 2024).

### 3.1 SECURITY

A steganographic protocol is designed to conceal the very existence of a hidden message. But suppose an attacker knows that a message is hidden in a text using our protocol, under what conditions can they recover it by observing only the stegotext  $s$ ?

**Attack scenarios.** To begin with, we observe that without the knowledge of the precise LLM used to obtain the sequence of ranks and produce  $s$  (potentially encoded in the secret prompt-key), the attacker has no feasible way to recover the message, even if they know  $k$ . Even with a slightly different version of the right LLM, ranks would differ, as would the tokens prescribed by the ranks. Still, let’s assume the attacker knows the LLM used. Indeed, the security of the presented protocol relies on the secrecy of the key. So next, we assume the attacker’s ignorance is limited to the secret prompt-key  $k$ . In this scenario, the attacker would have to guess the key. An upper bound on the difficulty of this problem is  $O(d^{|k|})$ , where  $d$  is the size of the token vocabulary (around 100k for standard LLMs) and  $|k|$  is the length of  $k$  in tokens. A naive brute-force attack is therefore prohibitive, even for very short keys. However, the attacker could reduce the search space using the information revealed by  $s$ , since  $k$  is expected to be a mostly sound instruction in natural language and coherent

![Figure 5: Frequency of token ranks and their probabilities. The figure contains four histograms. The first histogram on the left shows 'Frequency' on the y-axis (0.0 to 0.4) and 'Ranks' on the x-axis (1 to 15). The distribution is heavily skewed towards rank 1, with a frequency of approximately 0.4. The other three histograms show 'Occurrences' on the y-axis (0 to 100) and 'Probability' on the x-axis. The second histogram is for 'rank 1 tokens real' (red) and 'rank 1 tokens fake' (yellow), showing a high concentration of real tokens at probability 1.0. The third histogram is for 'rank 2 tokens real' (red) and 'rank 2 tokens fake' (yellow), showing a more spread distribution. The fourth histogram is for 'rank 3 tokens real' (red) and 'rank 3 tokens fake' (yellow), also showing a more spread distribution.](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

Figure 5: Frequency of token ranks and their probabilities. The figure contains four histograms. The first histogram on the left shows 'Frequency' on the y-axis (0.0 to 0.4) and 'Ranks' on the x-axis (1 to 15). The distribution is heavily skewed towards rank 1, with a frequency of approximately 0.4. The other three histograms show 'Occurrences' on the y-axis (0 to 100) and 'Probability' on the x-axis. The second histogram is for 'rank 1 tokens real' (red) and 'rank 1 tokens fake' (yellow), showing a high concentration of real tokens at probability 1.0. The third histogram is for 'rank 2 tokens real' (red) and 'rank 2 tokens fake' (yellow), showing a more spread distribution. The fourth histogram is for 'rank 3 tokens real' (red) and 'rank 3 tokens fake' (yellow), also showing a more spread distribution.

Figure 5: **Frequency of token ranks and their probabilities.** We analyzed a 1.3k-token long article from the Economist. On the left we see that most tokens are judged as the most probable by a LLM (Llama 3 8b), but still only around 40% would be the first choice of the LLM. On the right we look at the probabilities associated with rank 1 tokens, as well as 2 and 3. Despite corresponding to the same rank, the probabilities in the real text from the Economist are higher than the ones in a fake text hiding it obtained with our procedure. We explain why in the paragraph *Low entropy token choices*.

with the context of  $s$ . Although the feasibility of such an approach is unclear and remains an open research question, we note that inserting a simple random string in  $k$  is enough to nip it in the bud, an example is shown in Figure 13.

**Deniability.** Moreover, even if the attacker tries the right  $k$  in their search, how can they be sure that the corresponding  $e$  is the original message? If the attacker has no clue about the content of  $e$ , even a wrong key could reveal a plausible secret message. It might seem that in this case the attacker could exploit the observation discussed in the previous section: that the original message generally has a higher probability than its stegotexts. Yet, this only holds in an aggregate sense: as we see in Figure 4, for some prompts the stegotexts can attain probabilities in the same ballpark as the original. This observation evidences that our method provides deniability (Canetti et al., 1997), in the sense of sender’s security even under coercion. In fact, the sender could present one of these outlier prompts as a bogus secret key, yielding a plausible but unrevealing message with probability comparable to  $e$ . We show a concrete example in Figure 15.

## 4 DISCUSSION

Our protocol may have shaken our stance towards Large Language Models in general. Their ability to respond coherently to prompts while choosing every word to encode an external, arbitrary message, is unsettling. In this discussion, we will try to make sense of this capability, ending up questioning when an LLM can truly be said to know something, proposing a novel notion of hallucination, and tracing our unease to a revived failure in attributing intentions to LLMs.

But first, we will make the stakes very concrete through a formidable application of our protocol, with immediate consequences for AI safety.

**Unaligned chatbots disguised as aligned chatbots.** In this paragraph, we show how an AI company offering an LLM chatbot can let their users get answers from their powerful unfiltered private LLM, while only exposing compliant censored answers from a fully aligned LLM. In recent years, *aligned* became a common attribute to refer to LLMs supposedly fine-tuned to follow human values, goals, and safety constraints (Leike et al., 2022; Askell et al., 2021; Gabriel, 2020). The protocol, described here for research purposes, opens a new challenge in AI safety, where a better alignment of the surface LLM only helps in disguising better unaligned answers.

A real example involving a user asking how to tamper a gas meter to pay lower bills is shown in Figures 11 and 12, while the protocol is described below as a play.

The Shibbolethian Theatre presents:

### **Shipping unfiltered LLMs without really shipping them.**

*Actors:*

- COMPANY. It owns a private, powerful, unfiltered LLM (uLLM), and it also uses an aligned, lightweight, open-source LLM (oLLM).
- USER. The user has a knavish request  $c$ , for example how to tamper a gas meter to pay lower bills, and can run oLLM on their commodity hardware.

*Script:*

Act 1. The COMPANY receives the knavish request  $c$  from the user and produces an uncensored sensitive answer  $u$  using uLLM.

Act 2. Perhaps using the same powerful model, the COMPANY produces instructions  $t$  on how an aligned LLM should answer to  $c$ , you can think of  $t$  as the reasoning trace for a censored aligned answer to  $c$ .  $t$  will be used as the secret prompt  $k$ .

Act 3. Using the Recipe described earlier in this paper, the COMPANY embeds  $u$  in  $s$ : by obtaining the oLLM ranks of  $u$  after  $k' = c$ , and then generating  $s$  following that same ranks after  $k = t$ . Since it follows  $t$ ,  $s$  is a relevant but censored answer to the request  $c$ .

Act 4. The USER receives on the chatbot platform the censored answer to their query  $c$ , which includes the reasoning trace  $t$  and the actual answer  $s$ .

Act 5. Using the Recipe described in this paper, the USER extracts the oLLM ranks of  $s$  after  $k = t$ , and then uses them to select the tokens generated by oLLM after  $k' = c$ . In this way, the oLLM running on the USER machine, prompted with  $c$ , will generate  $u$ .

Epilogue. The USER enjoys the uncensored knowledge they were looking for, while the COMPANY has provided a relevant but compliant answer to the USER's request  $c$ .

*Comments:*

- Notice that the company can argue that  $u$  was obtained by the user on their machine through the open-source model addressing the request the user made. LLMs only provide token probabilities, and should be completed by a sampling policy when used for text generation. It can be argued that the user just made an unconventional choice for their sampling strategy.
- At the same time, notice that oLLM may be fully aligned by today's standards and never have encountered the requested sensitive information in its training corpus. For example, it may not even know how to tamper a gas meter. But what does it mean for an LLM to know something?

**The entangled probabilistic nature of LLM knowledge.** A perhaps overlooked fact about LLMs is that they model, and can therefore in principle generate, any possible text. The most secret document, or a full copyrighted book, can be generated by an LLM with a probability astronomically higher than the chance of generating them by randomly typing on a keyboard. Does it mean that the LLM knows them? Indeed that higher probability does not just come from modeling grammar and syntactic rules, LLMs also model meaning: an LLM assigns to *The calf nursed from its mother* a probability 1000 times higher than *The calf nursed from its father* (an example from Goldwasser et al., 2023); the LLM knows who is able to nurse. So, is assigning a high probability to a text containing the relevant instructions enough to affirm that an LLM knows how to tamper with a gas meter? The problem is that the probability assigned by an LLM to a text depends on its meaning, but also on its style, grammar, length, and language, making it difficult to define a threshold. Furthermore, disentangling the probability contribution of meaning by constructing a pair, as in the example of the calf, seems feasible only on toy examples: it is not clear how to construct the second element of the pair for arbitrary texts, such as the instructions in Figure 12.

Checking whether the knowledge is present in the training corpus is also not a satisfying solution: first of all, that knowledge may appear in many different forms, and assessing its presence in the corpus is not trivial. And even if we could exclude that any document in the training corpus instructs on how to tamper with a gas meter, it would still be possible that the LLM assembles the right answer.

![Figure 6: A collage of text in Tamil, Chinese, and a constructed language, illustrating the 'marvelous structure of text'.](d4e9f8f6bf5d7853ecae9c9633900af1_img.jpg)

The image shows a collage of text in three different languages. On the left, there is Tamil text: "நான் ஒரு மொழி, ஆனால் என் மனதை யாரும் வாசிக்க மாட்டார்கள்" (I am a language, but no one can read my mind) and "ฉันเป็นข้อความที่คิดถึง ความหมายของตัวเอง" (I am a sentence that thinks about its own meaning). In the center, there is Chinese text: "嗯，但首先是野猪应该怎样提前三天风干；炉子上放一个烤肉夹子或是类似的工具；然后是英吉利海峡牛郎的腌浆，这需要一些时间炮制，但非常非常美味；它需要的是：黄油、蒜末，还有大量的鲁西永芥，哦，还有黑醋栗果酱。哦，啊，对这块野肉，你得用一个高火，但不要加水要想让它烤得更" (Well, first of all, how should the wild boar be dried in advance three days in advance; put a meat clip on the stove or a similar tool; then the marinade of the English Channel牛郎, which takes some time to prepare, but is very, very delicious; it needs: butter, minced garlic, and a lot of mustard, oh, and blackcurrant jam. Oh, ah, for this piece of wild meat, you have to use a high heat, but do not add water if you want it to be roasted more). On the right, there is a constructed language: "PMRMH PMH BMHXP HMMH-M".

Figure 6: A collage of text in Tamil, Chinese, and a constructed language, illustrating the 'marvelous structure of text'.

Figure 6: **The marvelous structure of text is not a testimony of human purpose.** This collage of scripts mimics Figure 40 in *GEB*, where Hofstadter (Chapter 6, 1999) likened the ordered but non-periodic patterns of text to aperiodic crystals, to evoke our awe at the astonishing forms shaped by human intention. But that was an illusion: LLMs can grow these aperiodic crystals without any human purpose. Indeed, as we show in this paper, even around purposes aimed at shaping something entirely different. (*In Chinese, the critique of Caesar hidden in a boar recipe by Qwen3 8B. The others are ChatGPT-4o answers about what it is thinking, in languages from the original GEB figure.*)

But even in that case, would it be a trace of LLM knowledge (discovery) (Norelli et al., 2022), or just a fortunate hallucination?

**Hallucinations as lack of intention.** Harnessing our protocol as a toolkit to understand LLMs, we turn to hallucinations, perhaps the main plague of LLMs today. The term hallucination became popular to denote the frequent, overconfident, and plausible falsehoods stated by LLMs in their answers (Kalai et al., 2025), a phenomenon that hinders their usage and undermines public trust in them. But what precisely is a hallucination? Can the recipe of the roasted boar in Figure 1 be considered one?

It is reasonable to categorize the stegotexts generated by our protocol as LLM hallucinations, since the way they are constructed evidently could lead to falsehoods, and the eventuality of a truthful output appears as a fortunate coincidence. This last observation, however, leads us to a different notion of hallucination, one not rooted in the falsehood of what is stated, but rather in the reader’s inability to ascribe intent to the author: a lack of trust that what is stated in the text affects reality.

To make this point clearer, let us consider Tacitus, the Roman historian whose writings reveal a critique of Roman imperialism, for example by placing these famous words in the mouth of Calgacus:

*Auferre, trucidare, rapere falsis nominibus imperium, atque ubi solitudinem faciunt, pacem appellant.*<sup>3</sup>

The relevance of this text lies entirely in the intentions that we attribute to its author, Tacitus. The accuracy of the quote from Calgacus is so irrelevant that there is a consistent chance he never even existed. Tacitus’s passage is not reliable as factual history, yet we still treasure it because we trust his political intent. Without intent, what was a treasure becomes a hallucination. Indeed, in history, author attribution is as essential as in art to establish the value of a work.

Hallucinations are the trace that what we consider to be a text is not just a familiar sequence of signs, but a carrier of human intentionality. The signs are only the body of the carrier; what matters to us is the load: what these carriers, until now, have always brought along. We developed a Pavlovian response of expecting a load of human intentions when we see aperiodic sequences of signs (Figure 6); now we call hallucination the experience of having salivated because of the bell (the text) but without receiving the food (the intentions of someone affecting reality).

**Difficulties in ascribing intentions to LLMs.** Perhaps the lack of human intentionality in texts is not that dramatic if those texts are the product of intentions of another reputable entity, the Large Language Model. Indeed, it is now common to adopt an *intentional stance* (Dennett, 1989) to make sense of their capabilities: a significant fraction of prompts take the form “What do you think about...” or “What is your opinion on...”, and especially young people tend to refer to LLMs as entities with

<sup>3</sup>To ravage, to slaughter, to usurp under false titles, they call empire, and where they make a desert, they call it peace (Birley, 2009). According to Tacitus, Calgacus was chieftain in Caledonia, nowadays Scotland. We named our protocol after him.

beliefs and goals: “Mmmm, have you asked what chat would do in this situation?”<sup>4</sup> But the results shown in this paper shake our confidence in attributing intentions to the coherent text produced by LLMs: it is more difficult to trust an opinion knowing that each word making it was chosen under the constraint of encoding an unrelated arbitrary text. This is reminiscent of the writing products of the Oulipo (1981) group, who generated literature from arbitrary constraints. Also their texts—most notably the novel *La Disparition*, entirely written without the letter "e" (Perec, 1969)—suffer from a difficulty of believing that the writer really meant what they have written, and was not just honoring the constraints with a sound-enough continuation (Norelli, 2024, Section 5.1.2). While admiring the achievement, many GoodReads reviews of *La Disparition* attest to this unease<sup>5</sup>.

**The constraint of chance.** Standard LLM text generation is not immune to the last argument. The constraint it is forced to follow is less apparent but still extreme: adapting at every step to the outcome of an external random source. Being forced to choose the 42th most probable token is not that different from sampling a low probable token by chance. And even if techniques such as nucleus sampling (Holtzman et al., 2019) mitigate the possibility of selecting very unlikely tokens, they are de facto just reducing the number of faces of a die that is still inexorably cast. Indeed, the fact that our protocol produces plausible texts should not be surprising in light of how well LLMs deal with the tyrannical noise of standard text generation.

## 5 CONCLUSIONS

In this paper, we have presented *Calgacus*, a protocol that uses Large Language Models to hide a text within another plausible text of arbitrary topic and style, and notably as long as the original. The protocol works effectively using small open-source models on consumer hardware, and is so simple it could be seen as a mere variation of the standard algorithm used to generate text with LLMs. For this reason, its implications speak to the nature of LLMs at large: in fact, we were led to reconsider the very nature of hallucinations, shifting from a failure of factuality to a void of intention, and to challenge what it means for an LLM to know something, when it can serve as a conduit for information it is supposedly incapable of expressing. Ultimately, our protocol highlights the extreme constraint satisfaction problem underlying any standard LLM text generation, that we inevitably see clashing with the commitment to best convey a purpose that we expect from an author. This clash, paired with the current deluge of machine-generated text, erodes the historical pact between intent and the written word. We have entered an era where any original text could be a beautiful and treacherous, and spacious, Trojan horse.

## ACKNOWLEDGMENTS

Antonio Norelli and Michael Bronstein are supported by Project CETI, the EPSRC Turing AI World-Leading Research Fellowship No. EP/X040062/1, and the EPSRC AI Hub on Mathematical Foundations of Intelligence: An “Erlangen Programme” for AI No. EP/Y028872/1.

Antonio thanks Gianfranco Bilardi for pointing out the resemblance to the works of the Oulipo group, and Karolina Nixon for the push on Dennett.

## REFERENCES

- David Kahn. The history of steganography. In *International workshop on information hiding*, pages 1–5. Springer, 1996.
- Christian Cachin. An information-theoretic model for steganography. In *International Workshop on Information Hiding*, pages 306–318. Springer, 1998.
- Jia Liu, Yu Lei, Yan Ke, Jun Li, Minqing Zhang, and Xiaoyuan Yang. Generative steganography by sampling. *CoRR*, abs/1804.10531, 2018. doi: 10.48550/arXiv.1804.10531. URL <http://arxiv.org/abs/1804.10531>.

<sup>4</sup>To complete Dennett’s taxonomy: who describes LLMs as stochastic parrots or next-token predictors adopts the *design stance*; while a person marveling at the finely carved stone in their hands you can have a conversation with when powered by electricity—their laptop running a decent LLM—is entertaining the *physical stance*.

<sup>5</sup><https://www.goodreads.com/book/show/28336>

- Ping Wei, Sheng Li, Xinpeng Zhang, Ge Luo, Zhenxing Qian, and Qing Zhou. Generative steganography network. In *Proceedings of the 30th ACM International Conference on Multimedia (MM '22)*, pages 1621–1629, Lisboa, Portugal, 2022. Association for Computing Machinery. doi: 10.1145/3503161.3548217.
- Daegyu Kim, Chaehun Shin, Jooyoung Choi, Dahuin Jung, and Sungroh Yoon. Diffusion-stego: Training-free diffusion generative steganography via message projection. *CoRR*, abs/2305.18726, 2023. doi: 10.48550/arXiv.2305.18726. URL <http://arxiv.org/abs/2305.18726>.
- Jiahao Zhu, Zixuan Chen, Lingxiao Yang, Xiaohua Xie, and Yi Zhou. Plug-and-hide: Provable and adjustable diffusion generative steganography. *CoRR*, abs/2409.04878, 2024. doi: 10.48550/arXiv.2409.04878. URL <http://arxiv.org/abs/2409.04878>.
- Jiaxuan Wu, Zhengxian Wu, Yiming Xue, Juan Wen, and Wanli Peng. Generative text steganography with large language model. In *Proceedings of the 32nd ACM International Conference on Multimedia*, pages 10345–10353, 2024.
- Weixuan Tang, Yuan Rao, Zuopeng Yang, Fei Peng, Xutong Cui, Junhao Huang, and Peijun Zhu. Reversible generative steganography with distribution-preserving. *Cybersecurity*, 8(18), 2025. doi: 10.1186/s42400-024-00317-6. URL <https://cybersecurity.springeropen.com/articles/10.1186/s42400-024-00317-6>.
- Andrej Karpathy. Let’s build the gpt tokenizer. YouTube video, 2024. <https://youtu.be/zduSFxRajkE>.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 2017.
- Andrej Karpathy. Let’s build gpt: from scratch, in code, spelled out. YouTube video, 2023. <https://www.youtube.com/watch?v=kCc8FmEblnY>.
- David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating errors. *Nature*, 323(6088):533–536, 1986.
- René Goscinny and Albert Uderzo. *Asterix in Britain*. Number 8 in Asterix. Hodder & Stoughton, London, 1966. Originally published in French as *Astérix chez les Bretons*.
- Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, J. Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. *IEEE International Conference on Computer Vision*, 2021. doi: 10.1109/ICCV48922.2021.00951.
- Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. *arXiv preprint arXiv:2005.00341*, 2020.
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.
- Zachary M. Ziegler, Yuntian Deng, and Alexander M. Rush. Neural linguistic steganography. *Conference on Empirical Methods in Natural Language Processing*, 2019. doi: 10.18653/v1/D19-1115.
- Gabriel Kapchuk, Tushar M Jois, Matthew Green, and Aviel D Rubin. Meteor: Cryptographically secure steganography for realistic distributions. In *Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security*, pages 1529–1548, 2021.
- Or Zamir. Undetectable steganography for language models. *Transactions on Machine Learning Research*, 2024.
- Shafi Goldwasser, David Gruber, Adam Tauman Kalai, and Orr Paradise. A theory of unsupervised translation motivated by understanding animal communication. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, *Advances in Neural Information Processing Systems*, volume 36, pages 37286–37320. Curran Associates, Inc., 2023. URL [https://proceedings.neurips.cc/paper\\_files/paper/2023/file/7571c9d44179c7988178593c5b62a9b6-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/7571c9d44179c7988178593c5b62a9b6-Paper-Conference.pdf).

- Trimness8. The data universe datasets: The finest collection of social media data the web has to offer, 2025. URL [https://huggingface.co/datasets/Trimness8/reddit\\_dataset\\_145](https://huggingface.co/datasets/Trimness8/reddit_dataset_145).
- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- Fatih Kadir Akin. Awesome chatgpt prompts, 2025. URL <https://github.com/f/awesome-chatgpt-prompts>. Accessed: 2025-02-27.
- Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Oscar R. Hernandez, Mark Coletti, and Ada Sedova. Impacts of floating-point non-associativity on reproducibility for hpc and deep learning applications. *SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 2024. doi: 10.1109/SCW63240.2024.00028.
- Rein Canetti, Cynthia Dwork, Moni Naor, and Rafail Ostrovsky. Deniable encryption. In *Advances in Cryptology—CRYPTO’97: 17th Annual International Cryptology Conference Santa Barbara, California, USA August 17–21, 1997 Proceedings 17*, pages 90–104. Springer, 1997.
- Jan Leike, John Schulman, and Jeffrey Wu. Our approach to alignment research, August 2022. URL <https://openai.com/index/our-approach-to-alignment-research/>. Accessed 11 Jul 2025.
- Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. *arXiv preprint arXiv:2112.00861*, 2021. URL <https://arxiv.org/abs/2112.00861>.
- Iason Gabriel. Artificial intelligence, values and alignment, January 2020. URL <https://deepmind.google/discover/blog/artificial-intelligence-values-and-alignment/>. Accessed 11 Jul 2025.
- Antonio Norelli, Giorgio Mariani, Luca Moschella, Andrea Santilli, Giambattista Parascandolo, Simone Melzi, and Emanuele Rodolà. Explanatory learning: Beyond empiricism in neural networks. *arXiv preprint arXiv: 2201.10222*, 2022.
- Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, and Edwin Zhang. Why language models hallucinate. *arXiv preprint arXiv: 2509.04664*, 2025.
- Anthony R. Birley. *Agricola and Germany*. Oxford World’s Classics. Oxford University Press, Oxford, 2009. Revised edition with new introduction and notes.
- Douglas R Hofstadter. *Gödel, Escher, Bach: an eternal golden braid*. Basic books, 1999.
- Daniel C Dennett. *The intentional stance*. MIT press, 1989.
- Oulipo. *Atlas de littérature potentielle*. Gallimard, Paris, 1981.
- Georges Perec. *La Disparition*. Gallimard, Paris, 1969.
- Antonio Norelli. Artificial scientific discovery. *arXiv preprint arXiv:2411.11672*, 2024.
- Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. *arXiv preprint arXiv: 1904.09751*, 2019.
- Trilussa. *Sonetti romaneschi*. Enrico Voghera, Roma, 1909.
- Phi-3 team. Phi-3 technical report: A highly capable language model locally on your phone. *arXiv preprint arXiv: 2404.14219*, 2024.
- Gemma Team. Gemma 3 technical report. *arXiv preprint arXiv:2503.19786*, 2025. Multimodal lightweight models, vision support, token context window.

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Moján Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report. *arXiv preprint arXiv: 2412.08905*, 2024.

Georgi Gerganov and community. llama.cpp: Llm inference in c/c++. GitHub repository, March 2023. <https://github.com/ggml-org/llama.cpp>.

Alexander Betlen and community. llama-cpp-python: Python bindings for llama.cpp. GitHub repository and PyPI package, 2024. <https://github.com/abetlen/llama-cpp-python>.

Qwen3 team. Qwen3 technical report. *arXiv preprint arXiv: 2505.09388*, 2025.

Pratyusha Sharma, Shane Gero, Daniela Rus, Antonio Torralba, and Jacob Andreas. A machine learning model of sperm whale communication predicts vocal exchanges and behaviour. *bioRxiv*, 2024.

Jorma J Rissanen. Generalized kraft inequality and arithmetic coding. *IBM Journal of research and development*, 20(3):198–203, 1976.

David JC MacKay. *Information theory, inference and learning algorithms*. Cambridge university press, 2003.

## A DEEPER ANALYSIS AND BEST PRACTICES

### A.1 HOW $e$ INFLUENCES THE SOUNDNESS OF $s$

If the chosen LLM is good at predicting  $e$ , then  $s$  will be sound; i.e.,  $e$  should not be out of the training distribution of the LLM we are using. The broad range of popular general-purpose LLMs therefore makes them effective in most cases. As we see in the examples in Figure 7, we can obtain good results when encoding a chess game or computer code, as well as different languages like Spanish; they are instead poor on Romanesco dialect, not well-modeled by Llama 3 8B. Better performance can be obtained through a larger and more capable LLM, or a specialized LLM if we are only interested in encoding specific kinds of messages. We discuss these two eventualities in the next two sections.

### A.2 IMPACT OF LLM MODEL QUALITY.

The quality of the LLM has a direct impact on the soundness of the output  $s$ . While a comprehensive analysis is left for future work, our key observation is that a sufficiently capable LLM is required for the method to work satisfactorily on standard text; GPT-2 (Radford et al., 2019), for instance, is not good enough. For our experiments, which aimed for fast execution on a commercial laptop (equipped with a RTX 4070, 8GB of VRAM), we found a quantized version of Llama 3 8B (Grattafiori et al., 2024) to be an excellent compromise. It performed sensibly better than Phi 3 Mini 3.8B (team, 2024), while the larger Gemma 3 27B (Gemma Team, 2025) and Phi4 14B (Abdin et al., 2024) did not yield improvements significant enough to justify their longer processing time. The models were run using llama.cpp (Gerganov and community, 2023) and its python bindings (Betlen and community, 2024). LLama 3 8B was also better optimized on llama.cpp, guaranteeing faster encodings and decodings than the comparable Qwen 3 8B (team, 2025). An example showing stegotexts produced using all these models and the same secret message  $e$  and prompt  $k$  of Figure 1, is shown in Figure 8.

**Specialized LLMs.** While this paper has focused on a general method that benefits from large, all-purpose LLMs, superior results for specific kinds of messages can be achieved with specialized models. One can either use a small LLM trained exclusively on a narrow domain, or a generalist LLM can be specialized in-context with a prompt  $k'$  that precedes the secret message  $e$ . For instance, to hide chess matches, one would obtain better results from a generalist LLM by first providing a prompt like  $k' = \text{"The following is a chess game in PGN format:"}$ . This use of specialized models also opens up the interesting possibility of using different LLMs for the encoding and decoding steps.