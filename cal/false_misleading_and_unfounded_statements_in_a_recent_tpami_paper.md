# FALSE, MISLEADING, AND UNFOUNDED STATEMENTS
## IN A RECENT TPAMI PUBLICATION

**Anonymous authors**
Paper under double-blind review


ABSTRACT


A recent TPAMI response raises issues with the contents of a recent TPAMI comment and
the data collection underlying that comment. Several of the claims in that response are unfounded, inaccurate, misleading, false, invalid, or unsupported, as demonstrated by text in
the comment and cited work, and new analyses that we report. The response further ignores
key components of the work that it responds to.


1 INTRODUCTION


A recent response (Palazzo et al., 2024) raises issues with a recent comment (Bharadwaj et al.,
2023) and the data collection (Ahmed et al., 2021) underlying that comment. Several of the claims
in Palazzo et al. (2024) are unfounded, inaccurate, misleading, false, invalid, or unsupported, as
demonstrated by text in Bharadwaj et al. (2023) and Ahmed et al. (2021), and new analyses that we
report. Palazzo et al. (2024) further ignore key components of Bharadwaj et al. (2023) and Ahmed
et al. (2021). We clarify these below.


2 SIGNAL BLEEDING ACROSS TRIALS


Palazzo et al. (2024) claim that the interleaved design used by Bharadwaj et al. (2023) and Ahmed
et al. (2021) allows brain activity measured by EEG to bleed between adjacent trials. [1]


_On the contrary, interleaved-design experiments introduce several confounds that may sup-_
_press_ _the_ _very_ _response_ _that_ _one_ _would_ _hope_ _to_ _classify_ _with_ _machine_ _learning_ _methods._
_Indeed, object recognition in humans tends to last many hundreds of milliseconds (especially_
_when the items change rapidly)._ _This means that components such as the P300 and the N400_
_may still be processing the item from one class, when an item from the next class is presented_

_[14]._ _This response overlap certainly results in the signal bleeding into the subsequent trial._


Palazzo et al. (2024)


While this may be true for designs such as those used by Spampinato et al. (2017), Kavasidis et al.
(2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021), Li et al. (2021), and Ahmed et al. (2022)
where trials had duration 0.5 s and did not have any blanking between trials, the trials in Ahmed
et al. (2021), one of the datasets used by Bharadwaj et al. (2023), had duration 2 s with 1 s blanking
between trials.


_Each run started with 10 s of blanking, followed by 400 stimulus presentations, each lasting_
_2 s, with 1 s of blanking between adjacent stimulus presentations, followed by 10 s of blanking_
_at the end of the run._


Bharadwaj et al. (2023)


In the design of Ahmed et al. (2021), one of the datasets used by Bharadwaj et al. (2023), the items
do not change rapidly and the 1 s blanking between trials is likely to preclude significant signal
bleeding between adjacent trials. Thus the claim by Palazzo et al. (2024) that the interleaved design
used by Bharadwaj et al. (2023) and Ahmed et al. (2021) “certainly results in the signal bleeding
into the subsequent trial” is unfounded.


1All citation numbers in quoted text are those in the original.


1


3 SUBJECT ATTENTIVENESS


Palazzo et al. (2024) claim that block designs make the class more salient than interleaved designs
and raised a concern about the attentiveness of the subject in Ahmed et al. (2021).


_Additionally, when items are presented in a block, it is possible to make the class very salient_
_(i.e.,_ _the participant will notice that they have viewed 50 dogs in a row),_ _whereas the inter-_
_leaved_ _design_ _obscures_ _the_ _point_ _of_ _the_ _study._ _In_ _this_ _case,_ _if_ _the_ _subjects_ _were_ _even_ _mildly_
_inattentive,_ _they_ _would_ _certainly_ _fail_ _to_ _think_ _about_ _the_ _current_ _class,_ _something_ _that_ _is_ _far_
_harder to miss in the block-design._ _Obscuring the class like Bharadwaj et al._ _did, without re-_
_quiring an overt response from the subject, calls into question if the subject was even paying_
_attention to the stimuli,_ _whereas an overt response forces the subject to attend to and more_
_fully process the stimuli to the class level [14]._


Palazzo et al. (2024)


That may be an issue when presenting stimuli for 0.5 s with no blanking between stimuli, but is
likely to be less of an issue when presenting stimuli for 2 s with 1 s blanking between stimuli. But
beyond this, Ahmed et al. (2021) report strong evidence that the subject did attend to the stimuli.


_To check whether the subject consistently viewed the images presented, online trial averaging_
_of the EEG data was performed in every session to obtain evoked responses that are phase-_
_locked_ _to_ _the_ _onset_ _of_ _the_ _images._ _Data_ _from_ _two_ _occipital_ _channels_ _(C31_ _and_ _C32)_ _were_
_bandpass filtered in the 1–40 Hz range and epochs of 800 ms duration were segmented out_
_synchronously_ _following_ _the_ _onset_ _of_ _each_ _image._ _Epochs_ _with_ _peak-to-trough_ _fluctuations_
_exceeding 100 µV were discarded and the remaining epochs were averaged together to yield_
_an_ _800_ _ms-long_ _evoked_ _response._ _A_ _clear_ _and_ _robust_ _N1-P2_ _onset_ _response_ _pattern_ _was_
_discernible_ _in_ _the_ _evoked_ _response_ _traces_ _obtained_ _in_ _each_ _of_ _the_ _100_ _runs,_ _consistent_ _with_
_the subject viewing the images as instructed._ _Note that all online averaging procedures (_ e.g. _,_
_filtering) were done to data in a separate buffer; the raw unprocessed data from 96 channels_
_was saved for offline analysis._


Ahmed et al. (2021)


Further evidence of subject attentiveness is that Ahmed et al. (2021) report statistically significant
classification accuracy as high as 7.3% and Bharadwaj et al. (2023) report statistically significant
classification accuracy as high as 17.6% on a task where chance performance is 2.5%. Given the
randomized nature of the design, this would not be possible if the subject did not attend to the
stimuli. Thus the concern raised by Palazzo et al. (2024) about the subject in Ahmed et al. (2021) as
to whether “the subject was even paying attention to the stimuli” is unfounded.


4 SESSION LENGTH


Palazzo et al. (2024) claim that the data collection underlying Spampinato et al. (2017), Kavasidis
et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021) and had sessions lasting about 4 minutes.


_In the data collection carried out by Bharadwaj et al._ _in [7] and also employed in [1],_ _the_
_authors state that a subject underwent stimuli exposition for over 20 minutes, instead of about_
_4 minutes in [3]._


Palazzo et al. (2024)


Similar claims are made six times in Palazzo et al. (2020b). However Spampinato et al. (2017,
Table 1), Kavasidis et al. (2017, Table 1), and Palazzo et al. (2017, Table 1) state that session
running time was 350 s, _i.e._, 5 minutes and 50 s. This is more-or-less consistent with the protocol
described in Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al. (2017) where each
session contained 10 blocks, each block contained 50 stimuli, each stimulus lasted 0.5 s, and blocks
were separated by 10 s blanking. Thus the claim in Palazzo et al. (2024) that the data collection in
Spampinato et al. (2017) took “about 4 minutes” is inaccurate.


2


5 CROSS-SUBJECT VARIABILITY


Palazzo et al. (2024) claim that Li et al. (2021) observe large subject-to-subject variability in classification accuracy.


_Even Bharadwaj et al._ _in [21] observe large subject-to-subject variability in their reported_
_results,_ _as_ _classification_ _performance_ _of_ _their_ _own_ _proposed_ _method_ _varies_ _from_ _37.80%_ _to_
_70.50% (Table 4 in [21], and Tables 21–25 in [21]’s appendix)._


Palazzo et al. (2024)


Li et al. (2021, Tables 4, 21–25) discuss block runs. The central claim of Li et al. (2021) is that
the block runs suffer from a temporal confound and thus one cannot draw any conclusions about
stimulus processing from these block runs. In contrast, to assess cross-subject variability in Li et al.
(2021), one needs to limit consideration to Li et al. (2021, Tables 5, 26–30) because these report
randomized trials on image stimuli and the full 96 channels with bandpass filtering. These tables
do not differ from chance in a statistically significant fashion. Thus the claim of Palazzo et al.
(2024) that Li et al. (2021) “observe large subject-to-subject variability in their reported results” is
misleading.


6 SINGLE SUBJECT


Palazzo et al. (2024) claim that the supertrial method of Bharadwaj et al. (2023) was applied to only
a single subject.


_A_ _recent_ _comments_ _paper_ _[1]_ _by_ _Bharadwaj_ _et_ _al._ _discusses_ _the_ _results_ _presented_ _in_ _[2],_
_claiming that the above-chance accuracy reported by that method is due to confounds in the_
_experimental_ _design_ _(from_ _[3])._ _In_ _order_ _to_ _support_ _that_ _claim,_ _Bharadwaj_ _et_ _al._ _propose_
_a_ _new_ _dataset_ _that_ _is,_ _according_ _to_ _them,_ _free_ _from_ _those_ _confounds._ _The_ _key_ _aspect_ _of_ _this_
_dataset is that samples — or,_ _as they call them,_ _“supertrials”,_ _borrowing terminology from_

_[4] — are obtained by averaging a set of trials collected during EEG recording for a single_
_subject._


Palazzo et al. (2024)


They further state:


_The dataset used by Bharadwaj et al., introduced in [7], is the result of EEG data collection_
_on one subject only._ _Single-subject analysis is critical mainly because EEG data are known_
_to be highly replicable within a person [14],_ _but also highly specific from person to person_

_[14], [20]._


Palazzo et al. (2024)


This, however, ignores the fact that Bharadwaj et al. (2023) report not only the results of a supertrial
analysis on the single-subject data from Ahmed et al. (2021), but also on the data from Li et al.
(2021) on six subjects.


_We_ _repeat_ _this_ _same_ _method_ _to_ _all_ _six_ _subjects_ _of_ _the_ _image_ _rapid_ _event_ _data_ _from_ _Li_ _et_ _al._

_[10] and replicate the study of Ahmed et al._ _[2, inline unnumbered table 9] with supertrials_
_instead of trials, with five-fold leave-one-portion-out cross validation._


Bharadwaj et al. (2023)


These results are reported in the right half of Bharadwaj et al. (2023, Table 1). Bharadwaj et al.
(2023) further state:


_Here, we form supertrials by aggregating trials from a single subject._ _One could form super-_
_trials by aggregating trials from multiple subjects._


Bharadwaj et al. (2023)


3


Bharadwaj et al. (2023) report results for a total of seven subjects: the left half of Bharadwaj et al.
(2023, Table 1) reports results on one subject and the right half reports results on six subjects. Thus
the claim of Palazzo et al. (2024) that “The dataset used by Bharadwaj et al., introduced in [7], is
the result of EEG data collection on one subject only” is false.


7 EFFECT OF SUPERTRIALS ON SIGNAL SPECTRUM


Palazzo et al. (2024) claim that the supertrial method of Bharadwaj et al. (2023) attenuates higherfrequency bands in the signal:


_Interestingly,_ _EEGNet_ _outperforms_ _EEGChannelNet_ _at_ _lower_ _frequency_ _bands,_ _while_ _our_
_approach_ _performs_ _better_ _at_ _higher_ _frequency_ _bands,_ _thus_ _confirming_ _the_ _findings_ _of_ _[2]._
_Thus, EEGChannelNet works better at higher frequencies._ _However, higher frequencies are_
_unavoidably attenuated by the supertrial method, proposed by [1]._ _Averaging trials acts as_
_a low pass filter (high frequencies rarely align temporally;_ _therefore phase differences lead_
_to_ _averaging_ _out_ _over_ _trials_ _[14])._ _Simply_ _put,_ _the_ _authors_ _explicitly_ _test_ _the_ _model_ _using_
_low_ _frequency_ _information,_ _which_ _we_ _previously_ _reported_ _to_ _reduce_ _classification_ _accuracy_
_(as_ _shown_ _in_ _[2],_ _low_ _frequency_ _classification_ _accuracy_ _of_ _EEGChannelNet_ _is_ _30_ _percent_
_lower_ _w.r.t._ _high_ _frequency_ _classification)._ _Supertrials_ _necessarily_ _result_ _in_ _the_ _averaging_
_out of information with inconsistent phase but significant power in a specific frequency band,_
_which still contains useful neural information [14]._


Palazzo et al. (2024)


and this penalizes EEGChannelNet.


_Additionally, their specific supertrial setup seems designed to penalize EEGChannelNet [2],_
_since_ _it_ _has_ _been_ _shown_ _to_ _exploit_ _high-frequency_ _information,_ _which_ _are_ _practically_ _sup-_
_pressed by sample averaging._


Palazzo et al. (2024)


Bharadwaj et al. (2023) state:


_Here,_ _we_ _aggregate_ _supertrials_ _by_ _unweighted_ _average_ _in_ _the_ _time_ _domain._ _One_ _could_ _av-_
_erage_ _in_ _the_ _frequency_ _domain,_ _potentially_ _considering_ _only_ _certain_ _bands_ _(_ e.g. _,_ _induced_
_responses), weighting some samples or bands more than others, or more generally averaging_
_some nonlinear transform, learned or hard-coded, of single trials._


Bharadwaj et al. (2023)


Now, we repeat the analyses of Bharadwaj et al. (2023) on the data from Ahmed et al. (2021),
constructing supertrials by averaging in the frequency domain. We do this by performing an FFT on
each sample, averaging the magnitude and phase of the samples independently, and performing an
inverse FFT on the average. This is done independently on each channel.


Fig. 1 plots the spectra for the raw trials and supertrials of various sizes _N_, averaged over (super)trial
and channel. It can be seen that this does not attenuate higher-frequency components. In fact, it
amplifies them.


We further repeat the analysis of Bharadwaj et al. (2023, Table 1 left) on the data from Ahmed
et al. (2021) using this supertrial averaging method (Table 1). EEGChannelNet is still at chance,
while SVM, 1D CNN, EEGNet, and SyncNet are still above chance for various size supertrials,
validating the original claim of Bharadwaj et al. (2023). Thus the claim by Palazzo et al. (2024)
that “Supertrials necessarily result in the averaging out of information with inconsistent phase but
significant power in a specific frequency band, which still contains useful neural information [14]”
is invalid.


Beyond this, Bharadwaj et al. (2023) did not develop the supertrial method; they simply employed
methods of Isik et al. (2014), Cichy et al. (2016), Greene & Hansen (2020), and Zheng et al. (2020a).


4


Figure 1: Spectra for the raw data from Ahmed et al. (2021) and various sizes of supertrials constructed by averaging in the frequency domain.


Table 1: Replication of the analysis from Bharadwaj et al. (2023, Table 1 left) for various sizes _N_ of
supertrials. Starred values indicate statistical significance above chance ( _p_ _<_ 0.005) by a binomial
cmf. Note that when _N_ gets larger, the number of test samples gets smaller, increasing quantization
noise in the accuracy estimates, thus requiring higher accuracy to achieve significance.


_N_ LSTM _k_ -NN SVM MLP 1D CNN EEGNet SyncNet EEGChannelNet
1 2.2% 2.1% 5.5% _[∗]_ 2.5% 5.5% _[∗]_ 7.1% _[∗]_ 2.5% 2.5%
2 2.5% 2.3% 5.4% _[∗]_ 2.4% 5.0% _[∗]_ 7.9% _[∗]_ 2.7% 2.5%
4 2.4% 2.5% 6.3% _[∗]_ 2.6% 6.9% _[∗]_ 8.7% _[∗]_ 3.7% _[∗]_ 2.5%
5 2.1% 2.4% 6.0% _[∗]_ 2.7% 7.5% _[∗]_ 7.0% _[∗]_ 3.2% _[∗]_ 2.4%
8 2.3% 2.4% 3.2% _[∗]_ 2.4% 5.9% _[∗]_ 9.5% _[∗]_ 3.4% _[∗]_ 2.4%
10 2.2% 2.1% 2.6% 2.4% 4.5% _[∗]_ 7.9% _[∗]_ 3.2% _[∗]_ 2.6%
20 1.5% 2.0% 2.4% 2.7% 2.3% 7.9% _[∗]_ 3.0% 2.9%
25 3.4% 2.1% 2.3% 2.3% 2.9% 3.5% 2.6% 2.6%
40 2.2% 2.7% 2.2% 2.3% 2.0% 2.6% 3.4% 1.7%
50 2.1% 3.0% 2.8% 2.5% 2.8% 3.1% 3.6% 2.4%
100 4.0% 1.5% 3.5% 3.3% 3.0% 5.3% _[∗]_ 2.8% 2.8%


Since this work all predates Bharadwaj et al. (2023), and some of this work even predates Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021), Bharadwaj et al. (2023) could not have designed the supertrial setup to penalize EEGChannelNet. Thus
the claim by Palazzo et al. (2024) that “their specific supertrial setup seems designed to penalize
EEGChannelNet [2]” is inaccurate.


.


5


8 CONFOUNDS


Palazzo et al. (2024) claim that interleaved-design experiments (aka randomized stimulus presentation order) introduce several confounds.


_On the contrary, interleaved-design experiments introduce several confounds that may sup-_
_press the very response that one would hope to classify with machine learning methods._


Palazzo et al. (2024)


It is not clear what “several confounds” refers to. Nonetheless, none of the concerns raised by
Palazzo et al. (2024) about Bharadwaj et al. (2023) and Ahmed et al. (2021) constitute confounds,
even if they were true. According to APA (2024), a confound is:


_in_ _an_ _experiment,_ _an_ _independent_ _variable_ _that_ _is_ _conceptually_ _distinct_ _but_ _empirically_ _in-_
_separable from one or more other independent variables._ _Confounding makes it impossible_
_to_ _differentiate_ _that_ _variable’s_ _effects_ _in_ _isolation_ _from_ _its_ _effects_ _in_ _conjunction_ _with_ _other_
_variables._


APA (2024)


Palazzo et al. (2024) misuse the term “confound”.


The protocol of Spampinato et al. (2017), Kavasidis et al. (2017), Palazzo et al. (2017; 2018;
2020a;b; 2021), and the block runs of Li et al. (2021) and Ahmed et al. (2022), does suffer from
a confound, namely, a correlation between stimulus class and time since the start of the run, essentially a clock embedded in the signal. As a result, it is impossible to determine whether the
classifier is classifying stimulus class or the embedded clock. This temporal confound excessively
_overestimates_ the classification accuracy. Even if they were true, the concerns raised by Palazzo
et al. (2024) about Bharadwaj et al. (2023) and Ahmed et al. (2021) only would reduce the quality of
the data and _underestimate_ the classification accuracy. Any potential limitations of the interleaveddesign experiments would not constitute “confounds.” Thus the claim by Palazzo et al. (2024) that
“interleaved-design experiments introduce several confounds” is false.


Palazzo et al. (2024) claim that the protocol of Spampinato et al. (2017), Kavasidis et al. (2017), and
Palazzo et al. (2017; 2018; 2020a;b; 2021) does not suffer from a confound.


_The claim that classification in block-design experiments mainly relies on temporal correla-_
_tions has already been addressed in [13], where we showed that:_

    - _Models_ _are_ _not_ _able_ _to_ _classify_ _samples_ _from_ _a_ _rapid-design_ _setup_ _when_ _block-level_
_labels are artificially assigned;_

    - _Samples collected during blank screens between two blocks are unlikely to be classified_
_as coming from the class before or after the blank screen._


Palazzo et al. (2024)


This line of reasoning exhibits a logical fallacy. According to Frost (2024):


_You can’t prove a negative!_ _[...]_ _If your test fails to detect an effect, it’s not proof that the_
_effect doesn’t exist._ _It just means your sample contained an insufficient amount of evidence_
_to conclude that it exists._


Frost (2024)


The presence of a confound in the protocol used by Spampinato et al. (2017), Kavasidis et al. (2017),
and Palazzo et al. (2017; 2018; 2020a;b; 2021) is clearly demonstrated by the incorrect block-level
labels experiment reported in Li et al. (2021, Tables 9 and 10) wherein it is shown that classifiers can
decode incorrect block-level class labels that are unrelated to the actual stimuli used to elicit EEG
response from trials with randomized stimulus presentation order.


Luck (2014) references twenty three discussions of confounds in the index. Among them, Luck
(2014, p. 133) states:


6


_Ignorance and Lack of Imagination When someone says, “I can’t imagine how that little con-_
_found could explain my results,” this is a case of a general logical fallacy that philosophers_
_call the argument from ignorance._ _In fact,_ _it’s a special case that is called (with a touch of_
_humor) the argument from lack of imagination._ _The fact that someone can’t imagine how a_
_confound_ _could_ _produce_ _a_ _particular_ _effect_ _might_ _just_ _mean_ _that_ _the_ _person_ _doesn’t_ _have_ _a_
_very good imagination!_ _I myself have occasionally used the “I can’t imagine how ...”_ _type_
_of_ _reasoning_ _and_ _then_ _found_ _that_ _I_ _was_ _suffering_ _from_ _a_ _lack_ _of_ _imagination_ _(see,_ e.g. _,_ _box_
_4.5)._ _But_ _now_ _that_ _I_ _realize_ _that_ _this_ _is_ _not_ _a_ _compelling_ _form_ _of_ _argument,_ _I_ _usually_ _catch_
_myself before I say it._


Luck (2014)


Palazzo et al. (2020b) (reference [13] in Palazzo et al. (2024)) offers two analyses in attempt to
support their claim of a lack of a temporal confound in the data of Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021). Palazzo et al. (2020b, Table 2)
report an analysis whereby models are trained on BDVE, the original data used by Spampinato et al.
(2017), Kavasidis et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021), and tested on BDB,
a dataset constructed from EEG collected when subjects viewed blank screens.


_The_ _neural_ _signals_ _recorded_ _between_ _each_ _pair_ _of_ _classes,_ i.e. _,_ _the_ **BDB** **dataset** _,_ _can_ _help_
_address this question._ _Since the neural data in response to the blank screen is equidistant in_
_time from two classes, a strong temporal correlation would result in significantly greater than_
_chance classification of that data as either the class before or the class after the blank screen._
_Thus,_ _we verify whether a model trained on the block-design_ **BDVE** _dataset would classify_
_blank screen segments as either the preceding or subsequent class. Finding near chance level_
_classification accuracy here would indicate little to no impact of a temporal correlation._ _To_
_assess the temporal correlation we assign two class labels to each blank segment in the BDB_
_dataset, corresponding to the preceding class and the following class._ _Then, for each of the_
_models_ _trained_ _on_ _the_ _BDVE_ _dataset_ _and_ _whose_ _results_ _are_ _given_ _in_ _Table_ _1,_ _we_ _compute_
_the_ _classification_ _accuracy_ _of_ _the_ _BDB_ _dataset_ _as_ _the_ _ratio_ _of_ _blank_ _segments_ _classified_ _as_
_either_ _one_ _of_ _the_ _corresponding_ _classes._ _Results_ _are_ _shown_ _in_ _Table_ _2,_ _and_ _reveal_ _that_ _all_
_methods are at or slightly above chance accuracy (_ i.e. _, 5%, since for each segment has two_
_possible_ _correct_ _options_ _out_ _of_ _the_ _40_ _classes)._ **This** **seems** **to** **be** **a** **clear** **indication** **that**
**temporal correlation in [2]’s data is minimal, suggesting that block design experiments**
**(when properly pre-processed) are suitable for classification studies.**


Palazzo et al. (2020b)
(Emphasis in the original highlighted in bold.)


First note that Palazzo et al. (2020b, Table 2) do indeed report finding a temporal confound in the
data of Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b;
2021). Second, this analysis does not accurately assess the temporal confound in the original results
in Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021),
as described below.


Li et al. (2021) discuss two kinds of temporal confound, one where the training and test sets come
from the same blocks of the same runs (Li et al., 2021, Table 6) and one where the training and test
sets comes from temporally correlated blocks of two different runs (Li et al., 2021, § 3.7, Table 15).
Note that the former has considerably higher accuracy than the latter, yet both are considerably
above chance. This suggests that there is a strong temporal correlation within the blocks of the same
run and a weaker, but still present, temporal correlation between temporally correlated blocks of
different runs.


The BDB analysis of Palazzo et al. (2020b) measures the latter, not the former. It is thus expected
that the temporal correlation will be less than that present in Spampinato et al. (2017), Kavasidis
et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021) which is of the former kind. Thus, the
claims in Palazzo et al. (2020b), that the “temporal correlation in [2]’s data is minimal” and “that
block design experiments (when properly pre-processed) are suitable for classification studies”, and
the claim in Palazzo et al. (2024), that “The claim that classification in block-design experiments
mainly relies on temporal correlations has already been addressed in [13]”, are unfounded.


7


Further, the training and test samples in Spampinato et al. (2017) and Palazzo et al. (2017; 2018;
2020a;b; 2021) which come from the same block of the same run, have a uniformly distributed
temporal distance between 0.5 s and 25 s whereas the test samples in BDB come from the blanking
periods, not the stimulus periods. The temporal distance between the blanking periods and the
corresponding stimulus periods varies uniformly between 25 s and 35 s. Palazzo et al. (2020b) state:


_The_ _data_ _from_ _these_ _blank_ _screens_ _are_ _particularly_ _significant_ _because,_ _as_ _claimed_ _in_ _[1],_
_any contribution of a temporal correlation to classification accuracy should persist through-_
_out the blank screen interval (i.e., the blank interval should be consistently classified above_
_chance as either the class before or after the blank screen)_


Palazzo et al. (2020b)


Li et al. (2021) never claim this and we have no reason to believe that this is the case. It is likely
that the temporal confound proceeds like a clock throughout the recording session. Palazzo et al.
(2020b; 2024) misunderstand the nature of the confound in Spampinato et al. (2017), Kavasidis
et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021) reported by Li et al. (2021), Ahmed
et al. (2021; 2022), and Bharadwaj et al. (2023). Thus the claim by Palazzo et al. (2024) that “any
contribution of a temporal correlation to classification accuracy should persist throughout the blank
screen interval (i.e., the blank interval should be consistently classified above chance as either the
class before or after the blank screen)” is also not supported by the data.


Palazzo et al. (2020b, Table 4) report a second analysis, that replicates the analysis in Li et al.
(2021, Table 9), whereby models are trained on BDVE, and tested on RDVE, a dataset collected
with randomized trials (with half the samples per class than the datasets in either Li et al. 2021 or
Spampinato et al. 2017, Kavasidis et al. 2017, and Palazzo et al. 2017; 2018; 2020a;b; 2021), but
where the actual class labels are replaced with incorrect block-level labels. First note that Palazzo
et al. (2020b, Table 4) do indeed report finding a temporal confound in the data of Spampinato et al.
(2017), Kavasidis et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b; 2021).


_The classification accuracy, when using rapid-design data with incorrect block-level labels,_
_is at most 9 percent points above chance, suggesting that the rapid design carries some small_
_temporal correlations._


Palazzo et al. (2020b)


Many factors could contribute to observing a smaller effect than that observed by Li et al. (2021),
among them the fact that RDVE has half the samples per class. Thus the statement “at most 9
percent points above chance” is misleading when used to validate the use of data from Spampinato
et al. (2017) and the results from Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al.
(2017; 2018; 2020a;b; 2021).


Finally, Palazzo et al. (2024) state:


_In [13], we further elucidate that the single-subject analysis is problematic, by demonstrat-_
_ing_ _that_ _pooling_ _data_ _across_ _subjects_ _accounts_ _for_ _inter-subject_ _variability_ _by_ _reducing_ _the_
_subject-specific_ _representation_ _on_ _the_ _classifier._ _We_ _show_ _that_ _the_ _per-subject_ _variability_
_(measured in terms of standard deviation) decreases significantly when a classifier is trained_
_using_ _multiple_ _subjects’_ _data._ _Furthermore,_ _this_ _allows_ _the_ _model_ _to_ _focus_ _on_ _inter-subject_
_discriminative_ _features,_ _reducing_ _the_ _bias_ _due_ _to_ _possible_ _temporal_ _correlations_ _that_ _may_
_exist_ _in_ _a_ _single_ _subject’s_ _neural_ _responses._ _Thus,_ _the_ _large_ _inter-subject_ _differences_ _must_
_be overcome for any viable classification method._ _Importantly, averaged event-related data_
_from a random sample of about 10 subjects tends to look highly similar to another random_
_sample_ _of_ _10_ _subjects_ _[22],_ _[14]._ _Failure_ _to_ _pool_ _data_ _across_ _subjects_ _would,_ _again,_ _only_
_serve to increase the impact of any temporal correlation._


Palazzo et al. (2024)


We have no reason to believe that the temporal correlation proceeds at the same rate in different subjects. Li et al. (2021, Table 8) assess this via a leave-one-subject-out analysis on the data
from Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al. (2017; 2018; 2020a;b;
2021). The precipitous drop in classification accuracy from that reported by Spampinato et al.


8


(2017) and Palazzo et al. (2017; 2018; 2020a;b; 2021), while still “pooling training data across subjects,” strongly suggests that the high accuracy reported by Spampinato et al. (2017) and Palazzo
et al. (2017; 2018; 2020a;b; 2021) results from within-subject within-run temporal correlations that
are absent across subjects. Thus the claim in Palazzo et al. (2024) “that pooling data across subjects
accounts for inter-subject variability by reducing the subject-specific representation on the classifier”
is unfounded.


We know of no successful results on performing cross-subject classification of EEG recordings from
stimuli similar to those used in Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al.
(2017; 2018; 2020a;b; 2021) that do not suffer from confounds. EEG data collection is resource
limited. One can spend that resource collecting a smaller amount of data from multiple subjects
or a larger amount of data from a single subject. Ahmed et al. (2021) decided to do the latter as
cross-subject classification is infeasible at the current time and the intent was to assess the bounds of
classification accuracy with a feasible data collection effort. The data collection from Ahmed et al.
(2021) and Bharadwaj et al. (2023) was the largest known nonconfounded EEG dataset from stimuli
similar to those used in Spampinato et al. (2017), Kavasidis et al. (2017), and Palazzo et al. (2017;
2018; 2020a;b; 2021) at the time of publication. Moreover, the classification accuracies were the
highest known for nonconfounded data of that type at the time of publication. To our knowledge,
both of these are still the case.


9 CONCLUSION


The key claims in Bharadwaj et al. (2023) are stated in the conclusion.


_Palazzo et al._ _[14] claim that the data collected in Li et al._ _[10] lacks class information due_
_to lack of subject attentiveness during long sessions, and that classification failure is based_
_on_ _this._ _[...]_ _Table_ _I_ _demonstrates_ _that_ _the_ _data_ _of_ _Ahmed_ _et_ _al._ _[1]_ _and_ _Li_ _et_ _al._ _[10]_ _do_
_contain class information; it is just that some classifiers successfully extract it and some do_
_not._ _Thus our results here refute their claim._ _Table I further demonstrates that:_

    - _With_ _and_ _without_ _supertrials,_ _EEGChannelNet_ _yields_ _chance_ _accuracy_ _on_ _a_ _noncon-_
_founded dataset 20× larger than that of [15]._

    - _For some amounts of supertrial aggregation, EEGNet and SyncNet yield above chance_
_accuracy._
_This refutes the claim in [15] that EEGChannelNet outperforms EEGNet and SyncNet. More-_
_over, to the best of our knowledge, the classification accuracy of 17.5% obtained by EEGNet_
_with_ _N_ = 20 _is_ _the_ _highest_ _reported_ _for_ _a_ _40-class_ _EEG_ _classification_ _task_ _on_ _ImageNet_
_stimuli._ _Finally,_ _this_ _demonstrates_ _that_ _the_ _datasets_ _of_ _Ahmed_ _et_ _al._ _[1]_ _and_ _Li_ _et_ _al._ _[10]_
_do contain class information in the EEG signal; EEGNet, to some extent, and SyncNet, to a_
_lessor extent, can extract that class information._ _EEGChannelNet cannot._


Bharadwaj et al. (2023)


Nothing in Palazzo et al. (2024) refutes that claim.


AUTHOR CONTRIBUTIONS


Removed for blind review.


ACKNOWLEDGMENTS


Removed for blind review.


ETHICS STATEMENT


This work debunks nearly one hundred published papers whose results are based on the same confound: a correlation between stimulus class and temporal drift. This confound has been found in
eighteen available EEG datasets. Just as with an inconsistent set of axioms one can prove anything, a
confounded dataset can be used to support any claim, even ones that are false or absurd. That is what
many recent publications based on this confound do: things like generating high fidelity renderings
of images, or even 3D CAD models of objects, from EEG recordings.


9


A research community, knowingly or unknowingly, has discovered that one can use confounded
datasets to churn out a plethora of flawed results without reviewers noticing. They have also discovered that one can collect new confounded datasets to churn out even more flawed results without
reviewers noticing. The temptation to do this is so strong that the community continues to do so four
years after details of the confound were published.


It is conceivable that the flaws in these datasets may be a driving factor behind their frequent reuse.
When a dataset is severely confounded, it becomes relatively easy to achieve an extremely high
accuracy, which can in turn be used to support sensational claims, and ultimately directs further
attention to the dataset. In business, this phenomenon is referred to as “the bad money drives out the
good money.”


More prominent exposure of these flawed methods and consequent false results will allow resources
wasted on continued use of these confounded datasets and flawed methods to be reallocated. The
debunked work also causes direct ongoing harm:


    - grant proposals can be rejected due to preliminary results not being competitive with results
demonstrating falsely-inflated performance based on confounded data or faulty methods;

    - manuscripts can be rejected for the same reason;

    - grants can be awarded based on false pretenses

    - manuscripts can be accepted for the same reason;

    - degrees can be awarded for the same reason;

    - resources can be wasted attempting to replicate the debunked results;

    - resources can be wasted having people read and review flawed papers, and learn flawed
methods; and

    - because the debunked work relates to brain-computer interfaces—whose primary application is helping people with disabilities ( _e.g._, paralysis) interact with the world—the harm
caused is not merely scientific but also medical, with disproportionate impact on people
with disabilities.


This work is significant for the following reasons:


    - Nearly one hundred papers (An & Cho, 2016; Spampinato et al., 2016; Ben Said et al.,
2017; Bozal Chaves, 2017; Kavasidis et al., 2017; Palazzo et al., 2017; Parekh et al., 2017;
Spampinato et al., 2017; Zhang et al., 2017; Du et al., 2018; Fares et al., 2018; Kumar et al.,
2018; Palazzo et al., 2018; Piplani et al., 2018; Tirupattur et al., 2018; Wang et al., 2018;
Zhang & Liu, 2018; Zhang et al., 2018; Zhong et al., 2018; Du et al., 2019; Hwang et al.,
2019; Jiang et al., 2019; Jiao et al., 2019; Long et al., 2019; Mukherjee et al., 2019; Uys,
2019; Wang et al., 2019; Cudlenco et al., 2020; Fares et al., 2020; Li et al., 2020; Palazzo
et al., 2020a;b; Wang et al., 2020; Zheng et al., 2020b;c; Palazzo et al., 2021; Zheng &
Chen, 2021; Ma et al., 2021; Mo et al., 2021; Jiang et al., 2021; Lee et al., 2021; Cavazza
et al., 2022; Khaleghi et al., 2022; Lee et al., 2022; Mishra et al., 2022; Mishra, 2022;
Scharnagl & Groth, 2022; Shimizu & Srinivasan, 2022; Ahmadieh et al., 2023; Bai et al.,
2023; Du et al., 2023; Duan et al., 2023; Hasan & A, 2023; Imani et al., 2023; Lan et al.,
2023; Lee et al., 2023; Liu et al., 2023; Singh et al., 2023; Song et al., 2023; Wahengbam
et al., 2023; Zeng et al., 2023b;a; Fan et al., 2024; Ferrante et al., 2024a;b; Gou et al.,
2024; Lei et al., 2024; Liu et al., 2024a;b; Luvsansambuu et al., 2024; Mishra et al., 2024;
Mwata-Velu et al., 2024; Ngo et al., 2024; Palazzo et al., 2024; Pan et al., 2024; Qian et al.,
2024; Singh et al., 2024; Tang et al., 2024; de la Torre-Ortiz et al., 2024; Yang & Liu, 2024;
Ye et al., 2024; Zheng et al., 2024b;a; Zhu et al., 2024; Deng et al., 2025; Fares, 2025; Fu
et al., 2025; Lopez et al., 2025; Mehmood et al., 2025; Singh et al., 2025; Xiang et al.,
2025) draw flawed conclusions based on the confounded dataset from Spampinato et al.
(2017) and datasets suffering from the same confound.

    - A number of new datasets have been collected with this same confounded protocol (Gou
et al., 2024; Pan et al., 2024; Zhu et al., 2024; Qian et al., 2024; Uys, 2019; Shimizu &
Srinivasan, 2022; Liu et al., 2024b; Wang et al., 2019; 2020; Ma et al., 2021; Cudlenco
et al., 2020; Zheng et al., 2024b; Cavazza et al., 2022; Luvsansambuu et al., 2024; Liu
et al., 2023; Bai et al., 2023; Parekh et al., 2017).

    - A number of these have been publicly released and are used by others. For example, Singh
et al. (2023), Singh et al. (2024), and Lopez et al. (2025) use the dataset reported in Kumar


10


et al. (2018) and Duan et al. (2023), Singh et al. (2024), and Lopez et al. (2025) use the
dataset reported in Ma et al. (2021).

    - This is further egregious because Palazzo et al. (2020b; 2024) continue to claim that their
dataset (Spampinato et al., 2017), and their results that were obtained with that dataset
(Spampinato et al., 2017; Kavasidis et al., 2017; Palazzo et al., 2017; 2018; 2020a;b; 2021;
2024), are valid, despite the refutations in Li et al. (2021), Ahmed et al. (2021; 2022), and
Bharadwaj et al. (2023), in part, because of the arguments in Palazzo et al. (2024).

    - This has been used to justify continued publication of a large and growing body of flawed
work based on confounded datasets (Cavazza et al., 2022; Khaleghi et al., 2022; Lee et al.,
2022; Mishra et al., 2022; Mishra, 2022; Scharnagl & Groth, 2022; Shimizu & Srinivasan,
2022; Ahmadieh et al., 2023; Bai et al., 2023; Du et al., 2023; Duan et al., 2023; Hasan &
A, 2023; Imani et al., 2023; Lan et al., 2023; Lee et al., 2023; Liu et al., 2023; Singh et al.,
2023; Song et al., 2023; Wahengbam et al., 2023; Zeng et al., 2023b;a; Fan et al., 2024;
Ferrante et al., 2024a;b; Gou et al., 2024; Lei et al., 2024; Liu et al., 2024a;b; Luvsansambuu et al., 2024; Mishra et al., 2024; Mwata-Velu et al., 2024; Ngo et al., 2024; Palazzo
et al., 2024; Pan et al., 2024; Qian et al., 2024; Singh et al., 2024; Tang et al., 2024; de la
Torre-Ortiz et al., 2024; Yang & Liu, 2024; Ye et al., 2024; Zheng et al., 2024b;a; Zhu et al.,
2024; Deng et al., 2025; Fares, 2025; Fu et al., 2025; Lopez et al., 2025; Mehmood et al.,
2025; Singh et al., 2025; Xiang et al., 2025) even after the confound became known through
the work of Li et al. (2021), Ahmed et al. (2021; 2022), and Bharadwaj et al. (2023).


Current machine-learning conferences, and more generally, computer-science conferences and journals, are loathe to publish refutations. Observing this, Schaeffer et al. (2025) proposed that the field
of machine-learning establish a “refutations and critiques” track in prominent conferences. While
we applaud and support this proposal, the current lack of such a track should not be an impediment
to publishing refutations. Scientific journals in other fields have long done so, often resulting in
retraction of flawed work. Schaeffer et al. (2025) offer five example pieces of claimed flawed work
in machine learning. Each is an individual paper. These pale in comparison to the flaws we uncover
here: a systemic flaw of the entire peer review process across an entire field of inquiry, namely classification of stimulus image class from EEG recordings, that affects seventeen datasets and ninety
one papers. Moreover, none of the five examples in Schaeffer et al. (2025) are egregious; here the
authors of the flawed work continue to argue for its validity despite four refereed refutations and fifty
new flawed papers have been published subsequent to these four refereed refutations. This argues
for the need to make the community aware of the severity of the issue.


REPRODUCIBILITY STATEMENT


The raw data that produced these results is available at [https://dx.doi.org/10.21227/](https://dx.doi.org/10.21227/bc7e-6j47)
[bc7e-6j47.](https://dx.doi.org/10.21227/bc7e-6j47) Our code, which will be released upon publication, is built on top of the code in
[https://dx.doi.org/10.21227/bc7e-6j47.](https://dx.doi.org/10.21227/bc7e-6j47)


REFERENCES


Hajar Ahmadieh, Farnaz Gassemi, and Mohammad Hasan Moradi. A hybrid deep learning framework for automated visual image classification using EEG signals. _Neural Computing and Appli-_
_cations_, pp. 1–17, 2023.


Hamad Ahmed, Ronnie B. Wilbur, Hari M. Bharadwaj, and Jeffrey Mark Siskind. Object classification from randomized EEG trials. In _Computer Vision and Pattern Recognition_, pp. 3845–3854,
2021.


Hamad Ahmed, Ronnie B. Wilbur, Hari M. Bharadwaj, and Jeffrey Mark Siskind. Confounds in the
data—Comments on “Decoding brain representations by multimodal learning of neural activity
and visual features”. _Transactions on Pattern Analysis and Machine Intelligence_, 44(12):9217–
9220, 2022.


Jinwon An and Sungzoon Cho. Hand motion identification of grasp-and-lift task from electroencephalography recordings using recurrent neural networks. In _International_ _Conference_ _on_ _Big_
_Data and Smart Computing_, pp. 427–429, 2016.


APA. Dictionary of psychology, 2024.


11


Yunpeng Bai, Xintao Wang, Yan-pei Cao, Yixiao Ge, Chun Yuan, and Ying Shan. Dreamdiffusion:
Generating high-quality images from brain EEG signals. _arXiv_, 2306.16934, 2023.


Ahmed Ben Said, Amr Mohamed, Tarek Elfouly, Khaled Harras, and Z. Jane Wang. Multimodal
deep learning approach for joint EEG-EMG data compression and classification. In _Wireless_
_Communications and Networking Conference_, 2017.


Hari M. Bharadwaj, Ronnie B. Wilbur, and Jeffrey Mark Siskind. Still an ineffective method with
supertrials/ERPs—Comments on “Decoding brain representations by multimodal learning of neural activity and visual features”. _Transactions on Pattern Analysis and Machine Intelligence_, 45
(11):14052–14054, 2023.


Alberto Bozal Chaves. Personalized image classification from EEG signals using deep learning.
B.S. thesis, Universitat Polit`ecnica de Catalunya, 2017.


Jacopo Cavazza, Waqar Ahmed, Riccardo Volpi, Pietro Morerio, Francesco Bossi, Cesco Willemse,
Agnieszka Wykowska, and Vittorio Murino. Understanding action concepts from videos and
brain activity through subjects’ consensus. _Scientific Reports_, 12(1):19073, 2022.


Radoslaw Martin Cichy, Aditya Khosla, Dimitrios Pantazis, Antonio Torralba, and Aude Oliva.
Comparison of deep neural networks to spatio-temporal cortical dynamics of human visual object
recognition reveals hierarchical correspondence. _Scientific reports_, 6(1):1–13, 2016.


Nicolae Cudlenco, Nirvana Popescu, and Marius Leordeanu. Reading into the mind’s eye: Boosting
automatic visual recognition with EEG signals. _Neurocomputing_, 386:281–292, 2020.


Carlos de la Torre-Ortiz, Michiel M. Spap´e, Niklas Ravaja, and Tuukka Ruotsalo. Cross-subject
EEG feedback for implicit image generation. _Transactions_ _on_ _Cybernetics_, 54(10):6105–6117,
2024.


Xia Deng, Shen Chen, Jiale Zhou, and Lei Li. Mind2matter: Creating 3D models from EEG signals.
_arXiv_, 2504.11936, 2025.


Changde Du, Changying Du, and Huiguang He. Doubly semi-supervised multimodal adversarial
learning for classification, generation and retrieval. In _International Conference on Multimedia_,
pp. 13–18, 2019.


Changde Du, Kaicheng Fu, Jinpeng Li, and Huiguang He. Decoding visual neural representations
by multimodal learning of brain-visual-linguistic features. _Transactions on Pattern Analysis and_
_Machine Intelligence_, 45(9):10760–10777, 2023.


Changying Du, Changde Du, Xingyu Xie, Chen Zhang, and Hao Wang. Multi-view adversarially
learned inference for cross-domain joint distribution matching. In _International_ _Conference_ _on_
_Knowledge Discovery & Data Mining_, pp. 1348–1357, 2018.


Yiping Duan, Shuzhan Hu, Xin Ma, and Xiaoming Tao. Multi-class image generation from EEG
features with conditional generative adversarial networks. In _International Conference on Wire-_
_less Communications and Signal Processing_, pp. 534–539, 2023.


Xiaoya Fan, Yuntao Liu, and Zhong Wang. Electroencephalogram helps few-shot learning. In
_International Conference on Acoustics, Speech and Signal Processing_, pp. 8015–8019, 2024.


Ahmed Fares. A novel spatiotemporal framework for EEG-based visual image classification through
signal disambiguation. _Applied System Innovation_, 8(5):121, 2025.


Ahmed Fares, Shenghua Zhong, and Jianmin Jiang. Region level bi-directional deep learning framework for EEG-based image classification. In _International_ _Conference_ _on_ _Bioinformatics_ _and_
_Biomedicine_, pp. 368–373, 2018.


Ahmed Fares, Sheng-hua Zhong, and Jianmin Jiang. Brain-media: A dual conditioned and lateralization supported GAN (DCLS-GAN) towards visualization of image-evoked brain activities. In
_International Conference on Multimedia_, pp. 1764–1772, 2020.


12


Matteo Ferrante, Tommaso Boccato, Stefano Bargione, and Nicola Toschi. Decoding EEG signals
of visual brain representations with a CLIP based knowledge distillation. In _Learning from Time_
_Series For Health_, 2024a.


Matteo Ferrante, Tommaso Boccato, Stefano Bargione, and Nicola Toschi. Decoding visual brain
representations from electroencephalography through knowledge distillation and latent diffusion
models. _Computers in Biology and Medicine_, 178:108701, 2024b.


Jim Frost. Statistics by Jim: Failing to reject the null hypothesis, 2024.


Honghao Fu, Hao Wang, Jing Jih Chin, and Zhiqi Shen. BrainVis: Exploring the bridge between
brain and visual signals via image reconstruction. In _International_ _Conference_ _on_ _Acoustics,_
_Speech and Signal Processing_, pp. 1–5, 2025.


Mingyu Gou, Ying-Jie Zhang, Ren-Jie Dai, Hao-Long Yin, Tianzhen Chen, Fei Cheng, Bao-Liang
Lu, Jiang Du, and Wei-Long Zheng. Addressing temporal and auditory factors in meditative EEG
with self-supervised learning. In _International_ _Conference_ _on_ _Bioinformatics_ _and_ _Biomedicine_,
pp. 1954–1959, 2024.


Michelle R Greene and Bruce C Hansen. Disentangling the independent contributions of visual and
conceptual features to the spatiotemporal dynamics of scene categorization. _Journal_ _of_ _Neuro-_
_science_, 40(27):5283–5299, 2020.


Haitham S Hasan and Al-Sharqi Mais A. EEG-based image classification using an efficient geometric deep network based on functional connectivity. _Periodicals_ _of_ _Engineering_ _and_ _Natural_
_Sciences_, 11(1):208–215, 2023.


Sunhee Hwang, Kibeom Hong, Guiyoung Son, and Hyeran Byun. EZSL-GAN: EEG-based zeroshot learning approach using a generative adversarial network. In _International Winter Conference_
_on Brain-Computer Interface_, pp. 1–4, 2019.


Zahra Imani, Mehdi Ezoji, and Timoth´ee Masquelier. Brain-guided manifold transferring to improve
the performance of spiking neural networks in image classification. _Journal_ _of_ _Computational_
_Neuroscience_, 51(4):475–490, 2023.


Leyla Isik, Ethan M Meyers, Joel Z Leibo, and Tomaso Poggio. The dynamics of invariant object
recognition in the human visual system. _Journal of neurophysiology_, 111(1):91–102, 2014.


Jianmin Jiang, Ahmed Fares, and Sheng-Hua Zhong. A context-supported deep learning framework
for multimodal brain imaging classification. _Transactions_ _on_ _Human-Machine_ _Systems_, 49(6):
611–622, 2019.


Jianmin Jiang, Ahmed Fares, and Sheng-Hua Zhong. A brain-media deep framework towards seeing
imaginations inside brains. _Transactions on Multimedia_, 23:1454–1465, 2021.


Zhicheng Jiao, Haoxuan You, Fan Yang, Xin Li, Han Zhang, and Dinggang Shen. Decoding EEG by
visual-guided deep neural networks. In _International Joint Conference on Artificial Intelligence_,
2019.


Isaak Kavasidis, Simone Palazzo, Concetto Spampinato, Daniela Giordano, and Mubarak Shah.
Brain2Image: Converting brain signals into images. In _International Conference on Multimedia_,
pp. 1809–1817, 2017.


Nastaran Khaleghi, Tohid Yousefi Rezaii, Soosan Beheshti, Saeed Meshgini, Sobhan Sheykhivand,
and Sebelan Danishvar. Visual saliency and image reconstruction from EEG signals via an effective geometric deep network-based generative adversarial network. _Electronics_, 11(21):3637,
2022.


Pradeep Kumar, Rajkumar Saini, Partha Pratim Roy, Pawan Kumar Sahu, and Debi Prosad Dogra.
Envisioned speech recognition using EEG sensors. _Personal and Ubiquitous Computing_, 22(1):
185–199, 2018.


13


Yu-Ting Lan, Kan Ren, Yansen Wang, Wei-Long Zheng, Dongsheng Li, Bao-Liang Lu, and Lili Qiu.
Seeing through the brain: image reconstruction of visual perception from human brain signals.
_arXiv_, 2308.02510, 2023.


Pilhyeon Lee, Sunhee Hwang, Seogkyu Jeon, and Hyeran Byun. Subject adaptive EEG-based visual
recognition. In _Asian Conference on Pattern Recognition_, pp. 322–334, 2021.


Pilhyeon Lee, Sunhee Hwang, Jewook Lee, Minjung Shin, Seogkyu Jeon, and Hyeran Byun. Intersubject contrastive learning for subject adaptive EEG-based visual recognition. In _International_
_Winter Conference on Brain-Computer Interface_, pp. 1–6, 2022.


Pilhyeon Lee, Seogkyu Jeon, Sunhee Hwang, Minjung Shin, and Hyeran Byun. Source-free subject adaptation for EEG-based visual recognition. In _International Winter Conference on Brain-_
_Computer Interface_, pp. 1–6, 2023.


Weixian Lei, Yixiao Ge, Kun Yi, Jianfeng Zhang, Difei Gao, Dylan Sun, Yuying Ge, Ying Shan,
and Mike Zheng Shou. VIT-LENS: Towards omni-modal representations. In _Computer_ _Vision_
_and Pattern Recognition_, pp. 26637–26647, 2024.


Dan Li, Changde Du, and Huiguang He. Semi-supervised cross-modal image generation with generative adversarial networks. _Pattern Recognition_, 100, 2020.


Ren Li, Jared S. Johansen, Hamad Ahmed, Thomas V. Ilyevsky, Ronnie B. Wilbur, Hari M. Bharadwaj, and Jeffrey Mark Siskind. The perils and pitfalls of block design for EEG classification
experiments. _Transactions on Pattern Analysis and Machine Intelligence_, 43(1):316–333, 2021.


Dongjun Liu, Weichen Dai, Hangkui Zhang, Xuanyu Jin, Jianting Cao, and Wanzeng Kong. Brainmachine coupled learning method for facial emotion recognition. _Transactions on Pattern Anal-_
_ysis and Machine Intelligence_, 45(9):10703–10717, 2023.


Dongjun Liu, Jin Cui, Zeyu Pan, Hangkui Zhang, Jianting Cao, and Wanzeng Kong. Machine to
brain: facial expression recognition using brain machine generative adversarial networks. _Cogni-_
_tive Neurodynamics_, 18(3):863–875, 2024a.


Xuan-Hao Liu, Yan-Kai Liu, Yansen Wang, Kan Ren, Hanwen Shi, Zilong Wang, Dongsheng Li,
Bao-Liang Lu, and Wei-Long Zheng. EEG2video: Towards decoding dynamic visual perception
from EEG signals. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, pp. 72245–72273,
2024b.


Yanfang Long, Wanzeng Kong, Xuanyu Jin, Jili Shang, and Can Yang. Visualizing emotional states:
A method based on human brain activity. In _Human_ _Brain_ _and_ _Artificial_ _Intelligence_, pp. 248–
258, 2019.


Eleonora Lopez, Luigi Sigillo, Federica Colonnese, Massimo Panella, and Danilo Comminiello.
Guess what I think: Streamlined EEG-to-image generation with latent diffusion models. In _Inter-_
_national Conference on Acoustics, Speech and Signal Processing_, pp. 1–5, 2025.


SJ Luck. _An introduction to the event-related potential technique_ . MIT press, 2 edition, 2014.


Uurtsaikh Luvsansambuu, Tengis Tserendondog, Munkbayar Bat-Erdene, and Batmunkh Amar. A
deep learning model for classifying the thoughts of multiple individuals based on visual event. In
_International Conference on Electrical, Computer and Energy Technologies_, pp. 1–6, 2024.


Xin Ma, Yiping Duan, Shuzhan Hu, Xiaoming Tao, and Ning Ge. EEG based visual classification
with multi-feature joint learning. In _International Conference on Image Processing_, pp. 264–268,
2021.


Tariq Mehmood, Hamza Ahmad, Muhammad Haroon Shakeel, and Murtaza Taj. CATVis: Contextaware thought visualization. _arXiv_, 2507.11522, 2025.


Abhijit Mishra, Shreya Shukla, Jose Torres, Jacek Gwizdka, and Shounak Roychowdhury.
Thought2text: Text generation from EEG signal using large language models (llms). _arXiv_,
2410.07507, 2024.


14


Alankrit Mishra. Enhancing machine vision using human cognition from EEG analysis. Master’s
thesis, Lakehead University, 2022.


Alankrit Mishra, Nikhil Raj, and Garima Bajwa. EEG-based image feature extraction for visual
classification using deep learning. In _International Conference on Intelligent Data Science Tech-_
_nologies and Applications_, pp. 181–188, 2022.


Liangyan Mo, Yuhan Wang, Wenhui Zhou, Xingfa Shen, and Wanzeng Kong. A bi-LSTM based
network with attention mechanism for EEG visual classification. In _International Conference on_
_Unmanned Systems_, pp. 858–863, 2021.


Pranay Mukherjee, Abhirup Das, Ayan Kumar Bhunia, and Partha Pratim Roy. Cogni-Net: Cognitive feature learning through deep visual perception. In _International_ _Conference_ _on_ _Image_
_Processing_, pp. 4539–4543, 2019.


Tat’y Mwata-Velu, Erik Zamora, Juan Irving Vasquez-Gomez, Jose Ruiz-Pinales, and Humberto
Sossa. Multiclass classification of visual electroencephalogram based on channel selection, minimum norm estimation algorithm, and deep network architectures. _Sensors_, 24(12):3968, 2024.


Huyen Ngo, Khoi Do, Duong Nguyen, Viet Dung Nguyen, and Lan Dang. How homogenizing the
channel-wise magnitude can enhance EEG classification model? _arXiv_, 2407.20247, 2024.


Simone Palazzo, Concetto Spampinato, Isaak Kavasidis, Daniela Giordano, and Mubarak Shah.
Generative adversarial networks conditioned by brain signals. In _International_ _Conference_ _on_
_Computer Vision_, pp. 3410–3418, 2017.


Simone Palazzo, Concetto Spampinato, Isaak Kavasidis, Daniela Giordano, and Mubarak Shah.
Decoding brain representations by multimodal learning of neural activity and visual features.
_arXiv_, 1820.10974v1, 2018.


Simone Palazzo, Francesco Rundo, Sebastiano Battiato, Daniela Giordano, and Concetto Spampinato. Visual saliency detection guided by neural signals. In _International Conference on Auto-_
_matic Face and Gesture Recognition_, pp. 434–440, 2020a.


Simone Palazzo, Concetto Spampinato, Joseph Schmidt, Isaak Kavasidis, Daniela Giordano, and
Mubarak Shah. Correct block-design experiments mitigate temporal correlation bias in EEG
classification. _arXiv_, 2012.03849, 2020b.


Simone Palazzo, Concetto Spampinato, Isaak Kavasidis, Daniela Giordano, Joseph Schmidt, and
Mubarak Shah. Decoding brain representations by multimodal learning of neural activity and
visual features. _Transactions on Pattern Analysis and Machine Intelligence_, 43(11):3833–3849,
2021.


Simone Palazzo, Concetto Spampinato, Isaak Kavasidis, Daniela Giordano, Joseph Schmidt, and
Mubarak Shah. Rebuttal to “Comments on ‘Decoding brain representations by multimodal learning of neural activity and visual features’ ”. _Transactions on Pattern Analysis and Machine Intel-_
_ligence_, 46(12):11540–11542, 2024.


Hongguang Pan, Zhuoyi Li, Yunpeng Fu, Xuebin Qin, and Jianchen Hu. Reconstructing visual stimulus representation from EEG signals based on deep visual representation model. _Transactions_
_on Human-Machine Systems_, 54(6):711–722, 2024.


Viral Parekh, Ramanathan Subramanian, Dipanjan Roy, and CV Jawahar. An EEG-based image
annotation system. In _National_ _Conference_ _on_ _Computer_ _Vision,_ _Pattern_ _Recognition,_ _Image_
_Processing, and Graphics_, pp. 303–313, 2017.


Tanya Piplani, Nick Merill, and John Chuang. Faking it, making it: Fooling and improving brainbased authentication with generative adversarial networks. In _International Conference on Bio-_
_metrics Theory, Applications and Systems_, 2018.


Dongguan Qian, Hong Zeng, Wenjie Cheng, Yu Liu, Taha Bikki, and Jianjiang Pan. NeuroDM:
Decoding and visualizing human brain activity with EEG-guided diffusion model. _Computer_
_Methods and Programs in Biomedicine_, 251:108213, 2024.


15


Rylan Schaeffer, Joshua Kazdan, Yegor Denisov-Blanch, Brando Miranda, Matthias Gerstgrasser,
Susan Zhang, Andreas Haupt, Isha Gupta, Elyas Obbad, Jesse Dodge, Jessica Zosa Forde,
Francesco Orabona, Sanmi Koyejo, and David Donoho. Position: Machine learning conferences
should establish a “refutations and critiques” track. _arXiv_, 2506.19882, 2025.


Bastian Scharnagl and Christian Groth. Evaluation of different deep learning approaches for EEG
classification. In _International_ _Conference_ _on_ _Artificial_ _Intelligence_ _for_ _Industries_, pp. 42–47,
2022.


Hirokatsu Shimizu and Ramesh Srinivasan. Improving classification and reconstruction of imagined
images from EEG signals. _Plos one_, 17(9):e0274847, 2022.


Prajwal Singh, Pankaj Pandey, Krishna Miyapuram, and Shanmuganathan Raman. EEG2IMAGE:
Image reconstruction from EEG brain signals. In _International Conference on Acoustics, Speech_
_and Signal Processing_, pp. 1–5, 2023.


Prajwal Singh, Dwip Dalal, Gautam Vashishtha, Krishna Miyapuram, and Shanmuganathan Raman.
Learning robust deep visual representations from EEG brain recordings. In _Winter Conference on_
_Applications of Computer Vision_, pp. 7538–7547, 2024.


Pushapdeep Singh, Jyoti Nigam, Medicherla Vamsi Krishna, Arnav Bhavsar, and Aditya Nigam.
EAD: An EEG adapter for automated classification. _arXiv_, 2505.23107, 2025.


Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao. Decoding
natural images from EEG for object recognition. _arXiv_, 2308.13234, 2023.


Concetto Spampinato, Simone Palazzo, Isaak Kavasidis, Daniela Giordano, Mubarak Shah, and
Nasim Souly. Deep learning human mind for automated visual classification. _arXiv_, 1609.00344,
2016.


Concetto Spampinato, Simone Palazzo, Isaak Kavasidis, Daniela Giordano, Nasim Souly, and
Mubarak Shah. Deep learning human mind for automated visual classification. In _Computer_
_Vision and Pattern Recognition_, pp. 6809–6817, 2017.


Jiajia Tang, Yutao Yang, Qibin Zhao, Yu Ding, Jianhai Zhang, Yang Song, and Wanzeng Kong.
Visual-guided dual-spatial interaction network for fine-grained brain semantic decoding. _Trans-_
_actions on Instrumentation and Measurement_, 73:1–14, 2024.


Praveen Tirupattur, Yogesh Singh Rawat, Concetto Spampinato, and Mubarak Shah. ThoughtViz:
Visualizing human thoughts using generative adversarial network. In _International_ _Conference_
_on Multimedia_, pp. 950–958, 2018.


Pieter Johannes Uys. Image classification from EEG brain signals using machine learning and deep
learning techniques. Master’s thesis, Stellenbosch University, 2019.


Kanan Wahengbam, Kshetrimayum Linthoinganbi Devi, and Aheibam Dinamani Singh. Fortifying
brain signals for robust interpretation. _Transactions on Network Science and Engineering_, 10(2):
742–753, 2023.


Fang Wang, Sheng Hua Zhong, Jianfeng Peng, Jianmin Jiang, and Yan Liu. Data augmentation
for EEG-based emotion recognition with deep convolutional neural networks. _Lecture_ _Notes_ _in_
_Computer Science_, 10705:82–93, 2018.


Pan Wang, Danlin Peng, Ling Li, Liuqing Chen, Chao Wu, Xiaoyi Wang, Peter Childs, and Yike
Guo. Human-in-the-loop design with machine learning. In _International_ _Conference_ _on_ _Engi-_
_neering Design_, pp. 2577–2586, 2019.


Pan Wang, Shuo Wang, Danlin Peng, Liuqing Chen, Chao Wu, Zhen Wei, Peter Childs, Yike Guo,
and Ling Li. Neurocognition-inspired design with machine learning. _Design science_, 6:e33, 2020.


Xin Xiang, Wenhui Zhou, and Guojun Dai. Electroencephalography-driven three-dimensional object decoding with multi-view perception diffusion. _Engineering Applications of Artificial Intel-_
_ligence_, 156:111180, 2025.


16


Guangyu Yang and Jinguo Liu. A new framework combining diffusion models and the convolution
classifier for generating images from EEG signals. _Brain Sciences_, 14(5):478, 2024.


Zesheng Ye, Lina Yao, Yu Zhang, and Sylvia Gustin. Self-supervised cross-modal visual retrieval
from brain activities. _Pattern Recognition_, 145:109915, 2024.


Hong Zeng, Nianzhang Xia, Dongguan Qian, Motonobu Hattori, Chu Wang, and Wanzeng Kong.
DM-RE2I: A framework based on diffusion model for the reconstruction from EEG to image.
_Biomedical Signal Processing and Control_, 86:105125, 2023a.


Hong Zeng, Nianzhang Xia, Ming Tao, Deng Pan, Haohao Zheng, Chu Wang, Feifan Xu, Wael Zakaria, and Guojun Dai. DCAE: A dual conditional autoencoder framework for the reconstruction
from EEG into image. _Biomedical Signal Processing and Control_, 81:104440, 2023b.


Wenxiang Zhang and Qingshan Liu. Using the center loss function to improve deep learning performance for EEG signal classification. In _International Conference on Advanced Computational_
_Intelligence_, pp. 578–582, 2018.


X. Zhang, L. Yao, Q. Z. Sheng, S. S. Kanhere, T. Gu, and D. Zhang. Converting your thoughts to
texts: Enabling brain typing via deep feature learning of EEG signals. In _International Conference_
_on Pervasive Computing and Communications_, 2018.


Xiang Zhang, Lina Yao, Dalin Zhang, Xianzhi Wang, Quan Z. Sheng, and Tao Gu. Multi-person
brain activity recognition via comprehensive EEG signal analysis. In _International_ _Conference_
_on Mobile and Ubiquitous Systems:_ _Computing, Networking and Services_, 2017.


Linfeng Zheng, Peilin Chen, and Shiqi Wang. EidetiCom: A cross-modal brain-computer semantic
communication paradigm for decoding visual perception. _arXiv_, 2407.14936, 2024a.


Xianglin Zheng, Zehong Cao, and Quan Bai. An evoked potential-guided deep learning brain representation for visual classification. In _International Conference on Neural Information Processing_,
pp. 54–61, 2020a.


Xiao Zheng and Wanzhong Chen. An attention-based bi-LSTM method for visual object classification via EEG. _Biomedical Signal Processing and Control_, 63, 2021.


Xiao Zheng, Wanzhong Chen, Mingyang Li, Tao Zhang, Yang You, and Yun Jiang. Decoding human
brain activity with deep learning. _Biomedical Signal Processing and Control_, 56, 2020b.


Xiao Zheng, Wanzhong Chen, Yang You, Yun Jiang, Mingyang Li, and Tao Zhang. Ensemble deep
learning for automated visual classification using EEG signals. _Pattern Recognition_, 102, 2020c.


Xu Zheng, Ling Wang, Kanghao Chen, Yuanhuiyi Lyu, Jiazhou Zhou, and Lin Wang. EIT-1M: One
million EEG-image-text pairs for human visual-textual recognition and more. _arXiv_, 2407.01884,
2024b.


Saisai Zhong, Yadong Liu, Zongtan Zhou, and Dewen Hu. ELSTM-based visual decoding from singal [sic]-trial EEG recording. In _International Conference on Software Engineering and Service_
_Science_, pp. 1139–1142, 2018.


Shuqi Zhu, Ziyi Ye, Qingyao Ai, and Yiqun Liu. EEG-ImageNet: An electroencephalogram dataset
and benchmarks with image visual stimuli of multi-granularity labels. _arXiv_, 2406.07151, 2024.


17