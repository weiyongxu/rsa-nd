# RSA by brain parcel, frequency band and time

One `compute_rsa()` call scores two models at every combination of 68 cortical
parcels, 3 frequency bands and 16 time points: 2 × 68 × 3 × 16 scores. `mne-rsa`
alone searches parcels and time; `rsa-nd` adds the band axis and both models in the
same call. Data come from one participant of the MNE sample dataset and are shown
without statistical tests or confidence intervals.

**Data.** One icon per axis of the input. Parcels are the 68 labels of the
Desikan-Killiany (`aparc`) atlas, drawn in their atlas colours on the left
hemisphere. The 120 trials are the items: 30 each of auditory tones in the left or
right ear, and visual checkerboards in the left or right visual field. Single-trial
MEG source estimates (dSPM, `lambda2 = 1/9`, normal orientation, 7,498 sources) were
transformed with Morlet wavelets (2 cycles; 5, 7, 9, 12, 16 and 24 Hz; one sample
every 40 ms). Log power was averaged within each band (theta 4–8 Hz, alpha 8–13 Hz,
beta 13–30 Hz) and z-scored across trials, from −0.1 to 0.5 s.

**Models.** Two 120 × 120 model RDMs, with trials ordered by condition (Aud and Vis
for auditory and visual, L and R for left and right). Blue marks trial pairs that
the model says are alike, and red marks pairs it says differ. *Auditory vs visual*
contrasts the modality; *left vs right* contrasts the side of the stimulus,
whatever its modality.

**One call.** The code that produced the scores, shortened. Each searchlight
position uses all source vertices of one parcel as features, within one band and
within 40 ms either side of one time point. Data RDMs use correlation distance, and
are compared with each model RDM by Spearman correlation (r).

**Results.** Rows show the models and columns the bands. Each parcel is coloured by
its score at the time of that model's highest score across all parcels, bands and
times (shown next to the model name), on the inflated cortex of the sample subject:
lateral views on top, medial views below, for the left and right hemispheres. Each
row has its own colour scale, centred on zero; the medial wall, which has no
parcel, is grey. The parcel with the highest score is outlined in black on every brain
of its row, and named in the view that shows it best for the band with that score.
The line plots follow this parcel over time, with one line per band in the band's
colour; the dotted line marks the time shown on the brains.

The modality model peaks in the left superior temporal cortex, which holds the
auditory cortex, and reaches visual areas too, since both differ between tones and
checkerboards. The side model peaks in medial occipital cortex (lingual and
pericalcarine), the primary visual cortex: a stimulus in the left or right visual
field drives opposite hemispheres, so the pattern across a visual parcel differs
by side. A sound in either ear drives both auditory cortices, so auditory parcels
carry little information about the side.
