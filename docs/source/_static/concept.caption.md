# How rsa-nd works

Schematic with simulated values, not results. Each item's data is drawn as a cube
with axes for time (8 points), frequency (6) and space (5 sensors). `rsa-nd`
handles any number of named axes; three are drawn because three can be shown.
Colours show values on a red-blue scale centred on zero, as in MEG and EEG maps.

**How it works.** A worked example with 6 items, searching frequency and time
(`search_axes=("freq", "time")`). (1) A patch, outlined in black, is the part of a
cube used for one comparison: here one time point and one frequency, across all 5
sensors, taken from the same place in every item. Items 3 to 5 are not drawn.
(2) Each item's patch is flattened into one row of 5 values, giving a table of
6 items by 5 sensors. Items 1 to 3 share one spatial pattern and items 4 to 6 the
opposite one. (3) The data RDM compares every pair of rows: it has one row and one
column per item, so it is 6 by 6. The outlined cell compares the outlined rows 1
and 6. Blue means two items are alike and red means they differ (correlation
distance, from 0 to 2, drawn in lighter colours). The model RDM has the same size and predicts that items 1
to 3 are alike and items 4 to 6 are alike. (4) The RSA step itself: the Spearman
correlation between the data RDM and the model RDM gives one score, shown in the
circle. (5) The score fills the cell of the result at the patch's
time and frequency, outlined. Moving the patch fills all 8 × 6 cells. Space is not
an axis of the result, because the sensors were the features. All values, including
r, are computed from the simulated data drawn.

**Choose what to search.** Each cube shows the part of the data that one patch
covers, in grey. An orange arrow and label mark a searched axis, which has one score
per position in the result; every other axis is pooled into the features of the
patch. From left to right: searching time with a radius gives a time course, and
each patch spans neighbouring time points. Searching frequency and time gives a
time-frequency map, with one score per frequency (4 to 30 Hz). Searching all three
axes with radii gives a score for every point, from a small block around it.
Grouping frequencies into named bands first averages each band to one value, then
searches bands and time. Not drawn: `search_axes=()` pools everything into one score,
and a radius of 0 gives single points.
