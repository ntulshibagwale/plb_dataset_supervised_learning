5-2-2022

Caelin just pulled a new dataset, so I'm going to 
try and run the neural networks on the 20 and 40
and see how well it extrapolates to the other.s



So I just started looking at the analysis. Seems
like this dataset, when trained on 20 and 40 deg
was not really able to predict the other waveforms.

The model architecture is the same as experiment 02
and all the training hyperparameters are the same.
Just did worse. The training set is smaller, so maybe
that has something to do with it. Or maybe there
is something different about the way these waveforms
were collected since they were acquired on different days.


Something I could check: run this model trained on the newer
data to the OLD data. That would rule out whether it was the 
model or the data...I think.