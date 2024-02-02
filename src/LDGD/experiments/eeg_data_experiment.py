import numpy as np
import mogptk
import pandas as pd
import os
import matplotlib.pyplot as plt

np.random.seed(1)

working_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

dataset_pd = pd.read_csv(working_folder+'/data/eeg.csv', header=0, index_col=0)
cols = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG O1-Ref',
        'EEG O2-Ref']
t = dataset_pd['time'].values
y = dataset_pd[cols].values

data = mogptk.DataSet()
for i in range(len(cols)):
    data.append(mogptk.Data(t, y[:, i], name=cols[i]))

for i, channel in enumerate(data):
    channel.transform(mogptk.TransformNormalize())
    channel.remove_randomly(pct=0.4)

    if i not in [0, 1, 2, 3, 5, 7]:
        channel.remove_range(45, None)

# simulate sensor failure
data[0].remove_range(25, 35)
data[5].remove_range(None, 10)
data[7].remove_range(None, 10)


data.plot()
plt.show()

model = mogptk.MOHSM(data, Q=2, P=2)
model.init_parameters('BNSE')
model.train(method='Adam', lr=0.1, iters=100, verbose=True, error='MAE')

model.plot_gram()
plt.show()

model.plot_prediction(transformed=True)


