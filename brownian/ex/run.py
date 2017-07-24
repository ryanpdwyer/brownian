import subprocess
import psutil
import pandas as pd
import numpy as np
from brownian._rdump import dump_to_rdata
from kpfm.util import txt_filename
from cStringIO import StringIO

N = 4
Nsamples = 500
Nwarmup = 500
model = '../stanmodels/gamma2'
modelname = 'gamma2'
datafile = 'rdata.dump'
seed= 12344



@txt_filename
def split_header_data(fh):
    comments = []
    data = []
    for line in fh:
        if line.startswith('#'):
            comments.append(line)
        else:
            data.append(line)

    return (data, comments)

def cmdstan_sample(data, iterations, chains, warmup=None, datafile='data.dump',
                    model='stanmodels/gamma', modelname='gamma'):
    if data is not None:
        with open(datafile, 'w') as f:
            dump_to_rdata(f, **data)

    if warmup is None:
        warmup = iterations


    processes = [subprocess.Popen([model, 'sample', 'num_samples={}'.format(iterations),
                      'num_warmup={}'.format(warmup),
                      'data', 'file={}'.format(datafile),
                      'id={}'.format(i),
                      'output',
                      'file={}-{}.csv'.format(modelname, i)])
                for i in xrange(1, chains+1)]

    ls = [psutil.Process(p.pid) for p in processes]

    gone, alive = psutil.wait_procs(ls, timeout=1000)

    filenames = ['{}-{}.csv'.format(modelname, i)
                for i in xrange(1, chains+1)]

    files = [open(filename, 'r') for filename in filenames]

    for i, fh in enumerate(files):
        data, comments = split_header_data(fh)
        data_io = StringIO("".join(data))
        df = pd.read_csv(data_io)
        if i == 0:
            df_full = df
        else:
            df2 = df.set_index(np.arange(i*iterations, (i+1)*iterations))
            df_full = df_full.append(df2)

    return df_full