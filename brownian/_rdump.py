import numpy as np
import sys

from six import iteritems
from six.moves import zip as izip
from six.moves import xrange
from itertools import chain, repeat, islice

def intersperse(delimiter, seq):
    # http://stackoverflow.com/a/5655803/344821
    return islice(chain.from_iterable(izip(repeat(delimiter), seq)), 1, None)

def _write_vec(vec, output):
    output.write('c(')
    for thing in intersperse(', ', vec):
        output.write(str(thing))
    output.write(')')

def _write_rep(val, output):
    if np.isscalar(val) and np.isreal(val):
        output.write(str(val))
    elif isinstance(val, xrange) and abs(val[0] - val[1]) == 1:
        output.write('{}:{}'.format(val[0], val[-1]))
    else:
        mat = np.asarray(val)
        if mat.ndim == 1:
            _write_vec(mat, output)
        elif mat.ndim > 1:
            output.write('structure(')
            _write_vec(mat.T.flat, output)
            output.write(', .Dim = ')
            _write_vec(mat.shape, output)
            output.write(')')
        else:
            raise TypeError("Don't know how to handle data {!r}".format(val))

def dump_to_rdata(output=sys.stdout, **things):
    assert hasattr(output, 'write')

    for name, val in iteritems(things):
        output.write(name)
        output.write(' <- ')
        _write_rep(val, output)
        output.write('\n')


# if __name__ == '__main__':
    # N = 1024
    # x = np.arange(N, dtype=np.float64)/4 
    # seed = 2354463
    # np.random.seed(seed)
    # data = {
    # 'N': N,
    # 'x': x,
    # 'seed': seed,
    # 'y': np.random.randn(N)*10 + 3*x - 2,
    # 'mu_alpha': 0,
    # 'sigma_alpha': 5,
    # 'mu_beta': 2,
    # 'sigma_beta': 2,
    # 'mu_sigma': 1
    # }
    # with open('data.dump', 'w') as f:
    #     dump_to_rdata(f, **data)