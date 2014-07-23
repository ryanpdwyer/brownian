HDF5 Format for Brownian Motion Data
====================================

Here is a sample of the HDF5 file format.

```
/
attributes: {'date': '2014-07-22',
             'time': '13:00:00',
             'version': 'pos-PSD-1.0',
             'instrument': 'PSB-B19-AFM',
             'help': 'Nice help about the data structure; omitted here for brevity.'}
        f
        attributes: {'unit': 'Hz',
                     'help': 'Frequency array for PSD.'}
        [0, 0.5, 1, 1.5]

    PSD/
    attributes: {'n': 4,
                 'units': 'nm^2/Hz',
                 'help': 'Power spectral density of position fluctuations.'}
        mean        
            [1, 2.1, 2.9, 4]
        stdev
            [0.2, 0.3, 0.4, 0.5]
  
``` 

