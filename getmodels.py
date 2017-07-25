import urllib
import os
import stat
from numpy import ceil

# def download_file(url, filename):
#     """Streaming download of a large file.

#     Modified from stackoverflow, http://stackoverflow.com/a/16696317
#     """
#     # NOTE the stream=True parameter
#     r = requests.get(url, stream=True)
#     chunk_size = 8192
#     file_size = int(r.headers.get('content-length'))
#     total = int(ceil(file_size / chunk_size))
#     with open(filename, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=chunk_size): 
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)
#                 #f.flush() commented by recommendation from J.F.Sebastian


urls =  {
    "https://s3-us-west-2.amazonaws.com/brownian-stan/ryanpdwyer/stan-linear-ex/10/10.1/gamma": "brownian/stanmodels/gamma.linux",
    "https://s3-us-west-2.amazonaws.com/brownian-stan/ryanpdwyer/stan-linear-ex/10/10.1/gamma2": "brownian/stanmodels/gamma2.linux",
    "https://s3-us-west-2.amazonaws.com/brownian-stan/ryanpdwyer/stan-linear-ex/10/10.2/gamma": "brownian/stanmodels/gamma.osx",
    "https://s3-us-west-2.amazonaws.com/brownian-stan/ryanpdwyer/stan-linear-ex/10/10.2/gamma2": "brownian/stanmodels/gamma2.osx",
    "https://s3-us-west-2.amazonaws.com/brownian-stan/ryanpdwyer/stan-linear-ex/appveyor/01/gamma.exe":
                                "brownian/stanmodels/gamma.exe",
    "https://s3-us-west-2.amazonaws.com/brownian-stan/ryanpdwyer/stan-linear-ex/appveyor/01/gamma2.exe": "brownian/stanmodels/gamma2.exe"
}


# Download all the files
for url, file in urls.items():
    urllib.urlretrieve(url, file)
    st = os.stat(file)
    os.chmod(file, st.st_mode | stat.S_IEXEC)
