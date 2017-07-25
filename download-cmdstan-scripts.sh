#!/usr/bin/env bash
N_TRAVIS=10

aws s3 cp s3://brownian-stan/ryanpdwyer/stan-linear-ex/$N_TRAVIS/$N_TRAVIS.1/gamma brownian/stanmodels/gamma.linux
aws s3 cp s3://brownian-stan/ryanpdwyer/stan-linear-ex/$N_TRAVIS/$N_TRAVIS.1/gamma2 brownian/stanmodels/gamma2.linux
aws s3 cp s3://brownian-stan/ryanpdwyer/stan-linear-ex/$N_TRAVIS/$N_TRAVIS.2/gamma brownian/stanmodels/gamma.osx
aws s3 cp s3://brownian-stan/ryanpdwyer/stan-linear-ex/$N_TRAVIS/$N_TRAVIS.2/gamma2 brownian/stanmodels/gamma2.osx


aws s3 cp "s3://brownian-stan/ryanpdwyer/stan-linear-ex/appveyor/01/gamma.exe" brownian/stanmodels/gamma.exe
aws s3 cp "s3://brownian-stan/ryanpdwyer/stan-linear-ex/appveyor/01/gamma2.exe" brownian/stanmodels/gamma2.exe

cd brownian/stanmodels && chmod +x gamma*