### Federated learning for EV battery remaining useful life prediction

The implementation was created using Flower framework. See flower.dev

Base model is from https://www.mdpi.com/1996-1073/16/6/2837 and 
can be found here https://github.com/MichaelBosello/battery-rul-estimation

The base implementation is based on NASA Randomized dataset that can
be found here https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

There are four configurations:
- Baseline FL with no privacy and security methods;
- DP FL with differential privacy (built into Flower);
- HE FL with secure aggregation using homomorphic encryption from Pyfhel lib;
- DP HE FL the combination of the previous two;

The HE methods require changes in the (...)/flwr/common/parameter.py file.
See HOMOMORPHIC_ENCRYPTION/ readme file for more info.

If you want to test this, I would advise you to create two seperate virtual envs
- one for HE methods;
- and one for the others.

See https://github.com/adap/flower for the default server strategy and client source
