# AttackOlineRanker
To run the script:

python3 script.py [-h] [-L LENGTH] [-d DIMENSION] -T TIME [-r REPEAT] -cm
                 CLICK_MODELS [CLICK_MODELS ...] [-s SYNTHETIC] [-t TABULAR]
                 [-f FILENAME] -a ALGORITHMS [ALGORITHMS ...]

Run script for testing attack algorithms.

optional arguments:

  -h, --help            show this help message and exit

  -L LENGTH, --length LENGTH
                        Number of items to extract (default 1000)

  -d DIMENSION, --dimension DIMENSION
                        Dimension of feature space (default 5)

  -T TIME, --time TIME  Time t or Number of iterations required

  -r REPEAT, --repeat REPEAT
                        Number of runs (required)

  -cm CLICK_MODELS [CLICK_MODELS ...], --click_models CLICK_MODELS [CLICK_MODELS ...]
                        Click models to be used (cas or pbm)

  -s SYNTHETIC, --synthetic SYNTHETIC
                        Use synthetic item or not (default true)

  -t TABULAR, --tabular TABULAR
                        Use diagonal function to generate items or not (default false)

  -f FILENAME, --filename FILENAME
                        Use the provided dataset, default is MovieLens dataset

  -a ALGORITHMS [ALGORITHMS ...], --algorithms ALGORITHMS [ALGORITHMS ...]
                        The algorithms that would be attacked UCB stands for
                        pbm-UCB or cas-UCB, Top stands for Top ranker
                        required

OR

Just run python3 try_attack.py