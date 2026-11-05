This is a Julia library for bandit experiments.

We support both pure exploration and regret minimisation problems. We aim to adapt to a variety of structured bandit problems (Lipschitz, unimodal, categorical, ...), both for regret minimisation and for pure exploration objectives (including best arm, minimum threshold, largest profit, ... ).

The files in this directory are shared library files. They are included by the regret minimisation code in [regret_games_paper](regret_games_paper), and the pure exploration code in [purex_games_paper](purex_games_paper). To get started, visit either one.
