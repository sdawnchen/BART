# Overview

This version of the code includes the basic relation learning model and the importance-guided mapping algorithm for solving analogy problems. Code for the distance effect and the extra simulations involving test items that go beyond the training data in various ways (e.g., containing
different animals) are not included.


# Main Files

The following three files are the main programs to run to test the model:

`learn_test_relations.m`: This program trains the model on comparative relations and evaluates its generalization performance.  Many parameters can be set at the top of this file, for example, which input (ratings, Leuven, or topics) to use, whether to run BART or the baseline model, the number of training examples to use, etc.  The generalization results are saved in the `results/<input>/generalization` folder, and the resulting relational representations are saved in the `results/<input>/matrices` folder (see **Directory Structure** below for more details).

`analogy_test_all.m`: Once `learn_test_relations.m` has been run to generate the appropriate relational representations, this program can be run to test all the analogy types using those relational representations.  Many parameters can be changed in this file as well.  In particular,
`controlindx` can be used to control how much information about the relational representations (all means, variances, and covariances, only means and variances, or only means) to use.

`gen_learn_curve.m`: This program is for conveniently generating learning curves or running many simulations at once.


# Other Files

Some info about the other files:

`data\*.mat` files: Input files for ratings, Leuven, and topics.

`learn_all_predicates.m`: Learns representations for the single-place predicates (large, small, etc.).

`create_emp_prior.m`: Uses the training data for a comparative relation to select a single-place predidicate to construct the empirical prior for that relation.

`evaluate_relation.m`: Evaluates generalization performance on a relation, calculating absolute accuracy, relative accuracy, and Az.

`analogy_test_\*.m`: Tests each type of analogy problem.

`analogy_importance_mapping.m`: Uses importance-guided mapping to compute the analogical distance of a single four-term analogy.


# Directory Structure

The outputs are saved in the `results/<input>/ directory`, where `<input>` is `ratings`, `leuven`, or `topics`.  Each input directory has the same structure:

```
results
    |__ <input>
            |__ analogy
            |       |__ BART
            |       |__ baseline
            |
            |__ generalization
            |       |__ BART
            |       |__ baseline
            |
            |__ matrices
                    |__ predicates
                    |__ relations
                            |__ BART
                            |__ baseline
```

The `predicates` folder under `matrices` contains the learned predicate representations, whereas the `relations` folder contains the learned relational representations.