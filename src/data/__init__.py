import collections

# Container for input processed datasets.
#     input_features: The representation for the input x.
#         size = [num_examples, num_features]
#     output_probs: The prediction p(y|x).
#         size = [num_examples, num_classes]
#     confidences: The confidence (typically p(Y = 1 | x) or p(Y = Y_hat | x).
#         size = [num_examples]
#     labels: Binary label (typically either Y = 1 or Y = Y_hat).
#         size = [num_examples]
InputDataset = collections.namedtuple(
    'InputDataset',
    ('input_features', 'output_probs', 'confidences', 'labels'),
    defaults=(None, None, None, None))


# Same as InputDataset, but of size [num_batches, ...].
BatchedInputDataset = collections.namedtuple(
    'BatchedInputDataset',
    ('input_features', 'output_probs', 'confidences', 'labels'),
    defaults=(None, None, None, None))
