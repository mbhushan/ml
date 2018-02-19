def train(self, features, targets):
    ''' Train the network on batch of features and targets.

        Arguments
        ---------

        features: 2D array, each row is one data record, each column is a feature
        targets: 1D array of target values

    '''
    n_records = features.shape[0]
    delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
    delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
    for X, y in zip(features, targets):
        final_outputs, hidden_outputs = self.forward_pass_train(X)