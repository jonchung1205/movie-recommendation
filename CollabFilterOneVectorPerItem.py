'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.zeros(1),
            b_per_user=ag_np.zeros(n_users), # FIX dimensionality
            c_per_item=ag_np.zeros(n_items), # FIX dimensionality
            U=0.001 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.001 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        if mu is None:
            # mu = self.param_dict.get('mu', 0.0)
            mu = self.param_dict['mu']
        if b_per_user is None:
            b_per_user = self.param_dict.get('b_per_user', ag_np.zeros(1))
        if c_per_item is None:
            c_per_item = self.param_dict.get('c_per_item', ag_np.zeros(1))
        if U is None:
            U = self.param_dict.get('U', ag_np.zeros((1, self.n_factors)))
        if V is None:
            V = self.param_dict.get('V', ag_np.zeros((1, self.n_factors)))

        #try to do array operations instead of individual items
        N = user_id_N.size
        
        # Initialize the predictions array
        yhat_N = ag_np.zeros(N)
        yhat_N += mu  
        
        if b_per_user is not None:
            user_bias = b_per_user[user_id_N]
            yhat_N += user_bias
        
        if c_per_item is not None:
            item_bias = c_per_item[item_id_N]
            yhat_N += item_bias

        if U is not None and V is not None:
            user_factors = U[user_id_N]
            item_factors = V[item_id_N]
            
            interaction = ag_np.sum(user_factors * item_factors, axis=1)
            yhat_N += interaction

        return yhat_N

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        user_id_N, item_id_N, y_N = data_tuple
        
        # Unpack parameters from param_dict
        mu = param_dict.get('mu')
        b_per_user = param_dict.get('b_per_user')
        c_per_item = param_dict.get('c_per_item')
        U = param_dict.get('U')
        V = param_dict.get('V')

        # Predict using unpacked parameters
        yhat_N = self.predict(user_id_N, item_id_N, mu, b_per_user, c_per_item, U, V)

        # Compute mean squared error (MSE) loss
        mse_loss = ag_np.mean((yhat_N - y_N) ** 2)

        # Compute regularization loss
        reg_loss = 0.0
        for param in param_dict.values():
            reg_loss += ag_np.sum(param ** 2)

        # Combine MSE loss and regularization loss
        loss_total = mse_loss + self.alpha * reg_loss

        
        #   DONE IN CLASS ##
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], item_id_N, mu, b_per_user, c_per_item, U, V)
        return loss_total 


def check_shapes(data_tuple, name="Data"):
    """
    This function checks and prints the shape of each element in the data tuple.
    
    Parameters:
    ----------
    data_tuple : tuple
        A tuple containing user_ids, item_ids, and ratings (all as numpy arrays).
    name : str
        The name to print for identification (e.g., "train", "valid", "test").
    """
    user_ids, item_ids, ratings = data_tuple
    
    print(f"Checking shapes for {name}:")
    print(f"User IDs shape: {np.shape(user_ids)}")
    print(f"Item IDs shape: {np.shape(item_ids)}")
    print(f"Ratings shape: {np.shape(ratings)}")
    print("\n")

if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

    # Check the shape of train, valid, and test datasets
    check_shapes(train_tuple, name="train")
    check_shapes(valid_tuple, name="valid")
    check_shapes(test_tuple, name="test")