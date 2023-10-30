class LambdaObjectiveFunction:
    """
    This class is used to create a lambda objective function.
    """

    def __init__(self, lambda_function, name):
        self.lambda_function = lambda_function
        self.name = name

    def make_objective_function(self, *args, **kwargs):
        """
        This method is wrapper that takes an objective function and
        prepares it for use in the optimiser.
        """

        def objective_function(theta):
            return self.lambda_function(theta, *args, **kwargs)

        return objective_function
