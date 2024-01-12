class ModelTypeError(Exception):
    def __init__(self, model_type):
        message = "Unknown type of model: {}".format(model_type)
        super().__init__(message)
