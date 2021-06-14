class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                 train_folder: str, validation_folder: str, runs_folder: str,
                 start_epoch: int, experiment_name: str):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name



class HiDDenConfiguration():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, H: int, W: int,L: int, blocks_num: int,first_block: int
                 ):
        self.H = H
        self.W = W
        self.L = L
        self.blocks_num = blocks_num
        self.first_block = first_block
