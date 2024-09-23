import torch




class CNN1D(BaseSurrogate):

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.model = 
        self.likelihood = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def
