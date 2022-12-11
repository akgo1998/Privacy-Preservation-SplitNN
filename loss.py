import torch

class NoPeekLoss(torch.nn.modules.loss._Loss):
    
    def __init__(self, dcor_weighting: float = 0.1) -> None:
        super().__init__()
        self.dcor_weighting = dcor_weighting

        self.ce = torch.nn.CrossEntropyLoss()
        self.dcor = DistanceCorrelationLoss()

    def forward(self, inputs, intermediates, outputs, targets):
        return self.ce(outputs, targets) + self.dcor_weighting * self.dcor(inputs, intermediates)


class DistanceCorrelationLoss(torch.nn.modules.loss._Loss):
    def forward(self, input_data, intermediate_data):
        
        input_data = self._Normalize_matrix(input_data.view(input_data.size(0), -1))
        intermediate_data = self._Normalize_matrix(intermediate_data.view(intermediate_data.size(0), -1))

        covX = torch.matmul(input_data.T,input_data)
        covZ = torch.matmul(intermediate_data.T,intermediate_data)
        
        diag1inv = (torch.diag(torch.diag(1/covX)))
        diag2inv = (torch.diag(torch.diag(1/covZ)))

        XTZ = torch.matmul(input_data.T,intermediate_data)**2

        dcorr = torch.matmul(torch.matmul(diag1inv, XTZ), diag2inv)

        return dcorr.sum().sqrt()

    def _Normalize_matrix(self, data):

        row_mean = data.mean(dim=0, keepdim=True)
        col_mean = data.mean(dim=1, keepdim=True)
        data_mean = data.mean()

        return data - row_mean - col_mean + data_mean

