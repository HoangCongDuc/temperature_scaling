import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self
    

def get_temperature(logits, labels, max_iter=50):
    "Standalone function for temperature scaling on precomputed outputs"
    temperature = torch.tensor(1.0, dtype=torch.float32, device=logits.device, requires_grad=True)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(eval)
    
    return temperature.cpu().item()
    

def get_temperature_search(logits, labels, temperatures=None):
    ece_func = ECELoss()
    if temperatures is None:
        temperatures = (torch.arange(500, device=logits.device) + 1) / 100
    else:
        temperatures = torch.flatten(temperatures)
    logits_with_temp = logits / temperatures[:, None, None]
    eces = ece_func(logits_with_temp, labels)
    ece, idx = torch.min(eces, dim=0)
    temp_best = temperatures[idx]
    return temp_best.cpu().item(), ece.cpu().item()

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_lowers[0] = -1     # Include edge cases with 0 confidence
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        """
        logits shape: (T, N, C) or (N, C)
        labels shape: (N,)
        T is the number of temperatures, N is number of samples, C is number of classes
        To compute ECE for a single temperature, divide the model output logits by the temperature and
        feed to this function together with the labels
        To compute ECE for multiple temperatures, divide the output logits with each temperature,
        stack them in the extra first dimension, then feed to this function

        output: Tensor of Size([]) or of Size([T]), one element for each temperature
        """
        softmaxes = F.softmax(logits, dim=-1) # (T, N, C)
        confidences, predictions = torch.max(softmaxes, -1) # (T, N)
        corrects = predictions.eq(labels).float() # (T, N)

        device = logits.device
        temperature_dims = logits.shape[:-2]
        n_samples = logits.shape[-2]
        zeros = torch.tensor(0.0, device=device)
        ece = torch.zeros(temperature_dims, device=device) # (T,)
        bin_lowers = self.bin_lowers.to(device) # (B,)
        bin_uppers = self.bin_uppers.to(device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper) # (T, N)
            confidence_in_bin = confidences.where(in_bin, zeros).sum(dim=-1) # (T,)
            correct_in_bin = corrects.where(in_bin, zeros).sum(dim=-1) # (T,)
            count_in_bin = in_bin.sum(dim=-1) # (T,)
            # prop_in_bin = in_bin.float().mean(dim=-1)
            # ece_in_bin = torch.abs((confidence_in_bin - correct_in_bin) / count_in_bin)
            # ece_in_bin *= prop_in_bin
            ece_in_bin = torch.abs((confidence_in_bin - correct_in_bin) / n_samples) # Reduced from last 3 lines
            ece += ece_in_bin.where(count_in_bin > 0, zeros)

        return ece
