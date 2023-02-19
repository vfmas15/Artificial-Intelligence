import torch
import pytorch_lightning
import math
import torchmetrics

class ImageAnalyzer(pytorch_lightning.LightningModule):
    def __init__(self, img_width, img_height, channels,n_of_classes, number_of_conv_layers=3,
                depth_of_conv_layer=2, decrease_factor = 4, learning_rate = 2e-4):

        super(ImageAnalyzer,self).__init__()
        self.w, self.h, self.c = img_width, img_height, channels # Size of images
        self.n_conv_layers = number_of_conv_layers 
        self.depth_conv = depth_of_conv_layer
        self.classes = n_of_classes
        self.d_factor = decrease_factor
        self.lr = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.classes)

        # -------- BASE MODEL --------
        # n_conv_layers: Depth of every layer
        # For a depth of 2: Conv2D -> ReLu -> Conv2D -> ReLu
        self.model = torch.nn.Sequential()
        fractional_w = 0.0
        fractional_h = 0.0
        while self.n_conv_layers > 0:

            fractional_w, _ = math.modf(self.w / 2)
            fractional_h, _ = math.modf(self.h / 2)

            if fractional_w != 0 or fractional_h != 0:
                break

            for j in range( self.depth_conv):
                self.model.append(torch.nn.Conv2d(self.c, self.c*2,
                                                kernel_size = 3,stride = 1, padding = 1))
                self.model.append(torch.nn.ReLU())
                self.c = self.c*2

            self.model.append(torch.nn.MaxPool2d(kernel_size = 2, stride = 2))

            self.w = self.w / 2
            self.h = self.h / 2

            self.n_conv_layers -= 1

            
            

        # -------- TOP MODEL --------
        # d_factor: Factor that will make the neurons decrese
        # For a factor of 2: Linear(128, 64) -> Linear(64, 32) ...
        # For a factor of 3: Linear(128, 42) -> Linear(42, 14) ...
        self.model.append(torch.nn.Flatten())
        neurons = int(math.floor(self.w * self.h * self.c))
        while True:
            self.model.append(torch.nn.Linear(int(neurons),int(math.floor(neurons/self.d_factor))))
            self.model.append(torch.nn.ReLU())
            neurons = int(math.floor(neurons) / self.d_factor)
            if int(math.floor(neurons) / self.d_factor) <= self.classes:
                self.model.append(torch.nn.Linear(neurons, self.classes))
                break

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
        return torch.nn.functional.log_softmax(x, dim=1)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer    