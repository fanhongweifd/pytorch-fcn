from torchstat import stat
from torchsummary import summary


def summary_shape(model, input_shape):
    """
    显示每层输出的shape，以及每层的参数量
    :param model: nn.module
    :param input_shape: (c*h*w)
    """
    input_shape = tuple(input_shape)
    summary(model, input_shape)
    

def param_stat(model, input_shape):
    """
    显示每层的输出的shape，计算量flops,以及其他参数量
    :param model:n.module
    :param input_shape:
    """
    input_shape = tuple(input_shape)
    stat(model, input_shape)
    
    
if __name__ == "__main__":
    
    import torchvision.models as models
    model = models.vgg11()
    summary_shape(model, (3,224,224))
    param_stat(model, (3,224,224))
    print(models.resnet18())