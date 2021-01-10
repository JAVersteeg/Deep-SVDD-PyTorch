from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .fashionNet import FASHION_NET, FASHION_NET_Autoencoder
from .crackNet import Crack_Architecture_Encoder, Crack_Architecture_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'fashionNet', 'crackNet', 'crackNet128')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'fashionNet':
        net = FASHION_NET()

    if net_name == 'crackNet':
        net = Crack_Architecture_Encoder()
        
    if net_name == 'crackNet128':
        net = Crack_Architecture_Encoder(rep_dim=128)

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'fashionNet', 'crackNet', 'crackNet128')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'fashionNet':
        ae_net = FASHION_NET_Autoencoder()

    if net_name == 'crackNet':
        ae_net = Crack_Architecture_Autoencoder()
    
    if net_name == 'crackNet128':
        ae_net = Crack_Architecture_Autoencoder(rep_dim=128)

    return ae_net
