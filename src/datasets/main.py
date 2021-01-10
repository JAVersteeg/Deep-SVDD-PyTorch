from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .fashion import FASHION_Dataset
from .crack import CRACK_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'fashion', 'crack', 'crack128')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'fashion':
        dataset = FASHION_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'crack':
        dataset = CRACK_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'crack128':
        dataset = CRACK_Dataset(root=data_path, normal_class=normal_class, patch_size=128)

    return dataset
