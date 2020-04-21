import torch
import torch.utils.data as torchdata

from alr.acquisition import AcquisitionFunction


class DataManager:
    def __init__(self, acquirer: AcquisitionFunction, X_train: torch.Tensor, y_train: torch.Tensor,
                 X_pool: torch.Tensor, y_pool: torch.Tensor, **data_loader_params):
        """
        A stateful data manager class

        Training data and labels will be updated according to newly acquired samples
        as dictated by the provided `acquirer`.
        :py:attr:`training_data` returns said data
        as a :class:`~torch.utils.data.DataLoader` object with the specified `batch_size` in `data_loader_params`.

        :param acquirer: acquisition object
        :type acquirer: :class:`AcquisitionFunction`
        :param X_train: tensor object
        :type X_train: `torch.Tensor`
        :param y_train: tensor object
        :type y_train: `torch.Tensor`
        :param X_pool: tensor object
        :type X_pool: `torch.Tensor`
        :param y_pool: tensor object
        :type y_pool: `torch.Tensor`
        :param data_loader_params: keyword parameters to be passed into `DataLoader` when calling
            :py:attr:`training_data`
        """
        # TODO: accept as y_pool as None for actual use-cases
        self._acquirer = acquirer
        self._X_train = X_train
        self._y_train = y_train
        self._X_pool = X_pool
        self._y_pool = y_pool
        if not data_loader_params:
            self._data_loader_params = dict(shuffle=True, num_workers=2,
                                            pin_memory=True, batch_size=32)
        else:
            self._data_loader_params = data_loader_params

    def acquire(self, b: int) -> None:
        """
        Acquires `b` points from the provided `X_pool` according to `acquirer`.

        :param b: number of points to acquire
        :type b: int
        :return: None
        :rtype: NoneType
        """
        assert b <= self._X_pool.size(0)
        idxs = self._acquirer(self._X_pool, b)
        assert idxs.shape == (b,)
        self._X_train = torch.cat((self._X_train, self._X_pool[idxs]), dim=0)
        self._y_train = torch.cat((self._y_train, self._y_pool[idxs]), dim=0)
        mask = torch.ones(self._X_pool.size(0), dtype=torch.bool)
        mask[idxs] = 0
        self._X_pool = self._X_pool[mask]
        self._y_pool = self._y_pool[mask]

    @property
    def training_data(self) -> torchdata.DataLoader:
        """
        Returns current training data after being updated by :meth:`acquire`.

        :return: A `DataLoader` object than represents the latest updated training pool.
        :rtype: `DataLoader`
        """
        return torchdata.DataLoader(torchdata.TensorDataset(self._X_train, self._y_train),
                                    **self._data_loader_params)

    @property
    def train_size(self) -> int:
        """
        Current number of points in `X_train`.

        :return: `X_train.size(0)`
        :rtype: int
        """
        return self._X_train.size(0)


