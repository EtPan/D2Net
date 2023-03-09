import torch.utils.data as data
import numpy as np
import pyarrow as pa

class LMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        self.repeat = repeat

    def __getitem__(self, index):
        index = index % self.length
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get('{:08}'.format(index).encode('ascii'))

        datum = pa.deserialize(raw_datum)
        (channels,height,width,data) = datum
        flat_x = np.fromstring(data, dtype=np.float32)
        x = flat_x.reshape(channels, height, width)

        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


    