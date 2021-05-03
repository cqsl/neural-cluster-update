# Write and read data with a buffer size smaller than the whole data size
# Assume data_count % chunk_size == 0

import h5py
import numpy as np


class ChunkedDataWriter:
    def __init__(self,
                 filename,
                 proto,
                 chunk_size,
                 compression=True,
                 chunk_id_sep=':'):
        self.proto = proto
        self.chunk_size = chunk_size
        self.compression = compression
        self.chunk_id_sep = chunk_id_sep

        self.count = 0
        self.buffers = {
            key: np.empty(shape=(self.chunk_size, ) + (shape or ()),
                          dtype=dtype)
            for key, dtype, shape in self.proto
        }
        self.buffer_idx = 0
        self.h5file = h5py.File(filename, 'w')

    def __enter__(self):
        return self

    def write(self, *args):
        for (key, _, _), data in zip(self.proto, args):
            self.buffers[key][self.buffer_idx] = data

        self.count += 1
        self.buffer_idx += 1
        if self.buffer_idx >= self.chunk_size:
            self.flush()

    def write_batch(self, *args):
        batch_size = args[0].shape[0]
        assert all(x.shape[0] == batch_size for x in args[1:])

        data_idx = 0
        while data_idx < batch_size:
            data_rest = batch_size - data_idx
            buffer_rest = self.chunk_size - self.buffer_idx
            if data_rest < buffer_rest:
                for (key, _, _), data in zip(self.proto, args):
                    buffer = self.buffers[key]
                    buffer[self.buffer_idx:self.buffer_idx +
                           data_rest] = data[data_idx:]

                self.count += data_rest
                data_idx += data_rest
                self.buffer_idx += data_rest
            else:
                for (key, _, _), data in zip(self.proto, args):
                    buffer = self.buffers[key]
                    buffer[self.buffer_idx:] = data[data_idx:data_idx +
                                                    buffer_rest]

                self.count += buffer_rest
                data_idx += buffer_rest
                self.buffer_idx += buffer_rest
                self.flush()

    # Write data without chunk
    def create_dataset(self, key, data):
        self.h5file.create_dataset(
            key,
            data=data,
            compression='gzip' if self.compression else None,
            shuffle=self.compression)

    def flush(self):
        if self.buffer_idx <= 0:
            return

        for key, _, _ in self.proto:
            self.h5file.create_dataset(
                '{}{}{}'.format(key, self.chunk_id_sep,
                                self.count // self.chunk_size),
                data=self.buffers[key][:self.buffer_idx],
                compression='gzip' if self.compression else None,
                shuffle=self.compression)

        self.buffer_idx = 0
        self.h5file.flush()

    def close(self):
        self.flush()
        self.h5file.close()

    def __exit__(self, type, value, traceback):
        self.close()


class ChunkedDataReader:
    def __init__(self, filename, chunk_id_sep=':'):
        self.h5file = h5py.File(filename, 'r')
        self.chunk_id_sep = chunk_id_sep

        self.keys = self.h5file.keys()
        self.key_names = {key.split(self.chunk_id_sep)[0] for key in self.keys}

    def __enter__(self):
        return self

    def get_id(self, key):
        return int(key.split(self.chunk_id_sep)[1])

    def get(self, key, min_chunk=None, max_chunk=None):
        keys = filter(lambda x: x.startswith(key + self.chunk_id_sep),
                      self.keys)
        if min_chunk:
            keys = filter(lambda x: self.get_id(x) >= min_chunk, keys)
        if max_chunk:
            keys = filter(lambda x: self.get_id(x) < max_chunk, keys)
        keys = sorted(keys, key=lambda x: self.get_id(x))

        if not keys:
            raise KeyError(
                f'{key}, min_chunk={min_chunk}, max_chunk={max_chunk}')

        data = [self.h5file[key] for key in keys]
        data = np.concatenate(data)

        return data

    def __getitem__(self, key):
        if key in self.keys:
            # Data is without chunk
            return np.array(self.h5file[key])
        return self.get(key)

    def close(self):
        self.h5file.close()

    def __exit__(self, type, value, traceback):
        self.close()
