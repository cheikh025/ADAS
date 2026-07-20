"""Dependency-light loader for SciCode's official HDF5 targets."""

from __future__ import annotations

import os
from typing import Any

import h5py


def _process_list(group: h5py.Group) -> list[Any]:
    return [group[key][()] for key in group.keys()]


def _process_sparse_matrix(group: h5py.Group):
    import scipy.sparse

    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        return scipy.sparse.coo_matrix(
            (data, (group["row"][()], group["col"][()])), shape=shape
        )
    if "blocksize" in group:
        return scipy.sparse.bsr_matrix(
            (data, group["indices"][()], group["indptr"][()]),
            shape=shape,
            blocksize=tuple(group["blocksize"][()]),
        )
    return scipy.sparse.csr_matrix(
        (data, group["indices"][()], group["indptr"][()]), shape=shape
    )


def _process_dict(group: h5py.Group) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            result[key] = _process_sparse_matrix(obj["sparse_matrix"])
        elif isinstance(obj[()], bytes):
            result[key] = obj[()].decode("utf-8", errors="strict")
        else:
            try:
                result[float(key)] = obj[()]
            except ValueError:
                result[key] = obj[()]
    return result


def _process_data_group(group: h5py.Group):
    for key in group.keys():
        if key == "list":
            return _process_list(group[key])
        if key == "sparse_matrix":
            return _process_sparse_matrix(group[key])
        return _process_dict(group)
    return None


def process_hdf5_to_tuple(
    step_id: str,
    test_num: int,
    h5py_file: str | os.PathLike[str],
) -> list[Any]:
    data: list[Any] = []
    with h5py.File(h5py_file, "r") as handle:
        for test_id in range(test_num):
            group_path = f"{step_id}/test{test_id + 1}"
            if group_path not in handle:
                raise FileNotFoundError(
                    f"Path {group_path!r} was not found in {h5py_file}."
                )
            group = handle[group_path]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"Expected HDF5 group at {group_path!r}.")

            keys = list(group.keys())
            if len(keys) == 1:
                subgroup = group[keys[0]]
                if isinstance(subgroup, h5py.Dataset):
                    value = subgroup[()]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="strict")
                    data.append(value)
                elif isinstance(subgroup, h5py.Group):
                    data.append(_process_data_group(subgroup))
                continue

            values: list[Any] = []
            for key in keys:
                subgroup = group[key]
                if isinstance(subgroup, h5py.Dataset):
                    value = subgroup[()]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="strict")
                    values.append(value)
                elif isinstance(subgroup, h5py.Group):
                    values.append(_process_data_group(subgroup))
            data.append(tuple(values))
    return data
