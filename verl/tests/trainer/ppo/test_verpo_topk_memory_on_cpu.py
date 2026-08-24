import torch

from verl.trainer.distillation.verpo_zpd import _indexed_rows_view_when_contiguous


def test_contiguous_response_rows_are_selected_as_a_view():
    values = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5)

    selected = _indexed_rows_view_when_contiguous(values, torch.tensor([2, 3, 4]))

    torch.testing.assert_close(selected, values[2:5])
    assert selected.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()


def test_noncontiguous_response_rows_keep_advanced_indexing_semantics():
    values = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5)
    indices = torch.tensor([1, 3, 6])

    selected = _indexed_rows_view_when_contiguous(values, indices)

    torch.testing.assert_close(selected, values[indices])
    assert selected.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
