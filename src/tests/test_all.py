from ..game import n_ary_data

def test_n_ary_data():
    dataset = n_ary_data(n=2)
    assert len(dataset) == 4