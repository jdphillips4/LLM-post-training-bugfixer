from calculator import calculate_average


def test_average_of_positive_numbers():
    assert calculate_average([2, 4, 6]) == 4


def test_average_of_negative_numbers():
    assert calculate_average([-2, -4, -6]) == -4


def test_average_of_empty_list():
    assert calculate_average([]) == 0