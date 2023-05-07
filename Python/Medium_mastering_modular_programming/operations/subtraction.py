def diff(x: int, y: int) -> int:
    """this function subtracts two integers.

    Args:
        param 1: integer
        param 2: integer

    Returns:
        an integer that is the difference of the two arguments
    """
    if type(x) and type(y) == int:
        print(f"the difference of these two numbers is: {x-y}")
    else:
        print("arguments are not integers!")


if __name__ == "__main__":
    diff()