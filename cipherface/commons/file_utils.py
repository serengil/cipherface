#  built-in dependencies
import base64
from typing import Union


def write_data(file_name: str, data: Union[bytes, str]) -> None:
    """
    Write data to a file.
    Args:
        file_name (str): The name of the file.
        data (Union[bytes, str]): The data to write.
    """
    if isinstance(data, bytes):
        data = base64.b64encode(data)

    with open(file_name, "wb") as f:
        f.write(data)


def read_data(file_name: str) -> bytes:
    """
    Read data from a file.
    Args:
        file_name (str): The name of the file.
    Returns:
        data (bytes): The data read from the
    """
    with open(file_name, "rb") as f:
        data = f.read()

    # base64 to bytes
    return base64.b64decode(data)
