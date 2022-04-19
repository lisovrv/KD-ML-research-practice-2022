import typing
from typing import Type, Union

NoneType = type(None)


# In Python 3.8+, typing library includes get_origin and get_args methods.
# Unfortunately, Google Colab runs Python 3.7, in which there are no such functions.
# So I followed https://stackoverflow.com/a/50101934/7280039 and implemented these
# functions myself.


def get_origin(type_: Type):
    if hasattr(typing, "get_origin"):
        return typing.get_origin(type_)
    elif hasattr(type_, "__origin__"):
        return type_.__origin__
    else:
        return None


def get_args(type_: Type):
    if hasattr(typing, "get_args"):
        return typing.get_args(type_)
    elif hasattr(type_, "__args__"):
        return type_.__args__
    else:
        return ()


def is_optional(type_: Type):
    type_origin = get_origin(type_)
    type_args = get_args(type_)
    return type_origin is Union and len(type_args) == 2 and type_args[-1] is NoneType
