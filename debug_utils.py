import tensorflow as tf
from colorama import Fore, Style
from datetime import datetime


def format_object(o):
  if isinstance(o, tf.Tensor):
    return f"T[{o.shape}, {repr(o.dtype)}]"

  if isinstance(o, tuple):
    return tuple(format_object(x) for x in o)

  if isinstance(o, list):
    return [format_object(x) for x in o]

  return o


def p(*args, **kwargs):
  time = datetime.now().strftime("%H:%M:%S")

  items = []

  for arg in args:
    items.append(format_object(arg))

  for k, v in kwargs.items():
    items.append(f"{k}={format_object(v)}")

  print(f"{Fore.GREEN}[{time}]{Style.RESET_ALL}{Fore.LIGHTRED_EX}", *items, Style.RESET_ALL)