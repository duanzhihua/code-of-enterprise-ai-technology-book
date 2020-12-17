# -*- coding: utf-8 -*-
import pandas as pd


def read_data(filename):
    reader = pd.read_csv(filename, delimiter=",", encoding='GB18030', iterator=True)
    loop = True
    chunk_size = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("read data : Iteration Success!")
    df = pd.concat(chunks, ignore_index=True)
    return df
